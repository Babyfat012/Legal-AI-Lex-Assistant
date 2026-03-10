import re
import logging
from dataclasses import dataclass, field
from pydantic import BaseModel, model_validator
from langchain_text_splitters import RecursiveCharacterTextSplitter
from ingestion.loading import Document

logger = logging.getLogger(__name__)

@dataclass
class Chunk:
    """
    Represent a chunk of text split from a Document
    """
    text: str
    metadata: dict = field(default_factory=dict)
    chunk_index: int = 0


@dataclass
class ParentChildResult:
    """
    Kết quả chunking theo chiến lược Parent-Child.

    Luồng sử dụng trong RAG:
        1. Index vào Qdrant: CHỈ dùng ``children`` (nhỏ, embedding tập trung)
        2. Khi retrieve: Qdrant trả về child → dùng ``child.metadata["parent_id"]``
           để tra ``parent_store`` → lấy ``parent.text`` đưa vào LLM

    Attributes:
        parents:      Chunks lớn (toàn bộ nội dung 1 Điều) — context cho LLM.
        children:     Chunks nhỏ (đoạn trong Điều) — index để retrieval.
        parent_store: {parent_id → Chunk} để lookup O(1).
    """
    parents: list[Chunk]
    children: list[Chunk]
    parent_store: dict[str, Chunk]


# =============================================================================
# QUALITY CONTROL — Hàng rào kỹ thuật (Pydantic Schema + Validator)
# =============================================================================

class ChunkMetadataSchema(BaseModel):
    """
    Schema Pydantic kiểm tra chất lượng metadata của từng chunk luật pháp.
    Log ngay lập tức khi phát hiện thiếu metadata quan trọng.
    """

    luat: str = ""
    chuong: str = ""
    muc: str = ""
    dieu: str = ""
    chunk_index: int = 0
    source: str = ""

    @model_validator(mode="after")
    def warn_missing_legal_metadata(self) -> "ChunkMetadataSchema":
        """Log warning ngay khi phát hiện metadata luật bị rỗng."""
        if self.luat:  # Chỉ validate với tài liệu luật (đã xác định được tên luật)
            if not self.chuong:
                logger.warning(
                    "[QC] chunk #%d — 'chuong' rỗng | luat='%s' | source='%s'",
                    self.chunk_index,
                    self.luat,
                    self.source,
                )
            if not self.dieu:
                logger.warning(
                    "[QC] chunk #%d — 'dieu' rỗng | luat='%s' | source='%s'",
                    self.chunk_index,
                    self.luat,
                    self.source,
                )
        return self


class ChunkValidator:
    """
    Hàng rào kỹ thuật (Quality Control) cho chunking pipeline.

    - Mode mặc định (strict=False): log WARNING khi metadata thiếu, vẫn giữ chunk.
    - Strict mode (strict=True): REJECT chunk khi thiếu cả 'chuong' lẫn 'dieu'
      (dùng sau khi pipeline đã ổn định, tránh dữ liệu lỗi vào Vector DB).
    """

    def __init__(self, strict: bool = False):
        """
        Args:
            strict: True → Reject chunk khi thiếu metadata cấp Chương và Điều.
                    False (default) → Chỉ log warning, vẫn giữ chunk.
        """
        self.strict = strict
        self._rejected: int = 0
        self._total: int = 0

    @property
    def rejected_count(self) -> int:
        return self._rejected

    @property
    def total_count(self) -> int:
        return self._total

    def reset_stats(self) -> None:
        self._rejected = 0
        self._total = 0

    def is_valid(self, chunk: "Chunk") -> bool:
        """
        Validate 1 chunk qua Pydantic schema.
        Log ngay lập tức khi vi phạm. Trả về False nếu bị reject (strict mode).
        """
        self._total += 1
        meta = chunk.metadata

        # Chạy Pydantic validation — tự động log warning nếu thiếu trường
        ChunkMetadataSchema(
            luat=meta.get("luat", ""),
            chuong=meta.get("chuong", ""),
            muc=meta.get("muc", ""),
            dieu=meta.get("dieu", ""),
            chunk_index=meta.get("chunk_index", 0),
            source=meta.get("source", ""),
        )

        # Strict mode: reject nếu thiếu cả chuong lẫn dieu (khi đã biết tên luật)
        if self.strict and meta.get("luat"):
            if not meta.get("chuong") and not meta.get("dieu"):
                self._rejected += 1
                logger.error(
                    "[QC] REJECT chunk #%d | chuong='%s' | dieu='%s' "
                    "| preview='%.60s' | source='%s'",
                    meta.get("chunk_index", 0),
                    meta.get("chuong", ""),
                    meta.get("dieu", ""),
                    chunk.text[:60].replace("\n", " "),
                    meta.get("source", ""),
                )
                return False
        return True

    def filter(self, chunks: list["Chunk"]) -> list["Chunk"]:
        """
        Lọc và trả về các chunk hợp lệ. Log tổng kết sau khi filter.
        """
        valid = [c for c in chunks if self.is_valid(c)]
        if self._rejected > 0:
            logger.warning(
                "[QC] SUMMARY — Rejected %d/%d chunks do thiếu metadata.",
                self._rejected,
                self._total,
            )
        return valid

# Custom separators theo cấu trúc Markdown đã chuẩn hóa từ pre-processing
# Thứ tự ưu tiên giảm dần: heading lớn → heading nhỏ → paragraph → câu → từ
LEGAL_SEPARATORS = [
    "\n# ",            # PHẦN (h1)
    "\n## ",           # Chương (h2)
    "\n### ",          # Mục (h3)
    "\n#### ",         # Điều (h4)
    "\n\\d+\\. ",      # Khoản (1. 2. 3.)
    "\n[a-z]\\) ",     # Điểm (a) b) c))
    "\n\n",            # Paragraph break
    "\n",              # Xuống dòng
    "\\. ",            # Kết thúc câu
    " ",               # Khoảng trắng
    ""                 # Ký tự (fallback cuối cùng)
]


# =============================================================================
# CROSS-REFERENCE EXTRACTOR
# =============================================================================

class CrossReferenceExtractor:
    """
    Trích xuất tham chiếu chéo (cross-references) trong văn bản luật VN.

    Ví dụ các mẫu được nhận diện:
        "theo quy định tại Điều 15"          → ["Điều 15"]
        "căn cứ vào khoản 2 Điều 8"          → ["Điều 8"]
        "thực hiện theo Điều 20 của Luật này" → ["Điều 20"]
        "hướng dẫn tại Điều 12a"             → ["Điều 12a"]

    Metadata ``related_articles`` được gắn vào mỗi chunk, cho phép hệ thống
    gợi ý thêm các Điều liên quan ngay cả khi từ khóa không trùng khớp.
    """

    _PATTERNS = [
        # "quy định tại [khoản X] Điều Y"
        r"quy\s+định\s+tại\s+(?:khoản\s+\d+[a-z]?\s+)?(?:điều)\s+([\d]+[a-z]?)",
        # "căn cứ [vào] [khoản X] Điều Y"
        r"căn\s+cứ\s+(?:vào\s+)?(?:khoản\s+\d+[a-z]?\s+)?(?:điều)\s+([\d]+[a-z]?)",
        # "thực hiện theo [khoản X] Điều Y"
        r"thực\s+hiện\s+theo\s+(?:khoản\s+\d+[a-z]?\s+)?(?:điều)\s+([\d]+[a-z]?)",
        # "tại [khoản X] Điều Y của/Luật"
        r"tại\s+(?:khoản\s+\d+[a-z]?\s+)?(?:điều)\s+([\d]+[a-z]?)\s+(?:của|luật|này)",
        # "hướng dẫn tại Điều Y"
        r"hướng\s+dẫn\s+tại\s+(?:điều)\s+([\d]+[a-z]?)",
        # "theo Điều Y"
        r"theo\s+(?:điều)\s+([\d]+[a-z]?)",
    ]

    def extract(self, text: str) -> list[str]:
        """
        Tìm tất cả Điều được tham chiếu trong text.

        Returns:
            list[str]: Danh sách "Điều X" đã deduplicate và sort.
                       Ví dụ: ["Điều 12a", "Điều 15", "Điều 8"]
        """
        refs: set[str] = set()
        lower_text = text.lower()
        for pattern in self._PATTERNS:
            for match in re.finditer(pattern, lower_text):
                refs.add(f"Điều {match.group(1).strip()}")
        return sorted(refs, key=lambda x: (len(x), x))


# =============================================================================
# SELF-SUMMARY ENRICHER
# =============================================================================

class SelfSummaryEnricher:
    """
    Tạo tóm tắt 1 câu cho mỗi Điều luật — gắn vào đầu mỗi chunk.

    Tại sao cần?
        Embedding model ưu tiên trọng số phần đầu đoạn văn. Khi 1 child chunk
        chỉ chứa nội dung khoản/điểm nhỏ, không có heading, model khó hiểu
        context. Summary bù đắp thiếu hụt này.

    Hai chế độ:
        llm_fn=None  → Extractive: lấy câu đầu có nghĩa sau heading
        llm_fn=fn    → Abstractive: gọi LLM để tóm tắt

    Kết quả được prepend vào chunk text:
        "[Tóm tắt: Điều 15 quy định về tốc độ tối đa...]"
    """

    def __init__(self, llm_fn=None):
        """
        Args:
            llm_fn: callable(text: str) -> str
                    Hàm gọi LLM tóm tắt. None → extractive fallback.
        """
        self.llm_fn = llm_fn

    def summarize(self, text: str, dieu: str = "") -> str:
        """
        Tạo summary 1 câu cho đoạn text.

        Args:
            text: Nội dung cần tóm tắt (thường là toàn bộ 1 Điều).
            dieu: Tên Điều (vd: "Điều 15. Tốc độ") để làm prefix.

        Returns:
            str: Câu tóm tắt, rỗng nếu không extract được.
        """
        if self.llm_fn:
            try:
                return self.llm_fn(text)
            except Exception as exc:
                logger.warning(
                    "[SelfSummary] LLM failed: %s → extractive fallback", exc
                )
        return self._extractive_summary(text, dieu)

    def _extractive_summary(self, text: str, dieu: str = "") -> str:
        """Lấy câu đầu tiên có nghĩa sau heading làm summary."""
        lines = [
            ln.strip() for ln in text.split("\n")
            if ln.strip() and not ln.strip().startswith("#")
        ]
        if not lines:
            return ""

        first_line = lines[0][:200]
        # Bỏ bullet/đầu số: "1. ", "a) "
        first_line = re.sub(r"^\d+[\.。]\s*|^[a-z]\)\s*", "", first_line).strip()

        if dieu:
            return f"{dieu}: {first_line}"
        return first_line


# =============================================================================
# SEMANTIC CHUNK SPLITTER  (Token-aware + Semantic fallback)
# =============================================================================

class SemanticChunkSplitter:
    """
    Chia text thành các chunk không vượt quá ``max_tokens``.

    Logic 2 lớp:
        Layer 1 — Token guard:
            chunk ≤ max_tokens → giữ nguyên, không cắt thêm.

        Layer 2 — Semantic split (khi vượt max_tokens):
            a. Có ``embedder`` → tính cosine similarity giữa các câu.
               Cắt tại điểm similarity thấp nhất (ranh giới ngữ nghĩa thực sự),
               không cắt tùy tiện giữa câu có liên quan.
            b. Không có ``embedder`` → greedy sentence-packing fallback.

    Tại sao token-based thay vì char-based?
        AI hiểu tokens, không phải ký tự. "nghĩa vụ" = 2 tokens.
        Char limit 512 có thể = 100 tokens (quá ngắn) hoặc 200 tokens (quá dài)
        tùy nội dung. Token limit đảm bảo chunk luôn fit model embedding context.
    """

    def __init__(
        self,
        max_tokens: int = 256,
        tokenizer_encoding: str = "cl100k_base",
        similarity_threshold: float = 0.45,
        embedder=None,
    ):
        """
        Args:
            max_tokens: Số token tối đa mỗi chunk.
                        cl100k_base: ~4 chars/token với tiếng Việt.
            tokenizer_encoding: Tên tiktoken encoding.
                                ``cl100k_base`` = GPT-4 tokenizer, tương thích rộng.
            similarity_threshold: Ngưỡng cosine similarity để xác định cut point.
                Thấp hơn (0.3) → cắt ít, chunk lớn hơn.
                Cao hơn (0.6)  → cắt nhiều, chunk nhỏ hơn.
            embedder: callable(list[str]) -> list[list[float]]
                      Tương thích với ``EmbeddingService.embed_documents``.
                      None → sentence-packing fallback.
        """
        self.max_tokens = max_tokens
        self.similarity_threshold = similarity_threshold
        self.embedder = embedder
        self._tokenizer_encoding = tokenizer_encoding
        self._tokenizer = None  # lazy init

    @property
    def tokenizer(self):
        """Lazy init tiktoken. Trả về None nếu chưa cài (không crash)."""
        if self._tokenizer is None:
            try:
                import tiktoken
                self._tokenizer = tiktoken.get_encoding(self._tokenizer_encoding)
            except ImportError:
                logger.warning(
                    "[SemanticSplitter] tiktoken chưa cài → fallback char/4. "
                    "Cài: pip install tiktoken"
                )
                self._tokenizer = False  # sentinel: đã thử, không có
        return self._tokenizer if self._tokenizer is not False else None

    def count_tokens(self, text: str) -> int:
        """Đếm token chính xác (tiktoken) hoặc ước tính (char/4)."""
        tok = self.tokenizer
        if tok:
            return len(tok.encode(text))
        return max(1, len(text) // 4)

    def split(self, text: str) -> list[str]:
        """
        Chia text thành list[str], mỗi phần ≤ max_tokens.

        Returns:
            list[str] — luôn ít nhất 1 phần.
        """
        if self.count_tokens(text) <= self.max_tokens:
            return [text]

        if self.embedder:
            return self._semantic_split(text)
        return self._sentence_pack_split(text)

    # ------------------------------------------------------------------ #
    #  Internals                                                           #
    # ------------------------------------------------------------------ #

    def _semantic_split(self, text: str) -> list[str]:
        """
        Cắt tại ranh giới ngữ nghĩa thực sự bằng cosine similarity.

        Flow:
            1. Tách text → sentences
            2. Batch embed 1 lần (tiết kiệm API call)
            3. Tính cosine sim giữa câu liên tiếp → tìm valleys
            4. Cắt tại valleys → build chunks
            5. Recursively split chunk nào vẫn còn quá dài
        """
        try:
            import numpy as np
        except ImportError:
            logger.warning("[SemanticSplitter] numpy chưa cài → fallback sentence-pack")
            return self._sentence_pack_split(text)

        sentences = self._split_sentences(text)
        if len(sentences) <= 1:
            return [text]

        try:
            raw_embeddings = self.embedder(sentences)
            emb = np.array(raw_embeddings, dtype=np.float32)
        except Exception as exc:
            logger.warning("[SemanticSplitter] embedder failed: %s → fallback", exc)
            return self._sentence_pack_split(text)

        # Cosine similarity giữa câu i và câu i+1
        norms = np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8
        normed = emb / norms
        sims = (normed[:-1] * normed[1:]).sum(axis=1)  # shape (N-1,)

        # Cut points: nơi similarity xuống dưới ngưỡng
        cut_indices = [i + 1 for i, s in enumerate(sims.tolist())
                       if s < self.similarity_threshold]

        if not cut_indices:
            # Không có valley rõ → cắt tại điểm thấp nhất
            cut_indices = [int(np.argmin(sims)) + 1]

        chunks = self._build_chunks_from_cuts(sentences, cut_indices)

        # Recursively xử lý chunk vẫn còn quá dài
        result: list[str] = []
        for chunk in chunks:
            if self.count_tokens(chunk) > self.max_tokens:
                result.extend(self.split(chunk))
            else:
                result.append(chunk)
        return result

    def _sentence_pack_split(self, text: str) -> list[str]:
        """
        Greedy fallback: nhét câu vào chunk cho đến khi đầy token,
        sang chunk mới khi tràn.
        """
        sentences = self._split_sentences(text)
        chunks: list[str] = []
        current: list[str] = []
        current_tokens = 0

        for sent in sentences:
            sent_tokens = self.count_tokens(sent)
            if current and current_tokens + sent_tokens > self.max_tokens:
                chunks.append(" ".join(current))
                current = [sent]
                current_tokens = sent_tokens
            else:
                current.append(sent)
                current_tokens += sent_tokens

        if current:
            chunks.append(" ".join(current))
        return chunks or [text]

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        """
        Tách text thành câu, tối ưu cho tiếng Việt pháp lý.

        Cắt sau ". " khi ký tự tiếp theo là chữ hoa (bắt đầu câu mới).
        KHÔNG cắt: "khoản 1. Quy định", "10.000 đồng" — các số liên quan.
        """
        # Cắt tại điểm kết thúc câu trước chữ cái viết hoa
        _UPPER_VI = "A-ZÁÀẢÃẠĂẮẶẴẦẨÂẤẦẨẪẬĐÉÈẺẼẸÊẾỀỂỄỆÍÌỈĨỊÓÒỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÚÙỦŨỤƯỨỪỬỮỰÝỲỶỸỴ"
        parts = re.split(
            rf"(?<=[.!?])\s+(?=[{_UPPER_VI}])",
            text,
        )
        if len(parts) <= 1:
            parts = re.split(r"\.\s+", text)
        return [p.strip() for p in parts if p.strip()]

    @staticmethod
    def _build_chunks_from_cuts(sentences: list[str], cut_indices: list[int]) -> list[str]:
        """Ghép sentences thành chunks theo danh sách cut points."""
        chunks: list[str] = []
        prev = 0
        for cut in cut_indices:
            chunk = " ".join(sentences[prev:cut]).strip()
            if chunk:
                chunks.append(chunk)
            prev = cut
        tail = " ".join(sentences[prev:]).strip()
        if tail:
            chunks.append(tail)
        return chunks

class LegalMetadataExtractor:
    """
    Trích xuất metadata cấu trúc luật từ Markdown headings.
    Duy trì state để track context (Luật → Phần → Chương → Mục → Điều)
    xuyên suốt các chunks ("Kế thừa và Đổ đầy").

    Hierarchy reset rules (chặt chẽ):
        PHẦN mới  → reset Chương, Mục, Điều
        CHƯƠNG mới → reset Mục, Điều
        MỤC mới   → reset Điều
    """

    # Patterns match Markdown headings đã chuẩn hóa
    PATTERN_PHAN = re.compile(
        r"(?:^|\n)#\s+PHẦN\s+(THỨ\s+\w+|[IVXLCDM]+|\d+)[.,\s]*(.*?)(?:\n|$)",
        re.IGNORECASE,
    )
    PATTERN_LUAT = re.compile(
        r"(?:^|\n)#\s+(.+?)(?:\n|$)"
    )
    PATTERN_CHUONG = re.compile(
        r"(?:^|\n)##\s+Chương\s+([IVXLCDM]+|\d+)[.,\s]*(.*?)(?:\n|$)"
    )
    PATTERN_MUC = re.compile(
        r"(?:^|\n)###\s+Mục\s+(\d+)[.,\s]*(.*?)(?:\n|$)"
    )
    # Nâng cấp: hỗ trợ số có ký tự chữ (Điều 12a, Điều 12b)
    PATTERN_DIEU = re.compile(
        r"(?:^|\n)####\s+Điều\s+([\d]+[a-z]?)[.,\s]*(.*?)(?:\n|$)"
    )

    def __init__(self):
        self.current_phan: str = ""
        self.current_luat: str = ""
        self.current_chuong: str = ""
        self.current_muc: str = ""
        self.current_dieu: str = ""

    def reset(self):
        self.current_phan = ""
        self.current_luat = ""
        self.current_chuong = ""
        self.current_muc = ""
        self.current_dieu = ""

    def extract_from_text(self, text: str) -> dict:
        """
        Quét text của chunk và cập nhật context hiện tại (stateful).
        Reset cấp thấp hơn đúng theo hierarchy khi có cấp cao hơn mới.

        Returns:
            dict: metadata cấu trúc luật hiện tại
        """
        # PHẦN — cấp cao nhất: reset toàn bộ cấp thấp hơn
        match = self.PATTERN_PHAN.search(text)
        if match:
            num = match.group(1).strip()
            title = match.group(2).strip()
            self.current_phan = f"PHẦN {num}"
            if title:
                self.current_phan += f". {title}"
            self.current_chuong = ""
            self.current_muc = ""
            self.current_dieu = ""

        # Tên Luật (h1 không phải PHẦN) — chỉ set 1 lần duy nhất
        if not self.current_luat:
            match = self.PATTERN_LUAT.search(text)
            if match:
                candidate = match.group(1).strip()
                # Bỏ qua nếu là PHẦN heading (đã xử lý ở trên)
                if not re.match(r"^PHẦN\b", candidate, re.IGNORECASE):
                    self.current_luat = candidate

        # CHƯƠNG — reset Mục, Điều
        match = self.PATTERN_CHUONG.search(text)
        if match:
            num = match.group(1).strip()
            title = match.group(2).strip()
            self.current_chuong = f"Chương {num}"
            if title:
                self.current_chuong += f". {title}"
            self.current_muc = ""
            self.current_dieu = ""

        # MỤC — reset Điều
        match = self.PATTERN_MUC.search(text)
        if match:
            num = match.group(1).strip()
            title = match.group(2).strip()
            self.current_muc = f"Mục {num}"
            if title:
                self.current_muc += f". {title}"
            self.current_dieu = ""

        # ĐIỀU
        match = self.PATTERN_DIEU.search(text)
        if match:
            num = match.group(1).strip()
            title = match.group(2).strip()
            self.current_dieu = f"Điều {num}"
            if title:
                self.current_dieu += f". {title}"

        return {
            "phan": self.current_phan,
            "luat": self.current_luat,
            "chuong": self.current_chuong,
            "muc": self.current_muc,
            "dieu": self.current_dieu,
        }

    def build_context_prefix(self) -> str:
        """
        Tạo prefix string bắt buộc gắn vào đầu chunk.
        Embedding model ưu tiên trọng số các từ xuất hiện đầu đoạn văn.
        Điều này giúp retrieval chính xác hơn khi chunk không chứa heading.

        VD: [Luật Đường bộ - PHẦN I - Chương II - Mục 1 - Điều 12a. Tốc độ]
        """
        parts = []
        if self.current_luat:
            parts.append(self.current_luat)
        if self.current_phan:
            parts.append(self.current_phan)
        if self.current_chuong:
            parts.append(self.current_chuong)
        if self.current_muc:
            parts.append(self.current_muc)
        if self.current_dieu:
            parts.append(self.current_dieu)

        if parts:
            return "[" + " - ".join(parts) + "]\n"
        return ""


class TextChunker:
    """
    Chunking tài liệu luật Việt Nam sử dụng RecursiveCharacterTextSplitter
    với custom separators theo cấu trúc Markdown đã chuẩn hóa.

    Pre-processing đã chuyển:
        CHƯƠNG I    → ## Chương I
        Điều 8.     → #### Điều 8.

    Nên separators ở đây match Markdown headings, đảm bảo:
        - Ưu tiên cắt tại ranh giới Chương > Mục > Điều > Khoản > Điểm
        - Mỗi chunk được inject prefix: [Luật X - Chương Y - Điều Z]
    
    Tại sao cần chunking?
    - LLM có giới hạn context window
    - Embedding chất lượng hơn với đoạn text ngắn, tập trung 1 ý
    - Retrieval chính xác hơn khi chunk size phù hợp
    """

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        add_context_prefix: bool = True,
        validator: ChunkValidator = None,
    ):
        """
        Args:
                chunk_size: số ký tự tối đa cho mỗi chunk
            chunk_overlap: số ký tự overlap giữa các chunk liên tiếp
            add_context_prefix: có gắn [Luật - Chương - Điều] vào đầu chunk không
            validator: ChunkValidator instance để kiểm soát chất lượng chunk
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.add_context_prefix = add_context_prefix
        self.validator = validator or ChunkValidator(strict=False)

        self.splitter = RecursiveCharacterTextSplitter(
            separators=LEGAL_SEPARATORS,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            is_separator_regex=True,
            keep_separator="start",
            strip_whitespace=True,
        )

        self.metadata_extractor = LegalMetadataExtractor()

    def chunk_document(self, document: Document) -> list[Chunk]:
        """
        Chia document thành list[Chunk].

        Args:
            document: Document cần chia nhỏ (đã ở format Markdown)
        Returns:
            list[Chunk]
        """ 
        text = document.text
        if not text.strip():
            return []
        
        # Reset metadata extractor trước khi chunking document mới
        self.metadata_extractor.reset()

        # Nếu document metadata đã có tên luật, dùng luôn
        if "luat" in document.metadata and document.metadata["luat"]:
            self.metadata_extractor.current_luat = document.metadata["luat"]

        # Dùng langchain splitter
        raw_chunks = self.splitter.split_text(text)
        
        chunks = []
        for i, chunk_text in enumerate(raw_chunks):
            # Trích xuất metadata từ chunk text (cập nhật state kế thừa)
            legal_metadata = self.metadata_extractor.extract_from_text(chunk_text)

            # Context Injection: bắt buộc inject prefix vào đầu chunk
            # Embedding model ưu tiên trọng số những từ ở đầu đoạn văn
            context_prefix = ""
            if self.add_context_prefix:
                context_prefix = self.metadata_extractor.build_context_prefix()

            final_text = context_prefix + chunk_text if context_prefix else chunk_text

            chunks.append(Chunk(
                text=final_text,
                metadata={
                    **document.metadata,  # giữ nguyên metadata gốc
                    **legal_metadata,     # thêm metadata cấu trúc luật
                    "chunk_index": i,
                    "total_chunks": len(raw_chunks),
                    "context_prefix": context_prefix.strip(),
                },
                chunk_index=i,
            ))

        # QC: Validate và lọc chunks qua hàng rào Pydantic
        validated_chunks = self.validator.filter(chunks)
        return validated_chunks
    
    def chunk_documents(self, documents: list[Document]) -> list[Chunk]:
        """
        Chia nhiều document thành list[Chunk].

        Args:
            documents: list Document cần chia nhỏ
        Returns:
            list[Chunk]
        """ 
        all_chunks = []
        for doc in documents:
            chunks = self.chunk_document(doc)
            all_chunks.extend(chunks)

        rejected = self.validator.rejected_count
        qc_suffix = f" | [QC] {rejected} chunks bị reject" if rejected else ""
        logger.info(
            "Chunked %d documents \u2192 %d valid chunks%s",
            len(documents),
            len(all_chunks),
            qc_suffix,
        )
        return all_chunks


# =============================================================================
# PARENT-CHILD CHUNKER  (Chiến lược 2 cấp)
# =============================================================================

class ParentChildChunker:
    """
    Chiến lược chunking 2 cấp cho RAG chatbot pháp luật.

    Vấn đề cốt lõi — 2 mục tiêu xung đột:
        - Retrieval cần chunk NHỎ  → embedding tập trung, cosine sim cao → tìm trúng
        - Generation cần chunk LỚN → LLM đủ ngữ cảnh → trả lời đầy đủ

    Giải pháp Parent-Child:
    ┌──────────────────────────────────────────────────────────────┐
    │  Document → tách tại #### Điều → PARENT chunks              │
    │                       ↓                                     │
    │  Mỗi Parent → SemanticChunkSplitter → CHILD chunks          │
    │  child.metadata["parent_id"] → tra parent_store → full text │
    │                                                             │
    │  Qdrant index: CHỈ children (embedding nhỏ, chính xác)     │
    │  LLM context : parent.text  (đầy đủ nội dung Điều)         │
    └──────────────────────────────────────────────────────────────┘

    Enriched metadata (động):
        summary:          1 câu tóm tắt nội dung Điều (Self-Summary Context)
        related_articles: các Điều được tham chiếu trong nội dung (Cross-Reference)
    """

    def __init__(
        self,
        parent_max_tokens: int = 512,
        child_max_tokens: int = 128,
        tokenizer_encoding: str = "cl100k_base",
        similarity_threshold: float = 0.45,
        add_context_prefix: bool = True,
        add_self_summary: bool = True,
        summarizer_fn=None,
        embedder=None,
        validator: ChunkValidator = None,
    ):
        """
        Args:
            parent_max_tokens:    Token limit cho Parent. Mặc định 512.
                                  Điều dài → semantic split thêm.
            child_max_tokens:     Token limit cho Child. Mặc định 128.
                                  Phù hợp embedding model (text-embedding-3-small).
            tokenizer_encoding:   Tiktoken encoding. ``cl100k_base`` cho GPT-4.
            similarity_threshold: Ngưỡng cosine sim. 0.45 = cân bằng độ chính xác/kích thước.
            add_context_prefix:   Inject [Luật - Chương - Điều] vào đầu mỗi chunk.
            add_self_summary:     Prepend "[Tóm tắt: ...]" vào mỗi chunk.
            summarizer_fn:        callable(text: str) -> str
                                  None → extractive fallback (không cần GPU/API).
            embedder:             callable(list[str]) -> list[list[float]]
                                  Tương thích EmbeddingService.embed_documents.
                                  None → sentence-packing fallback (không cần API call).
            validator:            ChunkValidator để QC metadata.
        """
        self.add_context_prefix = add_context_prefix
        self.add_self_summary = add_self_summary

        # Sub-components
        self.metadata_extractor = LegalMetadataExtractor()
        self.cross_ref_extractor = CrossReferenceExtractor()
        self.summary_enricher = SelfSummaryEnricher(llm_fn=summarizer_fn)
        self.validator = validator or ChunkValidator(strict=False)

        # Parent splitter: đảm bảo parent ≤ 512 tokens (đủ cho LLM)
        self.parent_splitter = SemanticChunkSplitter(
            max_tokens=parent_max_tokens,
            tokenizer_encoding=tokenizer_encoding,
            similarity_threshold=similarity_threshold,
            embedder=embedder,
        )
        # Child splitter: đảm bảo child ≤ 128 tokens (tối ưu embedding)
        self.child_splitter = SemanticChunkSplitter(
            max_tokens=child_max_tokens,
            tokenizer_encoding=tokenizer_encoding,
            similarity_threshold=similarity_threshold,
            embedder=embedder,
        )

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def chunk_document(self, document: Document) -> ParentChildResult:
        """
        Chunk 1 document theo chiến lược parent-child.

        Args:
            document: Document đã qua preprocessing (Markdown format).

        Returns:
            ParentChildResult — parents, children, parent_store.
        """
        text = document.text
        if not text.strip():
            return ParentChildResult(parents=[], children=[], parent_store={})

        # Reset stateful metadata extractor cho document mới
        self.metadata_extractor.reset()
        if document.metadata.get("luat"):
            self.metadata_extractor.current_luat = document.metadata["luat"]

        # Step 1: tách tại ranh giới Điều → candidate parents
        parent_texts = self._split_into_parents(text)
        source = document.metadata.get("source", "doc")

        parents: list[Chunk] = []
        children: list[Chunk] = []
        parent_store: dict[str, Chunk] = {}

        for p_idx, parent_text in enumerate(parent_texts):
            if not parent_text.strip():
                continue

            # Trích xuất + kế thừa metadata cấu trúc luật (stateful)
            legal_meta = self.metadata_extractor.extract_from_text(parent_text)
            context_prefix = (
                self.metadata_extractor.build_context_prefix()
                if self.add_context_prefix else ""
            )

            # Cross-Reference: tìm các Điều được tham chiếu
            related_articles = self.cross_ref_extractor.extract(parent_text)

            # Self-Summary: 1 câu tóm tắt nội dung Điều
            summary = (
                self.summary_enricher.summarize(
                    parent_text, dieu=legal_meta.get("dieu", "")
                )
                if self.add_self_summary else ""
            )

            parent_id = f"{source}::p{p_idx}"
            parent_display = self._build_display_text(
                parent_text, context_prefix, summary
            )

            parent_chunk = Chunk(
                text=parent_display,
                metadata={
                    **document.metadata,
                    **legal_meta,
                    "chunk_type": "parent",
                    "chunk_id": parent_id,
                    "chunk_index": p_idx,
                    "context_prefix": context_prefix.strip(),
                    "summary": summary,
                    "related_articles": related_articles,
                },
                chunk_index=p_idx,
            )
            parents.append(parent_chunk)
            parent_store[parent_id] = parent_chunk

            # Step 2: tách parent → children (token-aware + semantic)
            child_texts = self.child_splitter.split(parent_text)

            for c_idx, child_text in enumerate(child_texts):
                if not child_text.strip():
                    continue
                child_id = f"{parent_id}::c{c_idx}"
                child_display = self._build_display_text(
                    child_text, context_prefix, summary
                )

                child_chunk = Chunk(
                    text=child_display,
                    metadata={
                        **document.metadata,
                        **legal_meta,
                        "chunk_type": "child",
                        "chunk_id": child_id,
                        "parent_id": parent_id,
                        "chunk_index": c_idx,
                        "context_prefix": context_prefix.strip(),
                        "summary": summary,
                        "related_articles": related_articles,
                    },
                    chunk_index=c_idx,
                )
                children.append(child_chunk)

        valid_children = self.validator.filter(children)
        valid_parents = self.validator.filter(parents)

        logger.info(
            "[ParentChild] %s → %d parents | %d children",
            document.metadata.get("filename", "?"),
            len(valid_parents),
            len(valid_children),
        )
        return ParentChildResult(
            parents=valid_parents,
            children=valid_children,
            parent_store=parent_store,
        )

    def chunk_documents(self, documents: list[Document]) -> ParentChildResult:
        """
        Chunk nhiều documents, merge kết quả.

        Args:
            documents: list[Document] đã qua preprocessing.

        Returns:
            ParentChildResult tổng hợp.
        """
        all_parents: list[Chunk] = []
        all_children: list[Chunk] = []
        all_parent_store: dict[str, Chunk] = {}

        for doc in documents:
            result = self.chunk_document(doc)
            all_parents.extend(result.parents)
            all_children.extend(result.children)
            all_parent_store.update(result.parent_store)

        rejected = self.validator.rejected_count
        qc_suffix = f" | [QC] {rejected} rejected" if rejected else ""
        logger.info(
            "[ParentChild] Total: %d docs → %d parents | %d children%s",
            len(documents), len(all_parents), len(all_children), qc_suffix,
        )
        return ParentChildResult(
            parents=all_parents,
            children=all_children,
            parent_store=all_parent_store,
        )

    # ------------------------------------------------------------------ #
    #  Internals                                                           #
    # ------------------------------------------------------------------ #

    def _split_into_parents(self, text: str) -> list[str]:
        """
        Tách document tại mỗi ``#### Điều`` heading → parent boundaries.

        Nếu 1 Điều vẫn vượt ``parent_max_tokens`` (Điều dài nhiều Khoản)
        → dùng ``parent_splitter`` (semantic) để cắt thêm.
        """
        # Prepend \n để regex luôn bắt Điều đầu tiên
        raw_parts = re.split(r"(?=\n####\s+Điều\s+)", "\n" + text)

        result: list[str] = []
        for part in raw_parts:
            part = part.strip()
            if not part:
                continue
            if self.parent_splitter.count_tokens(part) > self.parent_splitter.max_tokens:
                # Điều quá dài → semantic split để giữ parent trong giới hạn
                result.extend(self.parent_splitter.split(part))
            else:
                result.append(part)

        return result or [text]

    @staticmethod
    def _build_display_text(text: str, context_prefix: str, summary: str) -> str:
        """
        Ghép các phần thành display text cho chunk.

        Thứ tự: [Tóm tắt: ...] → [Context prefix] → nội dung thực
        Embedding model ưu tiên trọng số phần đầu → context/summary được
        encode với weight cao nhất.
        """
        parts: list[str] = []
        if summary:
            parts.append(f"[Tóm tắt: {summary}]")
        if context_prefix:
            parts.append(context_prefix.rstrip())
        parts.append(text)
        return "\n".join(parts)