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