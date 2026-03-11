import re
import logging
from dataclasses import dataclass, field
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
    """Kết quả chunking theo chiến lược Parent-Child."""

    parents: list[Chunk]
    children: list[Chunk]


class SemanticChunkSplitter:
    """Semantic/token-aware splitter dùng cho parent và child chunks."""

    def __init__(
        self,
        max_tokens: int = 256,
        tokenizer_encoding: str = "cl100k_base",
        similarity_threshold: float = 0.45,
        embedder=None,
    ):
        self.max_tokens = max_tokens
        self.similarity_threshold = similarity_threshold
        self.embedder = embedder
        self._tokenizer_encoding = tokenizer_encoding
        self._tokenizer = None

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            try:
                import tiktoken
                self._tokenizer = tiktoken.get_encoding(self._tokenizer_encoding)
            except ImportError:
                logger.warning(
                    "[SemanticSplitter] tiktoken chưa cài → fallback char/4"
                )
                self._tokenizer = False
        return self._tokenizer if self._tokenizer is not False else None

    def count_tokens(self, text: str) -> int:
        tok = self.tokenizer
        if tok:
            return len(tok.encode(text))
        return max(1, len(text) // 4)

    def split(self, text: str) -> list[str]:
        if self.count_tokens(text) <= self.max_tokens:
            return [text]
        chunks = self._semantic_split(text) if self.embedder else self._sentence_pack_split(text)
        result: list[str] = []
        for chunk in chunks:
            if self.count_tokens(chunk) > self.max_tokens:
                result.extend(self._force_split_long_text(chunk))
            else:
                result.append(chunk)
        return result or [text]

    def _semantic_split(self, text: str) -> list[str]:
        try:
            import numpy as np
        except ImportError:
            logger.warning("[SemanticSplitter] numpy chưa cài → fallback sentence-pack")
            return self._sentence_pack_split(text)

        sentences = self._split_sentences(text)
        if len(sentences) <= 1:
            return self._sentence_pack_split(text)

        try:
            emb = np.array(self.embedder(sentences), dtype=np.float32)
            if emb.ndim != 2 or emb.shape[0] < 2:
                return self._sentence_pack_split(text)
        except Exception as exc:
            logger.warning("[SemanticSplitter] embedder failed: %s → fallback", exc)
            return self._sentence_pack_split(text)

        norms = np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8
        normed = emb / norms
        sims = (normed[:-1] * normed[1:]).sum(axis=1)

        cut_indices = [i + 1 for i, s in enumerate(sims.tolist()) if s < self.similarity_threshold]
        if not cut_indices:
            cut_indices = [int(np.argmin(sims)) + 1]

        chunks = self._build_chunks_from_cuts(sentences, cut_indices)
        return chunks or [text]

    def _sentence_pack_split(self, text: str) -> list[str]:
        sentences = self._split_sentences(text)
        chunks: list[str] = []
        current: list[str] = []
        current_tokens = 0

        for sent in sentences:
            sent_tokens = self.count_tokens(sent)
            if sent_tokens > self.max_tokens:
                if current:
                    chunks.append(" ".join(current).strip())
                    current = []
                    current_tokens = 0
                chunks.extend(self._force_split_long_text(sent))
                continue

            if current and current_tokens + sent_tokens > self.max_tokens:
                chunks.append(" ".join(current).strip())
                current = [sent]
                current_tokens = sent_tokens
            else:
                current.append(sent)
                current_tokens += sent_tokens

        if current:
            chunks.append(" ".join(current).strip())
        return [c for c in chunks if c] or [text]

    def _force_split_long_text(self, text: str) -> list[str]:
        if self.count_tokens(text) <= self.max_tokens:
            return [text]
        step = max(80, int(self.max_tokens * 3.5))
        parts: list[str] = []
        i = 0
        while i < len(text):
            j = min(len(text), i + step)
            piece = text[i:j].strip()
            while piece and self.count_tokens(piece) > self.max_tokens and len(piece) > 1:
                piece = piece[: int(len(piece) * 0.8)].strip()
            if piece:
                parts.append(piece)
                i += len(piece)
            else:
                i = j
        return parts or [text]

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        _UPPER_VI = (
            "A-ZÁÀẢÃẠĂẮẶẴẦẨÂẤẦẨẪẬĐÉÈẺẼẸÊẾỀỂỄỆ"
            "ÍÌỈĨỊÓÒỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÚÙỦŨỤƯỨỪỬỮỰÝỲỶỸỴ"
        )
        parts = re.split(rf"(?<=[.!?])\s+(?=[{_UPPER_VI}])", text)
        if len(parts) <= 1:
            parts = re.split(r"\.\s+", text)
        return [p.strip() for p in parts if p.strip()]

    @staticmethod
    def _build_chunks_from_cuts(sentences: list[str], cut_indices: list[int]) -> list[str]:
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
    """Trích xuất metadata cấu trúc luật từ Markdown headings (stateful)."""

    PATTERN_PHAN = re.compile(
        r"(?:^|\n)#\s+PHẦN\s+(THỨ\s+\w+|[IVXLCDM]+|\d+)[.,\s]*(.*?)(?:\n|$)",
        re.IGNORECASE,
    )
    PATTERN_LUAT = re.compile(r"(?:^|\n)#\s+(.+?)(?:\n|$)")
    PATTERN_CHUONG = re.compile(
        r"(?:^|\n)##\s+Chương\s+([IVXLCDM]+|\d+)[.,\s]*(.*?)(?:\n|$)"
    )
    PATTERN_MUC = re.compile(r"(?:^|\n)###\s+Mục\s+(\d+)[.,\s]*(.*?)(?:\n|$)")
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
        match = self.PATTERN_PHAN.search(text)
        if match:
            num = match.group(1).strip()
            title = match.group(2).strip()
            self.current_phan = f"PHẦN {num}" + (f". {title}" if title else "")
            self.current_chuong = ""
            self.current_muc = ""
            self.current_dieu = ""

        if not self.current_luat:
            match = self.PATTERN_LUAT.search(text)
            if match:
                candidate = match.group(1).strip()
                if not re.match(r"^PHẦN\b", candidate, re.IGNORECASE):
                    self.current_luat = candidate

        match = self.PATTERN_CHUONG.search(text)
        if match:
            num = match.group(1).strip()
            title = match.group(2).strip()
            self.current_chuong = f"Chương {num}" + (f". {title}" if title else "")
            self.current_muc = ""
            self.current_dieu = ""

        match = self.PATTERN_MUC.search(text)
        if match:
            num = match.group(1).strip()
            title = match.group(2).strip()
            self.current_muc = f"Mục {num}" + (f". {title}" if title else "")
            self.current_dieu = ""

        match = self.PATTERN_DIEU.search(text)
        if match:
            num = match.group(1).strip()
            title = match.group(2).strip()
            self.current_dieu = f"Điều {num}" + (f". {title}" if title else "")

        return {
            "phan": self.current_phan,
            "luat": self.current_luat,
            "chuong": self.current_chuong,
            "muc": self.current_muc,
            "dieu": self.current_dieu,
        }

    def build_context_prefix(self) -> str:
        parts: list[str] = []
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
        return "[" + " - ".join(parts) + "]\n" if parts else ""


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
        embedder=None,
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
            embedder:             callable(list[str]) -> list[list[float]]
                                  Tương thích EmbeddingService.embed_documents.
                                  None → sentence-packing fallback (không cần API call).
        """
        self.add_context_prefix = add_context_prefix

        # Sub-components
        self.metadata_extractor = LegalMetadataExtractor()

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

        for p_idx, parent_text in enumerate(parent_texts):
            if not parent_text.strip():
                continue

            # Trích xuất + kế thừa metadata cấu trúc luật (stateful)
            legal_meta = self.metadata_extractor.extract_from_text(parent_text)
            context_prefix = (
                self.metadata_extractor.build_context_prefix()
                if self.add_context_prefix else ""
            )

            parent_id = f"{source}::p{p_idx}"
            parent_display = self._build_display_text(
                parent_text, context_prefix
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
                },
                chunk_index=p_idx,
            )
            parents.append(parent_chunk)

            # Step 2: tách parent → children (token-aware + semantic)
            child_texts = self.child_splitter.split(parent_text)

            for c_idx, child_text in enumerate(child_texts):
                if not child_text.strip():
                    continue
                child_id = f"{parent_id}::c{c_idx}"
                child_display = self._build_display_text(
                    child_text, context_prefix
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
                        "parent_content": parent_display,
                    },
                    chunk_index=c_idx,
                )
                children.append(child_chunk)

        logger.info(
            "[ParentChild] %s → %d parents | %d children",
            document.metadata.get("filename", "?"),
            len(parents),
            len(children),
        )
        return ParentChildResult(
            parents=parents,
            children=children,
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

        for doc in documents:
            result = self.chunk_document(doc)
            all_parents.extend(result.parents)
            all_children.extend(result.children)

        logger.info(
            "[ParentChild] Total: %d docs → %d parents | %d children",
            len(documents), len(all_parents), len(all_children),
        )
        return ParentChildResult(
            parents=all_parents,
            children=all_children,
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
    def _build_display_text(text: str, context_prefix: str, summary: str = "") -> str:
        """
        Ghép các phần thành display text cho chunk.

        Thứ tự: [Tóm tắt: ...] (optional) → [Context prefix] → nội dung thực
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