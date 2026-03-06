import re
from dataclasses import dataclass, field
from langchain_text_splitters import RecursiveCharacterTextSplitter
from ingestion.loading import Document

@dataclass
class Chunk:
    """
    Represent a chunk of text split from a Document
    """
    text: str
    metadata: dict = field(default_factory=dict)
    chunk_index: int = 0

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
    Duy trì state để track context (Luật → Chương → Mục → Điều) xuyên suốt các chunks.

    Vì pre-processing đã chuẩn hóa:
        ## Chương I. Quy định chung
        #### Điều 3. Giải thích từ ngữ
    → Chỉ cần match Markdown heading syntax.
    """
    # Patterns match Markdown headings đã chuẩn hóa
    PATTERN_LUAT = re.compile(
        r"(?:^|\n)#\s+(.+?)(?:\n|$)"
    )
    PATTERN_CHUONG = re.compile(
        r"(?:^|\n)##\s+Chương\s+([IVXLCDM]+|\d+)[\.\s]*(.*?)(?:\n|$)"
    )
    PATTERN_MUC = re.compile(
        r"(?:^|\n)###\s+Mục\s+(\d+)[\.\s]*(.*?)(?:\n|$)"
    )
    PATTERN_DIEU = re.compile(
        r"(?:^|\n)####\s+Điều\s+(\d+)[\.\s]*(.*?)(?:\n|$)"
    )

    def __init__(self):
        self.current_luat: str = ""
        self.current_chuong: str = ""
        self.current_muc: str = ""
        self.current_dieu: str = ""

    def reset(self):
        self.current_luat = ""
        self.current_chuong = ""
        self.current_muc = ""
        self.current_dieu = ""

    def extract_from_text(self, text: str) -> dict:
        """
        Quét text của chunk và cập nhật context hiện tại.

        Returns:
            dict: metadata cấu trúc luật hiện tại
        """
        # Tìm tên Luật (heading h1)
        match = self.PATTERN_LUAT.search(text)
        if match:
            self.current_luat = match.group(1).strip()

        # Tìm Chương (heading h2)
        match = self.PATTERN_CHUONG.search(text)
        if match:
            num = match.group(1).strip()
            title = match.group(2).strip()
            self.current_chuong = f"Chương {num}"
            if title:
                self.current_chuong += f". {title}"
            # Reset cấp thấp hơn khi có cấp cao hơn mới
            self.current_muc = ""
            self.current_dieu = ""

        # Tìm Mục (heading h3)
        match = self.PATTERN_MUC.search(text)
        if match:
            num = match.group(1).strip()
            title = match.group(2).strip()
            self.current_muc = f"Mục {num}"
            if title:
                self.current_muc += f". {title}"
            # Reset cấp thấp hơn khi có cấp cao hơn mới
            self.current_dieu = ""

        # Tìm Điều (heading h4)
        match = self.PATTERN_DIEU.search(text)
        if match:
            num = match.group(1).strip()
            title = match.group(2).strip()
            self.current_dieu = f"Điều {num}"
            if title:
                self.current_dieu += f". {title}"
        
        return {
            "luat": self.current_luat,
            "chuong": self.current_chuong,
            "muc": self.current_muc,
            "dieu": self.current_dieu,
        }
    
    def build_context_prefix(self) -> str:
        """
        Tạo prefix string gắn vào đầu chunk.
        Ví dụ: [Luật Đường bộ - Chương II - Điều 8. Đường nối]
        """
        parts = []
        if self.current_luat:
            parts.append(self.current_luat)
        if self.current_chuong:
            parts.append(self.current_chuong)
        if self.current_dieu:
            parts.append(self.current_dieu)

        if parts:
            return "[" + " - ".join(parts) + "]\n"
        else:
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
    ):
        """
        Args:
                chunk_size: số ký tự tối đa cho mỗi chunk
            chunk_overlap: số ký tự overlap giữa các chunk liên tiếp
            add_context_prefix: có gắn [Luật - Chương - Điều] vào đầu chunk không
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.add_context_prefix = add_context_prefix

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
            # Trích xuất metadata từ chunk text
            legal_metadata = self.metadata_extractor.extract_from_text(chunk_text)

            # Thêm prefix context
            context_prefix = ""
            if self.add_context_prefix:
                context_prefix = self.metadata_extractor.build_context_prefix()

            # Gán prefix vào đầu chunk
            final_text = context_prefix + chunk_text if context_prefix else chunk_text

            chunks.append(Chunk(
                text=final_text,
                metadata={
                    **document.metadata,  # giữ nguyên metadata gốc
                    **legal_metadata,     # thêm metadata cấu trúc luật
                    "chunk_index": i,      # đánh số thứ tự chunk trong document
                    "total_chunks": len(raw_chunks),
                    "context_prefix": context_prefix.strip(),
                },
                chunk_index=i,
            ))
        
        return chunks
    
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

        print(f"Chunked {len(documents)} documents into {len(all_chunks)} chunks")
        return all_chunks
    
    # def _split_text(self, text: str) -> list[str]:
    #     """
    #     Core logic để chia text thành các đoạn nhỏ hơn với overlap.
    #     """
    #     # Tách theo seperator
    #     paragraphs = text.split(self.separator)

    #     chunks = []
    #     current_chunk = ""

    #     for paragraph in paragraphs:
    #         paragraph = paragraph.strip()
    #         if not paragraph:
    #             continue

    #         # Nếu thêm paragraph vào mà vẫn <= chunk_size, thì thêm vào
    #         if len(current_chunk) + len(paragraph) + 1 <= self.chunk_size:
    #             current_chunk = (current_chunk + "\n" + paragraph).strip()
    #         else:
    #             # Lưu current chunk nếu nó không rỗng
    #             if current_chunk:
    #                 chunks.append(current_chunk)
            
    #         if len(paragraph) > self.chunk_size:
    #             # Nếu paragraph dài hơn chunk_size, thì cắt nó ra thành nhiều chunk nhỏ hơn
    #             sub_chunks = self._split_long_text(paragraph)
    #             chunks.extend(sub_chunks)
    #             current_chunk = ""
    #         else:
    #             current_chunk = paragraph

    #     # Thêm chunk cuối cùng
    #     if current_chunk:
    #         chunks.append(current_chunk)

    #     # Áp dụng overlap
    #     if self.chunk_overlap > 0 and len(chunks) > 1:
    #         chunks = self._apply_overlap(chunks)

    #     return chunks
    
    # def _split_long_text(self, text: str) -> list[str]:
    #     """
    #     Chia text dài hơn chunk size thành các phần nhỏ
    #     """
    #     chunks = []
    #     start = 0
    #     while start < len(text):
    #         end = start + self.chunk_size
    #         chunk = text[start:end]
    #         chunks.append(chunk.strip())
    #         start = end - self.chunk_overlap
    #     return chunks
    
    # def _apply_overlap(self, chunks: list[str]) -> list[str]:
    #     """
    #     Thêm overlap text từ chunk trước vào đầu chunk sau
    #     """
    #     overlapped = [chunks[0]]

    #     for i in range(1, len(chunks)):
    #         prev_chunk = chunks[i-1]
    #         overlap_text = prev_chunk[-self.chunk_overlap:]

    #         # Tìm vị trí ngắt từ gần nhất để không cắt giữa từ
    #         space_idx = overlap_text.find(" ")
    #         if space_idx != -1:
    #             overlap_text = overlap_text[space_idx + 1:]

    #         overlapped.append(overlap_text + " " + chunks[i])

    #     return overlapped        