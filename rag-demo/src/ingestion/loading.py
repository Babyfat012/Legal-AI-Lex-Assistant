from pathlib import Path # for handling file paths
from typing import Optional 
from dataclasses import dataclass, field

@dataclass
class Document:
    """ Represent a loaded document with text content and metadata"""
    text: str
    metadata: dict = field(default_factory=dict)



class DocumentLoader:
    """
    Load documents từ file hoặc thư mục.
    Hỗ trợ: .txt, .md, .pdf, .docx

    Nếu file là PDF/DOCX, tự động chuyển sang Markdown trước (pre-processing).
    """

    SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf", ".docx", ".doc"}

    def __init__(self, converter_backend: str = "markitdown"):
        """
        Args:
            converter_backend: backend dùng để chuyển PDF/DOCX sang Markdown 
            (mặc định là "markitdown")
        """
        self.converter_backend = converter_backend
        self._converter = None # khởi tạo lazy converter, chỉ khi cần mới tạo instance

    @property
    def converter(self):
        """
        Lazy initialization của MarkdownConverter
        """
        if self._converter is None:
            from ingestion.preprocessing import MarkdownConverter, ConverterBackend
            backend = ConverterBackend(self.converter_backend)
            self._converter = MarkdownConverter(backend=backend)
        return self._converter
    

    def load_file(self, file_path: str) -> list[Document]:
        """
        Load 1 file, trả về list[Document].
        PDF/DOCX sẽ được chuyển sang Markdown trước.

        Args:
            file_path: Đường dẫn file cần load

        Returns:
            list[Document]
        """
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        ext = path.suffix.lower()
        if ext not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported file type: {ext}")
        
        # Plain text or markdown thì chỉ cần đọc nội dung
        if ext in {".txt", ".md"}:
            text = path.read_text(encoding="utf-8")
        # PDF hoặc DOCX thì cần convert sang Markdown trước
        elif ext in {".pdf", ".docx", ".doc"}:
            text = self.converter.convert_file(str(path))
        else:
            raise ValueError(f"Unsupported file type: {ext}")
        
        return [Document(
            text=text,
            metadata={
                "source": str(path),
                "filename": path.name,
                "file_type": ext,
                "format": "markdown", # sau convert thì đều là markdown
            }
        )]

        
    
    # def _load_pdf(self, path: Path) -> list[Document]:
    #     """ Load .pdf file, each page is a Document """
    #     try:
    #         from pypdf import PdfReader
    #     except ImportError:
    #         raise ImportError("pypdf library is required to load PDF files.`")
        
    #     reader = PdfReader(str(path))
    #     documents = []

    #     for page_num, page in enumerate(reader.pages):
    #         text = page.extract_text()
    #         if text and text.strip():
    #             documents.append(Document(
    #                 text=text.strip(),
    #                 metadata={
    #                     "source": str(path),
    #                     "filename": path.name,
    #                     "file_type": ".pdf",
    #                     "page": page_num,
    #                     "total_pages": len(reader.pages),
    #                 }
    #             ))
    #     return documents
    

    # def _load_txt(self, path: Path) -> list[Document]:
    #     """ Load .txt file, whole file is 1 Document """
    #     text = path.read_text(encoding="utf-8")
    #     return [Document(
    #         text=text,
    #         metadata={
    #             "source": str(path),
    #             "filename": path.name,
    #             "file_type": path.suffix,
    #         }
    #     )]
    
    def load_directory(
            self, dir_path: str, extensions: Optional[list[str]] = None
    ) -> list[Document]:
        """
        Load tất cả files trong thư mục (recursive).

        Args:
            dir_path: Đường dẫn thư mục
            extensions: Lọc theo extensions (default: tất cả supported)

        Returns:
            list[Document]
        """
        dir_path = Path(dir_path)
        if not dir_path.is_dir():
            raise NotADirectoryError(f"Not a directory: {dir_path}")
        
        allowed_exts = set(extensions) if extensions else self.SUPPORTED_EXTENSIONS
        documents = []

        for file_path in sorted(dir_path.rglob("*")):
            if file_path.suffix.lower() in allowed_exts:
                try:
                    docs = self.load_file(str(file_path))
                    documents.extend(docs)
                    print(f"Loaded {file_path.name} -> {len(docs)} document(s)")
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")
        
        print(f"Total loaded {len(documents)} documents from {dir_path}")
        return documents
    
