from pathlib import Path
from typing import Optional, Iterator
from dataclasses import dataclass, field
import gc
import logging

logger = logging.getLogger(__name__)


@dataclass
class Document:
    text: str
    metadata: dict = field(default_factory=dict)


class DocumentLoader:
    SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf", ".docx"}

    DEFAULT_LARGE_FILE_THRESHOLD_MB: float = 10.0
    DEFAULT_PAGES_PER_BATCH: int = 20               
    DEFAULT_BATCH_OVERLAP: int = 2  

    def __init__(
        self,
        converter_backend: str = "markitdown",
        large_file_threshold_mb: float = DEFAULT_LARGE_FILE_THRESHOLD_MB,
        pages_per_batch: int = DEFAULT_PAGES_PER_BATCH,
        batch_overlap: int = DEFAULT_BATCH_OVERLAP,
    ):
        """
        Args:
            converter_backend: backend dùng để chuyển PDF/DOCX sang Markdown
                (``"markitdown"`` hoặc ``"docling"``).
            large_file_threshold_mb: Ngưỡng kích thước (MB) để bật streaming
                mode. File PDF nhỏ hơn ngưỡng này vẫn dùng converter đầy đủ.
                Mặc định: 10.0 MB.
            pages_per_batch: Số trang xử lý mỗi lần trong streaming mode.
                Giảm giá trị này nếu RAM vẫn bị tràn. Mặc định: 20.
            batch_overlap: Số trang cuối của batch trước được carry-over sang
                batch tiếp theo để bổ sung ngữ cảnh liên tục. Mặc định: 2.
        """
        self.converter_backend = converter_backend
        self.large_file_threshold_mb = large_file_threshold_mb
        self.pages_per_batch = pages_per_batch
        self.batch_overlap = batch_overlap
        self._converter = None

    @property
    def converter(self):
        """Lazy initialization của MarkdownConverter."""
        if self._converter is None:
            from ingestion.preprocessing import MarkdownConverter, ConverterBackend
            backend = ConverterBackend(self.converter_backend)
            self._converter = MarkdownConverter(backend=backend)
        return self._converter

    def _extract_law_name(self, filename: str) -> str:
        """
        Trích xuất tên Luật từ tên file để dùng làm metadata ``luat``.

        Sử dụng lookup dict để map tên file sang tên luật chuẩn.
        Nếu không tìm thấy, giữ nguyên stem của file.

        Ví dụ:
            ``luat_dan_su_2015.pdf``  → ``"Bộ luật Dân sự"``
            ``bo_luat_hinh_su.txt``   → ``"Bộ luật Hình sự"``
        """
        LAW_NAME_MAP = {
            "bo_luat_dan_su_2015": "Bộ luật Dân sự 2015",
            "luat_dat_dai": "Luật Đất đai",
            "luat_nha_o": "Luật Nhà ở",
            "luat_honnhan_giadinh": "Luật Hôn nhân và Gia đình",
            "luat_xu_ly_vphc": "Luật Xử lý vi phạm hành chính",
            "luat_36_2024_qh15": "Luật số 36/2024/QH15",
            "luat_35_2024_qh15": "Luật số 35/2024/QH15",
            "nghi_dinh_168": "Nghị định 168",
            "p1_vb_hop_nhat_blhs_2024": "Văn bản hợp nhất BLHS 2024",
        }

        stem = Path(filename).stem  # Bỏ extension
        base_name = stem.rsplit("_", 1)[0]  # Loại bỏ phần năm nếu có
        return LAW_NAME_MAP.get(base_name.lower(), stem.replace("_", " ").replace("-", " "))

    def load_file(self, file_path: str) -> list[Document]:
        """
        Load 1 file, trả về list[Document].

        Args:
            file_path: Đường dẫn file cần load.

        Returns:
            list[Document]
        """
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        ext = path.suffix.lower()
        if ext not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported file type: {ext}")

        if ext in {".txt", ".md"}:
            text = path.read_text(encoding="utf-8")
            return [Document(
                text=text,
                metadata={
                    "source": str(path),
                    "luat": self._extract_law_name(path.name),
                    "filename": path.name,
                    "file_type": ext,
                    "format": "markdown",
                },
            )]

        if ext == ".pdf":
            size_mb = path.stat().st_size / (1024 * 1024)
            if size_mb >= self.large_file_threshold_mb:
                logger.info(
                    "[Loader] Large PDF: %s (%.1f MB ≥ %.1f MB) → streaming mode "
                    "(%d pages/batch)",
                    path.name, size_mb, self.large_file_threshold_mb,
                    self.pages_per_batch,
                )
                return list(self._load_pdf_streaming(path))

            logger.info(
                "[Loader] Small PDF: %s (%.1f MB) → normal mode", path.name, size_mb
            )

        text = self.converter.convert_file(str(path))
        return [Document(
            text=text,
            metadata={
                "source": str(path),
                "luat": self._extract_law_name(path.name),
                "filename": path.name,
                "file_type": ext,
                "format": "markdown",
            },
        )]

    def load_directory(
        self, dir_path: str, extensions: Optional[list[str]] = None
    ) -> list[Document]:
        """
        Load tất cả files trong thư mục (recursive).

        Args:
            dir_path: Đường dẫn thư mục.
            extensions: Lọc theo extensions (default: tất cả supported).

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
                    logger.info("Loaded %s → %d document(s)", file_path.name, len(docs))
                except Exception as e:
                    logger.error("Error loading %s: %s", file_path, e)

        logger.info("Total loaded %d documents from %s", len(documents), dir_path)
        return documents

    # ------------------------------------------------------------------
    # Streaming internals
    # ------------------------------------------------------------------

    def _load_pdf_streaming(self, path: Path) -> Iterator[Document]:
        try:
            import pypdf
        except ImportError:
            raise ImportError(
                "pypdf is required for large PDF streaming. "
                "Install with: pip install pypdf"
            )

        size_mb = path.stat().st_size / (1024 * 1024)
        reader = pypdf.PdfReader(str(path))
        total_pages = len(reader.pages)
        batch_count = 0
        yielded_count = 0
        carry_over_text: str = ""           # Batch Overlap: text carry-over từ batch trước

        logger.info(
            "[Loader][Stream] %s | %.1f MB | %d pages | batch_size=%d",
            path.name, size_mb, total_pages, self.pages_per_batch,
        )

        for batch_start in range(0, total_pages, self.pages_per_batch):
            batch_end = min(batch_start + self.pages_per_batch, total_pages)

            page_texts: list[str] = []
            for page_idx in range(batch_start, batch_end):
                page_text = reader.pages[page_idx].extract_text() or ""
                page_texts.append(page_text)

            new_text = "\n\n".join(page_texts)
            has_carry = bool(carry_over_text)
            raw_text = (carry_over_text + "\n\n" + new_text) if carry_over_text else new_text

            # Cập nhật carry-over cho batch kế tiếp (giữ N trang cuối)
            carry_over_text = (
                "\n\n".join(page_texts[-self.batch_overlap :])
                if self.batch_overlap > 0 and page_texts
                else ""
            )

            if not raw_text.strip():
                logger.debug(
                    "[Loader][Stream] Batch %d (pages %d–%d): empty, skipping",
                    batch_count, batch_start + 1, batch_end,
                )
                batch_count += 1
                continue

            processed_text = self.converter._post_process_legal(raw_text)
            yield Document(
                text=processed_text,
                metadata={
                    "source": str(path),
                    "filename": path.name,
                    "file_type": ".pdf",
                    "format": "markdown",
                    "page_start": batch_start + 1,
                    "page_end": batch_end,
                    "total_pages": total_pages,
                    "batch_index": batch_count,
                    "streaming_mode": True,
                    "luat": self._extract_law_name(path.name),
                },
            )

            logger.info(
                "[Loader][Stream] Batch %d | pages %d–%d / %d | %d chars",
                batch_count, batch_start + 1, batch_end, total_pages, len(processed_text),
            )

            yielded_count += 1
            batch_count += 1

            # Giải phóng bộ nhớ ngay sau khi yield
            page_texts.clear()
            del raw_text, new_text, processed_text
            gc.collect()

        logger.info(
            "[Loader][Stream] Done: %s | %d pages → %d batches (%d non-empty)",
            path.name, total_pages, batch_count, yielded_count,
        )

