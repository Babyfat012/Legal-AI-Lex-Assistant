import re
import os
from pathlib import Path
from enum import Enum

class ConverterBackend(str, Enum):
    MARKITDOWN = "markitdown"
    DOCLING = "docling"

class MarkdownConverter:
    """
    Pre-processing: Chuyển PDF/DOCX -> Markdown text
    Giữ nguyên cấu trúc file dưới dạng heading markdown    

    Flow:
        file.pdf / file.docx
            -> converter -> raw markdown
            -> chuanar hóa heading theo cấu trúc luật VN
    """

    SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".doc", ".txt", ".md"}

    def __init__(self, backend: ConverterBackend = ConverterBackend.MARKITDOWN):
        """
        Args:
            backend: Engine chuyển đổi ("markitdown" hoặc "docling")
        """
        self.backend = backend
    
    def convert_file(self, file_path: str, output_path: str = None) -> str:
        """
        Chuyển 1 file sang Markdown

        Args:
            file_path: đường dẫn file gốc (pdf/docx/txt/md)
            output_path: đường dẫn lưu file markdown (None thì không lưu, chỉ trả về string)

        Return:
            str: nội dung markdown đã chuyển đổi
        """

        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        ext = path.suffix.lower()
        if ext not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file type: {ext}. Only {self.SUPPORTED_EXTENSIONS} are supported."
            )
        
        if ext in {".txt", ".md"}:
            # Nếu đã là txt/md thì chỉ cần đọc nội dung
            markdown = path.read_text(encoding="utf-8")
        else:
            if self.backend == ConverterBackend.MARKITDOWN:
                raw_md = self._convert_with_markitdown(path)
            elif self.backend == ConverterBackend.DOCLING:
                raw_md = self._convert_with_docling(path)
            else:
                raise ValueError(f"Unsupported converter backend: {self.backend}")
            
        # Post-process markdown để chuẩn hóa heading theo cấu trúc luật VN
        processed_md = self._post_process_legal(raw_md)

        # Lưu file markdown nếu output_path được cung cấp
        if output_path:
            output = Path(output_path)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(processed_md, encoding="utf-8")
            print(f"Saved markdown: {output}")
        
        return processed_md
    

    def convert_directory(
            self, dir_path: str, output_dir: str = None
    ) -> dict[str, str]:
        """
        Chuyển tất cả file trong thư mục sang markdown

        Args:
            dir_path: đường dẫn thư mục chứa file gốc
            output_dir: đường dẫn thư mục lưu file markdown (None thì không lưu, chỉ trả về dict)

        Return:
            dict[str, str]:  Mapping {filename, markdown_content}
        """
        dir_path = Path(dir_path)
        if not dir_path.is_dir():
            raise NotADirectoryError(f"Not a directory: {dir_path}")
        
        results = {}

        for file_path in sorted(dir_path.rglob("*")):
            if file_path.suffix.lower() in self.SUPPORTED_EXTENSIONS:
                try:
                    output_path = None
                    if output_dir:
                        # Giữ nguyên tên file nhưng đổi extension thành .md
                        relative = file_path.relative_to(dir_path)
                        output_path = str(
                            Path(output_dir) / relative.with_suffix(".md")
                        )

                    md_content = self.convert_file(str(file_path), output_path)
                    results[file_path.name] = md_content
                    print(
                        f"Converted {file_path.name} -> {len(md_content)} chars"
                    )
                except Exception as e:
                    print(f"Error converting {file_path.name}: {e}")

        print(f"Total converted {len(results)} files from {dir_path}")
        return results
    

    def _convert_with_markitdown(self, path: Path) -> str:
        """
        Chuyển đổi file sử dụng backend markitdown
        """
        try:
            from markitdown import MarkItDown
        except ImportError:
            raise ImportError(
                "markitdown is not installed. Please install it with `pip install markitdown`"
            )
        
        md = MarkItDown()
        result = md.convert(str(path))
        return result.text_content
    
    def _convert_with_docling(self, path: Path) -> str:
        """
        Chuyển đổi file sử dụng backend docling
        """
        try:
            from docling.document_converter import DocumentConverter
        except ImportError:
            raise ImportError(
                "docling is not installed. Please install it with `pip install docling`"
            )
        
        dl = DocumentConverter()
        result = dl.convert(str(path))
        return result.document.export_to_markdown()
    

    # Post-process markdown để chuẩn hóa heading theo cấu trúc luật VN
    def _post_process_legal(self, text: str) -> str:
        """
        Chuẩn hóa văn bản luật VN sang Markdown headings.

        Mapping:
            PHẦN ...      → # PHẦN ...
            CHƯƠNG ...    → ## Chương ...
            MỤC ...       → ### Mục ...
            Điều ...      → #### Điều ...
            1. 2. 3. ...  → giữ nguyên (Khoản)
            a) b) c) ...  → giữ nguyên (Điểm)

        Chuẩn hóa thêm:
            - Xóa khoảng trắng thừa
            - Thống nhất format số La Mã / số thường cho Chương
        """
        lines = text.split("\n")
        processed_lines = []

        for line in lines:
            stripped = line.strip()
            if not stripped:
                processed_lines.append("")

            processed_line = self._normalize_legal_heading(stripped)
            processed_lines.append(processed_line)

        result = "\n".join(processed_lines)

        # Xóa khoảng trắng thừa (nếu có)
        result = re.sub(r"\n{3,}", "\n\n", result)

        return result.strip()
    
    def _normalize_legal_heading(self, line: str) -> str:
        """
        Nhận diện và chuyển đổi 1 dòng thành Markdown heading nếu nó là
        tiêu đề cấu trúc luật.
        """
        # PHẦN (cấp cao nhất, ít gặp)
        match = re.match(
            r"^(?:#+\s*)?PHẦN\s+(THỨ\s+\w+|[IVXLCDM]+|\d+)[\.:]?\s*[-–]?\s*(.*)",
            line,
            re.IGNORECASE,
        )
        if match:
            num = match.group(1).strip()
            title = match.group(2).strip()
            heading = f"# PHẦN {num}"
            if title:
                heading += f". {title}"
            return heading

        # CHƯƠNG
        match = re.match(
            r"^(?:#+\s*)?CHƯƠNG\s+([IVXLCDM]+|\d+)[\.:]?\s*[-–]?\s*(.*)",
            line,
            re.IGNORECASE,
        )
        if match:
            num = match.group(1).strip()
            title = match.group(2).strip()
            heading = f"## Chương {num}"
            if title:
                heading += f". {title}"
            return heading

        # MỤC
        match = re.match(
            r"^(?:#+\s*)?MỤC\s+(\d+)[\.:]?\s*(.*)", line, re.IGNORECASE
        )
        if match:
            num = match.group(1).strip()
            title = match.group(2).strip()
            heading = f"### Mục {num}"
            if title:
                heading += f". {title}"
            return heading

        # ĐIỀU
        match = re.match(
            r"^(?:#+\s*)?Điều\s+(\d+)[\.:]?\s*(.*)", line, re.IGNORECASE
        )
        if match:
            num = match.group(1).strip()
            title = match.group(2).strip()
            heading = f"#### Điều {num}"
            if title:
                heading += f". {title}"
            return heading

        return line