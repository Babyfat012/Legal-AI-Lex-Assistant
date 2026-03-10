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
        self._md_instance = None       # lazy init — khởi tạo 1 lần khi cần
        self._docling_instance = None  # lazy init — khởi tạo 1 lần khi cần
    
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
            raw_md = path.read_text(encoding="utf-8")
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
        Chuyển đổi file sử dụng backend markitdown.
        Instance được khởi tạo 1 lần duy nhất (lazy init) và tái sử dụng
        cho mọi lần gọi tiếp theo.
        """
        if self._md_instance is None:
            try:
                from markitdown import MarkItDown
            except ImportError:
                raise ImportError(
                    "markitdown is not installed. Please install it with `pip install markitdown`"
                )
            self._md_instance = MarkItDown()

        result = self._md_instance.convert(str(path))
        return result.text_content
    
    def _convert_with_docling(self, path: Path) -> str:
        """
        Chuyển đổi file sử dụng backend docling.
        Instance được khởi tạo 1 lần duy nhất (lazy init) và tái sử dụng
        cho mọi lần gọi tiếp theo.
        """
        if self._docling_instance is None:
            try:
                from docling.document_converter import DocumentConverter
            except ImportError:
                raise ImportError(
                    "docling is not installed. Please install it with `pip install docling`"
                )
            self._docling_instance = DocumentConverter()

        result = self._docling_instance.convert(str(path))
        return result.document.export_to_markdown()
    

    # -------------------------------------------------------------------------
    # Post-processing: chuẩn hóa heading theo cấu trúc luật VN
    # -------------------------------------------------------------------------

    def _clean_raw_text(self, text: str) -> str:
        """
        Làm sạch text thô trước khi xử lý Markdown heading.

        Xử lý các artifacts phổ biến khi extract từ PDF:

        1. **Null bytes** (``\x00`` / ``\u0000``): thay bằng space.
           Đây là vấn đề cực kỳ phổ biến khi ``pypdf`` extract PDF dùng
           CID/Type0 fonts (ví dụ nhiều văn bản luật VN scan bằng OCR).
           Null bytes phá vỡ hoàn toàn regex heading, khiến ``CHƯƠNG\x0010``
           không bao giờ được normalize thành ``## Chương 10``, dẫn đến
           metadata ``chuong``/``dieu`` trống và retrieval hoàn toàn thất bại.

        2. **Nhiều space liên tiếp** (không phải newline) → 1 space:
           Sau bước 1, có thể xuất hiện double-space do thay null bằng space
           cạnh space sẵn có.

        3. **Quá nhiều dòng trống liên tiếp** → tối đa 2 dòng trống.
        """
        # 1. Null bytes → space
        text = text.replace("\x00", " ")
        # 2. Multiple non-newline whitespace → single space
        text = re.sub(r"[^\S\n]+", " ", text)
        # 3. Cap consecutive blank lines at 2
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text

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

        Pipeline:
            0. Làm sạch null bytes và artifacts PDF (``_clean_raw_text``)
            1. Gộp tiêu đề bị ngắt 2 dòng (look-ahead buffer)
            2. Chuẩn hóa từng dòng thành Markdown heading
            3. Dọn dẹp khoảng trắng thừa
        """
        text = self._clean_raw_text(text)
        lines = text.split("\n")
        lines = self._merge_broken_headings(lines)

        processed_lines = []
        for line in lines:
            stripped = line.strip()
            if not stripped:
                processed_lines.append("")
                continue

            processed_line = self._normalize_legal_heading(stripped)
            processed_lines.append(processed_line)

        result = "\n".join(processed_lines)
        result = re.sub(r"\n{3,}", "\n\n", result)
        return result.strip()

    def _merge_broken_headings(self, lines: list[str]) -> list[str]:
        """
        Gộp tiêu đề bị ngắt quãng thành 2 dòng (do converter PDF/DOCX).

        VD input:
            ["#### Điều 15.", "Quy định về biển báo hiệu đường bộ"]
        VD output:
            ["#### Điều 15. Quy định về biển báo hiệu đường bộ"]

        Logic: Khi gặp một dòng là tiêu đề "partial" (không có phần title),
        peek dòng tiếp theo. Nếu dòng tiếp không phải đầu mục mới → gộp lại.
        """
        result = []
        i = 0
        while i < len(lines):
            current = lines[i].strip()
            if i + 1 < len(lines) and self._is_partial_heading(current):
                next_line = lines[i + 1].strip()
                # Gộp nếu dòng tiếp KHÔNG phải heading mới và KHÔNG rỗng
                if next_line and not self._is_heading_candidate(next_line):
                    sep = " " if current.endswith(".") else ". "
                    result.append(current + sep + next_line)
                    i += 2
                    continue
            result.append(lines[i])
            i += 1
        return result

    def _is_partial_heading(self, line: str) -> bool:
        """
        Kiểm tra dòng có phải là tiêu đề chưa có phần title không.
        VD: "#### Điều 15.", "CHƯƠNG I", "**Mục 2.**"
        """
        # Strip markdown decorators trước khi kiểm tra
        clean = re.sub(r"^[\s*_#]+|[\s*_#]+$", "", line).strip()
        patterns = [
            r"^(?:PHẦN|CHƯƠNG|MỤC)\s+(?:THỨ\s+\w+|[IVXLCDM]+|\d+)[\.:]?\s*$",
            r"^(?:Điều|ĐIỀU)\s+[\d]+[a-z]?[\.:\s]*$",
        ]
        return any(re.match(p, clean, re.IGNORECASE) for p in patterns)

    def _is_heading_candidate(self, line: str) -> bool:
        """
        Kiểm tra dòng có phải là bắt đầu một đầu mục mới không.
        VD: "CHƯƠNG II", "Điều 8", "**MỤC 3**"
        """
        clean = re.sub(r"^[\s*_#]+|[\s*_#]+$", "", line).strip()
        return bool(
            re.match(r"^(?:PHẦN|CHƯƠNG|MỤC|Điều|ĐIỀU)\s+", clean, re.IGNORECASE)
        )

    def _normalize_legal_heading(self, line: str) -> str:
        """
        Nhận diện và chuyển đổi 1 dòng thành Markdown heading nếu là
        tiêu đề cấu trúc luật.

        Nâng cấp so với phiên bản cũ:
            - Strip Markdown decorators (**..**, *..*, _.._) trước khi match
            - Hỗ trợ số có ký tự chữ: Điều 12a, Điều 12b (thực tế văn bản luật)
        """
        # Bước 1: Strip Markdown decorators bao quanh dòng (**..**, *..*, _.._)
        # Giữ nguyên ký tự # (heading marker) — đã được xử lý bởi (?:#+\s*)? trong regex
        clean = re.sub(r"^\s*[*_]+\s*", "", line)
        clean = re.sub(r"\s*[*_]+\s*$", "", clean).strip()

        # PHẦN (cấp cao nhất)
        match = re.match(
            r"^(?:#+\s*)?PHẦN\s+(THỨ\s+\w+|[IVXLCDM]+|\d+)[\.:]?\s*[-–]?\s*(.*)",
            clean,
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
            clean,
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
            r"^(?:#+\s*)?MỤC\s+(\d+)[\.:]?\s*(.*)", clean, re.IGNORECASE
        )
        if match:
            num = match.group(1).strip()
            title = match.group(2).strip()
            heading = f"### Mục {num}"
            if title:
                heading += f". {title}"
            return heading

        # ĐIỀU — hỗ trợ số có ký tự chữ: Điều 12a, Điều 12b
        match = re.match(
            r"^(?:#+\s*)?(?:Điều|ĐIỀU)\s+([\d]+[a-z]?)[\.:\s\-]*(.*)",
            clean,
            re.IGNORECASE,
        )
        if match:
            num = match.group(1).strip()
            title = match.group(2).strip()
            heading = f"#### Điều {num}"
            if title:
                heading += f". {title}"
            return heading

        return line