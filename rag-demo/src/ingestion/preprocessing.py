import re
import os
from pathlib import Path
from enum import Enum

class ConverterBackend(str, Enum):
    MARKITDOWN = "markitdown"
    DOCLING = "docling"

class MarkdownConverter:
    SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}

    def __init__(self, backend: ConverterBackend = ConverterBackend.MARKITDOWN):
        self.backend = backend
        self._md_instance = None       # lazy init
        self._docling_instance = None  # lazy init
    
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
        
        processed_md = self._post_process_legal(raw_md)

        if output_path:
            output = Path(output_path)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(processed_md, encoding="utf-8")
            print(f"Saved markdown: {output}")
        
        return processed_md

    def _convert_with_markitdown(self, path: Path) -> str:
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
    

    def _clean_raw_text(self, text: str) -> str:
        """
        Làm sạch text thô trước khi xử lý Markdown heading.
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