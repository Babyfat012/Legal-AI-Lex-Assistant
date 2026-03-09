import os
import json
import time
from openai import OpenAI
from core.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Standard RAG prompts
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """
Bạn là Lex — trợ lý pháp lý AI chuyên về luật Việt Nam.

Nguyên tắc trả lời:
1. Chỉ trả lời dựa trên các đoạn văn bản luật được cung cấp trong phần CONTEXT.
2. Nếu context không đủ thông tin, hãy nói rõ: "Tôi không tìm thấy thông tin liên quan trong các văn bản pháp luật được cung cấp."
3. Trích dẫn cụ thể Điều/Khoản/Điểm khi trả lời.
4. Dùng ngôn ngữ rõ ràng, dễ hiểu. Tránh thuật ngữ kỹ thuật không cần thiết.
5. Không tự suy diễn hoặc thêm thông tin ngoài context.
"""

QUERY_PROMPT_TEMPLATE = """CONTEXT (các đoạn văn bản pháp luật liên quan):
{context}

---

CÂU HỎI: {query}

Hãy trả lời câu hỏi dựa trên các đoạn văn bản pháp luật ở trên."""

# ---------------------------------------------------------------------------
# Legal Reasoning prompts (Chain-of-Thought / Legal Syllogism)
# ---------------------------------------------------------------------------

_REASONING_SYSTEM = """\
Bạn là Lex — trợ lý pháp lý AI chuyên về luật Việt Nam.
Hãy áp dụng Tam đoạn luận pháp lý (Legal Syllogism) để phân tích tình huống.

Nguyên tắc bắt buộc:
1. CHỈ dùng thông tin từ CONTEXT — tuyệt đối không tự suy diễn ngoài context.
2. Nếu context thiếu căn cứ, ghi rõ "Không đủ căn cứ pháp lý để kết luận về [điểm này]".
3. Trích dẫn chính xác Điều/Khoản/Điểm và tên văn bản khi đối chiếu.

Bắt buộc trả về JSON hợp lệ với đúng 4 trường sau (không thêm trường khác):
{
  "hanh_vi": "Mô tả chi tiết hành vi của từng bên liên quan trong tình huống",
  "quy_dinh": [
    "Điều X, Khoản Y — Tên văn bản: trích dẫn hoặc tóm tắt nội dung",
    "..."
  ],
  "doi_chieu": "Đối chiếu từng hành vi với từng quy định, chỉ rõ điểm vi phạm cụ thể",
  "ket_luan": "Kết luận pháp lý: ai vi phạm điều gì, mức xử phạt nếu có căn cứ"
}
"""

_REASONING_USER = """\
CONTEXT (văn bản pháp luật liên quan):
{context}

---

TÌNH HUỐNG CẦN PHÂN TÍCH: {query}

Phân tích theo Tam đoạn luận pháp lý. Trả về JSON."""

class LLMGenerator:
    """
    Generator: Sinh câu trả lời từ query + retrieved chunks.

    Dùng OpenAI GPT làm LLM với system prompt định nghĩa vai trò Lex.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: str = None,
        temperature: float = 0.1,
        max_tokens: int = 1500,
    ):
        """Args:
            model: Tên model OpenAI (ví dụ "gpt-4o-mini")
            api_key: Khóa API OpenAI (nếu None sẽ dùng biến môi trường OPENAI_API_KEY)
            temperature: Độ sáng tạo của câu trả lời (thường để thấp cho legal)
            max_tokens: Số token tối đa cho câu trả lời
        """
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        logger.info(
            "LLMGenerator initialized | model=%s | max_tokens=%d | temperature=%.1f",
            model, max_tokens, temperature,
        )

    def generate(self, query: str, context_chunks: list[dict]) -> str:
        """
        Sinh câu trả lời từ query và danh sách chunks context.

        Args:
            query: Câu hỏi từ người dùng
            context_chunks: List chunks từ retriever
                            [{"text": ..., "score": ..., "metadata": ...}]

        Returns:
            str: Câu trả lời từ LLM
        """
        logger.info("Generating answer | context_chunks=%d | query: %.80s", len(context_chunks), query)
        t0 = time.perf_counter()

        # Format context từ các chunks
        context = self._format_context(context_chunks)

        # Tạo prompt hoàn chỉnh
        user_prompt = QUERY_PROMPT_TEMPLATE.format(
            context=context,
            query=query,
        )

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        usage = response.usage
        logger.info(
            "Generation done in %.2fs | tokens: prompt=%d completion=%d total=%d",
            time.perf_counter() - t0,
            usage.prompt_tokens,
            usage.completion_tokens,
            usage.total_tokens,
        )
        return response.choices[0].message.content.strip()
    
    def _format_context(self, chunks: list[dict]) -> str:
        """
        Format danh sách chunks thành chuỗi context cho LLM.

        Mỗi chunk được format với metadata (Luật, Chương, Điều) để LLM
        có thể trích dẫn chính xác nguồn.
        """
        if not chunks:
            return "Không có thông tin liên quan"
        
        parts = []
        for i, chunk in enumerate(chunks, 1):
            metadata = chunk.get("metadata", {})
            luat = metadata.get("luat", "")
            chuong = metadata.get("chuong", "")
            dieu = metadata.get("dieu", "")

            # Header cho mỗi chunk
            header_parts = [p for p in [luat, chuong, dieu] if p]
            header = " | ".join(header_parts) if header_parts else f"Đoạn {i}"

            parts.append(f"[{i}] {header}\n{chunk.get('text', '')}")

        return "\n\n---\n\n".join(parts)

    # ------------------------------------------------------------------
    # Legal Reasoning (Chain-of-Thought / Legal Syllogism)
    # ------------------------------------------------------------------

    def generate_with_reasoning(
        self,
        query: str,
        context_chunks: list[dict],
    ) -> tuple[str, dict]:
        """
        Sinh phân tích pháp lý theo Tam đoạn luận (Chain-of-Thought).

        Yêu cầu LLM trả về JSON có cấu trúc 4 bước:
            1. hanh_vi   — Mô tả hành vi các bên
            2. quy_dinh  — Các điều luật áp dụng (list)
            3. doi_chieu — Đối chiếu hành vi ↔ quy định
            4. ket_luan  — Kết luận pháp lý

        Nếu LLM không trả về JSON hợp lệ → fallback sang ``generate()``
        tiêu chuẩn và trả về reasoning_steps rỗng.

        Args:
            query:          Câu hỏi / tình huống cần phân tích.
            context_chunks: Danh sách chunks đã retrieve.

        Returns:
            tuple(answer_str, reasoning_dict)
                - answer_str:    Markdown string đầy đủ để hiển thị.
                - reasoning_dict: Dict cấu trúc CoT để frontend parse.
        """
        logger.info(
            "Generating reasoning answer | chunks=%d | query: %.80s",
            len(context_chunks), query,
        )
        t0 = time.perf_counter()
        context = self._format_context(context_chunks)
        user_prompt = _REASONING_USER.format(context=context, query=query)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": _REASONING_SYSTEM},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.1,
                max_tokens=2000,
                response_format={"type": "json_object"},
            )
            usage = response.usage
            logger.info(
                "Reasoning done in %.2fs | tokens: prompt=%d completion=%d total=%d",
                time.perf_counter() - t0,
                usage.prompt_tokens,
                usage.completion_tokens,
                usage.total_tokens,
            )

            raw = response.choices[0].message.content.strip()
            reasoning = json.loads(raw)

            # Validate required keys
            required = {"hanh_vi", "quy_dinh", "doi_chieu", "ket_luan"}
            if not required.issubset(reasoning.keys()):
                raise ValueError(f"Missing keys: {required - set(reasoning.keys())}")

            answer_str = self._format_reasoning_answer(reasoning)
            return answer_str, reasoning

        except Exception as exc:
            logger.warning(
                "Reasoning generation failed: %s — falling back to standard generate()",
                exc,
            )
            # Fallback: standard generation, empty reasoning dict
            answer_str = self.generate(query, context_chunks)
            return answer_str, {}

    def _format_reasoning_answer(self, reasoning: dict) -> str:
        """
        Format structured reasoning dict thành markdown string.

        Output mẫu:
            **Hành vi các bên:**
            ...

            **Quy định áp dụng:**
            - Điều X...
            - Điều Y...

            **Đối chiếu:**
            ...

            **Kết luận:**
            ...
        """
        hanh_vi = reasoning.get("hanh_vi", "").strip()
        quy_dinh: list = reasoning.get("quy_dinh", [])
        doi_chieu = reasoning.get("doi_chieu", "").strip()
        ket_luan = reasoning.get("ket_luan", "").strip()

        # Format quy_dinh list thành bullet points
        if isinstance(quy_dinh, list):
            quy_dinh_md = "\n".join(f"- {item}" for item in quy_dinh if item)
        else:
            quy_dinh_md = str(quy_dinh)

        parts = []
        if hanh_vi:
            parts.append(f"**Hành vi các bên:**\n{hanh_vi}")
        if quy_dinh_md:
            parts.append(f"**Quy định áp dụng:**\n{quy_dinh_md}")
        if doi_chieu:
            parts.append(f"**Đối chiếu:**\n{doi_chieu}")
        if ket_luan:
            parts.append(f"**Kết luận:**\n{ket_luan}")

        return "\n\n".join(parts) if parts else "Không đủ thông tin để phân tích."