import os
import time
from openai import OpenAI
from core.logger import get_logger

logger = get_logger(__name__)

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
        