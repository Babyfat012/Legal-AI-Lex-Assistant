import os
import json
import time
from openai import OpenAI
from core.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Standard RAG prompts
# ---------------------------------------------------------------------------

def _load_system_prompt() -> str:
    """Load system prompt from rag-demo/prompt_lex_system.md if available.

    Falls back to a minimal built-in prompt when the file is missing or unreadable.
    """
    base = os.path.dirname(__file__)
    prompt_path = os.path.normpath(os.path.join(base, '..', '..', 'prompt_lex_system.md'))
    try:
        with open(prompt_path, 'r', encoding='utf-8') as fh:
            content = fh.read()
        logger.info("Loaded system prompt from %s", prompt_path)
        return content
    except Exception as exc:
        logger.warning("Failed to load system prompt from %s, using fallback prompt: %s", prompt_path, exc)
        # Fallback minimal prompt
        return (
            "Bạn là Lex — trợ lý pháp lý AI chuyên về luật Việt Nam.\n\n"
            "Nguyên tắc trả lời:\n"
            "1. Chỉ trả lời dựa trên các đoạn văn bản luật được cung cấp trong phần CONTEXT.\n"
            "2. Nếu context không đủ thông tin, hãy nói rõ: \"Tôi không tìm thấy thông tin liên quan trong các văn bản pháp luật được cung cấp.\"\n"
            "3. Trích dẫn cụ thể Điều/Khoản/Điểm khi trả lời.\n"
            "4. Dùng ngôn ngữ rõ ràng, dễ hiểu. Tránh thuật ngữ kỹ thuật không cần thiết.\n"
            "5. Không tự suy diễn hoặc thêm thông tin ngoài context."
        )

SYSTEM_PROMPT = _load_system_prompt()

QUERY_PROMPT_TEMPLATE = """
Dưới đây là các Điều luật liên quan để trả lời câu hỏi:

{context}
---
CÂU HỎI: {query}

Lưu ý: Chỉ trả lời dựa trên các Điều luật trên. Nếu không tìm thấy thông tin, hãy nói rõ: "Tôi không tìm thấy thông tin liên quan trong các văn bản pháp luật được cung cấp.\"
"""

# Two-stage pipeline templates
CONDENSE_SYSTEM = """
Bạn là một chuyên gia phân tích truy vấn pháp lý. 
Nhiệm vụ của bạn là tạo ra một câu truy vấn tìm kiếm độc lập (Standalone Query).
QUY TẮC:
1. Nếu người dùng dùng đại từ thay thế (ví dụ: 'hành vi này', 'việc đó', 'tái phạm'), bạn phải thay bằng thực thể pháp lý cụ thể từ lịch sử (ví dụ: 'vi phạm nồng độ cồn xe máy').
2. KHÔNG được thay đổi lĩnh vực pháp lý. Nếu đang nói về Giao thông, câu truy vấn phải chứa từ khóa Giao thông.
3. Tuyệt đối không tự trả lời câu hỏi, chỉ được viết lại câu hỏi.
4. Trả về DUY NHẤT 1 câu hỏi standalone, không giải thích, không thêm thông tin ngoài lịch sử.
"""

CONDENSE_USER = """
Chat History:
{chat_history}

Follow-up Input: {question}

Standalone Question:
"""

PIPELINE_PROMPT_TEMPLATE = """
Dưới đây là các Điều luật liên quan để trả lời câu hỏi:

{context}
---
CHAT HISTORY:
{chat_history}
---
CÂU HỎI (đã làm rõ): {standalone_question}

Lưu ý: Chỉ trả lời dựa trên các Điều luật trên. Nếu không tìm thấy thông tin, hãy nói rõ: "Tôi không tìm thấy thông tin liên quan trong các văn bản pháp luật được cung cấp."
"""

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

    Hỗ trợ 2 model riêng biệt theo routing:
        simple_model    — gpt-4o-mini: nhanh, rẻ, đủ cho tra cứu thông tin.
        reasoning_model — gpt-4o (hoặc o3-mini): mạnh hơn, dùng cho tình
                          huống tranh chấp / Legal Syllogism CoT.
    """

    def __init__(
        self,
        simple_model: str = "gpt-4o-mini",
        reasoning_model: str = "gpt-4o-mini",
        api_key: str = None,
        temperature: float = 0.1,
        max_tokens: int = 1500,
    ):
        """
        Args:
            simple_model:    Model cho câu hỏi đơn giản (tra cứu, định nghĩa).
                             Mặc định: "gpt-4o-mini" (nhanh, rẻ).
            reasoning_model: Model cho câu hỏi phức tạp (tình huống, tranh chấp).
                             Mặc định: "gpt-4o-mini". Nâng lên "gpt-4o" hoặc
                             "o3-mini" để tăng chất lượng phân tích pháp lý.
            api_key:         OpenAI API key. None = đọc từ OPENAI_API_KEY.
            temperature:     Độ sáng tạo (thấp cho legal, mặc định 0.1).
            max_tokens:      Token tối đa cho câu trả lời.
        """
        self.simple_model = simple_model
        self.reasoning_model = reasoning_model
        self.model = simple_model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        logger.info(
            "LLMGenerator initialized | simple_model=%s | reasoning_model=%s"
            " | max_tokens=%d | temperature=%.1f",
            simple_model, reasoning_model, max_tokens, temperature,
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
        logger.info(
            "Generating answer | model=%s | context_chunks=%d | query: %.80s",
            self.simple_model, len(context_chunks), query,
        )
        t0 = time.perf_counter()

        # Format context từ các chunks
        context = self._format_context(context_chunks)

        # Tạo prompt hoàn chỉnh
        user_prompt = QUERY_PROMPT_TEMPLATE.format(
            context=context,
            query=query,
        )

        response = self.client.chat.completions.create(
            model=self.simple_model,
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

    def condense_question(self, chat_history: str, question: str) -> str:
        """Rewrite follow-up `question` into a standalone question using minimal context.

        Returns a single-line standalone question. On failure, returns the original `question`.
        """
        try:
            # Prepare / truncate chat history to avoid sending excessively long context
            prepared_history = self.prepare_chat_history(chat_history)
            user_prompt = CONDENSE_USER.format(chat_history=prepared_history or "", question=question)
            resp = self.client.chat.completions.create(
                model=self.reasoning_model,  # Use reasoning_model for better accuracy
                messages=[
                    {"role": "system", "content": CONDENSE_SYSTEM},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.0,
                max_tokens=128,
            )
            
            standalone = resp.choices[0].message.content.strip()
            # Keep single-line, trimmed
            standalone = " ".join(standalone.split())
            logger.info("Condensed question: %s", standalone)
            return standalone or question
        except Exception as exc:
            logger.warning("Condense question failed, returning original: %s", exc)
            return question

    def prepare_chat_history(self, chat_history, max_turns: int = 3, max_chars: int = 1200) -> str:
        """Prepare chat history for prompts.

        - Accepts `chat_history` as a string or list of strings (turns).
        - Keeps only the last `max_turns` turns, joining with newlines.
        - If resulting string exceeds `max_chars`, attempt to summarize it using LLM.
        - Adjusted to use max_turns=7 and consider token count for truncation.
        """
        if not chat_history:
            return ""

        # Normalize to list of turns
        if isinstance(chat_history, list):
            turns = chat_history[-7:]  # Increased max_turns to 7
            hist = "\n".join(turns)
        elif isinstance(chat_history, str):
            # split heuristically by newline or sentence
            lines = [ln.strip() for ln in chat_history.splitlines() if ln.strip()]
            hist = "\n".join(lines[-7:]) if len(lines) > 7 else "\n".join(lines)
        else:
            hist = str(chat_history)

        # Check token count instead of fixed max_chars
        token_count = len(hist.split())
        if token_count <= 1200:
            return hist

        # Summarize long history to keep important legal entities
        try:
            summary = self.summarize_chat_history(hist)
            logger.info("Chat history summarized (tokens %d -> %d)", token_count, len(summary.split()))
            return summary
        except Exception as exc:
            logger.warning("Failed to summarize chat history, truncating: %s", exc)
            return hist[-max_chars:]

    def summarize_chat_history(self, chat_history: str) -> str:
        """Use LLM to produce a short summary of chat history, preserving legal entities.

        Returns a 1-3 sentence summary suitable for including in prompts.
        """
        SUMMARIZE_SYSTEM = (
            "Bạn là một trợ lý tóm tắt chuyên nghiệp. Nhiệm vụ: tóm tắt ngắn gọn lịch sử hội thoại, "
            "giữ lại các thực thể pháp lý quan trọng (tên luật, tội danh, điều khoản, con số, thời gian, đối tượng)."
        )

        SUMMARIZE_USER = """
Chat History:
{chat_history}

Tóm tắt ngắn (1-3 câu), nhấn mạnh chủ đề pháp lý và các thực thể quan trọng:
"""

        resp = self.client.chat.completions.create(
            model=self.simple_model,
            messages=[
                {"role": "system", "content": SUMMARIZE_SYSTEM},
                {"role": "user", "content": SUMMARIZE_USER.format(chat_history=chat_history)},
            ],
            temperature=0.0,
            max_tokens=128,
        )
        summary = resp.choices[0].message.content.strip()
        # One-line normalize
        summary = " ".join(summary.split())
        return summary

    def generate_pipeline(self, question: str, chat_history: str, retriever, context_chunks: list[dict] = None) -> str:
        """Two-stage pipeline: condense question -> generate Lex answer.

        Args:
            question: latest user input (may be follow-up)
            chat_history: short chat history or summary (string)
            retriever: Retriever instance to fetch context
            context_chunks: (optional) pre-fetched context chunks (default None)

        Returns:
            str: assistant response generated by Lex
        """
        # Stage 1: condense to standalone question
        standalone = self.condense_question(chat_history, question)

        # Fetch new documents using the condensed question
        if context_chunks is None:
            context_chunks = retriever.retrieve(standalone)

        # Stage 2: prepare pipeline prompt and generate
        logger.info("Generating pipeline answer | model=%s | chunks=%d | question: %.80s", self.simple_model, len(context_chunks), standalone)
        t0 = time.perf_counter()
        context = self._format_context(context_chunks)
        prepared_history = self.prepare_chat_history(chat_history)
        user_prompt = PIPELINE_PROMPT_TEMPLATE.format(
            context=context,
            chat_history=(prepared_history or ""),
            standalone_question=standalone,
        )

        response = self.client.chat.completions.create(
            model=self.simple_model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        usage = getattr(response, 'usage', None)
        if usage:
            try:
                logger.info(
                    "Pipeline generation done in %.2fs | tokens: prompt=%d completion=%d total=%d",
                    time.perf_counter() - t0,
                    usage.prompt_tokens,
                    usage.completion_tokens,
                    usage.total_tokens,
                )
            except Exception:
                pass
        return response.choices[0].message.content.strip()
    
    def _format_context(self, chunks: list[dict]) -> str:
        """
        Format danh sách chunks thành chuỗi context cho LLM.

        Ưu tiên ``parent_content`` (toàn bộ Điều luật) thay vì child chunk
        để LLM có đủ ngữ cảnh cho generation chính xác.
        Mỗi Điều được đánh số và gắn header [Luật - Chương - Điều] để LLM
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

            # Header cho mỗi Điều luật
            header_parts = [p for p in [luat, chuong, dieu] if p]
            header = " - ".join(header_parts) if header_parts else f"Văn bản {i}"

            # Ưu tiên parent_content (đủ Điều luật) để LLM có đủ ngữ cảnh
            content = chunk.get("parent_content") or chunk.get("text", "")

            parts.append(f"[Văn bản {i}: {header}]\nNội dung: {content}")

        return "\n---\n".join(parts)

    # ------------------------------------------------------------------
    # Legal Reasoning (Chain-of-Thought / Legal Syllogism)
    # ------------------------------------------------------------------

    def generate_with_reasoning(
        self,
        query: str,
        context_chunks: list[dict],
        chat_history: str = "",
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
        # Ensure follow-up questions are rewritten into standalone form first
        standalone = self.condense_question(chat_history, query)
        logger.info(
            "Generating reasoning answer | model=%s | chunks=%d | query: %.80s",
            self.reasoning_model, len(context_chunks), standalone,
        )
        t0 = time.perf_counter()
        context = self._format_context(context_chunks)
        user_prompt = _REASONING_USER.format(context=context, query=standalone)

        try:
            response = self.client.chat.completions.create(
                model=self.reasoning_model,
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