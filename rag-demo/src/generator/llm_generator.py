import os
import json
import time
from openai import OpenAI
from core.logger import get_logger

logger = get_logger(__name__)


def _load_system_prompt() -> str:
    base = os.path.dirname(__file__)
    prompt_path = os.path.normpath(
        os.path.join(base, "..", "..", "prompt_lex_system.md")
    )
    try:
        with open(prompt_path, "r", encoding="utf-8") as fh:
            content = fh.read()
        logger.info("Loaded system prompt from %s", prompt_path)
        return content
    except Exception as exc:
        logger.warning(
            "Failed to load system prompt from %s, using fallback prompt: %s",
            prompt_path,
            exc,
        )
        return (
            "Bạn là Lex — trợ lý pháp lý AI chuyên về luật Việt Nam.\n\n"
            "Nguyên tắc trả lời:\n"
            "1. Chỉ trả lời dựa trên các đoạn văn bản luật được cung cấp trong phần CONTEXT.\n"
            '2. Nếu context không đủ thông tin, hãy nói rõ: "Tôi không tìm thấy thông tin liên quan trong các văn bản pháp luật được cung cấp."\n'
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

Lưu ý: Chỉ trả lời dựa trên các Điều luật trên. Nếu không tìm thấy thông tin, hãy nói rõ: "Tôi không tìm thấy thông tin liên quan trong các văn bản pháp luật được cung cấp.\""""

# ---------------------------------------------------------------------------
# PATCH 1/2: CONDENSE_SYSTEM
#
# Vấn đề cũ: CONDENSE_USER có dòng "Nếu câu hỏi đã đủ ngữ cảnh, trả về
# nguyên văn" → condenser bỏ qua toàn bộ history vì câu hỏi trông đủ ngữ
# pháp nhưng thiếu thực thể pháp lý (tuổi người bán, giao dịch vô hiệu...).
#
# Fix: Khi có history, LUÔN inject thực thể pháp lý vào câu condensed.
# Chỉ trả nguyên văn khi history thực sự rỗng.
# ---------------------------------------------------------------------------
CONDENSE_SYSTEM = """
Bạn là chuyên gia phân tích truy vấn pháp lý.
Nhiệm vụ: Viết lại câu follow-up thành một Standalone Query đầy đủ ngữ cảnh pháp lý.

QUY TẮC:
1. Khi có lịch sử hội thoại, LUÔN viết lại câu hỏi — KHÔNG trả về nguyên văn.
2. Bắt buộc đưa vào câu viết lại tất cả thực thể pháp lý quan trọng từ lịch sử:
   tuổi các bên, loại giao dịch, tình trạng pháp lý đã xác định (vô hiệu/hợp lệ),
   tài sản liên quan, v.v.
3. Giữ nguyên lĩnh vực pháp lý (Giao thông, Hình sự, Dân sự...).
4. Chỉ viết lại câu hỏi, KHÔNG trả lời, KHÔNG giải thích.
5. Trả về đúng 1 câu duy nhất.
6. TUYỆT ĐỐI KHÔNG thêm tên nghị định, số luật, số điều khoản
   nếu người dùng chưa đề cập trong câu hỏi của họ.
   Ví dụ sai: "...theo Nghị định 100/2019..."
   Ví dụ đúng: "...khi lái xe máy có nồng độ cồn vượt mức..."
7. Nếu KHÔNG có lịch sử hội thoại (rỗng), trả về nguyên văn câu hỏi gốc.

VÍ DỤ:
---
# Case 1: Follow-up về hậu quả — inject loại giao dịch và tuổi từ history
Chat History:
User: Em trai tôi 16 tuổi bán xe máy 20 triệu, gia đình không biết. Giao dịch có hợp pháp không?
Assistant: Giao dịch vô hiệu vì người 16 tuổi chưa có năng lực hành vi dân sự đầy đủ.
Follow-up: Nếu xe đã được sang tên rồi thì gia đình tôi có thể lấy lại xe không?
Standalone Question: Nếu xe máy đã sang tên sau khi người 16 tuổi bán xe (giao dịch dân sự vô hiệu), gia đình người bán có thể yêu cầu lấy lại xe không?

# Case 2: Follow-up có tham chiếu mờ — inject thực thể từ history
Chat History:
User: Mức phạt uống rượu lái xe máy là bao nhiêu?
Assistant: Nồng độ cồn vượt 80mg/100ml máu bị phạt 6-8 triệu đồng và tước GPLX 22-24 tháng.
Follow-up: Tái phạm thì sao?
Standalone Question: Mức phạt tái phạm hành vi lái xe máy có nồng độ cồn vượt 80mg/100ml máu là bao nhiêu?

# Case 3: Follow-up mở rộng điều kiện — inject tội danh từ history
Chat History:
User: Tội cướp tài sản bị xử lý thế nào?
Assistant: Tội cướp tài sản có khung phạt từ 3-10 năm tù.
Follow-up: Nếu có vũ khí thì khung hình phạt thay đổi không?
Standalone Question: Khung hình phạt tội cướp tài sản có sử dụng vũ khí là bao nhiêu?

# Case 4: Không có lịch sử — trả nguyên văn
Chat History: (trống)
Follow-up: Mức phạt cao nhất đối với người điều khiển xe máy có nồng độ cồn là bao nhiêu?
Standalone Question: Mức phạt cao nhất đối với người điều khiển xe máy có nồng độ cồn là bao nhiêu?
---
"""

# ---------------------------------------------------------------------------
# PATCH 2/2: CONDENSE_USER
#
# Xóa dòng "Nếu câu hỏi đã đủ ngữ cảnh, hãy trả về nguyên văn" — đây là
# dòng khiến condenser bỏ qua history ngay cả khi history có đầy đủ nội dung.
# ---------------------------------------------------------------------------
CONDENSE_USER = """
Chat History:
{chat_history}
Follow-up Input: {question}

Lưu ý: Nếu Chat History có nội dung, LUÔN viết lại câu hỏi với đầy đủ thực thể pháp lý từ history.
Chỉ trả về nguyên văn khi Chat History thực sự rỗng.
Standalone Question:"""


PIPELINE_PROMPT_TEMPLATE = """
Bạn là trợ lý pháp lý Lex. Nhiệm vụ duy nhất: trả lời câu hỏi dựa trên các Điều luật được cung cấp.

[CÁC ĐIỀU LUẬT LIÊN QUAN]
{context}

[LỊCH SỬ HỘI THOẠI - chỉ dùng để hiểu ngữ cảnh, KHÔNG dùng làm nguồn pháp lý]
{chat_history}

[CÂU HỎI CẦN TRẢ LỜI]
{standalone_question}

Yêu cầu trả lời:
1. Chỉ trích dẫn thông tin từ [CÁC ĐIỀU LUẬT LIÊN QUAN] ở trên, không dùng kiến thức nội tại.
2. Khi trích dẫn, ghi rõ tên văn bản và điều khoản theo đúng tên trong context.
   Ví dụ: "Theo Điều 6 Nghị định X, ..."
3. Nếu câu hỏi liên quan đến lượt trước (tái phạm, so sánh...),
   kết nối rõ ràng với điều luật hiện có — không dùng câu trả lời cũ làm nguồn.
4. Trả lời ngắn gọn, đúng trọng tâm. Không liệt kê thông tin thừa.
5. Xử lý các trường hợp thiếu thông tin:
   - Không tìm thấy: "Tôi không tìm thấy thông tin liên quan trong các văn bản được cung cấp."
   - Tìm thấy một phần: Trả lời phần có căn cứ, sau đó ghi rõ:
     "Lưu ý: Thông tin về [khía cạnh X] không có trong các văn bản được cung cấp."
"""

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
    def __init__(
        self,
        simple_model: str = "gpt-4o-mini",
        reasoning_model: str = "gpt-4o-mini",
        api_key: str = None,
        temperature: float = 0.0,
        max_tokens: int = 1500,
    ):
        self.simple_model = simple_model
        self.reasoning_model = reasoning_model
        self.model = simple_model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        logger.info(
            "LLMGenerator initialized | simple_model=%s | reasoning_model=%s"
            " | max_tokens=%d | temperature=%.1f",
            simple_model,
            reasoning_model,
            max_tokens,
            temperature,
        )

    def generate(self, query: str, context_chunks: list[dict]) -> str:
        logger.info(
            "Generating answer | model=%s | context_chunks=%d | query: %.80s",
            self.simple_model,
            len(context_chunks),
            query,
        )
        t0 = time.perf_counter()
        context = self._format_context(context_chunks)
        user_prompt = QUERY_PROMPT_TEMPLATE.format(context=context, query=query)
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
        """Rewrite follow-up into standalone question with full legal context from history."""
        try:
            prepared_history = self.prepare_chat_history(chat_history)
            user_prompt = CONDENSE_USER.format(
                chat_history=prepared_history or "", question=question
            )
            resp = self.client.chat.completions.create(
                model=self.reasoning_model,
                messages=[
                    {"role": "system", "content": CONDENSE_SYSTEM},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.0,
                max_tokens=200,  # tăng từ 128 → 200: câu có context pháp lý đầy đủ dài hơn
            )
            standalone = resp.choices[0].message.content.strip()
            standalone = " ".join(standalone.split())
            logger.info("Condensed question: %s", standalone)
            return standalone or question
        except Exception as exc:
            logger.warning("Condense question failed, returning original: %s", exc)
            return question

    def format_chat_history_structured(self, messages: list[dict]) -> str:
        lines = []
        for m in messages[-6:]:
            role = "User" if m["role"] == "user" else "Assistant"
            lines.append(f"{role}: {m['content']}")
        return "\n".join(lines)

    def prepare_chat_history(
        self, chat_history, max_turns: int = 3, max_chars: int = 1200
    ) -> str:
        if not chat_history:
            return ""
        if isinstance(chat_history, list):
            hist = self.format_chat_history_structured(chat_history)
        elif isinstance(chat_history, str):
            lines = [ln.strip() for ln in chat_history.splitlines() if ln.strip()]
            hist = "\n".join(lines[-7:]) if len(lines) > 7 else "\n".join(lines)
        else:
            hist = str(chat_history)

        token_count = len(hist.split())
        if token_count <= 1200:
            return hist

        try:
            summary = self.summarize_chat_history(hist)
            logger.info(
                "Chat history summarized (tokens %d -> %d)",
                token_count,
                len(summary.split()),
            )
            return summary
        except Exception as exc:
            logger.warning("Failed to summarize chat history, truncating: %s", exc)
            return hist[-max_chars:]

    def summarize_chat_history(self, chat_history: str) -> str:
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
                {
                    "role": "user",
                    "content": SUMMARIZE_USER.format(chat_history=chat_history),
                },
            ],
            temperature=0.0,
            max_tokens=128,
        )
        summary = resp.choices[0].message.content.strip()
        return " ".join(summary.split())

    def generate_pipeline(
        self,
        question: str,
        chat_history: str,
        retriever,
        context_chunks: list[dict] = None,
        language_instruction: str = "",
    ) -> tuple[str, list[dict]]:
        standalone = self.condense_question(chat_history, question)

        if context_chunks is None:
            context_chunks = retriever.retrieve(standalone)

        if not context_chunks:
            logger.warning(
                "No context chunks retrieved for question: %.80s", standalone
            )

        logger.info(
            "Generating pipeline answer | model=%s | chunks=%d | question: %.80s",
            self.simple_model,
            len(context_chunks),
            standalone,
        )
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
                {"role": "system", "content": SYSTEM_PROMPT + language_instruction},
                {"role": "user", "content": user_prompt},
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        usage = getattr(response, "usage", None)
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

        answer = response.choices[0].message.content.strip()
        return answer, context_chunks

    def _format_context(self, chunks: list[dict]) -> str:
        if not chunks:
            return "Không có thông tin liên quan"
        parts = []
        for i, chunk in enumerate(chunks, 1):
            metadata = chunk.get("metadata", {})
            luat = metadata.get("luat", "")
            chuong = metadata.get("chuong", "")
            dieu = metadata.get("dieu", "")
            header_parts = [p for p in [luat, chuong, dieu] if p]
            header = " - ".join(header_parts) if header_parts else f"Văn bản {i}"
            content = chunk.get("parent_content") or chunk.get("text", "")
            parts.append(f"[Văn bản {i}: {header}]\nNội dung: {content}")
        return "\n---\n".join(parts)

    def generate_with_reasoning(
        self,
        query: str,
        context_chunks: list[dict],
        chat_history: str = "",
    ) -> tuple[str, dict]:
        standalone = self.condense_question(chat_history, query)
        logger.info(
            "Generating reasoning answer | model=%s | chunks=%d | query: %.80s",
            self.reasoning_model,
            len(context_chunks),
            standalone,
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
                temperature=0,
                top_p=0,
                seed=123,
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
            answer_str = self.generate(query, context_chunks)
            return answer_str, {}

    def _format_reasoning_answer(self, reasoning: dict) -> str:
        hanh_vi = reasoning.get("hanh_vi", "").strip()
        quy_dinh: list = reasoning.get("quy_dinh", [])
        doi_chieu = reasoning.get("doi_chieu", "").strip()
        ket_luan = reasoning.get("ket_luan", "").strip()
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
