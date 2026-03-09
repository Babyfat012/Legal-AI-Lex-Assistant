"""
QueryAnalyzer: Nâng cao chất lượng retrieval cho câu hỏi pháp lý phức tạp.

Cung cấp 2 kỹ thuật:
    1. Query Decomposition — bẻ gãy câu hỏi tranh chấp/suy luận thành các
       sub-queries pháp lý độc lập, có thể tìm kiếm được trong văn bản luật.

    2. HyDE (Hypothetical Document Embeddings) — bảo LLM viết đoạn văn quy
       phạm giả định mang thuật ngữ chuyên môn, dùng chính đoạn đó để embed
       và search thay cho câu hỏi đời thường của người dùng.

Cả 2 kỹ thuật đều fallback an toàn về [query gốc] khi LLM lỗi.
"""

import os
import json
from openai import OpenAI
from core.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Prompts — Query Decomposition
# ---------------------------------------------------------------------------

_DECOMPOSE_SYSTEM = """\
Bạn là chuyên gia phân tích pháp lý Việt Nam.
Nhiệm vụ: phân rã câu hỏi pháp lý phức tạp thành các sub-queries độc lập,
có thể tìm kiếm được trong văn bản luật Việt Nam.

Nguyên tắc:
- Mỗi sub-query phải là 1 khía cạnh pháp lý cụ thể, riêng biệt
- Dùng ngôn ngữ quy phạm pháp luật: nhường đường, vi phạm, xử phạt,
  trách nhiệm pháp lý, quyền ưu tiên, tín hiệu giao thông, v.v.
- Bao phủ: quy tắc hành vi + mức xử phạt + trách nhiệm bồi thường (nếu liên quan)
- Tối đa 4 sub-queries

Trả về JSON array (không có text thêm, không markdown):
["sub-query 1", "sub-query 2", ...]
"""

_DECOMPOSE_USER = "Câu hỏi: {query}\n\nPhân rã thành sub-queries pháp lý (JSON array):"

# ---------------------------------------------------------------------------
# Prompts — HyDE
# ---------------------------------------------------------------------------

_HYDE_SYSTEM = """\
Bạn là chuyên gia soạn thảo văn bản pháp luật Việt Nam.
Nhiệm vụ: Viết một đoạn văn bản quy phạm giả định (phong cách nghị định/thông tư/luật)
có thể chứa câu trả lời cho tình huống được mô tả.

Yêu cầu:
- Dùng thuật ngữ pháp lý chuyên ngành (không dùng ngôn ngữ đời thường)
- Độ dài: 100–150 từ
- Chỉ viết đoạn văn, không có tiêu đề hay giải thích thêm
"""

_HYDE_USER = "Tình huống: {query}\n\nĐoạn văn bản quy phạm giả định:"

# ---------------------------------------------------------------------------
# Heuristics — phát hiện câu hỏi suy luận/tranh chấp
# ---------------------------------------------------------------------------

_REASONING_KEYWORDS = [
    # Tranh chấp lỗi
    "ai lỗi", "lỗi của ai", "xác định lỗi", "trách nhiệm thuộc về",
    # Tình huống
    "tình huống", "kịch bản", "trường hợp",
    # Tranh chấp pháp lý
    "tranh chấp", "phân tích", "có hợp pháp không",
    "có phạm luật không", "có vi phạm không", "có được phép không",
    "có đúng luật không", "có bị phạt không",
    # Tai nạn giao thông
    "đâm vào", "va chạm", "tai nạn", "quẹt vào", "húc vào",
    "đụng xe", "tông vào",
    # Trách nhiệm
    "trách nhiệm", "bồi thường", "ai chịu",
]


class QueryAnalyzer:
    """
    Phân tích và biến đổi câu hỏi pháp lý để tăng chất lượng retrieval.

    Workflow khi ``is_complex(query)`` trả về True:
        1. ``decompose(query)``   → list[str] sub-queries có tính pháp lý cao
        2. ``generate_hyde(query)`` → str hypothetical document để embed

    Cả 2 method đều:
        - Fallback an toàn về ``[query]`` hoặc ``query`` nếu LLM gặp lỗi
        - Log đầy đủ để debug
    """

    def __init__(self, model: str = "gpt-4o-mini", api_key: str = None):
        self.model = model
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        logger.info("QueryAnalyzer initialized | model=%s", model)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def is_complex(self, query: str) -> bool:
        """
        Phát hiện câu hỏi suy luận/tranh chấp bằng heuristics — không dùng LLM.

        Trả về True khi:
        - Query chứa keyword tình huống / tranh chấp, hoặc
        - Query dài (>80 ký tự) và có nhiều mệnh đề (dấu phẩy / "và")
        """
        q_lower = query.lower()
        keyword_match = any(kw in q_lower for kw in _REASONING_KEYWORDS)
        long_clause = len(query) > 80 and (
            "," in query or ";" in query or " và " in query
        )
        result = keyword_match or long_clause
        if result:
            logger.info("is_complex=True for query: %.80s", query)
        return result

    def decompose(self, query: str) -> list[str]:
        """
        Phân rã câu hỏi phức tạp thành sub-queries pháp lý độc lập.

        LLM được yêu cầu trả về JSON array thuần túy.
        Nếu parse thất bại → fallback về ``[query]``.

        Returns:
            list[str]: Tối đa 4 sub-queries, luôn có ít nhất 1 phần tử.
        """
        logger.info("[Decompose] query: %.80s", query)
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": _DECOMPOSE_SYSTEM},
                    {"role": "user", "content": _DECOMPOSE_USER.format(query=query)},
                ],
                temperature=0.2,
                max_tokens=400,
            )
            content = resp.choices[0].message.content.strip()

            # Strip markdown code fences nếu LLM trả về ```json ... ```
            if "```" in content:
                content = content.split("```")[1].lstrip("json").strip()

            sub_queries = json.loads(content)
            if isinstance(sub_queries, list):
                valid = [q for q in sub_queries if isinstance(q, str) and q.strip()][:4]
                if valid:
                    logger.info("[Decompose] → %d sub-queries: %s", len(valid), valid)
                    return valid

        except Exception as exc:
            logger.warning("[Decompose] failed: %s — fallback to original", exc)

        return [query]

    def generate_hyde(self, query: str) -> str:
        """
        Tạo Hypothetical Document (HyDE) để tăng chất lượng vector search.

        Văn bản giả định dùng thuật ngữ pháp lý (Legal Jargon) giúp
        embedding khớp với cách diễn đạt trong văn bản luật thực tế,
        thay vì ngôn ngữ đời thường của người dùng.

        Nếu LLM lỗi → fallback về ``query`` gốc.

        Returns:
            str: Đoạn văn bản quy phạm giả định (~100–150 từ).
        """
        logger.info("[HyDE] query: %.80s", query)
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": _HYDE_SYSTEM},
                    {"role": "user", "content": _HYDE_USER.format(query=query)},
                ],
                temperature=0.3,
                max_tokens=300,
            )
            hyde_doc = resp.choices[0].message.content.strip()
            logger.info("[HyDE] generated | %d chars", len(hyde_doc))
            return hyde_doc

        except Exception as exc:
            logger.warning("[HyDE] failed: %s — fallback to original query", exc)

        return query
