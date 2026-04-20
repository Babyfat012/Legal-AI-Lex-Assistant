"""
Intent Detector — Phân loại intent của user bằng LLM.

- legal_qa:      Câu hỏi pháp lý thông thường → RAG pipeline
- document_gen:  Yêu cầu tạo đơn/mẫu → Document Generation flow
"""

import json
import os
from openai import OpenAI
from core.logger import get_logger

logger = get_logger(__name__)

INTENT_SYSTEM_PROMPT = """\
Bạn là bộ phân loại intent cho trợ lý pháp lý Lex.
Phân tích câu hỏi của người dùng và trả về JSON.

OUTPUT FORMAT (chỉ trả JSON, không giải thích):
{"intent": "legal_qa" | "document_gen", "template_id": "..." | null, "confidence": 0.0-1.0}

CÁC TEMPLATE HỖ TRỢ:
- "ly_hon": Đơn xin ly hôn (thuận tình hoặc đơn phương)
- "khieu_nai": Đơn khiếu nại (khiếu nại quyết định hành chính, hành vi hành chính)
- "khoi_kien": Đơn khởi kiện (khởi kiện vụ án dân sự)

QUY TẮC:
1. Intent "document_gen" khi user YÊU CẦU TẠO/LÀM/SOẠN đơn, mẫu đơn, tờ đơn.
   Các từ khoá: tạo đơn, làm đơn, soạn đơn, viết đơn, mẫu đơn, lập đơn, tải đơn,
   giúp tôi làm đơn, cần đơn, muốn nộp đơn, fill form, create form, generate document.
2. Intent "legal_qa" cho mọi câu hỏi pháp lý khác (hỏi thông tin, tư vấn, tra cứu).
3. Nếu user yêu cầu tạo đơn nhưng KHÔNG rõ loại → template_id = null.
4. Trả confidence cao (>0.8) khi chắc chắn, thấp (<0.5) khi mơ hồ.

VÍ DỤ:
User: "Tạo đơn ly hôn" → {"intent": "document_gen", "template_id": "ly_hon", "confidence": 0.95}
User: "Mức phạt uống rượu lái xe" → {"intent": "legal_qa", "template_id": null, "confidence": 0.95}
User: "Tôi muốn làm đơn khiếu nại" → {"intent": "document_gen", "template_id": "khieu_nai", "confidence": 0.9}
User: "Làm đơn kiện hàng xóm" → {"intent": "document_gen", "template_id": "khoi_kien", "confidence": 0.85}
User: "Thủ tục ly hôn như thế nào" → {"intent": "legal_qa", "template_id": null, "confidence": 0.9}
User: "Giúp tôi soạn đơn" → {"intent": "document_gen", "template_id": null, "confidence": 0.85}
"""


class IntentDetector:
    def __init__(self, model: str = None):
        self.model = model or os.getenv("SIMPLE_MODEL", "gpt-4o-mini")
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def detect(self, user_message: str) -> dict:
        """
        Detect intent from user message.
        Returns: {"intent": str, "template_id": str|None, "confidence": float}
        """
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": INTENT_SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                temperature=0.0,
                max_tokens=100,
                response_format={"type": "json_object"},
            )
            raw = resp.choices[0].message.content.strip()
            result = json.loads(raw)

            # Validate
            intent = result.get("intent", "legal_qa")
            if intent not in ("legal_qa", "document_gen"):
                intent = "legal_qa"

            template_id = result.get("template_id")
            confidence = float(result.get("confidence", 0.5))

            logger.info(
                "Intent detected | intent=%s | template_id=%s | confidence=%.2f | query=%.60s",
                intent, template_id, confidence, user_message,
            )
            return {
                "intent": intent,
                "template_id": template_id,
                "confidence": confidence,
            }

        except Exception as exc:
            logger.warning("Intent detection failed: %s — defaulting to legal_qa", exc)
            return {"intent": "legal_qa", "template_id": None, "confidence": 0.0}
