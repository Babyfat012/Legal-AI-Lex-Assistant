import os
import uuid
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    import httpx
except ImportError:
    httpx = None

try:
    import chainlit as cl
except ImportError:

    class cl:
        @staticmethod
        def on_message(fn):
            return fn

        @staticmethod
        def on_action(fn):
            return fn

        @staticmethod
        def on_chat_start(fn):
            return fn

        class Message:
            def __init__(self, content=None, elements=None, actions=None):
                pass

            async def send(self):
                pass

        user_session = {}


API_URL = os.getenv("LEX_API_URL", "http://localhost:8000/api/v1/chat")
DEFAULT_TOP_K = int(os.getenv("CHAIN_TOP_K", "5"))


# --- UTILS ---


def _get_setting(name, default):
    return cl.user_session.get(name, default)


def _set_setting(name, value):
    cl.user_session.set(name, value)


def format_reasoning_md(reasoning: dict) -> str:
    if not reasoning:
        return ""
    parts = ["\n---\n### 🧠 Phân tích pháp lý:\n"]
    sections = {
        "hanh_vi": "1. Hành vi các bên",
        "quy_dinh": "2. Quy định áp dụng",
        "doi_chieu": "3. Đối chiếu pháp lý",
        "ket_luan": "4. Kết luận",
    }
    for key, label in sections.items():
        content = reasoning.get(key)
        if content:
            if isinstance(content, list):
                content = "\n".join([f"- {item}" for item in content])
            parts.append(f"**{label}**:\n{content}\n")
    return "\n".join(parts)


# --- CHAT START ---


@cl.on_chat_start
async def on_chat_start():
    session_id = str(uuid.uuid4())
    cl.user_session.set("session_id", session_id)
    cl.user_session.set("chat_history", [])
    logger.info("New chat session started | session_id=%s", session_id)


# --- MAIN CHAT ---


@cl.on_message
async def main(message: cl.Message):
    user_text = message.content.strip()

    if not user_text:
        await cl.Message(content="Vui lòng nhập câu hỏi pháp luật.").send()
        return

    # Xử lý Slash Commands
    if user_text.startswith("/"):
        cmd = user_text.split()
        if cmd[0] == "/settings":
            await cl.Message(
                content=f"Cài đặt: \n- TopK: {_get_setting('top_k', DEFAULT_TOP_K)}\n- Reasoning: {_get_setting('reasoning_mode', False)}"
            ).send()
        elif cmd[0] == "/reasoning":
            mode = cmd[1].lower() in ["on", "true", "1"] if len(cmd) > 1 else False
            _set_setting("reasoning_mode", mode)
            await cl.Message(
                content=f"Đã {'bật' if mode else 'tắt'} chế độ lập luận."
            ).send()
        return

    msg = cl.Message(content="🔍 Đang tìm kiếm thông tin liên quan...")
    await msg.send()

    session_id = cl.user_session.get("session_id")
    if not session_id:
        session_id = str(uuid.uuid4())
        cl.user_session.set("session_id", session_id)
        logger.warning(
            "session_id missing — created fallback | session_id=%s", session_id
        )

    reasoning_mode = _get_setting("reasoning_mode", False)
    top_k = _get_setting("top_k", DEFAULT_TOP_K)

    payload = {
        "query": user_text,
        "session_id": session_id,
        "top_k": int(top_k),
        "use_reranker": True,
        "reasoning_mode": bool(reasoning_mode),
        "chat_history": cl.user_session.get("chat_history", []),
    }

    if httpx is None:
        await cl.Message(content="Lỗi: Chưa cài đặt thư viện `httpx`").send()
        return

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(API_URL, json=payload)
            resp.raise_for_status()
            data = resp.json()
    except Exception as e:
        logger.error(f"API Error: {e}")
        await cl.Message(content=f"❌ Lỗi kết nối API: {str(e)}").send()
        return

    # 1. Gửi câu trả lời chính
    answer = data.get("answer", "Xin lỗi, tôi không tìm thấy thông tin phù hợp.")
    await cl.Message(content=answer).send()

    # 2. Gửi phân tích lý luận (nếu có)
    reasoning = data.get("reasoning_steps") or data.get("reasoning")
    if reasoning:
        await cl.Message(content=format_reasoning_md(reasoning)).send()

    # Cập nhật client-side history làm fallback
    history = cl.user_session.get("chat_history", []) or []
    history.append({"role": "user", "content": user_text})
    history.append({"role": "assistant", "content": answer})
    cl.user_session.set("chat_history", history[-10:])
