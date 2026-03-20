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


def format_sources_md(sources: list) -> str:
    if not sources:
        return ""
    parts = ["\n---\n### 📚 Nguồn tham chiếu (tóm tắt):\n"]
    for i, s in enumerate(sources, 1):
        title = s.get("luat") or s.get("filename") or f"Nguồn {i}"
        meta = []
        if s.get("dieu"):
            meta.append(f"Điều {s.get('dieu')}")
        if s.get("chuong"):
            meta.append(f"Chương {s.get('chuong')}")
        meta_str = f" ({', '.join(meta)})" if meta else ""
        text = s.get("parent_content") or s.get("text") or ""
        if len(text) > 500:
            text = text[:500].rsplit(" ", 1)[0] + "..."
        parts.append(
            f"<details>\n<summary><b>{i}. {title}{meta_str}</b></summary>\n\n{text}\n\n</details>"
        )
    return "\n".join(parts)


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


# --- CHAT START: khởi tạo session_id ---


@cl.on_chat_start
async def on_chat_start():
    """
    Tạo session_id duy nhất khi user mở conversation mới.
    Server dùng session_id này để lưu và truy xuất history
    — đảm bảo follow-up questions có đủ context.
    """
    session_id = str(uuid.uuid4())
    cl.user_session.set("session_id", session_id)
    cl.user_session.set("chat_history", [])
    logger.info("New chat session started | session_id=%s", session_id)


# --- INTERACTIVE HANDLERS ---


async def _on_action_impl(action):
    try:
        name = getattr(action, "name", None) or getattr(action, "id", None)
        value = getattr(action, "value", None) or getattr(action, "data", None)
    except Exception:
        return
    if not name:
        return
    if str(name).startswith("show_source_"):
        try:
            idx = int(str(value)) - 1
        except Exception:
            return
        sources = _get_setting("last_sources", []) or []
        if 0 <= idx < len(sources):
            src = sources[idx]
            full_text = (
                src.get("parent_content") or src.get("text") or "Không có nội dung."
            )
            title = src.get("luat") or f"Nguồn {idx+1}"
            try:
                text_element = getattr(cl, "Text", None)
                if text_element:
                    elem = text_element(name=title, content=full_text, display="inline")
                    await cl.Message(
                        content=f"Chi tiết nội dung của: **{title}**", elements=[elem]
                    ).send()
                else:
                    await cl.Message(content=f"**{title}**\n\n" + full_text).send()
            except Exception:
                await cl.Message(content=f"**{title}**\n\n" + full_text).send()


try:
    reg = getattr(cl, "on_action", None)
    if callable(reg):
        reg(_on_action_impl)
except Exception:
    pass


async def send_sources_ui(sources: list):
    _set_setting("last_sources", sources)
    actions = [
        cl.Action(
            name=f"show_source_{i}",
            value=str(i),
            label=f"Nguồn {i}",
            description="Xem đầy đủ văn bản",
            payload={"index": i},
        )
        for i, _ in enumerate(sources, 1)
    ]
    md_summary = format_sources_md(sources)
    await cl.Message(content=md_summary, actions=actions).send()


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

    # FIX: lấy session_id từ user_session để server nhận diện conversation
    # Nếu on_chat_start chưa chạy (edge case), tạo mới tại chỗ
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
        "session_id": session_id,  # FIX: thêm session_id vào payload
        "top_k": int(top_k),
        "use_reranker": True,
        "reasoning_mode": bool(reasoning_mode),
        # chat_history bỏ khỏi payload — server tự quản lý qua session_id
        # giữ lại làm fallback phòng server restart mất memory
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

    # 2. Gửi nguồn (Interactive)
    sources = data.get("sources", [])
    if sources:
        await send_sources_ui(sources[:5])

    # 3. Gửi phân tích lý luận (nếu có)
    reasoning = data.get("reasoning_steps") or data.get("reasoning")
    if reasoning:
        await cl.Message(content=format_reasoning_md(reasoning)).send()

    # Cập nhật client-side history làm fallback (phòng server restart)
    history = cl.user_session.get("chat_history", []) or []
    history.append({"role": "user", "content": user_text})
    history.append({"role": "assistant", "content": answer})
    cl.user_session.set("chat_history", history[-10:])
