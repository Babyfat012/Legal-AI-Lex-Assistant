import os
import uuid
import json
import logging
from urllib.parse import quote

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

from src.auth.data_layer import PostgreSQLDataLayer

# Đăng ký Data Layer cho Chainlit để bật tính năng Lịch sử & Sidebar
@cl.data_layer
def get_data_layer():
    return PostgreSQLDataLayer()

@cl.password_auth_callback
async def auth_callback(username: str, password: str) -> cl.User | None:
    from src.auth.database import AsyncSessionLocal, User
    from src.auth.security import verify_password
    from sqlalchemy import select

    async with AsyncSessionLocal() as db:
        result = await db.execute(
            select(User).where(User.email == username, User.is_active == True)
        )
        user = result.scalar_one_or_none()

    if user and verify_password(password, user.hashed_pw):
        return cl.User(
            identifier=username,
            metadata={"user_id": str(user.id), "full_name": user.full_name or ""}
        )
    return None


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


def format_web_sources_md(web_sources: list[dict]) -> str:
    """
    Render web sources từ Serper thành Markdown có clickable highlight URL.
    Mỗi link dùng #:~:text= (Scroll to Text Fragment) để trình duyệt
    tự scroll đến và highlight đoạn liên quan khi người dùng click.
    """
    if not web_sources:
        return ""
    lines = ["\n---\n### 🌐 Nguồn tham khảo từ web:\n"]
    for i, src in enumerate(web_sources, 1):
        title = src.get("title", f"Nguồn {i}")
        highlight_url = src.get("highlight_url") or src.get("url", "")
        snippet = src.get("snippet", "")
        # Hiển thị 100 ký tự đầu của snippet làm preview
        preview = snippet[:100].strip()
        if len(snippet) > 100:
            preview += "..."
        lines.append(
            f"{i}. **[{title}]({highlight_url})**\n"
            f"   > {preview}"
        )
    lines.append(
        "\n> 💡 *Click vào link để mở trang web và xem phần được highlight trực tiếp.*"
    )
    return "\n".join(lines)


def _extract_law_text(chunk_text: str) -> str:
    """
    Trích xuất nội dung THỰC TẾ của điều luật, bỏ qua các phần do hệ thống inject:
      - Context prefix:   "[Luật Việc Làm - Chương I - Điều 2. Giải thích từ ngữ]"
      - Markdown heading: "#### Điều 2. Giải thích từ ngữ"
      - Câu dẫn kết thúc bằng ':' trước danh sách khoản
        (VD: "Trong Luật này, các từ ngữ dưới đây được hiểu như sau:")

    Trả về chuỗi SINGLE-LINE (không có \n) an toàn cho URL fragment (#:~:text=).
    """
    import re
    if not chunk_text:
        return ""
    result_lines = []
    for line in chunk_text.split("\n"):
        stripped = line.strip()
        # Bỏ dòng context prefix: "[Luật... - Điều...]"
        if stripped.startswith("[") and stripped.endswith("]"):
            continue
        # Bỏ dòng markdown heading: "# / ## / ### / ####"
        if re.match(r"^#{1,6}\s+", stripped):
            continue
        result_lines.append(stripped)
    joined = " ".join(r for r in result_lines if r)
    result = re.sub(r" {2,}", " ", joined).strip()

    # Bỏ câu dẫn đầu kết thúc bằng ':' nếu theo sau là khoản đánh số
    # VD: "Trong Luật này, ... như sau: 1. Người lao động là..."
    # → chỉ giữ "1. Người lao động là..."
    intro_match = re.match(r"^(.+?:)\s+(\d+\.\s+.+)", result, re.DOTALL)
    if intro_match:
        remainder = intro_match.group(2).strip()
        if len(remainder) > 20:
            result = remainder

    return result or " ".join(chunk_text.split())


def _build_url_fragment(actual_text: str) -> str:
    """
    Tạo text fragment ngắn, kết thúc tại boundary tự nhiên (:, ;, .)
    để tránh span qua nhiều HTML element trên trang web nguồn.

    Vấn đề: website pháp luật thường render mỗi khoản a), b), c) thành
    separate <p> element. Fragment dài qua nhiều khoản sẽ không match.
    Giải pháp: cắt tại ký tự kết thúc câu đầu tiên sau ít nhất 30 ký tự.

    Ví dụ:
      Input:  "Thông tin thị trường lao động bao gồm: a) Thông tin về cung..."
      Output: "Thông tin thị trường lao động bao gồm:"   ← trong 1 element

      Input:  "1. Người lao động là công dân Việt Nam từ đủ 15 tuổi trở lên có khả năng lao động và có nhu cầu làm việc."
      Output: "1. Người lao động là công dân Việt Nam từ đủ 15 tuổi trở lên có khả năng lao động và có nhu cầu làm việc."
    """
    if not actual_text:
        return ""
    MIN_CHARS = 30   # tối thiểu: tránh fragment quá ngắn không đặc trưng
    MAX_CHARS = 200  # tối đa: fallback nếu không tìm được boundary

    for end_char in [":", ";", "."]:
        idx = actual_text.find(end_char, MIN_CHARS)
        if 0 < idx <= MAX_CHARS:
            return actual_text[: idx + 1].strip()

    # Không tìm được boundary → dùng 150 ký tự đầu
    return actual_text[:150].strip()


def format_rag_sources_md(sources: list[dict]) -> str:
    """
    Render RAG sources có source_url thành Markdown với highlight URL.

    Với mỗi chunk có source_url, tính on-the-fly:
        actual_text   = _extract_law_text(chunk.text)    # bỏ context prefix/heading
        highlight_url = source_url + "#:~:text=" + quote(actual_text[:120])

    Browser tự scroll đến và highlight đúng đoạn nội dung điều luật thực tế.
    """
    sources_with_url = [s for s in sources if s.get("source_url")]
    if not sources_with_url:
        return ""

    lines = ["\n---\n### 📚 Nguồn trong Knowledge Base:\n"]
    seen_urls: set[str] = set()  # tránh hiển thị trùng nguồn

    displayed = 0
    for src in sources_with_url:
        if displayed >= 3:  # tối đa 3 nguồn
            break
        source_url = src["source_url"]
        if source_url in seen_urls:
            continue
        seen_urls.add(source_url)

        # Lấy label hiển thị: ưu tiên Dieu > Luat > filename
        label = (
            src.get("dieu")
            or src.get("luat")
            or src.get("filename")
            or f"Nguồn {displayed + 1}"
        )

        # Strip context prefix & heading → nội dung thực tế của điều luật
        raw_text = src.get("text") or ""
        actual_text = _extract_law_text(raw_text)  # single-line, normalized

        # Chọn text tốt nhất cho URL fragment:
        # 1. Ưu tiên dùng tên Điều (dieu metadata) — luôn nằm trong 1 heading element
        #    → browser tìm được, highlight heading → scroll đúng điều luật
        # 2. Nếu không có dieu → dùng _build_url_fragment() cắt tại boundary đầu tiên
        dieu_name = (src.get("dieu") or "").strip()
        if dieu_name and len(dieu_name) > 5:
            # Dùng heading của điều luật: ngắn, trong 1 element, match chắc chắn
            fragment_text = dieu_name
        else:
            # Fallback: lấy clause đầu tiên kết thúc tại :, ;, .
            fragment_text = _build_url_fragment(actual_text)

        if fragment_text:
            encoded = quote(fragment_text, safe="")
            highlight_url = f"{source_url}#:~:text={encoded}"
        else:
            highlight_url = source_url

        # Preview: 150 ký tự đầu của nội dung thực tế
        preview = actual_text[:150].strip()
        if len(actual_text) > 150:
            preview += "..."

        lines.append(
            f"{displayed + 1}. **[{label}]({highlight_url})**\n"
            f"   > {preview}"
        )
        displayed += 1

    if displayed == 0:
        return ""

    lines.append(
        "\n> 💡 *Click để xem điều khoản được highlight trực tiếp trên trang nguồn.*"
    )
    return "\n".join(lines)


# --- CHAT START ---


@cl.on_chat_start
async def on_chat_start():
    # Chainlit manages thread ID automatically
    thread_id = cl.context.session.thread_id if hasattr(cl.context, 'session') and hasattr(cl.context.session, 'thread_id') else str(uuid.uuid4())
    cl.user_session.set("session_id", thread_id)
    cl.user_session.set("thread_id", thread_id)
    cl.user_session.set("chat_history", [])
    logger.info("New chat session started | session_id=%s", thread_id)


@cl.on_chat_resume
async def on_chat_resume(thread: dict):
    """Khôi phục session khi user click vào conversation cũ trong sidebar."""
    thread_id = thread.get("id", "")
    cl.user_session.set("session_id", thread_id)
    cl.user_session.set("thread_id", thread_id)
    cl.user_session.set("is_first_message", False)

    # Reconstruct chat_history from thread steps for context
    steps = thread.get("steps", []) or []
    history = []
    for step in steps:
        step_type = step.get("type", "")
        content = step.get("output", "")
        if not content:
            continue
        if step_type == "user_message":
            history.append({"role": "user", "content": content})
        elif step_type == "assistant_message":
            history.append({"role": "assistant", "content": content})
    # Keep last 10 messages for context window
    cl.user_session.set("chat_history", history[-10:])
    logger.info("Resumed chat session | thread_id=%s | history_len=%d", thread_id, len(history))


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
        # Fallback: use Chainlit thread_id if available
        session_id = cl.context.session.thread_id if hasattr(cl.context, 'session') and hasattr(cl.context.session, 'thread_id') else str(uuid.uuid4())
        cl.user_session.set("session_id", session_id)
        logger.warning(
            "session_id missing — using thread_id as fallback | session_id=%s", session_id
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

    # 3. Hiển thị nguồn tài liệu với highlight URL
    tool_used = data.get("tool_used", "rag")
    if tool_used == "rag":
        # RAG mode: tính highlight URL on-the-fly từ source_url + text
        rag_sources = data.get("sources", [])
        rag_md = format_rag_sources_md(rag_sources)
        if rag_md:
            await cl.Message(content=rag_md).send()
    else:
        # Web search fallback: highlight URL đã có sẵn từ Serper API
        web_sources = data.get("web_sources", [])
        if web_sources:
            web_md = format_web_sources_md(web_sources)
            if web_md:
                await cl.Message(content=web_md).send()

    # Cập nhật client-side history làm fallback
    history = cl.user_session.get("chat_history", []) or []
    history.append({"role": "user", "content": user_text})
    history.append({"role": "assistant", "content": answer})
    cl.user_session.set("chat_history", history[-10:])
    # Note: Chainlit automatically calls update_thread(name=first_message)
    # via flush_thread_queues on first interaction, which triggers our
    # data_layer.update_thread to persist a trimmed title to the DB.
