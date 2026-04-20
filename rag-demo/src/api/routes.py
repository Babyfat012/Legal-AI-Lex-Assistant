import os
import time
import uuid
import httpx
from collections import defaultdict
from datetime import datetime
from fastapi import APIRouter, HTTPException, Depends, UploadFile, File, Form
from functools import lru_cache

from api.schemas import (
    ChatRequest,
    ChatResponse,
    SourceChunk,
    WebSource,
    IngestRequest,
    IngestResponse,
    CollectionInfoResponse,
    HealthResponse,
)
from ingestion.pipeline import IngestionPipeline
from ingestion.qdrant_store import QdrantVectorStore
from embedding.embedding import EmbeddingService
from embedding.bm25_en import BM25Encoder
from retrieval.retriever import Retriever
from retrieval.reranker import Reranker
from retrieval.query_analyzer import QueryAnalyzer
from generator.llm_generator import LLMGenerator
from core.logger import get_logger
from core.database import IngestLog, SessionLocal, get_db
from sqlalchemy.orm import Session
from sqlalchemy import update, select, insert
from api.auth_routes import get_current_user
from auth.database import AsyncSessionLocal, Conversation, Message
from auth.database import AsyncSessionLocal, Message as DbMessage, Conversation as DbConversation

logger = get_logger(__name__)

router = APIRouter()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_BM25_VOCAB_PATH: str = os.getenv(
    "BM25_VOCAB_PATH",
    os.path.join(BASE_DIR, "data", "bm25_vocab.json"),
)

# ---------------------------------------------------------------------------
# Server-side conversation store
# ---------------------------------------------------------------------------
# Lưu history theo session_id để không phụ thuộc client gửi đúng.
# Key: session_id (str), Value: list[dict] với format {"role": ..., "content": ...}
# Giới hạn MAX_TURNS * 2 messages (user + assistant) để tránh memory leak.
# ---------------------------------------------------------------------------
_MAX_HISTORY_TURNS = 6  # 3 cặp user/assistant
_conversation_store: dict[str, list[dict]] = defaultdict(list)


def _get_history(session_id: str) -> list[dict]:
    """Lấy history của session, trả về list rỗng nếu chưa có."""
    return _conversation_store.get(session_id, [])


def _append_history(session_id: str, role: str, content: str) -> None:
    """Thêm 1 turn vào history, tự cắt bớt nếu vượt giới hạn."""
    store = _conversation_store[session_id]
    store.append({"role": role, "content": content})
    # Giữ tối đa _MAX_HISTORY_TURNS * 2 messages (rolling window)
    if len(store) > _MAX_HISTORY_TURNS * 2:
        _conversation_store[session_id] = store[-(_MAX_HISTORY_TURNS * 2) :]


# ---------------------------------------------------------------------------
# Language instruction for LLM responses
# ---------------------------------------------------------------------------
_LANG_INSTRUCTION = {
    "en": (
        "\n\n[LANGUAGE INSTRUCTION]: The user has selected English. "
        "You MUST respond entirely in English. Translate all legal terms, "
        "article names, and explanations into English while keeping the "
        "original Vietnamese legal reference names in parentheses for accuracy. "
        "Example: 'Article 123, Clause 2 — Law X of YYYY (Điều 123, Khoản 2 — Luật X năm YYYY)'"
    ),
    "vi": "",  # Vietnamese is the default, no extra instruction needed
}


def _get_lang_instruction(language: str) -> str:
    """Return language instruction to append to system prompt."""
    return _LANG_INSTRUCTION.get(language, "")


# ---------------------------------------------------------------------------
# Web Search Fallback — Serper API + Highlight URL (#:~:text=)
# ---------------------------------------------------------------------------
SERPER_API_KEY = os.getenv("SERPER_API_KEY", "")
SERPER_URL = "https://google.serper.dev/search"
WEB_SEARCH_SCORE_THRESHOLD = float(os.getenv("WEB_SEARCH_THRESHOLD", "0.1"))


async def _serper_search(query: str, num_results: int = 3) -> list[dict]:
    """
    Gọi Serper API — trả về list dict gồm title, url, snippet.
    Mỗi phần tử sau đó được wrap thành WebSource.build() để tạo highlight_url.
    """
    if not SERPER_API_KEY:
        logger.warning("SERPER_API_KEY not set — web search skipped")
        return []
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(
                SERPER_URL,
                headers={
                    "X-API-KEY": SERPER_API_KEY,
                    "Content-Type": "application/json",
                },
                json={
                    "q": f"{query} pháp luật Việt Nam",
                    "num": num_results,
                    "gl": "vn",
                    "hl": "vi",
                },
            )
            resp.raise_for_status()
            data = resp.json()
        results = []
        for item in data.get("organic", [])[:num_results]:
            results.append({
                "title":   item.get("title", ""),
                "url":     item.get("link", ""),
                "snippet": item.get("snippet", ""),
            })
        logger.info("Serper returned %d results for query: %.60s", len(results), query)
        return results
    except Exception as exc:
        logger.warning("Serper search failed: %s", exc)
        return []


def _web_synthesize(generator, query: str, web_results: list[dict], language: str = "vi") -> str:
    """
    Nhờ LLM tổng hợp câu trả lời từ web results.
    Sử dụng simple_model để tránh tốn chi phí.
    """
    search_context = "\n\n".join([
        f"Tiêu đề: {r['title']}\nNội dung: {r['snippet']}\nNguồn: {r['url']}"
        for r in web_results
    ])
    system_msg = (
        "Bạn là chuyên gia pháp luật Việt Nam. Tổng hợp câu trả lời ngắn gọn, "
        "chính xác từ các kết quả tìm kiếm web sau. Dùng **bold** cho số tiền, "
        "mức phạt, điều khoản quan trọ ng. Ngôn ngữ rõ ràng, dễ hiểu."
    )
    system_msg += _get_lang_instruction(language)
    user_msg = f"Kết quả tìm kiếm web:\n{search_context}\n\nCâu hỏi: {query}"
    try:
        resp = generator.client.chat.completions.create(
            model=generator.simple_model,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user",   "content": user_msg},
            ],
            temperature=0.1,
            max_tokens=800,
        )
        return resp.choices[0].message.content.strip()
    except Exception as exc:
        logger.warning("Web synthesize LLM call failed: %s", exc)
        # Fallback: snippet của result đầu tiên
        return web_results[0]["snippet"] if web_results else "Không tìm thấy thông tin."


# --- Dependency injection (singleton via lru_cache) ---


@lru_cache(maxsize=1)
def get_embedding_service() -> EmbeddingService:
    return EmbeddingService()


@lru_cache(maxsize=1)
def get_bm25_encoder() -> BM25Encoder:
    return BM25Encoder(vocab_path=_BM25_VOCAB_PATH)


@lru_cache(maxsize=1)
def get_vector_store() -> QdrantVectorStore:
    return QdrantVectorStore(
        collection_name=os.getenv("QDRANT_COLLECTION", "legal_documents"),
        dimension=get_embedding_service().dimension,
        url=os.getenv("QDRANT_URL", "http://localhost:6333"),
    )


@lru_cache(maxsize=1)
def get_retriever() -> Retriever:
    return Retriever(
        embedding_service=get_embedding_service(),
        bm25_encoder=get_bm25_encoder(),
        vector_store=get_vector_store(),
        reranker=Reranker(),
        initial_top_k=40,
        final_top_n=5,
        use_reranker=True,
    )


@lru_cache(maxsize=1)
def get_generator() -> LLMGenerator:
    return LLMGenerator(
        simple_model=os.getenv("SIMPLE_MODEL", "gpt-4o-mini"),
        reasoning_model=os.getenv("REASONING_MODEL", "gpt-4o-mini"),
    )


@lru_cache(maxsize=1)
def get_query_analyzer() -> QueryAnalyzer:
    return QueryAnalyzer()


@lru_cache(maxsize=1)
def get_pipeline() -> IngestionPipeline:
    return IngestionPipeline(
        embedding_service=get_embedding_service(),
        bm25_encoder=get_bm25_encoder(),
        vector_store=get_vector_store(),
    )


# --- Endpoints ---
@router.get("/health", response_model=HealthResponse, tags=["System"])
def health_check():
    """Kiểm tra trạng thái hệ thống."""
    logger.info("GET /health")
    qdrant_status = "unreachable"
    try:
        store = get_vector_store()
        info = store.get_collection_info()
        qdrant_status = f"ok ({info['points_count']} points)"
    except Exception as e:
        qdrant_status = f"error: {str(e)}"

    return HealthResponse(
        status="ok",
        qdrant=qdrant_status,
        openai_key_set=bool(os.getenv("OPENAI_API_KEY")),
    )


@router.post("/chat", response_model=ChatResponse, tags=["RAG"])
async def chat(
    request: ChatRequest,
    current_user: dict = Depends(get_current_user),
    retriever: Retriever = Depends(get_retriever),
    generator: LLMGenerator = Depends(get_generator),
):
    """
    RAG pipeline với 2 chế độ:

    **Standard mode** (``reasoning_mode=false``):
        Query → Hybrid Search → Rerank → ``simple_model`` generate

    **Reasoning mode** (``reasoning_mode=true``):
        Query → Decompose + HyDE → Multi-search → Rerank → ``reasoning_model`` + CoT

    Session management:
        Truyền ``session_id`` trong request để duy trì ngữ cảnh hội thoại.
        Nếu không truyền, mỗi request được coi là conversation mới (không có history).
    """
    logger.info(
        "POST /chat | query: %.80s | top_k=%d | use_reranker=%s | reasoning_mode=%s",
        request.query,
        request.top_k,
        request.use_reranker,
        request.reasoning_mode,
    )
    t0 = time.perf_counter()
    try:
        # Initialize variables
        sources = []
        web_sources: list[WebSource] = []
        tool_used = "rag"
        answer = ""
        retrieved_chunks = []

        retriever.use_reranker = request.use_reranker
        retriever.final_top_n = request.top_k

        mode = "reasoning" if request.reasoning_mode else "standard"
        sub_queries: list[str] = []
        hyde_doc: str | None = None
        reasoning_steps: dict | None = None

        # FIX: ưu tiên server-side history (đáng tin cậy hơn client-sent history)
        # Client có thể không gửi chat_history, hoặc gửi thiếu — server tự lưu
        session_id = getattr(request, "session_id", None) or ""
        language = getattr(request, "language", "vi") or "vi"
        server_history = _get_history(session_id) if session_id else []

        # Handle conversation creation/retrieval
        conv_uuid = None
        if session_id:
            try:
                conv_uuid = uuid.UUID(session_id)
                # Check if conversation exists and belongs to current user
                async with AsyncSessionLocal() as db:
                    result = await db.execute(
                        select(Conversation).where(
                            Conversation.id == conv_uuid,
                            Conversation.user_id == uuid.UUID(current_user["user_id"])
                        )
                    )
                    conversation = result.scalar_one_or_none()

                    # If conversation doesn't exist, create new one
                    if not conversation:
                        new_conv = Conversation(
                            id=conv_uuid,
                            user_id=uuid.UUID(current_user["user_id"]),
                            title=f"Hội thoại mới {datetime.utcnow().strftime('%Y-%m-%d %H:%M')}",
                            created_at=datetime.utcnow(),
                            updated_at=datetime.utcnow()
                        )
                        db.add(new_conv)
                        await db.commit()
                        logger.info("Created new conversation | id=%s | user=%s", conv_uuid, current_user["email"])
            except ValueError:
                # Invalid UUID format, create new conversation
                conv_uuid = uuid.uuid4()
                async with AsyncSessionLocal() as db:
                    new_conv = Conversation(
                        id=conv_uuid,
                        user_id=uuid.UUID(current_user["user_id"]),
                        title=f"Hội thoại mới {datetime.utcnow().strftime('%Y-%m-%d %H:%M')}",
                        created_at=datetime.utcnow(),
                        updated_at=datetime.utcnow()
                    )
                    db.add(new_conv)
                    await db.commit()
                session_id = str(conv_uuid)
                logger.info("Created new conversation with new UUID | id=%s | user=%s", conv_uuid, current_user["email"])

        # Fallback về client-sent history nếu server chưa có (lần đầu kết nối)
        if server_history:
            effective_history = server_history
            logger.debug(
                "Using server-side history | session=%s | turns=%d",
                session_id,
                len(server_history),
            )
        else:
            # Normalize client-sent history (list[dict] hoặc list[str])
            client_history = request.chat_history or []
            effective_history = [
                (
                    turn
                    if isinstance(turn, dict) and "role" in turn and "content" in turn
                    else {"role": "user", "content": str(turn)}
                )
                for turn in client_history
            ]
            logger.debug(
                "Using client-sent history | turns=%d",
                len(effective_history),
            )

        # Format history thành string cho generator
        history_turns = []
        for turn in effective_history:
            prefix = "User" if turn.get("role") == "user" else "Assistant"
            history_turns.append(f"{prefix}: {turn['content']}")

        chat_history_str = "\n".join(history_turns)

        logger.debug(
            "chat_history passed to generator | len=%d | preview=%.120s",
            len(chat_history_str),
            chat_history_str,
        )

        # --- Pipeline: Rewrite (condense) + Retrieve + Generate ---
        standalone = generator.condense_question(chat_history_str, request.query)
        retrieved_chunks = retriever.retrieve(standalone)

        # --- Web Search Fallback ---
        # Trigger khi không có chunk nào được truy xuất (RAG bị miss)
        tool_used = "rag"
        web_sources: list[WebSource] = []

        if not retrieved_chunks:
            logger.info(
                "RAG returned 0 chunks — triggering web search fallback | query: %.80s",
                standalone,
            )
            import asyncio
            raw_web = await _serper_search(standalone)

            if raw_web:
                tool_used = "web_search"
                # Xây dựng WebSource với highlight_url (#:~:text=)
                web_sources = [
                    WebSource.build(
                        title=r["title"],
                        url=r["url"],
                        snippet=r["snippet"],
                    )
                    for r in raw_web
                ]
                answer = _web_synthesize(generator, standalone, raw_web, language)
                if language == "en":
                    answer += (
                        "\n\n\u26a0\ufe0f *This information is from the web and has not been verified in the knowledge base.*"
                    )
                else:
                    answer += (
                        "\n\n\u26a0\ufe0f *Thông tin này lấy từ web, chưa được xác minh trong knowledge base.*"
                    )
                retrieved_chunks = []
            else:
                if language == "en":
                    answer = (
                        "Sorry, I couldn't find relevant information. "
                        "Please try again with a more specific question."
                    )
                else:
                    answer = (
                        "Xin lỗi, tôi không tìm thấy thông tin liên quan. "
                        "Vui lòng thử lại với câu hỏi cụ thể hơn."
                    )
        else:
            # RAG có kết quả — generate bình thường
            answer, retrieved_chunks = generator.generate_pipeline(
                question=request.query,
                chat_history=chat_history_str,
                retriever=retriever,
                context_chunks=retrieved_chunks,
                language_instruction=_get_lang_instruction(language),
            )

        # Lưu turn mới vào server-side store
        if session_id:
            _append_history(session_id, "user", request.query)
            _append_history(session_id, "assistant", answer)
            logger.debug(
                "History updated | session=%s | total_turns=%d",
                session_id,
                len(_conversation_store[session_id]),
            )

            # --- Persist messages to PostgreSQL ---
            try:
                now = datetime.utcnow()
                conv_uuid = uuid.UUID(session_id)
                async with AsyncSessionLocal() as db:
                    # Save user message
                    db.add(Message(
                        id=uuid.uuid4(),
                        conversation_id=conv_uuid,
                        role="user",
                        content=request.query,
                        sources=None,
                        created_at=now,
                    ))
                    # Save assistant message
                    db.add(Message(
                        id=uuid.uuid4(),
                        conversation_id=conv_uuid,
                        role="assistant",
                        content=answer,
                        sources=sources if sources else None,
                        created_at=now,
                    ))
                    # Update conversation.updated_at
                    await db.execute(
                        update(Conversation)
                        .where(Conversation.id == conv_uuid)
                        .values(updated_at=now)
                    )
                    await db.commit()
                    logger.debug("Messages persisted to DB | conv=%s", session_id)
            except Exception as db_err:
                logger.warning("Failed to persist messages to DB: %s", db_err)

        # Populate RAG sources
        sources = [
            {
                "text": chunk.get("text"),
                "parent_content": chunk["metadata"].get("parent_content"),
                "score": chunk.get("score"),
                "rerank_score": chunk["metadata"].get("rerank_score"),
                "luat": chunk["metadata"].get("luat"),
                "chuong": chunk["metadata"].get("chuong"),
                "muc": chunk["metadata"].get("muc"),
                "dieu": chunk["metadata"].get("dieu"),
                "filename": chunk["metadata"].get("filename"),
                "source_url": chunk["metadata"].get("source_url") or "",
            }
            for chunk in retrieved_chunks
        ]

        logger.info(
            "POST /chat done | mode=%s | tool=%s | %.2fs",
            mode,
            tool_used,
            time.perf_counter() - t0,
        )
        return ChatResponse(
            answer=answer,
            sources=sources,
            web_sources=web_sources,
            tool_used=tool_used,
            query=request.query,
            mode=mode,
            sub_queries=sub_queries,
            hyde_doc=hyde_doc,
            reasoning_steps=reasoning_steps,
        )

    except Exception as e:
        logger.error("Unexpected error in /chat: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ingest", response_model=IngestResponse, tags=["Ingestion"])
def ingest(
    request: IngestRequest,
    pipeline: IngestionPipeline = Depends(get_pipeline),
):
    """
    Ingest file PDF/DOCX vào vector database.

    Pipeline: Load → Pre-process → Chunk → Embed (Dense+Sparse) → Store
    """
    logger.info(
        "POST /ingest | file=%s | is_dir=%s | recreate=%s",
        request.file_path,
        request.is_directory,
        request.recreate_collection,
    )
    if not os.path.exists(request.file_path):
        raise HTTPException(
            status_code=400,
            detail=f"File not found: {request.file_path}",
        )

    try:
        if request.recreate_collection:
            store = get_vector_store()
            store.recreate_collection()
            get_vector_store.cache_clear()
            get_retriever.cache_clear()
            get_pipeline.cache_clear()

        result = pipeline.ingest(
            source=request.file_path,
            is_directory=request.is_directory,
            source_url=request.source_url or "",
        )

        return IngestResponse(
            status=result["status"],
            documents_loaded=result["documents_loaded"],
            chunks_created=result["chunks_created"],
            chunks_stored=result["chunks_stored"],
            collection_info=result["collection_info"],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/collection/info", response_model=CollectionInfoResponse, tags=["System"])
def collection_info(store: QdrantVectorStore = Depends(get_vector_store)):
    """Lấy thông tin collection Qdrant hiện tại."""
    try:
        info = store.get_collection_info()
        return CollectionInfoResponse(**info)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/ingest/logs", tags=["Ingestion"])
def get_ingest_logs(db: Session = Depends(get_db)):
    """Lấy lịch sử ingest từ PostgreSQL — dùng bởi Admin UI."""
    logs = db.query(IngestLog).order_by(IngestLog.upload_at.desc()).all()
    return [
        {
            "id": str(log.id),
            "file_name": log.file_name,
            "upload_at": log.upload_at.isoformat() if log.upload_at else None,
            "status": log.status,
            "chunk_count": log.chunk_count,
            "elapsed_secs": log.elapsed_secs,
            "error_msg": log.error_msg,
        }
        for log in logs
    ]


@router.delete("/collection", tags=["System"])
def delete_collection(store: QdrantVectorStore = Depends(get_vector_store)):
    """Xóa toàn bộ collection (dùng khi cần reset)."""
    try:
        store.delete_collection()
        get_vector_store.cache_clear()
        get_retriever.cache_clear()
        get_pipeline.cache_clear()
        return {"status": "deleted"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/collection/documents/{file_name}", tags=["System"])
def delete_document(file_name: str, store: QdrantVectorStore = Depends(get_vector_store)):
    """Xóa các points thuộc về một file cụ thể."""
    try:
        store.delete_points_by_filename(file_name)
        return {"status": "success", "file_name": file_name}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/ingest/upload", response_model=IngestResponse, tags=["Ingestion"])
async def ingest_upload(
    file: UploadFile = File(...),
    recreate_collection: bool = Form(False),
    source_url: str = Form("", description="URL gốc của tài liệu trên web"),
    pipeline: IngestionPipeline = Depends(get_pipeline),
):
    """
    Ingest file PDF/DOCX từ client gửi lên (binary).
    """
    logger.info("POST /ingest/upload | file=%s | recreate=%s", file.filename, recreate_collection)
    
    # Save file temporarily
    temp_dir = os.path.join(BASE_DIR, "data", "temp_uploads")
    os.makedirs(temp_dir, exist_ok=True)
    temp_path = os.path.join(temp_dir, file.filename)
    
    with open(temp_path, "wb") as f:
        f.write(await file.read())
    
    try:
        if recreate_collection:
            store = get_vector_store()
            store.recreate_collection()
            get_vector_store.cache_clear()
            get_retriever.cache_clear()
            get_pipeline.cache_clear()

        result = pipeline.ingest(
            source=temp_path,
            is_directory=False,
            source_url=source_url or "",
        )

        # Cleanup temp file
        os.remove(temp_path)

        return IngestResponse(
            status=result["status"],
            documents_loaded=result["documents_loaded"],
            chunks_created=result["chunks_created"],
            chunks_stored=result["chunks_stored"],
            collection_info=result["collection_info"],
        )
    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise HTTPException(status_code=500, detail=str(e))
