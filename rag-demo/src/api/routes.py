import os
import time
from collections import defaultdict
from fastapi import APIRouter, HTTPException, Depends
from functools import lru_cache

from api.schemas import (
    ChatRequest,
    ChatResponse,
    SourceChunk,
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
def chat(
    request: ChatRequest,
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
        retriever.use_reranker = request.use_reranker
        retriever.final_top_n = request.top_k

        mode = "reasoning" if request.reasoning_mode else "standard"
        sub_queries: list[str] = []
        hyde_doc: str | None = None
        reasoning_steps: dict | None = None

        # FIX: ưu tiên server-side history (đáng tin cậy hơn client-sent history)
        # Client có thể không gửi chat_history, hoặc gửi thiếu — server tự lưu
        session_id = getattr(request, "session_id", None) or ""
        server_history = _get_history(session_id) if session_id else []

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
        answer, retrieved_chunks = generator.generate_pipeline(
            question=request.query,
            chat_history=chat_history_str,
            retriever=retriever,
        )

        # FIX: lưu turn mới vào server-side store sau khi generate thành công
        if session_id:
            _append_history(session_id, "user", request.query)
            _append_history(session_id, "assistant", answer)
            logger.debug(
                "History updated | session=%s | total_turns=%d",
                session_id,
                len(_conversation_store[session_id]),
            )

        # Populate sources
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
            }
            for chunk in retrieved_chunks
        ]

        logger.info(
            "POST /chat done | mode=%s | %.2fs",
            mode,
            time.perf_counter() - t0,
        )
        return ChatResponse(
            answer=answer,
            sources=sources,
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
