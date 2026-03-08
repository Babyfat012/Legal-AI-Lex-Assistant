import os 
from fastapi import APIRouter, HTTPException, Depends
from functools import lru_cache

from api.schemas import (
    ChatRequest, ChatResponse, SourceChunk,
    IngestRequest, IngestResponse,
    CollectionInfoResponse,
    HealthResponse,
)
from ingestion.pipeline import IngestionPipeline
from ingestion.qdrant_store import QdrantVectorStore
from embedding.embedding import EmbeddingService
from embedding.bm25_en import BM25Encoder
from retrieval.retriever import Retriever
from retrieval.reranker import Reranker
from generator.llm_generator import LLMGenerator
import traceback
import logging

logging = logging.getLogger(__name__)

router = APIRouter()

# Base dir = src/
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# --- Dependency: Services (singleton via lru_cache) ---
@lru_cache(maxsize=1)
def get_embedding_service() -> EmbeddingService:
    return EmbeddingService()



@lru_cache(maxsize=1)
def get_bm25_encoder() -> BM25Encoder:
    bm25_path = os.getenv(
        "BM25_VOCAB_PATH",
        os.path.join(BASE_DIR, "data", "bm25_vocab.json"),
        )
    encoder = BM25Encoder(vocab_path=bm25_path)
    # Load vocab nếu đã fit trước đó
    if os.path.exists(bm25_path):
        encoder.load(bm25_path)
    return encoder

@lru_cache(maxsize=1)
def get_vector_store() -> QdrantVectorStore:
    embedding_service = get_embedding_service()
    return QdrantVectorStore(
        collection_name=os.getenv("QDRANT_COLLECTION", "legal_documents"),
        dimension=embedding_service.dimension,
        url=os.getenv("QDRANT_URL", "http://localhost:6333"),
    )

@lru_cache(maxsize=1)
def get_retriever() -> Retriever:
    return Retriever(
        embedding_service=get_embedding_service(),
        bm25_encoder=get_bm25_encoder(),
        vector_store=get_vector_store(),
        reranker=Reranker(),
        initial_top_k=20,
        final_top_n=5,
        use_reranker=True,
    )

@lru_cache(maxsize=1)
def get_generator() -> LLMGenerator:
    return LLMGenerator()

@lru_cache(maxsize=1)
def get_pipeline() -> IngestionPipeline:
    embedding_service = get_embedding_service()
    bm25_path = os.getenv("BM25_VOCAB_PATH", "data/bm25_vocab.json")
    return IngestionPipeline(
        embedding_service=embedding_service,
        bm25_encoder=BM25Encoder(vocab_path=bm25_path),
        vector_store=get_vector_store(),
    )


# --- Endpoints ---
@router.get("/health", response_model=HealthResponse, tags=["System"])
def health_check():
    """Kiểm tra trạng thái hệ thống."""
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
    RAG pipeline: Query → Hybrid Search → Rerank → Generate.

    1. Embed query (dense + sparse BM25)
    2. Hybrid search trong Qdrant (RRF fusion)
    3. Rerank top candidates
    4. Generate câu trả lời với GPT
    """
    try:
        # Override retriever config theo request
        retriever.use_reranker = request.use_reranker
        retriever.final_top_n = request.top_k

        # 1. Retrieve
        chunks = retriever.retrieve(request.query)

        if not chunks:
            return ChatResponse(
                answer="Tôi không tìm thấy thông tin liên quan trong các văn bản pháp luật được cung cấp.",
                sources=[],
                query=request.query,
            )
        
        # 2. Generate
        answer = generator.generate(
            query=request.query,
            context_chunks=chunks,
        )

        # 3. Format sources
        sources = [
            SourceChunk(
                text=chunk["text"],
                score=chunk.get("score", 0.0),
                rerank_score=chunk.get("rerank_score"),
                luat=chunk["metadata"].get("luat"),
                chuong=chunk["metadata"].get("chuong"),
                muc=chunk["metadata"].get("muc"),
                dieu=chunk["metadata"].get("dieu"),
                filename=chunk["metadata"].get("filename"),
            )
            for chunk in chunks
        ]

        return ChatResponse(
            answer=answer,
            sources=sources,
            query=request.query,
        )
    
    except RuntimeError as e:
        logging.error(f"Runtime error during /chat: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        # Log full traceback for unexpected errors
        logging.error(f"Error in /chat:\n{traceback.format_exc()}")
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
    if not os.path.exists(request.file_path):
        raise HTTPException(
            status_code=400,
            detail=f"File not found: {request.file_path}",
        )
    
    try:
        if request.recreate_collection:
            store = get_vector_store()
            store.recreate_collection()
            # Clear lru_cache để reinitialize
            get_vector_store.cache_clear()
            get_retriever.cache_clear()
            get_pipeline.cache_clear()

        result = pipeline.ingest(
            source=request.file_path,
            is_directory=request.is_directory,
        )

        # Reload BM25 encoder trong retriever sau khi ingest
        get_bm25_encoder.cache_clear()

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