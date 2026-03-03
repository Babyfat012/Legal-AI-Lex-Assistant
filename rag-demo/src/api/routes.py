from fastapi import APIRouter, HTTPException
from api.schemas import (
    EmbedRequest, EmbedResponse,
    IndexRequest, IndexResponse,
    RetrieveRequest, RetrieveResponse, RetrieveResult,
    GenerateRequest, GenerateResponse,
)
from embedding.embedding import EmbeddingService
from retrieval.vector_store import VectorStore
from retrieval.retriever import Retriever
from generator.llm_generator import LLMGenerator

# === Khởi tạo components (singleton) ===
embedding_service = EmbeddingService()
vector_store = VectorStore(dimension=1536)
retriever = Retriever(embedding_service, vector_store)
generator = LLMGenerator()

router = APIRouter()


@router.post("/embed", response_model=EmbedResponse, tags=["1. Embedding"],
             summary="Chuyển danh sách text thành embedding vectors")
def embed_texts(request: EmbedRequest):
    try:
        embeddings = embedding_service.embed_documents(request.texts)
        return EmbedResponse(
            embeddings=embeddings,
            dimension=len(embeddings[0]) if embeddings else 0,
            count=len(embeddings),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding error: {str(e)}")


@router.post("/index", response_model=IndexResponse, tags=["2. Indexing"],
             summary="Nạp tài liệu vào vector store (FAISS)")
def index_documents(request: IndexRequest):
    try:
        retriever.index_documents(request.texts)
        return IndexResponse(
            message="Documents indexed successfully.",
            indexed_count=len(request.texts),
            total_documents=vector_store.total_documents,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Indexing error: {str(e)}")


@router.post("/retrieve", response_model=RetrieveResponse, tags=["3. Retrieval"],
             summary="Tìm top-K tài liệu liên quan nhất")
def retrieve_documents(request: RetrieveRequest):
    try:
        results = retriever.retrieve(request.query, request.top_k)
        return RetrieveResponse(
            query=request.query,
            results=[RetrieveResult(**r) for r in results],
            count=len(results),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retrieval error: {str(e)}")


@router.post("/generate", response_model=GenerateResponse, tags=["4. RAG Generate"],
             summary="Full RAG: Retrieve + Generate câu trả lời")
def generate_answer(request: GenerateRequest):
    try:
        results = retriever.retrieve(request.question, request.top_k)
        context = retriever.retrieve_with_context(request.question, request.top_k)

        if not results:
            return GenerateResponse(
                question=request.question,
                answer="Không tìm thấy tài liệu liên quan trong cơ sở dữ liệu.",
                context="",
                sources=[],
            )

        answer = generator.generate(context, request.question)

        return GenerateResponse(
            question=request.question,
            answer=answer,
            context=context,
            sources=[RetrieveResult(**r) for r in results],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation error: {str(e)}")


@router.get("/stats", tags=["5. Utilities"], summary="Xem thống kê vector store")
def get_stats():
    return {
        "total_documents": vector_store.total_documents,
        "dimension": vector_store.dimension,
        "embedding_model": embedding_service.model_name,
        "llm_model": generator.model_name,
    }


@router.delete("/clear", tags=["5. Utilities"], summary="Xóa toàn bộ dữ liệu")
def clear_store():
    vector_store.clear()
    return {"message": "Vector store cleared.", "total_documents": 0}