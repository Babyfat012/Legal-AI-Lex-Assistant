from retrieval.vector_store import VectorStore
from embedding.embedding import EmbeddingService


class Retriever:
    """
    Orchestrator kết nối EmbeddingService + VectorStore.
    
    Luồng hoạt động:
        index_documents(): texts → embed → store vào FAISS
        retrieve():        query → embed → search FAISS → trả top-K docs
    """

    def __init__(self, embedding_service: EmbeddingService, vector_store: VectorStore):
        """
        Args:
            embedding_service: Service để chuyển text → vector
            vector_store: Kho lưu trữ và tìm kiếm vector (FAISS)
        """
        self.embedding_service = embedding_service
        self.vector_store = vector_store

    def index_documents(self, texts: list[str]) -> None:
        """
        Index (nhúng + lưu) danh sách tài liệu vào vector store.
        
        Args:
            texts: Danh sách văn bản cần index
        """
        if not texts:
            print("Warning: No texts to index.")
            return

        # 1. Embed tất cả documents (batch)
        embeddings = self.embedding_service.embed_documents(texts)

        # 2. Lưu vào vector store
        self.vector_store.add_documents(texts, embeddings)

        print(f"Indexed {len(texts)} documents successfully.")

    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        """
        Tìm top-K tài liệu liên quan nhất với câu query.
        
        Args:
            query: Câu hỏi / truy vấn của user
            top_k: Số kết quả trả về
            
        Returns:
            List[dict]: Mỗi dict gồm {"text": str, "score": float, "index": int}
        """
        # 1. Embed câu query (single text)
        query_embedding = self.embedding_service.embed_text(query)

        # 2. Tìm kiếm trong vector store
        results = self.vector_store.retrieve_similar(query_embedding, top_k)

        return results

    def retrieve_with_context(self, query: str, top_k: int = 5) -> str:
        """
        Tìm tài liệu liên quan và ghép thành 1 context string 
        (tiện để truyền thẳng vào prompt cho LLM).
        
        Args:
            query: Câu hỏi của user
            top_k: Số tài liệu trả về
            
        Returns:
            str: Context string từ các tài liệu liên quan
        """
        results = self.retrieve(query, top_k)

        if not results:
            return "Không tìm thấy tài liệu liên quan."

        context_parts = []
        for i, doc in enumerate(results, 1):
            context_parts.append(f"[{i}] (score: {doc['score']:.4f}) {doc['text']}")

        return "\n\n".join(context_parts)