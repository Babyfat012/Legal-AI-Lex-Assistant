from embedding.embedding import EmbeddingService
from embedding.bm25_en import BM25Encoder
from ingestion.qdrant_store import QdrantStore
from retrieval.reranker import Reranker


class Retriever:
    """
    Retriever kết hợp Hybrid Search + Reranking:

        Query
          ├── Dense Embedding (OpenAI)  ──→ Qdrant dense search  ─┐
          └── Sparse Embedding (BM25)   ──→ Qdrant sparse search ─┤
                                                                   ↓
                                                          RRF Fusion (top 20)
                                                                   ↓
                                                          Reranker (top 5)
    """

    def __init__(
            self, 
            embedding_service: EmbeddingService,
            bm25_encoder: BM25Encoder,
            vector_store: QdrantStore,
            reranker: Reranker = None,
            initial_top_k: int = 20,
            final_top_n: int = 5,
            use_reranker: bool = True, 
        ):
        """
        Args:
            initial_top_k: Số candidates lấy từ hybrid search
            final_top_n: Số kết quả cuối sau reranking
            use_reranker: Tắt reranker nếu cần tốc độ (chỉ dùng hybrid search)
        """
        self.embedding_service = embedding_service
        self.bm25_encoder = bm25_encoder
        self.vector_store = vector_store
        self.reranker = reranker or Reranker()
        self.initial_top_k = initial_top_k
        self.final_top_n = final_top_n
        self.use_reranker = use_reranker

    

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

    def retrieve(self, query: str) -> list[dict]:
        """
        Retrieve các chunks liên quan nhất với query.

        Args:
            query: Câu hỏi từ người dùng

        Returns:
            list[dict]: Top chunks, mỗi item gồm text + score + metadata
        """
        # 1. Embed câu query (single text)
        query_dense = self.embedding_service.embed_query(query)
        query_sparse = self.bm25_encoder.encode_query(query)

        # 2. Hybrid search trên Qdrant: Kết hợp dense + sparse
        candidates = self.vector_store.hybrid_search(
            query_dense=query_dense,
            query_sparse=query_sparse,
            top_k=self.initial_top_k,
        )

        if not candidates:
            return []
        
        # 3. Rerank nếu có candidates và đang bật reranker
        if self.use_reranker and len(candidates) > self.final_top_n:
            results = self.reranker.rerank(
                query=query,
                candidates=candidates,
                top_n=self.final_top_n,
            )
        else:
            results = candidates[:self.final_top_n]

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