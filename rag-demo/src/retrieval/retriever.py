import time
from embedding.embedding import EmbeddingService
from embedding.bm25_en import BM25Encoder
from ingestion.qdrant_store import QdrantVectorStore
from retrieval.reranker import Reranker
from core.logger import get_logger

logger = get_logger(__name__)


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
            vector_store: QdrantVectorStore,
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
        logger.info(
            "Retriever initialized | initial_top_k=%d | final_top_n=%d | use_reranker=%s",
            initial_top_k, final_top_n, use_reranker,
        )

    

    def index_documents(self, texts: list[str]) -> None:
        """
        Index (nhúng + lưu) danh sách tài liệu vào vector store.
        
        Args:
            texts: Danh sách văn bản cần index
        """
        if not texts:
            logger.warning("index_documents called with empty texts list")
            return

        # 1. Embed tất cả documents (batch)
        embeddings = self.embedding_service.embed_documents(texts)

        # 2. Lưu vào vector store
        self.vector_store.add_documents(texts, embeddings)

        logger.info("Indexed %d documents successfully", len(texts))

    def retrieve(self, query: str) -> list[dict]:
        """
        Retrieve các chunks liên quan nhất với query.

        Args:
            query: Câu hỏi từ người dùng

        Returns:
            list[dict]: Top chunks, mỗi item gồm text + score + metadata
        """
        logger.info("Retrieval START | query: %.80s", query)
        t0 = time.perf_counter()

        # 1. Embed câu query (single text)
        query_dense = self.embedding_service.embed_query(query)
        query_sparse = self.bm25_encoder.encode_query(query)
        logger.debug(
            "Query embedded | dense_dim=%d | sparse_nnz=%d",
            len(query_dense), len(query_sparse.indices),
        )

        # 2. Hybrid search trên Qdrant: Kết hợp dense + sparse
        candidates = self.vector_store.hybrid_search(
            query_dense=query_dense,
            query_sparse=query_sparse,
            top_k=self.initial_top_k,
        )
        logger.info("Hybrid search → %d candidates", len(candidates))

        if not candidates:
            logger.warning("No candidates found — returning empty list")
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

        logger.info(
            "Retrieval END → %d results | total=%.2fs",
            len(results), time.perf_counter() - t0,
        )
        return results

    # ------------------------------------------------------------------
    # Advanced retrieval (Query Decomposition + HyDE)
    # ------------------------------------------------------------------

    def _raw_search(self, query: str) -> list[dict]:
        """
        Hybrid search không rerank — trả raw candidates.
        Dùng nội bộ bởi ``retrieve_advanced()``.
        """
        query_dense = self.embedding_service.embed_query(query)
        query_sparse = self.bm25_encoder.encode_query(query)
        return self.vector_store.hybrid_search(
            query_dense=query_dense,
            query_sparse=query_sparse,
            top_k=self.initial_top_k,
        )

    def retrieve_advanced(
        self,
        query: str,
        query_analyzer=None,
        use_hyde: bool = True,
        use_decomposition: bool = True,
    ) -> tuple[list[dict], list[str], str | None]:
        """
        Advanced retrieval kết hợp Query Decomposition + HyDE.

        Flow:
            1. Decompose query → sub_queries (nếu use_decomposition=True)
            2. Generate HyDE document   (nếu use_hyde=True)
            3. Raw hybrid search cho mỗi query trong pool
               (original + sub-queries + HyDE doc)
            4. Merge + dedup theo (filename, chunk_index),
               giữ score cao nhất nếu cùng chunk xuất hiện nhiều lần
            5. Rerank toàn bộ merged pool MỘT LẦN với query gốc
               → đảm bảo ranking phù hợp với intent ban đầu

        Args:
            query:            Câu hỏi gốc của người dùng.
            query_analyzer:   QueryAnalyzer instance (tạo mới nếu None).
            use_hyde:         Bật/tắt HyDE.
            use_decomposition: Bật/tắt Query Decomposition.

        Returns:
            tuple(results, sub_queries, hyde_doc)
                - results:     list[dict] chunks đã rerank, top ``final_top_n``
                - sub_queries: list[str]  sub-queries đã tạo ra
                - hyde_doc:    str | None hypothetical document đã tạo
        """
        from retrieval.query_analyzer import QueryAnalyzer  # lazy import tránh circular
        analyzer = query_analyzer or QueryAnalyzer()

        logger.info(
            "Advanced retrieval START | use_hyde=%s | use_decomp=%s | query: %.80s",
            use_hyde, use_decomposition, query,
        )
        t0 = time.perf_counter()

        # 1. Query Decomposition
        sub_queries: list[str] = []
        if use_decomposition:
            decomposed = analyzer.decompose(query)
            # Luôn giữ query gốc trong pool để không bỏ sót context trực tiếp
            sub_queries = decomposed
            if query not in sub_queries:
                sub_queries = [query] + sub_queries
        else:
            sub_queries = [query]

        # 2. HyDE generation
        hyde_doc: str | None = None
        if use_hyde:
            hyde_doc = analyzer.generate_hyde(query)

        # 3. Raw search cho toàn bộ search pool
        search_pool = list(sub_queries)
        if hyde_doc and hyde_doc != query:
            search_pool.append(hyde_doc)

        logger.info(
            "Search pool: %d queries (original + %d sub + %s HyDE)",
            len(search_pool),
            len(sub_queries) - 1,
            "1" if (hyde_doc and hyde_doc != query) else "0",
        )

        # 4. Merge + dedup — key = (filename, chunk_index)
        seen: dict[tuple, dict] = {}
        for q in search_pool:
            for c in self._raw_search(q):
                meta = c.get("metadata", {})
                key = (
                    meta.get("filename", ""),
                    meta.get("chunk_index", hash(c["text"])),
                )
                # Giữ candidate với score cao nhất nếu cùng chunk
                if key not in seen or c["score"] > seen[key]["score"]:
                    seen[key] = c

        merged = list(seen.values())
        logger.info(
            "Merged pool: %d unique candidates from %d searches",
            len(merged), len(search_pool),
        )

        if not merged:
            return [], sub_queries, hyde_doc

        # 5. Rerank một lần duy nhất với original query
        if self.use_reranker and len(merged) > self.final_top_n:
            results = self.reranker.rerank(
                query=query,
                candidates=merged,
                top_n=self.final_top_n,
            )
        else:
            results = sorted(merged, key=lambda x: x.get("score", 0), reverse=True)[
                : self.final_top_n
            ]

        logger.info(
            "Advanced retrieval END → %d results | total=%.2fs",
            len(results), time.perf_counter() - t0,
        )
        return results, sub_queries, hyde_doc

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