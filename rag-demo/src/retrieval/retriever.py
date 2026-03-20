import time
from embedding.embedding import EmbeddingService
from embedding.bm25_en import BM25Encoder
from ingestion.qdrant_store import QdrantVectorStore
from retrieval.reranker import Reranker
from retrieval.query_analyzer import QueryAnalyzer
from core.logger import get_logger

logger = get_logger(__name__)


class Retriever:
    """
    Retriever kết hợp Hybrid Search + Query Expansion + Reranking:

        Query gốc    ──→ hybrid search (top 40) ─┐
        Query pháp lý ──→ hybrid search (top 40) ─┤
                                                  ↓
                                    Merge + dedup + boost expanded
                                                  ↓
                                    Sort by priority → top 30
                                                  ↓
                                   Pre-filter noise (min_score)
                                                  ↓
                                        Reranker (top 20)
                                                  ↓
                                   Dedup parent → top 5 Điều luật
    """

    # Ngưỡng score tối thiểu để đưa vào reranker.
    # Chunks dưới ngưỡng này thường là noise từ BLHS, Luật GT, v.v.
    # không liên quan đến câu hỏi dân sự/hợp đồng.
    # Nếu filter quá mạnh thì _MIN_CANDIDATES đảm bảo reranker vẫn có đủ input.
    _RERANK_MIN_SCORE: float = 0.15
    _RERANK_MIN_CANDIDATES: int = 10

    def __init__(
        self,
        embedding_service: EmbeddingService,
        bm25_encoder: BM25Encoder,
        vector_store: QdrantVectorStore,
        reranker: Reranker = None,
        initial_top_k: int = 40,
        final_top_n: int = 5,
        use_reranker: bool = True,
    ):
        self.embedding_service = embedding_service
        self.bm25_encoder = bm25_encoder
        self.vector_store = vector_store
        self.reranker = reranker or Reranker()
        self.initial_top_k = initial_top_k
        self.final_top_n = final_top_n
        self.use_reranker = use_reranker
        logger.info(
            "Retriever initialized | initial_top_k=%d | final_top_n=%d | use_reranker=%s",
            initial_top_k,
            final_top_n,
            use_reranker,
        )

    # ------------------------------------------------------------------
    # Query Expansion
    # ------------------------------------------------------------------

    def _expand_query(self, query: str) -> str:
        """
        Rewrite query sang thuật ngữ pháp lý để dense embedding
        match tốt hơn với corpus luật.
        """
        try:
            response = self.reranker.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "user",
                        "content": f"""Viết lại câu hỏi sau thành 1 câu ngắn dùng thuật ngữ pháp lý Việt Nam.
Chỉ trả về câu viết lại, không giải thích, không dấu ngoặc kép.

Ví dụ:
- "con tôi 15 tuổi ký hợp đồng được không" → "năng lực hành vi dân sự của người chưa thành niên khi xác lập giao dịch dân sự"
- "bị đánh có kiện được không" → "quyền khởi kiện dân sự khi bị xâm phạm thân thể sức khỏe"
- "16 tuổi bán xe máy có cần cha mẹ đồng ý không" → "năng lực hành vi dân sự người chưa thành niên xác lập giao dịch dân sự động sản phải đăng ký"

Câu hỏi: {query}

Câu viết lại:""",
                    }
                ],
                max_tokens=100,
                temperature=0.0,
            )
            expanded = response.choices[0].message.content.strip()
            logger.info(
                "Query expanded | original: %.60s | expanded: %.80s", query, expanded
            )
            return expanded
        except Exception as e:
            logger.warning("Query expansion failed: %s — dùng query gốc", e)
            return query

    # ------------------------------------------------------------------
    # Pre-filter trước reranker
    # ------------------------------------------------------------------

    def _prefilter_for_rerank(self, candidates: list[dict]) -> list[dict]:
        """
        Lọc bỏ candidates có priority_score thấp trước khi đẩy vào reranker.

        Mục đích: giảm noise trong context window của LLM reranker.
        Chunks từ BLHS, Luật GT, v.v. thường có score thấp (RRF tie ≈ 0.5)
        và làm loãng context, khiến LLM score sai các chunks BLDS thực sự liên quan.

        Luôn giữ ít nhất _RERANK_MIN_CANDIDATES candidates để reranker
        không bị thiếu input trong trường hợp corpus ít kết quả.
        """
        filtered = [
            c for c in candidates if c.get("score", 0.0) >= self._RERANK_MIN_SCORE
        ]

        # Safety net: nếu filter quá mạnh thì fallback về top-N theo score
        if len(filtered) < self._RERANK_MIN_CANDIDATES:
            filtered = candidates[: self._RERANK_MIN_CANDIDATES]
            logger.debug(
                "Pre-filter: result below min_candidates — using top-%d instead",
                self._RERANK_MIN_CANDIDATES,
            )

        logger.debug(
            "Pre-filter rerank: %d → %d candidates (min_score=%.2f)",
            len(candidates),
            len(filtered),
            self._RERANK_MIN_SCORE,
        )
        return filtered

    # ------------------------------------------------------------------
    # Core retrieval
    # ------------------------------------------------------------------

    def retrieve(
        self, query: str, top_n: int = None, use_reranker: bool = None
    ) -> list[dict]:
        """
        Retrieve các chunks liên quan nhất với query.

        Flow:
            1. Hybrid search với query gốc → 40 candidates
            2. Expand query → hybrid search → 40 candidates
            3. Merge 2 pools, boost score của expanded candidates
            4. Sort by priority → top 30
            5. Pre-filter noise theo min_score       ← MỚI
            6. Rerank với query gốc → top 20
            7. Dedup parent → top_n Điều luật
        """
        _top_n = top_n if top_n is not None else self.final_top_n
        _use_reranker = use_reranker if use_reranker is not None else self.use_reranker

        logger.info("Retrieval START | query: %.80s", query)
        t0 = time.perf_counter()

        # 1. Search với query gốc
        query_dense = self.embedding_service.embed_query(query)
        query_sparse = self.bm25_encoder.encode_query(query)
        logger.debug(
            "Query embedded | dense_dim=%d | sparse_nnz=%d",
            len(query_dense),
            len(query_sparse.indices),
        )
        original_candidates = self.vector_store.hybrid_search(
            query_dense=query_dense,
            query_sparse=query_sparse,
            top_k=self.initial_top_k,
        )

        # 2. Expand query và search
        expanded_query = self._expand_query(query)
        expanded_candidates = []
        if expanded_query and expanded_query != query:
            expanded_dense = self.embedding_service.embed_query(expanded_query)
            expanded_sparse = self.bm25_encoder.encode_query(expanded_query)
            logger.debug(
                "Expanded query embedded | sparse_nnz=%d", len(expanded_sparse.indices)
            )
            expanded_candidates = self.vector_store.hybrid_search(
                query_dense=expanded_dense,
                query_sparse=expanded_sparse,
                top_k=self.initial_top_k,
            )

        # 3. Merge 2 pools
        #    - Giữ score cao nhất nếu trùng chunk_id
        #    - Đánh dấu _in_expanded để boost priority
        seen: dict[str, dict] = {}

        for c in original_candidates:
            key = c["metadata"].get("chunk_id", c["text"][:80])
            seen[key] = {**c, "_in_expanded": False}

        for c in expanded_candidates:
            key = c["metadata"].get("chunk_id", c["text"][:80])
            if key not in seen:
                seen[key] = {**c, "_in_expanded": True}
            else:
                seen[key]["_in_expanded"] = True
                if c["score"] > seen[key]["score"]:
                    seen[key]["score"] = c["score"]

        all_candidates = list(seen.values())
        logger.info(
            "Merged candidates | original=%d + expanded=%d → unique=%d",
            len(original_candidates),
            len(expanded_candidates),
            len(all_candidates),
        )

        # Log BLDS candidates để debug
        blds_found = [
            c
            for c in all_candidates
            if "Bo_Luat_Dan_Su" in c["metadata"].get("filename", "")
        ]
        if blds_found:
            logger.debug("BLDS trong merged pool: %d chunks", len(blds_found))
            for c in blds_found[:3]:
                logger.debug(
                    "  BLDS | score=%.4f | expanded=%s | %s",
                    c["score"],
                    c.get("_in_expanded", False),
                    c["metadata"].get("dieu", "")[:60],
                )
        else:
            logger.warning(
                "BLDS KHÔNG có trong %d merged candidates!", len(all_candidates)
            )

        if not all_candidates:
            logger.warning("No candidates found — returning empty list")
            return []

        # 4. Sort by priority: expanded candidates được boost +0.15
        def priority_score(c: dict) -> float:
            base = c.get("score", 0.0)
            bonus = 0.15 if c.get("_in_expanded", False) else 0.0
            return base + bonus

        all_candidates.sort(key=priority_score, reverse=True)

        # 5. Lấy top 30 rồi pre-filter noise trước khi vào reranker
        rerank_input = all_candidates[:30]

        for i, c in enumerate(rerank_input[:5]):
            logger.debug(
                "Top-%d | score=%.4f | expanded=%s | %s",
                i + 1,
                c.get("score", 0),
                c.get("_in_expanded", False),
                c["text"][:100].replace("\n", " "),
            )

        # BUG FIX: pre-filter để loại chunks không liên quan (BLHS, Luật GT, v.v.)
        # trước khi đẩy vào reranker — tránh làm loãng context window của LLM
        if _use_reranker:
            rerank_input = self._prefilter_for_rerank(rerank_input)

        # 6. Rerank
        if _use_reranker and len(rerank_input) > _top_n:
            rerank_top_n = min(len(rerank_input), max(_top_n * 4, 20))
            scored = self.reranker.rerank(
                query=query,
                candidates=rerank_input,
                top_n=rerank_top_n,
            )
        else:
            scored = rerank_input

        # 7. Dedup theo parent_id → top _top_n Điều luật độc nhất
        results = self._deduplicate_by_parent(scored, max_parents=_top_n)

        logger.info(
            "Retrieval END → %d unique parents | total=%.2fs",
            len(results),
            time.perf_counter() - t0,
        )
        return results

    # ------------------------------------------------------------------
    # Advanced retrieval (Query Decomposition + HyDE)
    # ------------------------------------------------------------------

    def _raw_search(self, query: str) -> list[dict]:
        """Hybrid search không rerank — dùng nội bộ bởi retrieve_advanced()."""
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
        query_analyzer: QueryAnalyzer = None,
        use_hyde: bool = True,
        use_decomposition: bool = True,
        top_n: int = None,
        use_reranker: bool = None,
    ) -> tuple[list[dict], list[str], str | None]:
        """Advanced retrieval kết hợp Query Decomposition + HyDE."""
        _top_n = top_n if top_n is not None else self.final_top_n
        _use_reranker = use_reranker if use_reranker is not None else self.use_reranker
        analyzer = query_analyzer or QueryAnalyzer()

        logger.info(
            "Advanced retrieval START | use_hyde=%s | use_decomp=%s | query: %.80s",
            use_hyde,
            use_decomposition,
            query,
        )
        t0 = time.perf_counter()

        sub_queries: list[str] = []
        if use_decomposition:
            decomposed = analyzer.decompose(query)
            sub_queries = decomposed
            if query not in sub_queries:
                sub_queries = [query] + sub_queries
        else:
            sub_queries = [query]

        hyde_doc: str | None = None
        if use_hyde:
            hyde_doc = analyzer.generate_hyde(query)

        search_pool = list(sub_queries)
        if hyde_doc and hyde_doc != query:
            search_pool.append(hyde_doc)

        logger.info(
            "Search pool: %d queries (original + %d sub + %s HyDE)",
            len(search_pool),
            len(sub_queries) - 1,
            "1" if (hyde_doc and hyde_doc != query) else "0",
        )

        seen: dict[tuple, dict] = {}
        for q in search_pool:
            for c in self._raw_search(q):
                meta = c.get("metadata", {})
                key = (
                    meta.get("filename", ""),
                    meta.get("chunk_index", hash(c["text"])),
                )
                if key not in seen or c["score"] > seen[key]["score"]:
                    seen[key] = c

        merged = list(seen.values())
        logger.info(
            "Merged pool: %d unique candidates from %d searches",
            len(merged),
            len(search_pool),
        )

        if not merged:
            return [], sub_queries, hyde_doc

        # BUG FIX: áp dụng pre-filter cho advanced retrieval path luôn
        if _use_reranker:
            merged = self._prefilter_for_rerank(merged)

        rerank_pool_size = min(len(merged), max(_top_n * 4, 20))
        if _use_reranker and len(merged) > _top_n:
            scored = self.reranker.rerank(
                query=query, candidates=merged, top_n=rerank_pool_size
            )
        else:
            scored = sorted(merged, key=lambda x: x.get("score", 0), reverse=True)[
                :rerank_pool_size
            ]

        results = self._deduplicate_by_parent(scored, max_parents=_top_n)

        logger.info(
            "Advanced retrieval END → %d unique parents | total=%.2fs",
            len(results),
            time.perf_counter() - t0,
        )
        return results, sub_queries, hyde_doc

    def retrieve_with_context(self, query: str, top_k: int = 5) -> str:
        """Tìm tài liệu liên quan và ghép thành 1 context string."""
        results = self.retrieve(query, top_n=top_k)
        if not results:
            return "Không tìm thấy tài liệu liên quan."
        context_parts = [
            f"[{i}] (score: {doc['score']:.4f}) {doc.get('parent_content') or doc['text']}"
            for i, doc in enumerate(results, 1)
        ]
        return "\n\n".join(context_parts)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _deduplicate_by_parent(
        self, results: list[dict], max_parents: int
    ) -> list[dict]:
        """Khử trùng theo parent_id, giữ child score cao nhất mỗi Điều luật."""
        seen_parents: dict[str, dict] = {}
        for result in results:
            meta = result.get("metadata", {})
            parent_id = meta.get("parent_id") or meta.get("chunk_id", "")
            if parent_id not in seen_parents:
                seen_parents[parent_id] = result
                if len(seen_parents) >= max_parents:
                    break
        unique = list(seen_parents.values())
        logger.debug(
            "Dedup parent_id: %d candidates → %d unique parents",
            len(results),
            len(unique),
        )
        return unique
