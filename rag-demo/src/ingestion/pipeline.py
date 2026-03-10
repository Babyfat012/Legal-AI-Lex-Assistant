import os
import time
from pathlib import Path
from ingestion.loading import DocumentLoader
from ingestion.chunking import TextChunker
from ingestion.qdrant_store import QdrantVectorStore
from embedding.embedding import EmbeddingService
from embedding.bm25_en import BM25Encoder
from core.logger import get_logger

logger = get_logger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class IngestionPipeline:
    """
    Orchestrator cho toàn bộ ingestion pipeline:
        Load → Chunk → Embed → Store
    """
    
    def __init__(
        self,
        loader: DocumentLoader = None,
        chunker: TextChunker = None,
        embedding_service: EmbeddingService = None,
        bm25_encoder: BM25Encoder = None,
        vector_store: QdrantVectorStore = None,
        large_file_threshold_mb: float = 10.0,
        pages_per_batch: int = 20,
    ):
        self.loader = loader or DocumentLoader(
            large_file_threshold_mb=large_file_threshold_mb,
            pages_per_batch=pages_per_batch,
        )
        self.chunker = chunker or TextChunker(chunk_size=1024, chunk_overlap=100)
        self.embedding_service = embedding_service or EmbeddingService()
        default_bm25_path = os.path.join(BASE_DIR, "data", "bm25_vocab.json")

        self.bm25_encoder = bm25_encoder or BM25Encoder(
            vocab_path=default_bm25_path
        )
        self.vector_store = vector_store or QdrantVectorStore(
            dimension=self.embedding_service.dimension
        )

    
    def ingest(self, source: str, is_directory: bool = False) -> dict:
        """
        Chạy full pipeline: Load → Chunk → Embed → Store.

        Args:
            source: Đường dẫn file hoặc thư mục
            is_directory: True nếu source là thư mục

        Returns:
            dict: Thống kê kết quả pipeline
        """
        t_total = time.perf_counter()
        logger.info("=" * 55)
        logger.info("INGESTION PIPELINE | source=%s", source)
        logger.info("=" * 55)

        # === 1. LOAD (includes pre-processing: PDF/DOCX → Markdown) ===
        logger.info("[1/5] LOAD — Reading & converting to Markdown...")
        t1 = time.perf_counter()
        if is_directory:
            documents = self.loader.load_directory(source)
        else:
            documents = self.loader.load_file(source)

        streaming = any(d.metadata.get("streaming_mode") for d in documents)
        load_mode = "streaming (page-by-page)" if streaming else "normal"
        logger.info(
            "[1/5] LOAD done | %d document(s) | mode=%s | %.2fs",
            len(documents), load_mode, time.perf_counter() - t1,
        )

        if not documents:
            logger.error("No documents loaded from source: %s", source)
            return {"status": "error", "message": "No documents loaded from source."}
        
        # === 2. CHUNK (Recursive splitting by legal structure) ===
        logger.info("[2/5] CHUNK — Splitting by Chương > Điều > Khoản...")
        t2 = time.perf_counter()
        chunks = self.chunker.chunk_documents(documents)
        logger.info("[2/5] CHUNK done | %d chunks | %.2fs", len(chunks), time.perf_counter() - t2)

        # QC stats từ ChunkValidator
        qc_rejected = self.chunker.validator.rejected_count
        qc_total = self.chunker.validator.total_count
        if qc_rejected > 0:
            logger.warning(
                "[QC] Đã reject %d/%d chunks do thiếu metadata (chuong/dieu).",
                qc_rejected, qc_total,
            )

        chunks_texts = [chunk.text for chunk in chunks]
        
        # 3. FIT BM25 + ENCODE SPARSE
        logger.info("[3/5] BM25 — Fitting vocabulary + encoding sparse vectors...")
        t3 = time.perf_counter()
        self.bm25_encoder.fit(chunks_texts)
        sparse_embeddings = self.bm25_encoder.encode_documents(chunks_texts)
        logger.info("[3/5] BM25 done | %d sparse vectors | %.2fs", len(sparse_embeddings), time.perf_counter() - t3)

        
        # === 3. EMBED (OpenAI text-embedding-3-small) ===
        logger.info("[4/5] EMBED — Dense encoding with %s...", self.embedding_service.model)
        t4 = time.perf_counter()
        dense_embeddings = self.embedding_service.embed_documents(chunks_texts)
        logger.info("[4/5] EMBED done | %d vectors | dim=%d | %.2fs", len(dense_embeddings), len(dense_embeddings[0]), time.perf_counter() - t4)

        # === 4. STORE (Save to Qdrant) ===
        logger.info("[5/5] STORE — Saving to Qdrant '%s'...", self.vector_store.collection_name)
        t5 = time.perf_counter()
        stored_count = self.vector_store.store_chunks(
            chunks=chunks,
            dense_embeddings=dense_embeddings,
            sparse_embeddings=sparse_embeddings,
        )
        logger.info("[5/5] STORE done | %d points | %.2fs", stored_count, time.perf_counter() - t5)

        # === Final stats ===
        collection_info = self.vector_store.get_collection_info()
        summary = {
            "status": "success",
            "load_mode": load_mode,
            "documents_loaded": len(documents),
            "chunks_created": len(chunks),
            "chunks_rejected_by_qc": qc_rejected,
            "dense_embeddings": len(dense_embeddings),
            "sparse_embeddings": len(sparse_embeddings),
            "chunks_stored": stored_count,
            "collection_info": collection_info,
        }

        total_elapsed = time.perf_counter() - t_total
        logger.info("=" * 55)
        logger.info("PIPELINE COMPLETED | docs=%d | chunks=%d | stored=%d | total=%.2fs",
                    len(documents), len(chunks), stored_count, total_elapsed)
        logger.info("=" * 55)

        return summary
    
    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """
        Tìm kiếm hybrid: encode query (dense + sparse) → hybrid_search Qdrant.

        Args:
            query: Câu hỏi tìm kiếm
            top_k: Số kết quả trả về

        Returns:
            list[dict]: Kết quả gồm text, score, metadata
        """
        logger.info("Pipeline search | query: %.80s | top_k=%d", query, top_k)
        query_dense = self.embedding_service.embed_query(query)
        query_sparse = self.bm25_encoder.encode_query(query)
        results = self.vector_store.hybrid_search(query_dense, query_sparse, top_k=top_k)
        logger.info("Pipeline search → %d results", len(results))
        return results
    