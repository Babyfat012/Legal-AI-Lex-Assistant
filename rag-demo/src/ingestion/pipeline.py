import os
import time
from concurrent.futures import ThreadPoolExecutor
from ingestion.loading import DocumentLoader
from ingestion.chunking import ParentChildChunker
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
        chunker: ParentChildChunker = None,
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
        self.chunker = chunker or ParentChildChunker(
            parent_max_tokens=512,
            child_max_tokens=128,
        )
        self.embedding_service = embedding_service or EmbeddingService()
        default_bm25_path = os.path.join(BASE_DIR, "data", "bm25_vocab.json")

        self.bm25_encoder = bm25_encoder or BM25Encoder(
            vocab_path=default_bm25_path
        )
        self.vector_store = vector_store or QdrantVectorStore(
            dimension=self.embedding_service.dimension
        )

    
    def ingest(self, source: str, is_directory: bool = False, source_url: str = "") -> dict:
        """
        Chạy full pipeline: Load → Chunk → Embed → Store.

        Args:
            source: Đường dẫn file hoặc thư mục
            is_directory: True nếu source là thư mục
            source_url: URL gốc của tài liệu trên web (VD: https://thuvienphapluat.vn/...).
                Lưu vào Qdrant metadata để tạo highlight URL khi trả lời RAG.

        Returns:
            dict: Thống kê kết quả pipeline
        """
        t_total = time.perf_counter()
        logger.info("=" * 55)
        logger.info("INGESTION PIPELINE | source=%s | source_url=%s", source, source_url or "(none)")
        logger.info("=" * 55)

        # === 1. LOAD (includes pre-processing: PDF/DOCX → Markdown) ===
        logger.info("[1/5] LOAD — Reading & converting to Markdown...")
        t1 = time.perf_counter()
        if is_directory:
            documents = self.loader.load_directory(source, source_url=source_url)
        else:
            documents = self.loader.load_file(source, source_url=source_url)

        streaming = any(d.metadata.get("streaming_mode") for d in documents)
        load_mode = "streaming (page-by-page)" if streaming else "normal"
        logger.info(
            "[1/5] LOAD done | %d document(s) | mode=%s | %.2fs",
            len(documents), load_mode, time.perf_counter() - t1,
        )

        if not documents:
            logger.error("No documents loaded from source: %s", source)
            return {"status": "error", "message": "No documents loaded from source."}
        
        # === 2. CHUNK (Parent-Child) ===
        logger.info("[2/5] CHUNK — Parent-Child chunking...")
        t2 = time.perf_counter()
        chunk_result = self.chunker.chunk_documents(documents)
        chunks = chunk_result.children  # chỉ index children
        logger.info(
            "[2/5] CHUNK done | %d parents | %d children(indexed) | %.2fs",
            len(chunk_result.parents),
            len(chunks),
            time.perf_counter() - t2,
        )

        chunks_texts = [chunk.text for chunk in chunks]
        
        # === 3+4. BM25 SPARSE + DENSE EMBEDDING (song song) ===
        # Hai bước hoàn toàn độc lập nhau → chạy concurrent bằng ThreadPoolExecutor
        # Tiết kiệm thời gian bằng cách overlap I/O (OpenAI API) với CPU (BM25 encode)
        t34 = time.perf_counter()

        def _fit_and_encode_sparse() -> list:
            if not self.bm25_encoder._fitted:
                logger.info("[3/5] BM25 — Fitting vocabulary on new corpus (%d chunks)...", len(chunks_texts))
                self.bm25_encoder.fit(chunks_texts)
            else:
                logger.info("[3/5] BM25 — Using existing vocab (frozen) | vocab_size=%d", len(self.bm25_encoder.vocab))
            return self.bm25_encoder.encode_documents(chunks_texts)

        def _encode_dense() -> list:
            logger.info("[4/5] EMBED — Dense encoding with %s...", self.embedding_service.model)
            return self.embedding_service.embed_documents(chunks_texts)

        with ThreadPoolExecutor(max_workers=2) as executor:
            sparse_future = executor.submit(_fit_and_encode_sparse)
            dense_future  = executor.submit(_encode_dense)
            sparse_embeddings = sparse_future.result()
            dense_embeddings  = dense_future.result()

        logger.info(
            "[3+4/5] BM25 + EMBED done | sparse=%d | dense=%d | dim=%d | %.2fs",
            len(sparse_embeddings), len(dense_embeddings),
            len(dense_embeddings[0]) if dense_embeddings else 0,
            time.perf_counter() - t34,
        )

        # === 5. STORE (Save to Qdrant) ===
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
            "parents_created": len(chunk_result.parents),
            "chunks_created": len(chunks),
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
