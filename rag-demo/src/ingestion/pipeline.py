import os 
from pathlib import Path
from ingestion.loading import DocumentLoader
from ingestion.chunking import TextChunker
from ingestion.qdrant_store import QdrantVectorStore
from embedding.embedding import EmbeddingService
from embedding.bm25_en import BM25Encoder

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
    ):
        self.loader = loader or DocumentLoader()
        self.chunker = chunker or TextChunker(chunk_size=1024, chunk_overlap=100)
        self.embedding_service = embedding_service or EmbeddingService()
        default_bm25_path = os.path.join(BASE_DIR, "data", "bm25_vocab.json"),

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
        print(f"\n{'='*60}")
        print(f"  INGESTION PIPELINE")
        print(f"  Source: {source}")
        print(f"{'='*60}\n")

        # === 1. LOAD (includes pre-processing: PDF/DOCX → Markdown) ===
        print("[1/5] LOAD - Reading & converting to Markdown...")
        if is_directory:
            documents = self.loader.load_directory(source)
        else:
            documents = self.loader.load_file(source)
        print(f"  -> Loaded {len(documents)} documents.\n")

        if not documents:
            return {"status": "error", "message": "No documents loaded from source."}
        
        # === 2. CHUNK (Recursive splitting by legal structure) ===
        print("[2/5] CHUNK - Splitting by Chương > Điều > Khoản...")
        chunks = self.chunker.chunk_documents(documents)
        print(f"  -> Created {len(chunks)} chunks.\n")

        chunks_texts = [chunk.text for chunk in chunks]
        
        # 3. FIT BM25 + ENCODE SPARSE
        print("[3/5] FIT BM25 - Building vocabulary from chunks...")
        self.bm25_encoder.fit(chunks_texts)
        sparse_embeddings = self.bm25_encoder.encode_documents(chunks_texts)
        print(f"  → {len(sparse_embeddings)} sparse vectors encoded\n")

        
        # === 3. EMBED (OpenAI text-embedding-3-small) ===
        print(f"[4/5] EMBED - Dense encoding with {self.embedding_service.model}...")
        dense_embeddings = self.embedding_service.embed_documents(chunks_texts)
        print(f"  → {len(dense_embeddings)} dense vectors (dim={len(dense_embeddings[0])})\n")

        # === 4. STORE (Save to Qdrant) ===
        print(f"[5/5] STORE - Saving to Qdrant '{self.vector_store.collection_name}'...")
        stored_count = self.vector_store.store_chunks(
            chunks=chunks,
            dense_embeddings=dense_embeddings,
            sparse_embeddings=sparse_embeddings,
        )

        # === Final stats ===
        collection_info = self.vector_store.get_collection_info()
        summary = {
            "status": "success",
            "documents_loaded": len(documents),
            "chunks_created": len(chunks),
            "dense_embeddings": len(dense_embeddings),
            "sparse_embeddings": len(sparse_embeddings),
            "chunks_stored": stored_count,
            "collection_info": collection_info,
        }

        print(f"\n{'='*60}")
        print(f"  PIPELINE COMPLETED")
        print(f"  Documents : {len(documents)}")
        print(f"  Chunks    : {len(chunks)}")
        print(f"  Stored    : {stored_count} points (dense + sparse)")
        print(f"{'='*60}\n")

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
        print(f"  Query: '{query}'")
        query_dense = self.embedding_service.embed_query(query)
        query_sparse = self.bm25_encoder.encode_query(query)
        results = self.vector_store.hybrid_search(query_dense, query_sparse, top_k=top_k)
        print(f"  Found {len(results)} results")
        return results
    