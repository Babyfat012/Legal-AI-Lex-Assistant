import os 
import uuid 
from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams,
    SparseVectorParams,
    SparseIndexParams,
    Distance,
    PointStruct,
    SparseVector,
    Prefetch,
    FusionQuery,
    Fusion,
)
from ingestion.chunking import Chunk

DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"


class QdrantVectorStore:
    """
    Vector store hỗ trợ Hybrid Search:
        - Dense vector  : OpenAI text-embedding-3-small (cosine similarity)
        - Sparse vector : BM25 (exact term matching)
        - Fusion        : Reciprocal Rank Fusion (RRF)
    """

    def __init__(
            self,
            collection_name: str = "legal_documents",
            dimension: int = 1536,
            url: str = None,
    ):
        """
        Args: 
            collection_name: Tên collection trong Qdrant để lưu trữ vectors
            dimension: Dimension của embedding vector (phải khớp với model embedding)
            url: URL của Qdrant server (default: "http://localhost:6333")
        """
        self.collection_name = collection_name
        self.dimension = dimension
        self.url = url or os.getenv("QDRANT_URL", "http://localhost:6333")
        self.client = QdrantClient(url=self.url)
        self._ensure_collection()

    def _ensure_collection(self):
        """
        Tạo collection nếu chưa tồn tại.
        Nếu collection tồn tại nhưng sai schema (vd: dense-only cũ),
        tự động xóa và tạo lại với schema mới (named dense + sparse).
        """
        collections = [c.name for c in self.client.get_collections().collections]

        if self.collection_name in collections:
            info = self.client.get_collection(self.collection_name)
            existing_vectors = info.config.params.vectors

            # Kiểm tra schema: cần là dict có key "dense" (named vectors)
            # Nếu là VectorParams trực tiếp → schema cũ (unnamed), cần recreate
            is_old_schema = not isinstance(existing_vectors, dict) or \
                            DENSE_VECTOR_NAME not in existing_vectors

            if is_old_schema:
                print(f"  ⚠️  Collection '{self.collection_name}' có schema cũ (unnamed vectors).")
                print(f"  ⚠️  Đang xóa và tạo lại với schema mới (named dense + sparse)...")
                self.client.delete_collection(self.collection_name)
            else:
                print(f"  Using existing collection: '{self.collection_name}' ({info.points_count} points)")
                return

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config={
                DENSE_VECTOR_NAME: VectorParams(
                    size=self.dimension,
                    distance=Distance.COSINE,
                ),
            },
            sparse_vectors_config={
                SPARSE_VECTOR_NAME: SparseVectorParams(
                    index=SparseIndexParams()
                ),
            },
        )
        print(f"  Created collection '{self.collection_name}' (dense={self.dimension}d + sparse BM25)")

    
    def store_chunks(
            self, 
            chunks: list[Chunk],
            dense_embeddings: list[list[float]],
            sparse_embeddings: list[SparseVector],
            batch_size: int = 100,
    ) -> int:
        """
        Lưu chunks với cả dense + sparse embeddings.

        Args:
            chunks: Danh sách chunks
            dense_embeddings: Dense vectors từ OpenAI
            sparse_embeddings: Sparse vectors từ BM25Encoder
            batch_size: Số points mỗi batch
        """
        if not ((len(chunks) == len(dense_embeddings) == len(sparse_embeddings))):
            raise ValueError(
                f"Mismatch lengths: chunks={len(chunks)}, "
                f"dense={len(dense_embeddings)}, sparse={len(sparse_embeddings)}"
            )
        
        points = []
        for chunk, dense_vec, sparse_vec in zip(chunks, dense_embeddings, sparse_embeddings):
            points.append(
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector={
                        DENSE_VECTOR_NAME: dense_vec,
                        SPARSE_VECTOR_NAME: sparse_vec,
                    },
                    payload={
                        "text": chunk.text,
                        "chunk_index": chunk.chunk_index,
                        **chunk.metadata,
                    },
                )
            )

        # Upsert theo batch
        total_stored = 0
        for i in range(0, len(points), batch_size):
            batch = points[i: i + batch_size]
            self.client.upsert(
                collection_name = self.collection_name,
                points=batch,
            )
            total_stored += len(batch)
            print(f" Upserted batch {i // batch_size + 1}: {len(batch)} points")

        print(f"  Total stored: {total_stored} points into '{self.collection_name}'")
        return total_stored
    
    def hybrid_search(
        self,
        query_dense: list[float],
        query_sparse: SparseVector,
        top_k: int = 20,
    ) -> list[dict]:
        """
        Hybrid search: Dense + Sparse kết hợp bằng Reciprocal Rank Fusion.

        Args:
            query_dense: Dense embedding của query
            query_sparse: Sparse embedding của query
            top_k: Số kết quả trả về sau fusion

        Returns:
            list[dict]: Kết quả đã fusion, gồm text + score + metadata
        """
        results = self.client.query_points(
            collection_name=self.collection_name,
            prefetch=[
                # Nhánh 1: Dense vector search (semantic)
                Prefetch(
                    query=query_dense,
                    using=DENSE_VECTOR_NAME,
                    limit=top_k,
                ),
                # Nhánh 2: Sparse BM25 search (keyword)
                Prefetch(
                    query=query_sparse,
                    using=SPARSE_VECTOR_NAME,
                    limit=top_k,
                ),
            ],
            # Kết hợp 2 nhánh bằng RRF
            query=FusionQuery(fusion=Fusion.RRF),
            limit=top_k,
            with_payload=True,
        )

        return [
            {
                "text": point.payload.get("text", ""),
                "score": point.score,
                "metadata": {
                    k: v for k, v in point.payload.items() if k != "text"
                },
            }
            for point in results.points
        ]
    
    def vector_search(
            self, 
            query_dense: list[float],
            top_k: int = 20,
    ) -> list[dict]:
        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_dense,
            using=DENSE_VECTOR_NAME,
            limit=top_k,
            with_payload=True,
        )
        return [
            {
                "text": point.payload.get("text", ""),
                "score": point.score,
                "metadata": {
                    k: v for k, v in point.payload.items()
                },
            }
            for point in results.points
        ]



    def get_collection_info(self) -> dict:
        """
        Lấy thông tin collection hiện tại.
        """
        info = self.client.get_collection(self.collection_name)
        return {
            "name": self.collection_name,
            # "vectors_count": info.vectors_count,
            "points_count": info.points_count or 0,
            "status": info.status.value,
            "dimension": self.dimension,
        }
    
    def delete_collection(self):
        """
        Xóa collection hiện tại (cẩn thận khi dùng).
        """
        self.client.delete_collection(self.collection_name)
        print(f"Deleted collection: '{self.collection_name}'")

    def recreate_collection(self):
        """
        Xóa và tạo lại collection (dùng khi muốn reset dữ liệu).
        """
        self.delete_collection()
        self._ensure_collection()