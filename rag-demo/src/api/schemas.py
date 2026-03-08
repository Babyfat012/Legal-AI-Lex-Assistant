from pydantic import BaseModel, Field
from typing import Optional

# --- Request Schemas ---
class ChatRequest(BaseModel):
    query: str = Field(..., description="Câu hỏi của người dùng", min_length=1, max_length=1000)
    top_k: int = Field(default=5, ge=1, le=20, description="Số chunks trả về sau reranking")
    use_reranker: bool = Field(default=True, description="Có dùng reranker không")

class IngestRequest(BaseModel):
    """
    Request body cho endpoint /ingest.
    """
    file_path: str = Field(..., description="Đường dẫn tới file văn bản cần ingest")
    is_directory: bool = Field(default=False, description="True nếu file_path là thư mục")
    recreate_collection: bool = Field(default=False, description="True để xóa collection cũ trước khi ingest")

# --- Response Schemas ---
class SourceChunk(BaseModel):
    text: str = Field(..., description="Nội dung chunk")
    score: float = Field(..., description="RRF score từ hybrid search")
    luat: Optional[str] = Field(None, description="Tên văn bản luật (nếu có)")
    chuong: Optional[str] = Field(None, description="Chương")
    muc: Optional[str] = Field(None, description="Mục")
    dieu: Optional[str] = Field(None, description="Điều")
    filename: Optional[str] = Field(None, description="Tên file gốc")

class ChatResponse(BaseModel):
    answer: str = Field(..., description="")
    sources: list[SourceChunk] = Field(default_factory=list, description="Danh sách chunks nguồn đã sử dụng để trả lời")
    query: str = Field(..., description="Câu hỏi gốc của người dùng")

class IngestResponse(BaseModel):
    """
    Response body cho endpoint /ingest.
    """
    status: str
    documents_loaded: int
    chunks_created: int
    chunks_stored: int
    collection_info: dict

class CollectionInfoResponse(BaseModel):
    name: str
    points_count: int
    indexed_vectors_count: int
    segments_count: int
    status: str
    dimension: int

class HealthResponse(BaseModel):
    status: str
    qdrant: str
    openai_key_set: bool