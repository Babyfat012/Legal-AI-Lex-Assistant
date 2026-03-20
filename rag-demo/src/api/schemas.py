from pydantic import BaseModel, Field
from typing import Optional, Any


# --- Request Schemas ---


class ChatRequest(BaseModel):
    query: str = Field(
        ..., description="Câu hỏi của người dùng", min_length=1, max_length=2000
    )
    session_id: Optional[str] = Field(
        default=None,
        description=(
            "ID phiên hội thoại. Dùng để server tự lưu và duy trì ngữ cảnh "
            "giữa các câu hỏi follow-up. Nếu None, mỗi request là độc lập."
        ),
    )
    top_k: int = Field(
        default=5, ge=1, le=20, description="Số chunks trả về sau reranking"
    )
    use_reranker: bool = Field(default=True, description="Có dùng reranker không")
    reasoning_mode: bool = Field(
        default=False,
        description=(
            "Bật chế độ phân tích pháp lý nâng cao (Legal Syllogism). "
            "Khi True: Query Decomposition + HyDE + Chain-of-Thought generation. "
            "Phù hợp với câu hỏi tranh chấp / tình huống / xác định lỗi."
        ),
    )
    chat_history: Optional[list[dict]] = Field(
        default_factory=list,
        description="Lịch sử hội thoại (dạng list các turn: user/assistant). Dùng làm fallback nếu server chưa có session.",
    )


class IngestRequest(BaseModel):
    """Request body cho endpoint /ingest."""

    file_path: str = Field(..., description="Đường dẫn tới file văn bản cần ingest")
    is_directory: bool = Field(
        default=False, description="True nếu file_path là thư mục"
    )
    recreate_collection: bool = Field(
        default=False, description="True để xóa collection cũ trước khi ingest"
    )


# --- Response Schemas ---


class SourceChunk(BaseModel):
    text: str = Field(..., description="Nội dung child chunk (đoạn match với query)")
    parent_content: Optional[str] = Field(
        None,
        description="Nội dung đầy đủ của Điều luật (parent chunk — dùng cho LLM generation)",
    )
    score: float = Field(..., description="RRF score từ hybrid search")
    rerank_score: Optional[float] = Field(
        None, description="Điểm rerank từ LLM (nếu có)"
    )
    luat: Optional[str] = Field(None, description="Tên văn bản luật (nếu có)")
    chuong: Optional[str] = Field(None, description="Chương")
    muc: Optional[str] = Field(None, description="Mục")
    dieu: Optional[str] = Field(None, description="Điều")
    filename: Optional[str] = Field(None, description="Tên file gốc")


class ChatResponse(BaseModel):
    answer: str = Field(..., description="Câu trả lời (hoặc phân tích pháp lý đầy đủ)")
    sources: list[SourceChunk] = Field(
        default_factory=list, description="Danh sách chunks nguồn đã sử dụng"
    )
    query: str = Field(..., description="Câu hỏi gốc của người dùng")
    mode: str = Field(
        default="standard",
        description="Chế độ xử lý: 'standard' hoặc 'reasoning'",
    )
    sub_queries: list[str] = Field(
        default_factory=list,
        description="Sub-queries được tạo ra từ Query Decomposition (chỉ có ở reasoning mode)",
    )
    hyde_doc: Optional[str] = Field(
        None,
        description="Hypothetical Document được tạo ra bởi HyDE (chỉ có ở reasoning mode)",
    )
    reasoning_steps: Optional[dict[str, Any]] = Field(
        None,
        description=(
            "Kết quả phân tích Tam đoạn luận có cấu trúc. "
            "Chỉ có ở reasoning mode. "
            "Gồm: hanh_vi, quy_dinh (list), doi_chieu, ket_luan."
        ),
    )


class IngestResponse(BaseModel):
    """Response body cho endpoint /ingest."""

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
