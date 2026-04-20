from pydantic import BaseModel, Field
from typing import Optional, Any
from urllib.parse import quote


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
    language: str = Field(
        default="vi",
        description=(
            "Ngôn ngữ phản hồi: 'vi' (Tiếng Việt) hoặc 'en' (English). "
            "Khi language='en', LLM sẽ trả lời bằng tiếng Anh."
        ),
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
    source_url: Optional[str] = Field(
        default=None,
        description=(
            "URL gốc của tài liệu trên web (VD: https://thuvienphapluat.vn/van-ban/...). "
            "Được lưu vào metadata của mỗi vector trong Qdrant. "
            "Dùng để tạo Scroll-to-Text-Fragment URL khi trả lời RAG."
        ),
    )


# --- Response Schemas ---


class WebSource(BaseModel):
    """Nguồn web search — có URL highlight via Scroll to Text Fragment (#:~:text=)."""

    title: str = Field(..., description="Tiêu đề trang web")
    url: str = Field(..., description="URL gốc của trang")
    snippet: str = Field(..., description="Đoạn trích liên quan từ trang")
    highlight_url: str = Field(
        ...,
        description=(
            "URL đã nhúng Text Fragment (#:~:text=<encoded_snippet>) "
            "để trình duyệt tự scroll đến và highlight đoạn liên quan khi click."
        ),
    )

    @classmethod
    def build(cls, title: str, url: str, snippet: str) -> "WebSource":
        """Factory: tự động tạo highlight_url từ snippet."""
        # Lấy tối đa 120 ký tự đầu của snippet để tránh URL quá dài
        fragment_text = snippet[:120].strip()
        encoded = quote(fragment_text, safe="")
        highlight_url = f"{url}#:~:text={encoded}"
        return cls(title=title, url=url, snippet=snippet, highlight_url=highlight_url)


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
    source_url: Optional[str] = Field(
        None,
        description=(
            "URL gốc của tài liệu trên web. Nếu có, Chainlit sẽ tạo "
            "Scroll-to-Text-Fragment URL (#:~:text=<encoded_text>) "
            "để user click và highlight đoạn liên quan trực tiếp trên trang."
        ),
    )


class ChatResponse(BaseModel):
    answer: str = Field(..., description="Câu trả lời (hoặc phân tích pháp lý đầy đủ)")
    sources: list[SourceChunk] = Field(
        default_factory=list, description="Danh sách chunks nguồn đã sử dụng"
    )
    web_sources: list[WebSource] = Field(
        default_factory=list,
        description=(
            "Nguồn web search (chỉ có khi tool_used='web_search'). "
            "Mỗi source có highlight_url với #:~:text= fragment để trình duyệt "
            "tự scroll và highlight đoạn liên quan."
        ),
    )
    tool_used: str = Field(
        default="rag",
        description="Tool đã dùng: 'rag' hoặc 'web_search'",
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
