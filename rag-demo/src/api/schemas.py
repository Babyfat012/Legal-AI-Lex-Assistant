from pydantic import BaseModel, Field
from typing import List, Optional

class EmbedRequest(BaseModel):
    texts: List[str] = Field(..., description="Danh sách văn bản cần embedding")

class EmbedResponse(BaseModel):
    embeddings: List[List[float]] = Field(..., description="Danh sách embedding vectors")
    dimension: int = Field(..., description="Số chiều của 1 embedding vector")
    count: int = Field(..., description="Số lượng vectors")

class IndexRequest(BaseModel):
    texts: List[str] = Field(..., description="Danh sách documents cần index vào vector store")

class IndexResponse(BaseModel):
    message: str
    indexed_count: int
    total_documents: int


class RetrieveRequest(BaseModel):
    query: str = Field(..., description="Câu hỏi / truy vấn của user")
    top_k: int = Field(default=5, ge=1, le=20, description="Số lượng tài liệu liên quan cần trả về")

class RetrieveResult(BaseModel):
    text: str
    score: float
    index: int

class RetrieveResponse(BaseModel):
    query: str
    results: list[RetrieveResult]
    count: int

class GenerateRequest(BaseModel):
    question: str = Field(..., description="Câu hỏi của user")
    top_k: int = Field(default=5, ge=1, le=20, description="Số lượng tài liệu context")

class GenerateResponse(BaseModel):
    question: str
    answer: str
    context: str
    sources: List[RetrieveResult]
