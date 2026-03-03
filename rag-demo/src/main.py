import uvicorn
from fastapi import FastAPI
from api.routes import router

app = FastAPI(
    title="RAG Demo - Legal AI",
    description=(
        "## Demo RAG Pipeline gồm 3 phần chính:\n"
        "1. **Embedding** — Chuyển text → vector\n"
        "2. **Retrieval** — Tìm tài liệu liên quan bằng FAISS\n"
        "3. **Generation** — Sinh câu trả lời bằng LLM\n\n"
        "### Hướng dẫn test:\n"
        "1. Gọi `/index` để nạp tài liệu\n"
        "2. Gọi `/retrieve` để test tìm kiếm\n"
        "3. Gọi `/generate` để chạy full pipeline RAG"
    ),
    version="1.0.0",
)

app.include_router(router, prefix="/api/v1")


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)