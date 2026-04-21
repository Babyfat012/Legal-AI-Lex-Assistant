import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from core.logger import get_logger

logger = get_logger(__name__)

# src/main.py → lên 1 cấp là rag-demo/ → đó là nơi chứa .env
_env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".env")
load_dotenv(_env_path)

logger.debug("Loading .env from: %s", os.path.abspath(_env_path))
logger.info("OPENAI_API_KEY set: %s", bool(os.getenv("OPENAI_API_KEY")))
logger.info("QDRANT_URL: %s", os.getenv("QDRANT_URL", "not set"))

from api.routes import router
from api.auth_routes import router as auth_router
from api.conversation_routes import router as conv_router
from api.docgen_routes import router as docgen_router
from core.db_init import init_all_databases

app = FastAPI(
    title="Lex - Legal AI Assistant",
    description="""
## Trợ lý pháp lý AI cho luật Việt Nam

### Pipeline
1. **Ingest** — Upload văn bản luật (PDF/DOCX) → chunk → embed → store vào Qdrant
2. **Chat** — Hybrid Search (BM25 + Vector) → Rerank → GPT generate

### Models
- **Embedding**: `text-embedding-3-small` (1536 dims)
- **LLM**: `gpt-4o-mini`
- **Vector DB**: Qdrant với RRF fusion
    """,
    version="1.0.0",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routes
app.include_router(router, prefix="/api/v1")
app.include_router(auth_router, prefix="/api/v1/auth")
app.include_router(conv_router, prefix="/api/v1")
app.include_router(docgen_router, prefix="/api/v1")

@app.get("/", tags=["System"])
def root():
    return {
        "name": "Lex - Legal AI Assistant",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/api/v1/health",
    }

if __name__ == "__main__":
    # Initialize both databases
    init_all_databases()
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )