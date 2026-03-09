# Legal AI — Lex Assistant (RAG Demo)

Legal AI - Lex Assistant is a cloud-native Law Question Answering system built with Retrieval-Agumented Generation (RAG) architecture. The system's UI is built by Streamlit, and FastAPI, a vector search for knowledge retrieval, and an inference server, with Redis as a cached chat history for multi-run conversations. This system is deployed on Kubernetes, automated by Terraform, Helm and Jenkins CI/CD, fully observed and monitored using Prometheus, Grafana, ELK stacks (Elasicsearch, Logstack, Kibana) and tracing with Jaeger.

**Tech stack:** FastAPI · Qdrant (Hybrid Search RRF) · OpenAI `text-embedding-3-small` + `gpt-4o-mini` · BM25 · Python 3.12

---

## Kiến trúc

```
┌──────────────────────────────────────────────────────────────────────┐
│                         INGESTION PIPELINE                           │
│                                                                      │
│  PDF/DOCX → [Preprocess] → [Load] → [Chunk] → [Embed] → [Store]     │
│              Markdown      Document  Chunks   Dense+Sparse  Qdrant   │
└──────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────┐
│                           QUERY PIPELINE                             │
│                                                                      │
│  Query → [Embed] → [Hybrid Search (RRF)] → [Rerank] → [Generate]    │
│          Dense+Sparse   Qdrant native        LLM score  gpt-4o-mini  │
└──────────────────────────────────────────────────────────────────────┘
```

### Các thành phần chính

| Module | File | Mô tả |
|--------|------|-------|
| **Logger** | `core/logger.py` | Centralized logger: màu ANSI console + rotate file `logs/lex_assistant.log` |
| **Pre-processing** | `ingestion/preprocessing.py` | Chuyển PDF/DOCX → Markdown, chuẩn hóa headings `## Chương`, `#### Điều` |
| **Loading** | `ingestion/loading.py` | Đọc file → `Document(text, metadata)` |
| **Chunking** | `ingestion/chunking.py` | `RecursiveCharacterTextSplitter` với legal separators, inject prefix `[Luật - Chương - Điều]` |
| **Embedding** | `embedding/embedding.py` | OpenAI `text-embedding-3-small` (1536 dims), batch embed |
| **BM25** | `embedding/bm25_en.py` | Sparse encoder TF-IDF cho keyword matching, persist vocab JSON |
| **Vector Store** | `ingestion/qdrant_store.py` | Qdrant: named dense + sparse vectors, Hybrid Search RRF native |
| **Retriever** | `retrieval/retriever.py` | Orchestrate: embed query → hybrid search → rerank |
| **Reranker** | `retrieval/reranker.py` | Batch LLM scoring (`gpt-4o-mini`): 1 call cho tất cả candidates |
| **Generator** | `generator/llm_generator.py` | RAG generation với `gpt-4o-mini`, system prompt pháp luật VN |
| **API** | `api/routes.py` | FastAPI endpoints: `/health`, `/chat`, `/ingest`, `/collection/info` |
| **Pipeline** | `ingestion/pipeline.py` | Orchestrator: Load → Chunk → BM25 fit → Embed → Store |

---

## Cấu trúc thư mục

```
rag-demo/
├── .env                        # API keys và config
├── requirements.txt
├── logs/
│   └── lex_assistant.log       # Log file (rotate 5 MB × 3, tự động tạo)
├── documents/                  # Đặt file PDF/DOCX tại đây
├── infra/
│   └── docker-compose.yaml     # Qdrant container
└── src/
    ├── main.py                 # FastAPI app entry point
    ├── test-ingestion.py       # Script test pipeline
    ├── core/
    │   └── logger.py           # Centralized logger (get_logger)
    ├── api/
    │   ├── routes.py           # API endpoints
    │   └── schemas.py          # Pydantic Request/Response schemas
    ├── data/
    │   └── bm25_vocab.json     # BM25 vocabulary (tự động tạo sau ingest)
    ├── embedding/
    │   ├── embedding.py        # OpenAI EmbeddingService
    │   └── bm25_en.py          # BM25Encoder (fit + encode + persist)
    ├── ingestion/
    │   ├── preprocessing.py    # PDF/DOCX → Markdown
    │   ├── loading.py          # File → Document
    │   ├── chunking.py         # Document → Chunks (legal-aware)
    │   ├── qdrant_store.py     # Qdrant vector store (hybrid search RRF)
    │   └── pipeline.py         # Ingestion orchestrator
    ├── retrieval/
    │   ├── retriever.py        # Hybrid search + rerank pipeline
    │   └── reranker.py         # Batch LLM reranker
    └── generator/
        └── llm_generator.py    # RAG answer generation
```

---

## Cài đặt

### Yêu cầu

- Python 3.12+
- Docker
- `uv` (package manager)
- OpenAI API key

### 1. Clone & cài dependencies

```bash
cd rag-demo
uv venv .venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

### 2. Tạo file `.env`

```bash
cp ../example.env .env
```

Điền các giá trị:

```env
OPENAI_API_KEY=sk-...
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION=legal_documents  # optional, default: legal_documents
```

### 3. Khởi động Qdrant

```bash
cd infra
docker compose up -d
```

Kiểm tra Qdrant:
```bash
curl http://localhost:6333/healthz
# Dashboard: http://localhost:6333/dashboard
```

---

## Sử dụng

### Ingest tài liệu

Đặt file PDF/DOCX vào thư mục `documents/`, sau đó gọi API hoặc chạy script test:

```bash
cd src

# Test pipeline (không cần API key — chỉ load + chunk)
python test-ingestion.py ../documents/luat_vd.pdf

# Full pipeline (cần OPENAI_API_KEY + Qdrant)
python test-ingestion.py ../documents/luat_vd.pdf --full
```

Log output của full pipeline:
```
14:30:00 [INFO    ] ingestion.pipeline — ═══════════════════════════════════════════════════════
14:30:00 [INFO    ] ingestion.pipeline — INGESTION PIPELINE | source=../documents/luat_vd.pdf
14:30:00 [INFO    ] ingestion.pipeline — [1/5] LOAD — Reading & converting to Markdown...
14:30:01 [INFO    ] ingestion.pipeline — [1/5] LOAD done | 1 documents | 0.84s
14:30:01 [INFO    ] ingestion.pipeline — [2/5] CHUNK — Splitting by Chương > Điều > Khoản...
14:30:01 [INFO    ] ingestion.pipeline — [2/5] CHUNK done | 312 chunks | 0.12s
14:30:01 [INFO    ] ingestion.pipeline — [3/5] BM25 — Fitting vocabulary + encoding sparse vectors...
14:30:01 [INFO    ] ingestion.pipeline — [3/5] BM25 done | 312 sparse vectors | 0.43s
14:30:01 [INFO    ] ingestion.pipeline — [4/5] EMBED — Dense encoding with text-embedding-3-small...
14:30:07 [INFO    ] ingestion.pipeline — [4/5] EMBED done | 312 vectors | dim=1536 | 5.91s
14:30:07 [INFO    ] ingestion.pipeline — [5/5] STORE — Saving to Qdrant 'legal_documents'...
14:30:08 [INFO    ] ingestion.pipeline — PIPELINE COMPLETED | docs=1 | chunks=312 | stored=312 | total=7.31s
```

### Chạy API server

```bash
cd src
uvicorn main:app --reload --port 8000
```

Swagger UI: **http://localhost:8000/docs**

---

## API Endpoints

| Method | Endpoint | Mô tả |
|--------|----------|-------|
| `GET` | `/api/v1/health` | Kiểm tra trạng thái hệ thống (Qdrant + OpenAI key) |
| `POST` | `/api/v1/chat` | Full RAG: Hybrid Search → Rerank → Generate câu trả lời |
| `POST` | `/api/v1/ingest` | Ingest PDF/DOCX vào Qdrant |
| `GET` | `/api/v1/collection/info` | Thông tin collection (số points, status, dim) |
| `DELETE` | `/api/v1/collection` | Xóa toàn bộ collection |

### Request / Response

**`POST /api/v1/chat`**
```json
// Request
{
  "query": "Điều kiện để được hưởng trợ cấp thất nghiệp là gì?",
  "top_k": 5,
  "use_reranker": true
}

// Response
{
  "answer": "Theo Điều 49 Luật Việc làm...",
  "sources": [
    {
      "text": "Điều 49. Điều kiện hưởng trợ cấp...",
      "score": 0.87,
      "luat": "Luật Việc làm 2013",
      "dieu": "Điều 49"
    }
  ],
  "query": "Điều kiện để được hưởng trợ cấp thất nghiệp là gì?"
}
```

**`POST /api/v1/ingest`**
```json
// Request
{
  "file_path": "/app/documents/luat_viec_lam.pdf",
  "is_directory": false,
  "recreate_collection": false
}
```

### Ví dụ curl

```bash
# Health check
curl http://localhost:8000/api/v1/health

# Chat
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "Xe máy được phép chạy tối đa bao nhiêu km/h trong khu dân cư?", "top_k": 5}'

# Ingest thư mục
curl -X POST http://localhost:8000/api/v1/ingest \
  -H "Content-Type: application/json" \
  -d '{"file_path": "/app/documents/", "is_directory": true}'
```

---

## Logging

Toàn bộ codebase dùng logger tập trung tại `src/core/logger.py`.

### Output format

```
HH:MM:SS [LEVEL   ] module.submodule — message
```

- **Console**: màu ANSI theo level (DEBUG=cyan, INFO=green, WARNING=yellow, ERROR=red)
- **File**: `logs/lex_assistant.log` — plain text, rotate tự động 5 MB × 3 files

### Ví dụ log một RAG request

```
14:32:01 [INFO    ] api.routes — POST /chat | query: Điều kiện để được hưởng trợ cấp thất nghiệp...
14:32:01 [INFO    ] retrieval.retriever — Retrieval START | query: Điều kiện để được hưởng...
14:32:01 [DEBUG   ] retrieval.retriever — Query embedded | dense_dim=1536 | sparse_nnz=12
14:32:01 [DEBUG   ] ingestion.qdrant_store — Hybrid search | top_k=20 | dense_limit=60 | sparse_limit=60
14:32:01 [INFO    ] ingestion.qdrant_store — Hybrid search → 20 results | 0.041s
14:32:01 [INFO    ] retrieval.reranker — Reranking 20 candidates → top 5 | query: Điều kiện...
14:32:02 [INFO    ] retrieval.reranker — LLM batch score done in 0.91s | tokens: prompt=1823 completion=40 total=1863
14:32:02 [INFO    ] retrieval.reranker — Reranking done in 0.92s | top scores: ['0.90', '0.85', '0.80', '0.70', '0.60']
14:32:02 [INFO    ] retrieval.retriever — Retrieval END → 5 results | total=1.02s
14:32:02 [INFO    ] generator.llm_generator — Generating answer | context_chunks=5
14:32:03 [INFO    ] generator.llm_generator — Generation done in 1.23s | tokens: prompt=2100 completion=312 total=2412
```

### Dùng trong module mới

```python
from core.logger import get_logger

logger = get_logger(__name__)   # name = "my_module.submodule"

logger.debug("Chi tiết nội bộ: %s", value)
logger.info("Bước hoàn thành | count=%d | elapsed=%.2fs", n, t)
logger.warning("Dữ liệu thiếu: %s", field)
logger.error("Lỗi: %s", err, exc_info=True)
```

---

## Chi tiết kỹ thuật

### Chunking — Legal-aware splitting

`RecursiveCharacterTextSplitter` với separators theo cấu trúc luật VN:

```
Ưu tiên cắt: # PHẦN → ## Chương → ### Mục → #### Điều → Khoản → Điểm → \n → câu → từ
```

Mỗi chunk được inject context prefix để LLM biết nguồn gốc:
```
[Luật Đường bộ - Chương II - Điều 8. Quy tắc chung] Người tham gia giao thông...
```

### Hybrid Search — Qdrant Native RRF

```
Query
  ├── Dense (text-embedding-3-small) ──→ Qdrant cosine   (prefetch 60) ─┐
  └── Sparse (BM25)                  ──→ Qdrant keyword  (prefetch 60) ─┤
                                                                         ↓
                                              Reciprocal Rank Fusion (Qdrant native)
                                                       top 20 candidates
                                                                         ↓
                                                  LLM Batch Reranker (gpt-4o-mini)
                                                        1 call / request
                                                           top 5 results
```

| | Dense | Sparse (BM25) |
|--|--|--|
| **Tốt cho** | Semantic similarity ("xe cơ giới" ~ "phương tiện giao thông") | Exact match ("Điều 8", "khoản 2", tên luật cụ thể) |
| **Vector type** | `float[]` 1536 dims, cosine | Sparse `{index: weight}`, dot product |
| **Prefetch size** | `top_k × 3` | `top_k × 3` |

**RRF** kết hợp rank từ 2 nhánh không phụ thuộc score scale — hoàn toàn xử lý trong Qdrant engine, không cần post-processing.

### Reranker — Batch LLM Scoring

Thay vì gọi LLM `N` lần (1 call/chunk), reranker gộp tất cả candidates vào **1 prompt duy nhất**:

```
Đánh giá mức độ liên quan của từng đoạn với câu hỏi.
Trả về ĐÚNG 20 số (0-10), mỗi số trên 1 dòng...

[1] Điều 49. Điều kiện hưởng trợ cấp thất nghiệp...
[2] Điều 50. Mức hưởng trợ cấp thất nghiệp...
...
```

| | N calls (cũ) | 1 batch call (hiện tại) |
|--|--|--|
| **LLM calls** | N | 1 |
| **Latency** | O(N) × latency/call | O(1) × 1 call |
| **Cost tokens** | N × (system + chunk) | 1 × (system + all chunks) |

---

## Biến môi trường

| Biến | Bắt buộc | Mô tả |
|------|----------|-------|
| `OPENAI_API_KEY` | ✅ | Dùng cho embedding + LLM generation + reranking |
| `QDRANT_URL` | ✅ | URL Qdrant server (default: `http://localhost:6333`) |
| `QDRANT_COLLECTION` | ❌ | Tên collection (default: `legal_documents`) |
| `BM25_VOCAB_PATH` | ❌ | Đường dẫn vocab BM25 (default: `src/data/bm25_vocab.json`) |
