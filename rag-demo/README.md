# Legal AI — RAG Demo

Legal AI - Lex Assistant is a cloud-native Law Question Answering system built with Retrieval-Agumented Generation (RAG) architecture. The system's UI is built by Streamlit, and FastAPI, a vector search for knowledge retrieval, and an inference server, with Redis as a cached chat history for multi-run conversations. This system is deployed on Kubernetes, automated by Terraform, Helm and Jenkins CI/CD, fully observed and monitored using Prometheus, Grafana, ELK stacks (Elasicsearch, Logstack, Kibana) and tracing with Jaeger.

This repository is aim to learning an end-to-end RAG workflow, suitable for people who need search quickly  laws questions with a correct answer.

Hệ thống RAG (Retrieval-Augmented Generation) cho **văn bản pháp luật Việt Nam**. Hỗ trợ tìm kiếm ngữ nghĩa kết hợp keyword matching và sinh câu trả lời bằng LLM dựa trên nội dung điều luật thực tế.

---

## Kiến trúc

```
┌─────────────────────────────────────────────────────────────┐
│                      INGESTION PIPELINE                      │
│                                                              │
│  PDF/DOCX ──→ [Pre-process] ──→ [Load] ──→ [Chunk] ──→ [Embed] ──→ [Store] │
│               Markdown          Document    Chunks    Dense+Sparse  Qdrant  │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                      QUERY PIPELINE                          │
│                                                              │
│  User Query ──→ [Embed] ──→ [Hybrid Search] ──→ [Rerank] ──→ [Generate]    │
│                 Dense+Sparse   RRF Fusion       LLM Score    GPT-4o-mini    │
└─────────────────────────────────────────────────────────────┘
```

### Các thành phần chính

| Module | File | Mô tả |
|--------|------|-------|
| **Pre-processing** | `ingestion/preprocessing.py` | Chuyển PDF/DOCX → Markdown, chuẩn hóa headings `## Chương`, `#### Điều` |
| **Loading** | `ingestion/loading.py` | Đọc file → `Document(text, metadata)` |
| **Chunking** | `ingestion/chunking.py` | `RecursiveCharacterTextSplitter` với legal separators, inject prefix `[Luật - Chương - Điều]` |
| **Embedding** | `embedding/embedding.py` | OpenAI `text-embedding-3-small` (1536 dims) |
| **BM25** | `embedding/bm25_en.py` | Sparse encoder cho keyword matching |
| **Vector Store** | `ingestion/qdrant_store.py` | Qdrant với named dense + sparse vectors, Hybrid Search RRF |
| **Retriever** | `retrieval/retriever.py` | Hybrid Search (dense + BM25) → Reranker |
| **Reranker** | `retrieval/reranker.py` | LLM-based scoring với `gpt-4o-mini` |
| **Generator** | `generator/llm_generator.py` | RAG generation với `gpt-4o-mini`, system prompt pháp luật VN |
| **API** | `api/routes.py` | FastAPI endpoints: `/embed`, `/index`, `/retrieve`, `/generate` |
| **Pipeline** | `ingestion/pipeline.py` | Orchestrator: Load → Chunk → BM25 fit → Embed → Store |

---

## Cấu trúc thư mục

```
rag-demo/
├── .env                        # API keys và config
├── requirements.txt
├── data/
│   └── bm25_vocab.json         # BM25 vocabulary (tự động tạo sau ingest)
├── documents/                  # Đặt file PDF/DOCX tại đây
│   └── markdown_output/        # Markdown đã convert (để debug)
├── infra/
│   └── docker-compose.yaml     # Qdrant container
└── src/
    ├── main.py                 # FastAPI app entry point
    ├── test-ingestion.py       # Script test pipeline
    ├── api/
    │   ├── routes.py           # API endpoints
    │   └── schemas.py          # Request/Response models
    ├── embedding/
    │   ├── embedding.py        # OpenAI EmbeddingService
    │   └── bm25_en.py          # BM25Encoder
    ├── ingestion/
    │   ├── preprocessing.py    # PDF/DOCX → Markdown
    │   ├── loading.py          # File → Document
    │   ├── chunking.py         # Document → Chunks (legal-aware)
    │   ├── qdrant_store.py     # Qdrant vector store
    │   └── pipeline.py         # Ingestion orchestrator
    ├── retrieval/
    │   ├── retriever.py        # Hybrid search + rerank
    │   └── reranker.py         # LLM-based reranker
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
```

### 3. Khởi động Qdrant

```bash
cd infra
docker compose up -d
cd ..
```

Kiểm tra Qdrant:
```bash
curl http://localhost:6333/healthz
# Dashboard: http://localhost:6333/dashboard
```

---

## Sử dụng

### Ingest tài liệu

Đặt file PDF/DOCX vào thư mục `documents/`, sau đó chạy:

```bash
cd src

# Test pipeline (không cần API key — chỉ load + chunk)
python test-ingestion.py ../documents/luat_giao_thong.pdf

# Full pipeline (cần OPENAI_API_KEY + Qdrant)
python test-ingestion.py ../documents/luat_giao_thong.pdf --full
```

Output của `--full`:
```
[1/5] LOAD   - Reading & converting to Markdown...
[2/5] CHUNK  - Splitting by Chương > Điều > Khoản...
[3/5] BM25   - Fitting and encoding sparse vectors...
[4/5] EMBED  - Dense encoding with text-embedding-3-small...
[5/5] STORE  - Saving to Qdrant 'legal_documents'...
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
| `POST` | `/api/v1/embed` | Embed danh sách text → vectors |
| `POST` | `/api/v1/index` | Index documents vào vector store |
| `POST` | `/api/v1/retrieve` | Tìm top-K chunks liên quan |
| `POST` | `/api/v1/generate` | Full RAG: Retrieve + Generate câu trả lời |

### Ví dụ

**Retrieve:**
```bash
curl -X POST http://localhost:8000/api/v1/retrieve \
  -H "Content-Type: application/json" \
  -d '{"query": "Tốc độ tối đa trong khu đông dân cư", "top_k": 5}'
```

**Generate:**
```bash
curl -X POST http://localhost:8000/api/v1/generate \
  -H "Content-Type: application/json" \
  -d '{"question": "Xe máy được phép chạy tối đa bao nhiêu km/h trong khu dân cư?"}'
```

---

## Chi tiết kỹ thuật

### Chunking — Legal-aware splitting

`RecursiveCharacterTextSplitter` với separators theo cấu trúc luật VN:

```
Ưu tiên cắt: # PHẦN → ## Chương → ### Mục → #### Điều → Khoản → Điểm → \n → câu → từ
```

Mỗi chunk được inject context prefix:
```
[Luật Đường bộ - Chương II - Điều 8. Quy tắc chung] Người tham gia giao thông...
```

### Hybrid Search — RRF Fusion

```
Query
  ├── Dense (text-embedding-3-small) ──→ Qdrant cosine search  ─┐
  └── Sparse (BM25)                  ──→ Qdrant keyword search ─┤
                                                                 ↓
                                                    Reciprocal Rank Fusion
                                                         top 20 candidates
                                                                 ↓
                                               LLM Reranker (gpt-4o-mini)
                                                           top 5 results
```

- **Dense search**: Tốt cho semantic similarity ("xe cơ giới" ~ "phương tiện giao thông")
- **BM25 search**: Tốt cho exact match ("Điều 8", "khoản 2", tên điều luật cụ thể)
- **RRF Fusion**: Kết hợp rank từ 2 nhánh, không phụ thuộc score scale
- **LLM Reranker**: Score từng `(query, chunk)` pair, lọc ra chunk thực sự relevant

### LLM Generator

Model: `gpt-4o-mini` với system prompt chuyên biệt cho pháp luật Việt Nam:
- Chỉ trả lời dựa trên context được retrieve
- Trích dẫn điều luật cụ thể
- Thông báo rõ khi không có đủ dữ liệu

---

## Biến môi trường

| Biến | Bắt buộc | Mô tả |
|------|----------|-------|
| `OPENAI_API_KEY` | ✅ | Dùng cho embedding + LLM generation + reranking |
| `QDRANT_URL` | ✅ | URL Qdrant server (default: `http://localhost:6333`) |
| `LANGFUSE_SECRET_KEY` | ❌ | Observability/tracing (optional) |
| `LANGFUSE_PUBLIC_KEY` | ❌ | Observability/tracing (optional) |
| `LANGFUSE_HOST` | ❌ | Observability/tracing (optional) |
