# How to Run - Legal AI Lex Assistant System

## System Overview
The Legal AI Lex Assistant is a RAG (Retrieval-Augmented Generation) system for Vietnamese legal documents. It consists of:
- **FastAPI Backend**: Main API server with RAG pipeline
- **Chainlit Chat Interface**: Python-based chat UI
- **Streamlit Admin UI**: Python-based admin dashboard
- **React Admin UI**: Modern web-based admin interface
- **React Chat UI**: Modern web-based chat interface

## Prerequisites

### Required Services
1. **PostgreSQL Database** (port 5432)
2. **Qdrant Vector Database** (port 6333)
3. **Python 3.12+**
4. **Node.js 18+** (for React UIs)

### Required Environment Variables
Create `.env` file in the root directory:
```bash
# Required
OPENAI_API_KEY=your_openai_api_key_here
QDRANT_URL=http://localhost:6333
DATABASE_URL=postgresql://user:password@localhost:5432/lex_db

# Authentication
JWT_SECRET_KEY=your_jwt_secret_here
JWT_ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=60

# Optional
SERPER_API_KEY=your_serper_api_key  # For web search fallback
QDRANT_COLLECTION=legal_documents
BM25_VOCAB_PATH=./src/data/bm25_vocab.json
SIMPLE_MODEL=gpt-4o-mini
REASONING_MODEL=gpt-4o-mini
```

## Installation

### 1. Install Python Dependencies
```bash
pip install -r requirements.txt
```

### 2. Install Node.js Dependencies (for React UIs)
```bash
# For Admin UI React
cd src/admin_ui/web
npm install

# For Chat UI React
cd src/chat_ui/web
npm install
```

## Running Components

### 1. FastAPI Backend Server
```bash
# From root directory
python src/main.py

# Or using uvicorn
uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload
```

**Access Points:**
- API Documentation: http://localhost:8000/docs
- Health Check: http://localhost:8000/api/v1/health

### 2. Chainlit Chat Interface
```bash
# From root directory
chainlit run src/chainlit_app.py
```

**Access Point:**
- Chat Interface: http://localhost:8000 (Chainlit provides its own server)

### 3. Streamlit Admin UI
```bash
# From root directory
streamlit run src/admin_ui/app.py
```

**Access Point:**
- Admin Dashboard: http://localhost:8501

### 4. React Admin UI
```bash
# From root directory
cd src/admin_ui/web
npm run dev
```

**Access Point:**
- Admin Dashboard: http://localhost:5173

### 5. React Chat UI
```bash
# From root directory
cd src/chat_ui/web
npm run dev
```

**Access Point:**
- Chat Interface: http://localhost:5173

## Database Setup

### 1. PostgreSQL Database
```sql
-- Create database
CREATE DATABASE lex_db;

-- Create extension for UUIDs
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
```

### 2. Initialize Database Tables
The system will automatically create tables on first API request. You can also run:
```bash
# For core tables (ingest logs)
python -c "from src.core.database import init_db; init_db()"

# For auth tables (users, conversations, messages)
python -c "from src.auth.database import init_db; init_db()"
```

### 3. Qdrant Setup
```bash
# Start Qdrant Docker container
docker run -p 6333:6333 qdrant/qdrant

# Or if using Qdrant Cloud, update QDRANT_URL in .env
```

## Testing Components

### 1. API Testing
```bash
# Health check
curl http://localhost:8000/api/v1/health

# Test chat endpoint (requires authentication)
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your_token_here" \
  -d '{"query": "Hello", "session_id": "test-session"}'
```

### 2. Conversation Persistence Test
```bash
python test_conversation_fix.py
```

### 3. Database Connection Test
```bash
python test_db.py
```

## Common Issues and Solutions

### 1. Port Conflicts
- If port 8000 is taken, change it in `src/main.py`
- React UIs use port 5173 by default (can be changed in `vite.config.js`)
- Streamlit uses port 8501 by default

### 2. Database Connection Issues
- Ensure PostgreSQL is running
- Check DATABASE_URL format
- Verify database user permissions

### 3. Qdrant Connection Issues
- Ensure Qdrant is running
- Check QDRANT_URL
- Verify collection exists (will be created automatically on first ingest)

### 4. OpenAI API Issues
- Verify API key is correct
- Check API quota limits
- Ensure proper model names are configured

## Workflow

### Typical Development Workflow
1. Start PostgreSQL and Qdrant services
2. Start FastAPI backend
3. Start desired UI(s) for testing
4. Upload documents via admin UI
5. Test chat functionality
6. Monitor logs for debugging

### Production Deployment
1. Set up proper environment variables
2. Use production database (not SQLite)
3. Configure proper CORS origins
4. Set up proper authentication
5. Use reverse proxy (nginx) for serving
6. Set up proper logging and monitoring

## Additional Tools

### 1. Create Test User
```bash
python create_test_user.py
```

### 2. Create Templates
```bash
python create_templates.py
```

### 3. Test CLI
```bash
python test_cl.py
```

## Architecture Notes

- The system uses hybrid search (BM25 + Vector) with RRF fusion
- Documents are chunked using semantic chunking strategy
- Supports multiple document formats (PDF, DOCX, TXT)
- Includes web search fallback when confidence is low
- JWT-based authentication with session management
- Conversation history stored in PostgreSQL