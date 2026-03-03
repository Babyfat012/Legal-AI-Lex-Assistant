# RAG Demo Project

This project demonstrates a basic implementation of a Retrieval-Augmented Generation (RAG) system using FastAPI. The system consists of three main components: embedding, retrieval, and generation.

## Project Structure

```
rag-demo
├── src
│   ├── main.py              # Entry point for the FastAPI application
│   ├── config.py            # Configuration settings for the application
│   ├── embedding             # Module for generating embeddings
│   │   ├── __init__.py
│   │   └── embedder.py       # Class for generating embeddings from text
│   ├── retrieval             # Module for retrieving relevant embeddings
│   │   ├── __init__.py
│   │   ├── vector_store.py    # Class for managing vector embeddings
│   │   └── retriever.py       # Class for finding relevant embeddings
│   ├── generator             # Module for generating text responses
│   │   ├── __init__.py
│   │   └── llm_generator.py   # Class for generating responses using a language model
│   ├── api                  # Module for API routes and schemas
│   │   ├── __init__.py
│   │   ├── routes.py         # API routes linking to handlers
│   │   └── schemas.py        # Pydantic models for request and response
│   └── data                 # Module for document processing
│       └── documents.py      # Functions for loading and processing documents
├── documents
│   └── sample.txt           # Sample document for testing
├── requirements.txt          # Project dependencies
└── README.md                 # Project documentation
```

## Setup Instructions

1. **Clone the repository:**
   ```
   git clone <repository-url>
   cd rag-demo
   ```

2. **Create a virtual environment:**
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install dependencies:**
   ```
   pip install -r requirements.txt
   ```

4. **Run the application:**
   ```
   uvicorn src.main:app --reload
   ```

5. **Access the API:**
   Open your browser and navigate to `http://127.0.0.1:8000/docs` to view the Swagger UI and test the API endpoints.

## Usage Examples

- **Embedding:** Use the embedding endpoint to generate embeddings from input text.
- **Retrieval:** Query the retrieval endpoint to find relevant embeddings based on a query.
- **Generation:** Use the generation endpoint to get text responses based on the retrieved embeddings.

## Components Description

- **Embedding:** The `Embedder` class in `src/embedding/embedder.py` is responsible for generating embeddings from input text using a specified model.
- **Retrieval:** The `Retriever` class in `src/retrieval/retriever.py` utilizes the `VectorStore` class to find relevant embeddings based on a user query.
- **Generation:** The `LLMGenerator` class in `src/generator/llm_generator.py` generates text responses based on the input queries using a language model.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.