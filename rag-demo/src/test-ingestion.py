"""
Script test interactive cho Ingestion Pipeline:
    Pre-processing → Loading → Chunking → Embedding → Store

Chạy:
    cd rag-demo/src
    python test-ingestion.py ../documents/luat_giao_thong.docx
    python test-ingestion.py ../documents/luat_giao_thong.docx --full   # test cả embed + store
"""

import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))


def print_separator(title: str = ""):
    print(f"\n{'='*70}")
    if title:
        print(f"  {title}")
        print(f"{'='*70}")


def print_chunk_detail(chunk, index: int):
    """In chi tiết 1 chunk."""
    print(f"\n{'─'*50}")
    print(f"  CHUNK #{index}")
    print(f"{'─'*50}")
    print(f"  Length  : {len(chunk.text)} chars")
    print(f"  Luật    : {chunk.metadata.get('luat', '—')}")
    print(f"  Chương  : {chunk.metadata.get('chuong', '—')}")
    print(f"  Mục     : {chunk.metadata.get('muc', '—')}")
    print(f"  Điều    : {chunk.metadata.get('dieu', '—')}")
    print(f"  Prefix  : {chunk.metadata.get('context_prefix', '—')}")
    print(f"  Source  : {chunk.metadata.get('filename', '—')}")
    print(f"  Index   : {chunk.chunk_index}/{chunk.metadata.get('total_chunks', '?')}")
    print(f"\n  --- TEXT ---")
    text_preview = chunk.text[:500]
    if len(chunk.text) > 500:
        text_preview += "\n  ... [truncated]"
    for line in text_preview.split("\n"):
        print(f"  | {line}")


# =========================================================================
# TEST FUNCTIONS
# =========================================================================

def test_preprocessing(file_path: str) -> str:
    """Test bước Pre-processing: File → Markdown."""
    print_separator("STEP 1: PRE-PROCESSING (File → Markdown)")

    from ingestion.preprocessing import MarkdownConverter, ConverterBackend

    converter = MarkdownConverter(backend=ConverterBackend.MARKITDOWN)

    print(f"  Input file : {file_path}")
    print(f"  Backend    : markitdown")

    markdown_text = converter.convert_file(file_path)

    print(f"\n  Markdown output: {len(markdown_text)} chars")
    print(f"{'─'*50}")
    if len(markdown_text) <= 2000:
        print(markdown_text)
    else:
        print(markdown_text[:2000])
        print(f"\n  ... [truncated, total {len(markdown_text)} chars]")
    print(f"{'─'*50}")

    # Lưu file markdown
    output_dir = Path(file_path).parent / "markdown_output"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / (Path(file_path).stem + ".md")
    output_path.write_text(markdown_text, encoding="utf-8")
    print(f"  Saved to: {output_path}")

    return markdown_text


def test_loading(file_path: str):
    """Test bước Loading: File → Document."""
    print_separator("STEP 2: LOADING (File → Document)")

    from ingestion.loading import DocumentLoader

    loader = DocumentLoader(converter_backend="markitdown")
    documents = loader.load_file(file_path)

    print(f"  Loaded {len(documents)} document(s)")
    for i, doc in enumerate(documents):
        print(f"\n  Document #{i}:")
        print(f"    Text length : {len(doc.text)} chars")
        print(f"    Metadata    : {doc.metadata}")
        print(f"\n    --- FIRST 500 CHARS ---")
        for line in doc.text[:500].split("\n"):
            print(f"    | {line}")

    return documents


def test_chunking(documents: list, chunk_size: int = 1024, chunk_overlap: int = 100):
    """Test bước Chunking: Documents → Chunks."""
    print_separator(f"STEP 3: CHUNKING (size={chunk_size}, overlap={chunk_overlap})")

    from ingestion.chunking import TextChunker

    chunker = TextChunker(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_context_prefix=True,
    )

    chunks = chunker.chunk_documents(documents)

    print(f"\n  Total chunks: {len(chunks)}")
    chunk_lengths = [len(c.text) for c in chunks]
    print(f"  Min size : {min(chunk_lengths)} chars")
    print(f"  Max size : {max(chunk_lengths)} chars")
    print(f"  Avg size : {sum(chunk_lengths) / len(chunk_lengths):.0f} chars")

    print(f"\n{'='*70}")
    print(f"  ALL CHUNKS DETAIL")
    print(f"{'='*70}")

    for i, chunk in enumerate(chunks):
        print_chunk_detail(chunk, i)

    return chunks


def test_chunking_comparison(documents: list):
    """So sánh kết quả với các chunk_size khác nhau."""
    print_separator("COMPARISON: Different chunk sizes")

    from ingestion.chunking import TextChunker

    for size in [256, 512, 1024, 2048]:
        chunker = TextChunker(chunk_size=size, chunk_overlap=50)
        chunks = chunker.chunk_documents(documents)
        lengths = [len(c.text) for c in chunks]
        print(
            f"  chunk_size={size:>5} → {len(chunks):>3} chunks | "
            f"avg={sum(lengths)/len(lengths):>6.0f} | "
            f"min={min(lengths):>4} | max={max(lengths):>4}"
        )


def test_embedding(chunks: list):
    """Test bước Embedding: Chunks → Vectors."""
    print_separator("STEP 4: EMBEDDING (Chunks → Vectors)")

    from embedding.embedding import EmbeddingService

    embedding_service = EmbeddingService()

    print(f"  Model     : {embedding_service.model}")
    print(f"  Dimension : {embedding_service.dimension}")
    print(f"  Chunks    : {len(chunks)}")

    chunk_texts = [chunk.text for chunk in chunks]
    embeddings = embedding_service.embed_documents(chunk_texts)

    print(f"\n  Generated {len(embeddings)} embeddings")
    print(f"  Vector dim: {len(embeddings[0])}")

    # Preview first embedding
    print(f"\n  First embedding (first 10 values):")
    print(f"    {embeddings[0][:10]}")

    return embeddings, embedding_service


def test_store(chunks: list, dense_embeddings: list, sparse_embeddings: list, embedding_service):
    """Test bước Store: Dense + Sparse Vectors → Qdrant."""
    print_separator("STEP 5: STORE (Dense + Sparse → Qdrant)")

    from ingestion.qdrant_store import QdrantVectorStore

    store = QdrantVectorStore(
        collection_name="legal_documents_test",
        dimension=embedding_service.dimension,
    )
    store.recreate_collection()
    stored = store.store_chunks(chunks, dense_embeddings, sparse_embeddings)

    info = store.get_collection_info()
    print(f"\n  Collection: {info['name']}")
    print(f"  Points   : {info['points_count']}")
    print(f"  Status   : {info['status']}")

    return store


def test_hybrid_search(store, embedding_service, bm25_encoder):
    """Test Hybrid Search + Reranking."""
    print_separator("STEP 6: HYBRID SEARCH + RERANK")

    from retrieval.reranker import Reranker

    reranker = Reranker()

    test_queries = [
        "Quy định về tốc độ xe trong khu đông dân cư",
        "Điều kiện cấp giấy phép lái xe",
        "Xử phạt vượt đèn đỏ",
    ]

    for query in test_queries:
        print(f"\n  Query: '{query}'")
        print(f"  {'─'*50}")

        # Encode query
        query_dense = embedding_service.embed_query(query)
        query_sparse = bm25_encoder.encode_query(query)

        # Hybrid search (top 20)
        candidates = store.hybrid_search(
            query_dense=query_dense,
            query_sparse=query_sparse,
            top_k=20,
        )
        print(f"  Hybrid search → {len(candidates)} candidates (RRF fusion)")

        # Rerank (top 5)
        reranked = reranker.rerank(query, candidates, top_n=5)
        print(f"  After rerank  → top {len(reranked)}")

        for i, r in enumerate(reranked):
            rrf_score = r.get("score", 0)
            rerank_score = r.get("rerank_score", 0)
            dieu = r["metadata"].get("dieu", "—")
            print(f"\n  [{i+1}] RRF={rrf_score:.4f} | Rerank={rerank_score:.2f}")
            print(f"       {dieu}")
            print(f"       {r['text'][:120].replace(chr(10), ' ')}...")


def test_full_pipeline(file_path: str):
    """Test full pipeline qua IngestionPipeline orchestrator."""
    print_separator("FULL PIPELINE TEST (via IngestionPipeline)")

    from ingestion.pipeline import IngestionPipeline
    from ingestion.chunking import TextChunker
    from embedding.embedding import EmbeddingService
    from ingestion.qdrant_store import QdrantVectorStore

    embedding_service = EmbeddingService()

    pipeline = IngestionPipeline(
        chunker=TextChunker(chunk_size=1024, chunk_overlap=100),
        embedding_service=embedding_service,
        vector_store=QdrantVectorStore(
            collection_name="legal_documents",
            dimension=embedding_service.dimension,
        ),
    )

    result = pipeline.ingest(file_path, is_directory=False)

    print(f"\n  Pipeline result:")
    for k, v in result.items():
        print(f"    {k}: {v}")

    # Test search
    print(f"\n  --- Search test ---")
    search_results = pipeline.search("quy định về tốc độ xe", top_k=3)
    for i, r in enumerate(search_results):
        print(f"  [{i+1}] Score={r['score']:.4f}: {r['text'][:100]}...")

    return result


# =========================================================================
# MAIN
# =========================================================================

def main():
    default_file = os.path.join(
        os.path.dirname(__file__), "..", "documents", "sample_luat.txt"
    )

    # Parse arguments
    file_path = default_file
    full_mode = False

    for arg in sys.argv[1:]:
        if arg == "--full":
            full_mode = True
        else:
            file_path = arg

    file_path = os.path.abspath(file_path)

    print(f"{'#'*70}")
    print(f"  INGESTION PIPELINE TEST")
    print(f"  File: {file_path}")
    print(f"  Mode: {'FULL (embed + store)' if full_mode else 'BASIC (load + chunk only)'}")
    print(f"{'#'*70}")

    if not os.path.exists(file_path):
        print(f"\n  ERROR: File not found: {file_path}")
        print(f"\n  Usage:")
        print(f"    python test-ingestion.py <file>            # basic: load + chunk")
        print(f"    python test-ingestion.py <file> --full     # full: load + chunk + embed + store")
        print(f"\n  Examples:")
        print(f"    python test-ingestion.py ../documents/luat_giao_thong.docx")
        print(f"    python test-ingestion.py ../documents/luat_giao_thong.docx --full")
        sys.exit(1)

    # Step 1-3: Load + Chunk (no API key needed)
    markdown_text = test_preprocessing(file_path)
    documents = test_loading(file_path)
    chunks = test_chunking(documents, chunk_size=1024, chunk_overlap=100)
    test_chunking_comparison(documents)

    # Step 4-6: Embed + Store + Search (requires API key + Qdrant)
    if full_mode:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print(f"\n  ERROR: OPENAI_API_KEY not set in .env")
            sys.exit(1)

        # Dense embedding
        embeddings, embedding_service = test_embedding(chunks)

        # BM25 sparse encoding
        print_separator("STEP 4b: BM25 SPARSE ENCODING")
        from embedding.bm25_en import BM25Encoder
        bm25 = BM25Encoder(vocab_path="../data/bm25_vocab.json")
        bm25.fit([c.text for c in chunks])
        sparse_embeddings = bm25.encode_documents([c.text for c in chunks])
        print(f"  Encoded {len(sparse_embeddings)} sparse vectors")

        # Store
        store = test_store(chunks, embeddings, sparse_embeddings, embedding_service)

        # Hybrid search + rerank
        test_hybrid_search(store, embedding_service, bm25)

        # Bonus: test full pipeline orchestrator
        test_full_pipeline(file_path)

    # Summary
    print_separator("SUMMARY")
    print(f"  Input file     : {Path(file_path).name}")
    print(f"  Markdown chars : {len(markdown_text)}")
    print(f"  Documents      : {len(documents)}")
    print(f"  Chunks (1024)  : {len(chunks)}")

    if full_mode:
        print(f"  Embeddings     : ✅ stored in Qdrant")
        print(f"  Search         : ✅ tested")
    else:
        print(f"\n  To test embedding + store, run with --full flag:")
        print(f"    python test-ingestion.py {file_path} --full")


if __name__ == "__main__":
    main()