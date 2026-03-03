import numpy as np
import faiss


class VectorStore:
    def __init__(self, dimension: int = 1536):
        """ Initialize the vector store with a specified dimension for the vectors.
            Args:
                dimension: The dimensionality of the vectors to be stored 
                (default is 1536 for OpenAI embeddings).
        """
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)
        self.documents: list[dict] = []

    def add_documents(self, texts: list[str], embeddings: list[list[float]]) ->None:
        """ Add documents and their corresponding embeddings to the vector store.
            Args:
                texts: A list of document texts to be added.
                embeddings: A list of corresponding embeddings for the documents.
        """
        if len(texts) != len(embeddings):
            raise ValueError(f"texts {len(texts)} and embeddings {len(embeddings)} must have the same length.")
        
        # Convert to numpy array float32 (required by faiss)
        vectors = np.array(embeddings, dtype=np.float32)

        # Normalize the vectors to unit length (important for cosine similarity)
        faiss.normalize_L2(vectors)

        self.index.add(vectors)
        
        for text in texts:
            self.documents.append({"text": text})

        print(f"Added {len(texts)} documents. Total: {self.index.ntotal}")

    def add_single(self, text: str, embedding: list[float]) -> None:
        """ Add a single document and its corresponding embedding to the vector store.
            Args:
                text: The document text to be added.
                embedding: The corresponding embedding for the document.
        """
        self.add_documents([text], [embedding])

    def retrieve_similar(self, query_embedding: list[float], top_k: int = 5) -> list[dict]:
        """
        Find top-K most similar documents
        
        Args:
            query_embedding: Embedding vector for the query
            top_k: Number of similar documents to retrieve
            
        Returns:
            List[dict]: List of retrieved documents with their similarity scores and index
        """
        if self.index.ntotal == 0:
            print("Warning: Vector store is empty.")
            return []
        
        # Make sure top_k does not exceed the number of documents in the index
        top_k = min(top_k, self.index.ntotal)

        # Prepare the query vector for faiss (must be 2D array)
        query_vector = np.array([query_embedding], dtype=np.float32)
        faiss.normalize_L2(query_vector)

        # find top-K similar vectors
        scores, indices = self.index.search(query_vector, top_k)

        results = []
        for score, idx in zip (scores[0], indices[0]):
            if idx == -1:
                continue 
            results.append({
                "text": self.documents[idx]["text"],
                "score": float(score),
                "index": int(idx),
            })

        return results
    
    def clear(self) -> None:
        """ Clear the vector store and all stored documents. """
        self.index.reset()
        self.documents.clear()
        print("Vector store cleared.")

    @property
    def total_documents(self) -> int:
        """ Get the total number of documents currently stored in the vector store. """
        return self.index.ntotal