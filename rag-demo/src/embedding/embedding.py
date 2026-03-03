import os
from langchain_openai import OpenAIEmbeddings


class EmbeddingService:
    def __init__(self, model_name: str = "text-embedding-3-small"):
        self.model_name = model_name
        self.api_key = os.getenv("OPENAI_API_KEY")

        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set.")

        self.embedding_model = OpenAIEmbeddings(
            model=self.model_name,
            openai_api_key=self.api_key,
        )
        print(f"EmbeddingService initialized with model: {self.model_name}")

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed nhiều documents (batch)."""
        try:
            return self.embedding_model.embed_documents(texts)
        except Exception as e:
            print(f"Error embedding documents: {e}")
            raise

    def embed_text(self, text: str) -> list[float]:
        """Embed 1 câu query."""
        try:
            return self.embedding_model.embed_query(text)
        except Exception as e:
            print(f"Error embedding text: {e}")
            raise