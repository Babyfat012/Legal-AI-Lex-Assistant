import os
from openai import OpenAI
from core.logger import get_logger

logger = get_logger(__name__)


class EmbeddingService:
    """
    EmbeddingService: sử dụng OpenAI text-embedding-3-small
    """

    def __init__(
            self, 
            model: str = "text-embedding-3-small",
            api_key: str = None,
        ):
        """
        Args:
            model: Tên model embedding của OpenAI
            api_key: OpenAI API key (default: lấy từ env OPENAI_API_KEY)
        """
        self.model = model
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        logger.info("EmbeddingService initialized | model=%s | dim=%d", model, self.dimension)

        # Dimension mapping cho các model embedding phổ biến
        self._dimension = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536,
        }

    @property
    def dimension(self) -> int:
        """Trả về dimension của embedding vector cho model hiện tại."""
        return self._dimension.get(self.model, 1536)  # Mặc định 1536 nếu model không có trong mapping
    
    def embed_text(self, text: str) -> list[float]:
        """
        Embed 1 đoạn text thành vector.
        
        Args:
            text: Chuỗi văn bản cần embed.

        Returns:
            list[float]: Vector embedding của đoạn text.
        """
        response = self.client.embeddings.create(
            input=text,
            model=self.model,
        )
        return response.data[0].embedding

    def embed_documents(self, texts: list[str], batch_size: int = 100) -> list[list[float]]:
        """
        Embed nhiều đoạn text thành vectors
        Tự động chia batch để trách vượt rate limit

        Args:
            texts: Danh sách chuỗi văn bản cần embed.
            batch_size: Số text mỗi batch gửi lên API

        Returns:
            list[list[float]]: Danh sách vector embedding tương ứng với mỗi đoạn text.
        """
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]

            response = self.client.embeddings.create(
                input=batch,
                model=self.model,
            )

            # Sort by index để đảm bảo thứ tự đúng
            batch_embeddings = [item.embedding for item in sorted(
                response.data, key=lambda x: x.index
            )]
            all_embeddings.extend(batch_embeddings)

            logger.debug("Embedded batch %d/%d | %d texts",
                         i // batch_size + 1, -(-len(texts) // batch_size), len(batch))
        
        logger.info("Embedding complete | %d vectors | dim=%d", len(all_embeddings), len(all_embeddings[0]) if all_embeddings else 0)
        return all_embeddings
    
    def embed_query(self, query: str) -> list[float]:
        """
        Embed query text cho retrieval.
        Tách riêng để sau này có thể thêm logic khác cho query vs document.

        Args:
            query: Câu hỏi cần embed

        Returns:
            list[float]: Query embedding vector
        """
        return self.embed_text(query)
    
    