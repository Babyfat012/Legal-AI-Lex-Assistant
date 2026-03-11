import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from core.logger import get_logger

logger = get_logger(__name__)


class EmbeddingService:
    """
    EmbeddingService: sử dụng OpenAI text-embedding-3-small
    """

    # Dimension mapping — cập nhật khi thêm model mới
    _DIMENSION_MAP: dict[str, int] = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
    }

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

    @property
    def dimension(self) -> int:
        """Trả về dimension của embedding vector cho model hiện tại."""
        return self._DIMENSION_MAP.get(self.model, 1536)
    
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

    def embed_documents(self, texts: list[str], batch_size: int = 100, max_workers: int = 6) -> list[list[float]]:
        """
        Embed nhiều đoạn text thành vectors.
        Gửi các batch song song thay vì tuần tự → giảm latency đáng kể
        khi có nhiều batch (I/O-bound, ThreadPoolExecutor phù hợp).

        Args:
            texts:       Danh sách chuỗi văn bản cần embed.
            batch_size:  Số text mỗi batch gửi lên API. Mặc định 100.
            max_workers: Số thread song song. Mặc định 6 (cân bằng tốc độ / rate-limit).

        Returns:
            list[list[float]]: Danh sách vector embedding theo đúng thứ tự input.
        """
        if not texts:
            return []

        batches = [texts[i: i + batch_size] for i in range(0, len(texts), batch_size)]
        n_batches = len(batches)
        results: list[list[list[float]] | None] = [None] * n_batches

        def _embed_batch(batch_idx: int, batch: list[str]) -> tuple[int, list[list[float]]]:
            response = self.client.embeddings.create(input=batch, model=self.model)
            embeddings = [item.embedding for item in sorted(response.data, key=lambda x: x.index)]
            logger.debug("Embedded batch %d/%d | %d texts", batch_idx + 1, n_batches, len(batch))
            return batch_idx, embeddings

        workers = min(max_workers, n_batches)
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(_embed_batch, idx, batch): idx
                for idx, batch in enumerate(batches)
            }
            for future in as_completed(futures):
                batch_idx, embeddings = future.result()
                results[batch_idx] = embeddings

        all_embeddings: list[list[float]] = []
        for batch_embeddings in results:
            all_embeddings.extend(batch_embeddings)  # type: ignore[arg-type]

        logger.info(
            "Embedding complete | %d vectors | dim=%d | batches=%d | workers=%d",
            len(all_embeddings),
            len(all_embeddings[0]) if all_embeddings else 0,
            n_batches,
            workers,
        )
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
