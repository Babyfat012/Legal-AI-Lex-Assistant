import os
import time
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

    def embed_documents(
        self,
        texts: list[str],
        batch_size: int = 100,
        max_workers: int = 3,
        max_retries: int = 5,
    ) -> list[list[float]]:
        """
        Embed nhiều documents với retry logic cho rate limit.

        Args:
            texts: Danh sách văn bản cần embed
            batch_size: Số text mỗi batch (default: 100)
            max_workers: Số threads song song (default: 3)
            max_retries: Số lần retry khi gặp rate limit (default: 5)

        Returns:
            list[list[float]]: Danh sách embedding vectors
        """
        if not texts:
            return []

        batches = [texts[i: i + batch_size] for i in range(0, len(texts), batch_size)]
        n_batches = len(batches)
        results: list[list[list[float]] | None] = [None] * n_batches

        def _embed_batch(batch_idx: int, batch: list[str]) -> tuple[int, list[list[float]]]:
            for attempt in range(max_retries):
                try:
                    response = self.client.embeddings.create(
                        input=batch,
                        model=self.model,
                    )
                    embeddings = [
                        item.embedding
                        for item in sorted(response.data, key=lambda x: x.index)
                    ]
                    logger.debug(
                        "Embedded batch %d/%d | %d texts",
                        batch_idx + 1, n_batches, len(batch),
                    )
                    return batch_idx, embeddings

                except Exception as e:
                    if "429" in str(e) or "rate_limit" in str(e).lower():
                        wait = 2 ** attempt  # 1, 2, 4, 8, 16s
                        logger.warning(
                            "Rate limit | batch %d/%d | retry %d/%d | wait %ds",
                            batch_idx + 1, n_batches, attempt + 1, max_retries, wait,
                        )
                        time.sleep(wait)
                    else:
                        logger.error(
                            "Embedding error | batch %d/%d | %s",
                            batch_idx + 1, n_batches, e,
                        )
                        raise

            raise RuntimeError(
                f"Batch {batch_idx + 1}/{n_batches} failed sau {max_retries} retries"
            )

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

        Args:
            query: Câu hỏi cần embed

        Returns:
            list[float]: Query embedding vector
        """
        return self.embed_text(query)   