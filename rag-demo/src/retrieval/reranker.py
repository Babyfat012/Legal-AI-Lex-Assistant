import os
import time
from openai import OpenAI
from core.logger import get_logger

logger = get_logger(__name__)


class Reranker:
    """
    Reranker: Chọn lọc và sắp xếp lại candidates từ hybrid search.

    Dùng batch scoring: gộp tất cả candidates vào 1 LLM call duy nhất
    thay vì N calls riêng lẻ → giảm latency và cost đáng kể.

    Flow:
        hybrid_search(top_k=20) → reranker(top_n=5) → final results
    """

    def __init__(self, api_key: str = None):
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        logger.info("Reranker initialized (model=gpt-4o-mini, batch scoring)")

    def rerank(
        self,
        query: str,
        candidates: list[dict],
        top_n: int = 5,
    ) -> list[dict]:
        """
        Rerank bằng 1 LLM call duy nhất (batch scoring).

        Args:
            query: Câu hỏi gốc
            candidates: List kết quả từ hybrid search
            top_n: Số kết quả cuối trả về

        Returns:
            list[dict]: Candidates đã rerank, thêm field "rerank_score"
        """
        if not candidates:
            logger.warning("Reranker received empty candidates list")
            return []

        top_n = min(top_n, len(candidates))
        logger.info(
            "Reranking %d candidates → top %d | query: %.80s",
            len(candidates), top_n, query,
        )

        t0 = time.perf_counter()
        # Batch: 1 LLM call cho tất cả candidates
        scores = self._batch_score(query, candidates)
        
        scored = []
        for candidate, score in zip(candidates, scores):
            scored.append({
                **candidate,
                "rerank_score": score,
            })
        
        scored.sort(key=lambda x: x["rerank_score"], reverse=True)
        top_results = scored[:top_n]
        logger.info(
            "Reranking done in %.2fs | top scores: %s",
            time.perf_counter() - t0,
            [f"{r['rerank_score']:.2f}" for r in top_results],
        )
        return top_results
    
    def _batch_score(self, query: str, candidates: list[dict]) -> list[float]:
        """
        Score tất cả candidates trong 1 LLM call.
        Trả về list scores tương ứng với từng candidate.
        """
        # Tạo numbered lists các chunks
        chunks_text = ""
        for i, c in enumerate(candidates, 1):
            snippet = c["text"][:400].replace("\n", " ")  # Cắt ngắn và làm gọn text
            chunks_text += f"[{i}] {snippet}\n\n"


        prompt = f"""Đánh giá mức độ liên quan của từng đoạn văn bản luật với câu hỏi.
Trả về ĐÚNG {len(candidates)} số (0-10), mỗi số trên 1 dòng, theo thứ tự [1] đến [{len(candidates)}].
Không giải thích, chỉ trả số.

Câu hỏi: {query}

Các đoạn văn bản:
{chunks_text}
Điểm liên quan (0-10) cho từng đoạn:"""

        logger.debug(
            "Batch score request | candidates=%d | prompt_len=%d chars",
            len(candidates), len(prompt),
        )

        try:
            t0 = time.perf_counter()
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                # Ước lượng: ~5 tokens/score x N candidates, rất nhỏ so với embedding cost
                max_tokens=5 * len(candidates),
                temperature=0.0,
            )
            elapsed = time.perf_counter() - t0
            usage = response.usage
            logger.info(
                "LLM batch score done in %.2fs | tokens: prompt=%d completion=%d total=%d",
                elapsed,
                usage.prompt_tokens,
                usage.completion_tokens,
                usage.total_tokens,
            )

            # Fix: split thành lines, không iterate từng ký tự
            raw = response.choices[0].message.content.strip().split("\n")
            scores = []
            for line in raw:
                try:
                    score = float(line.strip().split()[0].replace(",", "."))  # Handle decimal comma
                    scores.append(min(max(score / 10.0, 0.0), 1.0))
                except (ValueError, IndexError):
                    scores.append(0.0)

            # Đảm bảo đủ số lượng scores
            while len(scores) < len(candidates):
                scores.append(0.0)

            logger.debug("Parsed scores: %s", [f"{s:.2f}" for s in scores[:len(candidates)]])
            return scores[:len(candidates)]

        except Exception as e:
            logger.error(
                "Batch scoring failed: %s — falling back to RRF order", e, exc_info=True
            )
            return [0.0] * len(candidates)
        