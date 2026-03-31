import os
import re
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
            len(candidates),
            top_n,
            query,
        )

        t0 = time.perf_counter()
        scores = self._batch_score(query, candidates)

        scored = []
        for candidate, score in zip(candidates, scores):
            scored.append(
                {
                    **candidate,
                    "rerank_score": score,
                }
            )

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

        BUG ĐÃ SỬA:
        - Strip markdown fences (``` ... ```) trước khi split → tránh lệch index
        - Validate len(scores) == len(candidates) trước khi dùng
        - Prompt chặt hơn: chỉ cho phép số nguyên 0 hoặc 10
        """
        chunks_text = ""
        for i, c in enumerate(candidates, 1):
            snippet = c["text"][:400].replace("\n", " ")
            chunks_text += f"[{i}] {snippet}\n\n"

        # FIX 3: Prompt chặt hơn — chỉ số nguyên, không markdown, không giải thích
        prompt = f"""Bạn là công cụ đánh giá mức độ liên quan của văn bản pháp luật.
Cho {len(candidates)} đoạn văn bản được đánh số từ [1] đến [{len(candidates)}].
Với mỗi đoạn, trả về 10 nếu liên quan đến câu hỏi, 0 nếu không liên quan.

QUY TẮC BẮT BUỘC:
- Chỉ trả về đúng {len(candidates)} số nguyên (0 hoặc 10)
- Mỗi số trên 1 dòng riêng, theo thứ tự [1] đến [{len(candidates)}]
- Tuyệt đối không dùng markdown, không giải thích, không ký tự thừa

Câu hỏi: {query}

Các đoạn văn bản:
{chunks_text}
Điểm liên quan:"""

        logger.debug(
            "Batch score request | candidates=%d | prompt_len=%d chars",
            len(candidates),
            len(prompt),
        )

        try:
            t0 = time.perf_counter()
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=10 * len(candidates),
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

            raw_content = response.choices[0].message.content
            logger.debug("RAW RERANKER OUTPUT: %r", raw_content)

            # FIX 1: Strip markdown fences trước khi split
            # Đây là bug gốc — nếu LLM trả về ```\n10\n10\n...```,
            # split("\n") giữ lại dòng "```" ở đầu làm lệch toàn bộ index
            cleaned = re.sub(r"```[^\n]*", "", raw_content).strip()
            raw_lines = [line.strip() for line in cleaned.split("\n") if line.strip()]

            logger.debug("Lines after fence strip: %s", raw_lines)

            # FIX 2: Soft recovery thay vì strict fallback
            # LLM đôi khi trả thừa/thiếu 1-2 dòng (thường do trailing newline)
            # → truncate nếu thừa, pad "0" nếu thiếu, thay vì bỏ hoàn toàn reranker
            n = len(candidates)
            diff = len(raw_lines) - n
            if diff != 0:
                if abs(diff) <= 2:
                    if diff > 0:
                        # Thừa dòng: bỏ phần đuôi
                        raw_lines = raw_lines[:n]
                        logger.warning(
                            "Score count off by +%d — truncated to %d", diff, n
                        )
                    else:
                        # Thiếu dòng: pad "0" vào cuối
                        raw_lines += ["0"] * (-diff)
                        logger.warning(
                            "Score count off by %d — padded %d zeros", diff, -diff
                        )
                else:
                    # Lệch quá nhiều (>2): thực sự có vấn đề, mới fallback
                    logger.error(
                        "Score count mismatch too large: got %d lines, expected %d — "
                        "falling back to RRF order. Raw output: %r",
                        len(raw_lines),
                        n,
                        raw_content,
                    )
                    return [0.0] * n

            scores = []
            for line in raw_lines:
                try:
                    score = float(line.split()[0].replace(",", "."))
                    scores.append(min(max(score / 10.0, 0.0), 1.0))
                except (ValueError, IndexError):
                    logger.warning(
                        "Could not parse score from line: %r — defaulting to 0.0", line
                    )
                    scores.append(0.0)

            logger.debug("Parsed scores: %s", [f"{s:.2f}" for s in scores])
            return scores

        except Exception as e:
            logger.error(
                "Batch scoring failed: %s — falling back to RRF order", e, exc_info=True
            )
            return [0.0] * len(candidates)
