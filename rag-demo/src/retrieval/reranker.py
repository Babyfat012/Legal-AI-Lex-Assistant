import os 
from openai import OpenAI


class Reranker:
    """
    Reranker: Chọn lọc và sắp xếp lại candidates từ hybrid search.

    Dùng cross-encoder approach với OpenAI để score từng (query, chunk) pair.
    Chỉ gọi reranker cho top candidates từ retrieval (không gọi cho toàn bộ DB).

    Flow:
        hybrid_search(top_k=20) → reranker(top_n=5) → final results
    """

    def __init__(self, api_key: str = None):
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))

    def rerank(
        self,
        query: str,
        candidates: list[dict],
        top_n: int = 5,
    ) -> list[dict]:
        """
        Rerank candidates dựa trên mức độ liên quan thực sự với query.

        Args:
            query: Câu hỏi gốc
            candidates: List kết quả từ hybrid search
                        [{"text": ..., "score": ..., "metadata": ...}]
            top_n: Số kết quả cuối cùng trả về

        Returns:
            list[dict]: Candidates đã rerank, thêm field "rerank_score"
        """
        if not candidates:
            return []
        
        top_n = min(top_n, len(candidates))
        
        # Tạo prompt de score từng candidate
        scored = []
        for candidate in candidates:
            score = self._score_relevance(query, candidate["text"])
            scored.append({
                **candidate,
                "rerank_score": score,
            })
        
        # Sắp xếp lại theo rerank_score giảm dần
        scored.sort(key=lambda x: x["rerank_score"], reverse=True)

        return scored[:top_n]
    
    def _score_relevance(self, query: str, chunk_text: str) -> float:
        """
        Score mức độ liên quan của 1 chunk với query.
        Dùng LLM để đánh giá từ 0-10.

        Args:
            query: Câu hỏi
            chunk_text: Nội dung chunk cần đánh giá

        Returns:
            float: Score từ 0.0 đến 1.0
        """
        prompt = f"""Đánh giá mức độ liên quan của đoạn văn bản luật sau với câu hỏi.
Chỉ trả về một số từ 0 đến 10, không giải thích.

    Câu hỏi: {query}

    Đoạn văn bản:
    {chunk_text[:800]}

    Điểm liên quan (0-10):"""
        
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{
                "role": "user",
                "content": prompt,
            }],
            max_tokens=5,
            temperature=0.0,
        )

        try:
            raw = response.choices[0].message.content.strip()
            score = float(raw.split()[0].replace(",", "."))
            return min(max(score / 10.0, 0.0), 1.0)  # Chuẩn hóa về 0-1
        except (ValueError, IndexError):
            return 0.0  # Nếu parsing lỗi, trả về score thấp nhất
        
        