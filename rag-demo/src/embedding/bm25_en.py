import os
import json
from pathlib import Path
from qdrant_client.models import SparseVector
from core.logger import get_logger

logger = get_logger(__name__)


class BM25Encoder:
    """
    BM25 Sparse Encoder cho văn bản luật Tiếng Việt.

    Chuyển text → sparse vector {index: weight} để lưu vào Qdrant
    và thực hiện exact/keyword matching.

    Sử dụng thư viện `rank_bm25` để tính IDF weights.
    """

    def __init__(self, vocab_path: str = None):
        """
        Args:
            vocab_path: Đường dẫn lưu/load vocabulary đã fit.
                        None = không persist, fit mới mỗi lần.
        """
        self.vocab_path = vocab_path
        self.vocab: dict[str, int] = {}   # token → index
        self.idf: dict[str, float] = {}   # token → idf weight
        self._fitted = False               # đã fit vocab/idf chưa

    def _tokenize(self, text: str) -> list[str]:
        """
        Tokenize đơn giản cho tiếng Việt:
        - Lowercase
        - Tách theo khoảng trắng + dấu câu
        - Giữ lại các ký tự có nghĩa (bao gồm tiếng Việt)
        """
        import re
        text = text.lower()
        # Tách theo ký tự không phải chữ/số/dấu tiếng việt
        tokens = re.split(r"[^\w\s]", text)
        tokens = " ".join(tokens).split()
        # Lọc token quá ngắn
        return [t for t in tokens if len(t) >= 2]
    
    def fit(self, texts: list[str]):
        """
        Fit BM25 trên corpus (danh sách documents).
        Tính IDF cho từng token.

        Args:
            texts: Toàn bộ corpus dùng để tính IDF
        """
        import math

        logger.info("Fitting BM25 on %d documents...", len(texts))

        # Tokenize tất cả documents
        tokenized_docs = [self._tokenize(t) for t in texts]

        # Build vocabulary
        all_tokens = set(token for doc in tokenized_docs for token in doc)
        self.vocab = {token: idx for idx, token in enumerate(sorted(all_tokens))}

        # Tính IDF: log((N - df + 0.5) / (df + 0.5)) cho mỗi token
        N = len(tokenized_docs)
        df: dict[str, int] = {}
        for doc_tokens in tokenized_docs:
            for token in set(doc_tokens):
                df[token] = df.get(token, 0) + 1
            
        self.idf = {
            token: math.log((N - freq + 0.5) / (freq + 0.5) + 1)
            for token, freq in df.items()
        }
        
        self._fitted = True
        logger.info("BM25 fit complete | vocab_size=%d tokens", len(self.vocab))

        # Lưu vocab/idf nếu có đường dẫn
        if self.vocab_path:
            self.save(self.vocab_path)

    def encode(self, text: str, is_query: bool = False) -> SparseVector:
        """
        Encode text thành SparseVector cho Qdrant.

        Args:
            text: Text cần encode
            is_query: True nếu là query (chỉ dùng IDF, không cần TF normalization)

        Returns:
            SparseVector với indices và values
        """
        if not self._fitted:
            raise RuntimeError(
                "BM25Encoder chưa được fit. Gọi fit(texts) hoặc load(path) trước."
            )
        
        tokens = self._tokenize(text)
        if not tokens:
            return SparseVector(indices=[], values=[])
        
        # Tính TF
        tf: dict[str, int] = {}
        for token in tokens:
            tf[token] = tf.get(token, 0) + 1

        # Tính weights: TF * IDF
        indices = []
        values = []

        for token, count in tf.items():
            if token not in self.vocab:
                continue # OOV token, bỏ qua

            idx = self.vocab[token]
            idf_weight = self.idf.get(token, 0.0)

            if is_query:
                # Query chỉ dùng IDF để tăng trọng số cho token hiếm
                weight = idf_weight
            else:
                # Document dùng TF * IDF voi bm25 normalization
                k1, b = 1.5, 0.75
                avg_len = 100
                tf_norm =  (count * (k1 + 1)) / (count + k1 * (1 - b  + b * len(tokens) / avg_len))
                weight = tf_norm * idf_weight

            if weight > 0:
                indices.append(idx)
                values.append(float(weight))

        return SparseVector(indices=indices, values=values)
        

    def encode_documents(self, texts: list[str]) -> list[SparseVector]:
        """Encode nhiều documents."""
        return [self.encode(text, is_query=False) for text in texts]
    

    def encode_query(self, query: str) -> SparseVector:
        """Encode query text."""
        return self.encode(query, is_query=True)
    
    def save(self, path: str):
        """Lưu vocab và idf ra file JSON."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        data = {
            "vocab": self.vocab,
            "idf": self.idf,
        }
        with open (path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info("BM25 vocab saved | path=%s | tokens=%d", path, len(self.vocab))

    def load(self, path: str):
        """Load vocab + IDF từ file JSON."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.vocab = data["vocab"]
        self.idf = data["idf"]
        self._fitted = True
        logger.info("BM25 vocab loaded | path=%s | tokens=%d", path, len(self.vocab))


    