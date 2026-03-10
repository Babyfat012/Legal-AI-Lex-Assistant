import os
import json
from pathlib import Path
from qdrant_client.models import SparseVector
from core.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Vietnamese Legal Stopwords
# TỚhp các từ xuất hiện ở hầu hết văn bản luật → IDF gần 0, không có giá trị phân biệt.
# Kết hợp với min_idf_threshold để loại thêm rác đặc thù ngành (xuất hiện khắp nơi trong corpus).
# ---------------------------------------------------------------------------
VIETNAMESE_LEGAL_STOPWORDS: frozenset[str] = frozenset([
    # Đại từ / Liên từ phổ biến
    "đây", "đó", "để", "đến", "được", "điều", "đó", "đi", "đã",
    "và", "với", "về", "vì", "vậy", "vẫn", "vào", "vẫng",
    "các", "của", "có", "còn", "cũng", "càng", "cần", "cầu", "cả",
    "là", "lên", "lại", "lúc", "loại",
    "trong", "theo", "tại", "từ", "tới", "tức", "tắt",
    "khi", "không", "khác", "khoản",
    "hoặc", "hơn", "hết",
    "này", "nếu", "như", "nhưng", "những",
    "bằng", "bất", "bên", "bỏi",
    "mà", "mỗi", "mọi", "một", "mới",
    "sẽ", "sau", "so",
    "quả", "qua",
    "chỉ", "cho", "chưa",
    "giữa", "gìn",
    "rằng", "rất",
    "thì", "thêm", "theo", "thảy", "thu",
    "hai", "ba", "bốn", "năm", "sáu", "bảy", "tám", "chín", "mười",
    # Rác đặc thù pháp lý xuất hiện khắp nơi
    "quy", "quy_định", "pháp", "luật", "nghị",
    "thực", "hiện", "thực_hiện",
    "cơ", "quan", "cơ_quan",
    "cá nhân", "tổ_chức",
    "người",
])


class BM25Encoder:
    """
    BM25 Sparse Encoder cho văn bản luật Tiếng Việt.

    Chuyển text → sparse vector {index: weight} để lưu vào Qdrant
    và thực hiện exact/keyword matching.

    Cải tiến so với phiên bản cũ:
        1. Underthesea word segmentation: giữ đúng ngữ nghĩa pháp lý.
           VD: ["hình", "sự"] → ["hình_sự"] không bị tách nát.
        2. Stopwords 2 lớp: danh sách tĩnh + ngưỡng IDF tối thiểu.
        3. avg_len động: tính lại mỗi lần fit, công bằng điều luật dài/ngắn.
        4. Persistent: lưu cả avg_len vào JSON, kết quả nhất quán sau khi restart.
    """

    def __init__(
        self,
        vocab_path: str = None,
        stopwords: frozenset[str] = None,
        min_idf_threshold: float = 0.1,
    ):
        """
        Args:
            vocab_path: Đường dẫn lưu/load vocabulary đã fit.
                        None = không persist, fit mới mỗi lần.
            stopwords: Tập từ dừng tũnh bổ sung.
                       None = dùng VIETNAMESE_LEGAL_STOPWORDS mặc định.
            min_idf_threshold: Ngưỡng IDF tối thiểu. Token có IDF < ngưỡng
                               (xuất hiện khắp nơi) bị loại khỏi vocab.
                               Mặc định: 0.1.
        """
        self.vocab_path = vocab_path
        self.stopwords: frozenset[str] = (
            stopwords if stopwords is not None else VIETNAMESE_LEGAL_STOPWORDS
        )
        self.min_idf_threshold = min_idf_threshold

        self.vocab: dict[str, int] = {}    # token → index
        self.idf: dict[str, float] = {}    # token → idf weight
        self.avg_len: float = 100.0        # avg document length (tokens), được tính trong fit()
        self._fitted = False               # đã fit vocab/idf chưa

    # ------------------------------------------------------------------
    # Tokenization
    # ------------------------------------------------------------------

    def _tokenize(self, text: str) -> list[str]:
        """
        Tokenize tiếng Việt bằng Underthesea word segmentation.

        Tại sao Underthesea?
            Tách theo khoảng trắng khiến cụm từ pháp lý bị vỡ:
                "hình sự"   → ["hình", "sự"]   (sai ngữ nghĩa)
            Underthesea gộp đúng:
                "hình sự"   → ["hình_sự"]     (chuẩn)
                "tòa án"    → ["tòa_án"]
                "dân sự"   → ["dân_sự"]

        Sau segmentation:
            - Lowercase toàn bộ
            - Loại bỏ dấu câu và token quá ngắn (< 2 ký tự)
            - Lọc stopwords tĩnh
        """
        import re
        try:
            from underthesea import word_tokenize
            # format="text" trả về string với cụm từ dùng dấu gạch dưới
            # VD: "hình sự" → "hình_sự"
            segmented = word_tokenize(text.lower(), format="text")
        except ImportError:
            logger.warning(
                "[BM25] underthesea chưa cài → fallback whitespace tokenizer. "
                "Cài: pip install underthesea"
            )
            segmented = text.lower()

        # Loại bỏ dấu câu (giữ lại chữ, số, dấu gạch dưới từ segmentation)
        tokens = re.split(r"[^\w]+", segmented)
        tokens = [
            t for t in tokens
            if len(t) >= 2 and t not in self.stopwords
        ]
        return tokens

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(self, texts: list[str]):
        """
        Fit BM25 trên corpus (danh sách documents).

        Pipeline:
            1. Tokenize (underthesea) + lọc stopwords tĩnh
            2. Tính IDF: log((N - df + 0.5) / (df + 0.5) + 1)
            3. Tính avg_len = total_tokens / total_docs  (động, không hardcode)
            4. Loại token có IDF < min_idf_threshold  (rác đặc thù ngành)
            5. Build vocab (chỉ giữ token đạt ngưỡng)
            6. Persist nếu có vocab_path

        Args:
            texts: Toàn bộ corpus dùng để tính IDF
        """
        import math

        logger.info("Fitting BM25 on %d documents...", len(texts))

        # Bước 1: Tokenize
        tokenized_docs = [self._tokenize(t) for t in texts]

        # Bước 2: Tính IDF trên toàn bộ corpus
        N = len(tokenized_docs)
        df: dict[str, int] = {}
        for doc_tokens in tokenized_docs:
            for token in set(doc_tokens):
                df[token] = df.get(token, 0) + 1

        raw_idf: dict[str, float] = {
            token: math.log((N - freq + 0.5) / (freq + 0.5) + 1)
            for token, freq in df.items()
        }

        # Bước 3: avg_len động — công bằng điều luật dài/ngắn
        total_tokens = sum(len(doc) for doc in tokenized_docs)
        self.avg_len = total_tokens / N if N > 0 else 1.0
        logger.info("avg_len computed: %.1f tokens/doc", self.avg_len)

        # Bước 4: Lọc stopwords đặc thù ngành (IDF thấp = xuất hiện khắp nơi)
        kept_tokens = {
            token for token, idf_val in raw_idf.items()
            if idf_val >= self.min_idf_threshold
        }
        filtered_count = len(raw_idf) - len(kept_tokens)
        if filtered_count:
            logger.info(
                "IDF filter: loại %d token (IDF < %.2f)",
                filtered_count, self.min_idf_threshold,
            )

        # Bước 5: Build vocab và idf chỉ với token đạt ngưỡng
        self.idf = {t: v for t, v in raw_idf.items() if t in kept_tokens}
        self.vocab = {
            token: idx
            for idx, token in enumerate(sorted(kept_tokens))
        }

        self._fitted = True
        logger.info(
            "BM25 fit complete | vocab_size=%d | avg_len=%.1f | idf_threshold=%.2f",
            len(self.vocab), self.avg_len, self.min_idf_threshold,
        )

        # Bước 6: Persist
        if self.vocab_path:
            self.save(self.vocab_path)

    # ------------------------------------------------------------------
    # Encode
    # ------------------------------------------------------------------

    def encode(self, text: str, is_query: bool = False) -> SparseVector:
        """
        Encode text thành SparseVector cho Qdrant.

        Args:
            text:     Text cần encode
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

        indices = []
        values = []

        for token, count in tf.items():
            if token not in self.vocab:
                continue  # OOV token, bỏ qua

            idx = self.vocab[token]
            idf_weight = self.idf.get(token, 0.0)

            if is_query:
                # Query: chỉ dùng IDF — tăng trọng số cho token hiếm
                weight = idf_weight
            else:
                # Document: BM25 TF normalization với avg_len động từ fit()
                k1, b = 1.5, 0.75
                tf_norm = (
                    (count * (k1 + 1))
                    / (count + k1 * (1 - b + b * len(tokens) / self.avg_len))
                )
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

    # ------------------------------------------------------------------
    # Persist
    # ------------------------------------------------------------------

    def save(self, path: str):
        """
        Lưu vocab, idf và avg_len ra file JSON.

        avg_len được lưu cùng để đảm bảo kết quả search nhất quán
        sau khi restart server, không phụ thuộc vào corpus gốc.
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        data = {
            "vocab": self.vocab,
            "idf": self.idf,
            "avg_len": self.avg_len,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(
            "BM25 saved | path=%s | tokens=%d | avg_len=%.1f",
            path, len(self.vocab), self.avg_len,
        )

    def load(self, path: str):
        """
        Load vocab, IDF và avg_len từ file JSON.

        avg_len được khôi phục để encode() cho kết quả giống hệt
        thời điểm fit() — không bị đổi khi corpus thay đổi sau.
        """
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.vocab = data["vocab"]
        self.idf = data["idf"]
        self.avg_len = data.get("avg_len", 100.0)  # backward-compat với file JSON cũ
        self._fitted = True
        logger.info(
            "BM25 loaded | path=%s | tokens=%d | avg_len=%.1f",
            path, len(self.vocab), self.avg_len,
        )