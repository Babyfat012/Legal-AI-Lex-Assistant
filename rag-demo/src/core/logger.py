import logging
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler

# rag-demo/logs/ — nằm ở root project, ngoài src/
LOG_DIR = Path(__file__).parent.parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

# ── ANSI escape codes ─────────────────────────────────────────────────────────
_RESET = "\033[0m"
_BOLD  = "\033[1m"
_DIM   = "\033[2m"
_BLUE  = "\033[34m"

_LEVEL_COLORS: dict[str, str] = {
    "DEBUG":    "\033[36m",   # Cyan
    "INFO":     "\033[32m",   # Green
    "WARNING":  "\033[33m",   # Yellow
    "ERROR":    "\033[31m",   # Red
    "CRITICAL": "\033[35m",   # Magenta
}


class _ColorFormatter(logging.Formatter):
    """Formatter màu ANSI cho console — dễ đọc khi debug."""

    def format(self, record: logging.LogRecord) -> str:
        color  = _LEVEL_COLORS.get(record.levelname, _RESET)
        time_s = self.formatTime(record, "%H:%M:%S")
        level  = f"{record.levelname:<8}"
        msg    = record.getMessage()

        if record.exc_info:
            msg = f"{msg}\n{self.formatException(record.exc_info)}"

        return (
            f"{_DIM}{time_s}{_RESET} "
            f"{_BOLD}{color}[{level}]{_RESET} "
            f"{_BLUE}{record.name}{_RESET} "
            f"— {msg}"
        )


class _PlainFormatter(logging.Formatter):
    """Formatter plain-text cho file — parseable bởi ELK / grep."""

    def __init__(self) -> None:
        super().__init__(
            fmt="{asctime} [{levelname:<8}] {name} — {message}",
            datefmt="%Y-%m-%d %H:%M:%S",
            style="{",
        )


def get_logger(name: str, level: int = logging.DEBUG) -> logging.Logger:
    """
    Trả về logger đã cấu hình cho module.

    - Console : màu ANSI, tất cả levels từ DEBUG trở lên.
    - File    : plain text, rotate 5 MB × 3 files
                → rag-demo/logs/lex_assistant.log

    Usage:
        from core.logger import get_logger
        logger = get_logger(__name__)

        logger.info("Starting pipeline...")
        logger.debug("dense_dim=%d", len(vec))
        logger.warning("Candidates empty, skipping rerank")
        logger.error("LLM call failed: %s", err, exc_info=True)

    Args:
        name : Truyền __name__ để tự động lấy tên module (vd: retrieval.reranker)
        level: Log level tối thiểu (default DEBUG — log tất cả)

    Returns:
        logging.Logger đã gắn console handler + file handler
    """
    logger = logging.getLogger(name)

    # Guard: tránh duplicate handlers khi module được import lại
    if logger.handlers:
        return logger

    logger.setLevel(level)
    logger.propagate = False  # Không bubble lên root logger

    # ── Console handler (stdout, màu) ─────────────────────────────────────────
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(_ColorFormatter())
    logger.addHandler(ch)

    # ── File handler (rotate 5 MB × 3 files) ─────────────────────────────────
    log_file = LOG_DIR / "lex_assistant.log"
    fh = RotatingFileHandler(
        log_file,
        maxBytes=5 * 1024 * 1024,   # 5 MB
        backupCount=3,
        encoding="utf-8",
    )
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(_PlainFormatter())
    logger.addHandler(fh)

    return logger
