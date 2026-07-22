"""Console logging setup.

Extensive human-friendly progress logging: which stage, which page, counts and
warnings. Defaults to stdout for the CLI, but can switch to stderr (required for
a future MCP stdio server, where stdout is reserved for the JSON-RPC stream).
"""

import logging
import sys

_FORMAT = "%(asctime)s [%(levelname)s] %(message)s"
_DATEFMT = "%H:%M:%S"
_CONFIGURED = False


def setup_logging(stream: str = "stdout", level: int = logging.INFO) -> None:
    """Configure the root logger once. stream is 'stdout' or 'stderr'."""
    global _CONFIGURED
    target = sys.stdout if stream == "stdout" else sys.stderr
    try:
        target.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    handler = logging.StreamHandler(target)
    handler.setFormatter(logging.Formatter(_FORMAT, datefmt=_DATEFMT))

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a module logger (call setup_logging() once at startup)."""
    return logging.getLogger(name)


# ---- convenience helpers for consistent progress lines ----
def log_stage(logger, index: int, total: int, name: str, message: str = "") -> None:
    """e.g. [1/6 Detection] starting ..."""
    logger.info("[%d/%d %s] %s", index, total, name, message)


def log_page(
    logger,
    stage_index: int,
    total_stages: int,
    stage: str,
    page: int,
    total_pages: int,
    message: str = "",
) -> None:
    """e.g. [1/6 Detection] Page 003/012 ...found 2 bubbles"""
    logger.info(
        "[%d/%d %s] Page %03d/%03d %s",
        stage_index,
        total_stages,
        stage,
        page,
        total_pages,
        message,
    )
