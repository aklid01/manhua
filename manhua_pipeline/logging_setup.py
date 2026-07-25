"""Console and file logging setup.

Extensive human-friendly progress logging: which stage, which page, counts and
warnings. Defaults to stdout for the CLI, but can switch to stderr (required for
a future MCP stdio server, where stdout is reserved for the JSON-RPC stream).

Key properties vs. the previous version:
- Separate console level (default INFO) and file level (default DEBUG).
- Full date/time in timestamps (YYYY-MM-DD HH:MM:SS).
- Handlers are added idempotently; existing handlers are located by a marker
  attribute instead of being blindly cleared on every call.
- A file handler can be attached in a second call without touching the console
  handler already in place.
- Logs are always UTF-8.
"""

import logging
import sys
from pathlib import Path

_TEXT_FORMAT = "%(asctime)s [%(levelname)s] %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def _find_handler(root: logging.Logger, marker: str) -> logging.Handler | None:
    for handler in root.handlers:
        if getattr(handler, "_manhua_marker", None) == marker:
            return handler
    return None


def setup_logging(
    *,
    stream: str = "stdout",
    console_level: int = logging.INFO,
    file_level: int = logging.DEBUG,
    log_dir: str | Path | None = None,
    run_id: str | None = None,
    # Legacy positional-style call: setup_logging(stream="stdout", log_dir=...).
    # The old signature had `level` as the second arg. Accept it via **kwargs so
    # existing call sites that pass `level=logging.INFO` still work.
    **_legacy_kwargs,
) -> None:
    """Configure the root logger. Safe to call multiple times.

    - First call: attaches the console handler.
    - Second call with log_dir: attaches a per-run file handler without
      disturbing the console handler.
    - Third+ calls: update levels/formatters on the existing handlers.
    """
    root = logging.getLogger()
    root.setLevel(min(console_level, file_level))
    formatter = logging.Formatter(_TEXT_FORMAT, datefmt=_DATE_FORMAT)

    # Console handler
    console = _find_handler(root, "console")
    if console is None:
        target = sys.stdout if stream == "stdout" else sys.stderr
        try:
            target.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
        console = logging.StreamHandler(target)
        console._manhua_marker = "console"  # type: ignore[attr-defined]
        root.addHandler(console)
    console.setLevel(console_level)
    console.setFormatter(formatter)

    # File handler - only when log_dir is supplied
    if log_dir is not None:
        path = Path(log_dir)
        path.mkdir(parents=True, exist_ok=True)

        marker = f"run-file:{run_id}" if run_id else "run-file:default"
        file_handler = _find_handler(root, marker)
        if file_handler is None:
            filename = f"run_{run_id}.log" if run_id else "run.log"
            file_handler = logging.FileHandler(
                path / filename,
                encoding="utf-8",
            )
            file_handler._manhua_marker = marker  # type: ignore[attr-defined]
            root.addHandler(file_handler)
        file_handler.setLevel(file_level)
        file_handler.setFormatter(formatter)


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
