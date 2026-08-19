# Console and file logging configuration.

import logging
import sys
from pathlib import Path

_TEXT_FORMAT = "%(asctime)s [%(levelname)s] %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


# Locate an existing logging handler by marker attribute
def _find_handler(root: logging.Logger, marker: str) -> logging.Handler | None:
    for handler in root.handlers:
        if getattr(handler, "_manhua_marker", None) == marker:
            return handler
    return None


# Configure console and file logging handlers idempotently
def setup_logging(
    *,
    stream: str = "stdout",
    console_level: int = logging.INFO,
    file_level: int = logging.DEBUG,
    log_dir: str | Path | None = None,
    run_id: str | None = None,
    **_legacy_kwargs,
) -> None:
    root = logging.getLogger()
    root.setLevel(min(console_level, file_level))
    formatter = logging.Formatter(_TEXT_FORMAT, datefmt=_DATE_FORMAT)

    console = _find_handler(root, "console")
    if console is None:
        target = sys.stdout if stream == "stdout" else sys.stderr
        try:
            target.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
        console = logging.StreamHandler(target)
        console._manhua_marker = "console"
        root.addHandler(console)
    console.setLevel(console_level)
    console.setFormatter(formatter)

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
            file_handler._manhua_marker = marker
            root.addHandler(file_handler)
        file_handler.setLevel(file_level)
        file_handler.setFormatter(formatter)


# Return named logger instance
def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


# Emit standardized stage progress log message
def log_stage(logger, index: int, total: int, name: str, message: str = "") -> None:
    logger.info("[%d/%d %s] %s", index, total, name, message)


# Emit standardized page progress log message
def log_page(
    logger,
    stage_index: int,
    total_stages: int,
    stage: str,
    page: int,
    total_pages: int,
    message: str = "",
) -> None:
    logger.info(
        "[%d/%d %s] Page %03d/%03d %s",
        stage_index,
        total_stages,
        stage,
        page,
        total_pages,
        message,
    )
