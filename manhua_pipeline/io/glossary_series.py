import json
from datetime import datetime, timezone
from pathlib import Path

from manhua_pipeline.logging_setup import get_logger

logger = get_logger(__name__)


def series_glossary_path(base_dir: Path, config) -> Path:
    return base_dir / getattr(config, "GLOSSARY_NAME", "glossary.json")


def load_series_glossary(base_dir: Path, config) -> dict:
    """Load the series-level glossary.json."""
    p = series_glossary_path(base_dir, config)
    if p.exists():
        try:
            with p.open("r", encoding="utf-8") as fh:
                return json.load(fh)
        except Exception as exc:
            logger.warning("Failed to load series glossary from %s: %s", p, exc)
    return {
        "version": "v1",
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "terms": [],
    }
