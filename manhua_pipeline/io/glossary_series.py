import json
from datetime import datetime, timezone
from pathlib import Path

from manhua_pipeline.logging_setup import get_logger

logger = get_logger(__name__)


def series_glossary_path(base_dir: Path, config) -> Path:
    return base_dir / getattr(config, "GLOSSARY_NAME", "glossary.json")


_EMPTY_GLOSSARY = {
    "version": "v1",
    "updated_at": None,
    "terms": [],
}


def load_series_glossary(base_dir: Path, config) -> dict:
    """Load the series-level glossary.json, creating it if missing."""
    p = series_glossary_path(base_dir, config)
    if p.exists():
        try:
            with p.open("r", encoding="utf-8") as fh:
                return json.load(fh)
        except Exception as exc:
            logger.warning("Failed to load series glossary from %s: %s", p, exc)
    else:
        g = dict(_EMPTY_GLOSSARY)
        g["updated_at"] = datetime.now(timezone.utc).isoformat()
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            with p.open("w", encoding="utf-8") as fh:
                json.dump(g, fh, ensure_ascii=False, indent=2)
            logger.info("[Glossary] Created empty series glossary: %s", p)
        except Exception as exc:
            logger.warning("[Glossary] Could not create %s: %s", p, exc)
        return g
    return dict(_EMPTY_GLOSSARY)


def append_glossary_term(
    base_dir: Path, config, source_term: str, target_term: str = "", locked: bool = True
) -> None:
    """Add a term stub if the source_term isn't already present."""
    p = series_glossary_path(base_dir, config)
    g = load_series_glossary(base_dir, config)
    if any(t.get("source_term") == source_term for t in g.get("terms", [])):
        return
    g.setdefault("terms", []).append(
        {
            "source_term": source_term,
            "target_term": target_term,
            "locked": locked,
            "notes": "auto-added: needs manual target",
        }
    )
    g["updated_at"] = datetime.now(timezone.utc).isoformat()
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as fh:
            json.dump(g, fh, ensure_ascii=False, indent=2)
    except Exception as exc:
        logger.warning("Failed to save series glossary to %s: %s", p, exc)

