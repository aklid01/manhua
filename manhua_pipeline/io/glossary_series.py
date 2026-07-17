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


def merge_glossary(base_dir: Path, new_terms: list[dict]) -> None:
    """Merge new terms into the series-level glossary without clobbering locked terms."""
    import config

    p = series_glossary_path(base_dir, config)
    g = load_series_glossary(base_dir, config)

    existing_terms = g.get("terms", [])
    existing_by_id = {t["term_id"]: t for t in existing_terms if "term_id" in t}

    modified = False
    for nt in new_terms:
        tid = nt.get("term_id")
        if not tid:
            continue
        if tid in existing_by_id:
            if existing_by_id[tid].get("locked"):
                continue
            existing_by_id[tid].update(nt)
            modified = True
        else:
            existing_terms.append(nt)
            existing_by_id[tid] = nt
            modified = True

    if modified or not p.exists():
        g["terms"] = existing_terms
        g["updated_at"] = datetime.now(timezone.utc).isoformat()
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            with p.open("w", encoding="utf-8") as fh:
                json.dump(g, fh, ensure_ascii=False, indent=2)
        except Exception as exc:
            logger.warning("Failed to save series glossary to %s: %s", p, exc)
