# Series-level glossary management, creation, and term appending.

import json
from datetime import datetime, timezone
from pathlib import Path

from manhua_pipeline.logging_setup import get_logger
from manhua_pipeline.manual_overrides import atomic_write_json

logger = get_logger(__name__)


# Return Path to series-level glossary JSON file
def series_glossary_path(base_dir: Path, config) -> Path:
    return base_dir / getattr(config, "GLOSSARY_NAME", "glossary.json")


_EMPTY_GLOSSARY = {
    "version": "v1",
    "updated_at": None,
    "terms": [],
}


# Load series-level glossary from disk, creating default schema if missing
def load_series_glossary(base_dir: Path, config) -> dict:
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


# Append new glossary term stub if not already present
def append_glossary_term(
    base_dir: Path,
    config,
    source_term: str,
    target_term: str = "",
    locked: bool = False,
) -> None:
    p = series_glossary_path(base_dir, config)
    g = load_series_glossary(base_dir, config)
    if any(t.get("source_term") == source_term for t in g.get("terms", [])):
        return
    g.setdefault("terms", []).append(
        {
            "source_term": source_term,
            "target_term": target_term,
            "locked": locked,
            "notes": "auto-added HINT; set target + set locked:true only if a real recurring name",
        }
    )
    g["updated_at"] = datetime.now(timezone.utc).isoformat()
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as fh:
            json.dump(g, fh, ensure_ascii=False, indent=2)
    except Exception as exc:
        logger.warning("Failed to save series glossary to %s: %s", p, exc)


# Fields a glossary term may carry. Unknown keys are dropped with a warning.
_ALLOWED_TERM_KEYS = {
    "source_term",
    "target_term",
    "locked",
    "notes",
    "category",
    "term_id",
    "auto_seeded",
    "source",
    "source_region",
    "request_id",
}

_OPTIONAL_KEYS = _ALLOWED_TERM_KEYS - {"source_term", "target_term"}


# Check whether text contains CJK characters
def _has_cjk(text: str) -> bool:
    return any(
        ("\u3400" <= ch <= "\u9fff") or ("\uf900" <= ch <= "\ufaff")
        for ch in (text or "")
    )


# Validate a submission entry's required fields. Return ((source, target), None)
# on success or (None, warning) when the entry must be rejected.
def _validate_entry_fields(
    raw: dict, index: int
) -> tuple[tuple[str, str] | None, str | None]:
    source = raw.get("source_term")
    if not isinstance(source, str) or not source.strip():
        return (
            None,
            f"Entry #{index} is missing required field 'source_term' - skipped.",
        )
    source = source.strip()

    target = raw.get("target_term")
    if not isinstance(target, str) or not target.strip():
        return None, (
            f"Entry #{index} ({source!r}) is missing required field "
            "'target_term' - skipped."
        )

    if not _has_cjk(source):
        return None, (
            f"Entry #{index} ({source!r}): 'source_term' must be CJK text - skipped."
        )

    return (source, target.strip()), None


# Validate and upsert glossary entries into the series glossary
def update_glossary_terms(base_dir: Path, config, entries: list) -> dict:
    if not isinstance(entries, list):
        raise ValueError("terms must be a list of JSON objects.")

    path = series_glossary_path(base_dir, config)
    glossary = load_series_glossary(base_dir, config)

    warnings: list[str] = []
    terms = glossary.get("terms")
    if not isinstance(terms, list):
        warnings.append("Glossary 'terms' is not a list - resetting it to empty.")
        terms = []
        glossary["terms"] = terms

    index = {t.get("source_term"): t for t in terms if isinstance(t, dict)}
    seen: set[str] = set()
    written = 0
    skipped = 0

    for i, raw in enumerate(entries, start=1):
        if not isinstance(raw, dict):
            warnings.append(f"Entry #{i} is not a JSON object - skipped.")
            skipped += 1
            continue

        fields, reject = _validate_entry_fields(raw, i)
        if reject is not None:
            warnings.append(reject)
            skipped += 1
            continue
        source, target = fields

        if source in seen:
            warnings.append(
                f"Duplicate source_term {source!r} in submission - "
                "the first occurrence was kept."
            )
            skipped += 1
            continue

        unknown = sorted(set(raw) - _ALLOWED_TERM_KEYS)
        if unknown:
            warnings.append(
                f"Entry #{i} ({source!r}) has unknown field(s) "
                f"{', '.join(unknown)} - ignored."
            )

        existing = index.get(source)
        if existing is not None:
            current = (existing.get("target_term") or "").strip()
            if existing.get("locked") and current and current != target:
                warnings.append(
                    f"Locked term {source!r} already maps to {current!r} - refusing "
                    f"to overwrite with {target!r}. Unlock it first if this change is "
                    "intended."
                )
                skipped += 1
                continue

            existing["target_term"] = target
            # Only override optional fields the caller actually supplied, so a
            # partial submission cannot silently unlock a term or wipe its notes.
            if "locked" in raw:
                if isinstance(raw["locked"], bool):
                    existing["locked"] = raw["locked"]
                else:
                    warnings.append(
                        f"Entry #{i} ({source!r}): 'locked' must be a boolean - "
                        "existing value kept."
                    )
            if "notes" in raw:
                if isinstance(raw["notes"], str):
                    existing["notes"] = raw["notes"]
                else:
                    warnings.append(
                        f"Entry #{i} ({source!r}): 'notes' must be a string - "
                        "existing value kept."
                    )
        else:
            locked = raw.get("locked", False)
            if not isinstance(locked, bool):
                warnings.append(
                    f"Entry #{i} ({source!r}): 'locked' must be a boolean - "
                    "defaulted to false."
                )
                locked = False
            notes = raw.get("notes", "")
            if not isinstance(notes, str):
                warnings.append(
                    f"Entry #{i} ({source!r}): 'notes' must be a string - "
                    "defaulted to empty."
                )
                notes = ""
            entry = {
                "source_term": source,
                "target_term": target,
                "locked": locked,
                "notes": notes,
            }
            for key in _OPTIONAL_KEYS:
                if key in raw and key not in ("locked", "notes"):
                    entry[key] = raw[key]
            terms.append(entry)
            index[source] = entry

        seen.add(source)
        written += 1

    summary = {
        "written": written,
        "skipped": skipped,
        "warnings": warnings,
        "warning_count": len(warnings),
        "path": str(path),
        "total_terms": len(terms),
    }

    if not written:
        return summary

    glossary["updated_at"] = datetime.now(timezone.utc).isoformat()
    atomic_write_json(path, glossary)
    logger.info(
        "[Glossary] Updated %s (%d written, %d skipped, %d warnings)",
        path,
        written,
        skipped,
        len(warnings),
    )
    return summary
