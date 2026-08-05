from manhua_pipeline.logging_setup import get_logger

logger = get_logger(__name__)

_STAGE_INDEX = 3
_TOTAL_STAGES = 7
_STAGE_NAME = "Translation"


def _locked_terms(glossary: dict) -> list[dict]:
    return [t for t in glossary.get("terms", []) if t.get("locked")]


def _build_locked_index(locked: list[dict]) -> dict[str, str]:
    return {
        t["source_term"]: t["target_term"]
        for t in locked
        if t.get("source_term") and t.get("target_term", "").strip()
    }


def _resolve_locked_glossary(
    regions: list[dict], locked_index: dict[str, str]
) -> tuple[dict[str, str], list[dict]]:
    resolved: dict[str, str] = {}
    remaining: list[dict] = []
    for region in regions:
        original = region.get("original_text", "")
        locked_value = locked_index.get(original)
        if locked_value and locked_value.strip():
            resolved[region["region_id"]] = locked_value
            logger.info(
                "[%d/%d %s] %s -> locked glossary match (%r)",
                _STAGE_INDEX,
                _TOTAL_STAGES,
                _STAGE_NAME,
                region["region_id"],
                locked_value,
            )
        else:
            remaining.append(region)
    return resolved, remaining


def _enforce_glossary(
    original_zh: str, translation: str, locked: list[dict]
) -> tuple[str, list[str], bool]:
    applied = []
    conflict = False
    for term in locked:
        source = term.get("source_term", "")
        target = term.get("target_term", "")
        term_id = term.get("term_id", "")
        if not source or not target:
            continue
        if source in original_zh:
            if target in translation:
                applied.append(term_id)
            else:
                applied.append(term_id)
                conflict = True
    return translation, applied, conflict
