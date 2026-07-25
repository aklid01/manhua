"""Pure validation and structured outcomes for translation responses.

Validation never calls input(), writes files, or manipulates UI.
All rejection reasons are preserved as typed fields.
"""

from dataclasses import dataclass
from typing import Any, Literal
import re

ValidationStatus = Literal["accepted", "missing", "rejected"]

_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")

_FULLWIDTH_MAP = {
    "\uff01": "!",
    "\uff1f": "?",
    "\uff0c": ",",
    "\u3002": ".",
    "\uff1a": ":",
    "\uff1b": ";",
    "\uff08": "(",
    "\uff09": ")",
    "\u201c": '"',
    "\u201d": '"',
    "\u2018": "'",
    "\u2019": "'",
    "\u3001": ",",
}


def _contains_cjk(text: str) -> bool:
    return bool(_CJK_RE.search(text or ""))


def _cjk_char_count(text: str) -> int:
    return sum(1 for c in (text or "") if _contains_cjk(c))


def _cjk_ratio(text: str) -> float:
    chars = [c for c in text if not c.isspace()]
    if not chars:
        return 0.0
    return sum(1 for c in chars if _contains_cjk(c)) / len(chars)


def normalize_newlines(text: str) -> str:
    return (
        text.replace("\\r\\n", "\n")
        .replace("\\n", "\n")
        .replace("\\r", "\n")
    )


def normalize_punct(text: str) -> str:
    return "".join(_FULLWIDTH_MAP.get(c, c) for c in text)


@dataclass(frozen=True)
class TranslationIssue:
    region_id: str
    status: ValidationStatus
    accepted_text: str | None = None
    raw_value: Any = None
    raw_value_type: str | None = None
    reason: str | None = None
    attempt: int = 1
    source: str = "ollama"


def validate_expected_regions(
    parsed: dict,
    expected_ids: set[str],
    *,
    attempt: int = 1,
    source: str = "ollama",
) -> dict[str, TranslationIssue]:
    """Validate a parsed response dict against expected region IDs.

    Each rejection reason is preserved distinctly:
      missing_key        - region_id absent from parsed dict
      non_string_value   - value is not a str
      empty_string       - value is ""
      whitespace_only    - value is non-empty but all whitespace
      cjk_ratio_exceeded - >30% Han ideographs remain in translation

    Accepted entries are normalized and stored in accepted_text.
    """
    results: dict[str, TranslationIssue] = {}

    for region_id in expected_ids:
        if region_id not in parsed:
            results[region_id] = TranslationIssue(
                region_id=region_id,
                status="missing",
                reason="missing_key",
                attempt=attempt,
                source=source,
            )
            continue

        raw = parsed[region_id]

        if not isinstance(raw, str):
            results[region_id] = TranslationIssue(
                region_id=region_id,
                status="rejected",
                raw_value=raw,
                raw_value_type=type(raw).__name__,
                reason="non_string_value",
                attempt=attempt,
                source=source,
            )
            continue

        if raw == "":
            results[region_id] = TranslationIssue(
                region_id=region_id,
                status="rejected",
                raw_value=raw,
                reason="empty_string",
                attempt=attempt,
                source=source,
            )
            continue

        if not raw.strip():
            results[region_id] = TranslationIssue(
                region_id=region_id,
                status="rejected",
                raw_value=raw,
                reason="whitespace_only",
                attempt=attempt,
                source=source,
            )
            continue

        normalized = normalize_punct(normalize_newlines(raw.strip()))

        if _cjk_ratio(normalized) > 0.30:
            results[region_id] = TranslationIssue(
                region_id=region_id,
                status="rejected",
                raw_value=raw,
                reason="cjk_ratio_exceeded",
                attempt=attempt,
                source=source,
            )
            continue

        results[region_id] = TranslationIssue(
            region_id=region_id,
            status="accepted",
            accepted_text=normalized,
            raw_value=raw,
            attempt=attempt,
            source=source,
        )

    return results


def is_meaningful_region(original_text: str, config) -> bool:
    """True if a region has CJK content that is meaningful (not trivially short/empty).

    Short CJK text is NOT trivial by length alone - 2-3 character names are
    meaningful and must not be silently dropped.
    """
    text = (original_text or "").strip()
    if not text:
        return False
    thresh = getattr(config, "OLLAMA_TRIVIAL_CJK_CHARS", 1)
    n = _cjk_char_count(text)
    if n == 0:
        return False
    # Only 0-char regions or single-char noise below the threshold are non-meaningful.
    # The threshold MUST stay at 1 (default) - raising it to 3 would silently drop names.
    return n > thresh
