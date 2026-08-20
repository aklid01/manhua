# Pure validation and structured outcome evaluation for translation responses.

import re
from dataclasses import dataclass
from typing import Any, Literal

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


# Check if string contains any CJK ideographs
def _contains_cjk(text: str) -> bool:
    return bool(_CJK_RE.search(text or ""))


# Count number of CJK characters in string
def _cjk_char_count(text: str) -> int:
    return sum(1 for c in (text or "") if _contains_cjk(c))


# Calculate ratio of CJK characters to total non-whitespace characters
def _cjk_ratio(text: str) -> float:
    chars = [c for c in text if not c.isspace()]
    if not chars:
        return 0.0
    return sum(1 for c in chars if _contains_cjk(c)) / len(chars)


_ESCAPED_NEWLINE_RE = re.compile(r"\\(?:r\\n|n|r)", re.IGNORECASE)


# Normalize escaped newline characters to actual newlines
def normalize_newlines(text: str) -> str:
    if not text:
        return text
    return _ESCAPED_NEWLINE_RE.sub("\n", text)


# Convert fullwidth CJK punctuation to ASCII equivalents
def normalize_punct(text: str) -> str:
    return "".join(_FULLWIDTH_MAP.get(c, c) for c in text)


# Container for region translation validation result and error details
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


_REGION_KEY_RE = re.compile(r"^[pP]?(\d+)[_\-rR]+(\d+)$")
_GENERIC_KEYS = {
    "translation",
    "text",
    "output",
    "result",
    "english",
    "response",
    "translated_text",
    "final_english",
    "content",
}


# Fuzzy-normalize region keys in parsed dictionary against expected region IDs
def canonicalize_region_keys(parsed: dict, expected_ids: set[str]) -> dict:
    if not parsed or not expected_ids:
        return parsed or {}

    canonical = {k: v for k, v in parsed.items() if k in expected_ids}
    remaining_parsed = {k: v for k, v in parsed.items() if k not in expected_ids}
    remaining_expected = expected_ids - set(canonical.keys())

    if not remaining_parsed or not remaining_expected:
        return {**canonical, **remaining_parsed}

    expected_norm: dict[tuple[int, int], str] = {}
    for eid in remaining_expected:
        m = _REGION_KEY_RE.match(eid)
        if m:
            expected_norm[(int(m.group(1)), int(m.group(2)))] = eid

    for k, v in list(remaining_parsed.items()):
        m = _REGION_KEY_RE.match(k)
        if m:
            key_norm = (int(m.group(1)), int(m.group(2)))
            if key_norm in expected_norm:
                target_id = expected_norm.pop(key_norm)
                canonical[target_id] = v
                remaining_parsed.pop(k)
                remaining_expected.remove(target_id)

    if len(remaining_expected) == 1 and len(remaining_parsed) == 1:
        target_id = next(iter(remaining_expected))
        k, v = next(iter(remaining_parsed.items()))
        if k.lower() in _GENERIC_KEYS:
            canonical[target_id] = v
            remaining_parsed.pop(k)

    return {**canonical, **remaining_parsed}


# Validate parsed translation dictionary against expected region IDs
def validate_expected_regions(
    parsed: dict,
    expected_ids: set[str],
    *,
    attempt: int = 1,
    source: str = "ollama",
) -> dict[str, TranslationIssue]:
    parsed = canonicalize_region_keys(parsed, expected_ids)
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


# Check whether region contains meaningful CJK dialogue beyond trivial noise
def is_meaningful_region(original_text: str, config) -> bool:
    text = (original_text or "").strip()
    if not text:
        return False
    thresh = getattr(config, "OLLAMA_TRIVIAL_CJK_CHARS", 1)
    n = _cjk_char_count(text)
    if n == 0:
        return False
    return n > thresh
