# JSON response parsing, auto-repair, and partial recovery for LLM backends.

import json
import re
from dataclasses import dataclass
from typing import Literal

ParseStatus = Literal[
    "success", "repaired", "partial_recovered", "blank", "invalid_json", "non_object"
]


# Encapsulates parsed JSON dictionary, parse status, and raw payload length
@dataclass(frozen=True)
class JsonParseResult:
    status: ParseStatus
    value: dict
    error: str | None = None
    raw_length: int = 0


_REGION_KV_RE = re.compile(
    r'["\'](P\d{3}_R\d{3})["\']\s*:\s*["\']((?:[^"\\]|\\.)*)["\']',
    re.DOTALL,
)
_TRAILING_COMMA_RE = re.compile(r",\s*([}\]])")
_UNQUOTED_KEY_RE = re.compile(r"([{,]\s*)(P\d{3}_R\d{3})\s*:")


# Strip markdown code fences and isolate JSON object block
def _strip_json_fences(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        inner = lines[1:] if len(lines) > 1 else lines
        if inner and inner[-1].strip() == "```":
            inner = inner[:-1]
        stripped = "\n".join(inner).strip()

    start = stripped.find("{")
    end = stripped.rfind("}")
    if start != -1 and end != -1 and end > start:
        return stripped[start : end + 1]

    return stripped


# Apply fast syntactic repairs to common LLM formatting flaws
def _repair_json_string(text: str) -> str:
    cleaned = text
    cleaned = _TRAILING_COMMA_RE.sub(r"\1", cleaned)
    cleaned = _UNQUOTED_KEY_RE.sub(r'\1"\2":', cleaned)
    if cleaned.startswith("{") and not cleaned.rstrip().endswith("}"):
        if cleaned.count('"') % 2 != 0:
            cleaned = cleaned.rstrip() + '"}'
        else:
            cleaned = cleaned.rstrip() + "}"
    return cleaned


# Extract all valid "PXXX_RYYY": "..." pairs via regex when full document parsing fails
def _harvest_partial_regions(text: str) -> dict[str, str]:
    results = {}
    for match in _REGION_KV_RE.finditer(text):
        rid, val = match.group(1), match.group(2)
        try:
            val = json.loads(f'"{val}"')
        except Exception:
            val = (
                val.replace('\\"', '"')
                .replace("\\n", "\n")
                .replace("\\r", "\r")
                .replace("\\t", "\t")
                .replace("\\\\", "\\")
            )
        results[rid] = val
    return results


# Parse backend response string into typed JsonParseResult with repair and partial fallback
def parse_json_result(text: str) -> JsonParseResult:
    if not text or not text.strip():
        return JsonParseResult(
            status="blank",
            value={},
            raw_length=len(text or ""),
        )

    raw_len = len(text)
    cleaned = _strip_json_fences(text)

    # 1. Fast path: native json.loads
    try:
        value = json.loads(cleaned)
        if isinstance(value, dict):
            return JsonParseResult(
                status="success",
                value=value,
                raw_length=raw_len,
            )
        return JsonParseResult(
            status="non_object",
            value={},
            raw_length=raw_len,
        )
    except json.JSONDecodeError as exc:
        first_error = str(exc)

    # 2. Syntax repair: trailing commas, unquoted keys, missing braces
    try:
        repaired = _repair_json_string(cleaned)
        value = json.loads(repaired)
        if isinstance(value, dict) and value:
            return JsonParseResult(
                status="repaired",
                value=value,
                raw_length=raw_len,
            )
    except json.JSONDecodeError:
        pass

    # 3. Partial key-value harvesting
    partial = _harvest_partial_regions(cleaned)
    if partial:
        return JsonParseResult(
            status="partial_recovered",
            value=partial,
            error=first_error,
            raw_length=raw_len,
        )

    return JsonParseResult(
        status="invalid_json",
        value={},
        error=first_error,
        raw_length=raw_len,
    )
