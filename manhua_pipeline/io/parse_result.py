# JSON response parsing and classification for LLM backends.

import json
from dataclasses import dataclass
from typing import Literal

ParseStatus = Literal["success", "blank", "invalid_json", "non_object"]


# Encapsulates parsed JSON dictionary, parse status, and raw payload length
@dataclass(frozen=True)
class JsonParseResult:
    status: ParseStatus
    value: dict
    error: str | None = None
    raw_length: int = 0


# Parse backend response string into typed JsonParseResult
def parse_json_result(text: str) -> JsonParseResult:
    if not text or not text.strip():
        return JsonParseResult(
            status="blank",
            value={},
            raw_length=len(text or ""),
        )

    cleaned = _strip_json_fences(text)

    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        return JsonParseResult(
            status="invalid_json",
            value={},
            error=str(exc),
            raw_length=len(text),
        )

    if not isinstance(value, dict):
        return JsonParseResult(
            status="non_object",
            value={},
            raw_length=len(text),
        )

    return JsonParseResult(
        status="success",
        value=value,
        raw_length=len(text),
    )


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
