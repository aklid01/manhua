import json
from dataclasses import dataclass
from typing import Literal

ParseStatus = Literal["success", "blank", "invalid_json", "non_object"]


@dataclass(frozen=True)
class JsonParseResult:
    status: ParseStatus
    value: dict
    error: str | None = None
    raw_length: int = 0


def parse_json_result(text: str) -> JsonParseResult:
    """Parse a backend JSON response and preserve the exact failure reason.

    Distinguishes four outcomes that must never be collapsed into one:
    - blank: empty or whitespace-only body
    - invalid_json: body present but not parseable
    - non_object: valid JSON but not a dict
    - success: a dict was returned
    """
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


def _strip_json_fences(text: str) -> str:
    """Remove markdown code fences and extract the first {...} object."""
    stripped = text.strip()
    # Remove ```json ... ``` or ``` ... ``` wrappers
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        inner = lines[1:] if len(lines) > 1 else lines
        # Drop trailing ``` line
        if inner and inner[-1].strip() == "```":
            inner = inner[:-1]
        stripped = "\n".join(inner).strip()

    # Find first { ... } block if there is surrounding prose
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start != -1 and end != -1 and end > start:
        return stripped[start : end + 1]

    return stripped
