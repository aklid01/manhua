# Pipeline event taxonomy and structured event dataclass.

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

# Backend request and JSON parsing events
BACKEND_REQUEST_STARTED = "BACKEND_REQUEST_STARTED"
BACKEND_REQUEST_SUCCEEDED = "BACKEND_REQUEST_SUCCEEDED"
BACKEND_REQUEST_FAILED = "BACKEND_REQUEST_FAILED"
BACKEND_RESPONSE_BLANK = "BACKEND_RESPONSE_BLANK"
BACKEND_JSON_INVALID = "BACKEND_JSON_INVALID"
BACKEND_JSON_NON_OBJECT = "BACKEND_JSON_NON_OBJECT"

# Translation region validation events
TRANSLATION_KEY_MISSING = "TRANSLATION_KEY_MISSING"
TRANSLATION_VALUE_EMPTY = "TRANSLATION_VALUE_EMPTY"
TRANSLATION_VALUE_WHITESPACE = "TRANSLATION_VALUE_WHITESPACE"
TRANSLATION_VALUE_NON_STRING = "TRANSLATION_VALUE_NON_STRING"
TRANSLATION_VALUE_CJK_REJECTED = "TRANSLATION_VALUE_CJK_REJECTED"
TRANSLATION_VALUE_SCHEMA_REJECTED = "TRANSLATION_VALUE_SCHEMA_REJECTED"
TRANSLATION_UNEXPECTED_REGION_ID = "TRANSLATION_UNEXPECTED_REGION_ID"

# Region resolution and retry events
LOCKED_GLOSSARY_APPLIED = "LOCKED_GLOSSARY_APPLIED"
REGION_OVERRIDE_APPLIED = "REGION_OVERRIDE_APPLIED"
REGION_OVERRIDE_SHADOWED = "REGION_OVERRIDE_SHADOWED"
ISOLATED_RETRY_STARTED = "ISOLATED_RETRY_STARTED"
ISOLATED_RETRY_SUCCEEDED = "ISOLATED_RETRY_SUCCEEDED"
ISOLATED_RETRY_FAILED = "ISOLATED_RETRY_FAILED"
TRANSLATION_REGION_RESOLVED = "TRANSLATION_REGION_RESOLVED"
TRANSLATION_COMPLETION_BLOCKED = "TRANSLATION_COMPLETION_BLOCKED"

# Manual intervention events
MANUAL_OVERRIDE_REQUIRED = "MANUAL_OVERRIDE_REQUIRED"
MANUAL_OVERRIDE_ACCEPTED = "MANUAL_OVERRIDE_ACCEPTED"
MANUAL_OVERRIDE_DEFERRED = "MANUAL_OVERRIDE_DEFERRED"
MANUAL_OVERRIDE_ABORTED = "MANUAL_OVERRIDE_ABORTED"
MANUAL_OVERRIDE_CONFLICT = "MANUAL_OVERRIDE_CONFLICT"
MANUAL_OVERRIDE_ALREADY_RESOLVED = "MANUAL_OVERRIDE_ALREADY_RESOLVED"

# Stage, chapter, QA, and rendering lifecycle events
STAGE_STARTED = "STAGE_STARTED"
STAGE_COMPLETED = "STAGE_COMPLETED"
STAGE_FAILED = "STAGE_FAILED"
STAGE_SKIPPED = "STAGE_SKIPPED"
CHAPTER_STARTED = "CHAPTER_STARTED"
CHAPTER_COMPLETED = "CHAPTER_COMPLETED"
CHAPTER_FAILED = "CHAPTER_FAILED"
QA_ISSUE_DETECTED = "QA_ISSUE_DETECTED"
RENDER_REGION_COMPLETED = "RENDER_REGION_COMPLETED"
RENDER_ERASE_MASK_LOW_OVERLAP = "RENDER_ERASE_MASK_LOW_OVERLAP"
RENDER_TEXT_TOO_SMALL = "RENDER_TEXT_TOO_SMALL"


# Structured pipeline event model for logging and telemetry
@dataclass(frozen=True)
class PipelineEvent:
    event: str
    run_id: str
    stage: str
    chapter: str
    level: str = "INFO"
    region_id: str | None = None
    page: int | None = None
    reason: str | None = None
    classification: str | None = None
    classification_confidence: float | None = None
    classification_evidence: tuple[str, ...] = ()
    backend: str | None = None
    model: str | None = None
    attempt: int | None = None
    batch: int | None = None
    original_text: str | None = None
    raw_value_type: str | None = None
    request_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    # Convert event to dictionary representation
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
