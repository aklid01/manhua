# Thread-safe JSONL event emission and logger integration.

import json
import logging
from pathlib import Path
from threading import Lock
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from manhua_pipeline.events import PipelineEvent

_LOGGABLE_FIELDS = (
    "region_id",
    "page",
    "reason",
    "classification",
    "backend",
    "model",
    "attempt",
    "batch",
    "request_id",
)


# Thread-safe structured event emitter writing to Python logger and JSONL event log
class EventEmitter:
    def __init__(
        self,
        *,
        logger: logging.Logger,
        event_path: Path,
        include_source_text: bool = False,
        max_text_chars: int = 200,
    ) -> None:
        self.logger = logger
        self.event_path = event_path
        self.include_source_text = include_source_text
        self.max_text_chars = max_text_chars
        self._lock = Lock()
        self.event_path.parent.mkdir(parents=True, exist_ok=True)

    # Sanitize event payload according to source text retention settings
    def _sanitized(self, event: "PipelineEvent") -> dict:
        payload = event.to_dict()
        if not self.include_source_text:
            payload["original_text"] = None
        elif payload.get("original_text"):
            payload["original_text"] = payload["original_text"][: self.max_text_chars]
        return payload

    # Format event log line and append JSON event to output file
    def emit(self, event: "PipelineEvent") -> None:
        payload = self._sanitized(event)

        fields = [
            f"event={event.event}",
            f"run_id={event.run_id}",
            f"stage={event.stage}",
            f"chapter={event.chapter}",
        ]
        for key in _LOGGABLE_FIELDS:
            value = getattr(event, key)
            if value is not None:
                fields.append(f"{key}={json.dumps(value, ensure_ascii=False)}")

        log_method = getattr(self.logger, event.level.lower(), self.logger.info)
        log_method(" ".join(fields))

        with self._lock:
            with self.event_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


# Initialize and return EventEmitter for target workspace
def get_event_emitter(
    workspace: Path | str,
    config: object,
    logger: logging.Logger | None = None,
    include_source_text: bool = False,
) -> EventEmitter:
    ws = Path(workspace)
    logs_dir = ws / getattr(config, "STAGE_FOLDERS", {}).get("logs", "logs")
    event_path = logs_dir / "events.jsonl"
    log = logger or logging.getLogger("manhua_pipeline")
    return EventEmitter(
        logger=log,
        event_path=event_path,
        include_source_text=include_source_text,
    )


# Helper to emit stage lifecycle event
def emit_stage_event(
    workspace: Path | str,
    config: object,
    stage_name: str,
    event_type: str,
    logger: logging.Logger | None = None,
    run_id: str | None = None,
    chapter: str | None = None,
    reason: str | None = None,
    metadata: dict | None = None,
    level: str = "INFO",
) -> None:
    ws = Path(workspace)
    ch = chapter or ws.name
    rid = run_id or getattr(config, "_run_id", "unknown")
    emitter = get_event_emitter(ws, config, logger=logger)
    from manhua_pipeline.events import PipelineEvent

    emitter.emit(
        PipelineEvent(
            event=event_type,
            run_id=rid,
            stage=stage_name,
            chapter=ch,
            level=level,
            reason=reason,
            metadata=metadata or {},
        )
    )
