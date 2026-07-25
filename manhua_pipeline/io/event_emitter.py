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


class EventEmitter:
    """Thread-safe structured event emitter that logs to a Python logger and
    appends JSON Lines to an event sink file.

    Note: the thread lock is sufficient for multi-threaded execution. For
    multi-process batch runs, use one event file per run/chapter and aggregate
    afterwards, or switch to a process-safe queue.
    """

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

    def _sanitized(self, event: "PipelineEvent") -> dict:
        payload = event.to_dict()
        if not self.include_source_text:
            payload["original_text"] = None
        elif payload.get("original_text"):
            payload["original_text"] = payload["original_text"][: self.max_text_chars]
        return payload

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
