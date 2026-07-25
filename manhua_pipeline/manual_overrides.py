"""Manual override request schema, interactive prompt, and JSONL queue.

Rules:
- prompt_for_override() is the ONLY place allowed to call input().
- handle_manual_request() checks RuntimePolicy.can_prompt before calling it.
- Batch/non-TTY/MCP: emit event, append to queue, return defer decision.
- Atomic writes prevent partial JSON corruption.
"""

import json
import os
import secrets
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from manhua_pipeline.interaction import RuntimePolicy

OverrideScope = Literal["region", "series_glossary"]
ManualAction = Literal["apply_region", "apply_glossary", "defer", "abort"]


@dataclass(frozen=True)
class ManualOverrideRequest:
    request_id: str
    run_id: str
    stage: str
    chapter: str
    region_id: str
    original_text: str
    reason: str
    final_validation_reason: str
    backend: str
    attempts: int
    suggested_scope: OverrideScope
    page: int | None = None
    classification: str | None = None
    classification_confidence: float | None = None
    classification_evidence: tuple[str, ...] = ()
    model: str | None = None
    context_before: tuple[str, ...] = ()
    context_after: tuple[str, ...] = ()
    crop_path: str | None = None
    status: str = "pending"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ManualOverrideDecision:
    action: ManualAction
    translation: str | None = None


def build_request_id(run_id: str, chapter: str, stage: str, region_id: str) -> str:
    return f"{run_id}:{chapter}:{stage}:{region_id}"


def make_override_request(
    *,
    run_id: str,
    chapter: str,
    region: dict,
    initial_reason: str,
    final_reason: str,
    backend: str,
    model: str | None,
    attempts: int,
    classification: str | None = None,
    classification_confidence: float | None = None,
    classification_evidence: tuple[str, ...] = (),
    context_before: tuple[str, ...] = (),
    context_after: tuple[str, ...] = (),
) -> ManualOverrideRequest:
    region_id = region["region_id"]
    request_id = build_request_id(run_id, chapter, "translation", region_id)
    suggested: OverrideScope = "region"
    return ManualOverrideRequest(
        request_id=request_id,
        run_id=run_id,
        stage="translation",
        chapter=chapter,
        region_id=region_id,
        original_text=region.get("original_text", ""),
        reason=initial_reason,
        final_validation_reason=final_reason,
        backend=backend,
        model=model,
        attempts=attempts,
        suggested_scope=suggested,
        page=region.get("page_number"),
        classification=classification,
        classification_confidence=classification_confidence,
        classification_evidence=classification_evidence,
        context_before=context_before,
        context_after=context_after,
    )


def prompt_for_override(request: ManualOverrideRequest) -> ManualOverrideDecision:
    """Interactive terminal prompt. Only called when RuntimePolicy.can_prompt is True."""
    print("\n[MANUAL OVERRIDE REQUIRED]")
    print(f"Chapter: {request.chapter}")
    print(f"Page: {request.page or 'unknown'}")
    print(f"Region: {request.region_id}")
    print(f"Source: {request.original_text}")
    print(f"Reason: {request.final_validation_reason}")
    if request.classification:
        print(f"Classification: {request.classification}")
    if request.classification_evidence:
        print("Evidence: " + ", ".join(request.classification_evidence))
    if request.context_before:
        print(f"Previous: {request.context_before[-1]}")
    if request.context_after:
        print(f"Next: {request.context_after[0]}")

    translation = input("Translation (blank to defer): ").strip()
    if not translation:
        return ManualOverrideDecision("defer")

    print("Save as:")
    print("  1. Region-only override (default)")
    print("  2. Locked series glossary entry")
    print("  3. Leave unresolved (defer)")
    print("  4. Abort stage")

    choice = input("Choice [1]: ").strip() or "1"

    if choice == "2":
        return ManualOverrideDecision("apply_glossary", translation)
    if choice == "4":
        return ManualOverrideDecision("abort")
    if choice == "3":
        return ManualOverrideDecision("defer")
    return ManualOverrideDecision("apply_region", translation)


class OverrideRequestStore:
    """JSONL-backed queue for pending manual override requests."""

    def __init__(self, chapter_dir: Path) -> None:
        self.path = chapter_dir / "manual_override_requests.jsonl"

    def append_pending(self, request: ManualOverrideRequest) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        record = asdict(request)
        record["queued_at"] = datetime.now(timezone.utc).isoformat()
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    def list_pending(self) -> list[dict]:
        if not self.path.exists():
            return []
        results = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                if record.get("status") == "pending":
                    results.append(record)
            except json.JSONDecodeError:
                continue
        return results


def atomic_write_json(path: str | Path, data: object) -> None:
    """Write JSON atomically via a temp file + os.replace."""
    from tempfile import NamedTemporaryFile

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        mode="w",
        delete=False,
        dir=destination.parent,
        suffix=".tmp",
        encoding="utf-8",
    ) as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, destination)


class LockedGlossaryConflict(Exception):
    def __init__(self, source_term: str, current: str, proposed: str) -> None:
        self.source_term = source_term
        self.current = current
        self.proposed = proposed
        super().__init__(
            f"Locked glossary conflict for {source_term!r}: "
            f"current={current!r}, proposed={proposed!r}"
        )


def save_region_override(
    overrides: dict,
    request: ManualOverrideRequest,
    translation: str,
    path: Path,
) -> None:
    overrides[request.region_id] = {
        "translation": translation,
        "source": "manual_interactive",
        "request_id": request.request_id,
        "original_text": request.original_text,
    }
    atomic_write_json(path, overrides)


def save_locked_glossary_entry(
    glossary: list[dict],
    request: ManualOverrideRequest,
    translation: str,
    path: Path,
) -> None:
    existing = next(
        (
            item
            for item in glossary
            if item.get("source_term") == request.original_text
        ),
        None,
    )
    if existing:
        current = (existing.get("target_term") or "").strip()
        if existing.get("locked") and current and current != translation:
            raise LockedGlossaryConflict(
                source_term=request.original_text,
                current=current,
                proposed=translation,
            )
        existing.update(
            target_term=translation,
            locked=True,
            source="manual_interactive",
            request_id=request.request_id,
        )
    else:
        glossary.append(
            {
                "source_term": request.original_text,
                "target_term": translation,
                "category": "character"
                if request.classification == "name_candidate"
                else "manual",
                "locked": True,
                "source": "manual_interactive",
                "request_id": request.request_id,
            }
        )
    atomic_write_json(path, glossary)
