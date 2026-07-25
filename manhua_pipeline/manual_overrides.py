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


def build_request_id(chapter: str, stage: str, region_id: str) -> str:
    """Stable, run-INDEPENDENT identity for a manual override request."""
    return f"{chapter}:{stage}:{region_id}"


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
    request_id = build_request_id(chapter, "translation", region_id)
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
    print(f"Chapter:  {request.chapter}")
    print(f"Page:     {request.page if request.page is not None else 'unknown'}")
    print(f"Region:   {request.region_id}")
    print(f"Source:   {request.original_text!r}")
    print(f"Reason:   {request.final_validation_reason or request.reason}")
    if request.classification:
        print(f"Class:    {request.classification}")
    if request.classification_evidence:
        print("Evidence: " + ", ".join(request.classification_evidence))
    if request.context_before:
        print(f"Previous: {request.context_before[-1]}")
    if request.context_after:
        print(f"Next:     {request.context_after[0]}")

    translation = input("Enter translation (blank to defer): ").strip()
    if not translation:
        return ManualOverrideDecision(action="defer")

    print("Save as:")
    print("  1. Region-only override  (this occurrence)")
    print("  2. Series glossary entry (locks this exact source everywhere)")
    print("  3. Leave unresolved      (defer)")
    print("  4. Abort stage")
    choice = input("Choice [1]: ").strip() or "1"

    if choice == "1":
        return ManualOverrideDecision(action="apply_region", translation=translation)
    if choice == "2":
        return ManualOverrideDecision(action="apply_glossary", translation=translation)
    if choice == "4":
        return ManualOverrideDecision(action="abort")
    return ManualOverrideDecision(action="defer")


def request_from_row(row: dict) -> ManualOverrideRequest:
    """Rebuild a ManualOverrideRequest from a stored queue row."""
    return ManualOverrideRequest(
        request_id=row["request_id"],
        run_id=row.get("run_id", "unknown"),
        stage=row.get("stage", "translation"),
        chapter=row.get("chapter", ""),
        region_id=row["region_id"],
        original_text=row.get("original_text", ""),
        reason=row.get("reason", ""),
        final_validation_reason=row.get("final_validation_reason", ""),
        backend=row.get("backend", "ollama"),
        attempts=int(row.get("attempts", 0)),
        suggested_scope=row.get("suggested_scope", "region"),
        page=row.get("page"),
        classification=row.get("classification"),
        classification_confidence=row.get("classification_confidence"),
        classification_evidence=tuple(row.get("classification_evidence", ())),
        model=row.get("model"),
        context_before=tuple(row.get("context_before", ())),
        context_after=tuple(row.get("context_after", ())),
        crop_path=row.get("crop_path"),
        status=row.get("status", "pending"),
        metadata=row.get("metadata", {}),
    )


class OverrideRequestStore:
    """JSON-backed queue of pending manual override requests.

    Dedup rule: one entry per request_id (chapter:stage:region). Re-queuing
    the same region UPDATES the existing pending entry rather than appending
    a duplicate. Resolved entries are marked, not deleted, for audit.
    """

    def __init__(self, workspace: Path):
        logs_dir = Path(workspace) / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        self.path = logs_dir / "manual_override_requests.json"

    def _load(self) -> list[dict]:
        if not self.path.exists():
            return []
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            return data if isinstance(data, list) else []
        except Exception:
            return []

    def _save(self, rows: list[dict]) -> None:
        import os
        from tempfile import NamedTemporaryFile
        with NamedTemporaryFile(
            "w", delete=False, dir=self.path.parent,
            suffix=".tmp", encoding="utf-8",
        ) as fh:
            json.dump(rows, fh, ensure_ascii=False, indent=2)
            tmp = Path(fh.name)
        os.replace(tmp, self.path)

    def upsert_pending(self, request: ManualOverrideRequest) -> None:
        """Insert a new pending request, or refresh an existing one by id."""
        rows = self._load()
        payload = asdict(request)
        payload["status"] = "pending"
        payload["queued_at"] = datetime.now(timezone.utc).isoformat()

        for i, row in enumerate(rows):
            if row.get("request_id") == request.request_id:
                payload["queued_at"] = row.get("queued_at", payload["queued_at"])
                if row.get("status") == "resolved":
                    return
                rows[i] = payload
                self._save(rows)
                return

        rows.append(payload)
        self._save(rows)

    def list_pending(self) -> list[dict]:
        return [r for r in self._load() if r.get("status") == "pending"]

    def mark_resolved(self, request_id: str, *, scope: str, translation: str) -> None:
        rows = self._load()
        for row in rows:
            if row.get("request_id") == request_id:
                row["status"] = "resolved"
                row["resolved_scope"] = scope
                row["resolved_translation"] = translation
                row["resolved_at"] = datetime.now(timezone.utc).isoformat()
                break
        self._save(rows)



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
