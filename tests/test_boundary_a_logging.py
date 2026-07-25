"""Tests for Boundary A: minimum logging and event foundation."""

import json
import logging
import sys
from pathlib import Path

import pytest

from manhua_pipeline.events import (
    BACKEND_JSON_INVALID,
    BACKEND_RESPONSE_BLANK,
    MANUAL_OVERRIDE_REQUIRED,
    TRANSLATION_VALUE_EMPTY,
    PipelineEvent,
)
from manhua_pipeline.io.event_emitter import EventEmitter
from manhua_pipeline.io.parse_result import JsonParseResult, parse_json_result
from manhua_pipeline.logging_setup import setup_logging
from manhua_pipeline.run_context import RunContext, create_run_context


# ---------------------------------------------------------------------------
# RunContext
# ---------------------------------------------------------------------------


def test_run_id_format():
    ctx = create_run_context()
    assert ctx.run_id, "run_id must be non-empty"
    # Format: 20260725T102930.123456Z-abc123
    parts = ctx.run_id.split("-")
    assert len(parts) == 2
    assert len(parts[1]) == 6, "hex token must be 6 chars"


def test_run_id_is_unique():
    ids = {create_run_context().run_id for _ in range(10)}
    assert len(ids) == 10, "run_ids must be unique across invocations"


def test_run_context_is_frozen():
    ctx = create_run_context()
    with pytest.raises((AttributeError, TypeError)):
        ctx.run_id = "tampered"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Logging setup - idempotency
# ---------------------------------------------------------------------------


def test_repeated_setup_does_not_duplicate_console_handlers():
    root = logging.getLogger()
    before = len(root.handlers)
    setup_logging(stream="stderr", console_level=logging.INFO)
    setup_logging(stream="stderr", console_level=logging.INFO)
    after = len(root.handlers)
    assert after == before + 1 or after == before, (
        "Repeated setup_logging must not add duplicate console handlers"
    )


def test_second_call_can_attach_file_handler(tmp_path):
    run_id = create_run_context().run_id
    root = logging.getLogger()

    setup_logging(stream="stderr", console_level=logging.WARNING)
    handlers_before = len(root.handlers)

    setup_logging(
        stream="stderr",
        console_level=logging.WARNING,
        log_dir=tmp_path,
        run_id=run_id,
    )
    handlers_after = len(root.handlers)
    assert handlers_after == handlers_before + 1, (
        "Second call with log_dir must attach exactly one new file handler"
    )

    log_file = tmp_path / f"run_{run_id}.log"
    assert log_file.exists(), "Log file must be created"


def test_console_and_file_levels_are_independent(tmp_path):
    run_id = create_run_context().run_id
    setup_logging(
        stream="stderr",
        console_level=logging.WARNING,
        file_level=logging.DEBUG,
        log_dir=tmp_path,
        run_id=run_id,
    )
    root = logging.getLogger()
    markers = {
        getattr(h, "_manhua_marker", ""): h.level for h in root.handlers
    }
    assert markers.get("console") == logging.WARNING
    assert markers.get(f"run-file:{run_id}") == logging.DEBUG


def test_log_file_uses_full_date_format(tmp_path):
    run_id = create_run_context().run_id
    setup_logging(
        stream="stderr",
        log_dir=tmp_path,
        run_id=run_id,
    )
    log = logging.getLogger("test_date_format")
    log.info("checking date format")
    log_file = tmp_path / f"run_{run_id}.log"
    content = log_file.read_text(encoding="utf-8")
    import re
    assert re.search(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", content), (
        "Log file must include full date (YYYY-MM-DD HH:MM:SS)"
    )


# ---------------------------------------------------------------------------
# PipelineEvent
# ---------------------------------------------------------------------------


def _make_event(**kwargs) -> PipelineEvent:
    defaults = dict(
        event=TRANSLATION_VALUE_EMPTY,
        run_id="run-001",
        stage="translation",
        chapter="chapter_001",
    )
    defaults.update(kwargs)
    return PipelineEvent(**defaults)


def test_pipeline_event_is_frozen():
    ev = _make_event()
    with pytest.raises((AttributeError, TypeError)):
        ev.event = "CHANGED"  # type: ignore[misc]


def test_pipeline_event_to_dict_is_serializable():
    ev = _make_event(region_id="P001_R001", reason="empty_string")
    data = ev.to_dict()
    assert json.dumps(data)  # must not raise


def test_pipeline_event_timestamp_present():
    ev = _make_event()
    assert ev.timestamp, "timestamp must be set automatically"


# ---------------------------------------------------------------------------
# EventEmitter
# ---------------------------------------------------------------------------


def _make_emitter(tmp_path, **kwargs) -> EventEmitter:
    return EventEmitter(
        logger=logging.getLogger("test_emitter"),
        event_path=tmp_path / "events.jsonl",
        **kwargs,
    )


def test_event_emitter_writes_valid_jsonl(tmp_path):
    emitter = _make_emitter(tmp_path)
    ev = _make_event(region_id="P001_R001")
    emitter.emit(ev)

    lines = (tmp_path / "events.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    parsed = json.loads(lines[0])
    assert parsed["event"] == TRANSLATION_VALUE_EMPTY
    assert parsed["region_id"] == "P001_R001"


def test_event_emitter_source_text_redacted_by_default(tmp_path):
    emitter = _make_emitter(tmp_path, include_source_text=False)
    ev = _make_event(original_text="季青临")
    emitter.emit(ev)

    lines = (tmp_path / "events.jsonl").read_text(encoding="utf-8").splitlines()
    parsed = json.loads(lines[0])
    assert parsed["original_text"] is None


def test_event_emitter_source_text_included_when_enabled(tmp_path):
    emitter = _make_emitter(tmp_path, include_source_text=True)
    ev = _make_event(original_text="季青临")
    emitter.emit(ev)

    lines = (tmp_path / "events.jsonl").read_text(encoding="utf-8").splitlines()
    parsed = json.loads(lines[0])
    assert parsed["original_text"] == "季青临"


def test_event_emitter_truncates_long_source_text(tmp_path):
    emitter = _make_emitter(tmp_path, include_source_text=True, max_text_chars=5)
    ev = _make_event(original_text="abcdefghij")
    emitter.emit(ev)

    lines = (tmp_path / "events.jsonl").read_text(encoding="utf-8").splitlines()
    assert json.loads(lines[0])["original_text"] == "abcde"


def test_event_emitter_jsonl_is_utf8(tmp_path):
    emitter = _make_emitter(tmp_path, include_source_text=True)
    emitter.emit(_make_event(original_text="季青临"))
    raw = (tmp_path / "events.jsonl").read_bytes()
    raw.decode("utf-8")  # must not raise


def test_event_emitter_appends_multiple_events(tmp_path):
    emitter = _make_emitter(tmp_path)
    emitter.emit(_make_event(event=TRANSLATION_VALUE_EMPTY))
    emitter.emit(_make_event(event=MANUAL_OVERRIDE_REQUIRED))
    lines = (tmp_path / "events.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["event"] == TRANSLATION_VALUE_EMPTY
    assert json.loads(lines[1])["event"] == MANUAL_OVERRIDE_REQUIRED


# ---------------------------------------------------------------------------
# parse_json_result
# ---------------------------------------------------------------------------


def test_blank_response_gives_blank_status():
    result = parse_json_result("")
    assert result.status == "blank"
    assert result.value == {}


def test_whitespace_response_gives_blank_status():
    result = parse_json_result("   \n  ")
    assert result.status == "blank"


def test_blank_parse_is_distinct_from_invalid_json():
    blank = parse_json_result("")
    invalid = parse_json_result("not json at all {{")
    assert blank.status == "blank"
    assert invalid.status == "invalid_json"


def test_invalid_json_preserves_error_message():
    result = parse_json_result("{bad json}")
    assert result.status == "invalid_json"
    assert result.error is not None


def test_non_object_json_gives_non_object_status():
    result = parse_json_result("[1, 2, 3]")
    assert result.status == "non_object"
    assert result.value == {}


def test_valid_json_object_gives_success():
    result = parse_json_result('{"P001_R001": "Hello"}')
    assert result.status == "success"
    assert result.value == {"P001_R001": "Hello"}


def test_json_inside_code_fence_is_parsed():
    fenced = '```json\n{"P001_R001": "Hello"}\n```'
    result = parse_json_result(fenced)
    assert result.status == "success"
    assert result.value["P001_R001"] == "Hello"


def test_empty_dict_gives_success():
    result = parse_json_result("{}")
    assert result.status == "success"
    assert result.value == {}


def test_raw_length_is_recorded():
    text = '{"P001_R001": "Hi"}'
    result = parse_json_result(text)
    assert result.raw_length == len(text)


def test_blank_parse_result_is_frozen():
    result = parse_json_result("")
    with pytest.raises((AttributeError, TypeError)):
        result.status = "success"  # type: ignore[misc]
