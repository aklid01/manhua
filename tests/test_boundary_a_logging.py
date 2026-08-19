# Unit tests for RunContext, logging setup, PipelineEvent, and EventEmitter.

import json
import logging
import re

import pytest
from PIL import Image

import config
import pipeline
from manhua_pipeline.events import (
    MANUAL_OVERRIDE_REQUIRED,
    TRANSLATION_VALUE_EMPTY,
    PipelineEvent,
)
from manhua_pipeline.io.event_emitter import EventEmitter
from manhua_pipeline.io.parse_result import parse_json_result
from manhua_pipeline.logging_setup import setup_logging
from manhua_pipeline.run_context import create_run_context


# Verify generated run_id conforms to expected timestamp and hex token format
def test_run_id_format():
    ctx = create_run_context()
    assert ctx.run_id, "run_id must be non-empty"
    parts = ctx.run_id.split("-")
    assert len(parts) == 2
    assert len(parts[1]) == 6, "hex token must be 6 chars"


# Verify run_id values are uniquely generated across consecutive calls
def test_run_id_is_unique():
    ids = {create_run_context().run_id for _ in range(10)}
    assert len(ids) == 10, "run_ids must be unique across invocations"


# Verify RunContext is immutable
def test_run_context_is_frozen():
    ctx = create_run_context()
    with pytest.raises((AttributeError, TypeError)):
        ctx.run_id = "tampered"  # type: ignore[misc]


# Verify repeated setup_logging calls do not attach duplicate console handlers
def test_repeated_setup_does_not_duplicate_console_handlers():
    root = logging.getLogger()
    before = len(root.handlers)
    setup_logging(stream="stderr", console_level=logging.INFO)
    setup_logging(stream="stderr", console_level=logging.INFO)
    after = len(root.handlers)
    assert after == before + 1 or after == before, (
        "Repeated setup_logging must not add duplicate console handlers"
    )


# Verify second setup_logging call attaches file handler when log_dir is provided
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


# Verify console and file handlers maintain independent log level filtering
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
    markers = {getattr(h, "_manhua_marker", ""): h.level for h in root.handlers}
    assert markers.get("console") == logging.WARNING
    assert markers.get(f"run-file:{run_id}") == logging.DEBUG


# Verify log file timestamps include full year, date, and time
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

    assert re.search(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", content), (
        "Log file must include full date (YYYY-MM-DD HH:MM:SS)"
    )


# Helper to construct sample PipelineEvent instances
def _make_event(**kwargs) -> PipelineEvent:
    defaults = dict(
        event=TRANSLATION_VALUE_EMPTY,
        run_id="run-001",
        stage="translation",
        chapter="chapter_001",
    )
    defaults.update(kwargs)
    return PipelineEvent(**defaults)


# Verify PipelineEvent instances are immutable
def test_pipeline_event_is_frozen():
    ev = _make_event()
    with pytest.raises((AttributeError, TypeError)):
        ev.event = "CHANGED"  # type: ignore[misc]


# Verify PipelineEvent.to_dict generates valid JSON-serializable dictionary
def test_pipeline_event_to_dict_is_serializable():
    ev = _make_event(region_id="P001_R001", reason="empty_string")
    data = ev.to_dict()
    assert json.dumps(data)


# Verify PipelineEvent automatically populates an ISO timestamp
def test_pipeline_event_timestamp_present():
    ev = _make_event()
    assert ev.timestamp, "timestamp must be set automatically"


# Helper to construct test EventEmitter bound to temporary log file
def _make_emitter(tmp_path, **kwargs) -> EventEmitter:
    return EventEmitter(
        logger=logging.getLogger("test_emitter"),
        event_path=tmp_path / "events.jsonl",
        **kwargs,
    )


# Verify EventEmitter serializes events to JSON lines
def test_event_emitter_writes_valid_jsonl(tmp_path):
    emitter = _make_emitter(tmp_path)
    ev = _make_event(region_id="P001_R001")
    emitter.emit(ev)

    lines = (tmp_path / "events.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    parsed = json.loads(lines[0])
    assert parsed["event"] == TRANSLATION_VALUE_EMPTY
    assert parsed["region_id"] == "P001_R001"


# Verify EventEmitter redacts raw source text when redaction is enabled
def test_event_emitter_source_text_redacted_by_default(tmp_path):
    emitter = _make_emitter(tmp_path, include_source_text=False)
    ev = _make_event(original_text="季青临")
    emitter.emit(ev)

    lines = (tmp_path / "events.jsonl").read_text(encoding="utf-8").splitlines()
    parsed = json.loads(lines[0])
    assert parsed["original_text"] is None


# Verify EventEmitter includes raw source text when explicitly enabled
def test_event_emitter_source_text_included_when_enabled(tmp_path):
    emitter = _make_emitter(tmp_path, include_source_text=True)
    ev = _make_event(original_text="季青临")
    emitter.emit(ev)

    lines = (tmp_path / "events.jsonl").read_text(encoding="utf-8").splitlines()
    parsed = json.loads(lines[0])
    assert parsed["original_text"] == "季青临"


# Verify EventEmitter truncates long source text to max_text_chars limit
def test_event_emitter_truncates_long_source_text(tmp_path):
    emitter = _make_emitter(tmp_path, include_source_text=True, max_text_chars=5)
    ev = _make_event(original_text="abcdefghij")
    emitter.emit(ev)

    lines = (tmp_path / "events.jsonl").read_text(encoding="utf-8").splitlines()
    assert json.loads(lines[0])["original_text"] == "abcde"


# Verify EventEmitter outputs UTF-8 encoded text for CJK characters
def test_event_emitter_jsonl_is_utf8(tmp_path):
    emitter = _make_emitter(tmp_path, include_source_text=True)
    emitter.emit(_make_event(original_text="季青临"))
    raw = (tmp_path / "events.jsonl").read_bytes()
    raw.decode("utf-8")


# Verify EventEmitter appends consecutive events without overwriting
def test_event_emitter_appends_multiple_events(tmp_path):
    emitter = _make_emitter(tmp_path)
    emitter.emit(_make_event(event=TRANSLATION_VALUE_EMPTY))
    emitter.emit(_make_event(event=MANUAL_OVERRIDE_REQUIRED))
    lines = (tmp_path / "events.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["event"] == TRANSLATION_VALUE_EMPTY
    assert json.loads(lines[1])["event"] == MANUAL_OVERRIDE_REQUIRED


# Verify empty response string returns blank parse status
def test_blank_response_gives_blank_status():
    result = parse_json_result("")
    assert result.status == "blank"
    assert result.value == {}


# Verify whitespace-only response string returns blank parse status
def test_whitespace_response_gives_blank_status():
    result = parse_json_result("   \n  ")
    assert result.status == "blank"


# Verify blank parse status is distinguished from invalid JSON syntax
def test_blank_parse_is_distinct_from_invalid_json():
    blank = parse_json_result("")
    invalid = parse_json_result("not json at all {{")
    assert blank.status == "blank"
    assert invalid.status == "invalid_json"


# Verify invalid JSON responses capture error details in result object
def test_invalid_json_preserves_error_message():
    result = parse_json_result("{bad json}")
    assert result.status == "invalid_json"
    assert result.error is not None


# Verify JSON non-object structures (e.g. lists) return non_object status
def test_non_object_json_gives_non_object_status():
    result = parse_json_result("[1, 2, 3]")
    assert result.status == "non_object"
    assert result.value == {}


# Verify valid JSON key-value dictionary returns success status
def test_valid_json_object_gives_success():
    result = parse_json_result('{"P001_R001": "Hello"}')
    assert result.status == "success"
    assert result.value == {"P001_R001": "Hello"}


# Verify Markdown code-fenced JSON responses are parsed successfully
def test_json_inside_code_fence_is_parsed():
    fenced = '```json\n{"P001_R001": "Hello"}\n```'
    result = parse_json_result(fenced)
    assert result.status == "success"
    assert result.value["P001_R001"] == "Hello"


# Verify empty JSON object string parses as empty success dictionary
def test_empty_dict_gives_success():
    result = parse_json_result("{}")
    assert result.status == "success"
    assert result.value == {}


# Verify raw response length is recorded in parsed result metadata
def test_raw_length_is_recorded():
    text = '{"P001_R001": "Hi"}'
    result = parse_json_result(text)
    assert result.raw_length == len(text)


# Verify parse result dataclass instance is immutable
def test_blank_parse_result_is_frozen():
    result = parse_json_result("")
    with pytest.raises((AttributeError, TypeError)):
        result.status = "success"  # type: ignore[misc]


# Verify CLI stage execution events attach the active run_id consistently
def test_cli_stage_events_never_use_unknown_run_id(tmp_path):
    ws = (tmp_path / "workspace").resolve()
    ws.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (100, 100), "white").save(ws / "001.png")

    ctx = pipeline.create_run_context()
    config._run_id = ctx.run_id
    pipeline._run_all_from(
        str(ws), config, start="import", input_path=str(ws), run_id=ctx.run_id
    )

    event_file = ws / "logs" / "events.jsonl"
    assert event_file.exists()
    events = [
        json.loads(line)
        for line in event_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert events
    assert all(e["run_id"] == ctx.run_id for e in events)
