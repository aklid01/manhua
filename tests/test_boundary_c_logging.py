"""Boundary C tests: All-stage logging revamp, lifecycle events, domain events, JSONL event persistence.

All tests are offline and operate on temporary workspaces.
"""

import json
from pathlib import Path

import pytest

import config
from manhua_pipeline.events import (
    CHAPTER_COMPLETED,
    CHAPTER_STARTED,
    QA_ISSUE_DETECTED,
    RENDER_REGION_COMPLETED,
    STAGE_COMPLETED,
    STAGE_STARTED,
)
from manhua_pipeline.io.event_emitter import EventEmitter, get_event_emitter
from manhua_pipeline.stages.stage0_import import run_import
from manhua_pipeline.stages.stage6_qa import run_qa
from manhua_pipeline.stages.stage7_package import run_package


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _setup_stage_ws(tmp_path, stage_name="import"):
    ws = tmp_path / "workspace"
    ws.mkdir(parents=True, exist_ok=True)
    manifest = {
        "chapter_id": "test_ch",
        "total_pages": 1,
        "pages": [{"page_number": 1, "filename": "001.png", "original_filename": "001.png", "skip": False, "width": 100, "height": 100}],
        "current_stage": stage_name,
        "completed_stages": [],
        "warning_count": 0,
        "status": "in_progress",
    }
    (ws / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (ws / "logs").mkdir(parents=True, exist_ok=True)
    return ws


def _read_events(ws: Path) -> list[dict]:
    event_file = ws / "stage_folders_logs" if (ws / "stage_folders_logs").exists() else ws / "logs" / "events.jsonl"
    if not event_file.exists():
        # Fall back to checking logs folder
        event_file = ws / "logs" / "events.jsonl"
    if not event_file.exists():
        return []
    lines = event_file.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


# ---------------------------------------------------------------------------
# Stage Lifecycle Events Tests
# ---------------------------------------------------------------------------


def test_stage0_import_emits_lifecycle_events(tmp_path):
    # Create dummy images folder
    src_dir = tmp_path / "raw_pages"
    src_dir.mkdir()
    from PIL import Image

    im = Image.new("RGB", (100, 100), (255, 255, 255))
    im.save(src_dir / "001.jpg")

    ws = tmp_path / "workspace"
    run_import(str(src_dir), str(ws), config)

    events = _read_events(ws)
    event_names = [e["event"] for e in events]
    assert STAGE_STARTED in event_names
    assert STAGE_COMPLETED in event_names

    start_event = next(e for e in events if e["event"] == STAGE_STARTED)
    assert start_event["stage"] == "import"

    complete_event = next(e for e in events if e["event"] == STAGE_COMPLETED)
    assert complete_event["stage"] == "import"
    assert complete_event["metadata"]["total_pages"] == 1


def test_stage5_render_emits_region_and_lifecycle_events(tmp_path):
    ws = _setup_stage_ws(tmp_path, "render")
    (ws / "pages").mkdir(parents=True, exist_ok=True)
    from PIL import Image

    im = Image.new("RGB", (100, 100), (255, 255, 255))
    im.save(ws / "pages" / "001.png")

    det = {
        "chapter_id": "test_ch",
        "stage": "detection",
        "generated_at": "now",
        "model": "dummy",
        "regions": [
            {
                "region_id": "P001_R001",
                "page_number": 1,
                "type": "speech_bubble",
                "bbox": {"x": 10, "y": 10, "w": 50, "h": 20},
                "reading_order": 1,
                "render": True,
                "erase_mask": {"type": "rect", "coords": [10, 10, 50, 20]},
            }
        ],
    }
    (ws / "stage1_detection").mkdir(parents=True, exist_ok=True)
    (ws / "stage1_detection" / "detection.json").write_text(json.dumps(det), encoding="utf-8")

    ocr = {
        "chapter_id": "test_ch",
        "stage": "ocr",
        "results": [
            {
                "region_id": "P001_R001",
                "page_number": 1,
                "has_usable_text": True,
            }
        ],
    }
    (ws / "stage2_ocr").mkdir(parents=True, exist_ok=True)
    (ws / "stage2_ocr" / "ocr.json").write_text(json.dumps(ocr), encoding="utf-8")

    para = {
        "chapter_id": "test_ch",
        "stage": "paraphrase",
        "results": [
            {
                "region_id": "P001_R001",
                "page_number": 1,
                "final_text": "Hello world!",
            }
        ],
    }
    (ws / "stage4_paraphrase").mkdir(parents=True, exist_ok=True)
    (ws / "stage4_paraphrase" / "paraphrase.json").write_text(json.dumps(para), encoding="utf-8")

    from manhua_pipeline.stages.stage5_render import run_render

    run_render(str(ws), config)

    events = _read_events(ws)
    event_names = [e["event"] for e in events]
    assert STAGE_STARTED in event_names
    assert RENDER_REGION_COMPLETED in event_names
    assert STAGE_COMPLETED in event_names

    region_ev = next(e for e in events if e["event"] == RENDER_REGION_COMPLETED)
    assert region_ev["region_id"] == "P001_R001"


def test_stage6_qa_emits_issue_and_lifecycle_events(tmp_path):
    ws = _setup_stage_ws(tmp_path, "qa")
    ocr = {"chapter_id": "test_ch", "results": [{"region_id": "P001_R001", "page_number": 1, "ocr_confidence": 0.1, "has_usable_text": True, "needs_correction": True}]}
    tr = {"chapter_id": "test_ch", "results": [{"region_id": "P001_R001", "page_number": 1, "literal_translation": "hi", "translated": True}]}
    para = {"chapter_id": "test_ch", "results": [{"region_id": "P001_R001", "page_number": 1, "final_text": "hi", "paraphrased": True}]}
    render = {"chapter_id": "test_ch", "results": [{"region_id": "P001_R001", "page_number": 1, "rendered": True}], "pages": [{"page_number": 1}]}

    (ws / "stage2_ocr").mkdir(parents=True, exist_ok=True)
    (ws / "stage2_ocr" / "ocr.json").write_text(json.dumps(ocr), encoding="utf-8")
    (ws / "stage3_translation").mkdir(parents=True, exist_ok=True)
    (ws / "stage3_translation" / "translation.json").write_text(json.dumps(tr), encoding="utf-8")
    (ws / "stage4_paraphrase").mkdir(parents=True, exist_ok=True)
    (ws / "stage4_paraphrase" / "paraphrase.json").write_text(json.dumps(para), encoding="utf-8")
    (ws / "stage5_render").mkdir(parents=True, exist_ok=True)
    (ws / "stage5_render" / "render.json").write_text(json.dumps(render), encoding="utf-8")

    run_qa(str(ws), config)

    events = _read_events(ws)
    event_names = [e["event"] for e in events]
    assert STAGE_STARTED in event_names
    assert QA_ISSUE_DETECTED in event_names
    assert STAGE_COMPLETED in event_names

    qa_ev = next(e for e in events if e["event"] == QA_ISSUE_DETECTED)
    assert qa_ev["region_id"] == "P001_R001"
    assert qa_ev["reason"] in ("low_ocr_confidence", "missing_paraphrase")


def test_stage7_package_emits_lifecycle_events(tmp_path):
    ws = _setup_stage_ws(tmp_path, "package")
    (ws / "stage5_render").mkdir(parents=True, exist_ok=True)
    (ws / "stage5_render" / "rendered").mkdir(parents=True, exist_ok=True)
    from PIL import Image

    im = Image.new("RGB", (10, 10), (0, 0, 0))
    im.save(ws / "stage5_render" / "rendered" / "001.png")

    run_package(str(ws), config, formats=["zip"])

    events = _read_events(ws)
    event_names = [e["event"] for e in events]
    assert STAGE_STARTED in event_names
    assert STAGE_COMPLETED in event_names


def test_batch_emits_chapter_lifecycle_events(tmp_path):
    # Setup batch input folder with one dummy chapter zip
    in_dir = tmp_path / "inputs"
    in_dir.mkdir()

    import zipfile
    from PIL import Image

    img_path = tmp_path / "001.jpg"
    Image.new("RGB", (10, 10)).save(img_path)

    cbz_path = in_dir / "ch01.cbz"
    with zipfile.ZipFile(cbz_path, "w") as zf:
        zf.write(img_path, arcname="001.jpg")

    series_dir = tmp_path / "series"
    series_dir.mkdir()

    # Run batch import only or mock run_all
    from pipeline import run_batch

    # Patch _run_all_from to quickly complete chapter
    import pipeline

    def mock_run_all(*args, **kwargs):
        ws = Path(args[0])
        m = {
            "chapter_id": ws.name,
            "current_stage": "complete",
            "completed_stages": ["import", "detect", "ocr", "translate", "paraphrase", "render", "qa"],
        }
        (ws / "manifest.json").write_text(json.dumps(m), encoding="utf-8")
        return 0

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(pipeline, "_run_all_from", mock_run_all)

    try:
        run_batch(str(in_dir), series_dir, config)
    finally:
        monkeypatch.undo()

    ch_ws = series_dir / "ch01"
    events = _read_events(ch_ws)
    event_names = [e["event"] for e in events]
    assert CHAPTER_STARTED in event_names
    assert CHAPTER_COMPLETED in event_names


def test_jsonl_event_sink_valid_schema(tmp_path):
    ws = _setup_stage_ws(tmp_path)
    emitter = get_event_emitter(ws, config)
    from manhua_pipeline.events import PipelineEvent

    ev = PipelineEvent(
        event="CUSTOM_TEST_EVENT",
        run_id="run-123",
        stage="test",
        chapter="ch01",
        region_id="P001_R001",
        page=1,
        reason="test_reason",
    )
    emitter.emit(ev)

    events = _read_events(ws)
    assert len(events) == 1
    e = events[0]
    assert e["event"] == "CUSTOM_TEST_EVENT"
    assert e["run_id"] == "run-123"
    assert e["stage"] == "test"
    assert e["chapter"] == "ch01"
    assert e["region_id"] == "P001_R001"
    assert e["page"] == 1
    assert e["reason"] == "test_reason"
    assert "timestamp" in e
