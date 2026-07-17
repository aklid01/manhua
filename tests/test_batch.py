"""Offline unit tests for the batch processing CLI command in pipeline.py."""

import json
from pathlib import Path
import pytest
import pipeline
import config


def _create_dummy_cbz(folder: Path, name: str):
    p = folder / name
    p.write_text("dummy cbz contents", encoding="utf-8")
    return p


def _setup_chapter(base_dir: Path, chapter: str, stage: str):
    ch_dir = base_dir / chapter
    ch_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "chapter_id": chapter,
        "total_pages": 1,
        "pages": [{"page_number": 1, "filename": "001.png", "skip": False}],
        "current_stage": stage,
        "completed_stages": ["import", "detect", "ocr", "translate"] if stage != "import" else [],
        "warning_count": 0,
        "status": "in_progress" if stage != "complete" else "success",
    }
    (ch_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_iter_batch_inputs(tmp_path):
    _create_dummy_cbz(tmp_path, "02_chap.cbz")
    _create_dummy_cbz(tmp_path, "01_chap.zip")
    _create_dummy_cbz(tmp_path, "03_chap.txt")
    (tmp_path / "subdir").mkdir()

    inputs = pipeline._iter_batch_inputs(tmp_path)
    assert len(inputs) == 2
    assert inputs[0].name == "01_chap.zip"
    assert inputs[1].name == "02_chap.cbz"


def test_run_batch_classifications(tmp_path, monkeypatch):
    input_folder = tmp_path / "inputs"
    input_folder.mkdir()
    _create_dummy_cbz(input_folder, "c1.cbz")
    _create_dummy_cbz(input_folder, "c2.cbz")
    _create_dummy_cbz(input_folder, "c3.cbz")
    _create_dummy_cbz(input_folder, "c4.cbz")

    base_dir = tmp_path / "workspace"
    base_dir.mkdir()

    # c1 is already complete -> should skip
    _setup_chapter(base_dir, "c1", "complete")

    # c2 exists but is pending @paraphrase -> should resume
    _setup_chapter(base_dir, "c2", "paraphrase")

    # c3 is new
    # c4 will crash

    runs = {}

    def mock_run_all(workspace, cfg, start="import", input_path=None, meta=None, fresh=False):
        ch = Path(workspace).name
        runs[ch] = {
            "start": start,
            "input_path": input_path,
            "fresh": fresh
        }
        if ch == "c4":
            raise RuntimeError("C4 blew up")
        if ch == "c3":
            _setup_chapter(base_dir, ch, "complete")
        return 0

    monkeypatch.setattr(pipeline, "_run_all_from", mock_run_all)
    monkeypatch.setattr(pipeline, "_clear_console_after", lambda d: None)

    rc = pipeline.run_batch(
        input_folder=str(input_folder),
        base_dir=base_dir,
        config=config,
        fresh=False,
        resume=True,
        clear_delay=0
    )

    assert rc == 0
    assert "c1" not in runs
    assert runs["c2"]["start"] == "import"
    assert runs["c2"]["fresh"] is False
    assert runs["c3"]["start"] == "import"
    assert runs["c3"]["fresh"] is False
    assert "c4" in runs

    log_files = list((base_dir / "logs").glob("batch_*.log"))
    assert len(log_files) == 1
    log_content = log_files[0].read_text(encoding="utf-8")
    assert "SKIP complete   | c1" in log_content
    assert "PENDING @paraphrase | c2" in log_content
    assert "DONE            | c3" in log_content
    assert "FAIL RuntimeError | c4" in log_content
    assert "done=1 pending=1 skipped=1 failed=1" in log_content

    err_files = list((base_dir / "logs").glob("error_*.log"))
    assert len(err_files) == 1
    assert "Error: C4 blew up" in err_files[0].read_text(encoding="utf-8")


def test_run_batch_no_resume(tmp_path, monkeypatch):
    input_folder = tmp_path / "inputs"
    input_folder.mkdir()
    _create_dummy_cbz(input_folder, "c1.cbz")

    base_dir = tmp_path / "workspace"
    base_dir.mkdir()

    _setup_chapter(base_dir, "c1", "paraphrase")

    runs = {}

    def mock_run_all(workspace, cfg, **kwargs):
        runs[Path(workspace).name] = True
        return 0

    monkeypatch.setattr(pipeline, "_run_all_from", mock_run_all)

    rc = pipeline.run_batch(
        input_folder=str(input_folder),
        base_dir=base_dir,
        config=config,
        resume=False
    )
    assert rc == 0
    assert "c1" not in runs
