# Unit tests for pipeline CLI parser, stage dispatching, and Stage 0 import.

import json
import zipfile
from unittest.mock import ANY, MagicMock, patch

import pytest
from PIL import Image, ImageDraw

import config
from manhua_pipeline.stages import (
    stage6_qa,
)
from pipeline import build_parser, main


# Fixture disabling subprocess execution for synchronous in-process test runs
@pytest.fixture(autouse=True)
def disable_batch_subprocess(monkeypatch):
    monkeypatch.setattr(config, "BATCH_SUBPROCESS", False)


# Verify CLI parser defines all required commands and flags
def test_build_parser():
    parser = build_parser()

    for cmd in ["detect", "ocr", "translate", "paraphrase", "render", "qa", "run-all"]:
        parsed = parser.parse_args([cmd])
        assert parsed.command == cmd
        assert parsed.workspace == "workspace"

    parsed = parser.parse_args(["import", "--input", "some/path"])
    assert parsed.command == "import"
    assert parsed.input == "some/path"


# Verify CLI execution dispatches each stage function with expected arguments
def test_pipeline_main_stage_execution(temp_workspace):
    (temp_workspace / config.MANIFEST_NAME).write_text(
        json.dumps({"chapter_id": "test_ch"})
    )
    from pipeline import STAGE_REGISTRY

    for stage_name in [k for k in STAGE_REGISTRY if k != "import"]:
        mock_fn = MagicMock()
        with patch("pipeline._load_stage", return_value=mock_fn) as mock_load:
            exit_code = main([stage_name, "--workspace", str(temp_workspace)])
            assert exit_code == 0 or exit_code is None
            mock_load.assert_called_once_with(stage_name)
            if stage_name in ["translate", "paraphrase"]:
                mock_fn.assert_called_once_with(
                    str(temp_workspace), config, allow_prompt=False
                )
            else:
                mock_fn.assert_called_once_with(str(temp_workspace), config)


# Verify 'import' command delegates to stage0_import.run_import
def test_pipeline_main_import_calls_run_import(temp_workspace):
    with patch("manhua_pipeline.stages.stage0_import.run_import") as mock_run:
        exit_code = main(
            [
                "import",
                "--input",
                "raw_cbz/0_001_.cbz",
                "--workspace",
                str(temp_workspace),
            ]
        )
        assert exit_code == 0
        mock_run.assert_called_once_with(
            "raw_cbz/0_001_.cbz",
            str(temp_workspace),
            config,
            title_romanized=None,
            title_english=None,
            source=None,
            fresh=False,
            skip_last=0,
        )


# Verify run-all --from-stage skips stages before the requested starting stage
def test_pipeline_run_all_from_stage(temp_workspace):
    mock_fns = {
        k: MagicMock()
        for k in ["import", "detect", "ocr", "translate", "paraphrase", "render", "qa"]
    }
    with patch("pipeline._load_stage", side_effect=lambda name: mock_fns[name]):
        exit_code = main(
            ["run-all", "--workspace", str(temp_workspace), "--from-stage", "ocr"]
        )
        assert exit_code == 0

        mock_fns["import"].assert_not_called()
        mock_fns["detect"].assert_not_called()
        mock_fns["ocr"].assert_called_once()
        mock_fns["translate"].assert_called_once()
        mock_fns["paraphrase"].assert_called_once()
        mock_fns["render"].assert_called_once()
        mock_fns["qa"].assert_called_once()


# Verify run-all fails with exit code 2 when given an invalid starting stage
def test_pipeline_run_all_invalid_stage(temp_workspace):
    exit_code = main(
        ["run-all", "--workspace", str(temp_workspace), "--from-stage", "invalid_stage"]
    )
    assert exit_code == 2


# Verify run-all --fresh starts at import even when manifest status is complete
def test_pipeline_run_all_fresh_bypasses_manifest_resume(temp_workspace, tmp_path):
    (temp_workspace / "manifest.json").write_text(
        json.dumps({"version": "v1", "current_stage": "complete"}), encoding="utf-8"
    )
    dummy_input = tmp_path / "dummy.cbz"
    dummy_input.write_text("dummy", encoding="utf-8")

    mock_fns = {
        k: MagicMock(return_value=None if k == "qa" else True)
        for k in ["import", "detect", "ocr", "translate", "paraphrase", "render", "qa"]
    }
    with patch("pipeline._load_stage", side_effect=lambda name: mock_fns[name]):
        exit_code = main(
            [
                "run-all",
                "--workspace",
                str(temp_workspace),
                "--input",
                str(dummy_input),
                "--fresh",
            ]
        )
        assert exit_code == 0
        mock_fns["import"].assert_called_once()
        assert mock_fns["import"].call_args.kwargs.get("fresh") is True


# Verify run-all --fresh ignores intermediate in-progress manifest state
def test_pipeline_run_all_fresh_bypasses_in_progress_manifest(temp_workspace, tmp_path):
    (temp_workspace / "manifest.json").write_text(
        json.dumps({"version": "v1", "current_stage": "translation"}), encoding="utf-8"
    )
    dummy_input = tmp_path / "dummy.cbz"
    dummy_input.write_text("dummy", encoding="utf-8")

    mock_fns = {
        k: MagicMock(return_value=None if k == "qa" else True)
        for k in ["import", "detect", "ocr", "translate", "paraphrase", "render", "qa"]
    }
    with patch("pipeline._load_stage", side_effect=lambda name: mock_fns[name]):
        exit_code = main(
            [
                "run-all",
                "--workspace",
                str(temp_workspace),
                "--input",
                str(dummy_input),
                "--fresh",
            ]
        )
        assert exit_code == 0
        mock_fns["import"].assert_called_once()


# Verify run-all from import without --input argument fails
def test_pipeline_run_all_missing_input(temp_workspace):
    exit_code = main(
        ["run-all", "--workspace", str(temp_workspace), "--from-stage", "import"]
    )
    assert exit_code == 2


# Verify importing image directory creates normalized page files and manifest
def test_import_folder(tmp_path):
    from manhua_pipeline.stages.stage0_import import run_import

    src = tmp_path / "src"
    src.mkdir()
    ws = tmp_path / "workspace"

    for i, name in enumerate(
        ["00000000_00010001.jpg", "00000000_00010002.jpg", "00000000_00010003.jpg"]
    ):
        img = Image.new("RGB", (860, 1214), color=(i * 80, 100, 200))
        img.save(src / name, "JPEG")

    run_import(str(src), str(ws), config)

    pages_dir = ws / "pages"
    assert (pages_dir / "001.jpg").exists()
    assert (pages_dir / "002.jpg").exists()
    assert (pages_dir / "003.jpg").exists()

    manifest_file = ws / "manifest.json"
    assert manifest_file.exists()
    manifest = json.loads(manifest_file.read_text())

    assert manifest["total_pages"] == 3
    assert manifest["input_format"] == "paginated"
    assert manifest["completed_stages"] == ["import"]
    assert manifest["current_stage"] == "detect"

    orig_names = [p["original_filename"] for p in manifest["pages"]]
    assert orig_names == [
        "00000000_00010001.jpg",
        "00000000_00010002.jpg",
        "00000000_00010003.jpg",
    ]


# Verify image directory import preserves mixed image extensions in manifest
def test_import_folder_preserves_mixed_extensions(tmp_path):
    from manhua_pipeline.stages.stage0_import import run_import

    src = tmp_path / "src"
    src.mkdir()
    ws = tmp_path / "workspace"
    Image.new("RGB", (100, 100), "white").save(src / "001.jpg")
    Image.new("RGB", (100, 100), "white").save(src / "002.png")
    Image.new("RGB", (100, 100), "white").save(src / "003.webp")

    run_import(str(src), str(ws), config)

    manifest = json.loads((ws / config.MANIFEST_NAME).read_text())
    assert [p["filename"] for p in manifest["pages"]] == [
        "001.jpg",
        "002.png",
        "003.webp",
    ]


# Verify tall strip images are split at safe seam regions during import
def test_import_slices_tall_page_at_safe_seams(tmp_path, monkeypatch):
    from manhua_pipeline.stages.stage0_import import run_import

    src = tmp_path / "src"
    src.mkdir()
    ws = tmp_path / "workspace"
    image = Image.new("RGB", (800, 10000), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((100, 1000, 700, 1800), fill="black")
    draw.rectangle((100, 4200, 700, 5000), fill="black")
    draw.rectangle((100, 7200, 700, 8000), fill="black")
    image.save(src / "strip.jpg", "JPEG")

    monkeypatch.setattr(config, "IMPORT_MAX_WIDTH", 0)
    monkeypatch.setattr(config, "IMPORT_SLICE_TARGET_HEIGHT", 3000)
    monkeypatch.setattr(config, "IMPORT_SLICE_MIN_HEIGHT", 4000)
    monkeypatch.setattr(config, "IMPORT_SLICE_SEAM_TOLERANCE", 0.10)
    monkeypatch.setattr(config, "IMPORT_SLICE_SEAM_BAND", 10)

    run_import(str(src), str(ws), config)

    manifest = json.loads((ws / config.MANIFEST_NAME).read_text())
    assert manifest["total_pages"] == 3
    assert [p["filename"] for p in manifest["pages"]] == [
        "001.jpg",
        "002.jpg",
        "003.jpg",
    ]
    assert all(p["slice_count"] == 3 for p in manifest["pages"])
    assert [p["slice_index"] for p in manifest["pages"]] == [1, 2, 3]
    assert manifest["warning_count"] == 0


# Verify tall images without safe horizontal seams are kept intact with warning
def test_import_keeps_tall_page_when_no_safe_seam(tmp_path, monkeypatch):
    from manhua_pipeline.stages.stage0_import import run_import

    src = tmp_path / "src"
    src.mkdir()
    ws = tmp_path / "workspace"
    Image.new("RGB", (800, 10000), "black").save(src / "solid.jpg", "JPEG")

    monkeypatch.setattr(config, "IMPORT_MAX_WIDTH", 0)
    monkeypatch.setattr(config, "IMPORT_SLICE_TARGET_HEIGHT", 3000)
    monkeypatch.setattr(config, "IMPORT_SLICE_MIN_HEIGHT", 4000)

    run_import(str(src), str(ws), config)

    manifest = json.loads((ws / config.MANIFEST_NAME).read_text())
    assert manifest["total_pages"] == 1
    assert manifest["pages"][0]["slice_count"] == 1
    assert manifest["warning_count"] == 1


# Verify importing CBZ archive unpacks and preserves true page order
def test_import_cbz(tmp_path):
    from manhua_pipeline.stages.stage0_import import run_import

    cbz = tmp_path / "1_001_.cbz"
    with zipfile.ZipFile(cbz, "w") as zf:
        for name in [
            "00000000_00010002.jpg",
            "00000000_00010000.jpg",
            "00000000_00010001.jpg",
        ]:
            img = Image.new("RGB", (860, 1214), color=(50, 100, 150))
            img_path = tmp_path / name
            img.save(img_path, "JPEG")
            zf.write(img_path, name)

    ws = tmp_path / "workspace"
    run_import(str(cbz), str(ws), config)

    pages_dir = ws / "pages"
    assert (pages_dir / "001.jpg").exists()
    assert (pages_dir / "002.jpg").exists()
    assert (pages_dir / "003.jpg").exists()

    manifest = json.loads((ws / "manifest.json").read_text())
    assert manifest["total_pages"] == 3
    orig_names = [p["original_filename"] for p in manifest["pages"]]
    assert orig_names == [
        "00000000_00010000.jpg",
        "00000000_00010001.jpg",
        "00000000_00010002.jpg",
    ]


# Verify re-running import on the same directory produces valid manifest without errors
def test_import_idempotent(tmp_path):
    from manhua_pipeline.stages.stage0_import import run_import

    src = tmp_path / "src"
    src.mkdir()
    ws = tmp_path / "workspace"
    img = Image.new("RGB", (860, 1214))
    img.save(src / "00000000_00010001.jpg", "JPEG")

    run_import(str(src), str(ws), config)
    run_import(str(src), str(ws), config)
    assert (ws / "manifest.json").exists()


# Verify importing empty folder raises ValueError
def test_import_empty_folder(tmp_path):
    from manhua_pipeline.stages.stage0_import import run_import

    src = tmp_path / "empty"
    src.mkdir()
    ws = tmp_path / "workspace"

    with pytest.raises(ValueError, match="No supported images"):
        run_import(str(src), str(ws), config)


# Verify importing non-existent input path raises FileNotFoundError
def test_import_missing_input(tmp_path):
    from manhua_pipeline.stages.stage0_import import run_import

    with pytest.raises(FileNotFoundError):
        run_import(str(tmp_path / "nope.cbz"), str(tmp_path / "ws"), config)


# Verify QA stage executes on empty workspace without crashing
def test_stage_stubs_run(temp_workspace):
    manifest = {
        "chapter_id": "test",
        "total_pages": 0,
        "pages": [],
        "completed_stages": [
            "import",
            "detect",
            "ocr",
            "translate",
            "paraphrase",
            "render",
        ],
        "current_stage": "render",
    }
    (temp_workspace / "manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    assert stage6_qa.run_qa(temp_workspace, config) is not None


# Verify unpadded filenames (p1..p10) sort in numerical rather than lexicographical order
def test_import_unpadded_numeric_order(tmp_path):
    from manhua_pipeline.stages.stage0_import import run_import

    src = tmp_path / "chapter_042"
    src.mkdir()
    ws = tmp_path / "workspace"
    names = [f"p{i}.jpg" for i in range(1, 11)]
    for name in names:
        Image.new("RGB", (860, 1214)).save(src / name, "JPEG")
    run_import(str(src), str(ws), config)
    manifest = json.loads((ws / "manifest.json").read_text())
    orig = [p["original_filename"] for p in manifest["pages"]]
    assert orig == [f"p{i}.jpg" for i in range(1, 11)]
    assert manifest["pages"][9]["page_number"] == 10


# Verify chapter_id is derived from immediate chapter directory name
def test_import_chapter_id_from_folder(tmp_path):
    from manhua_pipeline.stages.stage0_import import run_import

    src = tmp_path / "chapters" / "chapter_001"
    src.mkdir(parents=True)
    ws = tmp_path / "workspace"
    Image.new("RGB", (860, 1214)).save(src / "00000000_00010000.jpg", "JPEG")
    run_import(str(src), str(ws), config)
    manifest = json.loads((ws / "manifest.json").read_text())
    assert "chapter_001" in manifest["chapter_id"]
    assert manifest["chapter_id"] != "chapters"


# Verify CBZ import derives chapter_id from archive filename
def test_import_cbz_chapter_id(tmp_path):
    from manhua_pipeline.stages.stage0_import import run_import

    cbz = tmp_path / "1_001_.cbz"
    with zipfile.ZipFile(cbz, "w") as zf:
        for name in ["00000000_00010000.jpg", "00000000_00010001.jpg"]:
            img_path = tmp_path / name
            Image.new("RGB", (860, 1214)).save(img_path, "JPEG")
            zf.write(img_path, name)
    ws = tmp_path / "workspace"
    run_import(str(cbz), str(ws), config)
    manifest = json.loads((ws / "manifest.json").read_text())
    assert "1_001" in manifest["chapter_id"]
    assert "tmp" not in manifest["chapter_id"].lower()


# Verify re-importing fewer pages deletes old stale page files from disk
def test_import_idempotent_clears_stale(tmp_path):
    from manhua_pipeline.stages.stage0_import import run_import

    src = tmp_path / "src"
    src.mkdir()
    ws = tmp_path / "workspace"
    for i in range(1, 6):
        Image.new("RGB", (860, 1214)).save(src / f"{i:08d}.jpg", "JPEG")
    run_import(str(src), str(ws), config)
    assert (ws / "pages" / "005.jpg").exists()

    for f in src.iterdir():
        f.unlink()
    for i in range(1, 3):
        Image.new("RGB", (860, 1214)).save(src / f"{i:08d}.jpg", "JPEG")
    run_import(str(src), str(ws), config)
    pages = sorted((ws / "pages").glob("*.jpg"))
    assert [p.name for p in pages] == ["001.jpg", "002.jpg"]
    manifest = json.loads((ws / "manifest.json").read_text())
    assert manifest["total_pages"] == 2


# Verify optional metadata arguments (titles, source) are populated in manifest
def test_import_metadata_fields(tmp_path):
    from manhua_pipeline.stages.stage0_import import run_import

    src = tmp_path / "chapter_007"
    src.mkdir()
    ws = tmp_path / "workspace"
    Image.new("RGB", (860, 1214)).save(src / "00000000_00010000.jpg", "JPEG")
    run_import(
        str(src),
        str(ws),
        config,
        title_romanized="Wo Meizhou",
        title_english="Random Job",
        source="baozimh",
    )
    m = json.loads((ws / "manifest.json").read_text())
    assert m["title_english"] == "Random Job"
    assert m["title_romanized"] == "Wo Meizhou"
    assert m["source"] == "baozimh"


# Verify corrupt image files are recorded as skipped pages with error reasons
def test_import_failed_page_recorded(tmp_path):
    from manhua_pipeline.stages.stage0_import import run_import

    src = tmp_path / "src"
    src.mkdir()
    ws = tmp_path / "workspace"
    Image.new("RGB", (860, 1214)).save(src / "00000000_00010000.jpg", "JPEG")
    (src / "00000000_00010001.jpg").write_bytes(b"not a real image")
    run_import(str(src), str(ws), config)
    m = json.loads((ws / "manifest.json").read_text())
    assert m["total_pages"] == 2
    bad = [p for p in m["pages"] if p["skip"]]
    assert len(bad) == 1
    assert bad[0]["filename"] is None
    assert bad[0]["skip_reason"] == "read_error"
    assert m["warning_count"] >= 1


# Verify chapter_id strips unwanted trailing underscores
def test_import_chapter_id_no_trailing_underscore(tmp_path):
    from manhua_pipeline.stages.stage0_import import run_import

    cbz = tmp_path / "1_001_.cbz"
    with zipfile.ZipFile(cbz, "w") as zf:
        img_path = tmp_path / "00000000_00010000.jpg"
        Image.new("RGB", (860, 1214)).save(img_path, "JPEG")
        zf.write(img_path, "00000000_00010000.jpg")
    ws = tmp_path / "workspace"
    run_import(str(cbz), str(ws), config)
    m = json.loads((ws / "manifest.json").read_text())
    assert not m["chapter_id"].endswith("_")
    assert "1_001" in m["chapter_id"]


# Verify run-all executes sequential stage subprocesses in batch mode
def test_pipeline_run_all_subprocess(temp_workspace, monkeypatch):
    monkeypatch.setattr(config, "BATCH_SUBPROCESS", True)
    monkeypatch.setattr(config, "SUBPROCESS_STAGES", None)

    manifest = {
        "chapter_id": "test_ch",
        "current_stage": "ocr",
        "completed_stages": ["import", "detect"],
    }
    (temp_workspace / "manifest.json").write_text(json.dumps(manifest))

    with (
        patch("pipeline._run_stage_subprocess", return_value=0) as mock_run_sub,
        patch("manhua_pipeline.io.workspace.load_manifest") as mock_load_manifest,
    ):
        manifest_ocr = {"current_stage": "translate"}
        manifest_translate = {"current_stage": "paraphrase"}
        manifest_paraphrase = {"current_stage": "render"}
        manifest_render = {"current_stage": "qa"}
        manifest_qa = {"current_stage": "complete"}

        mock_load_manifest.side_effect = [
            manifest,
            manifest_ocr,
            manifest_translate,
            manifest_paraphrase,
            manifest_render,
            manifest_qa,
        ]

        exit_code = main(
            ["run-all", "--workspace", str(temp_workspace), "--from-stage", "ocr"]
        )
        assert exit_code == 0

        mock_run_sub.assert_any_call(
            "ocr",
            str(temp_workspace),
            input_path=None,
            meta={"title_romanized": None, "title_en": None, "source": None},
            fresh=False,
            skip_last=0,
            run_id=ANY,
        )
        mock_run_sub.assert_any_call(
            "translate",
            str(temp_workspace),
            input_path=None,
            meta={"title_romanized": None, "title_en": None, "source": None},
            fresh=False,
            skip_last=0,
            run_id=ANY,
        )
        mock_run_sub.assert_any_call(
            "paraphrase",
            str(temp_workspace),
            input_path=None,
            meta={"title_romanized": None, "title_en": None, "source": None},
            fresh=False,
            skip_last=0,
            run_id=ANY,
        )
        mock_run_sub.assert_any_call(
            "render",
            str(temp_workspace),
            input_path=None,
            meta={"title_romanized": None, "title_en": None, "source": None},
            fresh=False,
            skip_last=0,
            run_id=ANY,
        )
        mock_run_sub.assert_any_call(
            "qa",
            str(temp_workspace),
            input_path=None,
            meta={"title_romanized": None, "title_en": None, "source": None},
            fresh=False,
            skip_last=0,
            run_id=ANY,
        )


# Verify subprocess run-all pauses when a stage requires manual handoff
def test_pipeline_run_all_subprocess_handoff(temp_workspace, monkeypatch):
    monkeypatch.setattr(config, "BATCH_SUBPROCESS", True)
    monkeypatch.setattr(config, "SUBPROCESS_STAGES", None)

    manifest = {
        "chapter_id": "test_ch",
        "current_stage": "translate",
        "completed_stages": ["import", "detect", "ocr"],
    }
    (temp_workspace / "manifest.json").write_text(json.dumps(manifest))

    with (
        patch("pipeline._run_stage_subprocess", return_value=0) as mock_run_sub,
        patch("manhua_pipeline.io.workspace.load_manifest") as mock_load_manifest,
    ):
        mock_load_manifest.side_effect = [
            manifest,
            manifest,
        ]

        exit_code = main(
            ["run-all", "--workspace", str(temp_workspace), "--from-stage", "translate"]
        )
        assert exit_code == 0

        mock_run_sub.assert_called_once_with(
            "translate",
            str(temp_workspace),
            input_path=None,
            meta={"title_romanized": None, "title_en": None, "source": None},
            fresh=False,
            skip_last=0,
            run_id=ANY,
        )


# Verify _load_stage lazily loads stage entrypoint functions
def test_load_stage_lazy():
    from manhua_pipeline.stages import stage1_detection, stage2_ocr
    from pipeline import _load_stage

    assert _load_stage("ocr") == stage2_ocr.run_ocr
    assert _load_stage("detect") == stage1_detection.run_detection
