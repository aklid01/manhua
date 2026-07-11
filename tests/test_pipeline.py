import json
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

import config
from manhua_pipeline.stages import (
    stage1_detection,
    stage2_ocr,
    stage3_translation,
    stage4_paraphrase,
    stage5_render,
    stage6_qa,
)
from pipeline import STAGES, build_parser, main


def test_build_parser():
    """Test that build_parser constructs the CLI parser with all expected commands."""
    parser = build_parser()

    for cmd in ["detect", "ocr", "translate", "paraphrase", "render", "qa", "run-all"]:
        parsed = parser.parse_args([cmd])
        assert parsed.command == cmd
        assert parsed.workspace == "workspace"

    parsed = parser.parse_args(["import", "--input", "some/path"])
    assert parsed.command == "import"
    assert parsed.input == "some/path"


def test_pipeline_main_stage_execution(temp_workspace):
    """Test that non-import stages are called correctly from main."""
    non_import = {k: v for k, v in STAGES.items() if k != "import"}
    for stage_name in non_import:
        mock_fn = MagicMock()
        with patch.dict("pipeline.STAGES", {stage_name: mock_fn}):
            exit_code = main([stage_name, "--workspace", str(temp_workspace)])
            assert exit_code == 0 or exit_code is None
            mock_fn.assert_called_once_with(str(temp_workspace), config)


def test_pipeline_main_import_calls_run_import(temp_workspace):
    """Test that 'import' subcommand calls stage0_import.run_import with input_path."""
    with patch("pipeline.stage0_import.run_import") as mock_run:
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
            "raw_cbz/0_001_.cbz", str(temp_workspace), config
        )


def test_pipeline_run_all_from_stage(temp_workspace):
    """Test that run-all starting from a specific stage skips preceding stages."""
    with patch("pipeline.STAGES") as mock_stages:
        mock_stages.keys.return_value = [
            "import",
            "detect",
            "ocr",
            "translate",
            "paraphrase",
            "render",
            "qa",
        ]
        mock_fns = {k: MagicMock() for k in mock_stages.keys()}
        mock_stages.__getitem__.side_effect = lambda k: mock_fns[k]

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


def test_pipeline_run_all_invalid_stage(temp_workspace):
    """Test that run-all logs error and fails when given an invalid start stage."""
    exit_code = main(
        ["run-all", "--workspace", str(temp_workspace), "--from-stage", "invalid_stage"]
    )
    assert exit_code == 2


def test_pipeline_run_all_missing_input(temp_workspace):
    """Test that run-all from import without --input returns exit code 2."""
    exit_code = main(
        ["run-all", "--workspace", str(temp_workspace), "--from-stage", "import"]
    )
    assert exit_code == 2


def test_import_folder(tmp_path):
    """Stage0: import from a folder of synthetic images produces correct pages + manifest."""
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
    assert (pages_dir / "001.png").exists()
    assert (pages_dir / "002.png").exists()
    assert (pages_dir / "003.png").exists()

    manifest_file = ws / "manifest.json"
    assert manifest_file.exists()
    manifest = json.loads(manifest_file.read_text())

    assert manifest["total_pages"] == 3
    assert manifest["input_format"] == "paginated"
    assert manifest["completed_stages"] == ["import"]
    assert manifest["current_stage"] == "detection"

    orig_names = [p["original_filename"] for p in manifest["pages"]]
    assert orig_names == [
        "00000000_00010001.jpg",
        "00000000_00010002.jpg",
        "00000000_00010003.jpg",
    ]


def test_import_cbz(tmp_path):
    """Stage0: import from a CBZ file unpacks and produces correct pages."""
    import zipfile

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
    assert (pages_dir / "001.png").exists()
    assert (pages_dir / "002.png").exists()
    assert (pages_dir / "003.png").exists()

    manifest = json.loads((ws / "manifest.json").read_text())
    assert manifest["total_pages"] == 3
    orig_names = [p["original_filename"] for p in manifest["pages"]]
    assert orig_names == [
        "00000000_00010000.jpg",
        "00000000_00010001.jpg",
        "00000000_00010002.jpg",
    ]


def test_import_idempotent(tmp_path):
    """Stage0: re-running import overwrites pages deterministically."""
    from manhua_pipeline.stages.stage0_import import run_import

    src = tmp_path / "src"
    src.mkdir()
    ws = tmp_path / "workspace"

    for name in ["00000000_00010001.jpg"]:
        img = Image.new("RGB", (860, 1214))
        img.save(src / name, "JPEG")

    run_import(str(src), str(ws), config)

    run_import(str(src), str(ws), config)
    assert (ws / "manifest.json").exists()


def test_import_empty_folder(tmp_path):
    """Stage0: empty input folder raises ValueError."""
    from manhua_pipeline.stages.stage0_import import run_import

    src = tmp_path / "empty"
    src.mkdir()
    ws = tmp_path / "workspace"

    with pytest.raises(ValueError, match="No supported images"):
        run_import(str(src), str(ws), config)


def test_import_missing_input(tmp_path):
    """Stage0: non-existent input raises FileNotFoundError."""
    from manhua_pipeline.stages.stage0_import import run_import

    with pytest.raises(FileNotFoundError):
        run_import(str(tmp_path / "nope.cbz"), str(tmp_path / "ws"), config)


def test_stage_stubs_run(temp_workspace):
    """Remaining stage stubs run without raising exceptions."""
    assert stage1_detection.run_detection(temp_workspace, config) is not None
    assert stage2_ocr.run_ocr(temp_workspace, config) is not None
    assert stage3_translation.run_translation(temp_workspace, config) is not None
    assert stage4_paraphrase.run_paraphrase(temp_workspace, config) is not None
    assert stage5_render.run_render(temp_workspace, config) is not None
    assert stage6_qa.run_qa(temp_workspace, config) is not None
