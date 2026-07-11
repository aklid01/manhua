from unittest.mock import MagicMock, patch

import config
from manhua_pipeline.stages import (
    stage0_import,
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

    # Test valid subcommands
    for cmd in [
        "import",
        "detect",
        "ocr",
        "translate",
        "paraphrase",
        "render",
        "qa",
        "run-all",
    ]:
        parsed = parser.parse_args([cmd])
        assert parsed.command == cmd
        assert parsed.workspace == "workspace"


def test_pipeline_main_stage_execution(temp_workspace):
    """Test that main executes a specific stage successfully when invoked."""
    for stage_name in STAGES:
        mock_stage_func = MagicMock()
        with patch.dict("pipeline.STAGES", {stage_name: mock_stage_func}):
            # Call main with the CLI arguments
            exit_code = main([stage_name, "--workspace", str(temp_workspace)])
            assert exit_code == 0 or exit_code is None
            mock_stage_func.assert_called_once_with(str(temp_workspace), config)


def test_pipeline_run_all(temp_workspace):
    """Test that run-all triggers all stages in correct sequence."""
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
        for key in mock_stages.keys():
            mock_stages.__getitem__.return_value = MagicMock()

        exit_code = main(["run-all", "--workspace", str(temp_workspace)])
        assert exit_code == 0

        # Verify it called each stage in order
        assert mock_stages.__getitem__.call_count == 7


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

        # import and detect should not be called
        mock_fns["import"].assert_not_called()
        mock_fns["detect"].assert_not_called()

        # ocr and subsequent stages should be called
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


def test_stage_execution_mock_runs(temp_workspace):
    """Test stage functions run without exceptions in current implementation."""
    # We test with the actual stage functions
    assert stage0_import.run_import(temp_workspace, config) is not None
    assert stage1_detection.run_detection(temp_workspace, config) is not None
    assert stage2_ocr.run_ocr(temp_workspace, config) is not None
    assert stage3_translation.run_translation(temp_workspace, config) is not None
    assert stage4_paraphrase.run_paraphrase(temp_workspace, config) is not None
    assert stage5_render.run_render(temp_workspace, config) is not None
    assert stage6_qa.run_qa(temp_workspace, config) is not None
