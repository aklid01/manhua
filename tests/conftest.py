from pathlib import Path

import pytest


@pytest.fixture
def temp_workspace(tmp_path) -> Path:
    """Fixture that creates a temporary workspace directory mimicking the structure."""
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()

    # Create required staging folders if needed
    for stage_folder in [
        "pages",
        "stage1_detection",
        "stage2_ocr",
        "stage3_translation",
        "stage4_paraphrase",
        "stage5_render",
        "stage6_qa",
        "logs",
    ]:
        (workspace_dir / stage_folder).mkdir(parents=True, exist_ok=True)

    return workspace_dir
