import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# Mock heavy ML / framework dependencies for offline test runs
sys.modules["ultralytics"] = MagicMock()
sys.modules["paddleocr"] = MagicMock()


class MockFastMCP:
    # Minimal FastMCP decorator mock for testing MCP endpoints

    def __init__(self, *args, **kwargs):
        pass

    def tool(self, *args, **kwargs):
        return lambda func: func

    def resource(self, *args, **kwargs):
        return lambda func: func

    def prompt(self, *args, **kwargs):
        return lambda func: func


mock_fastmcp = MagicMock()
mock_fastmcp.FastMCP = MockFastMCP
sys.modules["fastmcp"] = mock_fastmcp


# Create a temporary workspace directory mimicking the pipeline staging structure
@pytest.fixture
def temp_workspace(tmp_path) -> Path:
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()

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
