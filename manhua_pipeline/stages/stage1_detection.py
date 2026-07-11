"""Stage 1: Detection.

YOLO bubble detection. v0 = speech_bubble + narration only. Model: ogkalu/comic-speech-bubble-detector-yolov8m. Outputs detection.json.
"""

from pathlib import Path

from manhua_pipeline.logging_setup import get_logger, log_page, log_stage

logger = get_logger(__name__)

_STAGE_INDEX = 1
_TOTAL_STAGES = 7
_STAGE_NAME = "Detection"


def run_detection(workspace, config):
    """Run the Detection stage over all pages in the workspace."""
    ws = Path(workspace)
    log_stage(logger, _STAGE_INDEX, _TOTAL_STAGES, _STAGE_NAME, "starting")

    # TODO: load YOLO model (config.MODEL_DETECTION) via ultralytics.
    # TODO: for each region record bbox, reading_order, style_hint,
    #       read_region, erase_mask, render flag.
    # TODO: load manifest to get the ordered page list
    total_pages = 0  # TODO: replace with real page count from manifest
    warnings = 0

    for i in range(1, total_pages + 1):
        log_page(
            logger, _STAGE_INDEX, _TOTAL_STAGES, _STAGE_NAME, i, total_pages, "..."
        )
        # TODO: real per-page work here

    # TODO: write output JSON to the stage folder
    output_path = ws / "TODO_output.json"

    log_stage(
        logger,
        _STAGE_INDEX,
        _TOTAL_STAGES,
        _STAGE_NAME,
        f"done: {total_pages} pages, {warnings} warnings -> {output_path}",
    )
    return output_path
