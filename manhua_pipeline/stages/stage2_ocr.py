"""Stage 2: OCR.

PaddleOCR over detected regions. v0 = horizontal text only. Records original Chinese + confidence; flags needs_correction below threshold.
"""

from pathlib import Path

from manhua_pipeline.logging_setup import get_logger, log_page, log_stage

logger = get_logger(__name__)

_STAGE_INDEX = 2
_TOTAL_STAGES = 7
_STAGE_NAME = "OCR"


def run_ocr(workspace, config):
    """Run the OCR stage over all pages in the workspace."""
    ws = Path(workspace)
    log_stage(logger, _STAGE_INDEX, _TOTAL_STAGES, _STAGE_NAME, "starting")

    # TODO: run PaddleOCR on each read_region.
    # TODO: flag needs_correction when confidence < config.OCR_CONFIDENCE_THRESHOLD.
    # NOTE: vertical text deferred to a later version.
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
