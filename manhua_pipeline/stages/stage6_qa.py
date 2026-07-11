"""Stage 6: QA.

Quality checks; reports only, never fixes. SUCCESS 0-2 / REVIEW 3-10 / FAILED >10 warnings. Outputs qa.json.
"""

from pathlib import Path

from manhua_pipeline.logging_setup import get_logger, log_page, log_stage

logger = get_logger(__name__)

_STAGE_INDEX = 6
_TOTAL_STAGES = 7
_STAGE_NAME = "QA"


def run_qa(workspace, config):
    """Run the QA stage over all pages in the workspace."""
    ws = Path(workspace)
    log_stage(logger, _STAGE_INDEX, _TOTAL_STAGES, _STAGE_NAME, "starting")

    # TODO: check missing translations, overflow, low OCR confidence,
    #       missing bubbles, rendering failures, unrendered required regions.
    # TODO: status via config.SUCCESS_MAX / config.REVIEW_MAX.
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
