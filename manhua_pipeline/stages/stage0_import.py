"""Stage 0: Import.

Normalize supported inputs (folder/ZIP/CBZ/PDF/strip) into ordered pages.
"""

from pathlib import Path

from manhua_pipeline.logging_setup import get_logger, log_page, log_stage

logger = get_logger(__name__)

_STAGE_INDEX = 0
_TOTAL_STAGES = 7
_STAGE_NAME = "Import"


def run_import(workspace, config):
    """Run the Import stage over all pages in the workspace."""
    ws = Path(workspace)
    log_stage(logger, _STAGE_INDEX, _TOTAL_STAGES, _STAGE_NAME, "starting")

    # NOTE: source filenames often look like 00000000_00010000.jpg
    #       ({volume}_{chapter}{page}). Sort by the FULL numeric filename and
    #       remap to sequential page numbers (001, 002, ...).
    # TODO: detect paginated vs long-strip and normalize accordingly.
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
