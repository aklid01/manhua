"""Stage 5: Rendering.

Mask-fill text removal + font rendering. Overflow ladder: rewrap -> resize -> ask LLM for shorter paraphrase -> warn.
"""

from pathlib import Path

from manhua_pipeline.logging_setup import get_logger, log_page, log_stage

logger = get_logger(__name__)

_STAGE_INDEX = 5
_TOTAL_STAGES = 7
_STAGE_NAME = "Rendering"


def run_render(workspace, config):
    """Run the Rendering stage over all pages in the workspace."""
    ws = Path(workspace)
    log_stage(logger, _STAGE_INDEX, _TOTAL_STAGES, _STAGE_NAME, "starting")

    # TODO: erase original text using erase_mask (not read_region).
    # TODO: render final_text with config.FONT_PATH; apply style_hint (spiky=bold).
    # TODO: overflow ladder -> rewrap, resize, ask-llm, then warn.
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
