"""Stage 3: Translation.

Literal translation. v0 = manual JSON handoff. Must honor glossary. Outputs translation.json.
"""

from pathlib import Path

from manhua_pipeline.logging_setup import get_logger, log_page, log_stage

logger = get_logger(__name__)

_STAGE_INDEX = 3
_TOTAL_STAGES = 7
_STAGE_NAME = "Translation"


def run_translation(workspace, config):
    """Run the Translation stage over all pages in the workspace."""
    ws = Path(workspace)
    log_stage(logger, _STAGE_INDEX, _TOTAL_STAGES, _STAGE_NAME, "starting")

    # TODO v0: emit a translation prompt JSON for manual handoff and read it back.
    # TODO: apply locked glossary terms.
    # Planned adapters: MCP, Ollama.
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
