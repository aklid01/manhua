"""Top-level CLI entry point for the manhua translation pipeline.

Provides one command per stage (maximally rerunnable) plus a `run-all`
convenience wrapper with resume support (--from-stage).
"""

import argparse
import sys

import config
from manhua_pipeline.logging_setup import get_logger, setup_logging
from manhua_pipeline.stages import (
    stage0_import,
    stage1_detection,
    stage2_ocr,
    stage3_translation,
    stage4_paraphrase,
    stage5_render,
    stage6_qa,
)

logger = get_logger(__name__)

# Map CLI command -> (label, run function)
STAGES = {
    "import": stage0_import.run_import,
    "detect": stage1_detection.run_detection,
    "ocr": stage2_ocr.run_ocr,
    "translate": stage3_translation.run_translation,
    "paraphrase": stage4_paraphrase.run_paraphrase,
    "render": stage5_render.run_render,
    "qa": stage6_qa.run_qa,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Manhua translation pipeline")
    parser.add_argument(
        "--workspace", default="workspace", help="Path to the workspace folder"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    for name in STAGES:
        sp = sub.add_parser(name, help=f"Run the {name} stage")
        sp.add_argument("--workspace", default="workspace")

    runall = sub.add_parser("run-all", help="Run every stage in order")
    runall.add_argument("--workspace", default="workspace")
    runall.add_argument(
        "--from-stage",
        default="import",
        help="Resume from this stage (import/detect/ocr/...)",
    )
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(stream="stdout")  # CLI logs to stdout; MCP adapter will use stderr

    if args.command == "run-all":
        return _run_all(args.workspace, args.from_stage)

    run_fn = STAGES[args.command]
    logger.info("Running stage: %s", args.command)
    run_fn(args.workspace, config)
    return 0


def _run_all(workspace: str, from_stage: str) -> int:
    order = ["import", "detect", "ocr", "translate", "paraphrase", "render", "qa"]
    if from_stage not in order:
        logger.error("Unknown --from-stage %r (expected one of %s)", from_stage, order)
        return 2
    start = order.index(from_stage)
    logger.info("run-all: starting from %r", from_stage)
    for name in order[start:]:
        logger.info("=" * 60)
        STAGES[name](workspace, config)
    logger.info("run-all: complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
