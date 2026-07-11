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

    import_sp = sub.add_parser("import", help="Run the import stage")
    import_sp.add_argument("--workspace", default="workspace")
    import_sp.add_argument(
        "--input", required=True, help="Path to a CBZ file or folder of images"
    )
    import_sp.add_argument("--title-romanized", default=None, dest="title_romanized")
    import_sp.add_argument("--title-en", default=None, dest="title_en")
    import_sp.add_argument("--source", default=None)

    for name in [n for n in STAGES if n != "import"]:
        sp = sub.add_parser(name, help=f"Run the {name} stage")
        sp.add_argument("--workspace", default="workspace")

    runall = sub.add_parser("run-all", help="Run every stage in order")
    runall.add_argument("--workspace", default="workspace")
    runall.add_argument(
        "--from-stage",
        default="import",
        help="Resume from this stage (import/detect/ocr/...)",
    )
    runall.add_argument(
        "--input", default=None, help="Source path (required when starting from import)"
    )
    runall.add_argument("--title-romanized", default=None, dest="title_romanized")
    runall.add_argument("--title-en", default=None, dest="title_en")
    runall.add_argument("--source", default=None)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(stream="stdout")  # CLI logs to stdout; MCP adapter will use stderr

    if args.command == "run-all":
        meta = {
            "title_romanized": getattr(args, "title_romanized", None),
            "title_en": getattr(args, "title_en", None),
            "source": getattr(args, "source", None),
        }
        return _run_all(
            args.workspace, args.from_stage, getattr(args, "input", None), meta
        )

    if args.command == "import":
        logger.info("Running stage: import")
        stage0_import.run_import(
            args.input,
            args.workspace,
            config,
            title_romanized=getattr(args, "title_romanized", None),
            title_english=getattr(args, "title_en", None),
            source=getattr(args, "source", None),
        )
        return 0

    run_fn = STAGES[args.command]
    logger.info("Running stage: %s", args.command)
    run_fn(args.workspace, config)
    return 0


def _run_all(
    workspace: str,
    from_stage: str,
    input_path: str | None = None,
    meta: dict | None = None,
) -> int:
    order = ["import", "detect", "ocr", "translate", "paraphrase", "render", "qa"]
    if from_stage not in order:
        logger.error("Unknown --from-stage %r (expected one of %s)", from_stage, order)
        return 2
    start = order.index(from_stage)
    if from_stage == "import" and input_path is None:
        logger.error("--input is required when run-all starts from import")
        return 2
    logger.info("run-all: starting from %r", from_stage)
    meta = meta or {}
    for name in order[start:]:
        logger.info("=" * 60)
        if name == "import":
            stage0_import.run_import(
                input_path,
                workspace,
                config,
                title_romanized=meta.get("title_romanized"),
                title_english=meta.get("title_en"),
                source=meta.get("source"),
            )
        else:
            STAGES[name](workspace, config)
    logger.info("run-all: complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
