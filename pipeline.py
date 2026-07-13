"""Top-level CLI entry point for the manhua translation pipeline.

Provides one command per stage (maximally rerunnable) plus a `run-all`
convenience wrapper with resume support (--from-stage).
"""

import argparse
import sys
from pathlib import Path

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
    "import": lambda *args, **kwargs: stage0_import.run_import(*args, **kwargs),
    "detect": lambda *args, **kwargs: stage1_detection.run_detection(*args, **kwargs),
    "ocr": lambda *args, **kwargs: stage2_ocr.run_ocr(*args, **kwargs),
    "translate": lambda *args, **kwargs: stage3_translation.run_translation(
        *args, **kwargs
    ),
    "paraphrase": lambda *args, **kwargs: stage4_paraphrase.run_paraphrase(
        *args, **kwargs
    ),
    "render": lambda *args, **kwargs: stage5_render.run_render(*args, **kwargs),
    "qa": lambda *args, **kwargs: stage6_qa.run_qa(*args, **kwargs),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Manhua translation pipeline")
    parser.add_argument(
        "--workspace", default="workspace", help="Path to the workspace folder"
    )
    parser.add_argument(
        "--output-dir", help="Override the series base directory for this run"
    )
    parser.add_argument(
        "--set-output-dir",
        help="Persist the series base directory to settings and exit",
    )
    sub = parser.add_subparsers(dest="command", required=False)

    import_sp = sub.add_parser("import", help="Run the import stage")
    import_sp.add_argument("--workspace", default="workspace")
    import_sp.add_argument(
        "--input", required=True, help="Path to a CBZ file or folder of images"
    )
    import_sp.add_argument("--title-romanized", default=None, dest="title_romanized")
    import_sp.add_argument("--title-en", default=None, dest="title_en")
    import_sp.add_argument("--source", default=None)
    import_sp.add_argument(
        "--fresh", action="store_true", help="Wipe prior stage outputs and prompts"
    )

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
    runall.add_argument(
        "--fresh",
        action="store_true",
        help="Wipe prior stage outputs and prompts when starting from import",
    )
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    setup_logging(stream="stdout")  # CLI logs to stdout; MCP adapter will use stderr

    # Handle --set-output-dir persisting base directory
    if getattr(args, "set_output_dir", None):
        from manhua_pipeline.io.settings import set_output_dir

        set_output_dir(args.set_output_dir)
        logger.info("Persisted series base directory: %s", args.set_output_dir)
        return 0

    if args.command is None:
        parser.print_help()
        return 1

    from manhua_pipeline.io.settings import resolve_base_dir

    if args.command == "run-all":
        if args.input:
            src = Path(args.input)
            chapter_stem = (
                src.stem if src.suffix.lower() in {".cbz", ".zip"} else src.name
            )
        else:
            chapter_stem = args.workspace

        if Path(chapter_stem).is_absolute():
            chapter_dir = Path(chapter_stem)
        else:
            base_dir = resolve_base_dir(args, config)
            chapter_dir = base_dir / chapter_stem

        meta = {
            "title_romanized": getattr(args, "title_romanized", None),
            "title_en": getattr(args, "title_en", None),
            "source": getattr(args, "source", None),
        }
        return _run_all_from(
            str(chapter_dir),
            config,
            start=args.from_stage,
            input_path=getattr(args, "input", None),
            meta=meta,
            fresh=getattr(args, "fresh", False),
        )

    if args.command == "import":
        src = Path(args.input)
        chapter_stem = src.stem if src.suffix.lower() in {".cbz", ".zip"} else src.name
        if Path(args.workspace).is_absolute():
            chapter_dir = Path(args.workspace)
        else:
            base_dir = resolve_base_dir(args, config)
            ws_arg = args.workspace
            if ws_arg == "workspace":
                chapter_dir = base_dir / chapter_stem
            else:
                chapter_dir = base_dir / ws_arg

        logger.info("Running stage: import")
        stage0_import.run_import(
            args.input,
            str(chapter_dir),
            config,
            title_romanized=getattr(args, "title_romanized", None),
            title_english=getattr(args, "title_en", None),
            source=getattr(args, "source", None),
            fresh=getattr(args, "fresh", False),
        )
        return 0

    # For other commands
    ws_arg = args.workspace
    if Path(ws_arg).is_absolute():
        chapter_dir = Path(ws_arg)
    else:
        base_dir = resolve_base_dir(args, config)
        chapter_dir = base_dir / ws_arg

    run_fn = STAGES[args.command]
    logger.info("Running stage: %s", args.command)
    run_fn(str(chapter_dir), config)
    return 0


def _run_all_from(
    workspace: str,
    config,
    start: str = "import",
    input_path: str | None = None,
    meta: dict | None = None,
    fresh: bool = False,
) -> int:
    order = ["import", "detect", "ocr", "translate", "paraphrase", "render", "qa"]
    if start not in order:
        logger.error("Unknown start stage %r (expected one of %s)", start, order)
        return 2

    # Add manifest-driven resume
    from manhua_pipeline.io.workspace import load_manifest

    manifest = load_manifest(workspace, config)
    if manifest:
        current_stage = manifest.get("current_stage")
        if current_stage == "complete":
            logger.info("run-all: chapter is already complete.")
            return 0
        if current_stage in order:
            if start == "import" or order.index(current_stage) > order.index(start):
                start = current_stage

    start_idx = order.index(start)
    if start == "import" and input_path is None:
        logger.error("--input is required when run-all starts from import")
        return 2
    logger.info("run-all: starting from %r", start)
    meta = meta or {}
    for name in order[start_idx:]:
        logger.info("=" * 60)
        if name == "import":
            res = stage0_import.run_import(
                input_path,
                workspace,
                config,
                title_romanized=meta.get("title_romanized"),
                title_english=meta.get("title_en"),
                source=meta.get("source"),
                fresh=fresh,
            )
        else:
            res = STAGES[name](workspace, config)

        if res is None:
            stage_title = name.title() if name != "ocr" else "OCR"
            logger.info(
                "Stopping run-all: manual handoff required at %s. Resume with: python pipeline.py run-all",
                stage_title,
            )
            return 0
    logger.info("run-all: complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
