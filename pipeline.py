"""Top-level CLI entry point for the manhua translation pipeline.

Provides one command per stage (maximally rerunnable) plus a `run-all`
convenience wrapper with resume support (--from-stage).

Manhua Translation Pipeline
Copyright (C) 2026 Ishan Dev Shakya

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU Affero General Public License as published by the Free
Software Foundation, either version 3 of the License, or (at your option) any
later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License along
with this program. If not, see <https://www.gnu.org/licenses/>.
"""

import argparse
import os
import sys
import time
# comment these if you want to see HF and transformers requests warnings
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
import traceback
from datetime import datetime, timezone
from pathlib import Path

import config
from manhua_pipeline.logging_setup import get_logger, setup_logging

logger = get_logger(__name__)

STAGE_REGISTRY = {
    "import": ("stage0_import", "run_import"),
    "detect": ("stage1_detection", "run_detection"),
    "ocr": ("stage2_ocr", "run_ocr"),
    "translate": ("stage3_translation", "run_translation"),
    "paraphrase": ("stage4_paraphrase", "run_paraphrase"),
    "render": ("stage5_render", "run_render"),
    "qa": ("stage6_qa", "run_qa"),
}


def _load_stage(name):
    """Import a single stage module on demand and return its run function."""
    import importlib

    mod_name, func_name = STAGE_REGISTRY[name]
    module = importlib.import_module(f"manhua_pipeline.stages.{mod_name}")
    return getattr(module, func_name)


def _iter_batch_inputs(folder: Path) -> list[Path]:
    """CBZ/ZIP files in a folder, lexically sorted (assumes user pre-sorted names)."""
    return sorted(
        (
            p
            for p in folder.iterdir()
            if p.is_file() and p.suffix.lower() in {".cbz", ".zip"}
        ),
        key=lambda p: p.name,
    )


def _clear_console_after(delay: int) -> None:
    """Show a countdown, then clear the console. Ctrl-C skips the wait."""
    if delay <= 0:
        return
    try:
        for remaining in range(delay, 0, -1):
            print(
                f"  clearing console in {remaining:>2}s… (Ctrl-C to skip)",
                end="\r",
                flush=True,
            )
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    os.system("cls" if os.name == "nt" else "clear")


def _append_batch_log(log_path: Path, line: str) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    with log_path.open("a", encoding="utf-8") as fh:
        fh.write(f"[{ts}] {line}\n")


def _write_error_log(base_dir: Path, chapter: str, exc: BaseException) -> Path:
    """Full traceback for a failed chapter (its own logs folder if present)."""
    ch_logs = base_dir / chapter / "logs"
    target_dir = ch_logs if (base_dir / chapter).exists() else (base_dir / "logs")
    target_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = target_dir / f"error_{ts}.log"
    with path.open("w", encoding="utf-8") as fh:
        fh.write(f"Chapter: {chapter}\n")
        fh.write(f"Error: {exc}\n\n")
        fh.write("".join(traceback.format_exception(type(exc), exc, exc.__traceback__)))
    return path


def run_batch(
    input_folder: str,
    base_dir: Path,
    config,
    fresh: bool = False,
    resume: bool = True,
    clear_delay: int = 10,
    meta: dict | None = None,
    formats: list[str] | None = None,
) -> int:
    """Process every CBZ in a folder. Continue-on-error; skip completed;
    resume pending; slim batch log + per-error logs; clean console per chapter."""
    from manhua_pipeline.io.workspace import load_manifest

    folder = Path(input_folder)
    if not folder.is_dir():
        logger.error("[batch] Not a folder: %s", folder)
        return 2

    cbz_files = _iter_batch_inputs(folder)
    if not cbz_files:
        logger.warning("[batch] No .cbz/.zip files in %s", folder)
        return 1

    run_ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    batch_log = base_dir / "logs" / f"batch_{run_ts}.log"
    meta = meta or {}

    summary = {"done": [], "pending": [], "skipped": [], "failed": []}
    total = len(cbz_files)
    logger.info("[batch] Found %d chapter file(s) in %s", total, folder)
    _append_batch_log(batch_log, f"BATCH START — {total} file(s) from {folder}")

    for idx, src in enumerate(cbz_files, start=1):
        chapter = src.stem
        chapter_dir = base_dir / chapter
        logger.info("=" * 60)
        logger.info("[batch] (%d/%d) %s", idx, total, chapter)

        # Skip completed / optionally skip any existing
        existing_manifest = (
            load_manifest(str(chapter_dir), config) if chapter_dir.exists() else None
        )
        if existing_manifest and existing_manifest.get("current_stage") == "complete" and not fresh:
            logger.warning("[batch] SKIP (already complete): %s", chapter)
            summary["skipped"].append(chapter)
            _append_batch_log(batch_log, f"SKIP complete   | {chapter}")
            continue
        if chapter_dir.exists() and not resume:
            logger.warning("[batch] SKIP (folder exists, --no-resume): %s", chapter)
            summary["skipped"].append(chapter)
            _append_batch_log(batch_log, f"SKIP exists     | {chapter}")
            continue

        # Run (fresh import OR manifest-driven resume, handled by _run_all_from)
        stage_after = None
        try:
            rc = _run_all_from(
                str(chapter_dir),
                config,
                start="import",
                input_path=str(src),
                meta={
                    "title_romanized": meta.get("title_romanized"),
                    "title_en": meta.get("title_en"),
                    "source": meta.get("source"),
                },
                fresh=fresh,
                formats=formats or [],
            )
            m = load_manifest(str(chapter_dir), config)
            stage_after = m.get("current_stage") if m else None

            if rc == 2:
                summary["failed"].append(chapter)
                _append_batch_log(batch_log, f"FAIL rc=2       | {chapter}")
                logger.error("[batch] %s failed (rc=2). Continuing.", chapter)
            elif stage_after == "complete":
                summary["done"].append(chapter)
                _append_batch_log(batch_log, f"DONE            | {chapter}")
            else:
                summary["pending"].append((chapter, stage_after))
                _append_batch_log(batch_log, f"PENDING @{stage_after} | {chapter}")
        except Exception as exc:  # continue-on-error
            err_path = _write_error_log(base_dir, chapter, exc)
            summary["failed"].append(chapter)
            _append_batch_log(
                batch_log, f"FAIL {type(exc).__name__} | {chapter} -> {err_path}"
            )
            logger.error(
                "[batch] %s crashed: %s (log: %s). Continuing.", chapter, exc, err_path
            )
            continue

        # Console clear ONLY on genuine completion (ignore handoff)
        if stage_after == "complete":
            _clear_console_after(clear_delay)

    _print_batch_summary(summary, batch_log)
    return 0


def _print_batch_summary(summary: dict, batch_log: Path) -> None:
    logger.info("=" * 60)
    logger.info("[batch] SUMMARY")
    logger.info("  done    : %d  %s", len(summary["done"]), summary["done"])
    logger.info(
        "  pending : %d  %s",
        len(summary["pending"]),
        [c for c, _ in summary["pending"]],
    )
    logger.info("  skipped : %d  %s", len(summary["skipped"]), summary["skipped"])
    logger.info("  failed  : %d  %s", len(summary["failed"]), summary["failed"])
    _append_batch_log(
        batch_log,
        f"BATCH END — done={len(summary['done'])} pending={len(summary['pending'])} "
        f"skipped={len(summary['skipped'])} failed={len(summary['failed'])}",
    )
    if summary["pending"]:
        logger.info(
            "[batch] %d chapter(s) awaiting handoff. Process them via MCP "
            "(list_pending → submit), then re-run the same batch command to finish "
            "render→QA (completed chapters are skipped automatically).",
            len(summary["pending"]),
        )


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

    for name in [n for n in STAGE_REGISTRY if n != "import"]:
        sp = sub.add_parser(name, help=f"Run the {name} stage")
        sp.add_argument("--workspace", default="workspace")
        sp.add_argument("--chapter", default=None)

    runall = sub.add_parser("run-all", help="Run every stage in order")
    runall.add_argument("--workspace", default="workspace")
    runall.add_argument("--chapter", default=None)
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
    runall.add_argument(
        "--package",
        default=None,
        help="Comma-separated formats to package after completion (zip,cbz,tar,pdf)",
    )

    batch_sp = sub.add_parser("batch", help="Process a folder of CBZ chapters")
    batch_sp.add_argument(
        "--input", required=True, help="Folder containing .cbz/.zip chapter files"
    )
    batch_sp.add_argument("--title-romanized", default=None, dest="title_romanized")
    batch_sp.add_argument("--title-en", default=None, dest="title_en")
    batch_sp.add_argument("--source", default=None)
    batch_sp.add_argument(
        "--fresh",
        action="store_true",
        help="Wipe prior stage outputs per chapter before importing",
    )
    batch_sp.add_argument(
        "--no-resume",
        action="store_true",
        help="Skip ANY existing chapter folder (default: resume pending, skip completed)",
    )
    batch_sp.add_argument(
        "--clear-delay",
        type=int,
        default=10,
        help="Seconds to show a completed chapter's output before clearing (0 = never)",
    )
    batch_sp.add_argument(
        "--package",
        default=None,
        help="Comma-separated formats to package after each chapter completes",
    )

    pkg_sp = sub.add_parser("package", help="Package rendered pages into archives")
    pkg_sp.add_argument(
        "--chapter", required=True, help="Chapter name in the series folder"
    )
    pkg_sp.add_argument(
        "--package",
        required=True,
        help="Comma-separated formats to package (zip,cbz,tar,pdf)",
    )

    return parser


def _resolve_run_all_chapter_dir(args, base_dir) -> Path:
    chapter = getattr(args, "chapter", None)
    if chapter:
        return base_dir / chapter
    if args.input:
        src = Path(args.input)
        chapter_stem = src.stem if src.suffix.lower() in {".cbz", ".zip"} else src.name
        return base_dir / chapter_stem
    chapter_stem = args.workspace
    if Path(chapter_stem).is_absolute():
        return Path(chapter_stem)
    return base_dir / chapter_stem


def _resolve_import_chapter_dir(args, base_dir) -> Path:
    src = Path(args.input)
    chapter_stem = src.stem if src.suffix.lower() in {".cbz", ".zip"} else src.name
    if Path(args.workspace).is_absolute():
        return Path(args.workspace)
    ws_arg = args.workspace
    if ws_arg == "workspace":
        return base_dir / chapter_stem
    return base_dir / ws_arg


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

    base_dir = resolve_base_dir(args, config)

    if args.command == "batch":
        return run_batch(
            args.input,
            base_dir,
            config,
            fresh=getattr(args, "fresh", False),
            resume=not getattr(args, "no_resume", False),
            clear_delay=getattr(args, "clear_delay", 10),
            meta={
                "title_romanized": getattr(args, "title_romanized", None),
                "title_en": getattr(args, "title_en", None),
                "source": getattr(args, "source", None),
            },
            formats=_parse_formats(getattr(args, "package", None)),
        )

    if args.command == "run-all":
        chapter_dir = _resolve_run_all_chapter_dir(args, base_dir)
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
            formats=_parse_formats(getattr(args, "package", None)),
        )

    if args.command == "import":
        chapter_dir = _resolve_import_chapter_dir(args, base_dir)
        logger.info("Running stage: import")
        _load_stage("import")(
            args.input,
            str(chapter_dir),
            config,
            title_romanized=getattr(args, "title_romanized", None),
            title_english=getattr(args, "title_en", None),
            source=getattr(args, "source", None),
            fresh=getattr(args, "fresh", False),
        )
        return 0

    if args.command == "package":
        from manhua_pipeline.stages import stage7_package

        chapter_dir = base_dir / args.chapter
        stage7_package.run_package(
            str(chapter_dir), config, _parse_formats(args.package)
        )
        return 0

    # For other commands
    chapter = getattr(args, "chapter", None)
    if chapter:
        chapter_dir = base_dir / chapter
    elif Path(args.workspace).is_absolute():
        chapter_dir = Path(args.workspace)
    else:
        chapter_dir = base_dir / args.workspace

    if not (chapter_dir / config.MANIFEST_NAME).exists():
        available = []
        if base_dir.exists():
            available = [
                p.name
                for p in base_dir.iterdir()
                if p.is_dir() and (p / config.MANIFEST_NAME).exists()
            ]
        logger.error(
            "No manifest at %s. Available chapters: %s. Pass --chapter <name>.",
            chapter_dir,
            available,
        )
        return 2

    run_fn = _load_stage(args.command)
    logger.info("Running stage: %s", args.command)
    run_fn(str(chapter_dir), config)
    return 0


def _parse_formats(raw: str | None) -> list[str]:
    return [f.strip() for f in raw.split(",") if f.strip()] if raw else []


def _run_stage_subprocess(
    stage, chapter_dir, *, input_path=None, meta=None, fresh=False
) -> int:
    import subprocess

    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        stage,
        "--workspace",
        str(chapter_dir),
    ]
    if stage == "import":
        if not input_path:
            raise ValueError("import subprocess requires input_path")
        cmd += ["--input", input_path]
        meta = meta or {}
        for flag, key in (
            ("--title-romanized", "title_romanized"),
            ("--title-en", "title_en"),
            ("--source", "source"),
        ):
            if meta.get(key):
                cmd += [flag, meta[key]]
        if fresh:
            cmd += ["--fresh"]
    return subprocess.run(cmd).returncode


def _run_all_from(
    workspace: str,
    config,
    start: str = "import",
    input_path: str | None = None,
    meta: dict | None = None,
    fresh: bool = False,
    formats: list[str] | None = None,
) -> int:
    order = config.STAGE_ORDER
    if start not in order:
        logger.error("Unknown start stage %r (expected one of %s)", start, order)
        return 2

    # Add manifest-driven resume
    from manhua_pipeline.io.workspace import load_manifest

    manifest = load_manifest(workspace, config)
    if manifest and not fresh:
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
        if getattr(config, "BATCH_SUBPROCESS", False):
            rc = _run_stage_subprocess(
                name, workspace, input_path=input_path, meta=meta, fresh=fresh
            )
            if rc != 0:
                logger.error(
                    "Subprocess for stage %r failed with return code %d", name, rc
                )
                return 2
            manifest = load_manifest(workspace, config)
            if not manifest:
                logger.error(
                    "Failed to load manifest after subprocess for stage %r", name
                )
                return 2
            current_stage = manifest.get("current_stage")
            if current_stage == name:
                stage_title = name.title() if name != "ocr" else "OCR"
                logger.info(
                    "Stopping run-all: manual handoff required at %s. Resume with: python pipeline.py run-all",
                    stage_title,
                )
                return 0
        else:
            if name == "import":
                res = _load_stage("import")(
                    input_path,
                    workspace,
                    config,
                    title_romanized=meta.get("title_romanized"),
                    title_english=meta.get("title_en"),
                    source=meta.get("source"),
                    fresh=fresh,
                )
            else:
                res = _load_stage(name)(workspace, config)

            if res is None:
                stage_title = name.title() if name != "ocr" else "OCR"
                logger.info(
                    "Stopping run-all: manual handoff required at %s. Resume with: python pipeline.py run-all",
                    stage_title,
                )
                return 0
    logger.info("run-all: complete")
    if formats:
        from manhua_pipeline.stages import stage7_package

        stage7_package.run_package(workspace, config, formats)
    return 0


if __name__ == "__main__":
    sys.exit(main())
