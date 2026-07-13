"""Stage 0: Import.

Normalize supported inputs (folder/ZIP/CBZ) into ordered pages and write manifest.
"""

import re
import tempfile
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageFile

from manhua_pipeline.io.workspace import ensure_workspace, save_manifest
from manhua_pipeline.logging_setup import get_logger, log_page, log_stage

ImageFile.LOAD_TRUNCATED_IMAGES = True

logger = get_logger(__name__)

_STAGE_INDEX = 0
_TOTAL_STAGES = 7
_STAGE_NAME = "Import"
_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
_STRIP_RATIO = 5.0


def _numeric_key(p: Path) -> tuple:
    """Sort key: tuple of integer runs from filename, raw name as stable tiebreak."""
    nums = tuple(int(n) for n in re.findall(r"\d+", p.name))
    return (nums, p.name)


def run_import(
    input_path: str,
    workspace: str,
    config,
    title_romanized: str | None = None,
    title_english: str | None = None,
    source: str | None = None,
    fresh: bool = False,
) -> Path:
    """Normalize input into ordered pages and write manifest.json."""
    t0 = time.monotonic()
    src = Path(input_path)
    ws = Path(workspace)

    if fresh:
        import shutil

        logger.info(
            "[%s] Running --fresh: cleaning prior chapter artifacts in %s",
            _STAGE_NAME,
            ws,
        )
        for folder_name in config.STAGE_FOLDERS.values():
            folder_path = ws / folder_name
            if folder_path.exists():
                shutil.rmtree(folder_path, ignore_errors=True)
        ov_path = ws / getattr(config, "OVERRIDES_NAME", "overrides.json")
        ov_path.unlink(missing_ok=True)

    ws = ensure_workspace(workspace, config)
    logger.info(
        "[%d/%d %s] Series: %s | Chapter: %s",
        _STAGE_INDEX,
        _TOTAL_STAGES,
        _STAGE_NAME,
        ws.parent.as_posix(),
        ws.name,
    )
    pages_dir = ws / config.STAGE_FOLDERS["pages"]

    log_stage(
        logger, _STAGE_INDEX, _TOTAL_STAGES, _STAGE_NAME, f"starting — source: {src}"
    )

    if not src.exists():
        raise FileNotFoundError(f"Input not found: {src}")

    meta = {
        "title_romanized": title_romanized,
        "title_english": title_english,
        "source": source,
    }

    if src.suffix.lower() in {".cbz", ".zip"}:
        # Pass the archive stem so chapter_id never comes from the temp dir
        manifest = _import_cbz(src, ws, pages_dir, config, t0, src.stem, meta)
    elif src.is_dir():
        manifest = _import_folder(src, ws, pages_dir, config, t0, src.name, meta)
    elif src.suffix.lower() == ".pdf":
        logger.warning(
            "[%s] PDF input detected — extraction deferred (TODO)", _STAGE_NAME
        )
        raise NotImplementedError("PDF import is deferred past v0")
    else:
        raise ValueError(f"Unsupported input: {src}")

    out = ws / config.MANIFEST_NAME
    save_manifest(workspace, config, manifest)
    elapsed = time.monotonic() - t0
    log_stage(
        logger,
        _STAGE_INDEX,
        _TOTAL_STAGES,
        _STAGE_NAME,
        f"done: {manifest['total_pages']} pages, {manifest['warning_count']} warnings, "
        f"format={manifest['input_format']} (elapsed {elapsed:.1f}s) -> {out}",
    )
    return out


def _import_cbz(
    src: Path,
    ws: Path,
    pages_dir: Path,
    config,
    t0: float,
    source_name: str,
    meta: dict,
) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        with zipfile.ZipFile(src, "r") as zf:
            zf.extractall(tmp_path)
        return _import_folder(tmp_path, ws, pages_dir, config, t0, source_name, meta)


def _import_folder(
    src: Path,
    ws: Path,
    pages_dir: Path,
    config,
    t0: float,
    source_name: str,
    meta: dict,
) -> dict:
    image_files = sorted(
        [p for p in src.iterdir() if p.suffix.lower() in _IMAGE_EXTS],
        key=_numeric_key,
    )

    if not image_files:
        raise ValueError(f"No supported images found in {src}")

    logger.info(
        "[%d/%d %s] Found %d image files in %s",
        _STAGE_INDEX,
        _TOTAL_STAGES,
        _STAGE_NAME,
        len(image_files),
        src,
    )
    logger.info(
        "[%d/%d %s] Sorted by numeric filename",
        _STAGE_INDEX,
        _TOTAL_STAGES,
        _STAGE_NAME,
    )

    # Clear stale pages from any prior run before writing new ones
    stale = list(pages_dir.glob("[0-9][0-9][0-9].png"))
    for old in stale:
        old.unlink()
    if stale:
        logger.info(
            "[%d/%d %s] Cleared %d stale page(s) from prior run",
            _STAGE_INDEX,
            _TOTAL_STAGES,
            _STAGE_NAME,
            len(stale),
        )

    pages = []
    warnings = 0
    total = len(image_files)
    strip_count = 0

    # Single-pass: open once, capture dimensions + ratio, convert, save
    for idx, src_file in enumerate(image_files, start=1):
        dest_name = f"{idx:03d}.png"
        dest = pages_dir / dest_name
        try:
            with Image.open(src_file) as im:
                w, h = im.size
                ratio = h / w if w else 1.0
                if ratio > _STRIP_RATIO:
                    strip_count += 1
                im.convert("RGB").save(dest, "PNG")
            log_page(
                logger,
                _STAGE_INDEX,
                _TOTAL_STAGES,
                _STAGE_NAME,
                idx,
                total,
                f"{src_file.name} -> {dest_name}  ({w}x{h})",
            )
            pages.append(
                {
                    "page_number": idx,
                    "filename": dest_name,
                    "original_filename": src_file.name,
                    "width": w,
                    "height": h,
                    "global_y_offset": None,
                    "skip": False,
                    "skip_reason": None,
                }
            )
        except Exception as exc:
            logger.warning(
                "[%s] Page %03d/%03d — FAILED to read %s: %s",
                _STAGE_NAME,
                idx,
                total,
                src_file.name,
                exc,
            )
            warnings += 1
            # Keep entry with filename=None so downstream detects the gap
            pages.append(
                {
                    "page_number": idx,
                    "filename": None,
                    "original_filename": src_file.name,
                    "width": 0,
                    "height": 0,
                    "global_y_offset": None,
                    "skip": True,
                    "skip_reason": "read_error",
                }
            )

    # input_format heuristic: majority-vote on strip ratio (height/width > _STRIP_RATIO=5.0).
    # A smarter boundary (very tall pixel height OR ratio, gray zone) is deferred past v0.
    input_format = "strip" if strip_count > total / 2 else "paginated"
    logger.info(
        "[%d/%d %s] Detected format: %s",
        _STAGE_INDEX,
        _TOTAL_STAGES,
        _STAGE_NAME,
        input_format,
    )
    if input_format == "strip":
        logger.warning(
            "[%s] Strip format detected — slicing deferred (v0). Recorded in manifest.",
            _STAGE_NAME,
        )

    now = datetime.now(timezone.utc).isoformat()
    # chapter_id: use slug from source_name; "chapter_0001" style is an option but
    # slug is kept here for simplicity — downstream can reformat if needed.
    return {
        "chapter_id": _chapter_id_from(source_name),
        # Optional title/source fields — null when not provided (deferred to manual entry)
        "title_romanized": meta.get("title_romanized"),
        "title_english": meta.get("title_english"),
        "source": meta.get("source"),
        "source_language": "zh",
        "target_language": "en-US",
        "input_format": input_format,
        # total_pages counts ALL enumerated pages, including failed/skipped ones
        # (filename=None, skip=True). Downstream stages MUST check skip==False and
        # filename is not None before opening any page file.
        "total_pages": len(pages),
        "current_stage": "detect",
        "completed_stages": ["import"],
        "warning_count": warnings,
        "status": "in_progress",
        "created_at": now,
        "updated_at": now,
        "pages": pages,
    }


def _chapter_id_from(name: str) -> str:
    """Derive a chapter_id slug from the source filename/folder name."""
    slug = name.lower().replace(" ", "_")
    for ch in (".", "-", "(", ")"):
        slug = slug.replace(ch, "_")
    return slug.strip("_")
