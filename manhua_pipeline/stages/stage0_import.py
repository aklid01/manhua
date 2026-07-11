"""Stage 0: Import.

Normalize supported inputs (folder/ZIP/CBZ) into ordered pages and write manifest.
"""

import tempfile
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

from manhua_pipeline.io.workspace import ensure_workspace, save_manifest
from manhua_pipeline.logging_setup import get_logger, log_page, log_stage

logger = get_logger(__name__)

_STAGE_INDEX = 0
_TOTAL_STAGES = 7
_STAGE_NAME = "Import"
_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
_STRIP_RATIO = 5.0
_PAGE_RATIO = 2.5


def run_import(input_path: str, workspace: str, config) -> Path:
    """Normalize input into ordered pages and write manifest.json."""
    t0 = time.monotonic()
    src = Path(input_path)
    ws = ensure_workspace(workspace, config)
    pages_dir = ws / config.STAGE_FOLDERS["pages"]

    log_stage(
        logger, _STAGE_INDEX, _TOTAL_STAGES, _STAGE_NAME, f"starting — source: {src}"
    )

    if not src.exists():
        raise FileNotFoundError(f"Input not found: {src}")

    if src.suffix.lower() in {".cbz", ".zip"}:
        manifest = _import_cbz(src, ws, pages_dir, config, t0)
    elif src.is_dir():
        manifest = _import_folder(src, ws, pages_dir, config, t0)
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


def _import_cbz(src: Path, ws: Path, pages_dir: Path, config, t0: float) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        with zipfile.ZipFile(src, "r") as zf:
            zf.extractall(tmp_path)
        return _import_folder(tmp_path, ws, pages_dir, config, t0)


def _import_folder(src: Path, ws: Path, pages_dir: Path, config, t0: float) -> dict:
    image_files = sorted(
        [p for p in src.iterdir() if p.suffix.lower() in _IMAGE_EXTS],
        key=lambda p: p.name,
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
        "[%d/%d %s] Sorted by filename", _STAGE_INDEX, _TOTAL_STAGES, _STAGE_NAME
    )

    ratios = []
    for p in image_files:
        try:
            with Image.open(p) as im:
                w, h = im.size
                ratios.append(h / w if w else 1.0)
        except Exception:
            ratios.append(1.0)

    strip_count = sum(1 for r in ratios if r > _STRIP_RATIO)
    input_format = "strip" if strip_count > len(image_files) / 2 else "paginated"
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

    pages = []
    warnings = 0
    total = len(image_files)

    for idx, src_file in enumerate(image_files, start=1):
        dest_name = f"{idx:03d}.png"
        dest = pages_dir / dest_name
        try:
            with Image.open(src_file) as im:
                w, h = im.size
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
            logger.error(
                "[%s] Page %03d/%03d — FAILED to read %s: %s",
                _STAGE_NAME,
                idx,
                total,
                src_file.name,
                exc,
            )
            warnings += 1
            pages.append(
                {
                    "page_number": idx,
                    "filename": dest_name,
                    "original_filename": src_file.name,
                    "width": 0,
                    "height": 0,
                    "global_y_offset": None,
                    "skip": True,
                    "skip_reason": "read_error",
                }
            )

    now = datetime.now(timezone.utc).isoformat()
    return {
        "chapter_id": _chapter_id_from(src.name if src.is_file() else src.parent.name),
        "source_language": "zh",
        "target_language": "en-US",
        "input_format": input_format,
        "total_pages": len(pages),
        "current_stage": "detection",
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
    return slug
