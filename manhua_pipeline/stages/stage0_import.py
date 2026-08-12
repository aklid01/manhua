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


def _row_content_densities(image: Image.Image, config) -> list[float]:
    """Return a cheap per-row estimate of non-background/content density."""
    gray = image.convert("L")
    sample_width = min(gray.width, 256)
    if sample_width != gray.width:
        gray = gray.resize((sample_width, gray.height), Image.Resampling.BILINEAR)
    pixels = list(gray.getdata())
    threshold = int(getattr(config, "IMPORT_SLICE_CONTENT_THRESHOLD", 245))
    densities = []
    for y in range(gray.height):
        row = pixels[y * sample_width : (y + 1) * sample_width]
        densities.append(sum(value < threshold for value in row) / sample_width)
    return densities


def _find_safe_seam(
    image: Image.Image, desired: int, config, densities: list[float]
) -> int | None:
    """Find the nearest low-content horizontal seam around ``desired``."""
    height = image.height
    target = max(1, int(getattr(config, "IMPORT_SLICE_TARGET_HEIGHT", 5000)))
    tolerance = max(
        1,
        int(
            target
            * float(getattr(config, "IMPORT_SLICE_SEAM_TOLERANCE", 0.20))
        ),
    )
    band = max(1, int(getattr(config, "IMPORT_SLICE_SEAM_BAND", 24)))
    max_density = float(
        getattr(config, "IMPORT_SLICE_MAX_CONTENT_DENSITY", 0.02)
    )
    low = max(band, desired - tolerance)
    high = min(height - band, desired + tolerance)
    if low > high:
        return None

    candidates = []
    for y in range(low, high + 1, max(1, image.width // 256)):
        start = max(0, y - band)
        end = min(height, y + band)
        score = sum(densities[start:end]) / max(1, end - start)
        if score <= max_density:
            candidates.append((score, abs(y - desired), y))
    if not candidates:
        return None
    return min(candidates)[2]


def _safe_slice_boundaries(image: Image.Image, config) -> list[tuple[int, int]] | None:
    """Return safe vertical slices, or None when a tall page cannot be split safely."""
    if not getattr(config, "IMPORT_SLICE_ENABLED", True):
        return [(0, image.height)]

    height = image.height
    width = image.width
    trigger_ratio = float(getattr(config, "IMPORT_SLICE_TRIGGER_RATIO", 5.0))
    min_height = int(getattr(config, "IMPORT_SLICE_MIN_HEIGHT", 4000))
    target = int(getattr(config, "IMPORT_SLICE_TARGET_HEIGHT", 5000))
    if (
        width <= 0
        or height < min_height
        or height / width < trigger_ratio
        or target <= 0
    ):
        return [(0, height)]

    densities = _row_content_densities(image, config)
    min_segment = max(1, int(target * 0.5))
    boundaries = []
    desired = target
    while height - desired >= min_segment:
        seam = _find_safe_seam(image, desired, config, densities)
        if seam is None:
            return None
        boundaries.append(seam)
        desired = seam + target

    points = [0, *boundaries, height]
    return list(zip(points, points[1:]))


def run_import(
    input_path: str,
    workspace: str,
    config,
    title_romanized: str | None = None,
    title_english: str | None = None,
    source: str | None = None,
    fresh: bool = False,
    skip_last: int = 0,
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
        logs_folder_name = config.STAGE_FOLDERS.get("logs", "logs")
        for folder_name in config.STAGE_FOLDERS.values():
            folder_path = ws / folder_name
            if folder_path.exists():
                if folder_name == logs_folder_name:
                    for log_file in folder_path.glob("*.log"):
                        try:
                            log_file.unlink()
                        except Exception:
                            pass
                else:
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
        logger, _STAGE_INDEX, _TOTAL_STAGES, _STAGE_NAME, f"starting - source: {src}"
    )
    from manhua_pipeline.events import STAGE_COMPLETED, STAGE_STARTED
    from manhua_pipeline.io.event_emitter import emit_stage_event

    emit_stage_event(ws, config, _STAGE_NAME.lower(), STAGE_STARTED, logger=logger)

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
            "[%s] PDF input detected - extraction deferred (TODO)", _STAGE_NAME
        )
        raise NotImplementedError("PDF import is deferred past v0")
    else:
        raise ValueError(f"Unsupported input: {src}")

    if skip_last and skip_last > 0:
        usable = [
            p
            for p in manifest.get("pages", [])
            if not p.get("skip") and p.get("filename")
        ]
        for p in usable[-skip_last:]:
            p["skip"] = True
            p["skip_reason"] = "trailing_skip"
        logger.warning(
            "[%s] Skipping last %d page(s) per user (--skip-last): %s",
            _STAGE_NAME,
            skip_last,
            [p["page_number"] for p in usable[-skip_last:]],
        )

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
    emit_stage_event(
        ws,
        config,
        _STAGE_NAME.lower(),
        STAGE_COMPLETED,
        logger=logger,
        metadata={"total_pages": manifest["total_pages"]},
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

    # Clear stale canonical pages from any prior run before writing new ones.
    stale = [
        p
        for p in pages_dir.iterdir()
        if p.is_file()
        and p.suffix.lower() in _IMAGE_EXTS
        and p.stem.isdigit()
    ]
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
    sliced_count = 0
    page_number = 0

    # Single-pass per source image: normalize dimensions, optionally slice, save.
    for source_index, src_file in enumerate(image_files, start=1):
        try:
            with Image.open(src_file) as im:
                im = im.convert("RGB")
                w, h = im.size
                max_w = getattr(config, "IMPORT_MAX_WIDTH", 0)
                if max_w and w > max_w:
                    im.thumbnail((max_w, max_w * 10), Image.Resampling.LANCZOS)
                    w, h = im.size
                ratio = h / w if w else 1.0
                if ratio > _STRIP_RATIO:
                    strip_count += 1

                boundaries = _safe_slice_boundaries(im, config)
                if boundaries is None:
                    logger.warning(
                        "[%s] %s is unusually tall but no safe seam was found; keeping intact",
                        _STAGE_NAME,
                        src_file.name,
                    )
                    warnings += 1
                    boundaries = [(0, h)]
                slice_count = len(boundaries)
                if slice_count > 1:
                    sliced_count += 1

                extension = src_file.suffix.lower()
                for slice_index, (top, bottom) in enumerate(boundaries, start=1):
                    page_number += 1
                    dest_name = f"{page_number:03d}{extension}"
                    dest = pages_dir / dest_name
                    piece = im.crop((0, top, w, bottom))
                    piece.save(dest)
                    piece_width, piece_height = piece.size
                    log_page(
                        logger,
                        _STAGE_INDEX,
                        _TOTAL_STAGES,
                        _STAGE_NAME,
                        page_number,
                        total,
                        f"{src_file.name} -> {dest_name}  ({piece_width}x{piece_height})",
                    )
                    pages.append(
                        {
                            "page_number": page_number,
                            "filename": dest_name,
                            "original_filename": src_file.name,
                            "source_filename": src_file.name,
                            "source_page_number": source_index,
                            "slice_index": slice_index,
                            "slice_count": slice_count,
                            "slice_top": top,
                            "slice_bottom": bottom,
                            "width": piece_width,
                            "height": piece_height,
                            "global_y_offset": None,
                            "skip": False,
                            "skip_reason": None,
                        }
                    )
        except Exception as exc:
            logger.warning(
                "[%s] Source %03d/%03d — FAILED to read %s: %s",
                _STAGE_NAME,
                source_index,
                total,
                src_file.name,
                exc,
            )
            warnings += 1
            # Keep entry with filename=None so downstream detects the gap
            page_number += 1
            pages.append(
                {
                    "page_number": page_number,
                    "filename": None,
                    "original_filename": src_file.name,
                    "source_filename": src_file.name,
                    "source_page_number": source_index,
                    "slice_index": 1,
                    "slice_count": 1,
                    "width": 0,
                    "height": 0,
                    "global_y_offset": None,
                    "skip": True,
                    "skip_reason": "read_error",
                }
            )

    # input_format heuristic: majority-vote on the source strip ratio.
    input_format = "strip" if strip_count > total / 2 else "paginated"
    logger.info(
        "[%d/%d %s] Detected format: %s (%d source page(s) sliced)",
        _STAGE_INDEX,
        _TOTAL_STAGES,
        _STAGE_NAME,
        input_format,
        sliced_count,
    )

    now = datetime.now(timezone.utc).isoformat()
    # chapter_id: use slug from source_name; "chapter_0001" style is an option but
    # slug is kept here for simplicity — downstream can reformat if needed.
    return {
        "chapter_id": _chapter_id_from(source_name),
        "source_archive": source_name,
        "title_romanized": meta.get("title_romanized"),
        "title_english": meta.get("title_english"),
        "source": meta.get("source"),
        "source_language": "zh",
        "target_language": "en-US",
        "input_format": input_format,
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
