# Stage 0: Normalize supported inputs (folder/ZIP/CBZ) into ordered pages and write manifest.

import os
import re
import shutil
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
_PASSTHROUGH_MODES = {"RGB", "L", "YCbCr"}


# Numeric sorting key extracting embedded integer runs from filename
def _numeric_key(p: Path) -> tuple:
    nums = tuple(int(n) for n in re.findall(r"\d+", p.name))
    return (nums, p.name)


# Compute per-row non-background content density across image height
def _row_content_densities(image: Image.Image, config) -> list[float]:
    import numpy as np

    gray = image.convert("L")
    sample_width = min(gray.width, 256)
    if sample_width != gray.width:
        gray = gray.resize((sample_width, gray.height), Image.Resampling.BILINEAR)
    threshold = int(getattr(config, "IMPORT_SLICE_CONTENT_THRESHOLD", 245))
    arr = np.frombuffer(gray.tobytes(), dtype=np.uint8).reshape(
        gray.height, sample_width
    )
    return (arr < threshold).mean(axis=1).tolist()


# Locate optimal low-content horizontal seam near desired split height
def _find_safe_seam(
    image: Image.Image, desired: int, config, densities: list[float]
) -> int | None:
    height = image.height
    target = max(1, int(getattr(config, "IMPORT_SLICE_TARGET_HEIGHT", 5000)))
    tolerance = max(
        1,
        int(target * float(getattr(config, "IMPORT_SLICE_SEAM_TOLERANCE", 0.20))),
    )
    band = max(1, int(getattr(config, "IMPORT_SLICE_SEAM_BAND", 24)))
    max_density = float(getattr(config, "IMPORT_SLICE_MAX_CONTENT_DENSITY", 0.02))
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


# Calculate safe vertical slice boundaries for oversized webtoon strips
def _safe_slice_boundaries(image: Image.Image, config) -> list[tuple[int, int]] | None:
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


# Normalize input images/archives into numbered pages and write manifest
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
    t0 = time.monotonic()
    src = Path(input_path)
    ws = Path(workspace)

    if fresh:
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


# Unpack CBZ/ZIP archive into temporary folder and run import
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


# Process image directory, convert modes, slice tall strips, and build manifest
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

    stale = [
        p
        for p in pages_dir.iterdir()
        if p.is_file() and p.suffix.lower() in _IMAGE_EXTS and p.stem.isdigit()
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
    max_w = getattr(config, "IMPORT_MAX_WIDTH", 0)

    stitch_small_enabled = getattr(config, "IMPORT_STITCH_SMALL_PAGES", True)
    min_page_h = int(getattr(config, "IMPORT_MIN_PAGE_HEIGHT", 600))
    target_stitch_h = int(getattr(config, "IMPORT_TARGET_STITCH_HEIGHT", 1800))
    max_stitch_h = int(getattr(config, "IMPORT_MAX_STITCH_HEIGHT", 2200))

    pending_slices = []

    def _flush_pending():
        nonlocal page_number, sliced_count
        if not pending_slices:
            return

        if len(pending_slices) == 1:
            im, src_file, src_idx, ext, needs_downscale, needs_convert = pending_slices[
                0
            ]
            w, h = im.size
            page_number += 1
            dest_name = f"{page_number:03d}{ext}"
            dest = pages_dir / dest_name
            if not needs_convert and not needs_downscale:
                shutil.copyfile(src_file, dest)
                log_page(
                    logger,
                    _STAGE_INDEX,
                    _TOTAL_STAGES,
                    _STAGE_NAME,
                    page_number,
                    total,
                    f"{src_file.name} -> {dest_name}  ({w}x{h}) [pass-through]",
                )
            else:
                im_to_save = im.convert("RGB")
                if needs_downscale:
                    im_to_save.thumbnail((max_w, max_w * 10), Image.Resampling.LANCZOS)
                im_to_save.save(dest)
                w, h = im_to_save.size
                log_page(
                    logger,
                    _STAGE_INDEX,
                    _TOTAL_STAGES,
                    _STAGE_NAME,
                    page_number,
                    total,
                    f"{src_file.name} -> {dest_name}  ({w}x{h})",
                )

            pages.append(
                {
                    "page_number": page_number,
                    "filename": dest_name,
                    "original_filename": src_file.name,
                    "source_filename": src_file.name,
                    "source_page_number": src_idx,
                    "slice_index": 1,
                    "slice_count": 1,
                    "slice_top": 0,
                    "slice_bottom": h,
                    "width": w,
                    "height": h,
                    "global_y_offset": None,
                    "skip": False,
                    "skip_reason": None,
                }
            )
            pending_slices.clear()
            return

        # Multiple small slices to stitch vertically
        stacked_w = pending_slices[0][0].width
        stacked_h = sum(s[0].height for s in pending_slices)
        stacked_im = Image.new("RGB", (stacked_w, stacked_h))

        curr_y = 0
        src_names = []
        for s_im, s_file, _, _, _, _ in pending_slices:
            stacked_im.paste(s_im.convert("RGB"), (0, curr_y))
            curr_y += s_im.height
            src_names.append(s_file.name)

        if max_w and stacked_w > max_w:
            stacked_im.thumbnail((max_w, max_w * 10), Image.Resampling.LANCZOS)
            stacked_w, stacked_h = stacked_im.size

        page_number += 1
        ext = pending_slices[0][3]
        dest_name = f"{page_number:03d}{ext}"
        dest = pages_dir / dest_name
        stacked_im.save(dest)

        merged_src_desc = "+".join(src_names)
        log_page(
            logger,
            _STAGE_INDEX,
            _TOTAL_STAGES,
            _STAGE_NAME,
            page_number,
            total,
            f"[{merged_src_desc}] -> {dest_name}  ({stacked_w}x{stacked_h}) [stitched {len(pending_slices)} slices]",
        )

        pages.append(
            {
                "page_number": page_number,
                "filename": dest_name,
                "original_filename": "+".join(src_names),
                "source_filename": src_names[0],
                "source_page_number": pending_slices[0][2],
                "slice_index": 1,
                "slice_count": 1,
                "slice_top": 0,
                "slice_bottom": stacked_h,
                "width": stacked_w,
                "height": stacked_h,
                "global_y_offset": None,
                "skip": False,
                "skip_reason": None,
            }
        )
        pending_slices.clear()

    for source_index, src_file in enumerate(image_files, start=1):
        try:
            with Image.open(src_file) as im:
                src_mode = im.mode
                w, h = im.size
                needs_downscale = bool(max_w and w > max_w)
                needs_convert = src_mode not in _PASSTHROUGH_MODES
                extension = src_file.suffix.lower()

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
                needs_slice = slice_count > 1

                if needs_slice:
                    _flush_pending()
                    sliced_count += 1
                    im = im.convert("RGB")
                    if needs_downscale:
                        im.thumbnail((max_w, max_w * 10), Image.Resampling.LANCZOS)
                        w, h = im.size

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
                    continue

                if stitch_small_enabled:
                    is_small = h < min_page_h
                    if is_small or (
                        pending_slices
                        and sum(s[0].height for s in pending_slices) < target_stitch_h
                        and h < target_stitch_h
                    ):
                        if not pending_slices:
                            if is_small:
                                pending_slices.append(
                                    (
                                        im.copy(),
                                        src_file,
                                        source_index,
                                        extension,
                                        needs_downscale,
                                        needs_convert,
                                    )
                                )
                                continue
                        else:
                            curr_pending_w = pending_slices[0][0].width
                            curr_pending_ext = pending_slices[0][3]
                            curr_pending_h = sum(s[0].height for s in pending_slices)
                            if (
                                w == curr_pending_w
                                and extension == curr_pending_ext
                                and curr_pending_h + h <= max_stitch_h
                            ):
                                pending_slices.append(
                                    (
                                        im.copy(),
                                        src_file,
                                        source_index,
                                        extension,
                                        needs_downscale,
                                        needs_convert,
                                    )
                                )
                                if curr_pending_h + h >= target_stitch_h:
                                    _flush_pending()
                                continue
                            else:
                                _flush_pending()
                                if is_small:
                                    pending_slices.append(
                                        (
                                            im.copy(),
                                            src_file,
                                            source_index,
                                            extension,
                                            needs_downscale,
                                            needs_convert,
                                        )
                                    )
                                    continue

                _flush_pending()

                page_number += 1
                dest_name = f"{page_number:03d}{extension}"
                dest = pages_dir / dest_name
                if not needs_convert and not needs_downscale:
                    shutil.copyfile(src_file, dest)
                    log_page(
                        logger,
                        _STAGE_INDEX,
                        _TOTAL_STAGES,
                        _STAGE_NAME,
                        page_number,
                        total,
                        f"{src_file.name} -> {dest_name}  ({w}x{h}) [pass-through]",
                    )
                else:
                    im_to_save = im.convert("RGB")
                    if needs_downscale:
                        im_to_save.thumbnail(
                            (max_w, max_w * 10), Image.Resampling.LANCZOS
                        )
                    im_to_save.save(dest)
                    w, h = im_to_save.size
                    log_page(
                        logger,
                        _STAGE_INDEX,
                        _TOTAL_STAGES,
                        _STAGE_NAME,
                        page_number,
                        total,
                        f"{src_file.name} -> {dest_name}  ({w}x{h})",
                    )

                pages.append(
                    {
                        "page_number": page_number,
                        "filename": dest_name,
                        "original_filename": src_file.name,
                        "source_filename": src_file.name,
                        "source_page_number": source_index,
                        "slice_index": 1,
                        "slice_count": 1,
                        "slice_top": 0,
                        "slice_bottom": h,
                        "width": w,
                        "height": h,
                        "global_y_offset": None,
                        "skip": False,
                        "skip_reason": None,
                    }
                )
        except Exception as exc:
            _flush_pending()
            logger.warning(
                "[%s] Source %03d/%03d - FAILED to read %s: %s",
                _STAGE_NAME,
                source_index,
                total,
                src_file.name,
                exc,
            )
            warnings += 1
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
    _flush_pending()

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


# Derive sanitized chapter ID slug from filename or folder name
def _chapter_id_from(name: str) -> str:
    slug = name.lower().replace(" ", "_")
    for ch in (".", "-", "(", ")"):
        slug = slug.replace(ch, "_")
    return slug.strip("_")
