"""Stage 5: Rendering.

Erases original Chinese text from bounding box areas and draws the paraphrased
US English text in its place. Utilizes Pillow to wrap, size, and style.
"""

import json
import time
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from manhua_pipeline.io.workspace import load_manifest, save_manifest
from manhua_pipeline.logging_setup import get_logger, log_stage

logger = get_logger(__name__)

_STAGE_INDEX = 5
_TOTAL_STAGES = 7
_STAGE_NAME = "Render"

_font_cache = {}


def _load_font(
    font_path: str, pt: int, config
) -> ImageFont.ImageFont | ImageFont.FreeTypeFont:
    """Load free type font and cache it by point size."""
    key = (font_path, pt)
    if key in _font_cache:
        return _font_cache[key]

    p = Path(font_path)
    if not p.exists():
        if getattr(config, "FONT_MISSING_HARD_ERROR", True):
            raise FileNotFoundError(
                f"Font file not found at {font_path}. Place ComicNeue-Regular.ttf in assets/fonts/ "
                "or set FONT_MISSING_HARD_ERROR = False in config.py to use PIL default font."
            )
        else:
            logger.warning(
                "Font not found at %s. Falling back to default font.", font_path
            )
            font = ImageFont.load_default()
            _font_cache[key] = font
            return font

    try:
        font = ImageFont.truetype(str(p), pt)
    except Exception as exc:
        if getattr(config, "FONT_MISSING_HARD_ERROR", True):
            raise exc
        else:
            logger.warning(
                "Failed to load truetype font %s: %s. Falling back to default font.",
                font_path,
                exc,
            )
            font = ImageFont.load_default()

    _font_cache[key] = font
    return font


def _estimate_bg(
    page_img: Image.Image, x: int, y: int, w: int, h: int
) -> tuple[int, int, int]:
    """Sample a 2px boundary ring around the mask bounding box and get its median color."""
    x0, y0 = max(0, x - 2), max(0, y - 2)
    x1, y1 = min(page_img.width - 1, x + w + 2), min(page_img.height - 1, y + h + 2)

    pixels = []
    # Sample top/bottom rows
    for px in range(x0, x1 + 1):
        pixels.append(page_img.getpixel((px, y0)))
        pixels.append(page_img.getpixel((px, y1)))
    # Sample left/right columns
    for py in range(y0, y1 + 1):
        pixels.append(page_img.getpixel((x0, py)))
        pixels.append(page_img.getpixel((x1, py)))

    # Keep only rgb
    rgb_pixels = []
    for p in pixels:
        if isinstance(p, tuple) and len(p) >= 3:
            rgb_pixels.append(p[:3])
        elif isinstance(p, int):
            rgb_pixels.append((p, p, p))

    if not rgb_pixels:
        return (255, 255, 255)

    # Median per R, G, B channel
    r_vals = sorted([p[0] for p in rgb_pixels])
    g_vals = sorted([p[1] for p in rgb_pixels])
    b_vals = sorted([p[2] for p in rgb_pixels])

    return (
        r_vals[len(r_vals) // 2],
        g_vals[len(g_vals) // 2],
        b_vals[len(b_vals) // 2],
    )


def _find_row_boundaries(
    is_white: list[list[bool]], width: int, height: int
) -> tuple[list[int], list[int]]:
    first_w = [-1] * height
    last_w = [-1] * height
    for y in range(height):
        for x in range(width):
            if is_white[y][x]:
                if first_w[y] == -1:
                    first_w[y] = x
                last_w[y] = x
    return first_w, last_w


def _find_col_boundaries(
    is_white: list[list[bool]], width: int, height: int
) -> tuple[list[int], list[int]]:
    first_w = [-1] * width
    last_w = [-1] * width
    for x in range(width):
        for y in range(height):
            if is_white[y][x]:
                if first_w[x] == -1:
                    first_w[x] = y
                last_w[x] = y
    return first_w, last_w


def _get_bubble_mask(bbox_img: Image.Image, config) -> Image.Image:
    """Generate a binary mask identifying the white/near-white bubble area."""
    img = bbox_img.convert("RGB")
    width, height = img.size

    threshold = getattr(config, "BUBBLE_WHITE_THRESHOLD", 220)

    # 1. Create a binary grid of white pixels
    is_white = []
    for y in range(height):
        row = []
        for x in range(width):
            r, g, b = img.getpixel((x, y))
            row.append(r > threshold and g > threshold and b > threshold)
        is_white.append(row)

    # 2. Find row/col boundaries for white pixels
    first_w_in_row, last_w_in_row = _find_row_boundaries(is_white, width, height)
    first_w_in_col, last_w_in_col = _find_col_boundaries(is_white, width, height)

    # 3. Create mask image
    mask = Image.new("L", (width, height), 0)
    for y in range(height):
        for x in range(width):
            if is_white[y][x]:
                mask.putpixel((x, y), 255)
            else:
                # Surrounded by white check
                if (
                    first_w_in_row[y] != -1
                    and first_w_in_row[y] < x
                    and last_w_in_row[y] > x
                    and first_w_in_col[x] != -1
                    and first_w_in_col[x] < y
                    and last_w_in_col[x] > y
                ):
                    mask.putpixel((x, y), 255)

    return mask


def _wrap_text(text: str, font: ImageFont.ImageFont, max_width: int) -> list[str]:
    """Word wraps lines to stay within max_width."""
    paragraphs = text.split("\n")
    all_lines = []

    for para in paragraphs:
        words = para.split()
        if not words:
            all_lines.append("")
            continue

        current_line = []
        for word in words:
            test_line = " ".join(current_line + [word])
            width = font.getlength(test_line)
            if width <= max_width or not current_line:
                current_line.append(word)
            else:
                all_lines.append(" ".join(current_line))
                current_line = [word]
        if current_line:
            all_lines.append(" ".join(current_line))

    return all_lines


def _block_height(
    lines: list[str], font: ImageFont.ImageFont, line_spacing: float
) -> int:
    """Calculates height of multiline text block."""
    if not lines:
        return 0
    bbox = font.getbbox("Hg")
    line_h = bbox[3] - bbox[1] if bbox else 10

    total_h = 0
    for i in range(len(lines)):
        if i > 0:
            total_h += int(line_h * line_spacing)
        else:
            total_h += line_h
    return total_h


def _fit_text(
    text: str,
    bbox_w: int,
    bbox_h: int,
    font_path: str,
    max_pt: int,
    min_pt: int,
    step_pt: int,
    line_spacing: float,
    config,
) -> tuple[ImageFont.ImageFont, list[str], int, bool]:
    """Fits text using the overflow ladder sizing step-down technique."""
    padding = getattr(config, "TEXT_PADDING_PX", 4)
    target_w = max(5, bbox_w - 2 * padding)
    target_h = max(5, bbox_h - 2 * padding)

    for pt in range(max_pt, min_pt - 1, -step_pt):
        font = _load_font(font_path, pt, config)
        lines = _wrap_text(text, font, target_w)
        h = _block_height(lines, font, line_spacing)
        if h <= target_h:
            return font, lines, pt, False

    # Fallback to min_pt and flag overflow
    font = _load_font(font_path, min_pt, config)
    lines = _wrap_text(text, font, target_w)
    return font, lines, min_pt, True


def _render_region(
    region: dict,
    page_img: Image.Image,
    draw: ImageDraw.ImageDraw,
    page_num: int,
    config,
) -> tuple[dict, int, int, int]:
    """Render a single region and return (result_dict, page_drawn, page_left, total_overflow)."""
    rid = region["region_id"]
    bbox = region["bbox"]
    render_flag = region["render"]
    has_usable = region["has_usable_text"]
    final_text = region["final_text"]
    style_hint = region["style_hint"]
    register = region["register"]
    glossary_conflict = region["glossary_conflict"]

    font_size = None
    lines_count = 0
    overflow = False
    page_drawn = 0
    page_left = 0
    total_overflow = 0

    # Text-Gated Erase verification
    if not render_flag:
        action = "left_original_not_render_type"
        rendered = False
        page_left = 1
    elif not has_usable:
        action = "left_original_no_text"
        rendered = False
        page_left = 1
        logger.info(
            "[%d/%d %s] Page %d %s -> left original (no usable text)",
            _STAGE_INDEX,
            _TOTAL_STAGES,
            _STAGE_NAME,
            page_num,
            rid,
        )
    elif not final_text:
        action = "missing_text"
        rendered = False
        page_left = 1
        logger.warning(
            "[%d/%d %s] Page %d %s -> left original (missing final_text)",
            _STAGE_INDEX,
            _TOTAL_STAGES,
            _STAGE_NAME,
            page_num,
            rid,
        )
    else:
        action = "drew"
        rendered = True
        page_drawn = 1

        # 1. Erase original text area
        mask_coords = region["erase_mask"]["coords"]
        mx = max(0, mask_coords[0])
        my = max(0, mask_coords[1])
        mw = min(page_img.width - mx, mask_coords[2])
        mh = min(page_img.height - my, mask_coords[3])

        bbox_img = page_img.crop((mx, my, mx + mw, my + mh))
        mask = _get_bubble_mask(bbox_img, config)
        white_img = Image.new("RGB", (mw, mh), (255, 255, 255))
        page_img.paste(white_img, (mx, my), mask=mask)

        # 2. Draw English final text
        text_to_draw = final_text
        if (register == "rude" or style_hint == "spiky") and getattr(
            config, "EMPHASIS_UPPERCASE", True
        ):
            text_to_draw = final_text.upper()

        max_pt = getattr(config, "FONT_MAX_PT", 18)
        min_pt = getattr(config, "FONT_MIN_PT", 9)
        step_pt = getattr(config, "FONT_STEP_PT", 1)
        line_spacing = getattr(config, "LINE_SPACING", 1.15)
        font_path = getattr(config, "FONT_PATH", "assets/fonts/ComicNeue-Regular.ttf")

        font, lines, font_size, overflow = _fit_text(
            text_to_draw,
            bbox["w"],
            bbox["h"],
            font_path,
            max_pt,
            min_pt,
            step_pt,
            line_spacing,
            config,
        )

        if overflow:
            total_overflow = 1
            logger.warning(
                "[%d/%d %s] Page %d %s -> OVERFLOW at %dpt, rendered clipped WARN",
                _STAGE_INDEX,
                _TOTAL_STAGES,
                _STAGE_NAME,
                page_num,
                rid,
                font_size,
            )
        else:
            logger.info(
                "[%d/%d %s] Page %d %s -> drew (%dpt, %d lines)",
                _STAGE_INDEX,
                _TOTAL_STAGES,
                _STAGE_NAME,
                page_num,
                rid,
                font_size,
                len(lines),
            )

        block_h = _block_height(lines, font, line_spacing)
        start_y = bbox["y"] + (bbox["h"] - block_h) // 2

        metrics_bbox = font.getbbox("Hg")
        line_h = metrics_bbox[3] - metrics_bbox[1] if metrics_bbox else 10

        for line in lines:
            line_w = font.getlength(line)
            start_x = bbox["x"] + (bbox["w"] - line_w) // 2
            draw.text((start_x, start_y), line, font=font, fill=(0, 0, 0))
            start_y += int(line_h * line_spacing)

        lines_count = len(lines)

    result_dict = {
        "region_id": rid,
        "page_number": page_num,
        "rendered": rendered,
        "action": action,
        "font_size_pt": font_size,
        "lines": lines_count,
        "overflow": overflow,
        "overflow_ask_llm": False,
        "register": register,
        "style_hint": style_hint,
        "glossary_conflict": glossary_conflict,
    }
    return result_dict, page_drawn, page_left, total_overflow


def _load_and_merge_regions(
    det_data: dict, ocr_data: dict, para_data: dict
) -> dict[int, list[dict]]:
    """Join detection, OCR, and paraphrase data by region_id and group by page."""
    ocr_map = {r["region_id"]: r for r in ocr_data.get("results", [])}
    para_map = {r["region_id"]: r for r in para_data.get("results", [])}

    regions_by_page = {}
    for region in det_data.get("regions", []):
        rid = region["region_id"]
        page_num = region["page_number"]
        ocr_entry = ocr_map.get(rid, {})
        para_entry = para_map.get(rid, {})

        merged = {
            "region_id": rid,
            "page_number": page_num,
            "bbox": region["bbox"],
            "erase_mask": region.get("erase_mask")
            or {
                "type": "rect",
                "coords": [
                    region["bbox"]["x"],
                    region["bbox"]["y"],
                    region["bbox"]["w"],
                    region["bbox"]["h"],
                ],
            },
            "render": region.get("render", True),
            "style_hint": region.get("style_hint") or "round",
            "has_usable_text": ocr_entry.get("has_usable_text", False),
            "final_text": para_entry.get("final_text") or "",
            "register": para_entry.get("register") or "neutral",
            "glossary_conflict": para_entry.get("glossary_conflict") or False,
        }
        regions_by_page.setdefault(page_num, []).append(merged)
    return regions_by_page


def _render_single_page(
    page: dict,
    ws: Path,
    regions_by_page: dict,
    render_dir: Path,
    config,
) -> tuple[list[dict], dict | None, int, int, int]:
    """Render regions on a single page, save output image, and return report components."""
    page_num = page["page_number"]
    filename = page["filename"]
    if page.get("skip") or not filename:
        logger.info(
            "[%d/%d %s] Skipping skipped/failed Page %d",
            _STAGE_INDEX,
            _TOTAL_STAGES,
            _STAGE_NAME,
            page_num,
        )
        return [], None, 0, 0, 0

    orig_path = ws / "pages" / filename
    if not orig_path.exists():
        logger.warning(
            "[%d/%d %s] Page %d file not found: %s",
            _STAGE_INDEX,
            _TOTAL_STAGES,
            _STAGE_NAME,
            page_num,
            orig_path,
        )
        return [], None, 0, 0, 0

    # Load page image
    try:
        page_img = Image.open(orig_path).convert("RGB")
    except Exception as exc:
        logger.error(
            "[%d/%d %s] Failed to load Page %d: %s",
            _STAGE_INDEX,
            _TOTAL_STAGES,
            _STAGE_NAME,
            page_num,
            exc,
        )
        return [], None, 0, 0, 0

    draw = ImageDraw.Draw(page_img)
    page_drawn = 0
    page_left = 0
    page_overflow = 0
    page_results = []
    page_regions = regions_by_page.get(page_num, [])

    for region in page_regions:
        res_dict, drawn, left, ovf = _render_region(
            region, page_img, draw, page_num, config
        )
        page_results.append(res_dict)
        page_drawn += drawn
        page_left += left
        page_overflow += ovf

    # Save output image
    out_filename = f"{page_num:03d}_render.png"
    out_path = render_dir / out_filename
    page_img.save(out_path)
    logger.info(
        "[%d/%d %s] Page %d -> saved %s (%d drawn, %d left)",
        _STAGE_INDEX,
        _TOTAL_STAGES,
        _STAGE_NAME,
        page_num,
        out_path.relative_to(ws),
        page_drawn,
        page_left,
    )

    page_report = {
        "page_number": page_num,
        "output_file": str(out_path.relative_to(ws)),
        "regions_drawn": page_drawn,
        "regions_left": page_left,
    }
    return page_results, page_report, page_drawn, page_left, page_overflow


def run_render(workspace: str, config) -> Path:
    """Run the Rendering stage."""
    t0 = time.monotonic()
    ws = Path(workspace)
    log_stage(logger, _STAGE_INDEX, _TOTAL_STAGES, _STAGE_NAME, "starting")

    manifest = load_manifest(workspace, config)
    if not manifest:
        raise ValueError("Manifest not found. Run import first.")

    # Check dependencies
    det_path = ws / config.STAGE_FOLDERS["detection"] / "detection.json"
    if not det_path.exists():
        raise FileNotFoundError("detection.json not found. Run detect first.")

    ocr_path = ws / config.STAGE_FOLDERS["ocr"] / "ocr.json"
    if not ocr_path.exists():
        raise FileNotFoundError("ocr.json not found. Run ocr first.")

    para_path = ws / config.STAGE_FOLDERS["paraphrase"] / "paraphrase.json"
    if not para_path.exists():
        raise FileNotFoundError("paraphrase.json not found. Run paraphrase first.")

    with det_path.open("r", encoding="utf-8") as fh:
        det_data = json.load(fh)
    with ocr_path.open("r", encoding="utf-8") as fh:
        ocr_data = json.load(fh)
    with para_path.open("r", encoding="utf-8") as fh:
        para_data = json.load(fh)

    logger.info(
        "[%d/%d %s] Font: %s",
        _STAGE_INDEX,
        _TOTAL_STAGES,
        _STAGE_NAME,
        getattr(config, "FONT_PATH", "assets/fonts/ComicNeue-Regular.ttf"),
    )

    # Joins
    regions_by_page = _load_and_merge_regions(det_data, ocr_data, para_data)

    # Render pages
    render_results = []
    page_reports = []
    total_drawn = 0
    total_left = 0
    total_overflow = 0

    render_dir = ws / config.STAGE_FOLDERS["render"]
    render_dir.mkdir(parents=True, exist_ok=True)

    for page in manifest.get("pages", []):
        results, report, drawn, left, ovf = _render_single_page(
            page, ws, regions_by_page, render_dir, config
        )
        if report:
            render_results.extend(results)
            page_reports.append(report)
            total_drawn += drawn
            total_left += left
            total_overflow += ovf

    now = datetime.now(timezone.utc).isoformat()
    output_report = {
        "chapter_id": manifest.get("chapter_id", "unknown"),
        "stage": "render",
        "generated_at": now,
        "font": getattr(config, "FONT_PATH", "assets/fonts/ComicNeue-Regular.ttf"),
        "results": render_results,
        "pages": page_reports,
    }

    report_path = render_dir / "render.json"
    with report_path.open("w", encoding="utf-8") as fh:
        json.dump(output_report, fh, ensure_ascii=False, indent=2)

    # Advance manifest
    completed = manifest.get("completed_stages", [])
    if "render" not in completed:
        completed.append("render")
    manifest["completed_stages"] = completed
    manifest["current_stage"] = "qa"
    manifest["updated_at"] = now
    save_manifest(ws, config, manifest)

    elapsed = time.monotonic() - t0
    log_stage(
        logger,
        _STAGE_INDEX,
        _TOTAL_STAGES,
        _STAGE_NAME,
        f"done: {len(page_reports)} pages, {total_drawn} drawn, {total_left} left, "
        f"{total_overflow} overflow -> {report_path} (elapsed {elapsed:.1f}s)",
    )
    return report_path
