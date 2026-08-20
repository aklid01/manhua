# Typography layout, font sizing, word wrapping, and mask clamping.

import re
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

from manhua_pipeline.logging_setup import get_logger
from manhua_pipeline.stages.stage5_render.types import Box

logger = get_logger(__name__)

_font_cache: dict = {}
_ESCAPED_NEWLINE_RE = re.compile(r"\\(?:r\\n|n|r)", re.IGNORECASE)


# Load cached FreeType font at specified point size
def _load_font(
    font_path: str, pt: int, config
) -> ImageFont.ImageFont | ImageFont.FreeTypeFont:
    key = (font_path, pt)
    if key in _font_cache:
        return _font_cache[key]

    p = Path(font_path)
    if not p.exists():
        if getattr(config, "FONT_MISSING_HARD_ERROR", True):
            raise FileNotFoundError(
                f"Font file not found at {font_path}. Place {Path(font_path).name} "
                "in assets/fonts/ (bundled with the repo) or set "
                "FONT_MISSING_HARD_ERROR = False in config.py to use the PIL default font."
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


# Calculate maximum font point size dynamically
def effective_max_pt(config, page_width: int | None) -> int:
    static_max = int(getattr(config, "FONT_MAX_PT", 40))
    frac = float(getattr(config, "FONT_MAX_PAGE_FRAC", 0.0) or 0.0)
    if page_width and frac > 0:
        return max(static_max, int(page_width * frac))
    return static_max


# Replace escaped newline characters with standard newline
def normalize_escaped_newlines(text: str) -> str:
    if not text:
        return text
    return _ESCAPED_NEWLINE_RE.sub("\n", text)


# Wrap paragraphs into lines fitting within maximum width
def _wrap_text(text: str, font: ImageFont.ImageFont, max_width: int) -> list[str]:
    text = normalize_escaped_newlines(text)
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
            left, _top, right, _bottom = font.getbbox(test_line)
            width = max(0, right - left)
            if width <= max_width or not current_line:
                current_line.append(word)
            else:
                all_lines.append(" ".join(current_line))
                current_line = [word]
        if current_line:
            all_lines.append(" ".join(current_line))

    return all_lines


# Calculate total vertical pixel height of wrapped text lines
def _block_height(
    lines: list[str], font: ImageFont.ImageFont, line_spacing: float
) -> int:
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


# Measure rendered ink pixel width of single text line
def _line_ink_width(line: str, font: ImageFont.ImageFont) -> int:
    left, _top, right, _bottom = font.getbbox(line)
    return max(0, right - left)


# Determine mask horizontal span at given page Y coordinate
def _get_available_width_at_y(
    eroded_mask: Image.Image,
    bubble_box: Box,
    sample_y_page: int,
) -> int:
    mask_w, mask_h = eroded_mask.size
    mask_y = max(0, min(mask_h - 1, sample_y_page - bubble_box.y))
    mask_pixels = eroded_mask.load()
    threshold = 128

    left_mask = None
    right_mask = None
    for mx in range(mask_w):
        if mask_pixels[mx, mask_y] >= threshold:
            if left_mask is None:
                left_mask = mx
            right_mask = mx

    if left_mask is None:
        return 0

    left_page = bubble_box.x + left_mask
    right_page = bubble_box.x + right_mask + 1
    return max(0, right_page - left_page)


# Check if any text line overflows target box or eroded bubble mask
def _any_line_overflows(
    lines: list[str],
    font: ImageFont.ImageFont,
    target_w: int,
    layout_box: Box | None = None,
    eroded_mask: Image.Image | None = None,
    bubble_box: Box | None = None,
    line_spacing: float = 1.15,
    draw: ImageDraw.ImageDraw | None = None,
) -> bool:
    if any(_line_ink_width(line, font) > target_w for line in lines):
        return True

    if eroded_mask is not None and bubble_box is not None and layout_box is not None:
        font_size = getattr(font, "size", 12)
        line_heights = []
        line_widths = []
        for line in lines:
            if draw is not None:
                left, top, right, bottom = draw.textbbox((0, 0), line, font=font)
                line_widths.append(right - left)
                line_heights.append(bottom - top)
            else:
                w = _line_ink_width(line, font)
                bbox = font.getbbox("Hg")
                h = bbox[3] - bbox[1] if bbox else font_size
                line_widths.append(w)
                line_heights.append(h)

        spacing_px = max(1, int(font_size * (line_spacing - 1.0)))
        block_height = sum(line_heights)
        if len(lines) > 1:
            block_height += spacing_px * (len(lines) - 1)

        cursor_y = layout_box.y + max(0, (layout_box.h - block_height) // 2)

        for line_w, line_h in zip(line_widths, line_heights):
            sample_y_page = cursor_y + line_h // 2
            avail_w = _get_available_width_at_y(eroded_mask, bubble_box, sample_y_page)
            if line_w > avail_w:
                return True
            cursor_y += line_h + spacing_px

    return False


# Iteratively fit text within bounding box by adjusting font size and line wrapping
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
    use_padding: bool = True,
    layout_box: Box | None = None,
    eroded_mask: Image.Image | None = None,
    bubble_box: Box | None = None,
    draw: ImageDraw.ImageDraw | None = None,
) -> tuple[ImageFont.ImageFont, list[str], int, bool]:
    padding = getattr(config, "TEXT_PADDING_PX", 8) if use_padding else 0
    target_w = max(5, bbox_w - 2 * padding)
    target_h = max(5, bbox_h - 2 * padding)

    abs_min = int(getattr(config, "FONT_ABS_MIN_PT", 9))
    readable_min = int(getattr(config, "FONT_MIN_PT", 12))
    hard_floor = max(1, min(abs_min, readable_min))

    for pt in range(max_pt, hard_floor - 1, -step_pt):
        font = _load_font(font_path, pt, config)

        for ratio in [0.65, 0.8, 0.95]:
            w_limit = max(5, int(target_w * ratio))
            lines = _wrap_text(text, font, w_limit)
            h = _block_height(lines, font, line_spacing)
            if h <= target_h and not _any_line_overflows(
                lines,
                font,
                target_w,
                layout_box=layout_box,
                eroded_mask=eroded_mask,
                bubble_box=bubble_box,
                line_spacing=line_spacing,
                draw=draw,
            ):
                overflow_flag = pt < readable_min
                return font, lines, pt, overflow_flag

    font = _load_font(font_path, hard_floor, config)
    lines = _wrap_text(text, font, target_w)
    return font, lines, hard_floor, True


# Calculate best font size, wrapped lines, and line placement coordinates
def fit_and_place_text(
    *,
    draw: ImageDraw.ImageDraw,
    final_text: str,
    font_path: str,
    layout_box: Box,
    config,
    page_width: int | None = None,
    bubble_mask: Image.Image | None = None,
    bubble_box: Box | None = None,
    pre_eroded_mask: Image.Image | None = None,
) -> tuple[ImageFont.ImageFont, list[str], int, bool, list[dict]]:
    resolved_max_pt = effective_max_pt(config, page_width)
    min_pt = getattr(config, "FONT_MIN_PT", 9)
    step_pt = getattr(config, "FONT_STEP_PT", 1)
    line_spacing = getattr(config, "LINE_SPACING", 1.15)

    if pre_eroded_mask is not None:
        eroded_mask = pre_eroded_mask
    elif bubble_mask is not None:
        erode_px = int(getattr(config, "LAYOUT_INTERIOR_ERODE_PX", 10))
        if erode_px > 0:
            k = erode_px * 2 + 1
            eroded_mask = bubble_mask.filter(ImageFilter.MinFilter(k))
        else:
            eroded_mask = bubble_mask
    else:
        eroded_mask = None

    font, lines, font_size, overflow = _fit_text(
        text=final_text,
        bbox_w=layout_box.w,
        bbox_h=layout_box.h,
        font_path=font_path,
        max_pt=resolved_max_pt,
        min_pt=min_pt,
        step_pt=step_pt,
        line_spacing=line_spacing,
        config=config,
        use_padding=False,
        layout_box=layout_box,
        eroded_mask=eroded_mask,
        bubble_box=bubble_box,
        draw=draw,
    )

    line_heights = []
    line_widths = []
    for line in lines:
        left, top, right, bottom = draw.textbbox((0, 0), line, font=font)
        line_widths.append(right - left)
        line_heights.append(bottom - top)

    spacing_px = max(1, int(font_size * (line_spacing - 1.0)))
    block_height = sum(line_heights)
    if len(lines) > 1:
        block_height += spacing_px * (len(lines) - 1)

    cursor_y = layout_box.y + max(0, (layout_box.h - block_height) // 2)
    placements = []

    for line, width, height in zip(lines, line_widths, line_heights):
        cursor_x = layout_box.x + max(0, (layout_box.w - width) // 2)
        cursor_x = max(
            layout_box.x, min(cursor_x, max(layout_box.x, layout_box.x2 - width))
        )
        cursor_y_clamped = max(
            layout_box.y, min(cursor_y, max(layout_box.y, layout_box.y2 - height))
        )
        placements.append(
            {
                "text": line,
                "x": cursor_x,
                "y": cursor_y_clamped,
                "w": width,
                "h": height,
            }
        )
        cursor_y += height + spacing_px

    return font, lines, font_size, overflow, placements


# Clamp line horizontal positions within interior bubble mask boundaries
def _clamp_placements_to_mask(
    placements: list[dict],
    bubble_mask: Image.Image,
    bubble_box: Box,
    erode_px: int = 10,
    pre_eroded_mask: Image.Image | None = None,
) -> list[dict]:
    if pre_eroded_mask is not None:
        mask = pre_eroded_mask
    elif erode_px > 0:
        k = erode_px * 2 + 1
        mask = bubble_mask.filter(ImageFilter.MinFilter(k))
    else:
        mask = bubble_mask

    mask_w, mask_h = mask.size
    mask_pixels = mask.load()
    threshold = 128

    result = []
    for p in placements:
        sample_y_page = p["y"] + p["h"] // 2
        mask_y = max(0, min(mask_h - 1, sample_y_page - bubble_box.y))

        left_mask = None
        right_mask = None
        for mx in range(mask_w):
            if mask_pixels[mx, mask_y] >= threshold:
                if left_mask is None:
                    left_mask = mx
                right_mask = mx

        if left_mask is None:
            result.append(p)
            continue

        left_page = bubble_box.x + left_mask
        right_page = bubble_box.x + right_mask + 1
        available_w = right_page - left_page
        line_w = p["w"]

        if available_w >= line_w:
            max_x = max(left_page, right_page - line_w)
            new_x = max(left_page, min(p["x"], max_x))
        else:
            new_x = left_page

        result.append({**p, "x": new_x})

    return result
