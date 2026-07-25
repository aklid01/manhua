"""Stage 5: Rendering.

Erases original Chinese text from bounding box areas and draws the paraphrased
US English text in its place. Utilizes Pillow to wrap, size, and style.
"""

import json
import random
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import median

from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont

from manhua_pipeline.io.settings import get_credits
from manhua_pipeline.io.workspace import load_manifest, save_manifest
from manhua_pipeline.logging_setup import get_logger, log_stage

logger = get_logger(__name__)

_STAGE_INDEX = 5
_TOTAL_STAGES = 7
_STAGE_NAME = "Render"

_font_cache = {}


@dataclass(frozen=True)
class Box:
    x: int
    y: int
    w: int
    h: int

    @property
    def x2(self) -> int:
        return self.x + self.w

    @property
    def y2(self) -> int:
        return self.y + self.h

    def as_xyxy(self) -> tuple[int, int, int, int]:
        return self.x, self.y, self.x2, self.y2

    def as_dict(self) -> dict[str, int]:
        return {"x": self.x, "y": self.y, "w": self.w, "h": self.h}


def box_from_dict(value: dict) -> Box:
    return Box(
        x=int(value["x"]),
        y=int(value["y"]),
        w=max(0, int(value["w"])),
        h=max(0, int(value["h"])),
    )


def box_from_coords(coords: list | tuple) -> Box:
    x, y, w, h = coords
    return Box(int(x), int(y), max(0, int(w)), max(0, int(h)))


def clip_box(box: Box, image_size: tuple[int, int]) -> Box:
    image_w, image_h = image_size
    x0 = max(0, min(image_w, box.x))
    y0 = max(0, min(image_h, box.y))
    x1 = max(x0, min(image_w, box.x2))
    y1 = max(y0, min(image_h, box.y2))
    return Box(x0, y0, x1 - x0, y1 - y0)


def expand_box(box: Box, padding: int, image_size: tuple[int, int]) -> Box:
    return clip_box(
        Box(
            x=box.x - padding,
            y=box.y - padding,
            w=box.w + 2 * padding,
            h=box.h + 2 * padding,
        ),
        image_size,
    )


def box_inside(inner: Box, outer: Box, *, margin: int = 0) -> bool:
    return (
        inner.x >= outer.x + margin
        and inner.y >= outer.y + margin
        and inner.x2 <= outer.x2 - margin
        and inner.y2 <= outer.y2 - margin
    )


def intersect_boxes(a: Box, b: Box) -> Box | None:
    x0 = max(a.x, b.x)
    y0 = max(a.y, b.y)
    x1 = min(a.x2, b.x2)
    y1 = min(a.y2, b.y2)
    if x1 <= x0 or y1 <= y0:
        return None
    return Box(x0, y0, x1 - x0, y1 - y0)


def bubble_interior_box(
    bubble_mask: Image.Image,
    bubble_box: Box,
    erode_px: int,
) -> Box | None:
    """Return an eroded interior bbox of the bubble mask, in page coords.

    Erosion pulls the usable region away from the curved rim and the
    speech tail so expanded English text stays visually centered in the
    body of the balloon.
    """
    mask = bubble_mask
    if erode_px and erode_px > 0:
        k = erode_px * 2 + 1  # odd kernel for MinFilter erosion
        mask = mask.filter(ImageFilter.MinFilter(k))

    box = mask.getbbox()
    if box is None:
        return None

    x0, y0, x1, y1 = box
    return Box(
        x=bubble_box.x + x0,
        y=bubble_box.y + y0,
        w=max(1, x1 - x0),
        h=max(1, y1 - y0),
    )


def grow_layout_box(
    text_box: Box,
    interior_box: Box | None,
    image_size: tuple[int, int],
    config,
) -> Box:
    """Expand the tight text box toward the eroded bubble interior,
    bounded by a growth cap and clipped to the interior and image.
    """
    layout_padding = int(getattr(config, "TEXT_LAYOUT_PADDING_PX", 8))

    base = Box(
        x=text_box.x + layout_padding,
        y=text_box.y + layout_padding,
        w=max(5, text_box.w - 2 * layout_padding),
        h=max(5, text_box.h - 2 * layout_padding),
    )

    if not getattr(config, "LAYOUT_EXPAND_ENABLED", True):
        return base
    if interior_box is None:
        return base

    # Candidate = union of base and interior, then clip to interior.
    union = Box(
        x=min(base.x, interior_box.x),
        y=min(base.y, interior_box.y),
        w=max(base.x2, interior_box.x2) - min(base.x, interior_box.x),
        h=max(base.y2, interior_box.y2) - min(base.y, interior_box.y),
    )
    candidate = intersect_boxes(union, interior_box) or base

    # Enforce growth cap vs. the original text area.
    max_factor = float(getattr(config, "LAYOUT_MAX_GROWTH_FACTOR", 3.0))
    text_area = max(1, text_box.w * text_box.h)
    if (candidate.w * candidate.h) > max_factor * text_area:
        candidate = _shrink_box_around_center(
            candidate,
            target_area=int(max_factor * text_area),
            anchor=text_box,
        )

    # Ensure the expanded box still overlaps the original text region.
    min_overlap = float(getattr(config, "LAYOUT_MIN_ANCHOR_OVERLAP", 0.5))
    overlap = intersect_boxes(candidate, text_box)
    overlap_ratio = (
        (overlap.w * overlap.h) / max(1, text_box.w * text_box.h) if overlap else 0.0
    )
    if overlap_ratio < min_overlap:
        return base  # anchor safety: do not drift off the source text

    return clip_box(candidate, image_size)


def _shrink_box_around_center(box: Box, target_area: int, anchor: Box) -> Box:
    """Shrink a box toward the anchor center to respect an area cap."""
    if box.w * box.h <= target_area:
        return box
    import math

    scale = math.sqrt(target_area / float(box.w * box.h))
    new_w = max(5, int(box.w * scale))
    new_h = max(5, int(box.h * scale))
    cx = anchor.x + anchor.w // 2
    cy = anchor.y + anchor.h // 2
    return Box(
        x=cx - new_w // 2,
        y=cy - new_h // 2,
        w=new_w,
        h=new_h,
    )


def select_render_geometry(
    region: dict,
    image_size: tuple[int, int],
    config,
    *,
    bubble_mask=None,
    bubble_box: Box | None = None,
) -> tuple[Box, Box, Box, Box]:
    text_value = region.get("read_region") or region.get("bbox")
    if not text_value:
        raise ValueError(f"Region {region.get('region_id')} has no read_region or bbox")

    text_box = clip_box(box_from_dict(text_value), image_size)

    parent_value = region.get("parent_bubble")
    if bubble_box is None:
        if parent_value:
            bubble_box = clip_box(box_from_dict(parent_value), image_size)
        else:
            erase_coords = region.get("erase_mask", {}).get("coords")
            bubble_box = (
                clip_box(box_from_coords(erase_coords), image_size)
                if erase_coords
                else text_box
            )

    erase_box = expand_box(
        text_box,
        int(getattr(config, "TEXT_ERASE_PADDING_PX", 12)),
        image_size,
    )
    if parent_value:
        erase_box = intersect_boxes(erase_box, bubble_box) or text_box

    interior_box = None
    if bubble_mask is not None:
        interior_box = bubble_interior_box(
            bubble_mask,
            bubble_box,
            int(getattr(config, "LAYOUT_INTERIOR_ERODE_PX", 10)),
        )

    layout_box = grow_layout_box(text_box, interior_box, image_size, config)

    return text_box, bubble_box, erase_box, layout_box


def estimate_text_background(
    image: Image.Image,
    text_box: Box,
    bubble_box: Box,
) -> tuple[int, int, int]:
    pad = max(6, min(text_box.w, text_box.h) // 10)
    expanded = Box(
        text_box.x - pad,
        text_box.y - pad,
        text_box.w + 2 * pad,
        text_box.h + 2 * pad,
    )
    ring_box = (
        intersect_boxes(
            clip_box(expanded, image.size),
            bubble_box,
        )
        or text_box
    )

    points: list[tuple[int, int, int]] = []
    step_x = max(1, ring_box.w // 24)
    step_y = max(1, ring_box.h // 24)

    for y in range(ring_box.y, ring_box.y2, step_y):
        for x in range(ring_box.x, ring_box.x2, step_x):
            if text_box.x <= x < text_box.x2 and text_box.y <= y < text_box.y2:
                continue
            pixel = image.getpixel((x, y))
            if isinstance(pixel, int):
                pixel = (pixel, pixel, pixel)
            points.append(tuple(pixel[:3]))

    if not points:
        return (255, 255, 255)

    return tuple(int(median(channel)) for channel in zip(*points))


def _count_mask_pixels(mask: Image.Image) -> int:
    histogram = mask.histogram()
    return histogram[255] if len(histogram) > 255 else 0


def erase_original_text(
    *,
    page_img: Image.Image,
    draw: ImageDraw.ImageDraw,
    text_box: Box,
    bubble_box: Box,
    erase_box: Box,
    bubble_mask: Image.Image,
    config,
) -> dict:
    fill = estimate_text_background(
        page_img,
        text_box,
        bubble_box,
    )

    local_erase = Box(
        x=erase_box.x - bubble_box.x,
        y=erase_box.y - bubble_box.y,
        w=erase_box.w,
        h=erase_box.h,
    )

    text_mask = Image.new("L", (bubble_box.w, bubble_box.h), 0)
    mask_draw = ImageDraw.Draw(text_mask)
    mask_draw.rectangle(local_erase.as_xyxy(), fill=255)

    dilation = int(getattr(config, "TEXT_ERASE_DILATION_PX", 5))
    if dilation > 1:
        if dilation % 2 == 0:
            dilation += 1
        text_mask = text_mask.filter(ImageFilter.MaxFilter(dilation))

    # bubble_mask is the filled interior mask.
    # Multiply clips rectangular corners outside curved balloons.
    erase_mask = ImageChops.multiply(text_mask, bubble_mask)

    fill_image = Image.new("RGB", (bubble_box.w, bubble_box.h), fill)
    page_img.paste(
        fill_image,
        (bubble_box.x, bubble_box.y),
        mask=erase_mask,
    )

    text_pixels = _count_mask_pixels(text_mask)
    erase_pixels = _count_mask_pixels(erase_mask)
    retention = (erase_pixels / text_pixels) if text_pixels else 0.0
    min_retention = float(getattr(config, "TEXT_ERASE_MIN_RETENTION", 0.65))
    low_overlap = retention < min_retention

    margin = int(getattr(config, "TEXT_ERASE_PARENT_MARGIN_PX", 8))
    geometrically_inside = box_inside(text_box, bubble_box, margin=margin)

    return {
        "erase_strategy": "text_mask_clipped_to_bubble",
        "erase_fill": list(fill),
        "dilation_px": dilation,
        "text_mask_pixels": text_pixels,
        "erase_mask_pixels": erase_pixels,
        "clip_retention_ratio": retention,
        "low_overlap": low_overlap,
        "safe_rect": geometrically_inside,
        "text_mask": text_mask,
        "erase_mask": erase_mask,
    }


def effective_max_pt(config, page_width: int | None) -> int:
    """Ceiling for font sizing, scaled to page WIDTH.

    Static FONT_MAX_PT is a floor-of-the-max so narrow pages keep a readable
    size. Width is used (not height) because stitched canvases and webtoon
    strips are tall but keep a normal width, so width is the stable reading
    unit across merged pages and strip formats.
    """
    static_max = int(getattr(config, "FONT_MAX_PT", 40))
    frac = float(getattr(config, "FONT_MAX_PAGE_FRAC", 0.0) or 0.0)
    if page_width and frac > 0:
        return max(static_max, int(page_width * frac))
    return static_max


def fit_and_place_text(
    *,
    draw: ImageDraw.ImageDraw,
    final_text: str,
    font_path: str,
    layout_box: Box,
    config,
    page_width: int | None = None,
) -> tuple[ImageFont.ImageFont, list[str], int, bool, list[dict]]:
    resolved_max_pt = effective_max_pt(config, page_width)
    min_pt = getattr(config, "FONT_MIN_PT", 9)
    step_pt = getattr(config, "FONT_STEP_PT", 1)
    line_spacing = getattr(config, "LINE_SPACING", 1.15)

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
        placements.append(
            {
                "text": line,
                "x": cursor_x,
                "y": cursor_y,
                "w": width,
                "h": height,
            }
        )
        cursor_y += height + spacing_px

    return font, lines, font_size, overflow, placements


def should_debug_region(region_id: str, config) -> bool:
    if not getattr(config, "RENDER_DEBUG_ARTIFACTS", False):
        return False
    selected = set(getattr(config, "RENDER_DEBUG_REGION_IDS", []) or [])
    return not selected or region_id in selected


def save_render_debug_image(
    image: Image.Image,
    *,
    debug_dir: Path,
    region_id: str,
    suffix: str,
) -> Path:
    debug_dir.mkdir(parents=True, exist_ok=True)
    path = debug_dir / f"{region_id}_{suffix}.png"
    image.save(path)
    return path


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


def _text_color_for_bg(bg: tuple[int, int, int]) -> tuple[int, int, int]:
    """Black text on light backgrounds, white text on dark - WCAG-style luminance."""
    r, g, b = bg[:3]
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    return (0, 0, 0) if luminance >= 128 else (255, 255, 255)


def _bfs(
    sx: int,
    sy: int,
    is_white: list[list[bool]],
    visited: list[list[bool]],
    width: int,
    height: int,
) -> list[tuple[int, int]]:
    comp = []
    queue = [(sx, sy)]
    visited[sy][sx] = True
    q_idx = 0
    while q_idx < len(queue):
        cx, cy = queue[q_idx]
        q_idx += 1
        comp.append((cx, cy))
        for nx, ny in [(cx - 1, cy), (cx + 1, cy), (cx, cy - 1), (cx, cy + 1)]:
            if 0 <= nx < width and 0 <= ny < height:
                if is_white[ny][nx] and not visited[ny][nx]:
                    visited[ny][nx] = True
                    queue.append((nx, ny))
    return comp


def _keep_largest_component(
    is_white: list[list[bool]],
    width: int,
    height: int,
    seed_box: Box | None = None,
) -> list[list[bool]]:
    """Keep component with highest seed_box overlap or largest area."""
    visited = [[False] * width for _ in range(height)]
    components = []

    for y in range(height):
        for x in range(width):
            if is_white[y][x] and not visited[y][x]:
                comp = _bfs(x, y, is_white, visited, width, height)
                components.append(comp)

    if not components:
        return [[False] * width for _ in range(height)]

    if seed_box is not None:
        margin = 8
        sx0 = max(0, seed_box.x - margin)
        sy0 = max(0, seed_box.y - margin)
        sx1 = min(width, seed_box.x2 + margin)
        sy1 = min(height, seed_box.y2 + margin)

        best_comp = None
        best_score = -1
        for comp in components:
            score = sum(1 for cx, cy in comp if sx0 <= cx < sx1 and sy0 <= cy < sy1)
            if score > best_score:
                best_score = score
                best_comp = comp

        if best_score > 0 and best_comp is not None:
            chosen = best_comp
        else:
            chosen = max(components, key=len)
    else:
        chosen = max(components, key=len)

    new_grid = [[False] * width for _ in range(height)]
    for cx, cy in chosen:
        new_grid[cy][cx] = True
    return new_grid


def _get_bubble_mask(
    bbox_img: Image.Image,
    mx: int,
    my: int,
    mw: int,
    mh: int,
    page_w: int,
    page_h: int,
    config,
    seed_box: Box | None = None,
) -> Image.Image:
    """Generate a binary mask identifying the white/near-white bubble area."""
    import numpy as np

    img = bbox_img.convert("RGB")
    width, height = img.size
    threshold = getattr(config, "BUBBLE_WHITE_THRESHOLD", 220)

    # 1. Create a binary grid of white pixels
    arr = np.asarray(img)
    is_white = (
        (arr[:, :, 0] > threshold)
        & (arr[:, :, 1] > threshold)
        & (arr[:, :, 2] > threshold)
    )

    # 2. Select component using seed_box overlap if available, else max area
    try:
        from scipy.ndimage import label

        labeled, num_features = label(is_white)
        if num_features > 0:
            if seed_box is not None:
                margin = 8
                sx0 = max(0, seed_box.x - margin)
                sy0 = max(0, seed_box.y - margin)
                sx1 = min(width, seed_box.x2 + margin)
                sy1 = min(height, seed_box.y2 + margin)

                best_label = 0
                best_score = -1
                for lab in range(1, num_features + 1):
                    comp = labeled == lab
                    score = int(comp[sy0:sy1, sx0:sx1].sum())
                    if score > best_score:
                        best_score = score
                        best_label = lab

                if best_score > 0:
                    selected_label = best_label
                else:
                    counts = np.bincount(labeled.ravel())
                    counts[0] = 0
                    selected_label = counts.argmax()
            else:
                counts = np.bincount(labeled.ravel())
                counts[0] = 0
                selected_label = counts.argmax()

            is_white = labeled == selected_label
        else:
            is_white = np.zeros_like(is_white)
    except ImportError:
        is_white_list = is_white.tolist()
        is_white_list = _keep_largest_component(
            is_white_list, width, height, seed_box=seed_box
        )
        is_white = np.array(is_white_list, dtype=bool)

    # 3. Find boundaries for enclosed-hole fill logic using numpy
    first_w_in_row = np.full(height, -1, dtype=int)
    last_w_in_row = np.full(height, -1, dtype=int)
    first_w_in_col = np.full(width, -1, dtype=int)
    last_w_in_col = np.full(width, -1, dtype=int)

    for y in range(height):
        row_trues = np.where(is_white[y, :])[0]
        if row_trues.size > 0:
            first_w_in_row[y] = row_trues[0]
            last_w_in_row[y] = row_trues[-1]

    for x in range(width):
        col_trues = np.where(is_white[:, x])[0]
        if col_trues.size > 0:
            first_w_in_col[x] = col_trues[0]
            last_w_in_col[x] = col_trues[-1]

    touches_top = (my == 0) and np.any(is_white[0, :])
    touches_bottom = (my + mh == page_h) and np.any(is_white[height - 1, :])
    touches_left = (mx == 0) and np.any(is_white[:, 0])
    touches_right = (mx + mw == page_w) and np.any(is_white[:, width - 1])

    # 4. Fill enclosed holes
    y_coords, x_coords = np.indices((height, width))
    has_left = touches_left | (
        (first_w_in_row[y_coords] != -1) & (first_w_in_row[y_coords] < x_coords)
    )
    has_right = touches_right | (last_w_in_row[y_coords] > x_coords)
    has_top = touches_top | (
        (first_w_in_col[x_coords] != -1) & (first_w_in_col[x_coords] < y_coords)
    )
    has_bottom = touches_bottom | (last_w_in_col[x_coords] > y_coords)

    mask_arr = is_white | (has_left & has_right & has_top & has_bottom)
    return Image.fromarray((mask_arr * 255).astype(np.uint8), mode="L")


def _wrap_text(text: str, font: ImageFont.ImageFont, max_width: int) -> list[str]:
    """Word wraps lines to stay within max_width."""
    text = text.replace("\\r\\n", "\n").replace("\\n", "\n").replace("\\r", "\n")
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


def _any_line_overflows(
    lines: list[str], font: ImageFont.ImageFont, target_w: int
) -> bool:
    return any(font.getlength(line) > target_w for line in lines)


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
) -> tuple[ImageFont.ImageFont, list[str], int, bool]:
    """Fits text using the overflow ladder sizing step-down technique with ratio wrapping search."""
    padding = getattr(config, "TEXT_PADDING_PX", 8) if use_padding else 0
    target_w = max(5, bbox_w - 2 * padding)
    target_h = max(5, bbox_h - 2 * padding)

    abs_min = int(getattr(config, "FONT_ABS_MIN_PT", 9))
    readable_min = int(getattr(config, "FONT_MIN_PT", 12))
    hard_floor = max(1, min(abs_min, readable_min))

    for pt in range(max_pt, hard_floor - 1, -step_pt):
        font = _load_font(font_path, pt, config)

        # Try multiple layout widths to encourage more lines and better aspect ratio
        for ratio in [0.65, 0.8, 0.95]:
            w_limit = max(5, int(target_w * ratio))
            lines = _wrap_text(text, font, w_limit)
            h = _block_height(lines, font, line_spacing)
            if h <= target_h and not _any_line_overflows(lines, font, target_w):
                # overflow means "below the readable target", not "below hard floor"
                overflow_flag = pt < readable_min
                return font, lines, pt, overflow_flag

    # Fallback to hard_floor and full target_w, and flag overflow
    font = _load_font(font_path, hard_floor, config)
    lines = _wrap_text(text, font, target_w)
    return font, lines, hard_floor, True


def _render_region(
    region: dict,
    page_img: Image.Image,
    draw: ImageDraw.ImageDraw,
    page_num: int,
    config,
    *,
    emitter=None,
    run_id: str = "unknown",
    chapter: str = "unknown",
    render_dir: Path | None = None,
) -> tuple[dict, int, int, int]:
    """Render a single region using explicit geometry roles and deterministic erasure."""
    rid = region["region_id"]
    render_flag = region.get("render", True)
    has_usable = region.get("has_usable_text", False)
    final_text = region.get("final_text", "")
    style_hint = region.get("style_hint", "round")
    register = region.get("register", "neutral")
    glossary_conflict = region.get("glossary_conflict", False)

    if not render_flag:
        result_dict = {
            "region_id": rid,
            "page_number": page_num,
            "rendered": False,
            "action": "left_original_not_render_type",
            "font_size_pt": None,
            "lines": 0,
            "overflow": False,
            "overflow_ask_llm": False,
            "register": register,
            "style_hint": style_hint,
            "glossary_conflict": glossary_conflict,
        }
        return result_dict, 0, 1, 0

    if not has_usable:
        logger.info(
            "[%d/%d %s] Page %d %s -> left original (no usable text)",
            _STAGE_INDEX,
            _TOTAL_STAGES,
            _STAGE_NAME,
            page_num,
            rid,
        )
        result_dict = {
            "region_id": rid,
            "page_number": page_num,
            "rendered": False,
            "action": "left_original_no_text",
            "font_size_pt": None,
            "lines": 0,
            "overflow": False,
            "overflow_ask_llm": False,
            "register": register,
            "style_hint": style_hint,
            "glossary_conflict": glossary_conflict,
        }
        return result_dict, 0, 1, 0

    if not final_text:
        logger.warning(
            "[%d/%d %s] Page %d %s -> left original (missing final_text)",
            _STAGE_INDEX,
            _TOTAL_STAGES,
            _STAGE_NAME,
            page_num,
            rid,
        )
        result_dict = {
            "region_id": rid,
            "page_number": page_num,
            "rendered": False,
            "action": "missing_text",
            "font_size_pt": None,
            "lines": 0,
            "overflow": False,
            "overflow_ask_llm": False,
            "register": register,
            "style_hint": style_hint,
            "glossary_conflict": glossary_conflict,
        }
        return result_dict, 0, 1, 0

    # 1. Geometry Selection (Initial parse to get text_box and bubble_box)
    text_box, bubble_box, _, _ = select_render_geometry(region, page_img.size, config)

    # 2. Bubble Mask for context / diagnostics (using local_text_box seed)
    local_text_box = Box(
        x=text_box.x - bubble_box.x,
        y=text_box.y - bubble_box.y,
        w=text_box.w,
        h=text_box.h,
    )
    bubble_crop = page_img.crop(bubble_box.as_xyxy())
    bubble_mask = _get_bubble_mask(
        bubble_crop,
        bubble_box.x,
        bubble_box.y,
        bubble_box.w,
        bubble_box.h,
        page_img.width,
        page_img.height,
        config,
        seed_box=local_text_box,
    )

    # Re-evaluate layout_box and erase_box now that we have the bubble mask
    text_box, bubble_box, erase_box, layout_box = select_render_geometry(
        region,
        page_img.size,
        config,
        bubble_mask=bubble_mask,
        bubble_box=bubble_box,
    )

    debug_dir = (render_dir / "debug") if render_dir else None
    do_debug = debug_dir and should_debug_region(rid, config)
    debug_paths = {}

    if do_debug:
        debug_paths["before"] = str(
            save_render_debug_image(
                page_img.crop(erase_box.as_xyxy()),
                debug_dir=debug_dir,
                region_id=rid,
                suffix="before",
            )
        )
        debug_paths["bubble_crop"] = str(
            save_render_debug_image(
                bubble_crop,
                debug_dir=debug_dir,
                region_id=rid,
                suffix="bubble_crop",
            )
        )
        debug_paths["bubble_mask"] = str(
            save_render_debug_image(
                bubble_mask,
                debug_dir=debug_dir,
                region_id=rid,
                suffix="bubble_mask",
            )
        )

    # 3. Erase Source Text
    erase_meta = erase_original_text(
        page_img=page_img,
        draw=draw,
        text_box=text_box,
        bubble_box=bubble_box,
        erase_box=erase_box,
        bubble_mask=bubble_mask,
        config=config,
    )

    if erase_meta.get("low_overlap"):
        min_retention = float(getattr(config, "TEXT_ERASE_MIN_RETENTION", 0.65))
        logger.warning(
            "[%d/%d %s] Page %d %s -> Low erase mask retention ratio (%.2f < %.2f)",
            _STAGE_INDEX,
            _TOTAL_STAGES,
            _STAGE_NAME,
            page_num,
            rid,
            erase_meta["clip_retention_ratio"],
            min_retention,
        )
        from manhua_pipeline.events import RENDER_ERASE_MASK_LOW_OVERLAP, PipelineEvent
        from manhua_pipeline.io.event_emitter import get_event_emitter

        em = emitter or (
            get_event_emitter(render_dir.parent, config, logger=logger)
            if render_dir
            else None
        )
        if em:
            em.emit(
                PipelineEvent(
                    event=RENDER_ERASE_MASK_LOW_OVERLAP,
                    run_id=run_id,
                    stage=_STAGE_NAME.lower(),
                    chapter=chapter,
                    region_id=rid,
                    page=page_num,
                    metadata={
                        "retention": erase_meta["clip_retention_ratio"],
                        "min_retention": min_retention,
                    },
                )
            )

    if do_debug:
        debug_paths["text_mask"] = str(
            save_render_debug_image(
                erase_meta["text_mask"],
                debug_dir=debug_dir,
                region_id=rid,
                suffix="text_mask",
            )
        )
        debug_paths["erase_mask_clipped"] = str(
            save_render_debug_image(
                erase_meta["erase_mask"],
                debug_dir=debug_dir,
                region_id=rid,
                suffix="erase_mask_clipped",
            )
        )
        debug_paths["after_erase"] = str(
            save_render_debug_image(
                page_img.crop(erase_box.as_xyxy()),
                debug_dir=debug_dir,
                region_id=rid,
                suffix="after_erase",
            )
        )

    # 4. Text Background & Fill
    text_bg = estimate_text_background(page_img, text_box, bubble_box)
    text_fill = _text_color_for_bg(text_bg)

    # Apply spiky/rude style emphasis
    text_to_draw = final_text
    if (register == "rude" or style_hint == "spiky") and getattr(
        config, "EMPHASIS_UPPERCASE", True
    ):
        text_to_draw = final_text.upper()

    font_path = getattr(config, "FONT_PATH", "assets/fonts/ComicNeue-Regular.ttf")

    # 5. Fit & Place English inside layout_box
    font, lines, font_size, overflow, placements = fit_and_place_text(
        draw=draw,
        final_text=text_to_draw,
        font_path=font_path,
        layout_box=layout_box,
        config=config,
        page_width=page_img.width,
    )

    for placement in placements:
        draw.text(
            (placement["x"], placement["y"]),
            placement["text"],
            fill=text_fill,
            font=font,
        )

    if do_debug:
        debug_paths["final"] = str(
            save_render_debug_image(
                page_img.crop(erase_box.as_xyxy()),
                debug_dir=debug_dir,
                region_id=rid,
                suffix="final",
            )
        )

    readable_min = int(getattr(config, "FONT_MIN_PT", 12))
    eff_max_pt = effective_max_pt(config, page_img.width)
    total_overflow = 1 if overflow else 0

    if font_size < readable_min or overflow:
        logger.warning(
            "[%d/%d %s] Page %d %s -> text below readable floor "
            "(pt=%d, min=%d, overflow=%s)",
            _STAGE_INDEX,
            _TOTAL_STAGES,
            _STAGE_NAME,
            page_num,
            rid,
            font_size,
            readable_min,
            overflow,
        )
        from manhua_pipeline.events import RENDER_TEXT_TOO_SMALL, PipelineEvent
        from manhua_pipeline.io.event_emitter import get_event_emitter

        em = emitter or (
            get_event_emitter(render_dir.parent, config, logger=logger)
            if render_dir
            else None
        )
        if em:
            em.emit(
                PipelineEvent(
                    event=RENDER_TEXT_TOO_SMALL,
                    run_id=run_id,
                    stage=_STAGE_NAME.lower(),
                    chapter=chapter,
                    region_id=rid,
                    page=page_num,
                    level="WARNING",
                    metadata={
                        "font_size_pt": font_size,
                        "readable_min_pt": readable_min,
                        "abs_min_pt": int(getattr(config, "FONT_ABS_MIN_PT", 9)),
                        "effective_max_pt": eff_max_pt,
                        "page_width": page_img.width,
                        "page_height": page_img.height,
                        "overflow": overflow,
                        "layout_box": layout_box.as_dict(),
                        "text_box": text_box.as_dict(),
                    },
                )
            )
    else:
        logger.info(
            "[%d/%d %s] Page %d %s -> drew (%dpt, %d lines, erase=%s)",
            _STAGE_INDEX,
            _TOTAL_STAGES,
            _STAGE_NAME,
            page_num,
            rid,
            font_size,
            len(lines),
            erase_meta["erase_strategy"],
        )

    layout_padding = int(getattr(config, "TEXT_LAYOUT_PADDING_PX", 8))
    base_w = max(5, text_box.w - 2 * layout_padding)
    base_h = max(5, text_box.h - 2 * layout_padding)
    base_area = base_w * base_h
    layout_area = layout_box.w * layout_box.h
    layout_expanded = layout_area > base_area
    layout_growth_factor = round(layout_area / max(1, base_area), 2)

    mask_bbox = bubble_mask.getbbox()
    mask_hist = bubble_mask.histogram()
    mask_pixels = mask_hist[255] if len(mask_hist) > 255 else 0
    mask_area = bubble_box.w * bubble_box.h
    mask_coverage = (mask_pixels / mask_area) if mask_area else 0.0

    result_dict = {
        "region_id": rid,
        "page_number": page_num,
        "rendered": True,
        "action": "drew",
        "font_size_pt": font_size,
        "effective_max_pt": eff_max_pt,
        "page_width": page_img.width,
        "page_height": page_img.height,
        "lines": len(lines),
        "readable_min_pt": readable_min,
        "abs_min_pt": int(getattr(config, "FONT_ABS_MIN_PT", 9)),
        "overflow": overflow,
        "overflow_ask_llm": overflow,
        "register": register,
        "style_hint": style_hint,
        "glossary_conflict": glossary_conflict,
        "erase_strategy": erase_meta["erase_strategy"],
        "text_box": text_box.as_dict(),
        "bubble_box": bubble_box.as_dict(),
        "erase_box": erase_box.as_dict(),
        "layout_box": layout_box.as_dict(),
        "layout_expanded": layout_expanded,
        "layout_growth_factor": layout_growth_factor,
        "text_fill": list(text_fill),
        "text_background": list(text_bg),
        "placements": placements,
        "bubble_mask_bbox": list(mask_bbox) if mask_bbox else None,
        "bubble_mask_pixels": mask_pixels,
        "bubble_mask_area": mask_area,
        "bubble_mask_coverage": mask_coverage,
    }
    if debug_paths:
        result_dict["debug_paths"] = debug_paths

    return result_dict, 1, 0, total_overflow


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
            "read_region": region.get("read_region"),
            "parent_bubble": region.get("parent_bubble"),
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
    run_id: str = "unknown",
) -> tuple[list[dict], dict | None, int, int, int]:
    """Render regions on a single page, save output image, and return report components."""
    page_num = page["page_number"]
    filename = page["filename"]
    orig_filename = page.get("original_filename")
    if page.get("skip") or not filename or not orig_filename:
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
            region,
            page_img,
            draw,
            page_num,
            config,
            render_dir=render_dir,
            run_id=run_id,
            chapter=ws.name,
        )
        page_results.append(res_dict)
        page_drawn += drawn
        page_left += left
        page_overflow += ovf

    # Save output image with sequential page number name
    rendered_dir = render_dir / "rendered"
    rendered_dir.mkdir(parents=True, exist_ok=True)
    out_path = rendered_dir / f"{page_num:03d}.png"
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


def _fit_font(draw, text, font_path, max_pt, max_w):
    pt = max_pt
    while pt >= 10:
        f = ImageFont.truetype(font_path, pt)
        if draw.textlength(text, font=f) <= max_w:
            return f
        pt -= 1
    return ImageFont.truetype(font_path, 10)


def _sample_bg(im, cx, cy, W, H, align):
    offs = (0.02, 0.10, 0.18) if align == "left" else (-0.12, -0.06, 0.06, 0.12)
    pts = []
    for dx in offs:
        x, y = int((cx + dx) * W), int(cy * H)
        if 0 <= x < W and 0 <= y < H:
            pts.append(im.getpixel((x, y)))
    if not pts:
        return (12, 14, 22)
    return tuple(sorted(c[i] for c in pts)[len(pts) // 2] for i in range(3))


def _render_credits_page(render_dir: Path, config, page_size) -> "Path | None":
    """Pick a random template, patch placeholders, draw credits. Never raises;
    excluded from manifest/QA."""
    try:
        credits = get_credits()
        templates = getattr(config, "CREDITS_TEMPLATES", {})
        if not templates:
            return None
        root = Path(__file__).resolve().parent.parent.parent
        cdir = root / getattr(config, "CREDITS_DIR", "assets/credits")
        available = [(n, s) for n, s in templates.items() if (cdir / n).exists()]
        if not available:
            logger.warning("[%s] No credit templates found in %s.", _STAGE_NAME, cdir)
            return None

        name, slots = random.choice(available)
        im = Image.open(cdir / name).convert("RGB")
        if page_size and getattr(config, "CREDITS_MATCH_PAGE_SIZE", True):
            im = im.resize(page_size, Image.Resampling.LANCZOS)
        W, H = im.size
        draw = ImageDraw.Draw(im)
        font_path = str(
            root / getattr(config, "FONT_PATH", "assets/fonts/ComicNeue-Bold.ttf")
        )
        fill = getattr(config, "CREDITS_TEXT_FILL", (240, 236, 225))

        for field, cx, cy, align, max_pt, pw, ph in slots:
            val = credits.get(field, "")
            if not val:
                continue
            pwx, phx = int(pw * W), int(ph * H)
            x0 = int(cx * W) - (pwx // 2 if align == "center" else int(0.005 * W))
            y0 = int(cy * H) - phx // 2
            draw.rectangle(
                [x0, y0, x0 + pwx, y0 + phx], fill=_sample_bg(im, cx, cy, W, H, align)
            )
            f = _fit_font(draw, val, font_path, max_pt, pwx - 10)
            tw = draw.textlength(val, font=f)
            tx = int(cx * W) - tw / 2 if align == "center" else x0 + 6
            ty = int(cy * H) - f.size / 2
            draw.text((tx, ty), val, font=f, fill=fill)

        out = render_dir / "zzz_credits.png"
        im.save(out)
        logger.info("[%s] Credits page (%s) -> %s", _STAGE_NAME, name, out.name)
        return out
    except Exception as exc:
        logger.warning("[%s] Credits page skipped (%s).", _STAGE_NAME, exc)
        return None


def run_render(workspace: str, config, run_id: str | None = None) -> Path:
    """Run the Rendering stage."""
    t0 = time.monotonic()
    ws = Path(workspace)
    resolved_run_id = run_id or getattr(config, "_run_id", "unknown")
    logger.info(
        "[%d/%d %s] Series: %s | Chapter: %s",
        _STAGE_INDEX,
        _TOTAL_STAGES,
        _STAGE_NAME,
        ws.parent.as_posix(),
        ws.name,
    )
    log_stage(logger, _STAGE_INDEX, _TOTAL_STAGES, _STAGE_NAME, "starting")
    from manhua_pipeline.events import (
        RENDER_REGION_COMPLETED,
        STAGE_COMPLETED,
        STAGE_STARTED,
    )
    from manhua_pipeline.io.event_emitter import emit_stage_event

    emit_stage_event(
        ws,
        config,
        _STAGE_NAME.lower(),
        STAGE_STARTED,
        logger=logger,
        run_id=resolved_run_id,
    )

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
            page, ws, regions_by_page, render_dir, config, run_id=resolved_run_id
        )
        if report:
            render_results.extend(results)
            page_reports.append(report)
            total_drawn += drawn
            total_left += left
            total_overflow += ovf

    now = datetime.now(timezone.utc).isoformat()

    last_size = None
    for page in reversed(manifest.get("pages", [])):
        if not page.get("skip") and page.get("width") and page.get("height"):
            last_size = (page["width"], page["height"])
            break
    rendered_dir = render_dir / "rendered"
    rendered_dir.mkdir(parents=True, exist_ok=True)
    _render_credits_page(rendered_dir, config, last_size)

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

    for r in render_results:
        if r.get("rendered"):
            from manhua_pipeline.events import PipelineEvent
            from manhua_pipeline.io.event_emitter import get_event_emitter

            get_event_emitter(ws, config, logger=logger).emit(
                PipelineEvent(
                    event=RENDER_REGION_COMPLETED,
                    run_id=resolved_run_id,
                    stage=_STAGE_NAME.lower(),
                    chapter=ws.name,
                    region_id=r["region_id"],
                    page=r.get("page_number"),
                    metadata={
                        "action": r.get("action"),
                        "text_box": r.get("text_box"),
                        "bubble_box": r.get("bubble_box"),
                        "erase_box": r.get("erase_box"),
                        "layout_box": r.get("layout_box"),
                        "erase_strategy": r.get("erase_strategy"),
                        "font_size_pt": r.get("font_size_pt"),
                        "lines": r.get("lines"),
                        "overflow": r.get("overflow"),
                        "placements": r.get("placements"),
                        "bubble_mask_bbox": r.get("bubble_mask_bbox"),
                        "bubble_mask_pixels": r.get("bubble_mask_pixels"),
                        "bubble_mask_area": r.get("bubble_mask_area"),
                        "bubble_mask_coverage": r.get("bubble_mask_coverage"),
                    },
                )
            )

    elapsed = time.monotonic() - t0
    log_stage(
        logger,
        _STAGE_INDEX,
        _TOTAL_STAGES,
        _STAGE_NAME,
        f"done: {len(page_reports)} pages, {total_drawn} drawn, {total_left} left, "
        f"{total_overflow} overflow -> {report_path} (elapsed {elapsed:.1f}s)",
    )
    emit_stage_event(
        ws,
        config,
        _STAGE_NAME.lower(),
        STAGE_COMPLETED,
        logger=logger,
        run_id=resolved_run_id,
        metadata={"total_drawn": total_drawn, "total_left": total_left},
    )
    return report_path
