"""Stage 5: Rendering.

Erases original Chinese text from bounding box areas and draws the paraphrased
US English text in its place. Utilizes Pillow to wrap, size, and style.
"""

import json
import random
import time
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from manhua_pipeline.io.settings import get_credits
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
    is_white: list[list[bool]], width: int, height: int
) -> list[list[bool]]:
    """Keep only the largest connected component of True values (the main bubble)."""
    visited = [[False] * width for _ in range(height)]
    largest_comp = []

    for y in range(height):
        for x in range(width):
            if is_white[y][x] and not visited[y][x]:
                comp = _bfs(x, y, is_white, visited, width, height)
                if len(comp) > len(largest_comp):
                    largest_comp = comp

    new_grid = [[False] * width for _ in range(height)]
    for cx, cy in largest_comp:
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

    # 2. Keep only the largest connected component of white pixels (noise filter)
    try:
        from scipy.ndimage import label

        labeled, num_features = label(is_white)
        if num_features > 0:
            counts = np.bincount(labeled.ravel())
            counts[0] = 0
            largest_label = counts.argmax()
            is_white = labeled == largest_label
        else:
            is_white = np.zeros_like(is_white)
    except ImportError:
        is_white_list = is_white.tolist()
        is_white_list = _keep_largest_component(is_white_list, width, height)
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
    """Fits text using the overflow ladder sizing step-down technique with ratio wrapping search."""
    padding = getattr(config, "TEXT_PADDING_PX", 8)
    target_w = max(5, bbox_w - 2 * padding)
    target_h = max(5, bbox_h - 2 * padding)

    # Try fitting text from max_pt down to absolute minimum floor (5pt) to completely avoid clipping
    abs_min = 5
    for pt in range(max_pt, abs_min - 1, -step_pt):
        font = _load_font(font_path, pt, config)

        # Try multiple layout widths to encourage more lines and better aspect ratio
        for ratio in [0.65, 0.8, 0.95]:
            w_limit = max(5, int(target_w * ratio))
            lines = _wrap_text(text, font, w_limit)
            h = _block_height(lines, font, line_spacing)
            if h <= target_h:
                # Ensure no single line overflows the absolute target width
                word_overflow = False
                for line in lines:
                    if font.getlength(line) > target_w:
                        word_overflow = True
                        break
                if not word_overflow:
                    # Mark as overflow only if we had to scale below config's FONT_MIN_PT
                    overflow_flag = pt < min_pt
                    return font, lines, pt, overflow_flag

    # Fallback to min_pt and full target_w, and flag overflow
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
        mask = _get_bubble_mask(
            bbox_img, mx, my, mw, mh, page_img.width, page_img.height, config
        )

        mask_bbox = mask.getbbox()
        mask_pixels = 0
        if mask_bbox is not None:
            hist = mask.histogram()
            mask_pixels = hist[255] if len(hist) > 255 else 0

        crop_area = mw * mh
        # If mask is valid and covers at least 10% of the crop, use it. Otherwise fallback to full box.
        if (
            mask_bbox is not None
            and crop_area > 0
            and (mask_pixels / crop_area) >= 0.10
        ):
            use_mask = True
            c_left, c_top, c_right, c_bottom = mask_bbox
            bubble_x = mx + c_left
            bubble_y = my + c_top
            bubble_w = max(5, c_right - c_left)
            bubble_h = max(5, c_bottom - c_top)
        else:
            use_mask = False
            bubble_x = bbox["x"]
            bubble_y = bbox["y"]
            bubble_w = bbox["w"]
            bubble_h = bbox["h"]

        interior = _estimate_bg(page_img, mx, my, mw, mh)
        is_dark_bubble = (
            0.299 * interior[0] + 0.587 * interior[1] + 0.114 * interior[2]
        ) < 128

        if use_mask and not is_dark_bubble:
            white_img = Image.new("RGB", (mw, mh), (255, 255, 255))
            page_img.paste(white_img, (mx, my), mask=mask)
            text_bg = (255, 255, 255)
        else:
            draw.rectangle([mx, my, mx + mw, my + mh], fill=interior)
            text_bg = interior

        text_fill = _text_color_for_bg(text_bg)

        # 3. Draw English final text
        # Apply spiky/rude style emphasis
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
            bubble_w,
            bubble_h,
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
                "[%d/%d %s] Page %d %s -> OVERFLOW at %dpt, rendered clipped (overflow)",
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
        start_y = bubble_y + (bubble_h - block_h) // 2

        metrics_bbox = font.getbbox("Hg")
        line_h = metrics_bbox[3] - metrics_bbox[1] if metrics_bbox else 10

        for line in lines:
            line_w = font.getlength(line)
            start_x = bubble_x + (bubble_w - line_w) // 2
            draw.text((start_x, start_y), line, font=font, fill=text_fill)
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
            region, page_img, draw, page_num, config
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


def run_render(workspace: str, config) -> Path:
    """Run the Rendering stage."""
    t0 = time.monotonic()
    ws = Path(workspace)
    logger.info(
        "[%d/%d %s] Series: %s | Chapter: %s",
        _STAGE_INDEX,
        _TOTAL_STAGES,
        _STAGE_NAME,
        ws.parent.as_posix(),
        ws.name,
    )
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
