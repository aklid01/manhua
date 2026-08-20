# Image inpainting, background color estimation, and bubble mask extraction.

import math
from collections import Counter
from statistics import median

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from manhua_pipeline.stages.stage5_render.geometry import (
    box_inside,
    clip_box,
    intersect_boxes,
)
from manhua_pipeline.stages.stage5_render.types import Box


# Check if RGB color luminance is below dark threshold
def _is_dark(c: tuple[int, int, int]) -> bool:
    return 0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2] < 96


# Validate fill color against text box brightness to prevent black boxes
def _sanitize_dark_fill(
    fill: tuple[int, int, int],
    page_img: Image.Image,
    text_box: Box,
) -> tuple[int, int, int]:
    if _is_dark(fill):
        tb = clip_box(text_box, page_img.size)
        if tb.w > 0 and tb.h > 0:
            tb_crop = page_img.crop(tb.as_xyxy())
            arr = np.asarray(tb_crop)
            bright_count = int(
                (
                    0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
                    > 180
                ).sum()
            )
            total = tb_crop.width * tb_crop.height
            if total > 0 and (bright_count / total) > 0.40:
                return (255, 255, 255)
    return fill


# Estimate dominant background color around text bounding box
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

    raw_fill = tuple(int(median(channel)) for channel in zip(*points))
    return _sanitize_dark_fill(raw_fill, image, text_box)


# Count total non-zero mask pixels in L-mode image
def _count_mask_pixels(mask: Image.Image) -> int:
    histogram = mask.histogram()
    return histogram[255] if len(histogram) > 255 else 0


# Perform image inpainting using OpenCV Telea algorithm
def _image_inpaint(
    page_img: Image.Image,
    bubble_box: Box,
    mask: Image.Image,
) -> bool:
    try:
        import cv2
    except ImportError:
        return False

    crop = np.asarray(page_img.crop(bubble_box.as_xyxy()).convert("RGB"))
    mask_arr = np.asarray(mask.convert("L"))
    repaired = cv2.inpaint(crop, mask_arr, 3, cv2.INPAINT_TELEA)
    page_img.paste(Image.fromarray(repaired, mode="RGB"), (bubble_box.x, bubble_box.y))
    return True


# Erase Chinese text within bounding box using mask and background fill
def erase_original_text(
    *,
    page_img: Image.Image,
    draw: ImageDraw.ImageDraw,
    text_box: Box,
    bubble_box: Box,
    erase_box: Box,
    bubble_mask: Image.Image,
    config,
    fill_color: tuple[int, int, int] | None = None,
) -> dict:
    if fill_color is not None:
        fill = fill_color
    else:
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

    text_pixels = _count_mask_pixels(text_mask)

    mask_overlap = Image.fromarray(
        np.minimum(np.asarray(text_mask), np.asarray(bubble_mask)).astype("uint8"),
        mode="L",
    )
    overlap_pixels = _count_mask_pixels(mask_overlap)
    retention = (overlap_pixels / text_pixels) if text_pixels else 0.0
    min_retention = float(getattr(config, "TEXT_ERASE_MIN_RETENTION", 0.65))
    low_overlap = retention < min_retention

    repaired_with_image_inpaint = False
    if low_overlap:
        erase_mask = text_mask
        is_colored = max(fill) - min(fill) > 18 and not all(
            channel >= 245 for channel in fill
        )
        if is_colored and getattr(config, "RENDER_IMAGE_INPAINT", True):
            repaired_with_image_inpaint = _image_inpaint(
                page_img,
                bubble_box,
                text_mask,
            )

        if not repaired_with_image_inpaint:
            fill_image = Image.new("RGB", (bubble_box.w, bubble_box.h), fill)
            page_img.paste(fill_image, (bubble_box.x, bubble_box.y), mask=erase_mask)
    else:
        erase_mask = bubble_mask
        fill_image = Image.new("RGB", (bubble_box.w, bubble_box.h), fill)
        page_img.paste(fill_image, (bubble_box.x, bubble_box.y), mask=erase_mask)

    erase_pixels = _count_mask_pixels(erase_mask)

    margin = int(getattr(config, "TEXT_ERASE_PARENT_MARGIN_PX", 8))
    geometrically_inside = box_inside(text_box, bubble_box, margin=margin)

    pad = int(getattr(config, "TEXT_ERASE_PADDING_PX", 12))
    img_w, img_h = page_img.size
    expanded_text = Box(
        max(0, text_box.x - pad),
        max(0, text_box.y - pad),
        min(img_w, text_box.x2 + pad) - max(0, text_box.x - pad),
        min(img_h, text_box.y2 + pad) - max(0, text_box.y - pad),
    )
    overflow_strips: list[tuple[int, int, int, int]] = []
    if expanded_text.y < bubble_box.y:
        overflow_strips.append(
            (
                expanded_text.x,
                expanded_text.y,
                expanded_text.x2,
                min(bubble_box.y, expanded_text.y2),
            )
        )
    if expanded_text.y2 > bubble_box.y2:
        overflow_strips.append(
            (
                expanded_text.x,
                max(bubble_box.y2, expanded_text.y),
                expanded_text.x2,
                expanded_text.y2,
            )
        )
    if expanded_text.x < bubble_box.x:
        overflow_strips.append(
            (
                expanded_text.x,
                expanded_text.y,
                min(bubble_box.x, expanded_text.x2),
                expanded_text.y2,
            )
        )
    if expanded_text.x2 > bubble_box.x2:
        overflow_strips.append(
            (
                max(bubble_box.x2, expanded_text.x),
                expanded_text.y,
                expanded_text.x2,
                expanded_text.y2,
            )
        )
    for x0, y0, x1, y1 in overflow_strips:
        if x1 > x0 and y1 > y0:
            strip_w, strip_h = x1 - x0, y1 - y0
            strip_fill = Image.new("RGB", (strip_w, strip_h), (255, 255, 255))
            page_img.paste(strip_fill, (x0, y0))

    return {
        "erase_strategy": (
            "text_mask_image_inpaint"
            if repaired_with_image_inpaint
            else "text_mask_clipped_to_bubble"
        ),
        "erase_fill": list(fill),
        "dilation_px": dilation,
        "text_mask_pixels": text_pixels,
        "erase_mask_pixels": erase_pixels,
        "clip_retention_ratio": retention,
        "low_overlap": low_overlap,
        "repaired_with_image_inpaint": repaired_with_image_inpaint,
        "safe_rect": geometrically_inside,
        "text_mask": text_mask,
        "erase_mask": erase_mask,
    }


# Sample median perimeter pixel colors to estimate region background
def _estimate_bg(
    page_img: Image.Image, x: int, y: int, w: int, h: int
) -> tuple[int, int, int]:
    x0, y0 = max(0, x - 2), max(0, y - 2)
    x1, y1 = min(page_img.width - 1, x + w + 2), min(page_img.height - 1, y + h + 2)

    pixels = []
    for px in range(x0, x1 + 1):
        pixels.append(page_img.getpixel((px, y0)))
        pixels.append(page_img.getpixel((px, y1)))
    for py in range(y0, y1 + 1):
        pixels.append(page_img.getpixel((x0, py)))
        pixels.append(page_img.getpixel((x1, py)))

    rgb_pixels = []
    for p in pixels:
        if isinstance(p, tuple) and len(p) >= 3:
            rgb_pixels.append(p[:3])
        elif isinstance(p, int):
            rgb_pixels.append((p, p, p))

    if not rgb_pixels:
        return (255, 255, 255)

    r_vals = sorted([p[0] for p in rgb_pixels])
    g_vals = sorted([p[1] for p in rgb_pixels])
    b_vals = sorted([p[2] for p in rgb_pixels])

    return (
        r_vals[len(r_vals) // 2],
        g_vals[len(g_vals) // 2],
        b_vals[len(b_vals) // 2],
    )


# Determine black or white text color based on background luminance
def _text_color_for_bg(bg: tuple[int, int, int]) -> tuple[int, int, int]:
    r, g, b = bg[:3]
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    return (0, 0, 0) if luminance >= 128 else (255, 255, 255)


# Breadth-first search finding connected white mask pixels
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


# Filter binary grid to retain largest connected component
def _keep_largest_component(
    is_white: list[list[bool]],
    width: int,
    height: int,
    seed_box: Box | None = None,
) -> list[list[bool]]:
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


# Compute dominant background fill color inside bubble crop
def _dominant_fill_color(
    bubble_crop: Image.Image,
    local_text_box: Box | None,
    config,
) -> tuple[int, int, int]:
    img = bubble_crop
    w, h = img.size
    if w == 0 or h == 0:
        return (255, 255, 255)

    if local_text_box is not None:
        pad = max(6, min(local_text_box.w, local_text_box.h) // 4)
        sx0 = max(0, local_text_box.x - pad)
        sy0 = max(0, local_text_box.y - pad)
        sx1 = min(w, local_text_box.x2 + pad)
        sy1 = min(h, local_text_box.y2 + pad)
    else:
        sx0, sy0, sx1, sy1 = 0, 0, w, h

    step_x = max(1, (sx1 - sx0) // 40)
    step_y = max(1, (sy1 - sy0) // 40)

    tx0 = tx1 = ty0 = ty1 = -1
    if local_text_box is not None:
        tx0, ty0 = local_text_box.x, local_text_box.y
        tx1, ty1 = local_text_box.x2, local_text_box.y2

    def _in_text(px: int, py: int) -> bool:
        return tx0 <= px < tx1 and ty0 <= py < ty1

    counter: Counter = Counter()
    samples: list[tuple[int, int, int]] = []

    for py in range(sy0, sy1, step_y):
        for px in range(sx0, sx1, step_x):
            if _in_text(px, py):
                continue
            r, g, b = img.getpixel((px, py))[:3]
            c = (r, g, b)
            samples.append(c)
            counter[(r // 16, g // 16, b // 16)] += 1

    if not counter:
        return (255, 255, 255)

    (qr, qg, qb), _ = counter.most_common(1)[0]

    matching = [
        c for c in samples if (c[0] // 16, c[1] // 16, c[2] // 16) == (qr, qg, qb)
    ]
    if not matching:
        raw_fill = (qr * 16 + 8, qg * 16 + 8, qb * 16 + 8)
    else:
        r = round(sum(c[0] for c in matching) / len(matching))
        g = round(sum(c[1] for c in matching) / len(matching))
        b = round(sum(c[2] for c in matching) / len(matching))

        if r >= 248 and g >= 248 and b >= 248:
            return (255, 255, 255)
        if r <= 6 and g <= 6 and b <= 6:
            raw_fill = (0, 0, 0)
        else:
            raw_fill = (r, g, b)

    if local_text_box is not None and _is_dark(raw_fill):
        tb_crop = img.crop(local_text_box.as_xyxy())
        arr = np.asarray(tb_crop)
        bright_count = int(
            (
                0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2] > 180
            ).sum()
        )
        total = tb_crop.width * tb_crop.height
        if total > 0 and (bright_count / total) > 0.40:
            return (255, 255, 255)

    return raw_fill


# Create boolean mask of pixels within color tolerance
def _color_close_mask(arr, fill_color: tuple[int, int, int], tol: int):
    r, g, b = fill_color
    dr = np.abs(arr[:, :, 0].astype(int) - r)
    dg = np.abs(arr[:, :, 1].astype(int) - g)
    db = np.abs(arr[:, :, 2].astype(int) - b)
    return (dr <= tol) & (dg <= tol) & (db <= tol)


# Generate solid white rectangular mask
def _solid_rect_mask(bubble_box: Box, erase_box: Box) -> Image.Image:
    mask = Image.new("L", (bubble_box.w, bubble_box.h), 0)
    draw = ImageDraw.Draw(mask)
    local = (
        erase_box.x - bubble_box.x,
        erase_box.y - bubble_box.y,
        erase_box.x2 - bubble_box.x,
        erase_box.y2 - bubble_box.y,
    )
    draw.rectangle(local, fill=255)
    return mask


# Extract thought bubble interior mask by tracing radial dark rays
def _get_radiating_thought_mask(  # noqa: C901
    bbox_img: Image.Image,
    generic_mask: Image.Image,
    config,
    *,
    seed_box: Box | None = None,
) -> Image.Image | None:
    if not getattr(config, "THOUGHT_BUBBLE_MASK_ENABLED", True) or seed_box is None:
        return None

    width, height = bbox_img.size
    if min(width, height) < 40:
        return None

    generic_coverage = _count_mask_pixels(generic_mask) / max(1, width * height)
    if generic_coverage < float(
        getattr(config, "THOUGHT_BUBBLE_GENERIC_MIN_COVERAGE", 0.92)
    ):
        return None

    rgb = np.asarray(bbox_img.convert("RGB"), dtype=np.uint8)
    luminance = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
    dark = luminance < int(getattr(config, "THOUGHT_BUBBLE_DARK_THRESHOLD", 130))

    sample_count = max(36, int(getattr(config, "THOUGHT_BUBBLE_ANGLE_SAMPLES", 144)))
    min_ray_length = max(2, int(getattr(config, "THOUGHT_BUBBLE_MIN_RAY_LENGTH_PX", 5)))
    cx, cy = (width - 1) / 2.0, (height - 1) / 2.0
    minimum_radius = max(18, int(min(width, height) * 0.30))
    text_pad = int(getattr(config, "TEXT_ERASE_PADDING_PX", 12))
    text_x0 = max(0, seed_box.x - text_pad)
    text_y0 = max(0, seed_box.y - text_pad)
    text_x1 = min(width, seed_box.x2 + text_pad)
    text_y1 = min(height, seed_box.y2 + text_pad)

    def _edge_distance(dx: float, dy: float) -> float:
        distances = []
        if abs(dx) > 1e-6:
            distances.append((width - 1 - cx) / dx if dx > 0 else cx / -dx)
        if abs(dy) > 1e-6:
            distances.append((height - 1 - cy) / dy if dy > 0 else cy / -dy)
        return min(distances)

    ray_starts: dict[int, int] = {}
    for sector in range(sample_count):
        angle = 2.0 * math.pi * sector / sample_count
        dx, dy = math.cos(angle), math.sin(angle)
        run_length = 0
        run_start = None
        for radius in range(minimum_radius, int(_edge_distance(dx, dy))):
            x = int(round(cx + radius * dx))
            y = int(round(cy + radius * dy))
            if text_x0 <= x < text_x1 and text_y0 <= y < text_y1:
                run_length = 0
                run_start = None
                continue

            x0, x1 = max(0, x - 1), min(width, x + 2)
            y0, y1 = max(0, y - 1), min(height, y + 2)
            if dark[y0:y1, x0:x1].any():
                if run_start is None:
                    run_start = radius
                run_length += 1
                if run_length >= min_ray_length:
                    ray_starts[sector] = run_start
                    break
            else:
                run_length = 0
                run_start = None

    ray_ratio = len(ray_starts) / sample_count
    min_ratio = float(getattr(config, "THOUGHT_BUBBLE_MIN_RAY_SECTOR_RATIO", 0.35))
    max_ratio = float(getattr(config, "THOUGHT_BUBBLE_MAX_RAY_SECTOR_RATIO", 0.75))
    if not min_ratio <= ray_ratio <= max_ratio:
        return None

    valid = np.array(sorted(ray_starts), dtype=float)
    radii = np.array([ray_starts[int(i)] for i in valid], dtype=float)
    expanded_indices = np.concatenate(
        (valid - sample_count, valid, valid + sample_count)
    )
    expanded_radii = np.tile(radii, 3)
    interpolated = np.interp(np.arange(sample_count), expanded_indices, expanded_radii)
    smoothed = np.empty(sample_count, dtype=float)
    for sector in range(sample_count):
        window = [
            interpolated[(sector + offset) % sample_count] for offset in range(-2, 3)
        ]
        smoothed[sector] = float(np.median(window))

    yy, xx = np.indices((height, width))
    distances = np.hypot(xx - cx, yy - cy)
    angles = (np.arctan2(yy - cy, xx - cx) + 2.0 * math.pi) % (2.0 * math.pi)
    positions = angles * sample_count / (2.0 * math.pi)
    lower = np.floor(positions).astype(int) % sample_count
    upper = (lower + 1) % sample_count
    fraction = positions - np.floor(positions)
    boundary = smoothed[lower] * (1.0 - fraction) + smoothed[upper] * fraction
    inset = int(getattr(config, "THOUGHT_BUBBLE_INTERIOR_INSET_PX", 6))
    interior = distances <= np.maximum(0, boundary - inset)

    coverage = float(interior.mean())
    min_coverage = float(getattr(config, "THOUGHT_BUBBLE_INTERIOR_MIN_COVERAGE", 0.28))
    max_coverage = float(getattr(config, "THOUGHT_BUBBLE_INTERIOR_MAX_COVERAGE", 0.80))
    if not min_coverage <= coverage <= max_coverage:
        return None

    coverage_x0 = max(0, seed_box.x)
    coverage_y0 = max(0, seed_box.y)
    coverage_x1 = min(width, seed_box.x2)
    coverage_y1 = min(height, seed_box.y2)
    if coverage_x0 >= coverage_x1 or coverage_y0 >= coverage_y1:
        return None
    text_coverage = float(
        interior[coverage_y0:coverage_y1, coverage_x0:coverage_x1].mean()
    )
    if text_coverage < float(getattr(config, "THOUGHT_BUBBLE_MIN_TEXT_COVERAGE", 0.85)):
        return None

    return Image.fromarray((interior * 255).astype(np.uint8), mode="L")


# Extract speech bubble container alpha mask using flood fill and connectivity
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
    fill_color: tuple[int, int, int] | None = None,
) -> Image.Image:

    img = bbox_img
    width, height = img.size
    mode = getattr(config, "BUBBLE_FILL_MODE", "white")

    if mode == "dominant" and fill_color is not None:
        tol = int(getattr(config, "BUBBLE_FILL_TOLERANCE", 42))
        is_fill = _color_close_mask(np.asarray(img), fill_color, tol)
    else:
        threshold = getattr(config, "BUBBLE_WHITE_THRESHOLD", 220)
        arr = np.asarray(img)
        is_fill = (
            (arr[:, :, 0] > threshold)
            & (arr[:, :, 1] > threshold)
            & (arr[:, :, 2] > threshold)
        )

    is_white = is_fill

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


# Decode base64 YOLO-seg bubble mask from region metadata
def load_region_bubble_mask(region: dict, bubble_box) -> "Image.Image | None":
    bm = region.get("bubble_mask")
    if not bm or not bm.get("valid"):
        return None
    try:
        import base64
        import io

        data = base64.b64decode(bm["mask_b64"])
        mask = Image.open(io.BytesIO(data)).convert("L")
        target = (bubble_box.w, bubble_box.h)
        if mask.size != target:
            mask = mask.resize(target, Image.NEAREST)
        return mask
    except Exception:
        return None
