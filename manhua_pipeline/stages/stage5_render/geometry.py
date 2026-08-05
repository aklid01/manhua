from PIL import Image, ImageFilter

from manhua_pipeline.stages.stage5_render.types import Box


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
    mask = bubble_mask
    if erode_px and erode_px > 0:
        k = erode_px * 2 + 1
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


def _shrink_box_away_from_siblings(
    candidate: Box,
    anchor: Box,
    siblings: list[Box],
) -> Box:
    x0, y0, x1, y1 = candidate.x, candidate.y, candidate.x2, candidate.y2
    ax0, ay0, ax1, ay1 = anchor.x, anchor.y, anchor.x2, anchor.y2

    for sib in siblings:
        sx0, sy0, sx1, sy1 = sib.x, sib.y, sib.x2, sib.y2

        if sx1 <= x0 or sx0 >= x1 or sy1 <= y0 or sy0 >= y1:
            continue

        candidates = []
        if sx0 >= ax1:
            candidates.append(("x1", sx0, x1 - sx0))
        if sx1 <= ax0:
            candidates.append(("x0", sx1, sx1 - x0))
        if sy0 >= ay1:
            candidates.append(("y1", sy0, y1 - sy0))
        if sy1 <= ay0:
            candidates.append(("y0", sy1, sy1 - y0))

        if not candidates:
            return anchor

        edge, value, _cost = min(candidates, key=lambda c: c[2])
        if edge == "x1":
            x1 = min(x1, value)
        elif edge == "x0":
            x0 = max(x0, value)
        elif edge == "y1":
            y1 = min(y1, value)
        elif edge == "y0":
            y0 = max(y0, value)

    x0 = min(x0, ax0)
    y0 = min(y0, ay0)
    x1 = max(x1, ax1)
    y1 = max(y1, ay1)
    return Box(x0, y0, x1 - x0, y1 - y0)


def grow_layout_box(
    text_box: Box,
    interior_box: Box | None,
    image_size: tuple[int, int],
    config,
    siblings: list[Box] | None = None,
) -> Box:
    layout_padding = int(getattr(config, "TEXT_LAYOUT_PADDING_PX", 8))

    base = Box(
        x=text_box.x + layout_padding,
        y=text_box.y + layout_padding,
        w=max(5, text_box.w - 2 * layout_padding),
        h=max(5, text_box.h - 2 * layout_padding),
    )

    if not getattr(config, "LAYOUT_EXPAND_ENABLED", True) or interior_box is None:
        return base

    union = Box(
        x=min(base.x, interior_box.x),
        y=min(base.y, interior_box.y),
        w=max(base.x2, interior_box.x2) - min(base.x, interior_box.x),
        h=max(base.y2, interior_box.y2) - min(base.y, interior_box.y),
    )
    candidate = intersect_boxes(union, interior_box) or base

    if siblings:
        candidate = _shrink_box_away_from_siblings(candidate, text_box, siblings)

    min_overlap = float(getattr(config, "LAYOUT_MIN_ANCHOR_OVERLAP", 0.5))
    overlap = intersect_boxes(candidate, text_box)
    overlap_ratio = (
        (overlap.w * overlap.h) / max(1, text_box.w * text_box.h) if overlap else 0.0
    )
    if overlap_ratio < min_overlap:
        return base

    return clip_box(candidate, image_size)


def select_render_geometry(
    region: dict,
    image_size: tuple[int, int],
    config,
    *,
    bubble_mask=None,
    bubble_box: Box | None = None,
    sibling_text_boxes: list[Box] | None = None,
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

    layout_box = grow_layout_box(
        text_box, interior_box, image_size, config, siblings=sibling_text_boxes
    )

    return text_box, bubble_box, erase_box, layout_box
