# Unit tests for Stage 5 font size scaling, interior mask expansion, and oval bubble boundary capping.

import os

import pytest
from PIL import Image, ImageDraw, ImageFilter

from manhua_pipeline.stages.stage5_render import (
    Box,
    _clamp_placements_to_mask,
    bubble_interior_box,
    fit_and_place_text,
    select_render_geometry,
)


# Mock render configuration settings
class _RenderConfig:
    FONT_PATH = "assets/fonts/ComicNeue-Bold.ttf"
    FONT_MAX_PT = 40
    FONT_MAX_PAGE_FRAC = 0.045
    FONT_MIN_PT = 12
    FONT_ABS_MIN_PT = 9
    FONT_STEP_PT = 1
    LINE_SPACING = 1.15
    TEXT_PADDING_PX = 8
    LAYOUT_EXPAND_ENABLED = True
    LAYOUT_INTERIOR_ERODE_PX = 10
    LAYOUT_MAX_GROWTH_FACTOR = 3.0
    LAYOUT_MIN_ANCHOR_OVERLAP = 0.5
    TEXT_LAYOUT_PADDING_PX = 8
    TEXT_ERASE_PADDING_PX = 12
    FONT_MISSING_HARD_ERROR = True


# Helper to construct test region with large speech bubble and short text
def _large_bubble_region():
    bubble_box = Box(100, 100, 800, 600)
    region = {
        "region_id": "P001_R001",
        "bbox": {"x": 460, "y": 380, "w": 120, "h": 60},
        "read_region": {"x": 460, "y": 380, "w": 120, "h": 60},
        "parent_bubble": bubble_box.as_dict(),
        "render": True,
        "has_usable_text": True,
        "final_text": "NO!!",
    }
    mask = Image.new("L", (bubble_box.w, bubble_box.h), 255)
    return region, bubble_box, mask


# Verify short text layout box expands to fill measured bubble interior
def test_short_text_in_large_bubble_fills_interior():
    cfg = _RenderConfig()
    region, bubble_box, mask = _large_bubble_region()

    _text_box, _bubble, _erase, layout_box = select_render_geometry(
        region,
        image_size=(1200, 1000),
        config=cfg,
        bubble_mask=mask,
        bubble_box=bubble_box,
    )

    interior = bubble_interior_box(mask, bubble_box, cfg.LAYOUT_INTERIOR_ERODE_PX)
    assert interior is not None

    interior_area = interior.w * interior.h
    layout_area = layout_box.w * layout_box.h
    fill_ratio = layout_area / interior_area
    assert fill_ratio >= 0.6


# Verify short dialogue in large bubbles scales up font size appropriately
def test_large_bubble_font_scales_up_for_short_line():
    cfg = _RenderConfig()
    if not os.path.exists(cfg.FONT_PATH):
        pytest.skip("ComicNeue-Bold.ttf not present; geometry guard covers root cause")

    region, bubble_box, mask = _large_bubble_region()
    _text_box, _bubble, _erase, layout_box = select_render_geometry(
        region,
        image_size=(1200, 1000),
        config=cfg,
        bubble_mask=mask,
        bubble_box=bubble_box,
    )

    img = Image.new("RGB", (1200, 1000), "white")
    draw = ImageDraw.Draw(img)
    _font, _lines, font_size, overflow, _placements = fit_and_place_text(
        draw=draw,
        final_text="NO!!",
        font_path=cfg.FONT_PATH,
        layout_box=layout_box,
        config=cfg,
        page_width=1200,
    )

    assert font_size >= 2 * cfg.FONT_MIN_PT
    assert overflow is False


# Verify fitted text remains completely within measured bubble interior boundaries
def test_rendered_text_stays_inside_bubble_interior():
    cfg = _RenderConfig()
    if not os.path.exists(cfg.FONT_PATH):
        pytest.skip("ComicNeue-Bold.ttf not present")

    region, bubble_box, mask = _large_bubble_region()
    region["final_text"] = "You should NOT be here right now!"

    _text_box, _bubble, _erase, layout_box = select_render_geometry(
        region,
        image_size=(1200, 1000),
        config=cfg,
        bubble_mask=mask,
        bubble_box=bubble_box,
    )
    interior = bubble_interior_box(mask, bubble_box, cfg.LAYOUT_INTERIOR_ERODE_PX)
    assert interior is not None

    img = Image.new("RGB", (1200, 1000), "white")
    draw = ImageDraw.Draw(img)
    _font, _lines, _pt, _overflow, placements = fit_and_place_text(
        draw=draw,
        final_text=region["final_text"],
        font_path=cfg.FONT_PATH,
        layout_box=layout_box,
        config=cfg,
        page_width=1200,
    )

    assert placements
    for p in placements:
        assert p["x"] >= interior.x
        assert p["y"] >= interior.y
        assert p["x"] + p["w"] <= interior.x2
        assert p["y"] + p["h"] <= interior.y2


# Verify emphasis all-caps dialogue remains inside layout box boundaries
def test_uppercase_allcaps_stays_inside_layout_box():
    cfg = _RenderConfig()
    if not os.path.exists(cfg.FONT_PATH):
        pytest.skip("ComicNeue-Bold.ttf not present")

    bubble_box = Box(10, 10, 400, 250)
    region = {
        "region_id": "P001_R001",
        "bbox": {"x": 50, "y": 50, "w": 320, "h": 180},
        "read_region": {"x": 50, "y": 50, "w": 320, "h": 180},
        "parent_bubble": bubble_box.as_dict(),
        "render": True,
        "has_usable_text": True,
    }
    mask = Image.new("L", (bubble_box.w, bubble_box.h), 255)

    _text_box, _bubble, _erase, layout_box = select_render_geometry(
        region,
        image_size=(500, 500),
        config=cfg,
        bubble_mask=mask,
        bubble_box=bubble_box,
    )

    img = Image.new("RGB", (500, 500), "white")
    draw = ImageDraw.Draw(img)

    allcaps_text = "THAT'S... THE GUARD BUREAU'S BLUE SEA MEDAL! IS GENERAL HU AWARDING IT TO NING YU?!"
    _font, _lines, _pt, _overflow, placements = fit_and_place_text(
        draw=draw,
        final_text=allcaps_text,
        font_path=cfg.FONT_PATH,
        layout_box=layout_box,
        config=cfg,
        page_width=500,
    )

    assert placements
    for p in placements:
        assert p["x"] >= layout_box.x
        assert p["x"] + p["w"] <= layout_box.x2
        assert p["y"] >= layout_box.y
        assert p["y"] + p["h"] <= layout_box.y2


# Verify text placements clamp within curved boundaries of oval speech bubbles
def test_oval_bubble_clamps_lines_to_curved_wall():
    cfg = _RenderConfig()
    if not os.path.exists(cfg.FONT_PATH):
        pytest.skip("ComicNeue-Bold.ttf not present")

    bw, bh = 400, 250
    bubble_box = Box(50, 50, bw, bh)
    mask = Image.new("L", (bw, bh), 0)
    md = ImageDraw.Draw(mask)
    md.ellipse([0, 0, bw - 1, bh - 1], fill=255)

    interior = bubble_interior_box(mask, bubble_box, cfg.LAYOUT_INTERIOR_ERODE_PX)
    assert interior is not None

    line_h = 30
    line_w = interior.w
    placements_in = [
        {"text": "line1", "x": interior.x, "y": interior.y, "w": line_w, "h": line_h},
        {
            "text": "line2",
            "x": interior.x,
            "y": interior.y + line_h + 5,
            "w": line_w // 2,
            "h": line_h,
        },
        {
            "text": "line3",
            "x": interior.x,
            "y": interior.y2 - line_h,
            "w": line_w,
            "h": line_h,
        },
    ]

    clamped = _clamp_placements_to_mask(
        placements_in, mask, bubble_box, erode_px=cfg.LAYOUT_INTERIOR_ERODE_PX
    )

    k = cfg.LAYOUT_INTERIOR_ERODE_PX * 2 + 1
    eroded = mask.filter(ImageFilter.MinFilter(k))

    for p in clamped:
        sample_y_page = p["y"] + p["h"] // 2
        mask_y = max(0, min(bh - 1, sample_y_page - bubble_box.y))

        row = [eroded.getpixel((mx, mask_y)) for mx in range(bw)]
        whites = [i for i, v in enumerate(row) if v >= 128]
        if not whites:
            continue
        left_wall = bubble_box.x + whites[0]
        right_wall = bubble_box.x + whites[-1] + 1
        available_w = right_wall - left_wall
        cur_w = p["w"]

        if available_w >= cur_w:
            assert p["x"] >= left_wall
            assert p["x"] + cur_w <= right_wall
        else:
            assert p["x"] >= left_wall


# Verify shape-aware font fitting sizes down lines to fit oval geometry
def test_shape_aware_fit_text_in_oval_bubble():
    cfg = _RenderConfig()
    if not os.path.exists(cfg.FONT_PATH):
        pytest.skip("ComicNeue-Bold.ttf not present")

    bw, bh = 350, 200
    bubble_box = Box(50, 50, bw, bh)
    mask = Image.new("L", (bw, bh), 0)
    md = ImageDraw.Draw(mask)
    md.ellipse([0, 0, bw - 1, bh - 1], fill=255)

    _text_box, _bubble, _erase, layout_box = select_render_geometry(
        {
            "region_id": "P001_R001",
            "bbox": {"x": 50, "y": 50, "w": bw, "h": bh},
            "parent_bubble": {"x": 50, "y": 50, "w": bw, "h": bh},
        },
        image_size=(500, 500),
        config=cfg,
        bubble_mask=mask,
        bubble_box=bubble_box,
    )

    img = Image.new("RGB", (500, 500), "white")
    draw = ImageDraw.Draw(img)

    long_text = "SO THE RUMORS ARE TRUE! BUT RAW TALENT ALONE DOESN'T MAKE YOU QUALIFIED TO TEACH!"

    _font, _lines, _size, _overflow, placements = fit_and_place_text(
        draw=draw,
        final_text=long_text,
        font_path=cfg.FONT_PATH,
        layout_box=layout_box,
        config=cfg,
        page_width=500,
        bubble_mask=mask,
        bubble_box=bubble_box,
    )

    assert placements
    k = cfg.LAYOUT_INTERIOR_ERODE_PX * 2 + 1
    eroded = mask.filter(ImageFilter.MinFilter(k))

    top_placement = placements[0]
    sample_y = top_placement["y"] + top_placement["h"] // 2
    mask_y = max(0, min(bh - 1, sample_y - bubble_box.y))
    row = [eroded.getpixel((mx, mask_y)) for mx in range(bw)]
    whites = [i for i, v in enumerate(row) if v >= 128]
    assert whites
    left_wall = bubble_box.x + whites[0]
    right_wall = bubble_box.x + whites[-1] + 1
    avail_w = right_wall - left_wall

    assert top_placement["w"] <= avail_w
