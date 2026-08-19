# Unit tests for Stage 5 readability metrics, minimum font size limits, and resolution-aware font scaling.

import json

from PIL import Image, ImageDraw

from manhua_pipeline.events import RENDER_TEXT_TOO_SMALL
from manhua_pipeline.io.event_emitter import get_event_emitter
from manhua_pipeline.stages.stage5_render import (
    Box,
    _draw_region_plan,
    _finalize_plan,
    _fit_text,
    _plan_and_erase_region,
    bubble_interior_box,
    effective_max_pt,
    fit_and_place_text,
    intersect_boxes,
    select_render_geometry,
)


# Verify dialogue inside large bubbles renders at or above minimum font floor
def test_large_bubble_renders_at_or_above_readable_floor():
    class TestConfig:
        FONT_PATH = "assets/fonts/ComicNeue-Bold.ttf"
        FONT_MAX_PT = 24
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
        FONT_MISSING_HARD_ERROR = True

    cfg = TestConfig()
    region = {
        "region_id": "P001_R001",
        "bbox": {"x": 200, "y": 200, "w": 60, "h": 40},
        "read_region": {"x": 200, "y": 200, "w": 60, "h": 40},
        "parent_bubble": {"x": 100, "y": 100, "w": 300, "h": 200},
    }

    mask = Image.new("L", (300, 200), 255)

    _text_box, _bubble_box, _erase_box, layout_box = select_render_geometry(
        region,
        image_size=(500, 500),
        config=cfg,
        bubble_mask=mask,
        bubble_box=Box(100, 100, 300, 200),
    )

    assert layout_box.w > 60
    assert layout_box.h > 40

    _font, _lines, pt, overflow = _fit_text(
        text="Hello world, this is a test of the readability fit.",
        bbox_w=layout_box.w,
        bbox_h=layout_box.h,
        font_path=cfg.FONT_PATH,
        max_pt=cfg.FONT_MAX_PT,
        min_pt=cfg.FONT_MIN_PT,
        step_pt=cfg.FONT_STEP_PT,
        line_spacing=cfg.LINE_SPACING,
        config=cfg,
        use_padding=False,
    )
    assert pt >= cfg.FONT_MIN_PT
    assert overflow is False


# Verify expanded layout box maintains required anchor overlap with detected text box
def test_expanded_layout_still_overlaps_text_box():
    class TestConfig:
        LAYOUT_EXPAND_ENABLED = True
        LAYOUT_INTERIOR_ERODE_PX = 10
        LAYOUT_MAX_GROWTH_FACTOR = 3.0
        LAYOUT_MIN_ANCHOR_OVERLAP = 0.5
        TEXT_LAYOUT_PADDING_PX = 8
        TEXT_ERASE_PADDING_PX = 12

    cfg = TestConfig()
    region = {
        "region_id": "P001_R001",
        "bbox": {"x": 200, "y": 200, "w": 60, "h": 40},
        "read_region": {"x": 200, "y": 200, "w": 60, "h": 40},
        "parent_bubble": {"x": 100, "y": 100, "w": 300, "h": 200},
    }
    mask = Image.new("L", (300, 200), 255)

    text_box, _bubble_box, _erase_box, layout_box = select_render_geometry(
        region,
        image_size=(500, 500),
        config=cfg,
        bubble_mask=mask,
        bubble_box=Box(100, 100, 300, 200),
    )
    overlap = intersect_boxes(layout_box, text_box)
    assert overlap is not None
    ratio = (overlap.w * overlap.h) / (text_box.w * text_box.h)
    assert ratio >= cfg.LAYOUT_MIN_ANCHOR_OVERLAP


# Verify bubble mask erosion excludes pointing tail geometry
def test_interior_erosion_excludes_tail_region():
    mask = Image.new("L", (100, 100), 0)
    draw = ImageDraw.Draw(mask)
    draw.rectangle((10, 10, 70, 70), fill=255)
    draw.rectangle((70, 70, 90, 90), fill=255)

    bubble_box = Box(0, 0, 100, 100)
    interior = bubble_interior_box(mask, bubble_box, erode_px=15)

    assert interior is not None
    assert interior.x2 <= 70
    assert interior.y2 <= 70


# Verify text fitter never produces font size below absolute minimum floor
def test_fit_text_never_returns_below_abs_min():
    class TestConfig:
        FONT_ABS_MIN_PT = 9
        FONT_MIN_PT = 12
        FONT_PATH = "assets/fonts/ComicNeue-Bold.ttf"
        FONT_MISSING_HARD_ERROR = True
        TEXT_PADDING_PX = 0

    cfg = TestConfig()
    _font, _lines, pt, overflow = _fit_text(
        text="A very long line that cannot fit in a tiny space no matter what we do, even at the absolute minimum point size.",
        bbox_w=20,
        bbox_h=10,
        font_path=cfg.FONT_PATH,
        max_pt=20,
        min_pt=cfg.FONT_MIN_PT,
        step_pt=1,
        line_spacing=1.15,
        config=cfg,
        use_padding=False,
    )
    assert pt >= cfg.FONT_ABS_MIN_PT
    assert overflow is True


# Verify rendering text below target font size emits RENDER_TEXT_TOO_SMALL event
def test_small_font_emits_render_text_too_small(tmp_path):
    class TestConfig:
        FONT_PATH = "assets/fonts/ComicNeue-Bold.ttf"
        FONT_MAX_PT = 21
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
        FONT_MISSING_HARD_ERROR = True
        TEXT_ERASE_PADDING_PX = 12
        EMPHASIS_UPPERCASE = True

    cfg = TestConfig()
    ws = tmp_path / "workspace"
    ws.mkdir()
    render_dir = ws / "stage5_render"
    render_dir.mkdir()

    emitter = get_event_emitter(ws, cfg)

    region = {
        "region_id": "P001_R001",
        "bbox": {"x": 10, "y": 10, "w": 30, "h": 20},
        "read_region": {"x": 10, "y": 10, "w": 30, "h": 20},
        "parent_bubble": {"x": 10, "y": 10, "w": 30, "h": 20},
        "render": True,
        "has_usable_text": True,
        "final_text": "This is a super long sentence that cannot possibly fit in 30x20 pixels.",
    }

    img = Image.new("RGB", (100, 100), "white")
    draw = ImageDraw.Draw(img)

    plan = _plan_and_erase_region(
        region,
        img,
        draw,
        page_num=1,
        config=cfg,
        render_dir=render_dir,
    )
    _draw_region_plan(plan, draw)
    _finalize_plan(
        plan,
        emitter=emitter,
        run_id="test_run",
        chapter="ch1",
        render_dir=render_dir,
        config=cfg,
    )

    events_file = ws / "logs" / "events.jsonl"
    assert events_file.exists()

    events = []
    with open(events_file, "r", encoding="utf-8") as f:
        for line in f:
            events.append(json.loads(line.strip()))

    too_small_events = [e for e in events if e["event"] == RENDER_TEXT_TOO_SMALL]
    assert len(too_small_events) > 0
    assert too_small_events[0]["metadata"]["font_size_pt"] < cfg.FONT_MIN_PT


# Verify layout expansion produces larger bounding area while remaining anchored
def test_p028_r001_layout_larger_than_text_box_but_anchored():
    class TestConfig:
        LAYOUT_EXPAND_ENABLED = True
        LAYOUT_INTERIOR_ERODE_PX = 10
        LAYOUT_MAX_GROWTH_FACTOR = 3.0
        LAYOUT_MIN_ANCHOR_OVERLAP = 0.5
        TEXT_LAYOUT_PADDING_PX = 8
        TEXT_ERASE_PADDING_PX = 12

    cfg = TestConfig()
    region_p028 = {
        "region_id": "P028_R001",
        "bbox": {"x": 566, "y": 874, "w": 334, "h": 99},
        "read_region": {"x": 566, "y": 874, "w": 334, "h": 99},
        "parent_bubble": {"x": 310, "y": 600, "w": 779, "h": 487},
    }
    mask = Image.new("L", (779, 487), 255)

    text_box, _bubble_box, _erase_box, layout_box = select_render_geometry(
        region_p028,
        image_size=(1280, 1707),
        config=cfg,
        bubble_mask=mask,
        bubble_box=Box(310, 600, 779, 487),
    )
    assert (layout_box.w * layout_box.h) >= (text_box.w * text_box.h)
    overlap = intersect_boxes(layout_box, text_box)
    assert overlap is not None


# Verify effective maximum font size scales proportionally with page width
def test_effective_max_pt_scales_with_width():
    class Cfg:
        FONT_MAX_PT = 40
        FONT_MAX_PAGE_FRAC = 0.045

    assert effective_max_pt(Cfg(), 700) == 40
    assert effective_max_pt(Cfg(), 1280) == 57


# Verify maximum font size uses static limit when page fraction is zero
def test_effective_max_pt_disabled_when_frac_zero():
    class Cfg:
        FONT_MAX_PT = 40
        FONT_MAX_PAGE_FRAC = 0.0

    assert effective_max_pt(Cfg(), 5000) == 40


# Verify missing page width defaults to static font size limit
def test_effective_max_pt_none_width_uses_static():
    class Cfg:
        FONT_MAX_PT = 40
        FONT_MAX_PAGE_FRAC = 0.045

    assert effective_max_pt(Cfg(), None) == 40


# Verify merged vertical strip uses page width rather than total height for scaling
def test_merged_canvas_ceiling_matches_single_page():
    class Cfg:
        FONT_MAX_PT = 40
        FONT_MAX_PAGE_FRAC = 0.045

    single_w = 900
    merged_w = 900
    assert effective_max_pt(Cfg(), single_w) == effective_max_pt(Cfg(), merged_w)


# Verify long vertical webtoon strips calculate font ceilings from strip width
def test_strip_ceiling_uses_width_not_height():
    class Cfg:
        FONT_MAX_PT = 40
        FONT_MAX_PAGE_FRAC = 0.045

    assert effective_max_pt(Cfg(), 800) == 40


# Verify identical text renders with larger font size on higher-resolution pages
def test_same_text_larger_on_wider_page():
    class TestConfig:
        FONT_PATH = "assets/fonts/ComicNeue-Bold.ttf"
        FONT_MAX_PT = 20
        FONT_MAX_PAGE_FRAC = 0.045
        FONT_MIN_PT = 12
        FONT_ABS_MIN_PT = 9
        FONT_STEP_PT = 1
        LINE_SPACING = 1.15
        TEXT_PADDING_PX = 8
        FONT_MISSING_HARD_ERROR = True

    cfg = TestConfig()
    text = "Short text."
    layout = Box(0, 0, 500, 300)
    img = Image.new("RGB", (1600, 1000), "white")
    draw = ImageDraw.Draw(img)

    _, _, pt_narrow, _, _ = fit_and_place_text(
        draw=draw,
        final_text=text,
        font_path=cfg.FONT_PATH,
        layout_box=layout,
        config=cfg,
        page_width=700,
    )
    _, _, pt_wide, _, _ = fit_and_place_text(
        draw=draw,
        final_text=text,
        font_path=cfg.FONT_PATH,
        layout_box=layout,
        config=cfg,
        page_width=1600,
    )
    assert pt_wide > pt_narrow


# Verify fitting text without page width parameter enforces static max font size
def test_fit_and_place_text_without_page_width_matches_static():
    class TestConfig:
        FONT_PATH = "assets/fonts/ComicNeue-Bold.ttf"
        FONT_MAX_PT = 20
        FONT_MAX_PAGE_FRAC = 0.045
        FONT_MIN_PT = 12
        FONT_ABS_MIN_PT = 9
        FONT_STEP_PT = 1
        LINE_SPACING = 1.15
        TEXT_PADDING_PX = 8
        FONT_MISSING_HARD_ERROR = True

    cfg = TestConfig()
    img = Image.new("RGB", (500, 500), "white")
    draw = ImageDraw.Draw(img)

    _, _, pt, _, _ = fit_and_place_text(
        draw=draw,
        final_text="Hi there",
        font_path=cfg.FONT_PATH,
        layout_box=Box(0, 0, 400, 300),
        config=cfg,
    )
    assert pt <= cfg.FONT_MAX_PT


# Verify render report records effective max font size and page dimensions
def test_render_json_records_effective_max_pt_and_width(tmp_path):
    class TestConfig:
        FONT_PATH = "assets/fonts/ComicNeue-Bold.ttf"
        FONT_MAX_PT = 20
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
        FONT_MISSING_HARD_ERROR = True
        TEXT_ERASE_PADDING_PX = 12
        EMPHASIS_UPPERCASE = True

    cfg = TestConfig()
    ws = tmp_path / "workspace"
    ws.mkdir()
    render_dir = ws / "stage5_render"
    render_dir.mkdir()

    region = {
        "region_id": "P001_R001",
        "bbox": {"x": 10, "y": 10, "w": 100, "h": 50},
        "read_region": {"x": 10, "y": 10, "w": 100, "h": 50},
        "parent_bubble": {"x": 10, "y": 10, "w": 100, "h": 50},
        "render": True,
        "has_usable_text": True,
        "final_text": "Hello world",
    }

    img = Image.new("RGB", (1280, 1600), "white")
    draw = ImageDraw.Draw(img)

    plan = _plan_and_erase_region(
        region,
        img,
        draw,
        page_num=1,
        config=cfg,
        render_dir=render_dir,
    )
    _draw_region_plan(plan, draw)
    result_dict, _, _, _ = _finalize_plan(
        plan,
        emitter=None,
        run_id="test_run",
        chapter="ch1",
        render_dir=render_dir,
        config=cfg,
    )

    assert result_dict["effective_max_pt"] == 57
    assert result_dict["page_width"] == 1280
    assert result_dict["page_height"] == 1600
