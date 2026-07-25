"""Regression tests for Stage 5 Render small font readability fix."""

import json

from PIL import Image, ImageDraw

from manhua_pipeline.stages.stage5_render import (
    Box,
    _fit_text,
    _render_region,
    bubble_interior_box,
    intersect_boxes,
    select_render_geometry,
)


def test_large_bubble_renders_at_or_above_readable_floor():
    # Big round bubble, short line, tight original text box.
    # After interior expansion, font must be >= FONT_MIN_PT.
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

    text_box, bubble_box, erase_box, layout_box = select_render_geometry(
        region,
        image_size=(500, 500),
        config=cfg,
        bubble_mask=mask,
        bubble_box=Box(100, 100, 300, 200),
    )

    # The layout_box should have expanded well beyond the original text box
    assert layout_box.w > 60
    assert layout_box.h > 40

    # Now try fitting text in it using _fit_text
    font, lines, pt, overflow = _fit_text(
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

    text_box, bubble_box, erase_box, layout_box = select_render_geometry(
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


def test_interior_erosion_excludes_tail_region():
    # Mask (100x100) with a protruding tail at bottom right
    mask = Image.new("L", (100, 100), 0)
    draw = ImageDraw.Draw(mask)
    draw.rectangle((10, 10, 70, 70), fill=255)  # bubble main body
    draw.rectangle((70, 70, 90, 90), fill=255)  # tail

    # eroded interior box
    bubble_box = Box(0, 0, 100, 100)
    interior = bubble_interior_box(mask, bubble_box, erode_px=15)

    assert interior is not None
    # The eroded box should be smaller and should exclude the tail (70..100)
    assert interior.x2 <= 70
    assert interior.y2 <= 70


def test_fit_text_never_returns_below_abs_min():
    class TestConfig:
        FONT_ABS_MIN_PT = 9
        FONT_MIN_PT = 12
        FONT_PATH = "assets/fonts/ComicNeue-Bold.ttf"
        FONT_MISSING_HARD_ERROR = True
        TEXT_PADDING_PX = 0

    cfg = TestConfig()
    font, lines, pt, overflow = _fit_text(
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


def test_small_font_emits_render_text_too_small(tmp_path):
    # Setup dummy directory and event emitter
    from manhua_pipeline.events import RENDER_TEXT_TOO_SMALL
    from manhua_pipeline.io.event_emitter import get_event_emitter

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

    _render_region(
        region,
        img,
        draw,
        page_num=1,
        config=cfg,
        emitter=emitter,
        run_id="test_run",
        chapter="ch1",
        render_dir=render_dir,
    )

    # Read the emitted events
    events_file = ws / "logs" / "events.jsonl"
    assert events_file.exists()

    events = []
    with open(events_file, "r", encoding="utf-8") as f:
        for line in f:
            events.append(json.loads(line.strip()))

    too_small_events = [e for e in events if e["event"] == RENDER_TEXT_TOO_SMALL]
    assert len(too_small_events) > 0
    assert too_small_events[0]["metadata"]["font_size_pt"] < cfg.FONT_MIN_PT


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

    text_box, bubble_box, erase_box, layout_box = select_render_geometry(
        region_p028,
        image_size=(1280, 1707),
        config=cfg,
        bubble_mask=mask,
        bubble_box=Box(310, 600, 779, 487),
    )
    # layout_box area should be larger than or equal to text_box area
    assert (layout_box.w * layout_box.h) >= (text_box.w * text_box.h)
    # layout_box should still overlap text_box
    overlap = intersect_boxes(layout_box, text_box)
    assert overlap is not None
