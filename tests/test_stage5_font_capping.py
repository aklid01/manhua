from PIL import Image, ImageDraw

from manhua_pipeline.stages.stage5_render import (
    Box,
    bubble_interior_box,
    fit_and_place_text,
    select_render_geometry,
)


class _RenderConfig:
    """Mirrors the real config knobs stage5_render reads. LAYOUT_MAX_GROWTH_FACTOR
    is intentionally left at the buggy 3.0 -- the whole point is that correct
    behaviour must NOT depend on tuning it."""

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
    LAYOUT_MAX_GROWTH_FACTOR = 3.0  # the culprit; behaviour must survive it
    LAYOUT_MIN_ANCHOR_OVERLAP = 0.5
    TEXT_LAYOUT_PADDING_PX = 8
    TEXT_ERASE_PADDING_PX = 12
    FONT_MISSING_HARD_ERROR = True


def _large_bubble_region():
    """A big 800x600 balloon with a tiny detected text box (a few words) and a
    full-white interior mask -- the classic dramatic-bubble layout."""
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


def test_short_text_in_large_bubble_fills_interior():
    """FAILS on current code: layout box is clamped to ~3x the tiny text box
    (~5% of the balloon interior). The layout box must instead be bounded by
    the measured interior mask, so short text can fill a large bubble.

    Pure geometry -- no font file required, so this is the CI-safe guard.
    """
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
    assert interior is not None, "eroded interior mask should yield a bbox"

    interior_area = interior.w * interior.h
    layout_area = layout_box.w * layout_box.h
    fill_ratio = layout_area / interior_area

    # Under the 3.0 growth cap this is ~0.05. It must fill most of the balloon.
    assert fill_ratio >= 0.6, (
        f"layout_box fills only {fill_ratio:.1%} of the bubble interior "
        f"(layout={layout_box.as_dict()}, interior={interior.as_dict()}). "
        "Short text will render tiny in a large balloon. Bound the layout box "
        "by the measured interior mask, not LAYOUT_MAX_GROWTH_FACTOR."
    )


def test_large_bubble_font_scales_up_for_short_line():
    """FAILS on current code: because the layout box is under-grown, the fitted
    font never approaches the size a large balloon can carry for a 4-char line.

    Needs the ComicNeue .ttf (same dependency as the existing readability
    tests). Skips cleanly if the font is absent so CI without assets stays green
    on this one while the geometry guard above still runs.
    """
    import os

    cfg = _RenderConfig()
    if not os.path.exists(cfg.FONT_PATH):
        import pytest

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

    # A 4-char line in an 800x600 balloon should comfortably clear ~2x the
    # readable floor once the layout box tracks the interior.
    assert font_size >= 2 * cfg.FONT_MIN_PT, (
        f"font size {font_size}pt is far below what a large balloon allows for "
        "a 4-char line; the layout box was under-grown by the growth cap."
    )
    assert overflow is False, "short text in a large balloon must not overflow"


def test_rendered_text_stays_inside_bubble_interior():
    """Permanent invariant: fitted text must stay INSIDE the bubble interior.

    Guards the 'capping that doesn't allow text outside the boundary, yet the
    font makes it spill' contradiction: geometry says the box, so the drawn
    glyphs must not exit the measured interior on any side.

    Needs the ComicNeue .ttf (same as the readability tests).
    """
    import os

    cfg = _RenderConfig()
    if not os.path.exists(cfg.FONT_PATH):
        import pytest

        pytest.skip("ComicNeue-Bold.ttf not present")

    region, bubble_box, mask = _large_bubble_region()
    # A longer line stresses the fit ladder toward the floor / spill path.
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

    assert placements, "expected at least one placed line"
    for p in placements:
        assert p["x"] >= interior.x, f"line spills past LEFT edge: {p}"
        assert p["y"] >= interior.y, f"line spills past TOP edge: {p}"
        assert p["x"] + p["w"] <= interior.x2, f"line spills past RIGHT edge: {p}"
        assert p["y"] + p["h"] <= interior.y2, f"line spills past BOTTOM edge: {p}"


def test_uppercase_allcaps_stays_inside_layout_box():
    """All-caps text (EMPHASIS_UPPERCASE path) must not spill outside layout_box.

    Capital letters have wider ink bounds than their pen advance, so getlength()
    under-estimates actual rendered width. This test catches the regression where
    _any_line_overflows and _wrap_text used getlength while fit_and_place_text
    measured with textbbox, letting glyphs escape the bubble boundary.

    Mirrors the real failure in pic 2: long all-caps lines in a medium bubble.
    """
    import os

    cfg = _RenderConfig()
    if not os.path.exists(cfg.FONT_PATH):
        import pytest

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

    # Mirrors the style_hint="spiky"/register="rude" uppercase path in the renderer
    allcaps_text = "THAT'S... THE GUARD BUREAU'S BLUE SEA MEDAL! IS GENERAL HU AWARDING IT TO NING YU?!"
    _font, _lines, _pt, _overflow, placements = fit_and_place_text(
        draw=draw,
        final_text=allcaps_text,
        font_path=cfg.FONT_PATH,
        layout_box=layout_box,
        config=cfg,
        page_width=500,
    )

    assert placements, "expected at least one placed line"
    for p in placements:
        assert p["x"] >= layout_box.x, f"line spills past LEFT edge of layout_box: {p}"
        assert p["x"] + p["w"] <= layout_box.x2, f"line spills past RIGHT edge of layout_box: {p}"
        assert p["y"] >= layout_box.y, f"line spills past TOP edge of layout_box: {p}"
        assert p["y"] + p["h"] <= layout_box.y2, f"line spills past BOTTOM edge of layout_box: {p}"

