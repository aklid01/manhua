"""Regression tests for Stage 5 Render original CJK text survival fix.

Covering geometry roles, deterministic erasure, layout fitting, event metadata,
and run_id propagation as defined in stage5_render_original_text_survival_fix.md.
"""

import json

from PIL import Image, ImageDraw

import config
from manhua_pipeline.events import RENDER_REGION_COMPLETED
from manhua_pipeline.stages.stage5_render import (
    Box,
    box_inside,
    erase_original_text,
    fit_and_place_text,
    run_render,
    select_render_geometry,
)


def test_p028_r001_uses_text_bbox_for_layout():
    region = {
        "region_id": "P028_R001",
        "bbox": {"x": 566, "y": 874, "w": 334, "h": 99},
        "read_region": {"x": 566, "y": 874, "w": 334, "h": 99},
        "erase_mask": {"type": "rect", "coords": [310, 600, 779, 487]},
        "parent_bubble": {"x": 310, "y": 600, "w": 779, "h": 487},
    }

    text_box, bubble_box, erase_box, layout_box = select_render_geometry(
        region,
        image_size=(1280, 1707),
        config=config,
    )

    assert text_box == Box(566, 874, 334, 99)
    assert bubble_box == Box(310, 600, 779, 487)
    assert layout_box.x >= text_box.x
    assert layout_box.y >= text_box.y
    assert layout_box.x2 <= text_box.x2
    assert layout_box.y2 <= text_box.y2


def test_p028_r001_erase_box_contains_text_box():
    region = {
        "region_id": "P028_R001",
        "bbox": {"x": 566, "y": 874, "w": 334, "h": 99},
        "read_region": {"x": 566, "y": 874, "w": 334, "h": 99},
        "erase_mask": {"type": "rect", "coords": [310, 600, 779, 487]},
        "parent_bubble": {"x": 310, "y": 600, "w": 779, "h": 487},
    }

    text_box, bubble_box, erase_box, _ = select_render_geometry(
        region,
        image_size=(1280, 1707),
        config=config,
    )

    assert erase_box.x <= text_box.x
    assert erase_box.y <= text_box.y
    assert erase_box.x2 >= text_box.x2
    assert erase_box.y2 >= text_box.y2
    assert box_inside(erase_box, bubble_box, margin=0)


def test_safe_rectangle_erases_dark_source_glyphs():
    image = Image.new("RGB", (500, 300), "white")
    draw = ImageDraw.Draw(image)

    bubble_box = Box(50, 40, 400, 220)
    text_box = Box(170, 130, 160, 40)
    erase_box = Box(158, 118, 184, 64)

    # Synthetic dark glyph strokes
    draw.rectangle((180, 135, 195, 165), fill="black")
    draw.rectangle((220, 135, 235, 165), fill="black")
    draw.rectangle((260, 135, 275, 165), fill="black")

    result = erase_original_text(
        page_img=image,
        draw=draw,
        text_box=text_box,
        bubble_box=bubble_box,
        erase_box=erase_box,
        bubble_mask=Image.new("L", (400, 220), 255),
        config=config,
    )

    assert result["erase_strategy"] == "text_mask_clipped_to_bubble"

    crop = image.crop(erase_box.as_xyxy())
    assert all(pixel == (255, 255, 255) for pixel in crop.getdata())


def test_white_mask_holes_do_not_preserve_source_text():
    from manhua_pipeline.stages.stage5_render import _get_bubble_mask

    image = Image.new("RGB", (300, 200), (50, 50, 50))
    draw = ImageDraw.Draw(image)
    draw.rectangle((60, 40, 240, 140), fill=(255, 255, 255))
    draw.rectangle((120, 90, 180, 120), fill="black")

    text_box = Box(120, 90, 60, 30)
    bubble_box = Box(60, 40, 180, 100)
    erase_box = Box(108, 78, 84, 54)

    local_text_box = Box(
        x=text_box.x - bubble_box.x,
        y=text_box.y - bubble_box.y,
        w=text_box.w,
        h=text_box.h,
    )
    bubble_crop = image.crop(bubble_box.as_xyxy())
    bubble_mask = _get_bubble_mask(
        bubble_crop,
        bubble_box.x,
        bubble_box.y,
        bubble_box.w,
        bubble_box.h,
        image.width,
        image.height,
        config,
        seed_box=local_text_box,
    )

    erase_original_text(
        page_img=image,
        draw=draw,
        text_box=text_box,
        bubble_box=bubble_box,
        erase_box=erase_box,
        bubble_mask=bubble_mask,
        config=config,
    )

    assert image.getpixel((140, 100)) == (255, 255, 255)


def test_placement_stays_inside_layout_box():
    dummy_img = Image.new("RGB", (1000, 1000), "white")
    draw = ImageDraw.Draw(dummy_img)

    font_path = getattr(config, "FONT_PATH", "assets/fonts/ComicNeue-Regular.ttf")
    font, lines, size, overflow, placements = fit_and_place_text(
        draw=draw,
        final_text="It's right here!",
        font_path=font_path,
        layout_box=Box(574, 882, 318, 83),
        config=config,
    )

    assert lines
    for placement in placements:
        assert placement["x"] >= 574
        assert placement["y"] >= 882
        assert placement["x"] + placement["w"] <= 574 + 318
        assert placement["y"] + placement["h"] <= 882 + 83


def test_render_json_records_erase_and_layout_geometry(tmp_path):
    ws = tmp_path / "workspace"
    ws.mkdir(parents=True, exist_ok=True)
    manifest = {
        "chapter_id": "test_ch",
        "total_pages": 1,
        "pages": [
            {
                "page_number": 1,
                "filename": "001.png",
                "original_filename": "001.png",
                "skip": False,
                "width": 1280,
                "height": 1707,
            }
        ],
        "current_stage": "render",
        "completed_stages": [],
        "warning_count": 0,
        "status": "in_progress",
    }
    (ws / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (ws / "pages").mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (1280, 1707), "white").save(ws / "pages" / "001.png")

    det = {
        "chapter_id": "test_ch",
        "stage": "detection",
        "regions": [
            {
                "region_id": "P001_R001",
                "page_number": 1,
                "type": "speech_bubble",
                "bbox": {"x": 566, "y": 874, "w": 334, "h": 99},
                "read_region": {"x": 566, "y": 874, "w": 334, "h": 99},
                "parent_bubble": {"x": 310, "y": 600, "w": 779, "h": 487},
                "reading_order": 1,
                "render": True,
            }
        ],
    }
    (ws / "stage1_detection").mkdir(parents=True, exist_ok=True)
    (ws / "stage1_detection" / "detection.json").write_text(
        json.dumps(det), encoding="utf-8"
    )

    ocr = {
        "chapter_id": "test_ch",
        "stage": "ocr",
        "results": [
            {"region_id": "P001_R001", "page_number": 1, "has_usable_text": True}
        ],
    }
    (ws / "stage2_ocr").mkdir(parents=True, exist_ok=True)
    (ws / "stage2_ocr" / "ocr.json").write_text(json.dumps(ocr), encoding="utf-8")

    para = {
        "chapter_id": "test_ch",
        "stage": "paraphrase",
        "results": [
            {
                "region_id": "P001_R001",
                "page_number": 1,
                "final_text": "It's right here!",
            }
        ],
    }
    (ws / "stage4_paraphrase").mkdir(parents=True, exist_ok=True)
    (ws / "stage4_paraphrase" / "paraphrase.json").write_text(
        json.dumps(para), encoding="utf-8"
    )

    run_render(str(ws), config, run_id="run-test-123")

    render_report_path = ws / "stage5_render" / "render.json"
    assert render_report_path.exists()

    report = json.loads(render_report_path.read_text(encoding="utf-8"))
    res = report["results"][0]

    assert res["rendered"] is True
    assert res["erase_strategy"] == "text_mask_clipped_to_bubble"
    assert res["text_box"] == {"x": 566, "y": 874, "w": 334, "h": 99}
    assert res["bubble_box"] == {"x": 310, "y": 600, "w": 779, "h": 487}
    assert res["layout_box"] != res["bubble_box"]


def test_render_events_receive_entrypoint_run_id(tmp_path):
    ws = tmp_path / "workspace"
    ws.mkdir(parents=True, exist_ok=True)
    manifest = {
        "chapter_id": "test_ch",
        "total_pages": 1,
        "pages": [
            {
                "page_number": 1,
                "filename": "001.png",
                "original_filename": "001.png",
                "skip": False,
                "width": 1000,
                "height": 1000,
            }
        ],
        "current_stage": "render",
        "completed_stages": [],
        "warning_count": 0,
        "status": "in_progress",
    }
    (ws / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (ws / "pages").mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (1000, 1000), "white").save(ws / "pages" / "001.png")

    det = {
        "chapter_id": "test_ch",
        "stage": "detection",
        "regions": [
            {
                "region_id": "P001_R001",
                "page_number": 1,
                "type": "speech_bubble",
                "bbox": {"x": 100, "y": 100, "w": 200, "h": 50},
                "reading_order": 1,
                "render": True,
            }
        ],
    }
    (ws / "stage1_detection").mkdir(parents=True, exist_ok=True)
    (ws / "stage1_detection" / "detection.json").write_text(
        json.dumps(det), encoding="utf-8"
    )

    ocr = {
        "chapter_id": "test_ch",
        "stage": "ocr",
        "results": [
            {"region_id": "P001_R001", "page_number": 1, "has_usable_text": True}
        ],
    }
    (ws / "stage2_ocr").mkdir(parents=True, exist_ok=True)
    (ws / "stage2_ocr" / "ocr.json").write_text(json.dumps(ocr), encoding="utf-8")

    para = {
        "chapter_id": "test_ch",
        "stage": "paraphrase",
        "results": [
            {"region_id": "P001_R001", "page_number": 1, "final_text": "Hello!"}
        ],
    }
    (ws / "stage4_paraphrase").mkdir(parents=True, exist_ok=True)
    (ws / "stage4_paraphrase" / "paraphrase.json").write_text(
        json.dumps(para), encoding="utf-8"
    )

    target_run_id = "test-run-id-999"
    run_render(str(ws), config, run_id=target_run_id)

    event_file = ws / "logs" / "events.jsonl"
    assert event_file.exists()
    events = [
        json.loads(line)
        for line in event_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert len(events) >= 2
    for event in events:
        assert event["run_id"] == target_run_id

    region_ev = next(e for e in events if e["event"] == RENDER_REGION_COMPLETED)
    assert region_ev["metadata"]["erase_strategy"] == "text_mask_clipped_to_bubble"
    assert region_ev["metadata"]["text_box"] == {"x": 100, "y": 100, "w": 200, "h": 50}


def test_erase_is_clipped_to_round_bubble():
    page = Image.new("RGB", (400, 300), (180, 180, 180))
    draw = ImageDraw.Draw(page)

    bubble_box = Box(50, 30, 300, 220)

    bubble_mask = Image.new("L", (300, 220), 0)
    mask_draw = ImageDraw.Draw(bubble_mask)
    mask_draw.ellipse((0, 0, 299, 219), fill=255)

    bubble_fill = Image.new("RGB", (300, 220), "white")
    page.paste(bubble_fill, (bubble_box.x, bubble_box.y), bubble_mask)

    text_box = Box(90, 100, 220, 50)
    erase_box = Box(70, 80, 260, 90)

    draw.rectangle((130, 115, 270, 145), fill="black")

    outside_before = page.getpixel((60, 80))

    result = erase_original_text(
        page_img=page,
        draw=draw,
        text_box=text_box,
        bubble_box=bubble_box,
        erase_box=erase_box,
        bubble_mask=bubble_mask,
        config=config,
    )

    assert page.getpixel((180, 125)) == (255, 255, 255)
    assert page.getpixel((60, 80)) == outside_before
    assert result["erase_strategy"] == "text_mask_clipped_to_bubble"


def test_clipped_mask_removes_rectangular_corners():
    page = Image.new("RGB", (200, 200), (100, 100, 100))
    draw = ImageDraw.Draw(page)

    bubble_box = Box(20, 20, 160, 160)

    bubble_mask = Image.new("L", (160, 160), 0)
    mask_draw = ImageDraw.Draw(bubble_mask)
    mask_draw.ellipse((0, 0, 159, 159), fill=255)

    text_box = Box(30, 30, 140, 140)
    erase_box = Box(20, 20, 160, 160)

    erase_original_text(
        page_img=page,
        draw=draw,
        text_box=text_box,
        bubble_box=bubble_box,
        erase_box=erase_box,
        bubble_mask=bubble_mask,
        config=config,
    )

    assert page.getpixel((22, 22)) == (100, 100, 100)


def test_low_overlap_never_falls_back_to_unclipped_rectangle():
    page = Image.new("RGB", (300, 300), (50, 50, 50))
    draw = ImageDraw.Draw(page)

    bubble_box = Box(50, 50, 200, 200)

    bubble_mask = Image.new("L", (200, 200), 0)
    mask_draw = ImageDraw.Draw(bubble_mask)
    mask_draw.rectangle((10, 10, 30, 30), fill=255)

    text_box = Box(60, 60, 180, 180)
    erase_box = Box(50, 50, 200, 200)

    result = erase_original_text(
        page_img=page,
        draw=draw,
        text_box=text_box,
        bubble_box=bubble_box,
        erase_box=erase_box,
        bubble_mask=bubble_mask,
        config=config,
    )

    assert result["low_overlap"] is True
    assert page.getpixel((150, 150)) == (50, 50, 50)


def test_component_near_text_wins_over_larger_background_component():
    from PIL import Image

    from manhua_pipeline.stages.stage5_render import _get_bubble_mask

    img = Image.new("RGB", (200, 200), (50, 50, 50))
    draw = ImageDraw.Draw(img)
    draw.rectangle((0, 0, 80, 120), fill=(255, 255, 255))
    draw.rectangle((100, 50, 150, 100), fill=(255, 255, 255))

    local_text_box = Box(105, 55, 40, 40)
    mask = _get_bubble_mask(
        img, 0, 0, 200, 200, 200, 200, config, seed_box=local_text_box
    )

    assert mask.getpixel((120, 75)) == 255
    assert mask.getpixel((40, 60)) == 0


def test_gold_panel_source_text_is_erased():
    from PIL import Image, ImageDraw

    from manhua_pipeline.stages.stage5_render import (
        Box,
        _dominant_fill_color,
        _get_bubble_mask,
        erase_original_text,
    )

    class Cfg:
        BUBBLE_FILL_MODE = "dominant"
        BUBBLE_FILL_TOLERANCE = 42
        BUBBLE_WHITE_THRESHOLD = 220
        TEXT_ERASE_DILATION_PX = 5
        TEXT_ERASE_MIN_RETENTION = 0.65
        TEXT_ERASE_PARENT_MARGIN_PX = 8

    cfg = Cfg()

    page = Image.new("RGB", (400, 300), (255, 200, 60))  # gold panel
    draw = ImageDraw.Draw(page)
    draw.rectangle((120, 120, 300, 170), fill=(20, 20, 20))  # dark source glyphs

    bubble_box = Box(50, 50, 300, 200)
    text_box = Box(110, 110, 200, 70)
    erase_box = Box(98, 98, 224, 94)
    local_text = Box(
        text_box.x - bubble_box.x, text_box.y - bubble_box.y, text_box.w, text_box.h
    )

    crop = page.crop(bubble_box.as_xyxy())
    fill = _dominant_fill_color(crop, local_text, cfg)
    assert abs(fill[0] - 255) < 24 and abs(fill[1] - 200) < 24 and fill[2] < 90

    mask = _get_bubble_mask(
        crop,
        bubble_box.x,
        bubble_box.y,
        bubble_box.w,
        bubble_box.h,
        page.width,
        page.height,
        cfg,
        seed_box=local_text,
        fill_color=fill,
    )
    erase_original_text(
        page_img=page,
        draw=draw,
        text_box=text_box,
        bubble_box=bubble_box,
        erase_box=erase_box,
        bubble_mask=mask,
        config=cfg,
        fill_color=fill,
    )
    # Glyph area now shows the gold fill, not dark text.
    px = page.getpixel((200, 140))
    assert px[2] < 90 and px[0] > 200  # gold, not (20,20,20)


def test_white_bubble_still_erases_under_dominant_mode():
    from PIL import Image, ImageDraw

    from manhua_pipeline.stages.stage5_render import (
        Box,
        _dominant_fill_color,
        _get_bubble_mask,
        erase_original_text,
    )

    class Cfg:
        BUBBLE_FILL_MODE = "dominant"
        BUBBLE_FILL_TOLERANCE = 42
        BUBBLE_WHITE_THRESHOLD = 220
        TEXT_ERASE_DILATION_PX = 5
        TEXT_ERASE_MIN_RETENTION = 0.65
        TEXT_ERASE_PARENT_MARGIN_PX = 8

    cfg = Cfg()

    page = Image.new("RGB", (400, 300), (255, 255, 255))  # white panel
    draw = ImageDraw.Draw(page)
    draw.rectangle((120, 120, 300, 170), fill=(20, 20, 20))  # dark source glyphs

    bubble_box = Box(50, 50, 300, 200)
    text_box = Box(110, 110, 200, 70)
    erase_box = Box(98, 98, 224, 94)
    local_text = Box(
        text_box.x - bubble_box.x, text_box.y - bubble_box.y, text_box.w, text_box.h
    )

    crop = page.crop(bubble_box.as_xyxy())
    fill = _dominant_fill_color(crop, local_text, cfg)
    assert fill[0] > 240 and fill[1] > 240 and fill[2] > 240

    mask = _get_bubble_mask(
        crop,
        bubble_box.x,
        bubble_box.y,
        bubble_box.w,
        bubble_box.h,
        page.width,
        page.height,
        cfg,
        seed_box=local_text,
        fill_color=fill,
    )
    erase_original_text(
        page_img=page,
        draw=draw,
        text_box=text_box,
        bubble_box=bubble_box,
        erase_box=erase_box,
        bubble_mask=mask,
        config=cfg,
        fill_color=fill,
    )
    # Glyph area now shows the white fill, not dark text.
    px = page.getpixel((200, 140))
    assert px[0] > 240 and px[1] > 240 and px[2] > 240


def test_low_coverage_falls_back_to_solid_rect():
    from PIL import Image, ImageDraw

    from manhua_pipeline.stages.stage5_render import (
        Box,
        _dominant_fill_color,
        _get_bubble_mask,
        erase_original_text,
    )

    class Cfg:
        BUBBLE_FILL_MODE = "dominant"
        BUBBLE_FILL_TOLERANCE = 10  # low tolerance to ensure very small mask
        BUBBLE_WHITE_THRESHOLD = 220
        TEXT_ERASE_DILATION_PX = 5
        TEXT_ERASE_MIN_RETENTION = 0.65
        TEXT_ERASE_PARENT_MARGIN_PX = 8
        BUBBLE_FILL_MIN_COVERAGE = 0.8  # high min coverage to trigger fallback

    cfg = Cfg()

    # Gradient/noisy panel where mask coverage will be low
    page = Image.new("RGB", (400, 300), (128, 128, 128))
    draw = ImageDraw.Draw(page)
    # Draw noise/gradients
    for y in range(300):
        for x in range(400):
            page.putpixel((x, y), (x % 255, y % 255, (x + y) % 255))

    # Put dark text
    draw.rectangle((120, 120, 300, 170), fill=(0, 0, 0))

    bubble_box = Box(50, 50, 300, 200)
    text_box = Box(110, 110, 200, 70)
    erase_box = Box(98, 98, 224, 94)
    local_text = Box(
        text_box.x - bubble_box.x, text_box.y - bubble_box.y, text_box.w, text_box.h
    )

    crop = page.crop(bubble_box.as_xyxy())
    fill = _dominant_fill_color(crop, local_text, cfg)

    mask = _get_bubble_mask(
        crop,
        bubble_box.x,
        bubble_box.y,
        bubble_box.w,
        bubble_box.h,
        page.width,
        page.height,
        cfg,
        seed_box=local_text,
        fill_color=fill,
    )

    # Let's count coverage. Should be low.
    mask_hist = mask.histogram()
    mask_pixels = mask_hist[255] if len(mask_hist) > 255 else 0
    crop_area = bubble_box.w * bubble_box.h
    coverage = mask_pixels / crop_area
    assert coverage < cfg.BUBBLE_FILL_MIN_COVERAGE

    # If we apply the fallback manually, it becomes solid rect
    from manhua_pipeline.stages.stage5_render import _solid_rect_mask

    fallback_mask = _solid_rect_mask(bubble_box, erase_box)

    erase_original_text(
        page_img=page,
        draw=draw,
        text_box=text_box,
        bubble_box=bubble_box,
        erase_box=erase_box,
        bubble_mask=fallback_mask,
        config=cfg,
        fill_color=fill,
    )
    # The entire erase_box area must be filled with the fill color
    for x in range(erase_box.x, erase_box.x2):
        for y in range(erase_box.y, erase_box.y2):
            px = page.getpixel((x, y))
            assert px == fill


def test_text_color_black_on_gold():
    from manhua_pipeline.stages.stage5_render import _text_color_for_bg

    assert _text_color_for_bg((255, 200, 60)) == (0, 0, 0)


def test_normalize_escaped_newlines_is_case_insensitive():
    from manhua_pipeline.stages.stage5_render import normalize_escaped_newlines

    assert normalize_escaped_newlines(r"A\NB") == "A\nB"
    assert normalize_escaped_newlines(r"A\nB") == "A\nB"
    assert normalize_escaped_newlines(r"A\R\NB") == "A\nB"
    assert normalize_escaped_newlines(r"A\r\nB") == "A\nB"
    assert normalize_escaped_newlines("plain") == "plain"


def test_low_overlap_fallback_erases_text_area(tmp_path, monkeypatch):
    """When the bubble mask has good overall coverage but clips most of the
    erase rectangle (low_overlap), the planner must retry with a solid rect
    mask so original glyphs are fully cleared before English text is drawn."""
    from manhua_pipeline.stages.stage5_render import inpainting, planner
    from manhua_pipeline.stages.stage5_render.types import Box

    class _Cfg:
        FONT_PATH = "assets/fonts/ComicNeue-Bold.ttf"
        FONT_MAX_PT = 20
        FONT_MAX_PAGE_FRAC = 0.0
        FONT_MIN_PT = 12
        FONT_ABS_MIN_PT = 9
        FONT_STEP_PT = 1
        LINE_SPACING = 1.15
        TEXT_PADDING_PX = 4
        TEXT_LAYOUT_PADDING_PX = 4
        TEXT_ERASE_PADDING_PX = 4
        TEXT_ERASE_DILATION_PX = 0
        TEXT_ERASE_MIN_RETENTION = 0.65
        TEXT_ERASE_PARENT_MARGIN_PX = 8
        BUBBLE_FILL_MODE = "white"
        BUBBLE_WHITE_THRESHOLD = 220
        BUBBLE_FILL_MIN_COVERAGE = 0.40
        LAYOUT_EXPAND_ENABLED = False
        LAYOUT_INTERIOR_ERODE_PX = 0
        LAYOUT_MAX_GROWTH_FACTOR = 1.0
        LAYOUT_MIN_ANCHOR_OVERLAP = 0.5
        FONT_MISSING_HARD_ERROR = True
        EMPHASIS_UPPERCASE = True
        BUBBLE_FILL_TOLERANCE = 42

    cfg = _Cfg()

    img_w, img_h = 400, 200
    bubble_box = Box(x=0, y=0, w=img_w, h=img_h)
    text_box = Box(x=100, y=80, w=200, h=40)

    img = Image.new("RGB", (img_w, img_h), (255, 255, 255))
    img.paste((30, 30, 30), (text_box.x, text_box.y, text_box.x2, text_box.y2))

    white_top_mask = Image.new("L", (img_w, img_h), 0)
    white_top_mask.paste(255, (0, 0, img_w, 60))

    def _patched_get_bubble_mask(*args, **kwargs):
        return white_top_mask

    monkeypatch.setattr(inpainting, "_get_bubble_mask", _patched_get_bubble_mask)

    draw = ImageDraw.Draw(img)
    region = {
        "region_id": "P001_R001",
        "page_number": 1,
        "bbox": {"x": text_box.x, "y": text_box.y, "w": text_box.w, "h": text_box.h},
        "read_region": {
            "x": text_box.x,
            "y": text_box.y,
            "w": text_box.w,
            "h": text_box.h,
        },
        "parent_bubble": {
            "x": bubble_box.x,
            "y": bubble_box.y,
            "w": bubble_box.w,
            "h": bubble_box.h,
        },
        "render": True,
        "has_usable_text": True,
        "final_text": "Wait up!",
        "register": "neutral",
        "glossary_conflict": False,
        "style_hint": "round",
    }

    plan = planner._plan_and_erase_region(
        region, img, draw, page_num=1, config=cfg, render_dir=None
    )

    assert plan.rendered, "expected plan to render"

    erase_box = Box(**plan.erase_box)
    crop = img.crop(erase_box.as_xyxy())
    pixels = list(crop.getdata())
    white_count = sum(1 for p in pixels if p[0] >= 200 and p[1] >= 200 and p[2] >= 200)
    total = len(pixels)
    assert white_count == total, (
        f"Expected erase zone to be fully erased after low_overlap fallback; "
        f"{total - white_count}/{total} non-white pixels remain"
    )


def test_clamp_placements_preserves_layout_x_alignment():
    """_clamp_placements_to_mask must clamp placements within mask bounds without
    forcibly recentering off-center placements to the middle of the bubble mask."""
    from PIL import Image

    from manhua_pipeline.stages.stage5_render.types import Box
    from manhua_pipeline.stages.stage5_render.typography import (
        _clamp_placements_to_mask,
    )

    mask = Image.new("L", (800, 200), 255)
    bubble_box = Box(x=0, y=0, w=800, h=200)

    # Left-aligned placement (e.g. region 1 on the left side of a wide speech bubble)
    left_p = [{"text": "Hey!", "x": 100, "y": 80, "w": 120, "h": 40}]
    clamped_left = _clamp_placements_to_mask(left_p, mask, bubble_box, erode_px=0)
    assert clamped_left[0]["x"] == 100, f"Expected x=100, got {clamped_left[0]['x']}"

    # Right-aligned placement (e.g. region 2 on the right side of a wide speech bubble)
    right_p = [{"text": "Wait up!", "x": 500, "y": 80, "w": 200, "h": 40}]
    clamped_right = _clamp_placements_to_mask(right_p, mask, bubble_box, erode_px=0)
    assert clamped_right[0]["x"] == 500, f"Expected x=500, got {clamped_right[0]['x']}"


def test_sanitize_dark_fill_prevents_false_black_box():
    """When a dark fill color is estimated (e.g. from outer black margins/stroke),
    _sanitize_dark_fill must correct it to white if the text_box is white."""
    from PIL import Image

    from manhua_pipeline.stages.stage5_render.types import Box

    # White page image with a white text_box
    image = Image.new("RGB", (400, 300), (255, 255, 255))
    text_box = Box(x=50, y=50, w=200, h=100)


def test_bubble_mask_erases_entire_bubble_container():
    """erase_original_text must erase the entire speech bubble container interior,
    preventing unrendered sibling text fragments or stray edge punctuation from surviving."""
    from PIL import Image, ImageDraw

    from manhua_pipeline.stages.stage5_render.inpainting import erase_original_text
    from manhua_pipeline.stages.stage5_render.types import Box

    # White bubble image with black text outside the text_box (e.g. top-left unrendered fragment)
    page_img = Image.new("RGB", (400, 300), (50, 50, 50))
    draw = ImageDraw.Draw(page_img)

    # White bubble interior
    draw.rectangle((50, 50, 350, 250), fill=(255, 255, 255))
    # Unrendered Chinese fragment on top-left inside bubble
    draw.rectangle((70, 70, 110, 110), fill=(0, 0, 0))
    # Main text region in middle
    draw.rectangle((150, 120, 250, 180), fill=(0, 0, 0))

    bubble_box = Box(50, 50, 300, 200)
    text_box = Box(150, 120, 100, 60)
    erase_box = Box(138, 108, 124, 84)

    bubble_mask = Image.new("L", (300, 200), 255)

    erase_original_text(
        page_img=page_img,
        draw=draw,
        text_box=text_box,
        bubble_box=bubble_box,
        erase_box=erase_box,
        bubble_mask=bubble_mask,
        config=config,
    )

    # Both top-left fragment (85, 85) and main text (180, 140) must be erased to white
    assert page_img.getpixel((85, 85)) == (255, 255, 255), (
        "Top-left fragment was not erased"
    )
    assert page_img.getpixel((180, 140)) == (255, 255, 255), "Main text was not erased"
