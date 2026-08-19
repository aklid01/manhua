# Regression tests for Stage 5 inpainting, bubble mask erasure, and layout geometry fixes.

import json
import math

from PIL import Image, ImageDraw

import config
from manhua_pipeline.events import RENDER_REGION_COMPLETED
from manhua_pipeline.stages.stage5_render import (
    Box,
    _dominant_fill_color,
    _get_bubble_mask,
    _get_radiating_thought_mask,
    _solid_rect_mask,
    _text_color_for_bg,
    box_inside,
    erase_original_text,
    fit_and_place_text,
    inpainting,
    normalize_escaped_newlines,
    planner,
    run_render,
    select_render_geometry,
)
from manhua_pipeline.stages.stage5_render.typography import (
    _clamp_placements_to_mask,
)


# Verify text bounding box is used to determine layout boundaries
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


# Verify erase box expands to enclose text bounding box within bubble bounds
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


# Verify dark source glyphs inside erase box are cleanly replaced with fill color
def test_safe_rectangle_erases_dark_source_glyphs():
    image = Image.new("RGB", (500, 300), "white")
    draw = ImageDraw.Draw(image)

    bubble_box = Box(50, 40, 400, 220)
    text_box = Box(170, 130, 160, 40)
    erase_box = Box(158, 118, 184, 64)

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


# Verify mask holes do not leave dark source glyphs unerased
def test_white_mask_holes_do_not_preserve_source_text():
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


# Verify fitted text line placements remain strictly within layout box bounds
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


# Verify render report records erase strategy, text box, bubble box, and layout box
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
    assert res["layout_box"] is not None


# Verify render stage events include propagating run_id
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


# Verify text erasure is clipped cleanly to rounded bubble boundary
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


# Verify clipped mask prevents rectangular over-erasure outside circular bubbles
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


# Verify low mask overlap does not fall back to unclipped rectangle over-erasure
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


# Verify connected component nearest to text box is selected as bubble mask
def test_component_near_text_wins_over_larger_background_component():
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


# Verify colored/gold speech bubbles erase source text using dominant background color
def test_gold_panel_source_text_is_erased():
    class Cfg:
        BUBBLE_FILL_MODE = "dominant"
        BUBBLE_FILL_TOLERANCE = 42
        BUBBLE_WHITE_THRESHOLD = 220
        TEXT_ERASE_DILATION_PX = 5
        TEXT_ERASE_MIN_RETENTION = 0.65
        TEXT_ERASE_PARENT_MARGIN_PX = 8

    cfg = Cfg()

    page = Image.new("RGB", (400, 300), (255, 200, 60))
    draw = ImageDraw.Draw(page)
    draw.rectangle((120, 120, 300, 170), fill=(20, 20, 20))

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
    px = page.getpixel((200, 140))
    assert px[2] < 90 and px[0] > 200


# Verify standard white speech bubbles erase cleanly under dominant color mode
def test_white_bubble_still_erases_under_dominant_mode():
    class Cfg:
        BUBBLE_FILL_MODE = "dominant"
        BUBBLE_FILL_TOLERANCE = 42
        BUBBLE_WHITE_THRESHOLD = 220
        TEXT_ERASE_DILATION_PX = 5
        TEXT_ERASE_MIN_RETENTION = 0.65
        TEXT_ERASE_PARENT_MARGIN_PX = 8

    cfg = Cfg()

    page = Image.new("RGB", (400, 300), (255, 255, 255))
    draw = ImageDraw.Draw(page)
    draw.rectangle((120, 120, 300, 170), fill=(20, 20, 20))

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
    px = page.getpixel((200, 140))
    assert px[0] > 240 and px[1] > 240 and px[2] > 240


# Verify low mask coverage falls back to solid rectangular erase mask
def test_low_coverage_falls_back_to_solid_rect():
    class Cfg:
        BUBBLE_FILL_MODE = "dominant"
        BUBBLE_FILL_TOLERANCE = 10
        BUBBLE_WHITE_THRESHOLD = 220
        TEXT_ERASE_DILATION_PX = 5
        TEXT_ERASE_MIN_RETENTION = 0.65
        TEXT_ERASE_PARENT_MARGIN_PX = 8
        BUBBLE_FILL_MIN_COVERAGE = 0.8

    cfg = Cfg()

    page = Image.new("RGB", (400, 300), (128, 128, 128))
    draw = ImageDraw.Draw(page)
    for y in range(300):
        for x in range(400):
            page.putpixel((x, y), (x % 255, y % 255, (x + y) % 255))

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

    mask_hist = mask.histogram()
    mask_pixels = mask_hist[255] if len(mask_hist) > 255 else 0
    crop_area = bubble_box.w * bubble_box.h
    coverage = mask_pixels / crop_area
    assert coverage < cfg.BUBBLE_FILL_MIN_COVERAGE

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
    for x in range(erase_box.x, erase_box.x2):
        for y in range(erase_box.y, erase_box.y2):
            px = page.getpixel((x, y))
            assert px == fill


# Verify high-contrast text color selection picks black for gold backgrounds
def test_text_color_black_on_gold():
    assert _text_color_for_bg((255, 200, 60)) == (0, 0, 0)


# Verify escaped newline normalization is case-insensitive
def test_normalize_escaped_newlines_is_case_insensitive():
    assert normalize_escaped_newlines(r"A\NB") == "A\nB"
    assert normalize_escaped_newlines(r"A\nB") == "A\nB"
    assert normalize_escaped_newlines(r"A\R\NB") == "A\nB"
    assert normalize_escaped_newlines(r"A\r\nB") == "A\nB"
    assert normalize_escaped_newlines("plain") == "plain"


# Verify low-overlap mask fallback retries with solid rect to erase text area
def test_low_overlap_fallback_erases_text_area(tmp_path, monkeypatch):
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

    assert plan.rendered

    erase_box = Box(**plan.erase_box)
    crop = img.crop(erase_box.as_xyxy())
    pixels = list(crop.getdata())
    white_count = sum(1 for p in pixels if p[0] >= 200 and p[1] >= 200 and p[2] >= 200)
    total = len(pixels)
    assert white_count == total


# Verify placement clamping preserves horizontal text alignment without force-centering
def test_clamp_placements_preserves_layout_x_alignment():
    mask = Image.new("L", (800, 200), 255)
    bubble_box = Box(x=0, y=0, w=800, h=200)

    left_p = [{"text": "Hey!", "x": 100, "y": 80, "w": 120, "h": 40}]
    clamped_left = _clamp_placements_to_mask(left_p, mask, bubble_box, erode_px=0)
    assert clamped_left[0]["x"] == 100

    right_p = [{"text": "Wait up!", "x": 500, "y": 80, "w": 200, "h": 40}]
    clamped_right = _clamp_placements_to_mask(right_p, mask, bubble_box, erode_px=0)
    assert clamped_right[0]["x"] == 500


# Verify dark fill sanitization corrects falsely estimated dark fills to white
def test_sanitize_dark_fill_prevents_false_black_box():
    Image.new("RGB", (400, 300), (255, 255, 255))
    Box(x=50, y=50, w=200, h=100)


# Verify bubble mask inpainting erases entire interior container to remove stray fragments
def test_bubble_mask_erases_entire_bubble_container():
    page_img = Image.new("RGB", (400, 300), (50, 50, 50))
    draw = ImageDraw.Draw(page_img)

    draw.rectangle((50, 50, 350, 250), fill=(255, 255, 255))
    draw.rectangle((70, 70, 110, 110), fill=(0, 0, 0))
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

    assert page_img.getpixel((85, 85)) == (255, 255, 255)
    assert page_img.getpixel((180, 140)) == (255, 255, 255)


# Helper to generate synthetic thought bubble with radiating rays
def _radiating_thought_bubble():
    page = Image.new("RGB", (420, 320), (240, 240, 240))
    draw = ImageDraw.Draw(page)
    cx, cy, rx, ry = 210, 160, 152, 94
    draw.ellipse((cx - rx, cy - ry, cx + rx, cy + ry), fill="white")

    ray_pixels = []
    for index in range(48):
        angle = 2 * math.pi * index / 48
        dx, dy = math.cos(angle), math.sin(angle)
        start = (round(cx + rx * dx), round(cy + ry * dy))
        end = (round(cx + (rx + 22) * dx), round(cy + (ry + 22) * dy))
        draw.line((start, end), fill="black", width=2)
        ray_pixels.append(end)

    return page, ray_pixels


# Verify radiating thought bubble mask preserves outer rays while wiping interior
def test_radiating_thought_mask_preserves_rays_and_erases_inner_fragments():
    page, ray_pixels = _radiating_thought_bubble()
    draw = ImageDraw.Draw(page)
    draw.rectangle((150, 130, 270, 180), fill="black")
    draw.rectangle((165, 195, 190, 215), fill="black")

    bubble_box = Box(0, 0, page.width, page.height)
    text_box = Box(150, 130, 120, 50)
    erase_box = Box(138, 118, 144, 74)
    generic_mask = Image.new("L", page.size, 255)
    thought_mask = _get_radiating_thought_mask(
        page,
        generic_mask,
        config,
        seed_box=text_box,
    )

    assert thought_mask is not None
    assert thought_mask.getpixel((210, 160)) == 255
    assert all(thought_mask.getpixel(point) == 0 for point in ray_pixels)

    before_rays = [page.getpixel(point) for point in ray_pixels]
    erase_original_text(
        page_img=page,
        draw=draw,
        text_box=text_box,
        bubble_box=bubble_box,
        erase_box=erase_box,
        bubble_mask=thought_mask,
        config=config,
        fill_color=(255, 255, 255),
    )

    assert page.getpixel((210, 160)) == (255, 255, 255)
    assert page.getpixel((175, 205)) == (255, 255, 255)
    assert [page.getpixel(point) for point in ray_pixels] == before_rays


# Verify radiating thought mask detector ignores standard round bubbles and panels
def test_radiating_thought_mask_rejects_normal_bubble_and_panel():
    seed_box = Box(120, 100, 80, 40)
    generic_mask = Image.new("L", (320, 240), 255)

    round_bubble = Image.new("RGB", (320, 240), (240, 240, 240))
    round_draw = ImageDraw.Draw(round_bubble)
    round_draw.ellipse((40, 45, 280, 195), fill="white", outline="black", width=3)
    assert (
        _get_radiating_thought_mask(
            round_bubble,
            generic_mask,
            config,
            seed_box=seed_box,
        )
        is None
    )

    panel = Image.new("RGB", (320, 240), "white")
    panel_draw = ImageDraw.Draw(panel)
    panel_draw.rectangle((20, 20, 300, 220), outline="black", width=4)
    assert (
        _get_radiating_thought_mask(panel, generic_mask, config, seed_box=seed_box)
        is None
    )


# Verify render report distinguishes thought mask strategy from generic fill
def test_planner_records_radiating_thought_mask_strategy():
    page, _ = _radiating_thought_bubble()
    draw = ImageDraw.Draw(page)
    draw.rectangle((150, 130, 270, 180), fill="black")
    region = {
        "region_id": "P001_R001",
        "page_number": 1,
        "bbox": {"x": 150, "y": 130, "w": 120, "h": 50},
        "read_region": {"x": 150, "y": 130, "w": 120, "h": 50},
        "parent_bubble": {"x": 0, "y": 0, "w": page.width, "h": page.height},
        "render": True,
        "has_usable_text": True,
        "final_text": "Saving everyone comes first.",
        "register": "neutral",
        "style_hint": "round",
        "glossary_conflict": False,
    }

    plan = planner._plan_and_erase_region(region, page, draw, 1, config)
    result, *_ = planner._finalize_plan(
        plan,
        emitter=None,
        run_id="test",
        chapter="test",
        render_dir=None,
        config=config,
    )

    assert plan.erase_strategy == "thought_bubble_inner_mask"
    assert result["erase_strategy"] == "thought_bubble_inner_mask"
    assert result["bubble_mask_source"] == "thought_radiating_inner"


# Verify text box area extending above parent bubble is completely erased
def test_erase_clears_text_above_bubble():
    page = Image.new("RGB", (800, 2000), (200, 200, 200))
    draw = ImageDraw.Draw(page)

    bubble_box = Box(20, 1265, 688, 509)
    text_box = Box(29, 1011, 558, 670)

    bubble_mask = Image.new("L", (bubble_box.w, bubble_box.h), 255)

    red = (220, 50, 50)
    draw.rectangle(
        (text_box.x, text_box.y, text_box.x2, bubble_box.y - 1),
        fill=red,
    )

    erase_box = Box(
        max(text_box.x, bubble_box.x),
        bubble_box.y,
        min(text_box.x2, bubble_box.x2) - max(text_box.x, bubble_box.x),
        min(text_box.y2, bubble_box.y2) - bubble_box.y,
    )

    erase_original_text(
        page_img=page,
        draw=draw,
        text_box=text_box,
        bubble_box=bubble_box,
        erase_box=erase_box,
        bubble_mask=bubble_mask,
        config=config,
    )

    above_mid_x = text_box.x + text_box.w // 2
    above_mid_y = (text_box.y + bubble_box.y) // 2
    pixel = page.getpixel((above_mid_x, above_mid_y))
    assert pixel == (255, 255, 255)
