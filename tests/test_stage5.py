# Unit tests for Stage 5 text layout, font fitting, inpainting erasure, and page rendering.

import json

from PIL import Image, ImageDraw, ImageFont

import config
from manhua_pipeline.stages import stage5_render
from manhua_pipeline.stages.stage5_render import (
    Box,
    _dominant_fill_color,
    _draw_region_plan,
    _finalize_plan,
    _plan_and_erase_region,
    grow_layout_box,
    intersect_boxes,
)


# Helper to configure mock workspace with test images and stage outputs
def _setup(ws, page_wh=(400, 600), det=None, ocr=None, para=None):
    (ws / "pages").mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", page_wh, (0, 255, 0))
    img.paste((255, 255, 255), (50, 50, 150, 100))
    img.paste((0, 0, 0), (70, 60, 130, 90))
    img.paste((255, 255, 255), (200, 300, 300, 350))
    img.paste((0, 0, 0), (220, 310, 280, 340))
    img.save(ws / "pages" / "001.png")

    manifest = {
        "chapter_id": "t",
        "total_pages": 1,
        "pages": [
            {
                "page_number": 1,
                "filename": "001.png",
                "original_filename": "001.png",
                "skip": False,
                "width": page_wh[0],
                "height": page_wh[1],
            }
        ],
        "current_stage": "render",
        "completed_stages": ["import", "detect", "ocr", "translate", "paraphrase"],
        "warning_count": 0,
        "status": "in_progress",
    }
    (ws / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (ws / "stage1_detection").mkdir(parents=True, exist_ok=True)
    (ws / "stage1_detection" / "detection.json").write_text(
        json.dumps(det), encoding="utf-8"
    )
    (ws / "stage2_ocr").mkdir(parents=True, exist_ok=True)
    (ws / "stage2_ocr" / "ocr.json").write_text(
        json.dumps(ocr, ensure_ascii=False), encoding="utf-8"
    )
    (ws / "stage4_paraphrase").mkdir(parents=True, exist_ok=True)
    (ws / "stage4_paraphrase" / "paraphrase.json").write_text(
        json.dumps(para, ensure_ascii=False), encoding="utf-8"
    )


# Helper to construct detection stage output dictionary
def _det(regions):
    return {
        "chapter_id": "t",
        "stage": "detection",
        "generated_at": "now",
        "model": "m",
        "regions": regions,
    }


# Helper to construct test detection region dictionary
def _region(rid, x, y, w, h):
    return {
        "region_id": rid,
        "page_number": 1,
        "type": "speech_bubble",
        "bbox": {"x": x, "y": y, "w": w, "h": h},
        "reading_order": 1,
        "style_hint": "round",
        "confidence": 0.9,
        "read_region": {"x": x, "y": y, "w": w, "h": h},
        "erase_mask": {"type": "rect", "coords": [x, y, w, h]},
        "render": True,
    }


# Verify text regions are erased and rendered while non-text areas are left untouched
def test_render_text_gated_erase(tmp_path, monkeypatch):
    monkeypatch.setattr(
        stage5_render, "_load_font", lambda p, pt, cfg: ImageFont.load_default()
    )
    ws = tmp_path / "workspace"
    det = _det(
        [
            _region("P001_R001", 50, 50, 100, 50),
            _region("P001_R002", 200, 300, 100, 50),
        ]
    )
    ocr = {
        "results": [
            {"region_id": "P001_R001", "has_usable_text": True},
            {"region_id": "P001_R002", "has_usable_text": False},
        ]
    }
    para = {
        "results": [
            {
                "region_id": "P001_R001",
                "final_text": "HI",
                "register": "neutral",
                "paraphrased": True,
            },
            {
                "region_id": "P001_R002",
                "final_text": "",
                "register": "neutral",
                "paraphrased": False,
            },
        ]
    }
    _setup(ws, det=det, ocr=ocr, para=para)
    stage5_render.run_render(str(ws), config)
    out = Image.open(ws / "stage5_render" / "rendered" / "001.jpg").convert("RGB")
    assert out.getpixel((75, 75)) == (255, 255, 255)
    px = out.getpixel((30, 30))
    assert px[1] > 200 and px[0] < 20 and px[2] < 20
    rep = json.loads((ws / "stage5_render" / "render.json").read_text(encoding="utf-8"))
    by_id = {r["region_id"]: r for r in rep["results"]}
    assert by_id["P001_R001"]["rendered"] is True
    assert by_id["P001_R002"]["rendered"] is False
    assert by_id["P001_R002"]["action"] == "left_original_no_text"


# Verify overlapping duplicate bubble regions are suppressed during rendering
def test_render_suppresses_duplicate_overlapping_bubble_regions(tmp_path, monkeypatch):
    monkeypatch.setattr(
        stage5_render, "_load_font", lambda p, pt, cfg: ImageFont.load_default()
    )
    ws = tmp_path / "workspace"
    primary = _region("P001_R001", 90, 85, 200, 220)
    primary["parent_bubble"] = {"x": 25, "y": 26, "w": 300, "h": 280}
    secondary = _region("P001_R002", 130, 170, 230, 100)
    secondary["parent_bubble"] = {"x": 36, "y": 150, "w": 330, "h": 170}
    det = _det([primary, secondary])
    ocr = {
        "results": [
            {
                "region_id": "P001_R001",
                "has_usable_text": True,
                "original_text": "商业触角透各行各",
            },
            {
                "region_id": "P001_R002",
                "has_usable_text": True,
                "original_text": "商业触角渗透各行各业，",
            },
        ]
    }
    para = {
        "results": [
            {"region_id": "P001_R001", "final_text": "The complete dialogue."},
            {"region_id": "P001_R002", "final_text": "The duplicate dialogue."},
        ]
    }
    _setup(ws, det=det, ocr=ocr, para=para)
    stage5_render.run_render(str(ws), config)
    rep = json.loads((ws / "stage5_render" / "render.json").read_text(encoding="utf-8"))
    by_id = {r["region_id"]: r for r in rep["results"]}
    assert by_id["P001_R001"]["rendered"] is True
    assert by_id["P001_R002"]["rendered"] is False
    assert by_id["P001_R002"]["action"] == "overlapping_duplicate"


# Verify text exceeding bubble geometry is rendered and flagged as overflow
def test_render_overflow_flagged(tmp_path, monkeypatch):
    monkeypatch.setattr(
        stage5_render, "_load_font", lambda p, pt, cfg: ImageFont.load_default()
    )
    ws = tmp_path / "workspace"
    det = _det([_region("P001_R001", 10, 10, 40, 20)])
    ocr = {"results": [{"region_id": "P001_R001", "has_usable_text": True}]}
    long_text = "This is an extremely long sentence that cannot possibly fit in a tiny bubble at all."
    para = {
        "results": [
            {
                "region_id": "P001_R001",
                "final_text": long_text,
                "register": "neutral",
                "paraphrased": True,
            }
        ]
    }
    _setup(ws, det=det, ocr=ocr, para=para)
    stage5_render.run_render(str(ws), config)
    rep = json.loads((ws / "stage5_render" / "render.json").read_text(encoding="utf-8"))
    r = rep["results"][0]
    assert r["overflow"] is True
    assert r["rendered"] is True


# Verify regions marked with render=False are skipped during inpainting and typesetting
def test_render_skips_non_render_type(tmp_path, monkeypatch):
    monkeypatch.setattr(
        stage5_render, "_load_font", lambda p, pt, cfg: ImageFont.load_default()
    )
    ws = tmp_path / "workspace"
    reg = _region("P001_R001", 50, 50, 100, 50)
    reg["render"] = False
    det = _det([reg])
    ocr = {"results": [{"region_id": "P001_R001", "has_usable_text": True}]}
    para = {
        "results": [
            {
                "region_id": "P001_R001",
                "final_text": "SIGN",
                "register": "neutral",
                "paraphrased": True,
            }
        ]
    }
    _setup(ws, det=det, ocr=ocr, para=para)
    stage5_render.run_render(str(ws), config)
    out = Image.open(ws / "stage5_render" / "rendered" / "001.jpg").convert("RGB")
    assert out.getpixel((75, 75)) == (0, 0, 0)
    rep = json.loads((ws / "stage5_render" / "render.json").read_text(encoding="utf-8"))
    assert rep["results"][0]["action"] == "left_original_not_render_type"


# Verify stage 5 completes rendering and advances manifest to QA
def test_render_advances_manifest_and_outputs(tmp_path, monkeypatch):
    monkeypatch.setattr(
        stage5_render, "_load_font", lambda p, pt, cfg: ImageFont.load_default()
    )
    ws = tmp_path / "workspace"
    det = _det([_region("P001_R001", 50, 50, 100, 50)])
    ocr = {"results": [{"region_id": "P001_R001", "has_usable_text": True}]}
    para = {
        "results": [
            {
                "region_id": "P001_R001",
                "final_text": "HI",
                "register": "neutral",
                "paraphrased": True,
            }
        ]
    }
    _setup(ws, det=det, ocr=ocr, para=para)
    stage5_render.run_render(str(ws), config)
    assert (ws / "stage5_render" / "rendered" / "001.jpg").exists()
    m = json.loads((ws / "manifest.json").read_text())
    assert m["current_stage"] == "qa"
    assert "render" in m["completed_stages"]


# Verify regions lacking paraphrase text are left unrendered
def test_render_missing_paraphrase_left(tmp_path, monkeypatch):
    monkeypatch.setattr(
        stage5_render, "_load_font", lambda p, pt, cfg: ImageFont.load_default()
    )
    ws = tmp_path / "workspace"
    det = _det([_region("P001_R001", 50, 50, 100, 50)])
    ocr = {"results": [{"region_id": "P001_R001", "has_usable_text": True}]}
    para = {"results": []}
    _setup(ws, det=det, ocr=ocr, para=para)
    stage5_render.run_render(str(ws), config)
    rep = json.loads((ws / "stage5_render" / "render.json").read_text(encoding="utf-8"))
    r = rep["results"][0]
    assert r["rendered"] is False
    assert r["action"] in ("missing_text", "left_original_no_text")


# Verify bubbles touching page seams erase edge text properly
def test_render_edge_touching_erase(tmp_path, monkeypatch):
    monkeypatch.setattr(
        stage5_render, "_load_font", lambda p, pt, cfg: ImageFont.load_default()
    )
    ws = tmp_path / "workspace"
    det = _det([_region("P001_R001", 50, 0, 100, 50)])
    ocr = {"results": [{"region_id": "P001_R001", "has_usable_text": True}]}
    para = {
        "results": [
            {
                "region_id": "P001_R001",
                "final_text": "HI",
                "register": "neutral",
                "paraphrased": True,
            }
        ]
    }
    (ws / "pages").mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (400, 600), (0, 255, 0))
    img.paste((255, 255, 255), (50, 0, 150, 50))
    img.paste((0, 0, 0), (70, 0, 130, 20))
    img.save(ws / "pages" / "001.png")

    manifest = {
        "chapter_id": "t",
        "total_pages": 1,
        "pages": [
            {
                "page_number": 1,
                "filename": "001.png",
                "original_filename": "001.png",
                "skip": False,
                "width": 400,
                "height": 600,
            }
        ],
        "current_stage": "render",
        "completed_stages": ["import", "detect", "ocr", "translate", "paraphrase"],
        "warning_count": 0,
        "status": "in_progress",
    }
    (ws / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (ws / "stage1_detection").mkdir(parents=True, exist_ok=True)
    (ws / "stage1_detection" / "detection.json").write_text(
        json.dumps(det), encoding="utf-8"
    )
    (ws / "stage2_ocr").mkdir(parents=True, exist_ok=True)
    (ws / "stage2_ocr" / "ocr.json").write_text(
        json.dumps(ocr, ensure_ascii=False), encoding="utf-8"
    )
    (ws / "stage4_paraphrase").mkdir(parents=True, exist_ok=True)
    (ws / "stage4_paraphrase" / "paraphrase.json").write_text(
        json.dumps(para, ensure_ascii=False), encoding="utf-8"
    )

    stage5_render.run_render(str(ws), config)
    out = Image.open(ws / "stage5_render" / "rendered" / "001.jpg").convert("RGB")
    assert out.getpixel((75, 10)) == (255, 255, 255)
    px = out.getpixel((30, 10))
    assert px[1] > 200 and px[0] < 20 and px[2] < 20


# Verify text wrapping respects literal escaped newlines
def test_wrap_text_literal_newlines(monkeypatch):
    font = ImageFont.load_default()
    raw = "Name: Lin\\nAge: 18\\r\\nGender: Male"
    lines = stage5_render._wrap_text(raw, font, 200)
    assert len(lines) == 3
    assert lines[0] == "Name: Lin"
    assert lines[1] == "Age: 18"
    assert lines[2] == "Gender: Male"


# Verify text color auto-selection picks high-contrast foreground based on background luminance
def test_text_color_for_bg():
    assert stage5_render._text_color_for_bg((255, 255, 255)) == (0, 0, 0)
    assert stage5_render._text_color_for_bg((200, 200, 200)) == (0, 0, 0)
    assert stage5_render._text_color_for_bg((0, 0, 0)) == (255, 255, 255)
    assert stage5_render._text_color_for_bg((20, 20, 20)) == (255, 255, 255)


# Verify dominant fill helper detects pure white for white speech bubbles
def test_dominant_fill_returns_pure_white_for_white_bubble():
    crop = Image.new("RGB", (200, 120), (255, 255, 255))
    assert _dominant_fill_color(crop, Box(40, 40, 60, 30), object()) == (255, 255, 255)


# Verify dominant fill helper preserves colored bubble fills
def test_dominant_fill_keeps_gold():
    crop = Image.new("RGB", (200, 120), (255, 200, 60))
    r, g, b = _dominant_fill_color(crop, Box(40, 40, 60, 30), object())
    assert r > 240 and 180 < g < 220 and b < 90


# Verify layout expansion avoids colliding with neighboring sibling text boxes
def test_expansion_does_not_overlap_sibling():
    class Cfg:
        LAYOUT_EXPAND_ENABLED = True
        LAYOUT_MAX_GROWTH_FACTOR = 3.0
        LAYOUT_MIN_ANCHOR_OVERLAP = 0.5
        TEXT_LAYOUT_PADDING_PX = 8

    text_box = Box(600, 600, 120, 60)
    sibling = Box(600, 720, 120, 60)
    interior = Box(560, 560, 400, 400)
    grown = grow_layout_box(text_box, interior, (2000, 2000), Cfg(), siblings=[sibling])
    assert intersect_boxes(grown, sibling) is None
    assert intersect_boxes(grown, text_box) is not None


# Verify multiple text boxes inside a shared bubble do not stack on top of each other
def test_shared_parent_regions_do_not_stack(monkeypatch):
    monkeypatch.setattr(
        "manhua_pipeline.stages.stage5_render._load_font",
        lambda p, pt, cfg: ImageFont.load_default(),
    )

    page_img = Image.new("RGB", (1000, 1000), (255, 255, 255))
    draw = ImageDraw.Draw(page_img)

    region = {
        "region_id": "P001_R001",
        "page_number": 1,
        "type": "speech_bubble",
        "bbox": {"x": 600, "y": 600, "w": 100, "h": 50},
        "read_region": {"x": 600, "y": 600, "w": 100, "h": 50},
        "parent_bubble": {"x": 500, "y": 500, "w": 300, "h": 300},
        "render": True,
        "has_usable_text": True,
        "final_text": "Hello World",
    }

    siblings = [Box(600, 700, 100, 50)]

    plan = _plan_and_erase_region(
        region, page_img, draw, 1, config, sibling_text_boxes=siblings
    )
    _draw_region_plan(plan, draw)
    res, _, _, _ = _finalize_plan(
        plan,
        emitter=None,
        run_id="test",
        chapter="ch",
        render_dir=None,
        config=config,
    )

    layout = Box(**res["layout_box"])
    assert intersect_boxes(layout, siblings[0]) is None
    assert layout.x == 608


# Verify lone text box expands into available parent bubble interior
def test_single_region_bubble_still_expands():
    class Cfg:
        LAYOUT_EXPAND_ENABLED = True
        LAYOUT_MAX_GROWTH_FACTOR = 3.0
        LAYOUT_MIN_ANCHOR_OVERLAP = 0.5
        TEXT_LAYOUT_PADDING_PX = 8

    text_box = Box(600, 600, 100, 50)
    interior = Box(500, 500, 300, 300)
    grown = grow_layout_box(text_box, interior, (1000, 1000), Cfg(), siblings=None)
    assert grown.w * grown.h > text_box.w * text_box.h


# Helper to construct test detection region with parent bubble coordinates
def _region_pb(rid, x, y, w, h, *, pb=None):
    r = _region(rid, x, y, w, h)
    if pb:
        r["parent_bubble"] = {"x": pb[0], "y": pb[1], "w": pb[2], "h": pb[3]}
    return r


# Verify two-pass rendering prevents erasing adjacent rendered text in overlapping bubbles
def test_two_overlapping_bubbles_neither_text_is_erased(tmp_path, monkeypatch):
    monkeypatch.setattr(
        stage5_render, "_load_font", lambda p, pt, cfg: ImageFont.load_default()
    )

    det = _det(
        [
            _region_pb("P001_R001", 50, 50, 100, 80, pb=(40, 40, 120, 110)),
            _region_pb("P001_R002", 80, 100, 100, 80, pb=(70, 90, 120, 110)),
        ]
    )
    ocr = {
        "results": [
            {"region_id": "P001_R001", "has_usable_text": True},
            {"region_id": "P001_R002", "has_usable_text": True},
        ]
    }
    para = {
        "results": [
            {
                "region_id": "P001_R001",
                "final_text": "First bubble full text.",
                "register": "neutral",
                "paraphrased": True,
            },
            {
                "region_id": "P001_R002",
                "final_text": "Second bubble full text.",
                "register": "neutral",
                "paraphrased": True,
            },
        ]
    }
    _setup(ws := tmp_path / "workspace", det=det, ocr=ocr, para=para)
    stage5_render.run_render(str(ws), config)

    render = json.loads((ws / "stage5_render" / "render.json").read_text())
    by_id = {r["region_id"]: r for r in render["results"]}
    assert by_id["P001_R001"]["rendered"] and by_id["P001_R002"]["rendered"]
    assert by_id["P001_R001"]["overflow"] is False
    assert by_id["P001_R002"]["overflow"] is False


# Verify all page erasures execute before any text drawing begins
def test_erase_all_happens_before_any_draw(tmp_path, monkeypatch):
    monkeypatch.setattr(
        stage5_render, "_load_font", lambda p, pt, cfg: ImageFont.load_default()
    )

    events = []
    real_erase = stage5_render.erase_original_text

    def spy_erase(**kw):
        events.append("erase")
        return real_erase(**kw)

    real_draw = stage5_render._draw_region_plan

    def spy_draw(plan, draw, render_dir=None):
        if plan.rendered:
            events.append("draw")
        return real_draw(plan, draw, render_dir=render_dir)

    monkeypatch.setattr(stage5_render, "erase_original_text", spy_erase)
    monkeypatch.setattr(stage5_render, "_draw_region_plan", spy_draw)

    det = _det(
        [
            _region_pb("P001_R001", 50, 50, 80, 60),
            _region_pb("P001_R002", 200, 300, 80, 60),
        ]
    )
    ocr = {
        "results": [
            {"region_id": "P001_R001", "has_usable_text": True},
            {"region_id": "P001_R002", "has_usable_text": True},
        ]
    }
    para = {
        "results": [
            {
                "region_id": "P001_R001",
                "final_text": "Alpha.",
                "register": "neutral",
                "paraphrased": True,
            },
            {
                "region_id": "P001_R002",
                "final_text": "Beta.",
                "register": "neutral",
                "paraphrased": True,
            },
        ]
    }
    _setup(ws := tmp_path / "workspace", det=det, ocr=ocr, para=para)
    stage5_render.run_render(str(ws), config)

    assert len(events) >= 2
    first_draw = next((i for i, e in enumerate(events) if e == "draw"), len(events))
    last_erase = max((i for i, e in enumerate(events) if e == "erase"), default=-1)
    assert last_erase < first_draw, f"Erase/draw ordering violated: {events}"


# Verify single bubble renders correctly under two-pass execution
def test_single_bubble_two_pass_unchanged(tmp_path, monkeypatch):
    monkeypatch.setattr(
        stage5_render, "_load_font", lambda p, pt, cfg: ImageFont.load_default()
    )

    det = _det([_region("P001_R001", 50, 50, 100, 50)])
    ocr = {"results": [{"region_id": "P001_R001", "has_usable_text": True}]}
    para = {
        "results": [
            {
                "region_id": "P001_R001",
                "final_text": "Hello.",
                "register": "neutral",
                "paraphrased": True,
            }
        ]
    }
    _setup(ws := tmp_path / "workspace", det=det, ocr=ocr, para=para)
    stage5_render.run_render(str(ws), config)

    render = json.loads((ws / "stage5_render" / "render.json").read_text())
    r = render["results"][0]
    assert r["rendered"] is True
    assert r["overflow"] is False


# Verify text layout clips at stitch seams while inpainting erases across boundaries
def test_stitched_page_clips_bubble_at_stitch_boundary(tmp_path, monkeypatch):
    monkeypatch.setattr(
        stage5_render, "_load_font", lambda p, pt, cfg: ImageFont.load_default()
    )

    stitch_y = 300
    page_h = stitch_y * 2

    ws = tmp_path / "workspace"
    (ws / "pages").mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (400, page_h), (255, 255, 255))
    ImageDraw.Draw(img).rectangle((55, 330, 75, 345), fill=(0, 0, 0))
    img.save(ws / "pages" / "001.png")

    manifest = {
        "chapter_id": "t",
        "total_pages": 1,
        "pages": [
            {
                "page_number": 1,
                "filename": "001.png",
                "original_filename": "001.png",
                "skip": False,
                "width": 400,
                "height": page_h,
                "global_y_offset": stitch_y,
            }
        ],
        "current_stage": "render",
        "completed_stages": [],
        "warning_count": 0,
        "status": "in_progress",
    }
    (ws / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    r001 = {
        "region_id": "P001_R001",
        "page_number": 1,
        "type": "speech_bubble",
        "bbox": {"x": 50, "y": 200, "w": 150, "h": 180},
        "read_region": {"x": 50, "y": 200, "w": 150, "h": 180},
        "parent_bubble": {"x": 20, "y": 180, "w": 200, "h": 200},
        "erase_mask": {"type": "rect", "coords": [20, 180, 200, 200]},
        "reading_order": 1,
        "style_hint": "round",
        "confidence": 0.95,
        "render": True,
    }
    r002 = {
        "region_id": "P001_R002",
        "page_number": 1,
        "type": "speech_bubble",
        "bbox": {"x": 50, "y": stitch_y + 20, "w": 150, "h": 80},
        "read_region": {"x": 50, "y": stitch_y + 20, "w": 150, "h": 80},
        "parent_bubble": {"x": 20, "y": stitch_y - 10, "w": 200, "h": 120},
        "erase_mask": {
            "type": "rect",
            "coords": [20, stitch_y - 10, 200, 120],
        },
        "reading_order": 2,
        "style_hint": "round",
        "confidence": 0.95,
        "render": True,
    }

    det = {
        "chapter_id": "t",
        "stage": "detection",
        "generated_at": "now",
        "model": "m",
        "regions": [r001, r002],
    }
    ocr = {
        "chapter_id": "t",
        "stage": "ocr",
        "results": [
            {"region_id": "P001_R001", "page_number": 1, "has_usable_text": True},
            {"region_id": "P001_R002", "page_number": 1, "has_usable_text": True},
        ],
    }
    para = {
        "chapter_id": "t",
        "stage": "paraphrase",
        "results": [
            {
                "region_id": "P001_R001",
                "page_number": 1,
                "final_text": "Hello.",
                "register": "neutral",
                "paraphrased": True,
            },
            {
                "region_id": "P001_R002",
                "page_number": 1,
                "final_text": "World.",
                "register": "neutral",
                "paraphrased": True,
            },
        ],
    }

    (ws / "stage1_detection").mkdir(parents=True, exist_ok=True)
    (ws / "stage1_detection" / "detection.json").write_text(
        json.dumps(det), encoding="utf-8"
    )
    (ws / "stage2_ocr").mkdir(parents=True, exist_ok=True)
    (ws / "stage2_ocr" / "ocr.json").write_text(json.dumps(ocr), encoding="utf-8")
    (ws / "stage4_paraphrase").mkdir(parents=True, exist_ok=True)
    (ws / "stage4_paraphrase" / "paraphrase.json").write_text(
        json.dumps(para), encoding="utf-8"
    )

    stage5_render.run_render(str(ws), config)

    render = json.loads((ws / "stage5_render" / "render.json").read_text())
    results = {r["region_id"]: r for r in render["results"]}

    assert results["P001_R001"]["rendered"] is True
    assert results["P001_R002"]["rendered"] is True

    r001_result = results["P001_R001"]
    if "text_box" in r001_result:
        text_bottom = r001_result["text_box"]["y"] + r001_result["text_box"]["h"]
        assert text_bottom <= stitch_y
    erase_text_box = r001_result["erase_text_box"]
    assert erase_text_box["y"] + erase_text_box["h"] > stitch_y
    assert r001_result["erase_box"]["y"] + r001_result["erase_box"]["h"] >= 346

    rendered = Image.open(ws / "stage5_render" / "rendered" / "001.jpg").convert("RGB")
    assert rendered.getpixel((65, 337)) == (255, 255, 255)
