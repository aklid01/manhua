"""Regression tests for Stage 5 Render original CJK text survival fix.

Covering geometry roles, deterministic erasure, layout fitting, event metadata,
and run_id propagation as defined in stage5_render_original_text_survival_fix.md.
"""

import json
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

import config
from manhua_pipeline.events import RENDER_REGION_COMPLETED
from manhua_pipeline.io.event_emitter import get_event_emitter
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

    assert result["erase_strategy"] == "padded_text_rectangle"

    crop = image.crop(erase_box.as_xyxy())
    assert all(pixel == (255, 255, 255) for pixel in crop.getdata())


def test_white_mask_holes_do_not_preserve_source_text():
    image = Image.new("RGB", (300, 200), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((120, 90, 180, 120), fill="black")

    bubble_mask = Image.new("L", (240, 140), 255)
    mask_draw = ImageDraw.Draw(bubble_mask)
    # Simulate holes caused by dark glyph pixels
    mask_draw.rectangle((70, 50, 130, 80), fill=0)

    text_box = Box(120, 90, 60, 30)
    bubble_box = Box(60, 40, 240, 140)
    erase_box = Box(108, 78, 84, 54)

    erase_original_text(
        page_img=image,
        draw=draw,
        text_box=text_box,
        bubble_box=bubble_box,
        erase_box=erase_box,
        bubble_mask=bubble_mask,
        config=config,
    )

    assert image.getpixel((140, 100)) != (0, 0, 0)


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
        "pages": [{"page_number": 1, "filename": "001.png", "original_filename": "001.png", "skip": False, "width": 1280, "height": 1707}],
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
    (ws / "stage1_detection" / "detection.json").write_text(json.dumps(det), encoding="utf-8")

    ocr = {
        "chapter_id": "test_ch",
        "stage": "ocr",
        "results": [{"region_id": "P001_R001", "page_number": 1, "has_usable_text": True}],
    }
    (ws / "stage2_ocr").mkdir(parents=True, exist_ok=True)
    (ws / "stage2_ocr" / "ocr.json").write_text(json.dumps(ocr), encoding="utf-8")

    para = {
        "chapter_id": "test_ch",
        "stage": "paraphrase",
        "results": [{"region_id": "P001_R001", "page_number": 1, "final_text": "It's right here!"}],
    }
    (ws / "stage4_paraphrase").mkdir(parents=True, exist_ok=True)
    (ws / "stage4_paraphrase" / "paraphrase.json").write_text(json.dumps(para), encoding="utf-8")

    run_render(str(ws), config, run_id="run-test-123")

    render_report_path = ws / "stage5_render" / "render.json"
    assert render_report_path.exists()

    report = json.loads(render_report_path.read_text(encoding="utf-8"))
    res = report["results"][0]

    assert res["rendered"] is True
    assert res["erase_strategy"] == "padded_text_rectangle"
    assert res["text_box"] == {"x": 566, "y": 874, "w": 334, "h": 99}
    assert res["bubble_box"] == {"x": 310, "y": 600, "w": 779, "h": 487}
    assert res["layout_box"] != res["bubble_box"]


def test_render_events_receive_entrypoint_run_id(tmp_path):
    ws = tmp_path / "workspace"
    ws.mkdir(parents=True, exist_ok=True)
    manifest = {
        "chapter_id": "test_ch",
        "total_pages": 1,
        "pages": [{"page_number": 1, "filename": "001.png", "original_filename": "001.png", "skip": False, "width": 1000, "height": 1000}],
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
    (ws / "stage1_detection" / "detection.json").write_text(json.dumps(det), encoding="utf-8")

    ocr = {
        "chapter_id": "test_ch",
        "stage": "ocr",
        "results": [{"region_id": "P001_R001", "page_number": 1, "has_usable_text": True}],
    }
    (ws / "stage2_ocr").mkdir(parents=True, exist_ok=True)
    (ws / "stage2_ocr" / "ocr.json").write_text(json.dumps(ocr), encoding="utf-8")

    para = {
        "chapter_id": "test_ch",
        "stage": "paraphrase",
        "results": [{"region_id": "P001_R001", "page_number": 1, "final_text": "Hello!"}],
    }
    (ws / "stage4_paraphrase").mkdir(parents=True, exist_ok=True)
    (ws / "stage4_paraphrase" / "paraphrase.json").write_text(json.dumps(para), encoding="utf-8")

    target_run_id = "test-run-id-999"
    run_render(str(ws), config, run_id=target_run_id)

    event_file = ws / "logs" / "events.jsonl"
    assert event_file.exists()
    events = [json.loads(line) for line in event_file.read_text(encoding="utf-8").splitlines() if line.strip()]

    assert len(events) >= 2
    for event in events:
        assert event["run_id"] == target_run_id

    region_ev = next(e for e in events if e["event"] == RENDER_REGION_COMPLETED)
    assert region_ev["metadata"]["erase_strategy"] == "padded_text_rectangle"
    assert region_ev["metadata"]["text_box"] == {"x": 100, "y": 100, "w": 200, "h": 50}
