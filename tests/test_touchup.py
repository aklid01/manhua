import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PIL import Image, ImageDraw
from PyQt6.QtCore import QRectF, Qt
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import QApplication

from touchup.app import TouchupApp
from touchup.cache import TouchupCache
from touchup.canvas import DraggableTextItem
from touchup.inpainter import build_text_erase_mask, inpaint_text_regions
from touchup.segmenter import SAMSegmenter
from touchup.translator import load_glossary, translate_text


@pytest.fixture(scope="module")
def qt_app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def touchup_window(qt_app):
    window = TouchupApp()
    yield window
    window.close()
    window.deleteLater()


def _rgb(color: QColor) -> tuple[int, int, int]:
    return color.red(), color.green(), color.blue()


def test_touchup_cache(tmp_path):
    cache = TouchupCache(tmp_path)
    edit = {
        "source_box": [10, 20, 100, 50],
        "original_text": "【万灵药】",
        "translation": "[Panacea]",
        "render": {"x": 10, "y": 20, "font_size": 20},
    }
    edit_id = cache.save_edit("001.png", edit)
    assert edit_id is not None

    edits = cache.get_edits("001.png")
    assert len(edits) == 1
    assert edits[0]["translation"] == "[Panacea]"

    # Update edit
    edit["translation"] = "[All-Healing Pill]"
    cache.save_edit("001.png", edit)
    edits = cache.get_edits("001.png")
    assert len(edits) == 1
    assert edits[0]["translation"] == "[All-Healing Pill]"

    # Delete edit
    assert cache.delete_edit("001.png", edit_id) is True
    assert len(cache.get_edits("001.png")) == 0


def test_touchup_selection_syncs_selected_bubble_and_resets_target(touchup_window):
    first = touchup_window.canvas.add_text_item("first\nline", QRectF(10, 10, 120, 80))
    first.set_line_style(
        -1,
        family="Impact",
        size=34,
        color=QColor(255, 215, 0),
        stroke_width=4,
        align=Qt.AlignmentFlag.AlignCenter,
        direction="vertical",
    )

    second = touchup_window.canvas.add_text_item("second", QRectF(20, 20, 120, 80))
    second.set_line_style(
        -1,
        family="Tahoma",
        size=18,
        color=QColor(0, 229, 255),
        stroke_width=1,
        align=Qt.AlignmentFlag.AlignRight,
        direction="horizontal",
    )

    touchup_window._on_item_selected(first)
    touchup_window.cb_target_line.setCurrentIndex(1)
    touchup_window._on_item_selected(second)

    assert touchup_window.cb_target_line.currentIndex() == 0
    assert touchup_window.cb_font.currentText() == "Tahoma"
    assert touchup_window.spn_font_size.value() == 18
    assert touchup_window.spn_stroke_width.value() == 1
    assert touchup_window.cb_align.currentText() == "Right"
    assert touchup_window.cb_direction.currentText() == "Horizontal"
    assert _rgb(touchup_window._text_color) == (0, 229, 255)


def test_touchup_size_change_does_not_overwrite_other_style(touchup_window):
    item = touchup_window.canvas.add_text_item("styled", QRectF(10, 10, 120, 80))
    item.set_line_style(
        -1,
        family="Impact",
        size=30,
        color=QColor(255, 215, 0),
        stroke_color=QColor(120, 10, 20),
        stroke_width=5,
        align=Qt.AlignmentFlag.AlignCenter,
        direction="vertical",
    )

    touchup_window._on_item_selected(item)
    touchup_window.spn_font_size.setValue(44)

    line = item._lines_data[0]
    assert line["font_size"] == 44
    assert line["font_family"] == "Impact"
    assert _rgb(line["color"]) == (255, 215, 0)
    assert _rgb(line["stroke_color"]) == (120, 10, 20)
    assert line["stroke_width"] == 5
    assert line["align"] == Qt.AlignmentFlag.AlignCenter
    assert line["direction"] == "vertical"


def test_touchup_mixed_state_does_not_flatten_until_field_changes(touchup_window):
    item = touchup_window.canvas.add_text_item("one\ntwo", QRectF(10, 10, 160, 100))
    item.set_line_style(
        0,
        family="Impact",
        size=22,
        color=QColor(255, 255, 255),
        stroke_width=2,
    )
    item.set_line_style(
        1,
        family="Tahoma",
        size=36,
        color=QColor(0, 229, 255),
        stroke_width=6,
    )

    touchup_window._on_item_selected(item)

    assert touchup_window.cb_target_line.currentIndex() == 0
    assert touchup_window.cb_font.currentText() == touchup_window.MIXED_STYLE_LABEL
    assert touchup_window.btn_text_color.text() == "Mixed…"
    assert item._lines_data[0]["font_size"] == 22
    assert item._lines_data[1]["font_size"] == 36

    touchup_window.cb_font.setCurrentText("Arial")

    assert item._lines_data[0]["font_family"] == "Arial"
    assert item._lines_data[1]["font_family"] == "Arial"
    assert item._lines_data[0]["font_size"] == 22
    assert item._lines_data[1]["font_size"] == 36
    assert _rgb(item._lines_data[0]["color"]) == (255, 255, 255)
    assert _rgb(item._lines_data[1]["color"]) == (0, 229, 255)


def test_touchup_cached_line_styles_survive_restore(touchup_window, tmp_path):
    touchup_window.cache = TouchupCache(tmp_path)
    touchup_window.current_image_path = tmp_path / "001.png"
    touchup_window.cache.save_edit(
        "001.png",
        {
            "source_box": [0, 0, 100, 60],
            "translation": "one\ntwo",
            "render": {
                "x": 10,
                "y": 20,
                "w": 120,
                "h": 80,
                "font_family": "Arial",
                "font_size": 20,
                "align": "left",
                "direction": "horizontal",
                "lines": [
                    {
                        "text": "one",
                        "font_family": "Impact",
                        "font_size": 28,
                        "color": [255, 215, 0],
                        "stroke_color": [0, 0, 0],
                        "stroke_width": 3,
                        "align": "center",
                        "direction": "vertical",
                    },
                    {
                        "text": "two",
                        "font_family": "Tahoma",
                        "font_size": 18,
                        "color": [0, 229, 255],
                        "stroke_color": [120, 10, 20],
                        "stroke_width": 1,
                        "align": "right",
                        "direction": "horizontal",
                    },
                ],
            },
        },
    )

    touchup_window._load_cached_edits()

    items = [
        item
        for item in touchup_window.canvas.scene.items()
        if isinstance(item, DraggableTextItem)
    ]
    assert len(items) == 1
    restored = items[0]
    assert restored._lines_data[0]["align"] == Qt.AlignmentFlag.AlignCenter
    assert restored._lines_data[0]["direction"] == "vertical"
    assert restored._lines_data[1]["align"] == Qt.AlignmentFlag.AlignRight
    assert _rgb(restored._lines_data[1]["stroke_color"]) == (120, 10, 20)


def test_inpainter_text_erasure():
    img = Image.new("RGB", (200, 200), color=(255, 200, 100))
    draw = ImageDraw.Draw(img)
    draw.rectangle((50, 50, 150, 100), fill=(0, 0, 0))

    text_boxes = [(50, 50, 150, 100)]
    sam_mask = np.ones((200, 200), dtype=bool)

    mask = build_text_erase_mask(
        (200, 200), text_boxes, sam_mask=sam_mask, padding_px=4
    )
    assert mask.shape == (200, 200)
    assert mask[75, 100] == 255
    assert mask[10, 10] == 0

    inpainted_img, out_mask = inpaint_text_regions(img, text_boxes, sam_mask=sam_mask)
    assert inpainted_img.size == (200, 200)
    # Text region should no longer be pure black
    pixel = inpainted_img.getpixel((75, 75))
    assert pixel != (0, 0, 0)


def test_sam_segmenter_box():
    segmenter = SAMSegmenter()
    img = Image.new("RGB", (100, 100), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.rectangle((20, 20, 80, 80), fill=(200, 50, 50))

    mask = segmenter.segment_box(img, (20, 20, 80, 80))
    assert mask.shape == (100, 100)
    assert mask.dtype == bool


def test_glossary_lookup(tmp_path):
    glossary = {
        "version": "v1",
        "terms": [
            {"source_term": "万灵药", "target_term": "Panacea"},
        ],
    }
    import json

    (tmp_path / "glossary.json").write_text(json.dumps(glossary), encoding="utf-8")

    terms = load_glossary(tmp_path)
    assert len(terms) == 1
    assert terms[0]["source_term"] == "万灵药"

    translated = translate_text("万灵药", series_dir=tmp_path)
    assert translated == "Panacea"
