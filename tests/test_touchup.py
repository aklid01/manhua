import numpy as np
from PIL import Image, ImageDraw

from touchup.cache import TouchupCache
from touchup.inpainter import build_text_erase_mask, inpaint_text_regions
from touchup.segmenter import SAMSegmenter
from touchup.translator import load_glossary, translate_text


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
