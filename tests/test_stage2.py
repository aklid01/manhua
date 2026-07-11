import json
from unittest.mock import MagicMock, patch

from PIL import Image

import config


def _setup(ws, page_h=1214, regions=None):
    (ws / "pages").mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (860, page_h)).save(ws / "pages" / "001.png")
    manifest = {
        "chapter_id": "t",
        "total_pages": 1,
        "pages": [
            {
                "page_number": 1,
                "filename": "001.png",
                "original_filename": "a.jpg",
                "width": 860,
                "height": page_h,
                "skip": False,
                "skip_reason": None,
                "global_y_offset": None,
            }
        ],
        "current_stage": "ocr",
        "completed_stages": ["import", "detect"],
        "warning_count": 0,
        "status": "in_progress",
    }
    (ws / "manifest.json").write_text(json.dumps(manifest))
    (ws / "stage1_detection").mkdir(parents=True, exist_ok=True)
    det = {
        "chapter_id": "t",
        "stage": "detection",
        "generated_at": "now",
        "model": "m",
        "regions": regions or [],
    }
    (ws / "stage1_detection" / "detection.json").write_text(json.dumps(det))


def _region(rid, x, y, w, h, page=1, rtype="speech_bubble"):
    return {
        "region_id": rid,
        "page_number": page,
        "type": rtype,
        "bbox": {"x": x, "y": y, "w": w, "h": h},
        "reading_order": 1,
        "style_hint": "round",
        "confidence": 0.9,
        "read_region": {"x": x, "y": y, "w": w, "h": h},
        "erase_mask": {"type": "rect", "coords": [x, y, w, h]},
        "render": True,
    }


def test_ocr_reads_text_and_flags(tmp_path):
    from manhua_pipeline.stages.stage2_ocr import run_ocr

    ws = tmp_path / "workspace"
    _setup(ws, regions=[_region("P001_R001", 100, 300, 200, 120)])
    with (
        patch("manhua_pipeline.stages.stage2_ocr._get_ocr") as mock_get,
        patch(
            "manhua_pipeline.stages.stage2_ocr._read_crop",
            return_value=("滚吧！", 0.94, 0.94),
        ),
    ):
        mock_get.return_value = MagicMock()
        run_ocr(str(ws), config)
    ocr = json.loads((ws / "stage2_ocr" / "ocr.json").read_text(encoding="utf-8"))
    e = ocr["results"][0]
    assert e["region_id"] == "P001_R001"
    assert e["original_text"] == "滚吧！"
    assert e["ocr_confidence"] == 0.94
    assert e["ocr_confidence_min"] == 0.94
    assert e["has_usable_text"] is True
    assert e["needs_correction"] is False
    assert e["edge_touching"] is False
    for k in [
        "region_id",
        "page_number",
        "type",
        "original_text",
        "text_direction",
        "ocr_confidence",
        "ocr_confidence_min",
        "has_usable_text",
        "do_not_render",
        "needs_correction",
        "edge_touching",
        "edge",
    ]:
        assert k in e


def test_ocr_low_confidence_flags_correction(tmp_path):
    from manhua_pipeline.stages.stage2_ocr import run_ocr

    ws = tmp_path / "workspace"
    _setup(ws, regions=[_region("P001_R001", 100, 300, 200, 120)])
    with (
        patch(
            "manhua_pipeline.stages.stage2_ocr._get_ocr",
            return_value=MagicMock(),
        ),
        patch(
            "manhua_pipeline.stages.stage2_ocr._read_crop",
            return_value=("朝阳集团", 0.41, 0.41),
        ),
    ):
        run_ocr(str(ws), config)
    e = json.loads((ws / "stage2_ocr" / "ocr.json").read_text(encoding="utf-8"))[
        "results"
    ][0]
    assert e["ocr_confidence"] == 0.41
    assert e["needs_correction"] is True


def test_ocr_edge_touching_split_bubble(tmp_path):
    from manhua_pipeline.stages.stage2_ocr import run_ocr

    ws = tmp_path / "workspace"
    # region flush to bottom edge: y + h == page_h
    _setup(ws, page_h=1000, regions=[_region("P001_R001", 100, 950, 200, 50)])
    with (
        patch(
            "manhua_pipeline.stages.stage2_ocr._get_ocr",
            return_value=MagicMock(),
        ),
        patch(
            "manhua_pipeline.stages.stage2_ocr._read_crop",
            return_value=("", 0.0, 0.0),
        ),
    ):
        run_ocr(str(ws), config)
    e = json.loads((ws / "stage2_ocr" / "ocr.json").read_text(encoding="utf-8"))[
        "results"
    ][0]
    assert e["edge_touching"] is True
    assert e["edge"] in ("bottom", "both")
    assert e["has_usable_text"] is False
    assert e["needs_correction"] is True
    assert e["note"]  # non-empty split note


def test_ocr_clamps_out_of_bounds(tmp_path):
    from manhua_pipeline.stages.stage2_ocr import run_ocr

    ws = tmp_path / "workspace"
    # bbox extends beyond width (860) and height
    _setup(ws, page_h=1000, regions=[_region("P001_R001", 800, 980, 200, 100)])
    with (
        patch(
            "manhua_pipeline.stages.stage2_ocr._get_ocr",
            return_value=MagicMock(),
        ),
        patch(
            "manhua_pipeline.stages.stage2_ocr._read_crop",
            return_value=("x", 0.8, 0.8),
        ),
    ):
        run_ocr(str(ws), config)  # must not raise
    ocr = json.loads((ws / "stage2_ocr" / "ocr.json").read_text(encoding="utf-8"))
    assert len(ocr["results"]) == 1


def test_ocr_zero_regions(tmp_path):
    from manhua_pipeline.stages.stage2_ocr import run_ocr

    ws = tmp_path / "workspace"
    _setup(ws, regions=[])
    with patch("manhua_pipeline.stages.stage2_ocr._get_ocr", return_value=MagicMock()):
        run_ocr(str(ws), config)
    ocr = json.loads((ws / "stage2_ocr" / "ocr.json").read_text(encoding="utf-8"))
    assert ocr["results"] == []
    m = json.loads((ws / "manifest.json").read_text(encoding="utf-8"))
    assert m["current_stage"] == "translate"
    assert "ocr" in m["completed_stages"]
