import json
from unittest.mock import MagicMock, patch

from PIL import Image

import config


def _make_manifest(ws, pages):
    (ws / "pages").mkdir(parents=True, exist_ok=True)
    for p in pages:
        if p["filename"]:
            Image.new("RGB", (p["width"], p["height"])).save(
                ws / "pages" / p["filename"]
            )
    manifest = {
        "chapter_id": "test_ch",
        "input_format": "paginated",
        "total_pages": len(pages),
        "pages": pages,
        "current_stage": "detection",
        "completed_stages": ["import"],
        "warning_count": 0,
        "status": "in_progress",
    }
    (ws / "manifest.json").write_text(json.dumps(manifest))
    return manifest


class _FakeBox:
    def __init__(self, xyxy, conf, cls_val=0):
        self.xyxy = [xyxy]  # mimic ultralytics tensor-ish access
        self.conf = [conf]
        mock_cls = MagicMock()
        mock_cls.item.return_value = cls_val
        self.cls = [mock_cls]


def _fake_predict_two_boxes(*args, **kwargs):
    # one result, two boxes: lower one FIRST to test sorting
    result = MagicMock()
    result.boxes = [
        _FakeBox((100, 400, 300, 500), 0.9),  # lower (y=400)
        _FakeBox((120, 80, 320, 180), 0.8),  # upper (y=80)
    ]
    result.names = {0: "speech_bubble", 1: "narration"}
    return [result]


@patch(
    "manhua_pipeline.stages.stage1_detection._resolve_model",
    side_effect=lambda x, *args: x,
)
def test_detection_builds_regions_in_reading_order(mock_resolve, tmp_path):
    from manhua_pipeline.stages.stage1_detection import run_detection

    ws = tmp_path / "workspace"
    pages = [
        {
            "page_number": 1,
            "filename": "001.png",
            "original_filename": "a.jpg",
            "width": 860,
            "height": 1214,
            "skip": False,
            "skip_reason": None,
            "global_y_offset": None,
        }
    ]
    _make_manifest(ws, pages)
    with patch("manhua_pipeline.stages.stage1_detection.YOLO") as MockYOLO:
        MockYOLO.return_value.predict.side_effect = _fake_predict_two_boxes
        run_detection(str(ws), config)
    det = json.loads((ws / "stage1_detection" / "detection.json").read_text())
    assert det["stage"] == "detection"
    regs = det["regions"]
    assert len(regs) == 2
    # upper box (y=80) should be reading_order 1
    assert regs[0]["reading_order"] == 1
    assert regs[0]["bbox"]["y"] == 80
    assert regs[1]["bbox"]["y"] == 400
    assert regs[0]["region_id"] == "P001_R001"
    for r in regs:
        assert {
            "region_id",
            "page_number",
            "type",
            "bbox",
            "reading_order",
            "style_hint",
            "confidence",
            "read_region",
            "erase_mask",
            "render",
        }.issubset(r)
        assert r["erase_mask"]["type"] == "rect"


@patch(
    "manhua_pipeline.stages.stage1_detection._resolve_model",
    side_effect=lambda x, *args: x,
)
def test_detection_skips_unusable_pages(mock_resolve, tmp_path):
    from manhua_pipeline.stages.stage1_detection import run_detection

    ws = tmp_path / "workspace"
    pages = [
        {
            "page_number": 1,
            "filename": None,
            "original_filename": "bad.jpg",
            "width": 0,
            "height": 0,
            "skip": True,
            "skip_reason": "read_error",
            "global_y_offset": None,
        },
        {
            "page_number": 2,
            "filename": "002.png",
            "original_filename": "b.jpg",
            "width": 860,
            "height": 1214,
            "skip": False,
            "skip_reason": None,
            "global_y_offset": None,
        },
    ]
    _make_manifest(ws, pages)
    with patch("manhua_pipeline.stages.stage1_detection.YOLO") as MockYOLO:
        MockYOLO.return_value.predict.side_effect = _fake_predict_two_boxes
        run_detection(str(ws), config)
        # predict called only once (page 2), never for the skipped page
        assert MockYOLO.return_value.predict.call_count == 1
    det = json.loads((ws / "stage1_detection" / "detection.json").read_text())
    assert all(r["page_number"] == 2 for r in det["regions"])


@patch(
    "manhua_pipeline.stages.stage1_detection._resolve_model",
    side_effect=lambda x, *args: x,
)
def test_detection_zero_boxes(mock_resolve, tmp_path):
    from manhua_pipeline.stages.stage1_detection import run_detection

    ws = tmp_path / "workspace"
    pages = [
        {
            "page_number": 1,
            "filename": "001.png",
            "original_filename": "a.jpg",
            "width": 860,
            "height": 1214,
            "skip": False,
            "skip_reason": None,
            "global_y_offset": None,
        }
    ]
    _make_manifest(ws, pages)

    def _empty_predict(*a, **k):
        r = MagicMock()
        r.boxes = []
        r.names = {}
        return [r]

    with patch("manhua_pipeline.stages.stage1_detection.YOLO") as MockYOLO:
        MockYOLO.return_value.predict.side_effect = _empty_predict
        run_detection(str(ws), config)
    det = json.loads((ws / "stage1_detection" / "detection.json").read_text())
    assert det["regions"] == []


@patch(
    "manhua_pipeline.stages.stage1_detection._resolve_model",
    side_effect=lambda x, *args: x,
)
def test_detection_model_loaded_once(mock_resolve, tmp_path):
    from manhua_pipeline.stages.stage1_detection import run_detection

    ws = tmp_path / "workspace"
    pages = [
        {
            "page_number": i,
            "filename": f"{i:03d}.png",
            "original_filename": f"{i}.jpg",
            "width": 860,
            "height": 1214,
            "skip": False,
            "skip_reason": None,
            "global_y_offset": None,
        }
        for i in range(1, 4)
    ]
    _make_manifest(ws, pages)
    with patch("manhua_pipeline.stages.stage1_detection.YOLO") as MockYOLO:
        MockYOLO.return_value.predict.side_effect = _fake_predict_two_boxes
        run_detection(str(ws), config)
        MockYOLO.assert_called_once()  # model constructed exactly once
        assert MockYOLO.return_value.predict.call_count == 3


@patch(
    "manhua_pipeline.stages.stage1_detection._resolve_model",
    side_effect=lambda x, *args: x,
)
def test_detection_overlay_saved_when_zero_regions(mock_resolve, tmp_path):
    """A processed page with zero detections must still produce an overlay PNG."""
    from manhua_pipeline.stages.stage1_detection import run_detection

    ws = tmp_path / "workspace"
    pages = [
        {
            "page_number": 1,
            "filename": "001.png",
            "original_filename": "a.jpg",
            "width": 860,
            "height": 1214,
            "skip": False,
            "skip_reason": None,
            "global_y_offset": None,
        }
    ]
    _make_manifest(ws, pages)

    def _empty_predict(*a, **k):
        r = MagicMock()
        r.boxes = []
        r.names = {}
        return [r]

    with patch("manhua_pipeline.stages.stage1_detection.YOLO") as MockYOLO:
        MockYOLO.return_value.predict.side_effect = _empty_predict
        run_detection(str(ws), config)

    det = json.loads((ws / "stage1_detection" / "detection.json").read_text())
    assert det["regions"] == []  # no regions detected
    overlay = ws / "stage1_detection" / "overlays" / "001_overlay.png"
    assert overlay.exists()  # but the clean overlay is still saved (Fix 1)
