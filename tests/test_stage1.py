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


class _YoloConfig:
    DETECTOR_BACKEND = "yolov8"

    def __getattr__(self, name):
        return getattr(config, name)


_yolo_config = _YoloConfig()


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
        run_detection(str(ws), _yolo_config)
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
        run_detection(str(ws), _yolo_config)
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
        run_detection(str(ws), _yolo_config)
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
        run_detection(str(ws), _yolo_config)
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
        run_detection(str(ws), _yolo_config)

    det = json.loads((ws / "stage1_detection" / "detection.json").read_text())
    assert det["regions"] == []  # no regions detected
    overlay = ws / "stage1_detection" / "overlays" / "001_overlay.png"
    assert overlay.exists()  # but the clean overlay is still saved (Fix 1)


@patch("manhua_pipeline.stages.stage1_detection._resolve_model", side_effect=lambda x, *args: x)
def test_detection_rtdetr_backend(mock_resolve, tmp_path):
    """Test RT-DETR backend integration: schema compatibility, parent matching, reading order."""
    import torch

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

    # Configure config for RT-DETR
    class MockConfig:
        DETECTOR_BACKEND = "rtdetr"
        RTDETR_REPO = "ogkalu/comic-text-and-bubble-detector"
        RTDETR_CONF = 0.30
        RTDETR_CLASS_BUBBLE = 0
        RTDETR_CLASS_TEXT_BUBBLE = 1
        RTDETR_CLASS_TEXT_FREE = 2
        STITCH_ENABLED = False

        def __getattr__(self, name):
            return getattr(config, name)

    cfg = MockConfig()

    # Setup mock processor and model
    mock_processor = MagicMock()
    mock_model = MagicMock()
    mock_model.device = "cpu"

    mock_outputs = MagicMock()
    mock_model.return_value = mock_outputs

    # Mock predictions:
    # classes: 1: text_bubble, 0: bubble, 2: text_free
    scores = torch.tensor([0.95, 0.92, 0.85, 0.90])
    labels = torch.tensor([1, 0, 2, 1])
    boxes = torch.tensor([
        [100.0, 150.0, 200.0, 250.0],  # text_bubble 1 (y=150, inside bubble)
        [90.0, 140.0, 250.0, 260.0],   # bubble 0 (y=140)
        [400.0, 50.0, 500.0, 100.0],   # text_free 2 (y=50, narration/SFX)
        [110.0, 450.0, 210.0, 550.0],  # text_bubble 1 (y=450, no bubble)
    ])
    mock_processor.post_process_object_detection.return_value = [{
        "scores": scores,
        "labels": labels,
        "boxes": boxes
    }]

    with patch("manhua_pipeline.stages.stage1_detection._get_rtdetr", return_value=(mock_model, mock_processor)):
        with patch("manhua_pipeline.stages.stage1_detection.YOLO") as MockYOLO:
            run_detection(str(ws), cfg)
            # YOLO should not be initialized
            MockYOLO.assert_not_called()

    det = json.loads((ws / "stage1_detection" / "detection.json").read_text())
    assert det["stage"] == "detection"
    assert det["model"] == "ogkalu/comic-text-and-bubble-detector"

    regs = det["regions"]
    # We should have 3 regions (2 speech_bubbles, 1 narration). Bubble is just a parent container, not a region.
    assert len(regs) == 3

    # Assert correct reading order sorting (by y coordinate first: 50 -> 150 -> 450)
    # 1. Narration (y=50)
    assert regs[0]["region_id"] == "P001_R001"
    assert regs[0]["type"] == "narration"
    assert regs[0]["bbox"]["y"] == 50
    assert regs[0]["is_free_text"] is True
    assert regs[0]["parent_bubble"] is None

    # 2. First text bubble (y=150)
    assert regs[1]["region_id"] == "P001_R002"
    assert regs[1]["type"] == "speech_bubble"
    assert regs[1]["bbox"]["y"] == 150
    assert regs[1]["is_free_text"] is False
    # Parent bubble mapping assertion
    assert regs[1]["parent_bubble"] is not None
    assert regs[1]["parent_bubble"]["x"] == 90
    assert regs[1]["parent_bubble"]["y"] == 140
    assert regs[1]["parent_bubble"]["w"] == 160  # 250 - 90
    assert regs[1]["parent_bubble"]["h"] == 120  # 260 - 140

    # 3. Second text bubble (y=450)
    assert regs[2]["region_id"] == "P001_R003"
    assert regs[2]["type"] == "speech_bubble"
    assert regs[2]["bbox"]["y"] == 450
    assert regs[2]["is_free_text"] is False
    assert regs[2]["parent_bubble"] is None

    # Verify all expected keys are present in all regions (schema validation check)
    expected_keys = {
        "region_id", "page_number", "type", "bbox", "reading_order",
        "style_hint", "confidence", "read_region", "erase_mask", "render",
        "is_free_text", "parent_bubble", "text_direction"
    }
    for r in regs:
        assert expected_keys.issubset(r)
        assert r["erase_mask"]["type"] == "rect"
        assert r["erase_mask"]["coords"] == [r["bbox"]["x"], r["bbox"]["y"], r["bbox"]["w"], r["bbox"]["h"]]

    # Verify visual debug overlay is written
    overlay = ws / "stage1_detection" / "overlays" / "001_overlay.png"
    assert overlay.exists()

