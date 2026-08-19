# Unit tests for Stage 1 text and speech bubble detection.

import json
from unittest.mock import MagicMock, patch

import torch
from PIL import Image

import config


# Helper to generate dummy workspace images and manifest for detection tests
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


# Mock YOLO bounding box tensor wrapper
class _FakeBox:
    def __init__(self, xyxy, conf, cls_val=0):
        self.xyxy = [xyxy]
        self.conf = [conf]
        mock_cls = MagicMock()
        mock_cls.item.return_value = cls_val
        self.cls = [mock_cls]


# Mock prediction returning two bounding boxes for reading order verification
def _fake_predict_two_boxes(*args, **kwargs):
    result = MagicMock()
    result.boxes = [
        _FakeBox((100, 400, 300, 500), 0.9),
        _FakeBox((120, 80, 320, 180), 0.8),
    ]
    result.names = {0: "speech_bubble", 1: "narration"}
    return [result]


# Mock config helper for YOLO detection parameters
class _YoloConfig:
    DETECTOR_BACKEND = "yolov8"
    OVERLAY_ENABLED = True

    def __getattr__(self, name):
        return getattr(config, name)


_yolo_config = _YoloConfig()


# Verify detected regions are sorted top-to-bottom and formatted to schema
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


# Verify skipped pages are omitted from detector prediction calls
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
        assert MockYOLO.return_value.predict.call_count == 1
    det = json.loads((ws / "stage1_detection" / "detection.json").read_text())
    assert all(r["page_number"] == 2 for r in det["regions"])


# Verify detection outputs empty region list when no bubbles are found
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


# Verify detection model is initialized once across multi-page chapter runs
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
        MockYOLO.assert_called_once()
        assert MockYOLO.return_value.predict.call_count == 3


# Verify debug overlay image is saved even when zero regions are detected
@patch(
    "manhua_pipeline.stages.stage1_detection._resolve_model",
    side_effect=lambda x, *args: x,
)
def test_detection_overlay_saved_when_zero_regions(mock_resolve, tmp_path):
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
    overlay = ws / "stage1_detection" / "overlays" / "001_overlay.png"
    assert overlay.exists()


# Verify RT-DETR model detection, parent-bubble association, and reading order sorting
@patch(
    "manhua_pipeline.stages.stage1_detection._resolve_model",
    side_effect=lambda x, *args: x,
)
def test_detection_rtdetr_backend(mock_resolve, tmp_path):
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

    class MockConfig:
        DETECTOR_BACKEND = "rtdetr"
        RTDETR_REPO = "ogkalu/comic-text-and-bubble-detector"
        RTDETR_CONF = 0.30
        RTDETR_CLASS_BUBBLE = 0
        RTDETR_CLASS_TEXT_BUBBLE = 1
        RTDETR_CLASS_TEXT_FREE = 2
        STITCH_ENABLED = False
        OVERLAY_ENABLED = True

        def __getattr__(self, name):
            return getattr(config, name)

    cfg = MockConfig()

    mock_processor = MagicMock()
    mock_model = MagicMock()
    mock_model.device = "cpu"
    mock_outputs = MagicMock()
    mock_model.return_value = mock_outputs

    scores = torch.tensor([0.95, 0.92, 0.85, 0.90])
    labels = torch.tensor([1, 0, 2, 1])
    boxes = torch.tensor(
        [
            [100.0, 150.0, 200.0, 250.0],
            [90.0, 140.0, 250.0, 260.0],
            [400.0, 50.0, 500.0, 100.0],
            [110.0, 450.0, 210.0, 550.0],
        ]
    )
    mock_processor.post_process_object_detection.return_value = [
        {"scores": scores, "labels": labels, "boxes": boxes}
    ]

    with patch(
        "manhua_pipeline.stages.stage1_detection._get_rtdetr",
        return_value=(mock_model, mock_processor),
    ):
        with patch("manhua_pipeline.stages.stage1_detection.YOLO") as MockYOLO:
            run_detection(str(ws), cfg)
            MockYOLO.assert_not_called()

    det = json.loads((ws / "stage1_detection" / "detection.json").read_text())
    assert det["stage"] == "detection"
    assert det["model"] == "ogkalu/comic-text-and-bubble-detector"

    regs = det["regions"]
    assert len(regs) == 3

    assert regs[0]["region_id"] == "P001_R001"
    assert regs[0]["type"] == "narration"
    assert regs[0]["bbox"]["y"] == 50
    assert regs[0]["is_free_text"] is True
    assert regs[0]["parent_bubble"] is None

    assert regs[1]["region_id"] == "P001_R002"
    assert regs[1]["type"] == "speech_bubble"
    assert regs[1]["bbox"]["y"] == 150
    assert regs[1]["is_free_text"] is False
    assert regs[1]["parent_bubble"] is not None
    assert regs[1]["parent_bubble"]["x"] == 90
    assert regs[1]["parent_bubble"]["y"] == 140
    assert regs[1]["parent_bubble"]["w"] == 160
    assert regs[1]["parent_bubble"]["h"] == 120

    assert regs[2]["region_id"] == "P001_R003"
    assert regs[2]["type"] == "speech_bubble"
    assert regs[2]["bbox"]["y"] == 450
    assert regs[2]["is_free_text"] is False
    assert regs[2]["parent_bubble"] is None

    expected_keys = {
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
        "is_free_text",
        "parent_bubble",
        "text_direction",
    }
    for r in regs:
        assert expected_keys.issubset(r)
        assert r["erase_mask"]["type"] == "rect"
        target_box = r["parent_bubble"] or r["bbox"]
        assert r["erase_mask"]["coords"] == [
            target_box["x"],
            target_box["y"],
            target_box["w"],
            target_box["h"],
        ]

    overlay = ws / "stage1_detection" / "overlays" / "001_overlay.png"
    assert overlay.exists()


# Verify intersection over smaller (IoS) metric detects offset nested duplicate boxes
def test_ios_catches_offset_size_mismatch_duplicate():
    from manhua_pipeline.stages.stage1_detection import _ios, _iou

    a = {"x": 76, "y": 1042, "w": 926, "h": 610}
    b = {"x": 85, "y": 1043, "w": 813, "h": 617}
    assert _iou(a, b) < 0.90
    assert _ios(a, b) >= 0.85


# Verify duplicate box deduplication retains higher-confidence box
def test_dedupe_drops_lower_confidence_ios_duplicate():
    from manhua_pipeline.stages.stage1_detection import _dedupe_raw_boxes

    class _Cfg:
        DEDUP_IOU_THRESHOLD = 0.90
        DEDUP_IOS_THRESHOLD = 0.85

    regions = [
        {
            "region_id": "P006_R002",
            "page_number": 6,
            "type": "speech_bubble",
            "bbox": {"x": 76, "y": 1042, "w": 926, "h": 610},
            "confidence": 0.73,
            "reading_order": 2,
        },
        {
            "region_id": "P006_R004",
            "page_number": 6,
            "type": "speech_bubble",
            "bbox": {"x": 85, "y": 1043, "w": 813, "h": 617},
            "confidence": 0.33,
            "reading_order": 4,
        },
    ]
    kept = _dedupe_raw_boxes(regions, _Cfg())
    assert len(kept) == 1
    assert kept[0]["region_id"] == "P006_R002"


# Verify disjoint speech bubbles are not merged by deduplication
def test_ios_does_not_merge_distinct_bubbles():
    from manhua_pipeline.stages.stage1_detection import _dedupe_raw_boxes

    class _Cfg:
        DEDUP_IOU_THRESHOLD = 0.90
        DEDUP_IOS_THRESHOLD = 0.85

    regions = [
        {
            "region_id": "P010_R001",
            "page_number": 10,
            "type": "speech_bubble",
            "bbox": {"x": 95, "y": 420, "w": 482, "h": 184},
            "confidence": 0.95,
            "reading_order": 1,
        },
        {
            "region_id": "P010_R002",
            "page_number": 10,
            "type": "speech_bubble",
            "bbox": {"x": 344, "y": 1336, "w": 628, "h": 429},
            "confidence": 0.97,
            "reading_order": 2,
        },
    ]
    kept = _dedupe_raw_boxes(regions, _Cfg())
    assert len(kept) == 2


# Verify overlapping boxes with different class labels are kept separate
def test_ios_does_not_merge_across_classes():
    from manhua_pipeline.stages.stage1_detection import _dedupe_raw_boxes

    class _Cfg:
        DEDUP_IOU_THRESHOLD = 0.90
        DEDUP_IOS_THRESHOLD = 0.85

    regions = [
        {
            "region_id": "A",
            "page_number": 6,
            "type": "speech_bubble",
            "bbox": {"x": 76, "y": 1042, "w": 926, "h": 610},
            "confidence": 0.7,
            "reading_order": 2,
        },
        {
            "region_id": "B",
            "page_number": 6,
            "type": "narration",
            "bbox": {"x": 76, "y": 1042, "w": 926, "h": 610},
            "confidence": 0.55,
            "reading_order": 3,
        },
    ]
    kept = _dedupe_raw_boxes(regions, _Cfg())
    assert len(kept) == 2


# Verify segmentation failure debug exporter saves crop and page images to failure directory
def test_save_seg_failure(tmp_path):
    from manhua_pipeline.stages.stage1_detection import _save_seg_failure

    crop = Image.new("RGB", (100, 100), color=(255, 0, 0))
    img_path = tmp_path / "page_001.png"
    crop.save(img_path)

    ws = tmp_path / "chapter_01"
    ws.mkdir()
    region = {"region_id": "P001_R001"}

    class _Cfg:
        YOLO_SEG_SAVE_FAILURES = True
        YOLO_SEG_FAILURES_DIR = str(tmp_path / "seg_failures")
        PROJECT_ROOT = tmp_path

    _save_seg_failure(crop, img_path, ws, 1, region, _Cfg())

    crop_out = tmp_path / "seg_failures" / "crops" / "chapter_01_P001_R001.png"
    page_out = tmp_path / "seg_failures" / "pages" / "chapter_01_P001.png"
    assert crop_out.exists()
    assert page_out.exists()
