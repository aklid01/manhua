# Unit tests for Stage 2 OCR text extraction, confidence scoring, and review flow.

import json
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

import config
from manhua_pipeline.stages.stage2_ocr import _read_crop


# Helper to configure mock workspace with test image, manifest, and detection JSON
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


# Helper to construct mock detection region dictionary
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


# Verify OCR stage extracts text, confidence scores, and schema fields
def test_ocr_reads_text_and_flags(tmp_path):
    from manhua_pipeline.stages.stage2_ocr import run_ocr

    ws = tmp_path / "workspace"
    _setup(ws, regions=[_region("P001_R001", 100, 300, 200, 120)])
    with (
        patch("manhua_pipeline.stages.stage2_ocr._get_ocr") as mock_get,
        patch(
            "manhua_pipeline.stages.stage2_ocr._read_crop",
            return_value=("滚吧！", 0.94, 0.94, False),
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


# Verify low confidence OCR results are flagged for manual correction
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
            return_value=("朝阳集团", 0.41, 0.41, False),
        ),
    ):
        run_ocr(str(ws), config)
    e = json.loads((ws / "stage2_ocr" / "ocr.json").read_text(encoding="utf-8"))[
        "results"
    ][0]
    assert e["ocr_confidence"] == 0.41
    assert e["needs_correction"] is True


# Verify regions touching page boundaries are flagged as edge_touching
def test_ocr_edge_touching_split_bubble(tmp_path):
    from manhua_pipeline.stages.stage2_ocr import run_ocr

    ws = tmp_path / "workspace"
    _setup(ws, page_h=1000, regions=[_region("P001_R001", 100, 950, 200, 50)])
    with (
        patch(
            "manhua_pipeline.stages.stage2_ocr._get_ocr",
            return_value=MagicMock(),
        ),
        patch(
            "manhua_pipeline.stages.stage2_ocr._read_crop",
            return_value=("", 0.0, 0.0, False),
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
    assert e["note"]


# Verify bounding boxes extending beyond page dimensions are clamped without errors
def test_ocr_clamps_out_of_bounds(tmp_path):
    from manhua_pipeline.stages.stage2_ocr import run_ocr

    ws = tmp_path / "workspace"
    _setup(ws, page_h=1000, regions=[_region("P001_R001", 800, 980, 200, 100)])
    with (
        patch(
            "manhua_pipeline.stages.stage2_ocr._get_ocr",
            return_value=MagicMock(),
        ),
        patch(
            "manhua_pipeline.stages.stage2_ocr._read_crop",
            return_value=("x", 0.8, 0.8, False),
        ),
    ):
        run_ocr(str(ws), config)
    ocr = json.loads((ws / "stage2_ocr" / "ocr.json").read_text(encoding="utf-8"))
    assert len(ocr["results"]) == 1


# Verify OCR stage handles pages with zero detected regions correctly
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


# Mock OCR predictor returning specified texts and scores
class _FakeOCR:
    def __init__(self, texts, scores):
        self._texts = texts
        self._scores = scores

    def predict(self, img):
        return [{"rec_texts": self._texts, "rec_scores": self._scores}]


# Verify OCR crop reading removes watermark lines while keeping dialogue
def test_read_crop_parses_and_filters_watermark():
    crop = Image.new("RGB", (200, 80))
    eng = _FakeOCR(["滚吧！", "www.baozimh.com"], [0.94, 0.99])
    text, mean_c, min_c, filtered = _read_crop(eng, crop, config)
    assert text == "滚吧！"
    assert filtered is True
    assert 0.9 <= mean_c <= 1.0


# Verify crop containing only watermarks returns empty string and zero confidence
def test_read_crop_empty_when_only_watermark():
    crop = Image.new("RGB", (200, 80))
    eng = _FakeOCR(["包子漫畫", "www.baozimh.com"], [0.9, 0.9])
    text, mean_c, min_c, filtered = _read_crop(eng, crop, config)
    assert text == ""
    assert filtered is True
    assert mean_c == 0.0 and min_c == 0.0


# Verify watermark-only regions are marked as unusable text
def test_ocr_watermark_only_region_not_usable(tmp_path):
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
            return_value=("", 0.0, 0.0, True),
        ),
    ):
        run_ocr(str(ws), config)
    e = json.loads((ws / "stage2_ocr" / "ocr.json").read_text(encoding="utf-8"))[
        "results"
    ][0]
    assert e["has_usable_text"] is False
    assert e["watermark_filtered"] is True
    assert e["note"]


# Verify dialogue lines are retained when watermarks are stripped
def test_ocr_dialogue_with_watermark_keeps_dialogue(tmp_path):
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
            return_value=("滚吧！", 0.94, 0.94, True),
        ),
    ):
        run_ocr(str(ws), config)
    e = json.loads((ws / "stage2_ocr" / "ocr.json").read_text(encoding="utf-8"))[
        "results"
    ][0]
    assert e["original_text"] == "滚吧！"
    assert e["has_usable_text"] is True
    assert e["watermark_filtered"] is True


# Verify regular clean text is not flagged as a watermark
def test_ocr_normal_text_not_flagged(tmp_path):
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
            return_value=("老子还不干了！", 0.89, 0.89, False),
        ),
    ):
        run_ocr(str(ws), config)
    e = json.loads((ws / "stage2_ocr" / "ocr.json").read_text(encoding="utf-8"))[
        "results"
    ][0]
    assert e["watermark_filtered"] is False
    assert e["has_usable_text"] is True


# Verify traditional Chinese watermark phrases are filtered
def test_read_crop_filters_traditional_watermark():
    crop = Image.new("RGB", (200, 80))
    eng = _FakeOCR(["快逃啊！", "最新免費漫畫"], [0.92, 0.95])
    text, mean_c, min_c, filtered = _read_crop(eng, crop, config)
    assert text == "快逃啊！"
    assert filtered is True


# Verify OCR stage results include success status tag
def test_ocr_results_include_status(tmp_path):
    from manhua_pipeline.stages.stage2_ocr import run_ocr

    ws = tmp_path / "workspace"
    _setup(ws, regions=[_region("P001_R001", 100, 300, 200, 120)])
    with (
        patch("manhua_pipeline.stages.stage2_ocr._get_ocr", return_value=MagicMock()),
        patch(
            "manhua_pipeline.stages.stage2_ocr._read_crop",
            return_value=("朝阳集团", 0.90, 0.90, False),
        ),
    ):
        run_ocr(str(ws), config)
    ocr = json.loads((ws / "stage2_ocr" / "ocr.json").read_text(encoding="utf-8"))
    res = ocr["results"][0]
    assert res["status"] == "success"


# Verify runtime error is raised if all detected regions produce zero OCR text
def test_ocr_all_regions_empty_raises_error(tmp_path):
    from manhua_pipeline.stages.stage2_ocr import run_ocr

    ws = tmp_path / "workspace"
    _setup(
        ws,
        regions=[
            _region("P001_R001", 100, 300, 200, 120),
            _region("P001_R002", 100, 450, 200, 120),
        ],
    )
    with (
        patch("manhua_pipeline.stages.stage2_ocr._get_ocr", return_value=MagicMock()),
        patch(
            "manhua_pipeline.stages.stage2_ocr._read_crop",
            return_value=("", 0.0, 0.0, False),
        ),
    ):
        with pytest.raises(
            RuntimeError, match="PaddleOCR returned no text for any region"
        ):
            run_ocr(str(ws), config)


# Verify fallback to alternative OCR engine when preferred initialization fails
def test_ocr_engine_fallback():
    import manhua_pipeline.stages.stage2_ocr as stage2_ocr
    from manhua_pipeline.stages.stage2_ocr import _get_ocr

    stage2_ocr._OCR_ENGINE = None
    stage2_ocr._ACTIVE_OCR_ENGINE = None

    class MockConfig:
        OCR_ENGINE = "transformers"
        OCR_LANG = "ch"
        OCR_VERSION = "PP-OCRv6"
        OCR_USE_GPU = False

    with patch("paddleocr.PaddleOCR") as mock_paddle_ocr:
        mock_paddle_ocr.side_effect = [ValueError("Failed preferred"), MagicMock()]

        engine = _get_ocr(MockConfig)
        assert engine is not None
        assert stage2_ocr._ACTIVE_OCR_ENGINE == "paddle"


# Verify _close_ocr releases engine resources and clears singleton reference
def test_ocr_close_clears_singleton():
    import manhua_pipeline.stages.stage2_ocr as stage2_ocr
    from manhua_pipeline.stages.stage2_ocr import _close_ocr

    stage2_ocr._OCR_ENGINE = None
    _close_ocr()
    assert stage2_ocr._OCR_ENGINE is None

    mock_engine = MagicMock()
    mock_engine.close = MagicMock()
    mock_engine.shutdown = MagicMock()

    mock_inner = MagicMock()
    mock_inner.close = MagicMock()
    mock_engine.paddlex_pipeline = mock_inner

    stage2_ocr._OCR_ENGINE = mock_engine
    _close_ocr()

    assert stage2_ocr._OCR_ENGINE is None
    mock_engine.close.assert_called_once()
    mock_inner.close.assert_called_once()


# Verify punctuation-only and empty text detection helpers
def test_ocr_suspicious_detector():
    from manhua_pipeline.stages.stage2_ocr import _has_cjk, _is_punct_or_empty

    assert _is_punct_or_empty("？") is True
    assert _is_punct_or_empty("") is True
    assert _is_punct_or_empty("！？") is True
    assert _is_punct_or_empty("   ") is True
    assert _is_punct_or_empty("诶？") is False
    assert _is_punct_or_empty("诶") is False
    assert _is_punct_or_empty("hi") is False
    assert _has_cjk("诶") is True
    assert _has_cjk("？") is False


# Verify end-to-end review flow flags suspicious regions and applies manual corrections
def test_ocr_review_flow(tmp_path, monkeypatch):
    from manhua_pipeline.stages import stage2_ocr as s2

    ws = tmp_path / "series" / "ch1"
    (ws / "pages").mkdir(parents=True)
    (ws / "stage1_detection").mkdir(parents=True)
    Image.new("RGB", (1100, 1800), (255, 255, 255)).save(ws / "pages" / "022.png")
    (ws / "manifest.json").write_text(
        json.dumps(
            {
                "chapter_id": "ch1",
                "pages": [{"page_number": 22, "filename": "022.png", "skip": False}],
                "completed_stages": ["import", "detect"],
                "current_stage": "ocr",
            }
        ),
        encoding="utf-8",
    )
    (ws / "stage1_detection" / "detection.json").write_text(
        json.dumps(
            {
                "regions": [
                    {
                        "region_id": "P022_R002",
                        "page_number": 22,
                        "type": "speech_bubble",
                        "read_region": {"x": 840, "y": 1571, "w": 131, "h": 93},
                        "bbox": {"x": 840, "y": 1571, "w": 131, "h": 93},
                        "is_free_text": False,
                    },
                    {
                        "region_id": "P022_R003",
                        "page_number": 22,
                        "type": "speech_bubble",
                        "read_region": {"x": 100, "y": 100, "w": 200, "h": 80},
                        "bbox": {"x": 100, "y": 100, "w": 200, "h": 80},
                        "is_free_text": False,
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    def fake_read_best(engine, crop_im, cfg):
        return (
            ("？", 0.95, 0.95, False)
            if crop_im.width < 160
            else ("你好呀", 0.92, 0.9, False)
        )

    monkeypatch.setattr(s2, "_read_best", fake_read_best)
    monkeypatch.setattr(
        s2, "_read_forced_ladder", lambda e, c, cfg: ("？", 0.95, 0.95, False)
    )
    monkeypatch.setattr(s2, "_get_ocr", lambda cfg: object())
    monkeypatch.setattr(s2, "_close_ocr", lambda: None)

    s2.run_ocr(str(ws), config)
    ocr = json.loads((ws / "stage2_ocr" / "ocr.json").read_text(encoding="utf-8"))
    by = {r["region_id"]: r for r in ocr["results"]}
    assert by["P022_R002"]["ocr_review"] is True
    assert by["P022_R002"]["ocr_review_reason"] == "punct_only"
    assert by["P022_R003"]["ocr_review"] is False
    rev = json.loads(
        (ws / "stage2_ocr" / "ocr_review.json").read_text(encoding="utf-8")
    )
    assert "P022_R002" in rev["regions"]
    assert "P022_R003" not in rev["regions"]
    assert (ws / "stage2_ocr" / "review_crops" / "P022_R002.png").exists()

    rev["regions"]["P022_R002"]["corrected_text"] = "诶？"
    (ws / "stage2_ocr" / "ocr_review.json").write_text(
        json.dumps(rev, ensure_ascii=False), encoding="utf-8"
    )
    s2.run_ocr(str(ws), config)
    ocr = json.loads((ws / "stage2_ocr" / "ocr.json").read_text(encoding="utf-8"))
    by = {r["region_id"]: r for r in ocr["results"]}
    assert by["P022_R002"]["original_text"] == "诶？"
    assert by["P022_R002"]["status"] == "manual_review"
    assert by["P022_R002"]["ocr_review"] is False


# Verify OCR_REVIEW_ENABLED=False skips review flagging and file generation
def test_ocr_review_toggle_off(tmp_path, monkeypatch):
    from manhua_pipeline.stages import stage2_ocr as s2

    monkeypatch.setattr(config, "OCR_REVIEW_ENABLED", False, raising=False)
    ws = tmp_path / "series" / "ch1"
    (ws / "pages").mkdir(parents=True)
    (ws / "stage1_detection").mkdir(parents=True)
    Image.new("RGB", (1100, 1800), (255, 255, 255)).save(ws / "pages" / "022.png")
    (ws / "manifest.json").write_text(
        json.dumps(
            {
                "chapter_id": "ch1",
                "pages": [{"page_number": 22, "filename": "022.png", "skip": False}],
                "completed_stages": ["import", "detect"],
                "current_stage": "ocr",
            }
        ),
        encoding="utf-8",
    )
    (ws / "stage1_detection" / "detection.json").write_text(
        json.dumps(
            {
                "regions": [
                    {
                        "region_id": "P022_R002",
                        "page_number": 22,
                        "type": "speech_bubble",
                        "read_region": {"x": 840, "y": 1571, "w": 131, "h": 93},
                        "bbox": {"x": 840, "y": 1571, "w": 131, "h": 93},
                        "is_free_text": False,
                    },
                    {
                        "region_id": "P022_R003",
                        "page_number": 22,
                        "type": "speech_bubble",
                        "read_region": {"x": 100, "y": 100, "w": 200, "h": 80},
                        "bbox": {"x": 100, "y": 100, "w": 200, "h": 80},
                        "is_free_text": False,
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        s2,
        "_read_best",
        lambda e, c, cfg: (
            ("？", 0.95, 0.95, False) if c.width < 160 else ("你好", 0.9, 0.9, False)
        ),
    )
    monkeypatch.setattr(s2, "_get_ocr", lambda cfg: object())
    monkeypatch.setattr(s2, "_close_ocr", lambda: None)

    s2.run_ocr(str(ws), config)
    ocr = json.loads((ws / "stage2_ocr" / "ocr.json").read_text(encoding="utf-8"))
    by = {r["region_id"]: r for r in ocr["results"]}
    assert by["P022_R002"]["ocr_review"] is False
    assert not (ws / "stage2_ocr" / "ocr_review.json").exists()
