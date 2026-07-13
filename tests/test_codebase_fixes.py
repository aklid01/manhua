import json

from PIL import Image

import config
import pipeline


def test_missing_chapter_lists_available(tmp_path, monkeypatch, capsys):
    base = tmp_path / "SeriesA"
    (base / "Ch1").mkdir(parents=True)
    (base / "Ch1" / config.MANIFEST_NAME).write_text(
        json.dumps({"chapter_id": "Ch1"})
    )
    monkeypatch.setattr(
        "manhua_pipeline.io.settings.resolve_base_dir", lambda a, c: base
    )
    rc = pipeline.main(["detect", "--chapter", "GhostCh"])
    assert rc == 2
    captured = capsys.readouterr()
    assert "Available chapters" in captured.out or "Available chapters" in captured.err


def test_qa_benign_no_text_not_failed(tmp_path):
    from manhua_pipeline.stages.stage6_qa import run_qa

    ws = tmp_path / "SeriesA" / "Ch1"
    ws.mkdir(parents=True)
    manifest = {
        "chapter_id": "Ch1",
        "current_stage": "complete",
        "pages": [{"page_number": 1, "filename": "001.png", "skip": False}],
    }
    (ws / "manifest.json").write_text(json.dumps(manifest))
    ocr = {
        "results": [
            {
                "region_id": f"P001_R00{i}",
                "page_number": 1,
                "has_usable_text": False,
                "needs_correction": True,
                "watermark_filtered": True,
                "edge_touching": False,
                "note": "watermark-only region; not rendered",
                "original_text": "",
            }
            for i in range(1, 9)
        ]
    }
    render = {
        "results": [
            {
                "region_id": f"P001_R00{i}",
                "page_number": 1,
                "rendered": False,
                "action": "left_original_no_text",
                "overflow": False,
            }
            for i in range(1, 9)
        ],
        "pages": [{"page_number": 1, "output_file": "rendered/001.png"}],
    }
    for sub, name, data in [
        ("stage2_ocr", "ocr.json", ocr),
        ("stage3_translation", "translation.json", {"results": []}),
        ("stage4_paraphrase", "paraphrase.json", {"results": []}),
        ("stage5_render", "render.json", render),
    ]:
        (ws / sub).mkdir(parents=True, exist_ok=True)
        (ws / sub / name).write_text(json.dumps(data), encoding="utf-8")

    run_qa(str(ws), config)
    qa = json.loads(
        (ws / "stage6_qa" / "qa.json").read_text(encoding="utf-8")
    )
    assert qa["status"] in ("SUCCESS", "REVIEW")  # NOT FAILED
    assert qa["checks"]["low_ocr_confidence"] >= 1


def test_import_current_stage_detect(tmp_path):
    from manhua_pipeline.stages.stage0_import import run_import

    base = tmp_path / "SeriesA"
    src = tmp_path / "Ch1"
    src.mkdir()
    Image.new("RGB", (860, 1214)).save(src / "00000000_00010000.jpg")
    run_import(str(src), str(base / "Ch1"), config)
    m = json.loads((base / "Ch1" / "manifest.json").read_text())
    assert m["current_stage"] == "detect"


def test_ocr_error_fallback_has_watermark_key(tmp_path):
    from manhua_pipeline.stages import stage2_ocr

    region = {
        "region_id": "P001_R001",
        "page_number": 1,
        "type": "speech_bubble",
        "read_region": {"x": 0, "y": 0, "w": 10, "h": 10},
    }
    page = {"filename": "001.png"}
    res, is_warning = stage2_ocr._process_single_region_ocr(
        region, page, object(), config, tmp_path
    )
    assert is_warning is True
    assert "watermark_filtered" in res
    assert res["watermark_filtered"] is False


def test_bubble_mask_fills_white_region(tmp_path):
    from manhua_pipeline.stages import stage5_render

    img = Image.new("RGB", (40, 40), (0, 0, 0))
    for y in range(10, 30):
        for x in range(10, 30):
            img.putpixel((x, y), (255, 255, 255))
    mask = stage5_render._get_bubble_mask(
        img, 0, 0, 40, 40, 40, 40, config
    )
    assert mask.getpixel((20, 20)) == 255
    assert mask.getpixel((2, 2)) == 0
