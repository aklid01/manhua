import json

from PIL import Image

import config


def _setup(ws, page_wh=(400, 600), det=None, ocr=None, para=None):
    (ws / "pages").mkdir(parents=True, exist_ok=True)
    # Start with a green background to verify outside is untouched
    img = Image.new("RGB", page_wh, (0, 255, 0))
    # Draw bubble 1: white background
    img.paste((255, 255, 255), (50, 50, 150, 100))
    # Draw dark characters inside bubble 1
    img.paste((0, 0, 0), (70, 60, 130, 90))
    # Draw bubble 2: white background with dark characters
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


def _det(regions):
    return {
        "chapter_id": "t",
        "stage": "detection",
        "generated_at": "now",
        "model": "m",
        "regions": regions,
    }


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


def test_render_text_gated_erase(tmp_path, monkeypatch):
    from PIL import ImageFont

    from manhua_pipeline.stages import stage5_render

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
    out = Image.open(ws / "stage5_render" / "001_render.png").convert("RGB")
    # Assert that the text inside the bubble was erased (became white)
    assert out.getpixel((75, 75)) == (255, 255, 255)
    # Assert that the green background outside the bubble was untouched
    assert out.getpixel((45, 45)) == (0, 255, 0)
    rep = json.loads((ws / "stage5_render" / "render.json").read_text(encoding="utf-8"))
    by_id = {r["region_id"]: r for r in rep["results"]}
    assert by_id["P001_R001"]["rendered"] is True
    assert by_id["P001_R002"]["rendered"] is False
    assert by_id["P001_R002"]["action"] == "left_original_no_text"


def test_render_overflow_flagged(tmp_path, monkeypatch):
    from PIL import ImageFont

    from manhua_pipeline.stages import stage5_render

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


def test_render_skips_non_render_type(tmp_path, monkeypatch):
    from PIL import ImageFont

    from manhua_pipeline.stages import stage5_render

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
    out = Image.open(ws / "stage5_render" / "001_render.png").convert("RGB")
    assert out.getpixel((75, 75)) == (0, 0, 0)  # untouched character block
    rep = json.loads((ws / "stage5_render" / "render.json").read_text(encoding="utf-8"))
    assert rep["results"][0]["action"] == "left_original_not_render_type"


def test_render_advances_manifest_and_outputs(tmp_path, monkeypatch):
    from PIL import ImageFont

    from manhua_pipeline.stages import stage5_render

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
    assert (ws / "stage5_render" / "001_render.png").exists()
    m = json.loads((ws / "manifest.json").read_text())
    assert m["current_stage"] == "qa"
    assert "render" in m["completed_stages"]


def test_render_missing_paraphrase_left(tmp_path, monkeypatch):
    from PIL import ImageFont

    from manhua_pipeline.stages import stage5_render

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


def test_render_edge_touching_erase(tmp_path, monkeypatch):
    from PIL import ImageFont

    from manhua_pipeline.stages import stage5_render

    monkeypatch.setattr(
        stage5_render, "_load_font", lambda p, pt, cfg: ImageFont.load_default()
    )
    ws = tmp_path / "workspace"
    # Bubble touching the top edge of page
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
    # Setup page
    (ws / "pages").mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (400, 600), (0, 255, 0))
    # Draw bubble 1: white background at top
    img.paste((255, 255, 255), (50, 0, 150, 50))
    # Draw character touching top edge y=0
    img.paste((0, 0, 0), (70, 0, 130, 20))
    img.save(ws / "pages" / "001.png")

    manifest = {
        "chapter_id": "t",
        "total_pages": 1,
        "pages": [
            {
                "page_number": 1,
                "filename": "001.png",
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
    out = Image.open(ws / "stage5_render" / "001_render.png").convert("RGB")
    # Assert character touching top y=0 was erased (became white)
    assert out.getpixel((75, 10)) == (255, 255, 255)
    # Assert green outside is untouched
    assert out.getpixel((45, 10)) == (0, 255, 0)
