import json

import config
import pytest


@pytest.fixture(autouse=True)
def disable_batch_subprocess(monkeypatch):
    monkeypatch.setattr(config, "BATCH_SUBPROCESS", False)


def test_settings_prompt_saved_and_reused(tmp_path, monkeypatch):
    from manhua_pipeline.io import settings as st

    monkeypatch.setattr(st, "SETTINGS_PATH", tmp_path / "settings.json")
    monkeypatch.setattr("builtins.input", lambda *_: str(tmp_path / "SeriesA"))
    base = st.resolve_base_dir(args=None, config=config)  # prompts, saves
    assert base == (tmp_path / "SeriesA")
    saved = json.loads((tmp_path / "settings.json").read_text())
    assert saved["output_dir"] == str(tmp_path / "SeriesA")
    # second call reuses without prompting
    monkeypatch.setattr(
        "builtins.input",
        lambda *_: (_ for _ in ()).throw(AssertionError("should not prompt")),
    )
    assert st.resolve_base_dir(args=None, config=config) == (tmp_path / "SeriesA")


def test_two_chapters_isolated(tmp_path):
    import json

    from PIL import Image

    from manhua_pipeline.stages.stage0_import import run_import

    base = tmp_path / "SeriesA"
    for stem in ("chap_a", "chap_b"):
        src = tmp_path / stem
        src.mkdir()
        Image.new("RGB", (860, 1214)).save(src / "00000000_00010000.jpg")
        run_import(
            str(src), str(base / stem), config
        )  # base/stem is the chapter dir now
    assert (base / "chap_a" / "manifest.json").exists()
    assert (base / "chap_b" / "manifest.json").exists()
    ma = json.loads((base / "chap_a" / "manifest.json").read_text())
    assert "chap_a" in ma["chapter_id"]


def test_series_glossary_inherited(tmp_path):
    import config
    from manhua_pipeline.io.glossary_series import load_series_glossary, merge_glossary

    base = tmp_path / "SeriesA"
    base.mkdir()
    merge_glossary(
        base,
        [
            {
                "term_id": "yu_lili",
                "source_term": "于丽丽",
                "target_term": "Yu Lili",
                "category": "person_name",
                "locked": True,
                "auto_seeded": True,
                "source_region": "P002_R002",
                "notes": "",
            }
        ],
    )
    g = load_series_glossary(base, config)
    assert any(t["target_term"] == "Yu Lili" for t in g["terms"])
    merge_glossary(
        base,
        [
            {
                "term_id": "lin_yi",
                "source_term": "林逸",
                "target_term": "Lin Yi",
                "category": "person_name",
                "locked": True,
                "auto_seeded": True,
                "source_region": "P005_R001",
                "notes": "",
            }
        ],
    )
    g2 = load_series_glossary(base, config)
    ids = {t["term_id"] for t in g2["terms"]}
    assert {"yu_lili", "lin_yi"} <= ids


def test_fresh_clears_chapter_artifacts(tmp_path):
    from PIL import Image

    from manhua_pipeline.stages.stage0_import import run_import

    base = tmp_path / "SeriesA"
    src = tmp_path / "chap_a"
    src.mkdir()
    Image.new("RGB", (860, 1214)).save(src / "00000000_00010000.jpg")
    run_import(str(src), str(base / "chap_a"), config)
    ch = base / "chap_a"
    (ch / "stage3_translation").mkdir(parents=True, exist_ok=True)
    (ch / "stage3_translation" / "translation_response.json").write_text("{}")
    run_import(str(src), str(base / "chap_a"), config, fresh=True)
    assert not (ch / "stage3_translation" / "translation_response.json").exists()


def test_render_original_filenames(tmp_path, monkeypatch):
    import json

    from PIL import Image, ImageFont

    from manhua_pipeline.stages import stage5_render

    monkeypatch.setattr(
        stage5_render, "_load_font", lambda p, pt, cfg: ImageFont.load_default()
    )
    ch = tmp_path / "SeriesA" / "chap_a"
    (ch / "pages").mkdir(parents=True)
    Image.new("RGB", (400, 600), (255, 255, 255)).save(ch / "pages" / "001.png")
    manifest = {
        "chapter_id": "chap_a",
        "total_pages": 1,
        "pages": [
            {
                "page_number": 1,
                "filename": "001.png",
                "original_filename": "00000000_00010000.jpg",
                "skip": False,
                "width": 400,
                "height": 600,
            }
        ],
        "current_stage": "render",
        "completed_stages": ["import", "detect", "ocr", "translate", "paraphrase"],
        "status": "in_progress",
    }
    (ch / "manifest.json").write_text(json.dumps(manifest))
    (ch / "stage1_detection").mkdir(parents=True)
    (ch / "stage1_detection" / "detection.json").write_text(
        json.dumps(
            {
                "regions": [
                    {
                        "region_id": "P001_R001",
                        "page_number": 1,
                        "type": "speech_bubble",
                        "bbox": {"x": 50, "y": 50, "w": 100, "h": 40},
                        "reading_order": 1,
                        "style_hint": "round",
                        "confidence": 0.9,
                        "read_region": {"x": 50, "y": 50, "w": 100, "h": 40},
                        "erase_mask": {"type": "rect", "coords": [50, 50, 100, 40]},
                        "render": True,
                    }
                ]
            }
        )
    )
    (ch / "stage2_ocr").mkdir(parents=True)
    (ch / "stage2_ocr" / "ocr.json").write_text(
        json.dumps({"results": [{"region_id": "P001_R001", "has_usable_text": True}]})
    )
    (ch / "stage4_paraphrase").mkdir(parents=True)
    (ch / "stage4_paraphrase" / "paraphrase.json").write_text(
        json.dumps(
            {
                "results": [
                    {
                        "region_id": "P001_R001",
                        "final_text": "HI",
                        "register": "neutral",
                        "paraphrased": True,
                    }
                ]
            }
        )
    )
    stage5_render.run_render(str(ch), config)
    assert (ch / "stage5_render" / "rendered" / "001.png").exists()


def test_two_chapters_independent_overrides_and_overlays(tmp_path):
    ch1 = tmp_path / "chap_a"
    ch2 = tmp_path / "chap_b"
    ch1.mkdir()
    ch2.mkdir()
    (ch1 / "overrides.json").write_text('{"P001_R001": "override_a"}')
    (ch2 / "overrides.json").write_text('{"P001_R001": "override_b"}')
    from manhua_pipeline.io.overrides import load_overrides

    ov1 = load_overrides(ch1, config)
    ov2 = load_overrides(ch2, config)
    assert ov1["P001_R001"] == "override_a"
    assert ov2["P001_R001"] == "override_b"


def test_run_all_stops_at_handoff(tmp_path, monkeypatch):
    import pipeline

    calls = []
    monkeypatch.setattr(
        pipeline.stage3_translation,
        "run_translation",
        lambda ws, cfg: calls.append("t") or None,
    )  # None => awaiting handoff
    monkeypatch.setattr(
        pipeline.stage4_paraphrase, "run_paraphrase", lambda ws, cfg: calls.append("p")
    )
    monkeypatch.setattr(
        pipeline.stage5_render, "run_render", lambda ws, cfg: calls.append("r")
    )
    pipeline._run_all_from(tmp_path, config, start="translate")
    assert calls == [
        "t"
    ]  # stopped after translation returned None; later stages NOT called


def test_bold_font_resolves_and_renders(tmp_path):
    import json

    from PIL import Image

    import config
    from manhua_pipeline.stages import stage5_render

    assert "ComicNeue-Bold.ttf" in str(config.FONT_PATH)

    ch = tmp_path / "chap_a"
    (ch / "pages").mkdir(parents=True)
    Image.new("RGB", (400, 600), (255, 255, 255)).save(ch / "pages" / "001.png")
    manifest = {
        "chapter_id": "chap_a",
        "total_pages": 1,
        "pages": [
            {
                "page_number": 1,
                "filename": "001.png",
                "original_filename": "00000000_00010000.jpg",
                "skip": False,
                "width": 400,
                "height": 600,
            }
        ],
        "current_stage": "render",
        "completed_stages": ["import", "detect", "ocr", "translate", "paraphrase"],
        "status": "in_progress",
    }
    (ch / "manifest.json").write_text(json.dumps(manifest))
    (ch / "stage1_detection").mkdir(parents=True)
    (ch / "stage1_detection" / "detection.json").write_text(
        json.dumps(
            {
                "regions": [
                    {
                        "region_id": "P001_R001",
                        "page_number": 1,
                        "type": "speech_bubble",
                        "bbox": {"x": 50, "y": 50, "w": 100, "h": 40},
                        "reading_order": 1,
                        "style_hint": "round",
                        "confidence": 0.9,
                        "read_region": {"x": 50, "y": 50, "w": 100, "h": 40},
                        "erase_mask": {"type": "rect", "coords": [50, 50, 100, 40]},
                        "render": True,
                    }
                ]
            }
        )
    )
    (ch / "stage2_ocr").mkdir(parents=True)
    (ch / "stage2_ocr" / "ocr.json").write_text(
        json.dumps({"results": [{"region_id": "P001_R001", "has_usable_text": True}]})
    )
    (ch / "stage4_paraphrase").mkdir(parents=True)
    (ch / "stage4_paraphrase" / "paraphrase.json").write_text(
        json.dumps(
            {
                "results": [
                    {
                        "region_id": "P001_R001",
                        "final_text": "HI",
                        "register": "neutral",
                        "paraphrased": True,
                    }
                ]
            }
        )
    )

    stage5_render.run_render(str(ch), config)
    assert (ch / "stage5_render" / "rendered" / "001.png").exists()
