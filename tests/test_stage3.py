# Unit tests for Stage 3 translation batching, glossary locking, triage, and SFX passthrough.

import json
from types import SimpleNamespace

import pytest

import config


# Fixture forcing manual translation backend for offline isolation
@pytest.fixture(autouse=True)
def _force_manual_backend(monkeypatch):
    monkeypatch.setattr(config, "TRANSLATOR_BACKEND", "manual", raising=False)


# Helper to configure mock workspace with manifest, OCR data, and optional glossary
def _setup(ws, ocr_results, glossary=None):
    ws.mkdir(parents=True, exist_ok=True)
    manifest = {
        "chapter_id": "t",
        "total_pages": 1,
        "pages": [{"page_number": 1, "filename": "001.png", "skip": False}],
        "current_stage": "translate",
        "completed_stages": ["import", "detect", "ocr"],
        "warning_count": 0,
        "status": "in_progress",
    }
    (ws / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (ws / "stage2_ocr").mkdir(parents=True, exist_ok=True)
    ocr = {
        "chapter_id": "t",
        "stage": "ocr",
        "generated_at": "now",
        "ocr_engine": "PaddleOCR",
        "results": ocr_results,
    }
    (ws / "stage2_ocr" / "ocr.json").write_text(
        json.dumps(ocr, ensure_ascii=False), encoding="utf-8"
    )
    if glossary is not None:
        (ws.parent / "glossary.json").write_text(
            json.dumps(glossary, ensure_ascii=False), encoding="utf-8"
        )


# Helper to construct mock OCR region result dictionary
def _ocr_entry(rid, text, usable=True):
    return {
        "region_id": rid,
        "page_number": 1,
        "type": "speech_bubble",
        "original_text": text,
        "text_direction": "horizontal",
        "ocr_confidence": 0.9,
        "ocr_confidence_min": 0.9,
        "has_usable_text": usable,
        "do_not_render": False,
        "needs_correction": not usable,
        "edge_touching": False,
        "edge": "none",
        "note": None,
        "watermark_filtered": False,
    }


# Verify manual translator backend emits prompt file and awaits user response
def test_translation_manual_awaits_response(tmp_path):
    from manhua_pipeline.stages.stage3_translation import run_translation

    ws = tmp_path / "workspace"
    _setup(ws, [_ocr_entry("P001_R001", "滚吧！")])
    result = run_translation(str(ws), config)
    assert result is None
    assert (ws / "stage3_translation" / "translation_prompt.json").exists()
    assert not (ws / "stage3_translation" / "translation.json").exists()
    m = json.loads((ws / "manifest.json").read_text())
    assert m["current_stage"] == "translate"


# Verify translation prompt includes only usable OCR text regions
def test_translation_prompt_only_usable(tmp_path):
    from manhua_pipeline.stages.stage3_translation import run_translation

    ws = tmp_path / "workspace"
    _setup(
        ws,
        [
            _ocr_entry("P001_R001", "滚吧！", usable=True),
            _ocr_entry("P001_R002", "", usable=False),
        ],
    )
    run_translation(str(ws), config)
    bundle = json.loads(
        (ws / "stage3_translation" / "translation_prompt.json").read_text(
            encoding="utf-8"
        )
    )
    ids = [item["region_id"] for item in bundle["regions"]]
    assert ids == ["P001_R001"]


# Verify translation stage ingests manual response file and updates manifest
def test_translation_ingests_response(tmp_path):
    from manhua_pipeline.stages.stage3_translation import run_translation

    ws = tmp_path / "workspace"
    _setup(
        ws,
        [
            _ocr_entry("P001_R001", "滚吧！", usable=True),
            _ocr_entry("P001_R002", "", usable=False),
        ],
    )
    run_translation(str(ws), config)
    resp = {"P001_R001": "Get out!"}
    (ws / "stage3_translation" / "translation_response.json").write_text(
        json.dumps(resp), encoding="utf-8"
    )
    out = run_translation(str(ws), config)
    assert out is not None
    tr = json.loads(
        (ws / "stage3_translation" / "translation.json").read_text(encoding="utf-8")
    )
    by_id = {r["region_id"]: r for r in tr["results"]}
    assert by_id["P001_R001"]["literal_translation"] == "Get out!"
    assert by_id["P001_R001"]["translated"] is True
    assert by_id["P001_R002"]["translated"] is False
    assert by_id["P001_R002"]["skip_reason"] == "no_usable_text"
    m = json.loads((ws / "manifest.json").read_text())
    assert m["current_stage"] == "paraphrase"
    assert "translate" in m["completed_stages"]


# Verify exact locked glossary terms translate directly without invoking backend
def test_translation_glossary_locked_term(tmp_path):
    from manhua_pipeline.stages.stage3_translation import run_translation

    ws = tmp_path / "workspace"
    glossary = {
        "version": "v1",
        "updated_at": "now",
        "terms": [
            {
                "term_id": "jingli",
                "source_term": "经理",
                "target_term": "Manager",
                "category": "title",
                "locked": True,
                "auto_seeded": False,
                "source_region": None,
                "notes": "",
            }
        ],
    }
    _setup(ws, [_ocr_entry("P001_R001", "经理")], glossary=glossary)
    out = run_translation(str(ws), config)
    assert out is not None
    tr = json.loads(
        (ws / "stage3_translation" / "translation.json").read_text(encoding="utf-8")
    )
    r = tr["results"][0]
    assert r["translated"] is True
    assert r["literal_translation"] == "Manager"
    assert r["translation_source"] == "locked_glossary"


# Verify translation stage advances when all regions are marked unusable
def test_translation_all_unusable_completes(tmp_path):
    from manhua_pipeline.stages.stage3_translation import run_translation

    ws = tmp_path / "workspace"
    _setup(ws, [_ocr_entry("P001_R001", "", usable=False)])
    out = run_translation(str(ws), config)
    assert out is not None
    tr = json.loads(
        (ws / "stage3_translation" / "translation.json").read_text(encoding="utf-8")
    )
    assert tr["results"][0]["translated"] is False
    m = json.loads((ws / "manifest.json").read_text())
    assert m["current_stage"] == "paraphrase"


# Verify locked glossary terms preserve exact target translations
def test_translation_glossary_conflict_does_not_mutate(tmp_path):
    from manhua_pipeline.stages.stage3_translation import run_translation

    ws = tmp_path / "workspace"
    glossary = {
        "version": "v1",
        "updated_at": "now",
        "terms": [
            {
                "term_id": "jingli",
                "source_term": "经理",
                "target_term": "Manager",
                "category": "title",
                "locked": True,
                "auto_seeded": False,
                "source_region": None,
                "notes": "",
            }
        ],
    }
    _setup(ws, [_ocr_entry("P001_R001", "经理")], glossary=glossary)
    out = run_translation(str(ws), config)
    assert out is not None
    tr = json.loads(
        (ws / "stage3_translation" / "translation.json").read_text(encoding="utf-8")
    )
    r = tr["results"][0]
    assert r["translated"] is True
    assert r["literal_translation"] == "Manager"
    assert r["glossary_conflict"] is False


# Verify translation output records active glossary version tag
def test_translation_glossary_version_from_glossary(tmp_path):
    from manhua_pipeline.stages.stage3_translation import run_translation

    ws = tmp_path / "workspace"
    glossary = {"version": "v3", "updated_at": "now", "terms": []}
    _setup(ws, [_ocr_entry("P001_R001", "滚吧！")], glossary=glossary)
    run_translation(str(ws), config)
    (ws / "stage3_translation" / "translation_response.json").write_text(
        json.dumps({"P001_R001": "Get lost!"}), encoding="utf-8"
    )
    run_translation(str(ws), config)
    tr = json.loads(
        (ws / "stage3_translation" / "translation.json").read_text(encoding="utf-8")
    )
    assert tr["glossary_version"] == "v3"


# Verify manual overrides rescue unusable OCR regions
def test_translation_override_rescues_unusable(tmp_path):
    from manhua_pipeline.stages.stage3_translation import run_translation

    ws = tmp_path / "workspace"
    _setup(ws, [_ocr_entry("P001_R001", "", usable=False)])
    (ws / "overrides.json").write_text(
        json.dumps({"_comment": "x", "P001_R001": "Manager? My ass!"}), encoding="utf-8"
    )
    out = run_translation(str(ws), config)
    assert out is not None
    tr = json.loads(
        (ws / "stage3_translation" / "translation.json").read_text(encoding="utf-8")
    )
    r = tr["results"][0]
    assert r["literal_translation"] == "Manager? My ass!"
    assert r["translated"] is True
    assert r["translation_source"] == "override"
    m = json.loads((ws / "manifest.json").read_text())
    assert m["current_stage"] == "paraphrase"


# Verify empty/whitespace overrides are ignored
def test_translation_empty_override_ignored(tmp_path):
    from manhua_pipeline.stages.stage3_translation import run_translation

    ws = tmp_path / "workspace"
    _setup(ws, [_ocr_entry("P001_R001", "", usable=False)])
    (ws / "overrides.json").write_text(
        json.dumps({"P001_R001": "   "}), encoding="utf-8"
    )
    run_translation(str(ws), config)
    tr = json.loads(
        (ws / "stage3_translation" / "translation.json").read_text(encoding="utf-8")
    )
    r = tr["results"][0]
    assert r["translated"] is False
    assert r.get("translation_source") in (None, "")


# Verify overrides for non-existent regions are discarded
def test_translation_unknown_override_ignored(tmp_path):
    from manhua_pipeline.stages.stage3_translation import run_translation

    ws = tmp_path / "workspace"
    _setup(ws, [_ocr_entry("P001_R001", "滚吧！", usable=True)])
    (ws / "overrides.json").write_text(
        json.dumps({"P999_R999": "ghost"}), encoding="utf-8"
    )
    run_translation(str(ws), config)
    tr_prompt = json.loads(
        (ws / "stage3_translation" / "translation_prompt.json").read_text(
            encoding="utf-8"
        )
    )
    ids = [i["region_id"] for i in tr_prompt["regions"]]
    assert ids == ["P001_R001"]


# Verify translation operates normally when no overrides file exists
def test_translation_no_overrides_file_unchanged(tmp_path):
    from manhua_pipeline.stages.stage3_translation import run_translation

    ws = tmp_path / "workspace"
    _setup(ws, [_ocr_entry("P001_R001", "滚吧！", usable=True)])
    result = run_translation(str(ws), config)
    assert result is None
    assert (ws / "stage3_translation" / "translation_prompt.json").exists()


# Verify batch response normalizes escaped newlines
def test_validate_batch_newline_normalization():
    from manhua_pipeline.stages.stage3_translation import OllamaBackend

    parsed = {"P001_R001": "Line 1\\nLine 2\\r\\nLine 3"}
    expected = {"P001_R001"}
    accepted, missing, unexpected = OllamaBackend._validate_batch(parsed, expected)
    assert accepted["P001_R001"] == "Line 1\nLine 2\nLine 3"
    assert missing == []
    assert unexpected == []


# Verify CJK character counting and trivial region classification
def test_cjk_char_count_and_is_trivial_region():
    from manhua_pipeline.stages.stage3_translation import (
        _cjk_char_count,
        _is_trivial_region,
    )

    assert _cjk_char_count("一") == 1
    assert _cjk_char_count("你好") == 2
    assert _cjk_char_count("Hello 123!") == 0

    assert _is_trivial_region("一", config) is True
    assert _is_trivial_region("你好", config) is False
    assert _is_trivial_region("ABC", config) is True


# Verify short CJK interjections and SFX route to sfx_passthrough
def test_short_cjk_dialogue_triage():
    from manhua_pipeline.stages.stage3_translation import triage_region

    assert triage_region("诶？", config).reason == "sfx_passthrough"
    assert triage_region("诶?", config).reason == "sfx_passthrough"
    assert triage_region("啊！", config).reason == "sfx_passthrough"
    assert triage_region("嗯", config).reason == "sfx_passthrough"

    assert triage_region("额……", config).reason == "sfx_passthrough"
    assert triage_region("额", config).reason == "sfx_passthrough"
    assert triage_region("呼", config).reason == "sfx_passthrough"
    assert triage_region("哈", config).reason == "sfx_passthrough"

    assert triage_region("嗷", config).reason == "sfx_passthrough"
    assert triage_region("噗", config).reason == "sfx_passthrough"
    assert triage_region("吼", config).reason == "sfx_passthrough"

    assert triage_region("砰", config).reason == "sfx_passthrough"
    assert triage_region("啪", config).reason == "sfx_passthrough"
    assert triage_region("咚", config).reason == "sfx_passthrough"

    assert triage_region("一", config).reason == "trivial"
    assert triage_region("王", config).reason == "trivial"

    assert triage_region("滚吧！", config).reason == "translatable"
    assert triage_region("", config).reason == "empty"
    assert triage_region("Hello", config).reason == "no_cjk"


# Verify sfx_passthrough regions are passed to paraphrase with raw CJK text
def test_sfx_passthrough_partition():
    from manhua_pipeline.stages.stage3_translation import _partition_by_triage

    region = {"region_id": "P003_R003", "page_number": 3, "original_text": "哈"}
    translatable, hard_skipped, trivial_fallback = _partition_by_triage(
        [region], config, {}
    )
    assert translatable == []
    assert trivial_fallback == []
    row = hard_skipped[0]
    assert row["translated"] is True
    assert row["literal_translation"] == "哈"
    assert row["translation_source"] == "sfx_passthrough"
    assert row["skip_reason"] is None


# Verify failed meaningful translations hand off to paraphrase when configured
def test_meaningful_translation_failure_hands_off_without_manual_override(
    tmp_path, monkeypatch
):
    from manhua_pipeline.stages.stage3_translation.engine import (
        _handle_unresolved_region,
    )

    monkeypatch.setattr(config, "TRANSLATOR_BACKEND", "ollama")
    monkeypatch.setattr(
        config, "TRANSLATION_FAILURE_HANDOFF_TO_PARAPHRASE", True, raising=False
    )
    region = {
        "region_id": "P001_R001",
        "page_number": 1,
        "original_text": "求……",
    }
    handled = _handle_unresolved_region(
        region=region,
        initial_reason="missing_after_batch",
        final_reason="missing_after_retry",
        runtime=SimpleNamespace(can_prompt=True),
        run_id="test-run",
        chapter="chapter",
        ws=tmp_path,
        config=config,
        translation_map={},
        glossary_data={},
    )

    assert handled is True
    assert region["translation_source"] == "translation_recovery"
    assert not (tmp_path / "logs" / "manual_override_requests.json").exists()


# Verify short speech bubbles with ASCII ellipses route to paraphrase
def test_short_speech_bubble_ascii_ellipsis_passthrough():
    from manhua_pipeline.stages.stage3_translation import _partition_by_triage

    region = {
        "region_id": "P024_R001",
        "page_number": 24,
        "type": config.TYPE_SPEECH,
        "render": True,
        "is_free_text": False,
        "original_text": "求....",
    }
    translatable, hard_skipped, trivial_fallback = _partition_by_triage(
        [region], config, {}
    )

    assert translatable == []
    assert trivial_fallback == []
    row = hard_skipped[0]
    assert row["translated"] is True
    assert row["literal_translation"] == "求...."
    assert row["translation_source"] == "speech_bubble_passthrough"
    assert row["skip_reason"] is None


# Verify bare single-character speech bubbles pass through to paraphrase
def test_bare_single_cjk_speech_bubble_passthrough():
    from manhua_pipeline.stages.stage3_translation import _partition_by_triage

    region = {
        "region_id": "P024_R003",
        "page_number": 24,
        "type": config.TYPE_SPEECH,
        "render": True,
        "is_free_text": False,
        "original_text": "求",
    }
    translatable, hard_skipped, trivial_fallback = _partition_by_triage(
        [region], config, {}
    )

    assert translatable == []
    assert trivial_fallback == []
    row = hard_skipped[0]
    assert row["translated"] is True
    assert row["literal_translation"] == "求"
    assert row["translation_source"] == "speech_bubble_passthrough"
    assert row["skip_reason"] is None


# Verify short CJK in non-speech narration remains classified as trivial
def test_short_background_sfx_remains_trivial():
    from manhua_pipeline.stages.stage3_translation import _partition_by_triage

    region = {
        "region_id": "P024_R002",
        "page_number": 24,
        "type": config.TYPE_NARRATION,
        "render": False,
        "is_free_text": True,
        "original_text": "求....",
    }
    translatable, hard_skipped, trivial_fallback = _partition_by_triage(
        [region], config, {}
    )

    assert translatable == []
    assert trivial_fallback == []
    row = hard_skipped[0]
    assert row["translated"] is False
    assert row["literal_translation"] == ""
    assert row["skip_reason"] == "trivial_manual_review"


# Verify disabling TRANSLATE_SHORT_CJK_DIALOGUE reverts to trivial skipping
def test_short_cjk_dialogue_toggle_off(monkeypatch):
    from manhua_pipeline.stages.stage3_translation import triage_region

    monkeypatch.setattr(config, "TRANSLATE_SHORT_CJK_DIALOGUE", False, raising=False)
    assert triage_region("诶？", config).reason == "trivial"


# Verify punctuation-only bubbles pass through with ASCII-normalized punctuation
def test_punct_only_region_passthrough():
    from manhua_pipeline.stages.stage3_translation import (
        _partition_by_triage,
        triage_region,
    )

    for t in ("？", "?", "！？", "?!", "……"):
        assert triage_region(t, config).reason == "punct_only", t
    assert triage_region("，", config).reason == "no_cjk"
    assert triage_region("Hello?", config).reason == "no_cjk"
    assert triage_region("一", config).reason == "trivial"

    region = {"region_id": "P022_R002", "page_number": 22, "original_text": "？"}
    translatable, hard_skipped, trivial_fallback = _partition_by_triage(
        [region], config, {}
    )
    assert translatable == []
    row = hard_skipped[0]
    assert row["translated"] is True
    assert row["literal_translation"] == "?"
    assert row["translation_source"] == "punct_passthrough"


# Verify disabling RENDER_PUNCT_ONLY_REGIONS treats punctuation-only bubbles as no_cjk
def test_punct_only_toggle_off(monkeypatch):
    from manhua_pipeline.stages.stage3_translation import triage_region

    monkeypatch.setattr(config, "RENDER_PUNCT_ONLY_REGIONS", False, raising=False)
    assert triage_region("？", config).reason == "no_cjk"


# Verify low-confidence real dialogue interjections are preserved via sfx_passthrough
def test_low_confidence_real_dialogue_not_dropped(tmp_path):
    from manhua_pipeline.stages.stage3_translation import run_translation

    ws = tmp_path / "workspace"
    low_conf_entry = {
        "region_id": "P001_R001",
        "page_number": 1,
        "type": "speech_bubble",
        "original_text": "啊……",
        "text_direction": "horizontal",
        "ocr_confidence": 0.15,
        "ocr_confidence_min": 0.15,
        "has_usable_text": False,
        "do_not_render": False,
        "needs_correction": True,
        "edge_touching": False,
        "edge": "none",
        "note": None,
        "watermark_filtered": False,
    }
    _setup(ws, [low_conf_entry])
    run_translation(str(ws), config)

    trans_path = ws / "stage3_translation" / "translation.json"
    assert trans_path.exists()
    data = json.loads(trans_path.read_text(encoding="utf-8"))
    region = next(
        (r for r in data.get("results", []) if r["region_id"] == "P001_R001"), None
    )
    assert region is not None
    assert region["translated"] is True
    assert region["translation_source"] == "sfx_passthrough"
