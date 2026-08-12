import json

import pytest

import config


@pytest.fixture(autouse=True)
def _force_manual_backend(monkeypatch):
    monkeypatch.setattr(config, "TRANSLATOR_BACKEND", "manual", raising=False)


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


def test_translation_glossary_locked_term(tmp_path):
    from manhua_pipeline.stages.stage3_translation import run_translation

    ws = tmp_path / "workspace"
    glossary = {
        "version": "v1",
        "updated_at": "now",
        "terms": [
            {
                "term_id": "jingli",
                "source_term": "\u7ecf\u7406",
                "target_term": "Manager",
                "category": "title",
                "locked": True,
                "auto_seeded": False,
                "source_region": None,
                "notes": "",
            }
        ],
    }
    _setup(ws, [_ocr_entry("P001_R001", "\u7ecf\u7406")], glossary=glossary)
    # Locked glossary pre-resolution: the region is resolved before the backend is called.
    # The response file with a conflicting value is never applied.
    out = run_translation(str(ws), config)
    # With manual backend and a locked exact match, the region goes to usable_for_backend=[]
    # so the backend isn't called and translation completes directly from the locked entry.
    assert out is not None
    tr = json.loads(
        (ws / "stage3_translation" / "translation.json").read_text(encoding="utf-8")
    )
    r = tr["results"][0]
    assert r["translated"] is True
    assert r["literal_translation"] == "Manager"
    assert r["translation_source"] == "locked_glossary"


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


def test_translation_glossary_conflict_does_not_mutate(tmp_path):
    from manhua_pipeline.stages.stage3_translation import run_translation

    ws = tmp_path / "workspace"
    glossary = {
        "version": "v1",
        "updated_at": "now",
        "terms": [
            {
                "term_id": "jingli",
                "source_term": "\u7ecf\u7406",
                "target_term": "Manager",
                "category": "title",
                "locked": True,
                "auto_seeded": False,
                "source_region": None,
                "notes": "",
            }
        ],
    }
    _setup(ws, [_ocr_entry("P001_R001", "\u7ecf\u7406")], glossary=glossary)
    out = run_translation(str(ws), config)
    # Locked glossary pre-resolution: "Manager" is applied directly.
    # The locked value wins - no conflict can arise and no backend is consulted.
    assert out is not None
    tr = json.loads(
        (ws / "stage3_translation" / "translation.json").read_text(encoding="utf-8")
    )
    r = tr["results"][0]
    assert r["translated"] is True
    assert r["literal_translation"] == "Manager"  # locked value, NOT defaced
    assert r["glossary_conflict"] is False  # no conflict: backend was never consulted


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


def test_translation_no_overrides_file_unchanged(tmp_path):
    from manhua_pipeline.stages.stage3_translation import run_translation

    ws = tmp_path / "workspace"
    _setup(ws, [_ocr_entry("P001_R001", "滚吧！", usable=True)])
    result = run_translation(str(ws), config)
    assert result is None
    assert (ws / "stage3_translation" / "translation_prompt.json").exists()


def test_validate_batch_newline_normalization():
    from manhua_pipeline.stages.stage3_translation import OllamaBackend

    parsed = {"P001_R001": "Line 1\\nLine 2\\r\\nLine 3"}
    expected = {"P001_R001"}
    accepted, missing, unexpected = OllamaBackend._validate_batch(parsed, expected)
    assert accepted["P001_R001"] == "Line 1\nLine 2\nLine 3"
    assert missing == []
    assert unexpected == []


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
    # Non-CJK (Latin) text is trivial: triage hard-skips it upstream, so it
    # is correctly non-blocking here. (Was stale: expected False.)
    assert _is_trivial_region("ABC", config) is True


def test_short_cjk_dialogue_triage():
    """Single-CJK interjections/SFX bypass translation and go to paraphrase directly.

    These chars are real dialogue but the translation model (small/local) returns empty
    for them consistently. They are routed sfx_passthrough: translated=True with the raw
    CJK as literal_translation so the stronger paraphrase model handles them in context.
    """
    from manhua_pipeline.stages.stage3_translation import triage_region

    # Single-CJK interjections with dialogue punct -> sfx_passthrough (not translatable)
    assert triage_region("诶？", config).reason == "sfx_passthrough"  # interjection + ?
    assert triage_region("诶?", config).reason == "sfx_passthrough"  # ASCII ?
    assert triage_region("啊！", config).reason == "sfx_passthrough"  # interjection + !
    assert (
        triage_region("嗯", config).reason == "sfx_passthrough"
    )  # bare known interjection

    # Regressions: chap 64 p23 (额……) and chap 62 p3 (呼)
    assert (
        triage_region("额……", config).reason == "sfx_passthrough"
    )  # hesitation + ellipsis
    assert triage_region("额", config).reason == "sfx_passthrough"  # bare hesitation
    assert triage_region("呼", config).reason == "sfx_passthrough"  # bare exhale
    assert (
        triage_region("哈", config).reason == "sfx_passthrough"
    )  # chap 62 p3 reported case

    # Other breath / exclamation chars
    assert triage_region("嗷", config).reason == "sfx_passthrough"  # pain cry
    assert triage_region("噗", config).reason == "sfx_passthrough"  # pfft
    assert triage_region("吼", config).reason == "sfx_passthrough"  # roar

    # Onomatopoeia sound effects
    assert triage_region("砰", config).reason == "sfx_passthrough"  # bang
    assert triage_region("啪", config).reason == "sfx_passthrough"  # slap
    assert triage_region("咚", config).reason == "sfx_passthrough"  # thud

    # Genuine debris / non-dialogue single chars -> still trivial (skipped)
    assert triage_region("一", config).reason == "trivial"  # numeral one, no signal
    assert triage_region("王", config).reason == "trivial"  # lone char, no signal

    # Multi-CJK -> normal translatable path unchanged
    assert triage_region("滚吧！", config).reason == "translatable"
    assert triage_region("", config).reason == "empty"
    assert triage_region("Hello", config).reason == "no_cjk"


def test_sfx_passthrough_partition():
    """sfx_passthrough regions are emitted as translated=True rows carrying raw CJK.

    This is the key contract: stage4 sees them as 'usable' (translated=True) and
    the paraphrase model receives the raw CJK as literal_translation.
    """
    from manhua_pipeline.stages.stage3_translation import _partition_by_triage

    region = {"region_id": "P003_R003", "page_number": 3, "original_text": "哈"}
    translatable, hard_skipped, trivial_fallback = _partition_by_triage(
        [region], config, {}
    )
    assert translatable == []
    assert trivial_fallback == []
    row = hard_skipped[0]
    assert row["translated"] is True
    assert row["literal_translation"] == "哈"  # raw CJK passed to paraphrase
    assert row["translation_source"] == "sfx_passthrough"
    assert row["skip_reason"] is None


def test_short_speech_bubble_ascii_ellipsis_passthrough():
    """ASCII OCR ellipses in short speech bubbles reach paraphrase."""
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


def test_short_background_sfx_remains_trivial():
    """Short CJK in a free-text/narration region is not promoted to paraphrase."""
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


def test_short_cjk_dialogue_toggle_off(monkeypatch):
    """TRANSLATE_SHORT_CJK_DIALOGUE=False restores the strict skip."""
    from manhua_pipeline.stages.stage3_translation import triage_region

    monkeypatch.setattr(config, "TRANSLATE_SHORT_CJK_DIALOGUE", False, raising=False)
    assert triage_region("诶？", config).reason == "trivial"


def test_punct_only_region_passthrough():
    """Punctuation-only bubbles ("？") render deterministically, no LLM call.

    Regression guard for P022_R002: PaddleOCR dropped "诶" leaving "？".
    Rendering "?" is still correct; it must NOT be dropped as no_cjk noise.
    """
    from manhua_pipeline.stages.stage3_translation import (
        _partition_by_triage,
        triage_region,
    )

    for t in ("？", "?", "！？", "?!", "……"):
        assert triage_region(t, config).reason == "punct_only", t
    assert triage_region("，", config).reason == "no_cjk"  # lone comma, no signal
    assert triage_region("Hello?", config).reason == "no_cjk"  # has letters
    assert triage_region("一", config).reason == "trivial"  # CJK numeral

    region = {"region_id": "P022_R002", "page_number": 22, "original_text": "？"}
    translatable, hard_skipped, trivial_fallback = _partition_by_triage(
        [region], config, {}
    )
    assert translatable == []
    row = hard_skipped[0]
    assert row["translated"] is True
    assert row["literal_translation"] == "?"
    assert row["translation_source"] == "punct_passthrough"


def test_punct_only_toggle_off(monkeypatch):
    from manhua_pipeline.stages.stage3_translation import triage_region

    monkeypatch.setattr(config, "RENDER_PUNCT_ONLY_REGIONS", False, raising=False)
    assert triage_region("？", config).reason == "no_cjk"


def test_low_confidence_real_dialogue_not_dropped(tmp_path):
    """Regions with real CJK dialogue and has_usable_text=False must not be dropped.

    Regression guard for the "啊……" case: OCR read one CJK char at low confidence,
    so has_usable_text=False. Single-CJK interjections now take the sfx_passthrough
    route - they bypass translation entirely and land in translation.json as
    translated=True, source="sfx_passthrough". The region must never be silently dropped.
    """
    from manhua_pipeline.stages.stage3_translation import run_translation

    ws = tmp_path / "workspace"
    low_conf_entry = {
        "region_id": "P001_R001",
        "page_number": 1,
        "type": "speech_bubble",
        "original_text": "\u554a\u2026\u2026",
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
    assert trans_path.exists(), "translation.json must be written"
    data = json.loads(trans_path.read_text(encoding="utf-8"))
    region = next(
        (r for r in data.get("results", []) if r["region_id"] == "P001_R001"), None
    )
    assert region is not None, "P001_R001 must appear in translation.json results"
    assert region["translated"] is True, (
        "sfx_passthrough must mark the region as translated=True so stage4 picks it up"
    )
    assert region["translation_source"] == "sfx_passthrough", (
        "single-CJK interjection must bypass the LLM and be marked sfx_passthrough"
    )
