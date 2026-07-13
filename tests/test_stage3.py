import json

import config


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
    run_translation(str(ws), config)
    (ws / "stage3_translation" / "translation_response.json").write_text(
        json.dumps({"P001_R001": "Boss"}), encoding="utf-8"
    )
    run_translation(str(ws), config)
    tr = json.loads(
        (ws / "stage3_translation" / "translation.json").read_text(encoding="utf-8")
    )
    r = tr["results"][0]
    assert r["glossary_conflict"] is True
    assert "jingli" in r["glossary_terms_applied"]


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
    run_translation(str(ws), config)
    (ws / "stage3_translation" / "translation_response.json").write_text(
        json.dumps({"P001_R001": "Boss"}), encoding="utf-8"
    )
    run_translation(str(ws), config)
    tr = json.loads(
        (ws / "stage3_translation" / "translation.json").read_text(encoding="utf-8")
    )
    r = tr["results"][0]
    assert r["glossary_conflict"] is True
    assert r["literal_translation"] == "Boss"  # text NOT defaced


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
