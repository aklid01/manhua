import json

import config


def _setup(ws, tr_results):
    ws.mkdir(parents=True, exist_ok=True)
    manifest = {
        "chapter_id": "t",
        "total_pages": 1,
        "pages": [{"page_number": 1, "filename": "001.png", "skip": False}],
        "current_stage": "paraphrase",
        "completed_stages": ["import", "detect", "ocr", "translate"],
        "warning_count": 0,
        "status": "in_progress",
    }
    (ws / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (ws / "stage3_translation").mkdir(parents=True, exist_ok=True)
    tr = {
        "chapter_id": "t",
        "stage": "translation",
        "generated_at": "now",
        "translator_backend": "manual",
        "glossary_version": "v1",
        "results": tr_results,
    }
    (ws / "stage3_translation" / "translation.json").write_text(
        json.dumps(tr, ensure_ascii=False), encoding="utf-8"
    )


def _tr_entry(rid, literal, translated=True, skip_reason=None):
    return {
        "region_id": rid,
        "page_number": 1,
        "original_text": "x",
        "literal_translation": literal,
        "translated": translated,
        "skip_reason": skip_reason,
        "glossary_terms_applied": [],
        "glossary_conflict": False,
    }


def test_paraphrase_manual_awaits_response(tmp_path):
    from manhua_pipeline.stages.stage4_paraphrase import run_paraphrase

    ws = tmp_path / "workspace"
    _setup(ws, [_tr_entry("P001_R001", "Get out!")])
    result = run_paraphrase(str(ws), config)
    assert result is None
    assert (ws / "stage4_paraphrase" / "paraphrase_prompt.json").exists()
    assert not (ws / "stage4_paraphrase" / "paraphrase.json").exists()
    m = json.loads((ws / "manifest.json").read_text())
    assert m["current_stage"] == "paraphrase"  # NOT advanced


def test_paraphrase_prompt_only_translated(tmp_path):
    from manhua_pipeline.stages.stage4_paraphrase import run_paraphrase

    ws = tmp_path / "workspace"
    _setup(
        ws,
        [
            _tr_entry("P001_R001", "Get out!", translated=True),
            _tr_entry("P001_R002", "", translated=False, skip_reason="no_usable_text"),
        ],
    )
    run_paraphrase(str(ws), config)
    bundle = json.loads(
        (ws / "stage4_paraphrase" / "paraphrase_prompt.json").read_text(
            encoding="utf-8"
        )
    )
    ids = [item["region_id"] for item in bundle["regions"]]
    assert ids == ["P001_R001"]


def test_paraphrase_ingests_response(tmp_path):
    from manhua_pipeline.stages.stage4_paraphrase import run_paraphrase

    ws = tmp_path / "workspace"
    _setup(
        ws,
        [
            _tr_entry("P001_R001", "Get out!", translated=True),
            _tr_entry("P001_R002", "", translated=False, skip_reason="no_usable_text"),
        ],
    )
    run_paraphrase(str(ws), config)
    (ws / "stage4_paraphrase" / "paraphrase_response.json").write_text(
        json.dumps({"P001_R001": "Get lost!"}), encoding="utf-8"
    )
    out = run_paraphrase(str(ws), config)
    assert out is not None
    pp = json.loads(
        (ws / "stage4_paraphrase" / "paraphrase.json").read_text(encoding="utf-8")
    )
    by_id = {r["region_id"]: r for r in pp["results"]}
    assert by_id["P001_R001"]["final_text"] == "Get lost!"
    assert by_id["P001_R001"]["paraphrased"] is True
    assert by_id["P001_R001"]["char_count"] == len("Get lost!")
    assert by_id["P001_R002"]["paraphrased"] is False
    assert by_id["P001_R002"]["final_text"] == ""
    assert by_id["P001_R002"]["skip_reason"] == "no_usable_text"
    m = json.loads((ws / "manifest.json").read_text())
    assert m["current_stage"] == "render"
    assert "paraphrase" in m["completed_stages"]


def test_paraphrase_register_heuristic(tmp_path):
    from manhua_pipeline.stages.stage4_paraphrase import run_paraphrase

    ws = tmp_path / "workspace"
    _setup(
        ws,
        [
            _tr_entry("P001_R001", "I am not doing this anymore.", translated=True),
            _tr_entry("P001_R002", "Screw this, I quit!", translated=True),
        ],
    )
    run_paraphrase(str(ws), config)
    (ws / "stage4_paraphrase" / "paraphrase_response.json").write_text(
        json.dumps(
            {
                "P001_R001": "I'm done here.",
                "P001_R002": "Screw this, I quit!",
            }
        ),
        encoding="utf-8",
    )
    run_paraphrase(str(ws), config)
    pp = json.loads(
        (ws / "stage4_paraphrase" / "paraphrase.json").read_text(encoding="utf-8")
    )
    by_id = {r["region_id"]: r for r in pp["results"]}
    assert by_id["P001_R002"]["register"] == "rude"
    assert by_id["P001_R001"]["register"] in ("neutral", "rude")


def test_paraphrase_all_passthrough_completes(tmp_path):
    from manhua_pipeline.stages.stage4_paraphrase import run_paraphrase

    ws = tmp_path / "workspace"
    _setup(
        ws,
        [_tr_entry("P001_R001", "", translated=False, skip_reason="no_usable_text")],
    )
    out = run_paraphrase(str(ws), config)
    assert out is not None
    pp = json.loads(
        (ws / "stage4_paraphrase" / "paraphrase.json").read_text(encoding="utf-8")
    )
    assert pp["results"][0]["paraphrased"] is False
    m = json.loads((ws / "manifest.json").read_text())
    assert m["current_stage"] == "render"


def test_paraphrase_passes_glossary_conflict(tmp_path):
    from manhua_pipeline.stages.stage4_paraphrase import run_paraphrase

    ws = tmp_path / "workspace"
    entry = _tr_entry("P001_R001", "Boss", translated=True)
    entry["glossary_conflict"] = True
    _setup(ws, [entry])
    run_paraphrase(str(ws), config)
    (ws / "stage4_paraphrase" / "paraphrase_response.json").write_text(
        json.dumps({"P001_R001": "Boss"}), encoding="utf-8"
    )
    run_paraphrase(str(ws), config)
    pp = json.loads(
        (ws / "stage4_paraphrase" / "paraphrase.json").read_text(encoding="utf-8")
    )
    assert pp["results"][0]["glossary_conflict"] is True


def test_paraphrase_bundle_includes_glossary(tmp_path):
    from manhua_pipeline.stages.stage4_paraphrase import run_paraphrase

    ws = tmp_path / "workspace"
    _setup(ws, [_tr_entry("P001_R001", "The manager is here.")])
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
    (ws.parent / "glossary.json").write_text(
        json.dumps(glossary, ensure_ascii=False), encoding="utf-8"
    )
    run_paraphrase(str(ws), config)
    bundle = json.loads(
        (ws / "stage4_paraphrase" / "paraphrase_prompt.json").read_text(
            encoding="utf-8"
        )
    )
    assert "glossary" in bundle
    assert any(t["target_term"] == "Manager" for t in bundle["glossary"])


def test_paraphrase_prompt_has_aggressive_clause(tmp_path):
    from manhua_pipeline.stages.stage4_paraphrase import run_paraphrase

    ws = tmp_path / "workspace"
    _setup(ws, [_tr_entry("P001_R001", "Get out!")])
    run_paraphrase(str(ws), config)
    bundle = json.loads(
        (ws / "stage4_paraphrase" / "paraphrase_prompt.json").read_text(
            encoding="utf-8"
        )
    )
    instr = bundle["READ_FIRST"].lower()
    assert "aggressively" in instr
    assert "verbatim" in instr


def test_paraphrase_missing_falls_back_to_literal(tmp_path):
    from manhua_pipeline.stages.stage4_paraphrase import run_paraphrase

    ws = tmp_path / "workspace"
    _setup(
        ws,
        [
            _tr_entry("P001_R001", "Get out!", translated=True),
            _tr_entry("P001_R002", "I quit!", translated=True),
        ],
    )
    run_paraphrase(str(ws), config)
    (ws / "stage4_paraphrase" / "paraphrase_response.json").write_text(
        json.dumps({"P001_R001": "Get lost!"}), encoding="utf-8"
    )
    run_paraphrase(str(ws), config)
    pp = json.loads(
        (ws / "stage4_paraphrase" / "paraphrase.json").read_text(encoding="utf-8")
    )
    by_id = {r["region_id"]: r for r in pp["results"]}
    assert by_id["P001_R002"]["paraphrased"] is False
    assert by_id["P001_R002"]["final_text"] == "I quit!"  # fell back to literal
    assert by_id["P001_R002"]["char_count"] == len("I quit!")


def test_paraphrase_override_used_verbatim(tmp_path):
    from manhua_pipeline.stages.stage4_paraphrase import run_paraphrase

    ws = tmp_path / "workspace"
    _setup(ws, [_tr_entry("P001_R001", "Manager? My ass!", translated=True)])
    (ws / "overrides.json").write_text(
        json.dumps({"P001_R001": "Manager? My ass!"}), encoding="utf-8"
    )
    out = run_paraphrase(str(ws), config)
    assert out is not None
    pp = json.loads(
        (ws / "stage4_paraphrase" / "paraphrase.json").read_text(encoding="utf-8")
    )
    r = pp["results"][0]
    assert r["final_text"] == "Manager? My ass!"
    assert r["paraphrase_source"] == "override"
    assert r["char_count"] == len("Manager? My ass!")
    m = json.loads((ws / "manifest.json").read_text())
    assert m["current_stage"] == "render"


def test_paraphrase_override_not_bundled(tmp_path):
    from manhua_pipeline.stages.stage4_paraphrase import run_paraphrase

    ws = tmp_path / "workspace"
    _setup(
        ws,
        [
            _tr_entry("P001_R001", "Get out!", translated=True),
            _tr_entry("P001_R002", "Boss", translated=True),
        ],
    )
    (ws / "overrides.json").write_text(
        json.dumps({"P001_R002": "Manager!"}), encoding="utf-8"
    )
    run_paraphrase(str(ws), config)
    bundle = json.loads(
        (ws / "stage4_paraphrase" / "paraphrase_prompt.json").read_text(
            encoding="utf-8"
        )
    )
    ids = [i["region_id"] for i in bundle["regions"]]
    assert ids == ["P001_R001"]


def test_validate_batch_newline_normalization():
    from manhua_pipeline.stages.stage4_paraphrase import OllamaBackend

    parsed = {"P001_R001": "Name: Lin\\nAge: 18\\r\\nGender: Male"}
    expected = {"P001_R001"}
    accepted, missing, unexpected = OllamaBackend._validate_batch(
        parsed, expected, literals={}
    )
    assert accepted["P001_R001"] == "Name: Lin\nAge: 18\nGender: Male"
    assert missing == []
    assert unexpected == []

