"""Offline tests for the Stage 4 Ollama paraphrase backend.

All Ollama HTTP calls are monkeypatched via OllamaBackend._call_ollama, so these
tests never touch the network and require no running `ollama serve`.
"""

import json

import pytest

import config


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _setup(ws, trans_results, glossary=None):
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
    trans = {
        "chapter_id": "t",
        "stage": "translation",
        "generated_at": "now",
        "translator_backend": "ollama",
        "glossary_version": "v1",
        "results": trans_results,
    }
    (ws / "stage3_translation" / "translation.json").write_text(
        json.dumps(trans, ensure_ascii=False), encoding="utf-8"
    )
    if glossary is not None:
        (ws.parent / "glossary.json").write_text(
            json.dumps(glossary, ensure_ascii=False), encoding="utf-8"
        )


def _trans_entry(rid, literal, translated=True):
    return {
        "region_id": rid,
        "page_number": 1,
        "original_text": "\u539f\u6587",
        "literal_translation": literal,
        "translated": translated,
        "skip_reason": None if translated else "no_usable_text",
        "glossary_terms_applied": [],
        "glossary_conflict": False,
        "translation_source": "llm" if translated else None,
    }


@pytest.fixture
def ollama_config(monkeypatch):
    monkeypatch.setattr(config, "PARAPHRASE_BACKEND", "ollama", raising=False)
    monkeypatch.setattr(config, "OLLAMA_PARA_HOST", "http://localhost:11434", raising=False)
    monkeypatch.setattr(config, "OLLAMA_PARA_MODEL", "qwen2.5:3b-instruct", raising=False)
    monkeypatch.setattr(config, "OLLAMA_PARA_BATCH_SIZE", 2, raising=False)
    monkeypatch.setattr(config, "OLLAMA_PARA_TIMEOUT", 30, raising=False)
    monkeypatch.setattr(config, "OLLAMA_PARA_TEMPERATURE", 0.7, raising=False)
    monkeypatch.setattr(config, "OLLAMA_PARA_MAX_RETRIES", 3, raising=False)
    monkeypatch.setattr(config, "OLLAMA_PARA_RETRY_BACKOFF", 0.0, raising=False)
    monkeypatch.setattr(config, "OLLAMA_PARA_MIN_COMPLETION_RATIO", 0.80, raising=False)
    monkeypatch.setattr(config, "OLLAMA_PARA_PROMPT_VERSION", "paraphrase-v1", raising=False)
    return config


def _patch_call(monkeypatch, responder):
    from manhua_pipeline.stages import stage4_paraphrase as s4
    monkeypatch.setattr(
        s4.OllamaBackend, "_call_ollama", lambda self, user_prompt: responder(user_prompt)
    )
    return s4


def _canned(mapping):
    payload = json.dumps(mapping, ensure_ascii=False)
    return lambda _prompt: payload


def _read_paraphrase(ws):
    return json.loads(
        (ws / "stage4_paraphrase" / "paraphrase.json").read_text(encoding="utf-8")
    )


# ---------------------------------------------------------------------------
# 1. Backend selection
# ---------------------------------------------------------------------------


def test_get_backend_returns_ollama(ollama_config):
    from manhua_pipeline.stages.stage4_paraphrase import OllamaBackend, _get_backend
    assert isinstance(_get_backend(ollama_config), OllamaBackend)


# ---------------------------------------------------------------------------
# 2. Successful paraphrase written + manifest advances
# ---------------------------------------------------------------------------


def test_successful_paraphrase_written(tmp_path, ollama_config, monkeypatch):
    _patch_call(monkeypatch, _canned({"P001_R001": "Get lost!"}))
    from manhua_pipeline.stages.stage4_paraphrase import run_paraphrase

    ws = tmp_path / "workspace"
    _setup(ws, [_trans_entry("P001_R001", "Get out of here!")])
    out = run_paraphrase(str(ws), ollama_config)

    assert out is not None
    pa = _read_paraphrase(ws)
    by_id = {r["region_id"]: r for r in pa["results"]}
    assert by_id["P001_R001"]["final_text"] == "Get lost!"
    assert by_id["P001_R001"]["paraphrased"] is True
    m = json.loads((ws / "manifest.json").read_text())
    assert m["current_stage"] == "render"


# ---------------------------------------------------------------------------
# 3. Multiple batches -> every region rewritten once
# ---------------------------------------------------------------------------


def test_multiple_batches(tmp_path, ollama_config, monkeypatch):
    calls = {"n": 0}

    def responder(prompt):
        calls["n"] += 1
        out = {}
        for rid in ["P001_R001", "P001_R002", "P001_R003"]:
            if rid in prompt:
                out[rid] = f"punchy {rid}"
        return json.dumps(out)

    _patch_call(monkeypatch, responder)
    from manhua_pipeline.stages.stage4_paraphrase import run_paraphrase

    ws = tmp_path / "workspace"
    _setup(ws, [
        _trans_entry("P001_R001", "a a a"),
        _trans_entry("P001_R002", "b b b"),
        _trans_entry("P001_R003", "c c c"),
    ])
    run_paraphrase(str(ws), ollama_config)

    pa = _read_paraphrase(ws)
    by_id = {r["region_id"]: r for r in pa["results"]}
    assert by_id["P001_R001"]["final_text"] == "punchy P001_R001"
    assert by_id["P001_R002"]["final_text"] == "punchy P001_R002"
    assert by_id["P001_R003"]["final_text"] == "punchy P001_R003"
    assert calls["n"] >= 2


# ---------------------------------------------------------------------------
# 4. Code-fenced JSON + parser helpers
# ---------------------------------------------------------------------------


def test_code_fenced_json_parses(tmp_path, ollama_config, monkeypatch):
    fenced = "```json\n" + json.dumps({"P001_R001": "Fenced!"}) + "\n```"
    _patch_call(monkeypatch, lambda _p: fenced)
    from manhua_pipeline.stages.stage4_paraphrase import run_paraphrase

    ws = tmp_path / "workspace"
    _setup(ws, [_trans_entry("P001_R001", "x")])
    run_paraphrase(str(ws), ollama_config)

    pa = _read_paraphrase(ws)
    assert pa["results"][0]["final_text"] == "Fenced!"


def test_parse_json_static_helpers():
    from manhua_pipeline.stages.stage4_paraphrase import OllamaBackend
    assert OllamaBackend._parse_json('```json\n{"a": "b"}\n```') == {"a": "b"}
    assert OllamaBackend._parse_json('prose {"a": "b"} tail') == {"a": "b"}
    assert OllamaBackend._parse_json("garbage") == {}
    assert OllamaBackend._parse_json("") == {}
    assert OllamaBackend._parse_json("[1,2,3]") == {}


# ---------------------------------------------------------------------------
# 5. Invalid JSON -> strict retry recovers
# ---------------------------------------------------------------------------


def test_invalid_json_recovers(tmp_path, ollama_config, monkeypatch):
    state = {"first": True}

    def responder(prompt):
        if state["first"]:
            state["first"] = False
            return "not json"
        out = {}
        for rid in ["P001_R001", "P001_R002"]:
            if rid in prompt:
                out[rid] = f"ok {rid}"
        return json.dumps(out)

    _patch_call(monkeypatch, responder)
    from manhua_pipeline.stages.stage4_paraphrase import run_paraphrase

    ws = tmp_path / "workspace"
    _setup(ws, [_trans_entry("P001_R001", "a"), _trans_entry("P001_R002", "b")])
    out = run_paraphrase(str(ws), ollama_config)

    assert out is not None
    pa = _read_paraphrase(ws)
    by_id = {r["region_id"]: r for r in pa["results"]}
    assert by_id["P001_R001"]["paraphrased"] is True
    assert by_id["P001_R002"]["paraphrased"] is True


# ---------------------------------------------------------------------------
# 6. Missing paraphrase -> literal fallback
# ---------------------------------------------------------------------------


def test_missing_falls_back_to_literal(tmp_path, ollama_config, monkeypatch):
    _patch_call(monkeypatch, _canned({"P001_R001": "Nice one!"}))
    from manhua_pipeline.stages.stage4_paraphrase import run_paraphrase

    ws = tmp_path / "workspace"
    _setup(ws, [_trans_entry("P001_R001", "a"), _trans_entry("P001_R002", "keep me literal")])
    run_paraphrase(str(ws), ollama_config)

    pa = _read_paraphrase(ws)
    by_id = {r["region_id"]: r for r in pa["results"]}
    assert by_id["P001_R001"]["paraphrased"] is True
    assert by_id["P001_R002"]["paraphrased"] is False
    assert by_id["P001_R002"]["final_text"] == "keep me literal"
    assert by_id["P001_R002"]["paraphrase_source"] == "literal_fallback"


# ---------------------------------------------------------------------------
# 7. Echo of literal is ALLOWED (not dropped)
# ---------------------------------------------------------------------------


def test_echo_of_literal_allowed():
    from manhua_pipeline.stages.stage4_paraphrase import OllamaBackend
    literals = {"P001_R001": "ACME Corp", "P001_R002": "hello there"}
    parsed = {"P001_R001": "ACME Corp", "P001_R002": "hey!"}
    accepted, missing, _ = OllamaBackend._validate_batch(
        parsed, {"P001_R001", "P001_R002"}, literals
    )
    assert accepted == {"P001_R001": "ACME Corp", "P001_R002": "hey!"}
    assert missing == []


# ---------------------------------------------------------------------------
# 8. Unexpected IDs rejected; invalid values dropped
# ---------------------------------------------------------------------------


def test_unexpected_ids_rejected():
    from manhua_pipeline.stages.stage4_paraphrase import OllamaBackend
    parsed = {"P001_R001": "good", "P999_R999": "ghost"}
    accepted, missing, unexpected = OllamaBackend._validate_batch(
        parsed, {"P001_R001"}, {"P001_R001": "g"}
    )
    assert accepted == {"P001_R001": "good"}
    assert unexpected == ["P999_R999"]
    assert missing == []


def test_invalid_values_dropped():
    from manhua_pipeline.stages.stage4_paraphrase import OllamaBackend
    parsed = {
        "P001_R001": "",
        "P001_R002": "   ",
        "P001_R003": None,
        "P001_R004": 5,
        "P001_R005": ["x"],
        "P001_R006": "valid",
    }
    expected = {f"P001_R00{i}" for i in range(1, 7)}
    accepted, missing, _ = OllamaBackend._validate_batch(parsed, expected, {})
    assert accepted == {"P001_R006": "valid"}
    assert set(missing) == expected - {"P001_R006"}


# ---------------------------------------------------------------------------
# 9. Connectivity failure -> clear RuntimeError
# ---------------------------------------------------------------------------


def test_connectivity_failure_raises(tmp_path, ollama_config, monkeypatch):
    import urllib.error
    from manhua_pipeline.stages import stage4_paraphrase as s4

    def boom(req, timeout=None):
        raise urllib.error.URLError("refused")

    monkeypatch.setattr(s4.urllib.request, "urlopen", boom)
    from manhua_pipeline.stages.stage4_paraphrase import run_paraphrase

    ws = tmp_path / "workspace"
    _setup(ws, [_trans_entry("P001_R001", "a")])
    with pytest.raises(RuntimeError) as exc:
        run_paraphrase(str(ws), ollama_config)
    msg = str(exc.value)
    assert "localhost:11434" in msg
    assert "qwen2.5:3b-instruct" in msg
    assert "ollama serve" in msg


# ---------------------------------------------------------------------------
# 10. Empty input -> no Ollama call, completes via short-circuit
# ---------------------------------------------------------------------------


def test_empty_input_no_call(tmp_path, ollama_config, monkeypatch):
    called = {"n": 0}

    def responder(_p):
        called["n"] += 1
        return "{}"

    _patch_call(monkeypatch, responder)
    from manhua_pipeline.stages.stage4_paraphrase import run_paraphrase

    ws = tmp_path / "workspace"
    _setup(ws, [_trans_entry("P001_R001", "", translated=False)])
    out = run_paraphrase(str(ws), ollama_config)

    assert out is not None
    assert called["n"] == 0
    m = json.loads((ws / "manifest.json").read_text())
    assert m["current_stage"] == "render"


# ---------------------------------------------------------------------------
# 11. Soft completion gate: below ratio -> writes but does NOT advance
# ---------------------------------------------------------------------------


def test_below_threshold_does_not_advance(tmp_path, ollama_config, monkeypatch):
    monkeypatch.setattr(config, "OLLAMA_PARA_BATCH_SIZE", 10, raising=False)
    # 1 of 3 = 0.33 < 0.80
    _patch_call(monkeypatch, _canned({"P001_R001": "one"}))
    from manhua_pipeline.stages.stage4_paraphrase import run_paraphrase

    ws = tmp_path / "workspace"
    _setup(ws, [
        _trans_entry("P001_R001", "a"),
        _trans_entry("P001_R002", "b"),
        _trans_entry("P001_R003", "c"),
    ])
    out = run_paraphrase(str(ws), ollama_config)

    assert out is None
    m = json.loads((ws / "manifest.json").read_text())
    assert m["current_stage"] == "paraphrase"
    assert (ws / "stage4_paraphrase" / "paraphrase.json").exists()


# ---------------------------------------------------------------------------
# 12. Config validation + provenance metadata
# ---------------------------------------------------------------------------


def test_config_validation(monkeypatch):
    from manhua_pipeline.stages.stage4_paraphrase import _validate_ollama_para_config

    monkeypatch.setattr(config, "OLLAMA_PARA_HOST", "http://localhost:11434", raising=False)
    monkeypatch.setattr(config, "OLLAMA_PARA_MODEL", "qwen2.5:3b-instruct", raising=False)
    monkeypatch.setattr(config, "OLLAMA_PARA_TIMEOUT", 30, raising=False)
    monkeypatch.setattr(config, "OLLAMA_PARA_TEMPERATURE", 0.7, raising=False)
    monkeypatch.setattr(config, "OLLAMA_PARA_MAX_RETRIES", 3, raising=False)

    monkeypatch.setattr(config, "OLLAMA_PARA_BATCH_SIZE", 0, raising=False)
    with pytest.raises(ValueError):
        _validate_ollama_para_config(config)

    monkeypatch.setattr(config, "OLLAMA_PARA_BATCH_SIZE", 2, raising=False)
    monkeypatch.setattr(config, "OLLAMA_PARA_HOST", "ftp://bad", raising=False)
    with pytest.raises(ValueError):
        _validate_ollama_para_config(config)

    monkeypatch.setattr(config, "OLLAMA_PARA_HOST", "http://localhost:11434", raising=False)
    monkeypatch.setattr(config, "OLLAMA_PARA_MODEL", "", raising=False)
    with pytest.raises(ValueError):
        _validate_ollama_para_config(config)


def test_provenance_recorded(tmp_path, ollama_config, monkeypatch):
    _patch_call(monkeypatch, _canned({"P001_R001": "hi!"}))
    from manhua_pipeline.stages.stage4_paraphrase import run_paraphrase

    ws = tmp_path / "workspace"
    _setup(ws, [_trans_entry("P001_R001", "a")])
    run_paraphrase(str(ws), ollama_config)

    pa = _read_paraphrase(ws)
    assert pa["paraphraser_backend"] == "ollama"
    assert pa["paraphraser_model"] == "qwen2.5:3b-instruct"
    assert pa["prompt_version"] == "paraphrase-v1"
    by_id = {r["region_id"]: r for r in pa["results"]}
    assert by_id["P001_R001"]["paraphrase_source"] == "ollama:qwen2.5:3b-instruct"
