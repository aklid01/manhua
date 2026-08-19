# Unit tests for Ollama backend Stage 4 dialogue paraphrasing.

import json
import urllib.error

import pytest

import config
from manhua_pipeline.stages import stage4_paraphrase
from manhua_pipeline.stages._backends import (
    OllamaSettings,
    validate_ollama_settings,
)
from manhua_pipeline.stages.stage4_paraphrase import (
    OllamaBackend,
    _get_backend,
    run_paraphrase,
)


# Helper to set up workspace with manifest and translation results
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


# Helper to construct mock translation region result dictionary
def _trans_entry(rid, literal, translated=True):
    return {
        "region_id": rid,
        "page_number": 1,
        "original_text": "原文",
        "literal_translation": literal,
        "translated": translated,
        "skip_reason": None if translated else "no_usable_text",
        "glossary_terms_applied": [],
        "glossary_conflict": False,
        "translation_source": "llm" if translated else None,
    }


# Fixture configuring Ollama paraphrase settings
@pytest.fixture
def ollama_config(monkeypatch):
    monkeypatch.setattr(config, "PARAPHRASE_BACKEND", "ollama", raising=False)
    monkeypatch.setattr(
        config, "OLLAMA_PARA_HOST", "http://localhost:11434", raising=False
    )
    monkeypatch.setattr(
        config, "OLLAMA_PARA_MODEL", "qwen2.5:3b-instruct", raising=False
    )
    monkeypatch.setattr(config, "OLLAMA_PARA_BATCH_SIZE", 2, raising=False)
    monkeypatch.setattr(config, "OLLAMA_PARA_TIMEOUT", 30, raising=False)
    monkeypatch.setattr(config, "OLLAMA_PARA_TEMPERATURE", 0.7, raising=False)
    monkeypatch.setattr(config, "OLLAMA_PARA_MAX_RETRIES", 3, raising=False)
    monkeypatch.setattr(config, "OLLAMA_PARA_RETRY_BACKOFF", 0.0, raising=False)
    monkeypatch.setattr(config, "OLLAMA_PARA_MIN_COMPLETION_RATIO", 0.80, raising=False)
    monkeypatch.setattr(
        config, "OLLAMA_PARA_PROMPT_VERSION", "paraphrase-v1", raising=False
    )
    return config


# Helper to mock OllamaBackend._call_ollama with custom responder
def _patch_call(monkeypatch, responder):
    monkeypatch.setattr(
        stage4_paraphrase.OllamaBackend,
        "_call_ollama",
        lambda self, user_prompt: responder(user_prompt),
    )
    return stage4_paraphrase


# Helper returning canned JSON dictionary responder
def _canned(mapping):
    payload = json.dumps(mapping, ensure_ascii=False)
    return lambda _prompt: payload


# Helper to read and deserialize paraphrase.json
def _read_paraphrase(ws):
    return json.loads(
        (ws / "stage4_paraphrase" / "paraphrase.json").read_text(encoding="utf-8")
    )


# Verify _get_backend returns OllamaBackend instance for paraphrase stage
def test_get_backend_returns_ollama(ollama_config):
    assert isinstance(_get_backend(ollama_config), OllamaBackend)


# Verify valid paraphrase response writes output and advances manifest
def test_successful_paraphrase_written(tmp_path, ollama_config, monkeypatch):
    _patch_call(monkeypatch, _canned({"P001_R001": "Get lost!"}))

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


# Verify multi-batch paraphrasing processes all regions
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

    ws = tmp_path / "workspace"
    _setup(
        ws,
        [
            _trans_entry("P001_R001", "a a a"),
            _trans_entry("P001_R002", "b b b"),
            _trans_entry("P001_R003", "c c c"),
        ],
    )
    run_paraphrase(str(ws), ollama_config)

    pa = _read_paraphrase(ws)
    by_id = {r["region_id"]: r for r in pa["results"]}
    assert by_id["P001_R001"]["final_text"] == "punchy P001_R001"
    assert by_id["P001_R002"]["final_text"] == "punchy P001_R002"
    assert by_id["P001_R003"]["final_text"] == "punchy P001_R003"
    assert calls["n"] >= 2


# Verify code-fenced JSON responses parse correctly
def test_code_fenced_json_parses(tmp_path, ollama_config, monkeypatch):
    fenced = "```json\n" + json.dumps({"P001_R001": "Fenced!"}) + "\n```"
    _patch_call(monkeypatch, lambda _p: fenced)

    ws = tmp_path / "workspace"
    _setup(ws, [_trans_entry("P001_R001", "x")])
    run_paraphrase(str(ws), ollama_config)

    pa = _read_paraphrase(ws)
    assert pa["results"][0]["final_text"] == "Fenced!"


# Verify static JSON parsing handles valid, markdown-fenced, and invalid inputs
def test_parse_json_static_helpers():
    assert OllamaBackend._parse_json('```json\n{"a": "b"}\n```') == {"a": "b"}
    assert OllamaBackend._parse_json('prose {"a": "b"} tail') == {"a": "b"}
    assert OllamaBackend._parse_json("garbage") == {}
    assert OllamaBackend._parse_json("") == {}
    assert OllamaBackend._parse_json("[1,2,3]") == {}


# Verify invalid JSON responses trigger retry and recover
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

    ws = tmp_path / "workspace"
    _setup(ws, [_trans_entry("P001_R001", "a"), _trans_entry("P001_R002", "b")])
    out = run_paraphrase(str(ws), ollama_config)

    assert out is not None
    pa = _read_paraphrase(ws)
    by_id = {r["region_id"]: r for r in pa["results"]}
    assert by_id["P001_R001"]["paraphrased"] is True
    assert by_id["P001_R002"]["paraphrased"] is True


# Verify missing paraphrase regions fall back to literal translation
def test_missing_falls_back_to_literal(tmp_path, ollama_config, monkeypatch):
    _patch_call(monkeypatch, _canned({"P001_R001": "Nice one!"}))

    ws = tmp_path / "workspace"
    _setup(
        ws,
        [_trans_entry("P001_R001", "a"), _trans_entry("P001_R002", "keep me literal")],
    )
    run_paraphrase(str(ws), ollama_config)

    pa = _read_paraphrase(ws)
    by_id = {r["region_id"]: r for r in pa["results"]}
    assert by_id["P001_R001"]["paraphrased"] is True
    assert by_id["P001_R002"]["paraphrased"] is False
    assert by_id["P001_R002"]["final_text"] == "keep me literal"
    assert by_id["P001_R002"]["paraphrase_source"] == "literal_fallback"


# Verify verbatim repetitions of literal translation are accepted as valid paraphrases
def test_echo_of_literal_allowed():
    literals = {"P001_R001": "ACME Corp", "P001_R002": "hello there"}
    parsed = {"P001_R001": "ACME Corp", "P001_R002": "hey!"}
    accepted, missing, _ = OllamaBackend._validate_batch(
        parsed, {"P001_R001", "P001_R002"}, literals
    )
    assert accepted == {"P001_R001": "ACME Corp", "P001_R002": "hey!"}
    assert missing == []


# Verify extraneous region IDs not in the batch are rejected
def test_unexpected_ids_rejected():
    parsed = {"P001_R001": "good", "P999_R999": "ghost"}
    accepted, missing, unexpected = OllamaBackend._validate_batch(
        parsed, {"P001_R001"}, {"P001_R001": "g"}
    )
    assert accepted == {"P001_R001": "good"}
    assert unexpected == ["P999_R999"]
    assert missing == []


# Verify null, numeric, list, or empty values are dropped
def test_invalid_values_dropped():
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


# Verify Ollama connection failures raise descriptive RuntimeError
def test_connectivity_failure_raises(tmp_path, ollama_config, monkeypatch):
    def boom(req, timeout=None):
        raise urllib.error.URLError("refused")

    monkeypatch.setattr("urllib.request.urlopen", boom)

    ws = tmp_path / "workspace"
    _setup(ws, [_trans_entry("P001_R001", "a")])
    with pytest.raises(RuntimeError) as exc:
        run_paraphrase(str(ws), ollama_config)
    msg = str(exc.value)
    assert "localhost:11434" in msg
    assert "qwen2.5:3b-instruct" in msg
    assert "ollama serve" in msg


# Verify chapters with zero usable regions bypass backend API calls
def test_empty_input_no_call(tmp_path, ollama_config, monkeypatch):
    called = {"n": 0}

    def responder(_p):
        called["n"] += 1
        return "{}"

    _patch_call(monkeypatch, responder)

    ws = tmp_path / "workspace"
    _setup(ws, [_trans_entry("P001_R001", "", translated=False)])
    out = run_paraphrase(str(ws), ollama_config)

    assert out is not None
    assert called["n"] == 0
    m = json.loads((ws / "manifest.json").read_text())
    assert m["current_stage"] == "render"


# Verify completion ratio below threshold writes output without advancing stage
def test_below_threshold_does_not_advance(tmp_path, ollama_config, monkeypatch):
    monkeypatch.setattr(config, "OLLAMA_PARA_BATCH_SIZE", 10, raising=False)
    _patch_call(monkeypatch, _canned({"P001_R001": "one"}))

    ws = tmp_path / "workspace"
    _setup(
        ws,
        [
            _trans_entry("P001_R001", "a"),
            _trans_entry("P001_R002", "b"),
            _trans_entry("P001_R003", "c"),
        ],
    )
    out = run_paraphrase(str(ws), ollama_config)

    assert out is None
    m = json.loads((ws / "manifest.json").read_text())
    assert m["current_stage"] == "paraphrase"
    assert (ws / "stage4_paraphrase" / "paraphrase.json").exists()


# Verify invalid Ollama host, model, or batch size configurations raise ValueError
def test_config_validation(monkeypatch):
    def _validate_ollama_para_config(cfg):
        s = OllamaSettings.from_config(cfg, prefix="OLLAMA_PARA")
        validate_ollama_settings(s, "OLLAMA_PARA")

    monkeypatch.setattr(
        config, "OLLAMA_PARA_HOST", "http://localhost:11434", raising=False
    )
    monkeypatch.setattr(
        config, "OLLAMA_PARA_MODEL", "qwen2.5:3b-instruct", raising=False
    )
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

    monkeypatch.setattr(
        config, "OLLAMA_PARA_HOST", "http://localhost:11434", raising=False
    )
    monkeypatch.setattr(config, "OLLAMA_PARA_MODEL", "", raising=False)
    with pytest.raises(ValueError):
        _validate_ollama_para_config(config)


# Verify paraphrase output metadata records backend model and prompt version
def test_provenance_recorded(tmp_path, ollama_config, monkeypatch):
    _patch_call(monkeypatch, _canned({"P001_R001": "hi!"}))

    ws = tmp_path / "workspace"
    _setup(ws, [_trans_entry("P001_R001", "a")])
    run_paraphrase(str(ws), ollama_config)

    pa = _read_paraphrase(ws)
    assert pa["paraphraser_backend"] == "ollama"
    assert pa["paraphraser_model"] == "qwen2.5:3b-instruct"
    assert pa["prompt_version"] == "paraphrase-v1"
    by_id = {r["region_id"]: r for r in pa["results"]}
    assert by_id["P001_R001"]["paraphrase_source"] == "ollama:qwen2.5:3b-instruct"
