"""Offline tests for the Stage 3 Ollama translation backend (v2, hardened).

All Ollama HTTP calls are monkeypatched via OllamaBackend._call_ollama, so these
tests never touch the network and require no running `ollama serve`.

Assumes the v2 backend described in stage3_ollama_implementation_v2.md is
implemented in manhua_pipeline/stages/stage3_translation.py:
  - class OllamaBackend with _call_ollama / _parse_json / _validate_batch / _translate_batch
  - _validate_ollama_config()
  - _get_backend() returns OllamaBackend for TRANSLATOR_BACKEND == "ollama"
  - run_translation() completion gate + _write_output(advance=...) + provenance
"""

import json

import pytest

import config


# ---------------------------------------------------------------------------
# Helpers (mirrors tests/test_stage3.py)
# ---------------------------------------------------------------------------


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


@pytest.fixture
def ollama_config(monkeypatch):
    """Point the pipeline at the Ollama backend with small, test-friendly knobs."""
    monkeypatch.setattr(config, "TRANSLATOR_BACKEND", "ollama", raising=False)
    monkeypatch.setattr(config, "OLLAMA_HOST", "http://localhost:11434", raising=False)
    monkeypatch.setattr(config, "OLLAMA_TRANSLATE_MODEL", "qwen2.5:3b", raising=False)
    monkeypatch.setattr(config, "OLLAMA_BATCH_SIZE", 2, raising=False)
    monkeypatch.setattr(config, "OLLAMA_TIMEOUT", 30, raising=False)
    monkeypatch.setattr(config, "OLLAMA_TEMPERATURE", 0.2, raising=False)
    monkeypatch.setattr(config, "OLLAMA_MAX_RETRIES", 3, raising=False)
    monkeypatch.setattr(config, "OLLAMA_RETRY_BACKOFF", 0.0, raising=False)
    monkeypatch.setattr(config, "OLLAMA_MIN_COMPLETION_RATIO", 0.95, raising=False)
    monkeypatch.setattr(config, "OLLAMA_PROMPT_VERSION", "translation-v2", raising=False)
    return config


def _patch_call(monkeypatch, responder):
    """Monkeypatch OllamaBackend._call_ollama with a callable(user_prompt)->str."""
    from manhua_pipeline.stages import stage3_translation as s3

    monkeypatch.setattr(
        s3.OllamaBackend, "_call_ollama", lambda self, user_prompt: responder(user_prompt)
    )
    return s3


def _canned(mapping):
    """Return a responder that always replies with the given mapping as JSON."""
    payload = json.dumps(mapping, ensure_ascii=False)
    return lambda _prompt: payload


def _read_translation(ws):
    return json.loads(
        (ws / "stage3_translation" / "translation.json").read_text(encoding="utf-8")
    )


# ---------------------------------------------------------------------------
# 1. Backend selection
# ---------------------------------------------------------------------------


def test_get_backend_returns_ollama(ollama_config):
    from manhua_pipeline.stages.stage3_translation import OllamaBackend, _get_backend

    assert isinstance(_get_backend(ollama_config), OllamaBackend)


# ---------------------------------------------------------------------------
# 2. Successful response
# ---------------------------------------------------------------------------


def test_successful_response_written(tmp_path, ollama_config, monkeypatch):
    _patch_call(monkeypatch, _canned({"P001_R001": "Get out!"}))
    from manhua_pipeline.stages.stage3_translation import run_translation

    ws = tmp_path / "workspace"
    _setup(ws, [_ocr_entry("P001_R001", "\u6efe\u5427\uff01")])
    out = run_translation(str(ws), ollama_config)

    assert out is not None
    tr = _read_translation(ws)
    by_id = {r["region_id"]: r for r in tr["results"]}
    assert by_id["P001_R001"]["literal_translation"] == "Get out!"
    assert by_id["P001_R001"]["translated"] is True
    m = json.loads((ws / "manifest.json").read_text())
    assert m["current_stage"] == "paraphrase"


# ---------------------------------------------------------------------------
# 3. Multiple batches
# ---------------------------------------------------------------------------


def test_multiple_batches_all_regions_once(tmp_path, ollama_config, monkeypatch):
    calls = {"n": 0}

    def responder(prompt):
        calls["n"] += 1
        out = {}
        for rid in ["P001_R001", "P001_R002", "P001_R003"]:
            if rid in prompt:
                out[rid] = f"EN {rid}"
        return json.dumps(out)

    _patch_call(monkeypatch, responder)
    from manhua_pipeline.stages.stage3_translation import run_translation

    ws = tmp_path / "workspace"
    _setup(
        ws,
        [
            _ocr_entry("P001_R001", "a"),
            _ocr_entry("P001_R002", "b"),
            _ocr_entry("P001_R003", "c"),
        ],
    )
    run_translation(str(ws), ollama_config)

    tr = _read_translation(ws)
    by_id = {r["region_id"]: r for r in tr["results"]}
    assert by_id["P001_R001"]["literal_translation"] == "EN P001_R001"
    assert by_id["P001_R002"]["literal_translation"] == "EN P001_R002"
    assert by_id["P001_R003"]["literal_translation"] == "EN P001_R003"
    assert calls["n"] >= 2


# ---------------------------------------------------------------------------
# 4. Code-fenced JSON
# ---------------------------------------------------------------------------


def test_code_fenced_json_parses(tmp_path, ollama_config, monkeypatch):
    fenced = "```json\n" + json.dumps({"P001_R001": "Fenced ok"}) + "\n```"
    _patch_call(monkeypatch, lambda _p: fenced)
    from manhua_pipeline.stages.stage3_translation import run_translation

    ws = tmp_path / "workspace"
    _setup(ws, [_ocr_entry("P001_R001", "x")])
    run_translation(str(ws), ollama_config)

    tr = _read_translation(ws)
    assert tr["results"][0]["literal_translation"] == "Fenced ok"


def test_parse_json_static_helpers():
    from manhua_pipeline.stages.stage3_translation import OllamaBackend

    assert OllamaBackend._parse_json('```json\n{"a": "b"}\n```') == {"a": "b"}
    assert OllamaBackend._parse_json('prose {"a": "b"} trailing') == {"a": "b"}
    assert OllamaBackend._parse_json("not json at all") == {}
    assert OllamaBackend._parse_json("") == {}
    assert OllamaBackend._parse_json("[1, 2, 3]") == {}


# ---------------------------------------------------------------------------
# 5. Invalid JSON triggers recovery
# ---------------------------------------------------------------------------


def test_invalid_json_recovers_via_retry(tmp_path, ollama_config, monkeypatch):
    state = {"first": True}

    def responder(prompt):
        if state["first"]:
            state["first"] = False
            return "total garbage, no json here"
        out = {}
        for rid in ["P001_R001", "P001_R002"]:
            if rid in prompt:
                out[rid] = f"recovered {rid}"
        return json.dumps(out)

    _patch_call(monkeypatch, responder)
    from manhua_pipeline.stages.stage3_translation import run_translation

    ws = tmp_path / "workspace"
    _setup(ws, [_ocr_entry("P001_R001", "a"), _ocr_entry("P001_R002", "b")])
    out = run_translation(str(ws), ollama_config)

    assert out is not None
    tr = _read_translation(ws)
    by_id = {r["region_id"]: r for r in tr["results"]}
    assert by_id["P001_R001"]["translated"] is True
    assert by_id["P001_R002"]["translated"] is True


# ---------------------------------------------------------------------------
# 6. Missing region identified
# ---------------------------------------------------------------------------


def test_missing_region_flagged(tmp_path, ollama_config, monkeypatch):
    _patch_call(monkeypatch, _canned({"P001_R001": "only one"}))
    from manhua_pipeline.stages.stage3_translation import run_translation

    ws = tmp_path / "workspace"
    _setup(ws, [_ocr_entry("P001_R001", "a"), _ocr_entry("P001_R002", "b")])
    out = run_translation(str(ws), ollama_config)

    tr = _read_translation(ws)
    by_id = {r["region_id"]: r for r in tr["results"]}
    assert by_id["P001_R001"]["translated"] is True
    assert by_id["P001_R002"]["translated"] is False
    m = json.loads((ws / "manifest.json").read_text())
    assert m["current_stage"] == "translate"
    assert out is None


# ---------------------------------------------------------------------------
# 7. Unexpected / cross-batch IDs rejected
# ---------------------------------------------------------------------------


def test_unexpected_ids_rejected():
    from manhua_pipeline.stages.stage3_translation import OllamaBackend

    parsed = {"P001_R001": "good", "P999_R999": "ghost", "P002_R002": "other batch"}
    accepted, missing, unexpected = OllamaBackend._validate_batch(
        parsed, {"P001_R001"}
    )
    assert accepted == {"P001_R001": "good"}
    assert set(unexpected) == {"P999_R999", "P002_R002"}
    assert missing == []


# ---------------------------------------------------------------------------
# 8. Invalid values never enter output
# ---------------------------------------------------------------------------


def test_invalid_values_dropped():
    from manhua_pipeline.stages.stage3_translation import OllamaBackend

    parsed = {
        "P001_R001": "",
        "P001_R002": "   ",
        "P001_R003": None,
        "P001_R004": 123,
        "P001_R005": ["a"],
        "P001_R006": {"k": "v"},
        "P001_R007": "valid",
    }
    expected = {f"P001_R00{i}" for i in range(1, 8)}
    accepted, missing, _ = OllamaBackend._validate_batch(parsed, expected)
    assert accepted == {"P001_R007": "valid"}
    assert set(missing) == expected - {"P001_R007"}


def test_validate_batch_no_overwrite():
    from manhua_pipeline.stages.stage3_translation import OllamaBackend

    accepted, _, _ = OllamaBackend._validate_batch(
        {"P001_R001": "keep"}, {"P001_R001"}
    )
    assert accepted == {"P001_R001": "keep"}


# ---------------------------------------------------------------------------
# 9. Connectivity failure -> clear RuntimeError
# ---------------------------------------------------------------------------


def test_connectivity_failure_raises_clear(tmp_path, ollama_config, monkeypatch):
    import urllib.error
    from manhua_pipeline.stages import stage3_translation as s3

    def boom(req, timeout=None):
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr(s3.urllib.request, "urlopen", boom)
    from manhua_pipeline.stages.stage3_translation import run_translation

    ws = tmp_path / "workspace"
    _setup(ws, [_ocr_entry("P001_R001", "a")])

    with pytest.raises(RuntimeError) as exc:
        run_translation(str(ws), ollama_config)
    msg = str(exc.value)
    assert "localhost:11434" in msg
    assert "qwen2.5:3b" in msg
    assert "ollama serve" in msg


# ---------------------------------------------------------------------------
# 10. Empty input -> no Ollama call
# ---------------------------------------------------------------------------


def test_empty_input_no_call(tmp_path, ollama_config, monkeypatch):
    called = {"n": 0}

    def responder(_p):
        called["n"] += 1
        return "{}"

    _patch_call(monkeypatch, responder)
    from manhua_pipeline.stages.stage3_translation import run_translation

    ws = tmp_path / "workspace"
    _setup(ws, [_ocr_entry("P001_R001", "", usable=False)])
    out = run_translation(str(ws), ollama_config)

    assert out is not None
    assert called["n"] == 0
    m = json.loads((ws / "manifest.json").read_text())
    assert m["current_stage"] == "paraphrase"


# ---------------------------------------------------------------------------
# 11. Completion threshold gates the manifest
# ---------------------------------------------------------------------------


def test_below_threshold_does_not_advance(tmp_path, ollama_config, monkeypatch):
    monkeypatch.setattr(config, "OLLAMA_BATCH_SIZE", 10, raising=False)
    _patch_call(monkeypatch, _canned({"P001_R001": "one"}))
    from manhua_pipeline.stages.stage3_translation import run_translation

    ws = tmp_path / "workspace"
    _setup(
        ws,
        [
            _ocr_entry("P001_R001", "a"),
            _ocr_entry("P001_R002", "b"),
            _ocr_entry("P001_R003", "c"),
        ],
    )
    out = run_translation(str(ws), ollama_config)

    assert out is None
    m = json.loads((ws / "manifest.json").read_text())
    assert m["current_stage"] == "translate"
    assert (ws / "stage3_translation" / "translation.json").exists()


def test_zero_translations_raises(tmp_path, ollama_config, monkeypatch):
    _patch_call(monkeypatch, _canned({}))
    from manhua_pipeline.stages.stage3_translation import run_translation

    ws = tmp_path / "workspace"
    _setup(ws, [_ocr_entry("P001_R001", "a")])
    with pytest.raises(ValueError):
        run_translation(str(ws), ollama_config)


# ---------------------------------------------------------------------------
# 12. Config validation
# ---------------------------------------------------------------------------


def test_config_validation_rejects_bad_values(monkeypatch):
    from manhua_pipeline.stages.stage3_translation import _validate_ollama_config

    monkeypatch.setattr(config, "OLLAMA_HOST", "http://localhost:11434", raising=False)
    monkeypatch.setattr(config, "OLLAMA_TRANSLATE_MODEL", "qwen2.5:3b", raising=False)
    monkeypatch.setattr(config, "OLLAMA_TIMEOUT", 30, raising=False)
    monkeypatch.setattr(config, "OLLAMA_TEMPERATURE", 0.2, raising=False)
    monkeypatch.setattr(config, "OLLAMA_MAX_RETRIES", 3, raising=False)

    monkeypatch.setattr(config, "OLLAMA_BATCH_SIZE", 0, raising=False)
    with pytest.raises(ValueError):
        _validate_ollama_config(config)

    monkeypatch.setattr(config, "OLLAMA_BATCH_SIZE", 2, raising=False)
    monkeypatch.setattr(config, "OLLAMA_HOST", "", raising=False)
    with pytest.raises(ValueError):
        _validate_ollama_config(config)

    monkeypatch.setattr(config, "OLLAMA_HOST", "ftp://bad", raising=False)
    with pytest.raises(ValueError):
        _validate_ollama_config(config)

    monkeypatch.setattr(config, "OLLAMA_HOST", "http://localhost:11434", raising=False)
    monkeypatch.setattr(config, "OLLAMA_TRANSLATE_MODEL", "", raising=False)
    with pytest.raises(ValueError):
        _validate_ollama_config(config)


def test_config_validation_accepts_good_values(monkeypatch):
    from manhua_pipeline.stages.stage3_translation import _validate_ollama_config

    monkeypatch.setattr(config, "OLLAMA_HOST", "http://localhost:11434", raising=False)
    monkeypatch.setattr(config, "OLLAMA_TRANSLATE_MODEL", "qwen2.5:3b", raising=False)
    monkeypatch.setattr(config, "OLLAMA_BATCH_SIZE", 15, raising=False)
    monkeypatch.setattr(config, "OLLAMA_TIMEOUT", 120, raising=False)
    monkeypatch.setattr(config, "OLLAMA_TEMPERATURE", 0.2, raising=False)
    monkeypatch.setattr(config, "OLLAMA_MAX_RETRIES", 3, raising=False)
    _validate_ollama_config(config)


# ---------------------------------------------------------------------------
# Bonus: provenance metadata recorded
# ---------------------------------------------------------------------------


def test_provenance_metadata_recorded(tmp_path, ollama_config, monkeypatch):
    _patch_call(monkeypatch, _canned({"P001_R001": "hi"}))
    from manhua_pipeline.stages.stage3_translation import run_translation

    ws = tmp_path / "workspace"
    _setup(ws, [_ocr_entry("P001_R001", "a")])
    run_translation(str(ws), ollama_config)

    tr = _read_translation(ws)
    assert tr["translator_backend"] == "ollama"
    assert tr["translator_model"] == "qwen2.5:3b"
    assert tr["prompt_version"] == "translation-v2"
    by_id = {r["region_id"]: r for r in tr["results"]}
    assert by_id["P001_R001"]["translation_source"] == "ollama:qwen2.5:3b"
