# Unit tests for Ollama backend Stage 3 translation execution and error handling.

import json
import urllib.error

import pytest

import config
from manhua_pipeline.stages import stage3_translation
from manhua_pipeline.stages._backends import (
    OllamaSettings,
    validate_ollama_settings,
)
from manhua_pipeline.stages.stage3_translation import (
    OllamaBackend,
    _contains_cjk,
    _get_backend,
    run_translation,
)


# Verify contextual retry is disabled by default
def test_contextual_retry_disabled_by_default():
    assert config.OLLAMA_CONTEXTUAL_RETRY_ENABLED is False


# Helper to set up workspace with manifest and OCR results
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


# Fixture configuring Ollama translator settings
@pytest.fixture
def ollama_config(monkeypatch):
    monkeypatch.setattr(config, "TRANSLATOR_BACKEND", "ollama", raising=False)
    monkeypatch.setattr(config, "OLLAMA_HOST", "http://localhost:11434", raising=False)
    monkeypatch.setattr(config, "OLLAMA_TRANSLATE_MODEL", "qwen2.5:3b", raising=False)
    monkeypatch.setattr(config, "OLLAMA_BATCH_SIZE", 2, raising=False)
    monkeypatch.setattr(config, "OLLAMA_TIMEOUT", 30, raising=False)
    monkeypatch.setattr(config, "OLLAMA_TEMPERATURE", 0.2, raising=False)
    monkeypatch.setattr(config, "OLLAMA_MAX_RETRIES", 3, raising=False)
    monkeypatch.setattr(config, "OLLAMA_RETRY_BACKOFF", 0.0, raising=False)
    monkeypatch.setattr(config, "OLLAMA_MIN_COMPLETION_RATIO", 0.95, raising=False)
    monkeypatch.setattr(
        config, "OLLAMA_PROMPT_VERSION", "translation-v2", raising=False
    )
    return config


# Helper to mock OllamaBackend._call_ollama with custom responder
def _patch_call(monkeypatch, responder):
    monkeypatch.setattr(
        stage3_translation.OllamaBackend,
        "_call_ollama",
        lambda self, user_prompt: responder(user_prompt),
    )
    return stage3_translation


# Helper returning canned JSON dictionary responder
def _canned(mapping):
    payload = json.dumps(mapping, ensure_ascii=False)
    return lambda _prompt: payload


# Helper to read and deserialize translation.json
def _read_translation(ws):
    return json.loads(
        (ws / "stage3_translation" / "translation.json").read_text(encoding="utf-8")
    )


# Verify _get_backend returns OllamaBackend instance when configured
def test_get_backend_returns_ollama(ollama_config):
    assert isinstance(_get_backend(ollama_config), OllamaBackend)


# Verify valid translation response writes output and advances manifest
def test_successful_response_written(tmp_path, ollama_config, monkeypatch):
    _patch_call(monkeypatch, _canned({"P001_R001": "Get out!"}))

    ws = tmp_path / "workspace"
    _setup(ws, [_ocr_entry("P001_R001", "滚吧！")])
    out = run_translation(str(ws), ollama_config)

    assert out is not None
    tr = _read_translation(ws)
    by_id = {r["region_id"]: r for r in tr["results"]}
    assert by_id["P001_R001"]["literal_translation"] == "Get out!"
    assert by_id["P001_R001"]["translated"] is True
    m = json.loads((ws / "manifest.json").read_text())
    assert m["current_stage"] == "paraphrase"


# Verify multi-batch translations query all regions across chunks
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

    ws = tmp_path / "workspace"
    _setup(
        ws,
        [
            _ocr_entry("P001_R001", "你好呀"),
            _ocr_entry("P001_R002", "早上好"),
            _ocr_entry("P001_R003", "再见了"),
        ],
    )
    run_translation(str(ws), ollama_config)

    tr = _read_translation(ws)
    by_id = {r["region_id"]: r for r in tr["results"]}
    assert by_id["P001_R001"]["literal_translation"] == "EN P001_R001"
    assert by_id["P001_R002"]["literal_translation"] == "EN P001_R002"
    assert by_id["P001_R003"]["literal_translation"] == "EN P001_R003"
    assert calls["n"] >= 2


# Verify code-fenced JSON responses are parsed successfully
def test_code_fenced_json_parses(tmp_path, ollama_config, monkeypatch):
    fenced = "```json\n" + json.dumps({"P001_R001": "Fenced ok"}) + "\n```"
    _patch_call(monkeypatch, lambda _p: fenced)

    ws = tmp_path / "workspace"
    _setup(ws, [_ocr_entry("P001_R001", "你好呀")])
    run_translation(str(ws), ollama_config)

    tr = _read_translation(ws)
    assert tr["results"][0]["literal_translation"] == "Fenced ok"


# Verify static JSON parser handles clean, embedded, empty, and invalid strings
def test_parse_json_static_helpers():
    assert OllamaBackend._parse_json('```json\n{"a": "b"}\n```') == {"a": "b"}
    assert OllamaBackend._parse_json('prose {"a": "b"} trailing') == {"a": "b"}
    assert OllamaBackend._parse_json("not json at all") == {}
    assert OllamaBackend._parse_json("") == {}
    assert OllamaBackend._parse_json("[1, 2, 3]") == {}


# Verify invalid JSON responses trigger retry and recover
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

    ws = tmp_path / "workspace"
    _setup(ws, [_ocr_entry("P001_R001", "你好呀"), _ocr_entry("P001_R002", "早上好")])
    out = run_translation(str(ws), ollama_config)

    assert out is not None
    tr = _read_translation(ws)
    by_id = {r["region_id"]: r for r in tr["results"]}
    assert by_id["P001_R001"]["translated"] is True
    assert by_id["P001_R002"]["translated"] is True


# Verify missing regions in response are flagged and recovered
def test_missing_region_flagged(tmp_path, ollama_config, monkeypatch):
    _patch_call(monkeypatch, _canned({"P001_R001": "only one"}))

    ws = tmp_path / "workspace"
    _setup(ws, [_ocr_entry("P001_R001", "你好啊"), _ocr_entry("P001_R002", "我知道")])
    out = run_translation(str(ws), ollama_config)

    tr = _read_translation(ws)
    by_id = {r["region_id"]: r for r in tr["results"]}
    assert by_id["P001_R001"]["translated"] is True
    assert by_id["P001_R002"]["translated"] is False
    m = json.loads((ws / "manifest.json").read_text())
    assert m["current_stage"] == "paraphrase"
    assert by_id["P001_R002"]["translation_source"] == "translation_recovery"
    assert out is not None


# Verify contextual single-region retries are skipped when disabled
def test_contextual_retry_is_skipped_by_default(tmp_path, ollama_config, monkeypatch):
    calls = []

    def responder(prompt):
        calls.append(prompt)
        return json.dumps({"P001_R001": "only one"})

    _patch_call(monkeypatch, responder)

    ws = tmp_path / "workspace"
    _setup(ws, [_ocr_entry("P001_R001", "你好啊"), _ocr_entry("P001_R002", "我知道")])
    run_translation(str(ws), ollama_config)

    assert not any("Translate the supplied Chinese region" in call for call in calls)
    by_id = {r["region_id"]: r for r in _read_translation(ws)["results"]}
    assert by_id["P001_R002"]["translation_source"] == "translation_recovery"


# Verify contextual retry runs when explicitly enabled
def test_contextual_retry_remains_available_when_enabled(
    tmp_path, ollama_config, monkeypatch
):
    calls = []

    def responder(prompt):
        calls.append(prompt)
        if "Translate the supplied Chinese region" in prompt:
            return json.dumps({"P001_R002": "context fallback"})
        return json.dumps({"P001_R001": "only one"})

    monkeypatch.setattr(config, "OLLAMA_CONTEXTUAL_RETRY_ENABLED", True)
    _patch_call(monkeypatch, responder)

    ws = tmp_path / "workspace"
    _setup(ws, [_ocr_entry("P001_R001", "你好啊"), _ocr_entry("P001_R002", "我知道")])
    run_translation(str(ws), ollama_config)

    assert any("Translate the supplied Chinese region" in call for call in calls)
    by_id = {r["region_id"]: r for r in _read_translation(ws)["results"]}
    assert by_id["P001_R002"]["translated"] is True
    assert by_id["P001_R002"]["literal_translation"] == "context fallback"


# Verify extraneous region IDs not in the requested batch are rejected
def test_unexpected_ids_rejected():
    parsed = {"P001_R001": "good", "P999_R999": "ghost", "P002_R002": "other batch"}
    accepted, missing, unexpected = OllamaBackend._validate_batch(parsed, {"P001_R001"})
    assert accepted == {"P001_R001": "good"}
    assert set(unexpected) == {"P999_R999", "P002_R002"}
    assert missing == []


# Verify null, numeric, list, or empty string values are filtered out
def test_invalid_values_dropped():
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


# Verify valid batch translations are preserved
def test_validate_batch_no_overwrite():
    accepted, _, _ = OllamaBackend._validate_batch({"P001_R001": "keep"}, {"P001_R001"})
    assert accepted == {"P001_R001": "keep"}


# Verify Ollama connection failures raise descriptive RuntimeError
def test_connectivity_failure_raises_clear(tmp_path, ollama_config, monkeypatch):
    def boom(req, timeout=None):
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr("urllib.request.urlopen", boom)

    ws = tmp_path / "workspace"
    _setup(ws, [_ocr_entry("P001_R001", "你好呀")])

    with pytest.raises(RuntimeError) as exc:
        run_translation(str(ws), ollama_config)
    msg = str(exc.value)
    assert "localhost:11434" in msg
    assert "qwen2.5:3b" in msg
    assert "ollama serve" in msg


# Verify chapters with zero usable regions bypass backend API calls
def test_empty_input_no_call(tmp_path, ollama_config, monkeypatch):
    called = {"n": 0}

    def responder(_p):
        called["n"] += 1
        return "{}"

    _patch_call(monkeypatch, responder)

    ws = tmp_path / "workspace"
    _setup(ws, [_ocr_entry("P001_R001", "", usable=False)])
    out = run_translation(str(ws), ollama_config)

    assert out is not None
    assert called["n"] == 0
    m = json.loads((ws / "manifest.json").read_text())
    assert m["current_stage"] == "paraphrase"


# Verify incomplete batches fall back to recovery and advance to paraphrase
def test_below_threshold_does_not_advance(tmp_path, ollama_config, monkeypatch):
    monkeypatch.setattr(config, "OLLAMA_BATCH_SIZE", 10, raising=False)
    _patch_call(monkeypatch, _canned({"P001_R001": "one"}))

    ws = tmp_path / "workspace"
    _setup(
        ws,
        [
            _ocr_entry("P001_R001", "你好啊"),
            _ocr_entry("P001_R002", "我知道"),
            _ocr_entry("P001_R003", "没问题"),
        ],
    )
    out = run_translation(str(ws), ollama_config)

    assert out is not None
    m = json.loads((ws / "manifest.json").read_text())
    assert m["current_stage"] == "paraphrase"
    tr = _read_translation(ws)
    by_id = {r["region_id"]: r for r in tr["results"]}
    assert by_id["P001_R002"]["translation_source"] == "translation_recovery"
    assert by_id["P001_R003"]["translation_source"] == "translation_recovery"
    assert (ws / "stage3_translation" / "translation.json").exists()


# Verify total translation failure hands off to stage 4 recovery
def test_zero_translations_hands_off_to_stage4(tmp_path, ollama_config, monkeypatch):
    _patch_call(monkeypatch, _canned({}))

    ws = tmp_path / "workspace"
    _setup(ws, [_ocr_entry("P001_R001", "你好呀")])
    out = run_translation(str(ws), ollama_config)

    assert out is not None
    translation = json.loads(
        (ws / "stage3_translation" / "translation.json").read_text(encoding="utf-8")
    )
    row = translation["results"][0]
    assert row["translated"] is False
    assert row["translation_source"] == "translation_recovery"
    manifest = json.loads((ws / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["current_stage"] == "paraphrase"


# Helper validating Ollama configuration settings
def _validate_ollama_config(cfg):
    s = OllamaSettings.from_config(cfg, prefix="OLLAMA")
    validate_ollama_settings(s, "OLLAMA")


# Verify invalid Ollama host, model, or batch size configurations raise ValueError
def test_config_validation_rejects_bad_values(monkeypatch):
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


# Verify valid Ollama settings pass validation
def test_config_validation_accepts_good_values(monkeypatch):
    monkeypatch.setattr(config, "OLLAMA_HOST", "http://localhost:11434", raising=False)
    monkeypatch.setattr(config, "OLLAMA_TRANSLATE_MODEL", "qwen2.5:3b", raising=False)
    monkeypatch.setattr(config, "OLLAMA_BATCH_SIZE", 15, raising=False)
    monkeypatch.setattr(config, "OLLAMA_TIMEOUT", 120, raising=False)
    monkeypatch.setattr(config, "OLLAMA_TEMPERATURE", 0.2, raising=False)
    monkeypatch.setattr(config, "OLLAMA_MAX_RETRIES", 3, raising=False)
    _validate_ollama_config(config)


# Verify translation output metadata records backend model and prompt version
def test_provenance_metadata_recorded(tmp_path, ollama_config, monkeypatch):
    _patch_call(monkeypatch, _canned({"P001_R001": "hi"}))

    ws = tmp_path / "workspace"
    _setup(ws, [_ocr_entry("P001_R001", "你好呀")])
    run_translation(str(ws), ollama_config)

    tr = _read_translation(ws)
    assert tr["translator_backend"] == "ollama"
    assert tr["translator_model"] == "qwen2.5:3b"
    assert tr["prompt_version"] == "translation-v2"
    by_id = {r["region_id"]: r for r in tr["results"]}
    assert by_id["P001_R001"]["translation_source"] == "ollama:qwen2.5:3b"


# Verify responses containing untranslated CJK characters are rejected
def test_residual_cjk_rejected_as_missing():
    parsed = {
        "P001_R001": "滚吧！",
        "P001_R002": "Get out!",
        "P001_R003": "Get out 滚吧了",
    }
    accepted, missing, _ = OllamaBackend._validate_batch(
        parsed, {"P001_R001", "P001_R002", "P001_R003"}
    )
    assert accepted == {"P001_R002": "Get out!"}
    assert set(missing) == {"P001_R001", "P001_R003"}


# Verify CJK detection helper
def test_contains_cjk_helper():
    assert _contains_cjk("滚吧")
    assert _contains_cjk("hello 你好")
    assert not _contains_cjk("hello!")
    assert not _contains_cjk("")
    assert not _contains_cjk(None)


# Verify missing trivial single-character regions bypass hold and advance
def test_trivial_missing_region_bypasses_hold_and_skips_glossary(
    tmp_path, ollama_config, monkeypatch
):
    ws = tmp_path / "workspace"
    _setup(
        ws,
        [
            _ocr_entry("P001_R001", "一", usable=True),
            _ocr_entry("P001_R002", "滚吧！", usable=True),
        ],
    )

    monkeypatch.setattr(
        stage3_translation.OllamaBackend,
        "_call_ollama",
        lambda self, prompt: json.dumps({"P001_R002": "Get out!"}),
    )

    result = stage3_translation.run_translation(str(ws), ollama_config)

    assert result is not None
    m = json.loads((ws / "manifest.json").read_text())
    assert m["current_stage"] == "paraphrase"

    glossary_path = ws.parent / "glossary.json"
    if glossary_path.exists():
        g_data = json.loads(glossary_path.read_text(encoding="utf-8"))
        terms = [t.get("source_term") for t in g_data.get("terms", [])]
        assert "一" not in terms
