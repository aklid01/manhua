# E2E regression: Stage 3 openai_compat failure handling.

import json

import config
from manhua_pipeline.stages import _openai_compat as oc


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


def _setup_translate(ws, ocr_results):
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
    (ws / "stage2_ocr" / "ocr.json").write_text(
        json.dumps(
            {
                "chapter_id": "t",
                "stage": "ocr",
                "generated_at": "now",
                "ocr_engine": "PaddleOCR",
                "results": ocr_results,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def _two_provider_chain(purpose, cfg):
    return (
        {
            "enabled": True,
            "max_retries_per_model": 2,
            "batch_size": 10,
            "concurrency": 1,
            "timeout": 30.0,
        },
        [
            {
                "name": "primary",
                "base_url": "https://primary.example.com/v1",
                "model": "primary-model",
                "api_key": "k1",
            },
            {
                "name": "secondary",
                "base_url": "https://secondary.example.com/v1",
                "model": "secondary-model",
                "api_key": "k2",
            },
        ],
    )


def _patch_chain(monkeypatch):
    monkeypatch.setattr(config, "TRANSLATOR_BACKEND", "openai_compat", raising=False)
    monkeypatch.setattr(config, "OPENAI_COMPAT_PROBE_ENABLED", False, raising=False)
    monkeypatch.setattr(
        config, "OPENAI_COMPAT_TRANSLATE_TEMPERATURE", 0.2, raising=False
    )
    monkeypatch.setattr(oc, "load_provider_chain", _two_provider_chain)
    monkeypatch.setattr(
        oc, "probe_and_rank_providers", lambda providers, timeout, lg=None: providers
    )


def _read_translation(ws):
    return json.loads(
        (ws / "stage3_translation" / "translation.json").read_text(encoding="utf-8")
    )


def test_transport_failure_fails_over_to_secondary(tmp_path, monkeypatch):
    _patch_chain(monkeypatch)
    from manhua_pipeline.stages.stage3_translation.engine import run_translation

    calls = []

    def fake_chat(base_url, headers, model, *a, **k):
        calls.append(model)
        if model == "primary-model":
            raise oc.ProviderError("timeout")
        return json.dumps({"P001_R001": "Get out!"})

    monkeypatch.setattr(oc, "call_openai_chat", fake_chat)
    monkeypatch.setattr(
        "manhua_pipeline.stages._openai_compat.call_openai_chat", fake_chat
    )

    ws = tmp_path / "ch-transport"
    _setup_translate(ws, [_ocr_entry("P001_R001", "滚吧！")])
    out = run_translation(str(ws), config)

    assert out is not None

def test_validation_gap_does_not_touch_secondary(tmp_path, monkeypatch):
    _patch_chain(monkeypatch)
    from manhua_pipeline.stages.stage3_translation.engine import run_translation

    calls = []

    def fake_chat(base_url, headers, model, *a, **k):
        calls.append(model)
        return json.dumps({})

    monkeypatch.setattr(oc, "call_openai_chat", fake_chat)
    monkeypatch.setattr(
        "manhua_pipeline.stages._openai_compat.call_openai_chat", fake_chat
    )

    ws = tmp_path / "ch-validation"
    _setup_translate(ws, [_ocr_entry("P001_R001", "滚吧！")])
    out = run_translation(str(ws), config)

    assert out is not None
    assert "secondary-model" not in calls
    by_id = {r["region_id"]: r for r in _read_translation(ws)["results"]}
    assert by_id["P001_R001"]["translation_source"] == "translation_recovery"


def test_partial_validation_keeps_accepted_without_secondary(tmp_path, monkeypatch):
    _patch_chain(monkeypatch)
    from manhua_pipeline.stages.stage3_translation.engine import run_translation

    calls = []

    def fake_chat(base_url, headers, model, *a, **k):
        calls.append(model)
        return json.dumps({"P001_R001": "Hello!"})

    monkeypatch.setattr(oc, "call_openai_chat", fake_chat)
    monkeypatch.setattr(
        "manhua_pipeline.stages._openai_compat.call_openai_chat", fake_chat
    )

    ws = tmp_path / "ch-partial"
    _setup_translate(
        ws,
        [_ocr_entry("P001_R001", "你好啊！"), _ocr_entry("P001_R002", "我知道了。")],
    )
    monkeypatch.setattr(config, "OLLAMA_MAX_MISSING", 5, raising=False)
    out = run_translation(str(ws), config)

    assert out is not None
    assert "secondary-model" not in calls
    by_id = {r["region_id"]: r for r in _read_translation(ws)["results"]}
    assert by_id["P001_R001"]["translation_source"] == "openai_compat"
    assert by_id["P001_R002"]["translation_source"] == "translation_recovery"
