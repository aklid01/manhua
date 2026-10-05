# Unit tests for OpenAI-compatible translation and paraphrase backends.

import json

import config
from manhua_pipeline.stages import _openai_compat as oc
from manhua_pipeline.stages import stage4_paraphrase as p_stage
from manhua_pipeline.stages.stage3_translation import backends as t_be


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


def _setup_paraphrase(ws, trans_results):
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
    (ws / "stage3_translation" / "translation.json").write_text(
        json.dumps(
            {
                "chapter_id": "t",
                "stage": "translation",
                "generated_at": "now",
                "translator_backend": "openai_compat",
                "glossary_version": "v1",
                "results": trans_results,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


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


def test_error_classification():
    fatal = oc._build_http_error(401, {}, '{"error": {"message": "invalid api key"}}')
    assert isinstance(fatal, oc.FatalProviderError)
    retriable = oc._build_http_error(429, {}, "rate limited")
    assert isinstance(retriable, oc.ProviderError)
    assert not isinstance(retriable, oc.FatalProviderError)
    no_channel = oc._build_http_error(
        500, {}, '{"error": "no available channel for model"}'
    )
    assert isinstance(no_channel, oc.FatalProviderError)


def test_parse_and_sanitize():
    assert oc.contains_cjk("hello 你好")
    assert not oc.contains_cjk("hello!")
    parsed = oc.parse_llm_json('```json\n{"P001_R001": "hi"}\n```')
    assert parsed == {"P001_R001": "hi"}
    assert oc.sanitize_dialogue_text("“hello”") == '"hello"'


def test_get_backend_openai_compat(monkeypatch):
    monkeypatch.setattr(config, "TRANSLATOR_BACKEND", "openai_compat", raising=False)
    assert isinstance(t_be._get_backend(config), t_be.OpenAICompatBackend)
    monkeypatch.setattr(config, "PARAPHRASE_BACKEND", "openai_compat", raising=False)
    assert isinstance(p_stage._get_backend(config), p_stage.OpenAICompatBackend)


def test_translation_hardened_prompt_differs_from_ollama():
    openai_be = t_be.OpenAICompatBackend()
    openai_be.gloss_txt = ""
    hardened = openai_be._build_openai_user_prompt(
        [{"region_id": "P001_R001", "original_text": "你好"}], strict=True
    )
    assert "ZERO" in hardened or "CJK" in hardened
    assert "Brother Wei" in hardened
    assert "STRICT" in hardened

    ollama_be = t_be.OllamaBackend()
    ollama_be.gloss_txt = ""
    plain = ollama_be._build_user_prompt(
        [{"region_id": "P001_R001", "original_text": "你好"}], strict=False
    )
    assert "Brother Wei" not in plain


def test_paraphrase_hardened_prompt_contains_zero_cjk():
    be = p_stage.OpenAICompatBackend()
    be.tone = ""
    be.shorten = ""
    be.gloss_txt = ""
    prompt = be._build_openai_user_prompt(
        [{"region_id": "P001_R001", "literal_translation": "Get out!"}]
    )
    assert "ZERO CHINESE" in prompt
    assert "Brother Wei" in prompt
    assert "REGIONS" in prompt
    assert "P001_R001" in prompt


def test_translate_end_to_end_with_mocked_provider(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "TRANSLATOR_BACKEND", "openai_compat", raising=False)
    monkeypatch.setattr(config, "OPENAI_COMPAT_PROBE_ENABLED", False, raising=False)
    monkeypatch.setattr(
        config, "OPENAI_COMPAT_TRANSLATE_TEMPERATURE", 0.2, raising=False
    )

    def fake_chain(purpose, cfg):
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
                    "name": "fake",
                    "base_url": "https://example.com/v1",
                    "model": "fake-model",
                    "api_key": "k",
                }
            ],
        )

    monkeypatch.setattr(oc, "load_provider_chain", fake_chain)
    monkeypatch.setattr(
        oc, "probe_and_rank_providers", lambda providers, timeout, lg=None: providers
    )

    def fake_chat(*a, **k):
        return json.dumps({"P001_R001": "Get out!"})

    monkeypatch.setattr(oc, "call_openai_chat", fake_chat)
    monkeypatch.setattr(
        "manhua_pipeline.stages._openai_compat.call_openai_chat", fake_chat
    )

    ws = tmp_path / "ch1"
    _setup_translate(ws, [_ocr_entry("P001_R001", "滚吧！")])
    from manhua_pipeline.stages.stage3_translation.engine import run_translation

    out = run_translation(str(ws), config)
    assert out is not None
    tr = json.loads(
        (ws / "stage3_translation" / "translation.json").read_text(encoding="utf-8")
    )
    by_id = {r["region_id"]: r for r in tr["results"]}
    assert by_id["P001_R001"]["literal_translation"] == "Get out!"
    assert by_id["P001_R001"]["translation_source"] == "openai_compat"
    assert tr["translator_backend"] == "openai_compat"
    assert tr["prompt_version"] == "translation-openai-v1"


def test_paraphrase_end_to_end_with_mocked_provider(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "PARAPHRASE_BACKEND", "openai_compat", raising=False)
    monkeypatch.setattr(config, "OPENAI_COMPAT_PROBE_ENABLED", False, raising=False)

    def fake_chain(purpose, cfg):
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
                    "name": "fake",
                    "base_url": "https://example.com/v1",
                    "model": "fake-model",
                    "api_key": "k",
                }
            ],
        )

    monkeypatch.setattr(oc, "load_provider_chain", fake_chain)
    monkeypatch.setattr(
        oc, "probe_and_rank_providers", lambda providers, timeout, lg=None: providers
    )

    def fake_chat(*a, **k):
        return json.dumps({"P001_R001": "Get lost!"})

    monkeypatch.setattr(oc, "call_openai_chat", fake_chat)

    ws = tmp_path / "ch1"
    _setup_paraphrase(ws, [_trans_entry("P001_R001", "Get out of here!")])
    out = p_stage.run_paraphrase(str(ws), config)
    assert out is not None
    pa = json.loads(
        (ws / "stage4_paraphrase" / "paraphrase.json").read_text(encoding="utf-8")
    )
    by_id = {r["region_id"]: r for r in pa["results"]}
    assert by_id["P001_R001"]["final_text"] == "Get lost!"
    assert by_id["P001_R001"]["paraphrase_source"] == "openai_compat"
    assert pa["paraphraser_backend"] == "openai_compat"
    assert pa["prompt_version"] == "paraphrase-openai-v1"
