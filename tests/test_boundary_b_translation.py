# Unit tests for translation validation, locked glossary matching, contextual retries, and override dispatch.

import json

import pytest

import config
from manhua_pipeline.interaction import InteractionMode, RuntimePolicy
from manhua_pipeline.translation_validation import (
    TranslationIssue,
    is_meaningful_region,
    normalize_newlines,
    normalize_punct,
    validate_expected_regions,
)


# Verify valid translation response normalizes escaped newlines and marks region as accepted
def test_accepted_normalizes_text():
    parsed = {"P001_R001": "Hello\\nWorld"}
    issues = validate_expected_regions(parsed, {"P001_R001"})
    issue = issues["P001_R001"]
    assert issue.status == "accepted"
    assert issue.accepted_text == "Hello\nWorld"


# Verify omitted region ID in translation response is marked missing
def test_missing_key_reason():
    issues = validate_expected_regions({}, {"P001_R001"})
    assert issues["P001_R001"].status == "missing"
    assert issues["P001_R001"].reason == "missing_key"


# Verify empty translation string is marked rejected
def test_empty_string_reason():
    issues = validate_expected_regions({"P001_R001": ""}, {"P001_R001"})
    assert issues["P001_R001"].status == "rejected"
    assert issues["P001_R001"].reason == "empty_string"


# Verify whitespace-only translation string is marked rejected
def test_whitespace_only_reason():
    issues = validate_expected_regions({"P001_R001": "   "}, {"P001_R001"})
    assert issues["P001_R001"].status == "rejected"
    assert issues["P001_R001"].reason == "whitespace_only"


# Verify non-string translation value type is marked rejected
def test_non_string_reason():
    issues = validate_expected_regions({"P001_R001": 42}, {"P001_R001"})
    assert issues["P001_R001"].status == "rejected"
    assert issues["P001_R001"].reason == "non_string_value"
    assert issues["P001_R001"].raw_value_type == "int"


# Verify untranslated Chinese text exceeding CJK ratio threshold is rejected
def test_cjk_ratio_exceeded_reason():
    issues = validate_expected_regions({"P001_R001": "你好世界"}, {"P001_R001"})
    assert issues["P001_R001"].status == "rejected"
    assert issues["P001_R001"].reason == "cjk_ratio_exceeded"


# Verify mixed text with minor CJK characters below threshold is accepted
def test_mixed_cjk_below_threshold_accepted():
    issues = validate_expected_regions({"P001_R001": "Hello 世"}, {"P001_R001"})
    assert issues["P001_R001"].status == "accepted"


# Verify retry attempt number is tracked in translation issue metadata
def test_attempt_tracked():
    parsed = {"P001_R001": "Good"}
    issues = validate_expected_regions(parsed, {"P001_R001"}, attempt=2)
    assert issues["P001_R001"].attempt == 2


# Verify multiple regions in a single batch validate independently
def test_multiple_regions_independent():
    parsed = {
        "P001_R001": "Good",
        "P001_R002": "",
        "P001_R003": "你好",
    }
    issues = validate_expected_regions(
        parsed, {"P001_R001", "P001_R002", "P001_R003", "P001_R004"}
    )
    assert issues["P001_R001"].status == "accepted"
    assert issues["P001_R002"].reason == "empty_string"
    assert issues["P001_R003"].reason == "cjk_ratio_exceeded"
    assert issues["P001_R004"].reason == "missing_key"


# Verify unexpected extra keys in translation response are filtered out
def test_unexpected_keys_not_in_result():
    parsed = {"P001_R001": "Good", "SPURIOUS": "extra"}
    issues = validate_expected_regions(parsed, {"P001_R001"})
    assert "SPURIOUS" not in issues
    assert issues["P001_R001"].status == "accepted"


# Verify TranslationIssue dataclass instance is immutable
def test_translation_issue_is_frozen():
    issue = TranslationIssue(region_id="r", status="accepted")
    with pytest.raises((AttributeError, TypeError)):
        issue.status = "rejected"


# Verify text with multi-character CJK is classified as meaningful dialogue
def test_meaningful_region_with_cjk():
    assert is_meaningful_region("你好", config) is True


# Verify single-character CJK is classified as trivial
def test_trivial_region_single_cjk():
    assert is_meaningful_region("一", config) is False


# Verify empty string is classified as trivial
def test_trivial_region_empty():
    assert is_meaningful_region("", config) is False


# Verify ASCII-only text without CJK is classified as trivial for translation
def test_trivial_region_only_ascii():
    assert is_meaningful_region("ABC 123", config) is False


# Verify escaped newlines (\n, \r\n, \r, \N) are normalized to standard newlines
def test_normalize_newlines_escaped():
    assert normalize_newlines("Hello\\nWorld") == "Hello\nWorld"
    assert normalize_newlines("A\\r\\nB") == "A\nB"
    assert normalize_newlines("A\\rB") == "A\nB"
    assert normalize_newlines("Hello\\NWorld") == "Hello\nWorld"
    assert normalize_newlines("A\\R\\NB") == "A\nB"


# Verify fullwidth Chinese punctuation marks are normalized to ASCII punctuation
def test_normalize_punct_fullwidth():
    assert normalize_punct("\uff01") == "!"
    assert normalize_punct("\u3002") == "."
    assert normalize_punct("Hello\uff01") == "Hello!"


# Verify NON_INTERACTIVE mode disables interactive prompts
def test_runtime_policy_non_interactive():
    policy = RuntimePolicy(mode=InteractionMode.NON_INTERACTIVE)
    assert policy.can_prompt is False


# Verify batch execution disables interactive prompts
def test_runtime_policy_batch_disables_prompt():
    policy = RuntimePolicy(mode=InteractionMode.INTERACTIVE, batch=True)
    assert policy.can_prompt is False


# Verify MCP execution mode disables interactive prompts
def test_runtime_policy_mcp_disables_prompt():
    policy = RuntimePolicy(mode=InteractionMode.MCP)
    assert policy.can_prompt is False


# Helper to construct mock chapter workspace with OCR and glossary files
def _make_ws(tmp_path, ocr_results, glossary=None):
    ws = tmp_path / "workspace"
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
    return ws


# Helper to construct mock OCR region result dictionary
def _ocr(rid, text, usable=True):
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


# Fixture forcing manual translator backend for offline test isolation
@pytest.fixture(autouse=True)
def _force_manual_backend(monkeypatch):
    monkeypatch.setattr(config, "TRANSLATOR_BACKEND", "manual", raising=False)


# Verify exact locked glossary term match translates directly without invoking backend
def test_locked_glossary_exact_match_bypasses_backend(tmp_path):
    from manhua_pipeline.stages.stage3_translation import run_translation

    glossary = {
        "version": "v1",
        "updated_at": "now",
        "terms": [
            {
                "source_term": "经理",
                "target_term": "Manager",
                "locked": True,
            }
        ],
    }
    ws = _make_ws(tmp_path, [_ocr("P001_R001", "经理")], glossary=glossary)
    out = run_translation(str(ws), config)
    assert out is not None
    tr = json.loads(
        (ws / "stage3_translation" / "translation.json").read_text(encoding="utf-8")
    )
    r = tr["results"][0]
    assert r["translated"] is True
    assert r["literal_translation"] == "Manager"
    assert r["translation_source"] == "locked_glossary"
    assert r["glossary_conflict"] is False


# Verify substring/partial glossary match does not bypass backend
def test_locked_glossary_partial_match_does_not_bypass(tmp_path):
    from manhua_pipeline.stages.stage3_translation import run_translation

    glossary = {
        "version": "v1",
        "updated_at": "now",
        "terms": [
            {
                "source_term": "经理",
                "target_term": "Manager",
                "locked": True,
            }
        ],
    }
    ws = _make_ws(tmp_path, [_ocr("P001_R001", "你好经理！")], glossary=glossary)
    run_translation(str(ws), config)
    assert not (ws / "stage3_translation" / "translation.json").exists()


# Verify locked glossary entry with empty target translation does not bypass backend
def test_locked_glossary_empty_target_does_not_bypass(tmp_path):
    from manhua_pipeline.stages.stage3_translation import run_translation

    glossary = {
        "version": "v1",
        "updated_at": "now",
        "terms": [
            {
                "source_term": "经理",
                "target_term": "",
                "locked": True,
            }
        ],
    }
    ws = _make_ws(tmp_path, [_ocr("P001_R001", "经理")], glossary=glossary)
    run_translation(str(ws), config)
    assert not (ws / "stage3_translation" / "translation.json").exists()


# Verify OllamaBackend._validate_batch returns accepted translations dictionary
def test_validate_batch_returns_accepted_dict():
    from manhua_pipeline.stages.stage3_translation import OllamaBackend

    parsed = {"P001_R001": "Hello", "P001_R002": "World"}
    accepted, missing, unexpected = OllamaBackend._validate_batch(
        parsed, {"P001_R001", "P001_R002"}
    )
    assert accepted == {"P001_R001": "Hello", "P001_R002": "World"}
    assert missing == []
    assert unexpected == []


# Verify empty string response moves region to missing list for retry
def test_validate_batch_empty_string_goes_to_missing():
    from manhua_pipeline.stages.stage3_translation import OllamaBackend

    parsed = {"P001_R001": ""}
    accepted, missing, unexpected = OllamaBackend._validate_batch(parsed, {"P001_R001"})
    assert "P001_R001" not in accepted
    assert "P001_R001" in missing


# Verify whitespace-only response moves region to missing list
def test_validate_batch_whitespace_only_goes_to_missing():
    from manhua_pipeline.stages.stage3_translation import OllamaBackend

    parsed = {"P001_R001": "   "}
    accepted, missing, unexpected = OllamaBackend._validate_batch(parsed, {"P001_R001"})
    assert "P001_R001" not in accepted
    assert "P001_R001" in missing


# Verify non-string response moves region to missing list
def test_validate_batch_non_string_goes_to_missing():
    from manhua_pipeline.stages.stage3_translation import OllamaBackend

    parsed = {"P001_R001": 42}
    accepted, missing, unexpected = OllamaBackend._validate_batch(parsed, {"P001_R001"})
    assert "P001_R001" not in accepted
    assert "P001_R001" in missing


# Verify spurious regions in response are reported in unexpected list
def test_validate_batch_unexpected_region_reported():
    from manhua_pipeline.stages.stage3_translation import OllamaBackend

    parsed = {"P001_R001": "Hi", "SPURIOUS": "extra"}
    accepted, missing, unexpected = OllamaBackend._validate_batch(parsed, {"P001_R001"})
    assert "SPURIOUS" in unexpected
    assert accepted == {"P001_R001": "Hi"}


# Verify OverrideRequestStore appends pending manual override requests
def test_override_request_store_appends(tmp_path):
    from manhua_pipeline.manual_overrides import (
        OverrideRequestStore,
        make_override_request,
    )

    store = OverrideRequestStore(tmp_path)
    request = make_override_request(
        run_id="run-1",
        chapter="ch01",
        region={"region_id": "P001_R001", "original_text": "你好", "page_number": 1},
        initial_reason="missing_after_batch",
        final_reason="blank_after_contextual_retry",
        backend="ollama",
        model="qwen2.5:3b",
        attempts=2,
    )
    store.upsert_pending(request)
    pending = store.list_pending()
    assert len(pending) == 1
    assert pending[0]["region_id"] == "P001_R001"
    assert pending[0]["status"] == "pending"


# Verify OverrideRequestStore deduplicates repeated upserts for the same region
def test_override_request_store_idempotent_list(tmp_path):
    from manhua_pipeline.manual_overrides import (
        OverrideRequestStore,
        make_override_request,
    )

    store = OverrideRequestStore(tmp_path)
    for _ in range(3):
        store.upsert_pending(
            make_override_request(
                run_id="run-1",
                chapter="ch01",
                region={
                    "region_id": "P001_R001",
                    "original_text": "你好",
                    "page_number": 1,
                },
                initial_reason="x",
                final_reason="y",
                backend="ollama",
                model=None,
                attempts=1,
            )
        )
    assert len(store.list_pending()) == 1


# Verify single-region contextual retry prompt includes target region ID and source text
def test_contextual_retry_prompt_contains_region_id():
    from manhua_pipeline.stages.stage3_translation import _build_contextual_retry_prompt

    region = {"region_id": "P001_R001", "original_text": "你好"}
    prompt = _build_contextual_retry_prompt(region, [])
    assert "P001_R001" in prompt
    assert "你好" in prompt


# Verify contextual retry prompt includes applicable locked glossary terms
def test_contextual_retry_prompt_includes_glossary():
    from manhua_pipeline.stages.stage3_translation import _build_contextual_retry_prompt

    region = {"region_id": "P001_R001", "original_text": "经理"}
    locked = [{"source_term": "经理", "target_term": "Manager"}]
    prompt = _build_contextual_retry_prompt(region, locked)
    assert "Manager" in prompt
    assert "经理" in prompt


# Verify load_overrides parses flat key-value string dictionary from overrides.json
def test_load_overrides_accepts_legacy_string(tmp_path):
    from manhua_pipeline.io.overrides import load_overrides

    (tmp_path / "overrides.json").write_text(
        json.dumps({"P001_R001": "Hello", "_comment": "skip"}),
        encoding="utf-8",
    )
    result = load_overrides(tmp_path, config)
    assert result == {"P001_R001": "Hello"}


# Verify load_overrides parses structured manual override records
def test_load_overrides_accepts_manual_record(tmp_path):
    from manhua_pipeline.io.overrides import load_overrides
    from manhua_pipeline.manual_overrides import (
        make_override_request,
        save_region_override,
    )

    req = make_override_request(
        run_id="run-1",
        chapter="ch1",
        region={"region_id": "P003_R012", "original_text": "季青临"},
        initial_reason="test",
        final_reason="test",
        backend="ollama",
        model=None,
        attempts=1,
    )
    save_region_override({}, req, "Ji Qinglin", tmp_path / "overrides.json")

    result = load_overrides(tmp_path, config)
    assert result == {"P003_R012": "Ji Qinglin"}
