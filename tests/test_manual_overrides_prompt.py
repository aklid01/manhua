# Unit tests for manual override terminal prompts, deduplication, and resolution dispatch.

import json
import sys
from pathlib import Path

import config
from manhua_pipeline.interaction import can_prompt_now
from manhua_pipeline.manual_overrides import (
    ManualOverrideDecision,
    OverrideRequestStore,
    build_request_id,
    make_override_request,
    save_locked_glossary_entry,
)
from pipeline import _run_all_from, resolve_overrides


# Verify terminal prompting is disabled in batch execution mode
def test_can_prompt_now_false_when_batch():
    assert can_prompt_now(batch=True) is False


# Verify terminal prompting is allowed when stdin and stdout are interactive TTYs
def test_can_prompt_now_true_when_tty(monkeypatch):
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True, raising=False)
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True, raising=False)
    assert can_prompt_now(batch=False) is True


# Verify terminal prompting is disabled when stdin is not a TTY
def test_can_prompt_now_false_when_not_tty(monkeypatch):
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False, raising=False)
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True, raising=False)
    assert can_prompt_now(batch=False) is False


# Verify override request ID is deterministic and independent of execution run_id
def test_request_id_is_run_independent():
    a = build_request_id("1", "translation", "P010_R001")
    b = build_request_id("1", "translation", "P010_R001")
    assert a == b == "1:translation:P010_R001"


# Verify pending override queue upsert deduplicates requests for the same region across runs
def test_queue_upsert_does_not_duplicate(tmp_path):
    store = OverrideRequestStore(tmp_path)
    region = {"region_id": "P010_R001", "original_text": "以及·", "page_number": 10}
    for run in ("runA", "runB", "runC"):
        req = make_override_request(
            run_id=run,
            chapter="1",
            region=region,
            initial_reason="missing_after_batch",
            final_reason="blank_after_contextual_retry",
            backend="ollama",
            model="qwen2.5:3b-instruct",
            attempts=2,
        )
        store.upsert_pending(req)
    assert len(store.list_pending()) == 1


# Verify interactive execution runs translate and paraphrase in-process for prompt capability
def test_run_all_forces_inprocess_for_translate_when_interactive(
    monkeypatch, temp_workspace
):
    (temp_workspace / config.MANIFEST_NAME).write_text(
        json.dumps({"chapter_id": "test_ch", "current_stage": "translate"})
    )

    monkeypatch.setattr("pipeline.can_prompt_now", lambda *, batch=False: True)
    monkeypatch.setattr(config, "BATCH_SUBPROCESS", True)

    subprocess_calls = []
    inprocess_calls = []

    def mock_subprocess(stage, chapter_dir, **kwargs):
        subprocess_calls.append(stage)
        return 0

    def mock_load_stage(name):
        def run_mock(*args, **kwargs):
            inprocess_calls.append(name)
            return Path("dummy")

        return run_mock

    monkeypatch.setattr("pipeline._run_stage_subprocess", mock_subprocess)
    monkeypatch.setattr("pipeline._load_stage", mock_load_stage)

    rc = _run_all_from(
        str(temp_workspace),
        config,
        start="translate",
        run_id="test_run",
        allow_prompt=True,
    )

    assert rc == 0
    assert "translate" in inprocess_calls
    assert "paraphrase" in inprocess_calls
    assert "translate" not in subprocess_calls
    assert "paraphrase" not in subprocess_calls


# Verify batch command execution runs stages in subprocesses and never prompts
def test_batch_command_never_prompts(monkeypatch, temp_workspace):
    (temp_workspace / config.MANIFEST_NAME).write_text(
        json.dumps({"chapter_id": "test_ch", "current_stage": "translate"})
    )

    monkeypatch.setattr("pipeline.can_prompt_now", lambda *, batch=False: True)
    monkeypatch.setattr(config, "BATCH_SUBPROCESS", True)
    monkeypatch.setattr(config, "SUBPROCESS_STAGES", None)

    subprocess_calls = []
    inprocess_calls = []

    def mock_subprocess(stage, chapter_dir, **kwargs):
        subprocess_calls.append(stage)
        manifest_path = temp_workspace / config.MANIFEST_NAME
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if stage == "translate":
            manifest["current_stage"] = "paraphrase"
        elif stage == "paraphrase":
            manifest["current_stage"] = "render"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        return 0

    def mock_load_stage(name):
        def run_mock(*args, **kwargs):
            inprocess_calls.append(name)
            return Path("dummy")

        return run_mock

    monkeypatch.setattr("pipeline._run_stage_subprocess", mock_subprocess)
    monkeypatch.setattr("pipeline._load_stage", mock_load_stage)

    rc = _run_all_from(
        str(temp_workspace),
        config,
        start="translate",
        run_id="test_run",
        allow_prompt=False,
    )

    assert rc == 0
    assert "translate" in subprocess_calls
    assert "paraphrase" in subprocess_calls
    assert "translate" not in inprocess_calls
    assert "paraphrase" not in inprocess_calls


# Verify resolve_overrides CLI command exits with error when not attached to a TTY
def test_resolve_overrides_requires_tty(monkeypatch, temp_workspace):
    monkeypatch.setattr(
        "manhua_pipeline.interaction.can_prompt_now", lambda *, batch=False: False
    )
    rc = resolve_overrides(temp_workspace, config)
    assert rc == 2


# Verify resolve_overrides applies single-region manual override to overrides.json
def test_resolve_overrides_writes_region_override(monkeypatch, temp_workspace):
    monkeypatch.setattr(
        "manhua_pipeline.interaction.can_prompt_now", lambda *, batch=False: True
    )

    store = OverrideRequestStore(temp_workspace)
    region = {"region_id": "P010_R001", "original_text": "以及·", "page_number": 10}
    req = make_override_request(
        run_id="test",
        chapter=temp_workspace.name,
        region=region,
        initial_reason="missing_after_batch",
        final_reason="blank_after_contextual_retry",
        backend="ollama",
        model="qwen",
        attempts=2,
    )
    store.upsert_pending(req)

    mock_decision = ManualOverrideDecision(action="apply_region", translation="And.")
    monkeypatch.setattr(
        "manhua_pipeline.manual_overrides.prompt_for_override", lambda r: mock_decision
    )

    rc = resolve_overrides(temp_workspace, config)
    assert rc == 0

    overrides_file = temp_workspace / "overrides.json"
    assert overrides_file.exists()
    overrides_data = json.loads(overrides_file.read_text(encoding="utf-8"))
    assert overrides_data["P010_R001"]["translation"] == "And."

    assert len(store.list_pending()) == 0
    all_rows = store._load()
    assert all_rows[0]["status"] == "resolved"
    assert all_rows[0]["resolved_scope"] == "region"
    assert all_rows[0]["resolved_translation"] == "And."


# Verify save_locked_glossary_entry writes structured glossary dictionary with locked term
def test_save_locked_glossary_entry_writes_dict(temp_workspace):
    glossary_path = temp_workspace / "glossary.json"
    region = {"region_id": "P010_R001", "original_text": "以及·", "page_number": 10}
    req = make_override_request(
        run_id="test",
        chapter=temp_workspace.name,
        region=region,
        initial_reason="missing_after_batch",
        final_reason="blank_after_contextual_retry",
        backend="ollama",
        model="qwen",
        attempts=2,
    )

    save_locked_glossary_entry([], req, "and..", glossary_path)

    assert glossary_path.exists()
    glossary_data = json.loads(glossary_path.read_text(encoding="utf-8"))
    assert isinstance(glossary_data, dict)
    assert glossary_data["version"] == "v1"
    assert len(glossary_data["terms"]) == 1
    assert glossary_data["terms"][0]["source_term"] == "以及·"
    assert glossary_data["terms"][0]["target_term"] == "and.."
    assert glossary_data["terms"][0]["locked"] is True


# Verify resolve_overrides applies glossary-level override to series glossary.json
def test_resolve_overrides_writes_glossary_override(monkeypatch, temp_workspace):
    monkeypatch.setattr(
        "manhua_pipeline.interaction.can_prompt_now", lambda *, batch=False: True
    )

    store = OverrideRequestStore(temp_workspace)
    region = {"region_id": "P010_R001", "original_text": "以及·", "page_number": 10}
    req = make_override_request(
        run_id="test",
        chapter=temp_workspace.name,
        region=region,
        initial_reason="missing_after_batch",
        final_reason="blank_after_contextual_retry",
        backend="ollama",
        model="qwen",
        attempts=2,
    )
    store.upsert_pending(req)

    mock_decision = ManualOverrideDecision(action="apply_glossary", translation="and..")
    monkeypatch.setattr(
        "manhua_pipeline.manual_overrides.prompt_for_override", lambda r: mock_decision
    )

    rc = resolve_overrides(temp_workspace, config)
    assert rc == 0

    glossary_file = temp_workspace.parent / "glossary.json"
    assert glossary_file.exists()
    glossary_data = json.loads(glossary_file.read_text(encoding="utf-8"))
    assert isinstance(glossary_data, dict)
    assert glossary_data["terms"][0]["target_term"] == "and.."
