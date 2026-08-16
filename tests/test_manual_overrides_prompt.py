import json
import sys
from pathlib import Path

import config
from manhua_pipeline.interaction import can_prompt_now
from manhua_pipeline.manual_overrides import (
    OverrideRequestStore,
    build_request_id,
    make_override_request,
)
from pipeline import _run_all_from, resolve_overrides

# --- interaction tests ---


def test_can_prompt_now_false_when_batch():
    assert can_prompt_now(batch=True) is False


def test_can_prompt_now_true_when_tty(monkeypatch):
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True, raising=False)
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True, raising=False)
    assert can_prompt_now(batch=False) is True


def test_can_prompt_now_false_when_not_tty(monkeypatch):
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False, raising=False)
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True, raising=False)
    assert can_prompt_now(batch=False) is False


# --- dedup tests ---


def test_request_id_is_run_independent():
    a = build_request_id("1", "translation", "P010_R001")
    b = build_request_id("1", "translation", "P010_R001")
    assert a == b == "1:translation:P010_R001"


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


# --- run-all execution mode tests ---


def test_run_all_forces_inprocess_for_translate_when_interactive(
    monkeypatch, temp_workspace
):
    # Setup manifest so it doesn't fail import stage validation
    (temp_workspace / config.MANIFEST_NAME).write_text(
        json.dumps({"chapter_id": "test_ch", "current_stage": "translate"})
    )

    monkeypatch.setattr(
        "pipeline.can_prompt_now", lambda *, batch=False: True
    )
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

    # Let's run from translate
    rc = _run_all_from(
        str(temp_workspace),
        config,
        start="translate",
        run_id="test_run",
        allow_prompt=True,
    )

    assert rc == 0
    # translate and paraphrase are prompt-capable and terminal is interactive -> in-process
    assert "translate" in inprocess_calls
    assert "paraphrase" in inprocess_calls
    assert "translate" not in subprocess_calls
    assert "paraphrase" not in subprocess_calls


def test_batch_command_never_prompts(monkeypatch, temp_workspace):
    (temp_workspace / config.MANIFEST_NAME).write_text(
        json.dumps({"chapter_id": "test_ch", "current_stage": "translate"})
    )

    monkeypatch.setattr(
        "pipeline.can_prompt_now", lambda *, batch=False: True
    )
    monkeypatch.setattr(config, "BATCH_SUBPROCESS", True)
    monkeypatch.setattr(config, "SUBPROCESS_STAGES", None)

    subprocess_calls = []
    inprocess_calls = []

    def mock_subprocess(stage, chapter_dir, **kwargs):
        subprocess_calls.append(stage)
        # Advance the manifest stage so run-all doesn't stop
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

    # If allow_prompt is False (like under batch command), stages run as subprocesses
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


# --- resolver tests ---


def test_resolve_overrides_requires_tty(monkeypatch, temp_workspace):
    monkeypatch.setattr(
        "manhua_pipeline.interaction.can_prompt_now", lambda *, batch=False: False
    )
    rc = resolve_overrides(temp_workspace, config)
    assert rc == 2


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

    # Mock prompt_for_override to return region decision with translation
    from manhua_pipeline.manual_overrides import ManualOverrideDecision

    mock_decision = ManualOverrideDecision(action="apply_region", translation="And.")
    monkeypatch.setattr(
        "manhua_pipeline.manual_overrides.prompt_for_override", lambda r: mock_decision
    )

    rc = resolve_overrides(temp_workspace, config)
    assert rc == 0

    # Verify overrides.json contains the resolved override
    overrides_file = temp_workspace / "overrides.json"
    assert overrides_file.exists()
    overrides_data = json.loads(overrides_file.read_text(encoding="utf-8"))
    assert overrides_data["P010_R001"]["translation"] == "And."

    # Verify queue status is resolved
    assert len(store.list_pending()) == 0
    all_rows = store._load()
    assert all_rows[0]["status"] == "resolved"
    assert all_rows[0]["resolved_scope"] == "region"
    assert all_rows[0]["resolved_translation"] == "And."


def test_save_locked_glossary_entry_writes_dict(temp_workspace):
    from manhua_pipeline.manual_overrides import (
        save_locked_glossary_entry,
    )

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

    # Mock prompt_for_override to return glossary decision
    from manhua_pipeline.manual_overrides import ManualOverrideDecision

    mock_decision = ManualOverrideDecision(action="apply_glossary", translation="and..")
    monkeypatch.setattr(
        "manhua_pipeline.manual_overrides.prompt_for_override", lambda r: mock_decision
    )

    rc = resolve_overrides(temp_workspace, config)
    assert rc == 0

    # Verify glossary.json contains the resolved entry as a dict
    glossary_file = temp_workspace.parent / "glossary.json"
    assert glossary_file.exists()
    glossary_data = json.loads(glossary_file.read_text(encoding="utf-8"))
    assert isinstance(glossary_data, dict)
    assert glossary_data["terms"][0]["target_term"] == "and.."
