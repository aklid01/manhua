import json
import logging
from pathlib import Path

import config


def _mk_chapter(base: Path, name: str, awaiting="translate"):
    ch = base / name
    ch.mkdir(parents=True, exist_ok=True)
    manifest = {
        "chapter_id": name,
        "current_stage": awaiting,
        "completed_stages": ["import", "detect", "ocr"],
        "pages": [{"page_number": 1, "filename": "001.png", "skip": False}],
    }
    (ch / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    if awaiting == "translate":
        (ch / config.STAGE_FOLDERS["translation"]).mkdir(parents=True, exist_ok=True)
        (ch / config.STAGE_FOLDERS["ocr"]).mkdir(parents=True, exist_ok=True)
        (ch / config.STAGE_FOLDERS["ocr"] / "ocr.json").write_text(
            json.dumps(
                {
                    "results": [
                        {
                            "region_id": "P001_R001",
                            "has_usable_text": True,
                            "original_text": "gun",
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        (
            ch / config.STAGE_FOLDERS["translation"] / config.TRANSLATION_PROMPT_NAME
        ).write_text(
            json.dumps(
                {"regions": [{"region_id": "P001_R001", "original_text": "gun"}]}
            ),
            encoding="utf-8",
        )
    elif awaiting == "paraphrase":
        (ch / config.STAGE_FOLDERS["paraphrase"]).mkdir(parents=True, exist_ok=True)
        (ch / config.STAGE_FOLDERS["translation"]).mkdir(parents=True, exist_ok=True)
        (ch / config.STAGE_FOLDERS["translation"] / "translation.json").write_text(
            json.dumps(
                {
                    "results": [
                        {
                            "region_id": "P001_R001",
                            "literal_translation": "Gun.",
                            "translated": True,
                            "paraphrased": False,
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        (
            ch / config.STAGE_FOLDERS["paraphrase"] / config.PARAPHRASE_PROMPT_NAME
        ).write_text(
            json.dumps(
                {"regions": [{"region_id": "P001_R001", "literal_translation": "Gun."}]}
            ),
            encoding="utf-8",
        )


def test_bundle_front_loads_directive(tmp_path, monkeypatch):
    from manhua_pipeline.adapters import mcp_server as srv
    from manhua_pipeline.stages import stage4_paraphrase as s4

    base = tmp_path / "SeriesA"
    base.mkdir()
    _mk_chapter(base, "chap_a", awaiting="paraphrase")
    monkeypatch.setattr(srv, "get_output_dir", lambda: str(base))

    bundle = srv.get_paraphrase_bundle("chap_a")
    assert "error" not in bundle
    keys = list(bundle.keys())
    assert keys[0] == "READ_FIRST"
    assert "READ_FIRST" in bundle
    assert "SPOKEN" in bundle["READ_FIRST"].upper()
    assert s4._PROMPT_INSTRUCTIONS.strip()[:20] in bundle["READ_FIRST"]


def test_get_paraphrase_bundle_docstring_has_rule():
    from manhua_pipeline.adapters import mcp_server as srv

    doc = srv.get_paraphrase_bundle.__doc__ or ""
    up = doc.upper()
    assert "REWRITE" in up
    assert "VERBATIM" in up
    assert "SUBMIT_PARAPHRASE" in up


def test_paraphrase_prompt_embeds_directive_and_lines(tmp_path, monkeypatch):
    from manhua_pipeline.adapters import mcp_server as srv
    from manhua_pipeline.stages import stage4_paraphrase as s4

    base = tmp_path / "SeriesA"
    base.mkdir()
    _mk_chapter(base, "chap_a", awaiting="paraphrase")
    monkeypatch.setattr(srv, "get_output_dir", lambda: str(base))

    text = srv.paraphrase_chapter_impl("chap_a")
    assert isinstance(text, str)
    assert s4._PROMPT_INSTRUCTIONS.strip()[:20] in text
    assert "P001_R001" in text
    assert "JSON" in text.upper()


def test_directive_single_source(tmp_path, monkeypatch):
    from manhua_pipeline.adapters import mcp_server as srv
    from manhua_pipeline.stages import stage4_paraphrase as s4

    base = tmp_path / "SeriesA"
    base.mkdir()
    _mk_chapter(base, "chap_a", awaiting="paraphrase")
    monkeypatch.setattr(srv, "get_output_dir", lambda: str(base))

    bundle = srv.get_paraphrase_bundle("chap_a")
    text = srv.paraphrase_chapter_impl("chap_a")
    snippet = s4._PROMPT_INSTRUCTIONS.strip()[:30]
    assert snippet in bundle["READ_FIRST"]
    assert snippet in text


def test_submit_logs_written_count(tmp_path, monkeypatch, caplog):
    from manhua_pipeline.adapters import mcp_server as srv

    base = tmp_path / "SeriesA"
    base.mkdir()
    _mk_chapter(base, "chap_a", awaiting="paraphrase")
    monkeypatch.setattr(srv, "get_output_dir", lambda: str(base))

    with caplog.at_level(logging.INFO):
        res = srv.submit_paraphrase(
            "chap_a", {"P001_R001": "Hell yeah!", "P001_R999": ""}
        )
    assert res["written"] == 1
    assert any("n=1" in m for m in caplog.messages)
