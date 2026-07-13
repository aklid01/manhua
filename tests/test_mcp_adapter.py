import json
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


def test_list_pending(tmp_path, monkeypatch):
    from manhua_pipeline.adapters import mcp_server as srv

    base = tmp_path / "SeriesA"
    base.mkdir()
    _mk_chapter(base, "chap_a", awaiting="translate")
    monkeypatch.setattr(srv, "get_output_dir", lambda: str(base))

    pending = srv.list_pending()
    assert any(
        p["chapter"] == "chap_a" and p["stage_awaiting"] == "translate" for p in pending
    )


def test_get_translation_bundle(tmp_path, monkeypatch):
    from manhua_pipeline.adapters import mcp_server as srv

    base = tmp_path / "SeriesA"
    base.mkdir()
    _mk_chapter(base, "chap_a", awaiting="translate")
    monkeypatch.setattr(srv, "get_output_dir", lambda: str(base))

    bundle = srv.get_translation_bundle("chap_a")
    assert "error" not in bundle
    ids = [r["region_id"] for r in bundle["regions"]]
    assert "P001_R001" in ids


def test_submit_translation_writes_response(tmp_path, monkeypatch):
    from manhua_pipeline.adapters import mcp_server as srv

    base = tmp_path / "SeriesA"
    base.mkdir()
    _mk_chapter(base, "chap_a", awaiting="translate")
    monkeypatch.setattr(srv, "get_output_dir", lambda: str(base))

    res = srv.submit_translation("chap_a", {"P001_R001": "Get lost!"})
    print("SUBMIT RESULT IS:", res)
    assert res["written"] == 1
    resp_file = (
        base
        / "chap_a"
        / config.STAGE_FOLDERS["translation"]
        / config.TRANSLATION_RESPONSE_NAME
    )
    resp = json.loads(resp_file.read_text(encoding="utf-8"))
    assert resp["P001_R001"] == "Get lost!"


def test_submit_unknown_chapter_errors(tmp_path, monkeypatch):
    from manhua_pipeline.adapters import mcp_server as srv

    base = tmp_path / "SeriesA"
    base.mkdir()
    monkeypatch.setattr(srv, "get_output_dir", lambda: str(base))

    res = srv.submit_translation("ghost", {"P001_R001": "x"})
    assert "error" in res


def test_submit_validation_drops_bad(tmp_path, monkeypatch):
    from manhua_pipeline.adapters import mcp_server as srv

    base = tmp_path / "SeriesA"
    base.mkdir()
    _mk_chapter(base, "chap_a", awaiting="translate")
    monkeypatch.setattr(srv, "get_output_dir", lambda: str(base))

    res = srv.submit_translation(
        "chap_a", {"P001_R001": "ok", "P001_R002": "", "P001_R003": 5}
    )
    assert res["written"] == 1
    assert len(res["warnings"]) > 0
    resp_file = (
        base
        / "chap_a"
        / config.STAGE_FOLDERS["translation"]
        / config.TRANSLATION_RESPONSE_NAME
    )
    resp = json.loads(resp_file.read_text(encoding="utf-8"))
    assert resp.get("P001_R001") == "ok"
    assert "P001_R002" not in resp and "P001_R003" not in resp


def test_submit_without_pending_bundle_errors(tmp_path, monkeypatch):
    from manhua_pipeline.adapters import mcp_server as srv

    base = tmp_path / "SeriesA"
    base.mkdir()
    ch = base / "chap_a"
    ch.mkdir()
    # Write manifest and ocr, but do NOT run translate (no translation_prompt.json)
    manifest = {
        "chapter_id": "chap_a",
        "current_stage": "translate",
        "completed_stages": ["import", "detect", "ocr"],
        "pages": [{"page_number": 1, "filename": "001.png", "skip": False}],
    }
    (ch / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(srv, "get_output_dir", lambda: str(base))

    res = srv.submit_translation("chap_a", {"P001_R001": "hello"})
    assert "error" in res
    assert "no pending bundle" in res["error"]


def test_get_paraphrase_bundle(tmp_path, monkeypatch):
    from manhua_pipeline.adapters import mcp_server as srv

    base = tmp_path / "SeriesA"
    base.mkdir()
    _mk_chapter(base, "chap_a", awaiting="paraphrase")
    monkeypatch.setattr(srv, "get_output_dir", lambda: str(base))

    bundle = srv.get_paraphrase_bundle("chap_a")
    assert "error" not in bundle
    ids = [r["region_id"] for r in bundle["regions"]]
    assert "P001_R001" in ids


def test_submit_paraphrase_writes_response(tmp_path, monkeypatch):
    from manhua_pipeline.adapters import mcp_server as srv

    base = tmp_path / "SeriesA"
    base.mkdir()
    _mk_chapter(base, "chap_a", awaiting="paraphrase")
    monkeypatch.setattr(srv, "get_output_dir", lambda: str(base))

    res = srv.submit_paraphrase("chap_a", {"P001_R001": "Oh yes!"})
    assert res["written"] == 1
    resp_file = (
        base
        / "chap_a"
        / config.STAGE_FOLDERS["paraphrase"]
        / config.PARAPHRASE_RESPONSE_NAME
    )
    resp = json.loads(resp_file.read_text(encoding="utf-8"))
    assert resp["P001_R001"] == "Oh yes!"


def test_submit_paraphrase_without_pending_bundle_errors(tmp_path, monkeypatch):
    from manhua_pipeline.adapters import mcp_server as srv

    base = tmp_path / "SeriesA"
    base.mkdir()
    ch = base / "chap_a"
    ch.mkdir()
    manifest = {
        "chapter_id": "chap_a",
        "current_stage": "paraphrase",
        "completed_stages": ["import", "detect", "ocr", "translate"],
        "pages": [{"page_number": 1, "filename": "001.png", "skip": False}],
    }
    (ch / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(srv, "get_output_dir", lambda: str(base))

    res = srv.submit_paraphrase("chap_a", {"P001_R001": "hello"})
    assert "error" in res
    assert "no pending bundle" in res["error"]
