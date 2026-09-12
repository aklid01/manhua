# Unit tests for FastMCP server adapter endpoints and payload validation.

import json
from pathlib import Path

import config


# Helper to set up mock chapter workspace awaiting translation or paraphrase
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


# Verify MCP list_pending tool discovers chapters awaiting translation or paraphrase
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


# Verify MCP get_translation_bundle retrieves regions for the requested chapter
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


# Verify MCP submit_translation writes validated translations to response file
def test_submit_translation_writes_response(tmp_path, monkeypatch):
    from manhua_pipeline.adapters import mcp_server as srv

    base = tmp_path / "SeriesA"
    base.mkdir()
    _mk_chapter(base, "chap_a", awaiting="translate")
    monkeypatch.setattr(srv, "get_output_dir", lambda: str(base))

    res = srv.submit_translation("chap_a", {"P001_R001": "Get lost!"})
    assert res["written"] == 1
    resp_file = (
        base
        / "chap_a"
        / config.STAGE_FOLDERS["translation"]
        / config.TRANSLATION_RESPONSE_NAME
    )
    resp = json.loads(resp_file.read_text(encoding="utf-8"))
    assert resp["P001_R001"] == "Get lost!"


# Verify MCP submit_translation returns error for non-existent chapter
def test_submit_unknown_chapter_errors(tmp_path, monkeypatch):
    from manhua_pipeline.adapters import mcp_server as srv

    base = tmp_path / "SeriesA"
    base.mkdir()
    monkeypatch.setattr(srv, "get_output_dir", lambda: str(base))

    res = srv.submit_translation("ghost", {"P001_R001": "x"})
    assert "error" in res


# Verify MCP submit_translation filters out empty and non-string values
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


# Verify MCP submit_translation rejects submission when no pending prompt exists
def test_submit_without_pending_bundle_errors(tmp_path, monkeypatch):
    from manhua_pipeline.adapters import mcp_server as srv

    base = tmp_path / "SeriesA"
    base.mkdir()
    ch = base / "chap_a"
    ch.mkdir()
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


# Verify MCP get_paraphrase_bundle retrieves regions awaiting paraphrase
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


# Verify MCP submit_paraphrase writes validated paraphrases to response file
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


# Verify MCP submit_paraphrase rejects submission when no prompt exists
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


# Write a seed glossary.json into the series base directory
def _seed_glossary(base: Path, terms: list) -> Path:
    path = base / config.GLOSSARY_NAME
    path.write_text(
        json.dumps(
            {
                "version": "v1.0",
                "updated_at": "2026-08-11T10:23:00.000000+00:00",
                "_readme": "Series glossary.",
                "terms": terms,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return path


# Read back the on-disk glossary as a dict
def _read_glossary(base: Path) -> dict:
    return json.loads((base / config.GLOSSARY_NAME).read_text(encoding="utf-8"))


# Verify MCP update_glossary appends new terms and updates existing ones in place
def test_update_glossary_upserts_new_and_existing(tmp_path, monkeypatch):
    from manhua_pipeline.adapters import mcp_server as srv

    base = tmp_path / "SeriesA"
    base.mkdir()
    _seed_glossary(
        base,
        [
            {
                "source_term": "姐姐",
                "target_term": "big sister",
                "locked": False,
                "notes": "existing note",
            }
        ],
    )
    monkeypatch.setattr(srv, "get_output_dir", lambda: str(base))

    res = srv.update_glossary(
        [
            {"source_term": "姐姐", "target_term": "Sis"},
            {"source_term": "哥哥", "target_term": "big brother", "locked": True},
        ]
    )

    assert "error" not in res
    assert res["written"] == 2
    assert res["total_terms"] == 2

    terms = {t["source_term"]: t for t in _read_glossary(base)["terms"]}
    # Existing entry updated in place, not duplicated.
    assert terms["姐姐"]["target_term"] == "Sis"
    # Omitted optional fields must not clobber what is on disk.
    assert terms["姐姐"]["notes"] == "existing note"
    assert terms["哥哥"]["target_term"] == "big brother"
    assert terms["哥哥"]["locked"] is True


# Verify MCP update_glossary refuses to overwrite a locked term
def test_update_glossary_warns_and_skips_locked(tmp_path, monkeypatch):
    from manhua_pipeline.adapters import mcp_server as srv

    base = tmp_path / "SeriesA"
    base.mkdir()
    _seed_glossary(
        base,
        [
            {
                "source_term": "姐姐",
                "target_term": "big sister",
                "locked": True,
                "notes": "",
            }
        ],
    )
    monkeypatch.setattr(srv, "get_output_dir", lambda: str(base))

    res = srv.update_glossary([{"source_term": "姐姐", "target_term": "Sis"}])

    assert res["written"] == 0
    assert res["skipped"] == 1
    assert res["warning_count"] == 1
    assert "姐姐" in res["warnings"][0]
    assert _read_glossary(base)["terms"][0]["target_term"] == "big sister"

    # Resubmitting the identical target is not a conflict, so it is allowed.
    res2 = srv.update_glossary([{"source_term": "姐姐", "target_term": "big sister"}])
    assert res2["written"] == 1
    assert res2["warning_count"] == 0


# Verify MCP update_glossary warns on invalid entries but still writes valid ones
def test_update_glossary_validation_warnings(tmp_path, monkeypatch):
    from manhua_pipeline.adapters import mcp_server as srv

    base = tmp_path / "SeriesA"
    base.mkdir()
    _seed_glossary(base, [])
    monkeypatch.setattr(srv, "get_output_dir", lambda: str(base))

    res = srv.update_glossary(
        [
            "not-an-object",
            {"target_term": "orphan"},
            {"source_term": "哥哥"},
            {"source_term": "姐姐", "target_term": "big sister", "locked": "yes"},
            {"source_term": "姐姐", "target_term": "Sis"},
            {"source_term": "妹妹", "target_term": "little sister", "bogus": 1},
        ]
    )

    assert res["written"] == 2
    assert res["skipped"] == 4
    assert res["warning_count"] == len(res["warnings"])
    joined = " ".join(res["warnings"])
    assert "not a JSON object" in joined
    assert "source_term" in joined
    assert "target_term" in joined
    assert "locked" in joined
    assert "bogus" in joined

    terms = {t["source_term"]: t for t in _read_glossary(base)["terms"]}
    # Duplicate source_term: first occurrence wins.
    assert terms["姐姐"]["target_term"] == "big sister"
    assert terms["姐姐"]["locked"] is False
    assert "bogus" not in terms["妹妹"]


# Verify MCP update_glossary rejects source terms that are not CJK text
def test_update_glossary_rejects_non_cjk_source(tmp_path, monkeypatch):
    from manhua_pipeline.adapters import mcp_server as srv

    base = tmp_path / "SeriesA"
    base.mkdir()
    _seed_glossary(base, [])
    monkeypatch.setattr(srv, "get_output_dir", lambda: str(base))

    res = srv.update_glossary(
        [
            {"source_term": "big sister", "target_term": "Sis"},
            {"source_term": "姐姐", "target_term": "big sister"},
        ]
    )

    assert "error" not in res
    assert res["written"] == 1
    assert res["skipped"] == 1
    assert res["warning_count"] == 1
    assert "must be CJK" in res["warnings"][0]
    assert "big sister" in res["warnings"][0]

    # Only the CJK source term reached the glossary.
    terms = {t["source_term"]: t for t in _read_glossary(base)["terms"]}
    assert set(terms) == {"姐姐"}
    assert terms["姐姐"]["target_term"] == "big sister"


# Verify MCP update_glossary preserves file metadata and applies optional-field defaults
def test_update_glossary_preserves_metadata_and_defaults(tmp_path, monkeypatch):
    from manhua_pipeline.adapters import mcp_server as srv

    base = tmp_path / "SeriesA"
    base.mkdir()
    path = _seed_glossary(base, [])
    before = path.read_text(encoding="utf-8")
    monkeypatch.setattr(srv, "get_output_dir", lambda: str(base))

    res = srv.update_glossary([{"source_term": "姐姐", "target_term": "big sister"}])

    assert res["written"] == 1
    data = _read_glossary(base)
    # Top-level metadata survives the round-trip.
    assert data["version"] == "v1.0"
    assert data["_readme"] == "Series glossary."
    assert data["updated_at"] != "2026-08-11T10:23:00.000000+00:00"
    # Optional fields default rather than being omitted.
    assert data["terms"][0]["locked"] is False
    assert data["terms"][0]["notes"] == ""

    # An all-invalid submission must not touch the file at all.
    res2 = srv.update_glossary([{"source_term": "  ", "target_term": "x"}])
    assert res2["written"] == 0
    assert res2["warning_count"] == 1
    assert path.read_text(encoding="utf-8") != before  # written above
    srv.update_glossary([])
    assert _read_glossary(base)["terms"][0]["target_term"] == "big sister"
