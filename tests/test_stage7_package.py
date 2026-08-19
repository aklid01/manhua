# Unit tests for Stage 7 archive packaging (ZIP/CBZ generation).

import json
import zipfile

from PIL import Image

import config
from manhua_pipeline.stages import stage7_package as s7


# Helper to configure mock workspace with rendered chapter pages
def _setup(tmp_path):
    ws = tmp_path / "0_001_"
    render = ws / config.STAGE_FOLDERS["render"] / "rendered"
    render.mkdir(parents=True)
    for n in ["001.png", "002.png", "010.png", "zzz_credits.png"]:
        Image.new("RGB", (50, 70), (200, 200, 200)).save(render / n)
    (ws / config.MANIFEST_NAME).write_text(
        json.dumps({"chapter_id": "t", "current_stage": "complete", "pages": []})
    )
    return ws


# Verify zip and cbz archives contain all rendered pages in lexicographical order
def test_zip_and_cbz_contents_and_order(tmp_path):
    ws = _setup(tmp_path)
    out = s7.run_package(str(ws), config, ["zip", "cbz"])
    assert {p.name for p in out} == {"0_001_.zip", "0_001_.cbz"}
    pkg = ws / config.STAGE_FOLDERS["package"]
    expected = ["001.png", "002.png", "010.png", "zzz_credits.png"]
    with zipfile.ZipFile(pkg / "0_001_.zip") as zf:
        assert zf.namelist() == expected
    with zipfile.ZipFile(pkg / "0_001_.cbz") as zf:
        assert zf.namelist() == expected


# Verify unsupported archive formats log a warning and produce no output
def test_unsupported_formats_are_ignored(tmp_path, caplog):
    ws = _setup(tmp_path)
    assert s7.run_package(str(ws), config, ["pdf", "tar"]) == []
    assert "Unknown format ignored" in caplog.text


# Verify unknown archive format requests filter down to supported types
def test_unknown_format_ignored(tmp_path):
    ws = _setup(tmp_path)
    out = s7.run_package(str(ws), config, ["zip", "rar"])
    assert {p.name for p in out} == {"0_001_.zip"}


# Verify packaging stage leaves manifest unmodified
def test_manifest_unchanged(tmp_path):
    ws = _setup(tmp_path)
    before = (ws / config.MANIFEST_NAME).read_text()
    s7.run_package(str(ws), config, ["cbz"])
    assert (ws / config.MANIFEST_NAME).read_text() == before


# Verify packaging preserves lossless raw image bytes inside archive
def test_source_bytes_preserved_in_archive(tmp_path):
    ws = _setup(tmp_path)
    render = ws / config.STAGE_FOLDERS["render"] / "rendered"
    source = render / "001.png"
    original_bytes = source.read_bytes()

    s7.run_package(str(ws), config, ["zip"])

    pkg = ws / config.STAGE_FOLDERS["package"]
    with zipfile.ZipFile(pkg / "0_001_.zip") as zf:
        assert zf.read("001.png") == original_bytes


# Verify rendered pages with mixed image extensions (jpg, webp, png) package correctly
def test_mixed_extensions_all_included(tmp_path):
    ws = tmp_path / "0_002_"
    render = ws / config.STAGE_FOLDERS["render"] / "rendered"
    render.mkdir(parents=True)
    Image.new("RGB", (50, 70)).save(render / "001.jpg", "JPEG")
    Image.new("RGB", (50, 70)).save(render / "002.webp", "WEBP")
    Image.new("RGB", (50, 70)).save(render / "003.png")
    (ws / config.MANIFEST_NAME).write_text(
        json.dumps({"chapter_id": "t", "current_stage": "complete", "pages": []})
    )

    out = s7.run_package(str(ws), config, ["zip"])
    pkg = ws / config.STAGE_FOLDERS["package"]
    with zipfile.ZipFile(pkg / "0_002_.zip") as zf:
        assert zf.namelist() == ["001.jpg", "002.webp", "003.png"]
