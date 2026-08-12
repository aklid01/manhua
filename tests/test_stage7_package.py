"""Offline tests for Stage 7 packaging."""

import json
import tarfile
import zipfile

from PIL import Image

import config


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


def test_zip_and_cbz_contents_and_order(tmp_path):
    from manhua_pipeline.stages import stage7_package as s7

    ws = _setup(tmp_path)
    out = s7.run_package(str(ws), config, ["zip", "cbz"])
    assert {p.name for p in out} == {"0_001_.zip", "0_001_.cbz"}
    pkg = ws / config.STAGE_FOLDERS["package"]
    with zipfile.ZipFile(pkg / "0_001_.zip") as zf:
        assert zf.namelist() == ["001.png"]
    with zipfile.ZipFile(pkg / "0_001_.cbz") as zf:
        assert zf.namelist() == ["001.png"]


def test_tar_contents(tmp_path):
    from manhua_pipeline.stages import stage7_package as s7

    ws = _setup(tmp_path)
    s7.run_package(str(ws), config, ["tar"])
    with tarfile.open(ws / config.STAGE_FOLDERS["package"] / "0_001_.tar") as tf:
        assert tf.getnames() == ["001.png"]


def test_pdf_written(tmp_path):
    from manhua_pipeline.stages import stage7_package as s7

    ws = _setup(tmp_path)
    s7.run_package(str(ws), config, ["pdf"])
    pdf = ws / config.STAGE_FOLDERS["package"] / "0_001_.pdf"
    assert pdf.exists()
    assert pdf.read_bytes()[:4] == b"%PDF"


def test_unknown_format_ignored(tmp_path):
    from manhua_pipeline.stages import stage7_package as s7

    ws = _setup(tmp_path)
    out = s7.run_package(str(ws), config, ["zip", "rar"])
    assert {p.name for p in out} == {"0_001_.zip"}


def test_manifest_unchanged(tmp_path):
    from manhua_pipeline.stages import stage7_package as s7

    ws = _setup(tmp_path)
    before = (ws / config.MANIFEST_NAME).read_text()
    s7.run_package(str(ws), config, ["cbz"])
    assert (ws / config.MANIFEST_NAME).read_text() == before


def test_pixel_budget_repartitions_final_group_for_credits(tmp_path, monkeypatch):
    from manhua_pipeline.stages import stage7_package as s7

    ws = _setup(tmp_path)
    monkeypatch.setattr(config, "PACKAGE_PIXEL_BUDGET", 200)
    render = ws / config.STAGE_FOLDERS["render"] / "rendered"
    groups = s7._build_groups(
        [render / "001.png", render / "002.png", render / "010.png", render / "zzz_credits.png"],
        config,
    )
    assert [[p.name for p in group] for group in groups] == [
        ["001.png", "002.png"],
        ["010.png", "zzz_credits.png"],
    ]


def test_credits_remain_separate_when_final_page_cannot_fit(tmp_path, monkeypatch):
    from manhua_pipeline.stages import stage7_package as s7

    ws = _setup(tmp_path)
    monkeypatch.setattr(config, "PACKAGE_PIXEL_BUDGET", 100)
    render = ws / config.STAGE_FOLDERS["render"] / "rendered"
    groups = s7._build_groups(
        [render / "001.png", render / "zzz_credits.png"],
        config,
    )
    assert [[p.name for p in group] for group in groups] == [
        ["001.png"],
        ["zzz_credits.png"],
    ]
