"""Offline tests for Stage 7 packaging."""

import json
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


def test_unsupported_formats_are_ignored(tmp_path, caplog):
    from manhua_pipeline.stages import stage7_package as s7

    ws = _setup(tmp_path)
    assert s7.run_package(str(ws), config, ["pdf", "tar"]) == []
    assert "Unknown format ignored" in caplog.text


def test_singleton_materialization_preserves_source_bytes(tmp_path, monkeypatch):
    from manhua_pipeline.stages import stage7_package as s7

    ws = _setup(tmp_path)
    source = ws / config.STAGE_FOLDERS["render"] / "rendered" / "001.png"
    before = source.read_bytes()
    out_dir = tmp_path / "package-pages"
    out_dir.mkdir()

    def fail_if_composed(*args, **kwargs):
        raise AssertionError("singleton groups must not be recomposed")

    monkeypatch.setattr(s7, "_compose_group", fail_if_composed)
    outputs = s7._materialize_groups([[source]], out_dir)

    assert [p.name for p in outputs] == ["001.png"]
    assert outputs[0].read_bytes() == before


def test_singleton_materialization_preserves_non_png_extension(tmp_path):
    from manhua_pipeline.stages import stage7_package as s7

    source = tmp_path / "source.jpg"
    Image.new("RGB", (20, 20), (20, 30, 40)).save(source, format="JPEG")
    out_dir = tmp_path / "package-pages"
    out_dir.mkdir()

    outputs = s7._materialize_groups([[source]], out_dir)

    assert outputs[0].name == "001.jpg"
    assert outputs[0].read_bytes() == source.read_bytes()


def test_only_multi_page_groups_are_composed(tmp_path, monkeypatch):
    from manhua_pipeline.stages import stage7_package as s7

    ws = _setup(tmp_path)
    render = ws / config.STAGE_FOLDERS["render"] / "rendered"
    out_dir = tmp_path / "package-pages"
    out_dir.mkdir()
    calls = []

    def record_composition(group, out):
        calls.append((group, out))

    monkeypatch.setattr(s7, "_compose_group", record_composition)
    s7._materialize_groups(
        [[render / "001.png"], [render / "002.png", render / "010.png"]],
        out_dir,
    )

    assert [[p.name for p in group] for group, _ in calls] == [["002.png", "010.png"]]


def test_oversized_page_above_hard_max_is_kept_intact_and_warns(
    tmp_path, monkeypatch, caplog
):
    from manhua_pipeline.stages import stage7_package as s7

    ws = _setup(tmp_path)
    render = ws / config.STAGE_FOLDERS["render"] / "rendered"
    page = render / "oversized.png"
    Image.new("RGB", (50, 250), (200, 200, 200)).save(page)
    monkeypatch.setattr(config, "PACKAGE_PIXEL_BUDGET", 100)
    monkeypatch.setattr(config, "PACKAGE_MAX_COMPOSITE_HEIGHT", 200)

    groups = s7._build_groups([page], config)

    assert [[p.name for p in group] for group in groups] == [["oversized.png"]]
    assert "hard maximum" in caplog.text


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
        [
            render / "001.png",
            render / "002.png",
            render / "010.png",
            render / "zzz_credits.png",
        ],
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
