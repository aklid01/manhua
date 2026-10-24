import json

from PIL import Image, ImageDraw

import config
from manhua_pipeline.stages.stage0_import import run_import


def _make_strip(path, size, blocks):
    image = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(image)
    for top, bottom in blocks:
        draw.rectangle((60, top, size[0] - 60, bottom), fill="black")
    image.save(path, "WEBP")


# Regression: real 720x4200 webtoon strips must be sliced before detection.
# With the old target of 5000px the cut loop required height >= 7500px, so every
# strip between IMPORT_SLICE_MIN_HEIGHT (4000) and 7500px passed the trigger but
# was silently kept intact and reached RT-DETR (640x640 squash) as one piece.
def test_webtoon_strip_720x4200_is_sliced(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    ws = tmp_path / "workspace"

    _make_strip(src / "001.webp", (720, 4200), [(400, 900), (2400, 2900), (3300, 3900)])

    run_import(str(src), str(ws), config)

    manifest = json.loads((ws / config.MANIFEST_NAME).read_text(encoding="utf-8"))
    pages = manifest["pages"]
    assert len(pages) >= 2
    assert all(p["slice_count"] == len(pages) for p in pages)
    assert [p["slice_index"] for p in pages] == list(range(1, len(pages) + 1))
    assert all(p["height"] <= 2600 for p in pages)
    assert all(p["width"] == 720 for p in pages)
    assert manifest["warning_count"] == 0
    for p in pages:
        imported = Image.open(ws / "pages" / p["filename"])
        assert imported.size == (p["width"], p["height"])


def test_very_long_strip_slices_into_multiple_pieces(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    ws = tmp_path / "workspace"

    _make_strip(
        src / "001.webp",
        (720, 10000),
        [(300, 800), (3000, 3500), (5200, 5700), (7400, 7900)],
    )

    run_import(str(src), str(ws), config)

    manifest = json.loads((ws / config.MANIFEST_NAME).read_text(encoding="utf-8"))
    pages = manifest["pages"]
    assert [p["height"] for p in pages] == [2200, 2200, 2200, 2200, 1200]
    assert all(p["slice_count"] == 5 for p in pages)
    assert manifest["warning_count"] == 0


def test_dense_strip_is_kept_intact_with_warning(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    ws = tmp_path / "workspace"

    Image.new("RGB", (720, 4200), "black").save(src / "001.webp", "WEBP")

    run_import(str(src), str(ws), config)

    manifest = json.loads((ws / config.MANIFEST_NAME).read_text(encoding="utf-8"))
    assert manifest["total_pages"] == 1
    assert manifest["pages"][0]["slice_count"] == 1
    assert manifest["pages"][0]["height"] == 4200
    assert manifest["warning_count"] == 1


def test_triggered_strip_below_slice_target_warns_instead_of_silent_passthrough(
    tmp_path, monkeypatch
):
    # If IMPORT_SLICE_TARGET_HEIGHT is raised past what the trigger gates allow
    # to be cut, the mismatch must fail loudly rather than pass through silently.
    monkeypatch.setattr(config, "IMPORT_SLICE_TARGET_HEIGHT", 5000)

    src = tmp_path / "src"
    src.mkdir()
    ws = tmp_path / "workspace"

    _make_strip(src / "001.webp", (720, 4200), [(400, 900), (2400, 2900)])

    run_import(str(src), str(ws), config)

    manifest = json.loads((ws / config.MANIFEST_NAME).read_text(encoding="utf-8"))
    assert manifest["total_pages"] == 1
    assert manifest["pages"][0]["slice_count"] == 1
    assert manifest["warning_count"] == 1


# Pages below the strip ratio gate are left intact by design; lowering
# IMPORT_SLICE_TRIGGER_RATIO would change this expectation.
def test_page_below_ratio_gate_is_not_sliced(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    ws = tmp_path / "workspace"

    _make_strip(src / "001.webp", (720, 3405), [(400, 900), (2400, 2900)])

    run_import(str(src), str(ws), config)

    manifest = json.loads((ws / config.MANIFEST_NAME).read_text(encoding="utf-8"))
    assert manifest["total_pages"] == 1
    assert manifest["pages"][0]["slice_count"] == 1
    assert manifest["pages"][0]["height"] == 3405
    assert manifest["warning_count"] == 0
