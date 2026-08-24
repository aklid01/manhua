import json

from PIL import Image

import config
from manhua_pipeline.stages.stage0_import import run_import


def test_small_slices_are_stitched_together(tmp_path):
    src = tmp_path / "raw_chapter"
    src.mkdir()
    ws = tmp_path / "workspace"

    # Create three small 400px high slices of width 800px
    for i in range(1, 4):
        im = Image.new("RGB", (800, 400), (i * 50, i * 50, i * 50))
        im.save(src / f"{i:03d}.jpg", "JPEG")

    run_import(str(src), str(ws), config)

    manifest_path = ws / config.MANIFEST_NAME
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    # Three 400px slices combined into one 1200px page
    assert manifest["total_pages"] == 1
    page = manifest["pages"][0]
    assert page["filename"] == "001.jpg"
    assert page["width"] == 800
    assert page["height"] == 1200

    imported_img = Image.open(ws / "pages" / "001.jpg")
    assert imported_img.size == (800, 1200)


def test_slices_with_different_widths_not_stitched(tmp_path):
    src = tmp_path / "raw_chapter"
    src.mkdir()
    ws = tmp_path / "workspace"

    # Slice 1: 800x400, Slice 2: 1000x400
    Image.new("RGB", (800, 400), "white").save(src / "001.jpg", "JPEG")
    Image.new("RGB", (1000, 400), "black").save(src / "002.jpg", "JPEG")

    run_import(str(src), str(ws), config)

    manifest = json.loads((ws / config.MANIFEST_NAME).read_text(encoding="utf-8"))
    assert manifest["total_pages"] == 2
    assert manifest["pages"][0]["width"] == 800
    assert manifest["pages"][1]["width"] == 1000


def test_no_hard_links_used(tmp_path):
    src = tmp_path / "raw_chapter"
    src.mkdir()
    ws = tmp_path / "workspace"

    src_file = src / "001.jpg"
    Image.new("RGB", (800, 1200), "white").save(src_file, "JPEG")

    run_import(str(src), str(ws), config)

    dest_file = ws / "pages" / "001.jpg"
    assert dest_file.exists()

    # Verify that modifying dest_file does NOT modify src_file (independent files)
    with open(dest_file, "wb") as f:
        f.write(b"modified_data")

    # src_file should still be a valid JPEG with original size
    with Image.open(src_file) as im:
        assert im.size == (800, 1200)


def test_normal_sized_pages_not_unnecessarily_stitched(tmp_path):
    src = tmp_path / "raw_chapter"
    src.mkdir()
    ws = tmp_path / "workspace"

    # Two standard 1500px tall pages
    Image.new("RGB", (800, 1500), "white").save(src / "001.jpg", "JPEG")
    Image.new("RGB", (800, 1500), "black").save(src / "002.jpg", "JPEG")

    run_import(str(src), str(ws), config)

    manifest = json.loads((ws / config.MANIFEST_NAME).read_text(encoding="utf-8"))
    assert manifest["total_pages"] == 2
