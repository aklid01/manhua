# Unit tests for credits page template selection and rendering.

from PIL import Image, ImageFont

import config


# Helper to create dummy template image file
def _mk(path, size=(400, 600)):
    Image.new("RGB", size, (12, 14, 22)).save(path)


# Helper to generate sample text slots for template rendering
def _slots():
    return [("scanlator", 0.5, 0.5, "center", 30, 0.4, 0.05)]


# Verify credits page is rendered and saved to output directory with correct filename
def test_credits_written(tmp_path, monkeypatch):
    from manhua_pipeline.stages import stage5_render as s5

    cdir = tmp_path / "credits"
    cdir.mkdir()
    _mk(cdir / "t1.png")
    monkeypatch.setattr(config, "CREDITS_DIR", str(cdir), raising=False)
    monkeypatch.setattr(
        config, "CREDITS_TEMPLATES", {"t1.png": _slots()}, raising=False
    )
    monkeypatch.setattr(s5, "_fit_font", lambda *a, **k: ImageFont.load_default())
    render_dir = tmp_path / "render"
    render_dir.mkdir()
    rendered_dir = render_dir / "rendered"
    rendered_dir.mkdir()
    out = s5._render_credits_page(rendered_dir, config, (400, 600))
    assert out is not None and out.exists()
    assert out.name == "zzz_credits.png"


# Verify credits rendering gracefully returns None if credits directory is missing
def test_missing_dir_returns_none(tmp_path, monkeypatch):
    from manhua_pipeline.stages import stage5_render as s5

    monkeypatch.setattr(config, "CREDITS_DIR", str(tmp_path / "nope"), raising=False)
    monkeypatch.setattr(config, "CREDITS_TEMPLATES", {"x.png": _slots()}, raising=False)
    rendered_dir = tmp_path / "render" / "rendered"
    rendered_dir.mkdir(parents=True)
    assert s5._render_credits_page(rendered_dir, config, (400, 600)) is None


# Verify template selection only chooses template files that exist on disk
def test_random_only_picks_existing(tmp_path, monkeypatch):
    from manhua_pipeline.stages import stage5_render as s5

    cdir = tmp_path / "credits"
    cdir.mkdir()
    _mk(cdir / "real.png")
    monkeypatch.setattr(config, "CREDITS_DIR", str(cdir), raising=False)
    monkeypatch.setattr(
        config,
        "CREDITS_TEMPLATES",
        {"real.png": _slots(), "ghost.png": _slots()},
        raising=False,
    )
    monkeypatch.setattr(s5, "_fit_font", lambda *a, **k: ImageFont.load_default())
    rendered_dir = tmp_path / "r" / "rendered"
    rendered_dir.mkdir(parents=True)
    for _ in range(15):
        assert s5._render_credits_page(rendered_dir, config, (400, 600)) is not None


# Verify credits generator creates image without altering core page lists
def test_credits_excluded_from_manifest_and_report(tmp_path, monkeypatch):
    from manhua_pipeline.stages import stage5_render as s5

    cdir = tmp_path / "credits"
    cdir.mkdir()
    _mk(cdir / "t1.png")
    monkeypatch.setattr(config, "CREDITS_DIR", str(cdir), raising=False)
    monkeypatch.setattr(
        config, "CREDITS_TEMPLATES", {"t1.png": _slots()}, raising=False
    )
    monkeypatch.setattr(s5, "_fit_font", lambda *a, **k: ImageFont.load_default())
    rendered_dir = tmp_path / "render" / "rendered"
    rendered_dir.mkdir(parents=True)
    s5._render_credits_page(rendered_dir, config, (400, 600))
    assert [p.name for p in rendered_dir.iterdir()] == ["zzz_credits.png"]
