"""Offline tests for the multi-template random credits page."""
import config
from PIL import Image, ImageFont


def _mk(path, size=(400, 600)):
    Image.new("RGB", size, (12, 14, 22)).save(path)


def _slots():
    return [("scanlator", 0.5, 0.5, "center", 30, 0.4, 0.05)]


def test_credits_written(tmp_path, monkeypatch):
    from manhua_pipeline.stages import stage5_render as s5
    cdir = tmp_path / "credits"
    cdir.mkdir()
    _mk(cdir / "t1.png")
    monkeypatch.setattr(config, "CREDITS_DIR", str(cdir), raising=False)
    monkeypatch.setattr(config, "CREDITS_TEMPLATES", {"t1.png": _slots()}, raising=False)
    monkeypatch.setattr(s5, "_fit_font", lambda *a, **k: ImageFont.load_default())
    render_dir = tmp_path / "render"
    render_dir.mkdir()
    out = s5._render_credits_page(render_dir, config, (400, 600))
    assert out is not None and out.exists()
    assert out.name == "zzz_credits.png"


def test_missing_dir_returns_none(tmp_path, monkeypatch):
    from manhua_pipeline.stages import stage5_render as s5
    monkeypatch.setattr(config, "CREDITS_DIR", str(tmp_path / "nope"), raising=False)
    monkeypatch.setattr(config, "CREDITS_TEMPLATES", {"x.png": _slots()}, raising=False)
    render_dir = tmp_path / "render"
    render_dir.mkdir()
    assert s5._render_credits_page(render_dir, config, (400, 600)) is None


def test_random_only_picks_existing(tmp_path, monkeypatch):
    from manhua_pipeline.stages import stage5_render as s5
    cdir = tmp_path / "credits"
    cdir.mkdir()
    _mk(cdir / "real.png")
    monkeypatch.setattr(config, "CREDITS_DIR", str(cdir), raising=False)
    monkeypatch.setattr(
        config, "CREDITS_TEMPLATES",
        {"real.png": _slots(), "ghost.png": _slots()},
        raising=False,
    )
    monkeypatch.setattr(s5, "_fit_font", lambda *a, **k: ImageFont.load_default())
    render_dir = tmp_path / "r"
    render_dir.mkdir()
    for _ in range(15):
        assert s5._render_credits_page(render_dir, config, (400, 600)) is not None


def test_credits_excluded_from_manifest_and_report(tmp_path, monkeypatch):
    """The credits helper must not append to render results or manifest pages."""
    from manhua_pipeline.stages import stage5_render as s5
    cdir = tmp_path / "credits"
    cdir.mkdir()
    _mk(cdir / "t1.png")
    monkeypatch.setattr(config, "CREDITS_DIR", str(cdir), raising=False)
    monkeypatch.setattr(config, "CREDITS_TEMPLATES", {"t1.png": _slots()}, raising=False)
    monkeypatch.setattr(s5, "_fit_font", lambda *a, **k: ImageFont.load_default())
    render_dir = tmp_path / "render"
    render_dir.mkdir()
    s5._render_credits_page(render_dir, config, (400, 600))
    assert [p.name for p in render_dir.iterdir()] == ["zzz_credits.png"]
