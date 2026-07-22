"""Offline tests for OCR confidence retry (escalating preprocessing, keep best)."""

from PIL import Image

import config


def _crop():
    return Image.new("RGB", (40, 20), (255, 255, 255))


def test_no_retry_when_confident(monkeypatch):
    from manhua_pipeline.stages import stage2_ocr as s2

    calls = {"n": 0}

    def fake(engine, img, cfg):
        calls["n"] += 1
        return ("hello", 0.85, 0.85, False)

    monkeypatch.setattr(s2, "_read_crop", fake)
    monkeypatch.setattr(config, "OCR_RETRY_ENABLED", True, raising=False)
    text, mean, mn, wm = s2._read_best(object(), _crop(), config)
    assert mean == 0.85
    assert calls["n"] == 1  # base only; already good -> no retry


def test_no_retry_below_floor(monkeypatch):
    from manhua_pipeline.stages import stage2_ocr as s2

    calls = {"n": 0}
    monkeypatch.setattr(
        s2,
        "_read_crop",
        lambda e, i, c: (
            calls.__setitem__("n", calls["n"] + 1) or ("", 0.10, 0.10, False)
        ),
    )
    monkeypatch.setattr(config, "OCR_RETRY_ENABLED", True, raising=False)
    s2._read_best(object(), _crop(), config)
    assert calls["n"] == 1  # below floor -> probably not text -> no retry


def test_retry_keeps_best(monkeypatch):
    from manhua_pipeline.stages import stage2_ocr as s2

    seq = iter(
        [("a", 0.50, 0.50, False), ("b", 0.45, 0.45, False), ("c", 0.72, 0.72, False)]
    )
    monkeypatch.setattr(s2, "_read_crop", lambda e, i, c: next(seq))
    monkeypatch.setattr(config, "OCR_RETRY_ENABLED", True, raising=False)
    monkeypatch.setattr(config, "OCR_RETRY_MAX", 2, raising=False)
    text, mean, mn, wm = s2._read_best(object(), _crop(), config)
    assert (text, mean) == ("c", 0.72)  # highest across attempts wins


def test_retry_bounded(monkeypatch):
    from manhua_pipeline.stages import stage2_ocr as s2

    calls = {"n": 0}
    monkeypatch.setattr(
        s2,
        "_read_crop",
        lambda e, i, c: (
            calls.__setitem__("n", calls["n"] + 1) or ("x", 0.50, 0.50, False)
        ),
    )
    monkeypatch.setattr(config, "OCR_RETRY_ENABLED", True, raising=False)
    monkeypatch.setattr(config, "OCR_RETRY_MAX", 2, raising=False)
    s2._read_best(object(), _crop(), config)
    assert calls["n"] == 3  # base + 2 retries, then stop (all mediocre)


def test_disabled_flag_short_circuits(monkeypatch):
    from manhua_pipeline.stages import stage2_ocr as s2

    calls = {"n": 0}
    monkeypatch.setattr(
        s2,
        "_read_crop",
        lambda e, i, c: (
            calls.__setitem__("n", calls["n"] + 1) or ("x", 0.50, 0.50, False)
        ),
    )
    monkeypatch.setattr(config, "OCR_RETRY_ENABLED", False, raising=False)
    s2._read_best(object(), _crop(), config)
    assert calls["n"] == 1  # disabled -> base read only
