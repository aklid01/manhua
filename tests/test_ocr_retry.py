# Unit tests for OCR confidence-based retries and preprocessing escalation.

from PIL import Image

import config


# Helper to generate dummy crop image
def _crop():
    return Image.new("RGB", (40, 20), (255, 255, 255))


# Verify confident initial OCR result skips unnecessary retry passes
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
    assert calls["n"] == 1


# Verify OCR confidence below minimum floor skips retry (likely non-text noise)
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
    assert calls["n"] == 1


# Verify multiple retry attempts preserve the highest-confidence OCR output
def test_retry_keeps_best(monkeypatch):
    from manhua_pipeline.stages import stage2_ocr as s2

    seq = iter(
        [("a", 0.50, 0.50, False), ("b", 0.45, 0.45, False), ("c", 0.72, 0.72, False)]
    )
    monkeypatch.setattr(s2, "_read_crop", lambda e, i, c: next(seq))
    monkeypatch.setattr(config, "OCR_RETRY_ENABLED", True, raising=False)
    monkeypatch.setattr(config, "OCR_RETRY_MAX", 2, raising=False)
    text, mean, mn, wm = s2._read_best(object(), _crop(), config)
    assert (text, mean) == ("c", 0.72)


# Verify retry attempts are strictly bounded by OCR_RETRY_MAX
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
    assert calls["n"] == 3


# Verify disabling OCR_RETRY_ENABLED performs only a single base OCR pass
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
    assert calls["n"] == 1
