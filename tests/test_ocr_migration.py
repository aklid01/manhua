import pytest
import numpy as np
from PIL import Image
from unittest.mock import MagicMock

import config
from manhua_pipeline.stages.stage2_ocr import _read_crop


class FakeGeneratorPredictor:
    def __init__(self, items):
        self.items = items

    def predict(self, img):
        # Return an iterable/generator to simulate PaddleOCR 3.x behavior
        return (item for item in self.items)


def test_read_crop_generator_success():
    crop = Image.new("RGB", (100, 50))
    # Generator returns one result dict
    predictor = FakeGeneratorPredictor([
        {"rec_texts": ["测试文本", "line2"], "rec_scores": [0.95, 0.88]}
    ])
    text, mean_conf, min_conf, watermark_filtered = _read_crop(predictor, crop, config)
    assert text == "测试文本\nline2"
    assert mean_conf == (0.95 + 0.88) / 2
    assert min_conf == 0.88
    assert watermark_filtered is False


def test_read_crop_generator_empty():
    crop = Image.new("RGB", (100, 50))
    # Generator is empty
    predictor = FakeGeneratorPredictor([])
    text, mean_conf, min_conf, watermark_filtered = _read_crop(predictor, crop, config)
    assert text == ""
    assert mean_conf == 0.0
    assert min_conf == 0.0
    assert watermark_filtered is False


def test_read_crop_schema_type_error():
    crop = Image.new("RGB", (100, 50))
    # Generator returns object without .get method
    predictor = FakeGeneratorPredictor([
        "invalid_string_instead_of_dict"
    ])
    with pytest.raises(TypeError, match="Unexpected PaddleOCR result type"):
        _read_crop(predictor, crop, config)


def test_read_crop_mismatched_texts_scores():
    crop = Image.new("RGB", (100, 50))
    # Mismatched lengths
    predictor = FakeGeneratorPredictor([
        {"rec_texts": ["text1", "text2"], "rec_scores": [0.95]}
    ])
    text, mean_conf, min_conf, watermark_filtered = _read_crop(predictor, crop, config)
    assert text == "text1\ntext2"
    assert mean_conf == (0.95 + 0.0) / 2
    assert min_conf == 0.0
    assert watermark_filtered is False


def test_read_crop_watermark_filtering():
    crop = Image.new("RGB", (100, 50))
    predictor = FakeGeneratorPredictor([
        {"rec_texts": ["测试文本", "www.baozimh.com"], "rec_scores": [0.95, 0.99]}
    ])
    text, mean_conf, min_conf, watermark_filtered = _read_crop(predictor, crop, config)
    assert text == "测试文本"
    assert mean_conf == 0.95
    assert min_conf == 0.95
    assert watermark_filtered is True
