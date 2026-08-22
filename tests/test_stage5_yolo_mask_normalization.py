import base64
import io
import numpy as np
from PIL import Image

from manhua_pipeline.stages.stage5_render import inpainting, types
from manhua_pipeline.stages.stage5_render.engine import _load_and_merge_regions


def _encode_mask(pil_mask: Image.Image) -> str:
    buf = io.BytesIO()
    pil_mask.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def test_load_and_merge_regions_preserves_bubble_mask():
    mock_det = {
        "regions": [
            {
                "region_id": "P001_R001",
                "page_number": 1,
                "bbox": {"x": 100, "y": 100, "w": 200, "h": 100},
                "parent_bubble": {"x": 80, "y": 80, "w": 240, "h": 140},
                "bubble_mask": {
                    "valid": True,
                    "conf": 0.85,
                    "mask_b64": "dummy_base64_data",
                },
                "render": True,
                "style_hint": "round",
            }
        ]
    }
    mock_ocr = {
        "results": [
            {
                "region_id": "P001_R001",
                "page_number": 1,
                "has_usable_text": True,
                "original_text": "你好",
            }
        ]
    }
    mock_para = {
        "results": [
            {
                "region_id": "P001_R001",
                "page_number": 1,
                "final_text": "Hello",
                "register": "neutral",
            }
        ]
    }

    merged = _load_and_merge_regions(mock_det, mock_ocr, mock_para)
    assert 1 in merged
    reg = merged[1][0]
    assert "bubble_mask" in reg
    assert reg["bubble_mask"]["valid"] is True
    assert reg["bubble_mask"]["conf"] == 0.85


def test_load_region_bubble_mask_normalizes_holes_and_protects_text():
    bw, bh = 200, 200
    bubble_box = types.Box(50, 50, bw, bh)
    local_text_box = types.Box(40, 40, 120, 100)

    # 1. Create a raw mask with an interior hole and an outer leak
    raw_mask_arr = np.zeros((bh, bw), dtype=np.uint8)
    # Main oval-ish blob
    raw_mask_arr[20:180, 20:180] = 255
    # Punch an interior hole in the middle
    raw_mask_arr[70:110, 70:110] = 0
    # Outer spiky leak bridge extending to edge
    raw_mask_arr[20:40, 170:200] = 255

    raw_mask_img = Image.fromarray(raw_mask_arr, mode="L")
    b64_str = _encode_mask(raw_mask_img)

    region = {
        "bubble_mask": {
            "valid": True,
            "conf": 0.88,
            "mask_b64": b64_str,
        }
    }

    # 2. Create a bubble crop with dark outline border at right edge
    crop_arr = np.full((bh, bw, 3), 255, dtype=np.uint8)
    # Dark border on the right edge
    crop_arr[:, 175:180] = (20, 20, 20)
    # Pink nebula background outside border
    crop_arr[:, 180:] = (240, 150, 180)
    bubble_crop = Image.fromarray(crop_arr, mode="RGB")

    normalized_mask = inpainting.load_region_bubble_mask(
        region,
        bubble_box,
        bubble_crop=bubble_crop,
        text_box=local_text_box,
    )

    assert normalized_mask is not None
    norm_arr = np.array(normalized_mask)

    # Assert 1: Interior hole was 100% filled
    assert np.all(norm_arr[70:110, 70:110] == 255)

    # Assert 2: Text box area is 100% solid white
    tx, ty, tw, th = (
        local_text_box.x,
        local_text_box.y,
        local_text_box.w,
        local_text_box.h,
    )
    assert np.all(norm_arr[ty : ty + th, tx : tx + tw] == 255)

    # Assert 3: Outer leak over dark border and pink background is pruned
    assert np.all(norm_arr[:, 180:] == 0)


def test_load_region_bubble_mask_returns_none_when_invalid():
    bubble_box = types.Box(0, 0, 100, 100)
    assert inpainting.load_region_bubble_mask({}, bubble_box) is None
    assert (
        inpainting.load_region_bubble_mask(
            {"bubble_mask": {"valid": False}}, bubble_box
        )
        is None
    )
