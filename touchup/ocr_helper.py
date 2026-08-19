import logging

import numpy as np
from PIL import Image

logger = logging.getLogger("touchup.ocr_helper")

_OCR_ENGINE = None


def get_ocr_engine():
    global _OCR_ENGINE
    if _OCR_ENGINE is not None:
        return _OCR_ENGINE

    try:
        from paddleocr import PaddleOCR

        _OCR_ENGINE = PaddleOCR(
            lang="ch",
            enable_mkldnn=False,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
        )
        logger.info("PaddleOCR engine initialized for touchup.")
    except Exception as exc:
        logger.warning("Could not initialize PaddleOCR: %s", exc)
        _OCR_ENGINE = None

    return _OCR_ENGINE


def _parse_predict_result(res: dict, x1: int, y1: int, x2: int, y2: int) -> list[dict]:
    lines = []
    raw_texts = res.get("rec_texts")
    texts = list(raw_texts) if raw_texts is not None else []

    raw_scores = res.get("rec_scores")
    scores = list(raw_scores) if raw_scores is not None else []

    raw_polys = res.get("dt_polys")
    if raw_polys is None:
        raw_polys = res.get("rec_polys")
    polys = list(raw_polys) if raw_polys is not None else []

    raw_boxes = res.get("rec_boxes")
    boxes = list(raw_boxes) if raw_boxes is not None else []

    for idx, text in enumerate(texts):
        if text is None:
            continue
        txt = str(text).strip()
        if not txt:
            continue
        score = float(scores[idx]) if idx < len(scores) else 1.0

        gx1, gy1, gx2, gy2 = x1, y1, x2, y2
        if idx < len(polys) and polys[idx] is not None:
            poly = np.asarray(polys[idx])
            if poly.ndim == 2 and poly.shape[0] >= 2:
                gx1 = int(round(x1 + float(np.min(poly[:, 0]))))
                gy1 = int(round(y1 + float(np.min(poly[:, 1]))))
                gx2 = int(round(x1 + float(np.max(poly[:, 0]))))
                gy2 = int(round(y1 + float(np.max(poly[:, 1]))))
        elif idx < len(boxes) and len(boxes[idx]) >= 4:
            b = boxes[idx]
            gx1 = int(round(x1 + float(b[0])))
            gy1 = int(round(y1 + float(b[1])))
            gx2 = int(round(x1 + float(b[2])))
            gy2 = int(round(y1 + float(b[3])))

        lines.append(
            {
                "text": txt,
                "box": (gx1, gy1, gx2, gy2),
                "score": score,
            }
        )
    return lines


def _parse_legacy_result(
    results: list, x1: int, y1: int, x2: int, y2: int
) -> list[dict]:
    lines = []
    if not results or not results[0]:
        return lines
    for entry in results[0]:
        if len(entry) >= 2 and isinstance(entry[1], (tuple, list)):
            pts = entry[0]
            txt = str(entry[1][0]).strip()
            score = float(entry[1][1]) if len(entry[1]) > 1 else 1.0
            if not txt:
                continue
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            lines.append(
                {
                    "text": txt,
                    "box": (
                        int(round(x1 + min(xs))),
                        int(round(y1 + min(ys))),
                        int(round(x1 + max(xs))),
                        int(round(y1 + max(ys))),
                    ),
                    "score": score,
                }
            )
    return lines


# Runs OCR on selected crop and returns detected text lines with global coordinates.
def extract_text_and_boxes(
    image: Image.Image,
    box_xyxy: tuple[int, int, int, int],
) -> tuple[str, list[dict]]:
    x1, y1, x2, y2 = [int(v) for v in box_xyxy]
    w, h = image.size
    x1, y1 = max(0, min(w, x1)), max(0, min(h, y1))
    x2, y2 = max(0, min(w, x2)), max(0, min(h, y2))

    if x2 <= x1 or y2 <= y1:
        return "", []

    crop = image.crop((x1, y1, x2, y2))
    crop_rgb = np.array(crop.convert("RGB"))
    crop_bgr = np.ascontiguousarray(crop_rgb[:, :, ::-1])

    engine = get_ocr_engine()
    if engine is None:
        return "", [{"text": "", "box": (x1, y1, x2, y2), "score": 1.0}]

    lines = []
    # Try PaddleX v3 predict API first, then fall back to legacy ocr()
    try:
        results = engine.predict(crop_bgr)
        res = next(iter(results), None)
        if res is not None and hasattr(res, "get"):
            lines = _parse_predict_result(res, x1, y1, x2, y2)
    except Exception as exc_predict:
        logger.warning("PaddleOCR predict() call failed: %s", exc_predict)
        try:
            results = engine.ocr(crop_rgb)
            lines = _parse_legacy_result(results, x1, y1, x2, y2)
        except Exception as exc_ocr:
            logger.warning("OCR legacy fallback also failed: %s", exc_ocr)

    lines.sort(key=lambda item: item["box"][1])
    full_text = "\n".join(item["text"] for item in lines)

    if not lines:
        lines.append(
            {
                "text": "",
                "box": (x1, y1, x2, y2),
                "score": 0.0,
            }
        )

    return full_text, lines
