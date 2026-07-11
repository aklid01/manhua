"""Stage 2: OCR.

PaddleOCR over detected regions. v0 = horizontal text only.
Records original Chinese + confidence; flags needs_correction below threshold.
"""

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

from manhua_pipeline.io.workspace import load_manifest, save_manifest
from manhua_pipeline.logging_setup import get_logger, log_stage

logger = get_logger(__name__)

_STAGE_INDEX = 2
_TOTAL_STAGES = 7
_STAGE_NAME = "OCR"


def _get_ocr(config):
    """Lazy-load and initialize PaddleOCR engine once."""
    import logging

    logging.getLogger("ppocr").setLevel(logging.WARNING)

    from paddleocr import PaddleOCR

    device = "gpu" if getattr(config, "OCR_USE_GPU", False) else "cpu"
    return PaddleOCR(
        lang=getattr(config, "OCR_LANG", "ch"),
        device=device,
        enable_mkldnn=False,
    )


def _read_crop(ocr_engine, crop_image, config) -> tuple:
    """Run PaddleOCR on a PIL image crop and return normalized text, confidence, and watermark flag.

    Returns:
        tuple: (original_text string, mean_confidence float, min_confidence float, watermark_filtered bool)
    """
    crop_bgr = np.array(crop_image.convert("RGB"))[:, :, ::-1]
    results = ocr_engine.predict(crop_bgr)

    if not results or not isinstance(results, list):
        return "", 0.0, 0.0, False

    res = results[0]
    texts = res.get("rec_texts", [])
    scores = res.get("rec_scores", [])

    if not texts:
        return "", 0.0, 0.0, False

    lines = []
    confidences = []
    filtered_any = False
    for txt, conf in zip(texts, scores):
        if not txt:
            continue

        # Check watermark
        is_wm = False
        for rx in getattr(config, "WATERMARK_REGEX", []):
            if rx.search(txt):
                is_wm = True
                break
        if is_wm:
            filtered_any = True
            continue

        lines.append(txt)
        confidences.append(float(conf))

    if not lines:
        return "", 0.0, 0.0, filtered_any

    text = "\n".join(lines)
    mean_conf = sum(confidences) / len(confidences)
    min_conf = min(confidences)
    return text, mean_conf, min_conf, filtered_any


def _ocr_region(region: dict, page: dict, ocr_engine, config, ws: Path) -> dict:
    """Run OCR on a single region and return the mapped result dictionary."""
    page_num = region["page_number"]
    img_path = ws / config.STAGE_FOLDERS["pages"] / page["filename"]
    if not img_path.exists():
        raise FileNotFoundError(f"Page file not found: {img_path}")

    # Load page size
    with Image.open(img_path) as page_im:
        W, H = page_im.size
        # Clamp bbox coordinates
        x, y, w, h = (
            region["read_region"]["x"],
            region["read_region"]["y"],
            region["read_region"]["w"],
            region["read_region"]["h"],
        )
        x0 = max(0, x)
        y0 = max(0, y)
        x1 = min(W, x + w)
        y1 = min(H, y + h)

        if x1 <= x0 or y1 <= y0:
            original_text, mean_conf, min_conf, watermark_filtered = (
                "",
                0.0,
                0.0,
                False,
            )
        else:
            crop_im = page_im.crop((x0, y0, x1, y1))
            original_text, mean_conf, min_conf, watermark_filtered = _read_crop(
                ocr_engine, crop_im, config
            )

    # Edge touching computation
    eps = config.EDGE_TOUCH_EPS
    touches_top = y <= eps
    touches_bottom = y + h >= H - eps
    edge_touching = touches_top or touches_bottom

    if touches_top and touches_bottom:
        edge = "both"
    elif touches_top:
        edge = "top"
    elif touches_bottom:
        edge = "bottom"
    else:
        edge = "none"

    has_usable_text = (original_text.strip() != "") and (
        mean_conf >= config.OCR_MIN_TEXT_CONF
    )
    needs_correction = (mean_conf < config.OCR_CONFIDENCE_THRESHOLD) or (
        edge_touching and not has_usable_text
    )

    notes = []
    if edge_touching and not has_usable_text:
        notes.append("possible split bubble; low/no text — render should not erase")
    if watermark_filtered and not has_usable_text:
        notes.append("watermark-only region; not rendered")

    note = "; ".join(notes) if notes else None

    return {
        "region_id": region["region_id"],
        "page_number": page_num,
        "type": region["type"],
        "original_text": original_text,
        "text_direction": "horizontal",
        "ocr_confidence": mean_conf,
        "ocr_confidence_min": min_conf,
        "has_usable_text": has_usable_text,
        "do_not_render": False,
        "needs_correction": needs_correction,
        "edge_touching": edge_touching,
        "edge": edge,
        "note": note,
        "watermark_filtered": watermark_filtered,
    }


def _process_single_region_ocr(
    region: dict, page: dict, ocr_engine, config, ws: Path
) -> tuple:
    """Run OCR for a single region, log progress details, handle exceptions gracefully.

    Returns:
        tuple: (result_dict, is_warning_bool)
    """
    page_num = region["page_number"]
    try:
        res = _ocr_region(region, page, ocr_engine, config, ws)
        # Log details per read region
        if res["needs_correction"] and not (
            res["edge_touching"] and not res["has_usable_text"]
        ):
            logger.info(
                "[%d/%d %s] Page %03d %s -> low confidence %.2f (needs_correction) ⚠",
                _STAGE_INDEX,
                _TOTAL_STAGES,
                _STAGE_NAME,
                page_num,
                res["region_id"],
                res["ocr_confidence"],
            )
        elif res["edge_touching"] and not res["has_usable_text"]:
            logger.info(
                "[%d/%d %s] Page %03d %s -> edge_touching(%s), no usable text — possible split ⚠",
                _STAGE_INDEX,
                _TOTAL_STAGES,
                _STAGE_NAME,
                page_num,
                res["region_id"],
                res["edge"],
            )
        else:
            snippet = res["original_text"].replace("\n", " ")
            if len(snippet) > 15:
                snippet = snippet[:15] + "..."
            logger.info(
                "[%d/%d %s] Page %03d %s -> %r (conf %.2f)",
                _STAGE_INDEX,
                _TOTAL_STAGES,
                _STAGE_NAME,
                page_num,
                res["region_id"],
                snippet,
                res["ocr_confidence"],
            )
        return res, False
    except Exception as exc:
        logger.warning(
            "[%s] Region %s — failed OCR: %s",
            _STAGE_NAME,
            region["region_id"],
            exc,
        )
        # Error isolation entry
        fallback = {
            "region_id": region["region_id"],
            "page_number": page_num,
            "type": region["type"],
            "original_text": "",
            "text_direction": "horizontal",
            "ocr_confidence": 0.0,
            "ocr_confidence_min": 0.0,
            "has_usable_text": False,
            "do_not_render": False,
            "needs_correction": True,
            "edge_touching": False,
            "edge": "none",
            "note": "ocr_error",
        }
        return fallback, True


def run_ocr(workspace: str, config) -> Path:
    """Run OCR over all detected regions in the workspace."""
    t0 = time.monotonic()
    ws = Path(workspace)
    log_stage(logger, _STAGE_INDEX, _TOTAL_STAGES, _STAGE_NAME, "starting")

    manifest = load_manifest(workspace, config)
    if not manifest:
        raise ValueError("Manifest not found. Run import first.")

    det_path = ws / config.STAGE_FOLDERS["detection"] / "detection.json"
    if not det_path.exists():
        raise FileNotFoundError("detection.json not found. Run detect first.")

    with det_path.open("r", encoding="utf-8") as fh:
        detection = json.load(fh)

    ocr_dir = ws / config.STAGE_FOLDERS["ocr"]
    ocr_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "[%d/%d %s] Initializing PaddleOCR (lang=%s, gpu=%s)",
        _STAGE_INDEX,
        _TOTAL_STAGES,
        _STAGE_NAME,
        config.OCR_LANG,
        config.OCR_USE_GPU,
    )
    ocr_engine = _get_ocr(config)

    results = []
    warnings = 0
    needs_correction_count = 0
    edge_touching_count = 0

    # Group by page for clear page-based logging
    pages_map = {p["page_number"]: p for p in manifest.get("pages", [])}

    for region in detection.get("regions", []):
        page_num = region["page_number"]
        page = pages_map.get(page_num)

        if not page or page.get("skip") or page.get("filename") is None:
            logger.warning(
                "[%s] Region %s references skipped/missing page %s",
                _STAGE_NAME,
                region["region_id"],
                page_num,
            )
            continue

        res, is_warning = _process_single_region_ocr(
            region, page, ocr_engine, config, ws
        )
        results.append(res)

        if is_warning:
            warnings += 1
        if res["needs_correction"]:
            needs_correction_count += 1
        if res["edge_touching"]:
            edge_touching_count += 1

    # Output OCR JSON
    now = datetime.now(timezone.utc).isoformat()
    output_json = {
        "chapter_id": manifest.get("chapter_id", "unknown_chapter"),
        "stage": "ocr",
        "generated_at": now,
        "ocr_engine": config.OCR_ENGINE,
        "results": results,
    }
    ocr_json_path = ocr_dir / "ocr.json"

    with ocr_json_path.open("w", encoding="utf-8") as fh:
        json.dump(output_json, fh, ensure_ascii=False, indent=2)

    # Update Manifest
    completed = manifest.get("completed_stages", [])
    if "ocr" not in completed:
        completed.append("ocr")
    manifest["completed_stages"] = completed
    manifest["current_stage"] = "translate"
    manifest["updated_at"] = now
    save_manifest(workspace, config, manifest)

    elapsed = time.monotonic() - t0
    log_stage(
        logger,
        _STAGE_INDEX,
        _TOTAL_STAGES,
        _STAGE_NAME,
        f"done: {len(results)} regions OCR'd, {needs_correction_count} need correction, "
        f"{edge_touching_count} edge-touching, {warnings} warnings -> {ocr_json_path} (elapsed {elapsed:.1f}s)",
    )
    return ocr_json_path
