"""Stage 2: OCR.

PaddleOCR over detected regions. v0 = horizontal text only.
Records original Chinese + confidence; flags needs_correction below threshold.
"""

import atexit
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter, ImageOps

from manhua_pipeline.io.workspace import load_manifest, save_manifest
from manhua_pipeline.logging_setup import get_logger, log_stage

logger = get_logger(__name__)

_STAGE_INDEX = 2
_TOTAL_STAGES = 7
_STAGE_NAME = "OCR"


_OCR_ENGINE = None
_ACTIVE_OCR_ENGINE = None


def _close_ocr():
    """Best-effort shutdown of the PaddleOCR/PaddleX engine and any worker it spawned."""
    global _OCR_ENGINE, _ACTIVE_OCR_ENGINE
    if _OCR_ENGINE is None:
        return
    engine = _OCR_ENGINE
    _OCR_ENGINE = None
    _ACTIVE_OCR_ENGINE = None
    for attr in ("close", "shutdown", "release", "__del__"):
        fn = getattr(engine, attr, None)
        if callable(fn):
            try:
                fn()
            except Exception:
                pass
    for attr in ("paddlex_pipeline", "_pipeline", "pipeline"):
        inner = getattr(engine, attr, None)
        if inner is not None:
            for m in ("close", "shutdown"):
                fn = getattr(inner, m, None)
                if callable(fn):
                    try:
                        fn()
                    except Exception:
                        pass
    try:
        import gc

        gc.collect()
    except Exception:
        pass


atexit.register(_close_ocr)


def _get_ocr(config):
    """Lazy-load and initialize PaddleOCR engine once (per process) with fallback."""
    global _OCR_ENGINE, _ACTIVE_OCR_ENGINE
    if _OCR_ENGINE is not None:
        return _OCR_ENGINE

    import logging

    for _name in ["ppocr", "paddlex", "paddle", "transformers", "huggingface_hub"]:
        logging.getLogger(_name).setLevel(logging.ERROR)

    from paddleocr import PaddleOCR

    requested_gpu = getattr(config, "OCR_USE_GPU", False)
    gpu_available = False
    if requested_gpu:
        try:
            import torch

            gpu_available = torch.cuda.is_available()
        except Exception:
            pass
        if not gpu_available:
            try:
                import paddle

                gpu_available = (
                    paddle.is_compiled_with_cuda()
                    and paddle.device.cuda.device_count() > 0
                )
            except Exception:
                pass
        if not gpu_available:
            logger.warning(
                "[%s] OCR_USE_GPU=True requested, but CUDA GPU support is not available. Falling back to device='cpu'.",
                _STAGE_NAME,
            )

    device = "gpu" if (requested_gpu and gpu_available) else "cpu"
    preferred_engine = getattr(config, "OCR_ENGINE", "paddle")
    if preferred_engine == "PaddleOCR":
        preferred_engine = "paddle"

    fallback_engine = "paddle" if preferred_engine == "transformers" else "transformers"

    try:
        logger.info(
            "[%s] Initializing PaddleOCR with preferred engine: %s (device=%s)",
            _STAGE_NAME,
            preferred_engine,
            device,
        )
        _OCR_ENGINE = PaddleOCR(
            lang=getattr(config, "OCR_LANG", "ch"),
            ocr_version=getattr(config, "OCR_VERSION", "PP-OCRv6"),
            device=device,
            engine=preferred_engine,
            enable_mkldnn=False,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
        )
        _ACTIVE_OCR_ENGINE = preferred_engine
        return _OCR_ENGINE
    except Exception as exc:
        logger.warning(
            "[%s] Failed to initialize preferred OCR engine %s: %s. Falling back to %s.",
            _STAGE_NAME,
            preferred_engine,
            exc,
            fallback_engine,
        )

    _OCR_ENGINE = PaddleOCR(
        lang=getattr(config, "OCR_LANG", "ch"),
        ocr_version=getattr(config, "OCR_VERSION", "PP-OCRv6"),
        device=device,
        engine=fallback_engine,
        enable_mkldnn=False,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
    )
    _ACTIVE_OCR_ENGINE = fallback_engine
    return _OCR_ENGINE


def _preprocess_variant(crop_im: "Image.Image", attempt: int) -> "Image.Image":
    """Escalating-preprocessing variant. attempt 0 = current (2x LANCZOS)."""
    if attempt <= 0:
        return crop_im.resize(
            (crop_im.width * 2, crop_im.height * 2), Image.Resampling.LANCZOS
        )
    if attempt == 1:
        g = ImageOps.grayscale(crop_im)
        g = g.resize((g.width * 3, g.height * 3), Image.Resampling.LANCZOS)
        return ImageOps.autocontrast(g)
    g = ImageOps.grayscale(crop_im)
    g = g.resize((g.width * 3, g.height * 3), Image.Resampling.LANCZOS)
    g = ImageOps.autocontrast(g)
    g = g.filter(ImageFilter.MedianFilter(size=3))
    return g.point(lambda p: 255 if p > 140 else 0)


def _read_best(ocr_engine, crop_im, config) -> tuple:
    """Base read; if confidence is in the retry window, escalate preprocessing and
    keep the highest-confidence attempt."""
    best = _read_crop(ocr_engine, _preprocess_variant(crop_im, 0), config)
    best_mean = best[1]

    if not getattr(config, "OCR_RETRY_ENABLED", False):
        return best

    floor = getattr(config, "OCR_RETRY_FLOOR", 0.30)
    ceil = getattr(config, "OCR_CONFIDENCE_THRESHOLD", 0.70)
    if not (floor <= best_mean < ceil):
        return best

    for attempt in range(1, getattr(config, "OCR_RETRY_MAX", 2) + 1):
        cand = _read_crop(ocr_engine, _preprocess_variant(crop_im, attempt), config)
        if cand[1] > best_mean:
            best, best_mean = cand, cand[1]
            logger.info(
                "[%s] OCR retry #%d improved confidence -> %.2f",
                _STAGE_NAME,
                attempt,
                best_mean,
            )
        if best_mean >= ceil:
            break
    return best


def _read_crop(ocr_engine, crop_image, config) -> tuple:
    """Run PaddleOCR on a PIL image crop and return normalized text, confidence, and watermark flag.

    Returns:
        tuple: (original_text string, mean_confidence float, min_confidence float, watermark_filtered bool)
    """
    crop_bgr = np.ascontiguousarray(np.array(crop_image.convert("RGB"))[:, :, ::-1])
    results = ocr_engine.predict(crop_bgr)
    res = next(iter(results), None)
    if res is None:
        return "", 0.0, 0.0, False
    if not hasattr(res, "get"):
        raise TypeError(f"Unexpected PaddleOCR result type: {type(res).__name__}")
    texts = list(res.get("rec_texts", []) or [])
    scores = list(res.get("rec_scores", []) or [])
    if not texts:
        return "", 0.0, 0.0, False
    if len(texts) != len(scores):
        logger.warning(
            "[%s] PaddleOCR returned %d texts but %d scores",
            _STAGE_NAME,
            len(texts),
            len(scores),
        )
    lines, confidences, filtered_any = [], [], False
    for index, txt in enumerate(texts):
        if txt is None:
            continue
        txt = str(txt)
        if not txt.strip():
            continue
        conf = float(scores[index]) if index < len(scores) else 0.0
        is_wm = False
        for rx in getattr(config, "WATERMARK_REGEX", []):
            if rx.search(txt):
                is_wm = True
                break
        if is_wm:
            filtered_any = True
            continue
        lines.append(txt)
        confidences.append(conf)
    if not lines:
        return "", 0.0, 0.0, filtered_any
    return (
        "\n".join(lines),
        sum(confidences) / len(confidences),
        min(confidences),
        filtered_any,
    )


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

        status = "success"
        if x1 <= x0 or y1 <= y0:
            original_text, mean_conf, min_conf, watermark_filtered = (
                "",
                0.0,
                0.0,
                False,
            )
            status = "no_text"
        else:
            crop_im = page_im.crop((x0, y0, x1, y1))
            original_text, mean_conf, min_conf, watermark_filtered = _read_best(
                ocr_engine, crop_im, config
            )
            if not original_text.strip():
                if watermark_filtered:
                    status = "watermark_only"
                else:
                    status = "no_prediction" if mean_conf == 0.0 else "no_text"

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
        "status": status,
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
                "[%d/%d %s] Page %03d %s -> low confidence %.2f (needs_correction) [WARNING]",
                _STAGE_INDEX,
                _TOTAL_STAGES,
                _STAGE_NAME,
                page_num,
                res["region_id"],
                res["ocr_confidence"],
            )
        elif res["edge_touching"] and not res["has_usable_text"]:
            logger.info(
                "[%d/%d %s] Page %03d %s -> edge_touching(%s), no usable text — possible split [WARNING]",
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
        logger.exception(
            "[%s] Region %s — failed OCR",
            _STAGE_NAME,
            region["region_id"],
        )
        status = "schema_error" if isinstance(exc, TypeError) else "inference_error"
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
            "watermark_filtered": False,
            "status": status,
        }
        return fallback, True


def run_ocr(workspace: str, config) -> Path:
    """Run OCR over all detected regions in the workspace."""
    t0 = time.monotonic()
    ws = Path(workspace)
    logger.info(
        "[%d/%d %s] Series: %s | Chapter: %s",
        _STAGE_INDEX,
        _TOTAL_STAGES,
        _STAGE_NAME,
        ws.parent.as_posix(),
        ws.name,
    )
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
        if getattr(config, "RTDETR_SKIP_TEXT_FREE", False) and region.get("is_free_text"):
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
    from importlib.metadata import version

    try:
        paddleocr_version = version("paddleocr")
    except Exception:
        paddleocr_version = "not_installed"

    now = datetime.now(timezone.utc).isoformat()
    output_json = {
        "chapter_id": manifest.get("chapter_id", "unknown_chapter"),
        "stage": "ocr",
        "generated_at": now,
        "ocr_engine": _ACTIVE_OCR_ENGINE or config.OCR_ENGINE,
        "ocr_version": getattr(config, "OCR_VERSION", None),
        "ocr_language": getattr(config, "OCR_LANG", "ch"),
        "ocr_device": "gpu" if getattr(config, "OCR_USE_GPU", False) else "cpu",
        "paddleocr_package_version": paddleocr_version,
        "results": results,
    }
    # Check if all regions failed (safety net)
    if results:
        if warnings == len(results):
            raise RuntimeError(
                "PaddleOCR failed for every processed region; OCR output not accepted."
            )
        if len(results) > 1:
            non_wm_results = [r for r in results if r.get("status") != "watermark_only"]
            if non_wm_results and all(
                r.get("status") in {"no_prediction", "schema_error", "inference_error"}
                for r in non_wm_results
            ):
                raise RuntimeError(
                    "PaddleOCR returned no text for any region; OCR output not accepted."
                )

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
    _close_ocr()
    return ocr_json_path
