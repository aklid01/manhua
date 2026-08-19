# Stage 2: PaddleOCR text extraction over detected dialogue and narration regions.

import atexit
import gc
import json
import time
from datetime import datetime, timezone
from importlib.metadata import version
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


# Clean shutdown and resource release for PaddleOCR engine
def _close_ocr():
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
        gc.collect()
    except Exception:
        pass


atexit.register(_close_ocr)


# Lazy-load PaddleOCR engine with device fallback handling
def _get_ocr(config):
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


# Generate escalating image preprocessing variants for OCR retry passes
def _preprocess_variant(crop_im: "Image.Image", attempt: int) -> "Image.Image":
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


# Read image crop with OCR and retry with higher preprocessing on low confidence
def _read_best(ocr_engine, crop_im, config) -> tuple:
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


# Run OCR on image crop and filter watermark lines
def _read_crop(ocr_engine, crop_image, config) -> tuple:
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
    watermark_regex = getattr(config, "WATERMARK_REGEX", [])
    lines, confidences, filtered_any = [], [], False
    for index, txt in enumerate(texts):
        if txt is None:
            continue
        txt = str(txt)
        if not txt.strip():
            continue
        conf = float(scores[index]) if index < len(scores) else 0.0
        is_wm = False
        for rx in watermark_regex:
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


# Check whether text contains CJK characters
def _has_cjk(text: str) -> bool:
    return any(
        ("\u3400" <= ch <= "\u9fff") or ("\uf900" <= ch <= "\ufaff")
        for ch in (text or "")
    )


# Check if text contains only punctuation or whitespace
def _is_punct_or_empty(text: str) -> bool:
    core = "".join(ch for ch in (text or "") if not ch.isspace())
    if not core:
        return True
    return not any(ch.isalnum() or _has_cjk(ch) for ch in core)


# Return region types eligible for manual OCR review
def _review_types(config) -> tuple:
    return tuple(getattr(config, "OCR_REVIEW_TYPES", ("speech_bubble", "narration")))


# Execute preprocessing escalation unconditionally on suspicious crops
def _read_forced_ladder(ocr_engine, crop_im, config):
    best = None
    best_score = -1.0
    for attempt in range(1, getattr(config, "OCR_RETRY_MAX", 2) + 1):
        cand = _read_crop(ocr_engine, _preprocess_variant(crop_im, attempt), config)
        has_real = 0 if _is_punct_or_empty(cand[0]) else 1
        score = has_real * 10 + float(cand[1])
        if score > best_score:
            best, best_score = cand, score
    return best


# Save cropped PNG of flagged bubble for user review
def _dump_review_crop(ws, ocr_dir, config, region_det, pages_map):
    page = pages_map.get(region_det.get("page_number"))
    if not page or not page.get("filename"):
        return None
    img_path = ws / config.STAGE_FOLDERS["pages"] / page["filename"]
    if not img_path.exists():
        return None
    box = region_det.get("read_region") or region_det.get("bbox")
    if not box:
        return None
    crops_dir = ocr_dir / "review_crops"
    crops_dir.mkdir(parents=True, exist_ok=True)
    out = crops_dir / f"{region_det['region_id']}.png"
    try:
        with Image.open(img_path) as im:
            W, H = im.size
            x, y, w, h = box["x"], box["y"], box["w"], box["h"]
            im.crop((max(0, x), max(0, y), min(W, x + w), min(H, y + h))).save(
                out, "PNG"
            )
    except Exception:
        return None
    return f"{config.STAGE_FOLDERS['ocr']}/review_crops/{out.name}"


# Write ocr_review.json containing flagged low-confidence regions
def _write_ocr_review(ws, ocr_dir, config, results, pages_map, region_by_id, prior):
    if not getattr(config, "OCR_REVIEW_ENABLED", True):
        return 0
    flagged = [r for r in results if r.get("ocr_review")]
    if not flagged:
        return 0
    entries = {}
    for r in flagged:
        rid = r["region_id"]
        prior_entry = prior.get(rid, {}) if isinstance(prior, dict) else {}
        crop_rel = None
        if getattr(config, "OCR_REVIEW_DUMP_CROPS", True):
            det = region_by_id.get(rid)
            if det is not None:
                det = dict(det)
                det.setdefault("page_number", r.get("page_number"))
                crop_rel = _dump_review_crop(ws, ocr_dir, config, det, pages_map)
        entries[rid] = {
            "page": r.get("page_number"),
            "ocr_text": r.get("original_text", ""),
            "reason": r.get("ocr_review_reason"),
            "crop": crop_rel,
            "corrected_text": (
                prior_entry.get("corrected_text", "")
                if isinstance(prior_entry, dict)
                else ""
            ),
        }
    payload = {
        "_README": (
            "OCR flagged these dialogue regions as likely dropped-character "
            "failures. Open the matching image under 'crop', type the Chinese "
            "you see into 'corrected_text', then re-run OCR. Leave blank to "
            "accept the rendered fallback. Non-blocking."
        ),
        "regions": entries,
    }
    review_path = ocr_dir / "ocr_review.json"
    with review_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    logger.warning(
        "[%s] %d region(s) flagged for OCR review -> %s "
        "(non-blocking; fill corrected_text and re-run).",
        _STAGE_NAME,
        len(flagged),
        review_path,
    )
    return len(flagged)


# Execute OCR extraction and calculate geometry flags on single region crop
def _ocr_region(
    region: dict, page: dict, page_im: "Image.Image", ocr_engine, config
) -> dict:
    page_num = region["page_number"]
    W, H = page_im.size
    ocr_review_enabled = getattr(config, "OCR_REVIEW_ENABLED", True)
    review_types = _review_types(config)

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
        if (
            ocr_review_enabled
            and region.get("type") in review_types
            and not watermark_filtered
            and _is_punct_or_empty(original_text)
        ):
            forced = _read_forced_ladder(ocr_engine, crop_im, config)
            if forced is not None and not _is_punct_or_empty(forced[0]):
                original_text, mean_conf, min_conf, watermark_filtered = forced
        if not original_text.strip():
            if watermark_filtered:
                status = "watermark_only"
            else:
                status = "no_prediction" if mean_conf == 0.0 else "no_text"

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

    ocr_review = (
        ocr_review_enabled
        and region.get("type") in review_types
        and not watermark_filtered
        and not edge_touching
        and _is_punct_or_empty(original_text)
    )
    ocr_review_reason = None
    if ocr_review:
        ocr_review_reason = "empty" if not original_text.strip() else "punct_only"

    notes = []
    if edge_touching and not has_usable_text:
        notes.append("possible split bubble; low/no text - render should not erase")
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
        "ocr_review": ocr_review,
        "ocr_review_reason": ocr_review_reason,
    }


# Process single region OCR with error isolation and logging
def _process_single_region_ocr(
    region: dict, page: dict, page_im: "Image.Image", ocr_engine, config
) -> tuple:
    page_num = region["page_number"]
    try:
        res = _ocr_region(region, page, page_im, ocr_engine, config)
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
                "[%d/%d %s] Page %03d %s -> edge_touching(%s), no usable text - possible split [WARNING]",
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
            "[%s] Region %s - failed OCR",
            _STAGE_NAME,
            region["region_id"],
        )
        status = "schema_error" if isinstance(exc, TypeError) else "inference_error"
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
            "ocr_review": False,
            "ocr_review_reason": None,
        }
        return fallback, True


# Execute full chapter OCR stage across all detected regions
def run_ocr(workspace: str, config) -> Path:
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
    from manhua_pipeline.events import STAGE_COMPLETED, STAGE_STARTED
    from manhua_pipeline.io.event_emitter import emit_stage_event

    emit_stage_event(ws, config, _STAGE_NAME.lower(), STAGE_STARTED, logger=logger)

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

    review_path = ocr_dir / "ocr_review.json"
    prior_review = {}
    ocr_corrections = {}
    if review_path.exists():
        try:
            with review_path.open("r", encoding="utf-8") as fh:
                _raw = json.load(fh)
            prior_review = _raw.get("regions", _raw) if isinstance(_raw, dict) else {}
            for _rid, _entry in prior_review.items():
                _ct = (_entry or {}).get("corrected_text", "")
                if isinstance(_ct, str) and _ct.strip():
                    ocr_corrections[_rid] = _ct.strip()
        except Exception:
            logger.warning(
                "[%s] Could not parse %s; ignoring corrections.",
                _STAGE_NAME,
                review_path,
            )

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

    pages_map = {p["page_number"]: p for p in manifest.get("pages", [])}
    region_by_id = {r["region_id"]: r for r in detection.get("regions", [])}
    skip_free_text = getattr(config, "RTDETR_SKIP_TEXT_FREE", False)

    regions_by_page: dict[int, list] = {}
    for region in detection.get("regions", []):
        regions_by_page.setdefault(region["page_number"], []).append(region)

    for page_num, page_regions in sorted(regions_by_page.items()):
        page = pages_map.get(page_num)
        if not page or page.get("skip") or page.get("filename") is None:
            for region in page_regions:
                logger.warning(
                    "[%s] Region %s references skipped/missing page %s",
                    _STAGE_NAME,
                    region["region_id"],
                    page_num,
                )
            continue

        img_path = ws / config.STAGE_FOLDERS["pages"] / page["filename"]
        with Image.open(img_path) as page_im:
            for region in page_regions:
                if skip_free_text and region.get("is_free_text"):
                    continue
                res, is_warning = _process_single_region_ocr(
                    region, page, page_im, ocr_engine, config
                )
                if res["region_id"] in ocr_corrections:
                    res["original_text"] = ocr_corrections[res["region_id"]]
                    res["status"] = "manual_review"
                    res["ocr_source"] = "manual_review"
                    res["ocr_review"] = False
                    res["ocr_review_reason"] = None
                    res["has_usable_text"] = True
                    res["needs_correction"] = False
                    is_warning = False
                    logger.info(
                        "[%s] %s: applied manual OCR correction.",
                        _STAGE_NAME,
                        res["region_id"],
                    )
                results.append(res)

                if is_warning:
                    warnings += 1
                if res["needs_correction"]:
                    needs_correction_count += 1
                if res["edge_touching"]:
                    edge_touching_count += 1

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

    completed = manifest.get("completed_stages", [])
    if "ocr" not in completed:
        completed.append("ocr")
    manifest["completed_stages"] = completed
    _write_ocr_review(
        ws, ocr_dir, config, results, pages_map, region_by_id, prior_review
    )

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
    emit_stage_event(
        ws,
        config,
        _STAGE_NAME.lower(),
        STAGE_COMPLETED,
        logger=logger,
        metadata={
            "total_regions": len(results),
            "needs_correction": needs_correction_count,
        },
    )
    _close_ocr()
    return ocr_json_path
