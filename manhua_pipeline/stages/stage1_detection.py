"""Stage 1: Detection.

YOLO bubble detection. v0 = speech_bubble + narration only.
Model: ogkalu/comic-speech-bubble-detector-yolov8m.
Outputs detection.json and optional visual debug overlays.
"""

import json
import time
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageDraw
from ultralytics import YOLO

from manhua_pipeline.io.workspace import load_manifest, save_manifest
from manhua_pipeline.logging_setup import get_logger, log_page, log_stage

logger = get_logger(__name__)

_STAGE_INDEX = 1
_TOTAL_STAGES = 7
_STAGE_NAME = "Detection"

_RTDETR_MODEL = None
_RTDETR_PROCESSOR = None


def _get_rtdetr(config):
    """Lazy-load the RT-DETR detector once per process."""
    global _RTDETR_MODEL, _RTDETR_PROCESSOR
    if _RTDETR_MODEL is not None:
        return _RTDETR_MODEL, _RTDETR_PROCESSOR
    import logging

    import torch
    from transformers import AutoImageProcessor, AutoModelForObjectDetection
    from transformers.utils import logging as hf_logging

    hf_logging.set_verbosity_error()
    logging.getLogger("huggingface_hub").setLevel(logging.WARNING)

    repo = getattr(config, "RTDETR_REPO", "ogkalu/comic-text-and-bubble-detector")
    _RTDETR_PROCESSOR = AutoImageProcessor.from_pretrained(repo)
    _RTDETR_MODEL = AutoModelForObjectDetection.from_pretrained(repo)
    _RTDETR_MODEL.eval()
    dev = "cuda" if torch.cuda.is_available() and getattr(config, "DETECTOR_USE_GPU", False) else "cpu"
    _RTDETR_MODEL.to(dev)
    logger.info("[%s] RT-DETR loaded (%s, cached)", _STAGE_NAME, repo)
    return _RTDETR_MODEL, _RTDETR_PROCESSOR


def _mk_region(config, page_num, idx, rec, *, type_, parent_bubble, is_free_text) -> dict:
    """Build a region dict matching the schema downstream expects."""
    x, y, w, h = rec["x"], rec["y"], rec["w"], rec["h"]
    style_hint = "narration" if type_ == config.TYPE_NARRATION else "round"
    
    return {
        "region_id": config.REGION_ID_FORMAT.format(page=page_num, idx=idx),
        "page_number": page_num,
        "type": type_,
        "bbox": {"x": x, "y": y, "w": w, "h": h},
        "reading_order": idx,
        "style_hint": style_hint,
        "confidence": rec.get("conf", 0.0),
        "read_region": {"x": x, "y": y, "w": w, "h": h},
        "erase_mask": {
            "type": "rect",
            "coords": [x, y, w, h],
        },
        "render": not is_free_text,
        "is_free_text": is_free_text,
        "parent_bubble": (
            {"x": parent_bubble["x"], "y": parent_bubble["y"],
             "w": parent_bubble["w"], "h": parent_bubble["h"]}
            if parent_bubble else None
        ),
        "text_direction": "horizontal",
    }


def _draw_debug_overlay_rtdetr(
    img_path: Path, sorted_boxes: list, bubbles: list, overlays_dir: Path, page_num: int
) -> None:
    """Draw bounding boxes and labels for RT-DETR (bubbles in green, speech regions in red, narration in blue)."""
    with Image.open(img_path) as im:
        overlay = im.copy()
        draw = ImageDraw.Draw(overlay)
        
        # 1. Draw parent bubble containers (green)
        for b in bubbles:
            bx, by, bw, bh = b["x"], b["y"], b["w"], b["h"]
            draw.rectangle([bx, by, bx + bw, by + bh], outline=(0, 255, 0), width=2)
            draw.text((bx + 5, by + 5), f"bubble ({b['conf']:.2f})", fill=(0, 255, 0))
            
        # 2. Draw sorted detected regions (speech/narration)
        for box in sorted_boxes:
            bx, by, bw, bh = box["bbox"]["x"], box["bbox"]["y"], box["bbox"]["w"], box["bbox"]["h"]
            if box["type"] == "speech_bubble":
                color = (255, 0, 0)  # Red for speech
                label = f"{box['region_id']} ({box['confidence']:.2f})"
                if box.get("parent_bubble"):
                    pb = box["parent_bubble"]
                    px, py = pb["x"] + pb["w"] // 2, pb["y"] + pb["h"] // 2
                    tx, ty = bx + bw // 2, by + bh // 2
                    draw.line([(tx, ty), (px, py)], fill=(255, 255, 0), width=1)
            else:
                color = (0, 0, 255)  # Blue for narration/SFX
                label = f"{box['region_id']} (SFX, {box['confidence']:.2f})"
            
            draw.rectangle([bx, by, bx + bw, by + bh], outline=color, width=3)
            draw.text((bx + 5, by + 18 if box["type"] == "speech_bubble" else by + 5), label, fill=color)

        dest_overlay = overlays_dir / f"{page_num:03d}_overlay.png"
        overlay.save(dest_overlay)


def _detect_page_regions_rtdetr(page, config, ws, overlays_dir) -> list:
    """RT-DETR detection for one page. Emits sorted regions mapping text_bubble (speech)
    and text_free (narration), maintaining parent bubble associations."""
    import torch
    from PIL import Image

    model, processor = _get_rtdetr(config)
    page_num = page["page_number"]
    img_path = ws / config.STAGE_FOLDERS["pages"] / page["filename"]
    if not img_path.exists():
        raise FileNotFoundError(f"Page file not found: {img_path}")

    image = Image.open(img_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(**inputs)
    target = torch.tensor([image.size[::-1]]).to(model.device)  # (H, W)
    det = processor.post_process_object_detection(
        outputs, target_sizes=target, threshold=getattr(config, "RTDETR_CONF", 0.30)
    )[0]

    C_BUB = getattr(config, "RTDETR_CLASS_BUBBLE", 0)
    C_TXT = getattr(config, "RTDETR_CLASS_TEXT_BUBBLE", 1)
    C_FREE = getattr(config, "RTDETR_CLASS_TEXT_FREE", 2)

    bubbles, raw_boxes = [], []
    for score, label, box in zip(det["scores"], det["labels"], det["boxes"]):
        x0, y0, x1, y1 = [int(v) for v in box.cpu().tolist()]
        rec = {"x": x0, "y": y0, "w": x1 - x0, "h": y1 - y0, "conf": float(score.item())}
        cid = int(label.item())
        
        if cid == C_BUB:
            bubbles.append(rec)
        elif cid == C_TXT:
            rec["type"] = config.TYPE_SPEECH
            rec["is_free_text"] = False
            raw_boxes.append(rec)
        elif cid == C_FREE:
            rec["type"] = config.TYPE_NARRATION
            rec["is_free_text"] = True
            raw_boxes.append(rec)

    def _parent_bubble(t):
        tx, ty, tw, th = t["x"], t["y"], t["w"], t["h"]
        best, best_area = None, 0
        for b in bubbles:
            ix = max(0, min(tx + tw, b["x"] + b["w"]) - max(tx, b["x"]))
            iy = max(0, min(ty + th, b["y"] + b["h"]) - max(ty, b["y"]))
            area = ix * iy
            if area > best_area:
                best, best_area = b, area
        return best

    # Sort all speech and narration regions in reading order
    page_height = image.height
    band = page_height * config.READING_ORDER_BAND_FRACTION
    sorted_boxes = _reading_order_sort(raw_boxes, band)

    regions = []
    for idx, box in enumerate(sorted_boxes, start=1):
        parent = _parent_bubble(box) if box["type"] == config.TYPE_SPEECH else None
        regions.append(_mk_region(
            config, page_num, idx, box, type_=box["type"],
            parent_bubble=parent, is_free_text=box["is_free_text"]
        ))

    if getattr(config, "OVERLAY_ENABLED", True):
        _draw_debug_overlay_rtdetr(img_path, regions, bubbles, overlays_dir, page_num)

    return regions


def _iter_boxes(result):
    """Decoupled helper to iterate boxes from an ultralytics YOLO prediction result.

    Yields:
        tuple: (xyxy coordinates [x1, y1, x2, y2], confidence score float, class_id int)
    """
    for box in getattr(result, "boxes", []):
        xyxy = box.xyxy
        if hasattr(xyxy, "tolist"):
            xyxy = xyxy.tolist()
        if isinstance(xyxy, list) and len(xyxy) > 0:
            xyxy = xyxy[0]

        conf = box.conf
        if hasattr(conf, "tolist"):
            conf = conf.tolist()
        if isinstance(conf, list) and len(conf) > 0:
            conf = conf[0]
        elif hasattr(conf, "item"):
            conf = conf.item()

        cls_id = int(box.cls[0].item() if hasattr(box.cls, "item") else box.cls[0])
        yield xyxy, conf, cls_id


def _reading_order_sort(boxes, band_height):
    """Sort boxes primarily top-to-bottom, then left-to-right within a vertical band."""
    bh = max(1.0, band_height)
    return sorted(boxes, key=lambda b: (round(b["y"] / bh), b["x"]))


def _resolve_model(model_name_or_path: str, config) -> str:
    """Resolve a model name/path. If it is a Hugging Face repo ID, download the .pt file."""
    path = Path(model_name_or_path)
    if path.exists():
        return str(path)

    # Check if it looks like a Hugging Face model repository: "username/repo"
    if "/" in model_name_or_path and not model_name_or_path.endswith(".pt"):
        logger.info(
            "[%s] Model path not found locally. Attempting to download weights from Hugging Face: %s",
            _STAGE_NAME,
            model_name_or_path,
        )
        try:
            from huggingface_hub import hf_hub_download

            filename = getattr(
                config,
                "DETECTION_WEIGHTS_FILENAME",
                "comic-speech-bubble-detector.pt",
            )
            downloaded_path = hf_hub_download(
                repo_id=model_name_or_path, filename=filename
            )
            logger.info(
                "[%s] Successfully downloaded model to %s",
                _STAGE_NAME,
                downloaded_path,
            )
            return downloaded_path
        except Exception as e:
            logger.error(
                "[%s] Failed to download model from Hugging Face: %s",
                _STAGE_NAME,
                e,
            )
            raise

    return model_name_or_path


def _process_predictions(result, page_num: int, page_height: int) -> list:
    """Process prediction results into a list of raw box dictionaries."""
    names = getattr(result, "names", {})
    raw_boxes = []
    for xyxy, conf, cls_id in _iter_boxes(result):
        x1, y1, x2, y2 = xyxy
        x = int(round(x1))
        y = int(round(y1))
        w = int(round(x2 - x1))
        h = int(round(y2 - y1))

        cls_name = names.get(cls_id, "speech_bubble").lower()
        region_type = "narration" if "narration" in cls_name else "speech_bubble"

        if region_type == "narration":
            style_hint = "narration"
        else:
            aspect_ratio = w / h if h else 1.0
            if aspect_ratio > 2.5 and (
                y < page_height * 0.15 or y + h > page_height * 0.85
            ):
                style_hint = "narration"
            else:
                style_hint = "round"

        raw_boxes.append(
            {
                "page_number": page_num,
                "type": region_type,
                "bbox": {"x": x, "y": y, "w": w, "h": h},
                "x": x,
                "y": y,
                "w": w,
                "h": h,
                "style_hint": style_hint,
                "confidence": float(conf),
                "read_region": {"x": x, "y": y, "w": w, "h": h},
                "erase_mask": {
                    "type": "rect",
                    "coords": [x, y, w, h],
                },
                "render": True,
            }
        )
    return raw_boxes


def _draw_debug_overlay(
    img_path: Path, sorted_boxes: list, overlays_dir: Path, page_num: int
) -> None:
    """Draw bounding boxes and labels onto a copy of the page image."""
    with Image.open(img_path) as im:
        overlay = im.copy()
        draw = ImageDraw.Draw(overlay)
        if not sorted_boxes:
            draw.text((10, 10), "0 regions", fill=(255, 0, 0))
        else:
            for box in sorted_boxes:
                color = (255, 0, 0) if box["type"] == "speech_bubble" else (0, 0, 255)
                bx, by, bw, bh = box["x"], box["y"], box["w"], box["h"]
                draw.rectangle([bx, by, bx + bw, by + bh], outline=color, width=3)
                label = f"{box['region_id']} ({box['confidence']:.2f})"
                draw.text((bx + 5, by + 5), label, fill=color)

        dest_overlay = overlays_dir / f"{page_num:03d}_overlay.png"
        overlay.save(dest_overlay)


def _detect_page_regions(
    page: dict, model, config, ws: Path, overlays_dir: Path
) -> list:
    """Detect regions on a single page, draw overlay, and return mapped clean regions."""
    page_num = page["page_number"]
    img_path = ws / config.STAGE_FOLDERS["pages"] / page["filename"]
    if not img_path.exists():
        raise FileNotFoundError(f"Page file not found: {img_path}")

    results = model.predict(str(img_path), conf=config.DETECTION_CONF, verbose=False)
    if not results:
        return []
    result = results[0]

    page_width = page.get("width", 0)
    page_height = page.get("height", 0)
    if page_width == 0 or page_height == 0:
        with Image.open(img_path) as tmp_im:
            page_width, page_height = tmp_im.size

    raw_boxes = _process_predictions(result, page_num, page_height)
    band = page_height * config.READING_ORDER_BAND_FRACTION
    sorted_boxes = _reading_order_sort(raw_boxes, band)

    page_regions = []
    for idx, box in enumerate(sorted_boxes, start=1):
        box["reading_order"] = idx
        box["region_id"] = config.REGION_ID_FORMAT.format(page=page_num, idx=idx)

        clean_region = {
            "region_id": box["region_id"],
            "page_number": box["page_number"],
            "type": box["type"],
            "bbox": box["bbox"],
            "reading_order": box["reading_order"],
            "style_hint": box["style_hint"],
            "confidence": box["confidence"],
            "read_region": box["read_region"],
            "erase_mask": box["erase_mask"],
            "render": box["render"],
        }
        page_regions.append(clean_region)

    if getattr(config, "OVERLAY_ENABLED", True):
        _draw_debug_overlay(img_path, sorted_boxes, overlays_dir, page_num)

    return page_regions


# ---- Stitching helpers (Feature 4) ----


def _xywh(box) -> dict:
    r = box.get("read_region") or box.get("bbox") or box
    return {"x": r["x"], "y": r["y"], "w": r["w"], "h": r["h"]}


def _box_touches_bottom(r, page_h, eps) -> bool:
    return (r["y"] + r["h"]) >= (page_h - eps)


def _box_touches_top(r, eps) -> bool:
    return r["y"] <= eps


def _x_overlap_frac(a, b) -> float:
    inter = max(0, min(a["x"] + a["w"], b["x"] + b["w"]) - max(a["x"], b["x"]))
    narrower = min(a["w"], b["w"]) or 1
    return inter / narrower


def _find_split_pairs(det_by_page: dict, pages_meta: list, config) -> list:
    """Return confirmed-by-geometry split candidates as tuples
    (page_num_a, page_num_b, box_a, box_b). Enforces pairwise-only via a `used` set."""
    eps = getattr(config, "STITCH_EDGE_EPS", 6)
    min_ov = getattr(config, "STITCH_MIN_X_OVERLAP", 0.5)
    order = [p for p in pages_meta if not p.get("skip") and p.get("filename")]
    used, pairs = set(), []
    for a, b in zip(order, order[1:]):
        na, nb = a["page_number"], b["page_number"]
        if na in used or nb in used:
            continue
        ha = a.get("height") or 0
        bottom = [
            x
            for x in det_by_page.get(na, [])
            if x.get("type") == config.TYPE_SPEECH
            and _box_touches_bottom(_xywh(x), ha, eps)
        ]
        top = [
            x
            for x in det_by_page.get(nb, [])
            if x.get("type") == config.TYPE_SPEECH and _box_touches_top(_xywh(x), eps)
        ]
        found = None
        for bx in bottom:
            for tx in top:
                if _x_overlap_frac(_xywh(bx), _xywh(tx)) >= min_ov:
                    found = (na, nb, bx, tx)
                    break
            if found:
                break
        if found:
            pairs.append(found)
            used.add(na)
            used.add(nb)
    return pairs


def _half_has_text(ws, page_meta, box, config, ocr_engine) -> bool:
    from manhua_pipeline.stages.stage2_ocr import _preprocess_variant, _read_crop

    img_path = ws / config.STAGE_FOLDERS["pages"] / page_meta["filename"]
    with Image.open(img_path) as im:
        r = _xywh(box)
        crop = im.crop((r["x"], r["y"], r["x"] + r["w"], r["y"] + r["h"]))
        text, mean, _min, _wm = _read_crop(
            ocr_engine, _preprocess_variant(crop, 0), config
        )
    return bool(text.strip()) and mean >= getattr(config, "OCR_MIN_TEXT_CONF", 0.30)


def _merge_page_images(ws, page_a, page_b, config) -> tuple:
    """Vertically concat A over B into A's filename slot. Returns (new_w, new_h, seam_y)."""
    pdir = ws / config.STAGE_FOLDERS["pages"]
    ia = Image.open(pdir / page_a["filename"]).convert("RGB")
    ib = Image.open(pdir / page_b["filename"]).convert("RGB")
    W, seam, H = max(ia.width, ib.width), ia.height, ia.height + ib.height
    merged = Image.new("RGB", (W, H), (255, 255, 255))
    merged.paste(ia, (0, 0))
    merged.paste(ib, (0, seam))
    merged.save(pdir / page_a["filename"])
    return W, H, seam


def _apply_stitching(
    ws, manifest: dict, per_page_regions: dict, model, config, overlays_dir
):
    """per_page_regions: {page_number: [region,...]} from the just-completed detection.
    Returns (new_pages_meta, new_regions_by_page) with merges applied + everything
    renumbered sequentially. No-op when disabled or no confirmed splits."""
    pages_meta = manifest.get("pages", [])
    if not getattr(config, "STITCH_ENABLED", False):
        return pages_meta, per_page_regions

    candidates = _find_split_pairs(per_page_regions, pages_meta, config)

    confirmed = []
    if candidates and getattr(config, "STITCH_TEXT_PROBE", True):
        from manhua_pipeline.stages.stage2_ocr import _get_ocr

        ocr_engine = _get_ocr(config)
        by_num = {p["page_number"]: p for p in pages_meta}
        for na, nb, bx, tx in candidates:
            if _half_has_text(
                ws, by_num[na], bx, config, ocr_engine
            ) and _half_has_text(ws, by_num[nb], tx, config, ocr_engine):
                confirmed.append((na, nb))
            else:
                logger.info(
                    "[%s] Split candidate %d+%d rejected (text guard).",
                    _STAGE_NAME,
                    na,
                    nb,
                )
    elif candidates:
        confirmed = [(na, nb) for na, nb, _bx, _tx in candidates]

    if not confirmed:
        return pages_meta, per_page_regions

    merged_away = {nb for _na, nb in confirmed}
    merge_map = {na: nb for na, nb in confirmed}

    new_pages, new_regions = [], {}
    seq = 0
    for p in pages_meta:
        n = p["page_number"]
        if n in merged_away:
            continue
        seq += 1
        if n in merge_map:
            nb = merge_map[n]
            pb = next(x for x in pages_meta if x["page_number"] == nb)
            W, H, seam = _merge_page_images(ws, p, pb, config)
            merged_page = dict(p)
            merged_page.update({"width": W, "height": H, "global_y_offset": seam})
            merged_page["page_number"] = seq
            backend = getattr(config, "DETECTOR_BACKEND", "yolov8")
            if backend == "rtdetr":
                regions = _detect_page_regions_rtdetr(merged_page, config, ws, overlays_dir)
            else:
                regions = _detect_page_regions(merged_page, model, config, ws, overlays_dir)
            logger.info(
                "[%s] Stitched pages %d+%d -> merged page (seam y=%d).",
                _STAGE_NAME,
                n,
                nb,
                seam,
            )
        else:
            merged_page = dict(p)
            regions = per_page_regions.get(n, [])
        merged_page["page_number"] = seq
        new_pages.append(merged_page)
        renum = []
        for idx, r in enumerate(regions, start=1):
            r = dict(r)
            r["page_number"] = seq
            r["region_id"] = config.REGION_ID_FORMAT.format(page=seq, idx=idx)
            renum.append(r)
        new_regions[seq] = renum

    manifest["total_pages"] = len(new_pages)
    return new_pages, new_regions


def run_detection(workspace: str, config) -> Path:
    """Run bubble and narration detection over all usable pages."""
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

    # 1. Resolve paths
    detect_dir = ws / config.STAGE_FOLDERS["detection"]
    detect_dir.mkdir(parents=True, exist_ok=True)
    overlays_dir = detect_dir / "overlays"
    if getattr(config, "OVERLAY_ENABLED", True):
        overlays_dir.mkdir(parents=True, exist_ok=True)

    # 2. Load model once (lazy-load via ultralytics YOLO)
    backend = getattr(config, "DETECTOR_BACKEND", "yolov8")
    logger.info("[%d/%d %s] Detector backend: %s", _STAGE_INDEX, _TOTAL_STAGES, _STAGE_NAME, backend)

    model = None
    if backend == "yolov8":
        resolved_model_path = _resolve_model(config.DETECTION_MODEL, config)
        logger.info(
            "[%d/%d %s] Loading model %s",
            _STAGE_INDEX,
            _TOTAL_STAGES,
            _STAGE_NAME,
            resolved_model_path,
        )
        model = YOLO(resolved_model_path)
    elif backend == "rtdetr":
        _get_rtdetr(config)

    total_pages = manifest.get("total_pages", 0)
    pages_processed = 0
    overlays_written = 0
    warnings = 0
    per_page_regions = {}

    # 3. Iterate usable pages
    for page in manifest.get("pages", []):
        page_num = page["page_number"]
        if page.get("skip") or page.get("filename") is None:
            logger.info(
                "[%d/%d %s] Page %03d/%03d skipped (%s)",
                _STAGE_INDEX,
                _TOTAL_STAGES,
                _STAGE_NAME,
                page_num,
                total_pages,
                page.get("skip_reason", "unknown reason"),
            )
            continue

        try:
            if backend == "rtdetr":
                page_regions = _detect_page_regions_rtdetr(page, config, ws, overlays_dir)
            else:
                page_regions = _detect_page_regions(page, model, config, ws, overlays_dir)
            per_page_regions[page_num] = page_regions
            pages_processed += 1
            if getattr(config, "OVERLAY_ENABLED", True):
                overlays_written += 1
            log_page(
                logger,
                _STAGE_INDEX,
                _TOTAL_STAGES,
                _STAGE_NAME,
                page_num,
                total_pages,
                f"found {len(page_regions)} region(s)",
            )
        except Exception as exc:
            logger.warning(
                "[%s] Page %03d — failed detection: %s",
                _STAGE_NAME,
                page_num,
                exc,
            )
            per_page_regions[page_num] = []
            warnings += 1

    # 3b. Stitching sub-step (Feature 4) — before writing detection.json
    new_pages_meta, new_regions_by_page = _apply_stitching(
        ws, manifest, per_page_regions, model, config, overlays_dir
    )
    manifest["pages"] = new_pages_meta
    regions = [r for pg in sorted(new_regions_by_page) for r in new_regions_by_page[pg]]

    # 4. Save output detection.json
    now = datetime.now(timezone.utc).isoformat()
    output_json = {
        "chapter_id": manifest.get("chapter_id", "unknown_chapter"),
        "stage": "detection",
        "generated_at": now,
        "model": getattr(config, "RTDETR_REPO", "ogkalu/comic-text-and-bubble-detector") if backend == "rtdetr" else config.DETECTION_MODEL,
        "regions": regions,
    }
    detection_json_path = detect_dir / "detection.json"

    with detection_json_path.open("w", encoding="utf-8") as fh:
        json.dump(output_json, fh, ensure_ascii=False, indent=2)

    # 5. Update Manifest
    completed = manifest.get("completed_stages", [])
    if "detect" not in completed:
        completed.append("detect")
    manifest["completed_stages"] = completed
    manifest["current_stage"] = "ocr"
    manifest["updated_at"] = now
    save_manifest(workspace, config, manifest)

    elapsed = time.monotonic() - t0
    logger.info(
        "[%d/%d %s] Wrote overlays for %d page(s)",
        _STAGE_INDEX,
        _TOTAL_STAGES,
        _STAGE_NAME,
        overlays_written,
    )
    log_stage(
        logger,
        _STAGE_INDEX,
        _TOTAL_STAGES,
        _STAGE_NAME,
        f"done: {total_pages} pages ({pages_processed} processed), {len(regions)} regions, "
        f"{overlays_written} overlays, {warnings} warnings -> {detection_json_path} (elapsed {elapsed:.1f}s)",
    )
    return detection_json_path
