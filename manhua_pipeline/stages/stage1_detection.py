# Stage 1: Comic speech bubble and narration text region detection.

import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageDraw

from manhua_pipeline.io.workspace import load_manifest, save_manifest
from manhua_pipeline.logging_setup import get_logger, log_page, log_stage

logger = get_logger(__name__)

_DEV_SAVE_BUBBLE_CROPS = False

_STAGE_INDEX = 1
_TOTAL_STAGES = 7
_STAGE_NAME = "Detection"

_RTDETR_MODEL = None
_RTDETR_PROCESSOR = None
_YOLO_SEG_MODEL = None


# Lazy-load RT-DETR model and image processor
def _get_rtdetr(config):
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
    dev = (
        "cuda"
        if torch.cuda.is_available() and getattr(config, "DETECTOR_USE_GPU", False)
        else "cpu"
    )
    _RTDETR_MODEL.to(dev)
    logger.info("[%s] RT-DETR loaded (%s, cached)", _STAGE_NAME, repo)
    return _RTDETR_MODEL, _RTDETR_PROCESSOR


# Unload RT-DETR from GPU memory and clear cache
def _unload_rtdetr() -> None:
    global _RTDETR_MODEL, _RTDETR_PROCESSOR
    _RTDETR_MODEL = None
    _RTDETR_PROCESSOR = None
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    logger.info("[%s] RT-DETR unloaded (YOLO-seg pass starting)", _STAGE_NAME)


# Lazy-load YOLO segmentation model
def _get_yolo_seg(config):
    global _YOLO_SEG_MODEL
    if _YOLO_SEG_MODEL is not None:
        return _YOLO_SEG_MODEL
    from ultralytics import YOLO

    model_path = Path(getattr(config, "YOLO_SEG_MODEL", "models/best.pt"))
    if not model_path.is_absolute():
        model_path = config.PROJECT_ROOT / model_path
    _YOLO_SEG_MODEL = YOLO(str(model_path))
    logger.info("[%s] YOLO-seg loaded: %s", _STAGE_NAME, model_path)
    return _YOLO_SEG_MODEL


# Save failed segmentation crop and page sample for debugging
def _save_seg_failure(
    crop: Image.Image,
    img_path: Path,
    ws: Path,
    page_num: int,
    region: dict,
    config,
) -> None:
    if not getattr(config, "YOLO_SEG_SAVE_FAILURES", False):
        return
    try:
        fail_dir = Path(getattr(config, "YOLO_SEG_FAILURES_DIR", "debug/seg_failures"))
        if not fail_dir.is_absolute():
            fail_dir = config.PROJECT_ROOT / fail_dir
        crops_dir = fail_dir / "crops"
        pages_dir = fail_dir / "pages"
        crops_dir.mkdir(parents=True, exist_ok=True)
        pages_dir.mkdir(parents=True, exist_ok=True)

        chapter_name = ws.name
        reg_id = region.get("region_id", f"P{page_num:03d}")
        crop_filename = f"{chapter_name}_{reg_id}.png"
        crop.save(crops_dir / crop_filename)

        page_ext = img_path.suffix or ".png"
        page_filename = f"{chapter_name}_P{page_num:03d}{page_ext}"
        target_page_path = pages_dir / page_filename
        if not target_page_path.exists():
            shutil.copy2(img_path, target_page_path)
    except Exception as exc:
        logger.warning("[%s] Failed saving seg failure sample: %s", _STAGE_NAME, exc)


# Encode PIL grayscale mask image as base64 PNG string
def _encode_mask_b64(pil_mask) -> str:
    import base64
    import io

    buf = io.BytesIO()
    pil_mask.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


# Post-detection pass applying YOLO segmentation to refine speech bubble masks
def _run_yolo_seg_all_pages(
    regions_by_page: dict,
    pages_meta: list,
    ws: Path,
    config,
) -> None:
    import numpy as np

    conf = float(getattr(config, "YOLO_SEG_CONF", 0.5))
    pad_frac = float(getattr(config, "YOLO_SEG_PAD_FRAC", 0.15))
    device = str(getattr(config, "YOLO_SEG_DEVICE", "cpu"))

    model = _get_yolo_seg(config)
    page_by_num = {p["page_number"]: p for p in pages_meta}
    pdir = ws / config.STAGE_FOLDERS["pages"]
    total_masked = 0

    for page_num, regions in regions_by_page.items():
        page_meta = page_by_num.get(page_num)
        if not page_meta or not page_meta.get("filename"):
            continue
        img_path = pdir / page_meta["filename"]
        if not img_path.exists():
            continue
        eligible = [
            r
            for r in regions
            if r.get("parent_bubble") and r.get("type") == config.TYPE_SPEECH
        ]
        if not eligible:
            continue

        with Image.open(img_path).convert("RGB") as image:
            W, H = image.size
            crops = []
            crop_meta = []
            for region in eligible:
                pb = region["parent_bubble"]
                px, py, pw, ph = pb["x"], pb["y"], pb["w"], pb["h"]
                pad_x = int(pw * pad_frac)
                pad_y = int(ph * pad_frac)
                x1 = max(0, px - pad_x)
                y1 = max(0, py - pad_y)
                x2 = min(W, px + pw + pad_x)
                y2 = min(H, py + ph + pad_y)
                crop = image.crop((x1, y1, x2, y2))
                crops.append(crop)
                crop_meta.append(
                    (crop.size[0], crop.size[1], x1, y1, px, py, pw, ph, region)
                )

            try:
                all_results = model.predict(
                    crops,
                    conf=conf,
                    imgsz=640,
                    device=device,
                    retina_masks=True,
                    verbose=False,
                )
            except Exception as exc:
                logger.warning(
                    "[%s] YOLO-seg batch failed on page %d: %s",
                    _STAGE_NAME,
                    page_num,
                    exc,
                )
                for crop, (*_, region) in zip(crops, crop_meta):
                    _save_seg_failure(crop, img_path, ws, page_num, region, config)
                continue

            for result, (crop_w, crop_h, x1, y1, px, py, pw, ph, region), crop in zip(
                all_results, crop_meta, crops
            ):
                masks = getattr(result, "masks", None)
                boxes = getattr(result, "boxes", None)
                if masks is None or not len(getattr(masks, "data", [])):
                    _save_seg_failure(crop, img_path, ws, page_num, region, config)
                    continue

                best_idx, best_conf = 0, 0.0
                if boxes is not None:
                    for i, box in enumerate(boxes):
                        c = float(box.conf[0])
                        if c > best_conf:
                            best_conf, best_idx = c, i

                mask_np = masks.data[best_idx].cpu().numpy()
                mask_at_crop = Image.fromarray(
                    (mask_np * 255).astype(np.uint8), mode="L"
                ).resize((crop_w, crop_h), Image.NEAREST)

                off_x, off_y = px - x1, py - y1
                mask_pb = mask_at_crop.crop((off_x, off_y, off_x + pw, off_y + ph))

                region["bubble_mask"] = {
                    "mask_b64": _encode_mask_b64(mask_pb),
                    "confidence": round(best_conf, 4),
                    "valid": True,
                }
                total_masked += 1

    logger.info(
        "[%d/%d %s] YOLO-seg: %d region(s) refined with bubble masks",
        _STAGE_INDEX,
        _TOTAL_STAGES,
        _STAGE_NAME,
        total_masked,
    )


# Construct region dictionary adhering to downstream schema
def _mk_region(
    config, page_num, idx, rec, *, type_, parent_bubble, is_free_text
) -> dict:
    x, y, w, h = rec["x"], rec["y"], rec["w"], rec["h"]
    eb = parent_bubble if parent_bubble else {"x": x, "y": y, "w": w, "h": h}
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
            "coords": [eb["x"], eb["y"], eb["w"], eb["h"]],
        },
        "render": not is_free_text,
        "is_free_text": is_free_text,
        "parent_bubble": (
            {
                "x": parent_bubble["x"],
                "y": parent_bubble["y"],
                "w": parent_bubble["w"],
                "h": parent_bubble["h"],
            }
            if parent_bubble
            else None
        ),
        "text_direction": "horizontal",
    }


# Save raw parent bubble ROI crops for training data
def _save_bubble_crops_dev(
    img_path: Path,
    bubbles: list,
    crops_dir: Path,
    page_num: int,
) -> None:
    crops_dir.mkdir(parents=True, exist_ok=True)
    with Image.open(img_path) as im:
        for i, b in enumerate(bubbles):
            x, y, w, h = b["x"], b["y"], b["w"], b["h"]
            pad_x = int(w * 0.15)
            pad_y = int(h * 0.15)
            crop = im.crop(
                (
                    max(0, x - pad_x),
                    max(0, y - pad_y),
                    min(im.width, x + w + pad_x),
                    min(im.height, y + h + pad_y),
                )
            )
            crop.save(crops_dir / f"{page_num:03d}_bubble_{i:02d}.png")


# Render bounding boxes and parent bubble links for RT-DETR debug overlay
def _draw_debug_overlay_rtdetr(
    img_path: Path, sorted_boxes: list, bubbles: list, overlays_dir: Path, page_num: int
) -> None:
    with Image.open(img_path) as im:
        overlay = im.copy()
        draw = ImageDraw.Draw(overlay)

        for b in bubbles:
            bx, by, bw, bh = b["x"], b["y"], b["w"], b["h"]
            draw.rectangle([bx, by, bx + bw, by + bh], outline=(0, 255, 0), width=2)
            draw.text((bx + 5, by + 5), f"bubble ({b['conf']:.2f})", fill=(0, 255, 0))

        for box in sorted_boxes:
            bx, by, bw, bh = (
                box["bbox"]["x"],
                box["bbox"]["y"],
                box["bbox"]["w"],
                box["bbox"]["h"],
            )
            if box["type"] == "speech_bubble":
                color = (255, 0, 0)
                label = f"{box['region_id']} ({box['confidence']:.2f})"
                if box.get("parent_bubble"):
                    pb = box["parent_bubble"]
                    px, py = pb["x"] + pb["w"] // 2, pb["y"] + pb["h"] // 2
                    tx, ty = bx + bw // 2, by + bh // 2
                    draw.line([(tx, ty), (px, py)], fill=(255, 255, 0), width=1)
            else:
                color = (0, 0, 255)
                label = f"{box['region_id']} (SFX, {box['confidence']:.2f})"

            draw.rectangle([bx, by, bx + bw, by + bh], outline=color, width=3)
            draw.text(
                (bx + 5, by + 18 if box["type"] == "speech_bubble" else by + 5),
                label,
                fill=color,
            )

        dest_overlay = overlays_dir / f"{page_num:03d}_overlay.png"
        overlay.save(dest_overlay)


# Execute RT-DETR detection on single page and map speech and narration regions
def _detect_page_regions_rtdetr(page, config, ws, overlays_dir) -> list:
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
    target = torch.tensor([image.size[::-1]]).to(model.device)
    det = processor.post_process_object_detection(
        outputs, target_sizes=target, threshold=getattr(config, "RTDETR_CONF", 0.30)
    )[0]

    C_BUB = getattr(config, "RTDETR_CLASS_BUBBLE", 0)
    C_TXT = getattr(config, "RTDETR_CLASS_TEXT_BUBBLE", 1)
    C_FREE = getattr(config, "RTDETR_CLASS_TEXT_FREE", 2)

    bubbles, raw_boxes = [], []
    for score, label, box in zip(det["scores"], det["labels"], det["boxes"]):
        x0, y0, x1, y1 = [int(v) for v in box.cpu().tolist()]
        rec = {
            "x": x0,
            "y": y0,
            "w": x1 - x0,
            "h": y1 - y0,
            "conf": float(score.item()),
        }
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

    raw_boxes = _dedupe_raw_boxes(raw_boxes, config)
    page_height = image.height
    band = page_height * config.READING_ORDER_BAND_FRACTION
    sorted_boxes = _reading_order_sort(raw_boxes, band)

    regions = []
    for idx, box in enumerate(sorted_boxes, start=1):
        parent = _parent_bubble(box) if box["type"] == config.TYPE_SPEECH else None
        regions.append(
            _mk_region(
                config,
                page_num,
                idx,
                box,
                type_=box["type"],
                parent_bubble=parent,
                is_free_text=box["is_free_text"],
            )
        )

    if _DEV_SAVE_BUBBLE_CROPS and bubbles:
        _save_bubble_crops_dev(
            img_path, bubbles, overlays_dir.parent / "dev_crops", page_num
        )

    if getattr(config, "OVERLAY_ENABLED", True):
        _draw_debug_overlay_rtdetr(img_path, regions, bubbles, overlays_dir, page_num)

    return regions


# Sort bounding boxes top-to-bottom and left-to-right within vertical bands
def _reading_order_sort(boxes, band_height):
    bh = max(1.0, band_height)
    return sorted(boxes, key=lambda b: (round(b["y"] / bh), b["x"]))


# Compute intersection-over-union metric for two bounding boxes
def _iou(a: dict, b: dict) -> float:
    ax0, ay0, ax1, ay1 = a["x"], a["y"], a["x"] + a["w"], a["y"] + a["h"]
    bx0, by0, bx1, by1 = b["x"], b["y"], b["x"] + b["w"], b["y"] + b["h"]
    iw = max(0, min(ax1, bx1) - max(ax0, bx0))
    ih = max(0, min(ay1, by1) - max(ay0, by0))
    inter = iw * ih
    if inter == 0:
        return 0.0
    union = a["w"] * a["h"] + b["w"] * b["h"] - inter
    return inter / union if union > 0 else 0.0


# Compute intersection-over-smaller metric for two bounding boxes
def _ios(a: dict, b: dict) -> float:
    ax0, ay0, ax1, ay1 = a["x"], a["y"], a["x"] + a["w"], a["y"] + a["h"]
    bx0, by0, bx1, by1 = b["x"], b["y"], b["x"] + b["w"], b["y"] + b["h"]

    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0, ix1 - ix0), max(0, iy1 - iy0)
    inter = iw * ih
    if inter == 0:
        return 0.0

    smaller = min(a["w"] * a["h"], b["w"] * b["h"])
    return inter / smaller if smaller > 0 else 0.0


# Filter near-duplicate overlapping bounding boxes keeping highest confidence
def _dedupe_raw_boxes(raw_boxes: list[dict], config) -> list[dict]:
    iou_thresh = float(getattr(config, "DEDUP_IOU_THRESHOLD", 0.90))
    ios_thresh = float(getattr(config, "DEDUP_IOS_THRESHOLD", 0.85))

    by_type: dict[str, list[dict]] = {}
    for r in raw_boxes:
        key = r.get("type", "")
        by_type.setdefault(key, []).append(r)

    kept: list[dict] = []
    dropped: list = []

    for _type, group in by_type.items():
        group_sorted = sorted(
            group, key=lambda r: r.get("confidence", r.get("conf", 0.0)), reverse=True
        )
        survivors: list[dict] = []
        for r in group_sorted:
            bbox_r = r.get("bbox") or {
                "x": r["x"],
                "y": r["y"],
                "w": r["w"],
                "h": r["h"],
            }
            is_dup = False
            for s in survivors:
                bbox_s = s.get("bbox") or {
                    "x": s["x"],
                    "y": s["y"],
                    "w": s["w"],
                    "h": s["h"],
                }
                iou = _iou(bbox_r, bbox_s)
                ios = _ios(bbox_r, bbox_s)
                if iou >= iou_thresh or ios >= ios_thresh:
                    is_dup = True
                    dropped.append(r)
                    logger.info(
                        "[%d/%d %s] Dedup drop %s (dup of %s) IoU=%.3f IoS=%.3f",
                        _STAGE_INDEX,
                        _TOTAL_STAGES,
                        _STAGE_NAME,
                        r.get("region_id", "?"),
                        s.get("region_id", "?"),
                        iou,
                        ios,
                    )
                    break
            if not is_dup:
                survivors.append(r)
        kept.extend(survivors)

    if dropped:
        logger.info(
            "[%d/%d %s] Dedup removed %d duplicate raw box(es)",
            _STAGE_INDEX,
            _TOTAL_STAGES,
            _STAGE_NAME,
            len(dropped),
        )

    return kept


# Extract {x, y, w, h} rectangle dictionary from region
def _xywh(box) -> dict:
    r = box.get("read_region") or box.get("bbox") or box
    return {"x": r["x"], "y": r["y"], "w": r["w"], "h": r["h"]}


# Resolve boundary rectangle for page stitching edge tests
def _stitch_box(box) -> dict:
    pb = box.get("parent_bubble")
    if pb and all(k in pb for k in ("x", "y", "w", "h")):
        return {"x": pb["x"], "y": pb["y"], "w": pb["w"], "h": pb["h"]}
    return _xywh(box)


# Check if rectangle touches or extends past bottom edge of page
def _box_touches_bottom(r, page_h, eps) -> bool:
    return (r["y"] + r["h"]) >= (page_h - eps)


# Check if rectangle touches top edge of page
def _box_touches_top(r, eps) -> bool:
    return r["y"] <= eps


# Calculate horizontal overlap ratio between two rectangles
def _x_overlap_frac(a, b) -> float:
    inter = max(0, min(a["x"] + a["w"], b["x"] + b["w"]) - max(a["x"], b["x"]))
    narrower = min(a["w"], b["w"]) or 1
    return inter / narrower


# Verify whether bottom bubble of page A continues into top bubble of page B
def _pages_link(a, b, det_by_page, config) -> bool:
    eps = getattr(config, "STITCH_EDGE_EPS", 6)
    min_ov = getattr(config, "STITCH_MIN_X_OVERLAP", 0.5)
    ha = a.get("height") or 0
    bottom = [
        x
        for x in det_by_page.get(a["page_number"], [])
        if x.get("type") == config.TYPE_SPEECH
        and _box_touches_bottom(_stitch_box(x), ha, eps)
    ]
    top = [
        x
        for x in det_by_page.get(b["page_number"], [])
        if x.get("type") == config.TYPE_SPEECH and _box_touches_top(_stitch_box(x), eps)
    ]
    for bx in bottom:
        for tx in top:
            ov = _x_overlap_frac(_stitch_box(bx), _stitch_box(tx))
            logger.info(
                "[%s]   link %d->%d overlap=%.2f (need >=%.2f) %s",
                _STAGE_NAME,
                a["page_number"],
                b["page_number"],
                ov,
                min_ov,
                "MATCH" if ov >= min_ov else "no",
            )
            if ov >= min_ov:
                return True
    return False


# Find consecutive sequences of pages split across a shared speech bubble
def _find_split_chains(det_by_page, pages_meta, config) -> list[list[int]]:
    max_chain = getattr(config, "STITCH_MAX_CHAIN", 4)
    order = [p for p in pages_meta if not p.get("skip") and p.get("filename")]
    chains, i = [], 0
    while i < len(order) - 1:
        chain = [order[i]["page_number"]]
        j = i
        while (
            j + 1 < len(order)
            and len(chain) < max_chain
            and _pages_link(order[j], order[j + 1], det_by_page, config)
        ):
            chain.append(order[j + 1]["page_number"])
            j += 1
        if len(chain) > 1:
            chains.append(chain)
            logger.info("[%s] Split chain: %s", _STAGE_NAME, chain)
            i = j + 1
        else:
            i += 1
    return chains


# Check if split bubble half contains readable text above confidence threshold
def _half_has_text(ws, page_meta, box, config, ocr_engine) -> bool:
    from manhua_pipeline.stages.stage2_ocr import _preprocess_variant, _read_crop

    img_path = ws / config.STAGE_FOLDERS["pages"] / page_meta["filename"]
    with Image.open(img_path) as im:
        r = _xywh(box)
        crop = im.crop((r["x"], r["y"], r["x"] + r["w"], r["y"] + r["h"]))
        text, mean, _min, _wm = _read_crop(
            ocr_engine, _preprocess_variant(crop, 0), config
        )
    stitch_floor = getattr(config, "STITCH_TEXT_MIN_CONF", 0.15)
    return bool(text.strip()) and mean >= stitch_floor


# Verify if either side of split bubble connection contains OCR text
def _link_has_text(ws, page_a, page_b, det_by_page, config, ocr_engine) -> bool:
    eps = getattr(config, "STITCH_EDGE_EPS", 6)
    min_ov = getattr(config, "STITCH_MIN_X_OVERLAP", 0.5)
    ha = page_a.get("height") or 0
    bottom = [
        x
        for x in det_by_page.get(page_a["page_number"], [])
        if x.get("type") == config.TYPE_SPEECH
        and _box_touches_bottom(_stitch_box(x), ha, eps)
    ]
    top = [
        x
        for x in det_by_page.get(page_b["page_number"], [])
        if x.get("type") == config.TYPE_SPEECH and _box_touches_top(_stitch_box(x), eps)
    ]
    for bx in bottom:
        for tx in top:
            if _x_overlap_frac(_stitch_box(bx), _stitch_box(tx)) >= min_ov:
                if _half_has_text(ws, page_a, bx, config, ocr_engine) or _half_has_text(
                    ws, page_b, tx, config, ocr_engine
                ):
                    return True
    return False


# Concatenate linked vertical pages into single combined image
def _merge_page_chain(ws, head_page, other_pages, config) -> tuple:
    pdir = ws / config.STAGE_FOLDERS["pages"]
    head_path = pdir / head_page["filename"]
    other_paths = [pdir / p["filename"] for p in other_pages]
    imgs = [Image.open(head_path).convert("RGB")]
    imgs += [Image.open(p).convert("RGB") for p in other_paths]
    W, H = max(im.width for im in imgs), sum(im.height for im in imgs)
    merged = Image.new("RGB", (W, H), (255, 255, 255))
    y = 0
    for im in imgs:
        merged.paste(im, (0, y))
        y += im.height
    for im in imgs:
        im.close()
    if head_path.exists():
        head_path.unlink()
    merged.save(head_path)
    for op in other_paths:
        if op.exists():
            op.unlink()
    return W, H, imgs[0].height


# Execute page stitching and renumbering on detected split chains
def _apply_stitching(ws, manifest: dict, per_page_regions: dict, config, overlays_dir):
    pages_meta = manifest.get("pages", [])
    if not getattr(config, "STITCH_ENABLED", False):
        return pages_meta, per_page_regions

    chains = _find_split_chains(per_page_regions, pages_meta, config)

    confirmed = []
    if chains and getattr(config, "STITCH_TEXT_PROBE", True):
        from manhua_pipeline.stages.stage2_ocr import _get_ocr

        ocr_engine = _get_ocr(config)
        by_num = {p["page_number"]: p for p in pages_meta}
        for chain in chains:
            ok = all(
                _link_has_text(
                    ws, by_num[na], by_num[nb], per_page_regions, config, ocr_engine
                )
                for na, nb in zip(chain, chain[1:])
            )
            if ok:
                confirmed.append(chain)
            else:
                logger.info("[%s] Chain %s rejected (text guard).", _STAGE_NAME, chain)
    else:
        confirmed = chains

    if not confirmed:
        return pages_meta, per_page_regions

    merged_away = {n for ch in confirmed for n in ch[1:]}
    head_of = {ch[0]: ch for ch in confirmed}

    new_pages, new_regions = [], {}
    seq = 0
    for p in pages_meta:
        n = p["page_number"]
        if n in merged_away:
            continue
        seq += 1
        if n in head_of:
            chain = head_of[n]
            others = [
                next(x for x in pages_meta if x["page_number"] == m) for m in chain[1:]
            ]
            W, H, seam = _merge_page_chain(ws, p, others, config)
            merged_page = dict(p)
            merged_page.update({"width": W, "height": H, "global_y_offset": seam})
            merged_page["page_number"] = seq
            regions = _detect_page_regions_rtdetr(merged_page, config, ws, overlays_dir)
            logger.info("[%s] Stitched chain %s -> merged page.", _STAGE_NAME, chain)
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


# Execute full chapter detection stage using the RT-DETR backend
def run_detection(workspace: str, config) -> Path:
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

    detect_dir = ws / config.STAGE_FOLDERS["detection"]
    detect_dir.mkdir(parents=True, exist_ok=True)
    overlays_dir = detect_dir / "overlays"
    overlay_enabled = getattr(config, "OVERLAY_ENABLED", True)
    if overlay_enabled:
        overlays_dir.mkdir(parents=True, exist_ok=True)

    _get_rtdetr(config)

    total_pages = manifest.get("total_pages", 0)
    pages_processed = 0
    overlays_written = 0
    warnings = 0
    per_page_regions = {}

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
            page_regions = _detect_page_regions_rtdetr(page, config, ws, overlays_dir)
            per_page_regions[page_num] = page_regions
            pages_processed += 1
            if overlay_enabled:
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
                "[%s] Page %03d - failed detection: %s",
                _STAGE_NAME,
                page_num,
                exc,
            )
            per_page_regions[page_num] = []
            warnings += 1

    new_pages_meta, new_regions_by_page = _apply_stitching(
        ws, manifest, per_page_regions, config, overlays_dir
    )
    manifest["pages"] = new_pages_meta

    if getattr(config, "YOLO_SEG_ENABLED", False):
        _unload_rtdetr()
        _run_yolo_seg_all_pages(new_regions_by_page, new_pages_meta, ws, config)

    regions = [r for pg in sorted(new_regions_by_page) for r in new_regions_by_page[pg]]

    now = datetime.now(timezone.utc).isoformat()
    output_json = {
        "chapter_id": manifest.get("chapter_id", "unknown_chapter"),
        "stage": "detection",
        "generated_at": now,
        "model": getattr(
            config, "RTDETR_REPO", "ogkalu/comic-text-and-bubble-detector"
        ),
        "regions": regions,
    }
    detection_json_path = detect_dir / "detection.json"

    with detection_json_path.open("w", encoding="utf-8") as fh:
        json.dump(output_json, fh, ensure_ascii=False, indent=2)

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
    emit_stage_event(
        ws,
        config,
        _STAGE_NAME.lower(),
        STAGE_COMPLETED,
        logger=logger,
        metadata={"total_regions": len(regions), "pages_processed": pages_processed},
    )
    return detection_json_path
