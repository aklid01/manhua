"""Stage 1: Detection.

YOLO bubble detection. v0 = speech_bubble + narration only.
Model: ogkalu/comic-speech-bubble-detector-yolov8m.
Outputs detection.json and optional visual debug overlays.
"""

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


def _resolve_model(model_name_or_path: str) -> str:
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

            filename = "comic-speech-bubble-detector.pt"
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

    if getattr(config, "OVERLAY_ENABLED", True) and sorted_boxes:
        _draw_debug_overlay(img_path, sorted_boxes, overlays_dir, page_num)

    return page_regions


def run_detection(workspace: str, config) -> Path:
    """Run bubble and narration detection over all usable pages."""
    t0 = time.monotonic()
    ws = Path(workspace)
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
    resolved_model_path = _resolve_model(config.DETECTION_MODEL)
    logger.info(
        "[%d/%d %s] Loading model %s",
        _STAGE_INDEX,
        _TOTAL_STAGES,
        _STAGE_NAME,
        resolved_model_path,
    )
    model = YOLO(resolved_model_path)

    total_pages = manifest.get("total_pages", 0)
    warnings = 0
    regions = []

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
            page_regions = _detect_page_regions(page, model, config, ws, overlays_dir)
            regions.extend(page_regions)
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
            warnings += 1

    # 4. Save output detection.json
    now = datetime.now(timezone.utc).isoformat()
    output_json = {
        "chapter_id": manifest.get("chapter_id", "unknown_chapter"),
        "stage": "detection",
        "generated_at": now,
        "model": config.DETECTION_MODEL,
        "regions": regions,
    }
    detection_json_path = detect_dir / "detection.json"
    import json

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
        len(regions),
    )
    log_stage(
        logger,
        _STAGE_INDEX,
        _TOTAL_STAGES,
        _STAGE_NAME,
        f"done: {total_pages} pages, {len(regions)} regions, {warnings} warnings -> {detection_json_path} (elapsed {elapsed:.1f}s)",
    )
    return detection_json_path
