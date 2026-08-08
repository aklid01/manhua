import copy
import json
import time
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path

from PIL import Image, ImageDraw

from manhua_pipeline.io.workspace import load_manifest, save_manifest
from manhua_pipeline.logging_setup import get_logger, log_stage
from manhua_pipeline.stages.stage5_render import credits, geometry, planner
from manhua_pipeline.stages.stage5_render.types import Box, RenderPlan

logger = get_logger(__name__)

_STAGE_INDEX = 5
_TOTAL_STAGES = 7
_STAGE_NAME = "Render"


def _box_overlap_ratio(a: dict | None, b: dict | None) -> float:
    """Return intersection / area of the smaller box."""
    if not a or not b:
        return 0.0
    ax0, ay0 = int(a.get("x", 0)), int(a.get("y", 0))
    ax1, ay1 = ax0 + int(a.get("w", 0)), ay0 + int(a.get("h", 0))
    bx0, by0 = int(b.get("x", 0)), int(b.get("y", 0))
    bx1, by1 = bx0 + int(b.get("w", 0)), by0 + int(b.get("h", 0))
    intersection = max(0, min(ax1, bx1) - max(ax0, bx0)) * max(
        0, min(ay1, by1) - max(ay0, by0)
    )
    smaller = min(
        max(0, int(a.get("w", 0)) * int(a.get("h", 0))),
        max(0, int(b.get("w", 0)) * int(b.get("h", 0))),
    )
    return intersection / smaller if smaller else 0.0


def _normalized_source_text(text: str) -> str:
    return "".join(ch.lower() for ch in (text or "") if ch.isalnum() or "\u3400" <= ch <= "\u9fff")


def _source_overlap_ratio(a: str, b: str) -> float:
    """Measure shared source characters, including separate matching spans."""
    left, right = _normalized_source_text(a), _normalized_source_text(b)
    if not left or not right:
        return 0.0
    matches = SequenceMatcher(None, left, right, autojunk=False).get_matching_blocks()
    matched = sum(block.size for block in matches)
    return matched / min(len(left), len(right))


def _suppress_overlapping_duplicate_regions(regions_by_page: dict[int, list[dict]]) -> None:
    """Suppress detector duplicates from one irregular speech bubble.

    RT-DETR can emit a large text box and a smaller box for the same connected
    bubble. Rendering both produces the exact overprinted English seen when
    their parent bubbles and OCR text substantially overlap. The smaller
    region remains in the render pass as erase-only so its original artwork is
    still removed, but it never contributes a second English text block.
    """
    for page_num, regions in regions_by_page.items():
        candidates = [
            r
            for r in regions
            if r.get("render", True)
            and r.get("has_usable_text")
            and r.get("final_text")
            and r.get("parent_bubble")
        ]
        candidates.sort(
            key=lambda r: int(r["bbox"].get("w", 0)) * int(r["bbox"].get("h", 0)),
            reverse=True,
        )

        for primary_index, primary in enumerate(candidates):
            if not primary.get("render", True):
                continue
            for secondary in candidates[primary_index + 1 :]:
                if not secondary.get("render", True):
                    continue
                parent_overlap = _box_overlap_ratio(
                    primary.get("parent_bubble"), secondary.get("parent_bubble")
                )
                text_overlap = _box_overlap_ratio(
                    primary.get("read_region") or primary.get("bbox"),
                    secondary.get("read_region") or secondary.get("bbox"),
                )
                source_overlap = _source_overlap_ratio(
                    primary.get("original_text", ""),
                    secondary.get("original_text", ""),
                )
                if (
                    parent_overlap < 0.65
                    or text_overlap < 0.55
                    or source_overlap < 0.55
                ):
                    continue

                secondary["render"] = False
                secondary["erase_only"] = True
                secondary["render_skip_reason"] = "overlapping_duplicate"
                secondary["overlap_duplicate_of"] = primary["region_id"]
                logger.warning(
                    "[%d/%d %s] Page %d %s suppressed as duplicate of %s "
                    "(parent=%.2f, text=%.2f, source=%.2f)",
                    _STAGE_INDEX,
                    _TOTAL_STAGES,
                    _STAGE_NAME,
                    page_num,
                    secondary["region_id"],
                    primary["region_id"],
                    parent_overlap,
                    text_overlap,
                    source_overlap,
                )


def _render_region(*_args, **_kwargs):
    raise NotImplementedError(
        "_render_region is retired; use _plan_and_erase_region + _draw_region_plan."
    )


def _load_and_merge_regions(
    det_data: dict, ocr_data: dict, para_data: dict
) -> dict[int, list[dict]]:
    ocr_map = {r["region_id"]: r for r in ocr_data.get("results", [])}
    para_map = {r["region_id"]: r for r in para_data.get("results", [])}

    regions_by_page = {}
    for region in det_data.get("regions", []):
        rid = region["region_id"]
        page_num = region["page_number"]
        ocr_entry = ocr_map.get(rid, {})
        para_entry = para_map.get(rid, {})

        merged = {
            "region_id": rid,
            "page_number": page_num,
            "bbox": region["bbox"],
            "read_region": region.get("read_region"),
            "parent_bubble": region.get("parent_bubble"),
            "erase_mask": region.get("erase_mask")
            or {
                "type": "rect",
                "coords": [
                    region["bbox"]["x"],
                    region["bbox"]["y"],
                    region["bbox"]["w"],
                    region["bbox"]["h"],
                ],
            },
            "render": region.get("render", True),
            "style_hint": region.get("style_hint") or "round",
            "has_usable_text": ocr_entry.get("has_usable_text", False),
            "original_text": ocr_entry.get("original_text", ""),
            "final_text": para_entry.get("final_text") or "",
            "register": para_entry.get("register") or "neutral",
            "glossary_conflict": para_entry.get("glossary_conflict") or False,
        }
        regions_by_page.setdefault(page_num, []).append(merged)
    _suppress_overlapping_duplicate_regions(regions_by_page)
    return regions_by_page


def _clip_region_to_stitch(region: dict, stitch_y: int) -> dict:
    """Clamp the layout boxes of a region so text is placed only within the first
    source page's half of a stitched image.

    Only bbox and read_region are clipped at stitch_y. parent_bubble and
    erase_mask are intentionally left at their original extent so the eraser
    still wipes the full bubble area (including Chinese text that may sit close
    to the seam). Restricting the layout inputs prevents English text from being
    placed where it would overlap with second-half content.
    """

    def _clip_dict_box(d: dict | None) -> dict | None:
        if d is None:
            return None
        y = int(d.get("y", 0))
        h = int(d.get("h", 0))
        new_h = max(0, min(h, stitch_y - y))
        if new_h == h:
            return d
        result = dict(d)
        result["h"] = new_h
        return result

    region = copy.copy(region)

    region["bbox"] = _clip_dict_box(region.get("bbox")) or region["bbox"]

    if region.get("read_region"):
        region["read_region"] = _clip_dict_box(region["read_region"])

    return region



def _render_single_page(
    page: dict,
    ws: Path,
    regions_by_page: dict,
    render_dir: Path,
    config,
    run_id: str = "unknown",
) -> tuple[list[dict], dict | None, int, int, int]:
    page_num = page["page_number"]
    filename = page["filename"]
    orig_filename = page.get("original_filename")
    if page.get("skip") or not filename or not orig_filename:
        logger.info(
            "[%d/%d %s] Skipping skipped/failed Page %d",
            _STAGE_INDEX,
            _TOTAL_STAGES,
            _STAGE_NAME,
            page_num,
        )
        return [], None, 0, 0, 0

    orig_path = ws / "pages" / filename
    if not orig_path.exists():
        logger.warning(
            "[%d/%d %s] Page %d file not found: %s",
            _STAGE_INDEX,
            _TOTAL_STAGES,
            _STAGE_NAME,
            page_num,
            orig_path,
        )
        return [], None, 0, 0, 0

    try:
        page_img = Image.open(orig_path).convert("RGB")
    except Exception as exc:
        logger.error(
            "[%d/%d %s] Failed to load Page %d: %s",
            _STAGE_INDEX,
            _TOTAL_STAGES,
            _STAGE_NAME,
            page_num,
            exc,
        )
        return [], None, 0, 0, 0

    draw = ImageDraw.Draw(page_img)
    page_regions = regions_by_page.get(page_num, [])

    stitch_y = page.get("global_y_offset")
    if stitch_y:
        clipped = []
        for r in page_regions:
            # A bubble can straddle the seam while its text belongs entirely to
            # the second source page. Use the text/layout anchor to decide
            # which half owns the region; the parent bubble may cross the seam
            # so erasing can still cover its border.
            text_box = r.get("read_region") or r["bbox"]
            region_top = int(text_box["y"])
            if region_top < stitch_y:
                r = _clip_region_to_stitch(r, stitch_y)
                if int(r["bbox"].get("h", 0)) == 0:
                    logger.warning(
                        "[%d/%d %s] Page %d %s bbox clipped to zero height at stitch boundary - skipping",
                        _STAGE_INDEX,
                        _TOTAL_STAGES,
                        _STAGE_NAME,
                        page_num,
                        r["region_id"],
                    )
                    continue
            clipped.append(r)
        page_regions = clipped

    render_regions = [r for r in page_regions if r.get("render", True)]

    def _text_box_of(r) -> Box:
        val = r.get("read_region") or r.get("bbox")
        return geometry.clip_box(geometry.box_from_dict(val), page_img.size)

    boxes_by_id = {r["region_id"]: _text_box_of(r) for r in render_regions}

    plans: list[RenderPlan] = []
    for region in page_regions:
        rid = region["region_id"]
        siblings = (
            [box for other_id, box in boxes_by_id.items() if other_id != rid]
            if region.get("render", True)
            else None
        )
        plan = planner._plan_and_erase_region(
            region,
            page_img,
            draw,
            page_num,
            config,
            render_dir=render_dir,
            sibling_text_boxes=siblings,
        )
        plans.append(plan)

    for plan in plans:
        planner._draw_region_plan(plan, draw, render_dir=render_dir)

    page_results = []
    page_drawn = page_left = page_overflow = 0
    for plan in plans:
        result, drawn, left, ovf = planner._finalize_plan(
            plan,
            emitter=None,
            run_id=run_id,
            chapter=ws.name,
            render_dir=render_dir,
            config=config,
        )
        page_results.append(result)
        page_drawn += drawn
        page_left += left
        page_overflow += ovf

    rendered_dir = render_dir / "rendered"
    rendered_dir.mkdir(parents=True, exist_ok=True)
    out_path = rendered_dir / f"{page_num:03d}.png"
    page_img.save(out_path)
    logger.info(
        "[%d/%d %s] Page %d -> saved %s (%d drawn, %d left)",
        _STAGE_INDEX,
        _TOTAL_STAGES,
        _STAGE_NAME,
        page_num,
        out_path.relative_to(ws),
        page_drawn,
        page_left,
    )

    page_report = {
        "page_number": page_num,
        "output_file": str(out_path.relative_to(ws)),
        "regions_drawn": page_drawn,
        "regions_left": page_left,
    }
    return page_results, page_report, page_drawn, page_left, page_overflow


def run_render(workspace: str, config, run_id: str | None = None) -> Path:
    t0 = time.monotonic()
    ws = Path(workspace)
    resolved_run_id = run_id or getattr(config, "_run_id", "unknown")
    logger.info(
        "[%d/%d %s] Series: %s | Chapter: %s",
        _STAGE_INDEX,
        _TOTAL_STAGES,
        _STAGE_NAME,
        ws.parent.as_posix(),
        ws.name,
    )
    log_stage(logger, _STAGE_INDEX, _TOTAL_STAGES, _STAGE_NAME, "starting")
    from manhua_pipeline.events import (
        RENDER_REGION_COMPLETED,
        STAGE_COMPLETED,
        STAGE_STARTED,
    )
    from manhua_pipeline.io.event_emitter import emit_stage_event

    emit_stage_event(
        ws,
        config,
        _STAGE_NAME.lower(),
        STAGE_STARTED,
        logger=logger,
        run_id=resolved_run_id,
    )

    manifest = load_manifest(workspace, config)
    if not manifest:
        raise ValueError("Manifest not found. Run import first.")

    det_path = ws / config.STAGE_FOLDERS["detection"] / "detection.json"
    if not det_path.exists():
        raise FileNotFoundError("detection.json not found. Run detect first.")

    ocr_path = ws / config.STAGE_FOLDERS["ocr"] / "ocr.json"
    if not ocr_path.exists():
        raise FileNotFoundError("ocr.json not found. Run ocr first.")

    para_path = ws / config.STAGE_FOLDERS["paraphrase"] / "paraphrase.json"
    if not para_path.exists():
        raise FileNotFoundError("paraphrase.json not found. Run paraphrase first.")

    with det_path.open("r", encoding="utf-8") as fh:
        det_data = json.load(fh)
    with ocr_path.open("r", encoding="utf-8") as fh:
        ocr_data = json.load(fh)
    with para_path.open("r", encoding="utf-8") as fh:
        para_data = json.load(fh)

    logger.info(
        "[%d/%d %s] Font: %s",
        _STAGE_INDEX,
        _TOTAL_STAGES,
        _STAGE_NAME,
        getattr(config, "FONT_PATH", "assets/fonts/ComicNeue-Bold.ttf"),
    )

    regions_by_page = _load_and_merge_regions(det_data, ocr_data, para_data)

    render_results = []
    page_reports = []
    total_drawn = 0
    total_left = 0
    total_overflow = 0

    render_dir = ws / config.STAGE_FOLDERS["render"]
    render_dir.mkdir(parents=True, exist_ok=True)

    for page in manifest.get("pages", []):
        results, report, drawn, left, ovf = _render_single_page(
            page, ws, regions_by_page, render_dir, config, run_id=resolved_run_id
        )
        if report:
            render_results.extend(results)
            page_reports.append(report)
            total_drawn += drawn
            total_left += left
            total_overflow += ovf

    now = datetime.now(timezone.utc).isoformat()

    last_size = None
    for page in reversed(manifest.get("pages", [])):
        if not page.get("skip") and page.get("width") and page.get("height"):
            last_size = (page["width"], page["height"])
            break
    rendered_dir = render_dir / "rendered"
    rendered_dir.mkdir(parents=True, exist_ok=True)
    credits._render_credits_page(rendered_dir, config, last_size)

    output_report = {
        "chapter_id": manifest.get("chapter_id", "unknown"),
        "stage": "render",
        "generated_at": now,
        "font": str(getattr(config, "FONT_PATH", "assets/fonts/ComicNeue-Bold.ttf")),
        "results": render_results,
        "pages": page_reports,
    }

    report_path = render_dir / "render.json"
    with report_path.open("w", encoding="utf-8") as fh:
        json.dump(output_report, fh, ensure_ascii=False, indent=2)

    completed = manifest.get("completed_stages", [])
    if "render" not in completed:
        completed.append("render")
    manifest["completed_stages"] = completed
    manifest["current_stage"] = "qa"
    manifest["updated_at"] = now
    save_manifest(ws, config, manifest)

    for r in render_results:
        if r.get("rendered"):
            from manhua_pipeline.events import PipelineEvent
            from manhua_pipeline.io.event_emitter import get_event_emitter

            get_event_emitter(ws, config, logger=logger).emit(
                PipelineEvent(
                    event=RENDER_REGION_COMPLETED,
                    run_id=resolved_run_id,
                    stage=_STAGE_NAME.lower(),
                    chapter=ws.name,
                    region_id=r["region_id"],
                    page=r.get("page_number"),
                    metadata={
                        "action": r.get("action"),
                        "text_box": r.get("text_box"),
                        "bubble_box": r.get("bubble_box"),
                        "erase_box": r.get("erase_box"),
                        "layout_box": r.get("layout_box"),
                        "erase_strategy": r.get("erase_strategy"),
                        "font_size_pt": r.get("font_size_pt"),
                        "lines": r.get("lines"),
                        "overflow": r.get("overflow"),
                        "placements": r.get("placements"),
                        "bubble_mask_bbox": r.get("bubble_mask_bbox"),
                        "bubble_mask_pixels": r.get("bubble_mask_pixels"),
                        "bubble_mask_area": r.get("bubble_mask_area"),
                        "bubble_mask_coverage": r.get("bubble_mask_coverage"),
                    },
                )
            )

    elapsed = time.monotonic() - t0
    log_stage(
        logger,
        _STAGE_INDEX,
        _TOTAL_STAGES,
        _STAGE_NAME,
        f"done: {len(page_reports)} pages, {total_drawn} drawn, {total_left} left, "
        f"{total_overflow} overflow -> {report_path} (elapsed {elapsed:.1f}s)",
    )
    emit_stage_event(
        ws,
        config,
        _STAGE_NAME.lower(),
        STAGE_COMPLETED,
        logger=logger,
        run_id=resolved_run_id,
        metadata={"total_drawn": total_drawn, "total_left": total_left},
    )
    return report_path
