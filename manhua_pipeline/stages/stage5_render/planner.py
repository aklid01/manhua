from pathlib import Path

from PIL import Image, ImageDraw

from manhua_pipeline.logging_setup import get_logger
from manhua_pipeline.stages.stage5_render import debug, geometry, inpainting, typography
from manhua_pipeline.stages.stage5_render.types import Box, RenderPlan

logger = get_logger(__name__)

_STAGE_INDEX = 5
_TOTAL_STAGES = 7
_STAGE_NAME = "Render"


def _plan_and_erase_region(
    region: dict,
    page_img: Image.Image,
    draw: ImageDraw.ImageDraw,
    page_num: int,
    config,
    *,
    render_dir: Path | None = None,
    sibling_text_boxes: list[Box] | None = None,
) -> RenderPlan:
    rid = region["region_id"]
    render_flag = region.get("render", True)
    has_usable = region.get("has_usable_text", False)
    final_text = region.get("final_text", "")
    style_hint = region.get("style_hint", "round")
    register = region.get("register", "neutral")
    glossary_conflict = region.get("glossary_conflict", False)

    def _left(action: str) -> RenderPlan:
        return RenderPlan(
            region_id=rid,
            page_number=page_num,
            rendered=False,
            action=action,
            register=register,
            style_hint=style_hint,
            glossary_conflict=glossary_conflict,
            page_width=page_img.width,
            page_height=page_img.height,
        )

    erase_only = bool(region.get("erase_only"))
    if not render_flag and not erase_only:
        return _left("left_original_not_render_type")
    if not has_usable:
        logger.info(
            "[%d/%d %s] Page %d %s -> left original (no usable text)",
            _STAGE_INDEX,
            _TOTAL_STAGES,
            _STAGE_NAME,
            page_num,
            rid,
        )
        return _left("left_original_no_text")
    if not final_text:
        logger.warning(
            "[%d/%d %s] Page %d %s -> left original (missing final_text)",
            _STAGE_INDEX,
            _TOTAL_STAGES,
            _STAGE_NAME,
            page_num,
            rid,
        )
        return _left("missing_text")

    text_box, bubble_box, _, _ = geometry.select_render_geometry(
        region,
        page_img.size,
        config,
        sibling_text_boxes=sibling_text_boxes,
    )

    local_text_box = Box(
        x=text_box.x - bubble_box.x,
        y=text_box.y - bubble_box.y,
        w=text_box.w,
        h=text_box.h,
    )
    bubble_crop = page_img.crop(bubble_box.as_xyxy())

    if getattr(config, "BUBBLE_FILL_MODE", "white") == "dominant":
        fill_color = inpainting._dominant_fill_color(
            bubble_crop, local_text_box, config
        )
    else:
        fill_color = None

    yolo_mask = inpainting.load_region_bubble_mask(region, bubble_box)
    if yolo_mask is not None:
        generic_bubble_mask = yolo_mask
        bubble_mask_source = "yolo_seg"
    else:
        generic_bubble_mask = inpainting._get_bubble_mask(
            bubble_crop,
            bubble_box.x,
            bubble_box.y,
            bubble_box.w,
            bubble_box.h,
            page_img.width,
            page_img.height,
            config,
            seed_box=local_text_box,
            fill_color=fill_color,
        )
        bubble_mask_source = "generic_fill"
    thought_bubble_mask = inpainting._get_radiating_thought_mask(
        bubble_crop,
        generic_bubble_mask,
        config,
        seed_box=local_text_box,
    )
    bubble_mask = (
        thought_bubble_mask if thought_bubble_mask is not None else generic_bubble_mask
    )
    bubble_mask_source = (
        "thought_radiating_inner"
        if thought_bubble_mask is not None
        else bubble_mask_source
    )

    shared_parent = False
    pb = region.get("parent_bubble")
    if pb and sibling_text_boxes:
        pbx0, pby0 = pb["x"], pb["y"]
        pbx1, pby1 = pb["x"] + pb["w"], pb["y"] + pb["h"]
        for sib in sibling_text_boxes:
            if sib.x >= pbx0 and sib.y >= pby0 and sib.x2 <= pbx1 and sib.y2 <= pby1:
                shared_parent = True
                break

    text_box, bubble_box, erase_box, layout_box = geometry.select_render_geometry(
        region,
        page_img.size,
        config,
        bubble_mask=None if shared_parent else bubble_mask,
        bubble_box=bubble_box,
        sibling_text_boxes=sibling_text_boxes,
    )

    erase_text_box = text_box
    preserved_erase_box = region.get("_erase_text_box")
    stitched_erase = bool(preserved_erase_box)
    if preserved_erase_box:
        erase_text_box = geometry.clip_box(
            geometry.box_from_dict(preserved_erase_box), page_img.size
        )
        _, _, erase_box, _ = geometry.select_render_geometry(
            region,
            page_img.size,
            config,
            bubble_mask=None,
            bubble_box=bubble_box,
            text_box_override=erase_text_box,
        )

    erase_bubble_mask = bubble_mask
    if stitched_erase:
        # A stitched text box can include source glyphs that the radiating
        # bubble mask classifies as part of the ray ring. The full OCR box is
        # the authoritative erase area; keep the measured mask for layout but
        # use a padded text rectangle for this seam-crossing erase.
        erase_bubble_mask = inpainting._solid_rect_mask(bubble_box, erase_box)

    min_cov = float(getattr(config, "BUBBLE_FILL_MIN_COVERAGE", 0.40))
    mask_hist = bubble_mask.histogram()
    mask_pixels = mask_hist[255] if len(mask_hist) > 255 else 0
    crop_area = max(1, bubble_box.w * bubble_box.h)
    coverage = mask_pixels / crop_area

    if coverage < min_cov and thought_bubble_mask is None:
        logger.warning(
            "[%d/%d %s] Page %d %s -> low fill-mask coverage (%.3f); "
            "falling back to padded text-box erase.",
            _STAGE_INDEX,
            _TOTAL_STAGES,
            _STAGE_NAME,
            page_num,
            rid,
            coverage,
        )
        bubble_mask = inpainting._solid_rect_mask(bubble_box, erase_box)
        bubble_mask_source = "text_rect_fallback"
        fill_color = (255, 255, 255)

    def _is_dark(c: tuple[int, int, int]) -> bool:
        return 0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2] < 96

    if fill_color is not None and _is_dark(fill_color) and coverage < min_cov:
        logger.warning(
            "[%d/%d %s] Page %d %s -> dark fill %s at low coverage %.2f; "
            "falling back to white.",
            _STAGE_INDEX,
            _TOTAL_STAGES,
            _STAGE_NAME,
            page_num,
            rid,
            fill_color,
            coverage,
        )
        fill_color = (255, 255, 255)

    debug_dir = (render_dir / "debug") if render_dir else None
    do_debug = debug_dir and debug.should_debug_region(rid, config)
    debug_paths = {}

    if do_debug:
        debug_paths["before"] = str(
            debug.save_render_debug_image(
                page_img.crop(erase_box.as_xyxy()),
                debug_dir=debug_dir,
                region_id=rid,
                suffix="before",
            )
        )
        debug_paths["bubble_crop"] = str(
            debug.save_render_debug_image(
                bubble_crop,
                debug_dir=debug_dir,
                region_id=rid,
                suffix="bubble_crop",
            )
        )
        debug_paths["bubble_mask"] = str(
            debug.save_render_debug_image(
                bubble_mask,
                debug_dir=debug_dir,
                region_id=rid,
                suffix="bubble_mask",
            )
        )

    erase_meta = inpainting.erase_original_text(
        page_img=page_img,
        draw=draw,
        text_box=erase_text_box,
        bubble_box=bubble_box,
        erase_box=erase_box,
        bubble_mask=erase_bubble_mask,
        config=config,
        fill_color=fill_color,
    )

    if erase_meta.get("low_overlap"):
        logger.warning(
            "[%d/%d %s] Page %d %s -> erase mask low overlap (%.2f); %s fallback.",
            _STAGE_INDEX,
            _TOTAL_STAGES,
            _STAGE_NAME,
            page_num,
            rid,
            erase_meta.get("clip_retention_ratio", 0.0),
            "using image inpaint"
            if erase_meta.get("repaired_with_image_inpaint")
            else "retrying with solid rect",
        )
        fallback_mask = inpainting._solid_rect_mask(bubble_box, erase_box)
        if erase_meta.get("repaired_with_image_inpaint"):
            # The original color mask is unsafe for placement too: its holes
            # are what forced text toward the right edge. Keep the repaired
            # image, but give typography the full parent-panel rectangle as a
            # neutral layout mask.
            bubble_mask = inpainting._solid_rect_mask(bubble_box, bubble_box)
            bubble_mask_source = "image_inpaint_layout"
        else:
            erase_meta = inpainting.erase_original_text(
                page_img=page_img,
                draw=draw,
                text_box=erase_text_box,
                bubble_box=bubble_box,
                erase_box=erase_box,
                bubble_mask=fallback_mask,
                config=config,
                fill_color=fill_color or (255, 255, 255),
            )
            bubble_mask = fallback_mask
            bubble_mask_source = "text_rect_fallback"
        _, _, _, layout_box = geometry.select_render_geometry(
            region,
            page_img.size,
            config,
            bubble_mask=bubble_mask,
            bubble_box=bubble_box,
            sibling_text_boxes=sibling_text_boxes,
        )
        _, _, erase_box, _ = geometry.select_render_geometry(
            region,
            page_img.size,
            config,
            bubble_mask=None,
            bubble_box=bubble_box,
            text_box_override=erase_text_box,
        )

    if do_debug:
        debug_paths["text_mask"] = str(
            debug.save_render_debug_image(
                erase_meta["text_mask"],
                debug_dir=debug_dir,
                region_id=rid,
                suffix="text_mask",
            )
        )
        debug_paths["erase_mask_clipped"] = str(
            debug.save_render_debug_image(
                erase_meta["erase_mask"],
                debug_dir=debug_dir,
                region_id=rid,
                suffix="erase_mask_clipped",
            )
        )
        debug_paths["after_erase"] = str(
            debug.save_render_debug_image(
                page_img.crop(erase_box.as_xyxy()),
                debug_dir=debug_dir,
                region_id=rid,
                suffix="after_erase",
            )
        )

    if erase_only:
        return RenderPlan(
            region_id=rid,
            page_number=page_num,
            rendered=False,
            action=region.get("render_skip_reason") or "erase_only",
            register=register,
            style_hint=style_hint,
            glossary_conflict=glossary_conflict,
            page_width=page_img.width,
            page_height=page_img.height,
            text_box=text_box.as_dict(),
            erase_text_box=erase_text_box.as_dict(),
            erase_box=erase_box.as_dict(),
            bubble_box=bubble_box.as_dict(),
            erase_strategy=erase_meta.get("erase_strategy", ""),
            debug_paths=debug_paths,
        )

    plan = RenderPlan(
        region_id=rid,
        page_number=page_num,
        rendered=True,
        action="drew",
        register=register,
        style_hint=style_hint,
        glossary_conflict=glossary_conflict,
        page_width=page_img.width,
        page_height=page_img.height,
        text_box=text_box.as_dict(),
        erase_text_box=erase_text_box.as_dict(),
        layout_box=layout_box.as_dict(),
        erase_box=erase_box.as_dict(),
        bubble_box=bubble_box.as_dict(),
        erase_strategy=(
            "thought_bubble_inner_mask"
            if bubble_mask_source == "thought_radiating_inner" and not stitched_erase
            else erase_meta.get("erase_strategy", "")
        ),
        debug_paths=debug_paths,
    )

    if erase_meta.get("low_overlap"):
        plan.emit_low_overlap = True
        plan.low_overlap_retention = erase_meta.get("clip_retention_ratio", 0.0)

    mask_bbox = bubble_mask.getbbox()
    mask_hist2 = bubble_mask.histogram()
    mask_px = mask_hist2[255] if len(mask_hist2) > 255 else 0
    mask_area = bubble_box.w * bubble_box.h
    plan.bubble_mask_bbox = list(mask_bbox) if mask_bbox else None
    plan.bubble_mask_pixels = mask_px
    plan.bubble_mask_area = mask_area
    plan.bubble_mask_coverage = (mask_px / mask_area) if mask_area else 0.0
    plan.bubble_mask_source = bubble_mask_source

    text_bg = fill_color or inpainting.estimate_text_background(
        page_img, text_box, bubble_box
    )
    plan.text_fill = inpainting._text_color_for_bg(text_bg)
    plan.text_bg = text_bg

    text_to_draw = final_text
    if (register == "rude" or style_hint == "spiky") and getattr(
        config, "EMPHASIS_UPPERCASE", True
    ):
        text_to_draw = final_text.upper()

    font_path = getattr(config, "FONT_PATH", "assets/fonts/ComicNeue-Bold.ttf")
    font, lines, font_size, overflow, placements = typography.fit_and_place_text(
        draw=draw,
        final_text=text_to_draw,
        font_path=font_path,
        layout_box=layout_box,
        config=config,
        page_width=page_img.width,
        bubble_mask=bubble_mask,
        bubble_box=bubble_box,
    )

    erode_px = int(getattr(config, "LAYOUT_INTERIOR_ERODE_PX", 10))
    placements = typography._clamp_placements_to_mask(
        placements, bubble_mask, bubble_box, erode_px=erode_px
    )

    plan.font = font
    plan.placements = placements
    plan.font_size_pt = font_size
    plan.lines = len(lines)
    plan.overflow = overflow
    plan.effective_max_pt = typography.effective_max_pt(config, page_img.width)
    plan.readable_min_pt = int(getattr(config, "FONT_MIN_PT", 12))
    plan.abs_min_pt = int(getattr(config, "FONT_ABS_MIN_PT", 9))

    layout_padding = int(getattr(config, "TEXT_LAYOUT_PADDING_PX", 8))
    base_w = max(5, text_box.w - 2 * layout_padding)
    base_h = max(5, text_box.h - 2 * layout_padding)
    plan.layout_expanded = layout_box.w * layout_box.h > base_w * base_h
    plan.layout_growth_factor = round(
        layout_box.w * layout_box.h / max(1, base_w * base_h), 2
    )

    if font_size < plan.readable_min_pt or overflow:
        plan.emit_too_small = True

    return plan


def _draw_region_plan(
    plan: RenderPlan,
    draw: ImageDraw.ImageDraw,
    render_dir: Path | None = None,
) -> None:
    if not plan.rendered or plan.font is None:
        return
    for p in plan.placements:
        draw.text(
            (p["x"], p["y"]), p["text"], fill=tuple(plan.text_fill), font=plan.font
        )

    if plan.debug_paths and render_dir:
        debug_dir = render_dir / "debug"
        erase_box = Box(**plan.erase_box)
        debug.save_render_debug_image(
            draw._image.crop(erase_box.as_xyxy()),  # type: ignore[attr-defined]
            debug_dir=debug_dir,
            region_id=plan.region_id,
            suffix="final",
        )


def _finalize_plan(
    plan: RenderPlan,
    *,
    emitter,
    run_id: str,
    chapter: str,
    render_dir,
    config,
) -> tuple[dict, int, int, int]:
    if not plan.rendered:
        result = {
            "region_id": plan.region_id,
            "page_number": plan.page_number,
            "rendered": False,
            "action": plan.action,
            "font_size_pt": None,
            "lines": 0,
            "overflow": False,
            "overflow_ask_llm": False,
            "register": plan.register,
            "style_hint": plan.style_hint,
            "glossary_conflict": plan.glossary_conflict,
        }
        return result, 0, 1, 0

    from manhua_pipeline.io.event_emitter import get_event_emitter

    em = emitter or (
        get_event_emitter(render_dir.parent, config, logger=logger)
        if render_dir
        else None
    )

    if plan.emit_low_overlap and em:
        from manhua_pipeline.events import RENDER_ERASE_MASK_LOW_OVERLAP, PipelineEvent

        min_retention = float(getattr(config, "TEXT_ERASE_MIN_RETENTION", 0.65))
        logger.warning(
            "[%d/%d %s] Page %d %s -> Low erase mask retention ratio (%.2f < %.2f)",
            _STAGE_INDEX,
            _TOTAL_STAGES,
            _STAGE_NAME,
            plan.page_number,
            plan.region_id,
            plan.low_overlap_retention,
            min_retention,
        )
        em.emit(
            PipelineEvent(
                event=RENDER_ERASE_MASK_LOW_OVERLAP,
                run_id=run_id,
                stage=_STAGE_NAME.lower(),
                chapter=chapter,
                region_id=plan.region_id,
                page=plan.page_number,
                metadata={
                    "retention": plan.low_overlap_retention,
                    "min_retention": min_retention,
                },
            )
        )

    if plan.emit_too_small and em:
        from manhua_pipeline.events import RENDER_TEXT_TOO_SMALL, PipelineEvent

        logger.warning(
            "[%d/%d %s] Page %d %s -> text below readable floor (pt=%d, min=%d, overflow=%s)",
            _STAGE_INDEX,
            _TOTAL_STAGES,
            _STAGE_NAME,
            plan.page_number,
            plan.region_id,
            plan.font_size_pt,
            plan.readable_min_pt,
            plan.overflow,
        )
        em.emit(
            PipelineEvent(
                event=RENDER_TEXT_TOO_SMALL,
                run_id=run_id,
                stage=_STAGE_NAME.lower(),
                chapter=chapter,
                region_id=plan.region_id,
                page=plan.page_number,
                level="WARNING",
                metadata={
                    "font_size_pt": plan.font_size_pt,
                    "readable_min_pt": plan.readable_min_pt,
                    "abs_min_pt": plan.abs_min_pt,
                    "effective_max_pt": plan.effective_max_pt,
                    "page_width": plan.page_width,
                    "page_height": plan.page_height,
                    "overflow": plan.overflow,
                    "layout_box": plan.layout_box,
                    "text_box": plan.text_box,
                },
            )
        )
    else:
        logger.info(
            "[%d/%d %s] Page %d %s -> drew (%dpt, %d lines, erase=%s)",
            _STAGE_INDEX,
            _TOTAL_STAGES,
            _STAGE_NAME,
            plan.page_number,
            plan.region_id,
            plan.font_size_pt,
            plan.lines,
            plan.erase_strategy,
        )

    result: dict = {
        "region_id": plan.region_id,
        "page_number": plan.page_number,
        "rendered": True,
        "action": "drew",
        "font_size_pt": plan.font_size_pt,
        "effective_max_pt": plan.effective_max_pt,
        "page_width": plan.page_width,
        "page_height": plan.page_height,
        "lines": plan.lines,
        "readable_min_pt": plan.readable_min_pt,
        "abs_min_pt": plan.abs_min_pt,
        "overflow": plan.overflow,
        "overflow_ask_llm": plan.overflow,
        "register": plan.register,
        "style_hint": plan.style_hint,
        "glossary_conflict": plan.glossary_conflict,
        "erase_strategy": plan.erase_strategy,
        "text_box": plan.text_box,
        "erase_text_box": plan.erase_text_box,
        "bubble_box": plan.bubble_box,
        "erase_box": plan.erase_box,
        "layout_box": plan.layout_box,
        "layout_expanded": plan.layout_expanded,
        "layout_growth_factor": plan.layout_growth_factor,
        "text_fill": list(plan.text_fill),
        "text_background": list(plan.text_bg),
        "placements": plan.placements,
        "bubble_mask_bbox": plan.bubble_mask_bbox,
        "bubble_mask_pixels": plan.bubble_mask_pixels,
        "bubble_mask_area": plan.bubble_mask_area,
        "bubble_mask_coverage": plan.bubble_mask_coverage,
        "bubble_mask_source": plan.bubble_mask_source,
    }
    if plan.debug_paths:
        result["debug_paths"] = plan.debug_paths
    return result, 1, 0, (1 if plan.overflow else 0)
