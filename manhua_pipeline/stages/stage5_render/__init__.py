"""Stage 5: Rendering package.

Erases original Chinese text from bounding box areas and draws the paraphrased
US English text in its place. Utilizes Pillow to wrap, size, and style.
"""

import sys
from types import ModuleType

from manhua_pipeline.stages.stage5_render import (
    credits,
    debug,
    engine,
    geometry,
    inpainting,
    planner,
    types,
    typography,
)
from manhua_pipeline.stages.stage5_render.credits import (
    _fit_font,
    _render_credits_page,
    _sample_bg,
)
from manhua_pipeline.stages.stage5_render.debug import (
    save_render_debug_image,
    should_debug_region,
)
from manhua_pipeline.stages.stage5_render.engine import (
    _load_and_merge_regions,
    _render_region,
    _render_single_page,
    run_render,
)
from manhua_pipeline.stages.stage5_render.geometry import (
    _shrink_box_away_from_siblings,
    box_from_coords,
    box_from_dict,
    box_inside,
    bubble_interior_box,
    clip_box,
    expand_box,
    grow_layout_box,
    intersect_boxes,
    select_render_geometry,
)
from manhua_pipeline.stages.stage5_render.inpainting import (
    _bfs,
    _color_close_mask,
    _count_mask_pixels,
    _dominant_fill_color,
    _estimate_bg,
    _get_bubble_mask,
    _get_radiating_thought_mask,
    _keep_largest_component,
    _solid_rect_mask,
    _text_color_for_bg,
    erase_original_text,
    estimate_text_background,
)
from manhua_pipeline.stages.stage5_render.planner import (
    _draw_region_plan,
    _finalize_plan,
    _plan_and_erase_region,
)
from manhua_pipeline.stages.stage5_render.types import Box, RenderPlan
from manhua_pipeline.stages.stage5_render.typography import (
    _any_line_overflows,
    _block_height,
    _clamp_placements_to_mask,
    _fit_text,
    _font_cache,
    _get_available_width_at_y,
    _line_ink_width,
    _load_font,
    _wrap_text,
    effective_max_pt,
    fit_and_place_text,
    normalize_escaped_newlines,
)

_SUBMODULE_MAP = {
    "_font_cache": typography,
    "_load_font": typography,
    "effective_max_pt": typography,
    "normalize_escaped_newlines": typography,
    "_wrap_text": typography,
    "_block_height": typography,
    "_line_ink_width": typography,
    "_get_available_width_at_y": typography,
    "_any_line_overflows": typography,
    "_fit_text": typography,
    "fit_and_place_text": typography,
    "_clamp_placements_to_mask": typography,
    "estimate_text_background": inpainting,
    "_count_mask_pixels": inpainting,
    "erase_original_text": inpainting,
    "_estimate_bg": inpainting,
    "_text_color_for_bg": inpainting,
    "_bfs": inpainting,
    "_keep_largest_component": inpainting,
    "_dominant_fill_color": inpainting,
    "_color_close_mask": inpainting,
    "_solid_rect_mask": inpainting,
    "_get_bubble_mask": inpainting,
    "_get_radiating_thought_mask": inpainting,
    "_plan_and_erase_region": planner,
    "_draw_region_plan": planner,
    "_finalize_plan": planner,
    "Box": types,
    "RenderPlan": types,
    "box_from_dict": geometry,
    "box_from_coords": geometry,
    "clip_box": geometry,
    "expand_box": geometry,
    "box_inside": geometry,
    "intersect_boxes": geometry,
    "bubble_interior_box": geometry,
    "_shrink_box_away_from_siblings": geometry,
    "grow_layout_box": geometry,
    "select_render_geometry": geometry,
    "should_debug_region": debug,
    "save_render_debug_image": debug,
    "_fit_font": credits,
    "_sample_bg": credits,
    "_render_credits_page": credits,
    "_render_region": engine,
    "_load_and_merge_regions": engine,
    "_render_single_page": engine,
    "run_render": engine,
}


class _Stage5Module(ModuleType):
    def __setattr__(self, name, value):
        super().__setattr__(name, value)
        if name in _SUBMODULE_MAP:
            setattr(_SUBMODULE_MAP[name], name, value)


sys.modules[__name__].__class__ = _Stage5Module

__all__ = [
    "Box",
    "RenderPlan",
    "box_from_dict",
    "box_from_coords",
    "clip_box",
    "expand_box",
    "box_inside",
    "intersect_boxes",
    "bubble_interior_box",
    "_shrink_box_away_from_siblings",
    "grow_layout_box",
    "select_render_geometry",
    "estimate_text_background",
    "_count_mask_pixels",
    "erase_original_text",
    "_estimate_bg",
    "_text_color_for_bg",
    "_bfs",
    "_keep_largest_component",
    "_dominant_fill_color",
    "_color_close_mask",
    "_solid_rect_mask",
    "_get_bubble_mask",
    "_get_radiating_thought_mask",
    "_font_cache",
    "_load_font",
    "effective_max_pt",
    "normalize_escaped_newlines",
    "_wrap_text",
    "_block_height",
    "_line_ink_width",
    "_get_available_width_at_y",
    "_any_line_overflows",
    "_fit_text",
    "fit_and_place_text",
    "_clamp_placements_to_mask",
    "_plan_and_erase_region",
    "_draw_region_plan",
    "_finalize_plan",
    "should_debug_region",
    "save_render_debug_image",
    "_fit_font",
    "_sample_bg",
    "_render_credits_page",
    "_render_region",
    "_load_and_merge_regions",
    "_render_single_page",
    "run_render",
]
