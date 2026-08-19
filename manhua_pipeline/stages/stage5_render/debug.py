# Rendering debug image export helpers.

from pathlib import Path

from PIL import Image


# Check whether region ID is selected for debug artifact export
def should_debug_region(region_id: str, config) -> bool:
    if not getattr(config, "RENDER_DEBUG_ARTIFACTS", False):
        return False
    selected = set(getattr(config, "RENDER_DEBUG_REGION_IDS", []) or [])
    return not selected or region_id in selected


# Save PIL debug image to specified debug directory
def save_render_debug_image(
    image: Image.Image,
    *,
    debug_dir: Path,
    region_id: str,
    suffix: str,
) -> Path:
    debug_dir.mkdir(parents=True, exist_ok=True)
    path = debug_dir / f"{region_id}_{suffix}.png"
    image.save(path)
    return path
