"""Stage 7: Packaging. Bundle rendered pages into zip/cbz/tar/pdf.
Opt-in; terminal; never advances the manifest."""

import re
import tarfile
import tempfile
import time
import zipfile
from pathlib import Path

from PIL import Image

from manhua_pipeline.io.workspace import load_manifest
from manhua_pipeline.logging_setup import get_logger, log_stage

logger = get_logger(__name__)
_STAGE_INDEX, _TOTAL_STAGES, _STAGE_NAME = 7, 7, "Package"


def _natural_key(p: Path) -> tuple:
    nums = tuple(int(n) for n in re.findall(r"\d+", p.name))
    return nums if nums else (float("inf"),)


def _collect(render_dir: Path, config) -> list[Path]:
    exts = getattr(config, "PACKAGE_IMAGE_EXTS", (".png", ".jpg", ".jpeg", ".webp"))
    rendered_dir = render_dir / "rendered"
    scan_dir = rendered_dir if rendered_dir.exists() else render_dir
    return sorted(
        (p for p in scan_dir.iterdir() if p.is_file() and p.suffix.lower() in exts),
        key=_natural_key,
    )


def _is_credits(path: Path, config) -> bool:
    if not getattr(config, "PACKAGE_CREDITS_ENABLED", True):
        return False
    pattern = getattr(
        config, "PACKAGE_CREDITS_PATTERN", r"^zzz_credits(?:\.[^.]+)?$"
    )
    try:
        return re.fullmatch(pattern, path.name, flags=re.IGNORECASE) is not None
    except re.error:
        return path.name.lower() == "zzz_credits.png"


def _image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        return image.size


def _group_height(group: list[Path], sizes: dict[Path, tuple[int, int]]) -> int:
    return sum(sizes[path][1] for path in group)


def _build_groups(images: list[Path], config) -> list[list[Path]]:
    """Build bounded vertical groups and attach credits to the final group."""
    sizes = {path: _image_size(path) for path in images}
    budget = max(1, int(getattr(config, "PACKAGE_PIXEL_BUDGET", 6000)))
    hard_max = max(
        budget,
        int(getattr(config, "PACKAGE_MAX_COMPOSITE_HEIGHT", 12000)),
    )
    credits = [path for path in images if _is_credits(path, config)]
    content = [path for path in images if path not in credits]

    groups: list[list[Path]] = []
    current: list[Path] = []
    current_height = 0
    for path in content:
        _, height = sizes[path]
        if height > budget:
            logger.warning(
                "[%s] Page %s is %dpx tall and exceeds the package budget of %dpx; keeping it alone.",
                _STAGE_NAME,
                path.name,
                height,
                budget,
            )
        if current and current_height + height > budget:
            groups.append(current)
            current = []
            current_height = 0
        current.append(path)
        current_height += height
        if current_height >= budget or height > budget:
            groups.append(current)
            current = []
            current_height = 0
    if current:
        groups.append(current)

    if not credits:
        return groups
    if not groups:
        return [credits]

    credit_height = _group_height(credits, sizes)
    final_group = groups[-1]
    if _group_height(final_group, sizes) + credit_height <= budget:
        final_group.extend(credits)
        return groups

    # Preserve the normal budget by moving only the final content page into a
    # new protected group, matching e.g. [39,40,41] -> [39,40], [41,credits].
    last_page = final_group[-1]
    _, last_height = sizes[last_page]
    if last_height + credit_height <= budget and last_height + credit_height <= hard_max:
        final_group.pop()
        if not final_group:
            groups.pop()
        groups.append([last_page, *credits])
        return groups

    logger.warning(
        "[%s] Credits (%dpx) cannot safely attach to final page %s within the %dpx budget; keeping credits separate.",
        _STAGE_NAME,
        credit_height,
        last_page.name,
        budget,
    )
    groups.append(credits)
    return groups


def _compose_group(group: list[Path], out: Path) -> None:
    """Vertically concatenate a group without scaling or cropping."""
    frames = []
    try:
        for path in group:
            with Image.open(path) as image:
                frames.append(image.convert("RGB"))
        width = max(frame.width for frame in frames)
        height = sum(frame.height for frame in frames)
        canvas = Image.new("RGB", (width, height), (255, 255, 255))
        y = 0
        for frame in frames:
            canvas.paste(frame, (0, y))
            y += frame.height
        canvas.save(out, "PNG")
        canvas.close()
    finally:
        for frame in frames:
            frame.close()


def _materialize_groups(groups: list[list[Path]], directory: Path) -> list[Path]:
    outputs = []
    for index, group in enumerate(groups, start=1):
        out = directory / f"{index:03d}.png"
        _compose_group(group, out)
        outputs.append(out)
    return outputs


def _base_name(ws: Path, manifest: dict) -> str:
    for key in ("source_archive", "source_filename", "original_name"):
        if manifest.get(key):
            return Path(manifest[key]).stem
    return ws.name


def _write_zip(images, out):
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        for img in images:
            zf.write(img, arcname=img.name)


def _write_tar(images, out):
    with tarfile.open(out, "w") as tf:
        for img in images:
            tf.add(img, arcname=img.name)


def _write_pdf(images, out):
    if not images:
        return
    frames = [Image.open(p).convert("RGB") for p in images]
    frames[0].save(out, format="PDF", save_all=True, append_images=frames[1:])
    for f in frames:
        f.close()


_WRITERS = {
    "zip": ("zip", _write_zip),
    "cbz": ("cbz", _write_zip),
    "tar": ("tar", _write_tar),
    "pdf": ("pdf", _write_pdf),
}


def run_package(workspace: str, config, formats: list[str]) -> list[Path]:
    t0 = time.monotonic()
    ws = Path(workspace)
    log_stage(logger, _STAGE_INDEX, _TOTAL_STAGES, _STAGE_NAME, "starting")
    from manhua_pipeline.events import STAGE_COMPLETED, STAGE_STARTED
    from manhua_pipeline.io.event_emitter import emit_stage_event

    emit_stage_event(ws, config, _STAGE_NAME.lower(), STAGE_STARTED, logger=logger)

    valid = set(getattr(config, "VALID_PACKAGE_FORMATS", ("zip", "cbz", "tar", "pdf")))
    requested = [f.lower().strip() for f in formats if f.strip()]
    for u in [f for f in requested if f not in valid]:
        logger.warning("[%s] Unknown format ignored: %r", _STAGE_NAME, u)
    requested = [f for f in requested if f in valid]
    if not requested:
        logger.info("[%s] No valid formats — nothing to package.", _STAGE_NAME)
        return []

    manifest = load_manifest(workspace, config)
    if not manifest:
        raise ValueError("Manifest not found. Run the pipeline first.")

    render_dir = ws / config.STAGE_FOLDERS["render"]
    if not render_dir.exists():
        raise FileNotFoundError("render output not found. Run render first.")

    source_images = _collect(render_dir, config)
    if not source_images:
        logger.warning("[%s] No rendered images in %s.", _STAGE_NAME, render_dir)
        return []

    groups = _build_groups(source_images, config)

    pkg_dir = ws / config.STAGE_FOLDERS["package"]
    pkg_dir.mkdir(parents=True, exist_ok=True)
    base = _base_name(ws, manifest)

    written = []
    with tempfile.TemporaryDirectory(prefix=".package_pages_", dir=pkg_dir) as tmp:
        package_images = _materialize_groups(groups, Path(tmp))
        for fmt in dict.fromkeys(requested):
            ext, writer = _WRITERS[fmt]
            out = pkg_dir / f"{base}.{ext}"
            try:
                writer(package_images, out)
                written.append(out)
                logger.info(
                    "[%s] Wrote %s (%d package pages from %d rendered pages)",
                    _STAGE_NAME,
                    out.name,
                    len(package_images),
                    len(source_images),
                )
            except Exception as exc:
                logger.error("[%s] Failed %s: %s", _STAGE_NAME, out.name, exc)

    log_stage(
        logger,
        _STAGE_INDEX,
        _TOTAL_STAGES,
        _STAGE_NAME,
        f"done: {len(written)} archive(s) -> {pkg_dir} (elapsed {time.monotonic() - t0:.1f}s)",
    )
    emit_stage_event(
        ws,
        config,
        _STAGE_NAME.lower(),
        STAGE_COMPLETED,
        logger=logger,
        metadata={
            "packages_written": len(written),
            "source_pages": len(source_images),
            "package_pages": len(groups),
        },
    )
    return written
