"""Stage 7: Packaging. Bundle rendered pages into zip/cbz archives.
Opt-in; terminal; never advances the manifest."""

import re
import time
import zipfile
from pathlib import Path

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


def _base_name(ws: Path, manifest: dict) -> str:
    for key in ("source_archive", "source_filename", "original_name"):
        if manifest.get(key):
            return Path(manifest[key]).stem
    return ws.name


def _write_zip(images: list[Path], out: Path) -> None:
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        for img in images:
            zf.write(img, arcname=img.name)


_WRITERS = {
    "zip": ("zip", _write_zip),
    "cbz": ("cbz", _write_zip),
}


def run_package(workspace: str, config, formats: list[str]) -> list[Path]:
    t0 = time.monotonic()
    ws = Path(workspace)
    log_stage(logger, _STAGE_INDEX, _TOTAL_STAGES, _STAGE_NAME, "starting")
    from manhua_pipeline.events import STAGE_COMPLETED, STAGE_STARTED
    from manhua_pipeline.io.event_emitter import emit_stage_event

    emit_stage_event(ws, config, _STAGE_NAME.lower(), STAGE_STARTED, logger=logger)

    valid = set(getattr(config, "VALID_PACKAGE_FORMATS", ("zip", "cbz")))
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

    pkg_dir = ws / config.STAGE_FOLDERS["package"]
    pkg_dir.mkdir(parents=True, exist_ok=True)
    base = _base_name(ws, manifest)

    written = []
    for fmt in dict.fromkeys(requested):
        ext, writer = _WRITERS[fmt]
        out = pkg_dir / f"{base}.{ext}"
        try:
            writer(source_images, out)
            written.append(out)
            logger.info(
                "[%s] Wrote %s (%d pages)",
                _STAGE_NAME,
                out.name,
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
        },
    )
    return written
