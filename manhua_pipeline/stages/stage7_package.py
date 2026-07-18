"""Stage 7: Packaging. Bundle rendered pages into zip/cbz/tar/pdf.
Opt-in; terminal; never advances the manifest."""

import re
import tarfile
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

    images = _collect(render_dir, config)
    if not images:
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
            writer(images, out)
            written.append(out)
            logger.info("[%s] Wrote %s (%d pages)", _STAGE_NAME, out.name, len(images))
        except Exception as exc:
            logger.error("[%s] Failed %s: %s", _STAGE_NAME, out.name, exc)

    log_stage(
        logger, _STAGE_INDEX, _TOTAL_STAGES, _STAGE_NAME,
        f"done: {len(written)} archive(s) -> {pkg_dir} (elapsed {time.monotonic()-t0:.1f}s)",
    )
    return written
