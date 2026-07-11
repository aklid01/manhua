"""Workspace management: create/resolve the stage folder structure and
read/write the manifest.
"""

import json
from pathlib import Path

from manhua_pipeline.logging_setup import get_logger

logger = get_logger(__name__)


def ensure_workspace(workspace, config) -> Path:
    """Create the workspace and all stage subfolders if missing."""
    ws = Path(workspace)
    for folder in config.STAGE_FOLDERS.values():
        (ws / folder).mkdir(parents=True, exist_ok=True)
    logger.info("Workspace ready at %s", ws)
    return ws


def manifest_path(workspace, config) -> Path:
    return Path(workspace) / config.MANIFEST_NAME


def load_manifest(workspace, config) -> dict:
    p = manifest_path(workspace, config)
    if not p.exists():
        logger.warning("No manifest at %s (run import first?)", p)
        return {}
    with p.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def save_manifest(workspace, config, manifest: dict) -> None:
    p = manifest_path(workspace, config)
    with p.open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=2)
    logger.info("Manifest written to %s", p)
