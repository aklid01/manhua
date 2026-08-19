# Workspace folder creation and chapter manifest operations.

import json
from pathlib import Path

from manhua_pipeline.logging_setup import get_logger

logger = get_logger(__name__)


# Create chapter workspace and required stage subfolders
def ensure_workspace(workspace, config) -> Path:
    ws = Path(workspace)
    for folder in config.STAGE_FOLDERS.values():
        (ws / folder).mkdir(parents=True, exist_ok=True)
    logger.info("Workspace ready at %s", ws)
    return ws


# Return Path to manifest JSON file
def manifest_path(workspace, config) -> Path:
    return Path(workspace) / config.MANIFEST_NAME


# Load and parse chapter manifest JSON file
def load_manifest(workspace, config) -> dict:
    p = manifest_path(workspace, config)
    if not p.exists():
        logger.warning("No manifest at %s (run import first?)", p)
        return {}
    with p.open("r", encoding="utf-8") as fh:
        return json.load(fh)


# Write chapter manifest dictionary to disk
def save_manifest(workspace, config, manifest: dict) -> None:
    p = manifest_path(workspace, config)
    with p.open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=2)
    logger.info("Manifest written to %s", p)
