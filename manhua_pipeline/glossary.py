"""Glossary load/save and auto-seeding from name_label / scene_text regions."""

import json
from pathlib import Path

from manhua_pipeline.logging_setup import get_logger

logger = get_logger(__name__)


def glossary_path(workspace, config) -> Path:
    return Path(workspace) / config.GLOSSARY_NAME


def load_glossary(workspace, config) -> dict:
    p = glossary_path(workspace, config)
    if not p.exists():
        logger.info("No glossary yet; starting empty")
        return {"version": "v1", "terms": []}
    with p.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def save_glossary(workspace, config, glossary: dict) -> None:
    p = glossary_path(workspace, config)
    with p.open("w", encoding="utf-8") as fh:
        json.dump(glossary, fh, ensure_ascii=False, indent=2)
    logger.info("Glossary written to %s (%d terms)", p, len(glossary.get("terms", [])))


def auto_seed_from_regions(glossary: dict, ocr_results: list) -> dict:
    """Seed glossary terms from name_label / scene_text OCR results."""
    # TODO: for each name_label/scene_text, add a candidate term if not present.
    logger.info("Auto-seed: TODO (name_label / scene_text -> glossary terms)")
    return glossary
