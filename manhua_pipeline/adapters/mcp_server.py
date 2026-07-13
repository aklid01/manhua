# ruff: noqa: E402
import json
import sys
from pathlib import Path

# Add project root to path so config and manhua_pipeline packages are discoverable
root_dir = Path(__file__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from fastmcp import FastMCP

import config
from manhua_pipeline.io.settings import get_output_dir
from manhua_pipeline.logging_setup import get_logger, setup_logging
from manhua_pipeline.stages.stage3_translation import (
    build_translation_bundle,
    write_translation_response,
)
from manhua_pipeline.stages.stage4_paraphrase import (
    build_paraphrase_bundle,
    write_paraphrase_response,
)

# CRITICAL: setup_logging with stream="stderr" so stdout is reserved for JSON-RPC
setup_logging(stream="stderr")
logger = get_logger(__name__)

mcp = FastMCP("Manhua Pipeline")


def _base() -> Path:
    output_dir = get_output_dir()
    if not output_dir:
        raise ValueError(
            "Series base directory is not set. Run with --output-dir <path> first."
        )
    return Path(output_dir)


def _check_chapter_pending(item: Path) -> str | None:
    manifest_path = item / "manifest.json"
    if not manifest_path.exists():
        return None
    try:
        with manifest_path.open("r", encoding="utf-8") as fh:
            manifest = json.load(fh)
        stage = manifest.get("current_stage")
        if stage == "translate":
            prompt = (
                item
                / config.STAGE_FOLDERS["translation"]
                / config.TRANSLATION_PROMPT_NAME
            )
            resp = (
                item
                / config.STAGE_FOLDERS["translation"]
                / config.TRANSLATION_RESPONSE_NAME
            )
            if prompt.exists() and not resp.exists():
                return "translate"
        elif stage == "paraphrase":
            prompt = (
                item
                / config.STAGE_FOLDERS["paraphrase"]
                / config.PARAPHRASE_PROMPT_NAME
            )
            resp = (
                item
                / config.STAGE_FOLDERS["paraphrase"]
                / config.PARAPHRASE_RESPONSE_NAME
            )
            if prompt.exists() and not resp.exists():
                return "paraphrase"
    except Exception:
        pass
    return None


@mcp.tool()
def list_pending() -> list[dict]:
    """List chapters currently awaiting manual translation or paraphrase handoff."""
    try:
        base = _base()
    except Exception as exc:
        return [{"error": str(exc)}]

    if not base.exists() or not base.is_dir():
        return []

    pending = []
    for item in base.iterdir():
        if item.is_dir():
            stage_awaiting = _check_chapter_pending(item)
            if stage_awaiting:
                pending.append({"chapter": item.name, "stage_awaiting": stage_awaiting})
    return pending


@mcp.tool()
def get_translation_bundle(chapter: str) -> dict:
    """Return lines to translate for a chapter.
    TRANSLATE each Chinese line faithfully and literally into US English.
    Preserve meaning, names, and terminology exactly. Do not paraphrase or localize slang.
    Apply the provided glossary terms exactly. Then call submit_translation(chapter, {region_id: translation})."""
    try:
        base = _base()
        ch_dir = base / chapter
        if not ch_dir.exists():
            return {"error": f"Chapter folder '{chapter}' not found."}
        return build_translation_bundle(ch_dir, config)
    except Exception as exc:
        return {"error": str(exc)}


@mcp.tool()
def submit_translation(chapter: str, mapping: dict) -> dict:
    """Write the translated {region_id: english} mapping for a chapter."""
    try:
        base = _base()
        ch_dir = base / chapter
        if not ch_dir.exists():
            return {"error": f"Chapter folder '{chapter}' not found."}

        prompt_path = (
            ch_dir
            / config.STAGE_FOLDERS["translation"]
            / config.TRANSLATION_PROMPT_NAME
        )
        if not prompt_path.exists():
            return {"error": "no pending bundle; run translate first"}

        res = write_translation_response(ch_dir, mapping)
        logger.info(
            "MCP submit_translation(chapter=%s, n=%d)", chapter, res.get("written", 0)
        )
        return res
    except Exception as exc:
        return {"error": str(exc)}


@mcp.tool()
def get_paraphrase_bundle(chapter: str) -> dict:
    """Return lines to rewrite for a chapter.
    REWRITE each line into natural, casual, SPOKEN US English for a comic bubble.
    Preserve meaning + crude/rude register (do NOT sanitize). Rewrite AGGRESSIVELY —
    use contractions, varied structures, and idiomatic speech; do NOT copy the source verbatim.
    Keep glossary terms exactly. Then call submit_paraphrase(chapter, {region_id: final_english})."""
    try:
        base = _base()
        ch_dir = base / chapter
        if not ch_dir.exists():
            return {"error": f"Chapter folder '{chapter}' not found."}
        return build_paraphrase_bundle(ch_dir, config)
    except Exception as exc:
        return {"error": str(exc)}


@mcp.tool()
def submit_paraphrase(chapter: str, mapping: dict) -> dict:
    """Write the paraphrased {region_id: final_english} mapping for a chapter."""
    try:
        base = _base()
        ch_dir = base / chapter
        if not ch_dir.exists():
            return {"error": f"Chapter folder '{chapter}' not found."}

        prompt_path = (
            ch_dir / config.STAGE_FOLDERS["paraphrase"] / config.PARAPHRASE_PROMPT_NAME
        )
        if not prompt_path.exists():
            return {"error": "no pending bundle; run paraphrase first"}

        res = write_paraphrase_response(ch_dir, mapping)
        logger.info(
            "MCP submit_paraphrase(chapter=%s, n=%d)", chapter, res.get("written", 0)
        )
        return res
    except Exception as exc:
        return {"error": str(exc)}


@mcp.tool()
def get_glossary() -> dict:
    """Return the series-level locked glossary."""
    try:
        from manhua_pipeline.io.glossary_series import load_series_glossary

        base = _base()
        return load_series_glossary(base, config)
    except Exception as exc:
        return {"error": str(exc)}


@mcp.resource("series://chapters")
def list_chapters_resource() -> str:
    """List of all chapters and their statuses."""
    try:
        base = _base()
        if not base.exists() or not base.is_dir():
            return "[]"
        chapters = []
        for item in base.iterdir():
            if item.is_dir():
                manifest_path = item / "manifest.json"
                if manifest_path.exists():
                    try:
                        with manifest_path.open("r", encoding="utf-8") as fh:
                            manifest = json.load(fh)
                        chapters.append(
                            {
                                "chapter": item.name,
                                "current_stage": manifest.get("current_stage"),
                                "status": manifest.get("status"),
                                "warning_count": manifest.get("warning_count", 0),
                            }
                        )
                    except Exception:
                        pass
        return json.dumps(chapters, indent=2)
    except Exception as exc:
        return json.dumps({"error": str(exc)})


def translate_chapter_impl(chapter: str) -> str:
    from manhua_pipeline.stages.stage3_translation import _PROMPT_INSTRUCTIONS

    base = _base()
    ch_dir = base / chapter
    bundle = build_translation_bundle(ch_dir, config)
    lines = "\n".join(
        f"{r['region_id']}: {r['original_text']}" for r in bundle["regions"]
    )
    glossary = bundle.get("glossary", [])
    gloss_txt = "\n".join(f"{t['source_term']} = {t['target_term']}" for t in glossary)
    return (
        _PROMPT_INSTRUCTIONS
        + ("\n\nGLOSSARY (keep exactly):\n" + gloss_txt if gloss_txt else "")
        + "\n\nLINES (translate each; reply ONLY as JSON {region_id: translation}):\n"
        + lines
    )


def paraphrase_chapter_impl(chapter: str) -> str:
    from manhua_pipeline.stages.stage4_paraphrase import _PROMPT_INSTRUCTIONS

    base = _base()
    ch_dir = base / chapter
    bundle = build_paraphrase_bundle(ch_dir, config)
    lines = "\n".join(
        f"{r['region_id']}: {r['literal_translation']}" for r in bundle["regions"]
    )
    glossary = bundle.get("glossary", [])
    gloss_txt = "\n".join(f"{t['source_term']} = {t['target_term']}" for t in glossary)
    return (
        _PROMPT_INSTRUCTIONS
        + ("\n\nGLOSSARY (keep exactly):\n" + gloss_txt if gloss_txt else "")
        + "\n\nLINES (rewrite each; reply ONLY as JSON {region_id: final_english}):\n"
        + lines
    )


@mcp.prompt()
def translate_chapter(chapter: str) -> str:
    """Build a ready-to-run literal-translation prompt for a chapter."""
    return translate_chapter_impl(chapter)


@mcp.prompt()
def paraphrase_chapter(chapter: str) -> str:
    """Build a ready-to-run paraphrase prompt for a chapter (full directives + lines)."""
    return paraphrase_chapter_impl(chapter)


if __name__ == "__main__":
    mcp.run()
