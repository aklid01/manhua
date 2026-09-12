# ruff: noqa: E402
# FastMCP server adapter exposing translation and paraphrasing workflows over JSON-RPC.

import json
import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from fastmcp import FastMCP

import config
from manhua_pipeline.io.glossary_series import (
    load_series_glossary,
    update_glossary_terms,
)
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

setup_logging(stream="stderr")
logger = get_logger(__name__)

mcp = FastMCP("Manhua Pipeline")


# Resolve base series output directory
def _base() -> Path:
    output_dir = get_output_dir()
    if not output_dir:
        raise ValueError(
            "Series base directory is not set. Run with --output-dir <path> first."
        )
    return Path(output_dir)


# Check whether chapter is awaiting translation or paraphrase response
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
    """List chapters in the series that are currently waiting for translation or paraphrase."""
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
    """Return the bundle of lines that need translation for a chapter, with glossary."""
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
    """Write the translation response mapping {region_id: english} for a chapter."""
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

    Follow the READ_FIRST instructions in the returned bundle: rewrite dialogue
    into natural spoken English; preserve locked glossary terms verbatim; do not
    leave lines in raw literal English; submit the rewritten mapping via
    submit_paraphrase.
    """
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
    """Write the paraphrase response mapping {region_id: final_english} for a chapter."""
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
    """Return the series-level locked glossary with character names and terms."""
    try:
        base = _base()
        return load_series_glossary(base, config)
    except Exception as exc:
        return {"error": str(exc)}


@mcp.tool()
def update_glossary(terms: list[dict]) -> dict:
    """Add or update series glossary terms.

    Pass a list of term objects shaped like entries in glossary.json:
    {"source_term": "姐姐", "target_term": "big sister", "locked": false, "notes": "..."}.

    'source_term' and 'target_term' are required. A 'source_term' must be CJK
    text; entries whose source term contains no CJK characters are rejected
    with a warning. 'locked' (default false) and
    'notes' (default "") are optional. Entries are matched against existing terms
    by 'source_term' and updated in place; new source_terms are appended. Terms
    already marked "locked": true are never overwritten - changing one requires
    unlocking it in a separate call.

    The response includes a 'warnings' array and 'warning_count'. If warnings is
    not empty, fix the flagged entries and resubmit - do not ignore them.
    """
    try:
        base = _base()
        res = update_glossary_terms(base, config, terms)
        logger.info(
            "MCP update_glossary(n=%d, written=%d, warnings=%d)",
            len(terms),
            res.get("written", 0),
            res.get("warning_count", 0),
        )
        return res
    except Exception as exc:
        return {"error": str(exc)}


@mcp.resource("series://chapters")
def list_chapters_resource() -> str:
    """List all chapters and their current statuses in JSON format."""
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


# Construct complete literal translation prompt text for chapter
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


# Construct complete paraphrase prompt text for chapter
def paraphrase_chapter_impl(chapter: str) -> str:
    from manhua_pipeline.stages.stage4_paraphrase import _PROMPT_INSTRUCTIONS

    base = _base()
    ch_dir = base / chapter
    bundle = build_paraphrase_bundle(ch_dir, config)
    lines = "\n".join(
        (
            f"{r['region_id']} [TRANSLATE_AND_PARAPHRASE]: {r.get('original_text', '')}"
            if r.get("translation_task") == "translate_and_paraphrase"
            else f"{r['region_id']}: {r['literal_translation']}"
        )
        for r in bundle["regions"]
    )
    glossary = bundle.get("glossary", [])
    gloss_txt = "\n".join(f"{t['source_term']} = {t['target_term']}" for t in glossary)
    return (
        _PROMPT_INSTRUCTIONS
        + ("\n\nGLOSSARY (keep exactly):\n" + gloss_txt if gloss_txt else "")
        + "\n\nFor lines marked TRANSLATE_AND_PARAPHRASE, translate the CJK text into "
        "English first and then paraphrase it. Rewrite normal lines as before. "
        "Reply ONLY as JSON {region_id: final_english}:\n" + lines
    )


@mcp.prompt()
def translate_chapter(chapter: str) -> str:
    """Build the prompt for translating a chapter's OCR lines."""
    return translate_chapter_impl(chapter)


@mcp.prompt()
def paraphrase_chapter(chapter: str) -> str:
    """Build the prompt for rewriting a chapter's literal translations."""
    return paraphrase_chapter_impl(chapter)


if __name__ == "__main__":
    mcp.run()
