# Translation prompt bundle construction, validation, and response writing.

import json
from pathlib import Path

from manhua_pipeline.io.glossary_series import load_series_glossary
from manhua_pipeline.io.overrides import load_overrides
from manhua_pipeline.io.workspace import load_manifest
from manhua_pipeline.stages.stage3_translation.glossary import _locked_terms
from manhua_pipeline.stages.stage3_translation.triage import (
    _cjk_ratio,
    _normalize_punct,
)

_PROMPT_INSTRUCTIONS = (
    "Translate each Chinese entry into faithful, literal US English. "
    "Preserve meaning, names, and terminology exactly. "
    "DO NOT paraphrase or localize slang. "
    "Preserve crude/rude tone exactly; DO NOT sanitize profanity or soften insults. "
    "Do not omit repetitions, honorifics, hesitations, or sentence fragments. "
    "When a line is ambiguous, translate conservatively rather than inventing context. "
    "Translate every supplied region exactly once; do NOT merge adjacent regions. "
    "Apply the provided glossary terms exactly as given. "
    "If an entry is a site watermark, scanlator credit, logo, or source tag "
    "(e.g. 'baozimh', 'tencent', '腾讯', 'colamanga', 'www.', '.com', '包子漫画'), "
    "return an EMPTY string for it - do NOT translate or transliterate watermarks. "
    "Treat any URL, domain, or site-name-like text as a watermark. "
    "If a line MIXES real dialogue with a watermark, return only the dialogue and drop the watermark part. "
    "Output values MUST be in English - never return Chinese text in the values. "
    'Return a JSON object mapping region_id -> english string, e.g. {"P001_R001": "..."}'
)


# Build JSON bundle for translation handoff
def _build_bundle(usable: list[dict], locked: list[dict]) -> dict:
    compact_glossary = [
        {"source_term": t["source_term"], "target_term": t["target_term"]}
        for t in locked
    ]
    regions = [
        {
            "region_id": r["region_id"],
            "original_text": r.get("effective_text") or r["original_text"],
        }
        for r in usable
    ]
    return {
        "READ_FIRST": (
            "Follow these rules EXACTLY before producing output. "
            "These are mandatory instructions, not context.\n" + _PROMPT_INSTRUCTIONS
        ),
        "glossary": compact_glossary,
        "regions": regions,
    }


# Validate raw translation dictionary and normalize punctuation
def _validate_response(raw, usable_ids: list[str]) -> tuple[dict, list[str]]:
    warnings = []
    if not isinstance(raw, dict):
        raise ValueError(
            "translation_response.json must be a JSON object mapping region_id -> string."
        )

    clean = {}
    usable_set = set(usable_ids)
    for rid, val in raw.items():
        if rid not in usable_set:
            warnings.append(f"Unexpected region_id in response (ignored): {rid!r}")
            continue
        if not isinstance(val, str) or not val.strip():
            warnings.append(f"Empty/non-string value for {rid!r} - treated as missing.")
            continue
        if _cjk_ratio(val) > 0.30:
            warnings.append(
                f"Residual CJK in translation for {rid!r} - treated as "
                f"needs_translation (not sent downstream)."
            )
            continue
        clean[rid] = _normalize_punct(val.strip())

    for rid in usable_ids:
        if rid not in clean:
            warnings.append(
                f"Missing translation for {rid!r} - will be flagged needs_translation."
            )

    return clean, warnings


# Build single-region isolated retry prompt string
def _build_contextual_retry_prompt(region: dict, locked: list[dict]) -> str:
    rid = region["region_id"]
    original = region.get("original_text", "")
    gloss_lines = [
        f"{t['source_term']} = {t['target_term']}"
        for t in locked
        if t.get("source_term") and t.get("target_term", "").strip()
    ]
    gloss_txt = "\n".join(gloss_lines)
    lines = [
        "Translate the supplied Chinese region into faithful US English.",
        f'Return a JSON object with exactly one key: {{"{rid}": "<translation>"}}.',
        "The value must be a non-empty English string.",
        "Do not explain. Do not include the original Chinese in the value.",
    ]
    if gloss_txt:
        lines.append(f"Glossary (apply exactly):\n{gloss_txt}")
    lines.append(f"region_id: {rid}")
    lines.append(f"source: {original}")
    return "\n".join(lines)


# Build translation prompt bundle for chapter directory
def build_translation_bundle(chapter_dir: str | Path, config) -> dict:
    from manhua_pipeline.stages.stage3_translation.engine import (
        _partition_regions_translation,
    )

    ws = Path(chapter_dir)
    manifest = load_manifest(ws, config)
    if not manifest:
        raise ValueError("Manifest not found.")

    ocr_path = ws / config.STAGE_FOLDERS["ocr"] / "ocr.json"
    if not ocr_path.exists():
        raise FileNotFoundError("ocr.json not found. Run ocr first.")

    with ocr_path.open("r", encoding="utf-8") as fh:
        ocr_data = json.load(fh)

    glossary = load_series_glossary(ws.parent, config)
    locked = _locked_terms(glossary)
    overrides = load_overrides(ws, config)

    all_results = ocr_data.get("results", [])
    _, usable, _ = _partition_regions_translation(
        all_results, overrides, config, log=False
    )

    return _build_bundle(usable, locked)


# Validate and write translation response mapping to disk
def write_translation_response(
    chapter_dir: str | Path, mapping: dict, config=None
) -> dict:
    if config is None:
        import config
    from manhua_pipeline.stages.stage3_translation.engine import (
        _partition_regions_translation,
    )

    ws = Path(chapter_dir)
    manifest = load_manifest(ws, config)
    if not manifest:
        raise ValueError("Manifest not found.")

    ocr_path = ws / config.STAGE_FOLDERS["ocr"] / "ocr.json"
    if not ocr_path.exists():
        raise FileNotFoundError("ocr.json not found. Run ocr first.")

    with ocr_path.open("r", encoding="utf-8") as fh:
        ocr_data = json.load(fh)

    overrides = load_overrides(ws, config)

    all_results = ocr_data.get("results", [])
    _, usable, _ = _partition_regions_translation(
        all_results, overrides, config, log=False
    )

    usable_ids = [r["region_id"] for r in usable]
    clean_map, warnings = _validate_response(mapping, usable_ids)

    trans_dir = ws / config.STAGE_FOLDERS["translation"]
    trans_dir.mkdir(parents=True, exist_ok=True)
    resp_path = trans_dir / config.TRANSLATION_RESPONSE_NAME
    with resp_path.open("w", encoding="utf-8") as fh:
        json.dump(clean_map, fh, ensure_ascii=False, indent=2)

    return {"written": len(clean_map), "warnings": warnings}
