"""Stage 3: Translation.

Literal translation via a pluggable backend. v0 = manual JSON handoff.
Emits translation_prompt.json; ingests translation_response.json on re-run.
Enforces locked glossary terms; stubs mcp/ollama for later.
"""

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

from manhua_pipeline.io.workspace import load_manifest, save_manifest
from manhua_pipeline.logging_setup import get_logger, log_stage

logger = get_logger(__name__)

_STAGE_INDEX = 3
_TOTAL_STAGES = 7
_STAGE_NAME = "Translation"

_PROMPT_INSTRUCTIONS = (
    "Translate each Chinese entry into faithful, literal US English. "
    "Preserve meaning, names, and terminology exactly. "
    "DO NOT paraphrase or localize slang. "
    "Apply the provided glossary terms exactly as given. "
    'Return a JSON object mapping region_id -> english string, e.g. {"P001_R001": "..."}'
)


# ---------------------------------------------------------------------------
# Backend protocol
# ---------------------------------------------------------------------------


class TranslatorBackend(Protocol):
    def request(self, bundle: dict, ws: Path, config) -> dict | None:
        """Return {region_id: english} mapping, or None if awaiting manual input."""


class ManualBackend:
    def request(self, bundle: dict, ws: Path, config) -> dict | None:
        trans_dir = ws / config.STAGE_FOLDERS["translation"]
        trans_dir.mkdir(parents=True, exist_ok=True)

        prompt_path = trans_dir / config.TRANSLATION_PROMPT_NAME
        response_path = trans_dir / config.TRANSLATION_RESPONSE_NAME

        with prompt_path.open("w", encoding="utf-8") as fh:
            json.dump(bundle, fh, ensure_ascii=False, indent=2)
        logger.info(
            "[%d/%d %s] Wrote request bundle -> %s",
            _STAGE_INDEX,
            _TOTAL_STAGES,
            _STAGE_NAME,
            prompt_path,
        )

        if not response_path.exists():
            logger.info(
                "[%d/%d %s] Manual handoff required.\n"
                "  1. Open: %s\n"
                "  2. Paste its contents to your coding assistant.\n"
                "  3. Save the assistant's JSON reply as: %s\n"
                "  4. Re-run: python pipeline.py translate\n"
                "  (Awaiting translation_response.json — no changes written.)",
                _STAGE_INDEX,
                _TOTAL_STAGES,
                _STAGE_NAME,
                prompt_path,
                response_path,
            )
            return None

        with response_path.open("r", encoding="utf-8") as fh:
            raw = json.load(fh)
        return raw


class McpBackend:
    def request(self, bundle: dict, ws: Path, config) -> dict | None:
        raise NotImplementedError(
            "MCP backend is not yet implemented. Set TRANSLATOR_BACKEND='manual' in config.py."
        )


class OllamaBackend:
    def request(self, bundle: dict, ws: Path, config) -> dict | None:
        raise NotImplementedError(
            "Ollama backend is not yet implemented. Set TRANSLATOR_BACKEND='manual' in config.py."
        )


def _get_backend(config) -> TranslatorBackend:
    name = getattr(config, "TRANSLATOR_BACKEND", "manual")
    if name == "manual":
        return ManualBackend()
    if name == "mcp":
        return McpBackend()
    if name == "ollama":
        return OllamaBackend()
    raise ValueError(
        f"Unknown TRANSLATOR_BACKEND: {name!r}. Use 'manual', 'mcp', or 'ollama'."
    )


# ---------------------------------------------------------------------------
# Glossary helpers
# ---------------------------------------------------------------------------


def _load_glossary(ws: Path, config) -> dict:
    path = ws / config.GLOSSARY_NAME
    if path.exists():
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    logger.info("[%s] glossary.json not found — creating empty glossary.", _STAGE_NAME)
    empty = {
        "version": "v1",
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "terms": [],
    }
    with path.open("w", encoding="utf-8") as fh:
        json.dump(empty, fh, ensure_ascii=False, indent=2)
    return empty


def _locked_terms(glossary: dict) -> list[dict]:
    return [t for t in glossary.get("terms", []) if t.get("locked")]


def _enforce_glossary(
    original_zh: str, translation: str, locked: list[dict]
) -> tuple[str, list[str], bool]:
    """Post-process: flag glossary conflicts without mutating translation text.

    If a locked source term appears in original_zh but the target is absent
    in translation, record the term_id and set conflict=True. The text is
    left unchanged so rendering is never defaced; QA surfaces conflicts.
    """
    applied = []
    conflict = False
    for term in locked:
        source = term.get("source_term", "")
        target = term.get("target_term", "")
        term_id = term.get("term_id", "")
        if not source or not target:
            continue
        if source in original_zh:
            if target in translation:
                applied.append(term_id)
            else:
                applied.append(term_id)
                conflict = True
    return translation, applied, conflict


def _maybe_seed_glossary(results: list[dict], glossary: dict) -> None:
    """Deferred stub: auto-seed glossary from name_label/scene_text regions.

    TODO: also consider skipping credits/title pages (P001) and author-note
    pages (P042) at Detection or Import level (page-level skip_reason).

    TODO: implement when detection emits name_label and scene_text types.
    """


# ---------------------------------------------------------------------------
# Bundle building and response validation
# ---------------------------------------------------------------------------


def _build_bundle(usable: list[dict], locked: list[dict]) -> dict:
    compact_glossary = [
        {"source_term": t["source_term"], "target_term": t["target_term"]}
        for t in locked
    ]
    regions = [
        {"region_id": r["region_id"], "original_text": r["original_text"]}
        for r in usable
    ]
    return {
        "instructions": _PROMPT_INSTRUCTIONS,
        "glossary": compact_glossary,
        "regions": regions,
    }


def _validate_response(raw, usable_ids: list[str]) -> tuple[dict, list[str]]:
    """Validate raw response dict; return (clean_map, warnings)."""
    warnings = []
    if not isinstance(raw, dict):
        raise ValueError(
            "translation_response.json must be a JSON object mapping region_id -> string."
        )

    clean = {}
    for rid, val in raw.items():
        if rid not in usable_ids:
            warnings.append(f"Unexpected region_id in response (ignored): {rid!r}")
            continue
        if not isinstance(val, str) or not val.strip():
            warnings.append(f"Empty/non-string value for {rid!r} — treated as missing.")
            continue
        clean[rid] = val.strip()

    for rid in usable_ids:
        if rid not in clean:
            warnings.append(
                f"Missing translation for {rid!r} — will be flagged needs_translation."
            )

    return clean, warnings


# ---------------------------------------------------------------------------
# Top-level orchestrator
# ---------------------------------------------------------------------------


def run_translation(workspace: str, config) -> Path | None:
    """Run the Translation stage.

    Returns the path to translation.json on success, or None if awaiting
    manual input (prompt written, no response yet).
    """
    t0 = time.monotonic()
    ws = Path(workspace)
    log_stage(logger, _STAGE_INDEX, _TOTAL_STAGES, _STAGE_NAME, "starting")

    manifest = load_manifest(workspace, config)
    if not manifest:
        raise ValueError("Manifest not found. Run import first.")

    ocr_path = ws / config.STAGE_FOLDERS["ocr"] / "ocr.json"
    if not ocr_path.exists():
        raise FileNotFoundError("ocr.json not found. Run ocr first.")

    with ocr_path.open("r", encoding="utf-8") as fh:
        ocr_data = json.load(fh)

    glossary = _load_glossary(ws, config)
    locked = _locked_terms(glossary)

    # Gate: split by has_usable_text
    all_results = ocr_data.get("results", [])
    usable = [r for r in all_results if r.get("has_usable_text")]
    skipped = [r for r in all_results if not r.get("has_usable_text")]

    logger.info(
        "[%d/%d %s] Backend: %s",
        _STAGE_INDEX,
        _TOTAL_STAGES,
        _STAGE_NAME,
        getattr(config, "TRANSLATOR_BACKEND", "manual"),
    )
    logger.info(
        "[%d/%d %s] %d usable regions, %d skipped (no_usable_text)",
        _STAGE_INDEX,
        _TOTAL_STAGES,
        _STAGE_NAME,
        len(usable),
        len(skipped),
    )

    # Short-circuit: nothing to translate
    if not usable:
        logger.info(
            "[%d/%d %s] No usable regions — completing without handoff.",
            _STAGE_INDEX,
            _TOTAL_STAGES,
            _STAGE_NAME,
        )
        return _write_output(
            ws, config, manifest, ocr_data, glossary, {}, usable, skipped, locked, t0
        )

    bundle = _build_bundle(usable, locked)
    backend = _get_backend(config)
    raw_response = backend.request(bundle, ws, config)

    if raw_response is None:
        return None

    # Validate
    usable_ids = [r["region_id"] for r in usable]
    try:
        translation_map, val_warnings = _validate_response(raw_response, usable_ids)
    except ValueError as exc:
        raise ValueError(
            f"translation_response.json is malformed: {exc}. "
            "Fix it and re-run: python pipeline.py translate"
        ) from exc

    for w in val_warnings:
        logger.warning("[%s] %s", _STAGE_NAME, w)

    return _write_output(
        ws,
        config,
        manifest,
        ocr_data,
        glossary,
        translation_map,
        usable,
        skipped,
        locked,
        t0,
    )


def _write_output(
    ws: Path,
    config,
    manifest: dict,
    ocr_data: dict,
    glossary: dict,
    translation_map: dict,
    usable: list[dict],
    skipped: list[dict],
    locked: list[dict],
    t0: float,
) -> Path:
    now = datetime.now(timezone.utc).isoformat()
    results = []
    translated_count = 0
    missing_count = 0
    conflict_count = 0
    applied_total = set()

    # Usable regions
    for region in usable:
        rid = region["region_id"]
        raw_text = translation_map.get(rid, "")
        if raw_text:
            final_text, applied, conflict = _enforce_glossary(
                region["original_text"], raw_text, locked
            )
            translated = True
            skip_reason = None
        else:
            final_text, applied, conflict = "", [], False
            translated = False
            skip_reason = None
            missing_count += 1

        if translated:
            translated_count += 1
        if conflict:
            conflict_count += 1
        applied_total.update(applied)

        results.append(
            {
                "region_id": rid,
                "page_number": region["page_number"],
                "original_text": region["original_text"],
                "literal_translation": final_text,
                "translated": translated,
                "skip_reason": skip_reason,
                "glossary_terms_applied": applied,
                "glossary_conflict": conflict,
            }
        )

    # Skipped regions
    for region in skipped:
        note = region.get("note") or ""
        skip_reason = "ocr_error" if "ocr_error" in note else "no_usable_text"
        results.append(
            {
                "region_id": region["region_id"],
                "page_number": region["page_number"],
                "original_text": region["original_text"],
                "literal_translation": "",
                "translated": False,
                "skip_reason": skip_reason,
                "glossary_terms_applied": [],
                "glossary_conflict": False,
            }
        )

    _maybe_seed_glossary(results, {})

    glossary_version = glossary.get("version", "v1")
    output = {
        "chapter_id": ocr_data.get("chapter_id", "unknown"),
        "stage": "translation",
        "generated_at": now,
        "translator_backend": getattr(config, "TRANSLATOR_BACKEND", "manual"),
        "glossary_version": glossary_version,
        "results": results,
    }

    trans_dir = ws / config.STAGE_FOLDERS["translation"]
    trans_dir.mkdir(parents=True, exist_ok=True)
    out_path = trans_dir / "translation.json"
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(output, fh, ensure_ascii=False, indent=2)

    # Advance manifest
    completed = manifest.get("completed_stages", [])
    if "translate" not in completed:
        completed.append("translate")
    manifest["completed_stages"] = completed
    manifest["current_stage"] = "paraphrase"
    manifest["updated_at"] = now
    save_manifest(ws, config, manifest)

    elapsed = time.monotonic() - t0
    logger.info(
        "[%d/%d %s] Glossary: applied %d locked terms; %d conflicts",
        _STAGE_INDEX,
        _TOTAL_STAGES,
        _STAGE_NAME,
        len(applied_total),
        conflict_count,
    )
    log_stage(
        logger,
        _STAGE_INDEX,
        _TOTAL_STAGES,
        _STAGE_NAME,
        f"done: {translated_count} translated, {len(skipped)} skipped, "
        f"{missing_count} missing -> {out_path} (elapsed {elapsed:.1f}s)",
    )
    return out_path
