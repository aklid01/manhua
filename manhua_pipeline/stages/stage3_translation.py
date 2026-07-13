"""Stage 3: Translation.

Literal translation via a pluggable backend. v0 = manual JSON handoff.
Emits translation_prompt.json; ingests translation_response.json on re-run.
Enforces locked glossary terms; utilizes mcp by default.
"""

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

from manhua_pipeline.io.glossary_series import load_series_glossary
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
        trans_dir = ws / config.STAGE_FOLDERS["translation"]
        trans_dir.mkdir(parents=True, exist_ok=True)

        prompt_path = trans_dir / config.TRANSLATION_PROMPT_NAME
        response_path = trans_dir / config.TRANSLATION_RESPONSE_NAME

        with prompt_path.open("w", encoding="utf-8") as fh:
            json.dump(bundle, fh, ensure_ascii=False, indent=2)
        logger.info(
            "[%d/%d %s] Wrote request bundle for MCP -> %s",
            _STAGE_INDEX,
            _TOTAL_STAGES,
            _STAGE_NAME,
            prompt_path,
        )

        if not response_path.exists():
            logger.info(
                "[%d/%d %s] MCP handoff required.\n"
                "  (Awaiting translation_response.json via MCP tool — no changes written.)",
                _STAGE_INDEX,
                _TOTAL_STAGES,
                _STAGE_NAME,
            )
            return None

        with response_path.open("r", encoding="utf-8") as fh:
            raw = json.load(fh)
        return raw


def _get_backend(config) -> TranslatorBackend:
    name = getattr(config, "TRANSLATOR_BACKEND", "manual")
    if name == "manual":
        return ManualBackend()
    if name == "mcp":
        return McpBackend()
    raise ValueError(
        f"Unknown TRANSLATOR_BACKEND: {name!r}. Use 'manual' or 'mcp'."
    )


# ---------------------------------------------------------------------------
# Glossary helpers
# ---------------------------------------------------------------------------


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
    # No-op in v0: auto-seeding awaits name_label/scene_text detection. Series glossary is currently hand-curated.
    pass


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
        "READ_FIRST": (
            "Follow these rules EXACTLY before producing output. "
            "These are mandatory instructions, not context.\n" + _PROMPT_INSTRUCTIONS
        ),
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


def _partition_regions_translation(
    all_results: list[dict], overrides: dict, config, log: bool = True
) -> tuple[list[dict], list[dict], list[dict]]:
    region_ids_set = {r["region_id"] for r in all_results}
    for k in overrides:
        if k not in region_ids_set:
            if log:
                logger.warning(
                    "[%s] override for unknown region %s ignored", _STAGE_NAME, k
                )

    overridden_regions = []
    usable = []
    skipped = []

    for r in all_results:
        rid = r["region_id"]
        if rid in overrides:
            overridden_regions.append(r)
        elif r.get("has_usable_text"):
            usable.append(r)
        else:
            skipped.append(r)

    for r in overridden_regions:
        rid = r["region_id"]
        if log:
            logger.info(
                "[%d/%d %s] %s -> using override (%r)",
                _STAGE_INDEX,
                _TOTAL_STAGES,
                _STAGE_NAME,
                rid,
                overrides[rid],
            )

    if log:
        logger.info(
            "[%d/%d %s] Backend: %s",
            _STAGE_INDEX,
            _TOTAL_STAGES,
            _STAGE_NAME,
            getattr(config, "TRANSLATOR_BACKEND", "manual"),
        )
        logger.info(
            "[%d/%d %s] %d usable regions, %d overridden, %d skipped",
            _STAGE_INDEX,
            _TOTAL_STAGES,
            _STAGE_NAME,
            len(usable),
            len(overridden_regions),
            len(skipped),
        )
    return overridden_regions, usable, skipped


def run_translation(workspace: str, config) -> Path | None:
    """Run the Translation stage.

    Returns the path to translation.json on success, or None if awaiting
    manual input (prompt written, no response yet).
    """
    t0 = time.monotonic()
    ws = Path(workspace)
    logger.info(
        "[%d/%d %s] Series: %s | Chapter: %s",
        _STAGE_INDEX,
        _TOTAL_STAGES,
        _STAGE_NAME,
        ws.parent.as_posix(),
        ws.name,
    )
    log_stage(logger, _STAGE_INDEX, _TOTAL_STAGES, _STAGE_NAME, "starting")

    manifest = load_manifest(workspace, config)
    if not manifest:
        raise ValueError("Manifest not found. Run import first.")

    ocr_path = ws / config.STAGE_FOLDERS["ocr"] / "ocr.json"
    if not ocr_path.exists():
        raise FileNotFoundError("ocr.json not found. Run ocr first.")

    with ocr_path.open("r", encoding="utf-8") as fh:
        ocr_data = json.load(fh)

    glossary = load_series_glossary(ws.parent, config)
    locked = _locked_terms(glossary)

    # Load overrides
    from manhua_pipeline.io.overrides import load_overrides

    overrides = load_overrides(ws, config)
    logger.info(
        "[%d/%d %s] Loaded %d overrides from overrides.json",
        _STAGE_INDEX,
        _TOTAL_STAGES,
        _STAGE_NAME,
        len(overrides),
    )

    all_results = ocr_data.get("results", [])
    overridden_regions, usable, skipped = _partition_regions_translation(
        all_results, overrides, config
    )

    # Short-circuit: nothing to translate via LLM
    if not usable:
        logger.info(
            "[%d/%d %s] No usable regions needing LLM translation — completing without handoff.",
            _STAGE_INDEX,
            _TOTAL_STAGES,
            _STAGE_NAME,
        )
        return _write_output(
            ws,
            config,
            manifest,
            ocr_data,
            glossary,
            {},
            usable,
            skipped,
            overridden_regions,
            overrides,
            locked,
            t0,
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
        overridden_regions,
        overrides,
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
    overridden_regions: list[dict],
    overrides: dict,
    locked: list[dict],
    t0: float,
) -> Path:
    now = datetime.now(timezone.utc).isoformat()
    results = []
    translated_count = 0
    missing_count = 0
    conflict_count = 0
    applied_total = set()

    # Overridden regions
    for region in overridden_regions:
        rid = region["region_id"]
        override_text = overrides[rid]
        final_text, applied, conflict = _enforce_glossary(
            region.get("original_text", ""), override_text, locked
        )
        results.append(
            {
                "region_id": rid,
                "page_number": region["page_number"],
                "original_text": region.get("original_text", ""),
                "literal_translation": final_text,
                "translated": True,
                "skip_reason": None,
                "glossary_terms_applied": applied,
                "glossary_conflict": conflict,
                "translation_source": "override",
            }
        )
        translated_count += 1
        if conflict:
            conflict_count += 1
        applied_total.update(applied)

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
            trans_source = "llm"
        else:
            final_text, applied, conflict = "", [], False
            translated = False
            skip_reason = None
            trans_source = None
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
                "translation_source": trans_source,
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
                "translation_source": None,
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


def build_translation_bundle(chapter_dir: str | Path, config) -> dict:
    """Build translation prompt bundle for the chapter."""
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

    from manhua_pipeline.io.overrides import load_overrides

    overrides = load_overrides(ws, config)

    all_results = ocr_data.get("results", [])
    _, usable, _ = _partition_regions_translation(all_results, overrides, config, log=False)

    return _build_bundle(usable, locked)


def write_translation_response(
    chapter_dir: str | Path, mapping: dict, config=None
) -> dict:
    """Validate and write the translation mapping to translation_response.json."""
    if config is None:
        import config
    ws = Path(chapter_dir)
    manifest = load_manifest(ws, config)
    if not manifest:
        raise ValueError("Manifest not found.")

    ocr_path = ws / config.STAGE_FOLDERS["ocr"] / "ocr.json"
    if not ocr_path.exists():
        raise FileNotFoundError("ocr.json not found. Run ocr first.")

    with ocr_path.open("r", encoding="utf-8") as fh:
        ocr_data = json.load(fh)

    from manhua_pipeline.io.overrides import load_overrides

    overrides = load_overrides(ws, config)

    all_results = ocr_data.get("results", [])
    _, usable, _ = _partition_regions_translation(all_results, overrides, config, log=False)

    usable_ids = [r["region_id"] for r in usable]
    clean_map, warnings = _validate_response(mapping, usable_ids)

    trans_dir = ws / config.STAGE_FOLDERS["translation"]
    trans_dir.mkdir(parents=True, exist_ok=True)
    resp_path = trans_dir / config.TRANSLATION_RESPONSE_NAME
    with resp_path.open("w", encoding="utf-8") as fh:
        json.dump(clean_map, fh, ensure_ascii=False, indent=2)

    return {"written": len(clean_map), "warnings": warnings}
