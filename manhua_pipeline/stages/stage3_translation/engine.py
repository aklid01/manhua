# Translation pipeline engine orchestrating triage, backends, overrides, and glossary locking.

import json
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from manhua_pipeline.io.glossary_series import (
    append_glossary_term,
    load_series_glossary,
)
from manhua_pipeline.io.overrides import load_overrides
from manhua_pipeline.io.workspace import load_manifest, save_manifest
from manhua_pipeline.logging_setup import get_logger, log_stage
from manhua_pipeline.manual_overrides import (
    OverrideRequestStore,
    make_override_request,
    prompt_for_override,
    save_locked_glossary_entry,
    save_region_override,
)
from manhua_pipeline.stages.stage3_translation import backends, bundle, glossary, triage
from manhua_pipeline.translation_validation import validate_expected_regions

logger = get_logger(__name__)

_STAGE_INDEX = 3
_TOTAL_STAGES = 7
_STAGE_NAME = "Translation"
_CJK_BLOCK_RE = re.compile(r"[\u3400-\u9fff]+")


# Partition chapter regions into overridden, usable, and skipped sets
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


# Route unresolved translation region to Stage 4 recovery or manual prompt
def _handle_unresolved_region(
    *,
    region: dict,
    initial_reason: str,
    final_reason: str,
    runtime,
    run_id: str,
    chapter: str,
    ws,
    config,
    translation_map: dict,
    glossary_data: dict,
) -> bool:
    if (
        getattr(config, "TRANSLATION_FAILURE_HANDOFF_TO_PARAPHRASE", True)
        and getattr(config, "TRANSLATOR_BACKEND", "") == "ollama"
    ):
        region["translation_source"] = "translation_recovery"
        logger.warning(
            "[%s] %s -> handing original CJK to Stage 4 translation recovery",
            _STAGE_NAME,
            region["region_id"],
        )
        return True

    request = make_override_request(
        run_id=run_id,
        chapter=chapter,
        region=region,
        initial_reason=initial_reason,
        final_reason=final_reason,
        backend=getattr(config, "TRANSLATOR_BACKEND", "ollama"),
        model=getattr(config, "OLLAMA_TRANSLATE_MODEL", None),
        attempts=2,
    )
    rid = region["region_id"]

    if runtime.can_prompt:
        decision = prompt_for_override(request)

        if decision.action == "apply_region" and decision.translation:
            translation_map[rid] = decision.translation
            overrides_path = ws / getattr(config, "OVERRIDES_NAME", "overrides.json")
            existing = _load_raw_overrides(overrides_path)
            save_region_override(
                existing, request, decision.translation, overrides_path
            )
            logger.info(
                "[%s] Manual override (region): %s -> %r",
                _STAGE_NAME,
                rid,
                decision.translation,
            )
            return True

        if decision.action == "apply_glossary" and decision.translation:
            translation_map[rid] = decision.translation
            glossary_terms = glossary_data.get("terms", [])
            glossary_path = ws.parent / getattr(
                config, "GLOSSARY_NAME", "glossary.json"
            )
            try:
                save_locked_glossary_entry(
                    glossary_terms, request, decision.translation, glossary_path
                )
                logger.info(
                    "[%s] Manual override saved to locked glossary: %r -> %r",
                    _STAGE_NAME,
                    request.original_text,
                    decision.translation,
                )
            except Exception as exc:
                logger.warning(
                    "[%s] Could not save locked glossary: %s", _STAGE_NAME, exc
                )
            return True

        if decision.action == "abort":
            raise RuntimeError(f"Translation aborted by user at region {rid}.")

    store = OverrideRequestStore(ws)
    store.upsert_pending(request)
    logger.warning(
        "[%s] MANUAL_OVERRIDE_DEFERRED: %s queued (%s)",
        _STAGE_NAME,
        rid,
        "user deferred" if runtime.can_prompt else "non-interactive runtime",
    )
    return False


# Load raw overrides JSON dictionary from disk
def _load_raw_overrides(path) -> dict:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


# Context container holding data and configuration for translation output writing
@dataclass
class TranslationWriteContext:
    ws: Path
    config: object
    manifest: dict
    ocr_data: dict
    glossary: dict
    translation_map: dict
    usable: list[dict]
    skipped: list[dict]
    overridden_regions: list[dict]
    overrides: dict
    locked: list[dict]
    t0: float
    advance: bool = True
    glossary_resolved_ids: frozenset = frozenset()


# Write translation.json artifact and update manifest stage progress
def _write_output(ctx: TranslationWriteContext) -> Path:
    ws, config, manifest = ctx.ws, ctx.config, ctx.manifest
    ocr_data, glossary_data, translation_map = (
        ctx.ocr_data,
        ctx.glossary,
        ctx.translation_map,
    )
    usable, skipped, overridden_regions = (
        ctx.usable,
        ctx.skipped,
        ctx.overridden_regions,
    )
    overrides, locked, t0, advance = ctx.overrides, ctx.locked, ctx.t0, ctx.advance

    now = datetime.now(timezone.utc).isoformat()
    translator_backend = getattr(config, "TRANSLATOR_BACKEND", "manual")
    is_ollama = translator_backend == "ollama"
    results = []
    translated_count = 0
    missing_count = 0
    conflict_count = 0
    applied_total = set()

    for region in overridden_regions:
        rid = region["region_id"]
        override_text = overrides[rid]
        final_text, applied, conflict = glossary._enforce_glossary(
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

    glossary_resolved_ids = ctx.glossary_resolved_ids
    for region in usable:
        rid = region["region_id"]
        raw_text = translation_map.get(rid, "")
        if raw_text:
            final_text, applied, conflict = glossary._enforce_glossary(
                region["original_text"], raw_text, locked
            )
            translated = True
            skip_reason = None
            if rid in glossary_resolved_ids:
                trans_source = "locked_glossary"
            elif is_ollama:
                trans_source = f"ollama:{getattr(config, 'OLLAMA_TRANSLATE_MODEL', '')}"
            else:
                trans_source = "llm"
        else:
            final_text, applied, conflict = "", [], False
            translated = False
            skip_reason = None
            trans_source = None
            missing_count += 1

            original = region.get("original_text", "")
            cjk_bits = _CJK_BLOCK_RE.findall(original)

            if region.get("translation_source") == "translation_recovery":
                logger.info(
                    "[%s] %s handed to Stage 4 recovery - not seeding glossary.",
                    _STAGE_NAME,
                    region["region_id"],
                )
            elif not cjk_bits:
                logger.info(
                    "[%s] %s untranslated but has no CJK (%r) - not seeding glossary.",
                    _STAGE_NAME,
                    region["region_id"],
                    original[:40],
                )
            elif triage._is_trivial_region(original, config):
                logger.info(
                    "[%s] %s left untranslated (trivial: %r) - not seeding glossary.",
                    _STAGE_NAME,
                    region["region_id"],
                    original,
                )
            else:
                snippet = cjk_bits[0]
                if len(snippet) <= 1:
                    logger.info(
                        "[%s] %s: %r is a single char - not seeding glossary.",
                        _STAGE_NAME,
                        region["region_id"],
                        snippet,
                    )
                else:
                    glossary_terms = {
                        t.get("source_term") for t in glossary_data.get("terms", [])
                    }
                    if snippet in glossary_terms:
                        logger.info(
                            "[%s]   Found %r in glossary.", _STAGE_NAME, snippet
                        )
                    else:
                        append_glossary_term(ws.parent, config, snippet)
                        logger.warning(
                            "[%s]   %r not in glossary. Added a stub - set its English "
                            "translation in %s/glossary.json and re-run.",
                            _STAGE_NAME,
                            snippet,
                            ws.parent.name,
                        )

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
                "translation_source": (
                    "translation_recovery"
                    if region.get("translation_source") == "translation_recovery"
                    else trans_source
                ),
            }
        )

    for region in skipped:
        if "skip_reason" in region and "translated" in region:
            results.append(
                {
                    "region_id": region["region_id"],
                    "page_number": region.get("page_number"),
                    "original_text": region.get("original_text", ""),
                    "literal_translation": region.get("literal_translation", ""),
                    "translated": region["translated"],
                    "skip_reason": region["skip_reason"],
                    "glossary_terms_applied": region.get("glossary_terms_applied", []),
                    "glossary_conflict": region.get("glossary_conflict", False),
                    "translation_source": region.get("translation_source"),
                }
            )
            if region["translated"]:
                translated_count += 1
        else:
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

    glossary_version = glossary_data.get("version", "v1")
    output = {
        "chapter_id": ocr_data.get("chapter_id", "unknown"),
        "stage": "translation",
        "generated_at": now,
        "translator_backend": translator_backend,
        "translator_model": getattr(config, "OLLAMA_TRANSLATE_MODEL", None)
        if is_ollama
        else None,
        "translator_temperature": getattr(config, "OLLAMA_TEMPERATURE", None)
        if is_ollama
        else None,
        "prompt_version": getattr(config, "OLLAMA_PROMPT_VERSION", "translation-v1"),
        "glossary_version": glossary_version,
        "results": results,
    }

    trans_dir = ws / config.STAGE_FOLDERS["translation"]
    trans_dir.mkdir(parents=True, exist_ok=True)
    out_path = trans_dir / "translation.json"
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(output, fh, ensure_ascii=False, indent=2)

    if advance:
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


# Execute full Stage 3 translation workflow over OCR regions
def run_translation(
    workspace: str, config, *, allow_prompt: bool = False
) -> Path | None:
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
    from manhua_pipeline.events import STAGE_STARTED
    from manhua_pipeline.io.event_emitter import emit_stage_event

    emit_stage_event(ws, config, _STAGE_NAME.lower(), STAGE_STARTED, logger=logger)

    manifest = load_manifest(workspace, config)
    if not manifest:
        raise ValueError("Manifest not found. Run import first.")

    ocr_path = ws / config.STAGE_FOLDERS["ocr"] / "ocr.json"
    if not ocr_path.exists():
        raise FileNotFoundError("ocr.json not found. Run ocr first.")

    with ocr_path.open("r", encoding="utf-8") as fh:
        ocr_data = json.load(fh)

    glossary_data = load_series_glossary(ws.parent, config)
    locked = glossary._locked_terms(glossary_data)

    overrides = load_overrides(ws, config)
    logger.info(
        "[%d/%d %s] Loaded %d overrides from overrides.json",
        _STAGE_INDEX,
        _TOTAL_STAGES,
        _STAGE_NAME,
        len(overrides),
    )

    all_results = ocr_data.get("results", [])

    locked_index_pre = glossary._build_locked_index(locked)
    non_override_results, override_results = [], []
    for r in all_results:
        (
            override_results if r["region_id"] in overrides else non_override_results
        ).append(r)

    translatable_results, triage_hard_skipped, trivial_fallback = (
        triage._partition_by_triage(non_override_results, config, locked_index_pre)
    )
    if triage_hard_skipped or trivial_fallback:
        logger.info(
            "[%d/%d %s] Triage: %d hard-skipped, %d trivial fallback, %d translatable",
            _STAGE_INDEX,
            _TOTAL_STAGES,
            _STAGE_NAME,
            len(triage_hard_skipped),
            len(trivial_fallback),
            len(translatable_results),
        )

    overridden_regions = override_results
    usable = translatable_results + trivial_fallback
    skipped: list[dict] = triage_hard_skipped

    all_ids = {r["region_id"] for r in all_results}
    for k in overrides:
        if k not in all_ids:
            logger.warning(
                "[%s] override for unknown region %s ignored", _STAGE_NAME, k
            )

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

    if not usable:
        logger.info(
            "[%d/%d %s] No usable regions needing LLM translation - completing without handoff.",
            _STAGE_INDEX,
            _TOTAL_STAGES,
            _STAGE_NAME,
        )
        return _write_output(
            TranslationWriteContext(
                ws=ws,
                config=config,
                manifest=manifest,
                ocr_data=ocr_data,
                glossary=glossary_data,
                translation_map={},
                usable=usable,
                skipped=skipped,
                overridden_regions=overridden_regions,
                overrides=overrides,
                locked=locked,
                t0=t0,
            )
        )

    locked_index = glossary._build_locked_index(locked)
    glossary_resolved, usable_for_backend = glossary._resolve_locked_glossary(
        usable, locked_index
    )
    if glossary_resolved:
        logger.info(
            "[%d/%d %s] %d region(s) resolved via locked glossary (no backend call needed)",
            _STAGE_INDEX,
            _TOTAL_STAGES,
            _STAGE_NAME,
            len(glossary_resolved),
        )

    bundle_data = bundle._build_bundle(usable_for_backend, locked)
    backend = backends._get_backend(config)
    raw_response = (
        backend.request(bundle_data, ws, config) if usable_for_backend else {}
    )

    if raw_response is None:
        return None

    backend_ids = [r["region_id"] for r in usable_for_backend]
    try:
        translation_map, val_warnings = bundle._validate_response(
            raw_response, backend_ids
        )
    except ValueError as exc:
        raise ValueError(
            f"translation_response.json is malformed: {exc}. "
            "Fix it and re-run: python pipeline.py translate"
        ) from exc

    for w in val_warnings:
        logger.warning("[%s] %s", _STAGE_NAME, w)

    translation_map.update(glossary_resolved)

    usable_ids = [r["region_id"] for r in usable]

    backend_name = getattr(config, "TRANSLATOR_BACKEND", "manual")
    trivial_missing: list[str] = []
    substantial_missing: list[str] = []
    if backend_name == "ollama" and usable_ids:
        orig_by_id = {r["region_id"]: r.get("original_text", "") for r in usable}
        missing_ids = [rid for rid in usable_ids if rid not in translation_map]
        trivial_missing = [
            rid
            for rid in missing_ids
            if triage._is_trivial_region(orig_by_id.get(rid, ""), config)
        ]
        trivial_missing_set = set(trivial_missing)
        substantial_missing = [
            rid for rid in missing_ids if rid not in trivial_missing_set
        ]

        substantial_total = max(1, len(usable_ids) - len(trivial_missing))
        ratio = (substantial_total - len(substantial_missing)) / substantial_total
        min_ratio = getattr(config, "OLLAMA_MIN_COMPLETION_RATIO", 0.95)
        max_missing = getattr(config, "OLLAMA_MAX_MISSING", 0)

    if len(translation_map) == 0 and not (
        getattr(config, "TRANSLATION_FAILURE_HANDOFF_TO_PARAPHRASE", True)
        and getattr(config, "TRANSLATOR_BACKEND", "") == "ollama"
        and usable
    ):
        raise ValueError(
            "Ollama returned zero usable translations. Not advancing. "
            "Check model output and `ollama serve`."
        )

    if trivial_missing:
        logger.info(
            "[%d/%d %s] %d trivial region(s) left untranslated (SFX/noise, not blocking): %s",
            _STAGE_INDEX,
            _TOTAL_STAGES,
            _STAGE_NAME,
            len(trivial_missing),
            sorted(trivial_missing),
        )

    contextual_retry_enabled = getattr(config, "OLLAMA_CONTEXTUAL_RETRY_ENABLED", False)
    if substantial_missing:
        logger.info(
            "[%d/%d %s] Contextual retry: %s (%d unresolved region(s))",
            _STAGE_INDEX,
            _TOTAL_STAGES,
            _STAGE_NAME,
            "enabled"
            if contextual_retry_enabled
            else "disabled; handing off to Stage 4",
            len(substantial_missing),
        )

    if substantial_missing and contextual_retry_enabled:
        region_by_id = {r["region_id"]: r for r in usable}
        backend_inst = backend if isinstance(backend, backends.OllamaBackend) else None
        for rid in list(substantial_missing):
            region = region_by_id.get(rid)
            if not region or not backend_inst:
                continue
            original = orig_by_id.get(rid, "")
            logger.info(
                "[%d/%d %s] Contextual retry for %s (%r)",
                _STAGE_INDEX,
                _TOTAL_STAGES,
                _STAGE_NAME,
                rid,
                original,
            )
            retry_prompt = bundle._build_contextual_retry_prompt(region, locked)
            try:
                retry_raw = backend_inst._call_ollama(retry_prompt)
                retry_parsed = backend_inst._parse_json(retry_raw)
                retry_issues = validate_expected_regions(retry_parsed, {rid}, attempt=2)
                issue = retry_issues.get(rid)
                if issue and issue.status == "accepted":
                    translation_map[rid] = issue.accepted_text
                    substantial_missing.remove(rid)
                    logger.info(
                        "[%s] Contextual retry succeeded for %s: %r",
                        _STAGE_NAME,
                        rid,
                        issue.accepted_text,
                    )
                else:
                    reason = issue.reason if issue else "unknown"
                    logger.warning(
                        "[%s] Contextual retry failed for %s: reason=%s",
                        _STAGE_NAME,
                        rid,
                        reason,
                    )
            except Exception as exc:
                logger.warning(
                    "[%s] Contextual retry error for %s: %s", _STAGE_NAME, rid, exc
                )

    if substantial_missing:
        from manhua_pipeline.interaction import InteractionMode, RuntimePolicy

        runtime = RuntimePolicy(
            mode=InteractionMode.INTERACTIVE
            if allow_prompt
            else InteractionMode.NON_INTERACTIVE,
            batch=not allow_prompt,
        )
        run_id = getattr(config, "_run_id", "unknown")
        chapter = ws.name
        region_by_id = {r["region_id"]: r for r in usable}

        for rid in list(substantial_missing):
            region = region_by_id[rid]
            resolved = _handle_unresolved_region(
                region=region,
                initial_reason="missing_after_batch",
                final_reason="blank_after_contextual_retry",
                runtime=runtime,
                run_id=run_id,
                chapter=chapter,
                ws=ws,
                config=config,
                translation_map=translation_map,
                glossary_data=glossary_data,
            )
            if resolved:
                substantial_missing.remove(rid)

        still_missing = [
            rid
            for rid in usable_ids
            if rid not in translation_map
            and not triage._is_trivial_region(orig_by_id.get(rid, ""), config)
            and region_by_id.get(rid, {}).get("translation_source")
            != "translation_recovery"
        ]
        new_substantial_total = max(1, len(usable_ids) - len(trivial_missing))
        ratio = (new_substantial_total - len(still_missing)) / new_substantial_total

        ratio_fail = ratio < min_ratio
        count_fail = len(still_missing) > max_missing
        if ratio_fail or count_fail:
            trigger = "ratio" if ratio_fail else "count"
            logger.error(
                "[%d/%d %s] Completion %.0f%% (%d substantial missing, %d allowed; trigger=%s) - "
                "NOT advancing. Re-run after investigating.",
                _STAGE_INDEX,
                _TOTAL_STAGES,
                _STAGE_NAME,
                ratio * 100,
                len(still_missing),
                max_missing,
                trigger,
            )
            _write_output(
                TranslationWriteContext(
                    ws=ws,
                    config=config,
                    manifest=manifest,
                    ocr_data=ocr_data,
                    glossary=glossary_data,
                    translation_map=translation_map,
                    usable=usable,
                    skipped=skipped,
                    overridden_regions=overridden_regions,
                    overrides=overrides,
                    locked=locked,
                    t0=t0,
                    advance=False,
                    glossary_resolved_ids=frozenset(glossary_resolved.keys()),
                )
            )
            return None
        if ratio < 1.0:
            logger.warning(
                "[%d/%d %s] Completion %.0f%% (advancing with warning).",
                _STAGE_INDEX,
                _TOTAL_STAGES,
                _STAGE_NAME,
                ratio * 100,
            )

    return _write_output(
        TranslationWriteContext(
            ws=ws,
            config=config,
            manifest=manifest,
            ocr_data=ocr_data,
            glossary=glossary_data,
            translation_map=translation_map,
            usable=usable,
            skipped=skipped,
            overridden_regions=overridden_regions,
            overrides=overrides,
            locked=locked,
            t0=t0,
            glossary_resolved_ids=frozenset(glossary_resolved.keys()),
        )
    )
