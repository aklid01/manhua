"""Stage 3: Translation.

Literal translation via a pluggable backend. v0 = manual JSON handoff.
Emits translation_prompt.json; ingests translation_response.json on re-run.
Enforces locked glossary terms; utilizes mcp by default.
"""

import json
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

from manhua_pipeline.io.glossary_series import load_series_glossary
from manhua_pipeline.io.workspace import load_manifest, save_manifest
from manhua_pipeline.logging_setup import get_logger, log_stage
from manhua_pipeline.stages import _backends as be

logger = get_logger(__name__)

_STAGE_INDEX = 3
_TOTAL_STAGES = 7
_STAGE_NAME = "Translation"

_PROMPT_INSTRUCTIONS = (
    "Translate each Chinese entry into faithful, literal US English. "
    "Preserve meaning, names, and terminology exactly. "
    "DO NOT paraphrase or localize slang. "
    "Preserve crude/rude tone exactly; DO NOT sanitize profanity or soften insults. "
    "Do not omit repetitions, honorifics, hesitations, or sentence fragments. "
    "When a line is ambiguous, translate conservatively rather than inventing context. "
    "Translate every supplied region exactly once; do NOT merge adjacent regions. "
    "Apply the provided glossary terms exactly as given. "
    'Return a JSON object mapping region_id -> english string, e.g. {"P001_R001": "..."}'
)

_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")

_FULLWIDTH_MAP = {
    "\uff01": "!", "\uff1f": "?", "\uff0c": ",", "\u3002": ".",
    "\uff1a": ":", "\uff1b": ";", "\uff08": "(", "\uff09": ")",
    "\u201c": '"', "\u201d": '"', "\u2018": "'", "\u2019": "'",
    "\u3001": ",",
}


def _contains_cjk(text: str) -> bool:
    """True if the string still contains Han ideographs (i.e. not fully translated)."""
    return bool(_CJK_RE.search(text or ""))


def _normalize_punct(text: str) -> str:
    """Replace full-width CJK punctuation with ASCII equivalents."""
    return "".join(_FULLWIDTH_MAP.get(c, c) for c in text)


# ---------------------------------------------------------------------------
# Backend protocol
# ---------------------------------------------------------------------------


class TranslatorBackend(Protocol):
    def request(self, bundle: dict, ws: Path, config) -> dict | None:
        """Return {region_id: english} mapping, or None if awaiting manual input."""


class ManualBackend:
    def request(self, bundle: dict, ws: Path, config) -> dict | None:
        paths = be.BackendPaths(
            "translation",
            config.TRANSLATION_PROMPT_NAME,
            config.TRANSLATION_RESPONSE_NAME,
        )
        return be.manual_request(
            bundle, ws, config, paths, logger,
            _STAGE_INDEX, _TOTAL_STAGES, _STAGE_NAME
        )


class McpBackend:
    def request(self, bundle: dict, ws: Path, config) -> dict | None:
        paths = be.BackendPaths(
            "translation",
            config.TRANSLATION_PROMPT_NAME,
            config.TRANSLATION_RESPONSE_NAME,
        )
        return be.mcp_request(
            bundle, ws, config, paths, logger,
            _STAGE_INDEX, _TOTAL_STAGES, _STAGE_NAME
        )


def _validate_ollama_config(config) -> None:
    s = be.OllamaSettings.from_config(config, prefix="OLLAMA")
    be.validate_ollama_settings(s, "OLLAMA")


class OllamaBackend:
    """Inline local translation via Ollama, with retries, batch validation,
    and malformed-output recovery. Returns {region_id: english}."""

    def request(self, bundle: dict, ws: Path, config) -> dict | None:
        s = be.OllamaSettings.from_config(config, prefix="OLLAMA")
        be.validate_ollama_settings(s, "OLLAMA")

        paths = be.BackendPaths(
            "translation",
            config.TRANSLATION_PROMPT_NAME,
            config.TRANSLATION_RESPONSE_NAME,
        )
        be.write_bundle(bundle, ws, config, paths)

        regions = bundle.get("regions", [])
        if not regions:
            return {}

        self.host = s.host
        self.model = s.model
        self.timeout = s.timeout
        self.temperature = s.temperature
        self.max_retries = s.max_retries
        self.backoff = s.backoff
        self.system_prompt = bundle.get("READ_FIRST", "")
        glossary = bundle.get("glossary", [])
        self.gloss_txt = "\n".join(
            f"{t['source_term']} = {t['target_term']}" for t in glossary
        )

        batch_size = getattr(config, "OLLAMA_BATCH_SIZE", 15)
        total = len(regions)
        n_batches = (total + batch_size - 1) // batch_size
        logger.info(
            "[%d/%d %s] Ollama: %d regions, %d batch(es), model=%s",
            _STAGE_INDEX, _TOTAL_STAGES, _STAGE_NAME, total, n_batches, self.model,
        )

        merged: dict = {}
        for bi in range(n_batches):
            chunk = regions[bi * batch_size:(bi + 1) * batch_size]
            t_batch = time.monotonic()
            accepted = self._translate_batch(chunk, depth=0)
            merged.update(accepted)
            expected = len(chunk)
            missing = expected - len(accepted)
            logger.info(
                "[%d/%d %s] Batch %d/%d: expected %d, accepted %d, missing %d (%.1fs)",
                _STAGE_INDEX, _TOTAL_STAGES, _STAGE_NAME,
                bi + 1, n_batches, expected, len(accepted), missing,
                time.monotonic() - t_batch,
            )

        trans_dir = ws / config.STAGE_FOLDERS["translation"]
        response_path = trans_dir / config.TRANSLATION_RESPONSE_NAME
        with response_path.open("w", encoding="utf-8") as fh:
            json.dump(merged, fh, ensure_ascii=False, indent=2)
        return merged

    def _translate_batch(self, chunk, depth: int) -> dict:
        expected_ids = {r["region_id"] for r in chunk}

        raw = self._call_ollama(self._build_user_prompt(chunk, strict=False))
        accepted, missing, unexpected = self._validate_batch(
            self._parse_json(raw), expected_ids
        )
        if unexpected:
            logger.warning("[%s] Rejected %d unexpected IDs", _STAGE_NAME, len(unexpected))
        if not missing:
            return accepted

        retry_chunk = [r for r in chunk if r["region_id"] in missing]
        raw = self._call_ollama(self._build_user_prompt(retry_chunk, strict=True))
        acc2, missing, _ = self._validate_batch(self._parse_json(raw), set(missing))
        accepted.update(acc2)
        if not missing:
            return accepted

        if len(retry_chunk) > 1 and depth < 4:
            still = [r for r in retry_chunk if r["region_id"] in missing]
            mid = len(still) // 2
            accepted.update(self._translate_batch(still[:mid], depth + 1))
            accepted.update(self._translate_batch(still[mid:], depth + 1))
        else:
            logger.warning(
                "[%s] Giving up on %d region(s) after recovery: %s",
                _STAGE_NAME, len(missing), sorted(missing),
            )
        return accepted

    def _build_user_prompt(self, chunk, strict: bool) -> str:
        payload = json.dumps(
            [{"region_id": r["region_id"], "original_text": r["original_text"]} for r in chunk],
            ensure_ascii=False,
        )
        strict_note = (
            "\nSTRICT: Return ONLY a valid JSON object. No prose, no code fences. "
            "One key per region_id. Do not merge or omit any region."
            if strict else ""
        )
        gloss = ("GLOSSARY (keep exactly):\n" + self.gloss_txt + "\n\n") if self.gloss_txt else ""
        return (
            gloss
            + "Translate each object's original_text. Return a JSON object mapping "
              "region_id -> english. REGIONS:\n" + payload + strict_note
        )

    def _call_ollama(self, user_prompt: str) -> str:
        s = be.OllamaSettings(
            host=self.host,
            model=self.model,
            timeout=self.timeout,
            temperature=self.temperature,
            max_retries=self.max_retries,
            backoff=self.backoff,
        )
        return be.call_ollama(s, self.system_prompt, user_prompt, logger, _STAGE_NAME)

    @staticmethod
    def _validate_batch(parsed: dict, expected_ids: set) -> tuple[dict, list, list]:
        accepted, unexpected = {}, []
        for k, v in parsed.items():
            if k not in expected_ids:
                unexpected.append(k)
                continue
            if not isinstance(v, str) or not v.strip():
                continue
            if _contains_cjk(v):
                continue
            if k not in accepted:
                accepted[k] = _normalize_punct(v.strip())
        missing = [i for i in expected_ids if i not in accepted]
        return accepted, missing, unexpected

    @staticmethod
    def _parse_json(text: str) -> dict:
        return be.parse_json(text)


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
        if _contains_cjk(val):
            warnings.append(
                f"Residual CJK in translation for {rid!r} — treated as "
                f"needs_translation (not sent downstream)."
            )
            continue
        clean[rid] = _normalize_punct(val.strip())

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
        return _write_output(TranslationWriteContext(
            ws=ws, config=config, manifest=manifest, ocr_data=ocr_data,
            glossary=glossary, translation_map={}, usable=usable, skipped=skipped,
            overridden_regions=overridden_regions, overrides=overrides,
            locked=locked, t0=t0,
        ))

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

    backend_name = getattr(config, "TRANSLATOR_BACKEND", "manual")
    if backend_name == "ollama" and usable_ids:
        ratio = len(translation_map) / len(usable_ids)
        min_ratio = getattr(config, "OLLAMA_MIN_COMPLETION_RATIO", 0.95)
        if len(translation_map) == 0:
            raise ValueError(
                "Ollama returned zero usable translations. Not advancing. "
                "Check model output and `ollama serve`."
            )
        if ratio < min_ratio:
            logger.error(
                "[%d/%d %s] Completion %.0f%% < %.0f%% threshold - writing artifacts "
                "but NOT advancing manifest. Re-run after investigating.",
                _STAGE_INDEX, _TOTAL_STAGES, _STAGE_NAME, ratio * 100, min_ratio * 100,
            )
            _write_output(TranslationWriteContext(
                ws=ws, config=config, manifest=manifest, ocr_data=ocr_data,
                glossary=glossary, translation_map=translation_map,
                usable=usable, skipped=skipped, overridden_regions=overridden_regions,
                overrides=overrides, locked=locked, t0=t0, advance=False,
            ))
            return None
        if ratio < 1.0:
            logger.warning(
                "[%d/%d %s] Completion %.0f%% (advancing with warning).",
                _STAGE_INDEX, _TOTAL_STAGES, _STAGE_NAME, ratio * 100,
            )

    return _write_output(TranslationWriteContext(
        ws=ws, config=config, manifest=manifest, ocr_data=ocr_data,
        glossary=glossary, translation_map=translation_map,
        usable=usable, skipped=skipped, overridden_regions=overridden_regions,
        overrides=overrides, locked=locked, t0=t0,
    ))


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


def _write_output(ctx: TranslationWriteContext) -> Path:
    ws, config, manifest = ctx.ws, ctx.config, ctx.manifest
    ocr_data, glossary, translation_map = ctx.ocr_data, ctx.glossary, ctx.translation_map
    usable, skipped, overridden_regions = ctx.usable, ctx.skipped, ctx.overridden_regions
    overrides, locked, t0, advance = ctx.overrides, ctx.locked, ctx.t0, ctx.advance

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
            trans_source = (
                f"ollama:{getattr(config, 'OLLAMA_TRANSLATE_MODEL', '')}"
                if getattr(config, "TRANSLATOR_BACKEND", "") == "ollama" else "llm"
            )
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

    glossary_version = glossary.get("version", "v1")
    output = {
        "chapter_id": ocr_data.get("chapter_id", "unknown"),
        "stage": "translation",
        "generated_at": now,
        "translator_backend": getattr(config, "TRANSLATOR_BACKEND", "manual"),
        "translator_model": getattr(config, "OLLAMA_TRANSLATE_MODEL", None)
            if getattr(config, "TRANSLATOR_BACKEND", "") == "ollama" else None,
        "translator_temperature": getattr(config, "OLLAMA_TEMPERATURE", None)
            if getattr(config, "TRANSLATOR_BACKEND", "") == "ollama" else None,
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
