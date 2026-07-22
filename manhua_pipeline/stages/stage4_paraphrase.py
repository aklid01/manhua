"""Stage 4: Paraphrase.

Rewrites literal translations into natural, casual US English optimized for comic speech bubbles.
Uses pluggable backends (manual, mcp) and supports manual JSON handoff.
Classifies register (rude, label, neutral) using a local heuristic.
"""

import json
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

from manhua_pipeline.io.glossary_series import load_series_glossary
from manhua_pipeline.io.workspace import load_manifest, save_manifest
from manhua_pipeline.logging_setup import get_logger, log_stage
from manhua_pipeline.stages import _backends as be

logger = get_logger(__name__)

_STAGE_INDEX = 4
_TOTAL_STAGES = 7
_STAGE_NAME = "Paraphrase"

_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")


def _contains_cjk(text: str) -> bool:
    return bool(_CJK_RE.search(text or ""))


def _cjk_ratio(text: str) -> float:
    """Fraction of non-space chars that are Han ideographs."""
    chars = [c for c in text if not c.isspace()]
    if not chars:
        return 0.0
    return sum(1 for c in chars if _contains_cjk(c)) / len(chars)


_PROMPT_INSTRUCTIONS = (
    "Rewrite each literal English line as natural, casual, SPOKEN US English for a "
    "comic speech bubble. Preserve meaning and emotional register (including crude/rude "
    "tone — do NOT sanitize). Rewrite AGGRESSIVELY: prefer idiomatic phrasing, "
    "contractions, and varied sentence structure over line-by-line literal wording. "
    "NEVER soften, censor, or euphemize profanity or insults; keep the same intensity. "
    "Do NOT copy the source line verbatim UNLESS it is a proper noun, brand/watermark, "
    "or fixed label. Keep names and glossary terms exactly. Prefer SHORT, punchy lines.\n"
    'Return a JSON object mapping region_id -> final english string, e.g. {"P001_R001": "..."}'
)


# ---------------------------------------------------------------------------
# Backend protocol
# ---------------------------------------------------------------------------


class ParaphraseBackend(Protocol):
    def request(self, bundle: dict, ws: Path, config) -> dict | None:
        """Return {region_id: final_english} mapping, or None if awaiting manual input."""


class ManualBackend:
    def request(self, bundle: dict, ws: Path, config) -> dict | None:
        paths = be.BackendPaths(
            "paraphrase",
            config.PARAPHRASE_PROMPT_NAME,
            config.PARAPHRASE_RESPONSE_NAME,
        )
        return be.manual_request(
            bundle, ws, config, paths, logger, _STAGE_INDEX, _TOTAL_STAGES, _STAGE_NAME
        )


class McpBackend:
    def request(self, bundle: dict, ws: Path, config) -> dict | None:
        paths = be.BackendPaths(
            "paraphrase",
            config.PARAPHRASE_PROMPT_NAME,
            config.PARAPHRASE_RESPONSE_NAME,
        )
        return be.mcp_request(
            bundle, ws, config, paths, logger, _STAGE_INDEX, _TOTAL_STAGES, _STAGE_NAME
        )


class OllamaBackend:
    """Inline local paraphrase via Ollama, with retries, batch validation, and
    malformed-output recovery. Returns {region_id: final_english}.

    No CJK guard (English->English). Echoes of the literal line are allowed
    (proper nouns / labels) but logged.
    """

    def request(self, bundle: dict, ws: Path, config) -> dict | None:
        s = be.OllamaSettings.from_config(config, prefix="OLLAMA_PARA")
        be.validate_ollama_settings(s, "OLLAMA_PARA")

        paths = be.BackendPaths(
            "paraphrase",
            config.PARAPHRASE_PROMPT_NAME,
            config.PARAPHRASE_RESPONSE_NAME,
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
        self.tone = bundle.get("tone_directive", "")
        self.shorten = bundle.get("shorten_hint", "")
        glossary = bundle.get("glossary", [])
        self.gloss_txt = "\n".join(
            f"{t['source_term']} = {t['target_term']}" for t in glossary
        )

        batch_size = getattr(config, "OLLAMA_PARA_BATCH_SIZE", 15)
        total = len(regions)
        n_batches = (total + batch_size - 1) // batch_size
        logger.info(
            "[%d/%d %s] Ollama: %d regions, %d batch(es), model=%s",
            _STAGE_INDEX,
            _TOTAL_STAGES,
            _STAGE_NAME,
            total,
            n_batches,
            self.model,
        )

        merged: dict = {}
        for bi in range(n_batches):
            chunk = regions[bi * batch_size : (bi + 1) * batch_size]
            t_batch = time.monotonic()
            accepted = self._paraphrase_batch(chunk, depth=0)
            merged.update(accepted)
            logger.info(
                "[%d/%d %s] Batch %d/%d: expected %d, accepted %d, missing %d (%.1fs)",
                _STAGE_INDEX,
                _TOTAL_STAGES,
                _STAGE_NAME,
                bi + 1,
                n_batches,
                len(chunk),
                len(accepted),
                len(chunk) - len(accepted),
                time.monotonic() - t_batch,
            )

        para_dir = ws / config.STAGE_FOLDERS["paraphrase"]
        response_path = para_dir / config.PARAPHRASE_RESPONSE_NAME
        with response_path.open("w", encoding="utf-8") as fh:
            json.dump(merged, fh, ensure_ascii=False, indent=2)
        return merged

    def _paraphrase_batch(self, chunk, depth: int) -> dict:
        expected_ids = {r["region_id"] for r in chunk}
        literals = {r["region_id"]: r.get("literal_translation", "") for r in chunk}

        raw = self._call_ollama(self._build_user_prompt(chunk, strict=False))
        accepted, missing, unexpected = self._validate_batch(
            self._parse_json(raw), expected_ids, literals
        )
        if unexpected:
            logger.warning(
                "[%s] Rejected %d unexpected IDs", _STAGE_NAME, len(unexpected)
            )
        if not missing:
            return accepted

        retry_chunk = [r for r in chunk if r["region_id"] in missing]
        raw = self._call_ollama(self._build_user_prompt(retry_chunk, strict=True))
        acc2, missing, _ = self._validate_batch(
            self._parse_json(raw), set(missing), literals
        )
        accepted.update(acc2)
        if not missing:
            return accepted

        if len(retry_chunk) <= 1 or depth >= 4:
            single_max = getattr(self, "single_retry_max", 3)
            for _attempt in range(single_max):
                if not missing:
                    break
                still = [r for r in chunk if r["region_id"] in missing]
                raw = self._call_ollama(self._build_user_prompt(still, strict=True))
                acc_n, missing, _ = self._validate_batch(
                    self._parse_json(raw), set(missing), literals
                )
                accepted.update(acc_n)
            if missing:
                logger.warning(
                    "[%s] Giving up on %d region(s); literal fallback will apply: %s",
                    _STAGE_NAME,
                    len(missing),
                    sorted(missing),
                )
            return accepted

        still = [r for r in retry_chunk if r["region_id"] in missing]
        mid = len(still) // 2
        accepted.update(self._paraphrase_batch(still[:mid], depth + 1))
        accepted.update(self._paraphrase_batch(still[mid:], depth + 1))
        return accepted

    def _build_user_prompt(self, chunk, strict: bool) -> str:
        payload = json.dumps(
            [
                {
                    "region_id": r["region_id"],
                    "literal_translation": r.get("literal_translation", ""),
                }
                for r in chunk
            ],
            ensure_ascii=False,
        )
        strict_note = (
            "\nSTRICT: Return ONLY a valid JSON object. No prose, no code fences. "
            "One key per region_id. Do not merge or omit any region."
            if strict
            else ""
        )
        gloss = (
            ("GLOSSARY (keep exactly):\n" + self.gloss_txt + "\n\n")
            if self.gloss_txt
            else ""
        )
        directives = ""
        if self.tone:
            directives += f"TONE: {self.tone}\n"
        if self.shorten:
            directives += f"{self.shorten}\n"
        return (
            directives
            + gloss
            + "Rewrite each object's literal_translation into punchy, natural spoken "
            "US English. Return a JSON object mapping region_id -> final_english. "
            "REGIONS:\n" + payload + strict_note
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
    def _validate_batch(
        parsed: dict, expected_ids: set, literals: dict
    ) -> tuple[dict, list, list]:
        """Accept any non-empty string. No CJK guard (English->English). Echoes allowed."""
        accepted, unexpected = {}, []
        for k, v in parsed.items():
            if k not in expected_ids:
                unexpected.append(k)
                continue
            if not isinstance(v, str) or not v.strip():
                continue
            val = v.strip()
            if _cjk_ratio(val) > 0.30:
                continue
            if literals.get(k, "").strip() and val == literals[k].strip():
                logger.info(
                    "[%s] %s: paraphrase echoes literal (allowed).", _STAGE_NAME, k
                )
            if k not in accepted:
                accepted[k] = val
        missing = [i for i in expected_ids if i not in accepted]
        return accepted, missing, unexpected

    @staticmethod
    def _parse_json(text: str) -> dict:
        return be.parse_json(text)


def _get_backend(config) -> ParaphraseBackend:
    name = getattr(config, "PARAPHRASE_BACKEND", "manual")
    if name == "manual":
        return ManualBackend()
    if name == "mcp":
        return McpBackend()
    if name == "ollama":
        return OllamaBackend()
    raise ValueError(
        f"Unknown PARAPHRASE_BACKEND: {name!r}. Use 'manual', 'mcp', or 'ollama'."
    )


# ---------------------------------------------------------------------------
# Glossary helpers
# ---------------------------------------------------------------------------


def _locked_terms(glossary: dict) -> list[dict]:
    return [t for t in glossary.get("terms", []) if t.get("locked")]


def _enforce_glossary(
    literal_en: str, paraphrased_en: str, locked: list[dict]
) -> tuple[str, bool]:
    conflict = False
    for term in locked:
        target = term.get("target_term", "")
        if not target:
            continue
        if target in literal_en:
            if target not in paraphrased_en:
                conflict = True
    return paraphrased_en, conflict


# ---------------------------------------------------------------------------
# Heuristic Register Classifier
# ---------------------------------------------------------------------------


def _detect_register(text: str, rude_markers: list[str]) -> str:
    """Classifies the tone/register of the final paraphrase string.

    Tags 'rude' (profanity, double exclamation, or all-caps),
    'label' (short text without ending punctuation, or contains dash/colon),
    otherwise 'neutral'.
    """
    lower_text = text.lower()

    # 1. Rude / Crude heuristic check
    if any(marker in lower_text for marker in rude_markers):
        return "rude"
    if "!!" in text or text.count("!") >= 2:
        return "rude"

    # All-caps words of length >= 2
    words = text.split()
    for w in words:
        w_clean = "".join(c for c in w if c.isalpha())
        if len(w_clean) >= 2 and w_clean.isupper():
            return "rude"

    # 2. Label heuristic check
    clean_ends = not any(lower_text.endswith(p) for p in [".", "?", "!"])
    if len(text) <= 25 and (clean_ends or "—" in text or "-" in text or ":" in text):
        return "label"

    return "neutral"


# ---------------------------------------------------------------------------
# Bundle building and response validation
# ---------------------------------------------------------------------------


def _build_bundle(usable: list[dict], locked: list[dict], config) -> dict:
    compact_glossary = [
        {"source_term": t["source_term"], "target_term": t["target_term"]}
        for t in locked
    ]
    regions = [
        {"region_id": r["region_id"], "literal_translation": r["literal_translation"]}
        for r in usable
    ]
    return {
        "READ_FIRST": (
            "Follow these rules EXACTLY before producing output. "
            "These are mandatory instructions, not context.\n" + _PROMPT_INSTRUCTIONS
        ),
        "tone_directive": getattr(
            config,
            "PARAPHRASE_TONE_DIRECTIVE",
            "preserve crude/rude register; casual US English",
        ),
        "shorten_hint": f"Target length: under {getattr(config, 'PARAPHRASE_MAX_CHARS', 90)} characters",
        "glossary": compact_glossary,
        "regions": regions,
    }


def _validate_response(raw, usable_ids: list[str]) -> tuple[dict, list[str]]:
    """Validate raw response dict; return (clean_map, warnings)."""
    warnings = []
    if not isinstance(raw, dict):
        raise ValueError(
            "paraphrase_response.json must be a JSON object mapping region_id -> string."
        )

    clean = {}
    for rid, val in raw.items():
        if rid not in usable_ids:
            warnings.append(f"Unexpected region_id in response (ignored): {rid!r}")
            continue
        if not isinstance(val, str) or not val.strip():
            warnings.append(f"Empty/non-string value for {rid!r} - treated as missing.")
            continue
        if _cjk_ratio(val) > 0.30:
            warnings.append(
                f"Residual CJK in paraphrase for {rid!r} - treated as missing."
            )
            continue
        clean[rid] = val.strip()

    for rid in usable_ids:
        if rid not in clean:
            warnings.append(
                f"Missing paraphrase for {rid!r} — will fall back to literal."
            )

    return clean, warnings


# ---------------------------------------------------------------------------
# Top-level orchestrator
# ---------------------------------------------------------------------------


def _partition_regions_paraphrase(
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
        elif r.get("translated"):
            usable.append(r)
        else:
            skipped.append(r)

    for r in overridden_regions:
        rid = r["region_id"]
        if log:
            logger.info(
                "[%d/%d %s] %s -> using override (final, not re-paraphrased)",
                _STAGE_INDEX,
                _TOTAL_STAGES,
                _STAGE_NAME,
                rid,
            )

    if log:
        logger.info(
            "[%d/%d %s] Backend: %s",
            _STAGE_INDEX,
            _TOTAL_STAGES,
            _STAGE_NAME,
            getattr(config, "PARAPHRASE_BACKEND", "manual"),
        )
        logger.info(
            "[%d/%d %s] %d paraphrasable regions, %d overridden, %d skipped",
            _STAGE_INDEX,
            _TOTAL_STAGES,
            _STAGE_NAME,
            len(usable),
            len(overridden_regions),
            len(skipped),
        )
    return overridden_regions, usable, skipped


def run_paraphrase(workspace: str, config) -> Path | None:
    """Run the Paraphrase stage.

    Returns the path to paraphrase.json on success, or None if awaiting manual input.
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

    trans_path = ws / config.STAGE_FOLDERS["translation"] / "translation.json"
    if not trans_path.exists():
        raise FileNotFoundError("translation.json not found. Run translate first.")

    with trans_path.open("r", encoding="utf-8") as fh:
        trans_data = json.load(fh)

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

    all_results = trans_data.get("results", [])
    overridden_regions, usable, skipped = _partition_regions_paraphrase(
        all_results, overrides, config
    )

    glossary = load_series_glossary(ws.parent, config)
    locked = _locked_terms(glossary)

    # Short-circuit: nothing to paraphrase
    if not usable:
        logger.info(
            "[%d/%d %s] No paraphrasable regions needing LLM paraphrase — completing without handoff.",
            _STAGE_INDEX,
            _TOTAL_STAGES,
            _STAGE_NAME,
        )
        return _write_output(
            ParaphraseWriteContext(
                ws=ws,
                config=config,
                manifest=manifest,
                trans_data=trans_data,
                paraphrase_map={},
                usable=usable,
                skipped=skipped,
                overridden_regions=overridden_regions,
                overrides=overrides,
                locked=locked,
                t0=t0,
            )
        )

    bundle = _build_bundle(usable, locked, config)
    backend = _get_backend(config)
    raw_response = backend.request(bundle, ws, config)

    if raw_response is None:
        return None

    # Validate
    usable_ids = [r["region_id"] for r in usable]
    try:
        paraphrase_map, val_warnings = _validate_response(raw_response, usable_ids)
    except ValueError as exc:
        raise ValueError(
            f"paraphrase_response.json is malformed: {exc}. "
            "Fix it and re-run: python pipeline.py paraphrase"
        ) from exc

    for w in val_warnings:
        logger.warning("[%s] %s", _STAGE_NAME, w)

    backend_name = getattr(config, "PARAPHRASE_BACKEND", "manual")
    if backend_name == "ollama" and usable_ids:
        ratio = len(paraphrase_map) / len(usable_ids)
        min_ratio = getattr(config, "OLLAMA_PARA_MIN_COMPLETION_RATIO", 0.80)
        if ratio < min_ratio:
            logger.error(
                "[%d/%d %s] Paraphrase completion %.0f%% < %.0f%% - writing artifacts "
                "(literal fallback applied) but NOT advancing manifest.",
                _STAGE_INDEX,
                _TOTAL_STAGES,
                _STAGE_NAME,
                ratio * 100,
                min_ratio * 100,
            )
            _write_output(
                ParaphraseWriteContext(
                    ws=ws,
                    config=config,
                    manifest=manifest,
                    trans_data=trans_data,
                    paraphrase_map=paraphrase_map,
                    usable=usable,
                    skipped=skipped,
                    overridden_regions=overridden_regions,
                    overrides=overrides,
                    locked=locked,
                    t0=t0,
                    advance=False,
                )
            )
            return None
        if ratio < 1.0:
            logger.warning(
                "[%d/%d %s] Paraphrase completion %.0f%% (advancing; literal fallback "
                "for the remainder).",
                _STAGE_INDEX,
                _TOTAL_STAGES,
                _STAGE_NAME,
                ratio * 100,
            )

    return _write_output(
        ParaphraseWriteContext(
            ws=ws,
            config=config,
            manifest=manifest,
            trans_data=trans_data,
            paraphrase_map=paraphrase_map,
            usable=usable,
            skipped=skipped,
            overridden_regions=overridden_regions,
            overrides=overrides,
            locked=locked,
            t0=t0,
        )
    )


@dataclass
class ParaphraseWriteContext:
    ws: Path
    config: object
    manifest: dict
    trans_data: dict
    paraphrase_map: dict
    usable: list[dict]
    skipped: list[dict]
    overridden_regions: list[dict]
    overrides: dict
    locked: list[dict]
    t0: float
    advance: bool = True


def _write_output(ctx: ParaphraseWriteContext) -> Path:
    ws, config, manifest = ctx.ws, ctx.config, ctx.manifest
    trans_data, paraphrase_map = ctx.trans_data, ctx.paraphrase_map
    usable, skipped, overridden_regions = (
        ctx.usable,
        ctx.skipped,
        ctx.overridden_regions,
    )
    overrides, locked, t0, advance = ctx.overrides, ctx.locked, ctx.t0, ctx.advance

    now = datetime.now(timezone.utc).isoformat()
    results = []
    paraphrased_count = 0
    missing_count = 0
    total_chars = 0
    rude_markers = getattr(config, "PARAPHRASE_RUDE_MARKERS", [])

    # Overridden regions
    for region in overridden_regions:
        rid = region["region_id"]
        override_text = overrides[rid]
        final_text, conflict = _enforce_glossary(
            region.get("literal_translation", ""), override_text, locked
        )
        reg = _detect_register(override_text, rude_markers)
        results.append(
            {
                "region_id": rid,
                "page_number": region["page_number"],
                "literal_translation": region.get("literal_translation")
                or override_text,
                "final_text": final_text,
                "paraphrased": True,
                "register": reg,
                "char_count": len(override_text),
                "skip_reason": None,
                "glossary_conflict": conflict
                or region.get("glossary_conflict")
                or False,
                "paraphrase_source": "override",
            }
        )
        paraphrased_count += 1
        total_chars += len(override_text)

    # Usable regions
    for region in usable:
        rid = region["region_id"]
        final_text = paraphrase_map.get(rid, "")
        if final_text:
            paraphrased = True
            skip_reason = None
            reg = _detect_register(final_text, rude_markers)
            para_source = (
                f"ollama:{getattr(config, 'OLLAMA_PARA_MODEL', '')}"
                if getattr(config, "PARAPHRASE_BACKEND", "") == "ollama"
                else "llm"
            )
        else:
            final_text = region.get("literal_translation") or ""
            paraphrased = False
            skip_reason = None
            reg = _detect_register(final_text, rude_markers)
            para_source = "literal_fallback"
            missing_count += 1

        if paraphrased:
            paraphrased_count += 1
        total_chars += len(final_text)

        # Re-enforce glossary on output text to catch compliance status
        final_text, conflict = _enforce_glossary(
            region.get("literal_translation") or "", final_text, locked
        )

        results.append(
            {
                "region_id": rid,
                "page_number": region["page_number"],
                "literal_translation": region.get("literal_translation") or "",
                "final_text": final_text,
                "paraphrased": paraphrased,
                "register": reg,
                "char_count": len(final_text),
                "skip_reason": skip_reason,
                "glossary_conflict": conflict
                or region.get("glossary_conflict")
                or False,
                "paraphrase_source": para_source,
            }
        )

    # Skipped regions (pass through)
    for region in skipped:
        results.append(
            {
                "region_id": region["region_id"],
                "page_number": region["page_number"],
                "literal_translation": region.get("literal_translation") or "",
                "final_text": region.get("final_text") or "",
                "paraphrased": False,
                "register": region.get("register") or "neutral",
                "char_count": region.get("char_count") or 0,
                "skip_reason": region.get("skip_reason") or "no_usable_text",
                "glossary_conflict": region.get("glossary_conflict") or False,
                "paraphrase_source": None,
            }
        )

    output = {
        "chapter_id": trans_data.get("chapter_id", "unknown"),
        "stage": "paraphrase",
        "generated_at": now,
        "paraphraser_backend": getattr(config, "PARAPHRASE_BACKEND", "manual"),
        "paraphraser_model": getattr(config, "OLLAMA_PARA_MODEL", None)
        if getattr(config, "PARAPHRASE_BACKEND", "") == "ollama"
        else None,
        "paraphraser_temperature": getattr(config, "OLLAMA_PARA_TEMPERATURE", None)
        if getattr(config, "PARAPHRASE_BACKEND", "") == "ollama"
        else None,
        "prompt_version": getattr(
            config, "OLLAMA_PARA_PROMPT_VERSION", "paraphrase-v1"
        ),
        "tone_directive": getattr(
            config,
            "PARAPHRASE_TONE_DIRECTIVE",
            "preserve crude/rude register; casual US English",
        ),
        "results": results,
    }

    para_dir = ws / config.STAGE_FOLDERS["paraphrase"]
    para_dir.mkdir(parents=True, exist_ok=True)
    out_path = para_dir / "paraphrase.json"
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(output, fh, ensure_ascii=False, indent=2)

    if advance:
        completed = manifest.get("completed_stages", [])
        if "paraphrase" not in completed:
            completed.append("paraphrase")
        manifest["completed_stages"] = completed
        manifest["current_stage"] = "render"
        manifest["updated_at"] = now
        save_manifest(ws, config, manifest)

    elapsed = time.monotonic() - t0
    avg_chars = int(total_chars / paraphrased_count) if paraphrased_count > 0 else 0
    log_stage(
        logger,
        _STAGE_INDEX,
        _TOTAL_STAGES,
        _STAGE_NAME,
        f"done: {paraphrased_count} paraphrased, {len(skipped)} passthrough, "
        f"{missing_count} missing; avg {avg_chars} chars -> {out_path} (elapsed {elapsed:.1f}s)",
    )
    return out_path


def build_paraphrase_bundle(chapter_dir: str | Path, config) -> dict:
    """Build paraphrase prompt bundle for the chapter."""
    ws = Path(chapter_dir)
    manifest = load_manifest(ws, config)
    if not manifest:
        raise ValueError("Manifest not found.")

    trans_path = ws / config.STAGE_FOLDERS["translation"] / "translation.json"
    if not trans_path.exists():
        raise FileNotFoundError("translation.json not found. Run translate first.")

    with trans_path.open("r", encoding="utf-8") as fh:
        trans_data = json.load(fh)

    from manhua_pipeline.io.overrides import load_overrides

    overrides = load_overrides(ws, config)

    all_results = trans_data.get("results", [])
    _, usable, _ = _partition_regions_paraphrase(
        all_results, overrides, config, log=False
    )

    glossary = load_series_glossary(ws.parent, config)
    locked = _locked_terms(glossary)

    return _build_bundle(usable, locked, config)


def write_paraphrase_response(
    chapter_dir: str | Path, mapping: dict, config=None
) -> dict:
    """Validate and write the paraphrase mapping to paraphrase_response.json."""
    if config is None:
        import config
    ws = Path(chapter_dir)
    manifest = load_manifest(ws, config)
    if not manifest:
        raise ValueError("Manifest not found.")

    trans_path = ws / config.STAGE_FOLDERS["translation"] / "translation.json"
    if not trans_path.exists():
        raise FileNotFoundError("translation.json not found. Run translate first.")

    with trans_path.open("r", encoding="utf-8") as fh:
        trans_data = json.load(fh)

    from manhua_pipeline.io.overrides import load_overrides

    overrides = load_overrides(ws, config)

    all_results = trans_data.get("results", [])
    _, usable, _ = _partition_regions_paraphrase(
        all_results, overrides, config, log=False
    )

    usable_ids = [r["region_id"] for r in usable]
    clean_map, warnings = _validate_response(mapping, usable_ids)

    para_dir = ws / config.STAGE_FOLDERS["paraphrase"]
    para_dir.mkdir(parents=True, exist_ok=True)
    resp_path = para_dir / config.PARAPHRASE_RESPONSE_NAME
    with resp_path.open("w", encoding="utf-8") as fh:
        json.dump(clean_map, fh, ensure_ascii=False, indent=2)

    return {"written": len(clean_map), "warnings": warnings}
