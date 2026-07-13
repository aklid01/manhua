"""Stage 4: Paraphrase.

Rewrites literal translations into natural, casual US English optimized for comic speech bubbles.
Uses pluggable backends (manual, mcp, ollama) and supports manual JSON handoff.
Classifies register (rude, label, neutral) using a local heuristic.
"""

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

from manhua_pipeline.io.workspace import load_manifest, save_manifest
from manhua_pipeline.logging_setup import get_logger, log_stage

logger = get_logger(__name__)

_STAGE_INDEX = 4
_TOTAL_STAGES = 7
_STAGE_NAME = "Paraphrase"

_PROMPT_INSTRUCTIONS = (
    "Rewrite each literal English line as natural, casual, SPOKEN US English for a "
    "comic speech bubble. Preserve meaning and emotional register (including crude/rude "
    "tone — do NOT sanitize). Rewrite AGGRESSIVELY: prefer idiomatic phrasing, "
    "contractions, and varied sentence structure over line-by-line literal wording. "
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
        para_dir = ws / config.STAGE_FOLDERS["paraphrase"]
        para_dir.mkdir(parents=True, exist_ok=True)

        prompt_path = para_dir / config.PARAPHRASE_PROMPT_NAME
        response_path = para_dir / config.PARAPHRASE_RESPONSE_NAME

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
                "  4. Re-run: python pipeline.py paraphrase\n"
                "  (Awaiting paraphrase_response.json — no changes written.)",
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
            "MCP backend is not yet implemented. Set PARAPHRASE_BACKEND='manual' in config.py."
        )


class OllamaBackend:
    def request(self, bundle: dict, ws: Path, config) -> dict | None:
        raise NotImplementedError(
            "Ollama backend is not yet implemented. Set PARAPHRASE_BACKEND='manual' in config.py."
        )


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


def _load_glossary(ws: Path, config) -> dict:
    path = ws / config.GLOSSARY_NAME
    if path.exists():
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    return {"version": "v1", "terms": []}


def _locked_terms(glossary: dict) -> list[dict]:
    return [t for t in glossary.get("terms", []) if t.get("locked")]


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
        "instructions": _PROMPT_INSTRUCTIONS,
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
            warnings.append(f"Empty/non-string value for {rid!r} — treated as missing.")
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
    all_results: list[dict], overrides: dict, config
) -> tuple[list[dict], list[dict], list[dict]]:
    region_ids_set = {r["region_id"] for r in all_results}
    for k in overrides:
        if k not in region_ids_set:
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
        logger.info(
            "[%d/%d %s] %s -> using override (final, not re-paraphrased)",
            _STAGE_INDEX,
            _TOTAL_STAGES,
            _STAGE_NAME,
            rid,
        )

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

    glossary = _load_glossary(ws, config)
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
            ws,
            config,
            manifest,
            trans_data,
            {},
            usable,
            skipped,
            overridden_regions,
            overrides,
            t0,
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

    return _write_output(
        ws,
        config,
        manifest,
        trans_data,
        paraphrase_map,
        usable,
        skipped,
        overridden_regions,
        overrides,
        t0,
    )


def _write_output(
    ws: Path,
    config,
    manifest: dict,
    trans_data: dict,
    paraphrase_map: dict,
    usable: list[dict],
    skipped: list[dict],
    overridden_regions: list[dict],
    overrides: dict,
    t0: float,
) -> Path:
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
        reg = _detect_register(override_text, rude_markers)
        results.append(
            {
                "region_id": rid,
                "page_number": region["page_number"],
                "literal_translation": region.get("literal_translation")
                or override_text,
                "final_text": override_text,
                "paraphrased": True,
                "register": reg,
                "char_count": len(override_text),
                "skip_reason": None,
                "glossary_conflict": region.get("glossary_conflict") or False,
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
            para_source = "llm"
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
                "glossary_conflict": region.get("glossary_conflict") or False,
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

    # Advance manifest
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
