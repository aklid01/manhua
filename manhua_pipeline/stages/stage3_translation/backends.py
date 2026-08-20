# Translation backends (manual, MCP, and Ollama batch processor).

import json
import time
from pathlib import Path
from typing import Protocol

from manhua_pipeline.logging_setup import get_logger
from manhua_pipeline.stages import _backends as be
from manhua_pipeline.translation_validation import validate_expected_regions

logger = get_logger(__name__)

_STAGE_INDEX = 3
_TOTAL_STAGES = 7
_STAGE_NAME = "Translation"


# Protocol defining interface for translation backend implementations
class TranslatorBackend(Protocol):
    def request(self, bundle: dict, ws: Path, config) -> dict | None:
        pass


# Backend implementing file-based manual translation handoff
class ManualBackend:
    def request(self, bundle: dict, ws: Path, config) -> dict | None:
        paths = be.BackendPaths(
            "translation",
            config.TRANSLATION_PROMPT_NAME,
            config.TRANSLATION_RESPONSE_NAME,
        )
        return be.manual_request(
            bundle, ws, config, paths, logger, _STAGE_INDEX, _TOTAL_STAGES, _STAGE_NAME
        )


# Backend implementing MCP tool-based translation handoff
class McpBackend:
    def request(self, bundle: dict, ws: Path, config) -> dict | None:
        paths = be.BackendPaths(
            "translation",
            config.TRANSLATION_PROMPT_NAME,
            config.TRANSLATION_RESPONSE_NAME,
        )
        return be.mcp_request(
            bundle, ws, config, paths, logger, _STAGE_INDEX, _TOTAL_STAGES, _STAGE_NAME
        )


# Backend implementing batch translation with Ollama and retry handling
class OllamaBackend:
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

        self.ws = ws
        self.config = config
        self._ollama_settings = s
        self.model = s.model
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
            self._batch_call_count = 0
            self._batch_response_chars = 0
            self._batch_unexpected_count = 0
            t_batch = time.monotonic()
            accepted = self._translate_batch(chunk, depth=0)
            merged.update(accepted)
            expected = len(chunk)
            missing = expected - len(accepted)
            logger.info(
                "[%d/%d %s] Batch %d/%d: expected %d, accepted %d, missing %d, unexpected %d, calls %d, response_chars %d (%.1fs)",
                _STAGE_INDEX,
                _TOTAL_STAGES,
                _STAGE_NAME,
                bi + 1,
                n_batches,
                expected,
                len(accepted),
                missing,
                self._batch_unexpected_count,
                self._batch_call_count,
                self._batch_response_chars,
                time.monotonic() - t_batch,
            )

        trans_dir = ws / config.STAGE_FOLDERS["translation"]
        response_path = trans_dir / config.TRANSLATION_RESPONSE_NAME
        with response_path.open("w", encoding="utf-8") as fh:
            json.dump(merged, fh, ensure_ascii=False, indent=2)
        return merged

    # Execute batch translation with targeted missing-subset retries (no recursive tree explosion)
    def _translate_batch(self, chunk, depth: int = 0) -> dict:
        expected_ids = {r["region_id"] for r in chunk}

        raw = self._call_ollama(self._build_user_prompt(chunk, strict=False))
        accepted, missing, unexpected = self._validate_batch(
            self._parse_json(raw, self), expected_ids
        )
        self._batch_unexpected_count = getattr(
            self, "_batch_unexpected_count", 0
        ) + len(unexpected)
        if unexpected:
            logger.warning(
                "[%s] Rejected %d unexpected IDs", _STAGE_NAME, len(unexpected)
            )
        if not missing:
            return accepted

        retry_chunk = [r for r in chunk if r["region_id"] in missing]
        raw = self._call_ollama(self._build_user_prompt(retry_chunk, strict=True))
        acc2, missing, unexpected = self._validate_batch(
            self._parse_json(raw, self), set(missing)
        )
        self._batch_unexpected_count = getattr(
            self, "_batch_unexpected_count", 0
        ) + len(unexpected)
        accepted.update(acc2)
        if not missing:
            return accepted

        retry_chunk2 = [r for r in retry_chunk if r["region_id"] in missing]
        raw = self._call_ollama(self._build_user_prompt(retry_chunk2, strict=True))
        acc3, missing, unexpected = self._validate_batch(
            self._parse_json(raw, self), set(missing)
        )
        self._batch_unexpected_count = getattr(
            self, "_batch_unexpected_count", 0
        ) + len(unexpected)
        accepted.update(acc3)

        if missing:
            logger.warning(
                "[%s] Giving up on %d region(s) after recovery: %s | last raw output: %r",
                _STAGE_NAME,
                len(missing),
                sorted(missing),
                raw[:300],
            )
        return accepted

    # Construct JSON user prompt payload for Ollama translation batch
    def _build_user_prompt(self, chunk, strict: bool) -> str:
        payload = json.dumps(
            [
                {
                    "region_id": r["region_id"],
                    "original_text": r.get("effective_text") or r["original_text"],
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
        return (
            gloss
            + "Translate each object's original_text. Return a JSON object mapping "
            "region_id -> english. REGIONS:\n" + payload + strict_note
        )

    # Send chat request to Ollama backend
    def _call_ollama(self, user_prompt: str) -> str:
        self._batch_call_count = getattr(self, "_batch_call_count", 0) + 1
        raw = be.call_ollama(
            self._ollama_settings, self.system_prompt, user_prompt, logger, _STAGE_NAME
        )
        self._batch_response_chars = getattr(self, "_batch_response_chars", 0) + len(
            raw
        )
        return raw

    # Validate returned batch dictionary against expected region IDs
    @staticmethod
    def _validate_batch(parsed: dict, expected_ids: set) -> tuple[dict, list, list]:
        unexpected = [k for k in parsed if k not in expected_ids]
        if unexpected:
            for k in unexpected:
                logger.debug(
                    "[%s] Unexpected region_id in response (ignored): %r",
                    _STAGE_NAME,
                    k,
                )

        issues = validate_expected_regions(
            {k: v for k, v in parsed.items() if k in expected_ids},
            expected_ids,
        )
        accepted = {
            rid: issue.accepted_text
            for rid, issue in issues.items()
            if issue.status == "accepted"
        }
        for rid, issue in issues.items():
            if issue.status != "accepted":
                logger.debug(
                    "[%s] Region %r rejected: reason=%s",
                    _STAGE_NAME,
                    rid,
                    issue.reason,
                )
        missing = [rid for rid, issue in issues.items() if issue.status != "accepted"]
        return accepted, missing, unexpected

    # Parse response JSON string and emit parsing telemetry events
    @staticmethod
    def _parse_json(text: str, self_obj=None) -> dict:
        from manhua_pipeline.events import (
            BACKEND_JSON_INVALID,
            BACKEND_JSON_NON_OBJECT,
            BACKEND_RESPONSE_BLANK,
            PipelineEvent,
        )
        from manhua_pipeline.io.parse_result import parse_json_result

        result = parse_json_result(text)
        if result.status in ("success", "repaired", "partial_recovered"):
            if result.status != "success":
                logger.info(
                    "[%s] JSON response %s (recovered %d keys)",
                    _STAGE_NAME,
                    result.status,
                    len(result.value),
                )
            return result.value

        event_by_status = {
            "blank": BACKEND_RESPONSE_BLANK,
            "invalid_json": BACKEND_JSON_INVALID,
            "non_object": BACKEND_JSON_NON_OBJECT,
        }
        event_name = event_by_status.get(result.status)
        ws = getattr(self_obj, "ws", None) if self_obj else None
        cfg = getattr(self_obj, "config", None) if self_obj else None
        if event_name and ws and cfg:
            from manhua_pipeline.io.event_emitter import get_event_emitter

            run_id = getattr(cfg, "_run_id", "unknown")
            chapter = ws.name if isinstance(ws, Path) else "unknown"
            get_event_emitter(ws, cfg, logger=logger).emit(
                PipelineEvent(
                    event=event_name,
                    run_id=run_id,
                    stage=_STAGE_NAME.lower(),
                    chapter=chapter,
                    reason=result.error or result.status,
                    backend="ollama",
                    model=getattr(self_obj, "model", None),
                    metadata={"raw_length": result.raw_length},
                )
            )
        return {}


# Instantiate configured translation backend instance
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
