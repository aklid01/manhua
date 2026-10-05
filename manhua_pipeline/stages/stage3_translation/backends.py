# Translation backends (manual, MCP, Ollama, and OpenAI-compatible batch processors).

import json
import time
from pathlib import Path
from typing import Protocol

from manhua_pipeline.logging_setup import get_logger
from manhua_pipeline.stages import _backends as be
from manhua_pipeline.translation_validation import validate_expected_regions

logger = get_logger(__name__)

# System prompt hardening for OpenAI-compatible translation models
_OPENAI_TRANSLATION_SYSTEM_HARDENING = (
    "\n\nOPENAI-COMPAT HARDENING (mandatory, overrides any conflicting style):\n"
    "- ZERO CJK IN VALUES: every output value must be 100% English (Latin alphabet). "
    "If any Chinese names, titles, or honorifics remain (e.g. \u536b\u5927\u54e5, \u5e08\u5085, \u5b97\u4e3b), "
    "romanize or translate them into natural English "
    "(e.g. 'Brother Wei', 'Master', 'Sect Master'). NEVER leave CJK characters in values.\n"
    "- LITERAL, NOT LOCALIZED: faithful literal translation. Do NOT paraphrase slang, "
    "do NOT sanitize profanity, do NOT drop repetitions/honorifics/fragments.\n"
    "- GLOSSARY VERBATIM: when the glossary maps X = Y and X appears in the source, "
    "the output must contain Y exactly.\n"
    "- JSON ONLY: return ONLY a valid JSON object mapping region_id -> english string. "
    "No markdown fences, no prose, no explanations. One key per input region. "
    "Do NOT merge, omit, or invent regions.\n"
    "- WATERMARKS: site tags, scanlator credits, logos, URLs/domains "
    "(baozimh, tencent, colamanga, www., .com, \u5305\u5b50\u6f2b\u753b) map to an empty string. "
    "Mixed dialogue + watermark: keep only the dialogue."
)

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
            parsed_last = self._parse_json(raw, self)
            missing_vals = {
                rid: parsed_last.get(rid, "<missing_key>") for rid in sorted(missing)
            }
            logger.warning(
                "[%s] Giving up on %d region(s) after recovery: %s | values: %r",
                _STAGE_NAME,
                len(missing),
                sorted(missing),
                missing_vals,
            )
        return accepted

    # Construct JSON user prompt payload for Ollama translation batch
    def _build_user_prompt(self, chunk, strict: bool) -> str:
        payload = json.dumps(
            [
                {
                    "region_id": r["region_id"],
                    "chinese_text": r.get("effective_text") or r["original_text"],
                }
                for r in chunk
            ],
            ensure_ascii=False,
        )
        if strict:
            strict_note = (
                "\nCRITICAL: You MUST translate every 'chinese_text' into fluent US ENGLISH. "
                "Output values MUST be in English (never return Chinese text). "
                "Return ONLY a valid JSON object mapping region_id -> english_text, e.g. "
                '{"P001_R001": "English translation here"}. No prose, no code fences.'
            )
        else:
            strict_note = (
                "\nTranslate each 'chinese_text' into natural US English. "
                "Output values MUST be in English. "
                'Return a JSON object mapping region_id -> english_translation, e.g. {"P001_R001": "..."}.'
            )
        gloss = (
            ("GLOSSARY (keep exactly):\n" + self.gloss_txt + "\n\n")
            if self.gloss_txt
            else ""
        )
        return (
            gloss
            + "TASK: Translate the following Chinese dialogue into faithful US English.\n"
            "INPUT REGIONS:\n" + payload + strict_note
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
        from manhua_pipeline.translation_validation import canonicalize_region_keys

        parsed = canonicalize_region_keys(parsed, expected_ids)
        unexpected = [k for k in parsed if k not in expected_ids]
        if unexpected:
            for k in unexpected:
                logger.debug(
                    "[%s] Unexpected region_id in response (ignored): %r",
                    _STAGE_NAME,
                    k,
                )

        issues = validate_expected_regions(
            parsed,
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


# Backend implementing OpenAI-compatible translation with provider failover
class OpenAICompatBackend:
    def request(self, bundle: dict, ws: Path, config) -> dict | None:
        from manhua_pipeline.stages._openai_compat import (
            load_provider_chain,
            probe_and_rank_providers,
        )

        global_cfg, providers = load_provider_chain("translate", config)
        if not global_cfg.get("enabled", True):
            logger.info(
                "[%d/%d %s] openai_compat disabled in settings.json.",
                _STAGE_INDEX,
                _TOTAL_STAGES,
                _STAGE_NAME,
            )
            return None
        if not providers:
            logger.warning(
                "[%d/%d %s] openai_compat: no providers in "
                "settings.json translate_fallback.",
                _STAGE_INDEX,
                _TOTAL_STAGES,
                _STAGE_NAME,
            )
            return None

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
        self.system_prompt = (bundle.get("READ_FIRST", "") or "").strip() + (
            _OPENAI_TRANSLATION_SYSTEM_HARDENING
        )
        glossary = bundle.get("glossary", [])
        self.gloss_txt = "\n".join(
            f"{t['source_term']} = {t['target_term']}" for t in glossary
        )
        self.max_retries = int(global_cfg.get("max_retries_per_model", 5))
        self.default_bs = int(global_cfg.get("batch_size", 20))
        self.default_concurrency = int(global_cfg.get("concurrency", 1))
        self.timeout = float(global_cfg.get("timeout", 120.0))
        self.temperature = float(
            getattr(
                config,
                "OPENAI_COMPAT_TRANSLATE_TEMPERATURE",
                getattr(config, "OLLAMA_TEMPERATURE", 0.2),
            )
        )

        if bool(getattr(config, "OPENAI_COMPAT_PROBE_ENABLED", True)):
            providers = probe_and_rank_providers(providers, self.timeout, logger)
        if not providers:
            logger.warning(
                "[%d/%d %s] openai_compat: no healthy providers.",
                _STAGE_INDEX,
                _TOTAL_STAGES,
                _STAGE_NAME,
            )
            return None

        merged: dict = {}
        for prov in providers:
            needed = [r for r in regions if r["region_id"] not in merged]
            if not needed:
                break
            name = prov.get("name", "Unknown")
            base_url = prov.get("base_url", "")
            model = prov.get("model", "")
            if not base_url or not model:
                continue
            prov_bs = int(prov.get("batch_size") or self.default_bs)
            prov_conc = int(prov.get("concurrency") or self.default_concurrency or 1)
            prov_conc = max(1, prov_conc)
            n_batches = (len(needed) + prov_bs - 1) // prov_bs
            logger.info(
                "[%d/%d %s] openai_compat: %s (%s) | %d pending -> %d batch(es), "
                "bs=%d conc=%d",
                _STAGE_INDEX,
                _TOTAL_STAGES,
                _STAGE_NAME,
                name,
                model,
                len(needed),
                n_batches,
                prov_bs,
                prov_conc,
            )
            chunks = [
                needed[bi * prov_bs : (bi + 1) * prov_bs] for bi in range(n_batches)
            ]
            import concurrent.futures
            import threading

            stop = threading.Event()
            lock = threading.Lock()

            def _one(bi: int, chunk: list[dict]):
                accepted = self._translate_via_provider(
                    prov, chunk, lock, bi, n_batches
                )
                if accepted is None:
                    stop.set()
                    return {}
                return accepted

            if prov_conc > 1 and len(chunks) > 1:
                with concurrent.futures.ThreadPoolExecutor(max_workers=prov_conc) as ex:
                    futs = [ex.submit(_one, bi, ch) for bi, ch in enumerate(chunks)]
                    for f in concurrent.futures.as_completed(futs):
                        try:
                            merged.update(f.result() or {})
                        except Exception:
                            pass
                        if stop.is_set():
                            break
            else:
                for bi, ch in enumerate(chunks):
                    merged.update(_one(bi, ch) or {})
                    if stop.is_set():
                        break

        trans_dir = ws / config.STAGE_FOLDERS["translation"]
        response_path = trans_dir / config.TRANSLATION_RESPONSE_NAME
        with response_path.open("w", encoding="utf-8") as fh:
            json.dump(merged, fh, ensure_ascii=False, indent=2)
        return merged

    def _translate_via_provider(self, prov, chunk, lock, bi: int, n_batches: int):
        from manhua_pipeline.stages._openai_compat import (
            FatalProviderError,
            _compute_backoff,
            call_openai_chat,
        )

        name = prov.get("name", "Unknown")
        expected_ids = {r["region_id"] for r in chunk}
        accepted: dict = {}
        with lock:
            self._batch_calls = 0
        for attempt in range(1, self.max_retries + 1):
            todo = [r for r in chunk if r["region_id"] not in accepted]
            if not todo:
                break
            to_send = todo
            send_ids = {r["region_id"] for r in to_send}
            strict = attempt > 1
            prompt = self._build_openai_user_prompt(to_send, strict=strict)
            try:
                raw = call_openai_chat(
                    prov.get("base_url", ""),
                    prov.get("headers", {}),
                    prov.get("model", ""),
                    self.system_prompt,
                    prompt,
                    api_key=prov.get("api_key", ""),
                    timeout=self.timeout,
                    stream=bool(prov.get("stream", True)),
                    temperature=self.temperature,
                )
                parsed = self._parse_json(raw)
                acc, missing, unexpected = self._validate_batch(parsed, send_ids)
                for k, v in acc.items():
                    if k not in accepted:
                        accepted[k] = v
                with lock:
                    logger.info(
                        "[%s] [%s Batch %d/%d] attempt %d/%d: accepted %d/%d",
                        _STAGE_NAME,
                        name,
                        bi + 1,
                        n_batches,
                        attempt,
                        self.max_retries,
                        len(accepted),
                        len(expected_ids),
                    )
                if unexpected:
                    logger.warning(
                        "[%s] Rejected %d unexpected IDs",
                        _STAGE_NAME,
                        len(unexpected),
                    )
                if len(accepted) == len(expected_ids):
                    break
            except FatalProviderError as exc:
                with lock:
                    logger.warning(
                        "[%s] [%s] fatal, abandoning provider: %s",
                        _STAGE_NAME,
                        name,
                        exc,
                    )
                return None
            except Exception as exc:
                retry_after = getattr(exc, "retry_after", None)
                with lock:
                    logger.warning(
                        "[%s] [%s Batch %d/%d] attempt %d failed: %s",
                        _STAGE_NAME,
                        name,
                        bi + 1,
                        n_batches,
                        attempt,
                        exc,
                    )
                if attempt < self.max_retries:
                    time.sleep(_compute_backoff(attempt, retry_after))
        return accepted

    # Construct JSON user prompt payload for OpenAI-compatible translation batch
    def _build_openai_user_prompt(self, chunk, strict: bool) -> str:
        payload = json.dumps(
            [
                {
                    "region_id": r["region_id"],
                    "chinese_text": r.get("effective_text") or r["original_text"],
                }
                for r in chunk
            ],
            ensure_ascii=False,
        )
        gloss = (
            ("GLOSSARY (keep exactly, verbatim in output):\n" + self.gloss_txt + "\n\n")
            if self.gloss_txt
            else ""
        )
        base = (
            gloss
            + "TASK: Translate each 'chinese_text' into faithful, literal US English.\n"
            "CRITICAL REQUIREMENTS:\n"
            "- Output values MUST be 100% English with ZERO Chinese/CJK characters. "
            "Romanize or translate any leftover names/titles/honorifics "
            "(e.g. \u536b\u5927\u54e5->'Brother Wei', \u5e08\u5085->'Master'). NEVER output Chinese text.\n"
            "- Translate every region exactly once. Do NOT merge, omit, or invent regions.\n"
            "- Apply glossary terms exactly as given.\n"
            "- Watermarks, site tags, logos, URLs/domains map to an empty string; "
            "mixed dialogue + watermark: keep only the dialogue.\n"
            "INPUT REGIONS:\n" + payload
        )
        if strict:
            return (
                base
                + "\nSTRICT: Return ONLY a valid JSON object mapping region_id -> english_text, "
                'e.g. {"P001_R001": "English here"}. No prose, no code fences, '
                "no explanations. Every input region_id must appear exactly once."
            )
        return (
            base + "\nReturn a JSON object mapping region_id -> english_translation, "
            'e.g. {"P001_R001": "..."}. No prose.'
        )

    # Validate returned batch dictionary against expected region IDs
    @staticmethod
    def _validate_batch(parsed: dict, expected_ids: set) -> tuple[dict, list, list]:
        from manhua_pipeline.translation_validation import canonicalize_region_keys

        parsed = canonicalize_region_keys(parsed, expected_ids)
        unexpected = [k for k in parsed if k not in expected_ids]
        issues = validate_expected_regions(parsed, expected_ids, source="openai_compat")
        accepted = {
            rid: issue.accepted_text
            for rid, issue in issues.items()
            if issue.status == "accepted"
        }
        missing = [rid for rid, issue in issues.items() if issue.status != "accepted"]
        return accepted, missing, unexpected

    # Parse JSON response from OpenAI-compatible model output
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
                    backend="openai_compat",
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
    if name == "openai_compat":
        return OpenAICompatBackend()
    raise ValueError(
        f"Unknown TRANSLATOR_BACKEND: {name!r}. Use 'manual', 'mcp', 'ollama', or 'openai_compat'."
    )
