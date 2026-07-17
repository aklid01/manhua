"""Shared LLM backend plumbing for Stage 3 (translate) and Stage 4 (paraphrase).

Stage-specific behavior (CJK guard vs echo-allowed batch validation, batch
orchestration, bundle building) stays in each stage module. This module holds
only the identical scaffolding: manual/mcp handoff loops and the Ollama HTTP
client with retries + tolerant JSON parsing.
"""

import json
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path


@dataclass
class BackendPaths:
    """Per-stage file/folder wiring."""
    folder_key: str          # e.g. "translation" | "paraphrase"
    prompt_name: str         # config.TRANSLATION_PROMPT_NAME | PARAPHRASE_PROMPT_NAME
    response_name: str       # config.TRANSLATION_RESPONSE_NAME | PARAPHRASE_RESPONSE_NAME


def _stage_dir(ws: Path, config, paths: BackendPaths) -> Path:
    d = ws / config.STAGE_FOLDERS[paths.folder_key]
    d.mkdir(parents=True, exist_ok=True)
    return d


def write_bundle(bundle: dict, ws: Path, config, paths: BackendPaths) -> Path:
    d = _stage_dir(ws, config, paths)
    p = d / paths.prompt_name
    with p.open("w", encoding="utf-8") as fh:
        json.dump(bundle, fh, ensure_ascii=False, indent=2)
    return p


def manual_request(bundle, ws, config, paths, logger, stage_index, total_stages,
                   stage_name) -> dict | None:
    """Manual handoff: write prompt; ingest response file if present, else wait."""
    write_bundle(bundle, ws, config, paths)
    d = _stage_dir(ws, config, paths)
    resp = d / paths.response_name
    if not resp.exists():
        logger.info(
            "[%d/%d %s] Manual handoff required.\n"
            "  1. Open: %s\n"
            "  2. Paste its contents to your coding assistant.\n"
            "  3. Save the assistant's JSON reply as: %s\n"
            "  4. Re-run: python pipeline.py %s\n"
            "  (Awaiting %s — no changes written.)",
            stage_index, total_stages, stage_name,
            d / paths.prompt_name, resp, stage_name.lower(), paths.response_name,
        )
        return None
    with resp.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def mcp_request(bundle, ws, config, paths, logger, stage_index, total_stages,
                stage_name) -> dict | None:
    """MCP handoff: write prompt; ingest response file if present, else wait."""
    write_bundle(bundle, ws, config, paths)
    d = _stage_dir(ws, config, paths)
    resp = d / paths.response_name
    if not resp.exists():
        logger.info(
            "[%d/%d %s] MCP handoff required.\n"
            "  (Awaiting %s via MCP tool — no changes written.)",
            stage_index, total_stages, stage_name, paths.response_name,
        )
        return None
    with resp.open("r", encoding="utf-8") as fh:
        return json.load(fh)


@dataclass
class OllamaSettings:
    host: str
    model: str
    timeout: float
    temperature: float
    max_retries: int
    backoff: float
    batch_size: int = 15

    @classmethod
    def from_config(cls, config, prefix: str) -> "OllamaSettings":
        """prefix = 'OLLAMA' (translate) or 'OLLAMA_PARA' (paraphrase)."""
        g = lambda k, d: getattr(config, f"{prefix}_{k}", d)
        model_key = "TRANSLATE_MODEL" if prefix == "OLLAMA" else "MODEL"
        model = getattr(config, f"{prefix}_{model_key}", "")
        if not model and prefix == "OLLAMA":
            model = getattr(config, "OLLAMA_MODEL", "")
        return cls(
            host=g("HOST", "http://localhost:11434"),
            model=model,
            timeout=g("TIMEOUT", 120),
            temperature=g("TEMPERATURE", 0.2),
            max_retries=g("MAX_RETRIES", 3),
            backoff=g("RETRY_BACKOFF", 1.0),
            batch_size=g("BATCH_SIZE", 15),
        )


def validate_ollama_settings(s: OllamaSettings, prefix: str) -> None:
    if not isinstance(s.batch_size, int) or s.batch_size <= 0:
        raise ValueError(f"{prefix}_BATCH_SIZE must be a positive integer.")
    if not isinstance(s.timeout, (int, float)) or s.timeout <= 0:
        raise ValueError(f"{prefix}_TIMEOUT must be a positive number.")
    if not isinstance(s.temperature, (int, float)) or s.temperature < 0:
        raise ValueError(f"{prefix}_TEMPERATURE must be numeric and non-negative.")
    if not s.host or not str(s.host).startswith(("http://", "https://")):
        raise ValueError(f"{prefix}_HOST must be a non-empty http(s) URL.")
    if not s.model:
        raise ValueError(f"{prefix}_MODEL must be non-empty.")
    if not isinstance(s.max_retries, int) or s.max_retries < 1:
        raise ValueError(f"{prefix}_MAX_RETRIES must be an integer >= 1.")


def call_ollama(s: OllamaSettings, system_prompt: str, user_prompt: str,
                logger, stage_name: str) -> str:
    payload = {
        "model": s.model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
        "format": "json",
        "options": {"temperature": s.temperature},
    }
    data = json.dumps(payload).encode("utf-8")
    last_exc = None
    for attempt in range(1, s.max_retries + 1):
        try:
            req = urllib.request.Request(
                f"{s.host.rstrip('/')}/api/chat",
                data=data, headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=s.timeout) as resp:
                body = json.loads(resp.read().decode("utf-8"))
            return body.get("message", {}).get("content", "")
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            last_exc = exc
            if attempt < s.max_retries:
                wait = s.backoff * (2 ** (attempt - 1))
                logger.warning(
                    "[%s] Ollama call failed (attempt %d/%d): %s; retrying in %.1fs",
                    stage_name, attempt, s.max_retries, exc, wait,
                )
                time.sleep(wait)
    raise RuntimeError(
        f"Ollama request failed after {s.max_retries} attempts "
        f"(host={s.host}, model={s.model}): {last_exc}. "
        "Is `ollama serve` running and the model pulled (`ollama pull ...`)?"
    )


def parse_json(text: str) -> dict:
    """Tolerant parse: strip code fences, extract outermost {...}."""
    if not text or not text.strip():
        return {}
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z]*\n?", "", cleaned)
        cleaned = re.sub(r"\n?```$", "", cleaned).strip()
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        cleaned = cleaned[start:end + 1]
    try:
        obj = json.loads(cleaned)
    except json.JSONDecodeError:
        return {}
    return obj if isinstance(obj, dict) else {}
