# Shared OpenAI-compatible transport for Stage 3 (translate) and Stage 4 (paraphrase).

import concurrent.futures
import json
import random
import re
import socket
import threading
import time
import urllib.error
import urllib.request

_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")


# Retriable provider/API failure (timeouts, 429, 5xx, bad payloads).
class ProviderError(Exception):
    retriable = True

    def __init__(self, message, status=None, retry_after=None, retriable=None):
        super().__init__(message)
        self.status = status
        self.retry_after = retry_after
        if retriable is not None:
            self.retriable = retriable


# Unrecoverable provider failure (bad key, unknown model, no channel).
class FatalProviderError(ProviderError):
    retriable = False


_RETRIABLE_STATUSES = {408, 409, 425, 429, 500, 502, 503, 504}
_FATAL_STATUSES = {400, 401, 402, 403, 404, 422}
_FATAL_PATTERNS = (
    "no available channel",
    "model_not_found",
    "model not found",
    "does not exist",
    "invalid api key",
    "incorrect api key",
    "invalid_api_key",
    "unauthorized",
)


def _short(obj, limit: int = 160) -> str:
    s = str(obj).replace("\n", " ").strip()
    return s if len(s) <= limit else s[: limit - 3] + "..."


def _extract_error_message(body: str) -> str:
    try:
        data = json.loads(body)
    except Exception:
        return ""
    if not isinstance(data, dict):
        return ""
    err = data.get("error")
    if isinstance(err, dict):
        msg = err.get("message") or err.get("code") or ""
        return (msg if isinstance(msg, str) else str(msg))[:300]
    if isinstance(err, str):
        return err[:300]
    for key in ("message", "msg", "detail"):
        val = data.get(key)
        if isinstance(val, str) and val.strip():
            return val[:300]
    return ""


def _build_http_error(status, headers, body) -> ProviderError:
    detail = _extract_error_message(body)
    if not detail and body:
        detail = _short(re.sub(r"\s+", " ", body), 200)
    label = f"HTTP {status}" + (f": {detail}" if detail else "")
    retry_after = None
    try:
        raw_ra = headers.get("Retry-After") if hasattr(headers, "get") else None
        if raw_ra:
            retry_after = float(raw_ra)
    except (TypeError, ValueError):
        retry_after = None
    haystack = f"{detail} {body[:500]}".lower()
    if status in _FATAL_STATUSES or any(p in haystack for p in _FATAL_PATTERNS):
        return FatalProviderError(label, status=status)
    if status in _RETRIABLE_STATUSES or status >= 500:
        return ProviderError(label, status=status, retry_after=retry_after)
    return FatalProviderError(label, status=status)


def _compute_backoff(
    attempt: int, retry_after=None, base: float = 2.0, cap: float = 45.0
) -> float:
    if retry_after:
        try:
            return min(cap, max(1.0, float(retry_after)))
        except (TypeError, ValueError):
            pass
    ceiling = min(cap, base * (2 ** max(0, attempt - 1)))
    return random.uniform(ceiling * 0.5, ceiling)


try:
    import requests as _requests
except Exception:
    _requests = None

_REQ_SESSION = None
_REQ_SESSION_LOCK = threading.Lock()


def _get_requests_session():
    global _REQ_SESSION
    if _requests is None:
        return None
    with _REQ_SESSION_LOCK:
        if _REQ_SESSION is None:
            _REQ_SESSION = _requests.Session()
            adapter = _requests.adapters.HTTPAdapter(
                pool_connections=8, pool_maxsize=16, max_retries=0
            )
            _REQ_SESSION.mount("https://", adapter)
            _REQ_SESSION.mount("http://", adapter)
    return _REQ_SESSION


def _extract_delta_content(chunk: dict) -> str:
    choices = chunk.get("choices") or []
    if not choices:
        return ""
    delta = choices[0].get("delta", {}) or {}
    content = delta.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            part.get("text", "") for part in content if isinstance(part, dict)
        )
    return ""


def _consume_sse(lines) -> str:
    collected = []
    for line in lines:
        if not line:
            continue
        if isinstance(line, (bytes, bytearray)):
            line = line.decode("utf-8", errors="replace")
        line = line.strip()
        if not line.startswith("data:"):
            continue
        data_str = line[5:].strip()
        if data_str == "[DONE]":
            break
        try:
            chunk = json.loads(data_str)
        except Exception:
            continue
        text = _extract_delta_content(chunk)
        if text:
            collected.append(text)
    full_content = "".join(collected)
    if not full_content:
        raise ProviderError("Empty completion collected from stream.")
    return full_content


def _extract_message_content(parsed: dict) -> str:
    choices = (
        parsed.get("choices")
        or (
            parsed.get("data", {}).get("choices")
            if isinstance(parsed.get("data"), dict)
            else None
        )
        or []
    )
    if not choices:
        raise ProviderError(f"Empty choices in API response: {_short(parsed)}")
    msg = choices[0].get("message", {}) or {}
    content = msg.get("content", "")
    if isinstance(content, list):
        content = "".join(
            part.get("text", "") for part in content if isinstance(part, dict)
        )
    content = str(content or "")
    if not content.strip():
        raise ProviderError("Empty content in API response.")
    return content


# Send an OpenAI-compatible chat request (streaming-aware, requests-first).
def call_openai_chat(
    base_url: str,
    headers: dict,
    model: str,
    system_prompt: str,
    user_prompt: str,
    api_key: str = "",
    timeout: float = 120.0,
    stream: bool = True,
    temperature: float = 0.7,
) -> str:
    url = base_url.rstrip("/")
    if not url.endswith("/chat/completions"):
        url = f"{url}/chat/completions"

    req_headers = {
        "Content-Type": "application/json",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    }
    if headers:
        req_headers.update(headers)
    if api_key and "Authorization" not in req_headers:
        req_headers["Authorization"] = f"Bearer {api_key}"

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "stream": stream,
    }

    session = _get_requests_session()
    if session is not None:
        return _chat_via_requests(session, url, req_headers, payload, timeout, stream)
    return _chat_via_urllib(url, req_headers, payload, timeout, stream)


def _chat_via_requests(session, url, req_headers, payload, timeout, stream) -> str:
    try:
        with session.post(
            url,
            headers=req_headers,
            json=payload,
            timeout=(15.0, float(timeout)),
            stream=True,
        ) as resp:
            if resp.status_code >= 400:
                body = ""
                try:
                    resp.encoding = "utf-8"
                    body = resp.text[:1500]
                except Exception:
                    pass
                raise _build_http_error(resp.status_code, resp.headers, body)
            content_type = resp.headers.get("Content-Type", "").lower()
            if stream and "text/event-stream" in content_type:
                return _consume_sse(resp.iter_lines(decode_unicode=False))
            resp.encoding = "utf-8"
            body = resp.text
    except ProviderError:
        raise
    except _requests.exceptions.Timeout as exc:
        raise ProviderError(f"Request timed out after {timeout}s") from exc
    except _requests.exceptions.ConnectionError as exc:
        raise ProviderError(f"Connection error: {_short(exc)}") from exc
    except _requests.exceptions.RequestException as exc:
        raise ProviderError(f"Request error: {_short(exc)}") from exc

    if stream and body.lstrip().startswith("data:"):
        return _consume_sse(body.splitlines())
    try:
        parsed = json.loads(body)
    except Exception as exc:
        raise ProviderError(f"Non-JSON response body: {_short(body)}") from exc
    return _extract_message_content(parsed)


def _chat_via_urllib(url, req_headers, payload, timeout, stream) -> str:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=req_headers, method="POST")
    try:
        resp = urllib.request.urlopen(req, timeout=timeout)
    except urllib.error.HTTPError as exc:
        body = ""
        try:
            body = exc.read().decode("utf-8", errors="replace")[:1500]
        except Exception:
            pass
        raise _build_http_error(exc.code, exc.headers or {}, body) from exc
    except urllib.error.URLError as exc:
        raise ProviderError(f"Connection error: {exc.reason}") from exc
    except (TimeoutError, socket.timeout) as exc:
        raise ProviderError(f"Request timed out after {timeout}s") from exc

    with resp:
        content_type = resp.headers.get("Content-Type", "").lower()
        if stream and "text/event-stream" in content_type:
            return _consume_sse(resp)
        body = resp.read().decode("utf-8", errors="replace")

    if stream and body.lstrip().startswith("data:"):
        return _consume_sse(body.splitlines())
    try:
        parsed = json.loads(body)
    except Exception as exc:
        raise ProviderError(f"Non-JSON response body: {_short(body)}") from exc
    return _extract_message_content(parsed)


_PROBE_CACHE: dict = {}
_PROBE_CACHE_TTL = 600.0
_PROBE_LOCK = threading.Lock()


def _probe_one(prov: dict, probe_timeout: float):
    """Returns (status, msg): True=healthy, False=fatal-dead, None=slow/down."""
    key = f"{prov.get('base_url', '')}|{prov.get('model', '')}"
    now = time.time()
    with _PROBE_LOCK:
        cached = _PROBE_CACHE.get(key)
        if cached and now - cached[0] < _PROBE_CACHE_TTL:
            return cached[1], cached[2]
    try:
        call_openai_chat(
            prov["base_url"],
            prov.get("headers", {}),
            prov["model"],
            "You are a connectivity check. Reply briefly.",
            "Reply with just: OK",
            api_key=prov.get("api_key", ""),
            timeout=probe_timeout,
            stream=False,
        )
        status, msg = True, ""
    except FatalProviderError as exc:
        status, msg = False, str(exc)
    except Exception as exc:
        status, msg = None, str(exc)
    with _PROBE_LOCK:
        _PROBE_CACHE[key] = (time.time(), status, msg)
    return status, msg


def clear_probe_cache() -> None:
    with _PROBE_LOCK:
        _PROBE_CACHE.clear()


# Ping each provider in parallel; drop fatal-dead, demote slow/down to the end.
def probe_and_rank_providers(
    providers: list[dict], timeout: float, logger=None
) -> list[dict]:
    candidates = [
        p
        for p in providers
        if not p.get("disabled", False)
        and p.get("enabled", True)
        and p.get("base_url")
        and p.get("model")
    ]
    if len(candidates) <= 1:
        return candidates

    probe_timeout = max(15.0, min(float(timeout), 45.0))
    if logger is not None:
        logger.info(
            "[openai_compat] Health-checking %d providers (timeout %.0fs)...",
            len(candidates),
            probe_timeout,
        )
    results: dict = {}

    def _worker(idx_prov):
        idx, prov = idx_prov
        results[idx] = _probe_one(prov, probe_timeout)

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=min(12, len(candidates))
    ) as executor:
        list(executor.map(_worker, enumerate(candidates)))

    healthy, demoted = [], []
    for idx, prov in enumerate(candidates):
        status, msg = results.get(idx, (None, "no result"))
        pname = prov.get("name", "Unknown")
        if status is True:
            healthy.append(prov)
            if logger is not None:
                logger.info("[openai_compat] [OK] %s", pname)
        elif status is False:
            if logger is not None:
                logger.info("[openai_compat] [SKIP] %s: %s", pname, _short(msg))
        else:
            demoted.append(prov)
            if logger is not None:
                logger.info(
                    "[openai_compat] [SLOW/DOWN - demoted] %s: %s", pname, _short(msg)
                )
    return healthy + demoted


# Load provider fallback chain and configuration from settings.json
def load_provider_chain(purpose: str, config) -> tuple[dict, list[dict]]:
    try:
        from manhua_pipeline.io.settings import load_settings

        data = load_settings()
    except Exception:
        data = {}
    section_key = (
        "translate_fallback" if purpose == "translate" else "paraphrase_fallback"
    )
    section = data.get(section_key, {}) or {}
    if not isinstance(section, dict):
        section = {}

    raw_providers = section.get("providers", []) or []
    disabled_list = set(section.get("disabled_providers", []) or [])
    active = [
        p
        for p in raw_providers
        if isinstance(p, dict)
        and not p.get("disabled", False)
        and p.get("enabled", True)
        and p.get("name") not in disabled_list
    ]

    prefix = "OPENAI_COMPAT"
    global_cfg = {
        "enabled": bool(section.get("enabled", True)),
        "max_retries_per_model": int(
            section.get(
                "max_retries_per_model",
                getattr(config, f"{prefix}_MAX_RETRIES", 5),
            )
        ),
        "batch_size": int(
            section.get("batch_size", getattr(config, f"{prefix}_BATCH_SIZE", 20))
        ),
        "concurrency": int(
            section.get("concurrency", getattr(config, f"{prefix}_CONCURRENCY", 3))
        ),
        "timeout": float(
            section.get("timeout", getattr(config, f"{prefix}_TIMEOUT", 120))
        ),
    }
    return global_cfg, active


# Sanitize dialogue text (mojibake/quote/dash cleanup, control-char strip).
def sanitize_dialogue_text(text: str) -> str:
    if not text:
        return ""
    if any(
        c in text
        for c in ("\x80", "\x93", "\x94", "\x98", "\x99", "\x9c", "\x9d", "\xa6")
    ):
        try:
            text = text.encode("latin-1").decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            pass

    mojibake_map = {
        "\u00e2\u20ac\u2014": "-",
        "\u00e2\u20ac\u2013": "-",
        "\u00e2\u20ac\u2122": "'",
        "\u00e2\u20ac\u02dc": "'",
        "\u00e2\u20ac\u0153": '"',
        "\u00e2\u20ac\u009d": '"',
        "\u00e2\u20ac\u00a6": "...",
        "\u00c2\u00a0": " ",
    }
    for bad, good in mojibake_map.items():
        if bad in text:
            text = text.replace(bad, good)

    text = text.replace("\u2014", "-")
    text = text.replace("\u2013", "-")
    text = text.replace("\u2015", "-")
    text = text.replace("\u2212", "-")

    text = text.replace("\u2018", "'").replace("\u2019", "'").replace("\u201b", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"').replace("\u201f", '"')

    text = text.replace("\u2026", "...")
    text = text.replace("\u00a0", " ")

    text = "".join(
        c for c in text if c == "\n" or not (0 <= ord(c) <= 31 or 127 <= ord(c) <= 159)
    )
    return text.strip()


# Parse an LLM JSON blob mapping region_id -> string (fence/boundary tolerant).
def parse_llm_json(text: str) -> dict:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        inner = lines[1:] if len(lines) > 1 else lines
        if inner and inner[-1].strip() == "```":
            inner = inner[:-1]
        cleaned = "\n".join(inner).strip()

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        cleaned = cleaned[start : end + 1]

    try:
        data = json.loads(cleaned)
        if isinstance(data, dict):
            return {str(k): sanitize_dialogue_text(str(v)) for k, v in data.items()}
    except Exception:
        pass

    matches = re.findall(
        r'["\'](P\d{3}_R\d{3})["\']\s*:\s*["\']((?:[^"\\]|\\.)*)["\']',
        text,
        re.DOTALL,
    )
    if matches:
        return {
            k: sanitize_dialogue_text(re.sub(r'\\(["\\])', r"\1", v))
            for k, v in matches
        }
    return {}


def contains_cjk(text: str) -> bool:
    return bool(_CJK_RE.search(text or ""))
