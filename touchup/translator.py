import json
import logging
import urllib.request
from pathlib import Path
from typing import Any

logger = logging.getLogger("touchup.translator")


def load_glossary(series_dir: Path | str | None) -> list[dict[str, Any]]:
    """Attempts to find and load glossary.json in the series directory."""
    if not series_dir:
        return []
    path = Path(series_dir) / "glossary.json"
    if not path.exists():
        path = Path(series_dir).parent / "glossary.json"
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data.get("terms", [])
            elif isinstance(data, list):
                return data
        except Exception as exc:
            logger.warning("Failed to parse glossary: %s", exc)
    return []


def resolve_ollama_model(host: str = "http://localhost:11434") -> str:
    """Finds the active or preferred Ollama model from config or localhost tags."""
    # 1. Try reading config.py
    try:
        import config

        for attr in ["OLLAMA_TRANSLATE_MODEL", "OLLAMA_PARA_MODEL", "OLLAMA_MODEL"]:
            val = getattr(config, attr, None)
            if val and isinstance(val, str) and val.strip():
                return val.strip()
    except Exception:
        pass

    # 2. Query Ollama /api/tags for installed models
    try:
        req = urllib.request.Request(f"{host}/api/tags")
        with urllib.request.urlopen(req, timeout=3) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            models = [m.get("name") for m in data.get("models", []) if m.get("name")]
            if models:
                # Prefer models with qwen in their name
                for m in models:
                    if "qwen" in m.lower():
                        return m
                return models[0]
    except Exception:
        pass

    return "qwen2.5:3b-instruct"


def translate_text(
    chinese_text: str,
    series_dir: Path | str | None = None,
    host: str = "http://localhost:11434",
) -> str:
    """Translates Chinese text into English using Ollama API and series glossary."""
    if not chinese_text.strip():
        return ""

    # Check for exact glossary match first
    terms = load_glossary(series_dir)
    clean_chinese = chinese_text.strip()
    for term in terms:
        if isinstance(term, dict):
            src = term.get("source_term", "")
            tgt = term.get("target_term", "")
            if src and src == clean_chinese and tgt:
                return tgt

    # Query Ollama endpoint
    try:
        model = resolve_ollama_model(host)

        glossary_context = ""
        if terms:
            glossary_lines = [
                f"- {t.get('source_term')}: {t.get('target_term')}"
                for t in terms
                if isinstance(t, dict) and t.get("source_term") and t.get("target_term")
            ]
            if glossary_lines:
                glossary_context = (
                    "Apply these glossary terms when applicable:\n"
                    + "\n".join(glossary_lines[:20])
                    + "\n\n"
                )

        prompt = (
            "You are an expert comic and webtoon translator. Translate this Chinese manhua text "
            "into clean, natural, immersive English. Maintain gaming/UI/cultivation terminology.\n"
            "Return ONLY the direct English translation without explanations, notes, or quotes.\n\n"
            f"{glossary_context}"
            f"Text to translate:\n{clean_chinese}"
        )

        req_data = {
            "model": model,
            "prompt": prompt,
            "stream": False,
        }

        req = urllib.request.Request(
            f"{host}/api/generate",
            data=json.dumps(req_data).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )

        with urllib.request.urlopen(req, timeout=20) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            trans = data.get("response", "").strip()
            if (trans.startswith('"') and trans.endswith('"')) or (
                trans.startswith("'") and trans.endswith("'")
            ):
                trans = trans[1:-1].strip()
            if trans:
                return trans

    except Exception as exc:
        logger.warning("Ollama translation request failed: %s", exc)

    return ""
