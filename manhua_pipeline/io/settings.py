# Persistent settings management and series base directory resolution.

import json
from pathlib import Path

from manhua_pipeline.logging_setup import get_logger

logger = get_logger(__name__)

SETTINGS_PATH = Path(__file__).resolve().parent.parent.parent / "settings.json"


# Load settings from settings.json
def load_settings() -> dict:
    if not SETTINGS_PATH.exists():
        return {}
    try:
        with SETTINGS_PATH.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception as exc:
        logger.warning("Failed to load settings from %s: %s", SETTINGS_PATH, exc)
        return {}


# Save settings dictionary to settings.json
def save_settings(settings: dict) -> None:
    try:
        with SETTINGS_PATH.open("w", encoding="utf-8") as fh:
            json.dump(settings, fh, ensure_ascii=False, indent=2)
    except Exception as exc:
        logger.warning("Failed to save settings to %s: %s", SETTINGS_PATH, exc)


# Retrieve saved series output directory from settings
def get_output_dir() -> str:
    return load_settings().get("output_dir", "")


# Persist configured series output directory to settings
def set_output_dir(path: str) -> None:
    settings = load_settings()
    settings["output_dir"] = str(Path(path).resolve())
    save_settings(settings)


# Resolve series base directory via CLI argument, saved setting, or prompt
def resolve_base_dir(args, config) -> Path:
    if args and getattr(args, "output_dir", None):
        return Path(args.output_dir).resolve()

    output_dir = get_output_dir()
    if output_dir:
        return Path(output_dir).resolve()

    try:
        ans = input(
            "Where should chapters be stored? This is your series folder: "
        ).strip()
    except (EOFError, OSError) as exc:
        raise ValueError(
            "No output dir set and not interactive. Use --output-dir or --set-output-dir."
        ) from exc

    if not ans:
        raise ValueError("Series folder path cannot be empty.")

    set_output_dir(ans)
    return Path(ans).resolve()


_CREDITS_DEFAULTS = {
    "scanlator": "aklid",
    "pipeline_name": "Manhua Pipeline",
    "pipeline_url": "github.com/aklid01",
}


# Retrieve credits dictionary merged with defaults
def get_credits() -> dict:
    saved = load_settings().get("credits", {}) or {}
    return {**_CREDITS_DEFAULTS, **{k: v for k, v in saved.items() if v}}
