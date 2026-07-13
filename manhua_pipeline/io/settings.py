import json
from pathlib import Path

from manhua_pipeline.logging_setup import get_logger

logger = get_logger(__name__)

SETTINGS_PATH = Path(__file__).resolve().parent.parent.parent / "settings.json"


def load_settings() -> dict:
    """Load settings from settings.json, returning empty dict on failure/missing."""
    if not SETTINGS_PATH.exists():
        return {}
    try:
        with SETTINGS_PATH.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception as exc:
        logger.warning("Failed to load settings from %s: %s", SETTINGS_PATH, exc)
        return {}


def save_settings(settings: dict) -> None:
    """Save settings dictionary to settings.json."""
    try:
        with SETTINGS_PATH.open("w", encoding="utf-8") as fh:
            json.dump(settings, fh, ensure_ascii=False, indent=2)
    except Exception as exc:
        logger.warning("Failed to save settings to %s: %s", SETTINGS_PATH, exc)
        pass


def get_output_dir() -> str:
    """Get the saved output_dir from settings."""
    return load_settings().get("output_dir", "")


def set_output_dir(path: str) -> None:
    """Set and persist the output_dir to settings."""
    settings = load_settings()
    settings["output_dir"] = str(Path(path).resolve())
    save_settings(settings)


def resolve_base_dir(args, config) -> Path:
    """Resolve the series base directory following the priority order."""
    # 1. Check CLI args if available
    if args and getattr(args, "output_dir", None):
        return Path(args.output_dir).resolve()

    # 2. Check saved settings
    output_dir = get_output_dir()
    if output_dir:
        return Path(output_dir).resolve()

    # 3. Prompt user (interactive input)
    try:
        ans = input("Where should chapters be stored? This is your series folder: ").strip()
    except (EOFError, Exception) as exc:
        raise ValueError(
            "No output dir set and not interactive. Use --output-dir or --set-output-dir."
        ) from exc

    if not ans:
        raise ValueError("Series folder path cannot be empty.")

    set_output_dir(ans)
    return Path(ans).resolve()
