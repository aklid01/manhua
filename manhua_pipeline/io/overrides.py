# Manual overrides loading and filtering.

import json
from pathlib import Path


# Load region overrides mapping, skipping comments and empty entries
def load_overrides(ws: Path, config) -> dict[str, str]:
    path = ws / getattr(config, "OVERRIDES_NAME", "overrides.json")
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(data, dict):
        return {}

    clean: dict[str, str] = {}
    for region_id, value in data.items():
        if region_id == "_comment":
            continue
        if isinstance(value, str):
            translation = value.strip()
        elif isinstance(value, dict):
            raw = value.get("translation")
            translation = raw.strip() if isinstance(raw, str) else ""
        else:
            translation = ""
        if translation:
            clean[region_id] = translation
    return clean
