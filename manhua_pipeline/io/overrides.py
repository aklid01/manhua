import json
from pathlib import Path


def load_overrides(ws: Path, config) -> dict:
    """Return {region_id: english} for NON-EMPTY overrides only. Missing file -> {}. Ignores the _comment key."""
    path = ws / getattr(config, "OVERRIDES_NAME", "overrides.json")
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return {
        k: v.strip()
        for k, v in data.items()
        if k != "_comment" and isinstance(v, str) and v.strip()
    }
