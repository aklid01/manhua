import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class TouchupCache:
    """Manages local cache for touchup edits per page or chapter."""

    def __init__(self, directory: Path | str):
        self.directory = Path(directory)
        self.cache_file = self.directory / ".touchup_cache.json"

    def _load_raw(self) -> dict[str, Any]:
        if not self.cache_file.exists():
            return {"version": 1, "pages": {}}
        try:
            return json.loads(self.cache_file.read_text(encoding="utf-8"))
        except Exception:
            return {"version": 1, "pages": {}}

    def _save_raw(self, data: dict[str, Any]) -> None:
        try:
            self.cache_file.write_text(
                json.dumps(data, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception as exc:
            print(f"[TouchupCache] Warning: Failed to save cache: {exc}")

    def get_edits(self, page_name: str) -> list[dict[str, Any]]:
        data = self._load_raw()
        return data.get("pages", {}).get(page_name, [])

    def save_edit(self, page_name: str, edit_dict: dict[str, Any]) -> str:
        data = self._load_raw()
        pages = data.setdefault("pages", {})
        edits = pages.setdefault(page_name, [])

        edit_id = edit_dict.get("edit_id")
        if not edit_id:
            edit_id = f"edit_{uuid.uuid4().hex[:8]}"
            edit_dict["edit_id"] = edit_id

        edit_dict["updated_at"] = datetime.now(timezone.utc).isoformat()

        # Update existing or append
        for idx, existing in enumerate(edits):
            if existing.get("edit_id") == edit_id:
                edits[idx] = edit_dict
                self._save_raw(data)
                return edit_id

        edits.append(edit_dict)
        self._save_raw(data)
        return edit_id

    def delete_edit(self, page_name: str, edit_id: str) -> bool:
        data = self._load_raw()
        pages = data.get("pages", {})
        if page_name not in pages:
            return False

        original_len = len(pages[page_name])
        pages[page_name] = [e for e in pages[page_name] if e.get("edit_id") != edit_id]
        if len(pages[page_name]) < original_len:
            self._save_raw(data)
            return True
        return False

    def clear_page_edits(self, page_name: str) -> None:
        data = self._load_raw()
        if page_name in data.get("pages", {}):
            del data["pages"][page_name]
            self._save_raw(data)
