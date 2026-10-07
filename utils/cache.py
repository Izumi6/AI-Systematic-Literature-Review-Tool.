"""
utils/cache.py
Disk-based JSON caching for API results, paper metadata, and LLM outputs.
"""

import json
import logging
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


class DiskCache:
    """
    Simple key-value store backed by individual JSON files on disk.
    Each key maps to a .json file inside the given directory.
    """

    def __init__(self, cache_dir: Path) -> None:
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        safe_key = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in key)
        return self.cache_dir / f"{safe_key}.json"

    def get(self, key: str) -> Optional[Any]:
        path = self._path(key)
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as fh:
                    return json.load(fh)
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning("Cache read failed for key '%s': %s", key, exc)
        return None

    def set(self, key: str, value: Any) -> None:
        path = self._path(key)
        try:
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(value, fh, ensure_ascii=False, indent=2)
        except OSError as exc:
            logger.warning("Cache write failed for key '%s': %s", key, exc)

    def exists(self, key: str) -> bool:
        return self._path(key).exists()

    def delete(self, key: str) -> None:
        path = self._path(key)
        if path.exists():
            path.unlink()

    def clear(self) -> None:
        for f in self.cache_dir.glob("*.json"):
            f.unlink()

    def keys(self) -> list[str]:
        return [f.stem for f in self.cache_dir.glob("*.json")]
