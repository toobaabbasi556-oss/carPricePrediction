"""Config loading + path resolution (SPEC §8: config-driven, no hardcoded paths).

Everything reads ``config/config.yaml``; nothing in ``.py`` files hardcodes params.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml


def repo_root() -> Path:
    """Repo root (this file lives at ``src/pkcar/config.py``)."""
    return Path(__file__).resolve().parents[2]


@lru_cache(maxsize=8)
def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load ``config/config.yaml`` (or an explicit path) as a plain dict."""
    cfg_path = Path(path) if path is not None else repo_root() / "config" / "config.yaml"
    with open(cfg_path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def resolve_path(rel: str | Path) -> Path:
    """Resolve a config path relative to the repo root (absolute paths pass through)."""
    p = Path(rel)
    return p if p.is_absolute() else repo_root() / p
