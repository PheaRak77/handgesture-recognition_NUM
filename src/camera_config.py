"""Per-camera settings for gesture detection (YAML)."""

from __future__ import annotations

import os
from typing import Any

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CONFIG_PATH = os.path.join(_PROJECT_ROOT, "config", "cameras.yaml")

DEFAULTS: dict[str, Any] = {
    "min_phrase_frames": 15,
    "min_hold_frames": 8,
    "ml_confidence": 0.55,
    "mirror": False,
    "use_ml": True,
}


def load_camera_config(camera_id: int | str, config_path: str | None = None) -> dict[str, Any]:
    """Merge defaults with optional per-camera overrides from YAML."""
    cfg = dict(DEFAULTS)
    path = config_path or DEFAULT_CONFIG_PATH

    if yaml is None or not os.path.isfile(path):
        return cfg

    with open(path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}

    cfg.update(raw.get("defaults") or {})
    cameras = raw.get("cameras") or {}
    overrides = cameras.get(str(camera_id)) or cameras.get(int(camera_id)) or {}
    cfg.update(overrides)
    return cfg
