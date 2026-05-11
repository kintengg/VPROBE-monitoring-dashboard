"""Counting / detection-line utilities.

Wraps the `inference_requirements/counting/` JSON config format used by the
Occlusion-Robust-RTDETR scripts. The actual line-zone tracking logic
(supervision.LineZone) lives in the inference adapter — this module only
parses configs and exposes them to the API.
"""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Optional

from .. import store

CONFIG_ROOT = store.BACKEND_DIR / "Occlusion-Robust-RTDETR" / "inference_requirements"
COUNTING_DIR = CONFIG_ROOT / "counting"
INFER_CONFIG_DIR = (
    store.BACKEND_DIR / "Occlusion-Robust-RTDETR" / "configs" / "rtdetr"
)
ANNOTATIONS_DIR = CONFIG_ROOT / "annotations"

VALID_REQUIREMENT_TYPES = ("infer-config", "annotations", "counting-config")


def list_counting_configs() -> list[str]:
    if not COUNTING_DIR.exists():
        return []
    return sorted(p.name for p in COUNTING_DIR.glob("*.json"))


def list_infer_configs() -> list[str]:
    if not INFER_CONFIG_DIR.exists():
        return []
    return sorted(p.name for p in INFER_CONFIG_DIR.glob("*.yml") if p.is_file()) + sorted(
        p.name for p in INFER_CONFIG_DIR.glob("*.yaml") if p.is_file()
    )


def read_counting_config(name: str) -> Optional[dict[str, Any]]:
    target = COUNTING_DIR / name
    if not target.exists():
        return None
    try:
        return json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def list_all_lines() -> list[dict[str, Any]]:
    """Flatten every line in every counting config into a single list."""
    out: list[dict[str, Any]] = []
    for filename in list_counting_configs():
        config = read_counting_config(filename) or {}
        for line in config.get("lines", []):
            entry = deepcopy(line)
            entry["sourceConfig"] = filename
            out.append(entry)
    return out


def requirement_target_dir(requirement_type: str) -> Path:
    if requirement_type == "infer-config":
        INFER_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        return INFER_CONFIG_DIR
    if requirement_type == "annotations":
        ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)
        return ANNOTATIONS_DIR
    if requirement_type == "counting-config":
        COUNTING_DIR.mkdir(parents=True, exist_ok=True)
        return COUNTING_DIR
    raise ValueError(
        f"Unknown requirementType '{requirement_type}'. Expected one of: {VALID_REQUIREMENT_TYPES}"
    )
