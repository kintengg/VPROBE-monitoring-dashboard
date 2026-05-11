"""Vehicle gate registry — geo coordinates, detection-line pixel coords, flow group.

Geo coordinates come from surveillance-system/components/maps/campus-osm-map.tsx.
Pixel coordinates come from
backend/Occlusion-Robust-RTDETR/inference_requirements/counting/counting_config_g*.json
and are loaded lazily so they stay in sync with whatever configs the user uploads.

Flow grouping mirrors the In/Out classification used in the original
surveillance-system store: gate2 + gate3 are entrances ("In"), gate2.9 +
gate3.2 + gate3.5 are exits ("Out").
"""

from __future__ import annotations

import json
import re
from copy import deepcopy
from pathlib import Path
from typing import Any, Optional

from .. import store

COUNTING_CONFIG_DIR = (
    store.BACKEND_DIR / "Occlusion-Robust-RTDETR" / "inference_requirements" / "counting"
)

FLOW_GROUP_IN: str = "In"
FLOW_GROUP_OUT: str = "Out"


GATE_SEEDS: list[dict[str, Any]] = [
    {
        "id": "gate-2",
        "name": "Gate 2",
        "normalizedName": "gate2",
        "latitude": 14.635825,
        "longitude": 121.074719,
        "flowGroup": FLOW_GROUP_IN,
        "countingConfig": "counting_config_g2.json",
        "defaultRoadLengthKm": 0.06,
        "defaultLaneCount": 2,
    },
    {
        "id": "gate-2-9",
        "name": "Gate 2.9",
        "normalizedName": "gate2.9",
        "latitude": 14.640421,
        "longitude": 121.074759,
        "flowGroup": FLOW_GROUP_OUT,
        "countingConfig": "counting_config_g2.9.json",
        "defaultRoadLengthKm": 0.05,
        "defaultLaneCount": 2,
    },
    {
        "id": "gate-3",
        "name": "Gate 3",
        "normalizedName": "gate3",
        "latitude": 14.640681,
        "longitude": 121.075508,
        "flowGroup": FLOW_GROUP_IN,
        "countingConfig": "counting_config_g3.json",
        "defaultRoadLengthKm": 0.07,
        "defaultLaneCount": 2,
    },
    {
        "id": "gate-3-2",
        "name": "Gate 3.2",
        "normalizedName": "gate3.2",
        "latitude": 14.640904,
        "longitude": 121.074872,
        "flowGroup": FLOW_GROUP_OUT,
        "countingConfig": "counting_config_g3.2.json",
        "defaultRoadLengthKm": 0.04,
        "defaultLaneCount": 1,
    },
    {
        "id": "gate-3-5",
        "name": "Gate 3.5",
        "normalizedName": "gate3.5",
        "latitude": 14.64119,
        "longitude": 121.07477,
        "flowGroup": FLOW_GROUP_OUT,
        "countingConfig": "counting_config_g3.5.json",
        "defaultRoadLengthKm": 0.06,
        "defaultLaneCount": 2,
    },
]


def normalize_gate_name(value: str) -> str:
    """Mirror Repo B's _normalize_gate_name — strip non [a-z0-9.] from a casefolded string."""
    return re.sub(r"[^a-z0-9.]", "", str(value or "").strip().lower())


def flow_group_from_name(name: str) -> Optional[str]:
    normalized = normalize_gate_name(name)
    for seed in GATE_SEEDS:
        if seed["normalizedName"] == normalized:
            return str(seed["flowGroup"])
    return None


def _load_counting_config(filename: str) -> Optional[dict[str, Any]]:
    config_path = COUNTING_CONFIG_DIR / filename
    try:
        return json.loads(config_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return None


def _detection_line_from_config(config: Optional[dict[str, Any]]) -> Optional[dict[str, Any]]:
    if not config:
        return None
    lines = config.get("lines") or []
    if not lines:
        return None
    primary = lines[0]
    start = primary.get("start")
    end = primary.get("end")
    anchors = primary.get("trigger_anchors") or ["CENTER"]
    if not (isinstance(start, (list, tuple)) and isinstance(end, (list, tuple))):
        return None
    return {
        "name": str(primary.get("name") or ""),
        "start": [int(start[0]), int(start[1])],
        "end": [int(end[0]), int(end[1])],
        "triggerAnchors": [str(a).upper() for a in anchors],
    }


_GATE_CACHE: Optional[tuple[float, list[dict[str, Any]]]] = None


def _build_gates() -> list[dict[str, Any]]:
    enriched: list[dict[str, Any]] = []
    for seed in GATE_SEEDS:
        config = _load_counting_config(seed["countingConfig"])
        gate = deepcopy(seed)
        gate["countingConfigPath"] = str(COUNTING_CONFIG_DIR / seed["countingConfig"])
        gate["countingConfigExists"] = config is not None
        gate["detectionLine"] = _detection_line_from_config(config)
        enriched.append(gate)
    return enriched


def list_gates() -> list[dict[str, Any]]:
    """Return all gates with their current detection-line config. Cached by config-dir mtime."""
    global _GATE_CACHE
    try:
        mtime = COUNTING_CONFIG_DIR.stat().st_mtime
    except OSError:
        mtime = 0.0
    if _GATE_CACHE is not None and _GATE_CACHE[0] == mtime:
        return deepcopy(_GATE_CACHE[1])
    gates = _build_gates()
    _GATE_CACHE = (mtime, gates)
    return deepcopy(gates)


def get_gate(gate_id: str) -> Optional[dict[str, Any]]:
    for gate in list_gates():
        if gate["id"] == gate_id:
            return gate
    return None


def list_counting_config_filenames() -> list[str]:
    if not COUNTING_CONFIG_DIR.exists():
        return []
    return sorted(p.name for p in COUNTING_CONFIG_DIR.glob("*.json"))
