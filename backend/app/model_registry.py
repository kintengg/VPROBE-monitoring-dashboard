from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from threading import Lock
from typing import Any, Optional

from . import store

REGISTRY_FILE = store.MODELS_DIR / "registry.json"
PEDESTRIAN_DIR = store.MODELS_DIR / "pedestrian"
VEHICLE_DIR = store.MODELS_DIR / "vehicle"

DOMAINS = ("pedestrian", "vehicle")

DOMAIN_DEFAULTS: dict[str, dict[str, Any]] = {
    "pedestrian": {
        "framework": "ultralytics-yolov8",
        "ultralyticsTag": "v8.3.228",
        "fallbackUltralyticsTag": "v8.3.50",
        "detectionClasses": ["person"],
    },
    "vehicle": {
        "framework": "rtdetr",
        "ultralyticsTag": "rtdetr-cli",
        "fallbackUltralyticsTag": "rtdetr-cli-cpu",
        "detectionClasses": ["car", "truck", "bus", "motorcycle"],
    },
}

SEED_WEIGHTS: dict[str, list[dict[str, Any]]] = {
    "pedestrian": [
        {
            "id": "hybrid_2000_fold3",
            "filename": "hybrid_2000_fold3.pt",
            "relativePath": "hybrid_2000_fold3.pt",
            "label": "Hybrid 2000 Fold-3 (default)",
        },
    ],
    "vehicle": [
        {
            "id": "fastervit-0",
            "filename": "fastervit-0.pth",
            "relativePath": "../../Occlusion-Robust-RTDETR/weights/fastervit/fastervit-0.pth",
            "label": "FasterViT-0 RT-DETR (default)",
        },
    ],
}

_LOCK = Lock()


def _resolve_path(relative_path: str) -> Path:
    return (store.MODELS_DIR / relative_path).resolve()


def _weight_record(domain: str, entry: dict[str, Any]) -> dict[str, Any]:
    relative_path = entry["relativePath"]
    resolved = _resolve_path(relative_path)
    size_bytes: Optional[int] = None
    exists = False
    try:
        size_bytes = resolved.stat().st_size
        exists = True
    except OSError:
        pass
    return {
        "id": entry["id"],
        "domain": domain,
        "filename": entry["filename"],
        "relativePath": relative_path,
        "absolutePath": str(resolved),
        "exists": exists,
        "sizeBytes": size_bytes,
        "label": entry.get("label"),
        "uploadedAt": entry.get("uploadedAt"),
        "isSeed": bool(entry.get("isSeed")),
    }


def _default_registry() -> dict[str, Any]:
    domains: dict[str, Any] = {}
    for domain in DOMAINS:
        seeds = [{**seed, "isSeed": True, "uploadedAt": None} for seed in SEED_WEIGHTS.get(domain, [])]
        domains[domain] = {
            **DOMAIN_DEFAULTS[domain],
            "domain": domain,
            "active": seeds[0]["id"] if seeds else None,
            "weights": seeds,
        }
    return {"version": 1, "domains": domains, "updatedAt": store._utc_timestamp()}


def _ensure_dirs() -> None:
    store.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    PEDESTRIAN_DIR.mkdir(parents=True, exist_ok=True)
    VEHICLE_DIR.mkdir(parents=True, exist_ok=True)


def _read_registry() -> dict[str, Any]:
    try:
        return json.loads(REGISTRY_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return _default_registry()


def _write_registry(data: dict[str, Any]) -> None:
    _ensure_dirs()
    data["updatedAt"] = store._utc_timestamp()
    REGISTRY_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _merged_seeds(registry: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    """Insert any missing seed weights / domain defaults. Returns (registry, changed)."""
    changed = False
    domains_block = registry.setdefault("domains", {})
    for domain, defaults in DOMAIN_DEFAULTS.items():
        if domain not in domains_block:
            domains_block[domain] = {**defaults, "domain": domain, "active": None, "weights": []}
            changed = True
        domain_block = domains_block[domain]
        for key, value in defaults.items():
            if key not in domain_block:
                domain_block[key] = value
                changed = True
        existing_ids = {w["id"] for w in domain_block.get("weights", [])}
        for seed in SEED_WEIGHTS.get(domain, []):
            if seed["id"] not in existing_ids:
                domain_block.setdefault("weights", []).append({**seed, "isSeed": True, "uploadedAt": None})
                changed = True
        if not domain_block.get("active") and domain_block.get("weights"):
            domain_block["active"] = domain_block["weights"][0]["id"]
            changed = True
    return registry, changed


def _sync_legacy_pedestrian_state(registry: dict[str, Any]) -> None:
    """Write the pedestrian-domain active weight back into state["model"] so the
    legacy inference engine (inference.ultralytics_status / resolve_model_path)
    sees the same value the Models page set."""
    block = registry.get("domains", {}).get("pedestrian", {})
    active_id = block.get("active")
    if not active_id:
        return
    weight = next((w for w in block.get("weights", []) if w["id"] == active_id), None)
    if not weight:
        return
    legacy = store.get_model_info()
    if legacy.get("currentModel") == weight["filename"]:
        return
    store.set_model(weight["filename"])


def init_registry() -> dict[str, Any]:
    """Ensure registry.json exists and contains all expected seeds. Writes only when changed."""
    with _LOCK:
        _ensure_dirs()
        registry, changed = _merged_seeds(_read_registry())
        if changed or not REGISTRY_FILE.exists():
            _write_registry(registry)
    _sync_legacy_pedestrian_state(registry)
    return deepcopy(registry)


def get_registry() -> dict[str, Any]:
    with _LOCK:
        registry, _changed = _merged_seeds(_read_registry())
        return deepcopy(registry)


def get_domain(domain: str) -> dict[str, Any]:
    if domain not in DOMAINS:
        raise ValueError(f"Unknown domain '{domain}'. Expected one of: {DOMAINS}")
    registry = get_registry()
    block = registry["domains"][domain]
    weights = [_weight_record(domain, w) for w in block.get("weights", [])]
    active_id = block.get("active")
    active = next((w for w in weights if w["id"] == active_id), None)
    return {
        "domain": domain,
        "framework": block.get("framework"),
        "ultralyticsTag": block.get("ultralyticsTag"),
        "fallbackUltralyticsTag": block.get("fallbackUltralyticsTag"),
        "detectionClasses": block.get("detectionClasses", []),
        "active": active,
        "activeWeightId": active_id,
        "weights": weights,
    }


def list_domains() -> list[dict[str, Any]]:
    return [get_domain(domain) for domain in DOMAINS]


def set_active(domain: str, weight_id: str) -> dict[str, Any]:
    if domain not in DOMAINS:
        raise ValueError(f"Unknown domain '{domain}'")
    with _LOCK:
        registry, _changed = _merged_seeds(_read_registry())
        block = registry["domains"][domain]
        ids = {w["id"] for w in block.get("weights", [])}
        if weight_id not in ids:
            raise KeyError(f"Weight '{weight_id}' not registered under domain '{domain}'")
        block["active"] = weight_id
        _write_registry(registry)
    if domain == "pedestrian":
        _sync_legacy_pedestrian_state(registry)
    return get_domain(domain)


def add_weight(
    domain: str,
    *,
    weight_id: str,
    filename: str,
    relative_path: str,
    label: Optional[str] = None,
    set_active_after: bool = False,
) -> dict[str, Any]:
    if domain not in DOMAINS:
        raise ValueError(f"Unknown domain '{domain}'")
    with _LOCK:
        registry, _changed = _merged_seeds(_read_registry())
        block = registry["domains"][domain]
        weights = block.setdefault("weights", [])
        if any(w["id"] == weight_id for w in weights):
            raise KeyError(f"Weight id '{weight_id}' already exists for domain '{domain}'")
        weights.append(
            {
                "id": weight_id,
                "filename": filename,
                "relativePath": relative_path,
                "label": label,
                "uploadedAt": store._utc_timestamp(),
                "isSeed": False,
            }
        )
        if set_active_after or not block.get("active"):
            block["active"] = weight_id
        _write_registry(registry)
    if domain == "pedestrian":
        _sync_legacy_pedestrian_state(registry)
    return get_domain(domain)


def remove_weight(domain: str, weight_id: str) -> dict[str, Any]:
    if domain not in DOMAINS:
        raise ValueError(f"Unknown domain '{domain}'")
    with _LOCK:
        registry, _changed = _merged_seeds(_read_registry())
        block = registry["domains"][domain]
        weights = block.get("weights", [])
        target = next((w for w in weights if w["id"] == weight_id), None)
        if target is None:
            raise KeyError(f"Weight '{weight_id}' not found in domain '{domain}'")
        if target.get("isSeed"):
            raise PermissionError(f"Cannot remove seed weight '{weight_id}' from domain '{domain}'")
        block["weights"] = [w for w in weights if w["id"] != weight_id]
        if block.get("active") == weight_id:
            block["active"] = block["weights"][0]["id"] if block["weights"] else None
        _write_registry(registry)
    if domain == "pedestrian":
        _sync_legacy_pedestrian_state(registry)
    resolved = _resolve_path(target["relativePath"])
    if resolved.exists() and resolved.is_file():
        try:
            resolved.unlink()
        except OSError:
            pass
    return get_domain(domain)


def active_weight_path(domain: str) -> Optional[Path]:
    info = get_domain(domain)
    active = info.get("active")
    if not active:
        return None
    path = Path(active["absolutePath"])
    return path if path.exists() else None
