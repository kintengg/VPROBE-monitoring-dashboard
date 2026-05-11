"""Vehicle-side state queries.

Reads the same persistent JSON store as the pedestrian layer (store.load_state)
but interprets records through a vehicle lens: gate flow groups, V/C ratios,
class breakdowns, etc.

Vehicle events are expected to live alongside pedestrian events in
state["events"] with type == "vehicle-detection" or with a `vehicleClass`
field populated. Until RT-DETR inference actually runs, the queries return
zero/empty state cleanly.
"""

from __future__ import annotations

import re
from typing import Any, Optional

from . import store
from .vehicle import gates as gate_registry
from .vehicle import los as los_module

VEHICLE_EVENT_TYPES = {"vehicle-detection", "vehicle-track"}


def normalize_vehicle_class_name(value: Any) -> Optional[str]:
    raw = str(value or "").strip().lower()
    if not raw:
        return None
    compact = re.sub(r"[^a-z0-9]+", "-", raw).strip("-")
    if not compact:
        return None
    aliases = {
        "auto": "car",
        "automobile": "car",
        "motorbike": "motorcycle",
        "bike": "bicycle",
    }
    return aliases.get(compact, compact)


def vehicle_class_label(class_name: Any) -> Optional[str]:
    normalized = normalize_vehicle_class_name(class_name)
    if not normalized:
        return None
    return " ".join(part.capitalize() for part in normalized.replace("_", "-").split("-") if part)


def _is_vehicle_event(event: dict[str, Any]) -> bool:
    if event.get("vehicleClass"):
        return True
    raw_type = str(event.get("type") or "").strip().lower()
    return raw_type in VEHICLE_EVENT_TYPES


def list_vehicle_events(date: Optional[str] = None, gate_id: Optional[str] = None) -> list[dict[str, Any]]:
    state = store.load_state()
    events = [event for event in state.get("events", []) if _is_vehicle_event(event)]
    if date:
        events = [event for event in events if str(event.get("date") or "")[:10] == date]
    if gate_id:
        target = gate_registry.get_gate(gate_id)
        if target is None:
            return []
        normalized = target["normalizedName"]
        events = [
            event for event in events
            if gate_registry.normalize_gate_name(event.get("gateName") or event.get("locationName") or "") == normalized
        ]
    return events


def class_breakdown(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    raw_counts: dict[str, int] = {}
    for event in events:
        normalized = normalize_vehicle_class_name(event.get("vehicleClass"))
        if not normalized:
            continue
        raw_counts[normalized] = raw_counts.get(normalized, 0) + 1
    total = sum(raw_counts.values())
    breakdown = []
    for class_name, count in sorted(raw_counts.items(), key=lambda kv: kv[1], reverse=True):
        breakdown.append(
            {
                "className": class_name,
                "label": vehicle_class_label(class_name),
                "count": count,
                "share": (count / total) if total else 0.0,
                "pceMultiplier": los_module.PCE_MULTIPLIERS.get(class_name, 1.0),
            }
        )
    return breakdown


def per_gate_los(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate detections per gate, compute V/C and LOS using gate defaults."""
    gates = gate_registry.list_gates()
    rows: list[dict[str, Any]] = []
    for gate in gates:
        gate_events = [
            event for event in events
            if gate_registry.normalize_gate_name(event.get("gateName") or event.get("locationName") or "")
               == gate["normalizedName"]
        ]
        counts = los_module.aggregate_class_counts(gate_events, class_field="vehicleClass")
        capacity = los_module.compute_capacity(
            gate.get("defaultRoadLengthKm", 0.0),
            int(gate.get("defaultLaneCount") or 0),
        )
        volume = los_module.compute_volume(counts)
        vc_ratio = los_module.compute_vc_ratio(counts, capacity) if capacity > 0 else None
        grade = los_module.get_los(vc_ratio)
        rows.append(
            {
                "gateId": gate["id"],
                "gateName": gate["name"],
                "flowGroup": gate["flowGroup"],
                "latitude": gate["latitude"],
                "longitude": gate["longitude"],
                "vehicleCount": sum(counts.values()),
                "volume": volume,
                "capacity": capacity,
                "vcRatio": vc_ratio,
                "los": grade,
                "losRank": los_module.los_rank(grade),
                "losDescription": los_module.los_description(grade),
            }
        )
    return rows


def vehicle_summary(date: Optional[str] = None) -> dict[str, Any]:
    events = list_vehicle_events(date)
    rows = per_gate_los(events)
    total_vehicles = sum(int(row.get("vehicleCount") or 0) for row in rows)
    avg_vc = None
    valid_vc = [row["vcRatio"] for row in rows if isinstance(row["vcRatio"], (int, float))]
    if valid_vc:
        avg_vc = sum(valid_vc) / len(valid_vc)
    dominant_los: Optional[str] = None
    if rows:
        graded = [row["los"] for row in rows if row["los"]]
        if graded:
            dominant_los = max(graded, key=lambda g: los_module.los_rank(g))
    return {
        "date": date,
        "totalVehicles": total_vehicles,
        "totalGates": len(rows),
        "averageVcRatio": avg_vc,
        "dominantLos": dominant_los,
        "perGateLos": rows,
    }


def ensure_gate_locations_seeded() -> int:
    """Idempotently insert each registry gate as a vehicle-domain Location row.

    Also backfills `flowGroup` on any pre-existing gate rows that are missing it.
    Returns the count of gates added or migrated.
    """
    state = store.load_state()
    existing_locations = state.get("locations", [])
    existing_by_id = {loc["id"]: loc for loc in existing_locations}
    existing_names = {str(loc.get("name", "")).strip().casefold(): loc for loc in existing_locations}
    changed = 0
    for gate in gate_registry.list_gates():
        gate_id = gate["id"]
        existing = existing_by_id.get(gate_id) or existing_names.get(str(gate["name"]).strip().casefold())
        if existing is not None:
            if existing.get("flowGroup") != gate["flowGroup"]:
                existing["flowGroup"] = gate["flowGroup"]
                changed += 1
            if existing.get("domain") != "vehicle":
                existing["domain"] = "vehicle"
                changed += 1
            continue
        existing_locations.append({
            "id": gate_id,
            "name": gate["name"],
            "latitude": gate["latitude"],
            "longitude": gate["longitude"],
            "description": f"{'Entrance' if gate['flowGroup'] == 'In' else 'Exit'} gate (seeded from registry).",
            "address": "",
            "roiCoordinates": None,
            "entryExitPoints": None,
            "walkableAreaM2": None,
            "domain": "vehicle",
            "roadLengthM": (gate.get("defaultRoadLengthKm") or 0.0) * 1000.0 or None,
            "laneCount": gate.get("defaultLaneCount"),
            "flowGroup": gate["flowGroup"],
        })
        changed += 1
    if changed:
        state["locations"] = existing_locations
        store.save_state(state)
    return changed


def vehicle_traffic_series(date: Optional[str] = None, bucket_minutes: int = 60) -> list[dict[str, Any]]:
    """Per-bucket In/Out counts. Empty list when no vehicle events exist yet."""
    events = list_vehicle_events(date)
    if not events:
        return []
    bucket_minutes = max(1, int(bucket_minutes))
    buckets: dict[str, dict[str, Any]] = {}
    for event in events:
        offset_seconds = event.get("offsetSeconds") or 0
        clock = event.get("clockTime") or event.get("startTime") or "00:00"
        try:
            hours, minutes = clock.split(":")[:2]
            total_minutes = int(hours) * 60 + int(minutes) + (int(offset_seconds) // 60)
        except (TypeError, ValueError):
            continue
        bucket_index = (total_minutes // bucket_minutes) * bucket_minutes
        bucket_key = f"{bucket_index // 60:02d}:{bucket_index % 60:02d}"
        bucket = buckets.setdefault(bucket_key, {"id": bucket_key, "time": bucket_key, "In": 0, "Out": 0})
        gate_name = event.get("gateName") or event.get("locationName") or ""
        flow_group = gate_registry.flow_group_from_name(gate_name)
        if flow_group:
            bucket[flow_group] = int(bucket.get(flow_group, 0)) + 1
    return [buckets[k] for k in sorted(buckets.keys())]
