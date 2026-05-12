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
        # Events don't carry their own date — join through the parent video.
        matching_video_ids = {
            video["id"]
            for video in state.get("videos", [])
            if str(video.get("date") or "")[:10] == date
        }
        events = [event for event in events if event.get("videoId") in matching_video_ids]
    if gate_id:
        target = gate_registry.get_gate(gate_id)
        if target is None:
            return []
        normalized = target["normalizedName"]
        events = [
            event for event in events
            if gate_registry.normalize_gate_name(
                event.get("gateName") or event.get("locationName") or event.get("location") or ""
            ) == normalized
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
    locations = store.load_state().get("locations", [])
    rows: list[dict[str, Any]] = []
    for gate in gates:
        location_override = next(
            (
                loc
                for loc in locations
                if loc.get("id") == gate["id"]
                or gate_registry.normalize_gate_name(loc.get("name") or "") == gate["normalizedName"]
            ),
            None,
        )
        road_length_km = gate.get("defaultRoadLengthKm", 0.0)
        lane_count = gate.get("defaultLaneCount")
        if location_override is not None:
            road_length_m = location_override.get("roadLengthM")
            if isinstance(road_length_m, (int, float)) and road_length_m > 0:
                road_length_km = float(road_length_m) / 1000.0
            lane_override = location_override.get("laneCount")
            if isinstance(lane_override, (int, float)) and lane_override > 0:
                lane_count = int(round(lane_override))
        gate_events = [
            event for event in events
            if gate_registry.normalize_gate_name(event.get("gateName") or event.get("locationName") or "")
               == gate["normalizedName"]
        ]
        counts = los_module.aggregate_class_counts(gate_events, class_field="vehicleClass")
        capacity = los_module.compute_capacity(
            road_length_km,
            int(lane_count or 0),
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


def vehicle_traffic_series(
    date: Optional[str] = None,
    time_range: str = "whole-day",
    start_time: Optional[str] = None,
    bucket_minutes: int = 60,
) -> list[dict[str, Any]]:
    """Per-bucket In/Out counts filtered by date and time window."""
    events = list_vehicle_events(date)
    if not events:
        return []
    state = store.load_state()
    video_start_by_id = {
        str(video.get("id")): str(video.get("startTime") or video.get("timestamp") or "")
        for video in state.get("videos", [])
    }
    bucket_minutes = max(1, int(bucket_minutes))
    window_start_min, window_end_min = _time_range_bounds(time_range, start_time)
    buckets: dict[str, dict[str, Any]] = {}
    for event in events:
        offset_seconds = float(event.get("offsetSeconds") or 0)
        clock = (
            event.get("clockTime")
            or event.get("startTime")
            or video_start_by_id.get(str(event.get("videoId")))
            or event.get("timestamp")
            or "00:00"
        )
        try:
            hours, minutes = str(clock).split(":")[:2]
            total_minutes = int(hours) * 60 + int(minutes) + int(offset_seconds) // 60
        except (TypeError, ValueError):
            continue
        # Apply time-window filter
        if window_start_min is not None and window_end_min is not None:
            if total_minutes < window_start_min or total_minutes >= window_end_min:
                continue
        bucket_index = (total_minutes // bucket_minutes) * bucket_minutes
        bucket_key = f"{bucket_index // 60:02d}:{bucket_index % 60:02d}"
        bucket = buckets.setdefault(bucket_key, {"id": bucket_key, "time": bucket_key, "In": 0, "Out": 0})
        gate_name = event.get("gateName") or event.get("locationName") or event.get("location") or ""
        flow_group = gate_registry.flow_group_from_name(gate_name)
        if flow_group:
            bucket[flow_group] = int(bucket.get(flow_group, 0)) + 1
    return [buckets[k] for k in sorted(buckets.keys())]


def _parse_time_to_minutes(value: str) -> "Optional[int]":
    """Parse 'HH:MM' or 'H:MM AM/PM' to total minutes since midnight."""
    try:
        cleaned = str(value).strip()
        parts = cleaned.split()
        time_part = parts[0]
        ampm = parts[1].upper() if len(parts) > 1 else None
        h, m = time_part.split(":")[:2]
        hours = int(h)
        minutes = int(m)
        if ampm == "PM" and hours != 12:
            hours += 12
        elif ampm == "AM" and hours == 12:
            hours = 0
        return hours * 60 + minutes
    except (IndexError, ValueError, TypeError):
        return None


def _time_range_bounds(time_range: str, start_time: "Optional[str]") -> "tuple[Optional[int], Optional[int]]":
    """Return (start_minutes, end_minutes) window, or (None, None) for whole-day."""
    if time_range == "whole-day":
        return None, None
    hours_map = {"12h": 12 * 60, "6h": 6 * 60, "4h": 4 * 60, "3h": 3 * 60, "2h": 2 * 60, "1h": 60, "30m": 30}
    window_minutes = hours_map.get(time_range)
    if window_minutes is None:
        return None, None
    start_minutes = _parse_time_to_minutes(start_time or "06:00") or (6 * 60)
    return start_minutes, start_minutes + window_minutes


def vehicle_analytics_series(
    date: "Optional[str]" = None,
    time_range: str = "whole-day",
    start_time: "Optional[str]" = None,
    gate_id: "Optional[str]" = None,
    bucket_minutes: int = 60,
) -> "list[dict[str, Any]]":
    """Per-bucket vehicle count series broken down by gate name.

    Each bucket point has gate names as keys so the frontend can render
    one series per gate. Supports date, time-range, start-time, and
    optional single-gate filtering.
    """
    events = list_vehicle_events(date, gate_id)
    if not events:
        return []

    state = store.load_state()
    video_start_by_id = {
        str(video.get("id")): str(video.get("startTime") or video.get("timestamp") or "")
        for video in state.get("videos", [])
    }

    bucket_minutes = max(1, int(bucket_minutes))
    window_start_min, window_end_min = _time_range_bounds(time_range, start_time)

    # Only count gate-crossing events (vehicle-detection with a gate name)
    gate_events = [
        e for e in events
        if str(e.get("type") or "") == "vehicle-detection" and (e.get("gateName") or "").strip()
    ]

    buckets: "dict[str, dict[str, Any]]" = {}
    all_gates: "set[str]" = set()

    for event in gate_events:
        offset_seconds = float(event.get("offsetSeconds") or 0)
        video_clock = video_start_by_id.get(str(event.get("videoId"))) or event.get("timestamp") or "00:00"
        start_min = _parse_time_to_minutes(str(video_clock))
        if start_min is None:
            continue
        total_minutes = start_min + int(offset_seconds) // 60

        # Apply time-range window filter
        if window_start_min is not None and window_end_min is not None:
            if total_minutes < window_start_min or total_minutes >= window_end_min:
                continue

        bucket_index = (total_minutes // bucket_minutes) * bucket_minutes
        bucket_key = f"{bucket_index // 60:02d}:{bucket_index % 60:02d}"
        gate_name = (event.get("gateName") or "").strip()
        all_gates.add(gate_name)

        if bucket_key not in buckets:
            buckets[bucket_key] = {"id": bucket_key, "time": bucket_key}
        buckets[bucket_key][gate_name] = int(buckets[bucket_key].get(gate_name, 0)) + 1

    if not buckets:
        return []

    # Fill missing gate keys with 0 so chart lines connect cleanly
    sorted_keys = sorted(buckets.keys())
    result = []
    for key in sorted_keys:
        point = dict(buckets[key])
        for gate in all_gates:
            point.setdefault(gate, 0)
        result.append(point)
    return result


def vehicle_los_series(
    date: "Optional[str]" = None,
    time_range: str = "whole-day",
    start_time: "Optional[str]" = None,
    gate_id: "Optional[str]" = None,
    bucket_minutes: int = 60,
) -> "list[dict[str, Any]]":
    """Per-bucket LOS series broken down by gate name.

    Each bucket point has keys `gateName__los` (letter grade) and
    `gateName__losRank` (1-6 int) so the VehicleAnalyticsChart can render
    per-gate LOS lines using the same logic as the pedestrian chart.
    """
    events = list_vehicle_events(date, gate_id)
    if not events:
        return []

    state = store.load_state()
    video_start_by_id = {
        str(video.get("id")): str(video.get("startTime") or video.get("timestamp") or "")
        for video in state.get("videos", [])
    }

    gates = gate_registry.list_gates()
    gate_name_by_normalized: dict[str, str] = {g["normalizedName"]: g["name"] for g in gates}

    # Build location overrides for road_length / lane_count
    locations = state.get("locations", [])
    location_by_gate_id: dict[str, dict[str, Any]] = {}
    for loc in locations:
        for gate in gates:
            if loc.get("id") == gate["id"] or gate_registry.normalize_gate_name(
                loc.get("name") or ""
            ) == gate["normalizedName"]:
                location_by_gate_id[gate["id"]] = loc

    gate_meta: dict[str, dict[str, Any]] = {}
    for gate in gates:
        loc_override = location_by_gate_id.get(gate["id"])
        road_length_km = gate.get("defaultRoadLengthKm", 0.0)
        lane_count = gate.get("defaultLaneCount") or 0
        if loc_override:
            rlm = loc_override.get("roadLengthM")
            if isinstance(rlm, (int, float)) and rlm > 0:
                road_length_km = float(rlm) / 1000.0
            lc = loc_override.get("laneCount")
            if isinstance(lc, (int, float)) and lc > 0:
                lane_count = int(round(lc))
        gate_meta[gate["name"]] = {
            "road_length_km": road_length_km,
            "lane_count": lane_count,
        }

    bucket_minutes = max(1, int(bucket_minutes))
    window_start_min, window_end_min = _time_range_bounds(time_range, start_time)

    # Only gate-crossing events
    gate_events = [
        e for e in events
        if str(e.get("type") or "") == "vehicle-detection" and (e.get("gateName") or "").strip()
    ]

    # Bucket events per gate name
    # Structure: { bucket_key: { gate_name: { class_name: count } } }
    buckets: dict[str, dict[str, dict[str, int]]] = {}
    all_gate_names: set[str] = set()

    for event in gate_events:
        offset_seconds = float(event.get("offsetSeconds") or 0)
        video_clock = video_start_by_id.get(str(event.get("videoId"))) or event.get("timestamp") or "00:00"
        start_min = _parse_time_to_minutes(str(video_clock))
        if start_min is None:
            continue
        total_minutes = start_min + int(offset_seconds) // 60

        if window_start_min is not None and window_end_min is not None:
            if total_minutes < window_start_min or total_minutes >= window_end_min:
                continue

        bucket_index = (total_minutes // bucket_minutes) * bucket_minutes
        bucket_key = f"{bucket_index // 60:02d}:{bucket_index % 60:02d}"
        gate_name = (event.get("gateName") or "").strip()
        vehicle_class = normalize_vehicle_class_name(event.get("vehicleClass")) or "car"
        all_gate_names.add(gate_name)

        bucket = buckets.setdefault(bucket_key, {})
        gate_bucket = bucket.setdefault(gate_name, {})
        gate_bucket[vehicle_class] = gate_bucket.get(vehicle_class, 0) + 1

    if not buckets:
        return []

    sorted_keys = sorted(buckets.keys())
    result = []
    for key in sorted_keys:
        point: dict[str, Any] = {"id": key, "time": key}
        gate_data = buckets[key]
        for gate_name in all_gate_names:
            counts = gate_data.get(gate_name, {})
            meta = gate_meta.get(gate_name, {"road_length_km": 0.0, "lane_count": 0})
            capacity = los_module.compute_capacity(
                meta["road_length_km"], meta["lane_count"]
            )
            vc_ratio = los_module.compute_vc_ratio(counts, capacity) if capacity > 0 else None
            grade = los_module.get_los(vc_ratio)
            rank = los_module.los_rank(grade)
            point[f"{gate_name}__los"] = grade
            point[f"{gate_name}__losRank"] = rank if rank is not None else 0
        result.append(point)
    return result
