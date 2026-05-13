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

import csv
import re
from pathlib import Path
from typing import Any, Iterable, Optional

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


def list_gate_crossing_events(date: Optional[str] = None, gate_id: Optional[str] = None) -> list[dict[str, Any]]:
    """Return only gate-crossing events (vehicle-detection type with a gate name).

    Used for class breakdown and LOS calculations so that tracking-only
    'first seen' events (vehicle-track) are not double-counted.
    """
    events = list_vehicle_events(date, gate_id)
    return [
        e for e in events
        if str(e.get("type") or "").strip().lower() == "vehicle-detection"
        and (e.get("gateName") or "").strip()
    ]


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


def per_gate_los(
    events: list[dict[str, Any]],
    *,
    congestion_samples: Optional[dict[str, list[tuple[int, float, Optional[str]]]]] = None,
) -> list[dict[str, Any]]:
    """Aggregate per-gate LOS using per-frame V/C from RT-DETR's congestion
    sidecar so the dashboard agrees with the in-video LOS overlay.

    `events` is still used to count crossings (vehicleCount). `congestion_samples`
    is the output of :func:`collect_congestion_samples` for the same time
    window. If samples aren't supplied (e.g., legacy callers), V/C and LOS
    fall back to None for that gate.
    """
    gates = gate_registry.list_gates()
    samples = congestion_samples or {}
    rows: list[dict[str, Any]] = []
    for gate in gates:
        gate_events = [
            event for event in events
            if gate_registry.normalize_gate_name(event.get("gateName") or event.get("locationName") or "")
               == gate["normalizedName"]
        ]
        counts = los_module.aggregate_class_counts(gate_events, class_field="vehicleClass")
        volume = los_module.compute_volume(counts)

        gate_samples = samples.get(gate["name"], [])
        mean_vc = _mean(vc for _, vc, _ in gate_samples) if gate_samples else None
        grade = los_module.get_los(mean_vc) if mean_vc is not None else None

        rows.append(
            {
                "gateId": gate["id"],
                "gateName": gate["name"],
                "flowGroup": gate["flowGroup"],
                "latitude": gate["latitude"],
                "longitude": gate["longitude"],
                "vehicleCount": sum(counts.values()),
                "volume": volume,
                # Capacity is implicit in per-frame V/C; surface a friendly
                # sample count instead so callers can spot empty windows.
                "capacity": float(len(gate_samples)),
                "vcRatio": mean_vc,
                "los": grade,
                "losRank": los_module.los_rank(grade),
                "losDescription": los_module.los_description(grade),
            }
        )
    return rows


def vehicle_summary(
    date: Optional[str] = None,
    time_range: str = "whole-day",
    start_time: Optional[str] = None,
) -> dict[str, Any]:
    """Aggregate vehicle summary over the selected day + sub-day window.

    The window is described by `time_range` (e.g., "whole-day", "30m", "1h",
    "6h", ...) and an optional `start_time` ("HH:MM"). Capacity used for
    V/C scales with the window length so the resulting LOS reflects the
    selected gate + time range, not just the whole day.
    """
    crossing_events = list_gate_crossing_events(date)
    window_start_min, window_end_min = _time_range_bounds(time_range, start_time)
    if window_start_min is not None and window_end_min is not None:
        window_hours = max(1 / 60.0, (window_end_min - window_start_min) / 60.0)
        crossing_events = _filter_events_to_window(crossing_events, window_start_min, window_end_min)
    else:
        window_hours = 24.0

    congestion_samples = collect_congestion_samples(date, window_start_min, window_end_min)
    rows = per_gate_los(crossing_events, congestion_samples=congestion_samples)
    total_vehicles = sum(int(row.get("vehicleCount") or 0) for row in rows)
    # "Average V/C" = mean of per-frame V/C across all gates' samples — this
    # weights time equally rather than letting a single low-traffic gate
    # bias the dashboard's headline average.
    all_vc_values = [vc for samples in congestion_samples.values() for _, vc, _ in samples]
    avg_vc = sum(all_vc_values) / len(all_vc_values) if all_vc_values else None
    # Average LOS uses the mean V/C across active (non-zero) gates and maps
    # the result to a single grade — matches the dashboard's "Average LOS" tile.
    average_los = los_module.get_los(avg_vc) if avg_vc is not None else None
    # Dominant LOS is the worst grade observed across gates that saw any flow.
    dominant_los: Optional[str] = None
    graded = [row["los"] for row in rows if row.get("los") and (row.get("vehicleCount") or 0) > 0]
    if graded:
        dominant_los = max(graded, key=lambda g: los_module.los_rank(g))
    return {
        "date": date,
        "totalVehicles": total_vehicles,
        "totalGates": len(rows),
        "averageVcRatio": avg_vc,
        "averageLos": average_los,
        "dominantLos": dominant_los,
        "perGateLos": rows,
        "timeRange": time_range,
        "startTime": start_time,
        "windowHours": window_hours,
    }


def _filter_events_to_window(
    events: list[dict[str, Any]],
    window_start_min: int,
    window_end_min: int,
) -> list[dict[str, Any]]:
    state = store.load_state()
    video_start_by_id = {
        str(video.get("id")): str(video.get("startTime") or video.get("timestamp") or "")
        for video in state.get("videos", [])
    }
    kept: list[dict[str, Any]] = []
    for event in events:
        offset_seconds = float(event.get("offsetSeconds") or 0)
        clock = video_start_by_id.get(str(event.get("videoId"))) or str(event.get("timestamp") or "00:00")
        start_min = _parse_time_to_minutes(clock)
        if start_min is None:
            continue
        total_minutes = start_min + int(offset_seconds) // 60
        if window_start_min <= total_minutes < window_end_min:
            kept.append(event)
    return kept


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
    events = list_gate_crossing_events(date)
    state = store.load_state()
    video_start_by_id = {
        str(video.get("id")): str(video.get("startTime") or video.get("timestamp") or "")
        for video in state.get("videos", [])
    }
    bucket_minutes = max(1, int(bucket_minutes))
    window_start_min, window_end_min = _time_range_bounds(time_range, start_time)
    buckets: dict[str, dict[str, Any]] = {}

    start_idx = 0
    end_idx = 24 * 60
    if window_start_min is not None and window_end_min is not None:
        start_idx = (window_start_min // bucket_minutes) * bucket_minutes
        end_idx = (window_end_min // bucket_minutes) * bucket_minutes

    for idx in range(start_idx, end_idx, bucket_minutes):
        bk = f"{idx // 60:02d}:{idx % 60:02d}"
        buckets[bk] = {"id": bk, "time": bk, "In": 0, "Out": 0}
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


# ---------------------------------------------------------------------------
# Per-frame congestion-CSV aggregation.
#
# RT-DETR writes a `<video_stem>_congestion.csv` next to each annotated video
# with one row per processed frame: timestamp (relative), V/C ratio, and LOS
# letter (computed against jam-density capacity in the realtime estimator).
# Aggregating these rows across the dashboard's selected window gives the
# *same* LOS distribution the in-video overlay shows, so the dashboard and
# the video playback agree.
# ---------------------------------------------------------------------------


def _congestion_csv_for_video(video: dict[str, Any]) -> Optional[Path]:
    processed = video.get("processedPath")
    if not processed:
        return None
    path = Path(processed)
    if not path.is_absolute():
        path = store.BACKEND_DIR / path
    candidate = path.with_suffix("")
    candidate = candidate.parent / f"{candidate.name}_congestion.csv"
    return candidate if candidate.exists() else None


def _absolute_minute_from_csv_timestamp(timestamp: str, video_start_min: int) -> Optional[int]:
    """Convert a `HH:MM:SS` offset from a congestion row to an absolute minute-of-day."""
    parts = timestamp.strip().split(":")
    if len(parts) != 3:
        return None
    try:
        hours = int(parts[0])
        minutes = int(parts[1])
        seconds = int(parts[2])
    except ValueError:
        return None
    offset_minutes = hours * 60 + minutes + (seconds // 60)
    return video_start_min + offset_minutes


def _gate_name_for_video(
    video: dict[str, Any],
    gate_name_by_normalized: dict[str, str],
) -> Optional[str]:
    candidate = video.get("location") or ""
    normalized = gate_registry.normalize_gate_name(str(candidate))
    if normalized in gate_name_by_normalized:
        return gate_name_by_normalized[normalized]
    # Fall back to the gate registry's loc id alias.
    location_id = str(video.get("locationId") or "")
    for gate in gate_registry.list_gates():
        if gate["id"] == location_id:
            return gate["name"]
    return None


def collect_congestion_samples(
    date: Optional[str],
    window_start_min: Optional[int],
    window_end_min: Optional[int],
    gate_id: Optional[str] = None,
) -> dict[str, list[tuple[int, float, Optional[str]]]]:
    """Walk every vehicle video for `date`, parse its congestion CSV, and
    return a dict keyed by gate name → list of (absolute_minute, V/C, LOS letter)."""
    state = store.load_state()
    locations_by_id = {str(loc.get("id")): loc for loc in state.get("locations", [])}
    gate_name_by_normalized = {g["normalizedName"]: g["name"] for g in gate_registry.list_gates()}
    target_gate_name: Optional[str] = None
    if gate_id:
        target = gate_registry.get_gate(gate_id)
        target_gate_name = target["name"] if target else None

    samples: dict[str, list[tuple[int, float, Optional[str]]]] = {}
    for video in state.get("videos", []):
        if date and str(video.get("date") or "")[:10] != date:
            continue
        # Vehicle-only: skip if its location isn't a vehicle gate.
        loc = locations_by_id.get(str(video.get("locationId")))
        if loc is not None and loc.get("domain") and loc["domain"] != "vehicle":
            continue
        gate_name = _gate_name_for_video(video, gate_name_by_normalized)
        if gate_name is None:
            continue
        if target_gate_name is not None and gate_name != target_gate_name:
            continue
        csv_path = _congestion_csv_for_video(video)
        if csv_path is None:
            continue
        video_start_min = _parse_time_to_minutes(str(video.get("startTime") or video.get("timestamp") or ""))
        if video_start_min is None:
            continue
        with csv_path.open("r", newline="") as handle:
            for row in csv.DictReader(handle):
                vc_raw = row.get("vc_ratio")
                if vc_raw in (None, ""):
                    continue
                try:
                    vc = float(vc_raw)
                except ValueError:
                    continue
                abs_min = _absolute_minute_from_csv_timestamp(row.get("timestamp", ""), video_start_min)
                if abs_min is None:
                    continue
                if window_start_min is not None and window_end_min is not None:
                    if abs_min < window_start_min or abs_min >= window_end_min:
                        continue
                los_letter = (row.get("los") or "").strip() or None
                samples.setdefault(gate_name, []).append((abs_min, vc, los_letter))
    return samples


def _mean(values: Iterable[float]) -> Optional[float]:
    items = [float(v) for v in values]
    if not items:
        return None
    return sum(items) / len(items)


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
    events = list_gate_crossing_events(date, gate_id)

    state = store.load_state()
    video_start_by_id = {
        str(video.get("id")): str(video.get("startTime") or video.get("timestamp") or "")
        for video in state.get("videos", [])
    }

    bucket_minutes = max(1, int(bucket_minutes))
    window_start_min, window_end_min = _time_range_bounds(time_range, start_time)

    buckets: "dict[str, dict[str, Any]]" = {}
    
    gates = gate_registry.list_gates()
    all_gates: "set[str]" = {g["name"] for g in gates}
    gate_name_by_normalized: dict[str, str] = {g["normalizedName"]: g["name"] for g in gates}
    if gate_id:
        target_gate = next((g["name"] for g in gates if str(g["id"]) == str(gate_id)), None)
        all_gates = {target_gate} if target_gate else set()

    start_idx = 0
    end_idx = 24 * 60
    if window_start_min is not None and window_end_min is not None:
        start_idx = (window_start_min // bucket_minutes) * bucket_minutes
        end_idx = (window_end_min // bucket_minutes) * bucket_minutes

    for idx in range(start_idx, end_idx, bucket_minutes):
        bk = f"{idx // 60:02d}:{idx % 60:02d}"
        buckets[bk] = {"id": bk, "time": bk}

    for event in events:
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
        raw_gate_name = (event.get("gateName") or event.get("locationName") or "").strip()
        normalized_gate = gate_registry.normalize_gate_name(raw_gate_name)
        gate_name = gate_name_by_normalized.get(normalized_gate, raw_gate_name)
        if not gate_name:
            continue
        all_gates.add(gate_name)

        if bucket_key not in buckets:
            buckets[bucket_key] = {"id": bucket_key, "time": bucket_key}
        buckets[bucket_key][gate_name] = int(buckets[bucket_key].get(gate_name, 0)) + 1

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
    events = list_gate_crossing_events(date, gate_id)

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
        lane_count = gate.get("defaultLaneCount") or 0
        if loc_override:
            lc = loc_override.get("laneCount")
            if isinstance(lc, (int, float)) and lc > 0:
                lane_count = int(round(lc))
        gate_meta[gate["name"]] = {
            "lane_count": lane_count,
        }

    bucket_minutes = max(1, int(bucket_minutes))
    window_start_min, window_end_min = _time_range_bounds(time_range, start_time)

    # Bucket events per gate name
    # Structure: { bucket_key: { gate_name: { class_name: count } } }
    buckets: dict[str, dict[str, dict[str, int]]] = {}
    all_gate_names: set[str] = {g["name"] for g in gates}

    if gate_id:
        target_gate = next((g["name"] for g in gates if str(g["id"]) == str(gate_id)), None)
        all_gate_names = {target_gate} if target_gate else set()

    start_idx = 0
    end_idx = 24 * 60
    if window_start_min is not None and window_end_min is not None:
        start_idx = (window_start_min // bucket_minutes) * bucket_minutes
        end_idx = (window_end_min // bucket_minutes) * bucket_minutes

    for idx in range(start_idx, end_idx, bucket_minutes):
        bk = f"{idx // 60:02d}:{idx % 60:02d}"
        buckets[bk] = {}

    for event in events:
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
        raw_gate_name = (event.get("gateName") or event.get("locationName") or "").strip()
        normalized_gate = gate_registry.normalize_gate_name(raw_gate_name)
        gate_name = gate_name_by_normalized.get(normalized_gate, raw_gate_name)
        if not gate_name:
            continue
        vehicle_class = normalize_vehicle_class_name(event.get("vehicleClass")) or "car"
        all_gate_names.add(gate_name)

        bucket = buckets.setdefault(bucket_key, {})
        gate_bucket = bucket.setdefault(gate_name, {})
        gate_bucket[vehicle_class] = gate_bucket.get(vehicle_class, 0) + 1

    # Per-bucket LOS comes from per-frame V/C in each video's _congestion.csv,
    # not from gate-crossing counts. Crossing counts and capacity-by-lane mix
    # flow with instantaneous occupancy, so we instead aggregate the same
    # values the realtime overlay uses.
    samples_by_gate = collect_congestion_samples(date, window_start_min, window_end_min, gate_id=gate_id)
    # Bucket samples per gate by absolute_minute → bucket_index.
    vc_by_bucket: dict[str, dict[str, list[float]]] = {}
    for gate_name, samples in samples_by_gate.items():
        for abs_min, vc, _los in samples:
            bucket_index = (abs_min // bucket_minutes) * bucket_minutes
            bucket_key = f"{bucket_index // 60:02d}:{bucket_index % 60:02d}"
            vc_by_bucket.setdefault(bucket_key, {}).setdefault(gate_name, []).append(vc)
            buckets.setdefault(bucket_key, {})
        all_gate_names.add(gate_name)

    sorted_keys = sorted(buckets.keys())
    result = []
    for key in sorted_keys:
        point: dict[str, Any] = {"id": key, "time": key}
        gate_data = buckets[key]
        gate_vc_data = vc_by_bucket.get(key, {})
        for gate_name in all_gate_names:
            counts = gate_data.get(gate_name, {})
            # Mean V/C for this bucket from per-frame samples; if no
            # congestion data, leave LOS empty so the chart skips the point.
            vc_samples = gate_vc_data.get(gate_name, [])
            mean_vc = sum(vc_samples) / len(vc_samples) if vc_samples else None
            grade = los_module.get_los(mean_vc) if mean_vc is not None else None
            rank = los_module.los_rank(grade)
            point[gate_name] = sum(counts.values())
            point[f"{gate_name}__los"] = grade
            point[f"{gate_name}__losRank"] = rank if rank is not None else 0
        result.append(point)
    return result
