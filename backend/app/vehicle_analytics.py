import json
import math
from datetime import datetime, timedelta
from typing import Any, Optional

def compute_vehicle_analytics(
    events: list[dict[str, Any]],
    videos: list[dict[str, Any]],
    resolved_date: datetime,
    time_range: str,
    start_time: Optional[str]
) -> dict[str, Any]:
    if isinstance(resolved_date, str):
        resolved_date = datetime.strptime(resolved_date, "%Y-%m-%d")

    # Parse start_time or use 00:00
    if start_time:
        try:
            st = datetime.strptime(start_time, "%H:%M")
            base_time = resolved_date.replace(hour=st.hour, minute=st.minute, second=0, microsecond=0)
        except ValueError:
            base_time = resolved_date.replace(hour=0, minute=0, second=0, microsecond=0)
    else:
        base_time = resolved_date.replace(hour=0, minute=0, second=0, microsecond=0)

    # Determine bucket size and number of buckets based on time_range
    if time_range == "1h":
        bucket_span = timedelta(minutes=5)
        num_buckets = 12
    elif time_range == "6h":
        bucket_span = timedelta(minutes=15)
        num_buckets = 24
    elif time_range == "12h":
        bucket_span = timedelta(minutes=30)
        num_buckets = 24
    else: # "whole-day"
        bucket_span = timedelta(hours=1)
        num_buckets = 24
        
    buckets = []
    for i in range(num_buckets):
        b_start = base_time + (bucket_span * i)
        if time_range == "1h" or time_range == "6h" or time_range == "12h":
            b_label = b_start.strftime("%I:%M %p")
        else:
            b_label = b_start.strftime("%I %p")
        buckets.append((b_start, b_label))

    end_time = base_time + (bucket_span * num_buckets)

    # Prepare series data
    series = []
    for b_start, b_label in buckets:
        series.append({
            "timestamp": b_label,
            "id": b_start.isoformat(),
            "time": b_label,
            "total": 0,
            "classes": {},
            "gates": {},
            "entry_cumulative": 0,
            "exit_cumulative": 0,
            "los_vc_ratio": 0.0,
            "los_level": "A",
            "capacity": 0.0
        })

    # Maps video ID to location (gate)
    video_locations = {v["id"]: v.get("location", "Unknown") for v in videos}

    # Track entry/exit cumulative counts
    # Entry gates: Gate 2, Gate 3
    # Exit gates: Gate 2.9, Gate 3.2, Gate 3.5
    entry_gates = ["Gate 2", "Gate 3"]
    exit_gates = ["Gate 2.9", "Gate 3.2", "Gate 3.5"]

    # We need to process events in chronological order for cumulative
    def get_event_time(evt):
        try:
            return datetime.strptime(f"{resolved_date.strftime('%Y-%m-%d')} {evt['timestamp']}", "%Y-%m-%d %H:%M:%S")
        except:
            try:
                return datetime.strptime(f"{resolved_date.strftime('%Y-%m-%d')} {evt['timestamp']}", "%Y-%m-%d %I:%M:%S %p")
            except:
                return resolved_date
    
    valid_events = []
    for evt in events:
        if evt.get("type") != "detection": continue
        evt_time = get_event_time(evt)
        valid_events.append((evt_time, evt))
    
    valid_events.sort(key=lambda x: x[0])

    running_entry = 0
    running_exit = 0

    # Count base cumulative before base_time
    for evt_time, evt in valid_events:
        if evt_time < base_time:
            loc = evt.get("location", video_locations.get(evt.get("videoId")))
            if any(g in loc for g in entry_gates):
                running_entry += 1
            elif any(g in loc for g in exit_gates):
                running_exit += 1

    # Populate buckets
    for i, (b_start, b_label) in enumerate(buckets):
        b_end = b_start + bucket_span
        
        # Events in this bucket
        bucket_events = [e for e in valid_events if b_start <= e[0] < b_end]
        
        # Update running cumulative based on bucket events
        bucket_entry_count = 0
        bucket_exit_count = 0
        for evt_time, evt in bucket_events:
            loc = evt.get("location", video_locations.get(evt.get("videoId")))
            if any(g in loc for g in entry_gates):
                bucket_entry_count += 1
                running_entry += 1
            elif any(g in loc for g in exit_gates):
                bucket_exit_count += 1
                running_exit += 1
        
        # Assign cumulatives
        series[i]["entry_cumulative"] = running_entry
        series[i]["exit_cumulative"] = running_exit

        # Totals and classes
        for evt_time, evt in bucket_events:
            series[i]["total"] += 1
            v_class = evt.get("vehicleClassLabel") or "Unknown"
            series[i]["classes"][v_class] = series[i]["classes"].get(v_class, 0) + 1
            
            loc = evt.get("location", video_locations.get(evt.get("videoId"))) or "Unknown"
            series[i]["gates"][loc] = series[i]["gates"].get(loc, 0) + 1

        # LOS Calculation
        # V/C ratio = Volume / Capacity. 
        # Capacity per hour = ~1000 vehicles for a typical campus road (just an estimate, or we base it on interval)
        # Let's say hourly capacity is 1200. For the bucket span, capacity is:
        hourly_capacity = 1200
        bucket_capacity = hourly_capacity * (bucket_span.total_seconds() / 3600.0)
        series[i]["capacity"] = bucket_capacity
        
        vc_ratio = series[i]["total"] / bucket_capacity if bucket_capacity > 0 else 0
        series[i]["los_vc_ratio"] = round(vc_ratio, 3)
        
        # LOS Levels: A (<0.2), B (<0.4), C (<0.6), D (<0.8), E (<1.0), F (>=1.0)
        if vc_ratio < 0.2: los = "A"
        elif vc_ratio < 0.4: los = "B"
        elif vc_ratio < 0.6: los = "C"
        elif vc_ratio < 0.8: los = "D"
        elif vc_ratio < 1.0: los = "E"
        else: los = "F"
        series[i]["los_level"] = los

    # Summary Statistics
    daily_total = sum(s["total"] for s in series)
    num_hours = len(buckets) * (bucket_span.total_seconds() / 3600.0)
    daily_average = round(daily_total / num_hours, 2) if num_hours > 0 else 0
    
    # Peak
    peak_bucket = max(series, key=lambda s: s["total"]) if series else None
    peak_volume = peak_bucket["total"] if peak_bucket else 0
    peak_hour = peak_bucket["timestamp"] if peak_bucket else "N/A"

    return {
        "timeRange": time_range,
        "bucketMinutes": int(bucket_span.total_seconds() / 60),
        "series": series,
        "summary": {
            "dailyAverage": daily_average,
            "peakVolume": peak_volume,
            "peakHour": peak_hour,
            "dailyTotal": daily_total
        }
    }
