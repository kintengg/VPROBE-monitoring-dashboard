"""Vehicle Level-of-Service (LOS) computation.

Ported from Occlusion-Robust-RTDETR/tools/traffic_congestion.py. Same V/C
thresholds and Passenger Car Equivalent (PCE) multipliers, but expressed as
small pure functions instead of an inheritable estimator class.
"""

from __future__ import annotations

from typing import Iterable, Mapping, Optional

LOS_GRADES: tuple[str, ...] = ("A", "B", "C", "D", "E", "F")

LOS_RANK: dict[str, int] = {grade: index + 1 for index, grade in enumerate(LOS_GRADES)}

LOS_THRESHOLDS: tuple[tuple[str, float, float], ...] = (
    ("A", 0.0, 0.20),
    ("B", 0.20, 0.50),
    ("C", 0.50, 0.70),
    ("D", 0.70, 0.85),
    ("E", 0.85, 1.00),
    ("F", 1.00, float("inf")),
)

LOS_DESCRIPTIONS: dict[str, str] = {
    "A": "Free flow — excellent conditions",
    "B": "Reasonably free flow",
    "C": "Stable flow",
    "D": "Approaching unstable flow",
    "E": "Unstable flow",
    "F": "Forced or breakdown flow",
}

PCE_MULTIPLIERS: dict[str, float] = {
    "car": 1.0,
    "motorcycle": 1.0,
    "bicycle": 0.5,
    "tricycle": 2.5,
    "bus": 2.0,
    "van": 1.5,
    "truck": 2.5,
    "jeepney": 1.5,
    "suv": 1.0,
}

JAM_DENSITY_VEH_PER_KM_PER_LANE: float = 145.0

# HCM 2010 arterial saturation flow rate per lane (vehicles/hour/lane).
# Used by the dashboard to compare bucketed crossing volume against a flow
# capacity rather than a static jam-density occupancy. compute_capacity()
# (jam-density) is still correct for the per-frame realtime overlay where
# both volume and capacity are instantaneous counts.
HCM_PER_LANE_HOURLY_CAPACITY: float = 1900.0


def get_los(vc_ratio: Optional[float]) -> Optional[str]:
    """Return the LOS letter grade for a V/C ratio, or None if input is None."""
    if vc_ratio is None:
        return None
    if vc_ratio < 0:
        vc_ratio = 0.0
    for grade, low, high in LOS_THRESHOLDS:
        if grade == "A":
            if low <= vc_ratio <= high:
                return grade
        else:
            if low < vc_ratio <= high:
                return grade
    return "F"


def los_rank(grade: Optional[str]) -> int:
    """Return a numeric rank (1=A … 6=F, 0=unknown) for ordering."""
    if not grade:
        return 0
    return LOS_RANK.get(grade, 0)


def los_description(grade: Optional[str]) -> Optional[str]:
    if not grade:
        return None
    return LOS_DESCRIPTIONS.get(grade)


def compute_volume(class_counts: Mapping[str, int]) -> float:
    """Sum vehicle counts weighted by their PCE multiplier."""
    volume = 0.0
    for class_name, count in class_counts.items():
        if not count:
            continue
        multiplier = PCE_MULTIPLIERS.get(str(class_name).lower(), 1.0)
        volume += float(count) * multiplier
    return volume


def compute_capacity(road_length_km: float, num_lanes: int, jam_density: float = JAM_DENSITY_VEH_PER_KM_PER_LANE) -> float:
    """Jam-density occupancy capacity: N_max = N_lanes * L * k_j.

    Use this only when comparing against an *instantaneous* vehicle count
    (e.g., the per-frame realtime overlay drawn by RT-DETR). For bucketed
    flow-volume V/C ratios in the dashboard, use compute_flow_capacity().
    """
    return max(0.0, float(num_lanes)) * max(0.0, float(road_length_km)) * float(jam_density)


def compute_flow_capacity(num_lanes: int, hours: float = 1.0, per_lane_hourly: float = HCM_PER_LANE_HOURLY_CAPACITY) -> float:
    """Flow capacity over a window of `hours`: lanes * 1900 PCU/h/lane * hours.

    Suitable for V/C ratios where the numerator is a count of crossings
    accumulated over the same time window (i.e., a flow, not an occupancy).
    """
    return max(0.0, float(num_lanes)) * float(per_lane_hourly) * max(0.0, float(hours))


def compute_vc_ratio(class_counts: Mapping[str, int], capacity: float) -> Optional[float]:
    if capacity <= 0:
        return None
    volume = compute_volume(class_counts)
    return volume / capacity


def aggregate_class_counts(events: Iterable[Mapping[str, object]], class_field: str = "vehicleClass") -> dict[str, int]:
    """Tally raw class -> count from a sequence of detection event dicts."""
    counts: dict[str, int] = {}
    for event in events:
        raw = event.get(class_field)
        if not raw:
            continue
        key = str(raw).strip().lower()
        if not key:
            continue
        counts[key] = counts.get(key, 0) + 1
    return counts
