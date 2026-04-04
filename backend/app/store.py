from __future__ import annotations

import json
import logging
import math
import os
import re
import shutil
from collections.abc import Callable
from copy import deepcopy
from datetime import datetime, timedelta
from pathlib import Path
from threading import Lock
from typing import Any, Optional, Union
from uuid import uuid4

logger = logging.getLogger(__name__)

BACKEND_DIR = Path(__file__).resolve().parents[1]
STORAGE_DIR = BACKEND_DIR / "storage"
MODELS_DIR = STORAGE_DIR / "models"
RAW_VIDEOS_DIR = STORAGE_DIR / "videos" / "raw"
PROCESSED_VIDEOS_DIR = STORAGE_DIR / "videos" / "processed"
EXPORTS_DIR = STORAGE_DIR / "exports"
DATA_FILE = STORAGE_DIR / "dev_data.json"
CLOCK_TIME_FORMATS = ("%H:%M", "%H:%M:%S", "%I:%M %p", "%I:%M:%S %p")
MIN_DRILLDOWN_BUCKET_MINUTES = 5
OWDI_CLASS_WEIGHTS = {0: 1, 1: 2, 2: 3}
PTSI_OCCLUSION_WEIGHTS = {0: 1, 1: 2, 2: 3}
PTSI_CONGESTION_WEIGHT = 0.85
PTSI_OCCLUSION_WEIGHT = 0.15
PTSI_ROI_TESTING_CAPACITY_PER_FULL_FRAME = 24.0
SEMANTIC_MIN_SCORE = 0.18
SEMANTIC_POSSIBLE_MATCH_SCORE = 0.28
SEMANTIC_STRONG_MATCH_SCORE = 0.45
PTSI_LOS_DESCRIPTIONS = {
    "A": "very high pedestrian space, free movement",
    "B": "high pedestrian space, comfortable movement",
    "C": "adequate space, noticeable interaction but manageable flow",
    "D": "limited space, constrained movement",
    "E": "crowded conditions, frequent interference",
    "F": "severely congested conditions, breakdown of smooth movement",
}
PTSI_LOS_RANKS = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5}
PTSI_LOS_BY_RANK = {rank: los for los, rank in PTSI_LOS_RANKS.items()}
PTSI_LOS_STATE_MAP = {
    "A": "clear",
    "B": "clear",
    "C": "clear",
    "D": "moderate",
    "E": "moderate",
    "F": "severe",
}
DEFAULT_EDSA_SEC_WALK_ROI = {
    "referenceSize": [1920, 1080],
    "includePolygonsNorm": [
        [[0.245870, 0.988037], [0.995651, 0.997306], [0.219792, 0.098426], [0.196354, 0.101481]],
        [[0.432646, 0.337861], [0.564266, 0.309009], [0.544432, 0.472491], [0.487635, 0.403574]],
        [[0.221693, 0.094250], [0.346099, 0.244907], [0.344297, 0.187213]],
    ],
}
LOCATION_PERSISTED_FIELDS = (
    "id",
    "name",
    "latitude",
    "longitude",
    "description",
    "address",
    "roiCoordinates",
    "walkableAreaM2",
)
SEARCH_STOPWORDS = {
    "a",
    "an",
    "and",
    "at",
    "for",
    "i",
    "im",
    "in",
    "is",
    "individual",
    "looking",
    "find",
    "person",
    "pedestrian",
    "people",
    "me",
    "of",
    "on",
    "or",
    "someone",
    "somebody",
    "that",
    "the",
    "to",
    "wearing",
    "with",
}
SEARCH_PHRASE_SYNONYMS = {
    "dark red": ("maroon", "burgundy", "wine", "red", "purple", "pink"),
    "flowy top": ("top", "upper clothing", "shirt", "blouse"),
}
SEARCH_TERM_SYNONYMS = {
    "beanie": ("head", "region", "hat"),
    "blouse": ("upper", "clothing", "top"),
    "burgundy": ("red", "purple", "maroon", "wine"),
    "cap": ("head", "region", "hat"),
    "crimson": ("red", "maroon", "pink"),
    "flowy": ("upper", "clothing", "top"),
    "hat": ("head", "region", "cap"),
    "hood": ("head", "region"),
    "leggings": ("lower", "clothing", "pants"),
    "maroon": ("red", "purple", "pink", "burgundy", "wine"),
    "pants": ("lower", "clothing", "trousers"),
    "scarlet": ("red", "maroon", "pink"),
    "shirt": ("upper", "clothing", "top"),
    "shorts": ("lower", "clothing", "pants"),
    "skirt": ("lower", "clothing"),
    "sleeveless": ("upper", "clothing", "top"),
    "tank": ("upper", "clothing", "top"),
    "tee": ("upper", "clothing", "shirt"),
    "top": ("upper", "clothing", "shirt", "blouse"),
    "trousers": ("lower", "clothing", "pants"),
    "wine": ("red", "purple", "maroon", "burgundy"),
}
SEARCH_REGION_ALIASES = {
    "head region": ("hat", "cap", "beanie", "hood", "head", "head region"),
    "upper clothing": (
        "top",
        "shirt",
        "blouse",
        "tank",
        "tee",
        "upper",
        "upper clothing",
        "sleeveless",
        "flowy top",
    ),
    "lower clothing": ("shorts", "pants", "trousers", "skirt", "leggings", "lower", "lower clothing"),
}
SEARCH_COLOR_ALIASES = {
    "black": ("black",),
    "blue": ("blue", "cyan"),
    "brown": ("brown",),
    "gray": ("gray", "grey"),
    "green": ("green",),
    "orange": ("orange",),
    "pink": ("pink",),
    "purple": ("purple",),
    "red": ("red", "dark red", "maroon", "burgundy", "wine", "crimson", "scarlet"),
    "white": ("white",),
    "yellow": ("yellow",),
}
SEARCH_COLOR_FAMILY_MATCHES = {
    "black": {"black"},
    "blue": {"blue", "cyan"},
    "brown": {"brown"},
    "gray": {"gray"},
    "green": {"green"},
    "orange": {"orange"},
    "pink": {"pink", "purple", "red", "maroon", "burgundy", "wine"},
    "purple": {"purple", "pink", "red", "maroon", "burgundy", "wine"},
    "red": {"red", "pink", "purple", "maroon", "burgundy", "wine"},
    "white": {"white"},
    "yellow": {"yellow"},
}
SEARCH_CAMERA_AMBIGUOUS_COLOR_MATCHES = {
    "gray": {"white"},
    "white": {"gray"},
}
QUERY_COLOR_JOINERS = {"and", "or"}
QUERY_COLOR_MODIFIERS = {"dark", "light"}
MAX_AI_SEARCH_CANDIDATES = 24
UPLOAD_STATUS_LOCK = Lock()
UPLOAD_STATUSES: dict[str, dict[str, Any]] = {}
UPLOAD_CANCEL_REQUESTS: set[str] = set()


def slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.strip().lower()).strip("-") or f"item-{uuid4().hex[:8]}"


def _utc_timestamp() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def set_upload_status(
    upload_id: str,
    *,
    state: str,
    progress_percent: Optional[int],
    message: str,
    phase: Optional[str] = None,
    video_id: Optional[str] = None,
    error: Optional[str] = None,
) -> dict[str, Any]:
    normalized_progress = None if progress_percent is None else max(0, min(100, int(round(progress_percent))))
    status = {
        "uploadId": upload_id,
        "state": state,
        "progressPercent": normalized_progress,
        "message": message,
        "phase": phase,
        "videoId": video_id,
        "error": error,
        "updatedAt": _utc_timestamp(),
    }

    with UPLOAD_STATUS_LOCK:
        UPLOAD_STATUSES[upload_id] = status
        if state in {"complete", "error", "cancelled"}:
            UPLOAD_CANCEL_REQUESTS.discard(upload_id)

    return deepcopy(status)


def get_upload_status(upload_id: str) -> Optional[dict[str, Any]]:
    with UPLOAD_STATUS_LOCK:
        status = UPLOAD_STATUSES.get(upload_id)
    return deepcopy(status) if status is not None else None


def request_upload_cancel(upload_id: str) -> bool:
    with UPLOAD_STATUS_LOCK:
        if upload_id not in UPLOAD_STATUSES:
            return False
        UPLOAD_CANCEL_REQUESTS.add(upload_id)
    return True


def is_upload_cancel_requested(upload_id: str) -> bool:
    with UPLOAD_STATUS_LOCK:
        return upload_id in UPLOAD_CANCEL_REQUESTS


def _location_video_cards(videos: list[dict[str, Any]], location_id: str) -> list[dict[str, Any]]:
    return [
        {
            key: video[key]
            for key in (
                "id",
                "timestamp",
                "date",
                "startTime",
                "endTime",
                "pedestrianCount",
                "rawPath",
                "processedPath",
            )
        }
        for video in videos
        if video["locationId"] == location_id
    ]


def _location_record(state: dict[str, Any], location_id: str, date: Optional[str] = None) -> dict[str, Any]:
    location = next((item for item in state["locations"] if item["id"] == location_id), None)
    if location is None:
        raise ValueError("Location not found")

    videos = state["videos"]
    if date:
        videos = [video for video in videos if video["date"] == date]

    return {**location, "videos": _location_video_cards(videos, location_id)}


def _ensure_location_name_is_unique(state: dict[str, Any], name: str, exclude_id: Optional[str] = None) -> None:
    normalized_name = name.strip().casefold()
    for location in state["locations"]:
        if exclude_id and location["id"] == exclude_id:
            continue
        if location["name"].strip().casefold() == normalized_name:
            raise ValueError("A location with that name already exists")


def _location_payload(location: dict[str, Any]) -> dict[str, Any]:
    return {field: location.get(field) for field in LOCATION_PERSISTED_FIELDS}


def seed_state() -> dict[str, Any]:
    return {
        "model": {"currentModel": "yolov8n-bytetrack.pt", "uploadedAt": None},
        "locations": [
            {
                "id": "edsa-sec-walk",
                "name": "EDSA Sec Walk",
                "latitude": 14.6397,
                "longitude": 121.0775,
                "description": "Approximate Xavier Hall camera anchor along the EDSA security walkway.",
                "address": "Ateneo de Manila University · Xavier Hall",
                "roiCoordinates": deepcopy(DEFAULT_EDSA_SEC_WALK_ROI),
                "walkableAreaM2": None,
            },
            {
                "id": "kostka-walk",
                "name": "Kostka Walk",
                "latitude": 14.6390,
                "longitude": 121.0781,
                "description": "Approximate Kostka Hall pedestrian corridor camera anchor.",
                "address": "Ateneo de Manila University · Kostka Hall",
                "roiCoordinates": None,
                "walkableAreaM2": None,
            },
            {
                "id": "gate-1-walkway",
                "name": "Gate 1 Walkway",
                "latitude": 14.6418,
                "longitude": 121.0758,
                "description": "Approximate Gate 1 walkway camera anchor.",
                "address": "Ateneo de Manila University · Gate 1",
                "roiCoordinates": None,
                "walkableAreaM2": None,
            },
            {
                "id": "gate-3-walkway",
                "name": "Gate 3 Walkway",
                "latitude": 14.6376,
                "longitude": 121.0742,
                "description": "Approximate Gate 3 walkway camera anchor.",
                "address": "Ateneo de Manila University · Gate 3",
                "roiCoordinates": None,
                "walkableAreaM2": None,
            },
        ],
        "videos": [],
        "events": [],
        "pedestrianTracks": [],
    }


def ensure_storage_layout() -> None:
    for path in (STORAGE_DIR, MODELS_DIR, RAW_VIDEOS_DIR, PROCESSED_VIDEOS_DIR, EXPORTS_DIR):
        path.mkdir(parents=True, exist_ok=True)
    if not DATA_FILE.exists():
        DATA_FILE.write_text(json.dumps(seed_state(), indent=2), encoding="utf-8")


def load_state() -> dict[str, Any]:
    ensure_storage_layout()
    state = json.loads(DATA_FILE.read_text(encoding="utf-8"))
    changed = False

    if "pedestrianTracks" not in state:
        state["pedestrianTracks"] = []
        changed = True

    for location in state.get("locations", []):
        if "roiCoordinates" not in location:
            location["roiCoordinates"] = None
            changed = True
        if "walkableAreaM2" not in location:
            location["walkableAreaM2"] = None
            changed = True
        if location.get("id") == "edsa-sec-walk" and not location.get("roiCoordinates"):
            location["roiCoordinates"] = deepcopy(DEFAULT_EDSA_SEC_WALK_ROI)
            changed = True

    for track in state.get("pedestrianTracks", []):
        if not isinstance(track.get("trajectorySamples"), list):
            track["trajectorySamples"] = _legacy_track_trajectory_samples(track)
            changed = True
        if not isinstance(track.get("semanticCrops"), list):
            track["semanticCrops"] = _legacy_track_semantic_crops(track)
            changed = True

    if changed:
        save_state(state)

    return state


def save_state(state: dict[str, Any]) -> None:
    DATA_FILE.write_text(json.dumps(state, indent=2), encoding="utf-8")


def _resolve_backend_path(path_value: Optional[str]) -> Optional[Path]:
    if not path_value:
        return None

    candidate = Path(path_value)
    if not candidate.is_absolute():
        candidate = BACKEND_DIR / candidate

    try:
        candidate.relative_to(BACKEND_DIR)
    except ValueError:
        return None

    return candidate


def resolve_video_source_path(video: Optional[dict[str, Any]]) -> Optional[Path]:
    if video is None:
        return None

    raw_path = _resolve_backend_path(video.get("rawPath"))
    if raw_path is not None and raw_path.exists():
        return raw_path

    processed_path = _resolve_backend_path(video.get("processedPath"))
    if processed_path is not None and processed_path.exists():
        return processed_path

    return None


def _legacy_track_semantic_crops(track: dict[str, Any]) -> list[dict[str, Any]]:
    thumbnail_path = str(track.get("thumbnailPath") or "").strip()
    if not thumbnail_path:
        return []
    return [
        {
            "label": "best",
            "path": thumbnail_path,
            "frame": track.get("bestFrame") or track.get("firstFrame"),
            "timestamp": track.get("bestTimestamp") or track.get("firstTimestamp"),
            "offsetSeconds": track.get("bestOffsetSeconds") if track.get("bestOffsetSeconds") is not None else track.get("firstOffsetSeconds"),
        }
    ]


def _refresh_semantic_index(state: dict[str, Any]) -> None:
    try:
        from . import semantic_search

        semantic_search.rebuild_index(state, backend_dir=BACKEND_DIR)
    except Exception:
        logger.exception("Semantic search index rebuild failed.")


def delete_video_assets(video: Optional[dict[str, Any]]) -> None:
    if video is None:
        return

    raw_path = _resolve_backend_path(video.get("rawPath"))
    if raw_path is not None:
        raw_path.unlink(missing_ok=True)

    processed_path = _resolve_backend_path(video.get("processedPath"))
    if processed_path is not None:
        processed_path.unlink(missing_ok=True)

    video_id = video.get("id")
    if video_id:
        shutil.rmtree(PROCESSED_VIDEOS_DIR / str(video_id), ignore_errors=True)


def list_locations(date: Optional[str] = None) -> list[dict[str, Any]]:
    state = load_state()
    videos = state["videos"]
    if date:
        videos = [video for video in videos if video["date"] == date]
    grouped: list[dict[str, Any]] = []
    for location in state["locations"]:
        grouped.append({**location, "videos": _location_video_cards(videos, location["id"])})
    return grouped


def add_location(payload: dict[str, Any]) -> dict[str, Any]:
    state = load_state()
    _ensure_location_name_is_unique(state, payload["name"])

    location_id = slugify(payload["name"])
    location = {**payload, "id": location_id}
    state["locations"].append(_location_payload(location))
    save_state(state)
    return _location_record(state, location_id)


def update_location(location_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    state = load_state()
    location = next((item for item in state["locations"] if item["id"] == location_id), None)
    if location is None:
        raise ValueError("Location not found")

    _ensure_location_name_is_unique(state, payload["name"], exclude_id=location_id)

    original_name = location["name"]
    location.update(
        {
            "name": payload["name"],
            "latitude": payload["latitude"],
            "longitude": payload["longitude"],
            "description": payload.get("description", ""),
            "address": payload.get("address", ""),
            "roiCoordinates": payload.get("roiCoordinates"),
            "walkableAreaM2": payload.get("walkableAreaM2"),
        }
    )

    affected_video_ids: set[str] = set()
    for video in state["videos"]:
        if video["locationId"] != location_id:
            continue
        video["location"] = location["name"]
        video["gpsLat"] = location["latitude"]
        video["gpsLng"] = location["longitude"]
        affected_video_ids.add(video["id"])

    for event in state["events"]:
        if event.get("videoId") in affected_video_ids or (event.get("videoId") is None and event.get("location") == original_name):
            event["location"] = location["name"]

    for track in state.get("pedestrianTracks", []):
        if track.get("videoId") in affected_video_ids or (track.get("videoId") is None and track.get("location") == original_name):
            track["location"] = location["name"]

    save_state(state)
    return _location_record(state, location_id)


def delete_location(location_id: str) -> bool:
    state = load_state()
    location = next((item for item in state["locations"] if item["id"] == location_id), None)
    if location is None:
        return False

    videos_to_delete = [deepcopy(video) for video in state["videos"] if video["locationId"] == location_id]
    removed_video_ids = {video["id"] for video in videos_to_delete}

    state["locations"] = [item for item in state["locations"] if item["id"] != location_id]
    state["videos"] = [video for video in state["videos"] if video["locationId"] != location_id]
    state["events"] = [
        event
        for event in state["events"]
        if event.get("videoId") not in removed_video_ids and not (event.get("videoId") is None and event.get("location") == location["name"])
    ]
    state["pedestrianTracks"] = [
        track
        for track in state.get("pedestrianTracks", [])
        if track.get("videoId") not in removed_video_ids and not (track.get("videoId") is None and track.get("location") == location["name"])
    ]
    save_state(state)
    _refresh_semantic_index(state)

    for video in videos_to_delete:
        delete_video_assets(video)

    return True


def list_videos() -> list[dict[str, Any]]:
    return load_state()["videos"]


def get_video(video_id: str) -> Optional[dict[str, Any]]:
    return next((video for video in load_state()["videos"] if video["id"] == video_id), None)


def add_video(payload: dict[str, Any]) -> dict[str, Any]:
    state = load_state()
    location = next((item for item in state["locations"] if item["id"] == payload["locationId"]), None)
    if location is None:
        raise ValueError("Unknown locationId")
    record = {
        "id": uuid4().hex[:8],
        "locationId": location["id"],
        "location": location["name"],
        "timestamp": payload["startTime"],
        "date": payload["date"],
        "startTime": payload["startTime"],
        "endTime": payload["endTime"],
        "gpsLat": location["latitude"],
        "gpsLng": location["longitude"],
        "pedestrianCount": payload.get("pedestrianCount", 0),
        "rawPath": payload.get("rawPath"),
        "processedPath": payload.get("processedPath"),
    }
    state["videos"].append(record)
    save_state(state)
    return record


def set_video_inference_result(
    video_id: str,
    pedestrian_count: int,
    processed_path: Optional[str],
    events: list[dict[str, Any]],
    pedestrian_tracks: Optional[list[dict[str, Any]]] = None,
) -> dict[str, Any]:
    state = load_state()
    video = next((item for item in state["videos"] if item["id"] == video_id), None)
    if video is None:
        raise ValueError("Video not found")

    video["pedestrianCount"] = pedestrian_count
    video["processedPath"] = processed_path
    state["events"] = [event for event in state["events"] if event.get("videoId") != video_id]
    state["events"].extend(events)
    state["pedestrianTracks"] = [track for track in state.get("pedestrianTracks", []) if track.get("videoId") != video_id]
    state["pedestrianTracks"].extend(pedestrian_tracks or [])
    save_state(state)
    _refresh_semantic_index(state)
    return deepcopy(video)


def remove_video(video_id: str) -> bool:
    state = load_state()
    original_video_count = len(state["videos"])
    state["videos"] = [video for video in state["videos"] if video["id"] != video_id]
    state["events"] = [event for event in state["events"] if event.get("videoId") != video_id]
    state["pedestrianTracks"] = [track for track in state.get("pedestrianTracks", []) if track.get("videoId") != video_id]
    removed = len(state["videos"]) != original_video_count
    if removed:
        save_state(state)
        _refresh_semantic_index(state)
    return removed


def list_events(video_id: Optional[str] = None) -> list[dict[str, Any]]:
    events = load_state()["events"]
    if video_id:
        return [event for event in events if event.get("videoId") == video_id]
    return events


def get_model_info() -> dict[str, Any]:
    return deepcopy(load_state()["model"])


def set_model(filename: str) -> dict[str, Any]:
    state = load_state()
    state["model"] = {"currentModel": filename, "uploadedAt": datetime.utcnow().isoformat(timespec="seconds") + "Z"}
    save_state(state)
    return deepcopy(state["model"])


def format_time_range_label(time_range: str) -> str:
    labels = {
        "whole-day": "Whole Day",
        "last-1h": "Last 1 Hour",
        "last-3h": "Last 3 Hours",
        "last-6h": "Last 6 Hours",
        "last-12h": "Last 12 Hours",
        "morning": "Morning",
        "afternoon": "Afternoon",
        "evening": "Evening",
    }
    return labels.get(time_range, time_range.replace("-", " ").title())


def _parse_clock_time(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None

    cleaned = str(value).strip()
    for time_format in CLOCK_TIME_FORMATS:
        try:
            return datetime.strptime(cleaned, time_format)
        except ValueError:
            continue
    return None


def _combine_date_and_time(date_value: str, time_value: Optional[str]) -> Optional[datetime]:
    parsed_time = _parse_clock_time(time_value)
    if parsed_time is None:
        return None

    try:
        base_date = datetime.strptime(date_value, "%Y-%m-%d")
    except ValueError:
        return None

    return base_date.replace(hour=parsed_time.hour, minute=parsed_time.minute, second=parsed_time.second, microsecond=0)


def _resolve_dashboard_date(videos: list[dict[str, Any]], date: Optional[str]) -> str:
    if date:
        return date
    if videos:
        return max((str(video.get("date", "")) for video in videos), default="") or datetime.utcnow().strftime("%Y-%m-%d")
    return datetime.utcnow().strftime("%Y-%m-%d")


def _filtered_dashboard_records(date: Optional[str]) -> tuple[dict[str, Any], str, list[dict[str, Any]], list[dict[str, Any]]]:
    state = load_state()
    resolved_date = _resolve_dashboard_date(state["videos"], date)
    videos = [video for video in state["videos"] if video.get("date") == resolved_date]
    video_ids = {video["id"] for video in videos}
    events = [event for event in state["events"] if event.get("videoId") in video_ids]
    return state, resolved_date, videos, events


def _filtered_pedestrian_tracks(state: dict[str, Any], videos: list[dict[str, Any]]) -> list[dict[str, Any]]:
    video_ids = {video["id"] for video in videos}
    return [track for track in state.get("pedestrianTracks", []) if track.get("videoId") in video_ids]


def _active_location_ids(videos: list[dict[str, Any]]) -> set[str]:
    return {str(video.get("locationId")) for video in videos if video.get("locationId")}


def _event_occlusion_class(event: dict[str, Any]) -> Optional[int]:
    raw_class = event.get("occlusionClass")
    try:
        normalized = int(raw_class)
    except (TypeError, ValueError):
        normalized = None

    if normalized in OWDI_CLASS_WEIGHTS:
        return normalized

    description = str(event.get("description", "")).lower()
    if "heavy occlusion" in description or "severe occlusion" in description:
        return 2
    if "moderate occlusion" in description or "partial occlusion" in description:
        return 1
    if "light occlusion" in description or "minor occlusion" in description:
        return 0
    return None


def _event_timestamp(event: dict[str, Any], video: dict[str, Any]) -> Optional[datetime]:
    event_time = _combine_date_and_time(str(video.get("date", "")), event.get("timestamp"))
    if event_time is not None:
        return event_time
    return _combine_date_and_time(str(video.get("date", "")), video.get("startTime"))


def _observation_time(video: dict[str, Any]) -> Optional[datetime]:
    return _combine_date_and_time(str(video.get("date", "")), video.get("startTime") or video.get("timestamp"))


def _floor_time(value: datetime, step_minutes: int) -> datetime:
    floored_minute = (value.minute // step_minutes) * step_minutes
    return value.replace(minute=floored_minute, second=0, microsecond=0)


def _format_window_time(value: datetime, day_end: datetime) -> str:
    if value >= day_end:
        return "24:00"
    return value.strftime("%H:%M")


def _is_detection_event(event: dict[str, Any]) -> bool:
    return str(event.get("type", "")).lower() == "detection"


def _tracked_pedestrian_key(event: dict[str, Any]) -> Optional[str]:
    video_id = event.get("videoId")
    pedestrian_id = event.get("pedestrianId")
    if not video_id or pedestrian_id is None:
        return None

    try:
        return f"{video_id}:{int(pedestrian_id)}"
    except (TypeError, ValueError):
        return None


def _tracked_pedestrian_track_key(track: dict[str, Any], video_id: str, fallback_index: int) -> str:
    pedestrian_id = track.get("pedestrianId")
    if pedestrian_id is not None:
        try:
            return f"{video_id}:{int(pedestrian_id)}"
        except (TypeError, ValueError):
            pass

    track_id = str(track.get("id") or "").strip()
    if track_id:
        return f"{video_id}:track:{track_id}"
    return f"{video_id}:track-fallback:{fallback_index}"


def _pedestrian_track_timestamp(track: dict[str, Any], video: dict[str, Any]) -> Optional[datetime]:
    track_time = _combine_date_and_time(str(video.get("date", "")), track.get("firstTimestamp") or track.get("bestTimestamp"))
    if track_time is not None:
        return track_time

    observed_at = _observation_time(video)
    if observed_at is None:
        return None

    try:
        offset_seconds = float(track.get("firstOffsetSeconds"))
    except (TypeError, ValueError):
        return observed_at
    return observed_at + timedelta(seconds=offset_seconds)


def _pedestrian_track_end_timestamp(track: dict[str, Any], video: dict[str, Any]) -> Optional[datetime]:
    track_time = _combine_date_and_time(str(video.get("date", "")), track.get("lastTimestamp") or track.get("bestTimestamp") or track.get("firstTimestamp"))
    if track_time is not None:
        return track_time

    observed_at = _observation_time(video)
    if observed_at is None:
        return None

    try:
        offset_seconds = float(track.get("lastOffsetSeconds"))
    except (TypeError, ValueError):
        return _pedestrian_track_timestamp(track, video)
    return observed_at + timedelta(seconds=offset_seconds)


def _normalized_foot_point(track: dict[str, Any]) -> Optional[tuple[float, float]]:
    raw_point = track.get("footPointNorm")
    if not isinstance(raw_point, (list, tuple)) or len(raw_point) < 2:
        return None

    try:
        x_value = float(raw_point[0])
        y_value = float(raw_point[1])
    except (TypeError, ValueError):
        return None

    if not (0.0 <= x_value <= 1.0 and 0.0 <= y_value <= 1.0):
        return None
    return (x_value, y_value)


def _trajectory_sample_second(sample: Any) -> Optional[int]:
    if not isinstance(sample, (list, tuple)) or not sample:
        return None

    try:
        return max(0, int(round(float(sample[0]))))
    except (TypeError, ValueError):
        return None


def _trajectory_sample_point(sample: Any) -> Optional[tuple[float, float]]:
    if not isinstance(sample, (list, tuple)) or len(sample) < 3:
        return None

    try:
        x_value = float(sample[1])
        y_value = float(sample[2])
    except (TypeError, ValueError):
        return None

    if not (0.0 <= x_value <= 1.0 and 0.0 <= y_value <= 1.0):
        return None
    return (x_value, y_value)


def _trajectory_sample_occlusion_class(sample: Any) -> Optional[int]:
    if not isinstance(sample, (list, tuple)) or len(sample) < 4:
        return None

    try:
        occlusion_class = int(sample[3])
    except (TypeError, ValueError):
        return None
    return occlusion_class if occlusion_class in PTSI_OCCLUSION_WEIGHTS else None


def _legacy_track_trajectory_samples(track: dict[str, Any]) -> list[list[Any]]:
    foot_point = _normalized_foot_point(track)
    if foot_point is None:
        return []

    raw_offset = track.get("bestOffsetSeconds")
    if raw_offset is None:
        raw_offset = track.get("firstOffsetSeconds")
    if raw_offset is None:
        raw_offset = track.get("lastOffsetSeconds")

    try:
        offset_seconds = max(0, int(round(float(raw_offset))))
    except (TypeError, ValueError):
        offset_seconds = 0

    return [[offset_seconds, foot_point[0], foot_point[1], track.get("occlusionClass")]]


def _normalized_trajectory_samples(track: dict[str, Any]) -> list[tuple[int, tuple[float, float], Optional[int]]]:
    raw_samples = track.get("trajectorySamples")
    sample_sets = [raw_samples] if isinstance(raw_samples, list) else []
    sample_sets.append(_legacy_track_trajectory_samples(track))

    for candidate_samples in sample_sets:
        normalized: list[tuple[int, tuple[float, float], Optional[int]]] = []
        for sample in candidate_samples:
            offset_second = _trajectory_sample_second(sample)
            point = _trajectory_sample_point(sample)
            if offset_second is None or point is None:
                continue
            normalized.append((offset_second, point, _trajectory_sample_occlusion_class(sample)))
        if normalized:
            return normalized
    return []


def _normalized_roi_polygons(location: dict[str, Any]) -> list[list[tuple[float, float]]]:
    roi_coordinates = location.get("roiCoordinates")
    if not isinstance(roi_coordinates, dict):
        return []

    raw_polygons = roi_coordinates.get("includePolygonsNorm")
    if not isinstance(raw_polygons, list):
        return []

    polygons: list[list[tuple[float, float]]] = []
    for raw_polygon in raw_polygons:
        if not isinstance(raw_polygon, list):
            continue

        polygon: list[tuple[float, float]] = []
        for raw_point in raw_polygon:
            if not isinstance(raw_point, (list, tuple)) or len(raw_point) < 2:
                polygon = []
                break
            try:
                point = (float(raw_point[0]), float(raw_point[1]))
            except (TypeError, ValueError):
                polygon = []
                break
            polygon.append(point)

        if len(polygon) >= 3:
            polygons.append(polygon)

    return polygons


def _point_on_segment(point: tuple[float, float], start: tuple[float, float], end: tuple[float, float]) -> bool:
    x_value, y_value = point
    x1, y1 = start
    x2, y2 = end
    cross_product = (x_value - x1) * (y2 - y1) - (y_value - y1) * (x2 - x1)
    if abs(cross_product) > 1e-9:
        return False
    return (
        min(x1, x2) - 1e-9 <= x_value <= max(x1, x2) + 1e-9
        and min(y1, y2) - 1e-9 <= y_value <= max(y1, y2) + 1e-9
    )


def _point_in_polygon(point: tuple[float, float], polygon: list[tuple[float, float]]) -> bool:
    inside = False
    previous_point = polygon[-1]
    for current_point in polygon:
        if _point_on_segment(point, previous_point, current_point):
            return True

        x_value, y_value = point
        x1, y1 = previous_point
        x2, y2 = current_point
        intersects = ((y1 > y_value) != (y2 > y_value)) and (
            x_value < ((x2 - x1) * (y_value - y1) / ((y2 - y1) or 1e-12)) + x1
        )
        if intersects:
            inside = not inside
        previous_point = current_point
    return inside


def _point_in_location_roi(point: tuple[float, float], location: dict[str, Any]) -> bool:
    polygons = _normalized_roi_polygons(location)
    if not polygons:
        return True
    return any(_point_in_polygon(point, polygon) for polygon in polygons)


def _track_in_location_roi(track: dict[str, Any], location: dict[str, Any]) -> bool:
    foot_point = _normalized_foot_point(track)
    if foot_point is None:
        return False
    return _point_in_location_roi(foot_point, location)


def _location_walkable_area_m2(location: dict[str, Any]) -> Optional[float]:
    raw_area = location.get("walkableAreaM2")
    try:
        area_value = float(raw_area)
    except (TypeError, ValueError):
        return None
    return area_value if area_value > 0 else None


def _location_ptsi_mode(location: dict[str, Any]) -> str:
    return "strict-fhwa" if _location_walkable_area_m2(location) is not None else "roi-testing"


def _polygon_area(polygon: list[tuple[float, float]]) -> float:
    if len(polygon) < 3:
        return 0.0

    area = 0.0
    for index, (x1, y1) in enumerate(polygon):
        x2, y2 = polygon[(index + 1) % len(polygon)]
        area += (x1 * y2) - (x2 * y1)
    return abs(area) / 2.0


def _location_roi_area_ratio(location: dict[str, Any]) -> float:
    polygons = _normalized_roi_polygons(location)
    if not polygons:
        return 1.0
    return min(1.0, max(0.0, sum(_polygon_area(polygon) for polygon in polygons)))


def _location_capacity_proxy(location: dict[str, Any]) -> float:
    return max(1.0, math.sqrt(_location_roi_area_ratio(location)) * PTSI_ROI_TESTING_CAPACITY_PER_FULL_FRAME)


def _ptsi_debug_enabled() -> bool:
    return os.getenv("PTSI_DEBUG", "").strip().lower() in {"1", "true", "yes", "on"}


def _ptsi_debug_log(event: str, **fields: Any) -> None:
    if not _ptsi_debug_enabled():
        return

    serialized_fields: dict[str, Any] = {"event": event}
    for key, value in fields.items():
        if isinstance(value, datetime):
            serialized_fields[key] = value.isoformat()
        else:
            serialized_fields[key] = value

    logger.info("PTSI_DEBUG %s", json.dumps(serialized_fields, sort_keys=True))


def _ptsi_occlusion_mix(light_count: int, moderate_count: int, heavy_count: int, visible_total: int) -> dict[str, float]:
    if visible_total <= 0:
        return {"lightPercent": 0.0, "moderatePercent": 0.0, "heavyPercent": 0.0}

    return {
        "lightPercent": round((light_count / visible_total) * 100.0, 1),
        "moderatePercent": round((moderate_count / visible_total) * 100.0, 1),
        "heavyPercent": round((heavy_count / visible_total) * 100.0, 1),
    }


def _resolve_root_window(
    date_value: str,
    time_range: str,
    observation_times: list[datetime],
) -> tuple[datetime, datetime, int]:
    day_start = datetime.strptime(date_value, "%Y-%m-%d")
    day_end = day_start + timedelta(days=1)

    if time_range in {"whole-day", "morning", "afternoon", "evening"}:
        hour_ranges = {
            "whole-day": (0, 24),
            "morning": (6, 12),
            "afternoon": (12, 18),
            "evening": (18, 24),
        }
        start_hour, end_hour = hour_ranges[time_range]
        window_end = day_end if end_hour == 24 else day_start.replace(hour=end_hour, minute=0, second=0, microsecond=0)
        return (
            day_start.replace(hour=start_hour, minute=0, second=0, microsecond=0),
            window_end,
            60,
        )

    last_windows = {
        "last-1h": (1, 5),
        "last-3h": (3, 10),
        "last-6h": (6, 15),
        "last-12h": (12, 30),
    }
    hours, step_minutes = last_windows.get(time_range, (12, 30))
    anchor = max(observation_times) if observation_times else (day_end - timedelta(minutes=1))
    anchor = min(max(anchor, day_start), day_end - timedelta(minutes=1))
    end_bucket = _floor_time(anchor, step_minutes)
    start_bucket = max(day_start, end_bucket - timedelta(hours=hours) + timedelta(minutes=step_minutes))
    window_end = min(day_end, end_bucket + timedelta(minutes=step_minutes))
    return start_bucket, window_end, step_minutes


def _next_zoom_bucket_minutes(current_bucket_minutes: int) -> Optional[int]:
    if current_bucket_minutes > 15:
        return 15
    if current_bucket_minutes > MIN_DRILLDOWN_BUCKET_MINUTES:
        return MIN_DRILLDOWN_BUCKET_MINUTES
    return None


def _bucket_minutes_for_zoom_level(root_bucket_minutes: int, zoom_level: int) -> tuple[int, int]:
    current_bucket_minutes = root_bucket_minutes
    actual_zoom_level = 0

    while actual_zoom_level < max(zoom_level, 0):
        next_bucket_minutes = _next_zoom_bucket_minutes(current_bucket_minutes)
        if next_bucket_minutes is None:
            break
        current_bucket_minutes = next_bucket_minutes
        actual_zoom_level += 1

    return current_bucket_minutes, actual_zoom_level


def _build_bucket_plan(
    date_value: str,
    time_range: str,
    observation_times: list[datetime],
    focus_time: Optional[str] = None,
    zoom_level: int = 0,
) -> tuple[list[tuple[str, datetime]], timedelta, dict[str, Any]]:
    day_start = datetime.strptime(date_value, "%Y-%m-%d")
    day_end = day_start + timedelta(days=1)

    root_window_start, root_window_end, root_bucket_minutes = _resolve_root_window(date_value, time_range, observation_times)
    requested_zoom_level = max(zoom_level, 0)
    if focus_time and requested_zoom_level == 0:
        requested_zoom_level = 1
    if not focus_time:
        requested_zoom_level = 0

    current_bucket_minutes, actual_zoom_level = _bucket_minutes_for_zoom_level(root_bucket_minutes, requested_zoom_level)
    window_start = root_window_start
    window_end = root_window_end
    normalized_focus_time = None

    if actual_zoom_level > 0 and focus_time:
        focused_at = _combine_date_and_time(date_value, focus_time)
        if focused_at is not None:
            parent_bucket_minutes, _ = _bucket_minutes_for_zoom_level(root_bucket_minutes, actual_zoom_level - 1)
            window_start = _floor_time(focused_at, parent_bucket_minutes)
            window_start = min(max(window_start, root_window_start), root_window_end)
            window_end = min(root_window_end, window_start + timedelta(minutes=parent_bucket_minutes))
            normalized_focus_time = window_start.strftime("%H:%M")
        else:
            current_bucket_minutes = root_bucket_minutes
            actual_zoom_level = 0
            window_start = root_window_start
            window_end = root_window_end

    buckets: list[tuple[str, datetime]] = []
    current = window_start
    while current < window_end and current < day_end:
        buckets.append((current.strftime("%H:%M"), current))
        current += timedelta(minutes=current_bucket_minutes)

    return buckets, timedelta(minutes=current_bucket_minutes), {
        "bucketMinutes": current_bucket_minutes,
        "zoomLevel": actual_zoom_level,
        "canZoomIn": _next_zoom_bucket_minutes(current_bucket_minutes) is not None,
        "isDrilldown": actual_zoom_level > 0,
        "focusTime": normalized_focus_time,
        "windowStart": buckets[0][0] if buckets else None,
        "windowEnd": _format_window_time(window_end, day_end) if buckets else None,
    }


def _build_analytics_samples(
    videos: list[dict[str, Any]],
    events: list[dict[str, Any]],
    pedestrian_tracks: Optional[list[dict[str, Any]]] = None,
) -> tuple[list[dict[str, Any]], dict[str, tuple[datetime, str]]]:
    videos_by_id = {video["id"]: video for video in videos}
    pedestrian_tracks = pedestrian_tracks or []
    event_backed_videos: set[str] = set()
    sample_index: dict[tuple[str, datetime], dict[str, Any]] = {}
    event_first_seen_by_video: dict[str, dict[str, tuple[datetime, str]]] = {}
    track_first_seen_by_video: dict[str, dict[str, tuple[datetime, str]]] = {}
    fallback_track_index_by_video: dict[str, int] = {}

    for event in events:
        if not _is_detection_event(event):
            continue

        video = videos_by_id.get(event.get("videoId") or "")
        if video is None:
            continue

        observed_at = _event_timestamp(event, video)
        if observed_at is None:
            continue

        sample = sample_index.setdefault(
            (video["id"], observed_at),
            {
                "observedAt": observed_at,
                "location": video["location"],
                "visibleTokens": set(),
                "classTokens": {0: set(), 1: set(), 2: set()},
            },
        )

        track_key = _tracked_pedestrian_key(event)
        visibility_token = track_key or f"event:{event.get('id') or observed_at.isoformat()}"
        sample["visibleTokens"].add(visibility_token)

        occlusion_class = _event_occlusion_class(event)
        if occlusion_class is not None:
            sample["classTokens"][occlusion_class].add(visibility_token)

        if track_key is not None:
            video_first_seen = event_first_seen_by_video.setdefault(video["id"], {})
            previous = video_first_seen.get(track_key)
            if previous is None or observed_at <= previous[0]:
                video_first_seen[track_key] = (observed_at, video["location"])

        event_backed_videos.add(video["id"])

    for track in pedestrian_tracks:
        video_id = str(track.get("videoId") or "")
        video = videos_by_id.get(video_id)
        if video is None:
            continue

        fallback_index = fallback_track_index_by_video.get(video_id, 0)
        fallback_track_index_by_video[video_id] = fallback_index + 1
        track_key = _tracked_pedestrian_track_key(track, video_id, fallback_index)
        observed_at = _pedestrian_track_timestamp(track, video)
        if observed_at is None:
            continue

        video_first_seen = track_first_seen_by_video.setdefault(video_id, {})
        previous = video_first_seen.get(track_key)
        location = str(track.get("location") or video.get("location") or "Unknown Location")
        if previous is None or observed_at <= previous[0]:
            video_first_seen[track_key] = (observed_at, location)

    first_seen_by_track: dict[str, tuple[datetime, str]] = {}
    for video in videos:
        video_id = video["id"]
        location = str(video.get("location") or "Unknown Location")
        source_entries = track_first_seen_by_video.get(video_id) or event_first_seen_by_video.get(video_id) or {}

        for track_key, first_seen in source_entries.items():
            previous = first_seen_by_track.get(track_key)
            if previous is None or first_seen[0] <= previous[0]:
                first_seen_by_track[track_key] = first_seen

        source_total = len(source_entries)
        target_total = max(source_total, int(video.get("pedestrianCount") or 0))
        if target_total <= source_total:
            continue

        observed_at = _observation_time(video)
        if observed_at is None:
            continue

        for index in range(target_total - source_total):
            fallback_track_key = f"{video_id}:fallback:{index}"
            first_seen_by_track[fallback_track_key] = (observed_at, location)

    for video in videos:
        if video["id"] in event_backed_videos:
            continue

        observed_at = _observation_time(video)
        pedestrian_count = max(
            int(video.get("pedestrianCount") or 0),
            len(track_first_seen_by_video.get(video["id"], {})),
        )
        if observed_at is None or pedestrian_count <= 0:
            continue

        sample = sample_index.setdefault(
            (video["id"], observed_at),
            {
                "observedAt": observed_at,
                "location": video["location"],
                "visibleTokens": set(),
                "classTokens": {0: set(), 1: set(), 2: set()},
            },
        )

        for index in range(pedestrian_count):
            fallback_track_key = f"{video['id']}:fallback:{index}"
            sample["visibleTokens"].add(fallback_track_key)
            first_seen_by_track[fallback_track_key] = (observed_at, video["location"])

    samples = [
        {
            "observedAt": sample["observedAt"],
            "location": sample["location"],
            "visibleCount": len(sample["visibleTokens"]),
            "classCounts": {class_id: len(tokens) for class_id, tokens in sample["classTokens"].items()},
        }
        for sample in sample_index.values()
    ]
    samples.sort(key=lambda item: item["observedAt"])
    return samples, first_seen_by_track


def _bucket_index(observed_at: datetime, first_bucket: datetime, bucket_seconds: float, bucket_count: int) -> Optional[int]:
    bucket_index = int((observed_at - first_bucket).total_seconds() // bucket_seconds)
    if 0 <= bucket_index < bucket_count:
        return bucket_index
    return None


def _location_unique_totals(
    first_seen_by_track: dict[str, tuple[datetime, str]],
    window_start: datetime,
    window_end: datetime,
) -> list[tuple[str, int]]:
    totals_by_location: dict[str, int] = {}
    for observed_at, location in first_seen_by_track.values():
        if observed_at < window_start or observed_at >= window_end:
            continue
        totals_by_location[location] = totals_by_location.get(location, 0) + 1

    totals = sorted(totals_by_location.items(), key=lambda item: item[1], reverse=True)
    return totals


def _traffic_series_from_samples(
    buckets: list[tuple[str, datetime]],
    bucket_span: timedelta,
    samples: list[dict[str, Any]],
    first_seen_by_track: dict[str, tuple[datetime, str]],
    root_window_start: datetime,
    location_names: list[str],
) -> list[dict[str, Union[float, int, str]]]:
    if not buckets:
        return []

    series: list[dict[str, Union[float, int, str]]] = [
        {
            "id": bucket_start.isoformat(),
            "time": label,
            "cumulativeUniquePedestrians": 0,
            "averageVisiblePedestrians": 0.0,
            **{location: 0 for location in location_names},
        }
        for label, bucket_start in buckets
    ]

    first_bucket = buckets[0][1]
    bucket_seconds = bucket_span.total_seconds()
    final_boundary = buckets[-1][1] + bucket_span
    visible_totals = [0.0 for _ in series]
    sample_counts = [0 for _ in series]
    first_seen_counts = [0 for _ in series]
    first_seen_counts_by_location = {location: [0 for _ in series] for location in location_names}

    for sample in samples:
        observed_at = sample["observedAt"]
        if observed_at < first_bucket or observed_at >= final_boundary:
            continue
        bucket_index = _bucket_index(observed_at, first_bucket, bucket_seconds, len(series))
        if bucket_index is None:
            continue
        visible_totals[bucket_index] += float(sample["visibleCount"])
        sample_counts[bucket_index] += 1

    baseline_unique_total = 0
    baseline_unique_by_location = {location: 0 for location in location_names}
    for observed_at, location in first_seen_by_track.values():
        if observed_at < root_window_start or observed_at >= final_boundary:
            continue
        if observed_at < first_bucket:
            baseline_unique_total += 1
            if location in baseline_unique_by_location:
                baseline_unique_by_location[location] += 1
            continue
        bucket_index = _bucket_index(observed_at, first_bucket, bucket_seconds, len(series))
        if bucket_index is not None:
            first_seen_counts[bucket_index] += 1
            if location in first_seen_counts_by_location:
                first_seen_counts_by_location[location][bucket_index] += 1

    running_total = baseline_unique_total
    running_total_by_location = dict(baseline_unique_by_location)
    for index, point in enumerate(series):
        running_total += first_seen_counts[index]
        point["cumulativeUniquePedestrians"] = running_total
        point["averageVisiblePedestrians"] = round(visible_totals[index] / sample_counts[index], 2) if sample_counts[index] else 0.0
        for location in location_names:
            running_total_by_location[location] += first_seen_counts_by_location[location][index]
            point[location] = running_total_by_location[location]

    return series


def _occlusion_series_from_samples(
    buckets: list[tuple[str, datetime]],
    bucket_span: timedelta,
    samples: list[dict[str, Any]],
) -> list[dict[str, Union[float, str]]]:
    if not buckets:
        return []

    series: list[dict[str, Union[float, str]]] = [
        {"id": bucket_start.isoformat(), "time": label, "Light": 0.0, "Moderate": 0.0, "Heavy": 0.0}
        for label, bucket_start in buckets
    ]

    first_bucket = buckets[0][1]
    bucket_seconds = bucket_span.total_seconds()
    final_boundary = buckets[-1][1] + bucket_span
    sample_counts = [0 for _ in series]
    class_totals = [{0: 0.0, 1: 0.0, 2: 0.0} for _ in series]
    label_by_class = {0: "Light", 1: "Moderate", 2: "Heavy"}

    for sample in samples:
        observed_at = sample["observedAt"]
        if observed_at < first_bucket or observed_at >= final_boundary:
            continue
        bucket_index = _bucket_index(observed_at, first_bucket, bucket_seconds, len(series))
        if bucket_index is None:
            continue
        sample_counts[bucket_index] += 1
        for class_id, count in sample["classCounts"].items():
            class_totals[bucket_index][class_id] += float(count)

    for index, point in enumerate(series):
        for class_id, label in label_by_class.items():
            point[label] = round(class_totals[index][class_id] / sample_counts[index], 2) if sample_counts[index] else 0.0

    return series


def _traffic_observations(videos: list[dict[str, Any]], events: list[dict[str, Any]]) -> list[tuple[str, datetime, int]]:
    videos_by_id = {video["id"]: video for video in videos}
    observations: list[tuple[str, datetime, int]] = []
    event_backed_videos: set[str] = set()

    for event in events:
        video = videos_by_id.get(event.get("videoId") or "")
        if video is None:
            continue
        observed_at = _event_timestamp(event, video)
        if observed_at is None:
            continue
        observations.append((video["location"], observed_at, 1))
        event_backed_videos.add(video["id"])

    for video in videos:
        if video["id"] in event_backed_videos:
            continue
        count = int(video.get("pedestrianCount") or 0)
        observed_at = _observation_time(video)
        if count <= 0 or observed_at is None:
            continue
        observations.append((video["location"], observed_at, count))

    return observations


def _bucket_counts(
    buckets: list[tuple[str, datetime]],
    bucket_span: timedelta,
    observations: list[tuple[str, datetime, int]],
    location_names: list[str],
) -> list[dict[str, Union[int, str]]]:
    if not buckets or not location_names:
        return []

    series: list[dict[str, Union[int, str]]] = [
        {"time": label, **{location: 0 for location in location_names}}
        for label, _ in buckets
    ]

    first_bucket = buckets[0][1]
    bucket_seconds = bucket_span.total_seconds()
    final_boundary = buckets[-1][1] + bucket_span

    for location_name, observed_at, count in observations:
        if location_name not in location_names:
            continue
        if observed_at < first_bucket or observed_at >= final_boundary:
            continue
        bucket_index = int((observed_at - first_bucket).total_seconds() // bucket_seconds)
        if 0 <= bucket_index < len(series):
            series[bucket_index][location_name] = int(series[bucket_index][location_name]) + count

    return series


def _traffic_location_totals(traffic_series: list[dict[str, Union[int, str]]]) -> list[tuple[str, int]]:
    if not traffic_series:
        return []

    location_names = [key for key in traffic_series[0].keys() if key != "time"]
    totals = [
        (location, sum(int(point.get(location, 0)) for point in traffic_series))
        for location in location_names
    ]
    totals.sort(key=lambda item: item[1], reverse=True)
    return totals


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0

    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])

    rank = (percentile / 100.0) * (len(ordered) - 1)
    lower_index = int(rank)
    upper_index = min(lower_index + 1, len(ordered) - 1)
    fraction = rank - lower_index
    lower_value = float(ordered[lower_index])
    upper_value = float(ordered[upper_index])
    return lower_value + ((upper_value - lower_value) * fraction)


def _ptsi_congestion_score(visible_count: int, walkable_area_m2: Optional[float]) -> float:
    if visible_count <= 0:
        return 0.0

    if walkable_area_m2 is None:
        return 0.0

    space_per_pedestrian = walkable_area_m2 / float(visible_count)
    if space_per_pedestrian > 5.6:
        return 0.0
    if space_per_pedestrian >= 3.7:
        return 0.2
    if space_per_pedestrian >= 2.2:
        return 0.4
    if space_per_pedestrian >= 1.4:
        return 0.6
    if space_per_pedestrian >= 0.75:
        return 0.8
    return 1.0


def _ptsi_los_from_space_per_pedestrian(space_per_pedestrian: Optional[float]) -> Optional[str]:
    if space_per_pedestrian is None or space_per_pedestrian <= 0:
        return None
    if space_per_pedestrian > 5.6:
        return "A"
    if space_per_pedestrian >= 3.7:
        return "B"
    if space_per_pedestrian >= 2.2:
        return "C"
    if space_per_pedestrian >= 1.4:
        return "D"
    if space_per_pedestrian >= 0.75:
        return "E"
    return "F"


def _ptsi_los_description(los: Optional[str]) -> Optional[str]:
    if los is None:
        return None
    return PTSI_LOS_DESCRIPTIONS.get(los)


def _ptsi_los_rank(los: Optional[str]) -> Optional[int]:
    if los is None:
        return None
    return PTSI_LOS_RANKS.get(los)


def _ptsi_los_from_rank(rank: Optional[int]) -> Optional[str]:
    if rank is None:
        return None
    clamped_rank = min(max(int(rank), 0), max(PTSI_LOS_BY_RANK))
    return PTSI_LOS_BY_RANK.get(clamped_rank)


def _ptsi_los_state(los: Optional[str], has_footage: bool, has_occlusion_data: bool) -> str:
    if not has_footage:
        return "no-footage"
    if not has_occlusion_data or los is None:
        return "no-data"
    return PTSI_LOS_STATE_MAP.get(los, "no-data")


def _ptsi_los_from_score(score: Optional[float]) -> Optional[str]:
    if score is None:
        return None
    if score < 15:
        return "A"
    if score < 33:
        return "B"
    if score < 50:
        return "C"
    if score < 66:
        return "D"
    if score < 85:
        return "E"
    return "F"


def _ptsi_space_per_pedestrian(visible_count: int, walkable_area_m2: Optional[float]) -> Optional[float]:
    if visible_count <= 0 or walkable_area_m2 is None or walkable_area_m2 <= 0:
        return None
    return walkable_area_m2 / float(visible_count)


def _ptsi_score_breakdown(visible_count: int, location: dict[str, Any], occlusion_value: float) -> dict[str, Any]:
    mode = _location_ptsi_mode(location)
    walkable_area_m2 = _location_walkable_area_m2(location)
    roi_area_ratio = _location_roi_area_ratio(location)
    capacity_proxy: Optional[float] = None
    space_per_pedestrian: Optional[float] = None
    los: Optional[str] = None

    if visible_count <= 0:
        congestion_score = 0.0
        score = 0.0
    elif mode == "strict-fhwa":
        space_per_pedestrian = _ptsi_space_per_pedestrian(visible_count, walkable_area_m2)
        los = _ptsi_los_from_space_per_pedestrian(space_per_pedestrian)
        congestion_score = _ptsi_congestion_score(visible_count, walkable_area_m2)
        score = 100.0 * ((PTSI_CONGESTION_WEIGHT * congestion_score) + (PTSI_OCCLUSION_WEIGHT * occlusion_value))
    else:
        capacity_proxy = max(1.0, math.sqrt(roi_area_ratio) * PTSI_ROI_TESTING_CAPACITY_PER_FULL_FRAME)
        congestion_score = min(1.0, visible_count / capacity_proxy)
        score = 100.0 * ((PTSI_CONGESTION_WEIGHT * congestion_score) + (PTSI_OCCLUSION_WEIGHT * occlusion_value))
        los = _ptsi_los_from_score(score)

    return {
        "mode": mode,
        "walkableAreaM2": walkable_area_m2,
        "roiAreaRatio": round(roi_area_ratio, 6),
        "capacityProxy": round(capacity_proxy, 3) if capacity_proxy is not None else None,
        "congestionScore": round(congestion_score, 4),
        "occlusionValue": round(occlusion_value, 4),
        "spacePerPedestrian": round(space_per_pedestrian, 3) if space_per_pedestrian is not None else None,
        "los": los,
        "losDescription": _ptsi_los_description(los),
        "score": round(min(max(score, 0.0), 100.0), 2),
    }


def _ptsi_score(visible_count: int, location: dict[str, Any], occlusion_value: float) -> float:
    return float(_ptsi_score_breakdown(visible_count, location, occlusion_value)["score"])


def _severity_state(score: Optional[float], has_footage: bool, has_occlusion_data: bool) -> str:
    if not has_footage:
        return "no-footage"
    if not has_occlusion_data or score is None:
        return "no-data"
    if score < 33:
        return "clear"
    if score < 66:
        return "moderate"
    return "severe"


def dashboard_summary(date: Optional[str] = None) -> dict[str, Any]:
    state, _, videos, events = _filtered_dashboard_records(date)
    pedestrian_tracks = _filtered_pedestrian_tracks(state, videos)
    _samples, first_seen_by_track = _build_analytics_samples(videos, events, pedestrian_tracks)
    return {
        "totalUniquePedestrians": len(first_seen_by_track),
        "averageFps": 29.7 if videos else 0.0,
        "totalHeavyOcclusions": sum(1 for event in events if _event_occlusion_class(event) == 2),
        "monitoredLocations": len(_active_location_ids(videos)),
    }


def dashboard_traffic(
    date: Optional[str] = None,
    time_range: str = "whole-day",
    focus_time: Optional[str] = None,
    zoom_level: int = 0,
) -> dict[str, Any]:
    state, resolved_date, videos, events = _filtered_dashboard_records(date)
    pedestrian_tracks = _filtered_pedestrian_tracks(state, videos)
    samples, first_seen_by_track = _build_analytics_samples(videos, events, pedestrian_tracks)
    observation_times = [sample["observedAt"] for sample in samples]
    if not observation_times:
        observation_times = [timestamp for timestamp in (_observation_time(video) for video in videos) if timestamp is not None]

    root_window_start, _root_window_end, _root_bucket_minutes = _resolve_root_window(resolved_date, time_range, observation_times)
    buckets, bucket_span, bucket_meta = _build_bucket_plan(resolved_date, time_range, observation_times, focus_time, zoom_level)
    active_location_ids = _active_location_ids(videos)
    active_location_names = [location["name"] for location in state["locations"] if location["id"] in active_location_ids]
    series = _traffic_series_from_samples(
        buckets,
        bucket_span,
        samples,
        first_seen_by_track,
        root_window_start,
        active_location_names,
    )

    if buckets:
        window_start = buckets[0][1]
        window_end = buckets[-1][1] + bucket_span
    else:
        window_start = root_window_start
        window_end = root_window_start

    active_location_name_set = set(active_location_names)
    location_totals = [
        {"location": location, "totalPedestrians": total}
        for location, total in _location_unique_totals(first_seen_by_track, window_start, window_end)
        if location in active_location_name_set
    ]

    return {
        "timeRange": time_range,
        "series": series,
        **bucket_meta,
        "locationTotals": location_totals,
    }


def dashboard_occlusion_trends(
    date: Optional[str] = None,
    time_range: str = "whole-day",
    focus_time: Optional[str] = None,
    zoom_level: int = 0,
) -> dict[str, Any]:
    state, resolved_date, videos, events = _filtered_dashboard_records(date)
    pedestrian_tracks = _filtered_pedestrian_tracks(state, videos)
    samples, _first_seen_by_track = _build_analytics_samples(videos, events, pedestrian_tracks)
    observation_times = [sample["observedAt"] for sample in samples]
    if not observation_times:
        observation_times = [timestamp for timestamp in (_observation_time(video) for video in videos) if timestamp is not None]

    buckets, bucket_span, bucket_meta = _build_bucket_plan(resolved_date, time_range, observation_times, focus_time, zoom_level)
    if not buckets:
        return {"timeRange": time_range, "series": [], **bucket_meta}

    series = _occlusion_series_from_samples(buckets, bucket_span, samples)
    return {"timeRange": time_range, "series": series, **bucket_meta}


def dashboard_occlusion(date: Optional[str] = None, time_range: str = "whole-day") -> dict[str, Any]:
    state, resolved_date, videos, _events = _filtered_dashboard_records(date)
    videos_by_id = {video["id"]: video for video in videos}
    locations_by_id = {location["id"]: location for location in state["locations"]}
    pedestrian_tracks = _filtered_pedestrian_tracks(state, videos)

    sample_observations: list[dict[str, Any]] = []
    observation_times: list[datetime] = []
    for fallback_index, track in enumerate(pedestrian_tracks):
        video_id = str(track.get("videoId") or "")
        video = videos_by_id.get(video_id)
        if video is None:
            continue

        location = locations_by_id.get(video["locationId"])
        if location is None:
            continue

        observed_at = _observation_time(video) or _pedestrian_track_timestamp(track, video)
        if observed_at is None:
            continue

        track_key = _tracked_pedestrian_track_key(track, video_id, fallback_index)
        for offset_second, point, occlusion_class in _normalized_trajectory_samples(track):
            if not _point_in_location_roi(point, location):
                continue

            sample_time = (observed_at + timedelta(seconds=offset_second)).replace(microsecond=0)
            observation_times.append(sample_time)
            sample_observations.append(
                {
                    "locationId": video["locationId"],
                    "trackKey": track_key,
                    "observedAt": sample_time,
                    "occlusionClass": occlusion_class,
                }
            )

    if not observation_times:
        observation_times = [timestamp for timestamp in (_observation_time(video) for video in videos) if timestamp is not None]

    buckets, bucket_span, _ = _build_bucket_plan(resolved_date, time_range, observation_times)
    window_start = buckets[0][1] if buckets else datetime.strptime(resolved_date, "%Y-%m-%d")
    window_end = (buckets[-1][1] + bucket_span) if buckets else (window_start + timedelta(days=1))

    second_metrics: dict[str, dict[datetime, dict[str, dict[str, Optional[int]]]]] = {}
    for sample in sample_observations:
        observed_at = sample["observedAt"]
        if observed_at < window_start or observed_at >= window_end:
            continue

        location_seconds = second_metrics.setdefault(sample["locationId"], {})
        second_entry = location_seconds.setdefault(observed_at, {"tracks": {}})
        track_occlusions = second_entry["tracks"]
        track_key = str(sample["trackKey"])
        incoming_occlusion = sample.get("occlusionClass")
        if track_key not in track_occlusions:
            track_occlusions[track_key] = incoming_occlusion
            continue

        existing_occlusion = track_occlusions[track_key]
        if incoming_occlusion is not None and (existing_occlusion is None or int(incoming_occlusion) > int(existing_occlusion)):
            track_occlusions[track_key] = incoming_occlusion

    available_hours: set[str] = set()
    active_location_ids = _active_location_ids(videos)
    locations_payload = []
    for location in state["locations"]:
        mode = _location_ptsi_mode(location)
        walkable_area_m2 = _location_walkable_area_m2(location)
        roi_area_ratio = _location_roi_area_ratio(location)
        capacity_proxy = _location_capacity_proxy(location) if mode == "roi-testing" else None
        has_footage = location["id"] in active_location_ids

        _ptsi_debug_log(
            "location_config",
            locationId=location["id"],
            locationName=location["name"],
            mode=mode,
            walkableAreaM2=walkable_area_m2,
            roiAreaRatio=round(roi_area_ratio, 6),
            capacityProxy=round(capacity_proxy, 3) if capacity_proxy is not None else None,
            hasFootage=has_footage,
        )

        hourly_rollups: dict[str, dict[str, Any]] = {}
        total_visible = 0
        total_light = 0
        total_moderate = 0
        total_heavy = 0
        total_sample_seconds = 0
        unique_track_keys: set[str] = set()

        for second_at, second_entry in sorted(second_metrics.get(location["id"], {}).items()):
            track_occlusions = second_entry.get("tracks") or {}
            visible_count = len(track_occlusions)
            if visible_count <= 0:
                continue

            light_count = sum(1 for value in track_occlusions.values() if value == 0)
            moderate_count = sum(1 for value in track_occlusions.values() if value == 1)
            heavy_count = sum(1 for value in track_occlusions.values() if value == 2)
            occlusion_value = (
                (light_count * PTSI_OCCLUSION_WEIGHTS[0])
                + (moderate_count * PTSI_OCCLUSION_WEIGHTS[1])
                + (heavy_count * PTSI_OCCLUSION_WEIGHTS[2])
            ) / (3.0 * visible_count)

            breakdown = _ptsi_score_breakdown(visible_count, location, occlusion_value)
            second_score = float(breakdown["score"])
            hour_label = second_at.strftime("%H:00")

            _ptsi_debug_log(
                "second_score",
                locationId=location["id"],
                locationName=location["name"],
                observedAt=second_at,
                hour=hour_label,
                visibleCount=visible_count,
                lightCount=light_count,
                moderateCount=moderate_count,
                heavyCount=heavy_count,
                occlusionValue=breakdown["occlusionValue"],
                congestionScore=breakdown["congestionScore"],
                capacityProxy=breakdown["capacityProxy"],
                roiAreaRatio=breakdown["roiAreaRatio"],
                walkableAreaM2=breakdown["walkableAreaM2"],
                spacePerPedestrian=breakdown["spacePerPedestrian"],
                los=breakdown["los"],
                score=second_score,
            )

            hour_rollup = hourly_rollups.setdefault(
                hour_label,
                {
                    "scores": [],
                    "visibleTotal": 0,
                    "sampleSeconds": 0,
                    "light": 0,
                    "moderate": 0,
                    "heavy": 0,
                    "losRanks": [],
                    "trackKeys": set(),
                },
            )
            hour_rollup["scores"].append(second_score)
            hour_rollup["visibleTotal"] += visible_count
            hour_rollup["sampleSeconds"] += 1
            hour_rollup["light"] += light_count
            hour_rollup["moderate"] += moderate_count
            hour_rollup["heavy"] += heavy_count
            los_rank = _ptsi_los_rank(breakdown["los"])
            if los_rank is not None:
                hour_rollup["losRanks"].append(los_rank)
            hour_rollup["trackKeys"].update(track_occlusions.keys())

            total_visible += visible_count
            total_light += light_count
            total_moderate += moderate_count
            total_heavy += heavy_count
            total_sample_seconds += 1
            unique_track_keys.update(track_occlusions.keys())

        hourly_scores = []
        for hour_label, rollup in sorted(hourly_rollups.items()):
            available_hours.add(hour_label)
            visible_total = int(rollup["visibleTotal"])
            sample_seconds = int(rollup["sampleSeconds"])
            hourly_score_value = round(_percentile(list(rollup["scores"]), 90), 2)
            los_rank_values = [int(rank) for rank in rollup["losRanks"]]
            hourly_los = _ptsi_los_from_score(hourly_score_value) if mode == "roi-testing" else _ptsi_los_from_rank(max(los_rank_values) if los_rank_values else None)
            hourly_score = {
                "hour": hour_label,
                "score": hourly_score_value,
                "mode": mode,
                "averagePedestrians": round(visible_total / sample_seconds, 2) if sample_seconds else 0.0,
                "uniquePedestrians": len(rollup["trackKeys"]),
                "occlusionMix": _ptsi_occlusion_mix(
                    int(rollup["light"]),
                    int(rollup["moderate"]),
                    int(rollup["heavy"]),
                    visible_total,
                ),
                "los": hourly_los,
                "losDescription": _ptsi_los_description(hourly_los),
            }
            hourly_scores.append(hourly_score)

            _ptsi_debug_log(
                "hour_rollup",
                locationId=location["id"],
                locationName=location["name"],
                hour=hour_label,
                sampleSeconds=sample_seconds,
                visibleTotal=visible_total,
                averagePedestrians=hourly_score["averagePedestrians"],
                uniquePedestrians=hourly_score["uniquePedestrians"],
                occlusionMix=hourly_score["occlusionMix"],
                los=hourly_score["los"],
                p90Score=hourly_score["score"],
            )

        has_occlusion_data = bool(hourly_scores)
        ranked_hours = sorted(hourly_scores, key=lambda score: (float(score["score"]), str(score["hour"])))
        peak_hour = ranked_hours[-1] if ranked_hours else None
        off_peak_hour = ranked_hours[0] if len(ranked_hours) > 1 else None
        overall_score = float(peak_hour["score"]) if peak_hour else None
        selected_mode = peak_hour["mode"] if peak_hour else mode
        selected_average_pedestrians = peak_hour["averagePedestrians"] if peak_hour else None
        selected_unique_pedestrians = peak_hour["uniquePedestrians"] if peak_hour else None
        selected_occlusion_mix = peak_hour["occlusionMix"] if peak_hour else None
        selected_los = peak_hour["los"] if peak_hour else None
        selected_los_description = peak_hour["losDescription"] if peak_hour else None
        state_label = _ptsi_los_state(selected_los, has_footage, has_occlusion_data)

        _ptsi_debug_log(
            "location_summary",
            locationId=location["id"],
            locationName=location["name"],
            availableHours=[score["hour"] for score in hourly_scores],
            peakHour=peak_hour["hour"] if peak_hour else None,
            peakHourScore=float(peak_hour["score"]) if peak_hour else None,
            offPeakHour=off_peak_hour["hour"] if off_peak_hour else None,
            offPeakHourScore=float(off_peak_hour["score"]) if off_peak_hour else None,
            score=overall_score,
            mode=selected_mode,
            averagePedestrians=selected_average_pedestrians,
            uniquePedestrians=selected_unique_pedestrians,
            occlusionMix=selected_occlusion_mix,
            los=selected_los,
            totalVisible=total_visible,
            totalSampleSeconds=total_sample_seconds,
            hasPTSIData=has_occlusion_data,
            state=state_label,
        )

        locations_payload.append(
            {
                "id": location["id"],
                "name": location["name"],
                "latitude": location["latitude"],
                "longitude": location["longitude"],
                "hasFootage": has_footage,
                "hasPTSIData": has_occlusion_data,
                "score": overall_score,
                "state": state_label,
                "mode": selected_mode,
                "averagePedestrians": selected_average_pedestrians,
                "uniquePedestrians": selected_unique_pedestrians,
                "occlusionMix": selected_occlusion_mix,
                "los": selected_los,
                "losDescription": selected_los_description,
                "peakHour": peak_hour["hour"] if peak_hour else None,
                "peakHourScore": float(peak_hour["score"]) if peak_hour else None,
                "offPeakHour": off_peak_hour["hour"] if off_peak_hour else None,
                "offPeakHourScore": float(off_peak_hour["score"]) if off_peak_hour else None,
                "hourlyScores": hourly_scores,
            }
        )

    return {
        "timeRange": time_range,
        "availableHours": sorted(available_hours),
        "locations": locations_payload,
    }


def ai_synthesis(date: str, time_range: str) -> dict[str, Any]:
    summary = dashboard_summary(date)
    traffic_response = dashboard_traffic(date, time_range)
    traffic_series = traffic_response["series"]
    location_totals = traffic_response.get("locationTotals", [])
    peak_point = None
    if traffic_series:
        peak_point = max(
            traffic_series,
            key=lambda point: float(point.get("averageVisiblePedestrians", 0.0)),
        )

    occlusion_response = dashboard_occlusion(date, time_range)
    hotspots = [location["name"] for location in occlusion_response["locations"] if location["state"] in {"moderate", "severe"}]

    if not traffic_series:
        return {
            "date": date,
            "timeRange": time_range,
            "sections": [
                {
                    "title": "Executive Overview",
                    "body": "No footage is available for the selected date and time range yet, so the dashboard is waiting for real observations before producing analytics.",
                    "badges": [{"label": "Locations", "value": str(summary["monitoredLocations"]), "tone": "blue"}],
                },
                {
                    "title": "Peak Traffic Events",
                    "body": "Upload or process footage to populate the pedestrian timeline and identify the busiest windows.",
                    "badges": [],
                },
                {
                    "title": "Spatial Breakdown",
                    "body": "The occlusion map will switch from neutral markers to OWDI severity states once occlusion-class detections are present in the selected footage.",
                    "badges": [{"label": "Heavy Occlusions", "value": str(summary["totalHeavyOcclusions"]), "tone": "red"}],
                },
            ],
        }

    top_location = str(location_totals[0].get("location")) if location_totals else "No location"
    peak_window = str(peak_point.get("time")) if peak_point else "N/A"
    peak_visible_average = float(peak_point.get("averageVisiblePedestrians", 0.0)) if peak_point else 0.0
    cumulative_total = int(traffic_series[-1].get("cumulativeUniquePedestrians", 0)) if traffic_series else 0
    return {
        "date": date,
        "timeRange": time_range,
        "sections": [
            {
                "title": "Executive Overview",
                "body": f"Across {summary['monitoredLocations']} active locations, the dashboard tracked {cumulative_total} unique pedestrians and processed footage at an average of {summary['averageFps']} FPS for the selected date.",
                "badges": [{"label": "Unique", "value": str(cumulative_total), "tone": "green"}, {"label": "FPS", "value": str(summary["averageFps"]), "tone": "purple"}],
            },
            {
                "title": "Peak Traffic Events",
                "body": f"{top_location} contributes the strongest unique footfall in this selection, while the busiest bucket averaged {peak_visible_average:.2f} visible pedestrians around {peak_window}.",
                "badges": [{"label": "Peak", "value": peak_window, "tone": "orange"}, {"label": "Location", "value": top_location, "tone": "blue"}],
            },
            {
                "title": "Spatial Breakdown",
                "body": (
                    f"Occlusion hotspots currently include {', '.join(hotspots)} based on the sustained peak density signal."
                    if hotspots
                    else "No occlusion hotspots have been detected yet in the selected footage."
                ),
                "badges": [{"label": "Occlusions", "value": str(summary["totalHeavyOcclusions"]), "tone": "red"}],
            },
        ],
    }


def export_dashboard_report(date: str, time_range: str) -> Path:
    ensure_storage_layout()

    summary = dashboard_summary(date)
    traffic_response = dashboard_traffic(date, time_range)
    traffic = traffic_response["series"]
    synthesis = ai_synthesis(date, time_range)
    model = get_model_info()

    location_totals = traffic_response.get("locationTotals", [])

    generated_at = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    timestamp_slug = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    filename = f"dashboard-report-{slugify(date)}-{slugify(time_range)}-{timestamp_slug}.md"
    target = EXPORTS_DIR / filename

    lines = [
        "# ALIVE Engine Dashboard Report",
        "",
        f"- Date: {date}",
        f"- Time Range: {format_time_range_label(time_range)}",
        f"- Generated At: {generated_at}",
        f"- Active Model: {model.get('currentModel') or 'No model selected'}",
        "",
        "## Summary",
        "",
        f"- Total Pedestrians: {summary['totalUniquePedestrians']}",
        f"- Average FPS: {summary['averageFps']}",
        f"- Heavy Occlusions: {summary['totalHeavyOcclusions']}",
        f"- Monitored Locations: {summary['monitoredLocations']}",
        "",
        "## Unique Pedestrian Totals By Location",
        "",
    ]

    if location_totals:
        lines.extend([
            f"- {entry['location']}: {entry['totalPedestrians']}"
            for entry in location_totals
        ])
    else:
        lines.append("- No traffic data available for the selected time range.")

    lines.extend(
        [
            "",
            "## AI Synthesis",
            "",
        ]
    )

    for section in synthesis["sections"]:
        lines.extend([f"### {section['title']}", "", section["body"], ""])
        if section.get("badges"):
            lines.append("Badges:")
            lines.extend([f"- {badge['label']}: {badge['value']} ({badge['tone']})" for badge in section["badges"]])
            lines.append("")

    target.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
    return target


def _search_terms(query: str) -> list[str]:
    normalized_query = query.lower()
    terms: list[str] = []
    seen: set[str] = set()

    def _add_term(value: str) -> None:
        for term in re.split(r"[^a-z0-9]+", value.lower()):
            if not term or term in SEARCH_STOPWORDS or term in seen:
                continue
            seen.add(term)
            terms.append(term)

    for phrase, synonyms in SEARCH_PHRASE_SYNONYMS.items():
        if phrase in normalized_query:
            for synonym in synonyms:
                _add_term(synonym)

    for raw_term in re.split(r"[^a-z0-9]+", normalized_query):
        if not raw_term or raw_term in SEARCH_STOPWORDS:
            continue
        _add_term(raw_term)
        for synonym in SEARCH_TERM_SYNONYMS.get(raw_term, ()):
            _add_term(synonym)

    return terms


def _normalized_search_text(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def _location_aliases(location: dict[str, Any]) -> list[str]:
    aliases: set[str] = set()
    for raw_value in (
        location.get("name"),
        location.get("address"),
        location.get("description"),
        str(location.get("id") or "").replace("-", " "),
    ):
        normalized = _normalized_search_text(str(raw_value or ""))
        if normalized:
            aliases.add(normalized)
        for part in re.split(r"[·,;/()]+", str(raw_value or "")):
            normalized_part = _normalized_search_text(part)
            if len(normalized_part) >= 4:
                aliases.add(normalized_part)
    return sorted(aliases, key=len, reverse=True)


def _match_query_location(query: str, locations: list[dict[str, Any]]) -> tuple[Optional[dict[str, Any]], Optional[str]]:
    normalized_query = _normalized_search_text(query)
    if not normalized_query:
        return None, None

    best_location: Optional[dict[str, Any]] = None
    best_alias: Optional[str] = None
    ambiguous = False
    for location in locations:
        for alias in _location_aliases(location):
            if len(alias) < 4 or alias not in normalized_query:
                continue
            if best_alias is None or len(alias) > len(best_alias):
                best_location = location
                best_alias = alias
                ambiguous = False
            elif len(alias) == len(best_alias) and best_location and location.get("id") != best_location.get("id"):
                ambiguous = True

    if ambiguous:
        return None, None
    return best_location, best_alias


def _strip_location_alias(query: str, alias: Optional[str]) -> str:
    normalized_query = _normalized_search_text(query)
    if not alias:
        return normalized_query
    return re.sub(rf"\b{re.escape(alias)}\b", " ", normalized_query).strip()


def _location_context_payload(locations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "id": str(location.get("id") or ""),
            "name": str(location.get("name") or ""),
            "address": str(location.get("address") or ""),
            "description": str(location.get("description") or ""),
        }
        for location in locations
    ]


def _coerce_region_color_requirements(raw_requirements: Any) -> dict[str, set[str]]:
    normalized: dict[str, set[str]] = {}
    if not isinstance(raw_requirements, list):
        return normalized

    valid_regions = set(SEARCH_REGION_ALIASES)
    valid_colors = set(SEARCH_COLOR_FAMILY_MATCHES)
    for item in raw_requirements:
        if not isinstance(item, dict):
            continue
        region_name = str(item.get("region") or "").strip().lower()
        if region_name not in valid_regions:
            continue
        colors = item.get("colors")
        if not isinstance(colors, list):
            continue
        for color_name in colors:
            normalized_color = str(color_name or "").strip().lower()
            if normalized_color not in valid_colors:
                continue
            normalized.setdefault(region_name, set()).update(SEARCH_COLOR_FAMILY_MATCHES.get(normalized_color, {normalized_color}))
    return normalized


def _merge_region_requirements(local_requirements: dict[str, set[str]], ai_requirements: dict[str, set[str]]) -> dict[str, set[str]]:
    merged = {region: set(colors) for region, colors in local_requirements.items()}
    for region, colors in ai_requirements.items():
        if region in merged:
            continue
        merged[region] = set(colors)
    return merged


def _unique_terms(values: list[str]) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    for value in values:
        term = str(value or "").strip().lower()
        if not term or term in SEARCH_STOPWORDS or term in seen:
            continue
        seen.add(term)
        normalized.append(term)
    return normalized


def _build_search_query_plan(
    query: str,
    locations: list[dict[str, Any]],
    query_parser: Optional[Callable[[str, list[dict[str, Any]]], dict[str, Any]]],
) -> dict[str, Any]:
    stripped_query = query.strip()
    matched_location, matched_alias = _match_query_location(stripped_query, locations)
    appearance_query = _strip_location_alias(stripped_query, matched_alias)

    local_terms = _search_terms(appearance_query or stripped_query)
    local_region_requirements = _query_region_color_requirements(appearance_query or stripped_query)

    parsed_query: dict[str, Any] = {}
    if query_parser is not None and stripped_query:
        try:
            parsed_query = query_parser(stripped_query, _location_context_payload(locations)) or {}
        except Exception:
            parsed_query = {}

    parsed_location_id = str(parsed_query.get("locationId") or "")
    parsed_location = next((location for location in locations if str(location.get("id") or "") == parsed_location_id), None)
    resolved_location = parsed_location or matched_location

    ai_terms = [str(term or "") for term in parsed_query.get("appearanceTerms") or [] if str(term or "").strip()]
    soft_terms = _unique_terms(
        [
            *[str(term or "") for term in parsed_query.get("softTerms") or []],
            *[str(term or "") for term in parsed_query.get("unsupportedTerms") or []],
        ]
    )
    region_requirements = _merge_region_requirements(local_region_requirements, _coerce_region_color_requirements(parsed_query.get("regionColorRequirements")))

    return {
        "locationId": str(resolved_location.get("id") or "") if resolved_location else None,
        "locationName": str((resolved_location or {}).get("name") or parsed_query.get("locationName") or "") or None,
        "locationAlias": matched_alias,
        "hardTerms": _unique_terms([*local_terms, *ai_terms]),
        "softTerms": soft_terms,
        "regionColorRequirements": region_requirements,
        "summary": str(parsed_query.get("summary") or "").strip(),
    }


def _query_tokens(query: str) -> list[str]:
    return [term for term in re.split(r"[^a-z0-9]+", query.lower()) if term]


def _find_phrase_mentions(tokens: list[str], aliases_by_name: dict[str, tuple[str, ...]]) -> list[tuple[int, int, str]]:
    patterns: list[tuple[int, tuple[str, ...], str]] = []
    for canonical_name, aliases in aliases_by_name.items():
        for alias in aliases:
            alias_tokens = tuple(part for part in re.split(r"[^a-z0-9]+", alias.lower()) if part)
            if alias_tokens:
                patterns.append((len(alias_tokens), alias_tokens, canonical_name))

    patterns.sort(key=lambda item: item[0], reverse=True)
    occupied = [False] * len(tokens)
    matches: list[tuple[int, int, str]] = []
    for _, alias_tokens, canonical_name in patterns:
        length = len(alias_tokens)
        for start in range(0, len(tokens) - length + 1):
            end = start + length
            if any(occupied[start:end]):
                continue
            if tuple(tokens[start:end]) != alias_tokens:
                continue
            matches.append((start, end, canonical_name))
            for index in range(start, end):
                occupied[index] = True

    matches.sort(key=lambda item: (item[0], item[1], item[2]))
    return matches


def _adjacent_region_colors(
    tokens: list[str],
    color_mentions: list[tuple[int, int, str]],
    region_start: int,
    region_end: int,
) -> list[str]:
    adjacent_colors: list[str] = []

    index = region_start
    while index > 0:
        color_match = next((match for match in color_mentions if match[1] == index), None)
        if color_match is not None:
            if color_match[2] not in adjacent_colors:
                adjacent_colors.append(color_match[2])
            index = color_match[0]
            continue
        preceding_token = tokens[index - 1]
        if preceding_token in QUERY_COLOR_JOINERS or preceding_token in QUERY_COLOR_MODIFIERS:
            index -= 1
            continue
        break

    index = region_end
    while index < len(tokens):
        color_match = next((match for match in color_mentions if match[0] == index), None)
        if color_match is not None:
            if color_match[2] not in adjacent_colors:
                adjacent_colors.append(color_match[2])
            index = color_match[1]
            continue
        following_token = tokens[index]
        if following_token in QUERY_COLOR_JOINERS or following_token in QUERY_COLOR_MODIFIERS:
            index += 1
            continue
        break

    return adjacent_colors


def _query_region_color_requirements(query: str) -> dict[str, set[str]]:
    tokens = _query_tokens(query)
    if not tokens:
        return {}

    color_mentions = _find_phrase_mentions(tokens, SEARCH_COLOR_ALIASES)
    region_mentions = _find_phrase_mentions(tokens, SEARCH_REGION_ALIASES)
    if not color_mentions or not region_mentions:
        return {}

    requirements: dict[str, set[str]] = {}
    for region_start, region_end, region_name in region_mentions:
        adjacent_colors = _adjacent_region_colors(tokens, color_mentions, region_start, region_end)
        if adjacent_colors:
            accepted_colors = requirements.setdefault(region_name, set())
            for color_name in adjacent_colors:
                accepted_colors.update(SEARCH_COLOR_FAMILY_MATCHES.get(color_name, {color_name}))
            continue

        nearest_gap: Optional[int] = None
        nearest_colors: list[str] = []
        for color_start, color_end, color_name in color_mentions:
            if color_end <= region_start:
                gap = region_start - color_end
            elif region_end <= color_start:
                gap = color_start - region_end
            else:
                gap = 0

            if nearest_gap is None or gap < nearest_gap:
                nearest_gap = gap
                nearest_colors = [color_name]
            elif gap == nearest_gap and color_name not in nearest_colors:
                nearest_colors.append(color_name)

        if nearest_gap is None or nearest_gap > 3:
            continue

        accepted_colors = requirements.setdefault(region_name, set())
        for color_name in nearest_colors:
            accepted_colors.update(SEARCH_COLOR_FAMILY_MATCHES.get(color_name, {color_name}))

    return requirements


def _track_region_colors(track: dict[str, Any]) -> dict[str, set[str]]:
    region_colors: dict[str, set[str]] = {}
    region_sources = [*(track.get("appearanceHints") or [])]
    if track.get("appearanceSummary"):
        region_sources.append(str(track.get("appearanceSummary") or ""))

    for source in region_sources:
        for match in re.finditer(r"(head region|upper clothing|lower clothing) appears ([a-z]+)", str(source).strip().lower()):
            region_name, color_name = match.groups()
            region_colors.setdefault(region_name, set()).add(color_name)
    return region_colors


def _camera_ambiguous_region_match(region_colors: set[str], accepted_colors: set[str]) -> bool:
    for accepted_color in accepted_colors:
        ambiguous_colors = SEARCH_CAMERA_AMBIGUOUS_COLOR_MATCHES.get(accepted_color) or set()
        if region_colors.intersection(ambiguous_colors):
            return True
    return False


def _occlusion_label(occlusion_class: Any) -> str:
    try:
        normalized = int(occlusion_class)
    except (TypeError, ValueError):
        return ""

    return {0: "light occlusion", 1: "moderate occlusion", 2: "heavy occlusion"}.get(normalized, "")


def _track_search_text(track: dict[str, Any]) -> str:
    parts: list[str] = [
        str(track.get("location") or ""),
        str(track.get("appearanceSummary") or ""),
        _occlusion_label(track.get("occlusionClass")),
    ]
    parts.extend(str(item) for item in (track.get("appearanceHints") or []) if item)
    parts.extend(str(item) for item in (track.get("visualLabels") or []) if item)
    parts.extend(str(item) for item in (track.get("visualObjects") or []) if item)
    parts.extend(str(item) for item in (track.get("visualLogos") or []) if item)
    parts.extend(str(item) for item in (track.get("visualText") or []) if item)
    if track.get("visualSummary"):
        parts.append(str(track.get("visualSummary") or ""))
    return " ".join(part for part in parts if part).lower()


def _track_candidate_score(
    track: dict[str, Any],
    terms: list[str],
    region_requirements: Optional[dict[str, set[str]]] = None,
    soft_terms: Optional[list[str]] = None,
) -> float:
    haystack = _track_search_text(track)
    visual_haystack = " ".join(
        str(item)
        for item in [
            *(track.get("visualLabels") or []),
            *(track.get("visualObjects") or []),
            *(track.get("visualLogos") or []),
            *(track.get("visualText") or []),
            str(track.get("visualSummary") or ""),
        ]
        if item
    ).lower()
    matched_terms = {term for term in terms if term in haystack}
    matched_visual_terms = {term for term in terms if term in visual_haystack}
    if terms and not matched_terms and not region_requirements:
        return 0.0

    color_terms = {
        "black",
        "blue",
        "brown",
        "burgundy",
        "cyan",
        "gray",
        "green",
        "maroon",
        "orange",
        "pink",
        "purple",
        "red",
        "white",
        "wine",
        "yellow",
    }
    score = 1.0 if not terms and not region_requirements else 0.0
    for term in matched_terms:
        score += 2.5 if term in color_terms else 1.5
    score += sum(0.8 for _ in matched_visual_terms)

    if region_requirements:
        track_region_colors = _track_region_colors(track)
        matched_regions = 0
        camera_ambiguous_regions = 0
        for region_name, accepted_colors in region_requirements.items():
            region_colors = track_region_colors.get(region_name) or set()
            if region_colors.intersection(accepted_colors):
                matched_regions += 1
                continue
            if _camera_ambiguous_region_match(region_colors, accepted_colors):
                camera_ambiguous_regions += 1
                continue
            return 0.0
        score += matched_regions * 4.0
        score += camera_ambiguous_regions * 2.25

    if soft_terms:
        for term in soft_terms:
            if term in visual_haystack:
                score += 0.75
            elif term in haystack:
                score += 0.4

    if track.get("thumbnailPath"):
        score += 0.4
    if track.get("appearanceSummary"):
        score += 0.6
    if track.get("visualSummary"):
        score += 0.4
    score += min(float(track.get("bestArea") or 0.0) / 10000.0, 2.0)
    return score


def _build_track_result(
    track: dict[str, Any],
    video: dict[str, Any],
    *,
    confidence: int,
    match_reason: str,
    semantic_score: Optional[float] = None,
    possible_match: bool = False,
    match_strategy: str = "metadata",
    thumbnail_path_override: Optional[str] = None,
    timestamp_override: Optional[str] = None,
    frame_override: Optional[int] = None,
    offset_seconds_override: Optional[float] = None,
) -> dict[str, Any]:
    timestamp = timestamp_override or track.get("bestTimestamp") or track.get("firstTimestamp") or video.get("timestamp") or "Unknown Time"
    frame = frame_override if frame_override is not None else (track.get("bestFrame") if track.get("bestFrame") is not None else track.get("firstFrame"))
    offset_seconds = (
        offset_seconds_override
        if offset_seconds_override is not None
        else (track.get("bestOffsetSeconds") if track.get("bestOffsetSeconds") is not None else track.get("firstOffsetSeconds"))
    )
    return {
        "id": str(track.get("id") or f"track-{video['id']}-{track.get('pedestrianId') or 'unknown'}"),
        "videoId": video["id"],
        "timestamp": str(timestamp),
        "date": str(video.get("date") or ""),
        "location": str(track.get("location") or video.get("location") or "Unknown Location"),
        "confidence": max(0, min(100, int(confidence))),
        "matchReason": match_reason,
        "pedestrianId": track.get("pedestrianId"),
        "frame": frame,
        "offsetSeconds": offset_seconds,
        "firstTimestamp": track.get("firstTimestamp"),
        "lastTimestamp": track.get("lastTimestamp"),
        "firstOffsetSeconds": track.get("firstOffsetSeconds"),
        "lastOffsetSeconds": track.get("lastOffsetSeconds"),
        "thumbnailPath": thumbnail_path_override or track.get("thumbnailPath"),
        "previewPath": video.get("processedPath") or video.get("rawPath"),
        "appearanceSummary": track.get("appearanceSummary"),
        "visualLabels": track.get("visualLabels") or [],
        "visualObjects": track.get("visualObjects") or [],
        "visualLogos": track.get("visualLogos") or [],
        "visualText": track.get("visualText") or [],
        "visualSummary": track.get("visualSummary"),
        "semanticScore": None if semantic_score is None else round(float(semantic_score), 4),
        "possibleMatch": bool(possible_match),
        "matchStrategy": match_strategy,
    }


def _interpolate_confidence(score: float, *, low_score: float, high_score: float, low_confidence: int, high_confidence: int) -> int:
    if high_score <= low_score:
        return high_confidence
    ratio = max(0.0, min(1.0, (score - low_score) / (high_score - low_score)))
    return int(round(low_confidence + ratio * (high_confidence - low_confidence)))


def _semantic_confidence(semantic_score: float) -> int:
    score = max(0.0, float(semantic_score))
    if score <= SEMANTIC_MIN_SCORE:
        return 35
    if score < SEMANTIC_POSSIBLE_MATCH_SCORE:
        return _interpolate_confidence(
            score,
            low_score=SEMANTIC_MIN_SCORE,
            high_score=SEMANTIC_POSSIBLE_MATCH_SCORE,
            low_confidence=35,
            high_confidence=59,
        )
    if score < SEMANTIC_STRONG_MATCH_SCORE:
        return _interpolate_confidence(
            score,
            low_score=SEMANTIC_POSSIBLE_MATCH_SCORE,
            high_score=SEMANTIC_STRONG_MATCH_SCORE,
            low_confidence=60,
            high_confidence=96,
        )
    return 96


def _semantic_track_matches(
    query: str,
    tracks: list[dict[str, Any]],
    videos_by_id: dict[str, dict[str, Any]],
    required_location_id: str,
) -> dict[str, dict[str, Any]]:
    if not query.strip() or not tracks:
        return {}

    try:
        from . import semantic_search

        raw_matches = semantic_search.search_tracks(query, backend_dir=BACKEND_DIR, limit=MAX_AI_SEARCH_CANDIDATES * 2)
    except Exception:
        return {}

    tracks_by_id = {str(track.get("id") or ""): track for track in tracks if track.get("id")}
    filtered: dict[str, dict[str, Any]] = {}
    for match in raw_matches:
        track_id = str(match.get("id") or "")
        track = tracks_by_id.get(track_id)
        if track is None:
            continue
        video = videos_by_id.get(track.get("videoId") or "")
        if video is None:
            continue
        if required_location_id and str(video.get("locationId") or "") != required_location_id:
            continue
        filtered[track_id] = match
    return filtered


def _ai_ranking_query(query: str, query_plan: dict[str, Any]) -> str:
    lines = [f"Original user query: {query.strip()}"]

    location_name = str(query_plan.get("locationName") or "").strip()
    if location_name:
        lines.append(f"Required location: {location_name}")

    region_requirements = query_plan.get("regionColorRequirements") or {}
    if isinstance(region_requirements, dict) and region_requirements:
        requirements_text = ", ".join(
            f"{region}: {'/'.join(sorted(colors))}"
            for region, colors in sorted(region_requirements.items())
            if colors
        )
        if requirements_text:
            lines.append(f"Strict body-region color requirements: {requirements_text}")

    hard_terms = query_plan.get("hardTerms") or []
    if hard_terms:
        lines.append(f"Searchable appearance terms: {', '.join(str(term) for term in hard_terms)}")

    soft_terms = query_plan.get("softTerms") or []
    if soft_terms:
        lines.append(
            "Soft preferences that may be absent from metadata: "
            + ", ".join(str(term) for term in soft_terms)
        )

    summary = str(query_plan.get("summary") or "").strip()
    if summary:
        lines.append(f"Structured interpretation: {summary}")

    return "\n".join(lines)


def _track_results(
    query: str,
    tracks: list[dict[str, Any]],
    videos_by_id: dict[str, dict[str, Any]],
    ai_ranker: Optional[Callable[[str, list[dict[str, Any]]], list[dict[str, Any]]]],
    query_plan: dict[str, Any],
) -> list[dict[str, Any]]:
    terms = list(query_plan.get("hardTerms") or [])
    region_requirements = query_plan.get("regionColorRequirements") or {}
    soft_terms = list(query_plan.get("softTerms") or [])
    required_location_id = str(query_plan.get("locationId") or "")
    semantic_matches = _semantic_track_matches(query, tracks, videos_by_id, required_location_id)

    if query.strip() and not terms and not region_requirements and not required_location_id and not semantic_matches:
        return []

    scored_tracks: list[tuple[float, dict[str, Any], dict[str, Any]]] = []
    for track in tracks:
        video = videos_by_id.get(track.get("videoId") or "")
        if video is None:
            continue
        if required_location_id and str(video.get("locationId") or "") != required_location_id:
            continue
        score = _track_candidate_score(track, terms, region_requirements, soft_terms)
        semantic_score = float((semantic_matches.get(str(track.get("id") or "")) or {}).get("semanticScore") or 0.0)
        if semantic_score >= SEMANTIC_MIN_SCORE:
            score += semantic_score * 12.0
        scored_tracks.append((score, track, video))

    scored_tracks.sort(
        key=lambda item: (
            item[0],
            float(item[1].get("bestArea") or 0.0),
            float(item[1].get("bestOffsetSeconds") or item[1].get("firstOffsetSeconds") or 0.0),
        ),
        reverse=True,
    )

    if not scored_tracks:
        return []

    positively_scored_tracks = [item for item in scored_tracks if item[0] > 0]
    semantic_positive_tracks = [
        item
        for item in scored_tracks
        if float((semantic_matches.get(str(item[1].get("id") or "")) or {}).get("semanticScore") or 0.0) >= SEMANTIC_MIN_SCORE
    ]
    if query.strip() and not positively_scored_tracks and not semantic_positive_tracks:
        return []

    if semantic_positive_tracks:
        semantic_results: list[dict[str, Any]] = []
        for _score, track, video in semantic_positive_tracks[:5]:
            semantic_match = semantic_matches.get(str(track.get("id") or "")) or {}
            semantic_score = float(semantic_match.get("semanticScore") or 0.0)
            confidence = _semantic_confidence(semantic_score)
            crop_label = str(semantic_match.get("cropLabel") or "representative")
            possible_match = semantic_score < SEMANTIC_POSSIBLE_MATCH_SCORE
            reason = f"Semantic appearance match from the {crop_label} crop."
            semantic_results.append(
                _build_track_result(
                    track,
                    video,
                    confidence=confidence,
                    match_reason=reason,
                    semantic_score=semantic_score,
                    possible_match=possible_match,
                    match_strategy="semantic",
                    thumbnail_path_override=semantic_match.get("matchedCropPath"),
                    timestamp_override=semantic_match.get("timestamp"),
                    frame_override=semantic_match.get("frame"),
                    offset_seconds_override=semantic_match.get("offsetSeconds"),
                )
            )
        if semantic_results:
            return semantic_results[:5]

    fallback_candidates = positively_scored_tracks[:MAX_AI_SEARCH_CANDIDATES]
    candidates = fallback_candidates[:MAX_AI_SEARCH_CANDIDATES]
    candidate_payload = [
        {
            "id": track["id"],
            "location": track.get("location") or video.get("location"),
            "timestamp": track.get("bestTimestamp") or track.get("firstTimestamp") or video.get("timestamp"),
            "pedestrianId": track.get("pedestrianId"),
            "appearanceSummary": track.get("appearanceSummary") or "Appearance summary unavailable.",
            "appearanceHints": track.get("appearanceHints") or [],
            "visualLabels": track.get("visualLabels") or [],
            "visualObjects": track.get("visualObjects") or [],
            "visualLogos": track.get("visualLogos") or [],
            "visualText": track.get("visualText") or [],
            "visualSummary": track.get("visualSummary") or "",
            "occlusion": _occlusion_label(track.get("occlusionClass")) or "clear view",
            "thumbnailAvailable": bool(track.get("thumbnailPath")),
            "semanticScore": round(float((semantic_matches.get(str(track.get("id") or "")) or {}).get("semanticScore") or 0.0), 4),
        }
        for _, track, video in candidates
        if track.get("id")
    ]

    candidate_map = {track["id"]: (track, video, score) for score, track, video in candidates if track.get("id")}
    ai_results: list[dict[str, Any]] = []
    if ai_ranker is not None and query.strip():
        try:
            ai_results = ai_ranker(_ai_ranking_query(query, query_plan), candidate_payload)
        except Exception:
            ai_results = []

    enriched_results: list[dict[str, Any]] = []
    used_ids: set[str] = set()
    for match in ai_results:
        candidate_id = str(match.get("id") or "")
        if candidate_id in used_ids or candidate_id not in candidate_map:
            continue
        track, video, _ = candidate_map[candidate_id]
        enriched_results.append(
            _build_track_result(
                track,
                video,
                confidence=int(match.get("confidence", 0)),
                match_reason=str(match.get("reason") or track.get("appearanceSummary") or "Potential visual match."),
                match_strategy="metadata",
            )
        )
        used_ids.add(candidate_id)

    if enriched_results:
        return enriched_results[:5]

    if not fallback_candidates:
        return []

    fallback_results: list[dict[str, Any]] = []
    for index, (score, track, video) in enumerate(fallback_candidates[:5], start=1):
        fallback_confidence = 68 if not query.strip() else min(96, int(round(62 + score * 10)) - (index - 1) * 2)
        fallback_results.append(
            _build_track_result(
                track,
                video,
                confidence=fallback_confidence,
                match_reason=str(track.get("appearanceSummary") or "Representative deduplicated pedestrian track."),
                match_strategy="metadata",
            )
        )
    return fallback_results[:5]


def search_results(
    query: str,
    ai_ranker: Optional[Callable[[str, list[dict[str, Any]]], list[dict[str, Any]]]] = None,
    query_parser: Optional[Callable[[str, list[dict[str, Any]]], dict[str, Any]]] = None,
) -> list[dict[str, Any]]:
    state = load_state()
    videos_by_id = {video["id"]: video for video in state["videos"]}
    query_plan = _build_search_query_plan(query, state.get("locations") or [], query_parser)

    pedestrian_tracks = state.get("pedestrianTracks") or []
    if pedestrian_tracks:
        track_results = _track_results(query, pedestrian_tracks, videos_by_id, ai_ranker, query_plan)
        if track_results:
            return track_results

    terms = list(query_plan.get("hardTerms") or [])
    required_location_id = str(query_plan.get("locationId") or "")
    if query.strip() and not terms and not required_location_id:
        return []

    results: list[dict[str, Any]] = []
    for event in state["events"]:
        video = videos_by_id.get(event.get("videoId") or "")
        if not video:
            continue
        if required_location_id and str(video.get("locationId") or "") != required_location_id:
            continue

        haystack = f"{event['location']} {event['description']}".lower()
        if terms and not any(term in haystack for term in terms):
            continue

        results.append(
            {
                "id": event["id"],
                "videoId": video["id"],
                "timestamp": event["timestamp"],
                "date": video["date"],
                "location": event["location"],
                "confidence": 80 + (len(results) * 3),
                "matchReason": event["description"],
                "pedestrianId": event.get("pedestrianId"),
                "frame": event.get("frame"),
                "offsetSeconds": event.get("offsetSeconds"),
                "firstTimestamp": None,
                "lastTimestamp": None,
                "firstOffsetSeconds": None,
                "lastOffsetSeconds": None,
                "thumbnailPath": None,
                "previewPath": video.get("processedPath") or video.get("rawPath"),
                "appearanceSummary": None,
                "visualLabels": [],
                "visualObjects": [],
                "visualLogos": [],
                "visualText": [],
                "visualSummary": None,
                "semanticScore": None,
                "possibleMatch": False,
                "matchStrategy": "event",
            }
        )
    if results:
        return results[:5]

    if query.strip():
        return []

    fallback = []
    for video in state["videos"][:5]:
        fallback.append(
            {
                "id": f"fallback-{video['id']}",
                "videoId": video["id"],
                "timestamp": video["timestamp"],
                "date": video["date"],
                "location": video["location"],
                "confidence": 75,
                "matchReason": f"Fallback result for query: {query}",
                "pedestrianId": None,
                "frame": None,
                "offsetSeconds": None,
                "firstTimestamp": None,
                "lastTimestamp": None,
                "firstOffsetSeconds": None,
                "lastOffsetSeconds": None,
                "thumbnailPath": None,
                "previewPath": video.get("processedPath") or video.get("rawPath"),
                "appearanceSummary": None,
                "visualLabels": [],
                "visualObjects": [],
                "visualLogos": [],
                "visualText": [],
                "visualSummary": None,
                "semanticScore": None,
                "possibleMatch": False,
                "matchStrategy": "event",
            }
        )
    return fallback
