from __future__ import annotations

import csv
import json
import logging
import math
import os
import re
import shutil
import zipfile
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
RTDETR_DIR = BACKEND_DIR / "Occlusion-Robust-RTDETR"
MODELS_DIR = RTDETR_DIR / "weights" / "models"
LEGACY_MODELS_DIR = STORAGE_DIR / "vehicle_models"
RAW_VIDEOS_DIR = STORAGE_DIR / "vehicle_videos" / "raw"
PROCESSED_VIDEOS_DIR = STORAGE_DIR / "vehicle_videos" / "processed"
EXPORTS_DIR = STORAGE_DIR / "vehicle_exports"
PORTABLE_DIR = STORAGE_DIR / "vehicle_portable"
PORTABLE_VIDEOS_DIR = PORTABLE_DIR / "videos"
QUEUE_HISTORY_JSON_FILE = PORTABLE_DIR / "vehicle_queue_history.json"
QUEUE_HISTORY_CSV_FILE = PORTABLE_DIR / "vehicle_queue_history.csv"
PORTABLE_MANIFEST_FILE = PORTABLE_DIR / "vehicle_manifest.json"
DATA_FILE = STORAGE_DIR / "dev_vehicle_data.json"
CLOCK_TIME_FORMATS = ("%H:%M", "%H:%M:%S", "%I:%M %p", "%I:%M:%S %p")
MIN_DRILLDOWN_BUCKET_MINUTES = 5
VIDEO_TIMELINE_MAX_BUCKETS = 120
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
SEARCH_QUERY_PARSER_MIN_TOKENS = 6
UPLOAD_STATUS_LOCK = Lock()
UPLOAD_STATUSES: dict[str, dict[str, Any]] = {}
UPLOAD_CANCEL_REQUESTS: set[str] = set()
TERMINAL_UPLOAD_STATES = {"complete", "error", "cancelled"}
UPLOAD_CANCEL_REQUESTED_MESSAGE_PREFIX = "cancellation requested"
RECOVERED_CANCELLED_UPLOAD_MESSAGE = "Video upload cancelled."
RECOVERED_INTERRUPTED_UPLOAD_MESSAGE = "Upload interrupted before completion. Please upload the video again."
UPLOAD_HISTORY_CSV_FIELDS = (
    "uploadId",
    "state",
    "progressPercent",
    "message",
    "phase",
    "videoId",
    "error",
    "fileName",
    "locationId",
    "locationName",
    "date",
    "startTime",
    "endTime",
    "fastMode",
    "createdAt",
    "startedAt",
    "completedAt",
    "updatedAt",
)


def slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.strip().lower()).strip("-") or f"item-{uuid4().hex[:8]}"


def _utc_timestamp() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def _format_video_offset_clock(offset_seconds: Any) -> str:
    try:
        total_seconds = max(0, int(float(offset_seconds)))
    except (TypeError, ValueError):
        total_seconds = 0

    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def _optional_string(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _write_json_file(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _csv_cell_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False)
    return value


def _csv_fieldnames(rows: list[dict[str, Any]], preferred: tuple[str, ...] = ()) -> list[str]:
    fieldnames: list[str] = []
    seen: set[str] = set()
    for fieldname in preferred:
        if fieldname not in seen:
            fieldnames.append(fieldname)
            seen.add(fieldname)
    for row in rows:
        for fieldname in row.keys():
            if fieldname not in seen:
                fieldnames.append(fieldname)
                seen.add(fieldname)
    return fieldnames


def _csv_text(rows: list[dict[str, Any]], preferred: tuple[str, ...] = ()) -> str:
    from io import StringIO

    normalized_rows = [{key: _csv_cell_value(value) for key, value in row.items()} for row in rows]
    fieldnames = _csv_fieldnames(normalized_rows, preferred)
    if not fieldnames:
        return ""

    buffer = StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(normalized_rows)
    return buffer.getvalue()


def _write_csv_rows(path: Path, rows: list[dict[str, Any]], preferred: tuple[str, ...] = ()) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_csv_text(rows, preferred=preferred), encoding="utf-8")


def _normalize_upload_status_record(record: Any) -> Optional[dict[str, Any]]:
    if not isinstance(record, dict):
        return None

    upload_id = _optional_string(record.get("uploadId"))
    state = _optional_string(record.get("state"))
    message = _optional_string(record.get("message"))
    updated_at = _optional_string(record.get("updatedAt"))
    if not upload_id or not state or not message or not updated_at:
        return None

    raw_progress = record.get("progressPercent")
    try:
        progress_percent = None if raw_progress in (None, "") else max(0, min(100, int(round(float(raw_progress)))))
    except (TypeError, ValueError):
        progress_percent = None

    return {
        "uploadId": upload_id,
        "state": state,
        "progressPercent": progress_percent,
        "message": message,
        "phase": _optional_string(record.get("phase")),
        "videoId": _optional_string(record.get("videoId")),
        "error": _optional_string(record.get("error")),
        "fileName": _optional_string(record.get("fileName")),
        "locationId": _optional_string(record.get("locationId")),
        "locationName": _optional_string(record.get("locationName")),
        "date": _optional_string(record.get("date")),
        "startTime": _optional_string(record.get("startTime")),
        "endTime": _optional_string(record.get("endTime")),
        "fastMode": bool(record.get("fastMode")) if record.get("fastMode") is not None else None,
        "createdAt": _optional_string(record.get("createdAt")) or updated_at,
        "startedAt": _optional_string(record.get("startedAt")),
        "completedAt": _optional_string(record.get("completedAt")),
        "updatedAt": updated_at,
    }


def _is_terminal_upload_state(state: Any) -> bool:
    return isinstance(state, str) and state in TERMINAL_UPLOAD_STATES


def _upload_message_requests_cancellation(message: Any) -> bool:
    return isinstance(message, str) and message.strip().lower().startswith(UPLOAD_CANCEL_REQUESTED_MESSAGE_PREFIX)


def _recover_interrupted_upload_statuses(statuses: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not statuses:
        return statuses

    recovered_at = _utc_timestamp()
    recovered_statuses: list[dict[str, Any]] = []
    changed = False

    for status in statuses:
        if _is_terminal_upload_state(status.get("state")):
            recovered_statuses.append(status)
            continue

        recovered = deepcopy(status)
        if _upload_message_requests_cancellation(status.get("message")):
            recovered["state"] = "cancelled"
            recovered["message"] = RECOVERED_CANCELLED_UPLOAD_MESSAGE
            recovered["error"] = None
        else:
            recovered["state"] = "error"
            recovered["message"] = RECOVERED_INTERRUPTED_UPLOAD_MESSAGE
            recovered["error"] = RECOVERED_INTERRUPTED_UPLOAD_MESSAGE

        recovered["phase"] = None
        recovered["progressPercent"] = None
        recovered["completedAt"] = recovered_at
        recovered["updatedAt"] = recovered_at
        recovered_statuses.append(recovered)
        changed = True

    if changed:
        _persist_upload_status_snapshot(recovered_statuses)

    return recovered_statuses


def _read_upload_status_history() -> list[dict[str, Any]]:
    ensure_storage_layout()
    if not QUEUE_HISTORY_JSON_FILE.exists():
        return []

    try:
        payload = json.loads(QUEUE_HISTORY_JSON_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        logger.exception("Failed to read upload history from %s", QUEUE_HISTORY_JSON_FILE)
        return []

    if isinstance(payload, dict):
        raw_items = payload.get("statuses") or []
    elif isinstance(payload, list):
        raw_items = payload
    else:
        raw_items = []

    statuses = [status for item in raw_items if (status := _normalize_upload_status_record(item)) is not None]
    statuses.sort(key=lambda item: (item.get("createdAt") or item["updatedAt"], item["uploadId"]))
    return statuses


def _persist_upload_status_snapshot(statuses: list[dict[str, Any]]) -> None:
    ordered_statuses = [status for item in statuses if (status := _normalize_upload_status_record(item)) is not None]
    ordered_statuses.sort(key=lambda item: (item.get("createdAt") or item["updatedAt"], item["uploadId"]))
    _write_json_file(QUEUE_HISTORY_JSON_FILE, ordered_statuses)
    _write_csv_rows(QUEUE_HISTORY_CSV_FILE, ordered_statuses, preferred=UPLOAD_HISTORY_CSV_FIELDS)


def _ensure_upload_statuses_loaded() -> None:
    ensure_storage_layout()
    with UPLOAD_STATUS_LOCK:
        if UPLOAD_STATUSES:
            return

    statuses = _recover_interrupted_upload_statuses(_read_upload_status_history())
    if not statuses:
        return

    with UPLOAD_STATUS_LOCK:
        if UPLOAD_STATUSES:
            return
        UPLOAD_CANCEL_REQUESTS.clear()
        UPLOAD_STATUSES.update({status["uploadId"]: status for status in statuses})


def set_upload_status(
    upload_id: str,
    *,
    state: str,
    progress_percent: Optional[int],
    message: str,
    phase: Optional[str] = None,
    video_id: Optional[str] = None,
    error: Optional[str] = None,
    file_name: Optional[str] = None,
    location_id: Optional[str] = None,
    location_name: Optional[str] = None,
    date: Optional[str] = None,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    fast_mode: Optional[bool] = None,
) -> dict[str, Any]:
    _ensure_upload_statuses_loaded()
    normalized_progress = None if progress_percent is None else max(0, min(100, int(round(progress_percent))))
    timestamp = _utc_timestamp()

    with UPLOAD_STATUS_LOCK:
        previous = UPLOAD_STATUSES.get(upload_id, {})
        status = {
            "uploadId": upload_id,
            "state": state,
            "progressPercent": normalized_progress,
            "message": message,
            "phase": phase if phase is not None else previous.get("phase"),
            "videoId": video_id if video_id is not None else previous.get("videoId"),
            "error": error if error is not None else (previous.get("error") if state == "error" else None),
            "fileName": file_name if file_name is not None else previous.get("fileName"),
            "locationId": location_id if location_id is not None else previous.get("locationId"),
            "locationName": location_name if location_name is not None else previous.get("locationName"),
            "date": date if date is not None else previous.get("date"),
            "startTime": start_time if start_time is not None else previous.get("startTime"),
            "endTime": end_time if end_time is not None else previous.get("endTime"),
            "fastMode": fast_mode if fast_mode is not None else previous.get("fastMode"),
            "createdAt": previous.get("createdAt") or timestamp,
            "startedAt": previous.get("startedAt") or (timestamp if state != "queued" else None),
            "completedAt": timestamp if state in {"complete", "error", "cancelled"} else previous.get("completedAt"),
            "updatedAt": timestamp,
        }
        UPLOAD_STATUSES[upload_id] = status
        if state in TERMINAL_UPLOAD_STATES:
            UPLOAD_CANCEL_REQUESTS.discard(upload_id)
        snapshot = [deepcopy(item) for item in UPLOAD_STATUSES.values()]

    _persist_upload_status_snapshot(snapshot)
    return deepcopy(status)


def get_upload_status(upload_id: str) -> Optional[dict[str, Any]]:
    _ensure_upload_statuses_loaded()
    with UPLOAD_STATUS_LOCK:
        status = UPLOAD_STATUSES.get(upload_id)
    return deepcopy(status) if status is not None else None


def request_upload_cancel(upload_id: str) -> bool:
    _ensure_upload_statuses_loaded()
    with UPLOAD_STATUS_LOCK:
        if upload_id not in UPLOAD_STATUSES:
            return False
        UPLOAD_CANCEL_REQUESTS.add(upload_id)
    return True


def is_upload_cancel_requested(upload_id: str) -> bool:
    _ensure_upload_statuses_loaded()
    with UPLOAD_STATUS_LOCK:
        return upload_id in UPLOAD_CANCEL_REQUESTS


def list_upload_statuses() -> list[dict[str, Any]]:
    _ensure_upload_statuses_loaded()
    with UPLOAD_STATUS_LOCK:
        statuses = [deepcopy(status) for status in UPLOAD_STATUSES.values()]
    statuses.sort(key=lambda item: (item.get("updatedAt") or "", item["uploadId"]), reverse=True)
    return statuses


def _portable_relative_path(path: Path) -> str:
    try:
        return str(path.relative_to(BACKEND_DIR))
    except ValueError:
        return str(path)


def _portable_video_directory(video_id: str) -> Path:
    return PORTABLE_VIDEOS_DIR / str(video_id)


def _read_portable_video_json(video_id: str, filename: str) -> Any:
    path = _portable_video_directory(video_id) / filename
    if not path.exists() or not path.is_file():
        return None

    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        logger.exception("Failed to read portable artifact %s", path)
        return None


def _portable_video_tracks(video_id: str) -> list[dict[str, Any]]:
    payload = _read_portable_video_json(video_id, "tracks.json")
    if not isinstance(payload, list):
        return []

    compact_tracks: list[dict[str, Any]] = []
    for fallback_index, track in enumerate(payload):
        if not isinstance(track, dict):
            continue

        first_offset = _track_window_offset(track.get("firstOffsetSeconds"))
        last_offset = _track_window_offset(track.get("lastOffsetSeconds"))
        if first_offset is None or last_offset is None:
            continue

        if last_offset < first_offset:
            first_offset, last_offset = last_offset, first_offset

        compact_tracks.append(
            {
                "id": str(track.get("trackId") or track.get("id") or f"{video_id}-portable-track-{fallback_index}"),
                "pedestrianId": track.get("pedestrianId"),
                "firstOffsetSeconds": first_offset,
                "lastOffsetSeconds": last_offset,
            }
        )

    return compact_tracks


def _portable_video_severity_summary(video_id: str) -> Optional[dict[str, Any]]:
    payload = _read_portable_video_json(video_id, "timeline.json")
    if not isinstance(payload, list):
        return None

    timeline_rows: list[dict[str, Any]] = []
    for row in payload:
        if not isinstance(row, dict):
            continue

        try:
            offset = max(0, float(row.get("offsetSeconds")))
        except (TypeError, ValueError):
            continue

        score_value: Optional[float] = None
        try:
            raw_score = row.get("ptsiScore")
            if raw_score is not None:
                score_value = float(raw_score)
        except (TypeError, ValueError):
            score_value = None

        severity_value = str(row.get("severity") or "").lower()
        if severity_value not in {"neutral", "light", "moderate", "heavy"}:
            severity_value = _timeline_severity_from_score(score_value)

        timeline_rows.append(
            {
                "offset": offset,
                "severity": severity_value,
                "score": score_value,
            }
        )

    if not timeline_rows:
        return None

    timeline_rows.sort(key=lambda item: item["offset"])

    buckets: list[dict[str, Any]] = []
    segment_start = float(timeline_rows[0]["offset"])
    current_severity = str(timeline_rows[0]["severity"])
    running_score_total = float(timeline_rows[0]["score"]) if timeline_rows[0]["score"] is not None else 0.0
    running_score_count = 1 if timeline_rows[0]["score"] is not None else 0

    for index in range(1, len(timeline_rows)):
        row = timeline_rows[index]
        row_offset = float(row["offset"])
        row_severity = str(row["severity"])

        if row_severity != current_severity:
            buckets.append(
                {
                    "startOffsetSeconds": segment_start,
                    "endOffsetSeconds": row_offset,
                    "severity": current_severity,
                    "score": (running_score_total / running_score_count) if running_score_count > 0 else None,
                }
            )
            segment_start = row_offset
            current_severity = row_severity
            running_score_total = 0.0
            running_score_count = 0

        if row["score"] is not None:
            running_score_total += float(row["score"])
            running_score_count += 1

    final_end = float(timeline_rows[-1]["offset"]) + 1.0
    buckets.append(
        {
            "startOffsetSeconds": segment_start,
            "endOffsetSeconds": final_end,
            "severity": current_severity,
            "score": (running_score_total / running_score_count) if running_score_count > 0 else None,
        }
    )

    return {
        "bucketCount": len(buckets),
        "sampledSeconds": len(timeline_rows),
        "buckets": buckets,
    }


def _severity_summary_from_timeline_rows(timeline_rows: list[dict[str, Any]]) -> Optional[dict[str, Any]]:
    if not timeline_rows:
        return None

    normalized: list[dict[str, Any]] = []
    for row in timeline_rows:
        try:
            offset = max(0.0, float(row.get("offsetSeconds")))
        except (TypeError, ValueError):
            continue

        score_value: Optional[float] = None
        try:
            raw_score = row.get("ptsiScore")
            if raw_score is not None:
                score_value = float(raw_score)
        except (TypeError, ValueError):
            score_value = None

        severity_value = str(row.get("severity") or "").lower()
        if severity_value not in {"neutral", "light", "moderate", "heavy"}:
            severity_value = _timeline_severity_from_score(score_value)

        normalized.append({"offset": offset, "score": score_value, "severity": severity_value})

    if not normalized:
        return None

    normalized.sort(key=lambda item: item["offset"])
    buckets: list[dict[str, Any]] = []
    segment_start = float(normalized[0]["offset"])
    current_severity = str(normalized[0]["severity"])
    running_score_total = float(normalized[0]["score"]) if normalized[0]["score"] is not None else 0.0
    running_score_count = 1 if normalized[0]["score"] is not None else 0

    for index in range(1, len(normalized)):
        row = normalized[index]
        row_offset = float(row["offset"])
        row_severity = str(row["severity"])

        if row_severity != current_severity:
            buckets.append(
                {
                    "startOffsetSeconds": segment_start,
                    "endOffsetSeconds": row_offset,
                    "severity": current_severity,
                    "score": (running_score_total / running_score_count) if running_score_count > 0 else None,
                }
            )
            segment_start = row_offset
            current_severity = row_severity
            running_score_total = 0.0
            running_score_count = 0

        if row["score"] is not None:
            running_score_total += float(row["score"])
            running_score_count += 1

    buckets.append(
        {
            "startOffsetSeconds": segment_start,
            "endOffsetSeconds": float(normalized[-1]["offset"]) + 1.0,
            "severity": current_severity,
            "score": (running_score_total / running_score_count) if running_score_count > 0 else None,
        }
    )

    return {"bucketCount": len(buckets), "sampledSeconds": len(normalized), "buckets": buckets}


def _severity_summary_has_signal(summary: Optional[dict[str, Any]]) -> bool:
    buckets = (summary or {}).get("buckets") or []
    for bucket in buckets:
        severity = str((bucket or {}).get("severity") or "").lower()
        if severity and severity != "neutral":
            return True

        try:
            score_value = (bucket or {}).get("score")
            if score_value is not None and float(score_value) > 0.0:
                return True
        except (TypeError, ValueError):
            continue

    return False


def _processed_congestion_csv_path(video: dict[str, Any]) -> Optional[Path]:
    processed_relative = _optional_string(video.get("processedPath"))
    if not processed_relative:
        return None

    processed_path = (BACKEND_DIR / processed_relative).resolve(strict=False)
    congestion_path = Path(f"{processed_path.with_suffix('')}_congestion.csv")
    if not congestion_path.exists() or not congestion_path.is_file():
        return None
    return congestion_path


def _parse_congestion_timestamp_offset(value: Any) -> Optional[int]:
    parsed_time = _parse_clock_time(_optional_string(value))
    if parsed_time is None:
        return None
    return (parsed_time.hour * 3600) + (parsed_time.minute * 60) + parsed_time.second


def _score_from_los(los: Optional[str], vc_ratio: Optional[float]) -> float:
    if los is not None:
        normalized = str(los).upper()
        los_scores = {
            "A": 8.0,
            "B": 24.0,
            "C": 42.0,
            "D": 58.0,
            "E": 75.0,
            "F": 92.0,
        }
        if normalized in los_scores:
            return los_scores[normalized]

    if vc_ratio is None:
        return 0.0

    return float(max(0.0, min(100.0, vc_ratio * 100.0)))


def _merge_congestion_timeline_rows(
    *,
    video: dict[str, Any],
    observed_at: Optional[datetime],
    timeline_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    congestion_path = _processed_congestion_csv_path(video)
    if congestion_path is None:
        return timeline_rows

    try:
        with congestion_path.open("r", encoding="utf-8", newline="") as file_handle:
            reader = csv.DictReader(file_handle)
            congestion_rows = [dict(row) for row in reader if isinstance(row, dict)]
    except (OSError, UnicodeDecodeError, csv.Error):
        return timeline_rows

    if not congestion_rows:
        return timeline_rows

    by_offset: dict[int, dict[str, Any]] = {}
    for row in congestion_rows:
        offset = _parse_congestion_timestamp_offset(row.get("timestamp"))
        if offset is None:
            continue

        los_value = _optional_string(row.get("los"))
        los_upper = los_value.upper() if los_value else None
        try:
            vc_ratio = float(row.get("vc_ratio") or "")
        except (TypeError, ValueError):
            vc_ratio = None
        try:
            volume = int(round(float(row.get("volume") or "0")))
        except (TypeError, ValueError):
            volume = 0
        try:
            counted_total = int(round(float(row.get("vehicles_counted_total") or "0")))
        except (TypeError, ValueError):
            counted_total = 0

        existing = by_offset.get(offset)
        score = _score_from_los(los_upper, vc_ratio)
        severity = _timeline_severity_from_score(score)
        payload = {
            "offsetSeconds": offset,
            "visiblePedestrians": max(0, volume),
            "detectedNow": max(0, volume),
            "cumulativeUniquePedestrians": max(0, counted_total),
            "totalPedestriansSoFar": max(0, counted_total),
            "ptsiScore": score,
            "los": los_upper,
            "losDescription": _ptsi_los_description(los_upper),
            "severity": severity,
        }

        if existing is None:
            by_offset[offset] = payload
            continue

        # Keep highest intensity when congestion CSV has repeated rows in the same second.
        existing["visiblePedestrians"] = max(int(existing.get("visiblePedestrians") or 0), int(payload["visiblePedestrians"]))
        existing["detectedNow"] = max(int(existing.get("detectedNow") or 0), int(payload["detectedNow"]))
        existing["cumulativeUniquePedestrians"] = max(
            int(existing.get("cumulativeUniquePedestrians") or 0),
            int(payload["cumulativeUniquePedestrians"]),
        )
        existing["totalPedestriansSoFar"] = max(int(existing.get("totalPedestriansSoFar") or 0), int(payload["totalPedestriansSoFar"]))
        if float(payload["ptsiScore"]) > float(existing.get("ptsiScore") or 0.0):
            existing["ptsiScore"] = payload["ptsiScore"]
            existing["los"] = payload["los"]
            existing["losDescription"] = payload["losDescription"]
            existing["severity"] = payload["severity"]

    if not by_offset:
        return timeline_rows

    timeline_by_offset = {int(item.get("offsetSeconds") or 0): item for item in timeline_rows}
    combined_offsets = sorted(set(timeline_by_offset.keys()) | set(by_offset.keys()))
    merged_rows: list[dict[str, Any]] = []
    running_total = 0

    for offset in combined_offsets:
        base = deepcopy(timeline_by_offset.get(offset) or {
            "videoId": str(video.get("id") or ""),
            "offsetSeconds": offset,
            "videoTime": _format_video_offset_clock(offset),
            "observedAt": None,
            "clockTime": None,
            "visiblePedestrians": 0,
            "detectedNow": 0,
            "cumulativeUniquePedestrians": 0,
            "totalPedestriansSoFar": 0,
            "lightOcclusionCount": 0,
            "moderateOcclusionCount": 0,
            "heavyOcclusionCount": 0,
            "occlusionValue": 0.0,
            "ptsiScore": 0.0,
            "los": None,
            "losDescription": None,
            "severity": "neutral",
            "mode": None,
            "walkableAreaM2": None,
            "roiAreaRatio": None,
            "capacityProxy": None,
            "congestionScore": 0.0,
            "spacePerPedestrianM2": None,
        })

        overlay = by_offset.get(offset)
        if overlay is not None:
            for key, value in overlay.items():
                base[key] = value

        if observed_at is not None:
            sample_timestamp = (observed_at + timedelta(seconds=offset)).replace(microsecond=0)
            base["observedAt"] = sample_timestamp.isoformat()
            base["clockTime"] = sample_timestamp.strftime("%H:%M:%S")

        running_total = max(running_total, int(base.get("totalPedestriansSoFar") or base.get("cumulativeUniquePedestrians") or 0))
        base["cumulativeUniquePedestrians"] = running_total
        base["totalPedestriansSoFar"] = running_total
        merged_rows.append(base)

    return merged_rows


def _event_offset_seconds(event: dict[str, Any]) -> Optional[float]:
    try:
        raw_offset = event.get("offsetSeconds")
        if raw_offset is not None:
            return max(0.0, float(raw_offset))
    except (TypeError, ValueError):
        pass

    parsed_clock = _parse_clock_time(_optional_string(event.get("timestamp")))
    if parsed_clock is None:
        return None

    return float((parsed_clock.hour * 3600) + (parsed_clock.minute * 60) + parsed_clock.second)


def _normalize_vehicle_class_name(value: Any) -> Optional[str]:
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


def _vehicle_class_label(class_name: Any) -> Optional[str]:
    normalized = _normalize_vehicle_class_name(class_name)
    if not normalized:
        return None
    return " ".join(part.capitalize() for part in normalized.replace("_", "-").split("-") if part)


def _infer_vehicle_class_from_description(description: Any) -> Optional[str]:
    text = str(description or "").strip().lower()
    if not text:
        return None

    matchers: list[tuple[str, re.Pattern[str]]] = [
        ("jeepney", re.compile(r"\bjeepney\b", re.IGNORECASE)),
        ("tricycle", re.compile(r"\btricycle\b", re.IGNORECASE)),
        ("motorcycle", re.compile(r"\bmotorcycle\b|\bmotorbike\b", re.IGNORECASE)),
        ("bicycle", re.compile(r"\bbicycle\b|\bbike\b", re.IGNORECASE)),
        ("truck", re.compile(r"\btruck\b", re.IGNORECASE)),
        ("bus", re.compile(r"\bbus\b", re.IGNORECASE)),
        ("van", re.compile(r"\bvan\b", re.IGNORECASE)),
        ("suv", re.compile(r"\bsuv\b", re.IGNORECASE)),
        ("car", re.compile(r"\bcar\b|\bauto\b|\bautomobile\b", re.IGNORECASE)),
    ]
    for class_name, pattern in matchers:
        if pattern.search(text):
            return class_name

    return None


def _video_detail_tracks_from_events(state: dict[str, Any], video_id: str) -> list[dict[str, Any]]:
    windows_by_pedestrian: dict[int, dict[str, float]] = {}

    for event in state.get("events", []):
        if event.get("videoId") != video_id:
            continue
        if str(event.get("type") or "").lower() != "detection":
            continue

        try:
            pedestrian_id = int(event.get("pedestrianId"))
        except (TypeError, ValueError):
            continue

        offset = _event_offset_seconds(event)
        if offset is None:
            continue

        current_window = windows_by_pedestrian.get(pedestrian_id)
        if current_window is None:
            windows_by_pedestrian[pedestrian_id] = {"start": offset, "end": offset}
            continue

        current_window["start"] = min(current_window["start"], offset)
        current_window["end"] = max(current_window["end"], offset)

    compact_tracks: list[dict[str, Any]] = []
    for index, (pedestrian_id, window) in enumerate(sorted(windows_by_pedestrian.items(), key=lambda item: item[0])):
        compact_tracks.append(
            {
                "id": f"{video_id}-event-track-{index}",
                "pedestrianId": pedestrian_id,
                "firstOffsetSeconds": window["start"],
                "lastOffsetSeconds": max(window["end"], window["start"] + 0.5),
            }
        )

    return compact_tracks


def _read_portable_manifest() -> dict[str, Any]:
    ensure_storage_layout()
    if not PORTABLE_MANIFEST_FILE.exists():
        payload: dict[str, Any] = {}
    else:
        try:
            payload = json.loads(PORTABLE_MANIFEST_FILE.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            logger.exception("Failed to read portable manifest from %s", PORTABLE_MANIFEST_FILE)
            payload = {}

    videos = payload.get("videos") if isinstance(payload, dict) else None
    return {
        "generatedAt": _optional_string(payload.get("generatedAt") if isinstance(payload, dict) else None) or _utc_timestamp(),
        "portableRoot": str(PORTABLE_DIR.relative_to(BACKEND_DIR)),
        "videos": videos if isinstance(videos, list) else [],
    }


def _write_portable_manifest_video_entry(entry: dict[str, Any]) -> None:
    manifest = _read_portable_manifest()
    videos = [item for item in manifest.get("videos", []) if str(item.get("videoId") or "") != str(entry.get("videoId") or "")]
    videos.append(entry)
    videos.sort(
        key=lambda item: (
            str(item.get("date") or ""),
            str(item.get("startTime") or ""),
            str(item.get("locationName") or ""),
            str(item.get("videoId") or ""),
        )
    )
    manifest["generatedAt"] = _utc_timestamp()
    manifest["videos"] = videos
    _write_json_file(PORTABLE_MANIFEST_FILE, manifest)


def _remove_portable_video_artifacts(video_id: Optional[str]) -> None:
    normalized_video_id = str(video_id or "").strip()
    if not normalized_video_id:
        return

    shutil.rmtree(_portable_video_directory(normalized_video_id), ignore_errors=True)

    manifest = _read_portable_manifest()
    manifest["generatedAt"] = _utc_timestamp()
    manifest["videos"] = [
        item for item in manifest.get("videos", []) if str(item.get("videoId") or "") != normalized_video_id
    ]
    _write_json_file(PORTABLE_MANIFEST_FILE, manifest)


def _video_event_rows(video_events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ordered_events = sorted(
        (deepcopy(event) for event in video_events),
        key=lambda item: (
            float(item.get("offsetSeconds") or 0.0),
            str(item.get("timestamp") or ""),
            str(item.get("id") or ""),
        ),
    )
    rows: list[dict[str, Any]] = []
    for event in ordered_events:
        rows.append(
            {
                "eventId": event.get("id"),
                "videoId": event.get("videoId"),
                "type": event.get("type"),
                "location": event.get("location"),
                "clockTime": event.get("timestamp"),
                "offsetSeconds": event.get("offsetSeconds"),
                "videoTime": _format_video_offset_clock(event.get("offsetSeconds")),
                "pedestrianId": event.get("pedestrianId"),
                "frame": event.get("frame"),
                "occlusionClass": event.get("occlusionClass"),
                "occlusionLabel": _occlusion_label(event.get("occlusionClass")),
                "vehicleClass": event.get("vehicleClass"),
                "vehicleClassLabel": event.get("vehicleClassLabel") or _vehicle_class_label(event.get("vehicleClass")),
                "description": event.get("description"),
            }
        )
    return rows


def _whole_footage_log_rows(
    video_id: str,
    timeline_rows: list[dict[str, Any]],
    second_metrics: dict[int, dict[str, Optional[int]]],
    track_identity_by_key: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for timeline_row in timeline_rows:
        offset_second = max(0, int(timeline_row.get("offsetSeconds") or 0))
        second_tracks = second_metrics.get(offset_second, {})
        base_row = {
            "videoId": video_id,
            "offsetSeconds": offset_second,
            "videoTime": timeline_row.get("videoTime"),
            "observedAt": timeline_row.get("observedAt"),
            "clockTime": timeline_row.get("clockTime"),
            "visiblePedestrians": timeline_row.get("visiblePedestrians"),
            "detectedNow": timeline_row.get("detectedNow"),
            "cumulativeUniquePedestrians": timeline_row.get("cumulativeUniquePedestrians"),
            "totalPedestriansSoFar": timeline_row.get("totalPedestriansSoFar"),
            "lightOcclusionCount": timeline_row.get("lightOcclusionCount"),
            "moderateOcclusionCount": timeline_row.get("moderateOcclusionCount"),
            "heavyOcclusionCount": timeline_row.get("heavyOcclusionCount"),
        }

        if not second_tracks:
            rows.append(
                {
                    **base_row,
                    "trackKey": None,
                    "trackId": None,
                    "pedestrianId": None,
                    "occlusionClass": None,
                    "occlusionLabel": "",
                    "insideROI": False,
                }
            )
            continue

        for track_key, occlusion_class in sorted(second_tracks.items()):
            identity = track_identity_by_key.get(track_key, {})
            rows.append(
                {
                    **base_row,
                    "trackKey": track_key,
                    "trackId": identity.get("trackId"),
                    "pedestrianId": identity.get("pedestrianId"),
                    "occlusionClass": occlusion_class,
                    "occlusionLabel": _occlusion_label(occlusion_class),
                    "insideROI": True,
                }
            )

    return rows


def _write_portable_video_artifacts(state: dict[str, Any], video: dict[str, Any]) -> None:
    video_id = str(video.get("id") or "").strip()
    if not video_id:
        return

    ensure_storage_layout()
    video_dir = _portable_video_directory(video_id)
    video_dir.mkdir(parents=True, exist_ok=True)

    location = next((item for item in state.get("locations", []) if item.get("id") == video.get("locationId")), {})
    video_events = [event for event in state.get("events", []) if event.get("videoId") == video_id]
    video_tracks = [track for track in state.get("pedestrianTracks", []) if track.get("videoId") == video_id]
    track_windows = {
        str(item.get("id") or ""): item for item in _video_detail_pedestrian_tracks(state, video_id) if item.get("id")
    }
    severity_summary = _video_severity_summary(state, video)
    observed_at = _observation_time(video)
    duration_seconds = _video_duration_seconds(video, video_tracks)

    second_metrics: dict[int, dict[str, Optional[int]]] = {}
    first_seen_by_track: dict[str, int] = {}
    track_identity_by_key: dict[str, dict[str, Any]] = {}
    trajectory_rows: list[dict[str, Any]] = []
    tracks_rows: list[dict[str, Any]] = []

    for fallback_index, track in enumerate(video_tracks):
        track_id = str(track.get("id") or f"{video_id}-track-{fallback_index}")
        track_key = _tracked_pedestrian_track_key(track, video_id, fallback_index)
        track_identity_by_key[track_key] = {
            "trackId": track_id,
            "pedestrianId": track.get("pedestrianId"),
        }
        samples = _normalized_trajectory_samples(track)
        sample_offsets = [float(offset_second) for offset_second, _point, _occlusion_class in samples]
        roi_qualified_offsets: list[int] = []
        max_occlusion_class: Optional[int] = None

        for offset_second, point, occlusion_class in samples:
            normalized_second = max(0, int(offset_second))
            inside_roi = _point_in_location_roi(point, location)
            if occlusion_class is not None and (max_occlusion_class is None or int(occlusion_class) > int(max_occlusion_class)):
                max_occlusion_class = int(occlusion_class)

            sample_timestamp = (observed_at + timedelta(seconds=normalized_second)).replace(microsecond=0) if observed_at is not None else None
            trajectory_rows.append(
                {
                    "videoId": video_id,
                    "trackKey": track_key,
                    "trackId": track_id,
                    "pedestrianId": track.get("pedestrianId"),
                    "offsetSeconds": normalized_second,
                    "videoTime": _format_video_offset_clock(normalized_second),
                    "observedAt": sample_timestamp.isoformat() if sample_timestamp is not None else None,
                    "clockTime": sample_timestamp.strftime("%H:%M:%S") if sample_timestamp is not None else None,
                    "xNorm": round(point[0], 6),
                    "yNorm": round(point[1], 6),
                    "occlusionClass": occlusion_class,
                    "occlusionLabel": _occlusion_label(occlusion_class),
                    "insideROI": inside_roi,
                }
            )

            if not inside_roi:
                continue

            roi_qualified_offsets.append(normalized_second)
            first_seen_by_track[track_key] = min(first_seen_by_track.get(track_key, normalized_second), normalized_second)

            second_tracks = second_metrics.setdefault(normalized_second, {})
            existing_occlusion = second_tracks.get(track_key)
            if existing_occlusion is None or (
                occlusion_class is not None and (existing_occlusion is None or int(occlusion_class) > int(existing_occlusion))
            ):
                second_tracks[track_key] = occlusion_class

        compact_track = track_windows.get(track_id, {})
        first_offset = compact_track.get("firstOffsetSeconds")
        last_offset = compact_track.get("lastOffsetSeconds")
        if first_offset is None and sample_offsets:
            first_offset = min(sample_offsets)
        if last_offset is None and sample_offsets:
            last_offset = max(sample_offsets)

        first_timestamp = _pedestrian_track_timestamp(track, video)
        last_timestamp = _pedestrian_track_end_timestamp(track, video)
        tracks_rows.append(
            {
                "videoId": video_id,
                "trackKey": track_key,
                "trackId": track_id,
                "pedestrianId": track.get("pedestrianId"),
                "firstOffsetSeconds": first_offset,
                "lastOffsetSeconds": last_offset,
                "bestOffsetSeconds": track.get("bestOffsetSeconds"),
                "firstObservedAt": first_timestamp.isoformat() if first_timestamp is not None else None,
                "lastObservedAt": last_timestamp.isoformat() if last_timestamp is not None else None,
                "firstTimestamp": track.get("firstTimestamp"),
                "lastTimestamp": track.get("lastTimestamp"),
                "bestTimestamp": track.get("bestTimestamp"),
                "sampleCount": len(samples),
                "roiQualifiedSampleCount": len(roi_qualified_offsets),
                "maxOcclusionClass": max_occlusion_class,
                "thumbnailPath": track.get("thumbnailPath"),
                "previewPath": track.get("previewPath"),
                "appearanceSummary": track.get("appearanceSummary"),
            }
        )

    new_tracks_by_second: dict[int, int] = {}
    for first_seen_second in first_seen_by_track.values():
        new_tracks_by_second[first_seen_second] = new_tracks_by_second.get(first_seen_second, 0) + 1

    timeline_rows: list[dict[str, Any]] = []
    cumulative_unique = 0
    for offset_second in range(max(1, duration_seconds)):
        cumulative_unique += new_tracks_by_second.get(offset_second, 0)
        track_occlusions = second_metrics.get(offset_second, {})
        visible_count = len(track_occlusions)
        light_count = sum(1 for value in track_occlusions.values() if value == 0)
        moderate_count = sum(1 for value in track_occlusions.values() if value == 1)
        heavy_count = sum(1 for value in track_occlusions.values() if value == 2)
        occlusion_value = 0.0
        ptsi_breakdown: dict[str, Any] = {
            "mode": _location_ptsi_mode(location),
            "walkableAreaM2": _location_walkable_area_m2(location),
            "roiAreaRatio": round(_location_roi_area_ratio(location), 6),
            "capacityProxy": round(_location_capacity_proxy(location), 3) if location else None,
            "congestionScore": 0.0,
            "occlusionValue": 0.0,
            "spacePerPedestrian": None,
            "los": None,
            "losDescription": None,
            "score": 0.0,
        }
        severity = "neutral"

        if visible_count > 0:
            occlusion_value = (
                (light_count * PTSI_OCCLUSION_WEIGHTS[0])
                + (moderate_count * PTSI_OCCLUSION_WEIGHTS[1])
                + (heavy_count * PTSI_OCCLUSION_WEIGHTS[2])
            ) / (3.0 * visible_count)
            ptsi_breakdown = _ptsi_score_breakdown(visible_count, location, occlusion_value)
            severity = _timeline_severity_from_score(float(ptsi_breakdown["score"]))

        sample_timestamp = (observed_at + timedelta(seconds=offset_second)).replace(microsecond=0) if observed_at is not None else None
        timeline_rows.append(
            {
                "videoId": video_id,
                "offsetSeconds": offset_second,
                "videoTime": _format_video_offset_clock(offset_second),
                "observedAt": sample_timestamp.isoformat() if sample_timestamp is not None else None,
                "clockTime": sample_timestamp.strftime("%H:%M:%S") if sample_timestamp is not None else None,
                "visiblePedestrians": visible_count,
                "detectedNow": visible_count,
                "cumulativeUniquePedestrians": cumulative_unique,
                "totalPedestriansSoFar": cumulative_unique,
                "lightOcclusionCount": light_count,
                "moderateOcclusionCount": moderate_count,
                "heavyOcclusionCount": heavy_count,
                "occlusionValue": round(occlusion_value, 4),
                "ptsiScore": ptsi_breakdown["score"],
                "los": ptsi_breakdown["los"],
                "losDescription": ptsi_breakdown["losDescription"],
                "severity": severity,
                "mode": ptsi_breakdown["mode"],
                "walkableAreaM2": ptsi_breakdown["walkableAreaM2"],
                "roiAreaRatio": ptsi_breakdown["roiAreaRatio"],
                "capacityProxy": ptsi_breakdown["capacityProxy"],
                "congestionScore": ptsi_breakdown["congestionScore"],
                "spacePerPedestrianM2": ptsi_breakdown["spacePerPedestrian"],
            }
        )

    timeline_rows = _merge_congestion_timeline_rows(
        video=video,
        observed_at=observed_at,
        timeline_rows=timeline_rows,
    )

    if not _severity_summary_has_signal(severity_summary):
        inferred_summary = _severity_summary_from_timeline_rows(timeline_rows)
        if inferred_summary is not None:
            severity_summary = inferred_summary

    whole_footage_rows = _whole_footage_log_rows(video_id, timeline_rows, second_metrics, track_identity_by_key)

    severity_rows = [
        {
            "startOffsetSeconds": bucket.get("startOffsetSeconds"),
            "endOffsetSeconds": bucket.get("endOffsetSeconds"),
            "severity": bucket.get("severity"),
            "score": bucket.get("score"),
        }
        for bucket in severity_summary.get("buckets", [])
    ]
    event_rows = _video_event_rows(video_events)

    metadata_payload = {
        "generatedAt": _utc_timestamp(),
        "video": deepcopy(video),
        "location": _location_payload(location) if location else None,
        "counts": {
            "pedestrianCount": int(video.get("pedestrianCount") or 0),
            "eventCount": len(event_rows),
            "trackCount": len(tracks_rows),
            "trajectorySampleCount": len(trajectory_rows),
            "timelineRowCount": len(timeline_rows),
            "wholeFootageLogRowCount": len(whole_footage_rows),
            "severityBucketCount": len(severity_rows),
        },
        "severitySummary": severity_summary,
    }

    artifacts = {
        "metadataJson": video_dir / "metadata.json",
        "manifestJson": video_dir / "manifest.json",
        "eventsCsv": video_dir / "events.csv",
        "eventsJson": video_dir / "events.json",
        "tracksCsv": video_dir / "tracks.csv",
        "tracksJson": video_dir / "tracks.json",
        "trajectorySamplesCsv": video_dir / "trajectory_samples.csv",
        "trajectorySamplesJson": video_dir / "trajectory_samples.json",
        "timelineCsv": video_dir / "timeline.csv",
        "timelineJson": video_dir / "timeline.json",
        "wholeFootageLogCsv": video_dir / "whole_footage_log.csv",
        "wholeFootageLogJson": video_dir / "whole_footage_log.json",
        "severityBucketsCsv": video_dir / "severity_buckets.csv",
        "severityBucketsJson": video_dir / "severity_buckets.json",
    }

    _write_json_file(artifacts["metadataJson"], metadata_payload)
    _write_csv_rows(artifacts["eventsCsv"], event_rows)
    _write_json_file(artifacts["eventsJson"], event_rows)
    _write_csv_rows(artifacts["tracksCsv"], tracks_rows)
    _write_json_file(artifacts["tracksJson"], tracks_rows)
    _write_csv_rows(artifacts["trajectorySamplesCsv"], trajectory_rows)
    _write_json_file(artifacts["trajectorySamplesJson"], trajectory_rows)
    _write_csv_rows(artifacts["timelineCsv"], timeline_rows)
    _write_json_file(artifacts["timelineJson"], timeline_rows)
    _write_csv_rows(artifacts["wholeFootageLogCsv"], whole_footage_rows)
    _write_json_file(artifacts["wholeFootageLogJson"], whole_footage_rows)
    _write_csv_rows(artifacts["severityBucketsCsv"], severity_rows)
    _write_json_file(artifacts["severityBucketsJson"], severity_rows)

    video_manifest = {
        "generatedAt": metadata_payload["generatedAt"],
        "videoId": video_id,
        "label": f"{video.get('location') or 'Unknown Location'} · {video.get('date') or ''} · {video.get('startTime') or video.get('timestamp') or ''}",
        "directory": _portable_relative_path(video_dir),
        "artifacts": {name: _portable_relative_path(path) for name, path in artifacts.items()},
    }
    _write_json_file(artifacts["manifestJson"], video_manifest)

    _write_portable_manifest_video_entry(
        {
            "videoId": video_id,
            "locationId": video.get("locationId"),
            "locationName": video.get("location"),
            "date": video.get("date"),
            "startTime": video.get("startTime"),
            "endTime": video.get("endTime"),
            "directory": _portable_relative_path(video_dir),
            "generatedAt": metadata_payload["generatedAt"],
            "pedestrianCount": int(video.get("pedestrianCount") or 0),
            "artifacts": video_manifest["artifacts"],
        }
    )


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
        "model": {"currentModel": "yolov8n-bytetrack.pt", "uploadedAt": None, "inferConfig": None},
        "locations": [
            {
                "id": "gate-2",
                "name": "Gate 2",
                "latitude": 14.6358,
                "longitude": 121.07469,
                "description": "Approximate camera anchor for Gate 2 pedestrian flow.",
                "address": "Ateneo de Manila University · Gate 2",
                "roiCoordinates": None,
                "walkableAreaM2": None,
            },
            {
                "id": "gate-2-9",
                "name": "Gate 2.9",
                "latitude": 14.63667,
                "longitude": 121.07472,
                "description": "Approximate camera anchor for Gate 2.9 pedestrian flow.",
                "address": "Ateneo de Manila University · Gate 2.9",
                "roiCoordinates": deepcopy(DEFAULT_EDSA_SEC_WALK_ROI),
                "walkableAreaM2": None,
            },
            {
                "id": "gate-3",
                "name": "Gate 3",
                "latitude": 14.64028,
                "longitude": 121.07472,
                "description": "Approximate camera anchor for Gate 3 pedestrian flow.",
                "address": "Ateneo de Manila University · Gate 3",
                "roiCoordinates": None,
                "walkableAreaM2": None,
            },
            {
                "id": "gate-3-2",
                "name": "Gate 3.2",
                "latitude": 14.64055,
                "longitude": 121.07475,
                "description": "Approximate camera anchor for Gate 3.2 pedestrian flow.",
                "address": "Ateneo de Manila University · Gate 3.2",
                "roiCoordinates": None,
                "walkableAreaM2": None,
            },
            {
                "id": "gate-3-5",
                "name": "Gate 3.5",
                "latitude": 14.6409,
                "longitude": 121.0748,
                "description": "Approximate camera anchor for Gate 3.5 pedestrian flow.",
                "address": "Ateneo de Manila University · Gate 3.5",
                "roiCoordinates": None,
                "walkableAreaM2": None,
            },
        ],
        "videos": [],
        "events": [],
        "pedestrianTracks": [],
    }


def ensure_storage_layout() -> None:
    for path in (
        STORAGE_DIR,
        MODELS_DIR,
        LEGACY_MODELS_DIR,
        RAW_VIDEOS_DIR,
        PROCESSED_VIDEOS_DIR,
        EXPORTS_DIR,
        PORTABLE_DIR,
        PORTABLE_VIDEOS_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)
    if not DATA_FILE.exists():
        DATA_FILE.write_text(json.dumps(seed_state(), indent=2), encoding="utf-8")
    if not QUEUE_HISTORY_JSON_FILE.exists():
        _write_json_file(QUEUE_HISTORY_JSON_FILE, [])
    if not QUEUE_HISTORY_CSV_FILE.exists():
        _write_csv_rows(QUEUE_HISTORY_CSV_FILE, [], preferred=UPLOAD_HISTORY_CSV_FIELDS)
    if not PORTABLE_MANIFEST_FILE.exists():
        _write_json_file(
            PORTABLE_MANIFEST_FILE,
            {
                "generatedAt": _utc_timestamp(),
                "portableRoot": str(PORTABLE_DIR.relative_to(BACKEND_DIR)),
                "videos": [],
            },
        )


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
        if location.get("id") == "gate-2-9" and not location.get("roiCoordinates"):
            location["roiCoordinates"] = deepcopy(DEFAULT_EDSA_SEC_WALK_ROI)
            changed = True

    model_state = state.get("model")
    if isinstance(model_state, dict) and "inferConfig" not in model_state:
        model_state["inferConfig"] = None
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
        _remove_portable_video_artifacts(str(video_id))


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


def _track_window_offset(value: Any) -> Optional[float]:
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        return None


def _video_detail_pedestrian_tracks(state: dict[str, Any], video_id: str) -> list[dict[str, Any]]:
    compact_tracks: list[dict[str, Any]] = []

    for fallback_index, track in enumerate(state.get("pedestrianTracks", [])):
        if track.get("videoId") != video_id:
            continue

        first_offset = _track_window_offset(track.get("firstOffsetSeconds"))
        last_offset = _track_window_offset(track.get("lastOffsetSeconds"))

        if first_offset is None or last_offset is None:
            sample_offsets = [float(offset_second) for offset_second, _point, _occlusion_class in _normalized_trajectory_samples(track)]
            if sample_offsets:
                if first_offset is None:
                    first_offset = min(sample_offsets)
                if last_offset is None:
                    last_offset = max(sample_offsets)

        if first_offset is None and last_offset is None:
            best_offset = _track_window_offset(track.get("bestOffsetSeconds"))
            if best_offset is None:
                continue
            first_offset = best_offset
            last_offset = best_offset
        elif first_offset is None:
            first_offset = last_offset
        elif last_offset is None:
            last_offset = first_offset

        if first_offset is None or last_offset is None:
            continue

        if last_offset < first_offset:
            first_offset, last_offset = last_offset, first_offset

        compact_tracks.append(
            {
                "id": str(track.get("id") or f"{video_id}-track-{fallback_index}"),
                "pedestrianId": track.get("pedestrianId"),
                "firstOffsetSeconds": first_offset,
                "lastOffsetSeconds": last_offset,
            }
        )

    return compact_tracks


def get_video_detail(video_id: str) -> Optional[dict[str, Any]]:
    state = load_state()
    video = next((item for item in state["videos"] if item["id"] == video_id), None)
    if video is None:
        return None

    detail = deepcopy(video)
    detail["severitySummary"] = _video_severity_summary(state, detail)
    if not _severity_summary_has_signal(detail.get("severitySummary")):
        portable_summary = _portable_video_severity_summary(video_id)
        if portable_summary is not None:
            detail["severitySummary"] = portable_summary

    detail["pedestrianTracks"] = _video_detail_pedestrian_tracks(state, video_id)
    if len(detail["pedestrianTracks"]) == 0:
        portable_tracks = _portable_video_tracks(video_id)
        if portable_tracks:
            detail["pedestrianTracks"] = portable_tracks
    if len(detail["pedestrianTracks"]) == 0:
        detail["pedestrianTracks"] = _video_detail_tracks_from_events(state, video_id)

    return detail


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
    end_time: Optional[str] = None,
) -> dict[str, Any]:
    state = load_state()
    video = next((item for item in state["videos"] if item["id"] == video_id), None)
    if video is None:
        raise ValueError("Video not found")

    video["pedestrianCount"] = pedestrian_count
    video["processedPath"] = processed_path
    if isinstance(end_time, str) and end_time.strip():
        video["endTime"] = end_time.strip()
    state["events"] = [event for event in state["events"] if event.get("videoId") != video_id]
    state["events"].extend(events)
    state["pedestrianTracks"] = [track for track in state.get("pedestrianTracks", []) if track.get("videoId") != video_id]
    state["pedestrianTracks"].extend(pedestrian_tracks or [])
    save_state(state)
    _write_portable_video_artifacts(state, video)
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
        _remove_portable_video_artifacts(video_id)
        _refresh_semantic_index(state)
    return removed


def list_events(video_id: Optional[str] = None) -> list[dict[str, Any]]:
    events = load_state()["events"]
    if video_id:
        filtered_events = [event for event in events if event.get("videoId") == video_id]
    else:
        filtered_events = events

    normalized_events: list[dict[str, Any]] = []
    for event in filtered_events:
        event_copy = deepcopy(event)
        inferred_offset = _event_offset_seconds(event_copy)
        if inferred_offset is not None:
            event_copy["offsetSeconds"] = inferred_offset

        normalized_vehicle_class = _normalize_vehicle_class_name(event_copy.get("vehicleClass"))
        if normalized_vehicle_class is None:
            normalized_vehicle_class = _infer_vehicle_class_from_description(event_copy.get("description"))

        event_copy["vehicleClass"] = normalized_vehicle_class
        event_copy["vehicleClassLabel"] = _vehicle_class_label(normalized_vehicle_class)
        normalized_events.append(event_copy)

    return normalized_events


def get_model_info() -> dict[str, Any]:
    return deepcopy(load_state()["model"])


def set_model(filename: str, infer_config: Optional[str] = None) -> dict[str, Any]:
    state = load_state()
    previous_model = state.get("model", {})
    next_infer_config = infer_config if infer_config is not None else previous_model.get("inferConfig")
    state["model"] = {
        "currentModel": filename,
        "uploadedAt": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "inferConfig": next_infer_config,
    }
    save_state(state)
    return deepcopy(state["model"])


def format_time_range_label(time_range: str) -> str:
    labels = {
        "12h": "12 hours",
        "6h": "6 hours",
        "4h": "4 hours",
        "3h": "3 hours",
        "2h": "2 hours",
        "1h": "1 hour",
        "30m": "30 minutes",
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


def _parse_non_negative_offset_seconds(value: Any) -> Optional[float]:
    try:
        offset_seconds = float(value)
    except (TypeError, ValueError):
        return None
    return max(0.0, offset_seconds)


def _video_end_time(video: dict[str, Any], video_start: Optional[datetime]) -> Optional[datetime]:
    video_end = _combine_date_and_time(str(video.get("date", "")), video.get("endTime"))
    if video_end is None:
        return None
    if video_start is not None and video_end < video_start:
        return video_end + timedelta(days=1)
    return video_end


def _is_likely_elapsed_counter_without_video_end(parsed_clock: datetime, event_time: datetime, video_start: datetime) -> bool:
    """Best-effort guard when endTime metadata is missing or invalid.

    With no known video end, "HH:MM:SS" can be ambiguous between:
    - wall-clock time (e.g., 08:15:00), and
    - elapsed counter encoded as clock text (e.g., 00:02:14).

    To avoid false reinterpretation of valid wall-clock values, only treat it as
    elapsed in this fallback path when the value strongly looks counter-like:
    - hour is exactly 0 (elapsed counters commonly start at 00:xx:xx), and
    - the wall-clock interpretation is implausibly far from video start.
    """

    if parsed_clock.hour != 0:
        return False

    return abs((event_time - video_start).total_seconds()) >= 6 * 3600


def _resolve_video_timestamp(
    raw_timestamp: Optional[str],
    video: dict[str, Any],
    *,
    explicit_offset_seconds: Optional[float],
    fallback_to_video_start_on_parse_error: bool = True,
) -> Optional[datetime]:
    video_start = _observation_time(video)

    # Prefer explicit offsets when available. They are already relative to video start and
    # avoid ambiguity between wall-clock times and elapsed counters.
    if video_start is not None and explicit_offset_seconds is not None:
        return (video_start + timedelta(seconds=explicit_offset_seconds)).replace(microsecond=0)

    event_time = _combine_date_and_time(str(video.get("date", "")), raw_timestamp)
    if event_time is None:
        if fallback_to_video_start_on_parse_error:
            return video_start
        return None

    if video_start is None:
        return event_time

    # Heuristic for detector outputs that emit elapsed counters (e.g. "00:00:04") instead
    # of wall-clock times. When the parsed wall-clock timestamp is implausibly far from the
    # video's known start, reinterpret HH:MM:SS as elapsed-from-start.
    if raw_timestamp and re.fullmatch(r"\d{1,2}:\d{2}:\d{2}", raw_timestamp):
        parsed_clock = _parse_clock_time(raw_timestamp)
        if parsed_clock is not None:
            elapsed_seconds = (parsed_clock.hour * 3600) + (parsed_clock.minute * 60) + parsed_clock.second
            elapsed_interpretation = (video_start + timedelta(seconds=elapsed_seconds)).replace(microsecond=0)

            # Parse end time when present so we can prefer the interpretation that lands
            # inside the known video window.
            video_end = _video_end_time(video, video_start)
            if video_end is not None:
                in_window_absolute = video_start <= event_time <= (video_end + timedelta(minutes=1))
                in_window_elapsed = video_start <= elapsed_interpretation <= (video_end + timedelta(minutes=1))
                if in_window_elapsed and not in_window_absolute:
                    return elapsed_interpretation

            # Fallback only when end metadata is unavailable and the value is strongly
            # counter-like. This keeps valid wall-clock HH:MM:SS values (for example
            # 08:15:00) from being broadly reinterpreted when endTime is missing/invalid.
            if _is_likely_elapsed_counter_without_video_end(parsed_clock, event_time, video_start):
                return elapsed_interpretation

    return event_time


def _event_timestamp(event: dict[str, Any], video: dict[str, Any]) -> Optional[datetime]:
    raw_timestamp = _optional_string(event.get("timestamp"))
    offset_seconds = _parse_non_negative_offset_seconds(event.get("offsetSeconds"))
    return _resolve_video_timestamp(raw_timestamp, video, explicit_offset_seconds=offset_seconds)


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
    first_timestamp = _optional_string(track.get("firstTimestamp"))
    first_offset_seconds = _parse_non_negative_offset_seconds(track.get("firstOffsetSeconds"))
    if first_timestamp is not None or first_offset_seconds is not None:
        resolved_first = _resolve_video_timestamp(
            first_timestamp,
            video,
            explicit_offset_seconds=first_offset_seconds,
            fallback_to_video_start_on_parse_error=False,
        )
        if resolved_first is not None:
            return resolved_first

    best_timestamp = _optional_string(track.get("bestTimestamp"))
    best_offset_seconds = _parse_non_negative_offset_seconds(track.get("bestOffsetSeconds"))
    if best_timestamp is not None or best_offset_seconds is not None:
        resolved_best = _resolve_video_timestamp(
            best_timestamp,
            video,
            explicit_offset_seconds=best_offset_seconds,
            fallback_to_video_start_on_parse_error=False,
        )
        if resolved_best is not None:
            return resolved_best

    return _observation_time(video)


def _pedestrian_track_end_timestamp(track: dict[str, Any], video: dict[str, Any]) -> Optional[datetime]:
    last_timestamp = _optional_string(track.get("lastTimestamp"))
    last_offset_seconds = _parse_non_negative_offset_seconds(track.get("lastOffsetSeconds"))
    if last_timestamp is not None or last_offset_seconds is not None:
        resolved_last = _resolve_video_timestamp(
            last_timestamp,
            video,
            explicit_offset_seconds=last_offset_seconds,
            fallback_to_video_start_on_parse_error=False,
        )
        if resolved_last is not None:
            return resolved_last

    best_timestamp = _optional_string(track.get("bestTimestamp"))
    best_offset_seconds = _parse_non_negative_offset_seconds(track.get("bestOffsetSeconds"))
    if best_timestamp is not None or best_offset_seconds is not None:
        resolved_best = _resolve_video_timestamp(
            best_timestamp,
            video,
            explicit_offset_seconds=best_offset_seconds,
            fallback_to_video_start_on_parse_error=False,
        )
        if resolved_best is not None:
            return resolved_best

    return _pedestrian_track_timestamp(track, video)


def _video_duration_seconds(video: dict[str, Any], pedestrian_tracks: list[dict[str, Any]]) -> int:
    start_time = _observation_time(video)
    end_time = _combine_date_and_time(str(video.get("date", "")), video.get("endTime"))

    if start_time is not None and end_time is not None:
        if end_time < start_time:
            end_time += timedelta(days=1)
        duration_seconds = int(math.ceil((end_time - start_time).total_seconds()))
        if duration_seconds > 0:
            return duration_seconds

    max_offset_seconds = 0
    for track in pedestrian_tracks:
        for offset_second, _point, _occlusion_class in _normalized_trajectory_samples(track):
            max_offset_seconds = max(max_offset_seconds, int(offset_second) + 1)

    return max(1, max_offset_seconds)


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
    start_time: Optional[str] = None,
) -> tuple[datetime, datetime, int]:
    day_start = datetime.strptime(date_value, "%Y-%m-%d")
    day_end = day_start + timedelta(days=1)

    duration_by_time_range = {
        "12h": 12 * 60,
        "6h": 6 * 60,
        "4h": 4 * 60,
        "3h": 3 * 60,
        "2h": 2 * 60,
        "1h": 1 * 60,
        "30m": 30,
    }
    if time_range in duration_by_time_range:
        parsed_start = _combine_date_and_time(date_value, start_time)
        start = parsed_start or day_start
        start = min(max(start, day_start), day_end)
        duration_minutes = duration_by_time_range[time_range]
        window_end = min(day_end, start + timedelta(minutes=duration_minutes))
        window_minutes = max(int((window_end - start).total_seconds() // 60), 1)
        step_minutes = max(1, math.ceil(window_minutes / 6))
        return start, window_end, step_minutes

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
    start_time: Optional[str] = None,
) -> tuple[list[tuple[str, datetime]], timedelta, dict[str, Any]]:
    day_start = datetime.strptime(date_value, "%Y-%m-%d")
    day_end = day_start + timedelta(days=1)

    root_window_start, root_window_end, root_bucket_minutes = _resolve_root_window(date_value, time_range, observation_times, start_time)
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


IN_GATE_LOCATION_IDS = {"gate-2", "gate-3"}
OUT_GATE_LOCATION_IDS = {"gate-2-9", "gate-3-2", "gate-3-5"}


def _normalize_gate_name(value: str) -> str:
    return re.sub(r"[^a-z0-9.]", "", str(value or "").strip().lower())


def _gate_flow_group_from_location_id(location_id: str) -> Optional[str]:
    normalized_id = str(location_id or "").strip().casefold()
    if normalized_id in IN_GATE_LOCATION_IDS:
        return "In"
    if normalized_id in OUT_GATE_LOCATION_IDS:
        return "Out"
    return None


def _gate_flow_group_from_location_name(location_name: str) -> Optional[str]:
    normalized_name = _normalize_gate_name(location_name)
    if normalized_name in {"gate2", "gate3"}:
        return "In"
    if normalized_name in {"gate2.9", "gate3.2", "gate3.5"}:
        return "Out"
    return None


def _track_video_id(track_key: str) -> Optional[str]:
    video_id, _separator, _suffix = str(track_key or "").partition(":")
    normalized_video_id = video_id.strip()
    return normalized_video_id or None


def _in_and_out_series_from_first_seen(
    buckets: list[tuple[str, datetime]],
    bucket_span: timedelta,
    first_seen_by_track: dict[str, tuple[datetime, str]],
    root_window_start: datetime,
    videos_by_id: dict[str, dict[str, Any]],
    footage_bucket_indexes: Optional[set[int]] = None,
) -> list[dict[str, Union[int, str, None]]]:
    if not buckets:
        return []

    series: list[dict[str, Union[int, str]]] = [
        {
            "id": bucket_start.isoformat(),
            "time": label,
            "In": 0,
            "Out": 0,
        }
        for label, bucket_start in buckets
    ]

    first_bucket = buckets[0][1]
    bucket_seconds = bucket_span.total_seconds()
    final_boundary = buckets[-1][1] + bucket_span
    baseline_counts = {"In": 0, "Out": 0}
    bucket_counts = {
        "In": [0 for _ in series],
        "Out": [0 for _ in series],
    }

    for track_key, (observed_at, location_name) in first_seen_by_track.items():
        video_id = _track_video_id(track_key)
        location_id = str((videos_by_id.get(video_id or "") or {}).get("locationId") or "")
        flow_group = _gate_flow_group_from_location_id(location_id)
        if flow_group is None:
            flow_group = _gate_flow_group_from_location_name(location_name)
        if flow_group is None:
            continue
        if observed_at < root_window_start or observed_at >= final_boundary:
            continue
        if observed_at < first_bucket:
            baseline_counts[flow_group] += 1
            continue

        bucket_index = _bucket_index(observed_at, first_bucket, bucket_seconds, len(series))
        if bucket_index is None:
            continue
        bucket_counts[flow_group][bucket_index] += 1

    running_in = baseline_counts["In"]
    running_out = baseline_counts["Out"]
    coverage_end_index = max(footage_bucket_indexes) if footage_bucket_indexes else None
    for index, point in enumerate(series):
        if coverage_end_index is not None and index > coverage_end_index:
            point["In"] = None
            point["Out"] = None
            continue

        running_in += bucket_counts["In"][index]
        running_out += bucket_counts["Out"][index]
        point["In"] = running_in
        point["Out"] = running_out

    return series


def _los_rank_from_los(los: Optional[str]) -> float:
    if los is None:
        return 0.0
    return float(PTSI_LOS_RANKS.get(los, 0) + 1)


def _los_series_from_samples(
    buckets: list[tuple[str, datetime]],
    bucket_span: timedelta,
    samples: list[dict[str, Any]],
    location: dict[str, Any],
    footage_bucket_indexes: Optional[set[int]] = None,
) -> list[dict[str, Union[float, str, None]]]:
    if not buckets:
        return []

    location_name = str(location.get("name") or "")
    series: list[dict[str, Union[float, str, None]]] = [
        {
            "id": bucket_start.isoformat(),
            "time": label,
            "los": None,
        }
        for label, bucket_start in buckets
    ]

    first_bucket = buckets[0][1]
    bucket_seconds = bucket_span.total_seconds()
    final_boundary = buckets[-1][1] + bucket_span

    visible_totals = [0.0 for _ in series]
    sample_counts = [0 for _ in series]
    light_totals = [0.0 for _ in series]
    moderate_totals = [0.0 for _ in series]
    heavy_totals = [0.0 for _ in series]
    coverage_end_index = max(footage_bucket_indexes) if footage_bucket_indexes else None

    for sample in samples:
        if str(sample.get("location") or "") != location_name:
            continue

        observed_at = sample["observedAt"]
        if observed_at < first_bucket or observed_at >= final_boundary:
            continue
        bucket_index = _bucket_index(observed_at, first_bucket, bucket_seconds, len(series))
        if bucket_index is None:
            continue

        visible_count = float(sample.get("visibleCount") or 0)
        class_counts = sample.get("classCounts") or {}
        sample_counts[bucket_index] += 1
        visible_totals[bucket_index] += visible_count
        light_totals[bucket_index] += float(class_counts.get(0) or 0)
        moderate_totals[bucket_index] += float(class_counts.get(1) or 0)
        heavy_totals[bucket_index] += float(class_counts.get(2) or 0)

    for index, point in enumerate(series):
        if coverage_end_index is not None and index > coverage_end_index:
            point["los"] = None
            continue

        if sample_counts[index] == 0:
            point["los"] = None
            continue

        average_visible = visible_totals[index] / sample_counts[index]
        rounded_visible = max(1, int(round(average_visible)))
        occlusion_value = 0.0
        if visible_totals[index] > 0:
            occlusion_value = (
                (light_totals[index] * PTSI_OCCLUSION_WEIGHTS[0])
                + (moderate_totals[index] * PTSI_OCCLUSION_WEIGHTS[1])
                + (heavy_totals[index] * PTSI_OCCLUSION_WEIGHTS[2])
            ) / (3.0 * visible_totals[index])

        los = _ptsi_score_breakdown(rounded_visible, location, occlusion_value).get("los")
        point["los"] = _los_rank_from_los(los)

    return series


def _footage_bucket_coverage(
    videos: list[dict[str, Any]],
    buckets: list[tuple[str, datetime]],
    bucket_span: timedelta,
    location_names: Optional[list[str]] = None,
) -> tuple[set[int], dict[str, set[int]]]:
    if not buckets:
        return set(), {}

    first_bucket = buckets[0][1]
    bucket_seconds = bucket_span.total_seconds()
    final_boundary = buckets[-1][1] + bucket_span

    location_coverage = {location: set() for location in (location_names or [])}
    all_coverage: set[int] = set()

    for video in videos:
        video_start = _observation_time(video)
        if video_start is None:
            continue

        video_end = _video_end_time(video, video_start)
        if video_end is None:
            video_end = final_boundary
        elif video_end <= video_start:
            video_end = video_start + timedelta(seconds=1)

        if video_end <= first_bucket or video_start >= final_boundary:
            continue

        start_index = _bucket_index(video_start, first_bucket, bucket_seconds, len(buckets))
        if start_index is None:
            start_index = 0 if video_start < first_bucket else len(buckets) - 1

        end_index = _bucket_index(
            video_end - timedelta(microseconds=1),
            first_bucket,
            bucket_seconds,
            len(buckets),
        )
        if end_index is None:
            end_index = len(buckets) - 1 if video_end >= final_boundary else 0

        if end_index < start_index:
            continue

        covered_indexes = set(range(start_index, end_index + 1))
        all_coverage.update(covered_indexes)

        location_name = str(video.get("location") or "")
        if location_names is not None and location_name not in location_coverage:
            continue

        location_coverage.setdefault(location_name, set()).update(covered_indexes)

    return all_coverage, location_coverage


def _dashboard_unique_pedestrian_rows(first_seen_by_track: dict[str, tuple[datetime, str]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    ordered_entries = sorted(first_seen_by_track.items(), key=lambda item: (item[1][0], item[1][1], item[0]))

    for track_key, (observed_at, location) in ordered_entries:
        video_id, _, track_suffix = str(track_key).partition(":")
        pedestrian_id: Optional[int] = None
        source_type = "trackedPedestrian"

        if track_suffix.isdigit():
            pedestrian_id = int(track_suffix)
            source_type = "pedestrianId"
        elif track_suffix.startswith("track:"):
            source_type = "trackId"
        elif track_suffix.startswith("fallback:") or track_suffix.startswith("track-fallback:"):
            source_type = "fallbackCount"

        rows.append(
            {
                "trackKey": track_key,
                "videoId": video_id or None,
                "location": location,
                "firstSeenAt": observed_at.isoformat(),
                "firstSeenDate": observed_at.strftime("%Y-%m-%d"),
                "firstSeenTime": observed_at.strftime("%H:%M:%S"),
                "pedestrianId": pedestrian_id,
                "sourceType": source_type,
                "countedInDashboardTotal": True,
            }
        )

    return rows


def _traffic_series_from_samples(
    buckets: list[tuple[str, datetime]],
    bucket_span: timedelta,
    samples: list[dict[str, Any]],
    first_seen_by_track: dict[str, tuple[datetime, str]],
    _root_window_start: datetime,
    location_names: list[str],
    active_locations_by_name: Optional[dict[str, dict[str, Any]]] = None,
    footage_bucket_indexes: Optional[set[int]] = None,
    footage_bucket_indexes_by_location: Optional[dict[str, set[int]]] = None,
) -> list[dict[str, Union[float, int, str, None]]]:
    if not buckets:
        return []

    include_location_los = bool(active_locations_by_name)

    series: list[dict[str, Union[float, int, str, None]]] = [
        {
            "id": bucket_start.isoformat(),
            "time": label,
            "cumulativeUniquePedestrians": 0,
            "averageVisiblePedestrians": 0.0,
            **{location: 0 for location in location_names},
            **({f"{location}__los": None for location in location_names} if include_location_los else {}),
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
    location_visible_totals = {location: [0.0 for _ in series] for location in location_names}
    location_sample_counts = {location: [0 for _ in series] for location in location_names}
    location_light_totals = {location: [0.0 for _ in series] for location in location_names}
    location_moderate_totals = {location: [0.0 for _ in series] for location in location_names}
    location_heavy_totals = {location: [0.0 for _ in series] for location in location_names}
    covered_bucket_indexes = footage_bucket_indexes if footage_bucket_indexes is not None else None
    covered_bucket_indexes_by_location = (
        footage_bucket_indexes_by_location if footage_bucket_indexes_by_location is not None else None
    )

    for sample in samples:
        observed_at = sample["observedAt"]
        if observed_at < first_bucket or observed_at >= final_boundary:
            continue
        bucket_index = _bucket_index(observed_at, first_bucket, bucket_seconds, len(series))
        if bucket_index is None:
            continue
        visible_totals[bucket_index] += float(sample["visibleCount"])
        sample_counts[bucket_index] += 1

        sample_location = str(sample.get("location") or "")
        if sample_location not in location_sample_counts:
            continue

        sample_visible_count = float(sample.get("visibleCount") or 0)
        sample_class_counts = sample.get("classCounts") or {}
        location_visible_totals[sample_location][bucket_index] += sample_visible_count
        location_sample_counts[sample_location][bucket_index] += 1
        location_light_totals[sample_location][bucket_index] += float(sample_class_counts.get(0) or 0)
        location_moderate_totals[sample_location][bucket_index] += float(sample_class_counts.get(1) or 0)
        location_heavy_totals[sample_location][bucket_index] += float(sample_class_counts.get(2) or 0)

    baseline_unique_total = 0
    baseline_unique_by_location = {location: 0 for location in location_names}
    effective_baseline_start = first_bucket
    for observed_at, location in first_seen_by_track.values():
        if observed_at < effective_baseline_start or observed_at >= final_boundary:
            continue
        bucket_index = _bucket_index(observed_at, first_bucket, bucket_seconds, len(series))
        if bucket_index is not None:
            first_seen_counts[bucket_index] += 1
            if location in first_seen_counts_by_location:
                first_seen_counts_by_location[location][bucket_index] += 1

    running_total = baseline_unique_total
    running_total_by_location = dict(baseline_unique_by_location)
    coverage_end_index = max(covered_bucket_indexes) if covered_bucket_indexes else None
    for index, point in enumerate(series):
        running_total += first_seen_counts[index]
        bucket_has_footage = (
            covered_bucket_indexes is None
            or (coverage_end_index is not None and index <= coverage_end_index)
        )
        point["cumulativeUniquePedestrians"] = running_total if bucket_has_footage else None
        point["averageVisiblePedestrians"] = (
            round(visible_totals[index] / sample_counts[index], 2)
            if bucket_has_footage and sample_counts[index]
            else (0.0 if bucket_has_footage else None)
        )
        for location in location_names:
            running_total_by_location[location] += first_seen_counts_by_location[location][index]

            location_has_footage = bucket_has_footage
            if covered_bucket_indexes_by_location is not None:
                location_coverage_indexes = covered_bucket_indexes_by_location.get(location, set())
                location_coverage_end_index = max(location_coverage_indexes) if location_coverage_indexes else None
                location_has_footage = location_coverage_end_index is not None and index <= location_coverage_end_index

            if not location_has_footage:
                point[location] = None
                if include_location_los:
                    point[f"{location}__los"] = None
                continue

            point[location] = running_total_by_location[location]

            if not include_location_los:
                continue

            location_details = (active_locations_by_name or {}).get(location)
            if location_details is None or location_sample_counts[location][index] == 0:
                point[f"{location}__los"] = None
                continue

            average_visible = location_visible_totals[location][index] / location_sample_counts[location][index]
            rounded_visible = max(1, int(round(average_visible)))
            occlusion_value = 0.0
            if location_visible_totals[location][index] > 0:
                occlusion_value = (
                    (location_light_totals[location][index] * PTSI_OCCLUSION_WEIGHTS[0])
                    + (location_moderate_totals[location][index] * PTSI_OCCLUSION_WEIGHTS[1])
                    + (location_heavy_totals[location][index] * PTSI_OCCLUSION_WEIGHTS[2])
                ) / (3.0 * location_visible_totals[location][index])

            point[f"{location}__los"] = _ptsi_score_breakdown(rounded_visible, location_details, occlusion_value).get("los")

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


def _timeline_severity_from_score(score: Optional[float]) -> str:
    if score is None:
        return "neutral"
    if score < 33:
        return "light"
    if score < 66:
        return "moderate"
    return "heavy"


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


def _video_severity_summary(state: dict[str, Any], video: dict[str, Any]) -> dict[str, Any]:
    video_id = str(video.get("id") or "")
    if not video_id:
        return {"bucketCount": 0, "sampledSeconds": 0, "buckets": []}

    pedestrian_tracks = [track for track in state.get("pedestrianTracks", []) if track.get("videoId") == video_id]
    if not pedestrian_tracks:
        return {"bucketCount": 0, "sampledSeconds": 0, "buckets": []}

    location = next((item for item in state.get("locations", []) if item.get("id") == video.get("locationId")), {})
    duration_seconds = _video_duration_seconds(video, pedestrian_tracks)
    bucket_count = max(1, min(duration_seconds, max(24, int(math.ceil(duration_seconds / 3.0))), VIDEO_TIMELINE_MAX_BUCKETS))
    bucket_span_seconds = max(float(duration_seconds) / float(bucket_count), 1.0)

    second_metrics: dict[int, dict[str, Optional[int]]] = {}
    for fallback_index, track in enumerate(pedestrian_tracks):
        track_key = _tracked_pedestrian_track_key(track, video_id, fallback_index)
        for offset_second, point, occlusion_class in _normalized_trajectory_samples(track):
            if not _point_in_location_roi(point, location):
                continue

            second_tracks = second_metrics.setdefault(max(0, int(offset_second)), {})
            existing_occlusion = second_tracks.get(track_key)
            if existing_occlusion is None or (
                occlusion_class is not None and (existing_occlusion is None or int(occlusion_class) > int(existing_occlusion))
            ):
                second_tracks[track_key] = occlusion_class

    if not second_metrics:
        return {"bucketCount": bucket_count, "sampledSeconds": 0, "buckets": []}

    bucket_scores: list[list[float]] = [[] for _ in range(bucket_count)]
    for offset_second, track_occlusions in sorted(second_metrics.items()):
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
        second_score = float(_ptsi_score_breakdown(visible_count, location, occlusion_value)["score"])
        bucket_index = min(bucket_count - 1, int(offset_second // bucket_span_seconds))
        bucket_scores[bucket_index].append(second_score)

    raw_buckets: list[dict[str, Any]] = []
    for bucket_index, scores in enumerate(bucket_scores):
        start_offset = round(bucket_index * bucket_span_seconds, 3)
        end_offset = round(
            float(duration_seconds) if bucket_index == bucket_count - 1 else min(float(duration_seconds), (bucket_index + 1) * bucket_span_seconds),
            3,
        )
        if end_offset <= start_offset:
            continue

        bucket_score = round(_percentile(scores, 90), 2) if scores else None
        raw_buckets.append(
            {
                "startOffsetSeconds": start_offset,
                "endOffsetSeconds": end_offset,
                "severity": _timeline_severity_from_score(bucket_score),
                "score": bucket_score,
            }
        )

    merged_buckets: list[dict[str, Any]] = []
    for bucket in raw_buckets:
        previous_bucket = merged_buckets[-1] if merged_buckets else None
        if previous_bucket and previous_bucket["severity"] == bucket["severity"]:
            previous_bucket["endOffsetSeconds"] = bucket["endOffsetSeconds"]
            if bucket.get("score") is not None:
                previous_score = previous_bucket.get("score")
                previous_bucket["score"] = bucket["score"] if previous_score is None else round(max(float(previous_score), float(bucket["score"])), 2)
            continue
        merged_buckets.append(bucket)

    return {
        "bucketCount": bucket_count,
        "sampledSeconds": len(second_metrics),
        "buckets": merged_buckets,
    }


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


def _windowed_dashboard_summary(date: str, time_range: str, start_time: Optional[str] = None) -> dict[str, Any]:
    state, resolved_date, videos, events = _filtered_dashboard_records(date)
    pedestrian_tracks = _filtered_pedestrian_tracks(state, videos)
    samples, first_seen_by_track = _build_analytics_samples(videos, events, pedestrian_tracks)
    videos_by_id = {str(video.get("id") or ""): video for video in videos}

    observation_times = [sample["observedAt"] for sample in samples]
    if not observation_times:
        observation_times = [timestamp for timestamp in (_observation_time(video) for video in videos) if timestamp is not None]

    buckets, bucket_span, _bucket_meta = _build_bucket_plan(resolved_date, time_range, observation_times, start_time=start_time)
    if buckets:
        window_start = buckets[0][1]
        window_end = buckets[-1][1] + bucket_span
    else:
        window_start = datetime.strptime(resolved_date, "%Y-%m-%d")
        window_end = window_start

    total_unique_pedestrians = sum(1 for first_seen_at, _location in first_seen_by_track.values() if window_start <= first_seen_at < window_end)
    total_heavy_occlusions = 0
    for event in events:
        if _event_occlusion_class(event) != 2:
            continue

        video = videos_by_id.get(str(event.get("videoId") or ""))
        if video is None:
            continue

        event_time = _event_timestamp(event, video)
        if event_time is None:
            continue
        if window_start <= event_time < window_end:
            total_heavy_occlusions += 1

    active_locations_in_window = {
        str(video.get("locationId") or "")
        for video in videos
        if (observed_at := _observation_time(video)) is not None and window_start <= observed_at < window_end
    }

    return {
        "totalUniquePedestrians": total_unique_pedestrians,
        "averageFps": 29.7 if active_locations_in_window else 0.0,
        "totalHeavyOcclusions": total_heavy_occlusions,
        "monitoredLocations": len(active_locations_in_window),
    }


def dashboard_traffic(
    date: Optional[str] = None,
    time_range: str = "12h",
    focus_time: Optional[str] = None,
    zoom_level: int = 0,
    start_time: Optional[str] = None,
) -> dict[str, Any]:
    state, resolved_date, videos, events = _filtered_dashboard_records(date)
    pedestrian_tracks = _filtered_pedestrian_tracks(state, videos)
    samples, first_seen_by_track = _build_analytics_samples(videos, events, pedestrian_tracks)
    observation_times = [sample["observedAt"] for sample in samples]
    if not observation_times:
        observation_times = [timestamp for timestamp in (_observation_time(video) for video in videos) if timestamp is not None]

    root_window_start, _root_window_end, _root_bucket_minutes = _resolve_root_window(resolved_date, time_range, observation_times, start_time)
    buckets, bucket_span, bucket_meta = _build_bucket_plan(resolved_date, time_range, observation_times, focus_time, zoom_level, start_time)
    active_location_ids = _active_location_ids(videos)
    active_location_names = [location["name"] for location in state["locations"] if location["id"] in active_location_ids]
    footage_bucket_indexes: Optional[set[int]] = None
    footage_bucket_indexes_by_location: Optional[dict[str, set[int]]] = None
    if time_range != "whole-day":
        footage_bucket_indexes, footage_bucket_indexes_by_location = _footage_bucket_coverage(
            videos,
            buckets,
            bucket_span,
            active_location_names,
        )
    series = _traffic_series_from_samples(
        buckets,
        bucket_span,
        samples,
        first_seen_by_track,
        root_window_start,
        active_location_names,
        None,
        footage_bucket_indexes,
        footage_bucket_indexes_by_location,
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


def dashboard_traffic_by_location(
    date: Optional[str] = None,
    time_range: str = "12h",
    focus_time: Optional[str] = None,
    zoom_level: int = 0,
    start_time: Optional[str] = None,
) -> dict[str, Any]:
    state, resolved_date, videos, events = _filtered_dashboard_records(date)
    pedestrian_tracks = _filtered_pedestrian_tracks(state, videos)
    samples, first_seen_by_track = _build_analytics_samples(videos, events, pedestrian_tracks)
    observation_times = [sample["observedAt"] for sample in samples]
    if not observation_times:
        observation_times = [timestamp for timestamp in (_observation_time(video) for video in videos) if timestamp is not None]

    root_window_start, _root_window_end, _root_bucket_minutes = _resolve_root_window(resolved_date, time_range, observation_times, start_time)
    buckets, bucket_span, bucket_meta = _build_bucket_plan(resolved_date, time_range, observation_times, focus_time, zoom_level, start_time)
    active_location_ids = _active_location_ids(videos)
    active_locations_by_name = {
        location["name"]: location for location in state["locations"] if location["id"] in active_location_ids
    }
    active_location_names = list(active_locations_by_name.keys())
    footage_bucket_indexes: Optional[set[int]] = None
    footage_bucket_indexes_by_location: Optional[dict[str, set[int]]] = None
    if time_range != "whole-day":
        footage_bucket_indexes, footage_bucket_indexes_by_location = _footage_bucket_coverage(
            videos,
            buckets,
            bucket_span,
            active_location_names,
        )
    series = _traffic_series_from_samples(
        buckets,
        bucket_span,
        samples,
        first_seen_by_track,
        root_window_start,
        active_location_names,
        active_locations_by_name,
        footage_bucket_indexes,
        footage_bucket_indexes_by_location,
    )

    return {
        "timeRange": time_range,
        "series": series,
        **bucket_meta,
    }


def dashboard_occlusion_trends(
    date: Optional[str] = None,
    time_range: str = "12h",
    focus_time: Optional[str] = None,
    zoom_level: int = 0,
    start_time: Optional[str] = None,
) -> dict[str, Any]:
    state, resolved_date, videos, events = _filtered_dashboard_records(date)
    pedestrian_tracks = _filtered_pedestrian_tracks(state, videos)
    samples, first_seen_by_track = _build_analytics_samples(videos, events, pedestrian_tracks)
    observation_times = [sample["observedAt"] for sample in samples]
    if not observation_times:
        observation_times = [timestamp for timestamp in (_observation_time(video) for video in videos) if timestamp is not None]

    root_window_start, _root_window_end, _root_bucket_minutes = _resolve_root_window(resolved_date, time_range, observation_times, start_time)
    buckets, bucket_span, bucket_meta = _build_bucket_plan(resolved_date, time_range, observation_times, focus_time, zoom_level, start_time)
    if not buckets:
        return {"timeRange": time_range, "series": [], **bucket_meta}

    videos_by_id = {str(video.get("id") or ""): video for video in videos}
    footage_bucket_indexes: Optional[set[int]] = None
    if time_range != "whole-day":
        footage_bucket_indexes, _footage_bucket_indexes_by_location = _footage_bucket_coverage(videos, buckets, bucket_span)
    series = _in_and_out_series_from_first_seen(
        buckets,
        bucket_span,
        first_seen_by_track,
        root_window_start,
        videos_by_id,
        footage_bucket_indexes,
    )
    return {"timeRange": time_range, "series": series, **bucket_meta}


def dashboard_los(
    date: Optional[str] = None,
    time_range: str = "12h",
    focus_time: Optional[str] = None,
    zoom_level: int = 0,
    location_id: Optional[str] = None,
    start_time: Optional[str] = None,
) -> dict[str, Any]:
    state, resolved_date, videos, events = _filtered_dashboard_records(date)
    selected_location_id = str(location_id or "").strip()
    location = next((item for item in state["locations"] if item["id"] == selected_location_id), None)

    if location is None:
        return {
            "timeRange": time_range,
            "series": [],
            "bucketMinutes": 60,
            "zoomLevel": 0,
            "canZoomIn": False,
            "isDrilldown": False,
            "focusTime": None,
            "windowStart": None,
            "windowEnd": None,
            "locationTotals": [],
        }

    location_videos = [video for video in videos if video.get("locationId") == selected_location_id]
    location_video_ids = {video["id"] for video in location_videos}
    location_events = [event for event in events if event.get("videoId") in location_video_ids]
    pedestrian_tracks = [track for track in _filtered_pedestrian_tracks(state, location_videos) if track.get("videoId") in location_video_ids]
    samples, _first_seen_by_track = _build_analytics_samples(location_videos, location_events, pedestrian_tracks)

    observation_times = [sample["observedAt"] for sample in samples]
    if not observation_times:
        observation_times = [timestamp for timestamp in (_observation_time(video) for video in location_videos) if timestamp is not None]

    buckets, bucket_span, bucket_meta = _build_bucket_plan(resolved_date, time_range, observation_times, focus_time, zoom_level, start_time)
    if not buckets:
        return {"timeRange": time_range, "series": [], **bucket_meta, "locationTotals": []}

    footage_bucket_indexes: Optional[set[int]] = None
    if time_range != "whole-day":
        _footage_bucket_indexes, footage_bucket_indexes_by_location = _footage_bucket_coverage(
            location_videos,
            buckets,
            bucket_span,
            [str(location.get("name") or "")],
        )
        footage_bucket_indexes = footage_bucket_indexes_by_location.get(str(location.get("name") or ""), set())

    series = _los_series_from_samples(
        buckets,
        bucket_span,
        samples,
        location,
        footage_bucket_indexes=footage_bucket_indexes,
    )
    return {
        "timeRange": time_range,
        "series": series,
        **bucket_meta,
        "locationTotals": [],
    }


def dashboard_occlusion(date: Optional[str] = None, time_range: str = "12h", start_time: Optional[str] = None) -> dict[str, Any]:
    state, resolved_date, videos, events = _filtered_dashboard_records(date)
    videos_by_id = {video["id"]: video for video in videos}
    locations_by_id = {location["id"]: location for location in state["locations"]}
    location_id_by_name = {
        str(video.get("location") or ""): str(video.get("locationId") or "")
        for video in videos
        if video.get("location") and video.get("locationId")
    }
    pedestrian_tracks = _filtered_pedestrian_tracks(state, videos)
    analytics_samples, _first_seen_by_track = _build_analytics_samples(videos, events, pedestrian_tracks)

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

    fallback_samples_by_second: dict[str, dict[datetime, dict[str, float]]] = {}
    for sample in analytics_samples:
        location_name = str(sample.get("location") or "")
        location_id = location_id_by_name.get(location_name)
        if not location_id:
            continue

        observed_at = sample.get("observedAt")
        if not isinstance(observed_at, datetime):
            continue
        second_at = observed_at.replace(microsecond=0)
        observation_times.append(second_at)

        location_seconds = fallback_samples_by_second.setdefault(location_id, {})
        second_rollup = location_seconds.setdefault(
            second_at,
            {
                "sampleCount": 0.0,
                "visibleTotal": 0.0,
                "lightTotal": 0.0,
                "moderateTotal": 0.0,
                "heavyTotal": 0.0,
            },
        )

        class_counts = sample.get("classCounts") or {}
        second_rollup["sampleCount"] += 1.0
        second_rollup["visibleTotal"] += float(sample.get("visibleCount") or 0.0)
        second_rollup["lightTotal"] += float(class_counts.get(0) or 0.0)
        second_rollup["moderateTotal"] += float(class_counts.get(1) or 0.0)
        second_rollup["heavyTotal"] += float(class_counts.get(2) or 0.0)

    if not observation_times:
        observation_times = [timestamp for timestamp in (_observation_time(video) for video in videos) if timestamp is not None]

    buckets, bucket_span, _ = _build_bucket_plan(resolved_date, time_range, observation_times, start_time=start_time)
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

    for location_id, samples_by_second in fallback_samples_by_second.items():
        location_seconds = second_metrics.setdefault(location_id, {})
        for second_at, counts in samples_by_second.items():
            if second_at in location_seconds:
                continue
            location_seconds[second_at] = {"counts": counts}

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
        all_scores: list[float] = []
        all_los_ranks: list[int] = []
        total_visible = 0
        total_light = 0
        total_moderate = 0
        total_heavy = 0
        total_sample_seconds = 0
        unique_track_keys: set[str] = set()

        for second_at, second_entry in sorted(second_metrics.get(location["id"], {}).items()):
            track_occlusions = second_entry.get("tracks") or {}
            counts = second_entry.get("counts") or {}
            has_track_details = bool(track_occlusions)

            if has_track_details:
                visible_count = len(track_occlusions)
                light_count = sum(1 for value in track_occlusions.values() if value == 0)
                moderate_count = sum(1 for value in track_occlusions.values() if value == 1)
                heavy_count = sum(1 for value in track_occlusions.values() if value == 2)
            else:
                sample_count = float(counts.get("sampleCount") or 0.0)
                if sample_count <= 0:
                    continue
                visible_count = int(round(float(counts.get("visibleTotal") or 0.0) / sample_count))
                light_count = int(round(float(counts.get("lightTotal") or 0.0) / sample_count))
                moderate_count = int(round(float(counts.get("moderateTotal") or 0.0) / sample_count))
                heavy_count = int(round(float(counts.get("heavyTotal") or 0.0) / sample_count))

            if visible_count <= 0:
                continue

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
                all_los_ranks.append(int(los_rank))
            if has_track_details:
                hour_rollup["trackKeys"].update(track_occlusions.keys())

            all_scores.append(second_score)
            total_visible += visible_count
            total_light += light_count
            total_moderate += moderate_count
            total_heavy += heavy_count
            total_sample_seconds += 1
            if has_track_details:
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
                "uniquePedestrians": len(rollup["trackKeys"]) if rollup["trackKeys"] else None,
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
        overall_score = round(_percentile(all_scores, 90), 2) if all_scores else None
        selected_mode = mode
        selected_average_pedestrians = round(total_visible / total_sample_seconds, 2) if total_sample_seconds else None
        selected_unique_pedestrians = len(unique_track_keys) if unique_track_keys else None
        selected_occlusion_mix = _ptsi_occlusion_mix(total_light, total_moderate, total_heavy, total_visible) if total_visible else None
        selected_los = _ptsi_los_from_score(overall_score) if mode == "roi-testing" else _ptsi_los_from_rank(max(all_los_ranks) if all_los_ranks else None)
        selected_los_description = _ptsi_los_description(selected_los)
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


def ai_synthesis(date: str, time_range: str, start_time: Optional[str] = None) -> dict[str, Any]:
    summary = _windowed_dashboard_summary(date, time_range, start_time)
    traffic_response = dashboard_traffic(date, time_range, start_time=start_time)
    traffic_series = traffic_response["series"]
    location_totals = traffic_response.get("locationTotals", [])

    def _to_float_or_none(value: Any) -> Optional[float]:
        if isinstance(value, bool):
            return None
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            stripped_value = value.strip()
            if not stripped_value:
                return None
            try:
                return float(stripped_value)
            except ValueError:
                return None
        return None

    peak_point = None
    peak_visible_average = 0.0
    if traffic_series:
        peak_candidates = [
            (point, parsed_average)
            for point in traffic_series
            if (parsed_average := _to_float_or_none(point.get("averageVisiblePedestrians"))) is not None
        ]
        if peak_candidates:
            peak_point, peak_visible_average = max(peak_candidates, key=lambda candidate: candidate[1])

    occlusion_response = dashboard_occlusion(date, time_range, start_time)
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
    cumulative_values = [
        int(value)
        for value in (point.get("cumulativeUniquePedestrians") for point in traffic_series)
        if value is not None
    ]
    cumulative_total = cumulative_values[-1] if cumulative_values else 0
    peak_events_body = (
        f"{top_location} contributes the strongest unique footfall in this selection, while the busiest bucket averaged {peak_visible_average:.2f} visible pedestrians around {peak_window}."
        if peak_point
        else f"{top_location} contributes the strongest unique footfall in this selection, but the selected timeline has no visible-pedestrian averages yet for peak detection."
    )

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
                "body": peak_events_body,
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


def export_dashboard_report(date: str, time_range: str, start_time: Optional[str] = None) -> Path:
    ensure_storage_layout()

    state, _, videos, events = _filtered_dashboard_records(date)
    pedestrian_tracks = _filtered_pedestrian_tracks(state, videos)
    _analytics_samples, first_seen_by_track = _build_analytics_samples(videos, events, pedestrian_tracks)
    summary = _windowed_dashboard_summary(date, time_range, start_time)
    traffic_response = dashboard_traffic(date, time_range, start_time=start_time)
    traffic = traffic_response["series"]
    occlusion_response = dashboard_occlusion(date, time_range, start_time)
    synthesis = ai_synthesis(date, time_range, start_time)
    model = get_model_info()

    location_totals = traffic_response.get("locationTotals", [])

    generated_at = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    timestamp_slug = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    filename = f"{slugify(date)}-{slugify(time_range)}-dashboard-report-{timestamp_slug}.zip"
    target = EXPORTS_DIR / filename
    report_name = f"{slugify(date)}-{slugify(time_range)}-dashboard-report-{timestamp_slug}.md"

    lines = [
        "# ALIVE Engine Dashboard Report",
        "",
        f"- Date: {date}",
        f"- Time Range: {format_time_range_label(time_range)}",
        f"- Generated At: {generated_at}",
        f"- Active Model: {model.get('currentModel') or 'No model selected'}",
        f"- Bundle Format: ZIP (Markdown + CSV + JSON portable artifacts)",
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
            "## PTSI / LOS Hotspots",
            "",
        ]
    )

    hotspot_locations = occlusion_response.get("locations", [])
    if hotspot_locations:
        for location in hotspot_locations:
            lines.append(
                f"- {location['name']}: LOS {location.get('los') or 'N/A'} · PTSI {location.get('score') if location.get('score') is not None else 'N/A'}"
            )
    else:
        lines.append("- No PTSI hotspot data available for the selected range.")

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

    markdown_text = "\n".join(lines).strip() + "\n"

    dashboard_summary_rows = [summary]
    traffic_rows = [dict(point) for point in traffic]
    location_total_rows = [dict(entry) for entry in location_totals]
    ptsi_rows = [dict(location) for location in occlusion_response.get("locations", [])]
    unique_pedestrian_rows = _dashboard_unique_pedestrian_rows(first_seen_by_track)
    video_total_rows = [
        {
            "videoId": video.get("id"),
            "locationId": video.get("locationId"),
            "locationName": video.get("location"),
            "date": video.get("date"),
            "startTime": video.get("startTime"),
            "endTime": video.get("endTime"),
            "pedestrianCount": int(video.get("pedestrianCount") or 0),
        }
        for video in videos
    ]

    for video in videos:
        _write_portable_video_artifacts(state, video)

    selected_videos = [video for video in state.get("videos", []) if str(video.get("date") or "") == str(date)]

    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(report_name, markdown_text)
        archive.writestr("dashboard/summary.json", json.dumps(summary, indent=2))
        archive.writestr("dashboard/summary.csv", _csv_text(dashboard_summary_rows))
        archive.writestr("dashboard/unique_pedestrians.json", json.dumps(unique_pedestrian_rows, indent=2))
        archive.writestr("dashboard/unique_pedestrians.csv", _csv_text(unique_pedestrian_rows))
        archive.writestr("dashboard/video_totals.json", json.dumps(video_total_rows, indent=2))
        archive.writestr("dashboard/video_totals.csv", _csv_text(video_total_rows))
        archive.writestr("dashboard/traffic.json", json.dumps(traffic_response, indent=2))
        archive.writestr("dashboard/traffic.csv", _csv_text(traffic_rows))
        archive.writestr("dashboard/location_totals.csv", _csv_text(location_total_rows))
        archive.writestr("dashboard/ptsi_map.json", json.dumps(occlusion_response, indent=2))
        archive.writestr("dashboard/ptsi_map.csv", _csv_text(ptsi_rows))
        archive.writestr("dashboard/ai_synthesis.json", json.dumps(synthesis, indent=2))

        if QUEUE_HISTORY_JSON_FILE.exists():
            archive.write(QUEUE_HISTORY_JSON_FILE, arcname="portable/queue_history.json")
        if QUEUE_HISTORY_CSV_FILE.exists():
            archive.write(QUEUE_HISTORY_CSV_FILE, arcname="portable/queue_history.csv")
        if PORTABLE_MANIFEST_FILE.exists():
            archive.write(PORTABLE_MANIFEST_FILE, arcname="portable/manifest.json")

        for video in selected_videos:
            video_dir = _portable_video_directory(str(video.get("id") or ""))
            if not video_dir.exists():
                continue
            for path in video_dir.rglob("*"):
                if path.is_dir():
                    continue
                archive.write(path, arcname=_portable_relative_path(path))

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
    if query_parser is not None and stripped_query and _should_parse_search_query(stripped_query):
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


def _should_parse_search_query(query: str) -> bool:
    return len(_query_tokens(query)) >= SEARCH_QUERY_PARSER_MIN_TOKENS


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
