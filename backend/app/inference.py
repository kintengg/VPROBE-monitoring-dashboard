from __future__ import annotations

import colorsys
import importlib
import importlib.util
import re
import sys
from collections.abc import Callable
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

from . import store, vision

PREFERRED_ULTRALYTICS_TAG = "v8.3.228"
FALLBACK_ULTRALYTICS_TAG = "v8.3.50"
VIDEO_SUFFIXES = (".mp4", ".avi", ".mov", ".mkv", ".m4v")
MAX_TRACK_EVENTS = 50
TRACK_THUMBNAIL_MAX_EDGE = 224
CLOCK_TIME_FORMATS = ("%H:%M", "%H:%M:%S", "%I:%M %p", "%I:%M:%S %p")


def _vendored_ultralytics_dir() -> Path:
    return store.BACKEND_DIR / "vendor" / "ultralytics"


def _prefer_vendored_ultralytics() -> Optional[Path]:
    vendor_dir = _vendored_ultralytics_dir()
    if not vendor_dir.exists():
        return None

    vendor_dir_str = str(vendor_dir.resolve())
    if sys.path[:1] != [vendor_dir_str]:
        try:
            sys.path.remove(vendor_dir_str)
        except ValueError:
            pass
        sys.path.insert(0, vendor_dir_str)
        importlib.invalidate_caches()

    return vendor_dir


def _module_is_within(path_value: Optional[str], root: Optional[Path]) -> bool:
    if not path_value or root is None:
        return False

    try:
        Path(path_value).resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def resolve_model_path(model_name: Optional[str]) -> Optional[Path]:
    if not model_name:
        return None

    candidate = store.MODELS_DIR / Path(model_name).name
    if candidate.exists():
        return candidate
    return None


def ultralytics_status() -> dict[str, Any]:
    model_info = store.get_model_info()
    model_name = model_info.get("currentModel")
    model_path = resolve_model_path(model_name)
    vendor_dir = _prefer_vendored_ultralytics()

    installed = importlib.util.find_spec("ultralytics") is not None
    version = None
    package_path = None
    if installed:
        try:
            from ultralytics import __file__ as ultralytics_file
            from ultralytics import __version__ as ultralytics_version

            version = ultralytics_version
            package_path = ultralytics_file
        except Exception:
            version = "installed"

    return {
        "installed": installed,
        "version": version,
        "packagePath": package_path,
        "vendoredPath": _backend_relative_path(vendor_dir),
        "usingVendoredCopy": _module_is_within(package_path, vendor_dir),
        "preferredTag": PREFERRED_ULTRALYTICS_TAG,
        "fallbackTag": FALLBACK_ULTRALYTICS_TAG,
        "currentModel": model_name,
        "modelPath": str(model_path.relative_to(store.BACKEND_DIR)) if model_path else None,
        "modelExists": model_path is not None,
        "ready": installed and model_path is not None,
    }


def _backend_relative_path(path: Optional[Path]) -> Optional[str]:
    if path is None:
        return None
    try:
        return str(path.relative_to(store.BACKEND_DIR))
    except ValueError:
        return str(path)


def _normalize_names(names: Any) -> dict[int, str]:
    if isinstance(names, dict):
        items = names.items()
    else:
        items = enumerate(names or [])
    return {int(idx): str(name).strip().lower() for idx, name in items}


def _normalized_label(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.strip().lower()).strip()


def _tracking_class_config(names: Any) -> tuple[Optional[list[int]], dict[int, int]]:
    normalized_names = {idx: _normalized_label(name) for idx, name in _normalize_names(names).items()}
    pedestrian_aliases = {"person", "pedestrian"}
    occlusion_alias_groups = {
        0: {"light", "light occlusion", "low occlusion", "minor occlusion"},
        1: {"moderate", "moderate occlusion", "medium occlusion", "partial occlusion"},
        2: {"heavy", "heavy occlusion", "severe occlusion", "high occlusion"},
    }

    class_ids: list[int] = []
    occlusion_classes: dict[int, int] = {}
    for idx, label in normalized_names.items():
        if label in pedestrian_aliases:
            class_ids.append(idx)
            continue

        for occlusion_class, aliases in occlusion_alias_groups.items():
            if label in aliases:
                class_ids.append(idx)
                occlusion_classes[idx] = occlusion_class
                break

    return (sorted(set(class_ids)) or None, occlusion_classes)


def _tracker_config_path() -> Path:
    return store.BACKEND_DIR / "vendor" / "ultralytics" / "ultralytics" / "cfg" / "trackers" / "bytetrack.yaml"


def preferred_inference_device() -> str:
    try:
        import torch
    except Exception:
        return "cpu"

    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and mps_backend.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "0"
    return "cpu"


def _scalar(value: Any, default: Any = None) -> Any:
    if value is None:
        return default
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError, RuntimeError):
            return value
    return value


def _find_processed_video(save_dir: Path, source_path: Path) -> Optional[Path]:
    for suffix in VIDEO_SUFFIXES:
        candidate = save_dir / (source_path.stem + suffix)
        if candidate.exists():
            return candidate
    return None


def _parse_clock_time(value: str) -> Optional[datetime]:
    cleaned = value.strip()
    for time_format in CLOCK_TIME_FORMATS:
        try:
            return datetime.strptime(cleaned, time_format)
        except ValueError:
            continue
    return None


def _format_event_timestamp(start_time: str, offset_seconds: float) -> str:
    base_time = _parse_clock_time(start_time)
    if base_time is None:
        return start_time or "Unknown Time"
    return (base_time + timedelta(seconds=max(0.0, offset_seconds))).strftime("%I:%M:%S %p").lstrip("0")


def _read_video_metadata(video_path: Path) -> tuple[float, Optional[int]]:
    try:
        import cv2
    except Exception:
        return 30.0, None

    capture = cv2.VideoCapture(str(video_path))
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    finally:
        capture.release()

    return (fps if fps > 0 else 30.0, frame_count if frame_count > 0 else None)


def _box_xyxy(box: Any) -> Optional[tuple[int, int, int, int]]:
    xyxy = getattr(box, "xyxy", None)
    if xyxy is None:
        return None

    raw_value = _scalar(xyxy, xyxy)
    if hasattr(raw_value, "tolist"):
        values = raw_value.tolist()
    else:
        values = raw_value

    if not isinstance(values, list) or not values:
        return None

    first = values[0] if isinstance(values[0], list) else values
    if not isinstance(first, list) or len(first) < 4:
        return None

    try:
        x1, y1, x2, y2 = (int(round(float(first[index]))) for index in range(4))
    except (TypeError, ValueError):
        return None

    return x1, y1, x2, y2


def _crop_frame(image: Any, bounds: tuple[int, int, int, int]) -> tuple[Optional[Any], float]:
    if image is None or not hasattr(image, "shape"):
        return None, 0.0

    height, width = image.shape[:2]
    x1, y1, x2, y2 = bounds
    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(x1 + 1, min(width, x2))
    y2 = max(y1 + 1, min(height, y2))
    crop = image[y1:y2, x1:x2]
    area = float(max(0, x2 - x1) * max(0, y2 - y1))
    if crop is None or getattr(crop, "size", 0) == 0:
        return None, 0.0
    return crop, area


def _resize_for_thumbnail(image: Any) -> Any:
    try:
        import cv2
    except Exception:
        return image

    height, width = image.shape[:2]
    max_edge = max(height, width)
    if max_edge <= TRACK_THUMBNAIL_MAX_EDGE:
        return image

    scale = TRACK_THUMBNAIL_MAX_EDGE / max_edge
    target_size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    return cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)


def _save_track_thumbnail(image: Any, target: Path) -> Optional[str]:
    try:
        import cv2
    except Exception:
        return None

    target.parent.mkdir(parents=True, exist_ok=True)
    thumbnail = _resize_for_thumbnail(image)
    success = cv2.imwrite(str(target), thumbnail, [int(cv2.IMWRITE_JPEG_QUALITY), 72])
    if not success:
        return None
    return _backend_relative_path(target)


def _average_bgr(region: Any) -> Optional[tuple[float, float, float]]:
    if region is None or getattr(region, "size", 0) == 0 or not hasattr(region, "mean"):
        return None
    mean = region.mean(axis=(0, 1))
    if not hasattr(mean, "tolist"):
        return None
    values = mean.tolist()
    if not isinstance(values, list) or len(values) < 3:
        return None
    return float(values[0]), float(values[1]), float(values[2])


def _color_name_from_bgr(color: Optional[tuple[float, float, float]]) -> Optional[str]:
    if color is None:
        return None

    blue, green, red = [max(0.0, min(255.0, channel)) / 255.0 for channel in color]
    hue, saturation, value = colorsys.rgb_to_hsv(red, green, blue)

    if value < 0.18:
        return "black"
    if saturation < 0.12 and value > 0.82:
        return "white"
    if saturation < 0.18:
        return "gray"
    if value < 0.45 and red > green and green > blue:
        return "brown"

    hue_degrees = hue * 360.0
    if hue_degrees < 12 or hue_degrees >= 345:
        return "red"
    if hue_degrees < 35:
        return "orange"
    if hue_degrees < 65:
        return "yellow"
    if hue_degrees < 160:
        return "green"
    if hue_degrees < 200:
        return "cyan"
    if hue_degrees < 255:
        return "blue"
    if hue_degrees < 320:
        return "purple"
    return "pink"


def _appearance_hints(crop: Any) -> list[str]:
    if crop is None or not hasattr(crop, "shape"):
        return []

    height, width = crop.shape[:2]
    if height < 24 or width < 12:
        return []

    left = int(round(width * 0.2))
    right = max(left + 1, int(round(width * 0.8)))
    centered = crop[:, left:right]
    section_boundaries = [0, max(1, int(height * 0.28)), max(2, int(height * 0.6)), height]
    region_specs = [
        ("head region", centered[section_boundaries[0] : section_boundaries[1]]),
        ("upper clothing", centered[section_boundaries[1] : section_boundaries[2]]),
        ("lower clothing", centered[section_boundaries[2] : section_boundaries[3]]),
    ]

    hints: list[str] = []
    for label, region in region_specs:
        color_name = _color_name_from_bgr(_average_bgr(region))
        if color_name:
            hints.append(f"{label} appears {color_name}")
    return hints


def _appearance_summary(hints: list[str], occlusion_class: Optional[int], vision_summary: Optional[str] = None) -> str:
    clauses: list[str] = []
    if hints:
        clauses.append("Representative crop suggests " + ", ".join(hints) + ".")
    if vision_summary:
        clauses.append(str(vision_summary).strip())
    if occlusion_class is not None:
        severity = {0: "light", 1: "moderate", 2: "heavy"}.get(occlusion_class)
        if severity:
            clauses.append(f"Visibility shows {severity} occlusion.")
    return " ".join(clauses) if clauses else "Representative pedestrian track with limited appearance detail."


def _track_thumbnail_file(track_summary: dict[str, Any]) -> Optional[Path]:
    thumbnail_path = str(track_summary.get("thumbnailPath") or "").strip()
    if not thumbnail_path:
        return None

    candidate = Path(thumbnail_path)
    if not candidate.is_absolute():
        candidate = store.BACKEND_DIR / candidate

    try:
        candidate.relative_to(store.BACKEND_DIR)
    except ValueError:
        return None

    return candidate if candidate.exists() else None


def _enrich_track_summaries_with_vision(
    track_summaries: list[dict[str, Any]],
    progress_callback: Optional[Callable[[dict[str, Any]], None]] = None,
) -> int:
    if not track_summaries or not vision.track_enrichment_enabled():
        return 0

    limit = vision.track_enrichment_limit()
    if limit <= 0:
        return 0

    eligible_tracks: list[tuple[dict[str, Any], Path]] = []
    for track_summary in sorted(
        track_summaries,
        key=lambda item: (
            float(item.get("bestArea") or 0.0),
            float(item.get("bestOffsetSeconds") or item.get("firstOffsetSeconds") or 0.0),
        ),
        reverse=True,
    ):
        thumbnail_file = _track_thumbnail_file(track_summary)
        if thumbnail_file is not None:
            eligible_tracks.append((track_summary, thumbnail_file))

    selected_tracks = eligible_tracks[:limit]
    if not selected_tracks:
        return 0

    if progress_callback is not None:
        progress_callback(
            {
                "progressPercent": 80,
                "phase": "vision",
                "message": f"Preparing Cloud Vision enrichment for {len(selected_tracks)} track thumbnails...",
            }
        )

    enriched_count = 0
    total_selected = len(selected_tracks)
    for index, (track_summary, thumbnail_file) in enumerate(selected_tracks, start=1):
        metadata = vision.enrich_track_thumbnail(thumbnail_file)
        if not metadata:
            if progress_callback is not None:
                progress_callback(
                    {
                        "progressPercent": min(98, 80 + int(round((index / total_selected) * 18))),
                        "phase": "vision",
                        "message": f"Cloud Vision analyzing track thumbnails ({index}/{total_selected})...",
                    }
                )
            continue

        track_summary["visualLabels"] = metadata.get("labels") or []
        track_summary["visualObjects"] = metadata.get("objects") or []
        track_summary["visualLogos"] = metadata.get("logos") or []
        track_summary["visualText"] = metadata.get("text") or []
        track_summary["visualSummary"] = metadata.get("summary") or None
        track_summary["appearanceSummary"] = _appearance_summary(
            track_summary.get("appearanceHints") or [],
            track_summary.get("occlusionClass"),
            track_summary.get("visualSummary"),
        )
        enriched_count += 1

        if progress_callback is not None:
            progress_callback(
                {
                    "progressPercent": min(98, 80 + int(round((index / total_selected) * 18))),
                    "phase": "vision",
                    "message": f"Cloud Vision analyzing track thumbnails ({index}/{total_selected})...",
                }
            )

    return enriched_count


def run_video_inference(
    video_path: Path,
    model_name: Optional[str] = None,
    video_record: Optional[dict[str, Any]] = None,
    fast_mode: bool = False,
    progress_callback: Optional[Callable[[dict[str, Any]], None]] = None,
) -> dict[str, Any]:
    status = ultralytics_status()
    if not status["installed"]:
        raise RuntimeError("Ultralytics is not installed yet. Install the pinned custom copy before running inference.")
    if not status["modelExists"]:
        raise RuntimeError("The active model file is missing from backend/storage/models.")

    model_path = resolve_model_path(model_name or status["currentModel"])
    if model_path is None:
        raise RuntimeError("Could not resolve the active model path for inference.")

    _prefer_vendored_ultralytics()
    from ultralytics import YOLO

    save_name = (video_record or {}).get("id") or video_path.stem
    save_dir = store.PROCESSED_VIDEOS_DIR / save_name
    save_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(model_path))
    classes, occlusion_classes = _tracking_class_config(getattr(model, "names", None))

    track_kwargs = {
        "source": str(video_path),
        "stream": True,
        "persist": True,
        "save": True,
        "device": preferred_inference_device(),
        "project": str(store.PROCESSED_VIDEOS_DIR),
        "name": save_name,
        "exist_ok": True,
        "verbose": False,
    }
    if classes is not None:
        track_kwargs["classes"] = classes
    tracker_config = _tracker_config_path()
    if tracker_config.exists():
        track_kwargs["tracker"] = str(tracker_config)
    if fast_mode:
        track_kwargs["imgsz"] = 512
        track_kwargs["vid_stride"] = 2

    location = (video_record or {}).get("location", "Unknown Location")
    start_timestamp = (video_record or {}).get("timestamp") or (video_record or {}).get("startTime") or "Unknown Time"
    video_id = (video_record or {}).get("id")
    fps, total_frames = _read_video_metadata(video_path)

    seen_track_ids: set[int] = set()
    max_people_in_frame = 0
    events: list[dict[str, Any]] = []
    pedestrian_tracks: dict[int, dict[str, Any]] = {}
    last_reported_percent = -1

    if progress_callback is not None:
        progress_callback({"progressPercent": 0, "phase": "tracking", "message": "Initializing detection and tracking..."})

    for frame_index, result in enumerate(model.track(**track_kwargs), start=1):
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            continue

        frame_image = getattr(result, "orig_img", None)

        frame_people = 0
        for box in boxes:
            frame_people += 1
            track_id = None
            if getattr(box, "is_track", False):
                track_id = _scalar(getattr(box, "id", None))
            if track_id is None:
                continue

            track_id = int(track_id)
            box_class = _scalar(getattr(box, "cls", None))
            occlusion_class = None if box_class is None else occlusion_classes.get(int(box_class))
            offset_seconds = round(max(0.0, (frame_index - 1) / fps), 2)
            timestamp = _format_event_timestamp(start_timestamp, offset_seconds)

            track_summary = pedestrian_tracks.get(track_id)
            if track_summary is None:
                track_summary = {
                    "id": uuid4().hex[:8],
                    "videoId": video_id,
                    "pedestrianId": track_id,
                    "location": location,
                    "firstTimestamp": timestamp,
                    "lastTimestamp": timestamp,
                    "bestTimestamp": timestamp,
                    "firstFrame": frame_index,
                    "lastFrame": frame_index,
                    "bestFrame": frame_index,
                    "firstOffsetSeconds": offset_seconds,
                    "lastOffsetSeconds": offset_seconds,
                    "bestOffsetSeconds": offset_seconds,
                    "thumbnailPath": None,
                    "appearanceHints": [],
                    "appearanceSummary": "Representative pedestrian track with limited appearance detail.",
                    "occlusionClass": occlusion_class,
                    "bestArea": 0.0,
                }
                pedestrian_tracks[track_id] = track_summary
            else:
                track_summary["lastTimestamp"] = timestamp
                track_summary["lastFrame"] = frame_index
                track_summary["lastOffsetSeconds"] = offset_seconds
                if occlusion_class is not None:
                    current_occlusion = track_summary.get("occlusionClass")
                    if current_occlusion is None or int(occlusion_class) > int(current_occlusion):
                        track_summary["occlusionClass"] = occlusion_class

            bounds = _box_xyxy(box)
            if bounds is not None and frame_image is not None:
                crop, crop_area = _crop_frame(frame_image, bounds)
                if crop is not None and crop_area > float(track_summary.get("bestArea") or 0.0):
                    track_summary["bestArea"] = crop_area
                    track_summary["bestFrame"] = frame_index
                    track_summary["bestTimestamp"] = timestamp
                    track_summary["bestOffsetSeconds"] = offset_seconds
                    thumbnail_target = save_dir / "tracks" / f"track-{track_id}.jpg"
                    thumbnail_path = _save_track_thumbnail(crop, thumbnail_target)
                    if thumbnail_path:
                        track_summary["thumbnailPath"] = thumbnail_path
                    hints = _appearance_hints(crop)
                    track_summary["appearanceHints"] = hints
                    track_summary["appearanceSummary"] = _appearance_summary(hints, track_summary.get("occlusionClass"))

            if track_id in seen_track_ids:
                continue

            seen_track_ids.add(track_id)
            if len(events) < MAX_TRACK_EVENTS:
                description = f"Pedestrian ID #{track_id} detected at frame {frame_index}"
                if occlusion_class is not None:
                    severity_labels = {0: "light", 1: "moderate", 2: "heavy"}
                    description = f"{severity_labels[occlusion_class].title()} occlusion pedestrian ID #{track_id} detected at frame {frame_index}"
                events.append(
                    {
                        "id": uuid4().hex[:8],
                        "type": "detection",
                        "location": location,
                        "timestamp": _format_event_timestamp(start_timestamp, offset_seconds),
                        "description": description,
                        "videoId": video_id,
                        "pedestrianId": track_id,
                        "frame": frame_index,
                        "offsetSeconds": offset_seconds,
                        "occlusionClass": occlusion_class,
                    }
                )

        max_people_in_frame = max(max_people_in_frame, frame_people)

        if progress_callback is None:
            continue

        if total_frames is not None:
            progress_percent = min(78, int(round((frame_index / total_frames) * 78)))
            if progress_percent >= last_reported_percent + 5 or frame_index >= total_frames:
                last_reported_percent = progress_percent
                progress_callback(
                    {
                        "progressPercent": progress_percent,
                        "phase": "tracking",
                        "message": "Running detection and tracking...",
                        "frameIndex": frame_index,
                        "totalFrames": total_frames,
                    }
                )
        elif frame_index == 1:
            progress_callback({"progressPercent": None, "phase": "tracking", "message": "Running detection and tracking..."})

    pedestrian_count = len(seen_track_ids) if seen_track_ids else max_people_in_frame
    if pedestrian_count > 0 and not events:
        events.append(
            {
                "id": uuid4().hex[:8],
                "type": "detection",
                "location": location,
                "timestamp": start_timestamp,
                "description": f"{pedestrian_count} pedestrians detected",
                "videoId": video_id,
                "pedestrianId": None,
                "frame": None,
                "offsetSeconds": None,
                "occlusionClass": None,
            }
        )

    track_results = list(pedestrian_tracks.values())
    _enrich_track_summaries_with_vision(track_results, progress_callback)

    if progress_callback is not None:
        progress_callback({"progressPercent": 99, "phase": "finalizing", "message": "Finalizing processed video..."})

    processed_path = _backend_relative_path(_find_processed_video(save_dir, video_path))
    return {
        "pedestrianCount": pedestrian_count,
        "processedPath": processed_path,
        "events": events,
        "pedestrianTracks": track_results,
    }