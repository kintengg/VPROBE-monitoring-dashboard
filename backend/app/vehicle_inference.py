"""Vehicle inference via the Occlusion-Robust-RTDETR research codebase.

Subprocess-invokes `Occlusion-Robust-RTDETR/tools/infer.py` against the
uploaded video, parses its `<output>_counts.csv` and `<output>_summary.txt`
sidecars, and converts them to events the rest of the pipeline understands.

This is a thin shim — the heavy lifting (model build, ByteTrack, LineZone)
all happens in the research codebase. We only translate I/O.
"""

from __future__ import annotations

import csv
import os
import re
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

from . import model_registry, store
from .vehicle import gates as gate_registry

RTDETR_DIR = store.BACKEND_DIR / "Occlusion-Robust-RTDETR"
INFER_SCRIPT = RTDETR_DIR / "tools" / "infer.py"
DEFAULT_RTDETR_CONFIG = RTDETR_DIR / "configs" / "rtdetr" / "rtdetr_fastervit0_final.yml"
COUNTING_DIR = RTDETR_DIR / "inference_requirements" / "counting"
DETECTION_THRESHOLD = "0.5"
PROCESSED_VIDEOS_DIR = store.PROCESSED_VIDEOS_DIR

VehicleProgressCallback = Callable[[dict[str, Any]], None]


class VehicleInferenceError(RuntimeError):
    """Raised when RT-DETR subprocess exits non-zero or sidecars are missing."""


def _resolve_counting_config(location_id: str, counting_config_name: Optional[str]) -> Optional[Path]:
    """Resolve counting-config JSON from an explicit name or seeded gate."""
    if counting_config_name:
        candidate = COUNTING_DIR / Path(counting_config_name).name
        if candidate.exists():
            return candidate
        raise VehicleInferenceError(
            f"Counting config '{counting_config_name}' not found in {COUNTING_DIR}"
        )

    gate = gate_registry.get_gate(location_id)
    if gate is None:
        return None
    candidate = COUNTING_DIR / gate["countingConfig"]
    return candidate if candidate.exists() else None


def _resolve_weights_path() -> Path:
    path = model_registry.active_weight_path("vehicle")
    if path is None:
        raise VehicleInferenceError(
            "No vehicle weight is registered. Visit /models to upload one."
        )
    return path


def _resolve_config_path() -> Path:
    if not DEFAULT_RTDETR_CONFIG.exists():
        raise VehicleInferenceError(
            f"RT-DETR config missing at {DEFAULT_RTDETR_CONFIG.relative_to(store.BACKEND_DIR)}"
        )
    return DEFAULT_RTDETR_CONFIG


def _device_for_runtime() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def _emit(callback: Optional[VehicleProgressCallback], **payload: Any) -> None:
    if callback is None:
        return
    try:
        callback(payload)
    except Exception:
        pass


def _stream_progress(process: subprocess.Popen[str], callback: Optional[VehicleProgressCallback]) -> None:
    """Read stderr for tqdm-style progress lines and forward to the callback.

    Polls a cancel-check (if `callback.cancel_check` is attached) at ~10 Hz
    via `select` so the subprocess can be terminated quickly even when
    RT-DETR has gone quiet (e.g., during model load). Cancellation
    terminates the subprocess and re-raises so the upload thread settles
    the queue entry as "cancelled".
    """
    import select

    progress_re = re.compile(r"(\d+)%\|.*?\|\s*(\d+)/(\d+)")
    if process.stderr is None:
        return
    cancel_check = getattr(callback, "cancel_check", None) if callback is not None else None
    last_percent = -1
    stderr_fd = process.stderr.fileno()

    def _terminate() -> None:
        try:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        except Exception:
            pass

    try:
        while True:
            if callable(cancel_check):
                cancel_check()  # raises InterruptedError when the user clicks cancel
            if process.poll() is not None:
                # Drain any remaining stderr after the subprocess exits.
                for raw_line in process.stderr:
                    pass  # discard; final state read upstream
                return
            # Wait up to 100ms for new stderr output; the short timeout keeps
            # the cancel check responsive.
            ready, _, _ = select.select([stderr_fd], [], [], 0.1)
            if not ready:
                continue
            raw_line = process.stderr.readline()
            if not raw_line:
                # EOF — subprocess has closed stderr. Loop back to check exit.
                continue
            line = raw_line.rstrip()
            match = progress_re.search(line) if line else None
            if not match:
                continue
            percent = int(match.group(1))
            if percent != last_percent:
                last_percent = percent
                _emit(
                    callback,
                    phase="tracking",
                    progressPercent=min(94, max(5, int(percent * 0.9) + 5)),
                    message=f"RT-DETR processing… frame {match.group(2)}/{match.group(3)}",
                )
    except InterruptedError:
        _terminate()
        raise


def _parse_counts_csv(csv_path: Path) -> list[dict[str, Any]]:
    if not csv_path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with csv_path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(row)
    return rows


def _parse_detections_csv(csv_path: Path) -> list[dict[str, Any]]:
    if not csv_path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with csv_path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(row)
    return rows


def _track_events_from_detections(
    rows: list[dict[str, Any]],
    video_id: str,
    video_record: dict[str, Any],
) -> list[dict[str, Any]]:
    """Collapse per-frame rows into one `vehicle-track` event per unique track_id.

    Each event carries the track's first-seen / last-seen offset so the page
    can answer 'how many vehicles have been detected so far' and 'how many are
    currently in frame' without needing a separate tracks table.
    """
    location_name = str(video_record.get("location") or "")
    base_timestamp = str(video_record.get("timestamp") or "")
    by_track: dict[int, dict[str, Any]] = {}
    for row in rows:
        try:
            track_id = int(row.get("track_id"))
        except (TypeError, ValueError):
            continue
        try:
            frame = int(row.get("frame_number") or 0)
        except (TypeError, ValueError):
            frame = 0
        offset = _parse_clock_offset_seconds(row.get("timestamp"))
        class_name = (row.get("class_name") or "").strip() or None
        existing = by_track.get(track_id)
        if existing is None:
            by_track[track_id] = {
                "trackId": track_id,
                "vehicleClass": class_name,
                "firstFrame": frame,
                "lastFrame": frame,
                "firstOffset": offset,
                "lastOffset": offset,
            }
            continue
        if frame < existing["firstFrame"]:
            existing["firstFrame"] = frame
            existing["firstOffset"] = offset
        if frame > existing["lastFrame"]:
            existing["lastFrame"] = frame
            existing["lastOffset"] = offset
        if existing.get("vehicleClass") is None and class_name:
            existing["vehicleClass"] = class_name

    events: list[dict[str, Any]] = []
    for track in by_track.values():
        events.append(
            {
                "id": str(uuid4()),
                "type": "vehicle-track",
                "location": location_name,
                "timestamp": base_timestamp,
                "description": (
                    f"{track['vehicleClass'] or 'vehicle'} #{track['trackId']} "
                    f"first seen at {track['firstOffset']:.1f}s, last seen at {track['lastOffset']:.1f}s"
                ),
                "videoId": video_id,
                "frame": track["firstFrame"],
                "offsetSeconds": track["firstOffset"],
                "vehicleClass": track["vehicleClass"],
                "trackId": track["trackId"],
                "lastOffsetSeconds": track["lastOffset"],
                "lastFrame": track["lastFrame"],
            }
        )
    return events


def _parse_clock_offset_seconds(value: Any) -> float:
    """RT-DETR's CSV stores `timestamp` as HH:MM:SS or MM:SS clock-from-video-start."""
    raw = str(value or "").strip()
    if not raw:
        return 0.0
    if ":" in raw:
        parts = raw.split(":")
        try:
            if len(parts) == 3:
                h, m, s = parts
                return float(h) * 3600 + float(m) * 60 + float(s)
            if len(parts) == 2:
                m, s = parts
                return float(m) * 60 + float(s)
        except (TypeError, ValueError):
            return 0.0
    try:
        return float(raw)
    except (TypeError, ValueError):
        return 0.0


def _events_from_csv(rows: list[dict[str, Any]], video_id: str, video_record: dict[str, Any]) -> list[dict[str, Any]]:
    location_name = str(video_record.get("location") or "")
    base_timestamp = str(video_record.get("timestamp") or "")
    events: list[dict[str, Any]] = []
    for row in rows:
        try:
            frame_number = int(row.get("frame_number") or 0)
        except (TypeError, ValueError):
            frame_number = 0
        offset_seconds = _parse_clock_offset_seconds(row.get("timestamp"))
        try:
            track_id = int(row.get("track_id")) if row.get("track_id") not in (None, "") else None
        except (TypeError, ValueError):
            track_id = None
        class_name = (row.get("class_name") or "").strip()
        gate_name = (row.get("line_name") or "").strip()
        raw_direction = (row.get("direction") or "").strip().lower()
        direction: Optional[str] = raw_direction if raw_direction in ("in", "out") else None
        events.append(
            {
                "id": str(uuid4()),
                "type": "vehicle-detection",
                "location": location_name,
                "timestamp": base_timestamp,
                "description": f"{class_name or 'vehicle'} crossing {gate_name or 'gate'} ({raw_direction or 'unknown direction'})",
                "videoId": video_id,
                "frame": frame_number,
                "offsetSeconds": offset_seconds,
                "vehicleClass": class_name or None,
                "gateName": gate_name or None,
                "trackId": track_id,
                "direction": direction,
            }
        )
    return events


def _vehicle_count_from_summary(summary_path: Path, csv_rows: list[dict[str, Any]]) -> int:
    """Best-effort total vehicle count: parse summary if present, otherwise count CSV rows."""
    if summary_path.exists():
        try:
            text = summary_path.read_text(encoding="utf-8")
        except OSError:
            text = ""
        match = re.search(r"Total\s+(?:vehicles|crossings)\s*[:=]\s*(\d+)", text, re.IGNORECASE)
        if match:
            return int(match.group(1))
    return len(csv_rows)


def _reencode_to_h264(raw_path: Path, callback: Optional[VehicleProgressCallback] = None) -> Optional[Path]:
    """Re-encode an OpenCV mp4v video to H.264 so browsers can play it.

    Returns the path of the re-encoded file, or None if ffmpeg is unavailable
    or the re-encode fails (the caller should fall back to the raw file).
    """
    import shutil as _shutil

    ffmpeg = _shutil.which("ffmpeg")
    if not ffmpeg:
        return None

    h264_path = raw_path.with_stem(raw_path.stem + "_h264")
    cmd = [
        ffmpeg,
        "-y",  # overwrite without asking
        "-i", str(raw_path),
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "23",
        "-movflags", "+faststart",  # enables streaming / progressive download
        "-an",  # strip audio (inference videos have none)
        str(h264_path),
    ]

    _emit(callback, phase="finalizing", progressPercent=96,
          message="Re-encoding output video to H.264 for browser playback…")
    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            timeout=600,
        )
        if result.returncode != 0 or not h264_path.exists():
            return None
        # Replace the raw file with the re-encoded one and return the same path.
        raw_path.unlink(missing_ok=True)
        h264_path.rename(raw_path)
        return raw_path
    except Exception:
        h264_path.unlink(missing_ok=True)
        return None


def run_vehicle_inference(
    video_path: Path,
    video_record: dict[str, Any],
    *,
    counting_config_name: Optional[str] = None,
    progress_callback: Optional[VehicleProgressCallback] = None,
) -> dict[str, Any]:
    """Run RT-DETR inference against a vehicle video.

    Returns a dict shaped to feed `store.set_video_inference_result`:
        { "vehicleCount": int, "processedPath": str | None, "events": list }
    """
    video_id = str(video_record.get("id") or "")
    if not video_id:
        raise VehicleInferenceError("video record is missing an id")

    weights = _resolve_weights_path()
    config = _resolve_config_path()
    counting_config = _resolve_counting_config(
        str(video_record.get("locationId") or ""),
        counting_config_name,
    )

    output_dir = PROCESSED_VIDEOS_DIR / video_id
    output_dir.mkdir(parents=True, exist_ok=True)
    output_video = output_dir / f"{video_path.stem}.mp4"
    sidecar_stem = str(output_video.with_suffix(""))
    counts_csv = Path(sidecar_stem + "_counts.csv")
    summary_txt = Path(sidecar_stem + "_summary.txt")
    detections_csv = Path(sidecar_stem + "_detections.csv")

    # Form input + storage uses meters; RT-DETR's --road-length expects kilometers.
    raw_road_length_m = video_record.get("roadLengthM")
    road_length_m = float(raw_road_length_m) if isinstance(raw_road_length_m, (int, float)) and raw_road_length_m > 0 else 80.3
    road_length_km = road_length_m / 1000.0

    cmd: list[str] = [
        sys.executable,
        str(INFER_SCRIPT),
        "-c",
        str(config),
        "-r",
        str(weights),
        "-v",
        str(video_path),
        "-o",
        str(output_video),
        "-a",
        "./inference_requirements/annotations/instances_train.json",
        "-d",
        _device_for_runtime(),
        "-t",
        DETECTION_THRESHOLD,
        "--tracking",
        "--congestion",
        "--road-length",
        f"{road_length_km:.6f}",
        "--lanes",
        str(video_record.get("laneCount") or 2)
    ]
    if counting_config is not None:
        cmd.extend(["--counting-config", str(counting_config)])

    _emit(
        progress_callback,
        phase="tracking",
        progressPercent=5,
        message=(
            "RT-DETR starting"
            + (f" with counting config {counting_config.name}" if counting_config else " (detection only)")
        ),
    )

    env = os.environ.copy()
    env.setdefault("PYTHONIOENCODING", "utf-8")
    process = subprocess.Popen(
        cmd,
        cwd=str(RTDETR_DIR),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )
    try:
        _stream_progress(process, progress_callback)
        stdout_tail = process.stdout.read() if process.stdout else ""
        stderr_tail = process.stderr.read() if process.stderr else ""
    finally:
        return_code = process.wait()

    if return_code != 0:
        raise VehicleInferenceError(
            f"RT-DETR exited with code {return_code}.\nstderr tail:\n{stderr_tail[-2000:]}"
        )

    csv_rows = _parse_counts_csv(counts_csv)
    detection_rows = _parse_detections_csv(detections_csv)
    crossing_events = _events_from_csv(csv_rows, video_id, video_record)
    track_events = _track_events_from_detections(detection_rows, video_id, video_record)
    events = track_events + crossing_events
    if counting_config_name:
        vehicle_count = len(crossing_events)
    else:
        vehicle_count = len(track_events) if track_events else _vehicle_count_from_summary(summary_txt, csv_rows)

    # Re-encode to H.264 so the browser's <video> element can play it.
    # OpenCV's mp4v codec is not natively supported by most browsers.
    if output_video.exists():
        _reencode_to_h264(output_video, progress_callback)

    processed_relative: Optional[str] = None
    if output_video.exists():
        try:
            processed_relative = str(output_video.relative_to(store.BACKEND_DIR))
        except ValueError:
            processed_relative = str(output_video)

    _emit(
        progress_callback,
        phase="finalizing",
        progressPercent=98,
        message=f"Indexed {vehicle_count} vehicle detections across {len(events)} events.",
    )

    return {
        "vehicleCount": vehicle_count,
        "processedPath": processed_relative,
        "events": events,
        "stdoutTail": stdout_tail[-1500:] if stdout_tail else "",
    }
