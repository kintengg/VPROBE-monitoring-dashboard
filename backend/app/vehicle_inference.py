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


def _resolve_counting_config(location_id: str) -> Optional[Path]:
    """Look up the counting-config JSON associated with the seeded gate."""
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
    """Read stderr for tqdm-style progress lines and forward to the callback."""
    progress_re = re.compile(r"(\d+)%\|.*?\|\s*(\d+)/(\d+)")
    if process.stderr is None:
        return
    last_percent = -1
    for raw_line in iter(process.stderr.readline, ""):
        line = raw_line.rstrip()
        if not line:
            continue
        match = progress_re.search(line)
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


def _parse_counts_csv(csv_path: Path) -> list[dict[str, Any]]:
    if not csv_path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with csv_path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(row)
    return rows


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


def run_vehicle_inference(
    video_path: Path,
    video_record: dict[str, Any],
    *,
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
    counting_config = _resolve_counting_config(str(video_record.get("locationId") or ""))

    output_dir = PROCESSED_VIDEOS_DIR / video_id
    output_dir.mkdir(parents=True, exist_ok=True)
    output_video = output_dir / f"{video_path.stem}.mp4"
    counts_csv = output_video.with_suffix("")
    counts_csv = Path(str(counts_csv) + "_counts.csv")
    summary_txt = output_video.with_suffix("")
    summary_txt = Path(str(summary_txt) + "_summary.txt")

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
        "-d",
        _device_for_runtime(),
        "-t",
        DETECTION_THRESHOLD,
        "--tracking",
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
    events = _events_from_csv(csv_rows, video_id, video_record)
    vehicle_count = _vehicle_count_from_summary(summary_txt, csv_rows)

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
