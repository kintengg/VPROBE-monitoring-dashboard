from __future__ import annotations

import colorsys
import csv
import os
import re
import selectors
import shutil
import signal
import subprocess
import sys
import time
from collections.abc import Callable
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

from . import vehicle_store as store, vision

PREFERRED_ULTRALYTICS_TAG = "rtdetr-cli"
FALLBACK_ULTRALYTICS_TAG = "rtdetr-cli-cpu"
VIDEO_SUFFIXES = (".mp4", ".avi", ".mov", ".mkv", ".m4v")
MAX_TRACK_EVENTS = 50
TRACK_THUMBNAIL_MAX_EDGE = 224
SEMANTIC_CROP_LABEL_ORDER = ("best", "early", "mid", "late")
CLOCK_TIME_FORMATS = ("%H:%M", "%H:%M:%S", "%I:%M %p", "%I:%M:%S %p")
INFERENCE_REQUIREMENTS_DIR = store.BACKEND_DIR / "Occlusion-Robust-RTDETR" / "inference_requirements"
LEGACY_INFERENCE_REQUIREMENTS_DIR = store.BACKEND_DIR / "inference_requirements"
LEGACY_STORAGE_INFERENCE_REQUIREMENTS_DIR = store.STORAGE_DIR / "inference_requirements"
CANONICAL_INFERENCE_CONFIGS_DIR = store.BACKEND_DIR / "Occlusion-Robust-RTDETR" / "configs" / "rtdetr"
INFERENCE_ANNOTATIONS_DIR = INFERENCE_REQUIREMENTS_DIR / "annotations"
INFERENCE_COUNTING_DIR = INFERENCE_REQUIREMENTS_DIR / "counting"
_DEFAULT_INFERENCE_MAX_RUNTIME_SECONDS = 1800.0
try:
    INFERENCE_MAX_RUNTIME_SECONDS = max(
        1.0,
        float(os.getenv("INFERENCE_MAX_RUNTIME_SECONDS", str(_DEFAULT_INFERENCE_MAX_RUNTIME_SECONDS))),
    )
except (TypeError, ValueError):
    INFERENCE_MAX_RUNTIME_SECONDS = _DEFAULT_INFERENCE_MAX_RUNTIME_SECONDS


def _foot_point_norm(bounds: tuple[int, int, int, int], frame_image: Any) -> Optional[list[float]]:
    shape = getattr(frame_image, "shape", None)
    if not shape or len(shape) < 2:
        return None

    frame_height = int(shape[0])
    frame_width = int(shape[1])
    if frame_width <= 0 or frame_height <= 0:
        return None

    x1, _y1, x2, y2 = bounds
    foot_x = min(max(((x1 + x2) / 2.0) / float(frame_width), 0.0), 1.0)
    foot_y = min(max(y2 / float(frame_height), 0.0), 1.0)
    return [round(foot_x, 6), round(foot_y, 6)]


def _thesis_root_dir() -> Path:
    # Project root for the surveillance-system workspace.
    return store.BACKEND_DIR.parent


def _is_within_project(path: Path) -> bool:
    try:
        path.resolve(strict=False).relative_to(_thesis_root_dir().resolve(strict=False))
        return True
    except ValueError:
        return False


def _candidate_occlusion_repo_dirs() -> list[Path]:
    candidates: list[Path] = []

    configured_repo_dir = os.getenv("OCCLUSION_RTDETR_DIR", "").strip()
    if configured_repo_dir:
        configured_candidate = Path(configured_repo_dir).expanduser()
        if not configured_candidate.is_absolute():
            configured_candidate = (_thesis_root_dir() / configured_candidate).resolve(strict=False)
        if _is_within_project(configured_candidate):
            candidates.append(configured_candidate)

    candidates.append(store.BACKEND_DIR / "Occlusion-Robust-RTDETR")
    candidates.append(_thesis_root_dir() / "Occlusion-Robust-RTDETR")
    candidates.append(store.BACKEND_DIR / "vendor" / "Occlusion-Robust-RTDETR")
    candidates.append(_thesis_root_dir() / "backend" / "vendor" / "Occlusion-Robust-RTDETR")

    deduped: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        normalized = str(candidate.resolve(strict=False))
        if normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(candidate)
    return deduped


def _looks_like_occlusion_repo(candidate: Path) -> bool:
    infer_candidates = [
        candidate / "src" / "zoo" / "rtdetr" / "infer.py",
    ]
    return any(path.exists() for path in infer_candidates)


def _occlusion_repo_dir() -> Path:
    candidates = _candidate_occlusion_repo_dirs()
    for candidate in candidates:
        if _looks_like_occlusion_repo(candidate):
            return candidate
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def occlusion_repo_dir() -> Path:
    return _occlusion_repo_dir()


def _ensure_inference_requirements_layout() -> None:
    if INFERENCE_REQUIREMENTS_DIR.exists():
        return

    fallback_dirs = [LEGACY_INFERENCE_REQUIREMENTS_DIR, LEGACY_STORAGE_INFERENCE_REQUIREMENTS_DIR]
    for fallback_dir in fallback_dirs:
        if not fallback_dir.exists():
            continue

        INFERENCE_REQUIREMENTS_DIR.mkdir(parents=True, exist_ok=True)
        for child in fallback_dir.iterdir():
            if child.name == "configs":
                continue
            target = INFERENCE_REQUIREMENTS_DIR / child.name
            if child.is_dir():
                shutil.copytree(child, target, dirs_exist_ok=True)
            elif child.is_file() and not target.exists():
                target.write_bytes(child.read_bytes())
        return

    INFERENCE_REQUIREMENTS_DIR.mkdir(parents=True, exist_ok=True)


def requirements_root_dir() -> Path:
    _ensure_inference_requirements_layout()
    return INFERENCE_REQUIREMENTS_DIR


def requirements_config_dir() -> Path:
    CANONICAL_INFERENCE_CONFIGS_DIR.mkdir(parents=True, exist_ok=True)
    return CANONICAL_INFERENCE_CONFIGS_DIR


def list_infer_config_names() -> list[str]:
    config_dir = requirements_config_dir()
    config_dir.mkdir(parents=True, exist_ok=True)
    return sorted(
        path.name
        for path in config_dir.glob("*")
        if path.is_file() and path.suffix.lower() in {".yml", ".yaml"}
    )


def requirements_annotations_dir() -> Path:
    _ensure_inference_requirements_layout()
    return INFERENCE_ANNOTATIONS_DIR


def requirements_counting_dir() -> Path:
    _ensure_inference_requirements_layout()
    return INFERENCE_COUNTING_DIR


def list_counting_config_names() -> list[str]:
    counting_dir = requirements_counting_dir()
    counting_dir.mkdir(parents=True, exist_ok=True)
    return sorted(path.name for path in counting_dir.glob("*.json") if path.is_file())


def _infer_script_path() -> Path:
    configured_script = str(os.getenv("RTDETR_INFER_SCRIPT") or "").strip()
    if configured_script:
        configured_path = Path(configured_script).expanduser()
        if not configured_path.is_absolute():
            configured_path = (_occlusion_repo_dir() / configured_path).resolve(strict=False)
        return configured_path

    repo_dir = _occlusion_repo_dir()
    preferred_candidates = [
        repo_dir / "tools" / "infer.py",
        repo_dir / "src" / "zoo" / "rtdetr" / "infer.py",
    ]

    for candidate in preferred_candidates:
        if candidate.exists():
            return candidate

    return preferred_candidates[0]


def _infer_config_path(selected_config_name: Optional[str] = None) -> Path:
    selected_value = str(selected_config_name or "").strip()
    if not selected_value:
        model_info = store.get_model_info()
        selected_value = str(model_info.get("inferConfig") or "").strip()

    if selected_value:
        normalized_name = Path(selected_value).name
        if Path(normalized_name).suffix.lower() not in {".yml", ".yaml"}:
            normalized_name = f"{normalized_name}.yml"
        selected_candidate = requirements_config_dir() / normalized_name
        return selected_candidate

    configured_path = str(os.getenv("RTDETR_INFER_CONFIG") or "").strip()
    if configured_path:
        configured_candidate = Path(configured_path).expanduser()
        if not configured_candidate.is_absolute():
            configured_candidate = (_occlusion_repo_dir() / configured_candidate).resolve(strict=False)
        return configured_candidate

    occlusion_repo = _occlusion_repo_dir()
    preferred_candidates = [
        occlusion_repo / "configs" / "rtdetr" / "rtdetr_r50_final.yml",
        CANONICAL_INFERENCE_CONFIGS_DIR / "rtdetr_r50_final.yml",
    ]

    for candidate in preferred_candidates:
        if candidate.exists():
            return candidate

    return preferred_candidates[0]


def _infer_annotations_path() -> Optional[Path]:
    configured_path = str(os.getenv("RTDETR_ANNOTATIONS_PATH") or "").strip()
    if configured_path:
        configured_candidate = Path(configured_path).expanduser()
        if not configured_candidate.is_absolute():
            configured_candidate = (_occlusion_repo_dir() / configured_candidate).resolve(strict=False)
        return configured_candidate

    local_candidate = INFERENCE_ANNOTATIONS_DIR / "instances_train.json"
    if local_candidate.exists():
        return local_candidate

    for candidate in INFERENCE_ANNOTATIONS_DIR.glob("*.json"):
        if candidate.is_file():
            return candidate

    default_candidate = _occlusion_repo_dir() / "configs" / "dataset" / "MergedAll" / "annotations" / "instances_train.json"
    if default_candidate.exists():
        return default_candidate

    for candidate in _occlusion_repo_dir().glob("**/instances_train.json"):
        if candidate.is_file():
            return candidate

    return None


def _required_annotations_path() -> Path:
    configured_path = str(os.getenv("RTDETR_ANNOTATIONS_PATH") or "").strip()
    if configured_path:
        configured_candidate = Path(configured_path).expanduser()
        if not configured_candidate.is_absolute():
            configured_candidate = (_occlusion_repo_dir() / configured_candidate).resolve(strict=False)
        return configured_candidate

    inferred_path = _infer_annotations_path()
    if inferred_path is not None:
        return inferred_path

    return INFERENCE_ANNOTATIONS_DIR / "instances_train.json"


def _normalized_counting_location_suffix(location_name: Optional[str]) -> Optional[str]:
    if location_name is None:
        return None

    normalized_location = re.sub(r"[^a-z0-9.]+", " ", str(location_name).strip().lower()).strip()
    compact_location = re.sub(r"[^a-z0-9.]+", "", normalized_location)
    if not compact_location:
        return None

    has_gate_keyword = (
        re.search(r"\bgate\b", normalized_location) is not None
        or re.search(r"\bgate\s*\d", normalized_location) is not None
        or compact_location.startswith("gate")
    )
    has_g_prefix = re.match(r"^g\d", compact_location) is not None
    if not has_gate_keyword and not has_g_prefix:
        return None

    gate_match: Optional[re.Match[str]] = None
    if has_g_prefix:
        gate_match = re.match(r"^g(?P<gate>\d\.\d|\d{2}|\d)", compact_location)

    if gate_match is None and has_gate_keyword:
        gate_match = re.search(r"\bgate\s*(?P<gate>\d\.\d|\d{2}|\d)\b", normalized_location)
        if gate_match is None:
            gate_match = re.search(r"gate(?P<gate>\d\.\d|\d{2}|\d)", compact_location)

    if gate_match is None:
        return None

    normalized_gate = {
        "2": "g2",
        "29": "g2.9",
        "2.9": "g2.9",
        "3": "g3",
        "32": "g3.2",
        "3.2": "g3.2",
        "35": "g3.5",
        "3.5": "g3.5",
    }.get(gate_match.group("gate"))
    return normalized_gate


def _infer_counting_config_path(location_name: Optional[str] = None) -> Path:
    fallback_path = INFERENCE_COUNTING_DIR / "counting_config_g2.9.json"
    suffix = _normalized_counting_location_suffix(location_name)
    if not suffix:
        return fallback_path

    configured_path = INFERENCE_COUNTING_DIR / f"counting_config_{suffix}.json"
    if configured_path.exists():
        return configured_path

    return fallback_path


def resolve_counting_config_path(
    selected_config_name: Optional[str] = None,
    location_name: Optional[str] = None,
) -> Path:
    if selected_config_name:
        raw_name = str(selected_config_name).strip()
        if raw_name:
            normalized_name = Path(raw_name).name
            if not normalized_name.lower().endswith(".json"):
                normalized_name = f"{normalized_name}.json"
            explicit_path = requirements_counting_dir() / normalized_name
            if explicit_path.exists() and explicit_path.is_file():
                return explicit_path
            raise FileNotFoundError(f"Selected counting config was not found: {normalized_name}")

    return _infer_counting_config_path(location_name)


def _first_missing_path(paths: list[Path]) -> Optional[Path]:
    return next((path for path in paths if not path.exists()), None)


def _model_search_roots() -> list[Path]:
    candidates = [
        store.MODELS_DIR,
        store.LEGACY_MODELS_DIR,
        store.STORAGE_DIR,
        _thesis_root_dir(),
        store.BACKEND_DIR,
    ]

    configured_model_dir = os.getenv("MODEL_SEARCH_DIR", "").strip()
    if configured_model_dir:
        configured_candidate = Path(configured_model_dir).expanduser()
        if not configured_candidate.is_absolute():
            configured_candidate = (_thesis_root_dir() / configured_candidate).resolve(strict=False)
        if _is_within_project(configured_candidate):
            candidates.insert(0, configured_candidate)

    deduped: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        normalized = str(candidate.resolve(strict=False))
        if normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(candidate)
    return deduped


def _find_models_by_name(filename: Optional[str] = None) -> list[Path]:
    suffixes = {".pt", ".pth"}
    ignored_dirs = {".git", ".next", "node_modules", "__pycache__", ".venv", "venv"}
    target_name = Path(filename).name if filename else None
    discovered: list[Path] = []
    seen: set[str] = set()

    for root in _model_search_roots():
        if not root.exists():
            continue

        for walk_root, dirnames, filenames in os.walk(root):
            dirnames[:] = [dirname for dirname in dirnames if dirname not in ignored_dirs]

            if target_name:
                if target_name not in filenames:
                    continue
                candidate = Path(walk_root) / target_name
                if candidate.suffix.lower() not in suffixes:
                    continue
                normalized_candidate = str(candidate.resolve(strict=False))
                if normalized_candidate in seen:
                    continue
                seen.add(normalized_candidate)
                discovered.append(candidate)
                continue

            for name in filenames:
                if Path(name).suffix.lower() not in suffixes:
                    continue
                candidate = Path(walk_root) / name
                normalized_candidate = str(candidate.resolve(strict=False))
                if normalized_candidate in seen:
                    continue
                seen.add(normalized_candidate)
                discovered.append(candidate)

    def _mtime(path: Path) -> float:
        try:
            return path.stat().st_mtime
        except OSError:
            return 0.0

    discovered.sort(key=_mtime, reverse=True)
    return discovered


def resolve_model_path(model_name: Optional[str]) -> Optional[Path]:
    allowed_suffixes = {".pt", ".pth"}
    model_value = str(model_name or "").strip()
    if model_value:
        explicit_path = Path(model_value).expanduser()
        if explicit_path.is_absolute() and explicit_path.exists() and explicit_path.is_file() and explicit_path.suffix.lower() in allowed_suffixes:
            return explicit_path

        for base_dir in (store.BACKEND_DIR, _thesis_root_dir()):
            candidate = (base_dir / explicit_path).resolve(strict=False)
            if (
                candidate.exists()
                and candidate.is_file()
                and candidate.suffix.lower() in allowed_suffixes
                and _is_within_project(candidate)
            ):
                return candidate

        default_models_dir_candidate = store.MODELS_DIR / Path(model_value).name
        if (
            default_models_dir_candidate.exists()
            and default_models_dir_candidate.is_file()
            and default_models_dir_candidate.suffix.lower() in allowed_suffixes
        ):
            return default_models_dir_candidate

        discovered_named_models = _find_models_by_name(model_value)
        if discovered_named_models:
            return discovered_named_models[0]

    discovered_models = _find_models_by_name()
    if discovered_models:
        return discovered_models[0]

    return None


def ultralytics_status() -> dict[str, Any]:
    _ensure_inference_requirements_layout()
    model_info = store.get_model_info()
    model_name = model_info.get("currentModel")
    infer_config_name = str(model_info.get("inferConfig") or "").strip() or None
    model_path = resolve_model_path(model_name)
    repo_dir = _occlusion_repo_dir()
    infer_config_path = _infer_config_path(infer_config_name)
    infer_script = _infer_script_path()
    fixed_required_paths = [
        repo_dir,
        infer_config_path,
        _infer_counting_config_path(),
        _required_annotations_path(),
    ]
    missing_fixed_path = _first_missing_path(fixed_required_paths)
    pipeline_installed = missing_fixed_path is None
    version = None
    if pipeline_installed:
        try:
            version = infer_script.stat().st_mtime_ns
            version = str(version)
        except OSError:
            version = "installed"

    return {
        "installed": pipeline_installed,
        "version": version,
        "packagePath": _project_relative_path(infer_script),
        "vendoredPath": _project_relative_path(repo_dir),
        "usingVendoredCopy": pipeline_installed,
        "preferredTag": PREFERRED_ULTRALYTICS_TAG,
        "fallbackTag": FALLBACK_ULTRALYTICS_TAG,
        "currentModel": model_name,
        "currentInferConfig": infer_config_name,
        "inferConfigPath": _project_relative_path(infer_config_path),
        "modelPath": _project_relative_path(model_path),
        "modelExists": model_path is not None,
        "ready": pipeline_installed and model_path is not None,
        "missingFixedPath": _project_relative_path(missing_fixed_path),
    }


def _project_relative_path(path: Optional[Path]) -> Optional[str]:
    if path is None:
        return None

    resolved_path = path.resolve(strict=False)
    relative_candidates: list[str] = []

    for root in (store.BACKEND_DIR.parent, _thesis_root_dir()):
        try:
            relative_candidates.append(str(resolved_path.relative_to(root.resolve(strict=False))))
        except ValueError:
            continue

    if relative_candidates:
        return min(relative_candidates, key=len)

    try:
        # Keep path relative even when outside known roots.
        return os.path.relpath(str(resolved_path), start=str(store.BACKEND_DIR.parent.resolve(strict=False)))
    except ValueError:
        return str(path.name)


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


def preferred_inference_device() -> str:
    try:
        import torch
    except Exception:
        return "cpu"

    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _inference_python_executable() -> Path:
    configured_python = str(os.getenv("INFERENCE_PYTHON_BIN") or "").strip()
    if configured_python:
        configured_path = Path(configured_python).expanduser()
        if configured_path.exists() and configured_path.is_file():
            return configured_path

    backend_venv_python = store.BACKEND_DIR / "venv" / "bin" / "python"
    if backend_venv_python.exists() and backend_venv_python.is_file():
        return backend_venv_python

    return Path(sys.executable)


def _build_rtdetr_command(
    *,
    model_path: Path,
    video_path: Path,
    output_path: Path,
    infer_config_path: Path,
    counting_config_path: Path,
    annotations_path: Path,
    enable_display: bool = False,
) -> list[str]:
    command = [
        str(_inference_python_executable()),
        str(_infer_script_path().resolve()),
        "--config",
        str(infer_config_path.resolve()),
        "-r",
        str(model_path.resolve()),
        "-v",
        str(video_path.resolve()),
        "-o",
        str(output_path.resolve()),
        "--tracking",
        "--counting-config",
        str(counting_config_path.resolve()),
        "--congestion",
        "--road-length",
        "0.0803",
        "--lanes",
        "2",
        "-d",
        preferred_inference_device(),
        "--batch-size",
        "32",
        "-a",
        str(annotations_path.resolve()),
    ]

    if enable_display:
        command.append("--display")

    return command


def _wait_for_process_with_cancellation(
    process: subprocess.Popen[Any],
    *,
    progress_callback: Optional[Callable[[dict[str, Any]], None]],
    timeout_seconds: Optional[float] = None,
    model_name: Optional[str] = None,
    infer_config_name: Optional[str] = None,
) -> tuple[int, str, str]:
    started_at = time.monotonic()
    effective_timeout_seconds = (
        max(1.0, float(timeout_seconds))
        if timeout_seconds is not None
        else INFERENCE_MAX_RUNTIME_SECONDS
    )
    stdout_chunks: list[str] = []
    stderr_chunks: list[str] = []
    stream_buffers: dict[str, str] = {"stdout": "", "stderr": ""}
    last_emitted_status = ""
    last_emitted_at = 0.0

    selector = selectors.DefaultSelector()
    if process.stdout is not None:
        selector.register(process.stdout, selectors.EVENT_READ, data="stdout")
    if process.stderr is not None:
        selector.register(process.stderr, selectors.EVENT_READ, data="stderr")

    def _emit_script_status(source: str, line: str) -> None:
        nonlocal last_emitted_status, last_emitted_at
        if progress_callback is None:
            return

        cleaned_line = line.strip()
        if not cleaned_line:
            return

        now = time.monotonic()
        status_line = f"RT-DETR {source}: {cleaned_line}"
        if status_line == last_emitted_status and (now - last_emitted_at) < 0.5:
            return

        progress_callback(
            {
                "phase": "tracking",
                "message": status_line,
            }
        )
        last_emitted_status = status_line
        last_emitted_at = now

    def _drain_stream_event(source: str, stream_obj: Any) -> None:
        try:
            raw_chunk = os.read(stream_obj.fileno(), 4096)
        except OSError:
            raw_chunk = b""

        if not raw_chunk:
            try:
                selector.unregister(stream_obj)
            except Exception:
                pass
            return

        text_chunk = raw_chunk.decode("utf-8", errors="replace")
        if source == "stdout":
            stdout_chunks.append(text_chunk)
        else:
            stderr_chunks.append(text_chunk)

        normalized_chunk = text_chunk.replace("\r", "\n")
        stream_buffers[source] += normalized_chunk
        lines = stream_buffers[source].split("\n")
        stream_buffers[source] = lines.pop() if lines else ""

        for line in lines:
            _emit_script_status(source, line)

    while True:
        _run_cancel_check(progress_callback)
        events = selector.select(timeout=0.2)
        for key, _mask in events:
            source = str(key.data)
            _drain_stream_event(source, key.fileobj)

        if process.poll() is not None:
            # Drain any remaining bytes and flush pending partial lines.
            for source, stream_obj in (("stdout", process.stdout), ("stderr", process.stderr)):
                if stream_obj is None:
                    continue
                while True:
                    previous_size = len(stdout_chunks) + len(stderr_chunks)
                    _drain_stream_event(source, stream_obj)
                    current_size = len(stdout_chunks) + len(stderr_chunks)
                    if current_size == previous_size:
                        break

            for source, trailing in stream_buffers.items():
                _emit_script_status(source, trailing)

            try:
                selector.close()
            except Exception:
                pass

            return process.returncode or 0, "".join(stdout_chunks), "".join(stderr_chunks)

        elapsed_seconds = time.monotonic() - started_at
        if elapsed_seconds > effective_timeout_seconds:
            _terminate_process(process)
            model_label = model_name or "unknown-model"
            config_label = infer_config_name or "unknown-config"
            raise RuntimeError(
                "RT-DETR inference command timed out "
                f"after {effective_timeout_seconds:.1f} seconds "
                f"(model={model_label}, config={config_label})."
            )


def _runtime_timeout_seconds(video_path: Path) -> float:
    detected_duration_seconds = detect_video_duration_seconds(video_path)
    if detected_duration_seconds is None:
        return INFERENCE_MAX_RUNTIME_SECONDS

    # Allow slower backbones enough runtime while still preventing indefinite hangs.
    multiplier = 12.0
    base_timeout = max(180.0, (float(detected_duration_seconds) * multiplier) + 120.0)
    return min(INFERENCE_MAX_RUNTIME_SECONDS, base_timeout)


def _terminate_process(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return

    terminated_with_group_signal = False
    if hasattr(os, "killpg") and hasattr(os, "getpgid"):
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            terminated_with_group_signal = True
        except ProcessLookupError:
            return
        except OSError:
            terminated_with_group_signal = False

    if not terminated_with_group_signal:
        process.terminate()

    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        if terminated_with_group_signal and hasattr(os, "killpg") and hasattr(os, "getpgid"):
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            except ProcessLookupError:
                return
            except OSError:
                process.kill()
        else:
            process.kill()
        process.wait(timeout=5)


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


def _resolve_processed_video_path(*, explicit_output_path: Path, save_dir: Path, source_path: Path) -> Path:
    if explicit_output_path.exists():
        return explicit_output_path

    for suffix in VIDEO_SUFFIXES:
        candidate = explicit_output_path.with_suffix(suffix)
        if candidate.exists():
            return candidate

    fallback_path = _find_processed_video(save_dir, source_path)
    if fallback_path is not None:
        return fallback_path

    raise RuntimeError(
        "RT-DETR inference finished but no processed output video was found. "
        f"Expected output near: {explicit_output_path}"
    )


def _ensure_browser_playable_mp4(video_path: Path) -> Path:
    if video_path.suffix.lower() != ".mp4":
        return video_path

    ffmpeg_path = shutil.which("ffmpeg")
    if not ffmpeg_path:
        return video_path

    transcoded_path = video_path.with_name(f"{video_path.stem}.h264.mp4")
    command = [
        ffmpeg_path,
        "-y",
        "-i",
        str(video_path),
        "-map",
        "0:v:0",
        "-map",
        "0:a?",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-c:a",
        "aac",
        str(transcoded_path),
    ]

    try:
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
    except Exception:
        return video_path

    if completed.returncode != 0 or not transcoded_path.exists():
        return video_path

    try:
        transcoded_path.replace(video_path)
    except OSError:
        return video_path

    return video_path


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


def detect_video_duration_seconds(video_path: Path) -> Optional[int]:
    fps, frame_count = _read_video_metadata(video_path)
    if frame_count is None or fps <= 0:
        return None

    duration_seconds = int(round(float(frame_count) / float(fps)))
    return duration_seconds if duration_seconds > 0 else None


def _counts_csv_path(output_video_path: Path) -> Path:
    return Path(f"{output_video_path.with_suffix('')}_counts.csv")


def _parse_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return int(text)
    except (TypeError, ValueError):
        return None


def _counts_description(track_id: Optional[int], line_name: str, direction: str) -> str:
    pedestrian_label = f"Pedestrian ID #{track_id}" if track_id is not None else "Pedestrian"
    crossing_line = line_name or "counting line"
    crossing_direction = direction or "unknown direction"
    return f"{pedestrian_label} crossed {crossing_line} ({crossing_direction})"


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


def _vehicle_class_label(class_name: Optional[str]) -> Optional[str]:
    if not class_name:
        return None

    return " ".join(part.capitalize() for part in class_name.replace("_", "-").split("-") if part)


def _parse_counts_csv(
    *,
    output_video_path: Path,
    video_record: Optional[dict[str, Any]],
) -> tuple[int, list[dict[str, Any]], list[dict[str, Any]]]:
    counts_path = _counts_csv_path(output_video_path)
    if not counts_path.exists() or not counts_path.is_file():
        return 0, [], []

    try:
        with counts_path.open("r", encoding="utf-8", newline="") as counts_file:
            reader = csv.DictReader(counts_file)
            rows = [dict(row) for row in reader if isinstance(row, dict)]
    except (OSError, UnicodeDecodeError, csv.Error):
        return 0, [], []

    if not rows:
        return 0, [], []

    record = video_record or {}
    default_timestamp = str(record.get("startTime") or "Unknown Time")
    location = str(record.get("location") or "Unknown Location")
    video_id = record.get("id")

    events: list[dict[str, Any]] = []
    all_track_ids: set[int] = set()
    person_track_ids: set[int] = set()
    saw_person_label = False
    track_summaries: dict[int, dict[str, Any]] = {}

    for index, row in enumerate(rows):
        raw_track_id = row.get("track_id")
        track_id = _parse_int(raw_track_id)
        frame_number = _parse_int(row.get("frame_number"))
        timestamp = str(row.get("timestamp") or "").strip() or default_timestamp
        line_name = str(row.get("line_name") or "").strip()
        direction = str(row.get("direction") or "").strip()
        class_name = str(row.get("class_name") or "").strip().lower()
        normalized_vehicle_class = _normalize_vehicle_class_name(class_name)
        if class_name == "person":
            saw_person_label = True

        if track_id is not None:
            all_track_ids.add(track_id)
            if class_name == "person":
                person_track_ids.add(track_id)

            track_summary = track_summaries.get(track_id)
            if track_summary is None:
                track_summary = {
                    "id": f"trk-{track_id}-{index:04d}",
                    "videoId": video_id,
                    "pedestrianId": track_id,
                    "firstTimestamp": timestamp,
                    "lastTimestamp": timestamp,
                    "firstFrame": frame_number,
                    "lastFrame": frame_number,
                    "firstOffsetSeconds": None,
                    "lastOffsetSeconds": None,
                    "appearanceSummary": "Track reconstructed from counting events.",
                    "thumbnailPath": None,
                    "semanticCrops": [],
                    "trajectorySamples": [],
                }
                track_summaries[track_id] = track_summary
            else:
                track_summary["lastTimestamp"] = timestamp
                if frame_number is not None:
                    first_frame = track_summary.get("firstFrame")
                    last_frame = track_summary.get("lastFrame")
                    if first_frame is None or frame_number < first_frame:
                        track_summary["firstFrame"] = frame_number
                    if last_frame is None or frame_number > last_frame:
                        track_summary["lastFrame"] = frame_number

        event_id_suffix = track_id if track_id is not None else "na"
        events.append(
            {
                "id": f"evt-{event_id_suffix}-{index:04d}",
                "type": "detection",
                "location": location,
                "timestamp": timestamp,
                "description": _counts_description(track_id, line_name, direction),
                "videoId": video_id,
                "pedestrianId": track_id,
                "frame": frame_number,
                "offsetSeconds": None,
                "occlusionClass": None,
                "vehicleClass": normalized_vehicle_class,
                "vehicleClassLabel": _vehicle_class_label(normalized_vehicle_class),
            }
        )

    pedestrian_count = len(person_track_ids) if saw_person_label else len(all_track_ids)
    pedestrian_tracks = list(track_summaries.values())
    return pedestrian_count, events, pedestrian_tracks


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
    return _project_relative_path(target)


def _semantic_crop_sort_key(crop: dict[str, Any]) -> tuple[int, float]:
    label = str(crop.get("label") or "")
    try:
        label_index = SEMANTIC_CROP_LABEL_ORDER.index(label)
    except ValueError:
        label_index = len(SEMANTIC_CROP_LABEL_ORDER)
    try:
        offset_seconds = float(crop.get("offsetSeconds") or 0.0)
    except (TypeError, ValueError):
        offset_seconds = 0.0
    return label_index, offset_seconds


def _upsert_semantic_crop(
    track_summary: dict[str, Any],
    *,
    label: str,
    image: Any,
    save_dir: Path,
    pedestrian_id: int,
    frame_index: int,
    timestamp: str,
    offset_seconds: float,
) -> None:
    target = save_dir / "tracks" / f"track-{pedestrian_id}-{label}.jpg"
    saved_path = _save_track_thumbnail(image, target)
    if not saved_path:
        return

    semantic_crops = track_summary.setdefault("semanticCrops", [])
    payload = {
        "label": label,
        "path": saved_path,
        "frame": frame_index,
        "timestamp": timestamp,
        "offsetSeconds": offset_seconds,
    }
    existing_index = next((index for index, item in enumerate(semantic_crops) if str(item.get("label") or "") == label), None)
    if existing_index is None:
        semantic_crops.append(payload)
    else:
        semantic_crops[existing_index] = payload
    semantic_crops.sort(key=_semantic_crop_sort_key)


def _update_semantic_crops(
    track_summary: dict[str, Any],
    *,
    image: Any,
    save_dir: Path,
    pedestrian_id: int,
    frame_index: int,
    timestamp: str,
    offset_seconds: float,
    is_new_best: bool,
) -> None:
    semantic_crops = track_summary.setdefault("semanticCrops", [])
    by_label = {str(item.get("label") or ""): item for item in semantic_crops}

    if "early" not in by_label:
        _upsert_semantic_crop(
            track_summary,
            label="early",
            image=image,
            save_dir=save_dir,
            pedestrian_id=pedestrian_id,
            frame_index=frame_index,
            timestamp=timestamp,
            offset_seconds=offset_seconds,
        )

    _upsert_semantic_crop(
        track_summary,
        label="late",
        image=image,
        save_dir=save_dir,
        pedestrian_id=pedestrian_id,
        frame_index=frame_index,
        timestamp=timestamp,
        offset_seconds=offset_seconds,
    )

    mid_crop = next((item for item in semantic_crops if str(item.get("label") or "") == "mid"), None)
    first_frame = int(track_summary.get("firstFrame") or frame_index)
    midpoint = (first_frame + frame_index) / 2.0
    current_distance = abs(frame_index - midpoint)
    existing_distance = None
    if mid_crop is not None:
        try:
            existing_distance = abs(float(mid_crop.get("frame") or frame_index) - midpoint)
        except (TypeError, ValueError):
            existing_distance = None
    if mid_crop is None or existing_distance is None or current_distance <= existing_distance:
        _upsert_semantic_crop(
            track_summary,
            label="mid",
            image=image,
            save_dir=save_dir,
            pedestrian_id=pedestrian_id,
            frame_index=frame_index,
            timestamp=timestamp,
            offset_seconds=offset_seconds,
        )

    if is_new_best:
        _upsert_semantic_crop(
            track_summary,
            label="best",
            image=image,
            save_dir=save_dir,
            pedestrian_id=pedestrian_id,
            frame_index=frame_index,
            timestamp=timestamp,
            offset_seconds=offset_seconds,
        )


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


def _run_cancel_check(progress_callback: Optional[Callable[[dict[str, Any]], None]]) -> None:
    cancel_check = getattr(progress_callback, "cancel_check", None)
    if callable(cancel_check):
        cancel_check()


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

    _run_cancel_check(progress_callback)

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
        _run_cancel_check(progress_callback)
        metadata = vision.enrich_track_thumbnail(thumbnail_file)
        _run_cancel_check(progress_callback)
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
    progress_callback: Optional[Callable[[dict[str, Any]], None]] = None,
) -> dict[str, Any]:
    status = ultralytics_status()
    if not status["installed"]:
        missing_path = status.get("missingFixedPath")
        if missing_path:
            raise RuntimeError(f"RT-DETR inference pipeline is not ready. Missing required path: {missing_path}")
        raise RuntimeError("RT-DETR inference pipeline is not ready.")
    if not status["modelExists"]:
        raise RuntimeError(
            "The active model file is missing. Expected under "
            f"{_project_relative_path(store.MODELS_DIR)} "
            "(legacy fallback: backend/storage/models)."
        )

    model_path = resolve_model_path(model_name or status["currentModel"])
    if model_path is None:
        raise RuntimeError("Could not resolve the active model path for inference.")

    save_name = str((video_record or {}).get("id") or video_path.stem)
    save_dir = store.PROCESSED_VIDEOS_DIR / save_name
    save_dir.mkdir(parents=True, exist_ok=True)
    output_video_path = save_dir / f"{video_path.stem}-processed.mp4"
    selected_counting_config_name = str((video_record or {}).get("countingConfig") or "").strip() or None
    show_live_preview = bool((video_record or {}).get("showLivePreview"))
    counting_config_path = resolve_counting_config_path(
        selected_config_name=selected_counting_config_name,
        location_name=(video_record or {}).get("location"),
    )
    infer_config_path = _infer_config_path()
    annotations_path = _required_annotations_path()
    if not annotations_path.exists() or not annotations_path.is_file():
        raise RuntimeError(
            "RT-DETR inference requires an annotations JSON file. "
            f"Missing required path: {_project_relative_path(annotations_path)}"
        )
    command = _build_rtdetr_command(
        model_path=model_path,
        video_path=video_path,
        output_path=output_video_path,
        infer_config_path=infer_config_path,
        counting_config_path=counting_config_path,
        annotations_path=annotations_path,
        enable_display=show_live_preview,
    )

    _run_cancel_check(progress_callback)
    if progress_callback is not None:
        progress_callback(
            {
                "progressPercent": 0,
                "phase": "tracking",
                "message": "Starting RT-DETR detection and tracking pipeline...",
            }
        )

    process: Optional[subprocess.Popen[str]] = None
    try:
        timeout_seconds = _runtime_timeout_seconds(video_path)
        process = subprocess.Popen(
            command,
            cwd=str(_occlusion_repo_dir().resolve()),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )

        return_code, _stdout, stderr = _wait_for_process_with_cancellation(
            process,
            progress_callback=progress_callback,
            timeout_seconds=timeout_seconds,
            model_name=model_path.name,
            infer_config_name=infer_config_path.name,
        )

        if progress_callback is not None:
            progress_callback(
                {
                    "progressPercent": 99,
                    "phase": "finalizing",
                    "message": "Finalizing processed video...",
                }
            )

        if return_code != 0:
            stderr_text = (stderr or "").strip()
            raise RuntimeError(
                "RT-DETR inference command failed"
                + (f": {stderr_text}" if stderr_text else ".")
            )
    except InterruptedError:
        if process is not None:
            _terminate_process(process)
        raise
    except Exception:
        if process is not None:
            _terminate_process(process)
        raise

    _run_cancel_check(progress_callback)
    resolved_output_video_path = _resolve_processed_video_path(
        explicit_output_path=output_video_path,
        save_dir=save_dir,
        source_path=video_path,
    )
    if not resolved_output_video_path.exists():
        raise RuntimeError(
            "RT-DETR inference finished but the resolved processed output file does not exist: "
            f"{resolved_output_video_path}"
        )

    resolved_output_video_path = _ensure_browser_playable_mp4(resolved_output_video_path)

    try:
        processed_path = str(resolved_output_video_path.relative_to(store.BACKEND_DIR))
    except ValueError:
        processed_path = _project_relative_path(resolved_output_video_path)
    if not processed_path:
        raise RuntimeError(
            "RT-DETR inference finished but could not resolve processedPath for API response."
        )

    pedestrian_count, events, pedestrian_tracks = _parse_counts_csv(
        output_video_path=output_video_path,
        video_record=video_record,
    )

    return {
        "pedestrianCount": pedestrian_count,
        "processedPath": processed_path,
        "events": events,
        "pedestrianTracks": pedestrian_tracks,
    }
