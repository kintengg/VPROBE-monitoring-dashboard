from __future__ import annotations

import json
import os
import re
import shutil
from threading import Lock, Thread
from pathlib import Path
from typing import Any, Optional
from urllib import error as urllib_error
from urllib import request as urllib_request
from uuid import uuid4

from fastapi import FastAPI, File, Form, HTTPException, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

from . import gemini, inference, model_registry, schemas, store, vehicle_inference, vehicle_store
from .vehicle import counting as vehicle_counting
from .vehicle import gates as vehicle_gates

REPO_ROOT = Path(__file__).resolve().parents[2]
LOCAL_ENV_PATHS = (REPO_ROOT / ".env.local", REPO_ROOT / "backend" / ".env.local")
GOOGLE_PLACES_TEXT_SEARCH_URL = "https://places.googleapis.com/v1/places:searchText"
GOOGLE_PLACES_FIELD_MASK = ",".join(
    [
        "places.id",
        "places.displayName",
        "places.formattedAddress",
        "places.location",
        "places.types",
    ]
)
ATENEO_LOCATION_BIAS = {
    "circle": {
        "center": {"latitude": 14.6397, "longitude": 121.0775},
        "radius": 15000.0,
    }
}
SEARCH_BACKFILL_LOCK = Lock()
SEARCH_BACKFILL_THREAD: Optional[Thread] = None


def load_local_env() -> None:
    for env_path in LOCAL_ENV_PATHS:
        if not env_path.exists():
            continue

        for line in env_path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if stripped.startswith("export "):
                stripped = stripped[7:].strip()
            if "=" not in stripped:
                continue

            key, value = stripped.split("=", 1)
            normalized_key = key.strip()
            normalized_value = value.strip().strip('"').strip("'")
            if normalized_key:
                os.environ.setdefault(normalized_key, normalized_value)


load_local_env()

app = FastAPI(title="ALIVE Engine Local API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
store.ensure_storage_layout()
app.mount("/storage", StaticFiles(directory=store.STORAGE_DIR, check_dir=False), name="storage")


def safe_filename(filename: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", Path(filename).name)
    return cleaned.strip("-") or "upload.bin"


def search_google_places(query: str) -> list[dict[str, Any]]:
    api_key = os.getenv("GOOGLE_MAPS_API_KEY")
    if not api_key:
        raise HTTPException(status_code=503, detail="Location search is unavailable. Set GOOGLE_MAPS_API_KEY on the backend.")

    request_body = {
        "textQuery": query,
        "pageSize": 5,
        "languageCode": "en",
        "regionCode": "PH",
        "locationBias": ATENEO_LOCATION_BIAS,
    }

    request = urllib_request.Request(
        GOOGLE_PLACES_TEXT_SEARCH_URL,
        data=json.dumps(request_body).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "X-Goog-Api-Key": api_key,
            "X-Goog-FieldMask": GOOGLE_PLACES_FIELD_MASK,
        },
        method="POST",
    )

    try:
        with urllib_request.urlopen(request, timeout=10) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib_error.HTTPError as exc:
        message = "Google Places search failed."
        try:
            error_payload = json.loads(exc.read().decode("utf-8"))
            message = error_payload.get("error", {}).get("message") or message
        except Exception:
            pass
        raise HTTPException(status_code=502, detail=message) from exc
    except urllib_error.URLError as exc:
        raise HTTPException(status_code=502, detail="Google Places search is unavailable right now.") from exc

    results: list[dict[str, Any]] = []
    for place in payload.get("places", []):
        location = place.get("location") or {}
        latitude = location.get("latitude")
        longitude = location.get("longitude")
        if latitude is None or longitude is None:
            continue

        display_name = place.get("displayName") or {}
        results.append(
            {
                "name": display_name.get("text") or place.get("formattedAddress") or query,
                "address": place.get("formattedAddress") or "",
                "latitude": latitude,
                "longitude": longitude,
                "placeId": place.get("id"),
                "types": place.get("types") or [],
            }
        )

    return results


@app.on_event("startup")
def startup() -> None:
    store.ensure_storage_layout()
    model_registry.init_registry()
    vehicle_store.ensure_gate_locations_seeded()


def _video_needs_search_backfill(state: dict[str, Any], video: dict[str, Any]) -> bool:
    video_id = str(video.get("id") or "")
    if not video_id:
        return False

    has_tracks = any(track.get("videoId") == video_id for track in state.get("pedestrianTracks", []))
    has_events = any(event.get("videoId") == video_id for event in state.get("events", []))
    if has_tracks or has_events:
        return False

    return store.resolve_video_source_path(video) is not None


def _latest_legacy_video_for_search() -> Optional[dict[str, Any]]:
    state = store.load_state()
    for video in reversed(state.get("videos", [])):
        if _video_needs_search_backfill(state, video):
            return video
    return None


def _backfill_legacy_video_search_metadata(video: dict[str, Any]) -> bool:
    refreshed_video = store.get_video(str(video.get("id") or ""))
    if refreshed_video is None:
        return False

    state = store.load_state()
    if not _video_needs_search_backfill(state, refreshed_video):
        return True

    source_path = store.resolve_video_source_path(refreshed_video)
    if source_path is None:
        return False

    try:
        result = inference.run_video_inference(source_path, None, refreshed_video, False, None)
    except Exception:
        return False

    store.set_video_inference_result(
        video_id=refreshed_video["id"],
        pedestrian_count=result.get("pedestrianCount", 0),
        processed_path=result.get("processedPath"),
        events=result.get("events", []),
        pedestrian_tracks=result.get("pedestrianTracks", []),
    )
    return True


def _search_backfill_worker() -> None:
    global SEARCH_BACKFILL_THREAD

    try:
        while True:
            video = _latest_legacy_video_for_search()
            if video is None:
                return
            if not _backfill_legacy_video_search_metadata(video):
                return
    finally:
        with SEARCH_BACKFILL_LOCK:
            SEARCH_BACKFILL_THREAD = None


def _ensure_search_metadata() -> None:
    global SEARCH_BACKFILL_THREAD

    if _latest_legacy_video_for_search() is None:
        return

    with SEARCH_BACKFILL_LOCK:
        if SEARCH_BACKFILL_THREAD is not None and SEARCH_BACKFILL_THREAD.is_alive():
            return
        SEARCH_BACKFILL_THREAD = Thread(target=_search_backfill_worker, name="search-metadata-backfill", daemon=True)
        SEARCH_BACKFILL_THREAD.start()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/locations", response_model=list[schemas.LocationRecord])
def get_locations(date: Optional[str] = None, domain: Optional[str] = None) -> list[dict]:
    return store.list_locations(date, domain)


@app.get("/api/locations/search", response_model=list[schemas.LocationSearchResult])
def search_locations(query: str) -> list[dict[str, Any]]:
    normalized_query = query.strip()
    if not normalized_query:
        return []
    return search_google_places(normalized_query)


@app.post("/api/locations", response_model=schemas.LocationRecord, status_code=201)
def create_location(payload: schemas.LocationCreate) -> dict:
    return store.add_location(payload.model_dump())


@app.put("/api/locations/{location_id}", response_model=schemas.LocationRecord)
def update_location(location_id: str, payload: schemas.LocationCreate) -> dict:
    try:
        return store.update_location(location_id, payload.model_dump())
    except ValueError as exc:
        if str(exc) == "Location not found":
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.delete("/api/locations/{location_id}", status_code=204)
def delete_location(location_id: str) -> Response:
    deleted = store.delete_location(location_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Location not found")
    return Response(status_code=204)


@app.get("/api/videos", response_model=list[schemas.VideoRecord])
def get_videos() -> list[dict]:
    return store.list_videos()


@app.get("/api/videos/{video_id}", response_model=schemas.VideoDetailRecord)
def get_video(video_id: str) -> dict:
    video = store.get_video_detail(video_id)
    if video is None:
        raise HTTPException(status_code=404, detail="Video not found")
    return video


@app.get("/api/videos/uploads/history", response_model=list[schemas.VideoUploadStatus])
def get_video_upload_history() -> list[dict[str, Any]]:
    return store.list_upload_statuses()


@app.get("/api/videos/uploads/{upload_id}", response_model=schemas.VideoUploadStatus)
def get_video_upload_status(upload_id: str) -> dict:
    status = store.get_upload_status(upload_id)
    if status is None:
        raise HTTPException(status_code=404, detail="Upload status not found")
    return status


@app.post("/api/videos/uploads/{upload_id}/cancel", response_model=schemas.VideoUploadStatus)
def cancel_video_upload(upload_id: str) -> dict:
    status = store.get_upload_status(upload_id)
    if status is None:
        raise HTTPException(status_code=404, detail="Upload status not found")

    if status["state"] in {"complete", "error", "cancelled"}:
        return status

    store.request_upload_cancel(upload_id)
    return store.set_upload_status(
        upload_id,
        state=status["state"],
        progress_percent=status.get("progressPercent"),
        message="Cancellation requested. Stopping upload...",
        phase=status.get("phase"),
        video_id=status.get("videoId"),
        error=status.get("error"),
    )


@app.post("/api/videos", response_model=schemas.VideoRecord, status_code=201)
async def upload_video(
    file: UploadFile = File(...),
    locationId: str = Form(...),
    date: str = Form(...),
    startTime: str = Form(...),
    endTime: str = Form(...),
    fastMode: bool = Form(False),
    uploadId: Optional[str] = Form(None),
    countingConfig: Optional[str] = Form(None),
    roadLengthM: Optional[float] = Form(None),
    laneCount: Optional[int] = Form(None),
) -> dict:
    target_location = next(
        (loc for loc in store.list_locations() if loc["id"] == locationId),
        None,
    )
    target_domain = (target_location.get("domain") if target_location else None) or "pedestrian"

    if target_domain == "pedestrian":
        status = inference.ultralytics_status()
        if not status["ready"]:
            if uploadId:
                store.set_upload_status(uploadId, state="error", progress_percent=None, message="Inference engine is not ready.", error="Inference engine is not ready. Upload a valid model before processing videos.")
            raise HTTPException(status_code=503, detail="Inference engine is not ready. Upload a valid model before processing videos.")
    else:
        if model_registry.active_weight_path("vehicle") is None:
            detail = "Vehicle inference model is missing. Visit /models to upload one."
            if uploadId:
                store.set_upload_status(uploadId, state="error", progress_percent=None, message=detail, error=detail)
            raise HTTPException(status_code=503, detail=detail)

    safe_name = safe_filename(file.filename or "video.mp4")
    raw_target = store.RAW_VIDEOS_DIR / f"{uuid4().hex[:8]}-{safe_name}"
    raw_target.write_bytes(await file.read())

    if uploadId:
        store.set_upload_status(
            uploadId,
            state="queued",
            progress_percent=0,
            message="Upload received. Preparing video for processing...",
            phase="queued",
            file_name=safe_name,
            location_id=locationId,
            date=date,
            start_time=startTime,
            end_time=endTime,
            fast_mode=fastMode,
        )

    try:
        video = store.add_video(
            {
                "locationId": locationId,
                "date": date,
                "startTime": startTime,
                "endTime": endTime,
                "rawPath": str(raw_target.relative_to(store.BACKEND_DIR)),
                "roadLengthM": roadLengthM,
                "laneCount": laneCount,
            }
        )
    except ValueError as exc:
        raw_target.unlink(missing_ok=True)
        if uploadId:
            store.set_upload_status(uploadId, state="error", progress_percent=None, message="Video upload could not be prepared.", video_id=None, error=str(exc))
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        def ensure_not_cancelled() -> None:
            if uploadId and store.is_upload_cancel_requested(uploadId):
                raise InterruptedError("Video upload cancelled by user.")

        ensure_not_cancelled()

        if uploadId:
            store.set_upload_status(
                uploadId,
                state="processing",
                progress_percent=0,
                message="Starting detection and tracking...",
                phase="tracking",
                video_id=video["id"],
                location_name=video["location"],
            )

        def handle_processing_progress(payload: dict) -> None:
            ensure_not_cancelled()
            if uploadId:
                store.set_upload_status(
                    uploadId,
                    state="processing",
                    progress_percent=payload.get("progressPercent"),
                    message=str(payload.get("message") or "Processing video..."),
                    phase=str(payload.get("phase") or "tracking"),
                    video_id=video["id"],
                )
            ensure_not_cancelled()

        handle_processing_progress.cancel_check = ensure_not_cancelled  # type: ignore[attr-defined]

        if target_domain == "vehicle":
            result = await run_in_threadpool(
                vehicle_inference.run_vehicle_inference,
                raw_target,
                video,
                counting_config_name=countingConfig,
                show_live_preview=fastMode,  # fastMode doubles as live-preview flag for vehicle uploads
                progress_callback=handle_processing_progress,
            )
            ensure_not_cancelled()
            if uploadId:
                store.set_upload_status(
                    uploadId,
                    state="processing",
                    progress_percent=96,
                    message="Persisting vehicle detections...",
                    phase="finalizing",
                    video_id=video["id"],
                )
            response = store.set_video_inference_result(
                video_id=video["id"],
                pedestrian_count=int(result.get("vehicleCount") or 0),
                processed_path=result.get("processedPath"),
                events=result.get("events", []),
                pedestrian_tracks=[],
            )
        else:
            result = await run_in_threadpool(
                inference.run_video_inference,
                raw_target,
                None,
                video,
                fastMode,
                handle_processing_progress,
            )
            ensure_not_cancelled()
            if uploadId:
                store.set_upload_status(
                    uploadId,
                    state="processing",
                    progress_percent=95,
                    message="Calculating Pedestrian Traffic Severity Index...",
                    phase="ptsi",
                    video_id=video["id"],
                )
            response = store.set_video_inference_result(
                video_id=video["id"],
                pedestrian_count=result.get("pedestrianCount", 0),
                processed_path=result.get("processedPath"),
                events=result.get("events", []),
                pedestrian_tracks=result.get("pedestrianTracks", []),
            )
        if uploadId:
            store.set_upload_status(uploadId, state="complete", progress_percent=100, message="Video upload and processing complete.", video_id=video["id"])
        return response
    except InterruptedError as exc:
        store.remove_video(video["id"])
        store.delete_video_assets(video)
        if uploadId:
            store.set_upload_status(uploadId, state="cancelled", progress_percent=None, message="Video upload cancelled.", video_id=video["id"])
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except RuntimeError as exc:
        store.remove_video(video["id"])
        store.delete_video_assets(video)
        if uploadId:
            store.set_upload_status(uploadId, state="error", progress_percent=None, message="Video processing failed.", video_id=video["id"], error=str(exc))
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        store.remove_video(video["id"])
        store.delete_video_assets(video)
        if uploadId:
            store.set_upload_status(uploadId, state="error", progress_percent=None, message="Video processing failed.", video_id=video["id"], error=str(exc))
        raise HTTPException(status_code=500, detail=f"Video inference failed: {exc}") from exc


@app.delete("/api/videos/{video_id}", status_code=204)
def delete_video(video_id: str) -> Response:
    video = store.get_video(video_id)
    if video is None:
        raise HTTPException(status_code=404, detail="Video not found")

    store.remove_video(video_id)
    store.delete_video_assets(video)
    return Response(status_code=204)


@app.get("/api/events", response_model=list[schemas.EventRecord])
def get_events(videoId: Optional[str] = None) -> list[dict]:
    return store.list_events(videoId)


@app.get("/api/dashboard/summary", response_model=schemas.DashboardSummary)
def get_dashboard_summary(date: Optional[str] = None) -> dict:
    return store.dashboard_summary(date)


@app.get("/api/dashboard/traffic", response_model=schemas.TrafficResponse)
def get_dashboard_traffic(
    date: Optional[str] = None,
    timeRange: str = "whole-day",
    focusTime: Optional[str] = None,
    zoomLevel: int = 0,
) -> dict[str, object]:
    return store.dashboard_traffic(date, timeRange, focusTime, zoomLevel)


@app.get("/api/dashboard/occlusion-trends", response_model=schemas.PTSITrendResponse)
def get_dashboard_occlusion_trends(
    date: Optional[str] = None,
    timeRange: str = "whole-day",
    focusTime: Optional[str] = None,
    zoomLevel: int = 0,
) -> dict[str, object]:
    return store.dashboard_occlusion_trends(date, timeRange, focusTime, zoomLevel)


@app.get("/api/dashboard/occlusion", response_model=schemas.PTSIMapResponse)
def get_dashboard_occlusion(date: Optional[str] = None, timeRange: str = "whole-day") -> dict[str, object]:
    return store.dashboard_occlusion(date, timeRange)


@app.get("/api/dashboard/ai-synthesis", response_model=schemas.AISynthesisResponse)
def get_ai_synthesis(date: str = "2026-03-15", timeRange: str = "whole-day") -> dict:
    return store.ai_synthesis(date, timeRange)


@app.get("/api/dashboard/export")
def export_dashboard_report(date: str = "2026-03-15", timeRange: str = "whole-day") -> FileResponse:
    report_path = store.export_dashboard_report(date, timeRange)
    return FileResponse(report_path, media_type="application/zip", filename=report_path.name)


@app.get("/api/search", response_model=list[schemas.SearchResult])
def search(query: str) -> list[dict]:
    _ensure_search_metadata()
    return store.search_results(query, ai_ranker=gemini.rank_pedestrian_matches, query_parser=gemini.parse_search_query)


@app.get("/api/models/current", response_model=schemas.ModelInfo)
def get_current_model() -> dict:
    return store.get_model_info()


@app.get("/api/inference/status", response_model=schemas.InferenceStatus)
def get_inference_status() -> dict:
    return inference.ultralytics_status()


@app.post("/api/models/upload", response_model=schemas.ModelInfo, status_code=201)
async def upload_model(file: UploadFile = File(...)) -> dict:
    filename = safe_filename(file.filename or "model.pt")
    if not filename.endswith(".pt"):
        raise HTTPException(status_code=400, detail="Only .pt model files are supported")
    target = store.MODELS_DIR / filename
    target.write_bytes(await file.read())
    return store.set_model(filename)


_DOMAIN_UPLOAD_EXTENSIONS = {
    "pedestrian": {".pt"},
    "vehicle": {".pt", ".pth"},
}


def _validate_domain(domain: str) -> None:
    if domain not in model_registry.DOMAINS:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown model domain '{domain}'. Expected one of: {list(model_registry.DOMAINS)}",
        )


@app.get("/api/models", response_model=schemas.ModelRegistryResponse)
def get_model_registry() -> dict[str, Any]:
    return {
        "domains": model_registry.list_domains(),
        "updatedAt": model_registry.get_registry().get("updatedAt"),
    }


@app.get("/api/models/{domain}", response_model=schemas.ModelDomainInfo)
def get_model_domain(domain: str) -> dict[str, Any]:
    _validate_domain(domain)
    return model_registry.get_domain(domain)


@app.post("/api/models/settings", response_model=schemas.ModelDomainInfo)
def update_model_settings(payload: schemas.ModelSettingsUpdate) -> dict[str, Any]:
    _validate_domain(payload.domain)
    try:
        return model_registry.set_active(payload.domain, payload.weightId)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post(
    "/api/models/{domain}/upload",
    response_model=schemas.ModelDomainInfo,
    status_code=201,
)
async def upload_domain_model(
    domain: str,
    file: UploadFile = File(...),
    label: Optional[str] = Form(None),
    setActive: bool = Form(True),
) -> dict[str, Any]:
    _validate_domain(domain)
    filename = safe_filename(file.filename or f"{domain}-model.pt")
    suffix = Path(filename).suffix.lower()
    allowed = _DOMAIN_UPLOAD_EXTENSIONS[domain]
    if suffix not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"Domain '{domain}' accepts only {sorted(allowed)} files",
        )
    domain_dir = store.MODELS_DIR / domain
    domain_dir.mkdir(parents=True, exist_ok=True)
    target = domain_dir / filename
    target.write_bytes(await file.read())
    weight_id = Path(filename).stem
    relative_path = f"{domain}/{filename}"
    try:
        return model_registry.add_weight(
            domain,
            weight_id=weight_id,
            filename=filename,
            relative_path=relative_path,
            label=label,
            set_active_after=bool(setActive),
        )
    except KeyError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.delete("/api/models/{domain}/weights/{weight_id}", response_model=schemas.ModelDomainInfo)
def delete_domain_weight(domain: str, weight_id: str) -> dict[str, Any]:
    _validate_domain(domain)
    try:
        return model_registry.remove_weight(domain, weight_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except PermissionError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


# --- Inference requirement configs (counting / infer YAMLs) ---


@app.get(
    "/api/inference/requirements/counting-configs",
    response_model=schemas.CountingConfigList,
)
def list_counting_configs_endpoint() -> dict[str, Any]:
    options = vehicle_counting.list_counting_configs()
    return {"options": options, "defaultConfig": options[0] if options else None}


@app.get(
    "/api/inference/requirements/infer-configs",
    response_model=schemas.InferConfigList,
)
def list_infer_configs_endpoint() -> dict[str, Any]:
    options = vehicle_counting.list_infer_configs()
    return {"options": options, "defaultConfig": options[0] if options else None}


@app.post(
    "/api/inference/requirements/upload",
    response_model=schemas.InferenceRequirementUploadResult,
    status_code=201,
)
async def upload_inference_requirement(
    file: UploadFile = File(...),
    requirementType: str = Form(...),
) -> dict[str, Any]:
    requirement_type = requirementType.strip().lower()
    if requirement_type not in vehicle_counting.VALID_REQUIREMENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid requirementType. Use one of: {list(vehicle_counting.VALID_REQUIREMENT_TYPES)}",
        )
    filename = safe_filename(file.filename or "upload.bin")
    suffix = Path(filename).suffix.lower()
    expected = {
        "infer-config": {".yml", ".yaml"},
        "annotations": {".json"},
        "counting-config": {".json"},
    }
    if suffix not in expected[requirement_type]:
        raise HTTPException(
            status_code=400,
            detail=f"{requirement_type} must be one of: {sorted(expected[requirement_type])}",
        )
    target_dir = vehicle_counting.requirement_target_dir(requirement_type)
    target_path = target_dir / filename
    target_path.write_bytes(await file.read())
    relative = target_path.relative_to(store.BACKEND_DIR.parent)
    return {
        "requirementType": requirement_type,
        "filename": filename,
        "savedPath": str(relative),
        "message": f"Uploaded {filename} to {target_dir.relative_to(store.BACKEND_DIR.parent)}",
    }


# --- Vehicle endpoints ---


@app.get("/api/vehicle/gates", response_model=list[schemas.VehicleGate])
def list_vehicle_gates() -> list[dict[str, Any]]:
    return vehicle_gates.list_gates()


@app.get("/api/vehicle/dashboard/summary", response_model=schemas.VehicleSummary)
def get_vehicle_summary(date: Optional[str] = None) -> dict[str, Any]:
    return vehicle_store.vehicle_summary(date)


@app.get("/api/vehicle/dashboard/los", response_model=list[schemas.VehicleGateLOS])
def get_vehicle_los(date: Optional[str] = None) -> list[dict[str, Any]]:
    events = vehicle_store.list_vehicle_events(date)
    return vehicle_store.per_gate_los(events)


@app.get("/api/vehicle/dashboard/class-breakdown", response_model=list[schemas.VehicleClassBreakdown])
def get_vehicle_class_breakdown(date: Optional[str] = None) -> list[dict[str, Any]]:
    events = vehicle_store.list_vehicle_events(date)
    return vehicle_store.class_breakdown(events)


@app.get("/api/vehicle/dashboard/traffic", response_model=schemas.VehicleTrafficResponse)
def get_vehicle_traffic(
    date: Optional[str] = None,
    timeRange: str = "whole-day",
    startTime: Optional[str] = None,
    bucketMinutes: int = 60,
) -> dict[str, Any]:
    series = vehicle_store.vehicle_traffic_series(date, timeRange, startTime, bucketMinutes)
    return {"timeRange": timeRange, "bucketMinutes": bucketMinutes, "series": series}


@app.get("/api/vehicle/dashboard/analytics")
def get_vehicle_analytics(
    date: Optional[str] = None,
    timeRange: str = "whole-day",
    startTime: Optional[str] = None,
    gateId: Optional[str] = None,
    bucketMinutes: int = 60,
) -> dict[str, Any]:
    """Per-gate time-bucketed vehicle count series for the analytics charts."""
    series = vehicle_store.vehicle_analytics_series(
        date=date,
        time_range=timeRange,
        start_time=startTime,
        gate_id=gateId,
        bucket_minutes=bucketMinutes,
    )
    return {"timeRange": timeRange, "bucketMinutes": bucketMinutes, "series": series}


@app.get("/api/vehicle/dashboard/los")
def get_vehicle_los_series(
    date: Optional[str] = None,
    timeRange: str = "whole-day",
    startTime: Optional[str] = None,
    gateId: Optional[str] = None,
    bucketMinutes: int = 60,
) -> dict[str, Any]:
    """Per-gate time-bucketed LOS series for the LOS chart."""
    series = vehicle_store.vehicle_los_series(
        date=date,
        time_range=timeRange,
        start_time=startTime,
        gate_id=gateId,
        bucket_minutes=bucketMinutes,
    )
    return {"timeRange": timeRange, "bucketMinutes": bucketMinutes, "series": series}


@app.get("/api/vehicle/events", response_model=list[schemas.EventRecord])
def get_vehicle_events(date: Optional[str] = None, gateId: Optional[str] = None) -> list[dict[str, Any]]:
    return vehicle_store.list_vehicle_events(date, gateId)


# --- Pedestrian aliases (mirror of /api/dashboard/* under symmetric namespace) ---


@app.get("/api/pedestrian/dashboard/summary", response_model=schemas.DashboardSummary)
def get_pedestrian_summary(date: Optional[str] = None) -> dict[str, Any]:
    return get_dashboard_summary(date)


@app.get("/api/pedestrian/dashboard/traffic", response_model=schemas.TrafficResponse)
def get_pedestrian_traffic(
    date: Optional[str] = None,
    timeRange: str = "12h",
    focusTime: Optional[str] = None,
    startTime: Optional[str] = None,
    zoomLevel: int = 0,
    locationId: Optional[str] = None,
) -> dict[str, Any]:
    return get_dashboard_traffic(date, timeRange, focusTime, startTime, zoomLevel, locationId)


@app.get("/api/pedestrian/dashboard/occlusion", response_model=schemas.PTSIMapResponse)
def get_pedestrian_occlusion(
    date: Optional[str] = None,
    timeRange: str = "12h",
    startTime: Optional[str] = None,
) -> dict[str, Any]:
    return get_dashboard_occlusion(date, timeRange, startTime)


@app.get("/api/pedestrian/dashboard/occlusion-trends", response_model=schemas.PTSITrendResponse)
def get_pedestrian_occlusion_trends(
    date: Optional[str] = None,
    timeRange: str = "12h",
    focusTime: Optional[str] = None,
    startTime: Optional[str] = None,
    zoomLevel: int = 0,
) -> dict[str, Any]:
    return get_dashboard_occlusion_trends(date, timeRange, focusTime, startTime, zoomLevel)


@app.get("/api/pedestrian/events", response_model=list[schemas.EventRecord])
def get_pedestrian_events(videoId: Optional[str] = None) -> list[dict[str, Any]]:
    return get_events(videoId)
