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

from . import gemini, inference, schemas, store

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
def get_locations(date: Optional[str] = None) -> list[dict]:
    return store.list_locations(date)


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
) -> dict:
    status = inference.ultralytics_status()
    if not status["ready"]:
        if uploadId:
            store.set_upload_status(uploadId, state="error", progress_percent=None, message="Inference engine is not ready.", error="Inference engine is not ready. Upload a valid model before processing videos.")
        raise HTTPException(status_code=503, detail="Inference engine is not ready. Upload a valid model before processing videos.")

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
