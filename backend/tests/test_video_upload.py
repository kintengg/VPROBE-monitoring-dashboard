from __future__ import annotations

import importlib
import io
import json
import logging
import os
import sys
import threading
import zipfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.app import inference, main, semantic_search, store


def configure_temp_storage(monkeypatch, tmp_path: Path) -> None:
    backend_dir = tmp_path / "backend"
    storage_dir = backend_dir / "storage"

    monkeypatch.setattr(store, "BACKEND_DIR", backend_dir)
    monkeypatch.setattr(store, "STORAGE_DIR", storage_dir)
    monkeypatch.setattr(store, "MODELS_DIR", storage_dir / "models")
    monkeypatch.setattr(store, "RAW_VIDEOS_DIR", storage_dir / "videos" / "raw")
    monkeypatch.setattr(store, "PROCESSED_VIDEOS_DIR", storage_dir / "videos" / "processed")
    monkeypatch.setattr(store, "EXPORTS_DIR", storage_dir / "exports")
    monkeypatch.setattr(store, "PORTABLE_DIR", storage_dir / "portable")
    monkeypatch.setattr(store, "PORTABLE_VIDEOS_DIR", storage_dir / "portable" / "videos")
    monkeypatch.setattr(store, "QUEUE_HISTORY_JSON_FILE", storage_dir / "portable" / "queue_history.json")
    monkeypatch.setattr(store, "QUEUE_HISTORY_CSV_FILE", storage_dir / "portable" / "queue_history.csv")
    monkeypatch.setattr(store, "PORTABLE_MANIFEST_FILE", storage_dir / "portable" / "manifest.json")
    monkeypatch.setattr(store, "DATA_FILE", storage_dir / "dev_data.json")

    store.ensure_storage_layout()
    with store.UPLOAD_STATUS_LOCK:
        store.UPLOAD_STATUSES.clear()
        store.UPLOAD_CANCEL_REQUESTS.clear()


def test_semantic_search_configures_openmp_compatibility_env(monkeypatch) -> None:
    monkeypatch.delenv("KMP_DUPLICATE_LIB_OK", raising=False)

    importlib.reload(semantic_search)

    assert os.environ["KMP_DUPLICATE_LIB_OK"] == "TRUE"


def test_semantic_confidence_scales_with_match_strength() -> None:
    assert store._semantic_confidence(0.18) == 35
    assert store._semantic_confidence(0.24) == 49
    assert store._semantic_confidence(0.28) == 60
    assert store._semantic_confidence(0.41) == 88
    assert store._semantic_confidence(0.52) == 96


def test_ultralytics_status_prefers_vendored_package(monkeypatch, tmp_path: Path) -> None:
    configure_temp_storage(monkeypatch, tmp_path)

    global_root = tmp_path / "global-site"
    global_pkg = global_root / "ultralytics"
    global_pkg.mkdir(parents=True, exist_ok=True)
    (global_pkg / "__init__.py").write_text('__version__ = "0.0.global"\n', encoding="utf-8")
    monkeypatch.syspath_prepend(str(global_root))

    vendored_pkg = store.BACKEND_DIR / "vendor" / "ultralytics" / "ultralytics"
    vendored_pkg.mkdir(parents=True, exist_ok=True)
    (vendored_pkg / "__init__.py").write_text('__version__ = "9.9.vendored"\n', encoding="utf-8")

    model_path = store.MODELS_DIR / "best.pt"
    model_path.write_bytes(b"fake-model")
    store.set_model(model_path.name)

    original_sys_path = sys.path.copy()
    for module_name in [name for name in list(sys.modules) if name == "ultralytics" or name.startswith("ultralytics.")]:
        monkeypatch.delitem(sys.modules, module_name, raising=False)

    try:
        status = inference.ultralytics_status()
    finally:
        sys.path[:] = original_sys_path
        for module_name in [name for name in list(sys.modules) if name == "ultralytics" or name.startswith("ultralytics.")]:
            sys.modules.pop(module_name, None)

    assert status["installed"] is True
    assert status["version"] == "9.9.vendored"
    assert status["vendoredPath"] == "vendor/ultralytics"
    assert status["usingVendoredCopy"] is True
    assert status["packagePath"].endswith("vendor/ultralytics/ultralytics/__init__.py")
    assert status["modelExists"] is True
    assert status["ready"] is True


def test_box_xyxy_accepts_multi_value_tensor_like_objects() -> None:
    class FakeTensor:
        def item(self):
            raise RuntimeError("a Tensor with 4 elements cannot be converted to Scalar")

        def tolist(self):
            return [[10.2, 20.4, 30.6, 40.8]]

    class FakeBox:
        xyxy = FakeTensor()

    assert inference._box_xyxy(FakeBox()) == (10, 20, 31, 41)


def test_video_detail_includes_compact_timeline_severity_summary(monkeypatch, tmp_path: Path) -> None:
    configure_temp_storage(monkeypatch, tmp_path)

    state = store.seed_state()
    state["locations"] = [
        {
            "id": "timeline-walk",
            "name": "Timeline Walk",
            "latitude": 14.6397,
            "longitude": 121.0775,
            "description": "",
            "address": "",
            "roiCoordinates": None,
            "walkableAreaM2": 4.0,
            "videos": [],
        }
    ]
    state["videos"] = [
        {
            "id": "video-timeline",
            "locationId": "timeline-walk",
            "location": "Timeline Walk",
            "timestamp": "10:00:00",
            "date": "2026-03-17",
            "startTime": "10:00:00",
            "endTime": "10:00:12",
            "gpsLat": 14.6397,
            "gpsLng": 121.0775,
            "pedestrianCount": 3,
            "rawPath": None,
            "processedPath": None,
        }
    ]
    state["pedestrianTracks"] = [
        {
            "id": "track-1",
            "videoId": "video-timeline",
            "pedestrianId": 1,
            "trajectorySamples": [[0, 0.2, 0.2, 0], [1, 0.2, 0.2, 0], [2, 0.2, 0.2, 0], [3, 0.2, 0.2, 0], [4, 0.2, 0.2, 1], [5, 0.2, 0.2, 1], [6, 0.2, 0.2, 1], [7, 0.2, 0.2, 1], [8, 0.2, 0.2, 2], [9, 0.2, 0.2, 2], [10, 0.2, 0.2, 2], [11, 0.2, 0.2, 2]],
        },
        {
            "id": "track-2",
            "videoId": "video-timeline",
            "pedestrianId": 2,
            "trajectorySamples": [[4, 0.4, 0.4, 1], [5, 0.4, 0.4, 1], [6, 0.4, 0.4, 1], [7, 0.4, 0.4, 1], [8, 0.4, 0.4, 2], [9, 0.4, 0.4, 2], [10, 0.4, 0.4, 2], [11, 0.4, 0.4, 2]],
        },
        {
            "id": "track-3",
            "videoId": "video-timeline",
            "pedestrianId": 3,
            "trajectorySamples": [[8, 0.6, 0.6, 2], [9, 0.6, 0.6, 2], [10, 0.6, 0.6, 2], [11, 0.6, 0.6, 2]],
        },
    ]
    store.save_state(state)

    with TestClient(main.app) as client:
        response = client.get("/api/videos/video-timeline")

    assert response.status_code == 200
    body = response.json()
    assert body["severitySummary"]["bucketCount"] == 12
    assert body["severitySummary"]["sampledSeconds"] == 12
    buckets = body["severitySummary"]["buckets"]
    assert [(bucket["startOffsetSeconds"], bucket["endOffsetSeconds"], bucket["severity"]) for bucket in buckets] == [
        (0.0, 4.0, "light"),
        (4.0, 8.0, "moderate"),
        (8.0, 12.0, "heavy"),
    ]
    assert [bucket["score"] for bucket in buckets] == pytest.approx([22.0, 61.0, 83.0], abs=0.01)
    assert body["pedestrianTracks"] == [
        {"id": "track-1", "pedestrianId": 1, "firstOffsetSeconds": 0.0, "lastOffsetSeconds": 11.0},
        {"id": "track-2", "pedestrianId": 2, "firstOffsetSeconds": 4.0, "lastOffsetSeconds": 11.0},
        {"id": "track-3", "pedestrianId": 3, "firstOffsetSeconds": 8.0, "lastOffsetSeconds": 11.0},
    ]


def test_enrich_track_summaries_with_vision_appends_visual_metadata(monkeypatch, tmp_path: Path) -> None:
    configure_temp_storage(monkeypatch, tmp_path)

    thumbnail = store.PROCESSED_VIDEOS_DIR / "video-1" / "tracks" / "track-7.jpg"
    thumbnail.parent.mkdir(parents=True, exist_ok=True)
    thumbnail.write_bytes(b"fake-jpeg-bytes")

    track = {
        "id": "track-7",
        "thumbnailPath": str(thumbnail.relative_to(store.BACKEND_DIR)),
        "appearanceHints": ["upper clothing appears red"],
        "appearanceSummary": "Representative crop suggests upper clothing appears red.",
        "occlusionClass": None,
        "bestArea": 3600.0,
        "bestOffsetSeconds": 2.0,
    }
    observed: dict[str, object] = {}
    progress_updates: list[dict[str, object]] = []

    monkeypatch.setattr(inference.vision, "track_enrichment_enabled", lambda: True)
    monkeypatch.setattr(inference.vision, "track_enrichment_limit", lambda: 10)

    def fake_enrich_track_thumbnail(path: Path):
        observed["thumbnail_path"] = path
        return {
            "labels": ["dress", "backpack"],
            "objects": ["bag"],
            "logos": ["nike"],
            "text": ["ATENEO"],
            "summary": "Cloud Vision labels: dress, backpack. Detected objects: bag. Detected logos: nike. Visible text: ATENEO.",
        }

    monkeypatch.setattr(inference.vision, "enrich_track_thumbnail", fake_enrich_track_thumbnail)

    enriched_count = inference._enrich_track_summaries_with_vision([track], progress_updates.append)

    assert enriched_count == 1
    assert observed["thumbnail_path"] == thumbnail
    assert track["visualLabels"] == ["dress", "backpack"]
    assert track["visualObjects"] == ["bag"]
    assert track["visualLogos"] == ["nike"]
    assert track["visualText"] == ["ATENEO"]
    assert "upper clothing appears red" in track["appearanceSummary"]
    assert "Cloud Vision labels: dress, backpack." in track["appearanceSummary"]
    assert progress_updates[0]["phase"] == "vision"
    assert progress_updates[0]["message"] == "Preparing Cloud Vision enrichment for 1 track thumbnails..."
    assert progress_updates[-1]["message"] == "Cloud Vision analyzing track thumbnails (1/1)..."


def test_upload_video_runs_inference_and_persists_results(monkeypatch, tmp_path: Path) -> None:
    configure_temp_storage(monkeypatch, tmp_path)

    monkeypatch.setattr(
        inference,
        "ultralytics_status",
        lambda: {"installed": True, "modelExists": True, "ready": True, "currentModel": "best.pt"},
    )

    upload_id = "upload-portable-artifacts"

    def fake_run_video_inference(video_path: Path, model_name=None, video_record=None, fast_mode: bool = False, progress_callback=None):
        processed_dir = store.PROCESSED_VIDEOS_DIR / video_record["id"]
        processed_dir.mkdir(parents=True, exist_ok=True)
        processed_file = processed_dir / f"{video_path.stem}.mp4"
        processed_file.write_bytes(b"processed-video")
        if progress_callback is not None:
            progress_callback({"progressPercent": 60, "message": "Running detection and tracking..."})
        return {
            "pedestrianCount": 2,
            "processedPath": str(processed_file.relative_to(store.BACKEND_DIR)),
            "events": [
                {
                    "id": "evt-1",
                    "type": "detection",
                    "location": video_record["location"],
                    "timestamp": "10:00:00 AM",
                    "description": "Pedestrian ID #7 detected at frame 1",
                    "videoId": video_record["id"],
                    "pedestrianId": 7,
                    "frame": 1,
                    "offsetSeconds": 0.0,
                }
            ],
            "pedestrianTracks": [
                {
                    "id": "trk-1",
                    "videoId": video_record["id"],
                    "pedestrianId": 7,
                    "location": video_record["location"],
                    "firstTimestamp": "10:00:00 AM",
                    "lastTimestamp": "10:00:05 AM",
                    "bestTimestamp": "10:00:02 AM",
                    "firstFrame": 1,
                    "lastFrame": 8,
                    "bestFrame": 3,
                    "firstOffsetSeconds": 0.0,
                    "lastOffsetSeconds": 5.0,
                    "bestOffsetSeconds": 2.0,
                    "thumbnailPath": str((processed_dir / "tracks" / "track-7.jpg").relative_to(store.BACKEND_DIR)),
                    "appearanceHints": ["head region appears blue", "lower clothing appears blue"],
                    "appearanceSummary": "Representative crop suggests head region appears blue, lower clothing appears blue.",
                    "occlusionClass": None,
                    "bestArea": 2400.0,
                    "trajectorySamples": [[0, 0.2, 0.2, 0], [1, 0.2, 0.2, 1], [2, 0.2, 0.2, 2]],
                }
            ],
        }

    monkeypatch.setattr(inference, "run_video_inference", fake_run_video_inference)

    with TestClient(main.app) as client:
        response = client.post(
            "/api/videos",
            data={
                "locationId": "edsa-sec-walk",
                "date": "2026-03-17",
                "startTime": "10:00",
                "endTime": "10:01",
                "uploadId": upload_id,
            },
            files={"file": ("clip.mp4", b"fake-video-bytes", "video/mp4")},
        )

    assert response.status_code == 201
    body = response.json()
    assert body["pedestrianCount"] == 2
    assert body["processedPath"].startswith("storage/videos/processed/")
    assert body["rawPath"].startswith("storage/videos/raw/")
    assert (store.BACKEND_DIR / body["rawPath"]).exists()
    assert (store.BACKEND_DIR / body["processedPath"]).exists()

    state = store.load_state()
    saved_video = next(video for video in state["videos"] if video["id"] == body["id"])
    saved_events = [event for event in state["events"] if event.get("videoId") == body["id"]]
    saved_tracks = [track for track in state["pedestrianTracks"] if track.get("videoId") == body["id"]]

    assert saved_video["processedPath"] == body["processedPath"]
    assert saved_video["pedestrianCount"] == 2
    assert len(saved_events) == 1
    assert saved_events[0]["pedestrianId"] == 7
    assert saved_events[0]["frame"] == 1
    assert saved_events[0]["offsetSeconds"] == 0.0
    assert len(saved_tracks) == 1
    assert saved_tracks[0]["pedestrianId"] == 7
    assert saved_tracks[0]["thumbnailPath"].endswith("track-7.jpg")

    portable_dir = store.PORTABLE_VIDEOS_DIR / body["id"]
    assert (portable_dir / "timeline.csv").exists()
    assert (portable_dir / "tracks.csv").exists()
    assert (portable_dir / "events.csv").exists()
    assert (portable_dir / "whole_footage_log.csv").exists()

    manifest = json.loads(store.PORTABLE_MANIFEST_FILE.read_text(encoding="utf-8"))
    manifest_entry = next(item for item in manifest["videos"] if item["videoId"] == body["id"])
    assert manifest_entry["artifacts"]["timelineCsv"].endswith(f"{body['id']}/timeline.csv")
    assert manifest_entry["artifacts"]["wholeFootageLogCsv"].endswith(f"{body['id']}/whole_footage_log.csv")

    timeline_text = (portable_dir / "timeline.csv").read_text(encoding="utf-8")
    assert "cumulativeUniquePedestrians" in timeline_text
    assert "totalPedestriansSoFar" in timeline_text

    whole_footage_text = (portable_dir / "whole_footage_log.csv").read_text(encoding="utf-8")
    assert "videoTime" in whole_footage_text
    assert "trackId" in whole_footage_text
    assert "cumulativeUniquePedestrians" in whole_footage_text

    queue_history = json.loads(store.QUEUE_HISTORY_JSON_FILE.read_text(encoding="utf-8"))
    history_item = next(item for item in queue_history if item["uploadId"] == upload_id)
    assert history_item["state"] == "complete"
    assert history_item["locationName"] == "EDSA Sec Walk"

    with TestClient(main.app) as client:
        history_response = client.get("/api/videos/uploads/history")

    assert history_response.status_code == 200
    assert any(item["uploadId"] == upload_id and item["state"] == "complete" for item in history_response.json())


def test_dashboard_export_returns_zip_bundle_with_portable_artifacts(monkeypatch, tmp_path: Path) -> None:
    configure_temp_storage(monkeypatch, tmp_path)

    video = store.add_video(
        {
            "locationId": "edsa-sec-walk",
            "date": "2026-03-17",
            "startTime": "10:00:00",
            "endTime": "10:00:05",
            "rawPath": None,
        }
    )
    store.set_video_inference_result(
        video_id=video["id"],
        pedestrian_count=1,
        processed_path=None,
        events=[
            {
                "id": "evt-export-1",
                "type": "detection",
                "location": video["location"],
                "timestamp": "10:00:00 AM",
                "description": "Pedestrian detected.",
                "videoId": video["id"],
                "pedestrianId": 1,
                "offsetSeconds": 0.0,
                "occlusionClass": 1,
            }
        ],
        pedestrian_tracks=[
            {
                "id": "trk-export-1",
                "videoId": video["id"],
                "pedestrianId": 1,
                "location": video["location"],
                "firstOffsetSeconds": 0.0,
                "lastOffsetSeconds": 2.0,
                "trajectorySamples": [[0, 0.3, 0.3, 0], [1, 0.3, 0.3, 1], [2, 0.3, 0.3, 2]],
            }
        ],
    )
    store.set_upload_status(
        "upload-export-bundle",
        state="complete",
        progress_percent=100,
        message="Video upload and processing complete.",
        video_id=video["id"],
        file_name="clip.mp4",
        location_id=video["locationId"],
        location_name=video["location"],
        date=video["date"],
        start_time=video["startTime"],
        end_time=video["endTime"],
        fast_mode=False,
    )

    store._remove_portable_video_artifacts(video["id"])
    assert not (store.PORTABLE_VIDEOS_DIR / video["id"] / "whole_footage_log.csv").exists()

    with TestClient(main.app) as client:
        response = client.get("/api/dashboard/export", params={"date": "2026-03-17", "timeRange": "whole-day"})

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/zip")
    assert "2026-03-17-whole-day-dashboard-report" in response.headers.get("content-disposition", "")

    archive = zipfile.ZipFile(io.BytesIO(response.content))
    archive_names = set(archive.namelist())

    assert any(name.endswith(".md") for name in archive_names)
    assert "dashboard/summary.csv" in archive_names
    assert "dashboard/unique_pedestrians.csv" in archive_names
    assert "dashboard/video_totals.csv" in archive_names
    assert "dashboard/traffic.json" in archive_names
    assert "portable/manifest.json" in archive_names
    assert "portable/queue_history.json" in archive_names
    assert f"storage/portable/videos/{video['id']}/timeline.csv" in archive_names
    assert f"storage/portable/videos/{video['id']}/whole_footage_log.csv" in archive_names
    assert f"storage/portable/videos/{video['id']}/tracks.json" in archive_names

    unique_csv_text = archive.read("dashboard/unique_pedestrians.csv").decode("utf-8")
    assert "countedInDashboardTotal" in unique_csv_text
    assert video["id"] in unique_csv_text

    whole_footage_csv_text = archive.read(f"storage/portable/videos/{video['id']}/whole_footage_log.csv").decode("utf-8")
    assert "00:00:00" in whole_footage_csv_text
    assert "trk-export-1" in whole_footage_csv_text
    assert (store.PORTABLE_VIDEOS_DIR / video["id"] / "whole_footage_log.csv").exists()


def test_upload_video_forwards_fast_mode(monkeypatch, tmp_path: Path) -> None:
    configure_temp_storage(monkeypatch, tmp_path)

    monkeypatch.setattr(
        inference,
        "ultralytics_status",
        lambda: {"installed": True, "modelExists": True, "ready": True, "currentModel": "best.pt"},
    )

    observed: dict[str, bool] = {}

    def fake_run_video_inference(video_path: Path, model_name=None, video_record=None, fast_mode: bool = False, progress_callback=None):
        observed["fast_mode"] = fast_mode
        if progress_callback is not None:
            progress_callback({"progressPercent": 25, "message": "Running detection and tracking..."})
        return {"pedestrianCount": 0, "processedPath": None, "events": [], "pedestrianTracks": []}

    monkeypatch.setattr(inference, "run_video_inference", fake_run_video_inference)

    with TestClient(main.app) as client:
        response = client.post(
            "/api/videos",
            data={
                "locationId": "edsa-sec-walk",
                "date": "2026-03-17",
                "startTime": "10:00",
                "endTime": "10:01",
                "fastMode": "true",
            },
            files={"file": ("clip.mp4", b"fake-video-bytes", "video/mp4")},
        )

    assert response.status_code == 201
    assert observed["fast_mode"] is True


def test_upload_video_status_endpoint_reports_processing_progress(monkeypatch, tmp_path: Path) -> None:
    configure_temp_storage(monkeypatch, tmp_path)

    monkeypatch.setattr(
        inference,
        "ultralytics_status",
        lambda: {"installed": True, "modelExists": True, "ready": True, "currentModel": "best.pt"},
    )

    upload_id = "upload-progress-check"
    progress_reported = threading.Event()
    allow_finish = threading.Event()
    upload_response: dict[str, object] = {}

    def fake_run_video_inference(video_path: Path, model_name=None, video_record=None, fast_mode: bool = False, progress_callback=None):
        if progress_callback is not None:
            progress_callback({"progressPercent": 35, "message": "Running detection and tracking..."})
        progress_reported.set()
        assert allow_finish.wait(timeout=3)
        if progress_callback is not None:
            progress_callback({"progressPercent": 90, "message": "Finalizing processed video..."})
        return {"pedestrianCount": 1, "processedPath": None, "events": [], "pedestrianTracks": []}

    monkeypatch.setattr(inference, "run_video_inference", fake_run_video_inference)

    def perform_upload() -> None:
        with TestClient(main.app) as client:
            upload_response["response"] = client.post(
                "/api/videos",
                data={
                    "locationId": "edsa-sec-walk",
                    "date": "2026-03-17",
                    "startTime": "10:00",
                    "endTime": "10:00:40",
                    "uploadId": upload_id,
                },
                files={"file": ("clip.mp4", b"fake-video-bytes", "video/mp4")},
            )

    upload_thread = threading.Thread(target=perform_upload)
    upload_thread.start()

    assert progress_reported.wait(timeout=3)

    with TestClient(main.app) as client:
        status_response = client.get(f"/api/videos/uploads/{upload_id}")

    assert status_response.status_code == 200
    status_body = status_response.json()
    assert status_body["state"] == "processing"
    assert status_body["progressPercent"] == 35
    assert status_body["phase"] == "tracking"
    assert status_body["message"] == "Running detection and tracking..."
    assert status_body["videoId"] is not None

    allow_finish.set()
    upload_thread.join(timeout=3)
    assert not upload_thread.is_alive()

    final_upload_response = upload_response["response"]
    assert final_upload_response.status_code == 201

    with TestClient(main.app) as client:
        final_status_response = client.get(f"/api/videos/uploads/{upload_id}")

    assert final_status_response.status_code == 200
    final_status = final_status_response.json()
    assert final_status["state"] == "complete"
    assert final_status["progressPercent"] == 100
    assert final_status["videoId"] == final_upload_response.json()["id"]


def test_upload_video_status_endpoint_reports_ptsi_phase_before_completion(monkeypatch, tmp_path: Path) -> None:
    configure_temp_storage(monkeypatch, tmp_path)

    monkeypatch.setattr(
        inference,
        "ultralytics_status",
        lambda: {"installed": True, "modelExists": True, "ready": True, "currentModel": "best.pt"},
    )

    upload_id = "upload-ptsi-phase-check"
    ptsi_phase_started = threading.Event()
    allow_finish = threading.Event()
    upload_response: dict[str, object] = {}
    original_set_video_inference_result = store.set_video_inference_result

    def fake_run_video_inference(video_path: Path, model_name=None, video_record=None, fast_mode: bool = False, progress_callback=None):
        if progress_callback is not None:
            progress_callback({"progressPercent": 60, "message": "Running detection and tracking..."})
        return {"pedestrianCount": 1, "processedPath": None, "events": [], "pedestrianTracks": []}

    def blocking_set_video_inference_result(*args, **kwargs):
        ptsi_phase_started.set()
        assert allow_finish.wait(timeout=3)
        return original_set_video_inference_result(*args, **kwargs)

    monkeypatch.setattr(inference, "run_video_inference", fake_run_video_inference)
    monkeypatch.setattr(store, "set_video_inference_result", blocking_set_video_inference_result)

    def perform_upload() -> None:
        with TestClient(main.app) as client:
            upload_response["response"] = client.post(
                "/api/videos",
                data={
                    "locationId": "edsa-sec-walk",
                    "date": "2026-03-17",
                    "startTime": "10:00",
                    "endTime": "10:01",
                    "uploadId": upload_id,
                },
                files={"file": ("clip.mp4", b"fake-video-bytes", "video/mp4")},
            )

    upload_thread = threading.Thread(target=perform_upload)
    upload_thread.start()

    assert ptsi_phase_started.wait(timeout=3)

    with TestClient(main.app) as client:
        status_response = client.get(f"/api/videos/uploads/{upload_id}")

    assert status_response.status_code == 200
    status_body = status_response.json()
    assert status_body["state"] == "processing"
    assert status_body["progressPercent"] == 95
    assert status_body["phase"] == "ptsi"
    assert status_body["message"] == "Calculating Pedestrian Traffic Severity Index..."
    assert status_body["videoId"] is not None

    allow_finish.set()
    upload_thread.join(timeout=3)
    assert not upload_thread.is_alive()

    final_upload_response = upload_response["response"]
    assert final_upload_response.status_code == 201

    with TestClient(main.app) as client:
        final_status_response = client.get(f"/api/videos/uploads/{upload_id}")

    assert final_status_response.status_code == 200
    final_status = final_status_response.json()
    assert final_status["state"] == "complete"
    assert final_status["progressPercent"] == 100
    assert final_status["videoId"] == final_upload_response.json()["id"]


def test_cancel_upload_endpoint_marks_upload_for_cancellation(monkeypatch, tmp_path: Path) -> None:
    configure_temp_storage(monkeypatch, tmp_path)

    monkeypatch.setattr(
        inference,
        "ultralytics_status",
        lambda: {"installed": True, "modelExists": True, "ready": True, "currentModel": "best.pt"},
    )

    upload_id = "upload-cancel-check"
    progress_reported = threading.Event()
    upload_response: dict[str, object] = {}

    def fake_run_video_inference(video_path: Path, model_name=None, video_record=None, fast_mode: bool = False, progress_callback=None):
        if progress_callback is not None:
            progress_callback({"progressPercent": 35, "phase": "tracking", "message": "Running detection and tracking..."})
        progress_reported.set()
        while True:
            if progress_callback is not None:
                progress_callback({"progressPercent": 40, "phase": "tracking", "message": "Running detection and tracking..."})

    monkeypatch.setattr(inference, "run_video_inference", fake_run_video_inference)

    def perform_upload() -> None:
        with TestClient(main.app) as client:
            upload_response["response"] = client.post(
                "/api/videos",
                data={
                    "locationId": "edsa-sec-walk",
                    "date": "2026-03-17",
                    "startTime": "10:00",
                    "endTime": "10:00:40",
                    "uploadId": upload_id,
                },
                files={"file": ("clip.mp4", b"fake-video-bytes", "video/mp4")},
            )

    upload_thread = threading.Thread(target=perform_upload)
    upload_thread.start()

    assert progress_reported.wait(timeout=3)

    with TestClient(main.app) as client:
        cancel_response = client.post(f"/api/videos/uploads/{upload_id}/cancel")

    assert cancel_response.status_code == 200
    assert cancel_response.json()["message"] == "Cancellation requested. Stopping upload..."

    upload_thread.join(timeout=3)
    assert not upload_thread.is_alive()

    final_upload_response = upload_response["response"]
    assert final_upload_response.status_code == 409

    with TestClient(main.app) as client:
        final_status_response = client.get(f"/api/videos/uploads/{upload_id}")

    assert final_status_response.status_code == 200
    assert final_status_response.json()["state"] == "cancelled"


def test_upload_video_cleans_up_when_inference_fails(monkeypatch, tmp_path: Path) -> None:
    configure_temp_storage(monkeypatch, tmp_path)

    monkeypatch.setattr(
        inference,
        "ultralytics_status",
        lambda: {"installed": True, "modelExists": True, "ready": True, "currentModel": "best.pt"},
    )
    monkeypatch.setattr(
        inference,
        "run_video_inference",
        lambda video_path, model_name=None, video_record=None, fast_mode=False, progress_callback=None: (_ for _ in ()).throw(RuntimeError("tracking failed")),
    )

    before = store.load_state()
    with TestClient(main.app) as client:
        response = client.post(
            "/api/videos",
            data={
                "locationId": "edsa-sec-walk",
                "date": "2026-03-17",
                "startTime": "10:00",
                "endTime": "10:01",
            },
            files={"file": ("clip.mp4", b"fake-video-bytes", "video/mp4")},
        )

    assert response.status_code == 503
    after = store.load_state()
    assert len(after["videos"]) == len(before["videos"])
    assert len(after["events"]) == len(before["events"])
    assert list(store.RAW_VIDEOS_DIR.iterdir()) == []


def test_update_and_delete_location_cascade_to_related_records(monkeypatch, tmp_path: Path) -> None:
    configure_temp_storage(monkeypatch, tmp_path)

    raw_file = store.RAW_VIDEOS_DIR / "clip.mp4"
    raw_file.write_bytes(b"raw-video")
    processed_dir = store.PROCESSED_VIDEOS_DIR / "video-1"
    processed_dir.mkdir(parents=True, exist_ok=True)
    processed_file = processed_dir / "clip.mp4"
    processed_file.write_bytes(b"processed-video")

    state = store.seed_state()
    state["videos"] = [
        {
            "id": "video-1",
            "locationId": "edsa-sec-walk",
            "location": "EDSA Sec Walk",
            "timestamp": "10:00",
            "date": "2026-03-17",
            "startTime": "10:00",
            "endTime": "10:40",
            "gpsLat": 14.6397,
            "gpsLng": 121.0775,
            "pedestrianCount": 3,
            "rawPath": str(raw_file.relative_to(store.BACKEND_DIR)),
            "processedPath": str(processed_file.relative_to(store.BACKEND_DIR)),
        }
    ]
    state["events"] = [
        {
            "id": "event-1",
            "type": "detection",
            "location": "EDSA Sec Walk",
            "timestamp": "10:00:00 AM",
            "description": "Pedestrian ID #3 detected at frame 1",
            "videoId": "video-1",
            "pedestrianId": 3,
            "frame": 1,
            "offsetSeconds": 0.0,
        },
        {
            "id": "event-2",
            "type": "alert",
            "location": "EDSA Sec Walk",
            "timestamp": "10:01:00 AM",
            "description": "Manual note",
            "videoId": None,
        },
    ]
    store.save_state(state)

    with TestClient(main.app) as client:
        update_response = client.put(
            "/api/locations/edsa-sec-walk",
            json={
                "name": "EDSA Sec Walk Updated",
                "latitude": 14.64,
                "longitude": 121.08,
                "description": "Updated camera view",
                "address": "Updated address",
                "roiCoordinates": {
                    "referenceSize": [1920, 1080],
                    "includePolygonsNorm": [
                        [[0.2, 0.2], [0.5, 0.2], [0.5, 0.6], [0.2, 0.6]],
                    ],
                },
                "walkableAreaM2": 54.5,
            },
        )

    assert update_response.status_code == 200
    updated_body = update_response.json()
    assert updated_body["name"] == "EDSA Sec Walk Updated"
    assert updated_body["latitude"] == 14.64
    assert updated_body["walkableAreaM2"] == 54.5
    assert updated_body["roiCoordinates"]["referenceSize"] == [1920, 1080]
    assert updated_body["videos"][0]["id"] == "video-1"

    updated_state = store.load_state()
    updated_location = next(location for location in updated_state["locations"] if location["id"] == "edsa-sec-walk")
    assert updated_state["videos"][0]["location"] == "EDSA Sec Walk Updated"
    assert updated_state["videos"][0]["gpsLat"] == 14.64
    assert updated_state["videos"][0]["gpsLng"] == 121.08
    assert updated_location["walkableAreaM2"] == 54.5
    assert updated_location["roiCoordinates"]["includePolygonsNorm"][0][0] == [0.2, 0.2]
    assert {event["location"] for event in updated_state["events"]} == {"EDSA Sec Walk Updated"}

    with TestClient(main.app) as client:
        delete_response = client.delete("/api/locations/edsa-sec-walk")

    assert delete_response.status_code == 204
    after_delete = store.load_state()
    assert all(location["id"] != "edsa-sec-walk" for location in after_delete["locations"])
    assert after_delete["videos"] == []
    assert after_delete["events"] == []
    assert not raw_file.exists()
    assert not processed_file.exists()
    assert not processed_dir.exists()


def test_location_search_proxies_google_places_results(monkeypatch, tmp_path: Path) -> None:
    configure_temp_storage(monkeypatch, tmp_path)
    monkeypatch.setenv("GOOGLE_MAPS_API_KEY", "test-google-key")

    captured: dict[str, object] = {}

    class FakeGoogleResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self) -> bytes:
            return json.dumps(
                {
                    "places": [
                        {
                            "id": "place-123",
                            "displayName": {"text": "SM North EDSA"},
                            "formattedAddress": "North Avenue corner EDSA, Quezon City, Metro Manila, Philippines",
                            "location": {"latitude": 14.6564, "longitude": 121.0309},
                            "types": ["shopping_mall", "point_of_interest", "establishment"],
                        }
                    ]
                }
            ).encode("utf-8")

    def fake_urlopen(request, timeout=0):
        captured["url"] = request.full_url
        captured["method"] = request.get_method()
        captured["body"] = json.loads(request.data.decode("utf-8"))
        captured["headers"] = {key.lower(): value for key, value in request.header_items()}
        captured["timeout"] = timeout
        return FakeGoogleResponse()

    monkeypatch.setattr(main.urllib_request, "urlopen", fake_urlopen)

    with TestClient(main.app) as client:
        response = client.get("/api/locations/search", params={"query": "SM North Edsa"})

    assert response.status_code == 200
    assert captured["url"] == main.GOOGLE_PLACES_TEXT_SEARCH_URL
    assert captured["method"] == "POST"
    assert captured["timeout"] == 10
    assert captured["headers"]["x-goog-api-key"] == "test-google-key"
    assert "places.displayName" in captured["headers"]["x-goog-fieldmask"]
    assert captured["body"] == {
        "textQuery": "SM North Edsa",
        "pageSize": 5,
        "languageCode": "en",
        "regionCode": "PH",
        "locationBias": main.ATENEO_LOCATION_BIAS,
    }
    assert response.json() == [
        {
            "name": "SM North EDSA",
            "address": "North Avenue corner EDSA, Quezon City, Metro Manila, Philippines",
            "latitude": 14.6564,
            "longitude": 121.0309,
            "placeId": "place-123",
            "types": ["shopping_mall", "point_of_interest", "establishment"],
        }
    ]


def test_location_search_requires_google_maps_api_key(monkeypatch, tmp_path: Path) -> None:
    configure_temp_storage(monkeypatch, tmp_path)
    monkeypatch.delenv("GOOGLE_MAPS_API_KEY", raising=False)

    with TestClient(main.app) as client:
        response = client.get("/api/locations/search", params={"query": "Blue Residences"})

    assert response.status_code == 503
    assert response.json()["detail"] == "Location search is unavailable. Set GOOGLE_MAPS_API_KEY on the backend."


def test_delete_video_removes_state_and_media(monkeypatch, tmp_path: Path) -> None:
    configure_temp_storage(monkeypatch, tmp_path)

    raw_file = store.RAW_VIDEOS_DIR / "clip.mp4"
    raw_file.write_bytes(b"raw-video")
    processed_dir = store.PROCESSED_VIDEOS_DIR / "video-1"
    processed_dir.mkdir(parents=True, exist_ok=True)
    processed_file = processed_dir / "clip.mp4"
    processed_file.write_bytes(b"processed-video")

    state = store.seed_state()
    state["videos"] = [
        {
            "id": "video-1",
            "locationId": "edsa-sec-walk",
            "location": "EDSA Sec Walk",
            "timestamp": "10:00",
            "date": "2026-03-17",
            "startTime": "10:00",
            "endTime": "11:00",
            "gpsLat": 14.6397,
            "gpsLng": 121.0775,
            "pedestrianCount": 3,
            "rawPath": str(raw_file.relative_to(store.BACKEND_DIR)),
            "processedPath": str(processed_file.relative_to(store.BACKEND_DIR)),
        }
    ]
    state["events"] = [
        {
            "id": "event-1",
            "type": "detection",
            "location": "EDSA Sec Walk",
            "timestamp": "10:00:00 AM",
            "description": "Pedestrian ID #3 detected at frame 1",
            "videoId": "video-1",
            "pedestrianId": 3,
            "frame": 1,
            "offsetSeconds": 0.0,
        }
    ]
    store.save_state(state)

    with TestClient(main.app) as client:
        response = client.delete("/api/videos/video-1")

    assert response.status_code == 204
    after = store.load_state()
    assert after["videos"] == []
    assert after["events"] == []
    assert after["pedestrianTracks"] == []
    assert not raw_file.exists()
    assert not processed_file.exists()
    assert not processed_dir.exists()


def test_search_endpoint_returns_ranked_pedestrian_track_matches(monkeypatch, tmp_path: Path) -> None:
    configure_temp_storage(monkeypatch, tmp_path)

    state = store.seed_state()
    state["videos"] = [
        {
            "id": "video-1",
            "locationId": "edsa-sec-walk",
            "location": "EDSA Sec Walk",
            "timestamp": "10:00",
            "date": "2026-03-17",
            "startTime": "10:00",
            "endTime": "10:30",
            "gpsLat": 14.6397,
            "gpsLng": 121.0775,
            "pedestrianCount": 3,
            "rawPath": None,
            "processedPath": None,
        }
    ]
    state["pedestrianTracks"] = [
        {
            "id": "track-blue",
            "videoId": "video-1",
            "pedestrianId": 7,
            "location": "EDSA Sec Walk",
            "firstTimestamp": "10:00:00 AM",
            "lastTimestamp": "10:00:05 AM",
            "bestTimestamp": "10:00:02 AM",
            "firstFrame": 1,
            "lastFrame": 12,
            "bestFrame": 4,
            "firstOffsetSeconds": 0.0,
            "lastOffsetSeconds": 5.0,
            "bestOffsetSeconds": 2.0,
            "thumbnailPath": "storage/videos/processed/video-1/tracks/track-7.jpg",
            "appearanceHints": ["head region appears blue", "lower clothing appears blue"],
            "appearanceSummary": "Representative crop suggests head region appears blue, upper clothing appears white, lower clothing appears blue.",
            "occlusionClass": None,
            "bestArea": 4200.0,
        },
        {
            "id": "track-other",
            "videoId": "video-1",
            "pedestrianId": 9,
            "location": "EDSA Sec Walk",
            "firstTimestamp": "10:04:00 AM",
            "lastTimestamp": "10:04:08 AM",
            "bestTimestamp": "10:04:03 AM",
            "firstFrame": 30,
            "lastFrame": 55,
            "bestFrame": 36,
            "firstOffsetSeconds": 240.0,
            "lastOffsetSeconds": 248.0,
            "bestOffsetSeconds": 243.0,
            "thumbnailPath": "storage/videos/processed/video-1/tracks/track-9.jpg",
            "appearanceHints": ["head region appears black", "lower clothing appears gray"],
            "appearanceSummary": "Representative crop suggests head region appears black, upper clothing appears gray, lower clothing appears gray.",
            "occlusionClass": None,
            "bestArea": 3600.0,
        },
    ]
    store.save_state(state)

    monkeypatch.setattr(
        main.gemini,
        "rank_pedestrian_matches",
        lambda query, candidates: [
            {"id": "track-blue", "confidence": 94, "reason": "Head region and lower clothing are both described as blue."}
        ],
    )

    with TestClient(main.app) as client:
        response = client.get("/api/search", params={"query": "im looking for a pedestrian wearing a blue hat and blue shorts"})

    assert response.status_code == 200
    body = response.json()
    assert len(body) == 1
    assert body[0]["id"] == "track-blue"
    assert body[0]["pedestrianId"] == 7
    assert body[0]["thumbnailPath"].endswith("track-7.jpg")
    assert body[0]["offsetSeconds"] == 2.0
    assert body[0]["previewPath"] is None
    assert body[0]["appearanceSummary"].startswith("Representative crop suggests")
    assert body[0]["matchReason"] == "Head region and lower clothing are both described as blue."


def test_search_endpoint_returns_semantic_track_matches_without_gemini(monkeypatch, tmp_path: Path) -> None:
    configure_temp_storage(monkeypatch, tmp_path)

    state = store.seed_state()
    state["videos"] = [
        {
            "id": "video-1",
            "locationId": "edsa-sec-walk",
            "location": "EDSA Sec Walk",
            "timestamp": "10:00",
            "date": "2026-03-17",
            "startTime": "10:00",
            "endTime": "10:30",
            "gpsLat": 14.6397,
            "gpsLng": 121.0775,
            "pedestrianCount": 1,
            "rawPath": None,
            "processedPath": None,
        }
    ]
    state["pedestrianTracks"] = [
        {
            "id": "track-semantic",
            "videoId": "video-1",
            "pedestrianId": 12,
            "location": "EDSA Sec Walk",
            "firstTimestamp": "10:00:00 AM",
            "lastTimestamp": "10:00:08 AM",
            "bestTimestamp": "10:00:03 AM",
            "firstFrame": 1,
            "lastFrame": 24,
            "bestFrame": 9,
            "firstOffsetSeconds": 0.0,
            "lastOffsetSeconds": 8.0,
            "bestOffsetSeconds": 3.0,
            "thumbnailPath": "storage/videos/processed/video-1/tracks/track-12.jpg",
            "semanticCrops": [
                {
                    "label": "late",
                    "path": "storage/videos/processed/video-1/tracks/track-12-late.jpg",
                    "frame": 20,
                    "timestamp": "10:00:06 AM",
                    "offsetSeconds": 6.0,
                }
            ],
            "appearanceHints": ["upper clothing appears white"],
            "appearanceSummary": "Representative crop suggests upper clothing appears white.",
            "occlusionClass": None,
            "bestArea": 3900.0,
        }
    ]
    store.save_state(state)

    monkeypatch.setattr(
        semantic_search,
        "search_tracks",
        lambda query, *, backend_dir, limit=24: [
            {
                "id": "track-semantic",
                "videoId": "video-1",
                "pedestrianId": 12,
                "location": "EDSA Sec Walk",
                "cropLabel": "late",
                "matchedCropPath": "storage/videos/processed/video-1/tracks/track-12-late.jpg",
                "frame": 20,
                "offsetSeconds": 6.0,
                "timestamp": "10:00:06 AM",
                "semanticScore": 0.41,
            }
        ],
    )

    ranker_called = False

    def fake_ranker(query: str, candidates: list[dict[str, object]]) -> list[dict[str, object]]:
        nonlocal ranker_called
        ranker_called = True
        return []

    monkeypatch.setattr(main.gemini, "rank_pedestrian_matches", fake_ranker)
    monkeypatch.setattr(main.gemini, "parse_search_query", lambda query, locations: (_ for _ in ()).throw(RuntimeError("parser unavailable")))

    with TestClient(main.app) as client:
        response = client.get("/api/search", params={"query": "person with white shirt"})

    assert response.status_code == 200
    body = response.json()
    assert len(body) == 1
    assert body[0]["id"] == "track-semantic"
    assert body[0]["matchStrategy"] == "semantic"
    assert body[0]["possibleMatch"] is False
    assert body[0]["confidence"] == 88
    assert body[0]["semanticScore"] == 0.41
    assert body[0]["thumbnailPath"].endswith("track-12-late.jpg")
    assert body[0]["frame"] == 20
    assert body[0]["offsetSeconds"] == 6.0
    assert body[0]["timestamp"] == "10:00:06 AM"
    assert "late crop" in body[0]["matchReason"]
    assert ranker_called is False


def test_load_state_backfills_legacy_semantic_crops_from_thumbnail(monkeypatch, tmp_path: Path) -> None:
    configure_temp_storage(monkeypatch, tmp_path)

    state = store.seed_state()
    state["videos"] = [
        {
            "id": "video-1",
            "locationId": "edsa-sec-walk",
            "location": "EDSA Sec Walk",
            "timestamp": "10:00",
            "date": "2026-03-17",
            "startTime": "10:00",
            "endTime": "10:30",
            "gpsLat": 14.6397,
            "gpsLng": 121.0775,
            "pedestrianCount": 1,
            "rawPath": None,
            "processedPath": None,
        }
    ]
    state["pedestrianTracks"] = [
        {
            "id": "track-legacy-semantic",
            "videoId": "video-1",
            "pedestrianId": 5,
            "location": "EDSA Sec Walk",
            "bestTimestamp": "10:00:02 AM",
            "bestFrame": 4,
            "bestOffsetSeconds": 2.0,
            "thumbnailPath": "storage/videos/processed/video-1/tracks/track-5.jpg",
            "appearanceSummary": "Representative crop suggests upper clothing appears red.",
        }
    ]
    store.DATA_FILE.write_text(json.dumps(state, indent=2), encoding="utf-8")

    loaded_state = store.load_state()
    track = loaded_state["pedestrianTracks"][0]

    assert track["semanticCrops"] == [
        {
            "label": "best",
            "path": "storage/videos/processed/video-1/tracks/track-5.jpg",
            "frame": 4,
            "timestamp": "10:00:02 AM",
            "offsetSeconds": 2.0,
        }
    ]


def test_search_endpoint_expands_descriptive_color_queries_for_ai_ranking(monkeypatch, tmp_path: Path) -> None:
    configure_temp_storage(monkeypatch, tmp_path)

    state = store.seed_state()
    state["videos"] = [
        {
            "id": "video-1",
            "locationId": "edsa-sec-walk",
            "location": "EDSA Sec Walk",
            "timestamp": "10:00",
            "date": "2026-03-17",
            "startTime": "10:00",
            "endTime": "10:30",
            "gpsLat": 14.6397,
            "gpsLng": 121.0775,
            "pedestrianCount": 2,
            "rawPath": None,
            "processedPath": None,
        }
    ]
    state["pedestrianTracks"] = [
        {
            "id": "track-purple",
            "videoId": "video-1",
            "pedestrianId": 3,
            "location": "EDSA Sec Walk",
            "firstTimestamp": "10:01:00 AM",
            "lastTimestamp": "10:01:07 AM",
            "bestTimestamp": "10:01:04 AM",
            "firstFrame": 12,
            "lastFrame": 28,
            "bestFrame": 20,
            "firstOffsetSeconds": 60.0,
            "lastOffsetSeconds": 67.0,
            "bestOffsetSeconds": 64.0,
            "thumbnailPath": "storage/videos/processed/video-1/tracks/track-3.jpg",
            "appearanceHints": ["head region appears gray", "upper clothing appears purple", "lower clothing appears gray"],
            "appearanceSummary": "Representative crop suggests head region appears gray, upper clothing appears purple, lower clothing appears gray.",
            "occlusionClass": None,
            "bestArea": 5200.0,
        },
        {
            "id": "track-neutral",
            "videoId": "video-1",
            "pedestrianId": 4,
            "location": "EDSA Sec Walk",
            "firstTimestamp": "10:02:00 AM",
            "lastTimestamp": "10:02:05 AM",
            "bestTimestamp": "10:02:02 AM",
            "firstFrame": 30,
            "lastFrame": 42,
            "bestFrame": 36,
            "firstOffsetSeconds": 120.0,
            "lastOffsetSeconds": 125.0,
            "bestOffsetSeconds": 122.0,
            "thumbnailPath": "storage/videos/processed/video-1/tracks/track-4.jpg",
            "appearanceHints": ["head region appears black", "upper clothing appears gray", "lower clothing appears gray"],
            "appearanceSummary": "Representative crop suggests head region appears black, upper clothing appears gray, lower clothing appears gray.",
            "occlusionClass": None,
            "bestArea": 4100.0,
        },
    ]
    store.save_state(state)

    observed: dict[str, object] = {}

    def fake_ranker(query: str, candidates: list[dict[str, object]]) -> list[dict[str, object]]:
        observed["query"] = query
        observed["candidates"] = candidates
        return [
            {
                "id": "track-purple",
                "confidence": 88,
                "reason": "Upper clothing appears purple, which is the closest available match to a maroon or dark red top.",
            }
        ]

    monkeypatch.setattr(main.gemini, "rank_pedestrian_matches", fake_ranker)

    with TestClient(main.app) as client:
        response = client.get("/api/search", params={"query": "sleeveless, maroon/dark red flowy top"})

    assert response.status_code == 200
    body = response.json()
    assert len(body) == 1
    assert body[0]["id"] == "track-purple"
    assert body[0]["offsetSeconds"] == 64.0
    assert "closest available match" in body[0]["matchReason"]
    candidates = observed["candidates"]
    assert isinstance(candidates, list)
    assert any(candidate.get("id") == "track-purple" for candidate in candidates)


def test_search_endpoint_uses_cloud_vision_metadata_for_apparel_queries(monkeypatch, tmp_path: Path) -> None:
    configure_temp_storage(monkeypatch, tmp_path)

    state = store.seed_state()
    state["videos"] = [
        {
            "id": "video-1",
            "locationId": "edsa-sec-walk",
            "location": "EDSA Sec Walk",
            "timestamp": "10:00",
            "date": "2026-03-17",
            "startTime": "10:00",
            "endTime": "10:30",
            "gpsLat": 14.6397,
            "gpsLng": 121.0775,
            "pedestrianCount": 1,
            "rawPath": None,
            "processedPath": None,
        }
    ]
    state["pedestrianTracks"] = [
        {
            "id": "track-vision",
            "videoId": "video-1",
            "pedestrianId": 14,
            "location": "EDSA Sec Walk",
            "firstTimestamp": "10:03:00 AM",
            "lastTimestamp": "10:03:07 AM",
            "bestTimestamp": "10:03:02 AM",
            "firstFrame": 40,
            "lastFrame": 60,
            "bestFrame": 45,
            "firstOffsetSeconds": 180.0,
            "lastOffsetSeconds": 187.0,
            "bestOffsetSeconds": 182.0,
            "thumbnailPath": "storage/videos/processed/video-1/tracks/track-14.jpg",
            "appearanceHints": ["head region appears black", "upper clothing appears red"],
            "appearanceSummary": "Representative crop suggests head region appears black, upper clothing appears red.",
            "occlusionClass": None,
            "bestArea": 5000.0,
            "visualLabels": ["dress", "backpack"],
            "visualObjects": ["bag"],
            "visualLogos": ["nike"],
            "visualText": ["ATENEO"],
            "visualSummary": "Cloud Vision labels: dress, backpack. Detected objects: bag. Detected logos: nike. Visible text: ATENEO.",
        }
    ]
    store.save_state(state)

    observed: dict[str, object] = {}
    monkeypatch.setattr(
        main.gemini,
        "parse_search_query",
        lambda query, locations: {
            "locationId": None,
            "locationName": None,
            "appearanceTerms": ["dress", "backpack", "nike"],
            "softTerms": [],
            "unsupportedTerms": [],
            "regionColorRequirements": [],
            "summary": "Use Cloud Vision apparel and logo evidence.",
        },
    )

    def fake_ranker(query: str, candidates: list[dict[str, object]]) -> list[dict[str, object]]:
        observed["query"] = query
        observed["candidates"] = candidates
        return [
            {
                "id": "track-vision",
                "confidence": 95,
                "reason": "Cloud Vision metadata shows a dress, backpack, and nike logo on the representative thumbnail.",
            }
        ]

    monkeypatch.setattr(main.gemini, "rank_pedestrian_matches", fake_ranker)

    with TestClient(main.app) as client:
        response = client.get("/api/search", params={"query": "looking for someone wearing a dress with a backpack and nike logo"})

    assert response.status_code == 200
    body = response.json()
    assert len(body) == 1
    assert body[0]["id"] == "track-vision"
    assert body[0]["pedestrianId"] == 14
    assert "Cloud Vision metadata" in body[0]["matchReason"]
    assert body[0]["visualLabels"] == ["dress", "backpack"]
    assert body[0]["visualObjects"] == ["bag"]
    assert body[0]["visualLogos"] == ["nike"]
    assert body[0]["visualText"] == ["ATENEO"]
    assert body[0]["visualSummary"].startswith("Cloud Vision labels")
    candidates = observed["candidates"]
    assert isinstance(candidates, list)
    assert candidates[0]["visualLabels"] == ["dress", "backpack"]
    assert candidates[0]["visualLogos"] == ["nike"]
    assert candidates[0]["visualText"] == ["ATENEO"]
    assert "Searchable appearance terms: dress, backpack, nike" in str(observed["query"])


def test_search_endpoint_requires_region_specific_color_matches(monkeypatch, tmp_path: Path) -> None:
    configure_temp_storage(monkeypatch, tmp_path)

    state = store.seed_state()
    state["videos"] = [
        {
            "id": "video-1",
            "locationId": "edsa-sec-walk",
            "location": "EDSA Sec Walk",
            "timestamp": "10:00",
            "date": "2026-03-17",
            "startTime": "10:00",
            "endTime": "10:30",
            "gpsLat": 14.6397,
            "gpsLng": 121.0775,
            "pedestrianCount": 2,
            "rawPath": None,
            "processedPath": None,
        }
    ]
    state["pedestrianTracks"] = [
        {
            "id": "track-blue-shirt",
            "videoId": "video-1",
            "pedestrianId": 7,
            "location": "EDSA Sec Walk",
            "firstTimestamp": "10:00:00 AM",
            "lastTimestamp": "10:00:05 AM",
            "bestTimestamp": "10:00:02 AM",
            "firstFrame": 1,
            "lastFrame": 12,
            "bestFrame": 4,
            "firstOffsetSeconds": 0.0,
            "lastOffsetSeconds": 5.0,
            "bestOffsetSeconds": 2.0,
            "thumbnailPath": "storage/videos/processed/video-1/tracks/track-7.jpg",
            "appearanceHints": ["head region appears gray", "upper clothing appears blue", "lower clothing appears gray"],
            "appearanceSummary": "Representative crop suggests head region appears gray, upper clothing appears blue, lower clothing appears gray.",
            "occlusionClass": None,
            "bestArea": 4200.0,
        }
    ]
    store.save_state(state)

    ranker_called = False

    def fake_ranker(query: str, candidates: list[dict[str, object]]) -> list[dict[str, object]]:
        nonlocal ranker_called
        ranker_called = True
        return []

    monkeypatch.setattr(main.gemini, "rank_pedestrian_matches", fake_ranker)

    with TestClient(main.app) as client:
        response = client.get("/api/search", params={"query": "blue hat"})

    assert response.status_code == 200
    assert response.json() == []
    assert ranker_called is False


def test_search_endpoint_understands_full_sentence_location_queries(monkeypatch, tmp_path: Path) -> None:
    configure_temp_storage(monkeypatch, tmp_path)

    state = store.seed_state()
    state["videos"] = [
        {
            "id": "video-edsa",
            "locationId": "edsa-sec-walk",
            "location": "EDSA Sec Walk",
            "timestamp": "10:00",
            "date": "2026-03-17",
            "startTime": "10:00",
            "endTime": "10:30",
            "gpsLat": 14.6397,
            "gpsLng": 121.0775,
            "pedestrianCount": 1,
            "rawPath": None,
            "processedPath": None,
        },
        {
            "id": "video-kostka",
            "locationId": "kostka-walk",
            "location": "Kostka Walk",
            "timestamp": "11:00",
            "date": "2026-03-17",
            "startTime": "11:00",
            "endTime": "11:30",
            "gpsLat": 14.6390,
            "gpsLng": 121.0781,
            "pedestrianCount": 1,
            "rawPath": None,
            "processedPath": None,
        },
    ]
    state["pedestrianTracks"] = [
        {
            "id": "track-red-edsa",
            "videoId": "video-edsa",
            "pedestrianId": 11,
            "location": "EDSA Sec Walk",
            "firstTimestamp": "10:00:00 AM",
            "lastTimestamp": "10:00:06 AM",
            "bestTimestamp": "10:00:03 AM",
            "firstFrame": 1,
            "lastFrame": 15,
            "bestFrame": 7,
            "firstOffsetSeconds": 0.0,
            "lastOffsetSeconds": 6.0,
            "bestOffsetSeconds": 3.0,
            "thumbnailPath": "storage/videos/processed/video-edsa/tracks/track-11.jpg",
            "appearanceHints": ["head region appears black", "upper clothing appears red", "lower clothing appears gray"],
            "appearanceSummary": "Representative crop suggests head region appears black, upper clothing appears red, lower clothing appears gray.",
            "occlusionClass": None,
            "bestArea": 4400.0,
        },
        {
            "id": "track-red-kostka",
            "videoId": "video-kostka",
            "pedestrianId": 22,
            "location": "Kostka Walk",
            "firstTimestamp": "11:00:00 AM",
            "lastTimestamp": "11:00:05 AM",
            "bestTimestamp": "11:00:02 AM",
            "firstFrame": 20,
            "lastFrame": 32,
            "bestFrame": 25,
            "firstOffsetSeconds": 0.0,
            "lastOffsetSeconds": 5.0,
            "bestOffsetSeconds": 2.0,
            "thumbnailPath": "storage/videos/processed/video-kostka/tracks/track-22.jpg",
            "appearanceHints": ["head region appears gray", "upper clothing appears red", "lower clothing appears gray"],
            "appearanceSummary": "Representative crop suggests head region appears gray, upper clothing appears red, lower clothing appears gray.",
            "occlusionClass": None,
            "bestArea": 4300.0,
        },
    ]
    store.save_state(state)

    observed: dict[str, object] = {}

    monkeypatch.setattr(
        main.gemini,
        "parse_search_query",
        lambda query, locations: {
            "locationId": "edsa-sec-walk",
            "locationName": "EDSA Sec Walk",
            "appearanceTerms": ["red"],
            "softTerms": ["short", "dress"],
            "unsupportedTerms": [],
            "regionColorRequirements": [{"region": "upper clothing", "colors": ["red"]}],
            "summary": "Interpret the query as a person at Xavier Hall / EDSA Sec Walk with red clothing; short and dress are soft preferences.",
        },
    )

    def fake_ranker(query: str, candidates: list[dict[str, object]]) -> list[dict[str, object]]:
        observed["query"] = query
        observed["candidate_ids"] = [candidate.get("id") for candidate in candidates]
        return [{"id": "track-red-edsa", "confidence": 92, "reason": "Red upper clothing at the requested Xavier Hall camera location."}]

    monkeypatch.setattr(main.gemini, "rank_pedestrian_matches", fake_ranker)

    with TestClient(main.app) as client:
        response = client.get("/api/search", params={"query": "i am looking for a short person who wears a red dress in xavier hall"})

    assert response.status_code == 200
    body = response.json()
    assert len(body) == 1
    assert body[0]["id"] == "track-red-edsa"
    assert body[0]["location"] == "EDSA Sec Walk"
    assert observed["candidate_ids"] == ["track-red-edsa"]
    assert "Required location: EDSA Sec Walk" in str(observed["query"])
    assert "Soft preferences" in str(observed["query"])


def test_search_endpoint_falls_back_when_query_parser_is_unavailable(monkeypatch, tmp_path: Path) -> None:
    configure_temp_storage(monkeypatch, tmp_path)

    state = store.seed_state()
    state["videos"] = [
        {
            "id": "video-1",
            "locationId": "edsa-sec-walk",
            "location": "EDSA Sec Walk",
            "timestamp": "10:00",
            "date": "2026-03-17",
            "startTime": "10:00",
            "endTime": "10:30",
            "gpsLat": 14.6397,
            "gpsLng": 121.0775,
            "pedestrianCount": 1,
            "rawPath": None,
            "processedPath": None,
        }
    ]
    state["pedestrianTracks"] = [
        {
            "id": "track-blue",
            "videoId": "video-1",
            "pedestrianId": 7,
            "location": "EDSA Sec Walk",
            "firstTimestamp": "10:00:00 AM",
            "lastTimestamp": "10:00:05 AM",
            "bestTimestamp": "10:00:02 AM",
            "firstFrame": 1,
            "lastFrame": 12,
            "bestFrame": 4,
            "firstOffsetSeconds": 0.0,
            "lastOffsetSeconds": 5.0,
            "bestOffsetSeconds": 2.0,
            "thumbnailPath": "storage/videos/processed/video-1/tracks/track-7.jpg",
            "appearanceHints": ["head region appears blue", "lower clothing appears blue"],
            "appearanceSummary": "Representative crop suggests head region appears blue, upper clothing appears white, lower clothing appears blue.",
            "occlusionClass": None,
            "bestArea": 4200.0,
        }
    ]
    store.save_state(state)

    monkeypatch.setattr(main.gemini, "parse_search_query", lambda query, locations: (_ for _ in ()).throw(RuntimeError("parser unavailable")))
    monkeypatch.setattr(
        main.gemini,
        "rank_pedestrian_matches",
        lambda query, candidates: [{"id": "track-blue", "confidence": 94, "reason": "Head region and lower clothing are both described as blue."}],
    )

    with TestClient(main.app) as client:
        response = client.get("/api/search", params={"query": "im looking for a pedestrian wearing a blue hat and blue shorts"})

    assert response.status_code == 200
    body = response.json()
    assert len(body) == 1
    assert body[0]["id"] == "track-blue"


def test_search_endpoint_matches_white_shirt_from_appearance_summary(monkeypatch, tmp_path: Path) -> None:
    configure_temp_storage(monkeypatch, tmp_path)

    state = store.seed_state()
    state["videos"] = [
        {
            "id": "video-1",
            "locationId": "edsa-sec-walk",
            "location": "EDSA Sec Walk",
            "timestamp": "10:00",
            "date": "2026-03-17",
            "startTime": "10:00",
            "endTime": "10:30",
            "gpsLat": 14.6397,
            "gpsLng": 121.0775,
            "pedestrianCount": 1,
            "rawPath": None,
            "processedPath": None,
        }
    ]
    state["pedestrianTracks"] = [
        {
            "id": "track-white-shirt",
            "videoId": "video-1",
            "pedestrianId": 5,
            "location": "EDSA Sec Walk",
            "firstTimestamp": "10:00:00 AM",
            "lastTimestamp": "10:00:05 AM",
            "bestTimestamp": "10:00:02 AM",
            "firstFrame": 1,
            "lastFrame": 10,
            "bestFrame": 4,
            "firstOffsetSeconds": 0.0,
            "lastOffsetSeconds": 5.0,
            "bestOffsetSeconds": 2.0,
            "thumbnailPath": "storage/videos/processed/video-1/tracks/track-5.jpg",
            "appearanceHints": ["head region appears blue", "lower clothing appears blue"],
            "appearanceSummary": "Representative crop suggests head region appears blue, upper clothing appears white, lower clothing appears blue.",
            "occlusionClass": None,
            "bestArea": 4200.0,
        }
    ]
    store.save_state(state)

    monkeypatch.setattr(main.gemini, "parse_search_query", lambda query, locations: (_ for _ in ()).throw(RuntimeError("parser unavailable")))
    monkeypatch.setattr(
        main.gemini,
        "rank_pedestrian_matches",
        lambda query, candidates: [{"id": "track-white-shirt", "confidence": 91, "reason": "Upper clothing is described as white in the appearance summary."}],
    )

    with TestClient(main.app) as client:
        response = client.get("/api/search", params={"query": "person wearing white shirt"})

    assert response.status_code == 200
    body = response.json()
    assert len(body) == 1
    assert body[0]["id"] == "track-white-shirt"
    assert body[0]["pedestrianId"] == 5


def test_search_endpoint_tolerates_white_gray_camera_shift_for_upper_clothing(monkeypatch, tmp_path: Path) -> None:
    configure_temp_storage(monkeypatch, tmp_path)

    state = store.seed_state()
    state["videos"] = [
        {
            "id": "video-1",
            "locationId": "edsa-sec-walk",
            "location": "EDSA Sec Walk",
            "timestamp": "10:00",
            "date": "2026-03-17",
            "startTime": "10:00",
            "endTime": "10:30",
            "gpsLat": 14.6397,
            "gpsLng": 121.0775,
            "pedestrianCount": 1,
            "rawPath": None,
            "processedPath": None,
        }
    ]
    state["pedestrianTracks"] = [
        {
            "id": "track-gray-shirt",
            "videoId": "video-1",
            "pedestrianId": 16,
            "location": "EDSA Sec Walk",
            "firstTimestamp": "10:02:00 AM",
            "lastTimestamp": "10:02:05 AM",
            "bestTimestamp": "10:02:02 AM",
            "firstFrame": 30,
            "lastFrame": 40,
            "bestFrame": 35,
            "firstOffsetSeconds": 120.0,
            "lastOffsetSeconds": 125.0,
            "bestOffsetSeconds": 122.0,
            "thumbnailPath": "storage/videos/processed/video-1/tracks/track-16.jpg",
            "appearanceHints": ["head region appears black", "upper clothing appears gray", "lower clothing appears black"],
            "appearanceSummary": "Representative crop suggests head region appears black, upper clothing appears gray, lower clothing appears black.",
            "occlusionClass": None,
            "bestArea": 4100.0,
        }
    ]
    store.save_state(state)

    observed: dict[str, object] = {}
    monkeypatch.setattr(main.gemini, "parse_search_query", lambda query, locations: (_ for _ in ()).throw(RuntimeError("parser unavailable")))

    def fake_ranker(query: str, candidates: list[dict[str, object]]) -> list[dict[str, object]]:
        observed["candidate_ids"] = [candidate.get("id") for candidate in candidates]
        return [
            {
                "id": "track-gray-shirt",
                "confidence": 83,
                "reason": "Upper clothing appears gray, which can still align with a white shirt query under dim camera lighting.",
            }
        ]

    monkeypatch.setattr(main.gemini, "rank_pedestrian_matches", fake_ranker)

    with TestClient(main.app) as client:
        response = client.get("/api/search", params={"query": "person wearing white shirt"})

    assert response.status_code == 200
    body = response.json()
    assert observed["candidate_ids"] == ["track-gray-shirt"]
    assert len(body) == 1
    assert body[0]["id"] == "track-gray-shirt"


def test_search_endpoint_accepts_mixed_white_gray_shirt_queries(monkeypatch, tmp_path: Path) -> None:
    configure_temp_storage(monkeypatch, tmp_path)

    state = store.seed_state()
    state["videos"] = [
        {
            "id": "video-1",
            "locationId": "edsa-sec-walk",
            "location": "EDSA Sec Walk",
            "timestamp": "10:00",
            "date": "2026-03-17",
            "startTime": "10:00",
            "endTime": "10:30",
            "gpsLat": 14.6397,
            "gpsLng": 121.0775,
            "pedestrianCount": 1,
            "rawPath": None,
            "processedPath": None,
        }
    ]
    state["pedestrianTracks"] = [
        {
            "id": "track-white-gray-shirt",
            "videoId": "video-1",
            "pedestrianId": 15,
            "location": "EDSA Sec Walk",
            "firstTimestamp": "10:01:00 AM",
            "lastTimestamp": "10:01:06 AM",
            "bestTimestamp": "10:01:02 AM",
            "firstFrame": 15,
            "lastFrame": 28,
            "bestFrame": 20,
            "firstOffsetSeconds": 60.0,
            "lastOffsetSeconds": 66.0,
            "bestOffsetSeconds": 62.0,
            "thumbnailPath": "storage/videos/processed/video-1/tracks/track-15.jpg",
            "appearanceHints": ["head region appears black", "lower clothing appears gray"],
            "appearanceSummary": "Representative crop suggests head region appears black, upper clothing appears white, lower clothing appears gray.",
            "occlusionClass": None,
            "bestArea": 4300.0,
        }
    ]
    store.save_state(state)

    observed: dict[str, object] = {}
    monkeypatch.setattr(main.gemini, "parse_search_query", lambda query, locations: (_ for _ in ()).throw(RuntimeError("parser unavailable")))

    def fake_ranker(query: str, candidates: list[dict[str, object]]) -> list[dict[str, object]]:
        observed["candidate_ids"] = [candidate.get("id") for candidate in candidates]
        return [
            {
                "id": "track-white-gray-shirt",
                "confidence": 90,
                "reason": "Upper clothing is described as white, which fits the requested light white/gray shirt.",
            }
        ]

    monkeypatch.setattr(main.gemini, "rank_pedestrian_matches", fake_ranker)

    with TestClient(main.app) as client:
        response = client.get("/api/search", params={"query": "person wearing light white/gray shirt"})

    assert response.status_code == 200
    body = response.json()
    assert observed["candidate_ids"] == ["track-white-gray-shirt"]
    assert len(body) == 1
    assert body[0]["id"] == "track-white-gray-shirt"


def test_search_endpoint_backfills_legacy_video_metadata(monkeypatch, tmp_path: Path) -> None:
    configure_temp_storage(monkeypatch, tmp_path)

    raw_file = store.RAW_VIDEOS_DIR / "legacy.mp4"
    raw_file.write_bytes(b"legacy-video")

    state = store.seed_state()
    state["videos"] = [
        {
            "id": "legacy-video-1",
            "locationId": "edsa-sec-walk",
            "location": "EDSA Sec Walk",
            "timestamp": "10:00",
            "date": "2026-03-17",
            "startTime": "10:00",
            "endTime": "10:30",
            "gpsLat": 14.6397,
            "gpsLng": 121.0775,
            "pedestrianCount": 0,
            "rawPath": str(raw_file.relative_to(store.BACKEND_DIR)),
            "processedPath": None,
        }
    ]
    store.save_state(state)

    observed: dict[str, object] = {}

    def fake_run_video_inference(video_path: Path, model_name=None, video_record=None, fast_mode: bool = False, progress_callback=None):
        observed["video_path"] = video_path
        observed["video_id"] = video_record["id"]
        observed["fast_mode"] = fast_mode
        return {
            "pedestrianCount": 1,
            "processedPath": str((store.PROCESSED_VIDEOS_DIR / video_record["id"] / "legacy.mp4").relative_to(store.BACKEND_DIR)),
            "events": [
                {
                    "id": "evt-legacy",
                    "type": "detection",
                    "location": video_record["location"],
                    "timestamp": "10:00:01 AM",
                    "description": "Pedestrian ID #4 detected at frame 2",
                    "videoId": video_record["id"],
                    "pedestrianId": 4,
                    "frame": 2,
                    "offsetSeconds": 1.0,
                }
            ],
            "pedestrianTracks": [
                {
                    "id": "track-legacy",
                    "videoId": video_record["id"],
                    "pedestrianId": 4,
                    "location": video_record["location"],
                    "firstTimestamp": "10:00:00 AM",
                    "lastTimestamp": "10:00:05 AM",
                    "bestTimestamp": "10:00:02 AM",
                    "firstFrame": 1,
                    "lastFrame": 8,
                    "bestFrame": 3,
                    "firstOffsetSeconds": 0.0,
                    "lastOffsetSeconds": 5.0,
                    "bestOffsetSeconds": 2.0,
                    "thumbnailPath": None,
                    "appearanceHints": ["head region appears blue", "lower clothing appears blue"],
                    "appearanceSummary": "Representative crop suggests head region appears blue and lower clothing appears blue.",
                    "occlusionClass": None,
                    "bestArea": 2400.0,
                }
            ],
        }

    monkeypatch.setattr(main.inference, "run_video_inference", fake_run_video_inference)
    monkeypatch.setattr(
        main.gemini,
        "rank_pedestrian_matches",
        lambda query, candidates: [
            {"id": "track-legacy", "confidence": 91, "reason": "Blue appearance hints align with the query."}
        ],
    )

    with TestClient(main.app) as client:
        response = client.get("/api/search", params={"query": "blue hat and blue shorts"})

    assert response.status_code == 200
    body = response.json()
    assert observed["video_path"] == raw_file
    assert observed["video_id"] == "legacy-video-1"
    assert observed["fast_mode"] is False
    assert len(body) == 1
    assert body[0]["id"] == "track-legacy"
    assert body[0]["offsetSeconds"] == 2.0
    assert body[0]["previewPath"] == str((store.PROCESSED_VIDEOS_DIR / "legacy-video-1" / "legacy.mp4").relative_to(store.BACKEND_DIR))

    saved_state = store.load_state()
    saved_tracks = [track for track in saved_state["pedestrianTracks"] if track.get("videoId") == "legacy-video-1"]
    assert len(saved_tracks) == 1
    assert saved_state["videos"][0]["pedestrianCount"] == 1


def test_search_endpoint_returns_event_offset_seconds(monkeypatch, tmp_path: Path) -> None:
    configure_temp_storage(monkeypatch, tmp_path)

    state = store.seed_state()
    state["videos"] = [
        {
            "id": "video-1",
            "locationId": "edsa-sec-walk",
            "location": "EDSA Sec Walk",
            "timestamp": "10:00",
            "date": "2026-03-17",
            "startTime": "10:00",
            "endTime": "10:30",
            "gpsLat": 14.6397,
            "gpsLng": 121.0775,
            "pedestrianCount": 1,
            "rawPath": None,
            "processedPath": None,
        }
    ]
    state["events"] = [
        {
            "id": "evt-1",
            "type": "detection",
            "location": "EDSA Sec Walk",
            "timestamp": "10:00:08 AM",
            "description": "Sleeveless maroon top pedestrian detected near crosswalk",
            "videoId": "video-1",
            "pedestrianId": 11,
            "frame": 24,
            "offsetSeconds": 8.0,
        }
    ]
    store.save_state(state)

    with TestClient(main.app) as client:
        response = client.get("/api/search", params={"query": "Sleeveless maroon top"})

    assert response.status_code == 200
    body = response.json()
    assert len(body) == 1
    assert body[0]["id"] == "evt-1"
    assert body[0]["offsetSeconds"] == 8.0


def test_search_endpoint_does_not_match_generic_events_for_descriptive_queries(monkeypatch, tmp_path: Path) -> None:
    configure_temp_storage(monkeypatch, tmp_path)

    state = store.seed_state()
    state["videos"] = [
        {
            "id": "video-1",
            "locationId": "edsa-sec-walk",
            "location": "EDSA Sec Walk",
            "timestamp": "10:00",
            "date": "2026-03-17",
            "startTime": "10:00",
            "endTime": "10:30",
            "gpsLat": 14.6397,
            "gpsLng": 121.0775,
            "pedestrianCount": 1,
            "rawPath": None,
            "processedPath": None,
        }
    ]
    state["events"] = [
        {
            "id": "evt-generic",
            "type": "detection",
            "location": "EDSA Sec Walk",
            "timestamp": "10:00:08 AM",
            "description": "Pedestrian ID #11 detected at frame 24",
            "videoId": "video-1",
            "pedestrianId": 11,
            "frame": 24,
            "offsetSeconds": 8.0,
        }
    ]
    store.save_state(state)

    with TestClient(main.app) as client:
        response = client.get("/api/search", params={"query": "Sleeveless maroon top"})

    assert response.status_code == 200
    assert response.json() == []


def test_dashboard_endpoints_only_surface_real_footage_and_neutral_empty_locations(monkeypatch, tmp_path: Path) -> None:
    configure_temp_storage(monkeypatch, tmp_path)

    state = store.seed_state()
    for location in state["locations"]:
        if location["id"] == "edsa-sec-walk":
            location["roiCoordinates"] = {
                "referenceSize": [1920, 1080],
                "includePolygonsNorm": [
                    [[0.1, 0.1], [0.5, 0.1], [0.5, 0.5], [0.1, 0.5]],
                ],
            }
            location["walkableAreaM2"] = 4.0

    state["videos"] = [
        {
            "id": "video-1",
            "locationId": "edsa-sec-walk",
            "location": "EDSA Sec Walk",
            "timestamp": "10:00",
            "date": "2026-03-17",
            "startTime": "10:00",
            "endTime": "10:30",
            "gpsLat": 14.6397,
            "gpsLng": 121.0775,
            "pedestrianCount": 3,
            "rawPath": None,
            "processedPath": None,
        }
    ]
    state["pedestrianTracks"] = [
        {
            "id": "track-in-roi",
            "videoId": "video-1",
            "pedestrianId": 1,
            "location": "EDSA Sec Walk",
            "firstTimestamp": "10:01:00 AM",
            "lastTimestamp": "10:01:30 AM",
            "bestTimestamp": "10:01:10 AM",
            "firstFrame": 10,
            "lastFrame": 30,
            "bestFrame": 15,
            "firstOffsetSeconds": 60.0,
            "lastOffsetSeconds": 90.0,
            "bestOffsetSeconds": 70.0,
            "footPointNorm": [0.3, 0.3],
            "occlusionClass": 1,
        },
        {
            "id": "track-outside-roi",
            "videoId": "video-1",
            "pedestrianId": 2,
            "location": "EDSA Sec Walk",
            "firstTimestamp": "10:05:00 AM",
            "lastTimestamp": "10:05:20 AM",
            "bestTimestamp": "10:05:10 AM",
            "firstFrame": 40,
            "lastFrame": 60,
            "bestFrame": 48,
            "firstOffsetSeconds": 300.0,
            "lastOffsetSeconds": 320.0,
            "bestOffsetSeconds": 310.0,
            "footPointNorm": [0.8, 0.8],
            "occlusionClass": 2,
        },
        {
            "id": "track-in-roi-no-occlusion",
            "videoId": "video-1",
            "pedestrianId": 3,
            "location": "EDSA Sec Walk",
            "firstTimestamp": "10:08:00 AM",
            "lastTimestamp": "10:08:15 AM",
            "bestTimestamp": "10:08:05 AM",
            "firstFrame": 70,
            "lastFrame": 82,
            "bestFrame": 74,
            "firstOffsetSeconds": 480.0,
            "lastOffsetSeconds": 495.0,
            "bestOffsetSeconds": 485.0,
            "footPointNorm": [0.25, 0.25],
            "occlusionClass": None,
        },
    ]
    state["events"] = [
        {
            "id": "event-1",
            "type": "detection",
            "location": "EDSA Sec Walk",
            "timestamp": "10:01:00",
            "description": "Light occlusion pedestrian ID #1 detected at frame 10",
            "videoId": "video-1",
            "pedestrianId": 1,
            "frame": 10,
            "offsetSeconds": 1.0,
            "occlusionClass": 0,
        },
        {
            "id": "event-2",
            "type": "detection",
            "location": "EDSA Sec Walk",
            "timestamp": "10:05:00",
            "description": "Moderate occlusion pedestrian ID #2 detected at frame 20",
            "videoId": "video-1",
            "pedestrianId": 2,
            "frame": 20,
            "offsetSeconds": 5.0,
            "occlusionClass": 1,
        },
        {
            "id": "event-3",
            "type": "detection",
            "location": "EDSA Sec Walk",
            "timestamp": "10:08:00",
            "description": "Pedestrian ID #3 detected at frame 30",
            "videoId": "video-1",
            "pedestrianId": 3,
            "frame": 30,
            "offsetSeconds": 8.0,
            "occlusionClass": None,
        },
    ]
    store.save_state(state)

    with TestClient(main.app) as client:
        summary_response = client.get("/api/dashboard/summary", params={"date": "2026-03-17"})
        traffic_response = client.get("/api/dashboard/traffic", params={"date": "2026-03-17", "timeRange": "whole-day"})
        drilldown_response = client.get(
            "/api/dashboard/traffic",
            params={"date": "2026-03-17", "timeRange": "whole-day", "focusTime": "10:00", "zoomLevel": 1},
        )
        nested_drilldown_response = client.get(
            "/api/dashboard/traffic",
            params={"date": "2026-03-17", "timeRange": "whole-day", "focusTime": "10:00", "zoomLevel": 2},
        )
        occlusion_trends_response = client.get(
            "/api/dashboard/occlusion-trends",
            params={"date": "2026-03-17", "timeRange": "whole-day", "focusTime": "10:00", "zoomLevel": 1},
        )
        occlusion_response = client.get("/api/dashboard/occlusion", params={"date": "2026-03-17", "timeRange": "whole-day"})

    assert summary_response.status_code == 200
    assert traffic_response.status_code == 200
    assert drilldown_response.status_code == 200
    assert nested_drilldown_response.status_code == 200
    assert occlusion_trends_response.status_code == 200
    assert occlusion_response.status_code == 200

    summary = summary_response.json()
    assert summary["monitoredLocations"] == 1
    assert summary["totalHeavyOcclusions"] == 0

    traffic = traffic_response.json()
    assert traffic["timeRange"] == "whole-day"
    assert traffic["series"]
    assert len({point["id"] for point in traffic["series"]}) == len(traffic["series"])
    assert traffic["bucketMinutes"] == 60
    assert traffic["zoomLevel"] == 0
    assert traffic["canZoomIn"] is True
    assert traffic["isDrilldown"] is False
    assert traffic["locationTotals"][0] == {"location": "EDSA Sec Walk", "totalPedestrians": 3}
    whole_day_bucket = next(point for point in traffic["series"] if point["time"] == "10:00")
    assert set(whole_day_bucket.keys()) == {"id", "time", "cumulativeUniquePedestrians", "averageVisiblePedestrians", "EDSA Sec Walk"}
    assert whole_day_bucket["id"].startswith("2026-03-17T10:00:00")
    assert whole_day_bucket["cumulativeUniquePedestrians"] == 3
    assert whole_day_bucket["EDSA Sec Walk"] == 3
    assert whole_day_bucket["averageVisiblePedestrians"] == 1.0

    drilldown = drilldown_response.json()
    assert drilldown["bucketMinutes"] == 15
    assert drilldown["zoomLevel"] == 1
    assert drilldown["canZoomIn"] is True
    assert drilldown["isDrilldown"] is True
    assert drilldown["focusTime"] == "10:00"
    assert drilldown["windowStart"] == "10:00"
    assert drilldown["windowEnd"] == "11:00"
    assert any(point["time"] == "10:00" for point in drilldown["series"])
    first_drill_bucket = next(point for point in drilldown["series"] if point["time"] == "10:00")
    assert first_drill_bucket["cumulativeUniquePedestrians"] == 3
    assert first_drill_bucket["EDSA Sec Walk"] == 3
    assert first_drill_bucket["averageVisiblePedestrians"] == 1.0

    nested_drilldown = nested_drilldown_response.json()
    assert nested_drilldown["bucketMinutes"] == 5
    assert nested_drilldown["zoomLevel"] == 2
    assert nested_drilldown["canZoomIn"] is False
    assert nested_drilldown["windowStart"] == "10:00"
    assert nested_drilldown["windowEnd"] == "10:15"
    nested_bucket = next(point for point in nested_drilldown["series"] if point["time"] == "10:05")
    assert nested_bucket["cumulativeUniquePedestrians"] == 3
    assert nested_bucket["averageVisiblePedestrians"] == 1.0

    occlusion_trends = occlusion_trends_response.json()
    assert len({point["id"] for point in occlusion_trends["series"]}) == len(occlusion_trends["series"])
    trend_bucket = next(point for point in occlusion_trends["series"] if point["time"] == "10:00")
    assert occlusion_trends["bucketMinutes"] == 15
    assert occlusion_trends["zoomLevel"] == 1
    assert occlusion_trends["canZoomIn"] is True
    assert trend_bucket["id"].startswith("2026-03-17T10:00:00")
    assert trend_bucket["Light"] == pytest.approx(0.33, abs=0.01)
    assert trend_bucket["Moderate"] == pytest.approx(0.33, abs=0.01)
    assert trend_bucket["Heavy"] == 0

    occlusion = occlusion_response.json()
    assert "10:00" in occlusion["availableHours"]

    edsa_sec_walk = next(location for location in occlusion["locations"] if location["id"] == "edsa-sec-walk")
    kostka_walk = next(location for location in occlusion["locations"] if location["id"] == "kostka-walk")

    assert edsa_sec_walk["hasFootage"] is True
    assert edsa_sec_walk["hasPTSIData"] is True
    assert edsa_sec_walk["state"] == "clear"
    assert edsa_sec_walk["los"] == "B"
    assert edsa_sec_walk["losDescription"] == "high pedestrian space, comfortable movement"
    assert edsa_sec_walk["score"] == pytest.approx(26.0, abs=0.01)
    assert edsa_sec_walk["mode"] == "strict-fhwa"
    assert edsa_sec_walk["averagePedestrians"] == pytest.approx(1.0, abs=0.01)
    assert edsa_sec_walk["uniquePedestrians"] == 2
    assert edsa_sec_walk["occlusionMix"] == {
        "lightPercent": pytest.approx(0.0, abs=0.01),
        "moderatePercent": pytest.approx(50.0, abs=0.01),
        "heavyPercent": pytest.approx(0.0, abs=0.01),
    }
    assert edsa_sec_walk["peakHour"] == "10:00"
    assert edsa_sec_walk["peakHourScore"] == pytest.approx(26.0, abs=0.01)
    assert edsa_sec_walk["offPeakHour"] is None
    assert edsa_sec_walk["offPeakHourScore"] is None
    assert edsa_sec_walk["hourlyScores"] == [
        {
            "hour": "10:00",
            "score": pytest.approx(26.0, abs=0.01),
            "mode": "strict-fhwa",
            "averagePedestrians": pytest.approx(1.0, abs=0.01),
            "uniquePedestrians": 2,
            "occlusionMix": {
                "lightPercent": pytest.approx(0.0, abs=0.01),
                "moderatePercent": pytest.approx(50.0, abs=0.01),
                "heavyPercent": pytest.approx(0.0, abs=0.01),
            },
            "los": "B",
            "losDescription": "high pedestrian space, comfortable movement",
        }
    ]
    assert kostka_walk["hasFootage"] is False
    assert kostka_walk["state"] == "no-footage"


def test_dashboard_occlusion_uses_per_second_trajectory_samples_for_ptsi(monkeypatch, tmp_path: Path) -> None:
    configure_temp_storage(monkeypatch, tmp_path)

    state = store.seed_state()
    for location in state["locations"]:
        if location["id"] == "edsa-sec-walk":
            location["roiCoordinates"] = {
                "referenceSize": [1920, 1080],
                "includePolygonsNorm": [[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]],
            }
            location["walkableAreaM2"] = None

    state["videos"] = [
        {
            "id": "video-ptsi-1",
            "locationId": "edsa-sec-walk",
            "location": "EDSA Sec Walk",
            "timestamp": "10:00",
            "date": "2026-03-17",
            "startTime": "10:00",
            "endTime": "10:10",
            "gpsLat": 14.6397,
            "gpsLng": 121.0775,
            "pedestrianCount": 3,
            "rawPath": None,
            "processedPath": None,
        }
    ]
    state["pedestrianTracks"] = [
        {
            "id": "track-a",
            "videoId": "video-ptsi-1",
            "pedestrianId": 1,
            "location": "EDSA Sec Walk",
            "trajectorySamples": [[0, 0.2, 0.2, None]],
        },
        {
            "id": "track-b",
            "videoId": "video-ptsi-1",
            "pedestrianId": 2,
            "location": "EDSA Sec Walk",
            "trajectorySamples": [[1, 0.35, 0.35, 2]],
        },
        {
            "id": "track-c",
            "videoId": "video-ptsi-1",
            "pedestrianId": 3,
            "location": "EDSA Sec Walk",
            "trajectorySamples": [[1, 0.45, 0.45, None]],
        },
    ]
    store.save_state(state)

    with TestClient(main.app) as client:
        response = client.get("/api/dashboard/occlusion", params={"date": "2026-03-17", "timeRange": "whole-day"})

    assert response.status_code == 200
    payload = response.json()
    edsa_sec_walk = next(location for location in payload["locations"] if location["id"] == "edsa-sec-walk")

    assert edsa_sec_walk["hasPTSIData"] is True
    assert edsa_sec_walk["mode"] == "roi-testing"
    assert edsa_sec_walk["state"] == "clear"
    assert edsa_sec_walk["los"] == "A"
    assert edsa_sec_walk["losDescription"] == "very high pedestrian space, free movement"
    assert edsa_sec_walk["score"] == pytest.approx(13.48, abs=0.01)
    assert edsa_sec_walk["averagePedestrians"] == pytest.approx(1.5, abs=0.01)
    assert edsa_sec_walk["uniquePedestrians"] == 3
    assert edsa_sec_walk["occlusionMix"] == {
        "lightPercent": pytest.approx(0.0, abs=0.01),
        "moderatePercent": pytest.approx(0.0, abs=0.01),
        "heavyPercent": pytest.approx(33.3, abs=0.1),
    }
    assert edsa_sec_walk["peakHour"] == "10:00"
    assert edsa_sec_walk["peakHourScore"] == pytest.approx(13.48, abs=0.01)
    assert edsa_sec_walk["offPeakHour"] is None
    assert edsa_sec_walk["offPeakHourScore"] is None
    assert edsa_sec_walk["hourlyScores"] == [
        {
            "hour": "10:00",
            "score": pytest.approx(13.48, abs=0.01),
            "mode": "roi-testing",
            "averagePedestrians": pytest.approx(1.5, abs=0.01),
            "uniquePedestrians": 3,
            "occlusionMix": {
                "lightPercent": pytest.approx(0.0, abs=0.01),
                "moderatePercent": pytest.approx(0.0, abs=0.01),
                "heavyPercent": pytest.approx(33.3, abs=0.1),
            },
            "los": "A",
            "losDescription": "very high pedestrian space, free movement",
        }
    ]


def test_dashboard_occlusion_all_hours_summary_uses_selected_range_totals(monkeypatch, tmp_path: Path) -> None:
    configure_temp_storage(monkeypatch, tmp_path)

    state = store.seed_state()
    for location in state["locations"]:
        if location["id"] == "edsa-sec-walk":
            location["roiCoordinates"] = {
                "referenceSize": [1920, 1080],
                "includePolygonsNorm": [[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]],
            }
            location["walkableAreaM2"] = None

    state["videos"] = [
        {
            "id": "video-ptsi-all-hours",
            "locationId": "edsa-sec-walk",
            "location": "EDSA Sec Walk",
            "timestamp": "10:00",
            "date": "2026-03-17",
            "startTime": "10:00",
            "endTime": "12:00",
            "gpsLat": 14.6397,
            "gpsLng": 121.0775,
            "pedestrianCount": 3,
            "rawPath": None,
            "processedPath": None,
        }
    ]
    state["pedestrianTracks"] = [
        {
            "id": "track-all-hours-a",
            "videoId": "video-ptsi-all-hours",
            "pedestrianId": 1,
            "location": "EDSA Sec Walk",
            "trajectorySamples": [[0, 0.2, 0.2, None], [3600, 0.2, 0.2, None]],
        },
        {
            "id": "track-all-hours-b",
            "videoId": "video-ptsi-all-hours",
            "pedestrianId": 2,
            "location": "EDSA Sec Walk",
            "trajectorySamples": [[0, 0.35, 0.35, None]],
        },
        {
            "id": "track-all-hours-c",
            "videoId": "video-ptsi-all-hours",
            "pedestrianId": 3,
            "location": "EDSA Sec Walk",
            "trajectorySamples": [[3600, 0.45, 0.45, 2]],
        },
    ]
    store.save_state(state)

    with TestClient(main.app) as client:
        response = client.get("/api/dashboard/occlusion", params={"date": "2026-03-17", "timeRange": "whole-day"})

    assert response.status_code == 200
    payload = response.json()
    assert {"10:00", "11:00"}.issubset(set(payload["availableHours"]))

    edsa_sec_walk = next(location for location in payload["locations"] if location["id"] == "edsa-sec-walk")
    ten_oclock = next(score for score in edsa_sec_walk["hourlyScores"] if score["hour"] == "10:00")
    eleven_oclock = next(score for score in edsa_sec_walk["hourlyScores"] if score["hour"] == "11:00")

    assert ten_oclock["uniquePedestrians"] == 2
    assert eleven_oclock["uniquePedestrians"] == 2
    assert edsa_sec_walk["peakHour"] == "11:00"
    assert edsa_sec_walk["offPeakHour"] == "10:00"
    assert edsa_sec_walk["peakHourScore"] == pytest.approx(eleven_oclock["score"], abs=0.01)
    assert edsa_sec_walk["uniquePedestrians"] == 3
    assert edsa_sec_walk["averagePedestrians"] == pytest.approx(2.0, abs=0.01)
    assert edsa_sec_walk["uniquePedestrians"] > eleven_oclock["uniquePedestrians"]


@pytest.mark.parametrize(
    ("space_per_pedestrian", "expected_los"),
    [
        (5.7, "A"),
        (5.6, "B"),
        (3.7, "B"),
        (3.69, "C"),
        (2.2, "C"),
        (2.19, "D"),
        (1.4, "D"),
        (1.39, "E"),
        (0.75, "E"),
        (0.74, "F"),
    ],
)
def test_ptsi_los_from_space_per_pedestrian_uses_fhwa_walkway_thresholds(space_per_pedestrian: float, expected_los: str) -> None:
    assert store._ptsi_los_from_space_per_pedestrian(space_per_pedestrian) == expected_los


@pytest.mark.parametrize(
    ("los", "expected_state"),
    [("A", "clear"), ("B", "clear"), ("C", "clear"), ("D", "moderate"), ("E", "moderate"), ("F", "severe")],
)
def test_ptsi_los_state_collapses_fhwa_bands_into_map_colors(los: str, expected_state: str) -> None:
    assert store._ptsi_los_state(los, has_footage=True, has_occlusion_data=True) == expected_state


@pytest.mark.parametrize(
    ("score", "expected_los"),
    [
        (14.99, "A"),
        (15.0, "B"),
        (32.99, "B"),
        (33.0, "C"),
        (49.99, "C"),
        (50.0, "D"),
        (65.7, "D"),
        (66.0, "E"),
        (84.99, "E"),
        (85.0, "F"),
    ],
)
def test_ptsi_los_from_score_uses_provisional_score_bands(score: float, expected_los: str) -> None:
    assert store._ptsi_los_from_score(score) == expected_los


def test_dashboard_occlusion_emits_ptsi_debug_logs_when_enabled(monkeypatch, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    configure_temp_storage(monkeypatch, tmp_path)
    monkeypatch.setenv("PTSI_DEBUG", "1")

    state = store.seed_state()
    for location in state["locations"]:
        if location["id"] == "edsa-sec-walk":
            location["roiCoordinates"] = {
                "referenceSize": [1920, 1080],
                "includePolygonsNorm": [[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]],
            }
            location["walkableAreaM2"] = None

    state["videos"] = [
        {
            "id": "video-ptsi-debug",
            "locationId": "edsa-sec-walk",
            "location": "EDSA Sec Walk",
            "timestamp": "10:00",
            "date": "2026-03-17",
            "startTime": "10:00",
            "endTime": "10:10",
            "gpsLat": 14.6397,
            "gpsLng": 121.0775,
            "pedestrianCount": 2,
            "rawPath": None,
            "processedPath": None,
        }
    ]
    state["pedestrianTracks"] = [
        {
            "id": "track-debug-a",
            "videoId": "video-ptsi-debug",
            "pedestrianId": 1,
            "location": "EDSA Sec Walk",
            "trajectorySamples": [[0, 0.25, 0.25, 0]],
        },
        {
            "id": "track-debug-b",
            "videoId": "video-ptsi-debug",
            "pedestrianId": 2,
            "location": "EDSA Sec Walk",
            "trajectorySamples": [[0, 0.35, 0.35, 2]],
        },
    ]
    store.save_state(state)

    with caplog.at_level(logging.INFO, logger="backend.app.store"):
        with TestClient(main.app) as client:
            response = client.get("/api/dashboard/occlusion", params={"date": "2026-03-17", "timeRange": "whole-day"})

    assert response.status_code == 200
    log_messages = [record.getMessage() for record in caplog.records if record.name == "backend.app.store"]
    assert any('PTSI_DEBUG {"capacityProxy": 24.0, "event": "location_config"' in message for message in log_messages)
    assert any('"event": "second_score"' in message and '"visibleCount": 2' in message for message in log_messages)
    assert any('"event": "hour_rollup"' in message and '"p90Score"' in message for message in log_messages)
    assert any('"event": "location_summary"' in message and '"peakHour": "10:00"' in message for message in log_messages)


def test_dashboard_traffic_uses_full_track_totals_instead_of_truncated_events(monkeypatch, tmp_path: Path) -> None:
    configure_temp_storage(monkeypatch, tmp_path)

    state = store.seed_state()
    state["videos"] = [
        {
            "id": "video-kostka-1",
            "locationId": "kostka-walk",
            "location": "Kostka Walk",
            "timestamp": "10:00",
            "date": "2026-03-17",
            "startTime": "10:00",
            "endTime": "10:10",
            "gpsLat": 14.6390,
            "gpsLng": 121.0781,
            "pedestrianCount": 5,
            "rawPath": None,
            "processedPath": None,
        },
        {
            "id": "video-kostka-2",
            "locationId": "kostka-walk",
            "location": "Kostka Walk",
            "timestamp": "10:10",
            "date": "2026-03-17",
            "startTime": "10:10",
            "endTime": "10:20",
            "gpsLat": 14.6390,
            "gpsLng": 121.0781,
            "pedestrianCount": 3,
            "rawPath": None,
            "processedPath": None,
        },
    ]
    state["events"] = [
        {
            "id": "evt-kostka-1",
            "type": "detection",
            "location": "Kostka Walk",
            "timestamp": "10:00:00 AM",
            "description": "Pedestrian ID #1 detected at frame 1",
            "videoId": "video-kostka-1",
            "pedestrianId": 1,
            "frame": 1,
            "offsetSeconds": 0.0,
        },
        {
            "id": "evt-kostka-2",
            "type": "detection",
            "location": "Kostka Walk",
            "timestamp": "10:02:00 AM",
            "description": "Pedestrian ID #2 detected at frame 20",
            "videoId": "video-kostka-1",
            "pedestrianId": 2,
            "frame": 20,
            "offsetSeconds": 120.0,
        },
        {
            "id": "evt-kostka-3",
            "type": "detection",
            "location": "Kostka Walk",
            "timestamp": "10:10:00 AM",
            "description": "Pedestrian ID #1 detected at frame 1",
            "videoId": "video-kostka-2",
            "pedestrianId": 1,
            "frame": 1,
            "offsetSeconds": 0.0,
        },
    ]
    state["pedestrianTracks"] = [
        {
            "id": f"track-kostka-1-{pedestrian_id}",
            "videoId": "video-kostka-1",
            "pedestrianId": pedestrian_id,
            "location": "Kostka Walk",
            "firstTimestamp": timestamp,
            "bestTimestamp": timestamp,
            "firstOffsetSeconds": float(offset_seconds),
            "bestOffsetSeconds": float(offset_seconds),
        }
        for pedestrian_id, timestamp, offset_seconds in [
            (1, "10:00:00 AM", 0),
            (2, "10:00:10 AM", 10),
            (3, "10:00:20 AM", 20),
            (4, "10:00:30 AM", 30),
            (5, "10:00:40 AM", 40),
        ]
    ] + [
        {
            "id": f"track-kostka-2-{pedestrian_id}",
            "videoId": "video-kostka-2",
            "pedestrianId": pedestrian_id,
            "location": "Kostka Walk",
            "firstTimestamp": timestamp,
            "bestTimestamp": timestamp,
            "firstOffsetSeconds": float(offset_seconds),
            "bestOffsetSeconds": float(offset_seconds),
        }
        for pedestrian_id, timestamp, offset_seconds in [
            (1, "10:10:00 AM", 0),
            (2, "10:10:10 AM", 10),
            (3, "10:10:20 AM", 20),
        ]
    ]
    store.save_state(state)

    with TestClient(main.app) as client:
        summary_response = client.get("/api/dashboard/summary", params={"date": "2026-03-17"})
        traffic_response = client.get("/api/dashboard/traffic", params={"date": "2026-03-17", "timeRange": "whole-day"})

    assert summary_response.status_code == 200
    assert traffic_response.status_code == 200

    summary = summary_response.json()
    traffic = traffic_response.json()
    whole_day_bucket = next(point for point in traffic["series"] if point["time"] == "10:00")

    assert summary["totalUniquePedestrians"] == 8
    assert traffic["locationTotals"] == [{"location": "Kostka Walk", "totalPedestrians": 8}]
    assert whole_day_bucket["cumulativeUniquePedestrians"] == 8
    assert whole_day_bucket["Kostka Walk"] == 8
    assert whole_day_bucket["averageVisiblePedestrians"] == 1.0


def test_dashboard_traffic_returns_per_location_cumulative_series(monkeypatch, tmp_path: Path) -> None:
    configure_temp_storage(monkeypatch, tmp_path)

    state = store.seed_state()
    state["videos"] = [
        {
            "id": "video-edsa-1",
            "locationId": "edsa-sec-walk",
            "location": "EDSA Sec Walk",
            "timestamp": "10:00",
            "date": "2026-03-17",
            "startTime": "10:00",
            "endTime": "10:10",
            "gpsLat": 14.6397,
            "gpsLng": 121.0775,
            "pedestrianCount": 2,
            "rawPath": None,
            "processedPath": None,
        },
        {
            "id": "video-kostka-1",
            "locationId": "kostka-walk",
            "location": "Kostka Walk",
            "timestamp": "10:20",
            "date": "2026-03-17",
            "startTime": "10:20",
            "endTime": "10:30",
            "gpsLat": 14.6390,
            "gpsLng": 121.0781,
            "pedestrianCount": 1,
            "rawPath": None,
            "processedPath": None,
        },
    ]
    state["events"] = [
        {
            "id": "event-edsa-1",
            "type": "detection",
            "location": "EDSA Sec Walk",
            "timestamp": "10:01:00",
            "description": "Pedestrian ID #1 detected at frame 10",
            "videoId": "video-edsa-1",
            "pedestrianId": 1,
            "frame": 10,
            "offsetSeconds": 1.0,
        },
        {
            "id": "event-edsa-2",
            "type": "detection",
            "location": "EDSA Sec Walk",
            "timestamp": "10:05:00",
            "description": "Pedestrian ID #2 detected at frame 20",
            "videoId": "video-edsa-1",
            "pedestrianId": 2,
            "frame": 20,
            "offsetSeconds": 5.0,
        },
        {
            "id": "event-kostka-1",
            "type": "detection",
            "location": "Kostka Walk",
            "timestamp": "10:21:00",
            "description": "Pedestrian ID #1 detected at frame 15",
            "videoId": "video-kostka-1",
            "pedestrianId": 1,
            "frame": 15,
            "offsetSeconds": 1.0,
        },
    ]
    store.save_state(state)

    with TestClient(main.app) as client:
        traffic_response = client.get("/api/dashboard/traffic", params={"date": "2026-03-17", "timeRange": "whole-day"})
        drilldown_response = client.get(
            "/api/dashboard/traffic",
            params={"date": "2026-03-17", "timeRange": "whole-day", "focusTime": "10:00", "zoomLevel": 1},
        )

    assert traffic_response.status_code == 200
    assert drilldown_response.status_code == 200

    traffic = traffic_response.json()
    assert all("id" in point for point in traffic["series"])
    assert traffic["locationTotals"] == [
        {"location": "EDSA Sec Walk", "totalPedestrians": 2},
        {"location": "Kostka Walk", "totalPedestrians": 1},
    ]

    whole_day_bucket = next(point for point in traffic["series"] if point["time"] == "10:00")
    assert whole_day_bucket["cumulativeUniquePedestrians"] == 3
    assert whole_day_bucket["EDSA Sec Walk"] == 2
    assert whole_day_bucket["Kostka Walk"] == 1

    drilldown = drilldown_response.json()
    ten_oclock_bucket = next(point for point in drilldown["series"] if point["time"] == "10:00")
    ten_fifteen_bucket = next(point for point in drilldown["series"] if point["time"] == "10:15")

    assert ten_oclock_bucket["cumulativeUniquePedestrians"] == 2
    assert ten_oclock_bucket["EDSA Sec Walk"] == 2
    assert ten_oclock_bucket["Kostka Walk"] == 0
    assert ten_fifteen_bucket["cumulativeUniquePedestrians"] == 3
    assert ten_fifteen_bucket["EDSA Sec Walk"] == 2
    assert ten_fifteen_bucket["Kostka Walk"] == 1