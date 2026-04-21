from backend.app import store

VIDEO_ID = "8e6835a6"
state = store.load_state()
video = next(video for video in state["videos"] if video.get("id") == VIDEO_ID)
location = next(location for location in state["locations"] if location.get("id") == video.get("locationId"))
config = store._normalized_entry_exit_points(location)

tracks = [track for track in state.get("pedestrianTracks", []) if track.get("videoId") == VIDEO_ID]
examples = []

for index, track in enumerate(tracks):
    samples = store._interpolated_trajectory_points(track)
    if not samples:
        continue

    visits = []
    strip_points = []
    for offset, point in samples:
        zone = store._directional_zone_for_point(point, config)
        if zone is not None:
            strip_points.append((float(offset), point, zone))
            if not visits or visits[-1] != zone:
                visits.append(zone)

    if len(visits) != 1 or not strip_points:
        continue

    xs = [point[0] for _offset, point, _zone in strip_points]
    ys = [point[1] for _offset, point, _zone in strip_points]
    full_xs = [point[0] for _offset, point in samples]
    full_ys = [point[1] for _offset, point in samples]
    examples.append(
        {
            "trackId": str(track.get("id") or f"fallback-{index}"),
            "pedestrianId": track.get("pedestrianId"),
            "zone": visits[0],
            "stripFirstOffset": round(strip_points[0][0], 3),
            "stripLastOffset": round(strip_points[-1][0], 3),
            "stripDuration": round(strip_points[-1][0] - strip_points[0][0], 3),
            "stripXRange": [round(min(xs), 3), round(max(xs), 3)],
            "stripYRange": [round(min(ys), 3), round(max(ys), 3)],
            "fullXRange": [round(min(full_xs), 3), round(max(full_xs), 3)],
            "fullYRange": [round(min(full_ys), 3), round(max(full_ys), 3)],
        }
    )

examples.sort(key=lambda item: (item["zone"], -item["stripDuration"], item["trackId"]))
print("single_strip_examples", examples)
