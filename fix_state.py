import json
from pathlib import Path

state_path = Path("backend/storage/dev_data.json")
state = json.loads(state_path.read_text())

# Map location IDs to their domain
locations = state.get("locations", [])
loc_domains = {str(loc["id"]): loc.get("domain", "pedestrian") for loc in locations}

changed = False
for video in state.get("videos", []):
    loc_id = str(video.get("locationId"))
    if loc_domains.get(loc_id) == "vehicle":
        events = [e for e in state.get("events", []) if e.get("videoId") == video["id"] and e.get("type") == "vehicle-detection"]
        if events:
            new_count = len(events)
            if int(video.get("pedestrianCount", 0)) != new_count:
                print(f"Video {video['id']}: updating count from {video.get('pedestrianCount')} to {new_count}")
                video["pedestrianCount"] = new_count
                changed = True

if changed:
    state_path.write_text(json.dumps(state, indent=2))
    print("State updated")
else:
    print("No changes needed")
