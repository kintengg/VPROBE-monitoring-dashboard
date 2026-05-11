# Vendored Ultralytics forks (side-by-side)

The pedestrian and vehicle pipelines need different Ultralytics builds:

| Domain     | Tag                    | Purpose                                              |
| ---------- | ---------------------- | ---------------------------------------------------- |
| pedestrian | `v8.3.228` (preferred), `v8.3.50` (fallback) | YOLOv8 person detection used by `inference.py`. |
| vehicle    | `rtdetr-cli` (preferred), `rtdetr-cli-cpu` (fallback) | RT-DETR with the `Occlusion-Robust-RTDETR` weights. |

## Layout

```
backend/vendor/
├── ultralytics/             # pedestrian fork (YOLOv8) — installed editable
└── ultralytics-rtdetr/      # vehicle fork (rtdetr-cli) — installed editable
```

The inference adapter resolves which fork to import based on the active
domain (looked up via `model_registry.active_weight_path(domain)` and the
domain's `ultralyticsTag` in `registry.json`).

## Installation

```bash
# from repo root
mkdir -p backend/vendor

# pedestrian (existing pip install also works)
git clone --branch v8.3.228 https://github.com/ultralytics/ultralytics.git \
  backend/vendor/ultralytics
pip install -e backend/vendor/ultralytics

# vehicle (RT-DETR fork — substitute the actual fork URL the project uses)
git clone --branch rtdetr-cli <fork-url> backend/vendor/ultralytics-rtdetr
pip install -e backend/vendor/ultralytics-rtdetr
```

Until both forks are vendored, vehicle inference will not run; the API
still serves gate metadata, LOS scaffolding, and the empty-state dashboards.
The Models page will surface the active weight + framework so you can spot
when the vendor folder is missing.

This README is the only file in `backend/vendor/` until you install. It is
checked in so the layout is documented; the actual `ultralytics/` and
`ultralytics-rtdetr/` directories should be kept in `.gitignore`.
