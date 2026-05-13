# VPROBE — AI-Powered Video Surveillance Dashboard

A real-time video management and analytics system for pedestrian and vehicle monitoring, built on Next.js and FastAPI with deep learning inference via Ultralytics models.

## Features

- **Unified Dashboard** — Real-time level-of-service (LOS) map, traffic counts, occlusion tracking, and AI-generated event summaries
- **Vehicle Monitoring** — Gate-level vehicle counts, speed estimation, and per-location analytics
- **Pedestrian Surveillance** — Multi-camera pedestrian tracking, queue detection, and flow analysis
- **Video Management** — Upload, playback, and search across surveillance footage with thumbnail generation
- **AI-Powered Search** — Semantic video search using natural language queries
- **Model Management** — Upload and switch between object detection models at runtime

## Tech Stack

| Layer    | Technology                                      |
| -------- | ----------------------------------------------- |
| Frontend | [Next.js 16](https://nextjs.org), React 19, Tailwind CSS 4, shadcn/ui, Leaflet |
| Backend  | [FastAPI](https://fastapi.tiangolo.com), Uvicorn, Python 3 |
| ML/AI    | Ultralytics RT-DETR, ByteTrack, YOLO |
| Storage  | Local filesystem (`backend/storage/`)           |

## Project Structure

```
├── app/                  # Next.js frontend (App Router)
│   ├── pedestrian/       # Pedestrian surveillance pages
│   ├── vehicle/          # Vehicle monitoring pages
│   ├── queue/            # Queue analysis pages
│   ├── video/            # Video management pages
│   ├── search/           # Semantic search pages
│   └── models/           # Model management pages
├── components/           # Shared React components (shadcn/ui)
├── lib/                  # API client, hooks, utilities
├── hooks/                # Custom React hooks
├── backend/              # FastAPI backend
│   └── app/
│       ├── main.py       # API routes & application entry point
│       ├── inference.py  # Pedestrian detection inference
│       ├── vehicle_inference.py  # Vehicle detection inference
│       ├── store.py      # Pedestrian data store
│       ├── vehicle_store.py      # Vehicle data store
│       ├── gemini.py     # AI synthesis integration
│       └── schemas.py    # Pydantic models
├── requirements.txt      # Python dependencies
├── package.json          # Node.js dependencies (pnpm)
└── tmp/                  # Temporary/example scripts
```

## Getting Started

### Prerequisites

- **Node.js** ≥ 18
- **pnpm** (install via `npm install -g pnpm`)
- **Python** ≥ 3.10

### Quick Setup

```bash
chmod +x setup.sh run.sh
./setup.sh
```

This installs frontend dependencies, creates a Python virtual environment, and installs all Python packages.

### Manual Setup

```bash
# Frontend
pnpm install

# Backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Configuration

Create a `.env.local` file in the project root (or in `backend/`) with the required environment variables:

```
GOOGLE_PLACES_API_KEY=your_google_places_api_key
GEMINI_API_KEY=your_gemini_api_key      # Optional, for AI synthesis
```

### Running

```bash
./run.sh
```

This starts both services:
- **Frontend** — [http://localhost:3000](http://localhost:3000)
- **Backend** — [http://localhost:8000](http://localhost:8000)

To run manually:

```bash
# Terminal 1: Frontend
pnpm dev

# Terminal 2: Backend
source venv/bin/activate
uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000
```

### API Documentation

Once the backend is running, interactive API docs are available at:

- Swagger UI — [http://localhost:8000/docs](http://localhost:8000/docs)
- Health check — [http://localhost:8000/health](http://localhost:8000/health)

## Storage Layout

```
backend/storage/
├── models/              # YOLO/RT-DETR .pt weights
├── videos/
│   ├── raw/             # Uploaded source footage
│   └── processed/       # Rendered detections / tracked exports
└── exports/             # Reports / downloadable artifacts
```
