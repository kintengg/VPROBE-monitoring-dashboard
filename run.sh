#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Ensure dependencies are installed
if [ ! -d "node_modules" ]; then
    echo "node_modules/ not found. Run ./setup.sh first."
    exit 1
fi

if [ ! -d "venv" ]; then
    echo "venv/ not found. Run ./setup.sh first."
    exit 1
fi

# Activate virtual environment
source venv/bin/activate

echo "=========================================="
echo " VPROBE — Starting Services"
echo "=========================================="
echo ""
echo " Frontend  → http://localhost:3000"
echo " Backend   → http://localhost:8000"
echo " API Docs  → http://localhost:8000/docs"
echo ""
echo " Press Ctrl+C to stop all services."
echo "=========================================="
echo ""

# Trap to kill both processes on exit
cleanup() {
    echo ""
    echo "Shutting down..."
    kill $FRONTEND_PID $BACKEND_PID 2>/dev/null || true
    wait $FRONTEND_PID $BACKEND_PID 2>/dev/null || true
    echo "All services stopped."
}
trap cleanup EXIT INT TERM

# Start backend
uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!

# Start frontend
pnpm dev &
FRONTEND_PID=$!

# Wait for both background processes (Ctrl+C triggers trap)
wait
