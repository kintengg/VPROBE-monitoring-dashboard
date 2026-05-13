#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=========================================="
echo " VPROBE — Project Setup"
echo "=========================================="

# ── Frontend dependencies ──────────────────
echo ""
echo "[1/3] Installing frontend dependencies (pnpm)..."
pnpm install

# ── Python virtual environment ─────────────
echo ""
echo "[2/3] Setting up Python virtual environment..."

if [ -d "venv" ]; then
    echo "  venv already exists, skipping creation."
else
    python3 -m venv venv
    echo "  Created venv/"
fi

# ── Python dependencies ────────────────────
echo ""
echo "[3/3] Installing Python dependencies..."
source venv/bin/activate
pip install --upgrade pip -q
pip install -r requirements.txt

echo ""
echo "=========================================="
echo " Setup complete!"
echo ""
echo " Start the app:"
echo "   ./run.sh"
echo ""
echo " Or manually:"
echo "   source venv/bin/activate"
echo "   pnpm dev          # Frontend → http://localhost:3000"
echo "   uvicorn backend.app.main:app --reload  # Backend → http://localhost:8000"
echo "=========================================="
