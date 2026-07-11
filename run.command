#!/bin/bash
# memorandum launcher for macOS (double-click in Finder).
# Creates the venv and installs dependencies on first run.
set -u
cd "$(dirname "$0")"

fail() {
    echo ""
    echo "[setup] $1"
    read -r -n 1 -p "Press any key to close..."
    exit 1
}

if [ ! -x "venv/bin/python" ]; then
    echo "[setup] Creating virtual environment..."
    python3 -m venv venv || fail "Failed to create venv. Install Python 3.9+ first."
    echo "[setup] Installing dependencies..."
    venv/bin/pip install -r requirements.txt || fail \
        "Failed to install dependencies. If PyAudio failed, run: brew install portaudio"
fi

venv/bin/python memorandum.py "$@"
echo ""
read -r -n 1 -p "[memorandum] Finished. Press any key to close..."
