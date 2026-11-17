#!/bin/bash
cd "$(dirname "$0")/.."
echo "Starting AIVIE Graph UI at http://127.0.0.1:8765"
exec python3 scripts/run_viz.py
