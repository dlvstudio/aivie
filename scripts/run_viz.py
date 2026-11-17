#!/usr/bin/env python3
"""Build web UI (if needed) and start Graph UI server on :8765."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from nn_aivie.core.paths import viz_default_checkpoint

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "viz" / "web"
DIST = WEB / "dist"


def main() -> int:
    if not DIST.exists():
        print("Building viz/web (npm run build)…")
        if not (WEB / "node_modules").exists():
            subprocess.run(["npm", "install"], cwd=WEB, check=True)
        subprocess.run(["npm", "run", "build"], cwd=WEB, check=True)
    if "AIVIE_CHECKPOINT" not in os.environ:
        os.environ["AIVIE_CHECKPOINT"] = str(viz_default_checkpoint())
    ckpt = os.environ["AIVIE_CHECKPOINT"]
    print(f"Checkpoint: {ckpt}")
    print("Starting server at http://127.0.0.1:8765")
    return subprocess.call([sys.executable, "-m", "viz.server.main"], cwd=ROOT)


if __name__ == "__main__":
    raise SystemExit(main())
