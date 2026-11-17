#!/usr/bin/env python
"""TK_051 ship — CLI ghép ≥2 cha (defaults = P203).

Xem RUNBOOK_COMPOSE_P203_SHIP.md. Nhãn: dual retrieve + bản lề W_skip
(không quảng cáo sinh sáng tạo tự do).

  PYTHONPATH=src .venv/bin/python scripts/gen_compose_2cha.py \\
    --checkpoint checkpoints/network_v12_subst.pkl \\
    --prompt "Trong các đợt tiến công"
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from nn_aivie.core.checkpoint_io import load_checkpoint  # noqa: E402
from nn_aivie.hca.compose_2cha import ComposeConfig, generate_compose_2cha  # noqa: E402
from nn_aivie.hca.readout import decode_surface  # noqa: E402
from nn_aivie.util.progress import get_run_logger  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="Sinh compose_2cha (P203 defaults)")
    ap.add_argument("--checkpoint", default="checkpoints/network_v12_subst.pkl")
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--label-map", default="out/token_labels_v12.json")
    ap.add_argument("--ws-model", default="out/models/ws_crf_uts_full")
    ap.add_argument("--join-d-max", type=int, default=8)
    ap.add_argument("--join-min-count", type=float, default=1.0)
    ap.add_argument("--top-n", type=int, default=24)
    ap.add_argument("--max-thay", type=int, default=2)
    ap.add_argument("--min-gain", type=int, default=1)
    ap.add_argument("--slot-rank", choices=("bam", "ncand"), default="bam")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    logger = get_run_logger("gen_compose_2cha")
    cfg = ComposeConfig(
        join_d_max=args.join_d_max,
        join_min_count=args.join_min_count,
        top_n=args.top_n,
        max_thay=args.max_thay,
        min_gain=args.min_gain,
        slot_rank=args.slot_rank,
        seed=args.seed,
        label_map=args.label_map,
        ws_model=args.ws_model,
        no_co_graph=True,
    )

    t0 = time.time()
    import gc
    gc.disable()
    state, _ = load_checkpoint(str(_ROOT / args.checkpoint))
    gc.enable()
    logger.log(f"loaded ({time.time()-t0:.0f}s)")

    ids, meta = generate_compose_2cha(
        state, args.prompt, cfg=cfg, root=_ROOT)
    print(decode_surface(state, ids).strip())
    print("# meta:", json.dumps(meta, ensure_ascii=False), file=sys.stderr)


if __name__ == "__main__":
    main()
