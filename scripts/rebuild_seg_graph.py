#!/usr/bin/env python3
"""
TK_013 — replay ISG trên checkpoint có parents (migrate hoặc so sánh với decompose_repair).

  python scripts/rebuild_seg_graph.py --checkpoint checkpoints/grade_v3_big_a2_t3.pkl
  python scripts/rebuild_seg_graph.py --checkpoint ... --log-file out/rebuild_isg.log
  python scripts/rebuild_seg_graph.py --checkpoint ... --legacy-repair  # so sánh Re-Pair
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

from nn_aivie.core.checkpoint_io import load_checkpoint, save_checkpoint
from nn_aivie.data.qa import DAP_LABEL
from nn_aivie.util.progress import FileProgressLog

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser(description="TK_013: rebuild incremental segment graph")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", default=None, help="Ghi checkpoint mới (mặc định: ghi đè input)")
    ap.add_argument("--min-frequency", type=int, default=3)
    ap.add_argument("--log-every", type=int, default=500, help="Log tiến trình mỗi N cha/promote")
    ap.add_argument(
        "--max-promotions",
        type=int,
        default=40000,
        help="Giới hạn số cụm promote (0=không giới hạn, mặc định 40000 như decompose_repair)",
    )
    ap.add_argument(
        "--log-file",
        default=None,
        help="File log tiến trình (mặc định: out/rebuild_isg_<timestamp>.log)",
    )
    ap.add_argument("--legacy-repair", action="store_true")
    ap.add_argument("--no-save", action="store_true")
    args = ap.parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = Path(args.log_file) if args.log_file else ROOT / "out" / f"rebuild_isg_{ts}.log"
    flog = FileProgressLog(log_path)
    flog.log(f"START checkpoint={args.checkpoint} min_freq={args.min_frequency}")

    ckpt = ROOT / args.checkpoint
    t_load = time.time()
    state, cache = load_checkpoint(str(ckpt))
    ps = state.parents
    if ps is None:
        flog.log("ERROR: No ParentStore in checkpoint")
        return 1
    flog.log(f"loaded parents={len(ps.parents)} ({time.time()-t_load:.1f}s)")

    ps.seg_min_frequency = args.min_frequency
    dap_id = state.vocab.id_of(DAP_LABEL)
    t0 = time.time()
    summary: dict = {"checkpoint": str(args.checkpoint), "min_frequency": args.min_frequency}

    if args.legacy_repair:
        flog.log("mode=legacy_repair decompose_repair …")
        res = ps.decompose_repair(state.vocab, min_frequency=args.min_frequency, max_merges=40000)
        flog.log(f"decompose_repair done: {res} ({time.time()-t0:.1f}s) phrases={len(ps.phrases)}")
        summary["mode"] = "legacy_repair"
        summary["result"] = res
    else:
        flog.log("mode=ISG rebuild_seg_graph …")
        res = ps.rebuild_seg_graph(
            state.vocab,
            dap_id=dap_id,
            progress=flog.log,
            log_every=args.log_every,
            max_promotions=args.max_promotions,
        )
        flog.log(f"ISG rebuild done: {res} ({time.time()-t0:.1f}s)")
        summary["mode"] = "isg"
        summary["result"] = res
        odau = None
        for ci, c in enumerate(ps.seg.clusters):
            if state.vocab.token_of(c.left) == "ở" and state.vocab.token_of(c.right) == "đâu":
                odau = {"idx": ci, "power": c.power}
        if odau:
            flog.log(f"cluster ở→đâu: {odau}")
            summary["odau_cluster"] = odau

    ps.ensure_skip_ledger(min_count=4, min_pmi=1.0)
    ps.flush_skip_ledger(state.vocab, logger=flog)

    if not args.no_save:
        out = str(ROOT / (args.out or args.checkpoint))
        flog.log(f"saving checkpoint → {out} …")
        t_save = time.time()
        save_checkpoint(state, out, cache)
        flog.log(f"saved ({time.time()-t_save:.1f}s)")
        summary["out"] = out

    summary["wall_seconds"] = round(time.time() - t0, 1)
    summary["log_file"] = str(log_path)
    summary_path = log_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    flog.log(f"DONE summary → {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
