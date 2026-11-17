#!/usr/bin/env python3
"""
TK_044 P3.1 — DRIVER: chạy `run_coref_pass` trên graph + lưu lại checkpoint (cho M4).

    ⚠️ CHẠY M4. PYTHONPATH=src python -u scripts/run_coref_pass.py \
        --checkpoint checkpoints/network_v11.pkl \
        --out checkpoints/network_v11_coref.pkl --log-file out/coref_pass.log

TIỀN ĐỀ (M4 làm trước — RUNBOOK_TK_044_P3.1.md §1/§4):
  * GraphState có field `coref` + `after_role` (state.py) và SERIALIZE trong save/load.
  Nếu chưa, pass vẫn chạy + set attr, nhưng `state.save` sẽ KHÔNG persist → dump sau ra rỗng.

Quét narrative có tín hiệu (out/coref_signal_scan.txt): viwiki + history + conv. Record = string thô.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from nn_aivie.core.checkpoint_io import load_checkpoint  # noqa: E402
from nn_aivie.training.coref_pass import run_coref_pass  # noqa: E402

CORPUS = ["viwiki", "history_articles", "conv_stories"]   # nguồn coref cao (bỏ paragraph/dict/qa)


def records(names, limit_each):
    for name in names:
        p = _ROOT / "data" / "text_corpus" / f"{name}.jsonl"
        if not p.exists():
            continue
        with open(p) as f:
            for i, line in enumerate(f):
                if limit_each and i >= limit_each:
                    break
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                yield rec if isinstance(rec, str) else str(rec.get("text", rec))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--checkpoint", default="checkpoints/network_v11.pkl")
    ap.add_argument("--out", default="checkpoints/network_v11_coref.pkl")
    ap.add_argument("--limit-each", type=int, default=0, help="0 = toàn bộ; >0 = giới hạn/nguồn (thử nhanh)")
    ap.add_argument("--gidf-min", type=float, default=3.0)
    ap.add_argument("--gidf-solo", type=float, default=6.0, help="ngưỡng gidf cho single-token entity")
    ap.add_argument("--log-file", default="out/coref_pass.log")
    args = ap.parse_args()

    logf = open(_ROOT / args.log_file, "w")
    t0 = time.time()

    def log(m=""):
        line = f"[{time.time()-t0:6.0f}s] {m}"
        print(line, flush=True); logf.write(line + "\n"); logf.flush()

    log(f"load {args.checkpoint} (gc.disable) …")
    gc.disable()
    state, _ = load_checkpoint(str(_ROOT / args.checkpoint))
    gc.enable()
    log(f"✓ loaded · parents {len(state.parents):,} · phrases {len(getattr(state.parents,'phrases',[])):,}")

    stats = run_coref_pass(state, records(CORPUS, args.limit_each),
                           gidf_min=args.gidf_min, gidf_solo=args.gidf_solo, log=log)
    log(f"stats: {stats}")

    if not getattr(state, "coref", None):
        log("⚠ state.coref RỖNG — kiểm occurrence-detect (gidf_min? phrases?) TRƯỚC khi lưu.")
    log(f"lưu → {args.out} (cần state.save persist coref/after_role — RUNBOOK §4)")
    state.save(str(_ROOT / args.out))
    log("✓ xong. Kế: scripts/dump_coref_edges.py --checkpoint " + args.out)
    logf.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
