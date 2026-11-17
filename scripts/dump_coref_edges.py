#!/usr/bin/env python3
"""
TK_044 P3.1 — CỔNG: dump cạnh coreference đã học, đọc tay (cho M4).

    ⚠️ CHẠY M4. PYTHONPATH=src python -u scripts/dump_coref_edges.py \
        --checkpoint checkpoints/network_v11_coref.pkl --top 30 --log-file out/coref_edges_P31.log

CỔNG P3.1 (RUNBOOK §5): đọc tay ~10 cạnh single — SIG-nhắc-lại → SIG-giới-thiệu có HỢP LÝ không?
  vd «`.`/`,` [ENT] `là/của`…» (nhắc lại) → «đầu-đoạn [ENT]» (giới thiệu).
Hợp lý → P3.2 (kênh copy). Rỗng/loạn → xem occurrence-detect / SIG-width / purity (RUNBOOK §6).
"""

from __future__ import annotations

import argparse
import gc
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from nn_aivie.core.checkpoint_io import load_checkpoint  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--checkpoint", default="checkpoints/network_v11_coref.pkl")
    ap.add_argument("--top", type=int, default=30)
    ap.add_argument("--log-file", default="out/coref_edges_P31.log")
    args = ap.parse_args()

    logf = open(_ROOT / args.log_file, "w")
    t0 = time.time()

    def log(m=""):
        print(m, flush=True); logf.write(str(m) + "\n"); logf.flush()

    log(f"[{time.time()-t0:.0f}s] load {args.checkpoint} (gc.disable) …")
    gc.disable()
    state, _ = load_checkpoint(str(_ROOT / args.checkpoint))
    gc.enable()
    vocab = state.vocab
    sur = vocab.token_of
    coref = getattr(state, "coref", {}) or {}
    after_role = getattr(state, "after_role", {}) or {}
    log(f"[{time.time()-t0:.0f}s] ✓ loaded · coref edges {len(coref):,} · after_role {len(after_role):,}")
    if not coref:
        log("⚠ RỖNG — pass chưa persist (state.save/load thiếu coref? RUNBOOK §4) hoặc occurrence=0.")
        return 1

    def s(ctx):
        return " ".join("<ENT>" if t == -1 else (sur(t) or f"?{t}") for t in ctx) or "∅"

    def show_sig(sig):
        left, right = sig
        return f"«{s(left)} [ENT] {s(right)}»"

    log(f"\n═══ TOP {args.top} cạnh coref (SIG-nhắc-lại → SIG-antecedent) — đọc tay khuôn ═══")
    for i, (a, b) in enumerate(list(coref.items())[: args.top]):
        ar = after_role.get(a)
        ars = f" · after_role={sur(ar)!r}" if ar is not None else ""
        log(f"\n  [{i}] {show_sig(a)}")
        log(f"      → {show_sig(b)}{ars}")
    log(f"\n[{time.time()-t0:.0f}s] → CỔNG: ≥ ~7/10 cạnh khuôn hợp lý (nhắc-lại→giới-thiệu) ⇒ P3.2. "
        "Loạn/rỗng ⇒ xem occurrence-detect / SIG-width / purity (RUNBOOK §6).")
    logf.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
