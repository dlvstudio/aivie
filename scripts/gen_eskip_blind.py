#!/usr/bin/env python
"""PHA 1 cổng G-emb(b) — E_skip làm kênh NEO-XA (`ar_distant_mode="e"`) có hơn E_old không?

E vào đường sinh AR CHỈ qua kênh `distant_content` khi `ar_distant_mode="e"` (β 0.4). Ba cấu hình,
đường AR (ar_no_agenda), chấm MÙ:
  A ppmi   — mode mặc định, E KHÔNG dùng (mốc)
  B e_old  — mode="e", embedding gốc
  C e_skip — mode="e", embedding W_skip (§12.114 Pha 1)
"""
from __future__ import annotations

import argparse
import gc
import json
import random
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from nn_aivie.core.checkpoint_io import load_checkpoint  # noqa: E402
from nn_aivie.hca.config import HCAConfig  # noqa: E402
from nn_aivie.hca.embedding_skip import build_eskip  # noqa: E402
from nn_aivie.hca.generate_hebbian import generate  # noqa: E402
from nn_aivie.hca.readout import decode_surface  # noqa: E402

from gen_samples_blind import PROMPTS  # noqa: E402

BASE = dict(ar_matrix_core=True, ar_style_payoff=False, ar_matrix_stack_weight=0.0,
            ar_no_agenda=True, ar_sent_agenda_handoff=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--max-len", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    t0 = time.time()
    gc.disable()
    state, _ = load_checkpoint(str(_ROOT / args.checkpoint))
    gc.enable()
    E_old = state.embedding
    E_skip = build_eskip(state, dim=96, d_max=8, lam=0.85)
    print(f"[{time.time()-t0:.0f}s] E_old {E_old.matrix.shape} · E_skip {E_skip.matrix.shape}",
          flush=True)

    configs = [
        ("A_ppmi", dict(ar_distant_mode="ppmi"), E_old),
        ("B_e_old", dict(ar_distant_mode="e"), E_old),
        ("C_e_skip", dict(ar_distant_mode="e"), E_skip),
    ]
    rows = []
    for name, over, emb in configs:
        state.embedding = emb                         # thả embedding tương ứng vào state
        cfg = HCAConfig(**{**BASE, **over})
        for p in PROMPTS:
            out = generate(state, p, max_len=args.max_len, config=cfg, seed=args.seed)
            pids = list(state.vocab.encode_prompt(p))
            cont = decode_surface(state, out[len(pids):]).strip()
            rows.append((name, p, cont))
        print(f"[{time.time()-t0:.0f}s] xong {name}", flush=True)

    rng = random.Random(20260730)
    ids = [f"E{n:03d}" for n in range(len(rows))]
    rng.shuffle(ids)
    key, blind = {}, []
    for sid, (name, p, cont) in zip(ids, rows):
        key[sid] = {"config": name, "prompt": p}
        blind.append((sid, p, cont))
    blind.sort(key=lambda r: r[0])

    outdir = _ROOT / "out"
    with open(outdir / "eskip_samples.txt", "w") as f:
        f.write("# CHẤM MÙ G-emb(b) — không lộ cấu hình.\n\n")
        for sid, p, cont in blind:
            f.write(f"[{sid}]  PROMPT: {p}\n        SINH  : {cont}\n\n")
    with open(outdir / "eskip_key.json", "w") as f:
        json.dump(key, f, ensure_ascii=False, indent=1)
    print(f"[{time.time()-t0:.0f}s] {len(rows)} mẫu → out/eskip_samples.txt", flush=True)


if __name__ == "__main__":
    main()
