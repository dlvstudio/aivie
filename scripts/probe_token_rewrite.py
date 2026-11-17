#!/usr/bin/env python3
"""
PROBE N→N REWRITE — mỗi token prompt → top liên quan (thí nghiệm FFN-ý) (cho M4).

    PYTHONPATH=src python -u scripts/probe_token_rewrite.py \
        --checkpoint checkpoints/network_v11.pkl --log-file out/probe_token_rewrite.log

## Ý (2026-07-23)

LLM: N vị trí → N biểu diễn đã pha tri thức. Hebbian thử: mỗi t_i → top-k u_i
(CO_ACT W×IDF / PPMI·C), vẫn N slot. Đọc tay:

  * Entity → fact-anchor hợp lý? HOW («hài hước») → sắc-thái?
  * Hư-từ / dấu → KEEP (không rewrite)?
  * Chuỗi u_1..u_N đọc được hơn / đúng nghĩa hơn t không?

Dương → wire residual t⊕u vào A3. Âm (hub/salad) → giữ spreading có hướng / multizone.
"""

from __future__ import annotations

import argparse
import gc
import math
import string
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from nn_aivie.core.checkpoint_io import load_checkpoint  # noqa: E402
from nn_aivie.core.edge_kinds import EdgeKind  # noqa: E402

PROMPTS = [
    "So sánh Nhà Lý và Nhà Trần",
    "Trần Hưng Đạo và trận Bạch Đằng",
    "Viết một đoạn văn hài hước về Ngô Quyền",
    "Kể buồn về mùa thu Hà Nội",
]

_PUNCT = set(string.punctuation + "…“”‘’—–·•")


def _core(tok: str) -> str:
    if tok.startswith("Ġ") or tok.startswith("Ċ"):
        tok = tok[1:]
    return tok


def _artifact(tok: str) -> bool:
    c = _core(tok)
    if not c or c.endswith(":"):
        return True
    return all(ch in _PUNCT or ch.isspace() for ch in c)


def _row_sum(co, i: int) -> float:
    return float(sum(co.row(i).values()))


def _neighbors(
    co, tid: int, *, sur, gidf, gidf_min: float, c_min: float, top: int, max_cand: int,
) -> list[tuple[int, float, float]]:
    """Trả [(j, w_idf, ppmi_c), ...] đã lọc artifact / self."""
    row = co.row(tid)
    if not row:
        return []
    seed_core = _core(sur(tid)).lower()
    items = sorted(row.items(), key=lambda x: -x[1])[:max_cand]
    c_seed = _row_sum(co, tid) or 1.0
    strength = {j: _row_sum(co, j) for j, _ in items}
    n_ego = c_seed + sum(strength.values())
    scored: list[tuple[float, float, int]] = []
    for j, c_ij in items:
        if j == tid or c_ij < c_min:
            continue
        tok = sur(j)
        if _artifact(tok) or _core(tok).lower() == seed_core:
            continue
        if tok.startswith("<"):
            continue
        g = gidf(j)
        if g < gidf_min:
            continue
        w_idf = float(c_ij) * g
        c_j = strength.get(j, 0.0)
        if c_j <= 0 or n_ego <= 0:
            continue
        pmi = math.log((c_ij * n_ego) / (c_seed * c_j) + 1e-12)
        if pmi <= 0:
            continue
        scored.append((pmi * math.log(1.0 + c_ij), w_idf, j))
    scored.sort(reverse=True)
    return [(j, w_idf, pc) for pc, w_idf, j in scored[:top]]


def _role(tid: int, vocab, ps, *, gidf_min: float) -> str:
    tok = vocab.token_of(tid)
    if not vocab.is_surface(tid) or _artifact(tok):
        return "KEEP"
    if vocab.cap_class(tid) >= 1:
        return "ENT"
    g = ps._gidf(tid)  # noqa: SLF001
    if g < gidf_min:
        return "KEEP"
    return "HOW" if g >= 4.5 else "CTX"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--checkpoint", default="checkpoints/network_v11.pkl")
    ap.add_argument("--top", type=int, default=3)
    ap.add_argument("--gidf-min", type=float, default=2.5)
    ap.add_argument("--c-min", type=float, default=2.0)
    ap.add_argument("--max-cand", type=int, default=800)
    ap.add_argument("--log-file", default="out/probe_token_rewrite.log")
    args = ap.parse_args()

    logf = open(_ROOT / args.log_file, "w")
    t0 = time.time()

    def log(m=""):
        print(m, flush=True)
        logf.write(str(m) + "\n")
        logf.flush()

    log(f"[{time.time()-t0:.0f}s] load {args.checkpoint} …")
    gc.disable()
    state, _ = load_checkpoint(str(_ROOT / args.checkpoint))
    gc.enable()
    vocab, ps = state.vocab, state.parents
    sur = vocab.token_of
    gidf = ps._gidf  # noqa: SLF001
    co = state.adjacency.layer(EdgeKind.CO_ACTIVATION)
    log(f"[{time.time()-t0:.0f}s] ✓ parents {len(ps):,}")
    log("    role: KEEP=giữ · ENT=viết-hoa · HOW=gidf cao thường · CTX=thường")
    log("    u* = top-1 PPMI·C (KEEP → chính t_i)\n")

    for prompt in PROMPTS:
        ids = list(vocab.encode_prompt(prompt))
        log(f"═══ {prompt!r}  (N={len(ids)}) ═══")
        rewritten: list[str] = []
        for i, tid in enumerate(ids):
            role = _role(tid, vocab, ps, gidf_min=args.gidf_min)
            t_sur = sur(tid)
            if role == "KEEP":
                rewritten.append(t_sur)
                log(f"  [{i:02d}] {role:4} {t_sur!r:16} → KEEP")
                continue
            neigh = _neighbors(
                co, tid, sur=sur, gidf=gidf,
                gidf_min=args.gidf_min, c_min=args.c_min,
                top=args.top, max_cand=args.max_cand,
            )
            if not neigh:
                rewritten.append(t_sur)
                log(f"  [{i:02d}] {role:4} {t_sur!r:16} → ∅ (giữ)")
                continue
            parts = [f"{sur(j)!r}({pc:.2f})" for j, _w, pc in neigh]
            u0 = sur(neigh[0][0])
            rewritten.append(u0)
            log(f"  [{i:02d}] {role:4} {t_sur!r:16} → {'  '.join(parts)}")

        # detokenize thô: nối surface (đủ đọc tay)
        def _join(toks: list[str]) -> str:
            out = []
            for t in toks:
                if t.startswith("Ġ"):
                    out.append(" " + t[1:])
                elif t.startswith("Ċ"):
                    out.append("\n" + t[1:])
                else:
                    out.append(t)
            return "".join(out).strip()

        log(f"  t : {_join([sur(t) for t in ids])!r}")
        log(f"  u*: {_join(rewritten)!r}")
        log()

    log(f"[{time.time()-t0:.0f}s] → đọc tay: u* có nghĩa hơn t? ENT→đúng entity? HOW→sắc-thái? "
        "Hub/salad → không wire rewrite cứng.")
    logf.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
