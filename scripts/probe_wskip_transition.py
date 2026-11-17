#!/usr/bin/env python
"""PHA 3+ (§12.114 ②) — TOÁN TỬ CHUYỂN W_skip: đám mây khả-đạt ΣγᵏPᵏ làm tín hiệu chống trôi.

P = W_skip[d=1] content-position, chuẩn hoá hàng = toán tử chuyển Markov trên token-nội-dung.
Đám mây từ prompt: cloud = Σ_{k=1..K} γ^k (v0 @ P^k), v0 = phân bố token-nội-dung prompt. Điểm mẫu
= trung bình cloud-mass trên token-nội-dung phần sinh — "phần sinh có nằm trong quỹ đạo prompt không".

KHÁC PMI-cross (§12.117): P^k KẾT HỢP bước chuyển ⇒ SINH đường mới (khái quát), không chỉ chứng
thực cặp đã thấy. Kỳ vọng: tránh được bẫy ECHO của PMI (lặp từ prompt) vì P ít tự-vòng.

So 3 thước trên cùng nhãn C (bám prompt) chấm mù. Luật 0: đếm + chuẩn hoá + nhân ma trận, closed-form.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import statistics as stx
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from nn_aivie.core.checkpoint_io import load_checkpoint  # noqa: E402
from nn_aivie.hca.context_positional import is_structural_token  # noqa: E402
from nn_aivie.hca.embedding_skip import build_eskip, wskip_cooccur  # noqa: E402
from nn_aivie.hca.sent_intra import _content_positions, iter_sentence_spans  # noqa: E402


def spearman(xs, ys):
    def rank(a):
        o = sorted(range(len(a)), key=lambda i: a[i])
        r = [0.0] * len(a)
        i = 0
        while i < len(a):
            j = i
            while j + 1 < len(a) and a[o[j + 1]] == a[o[i]]:
                j += 1
            for k in range(i, j + 1):
                r[o[k]] = (i + j) / 2.0
            i = j + 1
        return r
    rx, ry = rank(xs), rank(ys)
    n = len(xs)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((rx[i] - mx) * (ry[i] - my) for i in range(n))
    dx = math.sqrt(sum((x - mx) ** 2 for x in rx))
    dy = math.sqrt(sum((y - my) ** 2 for y in ry))
    return num / (dx * dy) if dx > 0 and dy > 0 else 0.0


def build_P(state, struct):
    """Toán tử chuyển content-bigram, chuẩn hoá hàng (row-stochastic)."""
    v = state.vocab
    n = len(v)
    cnt = defaultdict(float)
    for rec in state.parents.parents:
        sq = list(rec.answer_ids if (rec.is_qa and rec.answer_ids) else rec.emit_ids)
        for sp in iter_sentence_spans(sq, v):
            tk = [int(t) for t in _content_positions(sp, v, content_only=True, struct_mask=struct)]
            for a, b in zip(tk, tk[1:]):
                cnt[(a, b)] += 1.0
    r = np.fromiter((a for (a, _b) in cnt), dtype=np.int64, count=len(cnt))
    c = np.fromiter((b for (_a, b) in cnt), dtype=np.int64, count=len(cnt))
    d = np.fromiter(cnt.values(), dtype=np.float64, count=len(cnt))
    M = csr_matrix((d, (r, c)), shape=(n, n))
    rs = np.asarray(M.sum(1)).ravel()
    rs[rs == 0] = 1.0
    from scipy.sparse import diags
    return diags(1.0 / rs) @ M            # P[a,b] = xác suất b theo a (content d=1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--key", default="out/blind_key.json")
    ap.add_argument("--scores", required=True)
    ap.add_argument("--K", type=int, default=3, help="số bước chuyển")
    ap.add_argument("--gamma", type=float, default=0.6)
    ap.add_argument("--d-max", type=int, default=8)
    ap.add_argument("--lam", type=float, default=0.85)
    args = ap.parse_args()

    t0 = time.time()
    gc.disable()
    state, _ = load_checkpoint(str(_ROOT / args.checkpoint))
    gc.enable()
    v = state.vocab
    n = len(v)
    struct = [is_structural_token(v, i) for i in range(n)]

    P = build_P(state, struct)
    print(f"[{time.time()-t0:.0f}s] toán tử chuyển P: {P.nnz:,} cạnh", flush=True)

    # PMI-cross (§12.117) để so
    c, u, ut = wskip_cooccur(state, d_max=args.d_max, lam=args.lam, content_only=True)
    tot = sum(c.values()) or 1.0
    def pmi(a, b):
        w = c.get((a, b), 0.0)
        if w <= 0:
            return 0.0
        q = (u[a] / ut) * (u[b] / ut)
        return max(0.0, math.log((w / tot) / q)) if q > 0 else 0.0
    E_skip = build_eskip(state, dim=96, d_max=args.d_max, lam=args.lam)
    Em = E_skip.matrix

    def content(ids):
        return [int(t) for t in ids
                if 0 <= int(t) < n and not struct[int(t)] and v.is_surface(int(t))]

    cloud_cache: dict[tuple, np.ndarray] = {}
    def cloud_of(P_ids):
        keyt = tuple(sorted(set(P_ids)))
        if keyt in cloud_cache:
            return cloud_cache[keyt]
        v0 = np.zeros(n)
        for t in P_ids:
            v0[t] += 1.0
        s = v0.sum()
        if s > 0:
            v0 /= s
        acc = np.zeros(n)
        vk = v0
        for k in range(1, args.K + 1):
            vk = vk @ P                      # một bước chuyển (vector thưa × ma trận)
            vk = np.asarray(vk).ravel()
            acc += (args.gamma ** k) * vk
        cloud_cache[keyt] = acc
        return acc

    def cen(ids):
        ii = [t for t in ids if 0 <= t < Em.shape[0]]
        if not ii:
            return None
        x = Em[ii].mean(0)
        nr = float(np.linalg.norm(x))
        return x / nr if nr > 1e-9 else None

    key = json.load(open(_ROOT / args.key))
    sc = json.load(open(args.scores))
    rows = []  # (C, transition, pmi, cos)
    for sid, meta in key.items():
        if sid not in sc:
            continue
        P_ids = content(meta.get("prompt_ids", []))
        Q = content(meta.get("cont_ids", []))
        if not P_ids or not Q:
            continue
        cl = cloud_of(P_ids)
        trans = float(np.mean([cl[q] for q in Q]))
        pm = float(np.mean([max((pmi(p, q) for p in P_ids), default=0.0) for q in Q]))
        cp, cq = cen(meta.get("prompt_ids", [])), cen(meta.get("cont_ids", []))
        co = float(cp @ cq) if cp is not None and cq is not None else 0.0
        rows.append((sc[sid][1], trans, pm, co, sid, meta["prompt"][:20]))

    Cs = [r[0] for r in rows]
    print(f"\n{len(rows)} mẫu · K={args.K} γ={args.gamma}\n")
    names = [("transition(cloud)", 1), ("pmi-cross(§12.117)", 2), ("cos(E_skip)", 3)]
    print("Spearman thước↔C (cao = theo sát người hơn):")
    for nm, k in names:
        print(f"    {nm:20s} {spearman([r[k] for r in rows], Cs):+.3f}")
    print("\nTách chuẩn-hoá (C≥2 vs C≤1)/sd:")
    for nm, k in names:
        col = [r[k] for r in rows]
        hi = [col[i] for i in range(len(col)) if Cs[i] >= 2]
        lo = [col[i] for i in range(len(col)) if Cs[i] <= 1]
        sd = stx.pstdev(col) or 1.0
        mh = sum(hi) / len(hi) if hi else float("nan")
        ml = sum(lo) / len(lo) if lo else float("nan")
        print(f"    {nm:20s} {(mh-ml)/sd:+.3f}")

    # bẫy echo? xem top-6 transition có phải mẫu lặp/xáo không
    rows.sort(key=lambda r: -r[1])
    print("\nTOP-4 transition cao nhất (kiểm bẫy echo):")
    for C, tr, pm, co, sid, p in rows[:4]:
        print(f"  cloud {tr:.4f} C{C} | {p!r}")


if __name__ == "__main__":
    main()
