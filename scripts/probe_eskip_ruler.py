#!/usr/bin/env python
"""② §12.113 — E_skip có làm THƯỚC bám-chủ-đề tốt hơn E_old không?

Bối cảnh: §12.112 quyết dùng embedding để đo TRÔI ĐỀ (cos(prompt, phần sinh)). Nhưng E_old gần
CHẾT (§12.98: cosE_old top1 0.004). E_skip hơn 34× khi đứng một mình — nhưng nó dựng cho next-token.
Câu hỏi MỚI: dùng làm thước bám-chủ-đề (chứ không xếp hạng), E_skip có tách "đúng đề" khỏi "trôi"
tốt hơn E_old không? Nhãn thật = điểm C (bám prompt /3) tôi đã chấm MÙ ở §12.112.

Thước: với mỗi mẫu, cos(centroid prompt, centroid phần sinh) trong E_old và E_new. Rồi:
  (a) tương quan Spearman với C — thước nào theo sát nhãn người hơn.
  (b) tách tuyệt đối: cos trung bình cho C≥2 (đúng đề) vs C≤1 (trôi).

KHÔNG đổi state.embedding. Chỉ đo. Đọc out/blind_key.json (có token-id) + file điểm C.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from nn_aivie.core.checkpoint_io import load_checkpoint  # noqa: E402
from nn_aivie.core.embedding import truncated_svd  # noqa: E402
from nn_aivie.hca.context_positional import is_structural_token  # noqa: E402
from nn_aivie.hca.sent_intra import _content_positions, iter_sentence_spans  # noqa: E402


def build_eskip(state, dmax: int, lam: float, dim: int):
    v = state.vocab
    n = len(v)
    struct = [is_structural_token(v, i) for i in range(n)]
    c: dict[tuple[int, int], float] = defaultdict(float)
    u: dict[int, float] = defaultdict(float)
    ut = 0.0
    for rec in state.parents.parents:
        sq = list(rec.answer_ids if (rec.is_qa and rec.answer_ids) else rec.emit_ids)
        for sp in iter_sentence_spans(sq, v):
            tk = [int(t) for t in _content_positions(sp, v, content_only=True,
                                                     struct_mask=struct)]
            for t in tk:
                u[t] += 1.0
                ut += 1.0
            for i in range(len(tk)):
                for j in range(i + 1, min(len(tk), i + 1 + dmax)):
                    w = lam ** (j - i - 1)
                    c[(tk[i], tk[j])] += w      # ĐỐI XỨNG hoá: embedding lo TƯƠNG ĐỒNG
                    c[(tk[j], tk[i])] += w
    tot = sum(c.values()) or 1.0
    r, cc, vv = [], [], []
    for (a, b), w in c.items():
        q = (u[a] / ut) * (u[b] / ut)
        if q <= 0:
            continue
        x = math.log((w / tot) / q)
        if x > 0:
            r.append(a); cc.append(b); vv.append(x)
    ppmi = coo_matrix((vv, (r, cc)), shape=(n, n)).tocsr()
    return truncated_svd(ppmi, dim)


def centroid(M, ids):
    if M is None:
        return None
    ii = [int(t) for t in ids if 0 <= int(t) < M.shape[0]]
    if not ii:
        return None
    x = np.asarray(M[ii]).mean(0)
    nr = float(np.linalg.norm(x))
    return x / nr if nr > 1e-9 else None


def spearman(xs, ys):
    def rank(a):
        order = sorted(range(len(a)), key=lambda i: a[i])
        rk = [0.0] * len(a)
        i = 0
        while i < len(a):
            j = i
            while j + 1 < len(a) and a[order[j + 1]] == a[order[i]]:
                j += 1
            avg = (i + j) / 2.0
            for k in range(i, j + 1):
                rk[order[k]] = avg
            i = j + 1
        return rk
    rx, ry = rank(xs), rank(ys)
    n = len(xs)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((rx[i] - mx) * (ry[i] - my) for i in range(n))
    dx = math.sqrt(sum((rx[i] - mx) ** 2 for i in range(n)))
    dy = math.sqrt(sum((ry[i] - my) ** 2 for i in range(n)))
    return num / (dx * dy) if dx > 0 and dy > 0 else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--key", default="out/blind_key.json")
    ap.add_argument("--scores", required=True, help="json Sxxx -> [F,C,N,P]")
    ap.add_argument("--emb-dmax", type=int, default=8)
    ap.add_argument("--emb-lam", type=float, default=0.85)
    ap.add_argument("--emb-dim", type=int, default=96)
    args = ap.parse_args()

    t0 = time.time()
    gc.disable()
    state, _ = load_checkpoint(str(_ROOT / args.checkpoint))
    gc.enable()
    v = state.vocab
    E_old = getattr(getattr(state, "embedding", None), "matrix", None)
    print(f"[{time.time()-t0:.0f}s] E_old {None if E_old is None else E_old.shape}", flush=True)
    E_new = build_eskip(state, args.emb_dmax, args.emb_lam, args.emb_dim)
    print(f"[{time.time()-t0:.0f}s] E_skip {E_new.shape}", flush=True)

    key = json.load(open(_ROOT / args.key))
    scores = json.load(open(args.scores))

    rows = []  # (C, cos_old, cos_new)
    for sid, meta in key.items():
        if sid not in scores:
            continue
        C = scores[sid][1]
        pid, cid = meta.get("prompt_ids"), meta.get("cont_ids")
        if not pid or not cid:
            continue
        po, pn = centroid(E_old, pid), centroid(E_new, pid)
        co, cn = centroid(E_old, cid), centroid(E_new, cid)
        cos_o = float(po @ co) if po is not None and co is not None else float("nan")
        cos_n = float(pn @ cn) if pn is not None and cn is not None else float("nan")
        rows.append((C, cos_o, cos_n))

    valid = [r for r in rows if not (math.isnan(r[1]) or math.isnan(r[2]))]
    print(f"\n{len(valid)}/{len(rows)} mẫu có đủ cos cả hai embedding\n")
    Cs = [r[0] for r in valid]
    co = [r[1] for r in valid]
    cn = [r[2] for r in valid]

    print("(a) Spearman cos↔C (nhãn bám-prompt người chấm; cao = thước theo sát người hơn):")
    print(f"    E_old : {spearman(co, Cs):+.3f}")
    print(f"    E_skip: {spearman(cn, Cs):+.3f}")

    def split(vals):
        hi = [vals[i] for i in range(len(vals)) if Cs[i] >= 2]
        lo = [vals[i] for i in range(len(vals)) if Cs[i] <= 1]
        mh = sum(hi) / len(hi) if hi else float("nan")
        ml = sum(lo) / len(lo) if lo else float("nan")
        return mh, ml, mh - ml
    print("\n(b) Tách tuyệt đối cos: đúng-đề (C≥2) vs trôi (C≤1) — cao = tách rõ hơn:")
    for nm, vals in (("E_old", co), ("E_skip", cn)):
        mh, ml, d = split(vals)
        print(f"    {nm:6s}  C≥2 {mh:+.3f}  C≤1 {ml:+.3f}  TÁCH {d:+.3f}")


if __name__ == "__main__":
    main()
