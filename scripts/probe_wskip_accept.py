#!/usr/bin/env python
"""PHA 3 (§12.114) — W_skip làm tín hiệu NGHIỆM THU toàn cục. Đánh vào §12.104.

Nút chặn thật (§12.113①b, đo trên CẢ HAI trục): nền chắc ở chỗ chép, phân kỳ thì salad — vì không
có tín hiệu phân biệt "mới-mà-đúng" với "mới-mà-què". Lỗi CHÍNH quan sát được là TRÔI ĐỀ (nhảy sang
tài liệu khác), không phải splice. Mà trôi đề = token sinh KHÔNG đồng-hiện skip với token prompt.

Thử: chấm mỗi mẫu bằng **cross-support** prompt↔sinh — với mỗi token-nội-dung sinh, nó có đồng-hiện
skip (trong TRAIN, d≤D) với MỘT token-nội-dung prompt nào không. Hai biến thể:
  attest  = tỉ lệ token sinh có ÍT NHẤT một cặp được chứng thực với prompt
  pmi     = trung bình (max PMI skip tới prompt)

So với thước Pha 1 (cos E_skip centroid). Nhãn = C (bám prompt) chấm mù.

Dự đoán ĐĂNG KÝ TRƯỚC (thành thật): §12.104 cho attestation LIỀN trượt vì mù với splice. Nhưng đây
đo TRÔI (cặp không-đồng-hiện), khác splice — nên có CƠ HỘI dương cho lỗi trôi, dù có thể vẫn mù
splice. Cổng G-acc: tách (C≥2 vs C≤1) tốt hơn cos-E_skip ⇒ Pha 3 mở; nếu không ⇒ đóng, về §12.104e.

Luật 0: đếm đồng-hiện + PMI, closed-form.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from nn_aivie.core.checkpoint_io import load_checkpoint  # noqa: E402
from nn_aivie.hca.context_positional import is_structural_token  # noqa: E402
from nn_aivie.hca.embedding_skip import build_eskip, wskip_cooccur  # noqa: E402


def spearman(xs, ys):
    def rank(a):
        order = sorted(range(len(a)), key=lambda i: a[i])
        rk = [0.0] * len(a)
        i = 0
        while i < len(a):
            j = i
            while j + 1 < len(a) and a[order[j + 1]] == a[order[i]]:
                j += 1
            for k in range(i, j + 1):
                rk[order[k]] = (i + j) / 2.0
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
    ap.add_argument("--scores", required=True)
    ap.add_argument("--d-max", type=int, default=8)
    ap.add_argument("--lam", type=float, default=0.85)
    args = ap.parse_args()

    t0 = time.time()
    gc.disable()
    state, _ = load_checkpoint(str(_ROOT / args.checkpoint))
    gc.enable()
    v = state.vocab
    struct = [is_structural_token(v, i) for i in range(len(v))]

    c, u, ut = wskip_cooccur(state, d_max=args.d_max, lam=args.lam, content_only=True)
    tot = sum(c.values()) or 1.0
    print(f"[{time.time()-t0:.0f}s] đồng-hiện skip: {len(c):,} cặp", flush=True)

    def pmi(a, b):
        w = c.get((a, b), 0.0)
        if w <= 0:
            return None
        q = (u[a] / ut) * (u[b] / ut)
        if q <= 0:
            return None
        x = math.log((w / tot) / q)
        return x if x > 0 else 0.0

    def content(ids):
        return [int(t) for t in ids
                if 0 <= int(t) < len(v) and not struct[int(t)] and v.is_surface(int(t))]

    E_skip = build_eskip(state, dim=96, d_max=args.d_max, lam=args.lam)

    def cos(pids, cids):
        import numpy as np
        M = E_skip.matrix

        def cen(ids):
            ii = [t for t in ids if 0 <= t < M.shape[0]]
            if not ii:
                return None
            x = M[ii].mean(0)
            nr = float(np.linalg.norm(x))
            return x / nr if nr > 1e-9 else None
        a, b = cen(pids), cen(cids)
        return float(a @ b) if a is not None and b is not None else None

    key = json.load(open(_ROOT / args.key))
    sc = json.load(open(args.scores))
    rows = []  # (C, attest, pmi_score, cos_eskip)
    for sid, meta in key.items():
        if sid not in sc:
            continue
        P = content(meta.get("prompt_ids", []))
        Q = content(meta.get("cont_ids", []))
        if not P or not Q:
            continue
        att = sum(1 for q in Q if any(c.get((p, q), 0) > 0 for p in P)) / len(Q)
        pm = []
        for q in Q:
            best = max((pmi(p, q) or 0.0 for p in P), default=0.0)
            pm.append(best)
        pmi_s = sum(pm) / len(pm) if pm else 0.0
        ce = cos(meta.get("prompt_ids", []), meta.get("cont_ids", []))
        rows.append((sc[sid][1], att, pmi_s, ce if ce is not None else 0.0))

    Cs = [r[0] for r in rows]
    print(f"\n{len(rows)} mẫu · nhãn C (bám prompt)\n")
    print("Spearman thước↔C (cao = theo sát người hơn):")
    names = ["attest(skip cross)", "pmi(skip cross)", "cos(E_skip)"]
    for k, nm in enumerate(names, start=1):
        col = [r[k] for r in rows]
        print(f"    {nm:20s} {spearman(col, Cs):+.3f}")

    print("\nTách tuyệt đối: đúng-đề (C≥2) vs trôi (C≤1) — cao = tách rõ:")
    for k, nm in enumerate(names, start=1):
        col = [r[k] for r in rows]
        hi = [col[i] for i in range(len(col)) if Cs[i] >= 2]
        lo = [col[i] for i in range(len(col)) if Cs[i] <= 1]
        mh = sum(hi) / len(hi) if hi else float("nan")
        ml = sum(lo) / len(lo) if lo else float("nan")
        # chuẩn hoá tách theo độ lệch chuẩn gộp để so 3 thước khác thang
        import statistics as stx
        sd = stx.pstdev(col) or 1.0
        print(f"    {nm:20s} C≥2 {mh:+.3f}  C≤1 {ml:+.3f}  TÁCH {mh-ml:+.3f}  (chuẩn hoá {(mh-ml)/sd:+.3f})")

    print("\n  G-acc: skip-attestation tách TỐT HƠN cos(E_skip) ⇒ Pha 3 mở; nếu không ⇒ đóng, §12.104e.")


if __name__ == "__main__":
    main()
