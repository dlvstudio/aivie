#!/usr/bin/env python
"""P62 (TK_045 §13.26) — tín hiệu MỐI NỐI: cái nào phân biệt nối THẬT với nối BẢN-LỀ-ĐỒNG-ÂM?

**Bệnh (chấm mù vòng ba §13.25 chỉ ra, ở CẢ HAI cổng):** mối nối bắc bằng **từ đồng âm**:
`theo dõi bệnh **nhân**` → `2 **nhân** 3 bằng 6` · `sông **Hậu**` → `**Hậu** quả gồm nắng
nóng` · `vương tôn` → `vương **tôn** giáo`.

**Gốc:** `join_candidates` nối bằng **ĐỒNG NHẤT MỘT TOKEN** —
`W_skip[d][last_content(A), first_content(B)]`. Tiếng Việt đồng âm dày đặc ⇒ `nhân`(người)
và `nhân`(phép nhân) là **cùng một token-id** ⇒ mối nối hợp lệ về **đếm**, vô nghĩa về **nghĩa**.
Đây đúng căn bệnh mặt-chữ đã giết hai cổng fold, nhưng nằm ở **chỗ ghép** — không cổng nào
với tới.

## ĐỐI CHỨNG VÀNG dựng TRƯỚC (bài học §13.23)

* **Nối THẬT** = hai mệnh đề **liền kề trong cùng một cha** — đúng theo định nghĩa (văn gốc).
* **Nối GIẢ** = ứng viên mà `join_candidates` **thật sự đề xuất** cho cùng mệnh đề nguồn,
  nhưng từ cha khác. Đây là âm bản **thực tế**, không phải cặp ngẫu nhiên — nếu lấy ngẫu
  nhiên thì mọi tín hiệu đều tách được và phép đo thành vô nghĩa.

Thước: **AUC** = xác suất tín hiệu xếp một nối THẬT trên một nối GIẢ. 0.5 = mù hoàn toàn.

## Tín hiệu đem so

| | công thức | ý |
|---|---|---|
| `S1_hien_tai` | `Σ_d W_skip[d][lastA, firstB]` | **đang dùng** — một token bản lề |
| `S2_cua_so` | `Σ_{a∈đuôi A, b∈đầu B} λ^… · W_skip[d][a,b]` | nhiều token hai phía ⇒ đồng âm bị ngữ cảnh khử |
| `S3_pmi` | `pmi_cross(A, B)` | đồng-hiện toàn mệnh đề |

**CỔNG G-join (nêu TRƯỚC):** một tín hiệu thay thế chỉ đáng wire khi **AUC ≥ 0.65** VÀ
**hơn `S1_hien_tai` ≥ +0.05**. Dưới mức đó thì nó không tách nổi bệnh và wire chỉ thêm nhiễu.

Chạy:
  PYTHONPATH=src .venv/bin/python scripts/probe_join_signal.py \\
      --checkpoint checkpoints/mini_sent_ep_qa.pkl
"""
from __future__ import annotations

import argparse
import gc
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from nn_aivie.core.checkpoint_io import load_checkpoint  # noqa: E402
from nn_aivie.hca.clause_layer import (  # noqa: E402
    build_clause_layer,
    content_folds,
    join_candidates,
)
from nn_aivie.hca.metrics_pmi import PmiCrossIndex  # noqa: E402
from nn_aivie.hca.w_skip import ensure_w_skip  # noqa: E402
from nn_aivie.util.progress import get_run_logger  # noqa: E402

GATE_AUC_MIN = 0.65
GATE_AUC_GAIN = 0.05


def _auc(pos: list[float], neg: list[float]) -> float:
    """P(điểm nối THẬT > điểm nối GIẢ), hoà tính 0.5. Mann–Whitney."""
    if not pos or not neg:
        return float("nan")
    allv = sorted(pos + neg)
    rank = {}
    i = 0
    while i < len(allv):
        j = i
        while j < len(allv) and allv[j] == allv[i]:
            j += 1
        r = (i + j - 1) / 2.0 + 1.0
        rank[allv[i]] = r
        i = j
    rp = sum(rank[v] for v in pos)
    n1, n2 = len(pos), len(neg)
    return (rp - n1 * (n1 + 1) / 2.0) / (n1 * n2)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/mini_sent_ep_qa.pkl")
    ap.add_argument("--n-pairs", type=int, default=500)
    ap.add_argument("--d-join", type=int, default=4)
    ap.add_argument("--tail-k", type=int, default=4, help="số token nội dung ĐUÔI của A")
    ap.add_argument("--head-k", type=int, default=4, help="số token nội dung ĐẦU của B")
    ap.add_argument("--lam", type=float, default=0.6)
    ap.add_argument("--pmi-cut", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-json", default="out/probe_join_signal.json")
    args = ap.parse_args()

    logger = get_run_logger("probe_join_signal")
    t0 = time.time()
    logger.log(f"P62 — tín hiệu mối nối · {args.checkpoint}")
    logger.log(f"CỔNG G-join (nêu TRƯỚC): AUC ≥ {GATE_AUC_MIN} VÀ hơn S1 ≥ +{GATE_AUC_GAIN}")

    gc.disable()
    state, _ = load_checkpoint(str(_ROOT / args.checkpoint))
    gc.enable()
    v = state.vocab
    layer = build_clause_layer(state, pmi_theta=args.pmi_cut, logger=logger)
    ws = ensure_w_skip(state, d_max=8, min_count=1, content_only=False)
    pmi = PmiCrossIndex(state)
    logger.log(f"tầng mệnh đề {len(layer):,} ({time.time()-t0:.0f}s)")

    def content_seq(cl):
        return [t for t in cl.tokens if content_folds(state, (int(t),), gidf_min=0.0)]

    def S1(A, B):
        s = 0.0
        for d in range(1, args.d_join + 1):
            if d > ws.d_max:
                break
            row = ws.mats[d].getrow(int(A.last_content))
            if row.nnz:
                idx = np.flatnonzero(row.indices == int(B.first_content))
                if idx.size:
                    s += float(row.data[idx[0]])
        return s

    def S2(A, B):
        """Cửa sổ hai phía: đuôi A × đầu B, giảm theo khoảng cách tới mối nối."""
        ta = content_seq(A)[-args.tail_k:]
        hb = content_seq(B)[: args.head_k]
        if not ta or not hb:
            return 0.0
        s = 0.0
        for ia, a in enumerate(reversed(ta)):        # ia=0 là token sát mối nối
            for ib, b in enumerate(hb):
                w = (args.lam ** (ia + ib))
                for d in range(1, args.d_join + 1):
                    if d > ws.d_max:
                        break
                    row = ws.mats[d].getrow(int(a))
                    if row.nnz:
                        k = np.flatnonzero(row.indices == int(b))
                        if k.size:
                            s += w * float(row.data[k[0]])
        return s

    def S3(A, B):
        return float(pmi.score(list(A.tokens), list(B.tokens)))

    # --- ĐỐI CHỨNG VÀNG: nối THẬT = mệnh đề liền kề CÙNG cha ---
    gold = []
    for _pi, cids in layer.by_parent.items():
        for a, b in zip(cids, cids[1:]):
            A, B = layer.clauses[a], layer.clauses[b]
            if len(A.tokens) >= 4 and len(B.tokens) >= 4:
                gold.append((A, B))
    random.Random(args.seed).shuffle(gold)
    gold = gold[: args.n_pairs]

    # --- ÂM BẢN THỰC TẾ: ứng viên mà join_candidates THẬT SỰ đề xuất, từ cha KHÁC ---
    rng = random.Random(args.seed + 1)
    fake = []
    for A, _B in gold:
        tw = join_candidates(layer, ws, A.last_content, d_join=args.d_join,
                             hub_beta=1.0, vocab=v, require_word_start=True)
        if not tw:
            continue
        opener = rng.choice(list(tw.keys()))
        cand = [c for c in layer.by_opener[opener]
                if layer.clauses[c].parent_idx != A.parent_idx
                and len(layer.clauses[c].tokens) >= 4]
        if cand:
            fake.append((A, layer.clauses[rng.choice(cand)]))
    logger.log(f"nối THẬT {len(gold)} · nối GIẢ {len(fake)} (ứng viên thật của cùng nguồn)")

    rows = []
    for name, fn in (("S1_hien_tai", S1), ("S2_cua_so", S2), ("S3_pmi", S3)):
        p = [fn(a, b) for a, b in gold]
        n = [fn(a, b) for a, b in fake]
        auc = _auc(p, n)
        rows.append({"signal": name, "auc": auc,
                     "mean_gold": float(np.mean(p)), "mean_fake": float(np.mean(n))})
        logger.log(f"  {name:14s} AUC={auc:.3f}  (thật {np.mean(p):.3f} · giả {np.mean(n):.3f})"
                   f"  [{time.time()-t0:.0f}s]")

    base = next(r for r in rows if r["signal"] == "S1_hien_tai")
    winners = [r for r in rows if r["signal"] != "S1_hien_tai"
               and r["auc"] >= GATE_AUC_MIN and r["auc"] - base["auc"] >= GATE_AUC_GAIN]
    ok = bool(winners)
    logger.log(f"\n=== P62 ===")
    logger.log(f"  {'tín hiệu':16s} {'AUC':>7s} {'so S1':>8s}")
    for r in rows:
        logger.log(f"  {r['signal']:16s} {r['auc']:7.3f} {r['auc']-base['auc']:+8.3f}")
    logger.log(f"\n  G-join: {'ĐẠT' if ok else 'TRƯỢT'}"
               + (f" — {', '.join(w['signal'] for w in winners)}" if ok else ""))
    if not ok:
        logger.log("  ⇒ không tín hiệu nào tách nổi nối-thật khỏi nối-đồng-âm ở mức dùng được."
                   " Wire thêm chỉ tăng nhiễu. Cần ý tưởng KHÁC, không phải chỉnh λ/k.")
    if base["auc"] < 0.55:
        logger.log(f"  ⚠️ S1 (đang dùng) AUC={base['auc']:.3f} ≈ MÙ HOÀN TOÀN — mối nối hiện"
                   f" tại gần như không phân biệt thật/giả. Khớp với chấm mù §13.25.")

    out = _ROOT / args.out_json
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "checkpoint": args.checkpoint, "n_gold": len(gold), "n_fake": len(fake),
        "gate": {"auc_min": GATE_AUC_MIN, "auc_gain": GATE_AUC_GAIN},
        "rows": rows, "gate_pass": ok, "seconds": round(time.time() - t0, 1),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.log(f"  → {out}")


if __name__ == "__main__":
    main()
