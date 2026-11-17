#!/usr/bin/env python
"""P65 (TK_045 §13.31) — bộ CỨU mối nối: cứu được các lựa chọn SAI của chính S1 không?

## Vì sao đối chứng phải đổi

§13.26 dựng âm bản = ứng viên **ngẫu nhiên** trong số `join_candidates` đề xuất. `S3_pmi`
đạt AUC .814 ở đó — rồi **thua chấm mù 59.7%** (§13.30). Vì sao: âm bản ngẫu nhiên **không
phải** thứ hệ thật sự chọn. Hệ chọn `argmax S1`, và đúng những cái đó mới là bệnh.

**Đối chứng ở đây:** âm bản = **ứng viên đứng đầu theo S1** từ cha khác — tức chính mệnh đề
mà cơ chế đang chạy **sẽ nối vào**. Trên tập này:

* AUC của **S1 vô nghĩa** (âm bản là argmax của nó ⇒ nó thua theo định nghĩa). Không đọc.
* AUC của **một tín hiệu khác** = xác suất nó xếp mối nối THẬT trên **lựa chọn sai của S1**
  ⇒ đúng nghĩa "cứu được bao nhiêu". Đây là con số dùng để quyết.

Cùng lý đó, một âm bản thứ hai được giữ: `argmax S3` (mạch lạc cục bộ nhưng lạc đề) — hạng
mà `S3` rơi vào ở §13.30. Bộ cứu phải làm được **cả hai**, nếu không ta lại wire một thứ
đẹp trên giấy rồi thua chấm mù lần nữa.

## Tín hiệu

| | công thức | giả thuyết |
|---|---|---|
| `S4_prompt` | `pmi_cross(prompt, cand)` | neo **toàn cục** vào đề, không phải mạch cục bộ |
| `S6_khu_ban_le` | `W_skip(đuôi A **bỏ token bản lề**, đầu B)` | bệnh là *đồng âm ở token bản lề*; bỏ nó ra thì ngữ cảnh hai bên phải tự khớp |
| `S7_ket_hop` | `z(S6) + z(S4)` | vừa nối được vừa còn trong đề |

`S6` là mũi nhọn: `join_candidates` nối bằng **một** token (`W_skip[d][last_content(A), .]`),
và tiếng Việt gộp `nhân`(người) với `nhân`(phép nhân) vào **cùng một hàng**. Bỏ chính token
đó ra khỏi phép chấm ⇒ lối tắt đồng âm biến mất, chỉ còn ngữ cảnh thật đỡ.

## CỔNG G-rescue (nêu TRƯỚC)

Một tín hiệu chỉ đáng đem đi chấm mù khi **AUC ≥ 0.65 trên CẢ HAI** lớp âm bản.
Chỉ đạt một lớp ⇒ nó vá được một dạng lỗi và mở dạng kia — đúng vết xe `S3`.

Chạy:
  PYTHONPATH=src .venv/bin/python scripts/probe_join_prompt.py \\
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
from nn_aivie.hca.readout import decode_surface  # noqa: E402
from nn_aivie.hca.w_skip import ensure_w_skip  # noqa: E402
from nn_aivie.util.progress import get_run_logger  # noqa: E402

GATE_AUC_MIN = 0.65


def _auc(pos: list[float], neg: list[float]) -> float:
    """P(điểm THẬT > điểm GIẢ), hoà tính 0.5. Mann–Whitney."""
    if not pos or not neg:
        return float("nan")
    allv = sorted(pos + neg)
    rank: dict[float, float] = {}
    i = 0
    while i < len(allv):
        j = i
        while j < len(allv) and allv[j] == allv[i]:
            j += 1
        rank[allv[i]] = (i + j - 1) / 2.0 + 1.0
        i = j
    n1, n2 = len(pos), len(neg)
    return (sum(rank[v] for v in pos) - n1 * (n1 + 1) / 2.0) / (n1 * n2)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/mini_sent_ep_qa.pkl")
    ap.add_argument("--n-pairs", type=int, default=300)
    ap.add_argument("--d-join", type=int, default=4)
    ap.add_argument("--tail-k", type=int, default=4)
    ap.add_argument("--head-k", type=int, default=4)
    ap.add_argument("--lam", type=float, default=0.6)
    ap.add_argument("--n-cand", type=int, default=24, help="ứng viên xét mỗi nguồn")
    ap.add_argument("--pmi-cut", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dump", type=int, default=5)
    ap.add_argument("--out-json", default="out/probe_join_prompt.json")
    args = ap.parse_args()

    logger = get_run_logger("probe_join_prompt")
    t0 = time.time()
    logger.log(f"P65 — bộ CỨU mối nối · {args.checkpoint}")
    logger.log(f"CỔNG G-rescue (nêu TRƯỚC): AUC ≥ {GATE_AUC_MIN} trên CẢ HAI lớp âm bản")

    gc.disable()
    state, _ = load_checkpoint(str(_ROOT / args.checkpoint))
    gc.enable()
    layer = build_clause_layer(state, pmi_theta=args.pmi_cut, logger=logger)
    ws = ensure_w_skip(state, d_max=8, min_count=1, content_only=False)
    pmi = PmiCrossIndex(state)
    logger.log(f"tầng mệnh đề {len(layer):,} ({time.time()-t0:.0f}s)")

    _cs: dict[int, list[int]] = {}

    def content_seq(cl) -> list[int]:
        key = id(cl)
        got = _cs.get(key)
        if got is None:
            got = [int(t) for t in cl.tokens
                   if content_folds(state, (int(t),), gidf_min=0.0)]
            _cs[key] = got
        return got

    def _ws_pair(a: int, b: int) -> float:
        s = 0.0
        for d in range(1, min(args.d_join, ws.d_max) + 1):
            row = ws.mats[d].getrow(a)
            if row.nnz:
                k = np.flatnonzero(row.indices == b)
                if k.size:
                    s += float(row.data[k[0]])
        return s

    def S1(P, A, B) -> float:
        """ĐANG CHẠY: một token bản lề."""
        return _ws_pair(int(A.last_content), int(B.first_content))

    def S3(P, A, B) -> float:
        return float(pmi.score(list(A.tokens), list(B.tokens)))

    def S4(P, A, B) -> float:
        """Neo vào PROMPT, không phải vào mệnh đề trước."""
        return float(pmi.score(list(P.tokens), list(B.tokens)))

    def S6(P, A, B) -> float:
        """Đuôi A **BỎ token bản lề** × đầu B ⇒ lối tắt đồng âm bị cắt."""
        ta = content_seq(A)[-args.tail_k:][:-1]      # bỏ chính last_content
        hb = content_seq(B)[: args.head_k]
        if not ta or not hb:
            return 0.0
        s = 0.0
        for ia, a in enumerate(reversed(ta)):        # ia=0 sát mối nối nhất
            for ib, b in enumerate(hb):
                w = args.lam ** (ia + ib)
                if w > 1e-3:
                    s += w * _ws_pair(a, b)
        return s

    # --- ĐỐI CHỨNG VÀNG: (prompt = mệnh đề mở của cha, A, B liền kề sau đó) ---
    gold = []
    for _pi, cids in layer.by_parent.items():
        if len(cids) < 3:
            continue
        P = layer.clauses[cids[0]]
        for a, b in zip(cids[1:], cids[2:]):
            A, B = layer.clauses[a], layer.clauses[b]
            if len(A.tokens) >= 4 and len(B.tokens) >= 4:
                gold.append((P, A, B))
    random.Random(args.seed).shuffle(gold)
    gold = gold[: args.n_pairs]

    # --- ÂM BẢN: ứng viên ĐỨNG ĐẦU theo S1 / theo S3, từ cha KHÁC ---
    rng = random.Random(args.seed + 1)
    neg_s1, neg_s3, examples = [], [], []
    for P, A, _B in gold:
        tw = join_candidates(layer, ws, A.last_content, d_join=args.d_join,
                             hub_beta=1.0, vocab=state.vocab, require_word_start=True)
        cands = [layer.clauses[c] for op in tw for c in layer.by_opener[op]
                 if layer.clauses[c].parent_idx != A.parent_idx
                 and len(layer.clauses[c].tokens) >= 4]
        if len(cands) < 3:
            continue
        if len(cands) > args.n_cand:
            cands = rng.sample(cands, args.n_cand)
        b1 = max(cands, key=lambda C: S1(P, A, C))
        b3 = max(cands, key=lambda C: S3(P, A, C))
        neg_s1.append((P, A, b1))
        neg_s3.append((P, A, b3))
        if len(examples) < args.dump:
            examples.append(
                f"[{decode_surface(state, list(P.tokens)).strip()[:34]}] "
                f"…{decode_surface(state, list(A.tokens)).strip()[-34:]}"
                f"  ⇒argmaxS1⇒  {decode_surface(state, list(b1.tokens)).strip()[:46]}")
    logger.log(f"vàng {len(gold)} · âm-argmaxS1 {len(neg_s1)} · âm-argmaxS3 {len(neg_s3)}")
    logger.log("mẫu lựa chọn SAI của cơ chế đang chạy:")
    for e in examples:
        logger.log(f"    | {e}")

    sigs = [("S4_prompt", S4), ("S6_khu_ban_le", S6)]
    cache: dict[str, dict[str, list[float]]] = {}
    for name, fn in [("S1_dang_chay", S1), ("S3_pmi", S3), *sigs]:
        cache[name] = {
            "gold": [fn(*g) for g in gold],
            "n1": [fn(*n) for n in neg_s1],
            "n3": [fn(*n) for n in neg_s3],
        }

    def _z(xs: list[float], mu: float, sd: float) -> list[float]:
        return [(x - mu) / sd for x in xs] if sd > 0 else [0.0] * len(xs)

    # S7 = z(S6) + z(S4); chuẩn hoá bằng thống kê của tập VÀNG (không nhìn âm bản)
    comb: dict[str, list[float]] = {"gold": [], "n1": [], "n3": []}
    for split in comb:
        acc = np.zeros(len(cache["S4_prompt"][split]))
        for name in ("S6_khu_ban_le", "S4_prompt"):
            g = cache[name]["gold"]
            acc += np.array(_z(cache[name][split], float(np.mean(g)), float(np.std(g))))
        comb[split] = acc.tolist()
    cache["S7_ket_hop"] = comb

    rows = []
    for name in ("S1_dang_chay", "S3_pmi", "S4_prompt", "S6_khu_ban_le", "S7_ket_hop"):
        c = cache[name]
        a1, a3 = _auc(c["gold"], c["n1"]), _auc(c["gold"], c["n3"])
        rows.append({"signal": name, "auc_vs_argmaxS1": a1, "auc_vs_argmaxS3": a3,
                     "auc_min": min(a1, a3)})

    logger.log(f"\n=== P65 n_vàng={len(gold)} ===")
    logger.log(f"  {'tín hiệu':16s} {'vs argmaxS1':>12s} {'vs argmaxS3':>12s} {'min':>7s}")
    for r in rows:
        note = ""
        if r["signal"] == "S1_dang_chay":
            note = "  ← vô nghĩa (âm bản là argmax của nó)"
        elif r["signal"] == "S3_pmi":
            note = "  ← vô nghĩa ở cột argmaxS3"
        logger.log(f"  {r['signal']:16s} {r['auc_vs_argmaxS1']:12.3f} "
                   f"{r['auc_vs_argmaxS3']:12.3f} {r['auc_min']:7.3f}{note}")

    winners = [r for r in rows if r["signal"] in [s for s, _ in sigs] + ["S7_ket_hop"]
               and r["auc_min"] >= GATE_AUC_MIN]
    ok = bool(winners)
    logger.log(f"\n  G-rescue: {'ĐẠT' if ok else 'TRƯỢT'}"
               + (f" — {', '.join(w['signal'] for w in winners)}" if ok else ""))
    if not ok:
        logger.log("  ⇒ KHÔNG đem đi chấm mù. Không tín hiệu nào cứu nổi lựa chọn sai của S1"
                   " trên cả hai lớp; wire vào chỉ đổi dạng lỗi. §13.30 lặp lại.")

    out = _ROOT / args.out_json
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "checkpoint": args.checkpoint, "n_gold": len(gold), "n_neg": len(neg_s1),
        "gate": {"auc_min": GATE_AUC_MIN}, "rows": rows, "gate_pass": ok,
        "seconds": round(time.time() - t0, 1),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.log(f"  → {out}")


if __name__ == "__main__":
    main()
