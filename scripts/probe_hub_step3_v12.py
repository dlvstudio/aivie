#!/usr/bin/env python3
"""Đo BƯỚC 3 của hub_spread trên v12 — lan ngược từ nhãn ra thành viên mới.

Ý tác giả (2026-08-10): lấy nơ-ron W_skip làm NHÃN rồi lan ba bước::

    nghịch ngợm → {chó, mèo} → động vật → {chó, mèo, KHỈ}
                                          bước 3 = ứng viên CHƯA từng thấy

Bước 1–2 đã có ở `analogy_la`. Bước 3 (xuống lại từ nhãn) **chưa wire**. Ticket
ghi `hub_spread` "synthetic ĐẠT, trên isg_pw CÂM"; nguyên nhân đã tìm ra và vá
(tách kênh typed, commit e4324de). Câu hỏi còn lại: **vá đó có sống ở v12 không?**

THƯỚC NẰM TRONG ĐỒ THỊ — không chấm cảm tính:

* "hub có phải NHÃN không" = **bậc-vào trên cạnh typed** (`X là <hub>` với bao
  nhiêu X khác nhau). `động vật` có bậc-vào cao; `xóm` ≈ 0. Không ai phán xét.
* "có sáng tạo không" = thành viên lan ngược mà cặp (thành viên, thuộc tính)
  **chưa từng đồng hiện**.

Ba cột ĐỐI CHỨNG chạy cùng bộ seed: `typed_first` (mới) · `mixed` (cũ) · `cooc`.

    PYTHONPATH=src python scripts/probe_hub_step3_v12.py \
      --checkpoint checkpoints/network_v12_subst_isg_pw.pkl
"""
from __future__ import annotations

import argparse
import gc
import json
import random
import resource
import sys
import time
from collections import defaultdict
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from nn_aivie.core.checkpoint_io import load_checkpoint  # noqa: E402
from nn_aivie.hca.hub_spread import (  # noqa: E402
    _surf,
    build_hub_spread_index,
    reverse_peers,
    score_hubs_from,
)

T0 = time.monotonic()
_LOG: str | None = None

# Cổng nêu TRƯỚC khi có số (tiền đăng ký).
GATE_TYPED_EDGES = 1000     # (1) kênh typed phải tồn tại ở v12
GATE_LABEL_INDEG = 3        # hub được coi là NHÃN khi có ≥3 chủ thể khác nhau
GATE_LABEL_RATE = 0.30      # (2) ≥30% seed cho hub-đầu-bảng là NHÃN
GATE_LIFT_VS_COOC = 1.5     # (3) typed_first ≥1,5× cooc ở tỉ lệ nhãn
GATE_NEW_MEMBER = 0.20      # (4) ≥20% seed có ≥1 thành viên MỚI ở bước 3


def log(msg: str) -> None:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1073741824
    line = f"[{time.monotonic() - T0:7.1f}s RSS {rss:5.2f}GB] {msg}"
    print(line, flush=True)
    if _LOG:
        with open(_LOG, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")


def main() -> int:
    global _LOG
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/network_v12_subst_isg_pw.pkl")
    ap.add_argument("--n-seeds", type=int, default=200)
    ap.add_argument("--seed", type=int, default=46)
    ap.add_argument("--d-max", type=int, default=3)
    ap.add_argument("--min-count", type=int, default=2)
    ap.add_argument("--gidf-min", type=float, default=0.0)
    ap.add_argument("--peer-topk", type=int, default=20)
    ap.add_argument("--out", default="out/hub_step3_v12.json")
    ap.add_argument("--log", default="")
    # Ca cụ thể M4 đã lộ ra: `analogy_la` gán tên người cho con mèo.
    ap.add_argument("--case-prompts", default="mèo,chó,Trần,học sinh,Việt Nam")
    args = ap.parse_args()
    _LOG = args.log or f"out/hub_step3_{Path(args.checkpoint).stem}.log"
    Path(_LOG).parent.mkdir(parents=True, exist_ok=True)

    gc.disable()
    log(f"nạp {args.checkpoint} …")
    state, _ = load_checkpoint(str(_ROOT / args.checkpoint))
    gc.enable()
    v = state.vocab
    log(f"nạp xong · vocab={len(v)} parents={len(state.parents)}")

    log("dựng hub_spread index (W_skip + mine via-pivot) …")
    idx = build_hub_spread_index(
        state, d_max=args.d_max, min_count=args.min_count)
    n_typed_src = len(idx.via_pivot)
    n_typed_edges = sum(len(d) for d in idx.via_pivot.values())
    log(f"kênh typed: {n_typed_src} nguồn · {n_typed_edges} cạnh")

    # --- THƯỚC 1: bậc-vào typed = "hub này có phải NHÃN không" -------------
    indeg: dict[int, int] = defaultdict(int)
    for _src, dsts in idx.via_pivot.items():
        for dst in dsts:
            indeg[int(dst)] += 1
    top_label = sorted(indeg.items(), key=lambda kv: -kv[1])[:15]
    log("nhãn mạnh nhất theo bậc-vào typed: "
        + ", ".join(f"{_surf(v, t)}({c})" for t, c in top_label))

    # --- bộ seed ĐÓNG BĂNG: chủ thể có cạnh typed đi ra --------------------
    rng = random.Random(args.seed)
    cand = sorted(idx.via_pivot.keys())
    rng.shuffle(cand)
    seeds = cand[: args.n_seeds]
    log(f"seed đóng băng: {len(seeds)} (từ {len(cand)} chủ thể có cạnh typed)")

    rep: dict[str, object] = {
        "checkpoint": args.checkpoint,
        "n_vocab": len(v), "n_parents": len(state.parents),
        "n_typed_src": n_typed_src, "n_typed_edges": n_typed_edges,
        "top_labels": [{"surface": _surf(v, t), "indeg": c} for t, c in top_label],
        "n_seeds": len(seeds),
        "gates_pre": {
            "typed_edges": GATE_TYPED_EDGES, "label_indeg": GATE_LABEL_INDEG,
            "label_rate": GATE_LABEL_RATE, "lift_vs_cooc": GATE_LIFT_VS_COOC,
            "new_member": GATE_NEW_MEMBER,
        },
    }

    # --- ba cột đối chứng --------------------------------------------------
    cols: dict[str, dict[str, float]] = {}
    per_col_examples: dict[str, list] = {}
    for ch in ("typed_first", "mixed", "cooc"):
        n_label = 0
        n_new = 0
        n_have_hub = 0
        ex: list = []
        t = time.monotonic()
        for sid in seeds:
            hubs = score_hubs_from(
                state, idx, [sid], topk=1,
                gidf_min=args.gidf_min, ban_func=True, channel=ch,
            )
            if not hubs:
                continue
            n_have_hub += 1
            h = hubs[0]
            is_label = indeg.get(h.tid, 0) >= GATE_LABEL_INDEG
            n_label += int(is_label)

            # BƯỚC 3: từ nhãn lan ngược ra thành viên
            peers = reverse_peers(
                state, idx, [h.tid], exclude={sid}, topk=args.peer_topk)
            # "MỚI" = thành viên chưa từng đồng hiện trực tiếp với seed
            seen = set(idx.fwd.get(int(sid), {})) | set(idx.rev.get(int(sid), {}))
            moi = [p for p in peers if p.tid not in seen]
            n_new += int(bool(moi))
            if len(ex) < 12:
                ex.append({
                    "seed": _surf(v, sid),
                    "hub": h.surface, "typed": h.typed,
                    "hub_indeg": indeg.get(h.tid, 0), "la_nhan": is_label,
                    "peers": [p.surface for p in peers[:6]],
                    "thanh_vien_moi": [p.surface for p in moi[:6]],
                })
        n = max(n_have_hub, 1)
        cols[ch] = {
            "n_co_hub": n_have_hub,
            "ty_le_nhan": round(n_label / n, 4),
            "ty_le_co_thanh_vien_moi": round(n_new / n, 4),
            "giay": round(time.monotonic() - t, 1),
        }
        per_col_examples[ch] = ex
        log(f"[{ch:11}] có hub {n_have_hub}/{len(seeds)} · "
            f"nhãn {cols[ch]['ty_le_nhan']:.1%} · "
            f"thành viên mới {cols[ch]['ty_le_co_thanh_vien_moi']:.1%} "
            f"({cols[ch]['giay']}s)")
    rep["cols"] = cols
    rep["examples"] = per_col_examples

    # --- ca cụ thể: hub có chặn `con mèo là Vương` không? ------------------
    cases = []
    for w in [s.strip() for s in args.case_prompts.split(",") if s.strip()]:
        ids = [i for i in range(len(v)) if _surf(v, i) == w]
        if not ids:
            cases.append({"tu": w, "loi": "khong_co_trong_vocab"})
            continue
        hubs = score_hubs_from(
            state, idx, ids, topk=3, gidf_min=args.gidf_min,
            ban_func=True, channel="typed_first")
        members = []
        if hubs:
            members = [p.surface for p in reverse_peers(
                state, idx, [hubs[0].tid], exclude=set(ids), topk=30)]
        cases.append({
            "tu": w,
            "hub": [h.surface for h in hubs],
            "hub_typed": [h.typed for h in hubs],
            "hub_indeg": [indeg.get(h.tid, 0) for h in hubs],
            "thanh_vien_hub": members[:20],
        })
        log(f"  ca {w!r}: hub={[h.surface for h in hubs]} "
            f"thành viên={members[:8]}")
    rep["cases"] = cases

    # --- phán cổng ---------------------------------------------------------
    tf, cc = cols["typed_first"], cols["cooc"]
    lift = (tf["ty_le_nhan"] / cc["ty_le_nhan"]) if cc["ty_le_nhan"] else float("inf")
    g = {
        "1_typed_edges": n_typed_edges >= GATE_TYPED_EDGES,
        "2_label_rate": tf["ty_le_nhan"] >= GATE_LABEL_RATE,
        "3_lift_vs_cooc": lift >= GATE_LIFT_VS_COOC,
        "4_new_member": tf["ty_le_co_thanh_vien_moi"] >= GATE_NEW_MEMBER,
    }
    g["pass"] = all(v_ for k, v_ in g.items() if k != "pass")
    rep["lift_vs_cooc"] = round(lift, 3) if lift != float("inf") else None
    rep["gate"] = g
    log(f"CỔNG {g} · lift={rep['lift_vs_cooc']}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(
        json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    log(f"ghi {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
