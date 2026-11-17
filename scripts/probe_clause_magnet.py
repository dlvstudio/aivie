#!/usr/bin/env python
"""P69 (TK_045 §13.35) — **MỆNH ĐỀ NAM CHÂM RÁC**: một nhúm mệnh đề chiếm phần lớn lần phát?

**Vì sao.** §13.34: cụm `km 1661+100 QL1` là **hapax** trong corpus (1/829.513 đơn vị) mà xuất
hiện ở **10/104 đề** khác nhau khi sinh. Chẩn: nó không thắng vì hiếm — nó thắng vì fold của
nó là **rác §13.20** (`năm`, chữ số) nên **khớp với gần như mọi đề**.

Nếu đúng, phân bố "mỗi mệnh đề được phát ra ở bao nhiêu ĐỀ khác nhau" phải **lệch nặng**, và
đó là mục tiêu can thiệp **cụ thể, đếm được** — khác mọi thứ §13 đã thử, vì nó tấn công *cái
bị chọn quá nhiều*, không phải *cách chọn*.

**Đối chứng** (bắt buộc — §13.30 dạy: đừng đo một đại lượng mà không hỏi nó là bằng chứng cho
gì): chạy lại **cùng đề, cùng cổng**, nhưng **xáo trật tự** ứng viên bằng `join_rescore` ngẫu
nhiên. Chênh giữa hai bên tách được hai nguyên nhân:

- tập trung ở **cả hai** ⇒ do **bể ứng viên / cổng** (cái gì tới được), không phải do xếp hạng;
- tập trung **chỉ ở bản thật** ⇒ do **khâu xếp hạng** — đúng chẩn đoán nam châm.

Script chỉ **đo và in**. Không sửa cơ chế, không chỉnh tham số.

Chạy (v12 — cần máy có RAM, xem PHẦN G):
  PYTHONPATH=src .venv/bin/python scripts/probe_clause_magnet.py \\
      --checkpoint checkpoints/network_v12_sent.pkl --n-prompt 400
Smoke (mini):
  PYTHONPATH=src python scripts/probe_clause_magnet.py --n-prompt 40
"""
from __future__ import annotations

import argparse
import gc
import json
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from nn_aivie.core.checkpoint_io import load_checkpoint  # noqa: E402
from nn_aivie.hca.clause_layer import (  # noqa: E402
    CoverageGate,
    build_clause_layer,
    compose_chain,
    content_folds,
)
from nn_aivie.hca.readout import decode_surface  # noqa: E402
from nn_aivie.hca.w_skip import ensure_w_skip  # noqa: E402
from nn_aivie.util.progress import get_run_logger  # noqa: E402

# --- CỔNG G-magnet, nêu TRƯỚC khi chạy (§13.35) ------------------------------------------
# Hiệu chỉnh từ quan sát v12 THẬT (§13.34: cụm hapax `km 1661+100 QL1` phát ở 10/104 đề =
# 9,6%), KHÔNG từ mini — §13.32 đã dạy: mini dùng để BÁC, không dùng để ĐỊNH CỠ. Và smoke mini
# ở đây **không kiểm được** câu hỏi này: 103 lần phát trên 10.283 mệnh đề thì trùng lặp gần
# như không xảy ra dù cơ chế có thật. Smoke chỉ chứng minh script chạy thông.
GATE_MAX_PROMPT_SHARE = 0.05  # (1) CHÍNH: có ≥1 mệnh đề phát ở ≥5% số đề (nửa mức đã thấy)
GATE_TOP10_SHARE = 0.10       # (2) top-10 mệnh đề chiếm ≥10% tổng lần phát
GATE_RATIO_VS_CTRL = 1.5      # (3) QUY NHÂN: ≥1.5× đối chứng xáo ngẫu nhiên ⇒ do XẾP HẠNG


def _key(c) -> tuple[int, int]:
    return (int(c.parent_idx), int(c.pos_in_parent))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/mini_sent_ep_qa.pkl")
    ap.add_argument("--n-prompt", type=int, default=400)
    ap.add_argument("--max-clauses", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="out/probe_clause_magnet.json")
    args = ap.parse_args()

    logger = get_run_logger("probe_clause_magnet")
    t0 = time.time()
    logger.log(f"P69 nam châm rác · {args.checkpoint} · {args.n_prompt} đề")
    logger.log(f"cổng: top10≥{GATE_TOP10_SHARE:.0%} · ≥{GATE_RATIO_VS_CTRL}× đối chứng · "
               f"đỉnh ≥{GATE_MAX_PROMPT_SHARE:.0%} số đề")

    gc.disable()
    state, _ = load_checkpoint(str(_ROOT / args.checkpoint))
    gc.enable()
    layer = build_clause_layer(state, pmi_theta=0.0, logger=logger)
    ws = ensure_w_skip(state, d_max=8, min_count=1, content_only=False)
    logger.log(f"tầng mệnh đề {len(layer):,} ({time.time()-t0:.0f}s)")

    rng = random.Random(args.seed)
    openers = [c for c in layer.clauses if c.opens_parent and len(c.tokens) >= 4]
    rng.shuffle(openers)
    prompts = openers[: args.n_prompt]

    def run(shuffle: bool) -> tuple[Counter, dict[tuple[int, int], set[int]], int]:
        """Trả (đếm lần phát, mệnh đề → tập ĐỀ đã phát nó, tổng lần phát)."""
        emit: Counter = Counter()
        seen: dict[tuple[int, int], set[int]] = defaultdict(set)
        total = 0
        noise = random.Random(args.seed + 1)
        rs = (lambda _a, _b: noise.random()) if shuffle else None
        for i, p in enumerate(prompts):
            if i and i % 50 == 0:
                logger.log(f"  {'ĐC' if shuffle else 'thật'} {i}/{len(prompts)} "
                           f"({time.time()-t0:.0f}s)")
            gate = CoverageGate(
                theme_folds=frozenset(content_folds(state, p.tokens)), anchor_min_folds=1)
            ch, _ = compose_chain(
                state, layer, ws, p.tokens, max_clauses=args.max_clauses,
                gate=gate, rng=random.Random(args.seed), join_rescore=rs,
            )
            for c in ch:
                k = _key(c)
                emit[k] += 1
                seen[k].add(i)
                total += 1
        return emit, seen, total

    def summarize(emit: Counter, seen: dict, total: int) -> dict:
        if not total:
            return {"tong_lan_phat": 0}
        top = emit.most_common(10)
        top10 = sum(n for _, n in top) / total
        share = {k: len(v) / len(prompts) for k, v in seen.items()}
        peak = max(share.values()) if share else 0.0
        n_ge5 = sum(1 for v in share.values() if v >= GATE_MAX_PROMPT_SHARE)
        return {
            "tong_lan_phat": total,
            "menh_de_khac_nhau": len(emit),
            "top10_share": top10,
            "dinh_ty_le_de": peak,
            "so_menh_de_tren_5pc_de": n_ge5,
        }

    emit_r, seen_r, tot_r = run(shuffle=False)
    real = summarize(emit_r, seen_r, tot_r)
    emit_c, seen_c, tot_c = run(shuffle=True)
    ctrl = summarize(emit_c, seen_c, tot_c)

    print(f"\n{'':<26}{'THẬT':>10}{'ĐỐI CHỨNG':>12}")
    for k in ("tong_lan_phat", "menh_de_khac_nhau", "top10_share",
              "dinh_ty_le_de", "so_menh_de_tren_5pc_de"):
        a, b = real.get(k, 0), ctrl.get(k, 0)
        f = (lambda v: f"{v:.1%}") if isinstance(a, float) else (lambda v: f"{v:,}")
        print(f"{k:<26}{f(a):>10}{f(b):>12}")

    ratio = (real["top10_share"] / ctrl["top10_share"]) if ctrl.get("top10_share") else 0.0
    g1 = real.get("top10_share", 0) >= GATE_TOP10_SHARE
    g2 = ratio >= GATE_RATIO_VS_CTRL
    g3 = real.get("dinh_ty_le_de", 0) >= GATE_MAX_PROMPT_SHARE
    print(f"\ntỷ số top10 thật/đối chứng = {ratio:.2f}×")
    print(f"CỔNG G-magnet: (1) {'✅' if g1 else '❌'} · (2) {'✅' if g2 else '❌'} · "
          f"(3) {'✅' if g3 else '❌'}  ⇒ {'ĐẠT' if (g1 and g2 and g3) else 'TRƯỢT'}")

    print("\nTOP mệnh đề bị phát nhiều nhất (bản THẬT):")
    for k, n in emit_r.most_common(8):
        cl = next((c for c in layer.clauses if _key(c) == k), None)
        txt = decode_surface(state, [int(t) for t in cl.tokens])[:70] if cl else "?"
        print(f"  {n:>4} lần · {len(seen_r[k]):>3} đề · {txt}")

    out = {"checkpoint": args.checkpoint, "n_prompt": len(prompts),
           "gate": {"top10_share_min": GATE_TOP10_SHARE, "ratio_min": GATE_RATIO_VS_CTRL,
                    "peak_min": GATE_MAX_PROMPT_SHARE},
           "that": real, "doi_chung": ctrl, "ty_so": ratio,
           "gate_pass": bool(g1 and g2 and g3),
           "top": [{"lan_phat": n, "so_de": len(seen_r[k])} for k, n in emit_r.most_common(20)],
           "seconds": round(time.time() - t0, 1)}
    op = _ROOT / args.out
    op.parent.mkdir(parents=True, exist_ok=True)
    op.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    logger.log(f"→ {args.out} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
