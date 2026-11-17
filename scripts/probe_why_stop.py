#!/usr/bin/env python
"""P70 (TK_045 §13.36) — **VÌ SAO CHUỖI DỪNG SỚM**.

**Vì sao.** §13.35 (P69, v12, 400 đề): **833 lần phát / 400 đề = 2,08 mệnh đề mỗi đề**, trong
khi `max_clauses = 5`. Cộng `dead_rate` 81% của cổng fold (§12.154). Suốt §13 mọi vòng đo
*chọn cái gì*; **chưa vòng nào** đo *vì sao dừng*. Đây là con số lớn nhất còn lại.

**Bốn lối ra khả dĩ của vòng soạn** (`compose_chain` nay tự khai báo qua `stats["stop"]`):

| mã | nghĩa | nếu chi phối thì bệnh nằm ở |
|---|---|---|
| `du_max` | chạy hết `max_clauses` | **không tắc** — chuỗi đủ dài |
| `khong_neo` | không còn token nội dung làm neo | `gidf_min` / lọc nội dung |
| `khong_ung_vien` | neo có, `W_skip` không cho token-mở nào | **truy hồi** (đồ thị thưa) |
| `loai_het` | có ứng viên nhưng bị loại sạch | **bộ lọc**: cổng / trùng / dính |

Với `loai_het` còn tách được thủ phạm: `n_gate` (cổng phủ) · `n_dup` (đã dùng) · `n_glue`
(không phải biên từ).

**Đối chứng bắt buộc:** chạy lại **cùng đề, KHÔNG cổng**. Nếu bỏ cổng mà chuỗi vẫn dừng sớm
⇒ bệnh ở **truy hồi**, không phải ở cổng — và mọi việc chỉnh cổng đều vô ích.

Script chỉ đo và in. Không sửa cơ chế.

Chạy (mini — đủ để trả lời vì đây là câu hỏi CƠ CHẾ, không phải định cỡ):
  PYTHONPATH=src python scripts/probe_why_stop.py --n-prompt 300
Chạy (v12 — cần máy có RAM):
  … --checkpoint checkpoints/network_v12_sent.pkl --n-prompt 400
"""
from __future__ import annotations

import argparse
import gc
import json
import random
import statistics
import sys
import time
from collections import Counter
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
from nn_aivie.hca.w_skip import ensure_w_skip  # noqa: E402
from nn_aivie.util.progress import get_run_logger  # noqa: E402

STOPS = ("du_max", "khong_neo", "khong_ung_vien", "loai_het")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/mini_sent_ep_qa.pkl")
    ap.add_argument("--n-prompt", type=int, default=300)
    ap.add_argument("--max-clauses", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="out/probe_why_stop.json")
    args = ap.parse_args()

    logger = get_run_logger("probe_why_stop")
    t0 = time.time()
    logger.log(f"P70 vì sao dừng · {args.checkpoint} · {args.n_prompt} đề")

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

    def run(with_gate: bool) -> dict:
        stops: Counter[str] = Counter()
        lens: list[int] = []
        gate_r = dup_r = glue_r = 0
        pools: list[int] = []
        for i, p in enumerate(prompts):
            if i and i % 100 == 0:
                logger.log(f"  {'cổng' if with_gate else 'KHÔNG cổng'} {i}/{len(prompts)} "
                           f"({time.time()-t0:.0f}s)")
            gate = CoverageGate(
                theme_folds=frozenset(content_folds(state, p.tokens)), anchor_min_folds=1,
            ) if with_gate else None
            _ch, st = compose_chain(
                state, layer, ws, p.tokens, max_clauses=args.max_clauses,
                gate=gate, rng=random.Random(args.seed),
            )
            stops[st["stop"]] += 1
            lens.append(st["n_clauses"])
            gate_r += st["n_gate"]
            dup_r += st["n_dup"]
            glue_r += st["n_glue"]
            if st["stop"] == "loai_het":
                pools.append(st["pool_at_stop"])
        n = len(prompts)
        return {
            "stops": {k: stops.get(k, 0) for k in STOPS},
            "stop_rate": {k: stops.get(k, 0) / n for k in STOPS},
            "menh_de_tb": sum(lens) / n,
            "chuoi_rong": sum(1 for x in lens if x == 0) / n,
            "loai_boi": {"cong": gate_r, "trung": dup_r, "dinh": glue_r},
            "bể_luc_tac_trung_vi": statistics.median(pools) if pools else 0,
        }

    have = run(True)
    none = run(False)

    print(f"\n{'lý do dừng':<20}{'CÓ CỔNG':>12}{'KHÔNG CỔNG':>14}")
    for k in STOPS:
        print(f"{k:<20}{have['stop_rate'][k]:>11.1%}{none['stop_rate'][k]:>14.1%}")
    print(f"{'mệnh đề TB / đề':<20}{have['menh_de_tb']:>11.2f}{none['menh_de_tb']:>14.2f}")
    print(f"{'chuỗi RỖNG':<20}{have['chuoi_rong']:>11.1%}{none['chuoi_rong']:>14.1%}")

    lb = have["loai_boi"]
    tot = sum(lb.values()) or 1
    print("\nKhi có ứng viên mà vẫn loại (bản CÓ CỔNG) — thủ phạm:")
    for k, v in lb.items():
        print(f"  {k:<8}{v:>9,} ({v / tot:5.1%})")
    print(f"  bể lúc tắc, trung vị: {have['bể_luc_tac_trung_vi']} token-mở")

    out = {"checkpoint": args.checkpoint, "n_prompt": len(prompts),
           "max_clauses": args.max_clauses, "co_cong": have, "khong_cong": none,
           "seconds": round(time.time() - t0, 1)}
    op = _ROOT / args.out
    op.parent.mkdir(parents=True, exist_ok=True)
    op.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    logger.log(f"→ {args.out} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
