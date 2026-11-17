#!/usr/bin/env python
"""P72 (TK_045 §13.38) — **LẶP CỤM ĐẾN CẠN NGHĨA**: đo kiểu hỏng 3 của giám khảo.

**Vì sao.** §13.33 giám khảo mù lập bản kê 5 kiểu hỏng. Bốn kiểu đã đo (§13.34–§13.37). Kiểu 3
là món **cuối chưa ai đo**: `dữ liệu được lưu trữ. dữ liệu có thể được lưu trữ` ·
`học sinh học sinh học sinh` · nhồi `thời gian` sáu lần thành **`thời gian khổ`**.

**Đối chứng — trong cùng đồ thị, không cần corpus ngoài.** Văn thật cũng lặp (`của`, `và`,
`là`); không có đối chứng thì mọi con số lặp đều "trông cao". Đối chứng ở đây là **văn THẬT
lấy từ chính mạng**: k mệnh đề **liên tiếp của cùng một cha** — cùng bộ tách, cùng tokenizer,
cùng phân bố độ dài, và **đúng là văn người viết**.

⚠️ Bài học §13.35 (lỗi mẫu số): **không** đo trên file cặp chấm mù — nó đã lọc. Probe này sinh
mới trên **toàn bộ** bộ đề, và báo cả tỷ lệ đề rỗng để mẫu số minh bạch.

Chạy:
  PYTHONPATH=src python scripts/probe_repetition.py --n-prompt 300
"""
from __future__ import annotations

import argparse
import gc
import json
import random
import re
import statistics
import sys
import time
import unicodedata
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
from nn_aivie.hca.readout import decode_surface  # noqa: E402
from nn_aivie.hca.w_skip import ensure_w_skip  # noqa: E402
from nn_aivie.util.progress import get_run_logger  # noqa: E402

_WORD = re.compile(r"[^\W_]+", re.UNICODE)


def words(s: str) -> list[str]:
    return _WORD.findall(unicodedata.normalize("NFC", s).lower())


def metrics(text: str) -> dict | None:
    w = words(text)
    if len(w) < 8:            # quá ngắn thì tỷ lệ lặp là nhiễu
        return None
    bi = [(w[i], w[i + 1]) for i in range(len(w) - 1)]
    cnt = Counter(w)
    return {
        # distinct-2 THẤP = lặp nhiều
        "distinct2": len(set(bi)) / len(bi),
        # chữ bị nhắc lại nhiều nhất trong MỘT đầu ra (`thời gian` ×6 của §13.33)
        "max_lap_1_chu": max(cnt.values()),
        # có cặp 2-gram nào xuất hiện ≥2 lần không
        "co_bigram_lap": 1.0 if len(set(bi)) < len(bi) else 0.0,
        # lặp KỀ NHAU — `học sinh học sinh`
        "co_lap_ke": 1.0 if any(w[i] == w[i + 1] for i in range(len(w) - 1)) else 0.0,
        "n_chu": len(w),
    }


def agg(rows: list[dict]) -> dict:
    if not rows:
        return {}
    out = {"n_mau": len(rows)}
    for k in ("distinct2", "max_lap_1_chu", "co_bigram_lap", "co_lap_ke", "n_chu"):
        out[k] = statistics.mean(r[k] for r in rows)
    out["distinct2_p25"] = sorted(r["distinct2"] for r in rows)[len(rows) // 4]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/mini_sent_ep_qa.pkl")
    ap.add_argument("--n-prompt", type=int, default=300)
    ap.add_argument("--max-clauses", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="out/probe_repetition.json")
    args = ap.parse_args()

    logger = get_run_logger("probe_repetition")
    t0 = time.time()
    logger.log(f"P72 lặp cụm · {args.checkpoint} · {args.n_prompt} đề")

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

    def dec(clauses) -> str:
        return decode_surface(state, [int(t) for c in clauses for t in c.tokens]).strip()

    gen_rows, ctrl_rows = [], []
    n_empty = n_short = 0
    n_dup_moi_noi = n_moi_noi = n_dup_trong = n_cap_trong = 0
    # cha có ≥2 mệnh đề — kho văn THẬT cho đối chứng
    multi = [p for p, ids in layer.by_parent.items() if len(ids) >= 2]
    crng = random.Random(args.seed + 7)

    for i, p in enumerate(prompts):
        if i and i % 100 == 0:
            logger.log(f"  {i}/{len(prompts)} ({time.time()-t0:.0f}s)")
        gate = CoverageGate(
            theme_folds=frozenset(content_folds(state, p.tokens)), anchor_min_folds=1)
        ch, _st = compose_chain(
            state, layer, ws, p.tokens, max_clauses=args.max_clauses,
            gate=gate, rng=random.Random(args.seed),
        )
        if not ch:
            n_empty += 1
            continue
        m = metrics(dec(ch))
        if m is None:
            n_short += 1
            continue
        gen_rows.append(m)
        # Lặp KỀ nằm Ở ĐÂU? Nếu nó tụ ở **mối nối** (chữ cuối mệnh đề A == chữ đầu mệnh
        # đề B) thì đó là khuyết tật của khâu NỐI, sửa được; nếu rải đều trong lòng mệnh
        # đề thì nó là văn nguồn, không phải lỗi hệ.
        wc = [words(dec([c])) for c in ch]
        for j in range(len(wc) - 1):
            if wc[j] and wc[j + 1] and wc[j][-1] == wc[j + 1][0]:
                n_dup_moi_noi += 1
            n_moi_noi += 1
        for ws_ in wc:
            n_dup_trong += sum(1 for x in range(len(ws_) - 1) if ws_[x] == ws_[x + 1])
            n_cap_trong += max(0, len(ws_) - 1)
        # đối chứng: k mệnh đề LIÊN TIẾP của một cha thật, cùng k
        k = len(ch)
        for _ in range(12):
            ids = layer.by_parent[multi[crng.randrange(len(multi))]]
            if len(ids) >= k:
                s = crng.randrange(0, len(ids) - k + 1)
                mc = metrics(dec([layer.clauses[j] for j in ids[s:s + k]]))
                if mc is not None:
                    ctrl_rows.append(mc)
                break

    gen, ctrl = agg(gen_rows), agg(ctrl_rows)
    print(f"\nđề: {len(prompts)} · rỗng {n_empty} ({n_empty/len(prompts):.1%}) · "
          f"quá ngắn {n_short} · đo được {len(gen_rows)}")
    print(f"\n{'':<20}{'SINH':>10}{'VĂN THẬT':>12}")
    for k, lab in (("distinct2", "distinct-2"), ("distinct2_p25", "distinct-2 p25"),
                   ("max_lap_1_chu", "lặp 1 chữ tối đa"), ("co_bigram_lap", "có bigram lặp"),
                   ("co_lap_ke", "có lặp KỀ"), ("n_chu", "số chữ")):
        a, b = gen.get(k, 0), ctrl.get(k, 0)
        f = (lambda v: f"{v:.1%}") if k.startswith("co_") else (lambda v: f"{v:.2f}")
        print(f"{lab:<20}{f(a):>10}{f(b):>12}")

    r_join = n_dup_moi_noi / n_moi_noi if n_moi_noi else 0.0
    r_in = n_dup_trong / n_cap_trong if n_cap_trong else 0.0
    print(f"\nLặp KỀ nằm ở đâu:")
    print(f"  tại MỐI NỐI  {n_dup_moi_noi:>4}/{n_moi_noi:<5} = {r_join:.2%}")
    print(f"  TRONG mệnh đề {n_dup_trong:>4}/{n_cap_trong:<5} = {r_in:.2%}")
    print(f"  ⇒ mối nối gấp {(r_join/r_in if r_in else 0):.1f}× trong lòng mệnh đề")

    worse = gen.get("distinct2", 1) < ctrl.get("distinct2", 1)
    print(f"\n⇒ {'SINH lặp NHIỀU HƠN văn thật' if worse else 'KHÔNG thấy lặp quá văn thật'}")

    out = {"checkpoint": args.checkpoint, "n_prompt": len(prompts), "n_empty": n_empty,
           "n_short": n_short, "sinh": gen, "van_that": ctrl,
           "sinh_lap_nhieu_hon": bool(worse),
           "lap_ke_moi_noi": r_join, "lap_ke_trong_menh_de": r_in,
           "n_moi_noi": n_moi_noi, "n_cap_trong": n_cap_trong, "seconds": round(time.time() - t0, 1)}
    op = _ROOT / args.out
    op.parent.mkdir(parents=True, exist_ok=True)
    op.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    logger.log(f"→ {args.out}")


if __name__ == "__main__":
    main()
