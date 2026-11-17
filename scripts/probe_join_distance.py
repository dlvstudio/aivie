#!/usr/bin/env python
"""P78 (TK_045 §13.44) — mối nối hệ đang dùng **cách mối nối trong văn thật bao xa**?

**Vì sao.** §13.43 (mốc gốc tuyệt đối): **0,5%** đầu ra dùng được, và **55–60%** ca hỏng là
**nhảy chủ đề ngay tại mối nối**. Chín vòng xếp lại/chặn bớt trên `W_skip` đều hoà hoặc thua
⇒ vấn đề không ở *cách xếp* mà ở **đại lượng**. Vòng này đo đại lượng đó lệch bao xa.

**Giả thuyết H — có thể sai, và đo được:**

Tầng mệnh đề **cắt ở thung lũng PMI** (§13.6: biên mệnh đề = `pmi(a→b) ≤ θ`, tức chỗ liên kết
**YẾU nhất**). Nhưng mối nối lại **chọn theo `W_skip` MẠNH nhất**. Nếu H đúng thì hệ đang
**nối ở đúng chỗ văn thật NGẮT** — và đó là lý do cấu trúc khiến `quốc gia` + `gia vị` trông
hợp lệ với nó: hai âm tiết dính chặt là **từ**, không phải **biên**.

**Đối chứng nằm trong mạng:** mối nối THẬT = chuyển tiếp giữa hai mệnh đề **liên tiếp của cùng
một cha** — đúng chỗ người viết ngắt.

**Thiên lệch có lợi cho việc bác H, ghi rõ:** mối nối THẬT nằm ngay trong `W_seq` nên nó **tự
đếm chính nó** ⇒ pmi của nó bị **thổi LÊN**. Nếu đo ra mối nối SINH **vẫn cao hơn**, kết luận
càng chắc, vì thiên lệch chạy **ngược** hướng H. (Đây là bài học §13.39: đối chứng tự đếm.)

Chỉ đọc `W_seq` ⇒ Luật 0 sạch. Script chỉ đo.

Chạy:
  PYTHONPATH=src python scripts/probe_join_distance.py --n-prompt 600
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import random
import statistics
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from nn_aivie.core.checkpoint_io import load_checkpoint  # noqa: E402
from nn_aivie.hca.clause_layer import (  # noqa: E402
    CoverageGate,
    build_clause_layer,
    compose_chain,
    content_folds,
    ensure_boundary_signal,
)
from nn_aivie.hca.w_skip import ensure_w_skip  # noqa: E402
from nn_aivie.util.progress import get_run_logger  # noqa: E402


def summ(xs: list[float]) -> dict:
    fin = [x for x in xs if math.isfinite(x)]
    if not xs:
        return {}
    fin_sorted = sorted(fin)
    return {
        "n": len(xs),
        "ty_le_KHONG_chung_thuc": 1 - len(fin) / len(xs),   # pmi = −inf ⇒ cặp chưa từng thấy
        "trung_vi": statistics.median(fin_sorted) if fin else float("nan"),
        "p25": fin_sorted[len(fin_sorted) // 4] if fin else float("nan"),
        "p75": fin_sorted[3 * len(fin_sorted) // 4] if fin else float("nan"),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/mini_sent_ep_qa.pkl")
    ap.add_argument("--n-prompt", type=int, default=600)
    ap.add_argument("--max-clauses", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="out/probe_join_distance.json")
    args = ap.parse_args()

    logger = get_run_logger("probe_join_distance")
    t0 = time.time()
    logger.log(f"P78 khoảng cách mối nối · {args.checkpoint} · {args.n_prompt} đề")

    gc.disable()
    state, _ = load_checkpoint(str(_ROOT / args.checkpoint))
    gc.enable()
    layer = build_clause_layer(state, pmi_theta=0.0, logger=logger)
    ws = ensure_w_skip(state, d_max=8, min_count=1, content_only=False)
    bd = ensure_boundary_signal(state)
    logger.log(f"tầng mệnh đề {len(layer):,} ({time.time()-t0:.0f}s)")

    def trans(a, b) -> tuple[int, int]:
        """Chuyển tiếp BỀ MẶT: token CUỐI của A → token ĐẦU của B (không phải last_content).

        Dùng token cuối/đầu thật vì đó là chỗ hai mệnh đề **chạm nhau** khi đọc — chính chỗ
        `quốc gia` + `gia vị` dính lại thành từ.
        """
        return int(a.tokens[-1]), int(b.tokens[0])

    rng = random.Random(args.seed)
    openers = [c for c in layer.clauses if c.opens_parent and len(c.tokens) >= 4]
    rng.shuffle(openers)
    prompts = openers[: args.n_prompt]

    gen: list[float] = []
    for i, p in enumerate(prompts):
        if i and i % 200 == 0:
            logger.log(f"  sinh {i}/{len(prompts)} ({time.time()-t0:.0f}s)")
        gate = CoverageGate(
            theme_folds=frozenset(content_folds(state, p.tokens)), anchor_min_folds=1)
        ch, _ = compose_chain(
            state, layer, ws, p.tokens, max_clauses=args.max_clauses,
            gate=gate, rng=random.Random(args.seed),
        )
        for a, b in zip(ch, ch[1:]):
            if a.parent_idx != b.parent_idx:      # chỉ mối nối KHÁC CHA — đúng ca lúc sinh
                gen.append(bd.pmi(*trans(a, b)))

    real: list[float] = []
    for ids in layer.by_parent.values():
        if len(ids) < 2:
            continue
        cls = [layer.clauses[j] for j in ids]
        for a, b in zip(cls, cls[1:]):
            real.append(bd.pmi(*trans(a, b)))

    g, r = summ(gen), summ(real)
    print(f"\n{'':<26}{'SINH':>12}{'THẬT':>12}")
    for k, lab in (("n", "số mối nối"), ("ty_le_KHONG_chung_thuc", "cặp CHƯA từng thấy"),
                   ("p25", "pmi p25"), ("trung_vi", "pmi trung vị"), ("p75", "pmi p75")):
        a, b = g.get(k, 0), r.get(k, 0)
        f = ((lambda v: f"{v:,}") if k == "n" else
             (lambda v: f"{v:.1%}") if k.startswith("ty_le") else (lambda v: f"{v:+.2f}"))
        print(f"{lab:<26}{f(a):>12}{f(b):>12}")

    d = g.get("trung_vi", 0) - r.get("trung_vi", 0)
    print(f"\nchênh pmi trung vị (SINH − THẬT) = {d:+.2f} bit")
    print("⇒ " + ("H ĐỨNG: hệ nối ở chỗ liên kết MẠNH, văn thật ngắt ở chỗ YẾU"
                  if d > 0 else "H BỊ BÁC: mối nối sinh KHÔNG mạnh hơn biên thật"))
    print("   (thiên lệch chạy NGƯỢC: mối nối THẬT tự đếm chính nó nên pmi của nó bị thổi LÊN)")

    out = {"checkpoint": args.checkpoint, "n_prompt": len(prompts), "sinh": g, "that": r,
           "chenh_trung_vi": d, "H_dung": bool(d > 0), "seconds": round(time.time() - t0, 1)}
    op = _ROOT / args.out
    op.parent.mkdir(parents=True, exist_ok=True)
    op.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    logger.log(f"→ {args.out}")


if __name__ == "__main__":
    main()
