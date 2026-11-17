#!/usr/bin/env python
"""P80 (TK_045 §13.47) — **MIỀN VĂN BẢN**: hệ có tín hiệu nào cho "cùng miền" không?

**Vì sao.** §13.46: siết điều kiện mối nối ở **bề mặt** thất bại (cái thứ mười trong họ đó),
dù nó xoá sạch 46,6% ghép chưa-từng-thấy. Giám khảo nói bệnh thật là *"bám vài chữ cuối của đề
rồi **nhảy sang một miền văn bản hoàn toàn khác**"* — tiên hiệp ghép vào bài kỹ thuật —
chiếm **~3/4** mẫu hỏng **có chữ**.

`W_skip` và `W_seq` đều là thống kê **hai token cục bộ**; không cái nào mang **miền**. Câu hỏi
vòng này: **miền có phải một đại lượng ĐẾM ĐƯỢC không**, và mối nối sinh lệch bao xa khỏi mối
nối thật theo đại lượng đó.

**Đối chứng phải là CROSS-PARENT.** §13.44 đã mắc lỗi vòng vo vì lấy biên trong-cùng-cha làm
chuẩn (biên đó có pmi thấp *theo định nghĩa* của bộ cắt). Lần này chuẩn là **hai cha LIÊN TIẾP
trong cùng một episode** — tức hai câu thật, khác cha, mà người viết đặt cạnh nhau. Đó đúng là
việc hệ đang cố làm.

Ba nhóm, cùng một thước:
- **SINH**  — (cha của A, cha của B) ở mối nối do hệ chọn;
- **THẬT**  — hai cha liên tiếp trong một episode;
- **NGẪU NHIÊN** — hai cha bốc ngẫu nhiên (sàn).

Thước: **Jaccard trên fold nội dung** của hai cha. Thuần đếm ⇒ Luật 0 sạch.

Chạy:
  PYTHONPATH=src python scripts/probe_domain_match.py --n-prompt 600
"""
from __future__ import annotations

import argparse
import gc
import json
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
)
from nn_aivie.hca.w_skip import ensure_w_skip  # noqa: E402
from nn_aivie.util.progress import get_run_logger  # noqa: E402


def summ(xs: list[float]) -> dict:
    if not xs:
        return {}
    s = sorted(xs)
    return {"n": len(xs), "trung_binh": statistics.mean(s), "trung_vi": statistics.median(s),
            "p25": s[len(s) // 4], "p75": s[3 * len(s) // 4],
            "ty_le_bang_0": sum(1 for x in s if x == 0.0) / len(s)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/mini_sent_ep_qa.pkl")
    ap.add_argument("--n-prompt", type=int, default=600)
    ap.add_argument("--max-clauses", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="out/probe_domain_match.json")
    args = ap.parse_args()

    logger = get_run_logger("probe_domain_match")
    t0 = time.time()
    logger.log(f"P80 miền văn bản · {args.checkpoint} · {args.n_prompt} đề")

    gc.disable()
    state, _ = load_checkpoint(str(_ROOT / args.checkpoint))
    gc.enable()
    layer = build_clause_layer(state, pmi_theta=0.0, logger=logger)
    ws = ensure_w_skip(state, d_max=8, min_count=1, content_only=False)
    ps = state.parents
    logger.log(f"tầng mệnh đề {len(layer):,} · cha {len(ps.parents):,} · "
               f"episode {len(ps.episodes):,} ({time.time()-t0:.0f}s)")

    _fold: dict[int, frozenset] = {}

    def folds_of(pi: int) -> frozenset:
        f = _fold.get(pi)
        if f is None:
            f = frozenset(content_folds(state, ps.parents[pi].emit_ids))
            _fold[pi] = f
        return f

    def jac(a: int, b: int) -> float | None:
        fa, fb = folds_of(a), folds_of(b)
        if not fa or not fb:
            return None
        return len(fa & fb) / len(fa | fb)

    # --- SINH -------------------------------------------------------------------------
    rng = random.Random(args.seed)
    openers = [c for c in layer.clauses if c.opens_parent and len(c.tokens) >= 4]
    rng.shuffle(openers)
    gen: list[float] = []
    for i, p in enumerate(openers[: args.n_prompt]):
        if i and i % 200 == 0:
            logger.log(f"  sinh {i}/{args.n_prompt} ({time.time()-t0:.0f}s)")
        gate = CoverageGate(
            theme_folds=frozenset(content_folds(state, p.tokens)), anchor_min_folds=1)
        ch, _ = compose_chain(
            state, layer, ws, p.tokens, max_clauses=args.max_clauses,
            gate=gate, rng=random.Random(args.seed))
        # mối nối đầu tiên là ĐỀ → mệnh đề 1: cha của đề cũng là một cha thật
        seq = [p] + list(ch)
        for a, b in zip(seq, seq[1:]):
            if a.parent_idx != b.parent_idx:
                v = jac(a.parent_idx, b.parent_idx)
                if v is not None:
                    gen.append(v)

    # --- THẬT: hai cha LIÊN TIẾP trong một episode --------------------------------------
    real: list[float] = []
    for ep in ps.episodes:
        mem = [int(m[0]) for m in ep.members]
        for a, b in zip(mem, mem[1:]):
            if a != b:
                v = jac(a, b)
                if v is not None:
                    real.append(v)

    # --- NGẪU NHIÊN: sàn ----------------------------------------------------------------
    rr = random.Random(args.seed + 3)
    npar = len(ps.parents)
    rand: list[float] = []
    for _ in range(min(20000, max(2000, len(real)))):
        a, b = rr.randrange(npar), rr.randrange(npar)
        if a != b:
            v = jac(a, b)
            if v is not None:
                rand.append(v)

    g, r, z = summ(gen), summ(real), summ(rand)
    print(f"\n{'':<20}{'SINH':>10}{'THẬT':>10}{'NGẪU NHIÊN':>13}")
    for k, lab in (("n", "số mối nối"), ("trung_binh", "Jaccard TB"),
                   ("trung_vi", "Jaccard trung vị"), ("p75", "Jaccard p75"),
                   ("ty_le_bang_0", "KHÔNG fold chung")):
        f = (lambda v: f"{v:,}") if k == "n" else (
            (lambda v: f"{v:.1%}") if k.startswith("ty_le") else (lambda v: f"{v:.4f}"))
        print(f"{lab:<20}{f(g.get(k,0)):>10}{f(r.get(k,0)):>10}{f(z.get(k,0)):>13}")

    gm, rm, zm = g.get("trung_binh", 0), r.get("trung_binh", 0), z.get("trung_binh", 0)
    # SINH nằm ở đâu giữa sàn ngẫu nhiên và trần văn thật? 0 = như ngẫu nhiên, 1 = như thật.
    pos = (gm - zm) / (rm - zm) if (rm - zm) else float("nan")
    print(f"\nSINH nằm ở {pos:.0%} quãng đường từ NGẪU NHIÊN → THẬT")
    print("⇒ " + ("miền là đại lượng ĐẾM ĐƯỢC, và hệ đang gần NGẪU NHIÊN" if pos < 0.35 else
                  "hệ đã giữ được miền đáng kể — chẩn 'đứt mạch miền' YẾU đi"))

    out = {"checkpoint": args.checkpoint, "sinh": g, "that": r, "ngau_nhien": z,
           "vi_tri_sinh": pos, "seconds": round(time.time() - t0, 1)}
    op = _ROOT / args.out
    op.parent.mkdir(parents=True, exist_ok=True)
    op.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    logger.log(f"→ {args.out}")


if __name__ == "__main__":
    main()
