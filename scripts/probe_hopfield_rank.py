#!/usr/bin/env python
"""P48 (TK_045 §13 Trụ D) — Hopfield hiện đại vs `rank_matches` trên thu hồi cha nguồn.

**Dùng ĐÚNG bộ query của P31** (`probe_rank_matches_d.py`: cùng cách lấy `first_seq`, cùng
`seed`/`seed+1`, cùng `_expected_rank`/`_tie_stats`) ⇒ số so trực tiếp được với §12.126.

Bốn cấu hình:

| tên | là gì |
|---|---|
| `A_rank_matches` | **baseline production** — coverage phân tầng × distinct_ratio² × power × dir × priority |
| `B_hop_s1` | **ĐỐI CHỨNG**: cosine trần trên A (softmax đơn điệu ⇒ steps=1 KHÔNG phải Hopfield). Trả lời: đống cổng chỉnh tay có thắng cosine không? |
| `C_hop_s2` | Hopfield 2 bước — query được HOÀN NGUYÊN rồi khớp lại. Đây là chỗ duy nhất cơ chế có thể ăn |
| `D_hop_s3` | 3 bước — dò xem lặp thêm giúp hay làm mờ |

**CỔNG G-hop (nêu TRƯỚC, theo tiền lệ G-tie §12.126):** ở `q_len=3`, cấu hình Hopfield tốt
nhất so với `A_rank_matches` phải đạt **rank-1 ≥ +5pp** HOẶC **tie-rate giảm ≥20% tương đối**.

**Định kiến ÂM đã đo, ghi trước:** §12.118 (toán tử chuyển, Spearman +0.06 = làm mờ), §12.108
(tầm d), §0.W (*"Cặp gần-bất-ngờ > khả-đạt xa/nhiều-bước"*). `steps≥2` đi NGƯỢC định kiến này.
Lý do vẫn đo: các phép đó **không có nhiệt độ**; β là núm chống-mờ. β cao mà vẫn mờ ⇒ hướng
đóng, kết quả sạch.

Luật 0: chỉ đọc đồ thị, dạng đóng, không backprop.

Chạy:
  PYTHONPATH=src .venv/bin/python scripts/probe_hopfield_rank.py \\
      --checkpoint checkpoints/network_v12_sent.pkl
"""
from __future__ import annotations

import argparse
import gc
import json
import random
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from nn_aivie.core.checkpoint_io import load_checkpoint  # noqa: E402
from nn_aivie.hca.context_positional import is_structural_token  # noqa: E402
from nn_aivie.hca.hopfield import ensure_hopfield_memory, retrieve  # noqa: E402
from nn_aivie.hca.sent_intra import (  # noqa: E402
    _content_positions,
    iter_sentence_spans,
)
from nn_aivie.util.progress import get_run_logger  # noqa: E402

GATE_R1_PP = 5.0        # rank-1 ≥ +5pp
GATE_TIE_REL = 20.0     # hoặc tie-rate ↓ ≥20% tương đối …

# ⚠️ SỬA CỔNG 2026-08-02 sau lần chạy đầu (ghi lại, KHÔNG xoá dấu vết — §13.9):
#   Cổng gốc là `Δrank-1 ≥ +5pp` **HOẶC** `Δtie ≥ 20% rel`. OR trần ⇒ nó báo **ĐẠT** cho một
#   cấu hình làm rank-1 **TỤT 9.17pp**, chỉ vì tie-rate về 0. Sai ở đâu: tiêu chí tie mượn từ
#   G-tie §12.126, nơi nó so HAI BIẾN THỂ CỦA CÙNG một bộ chấm — ở đó "hoà" nghĩa là thật sự
#   không phân biệt được. Ở đây nó so hai HỌ bộ chấm khác nhau: cosine trên trọng số liên tục
#   **gần như không bao giờ hoà** — tie=0 là **hiện vật của kiểu số**, không mang tin gì về
#   chất lượng truy hồi. ⇒ **Tiêu chí tie KHÔNG chuyển được qua họ bộ chấm khác.**
#   Sửa: tie chỉ tính KHI chất lượng không tụt.
GATE_R1_NO_REGRESS_PP = -0.5   # dung sai nhiễu


def _gate_pass(d_r1: float, d_tie: float) -> bool:
    if d_r1 >= GATE_R1_PP:
        return True
    return d_tie >= GATE_TIE_REL and d_r1 >= GATE_R1_NO_REGRESS_PP


def _positive(sc) -> bool:
    """Điểm có thể là float (A–D) hoặc CẶP (chính, hopfield) ở cấu hình E."""
    return (sc[0] > 0) if isinstance(sc, tuple) else (sc > 0)


def _expected_rank(ranked, src: int, n_par: int) -> float:
    """Hạng kỳ vọng có xét hoà: #(cao hơn) + (#hoà+1)/2. Ngoài list → n_par. (≡ P31)"""
    by_pi = {pi: sc for pi, sc in ranked}
    ssrc = by_pi.get(src)
    if ssrc is None or not _positive(ssrc):
        return float(n_par)
    higher = sum(1 for sc in by_pi.values() if sc > ssrc)
    ties = sum(1 for sc in by_pi.values() if sc == ssrc)
    return higher + (ties + 1) / 2.0


def _tie_stats(ranked: list[tuple[int, float]]) -> tuple[bool, int]:
    if not ranked:
        return False, 0
    top = ranked[0][1]
    n = sum(1 for _, sc in ranked if sc == top)
    return n >= 2, n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--n-query", type=int, default=800)
    ap.add_argument("--q-len", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--betas", default="32.0",
                    help="β=32 là bản thắng trên mini (§13.9). Sweep đầy đủ chỉ nên "
                         "chạy trên mini — ở v12 mỗi β thêm 4 cấu hình, xem --only")
    ap.add_argument("--only", default="",
                    help="lọc cấu hình theo tiền tố, vd 'A,E' = chỉ baseline + tiebreak "
                         "(đủ trả lời câu hỏi v12, rẻ hơn ~2.5×)")
    ap.add_argument("--top-k", type=int, default=64)
    ap.add_argument("--top-m", type=int, default=64)
    ap.add_argument("--out-json", default="out/probe_hopfield_rank.json")
    args = ap.parse_args()

    logger = get_run_logger("probe_hopfield_rank")
    t0 = time.time()
    logger.log(f"P48 Trụ D — Hopfield vs rank_matches · {args.checkpoint}")
    logger.log(f"CỔNG G-hop (nêu TRƯỚC): rank-1 ≥ +{GATE_R1_PP}pp HOẶC "
               f"tie-rate ↓ ≥{GATE_TIE_REL}% tương đối, so A_rank_matches @ q_len=3")

    gc.disable()
    state, _ = load_checkpoint(str(_ROOT / args.checkpoint))
    gc.enable()
    v = state.vocab
    ps = state.parents
    n_par = len(ps.parents)
    struct = [is_structural_token(v, i) for i in range(len(v))]
    logger.log(f"load {time.time()-t0:.0f}s · {n_par:,} cha")

    # --- bộ query: y hệt P31 (probe_rank_matches_d) ---
    first_seq: list[list[int]] = []
    for rec in ps.parents:
        sq = list(rec.answer_ids if (rec.is_qa and rec.answer_ids) else rec.emit_ids)
        toks: list[int] = []
        for sp in iter_sentence_spans(sq, v):
            toks += [int(t) for t in _content_positions(
                sp, v, content_only=True, struct_mask=struct)]
        first_seq.append(toks)

    idx = list(range(n_par))
    random.Random(args.seed).shuffle(idx)
    q_ids = [i for i in idx if len(first_seq[i]) >= args.q_len][: args.n_query]
    rng = random.Random(args.seed + 1)
    queries: list[tuple[int, list[int]]] = []
    for src in q_ids:
        seq = first_seq[src]
        st_ = rng.randint(0, max(0, len(seq) - args.q_len))
        queries.append((src, seq[st_: st_ + args.q_len]))
    logger.log(f"{len(queries)} query (≥{args.q_len} token nội dung) — cùng seed với P31")

    mem = ensure_hopfield_memory(state)
    logger.log(f"HopfieldMemory: A {mem.A_csr.shape} nnz={mem.A_csr.nnz:,} "
               f"({time.time()-t0:.0f}s)")

    def _idf_w(q: list[int]) -> list[float]:
        return [float(ps._gidf(int(t))) for t in q]  # noqa: SLF001

    def _tiebreak(ranked, query, kn):
        """E — GIỮ NGUYÊN thứ hạng rank_matches, chỉ xếp lại NHÓM HOÀ bằng điểm Hopfield.

        Không đụng query không hoà (theo cấu tạo) ⇒ chỉ có thể ăn ở đúng 38% query mà
        §12.126 đã chỉ là nút thắt. Giữ thứ tự các bậc điểm; trong mỗi bậc xếp theo Hopfield.
        """
        if len(ranked) < 2:
            return ranked
        hop = dict(retrieve(
            mem, query, _idf_w(query), beta=kn["beta"], steps=kn["steps"],
            top_k=args.top_k, top_m=args.top_m, top_n=0,
        ))
        # Điểm = CẶP (chính, hopfield). Tuple so sánh được ⇒ `_expected_rank`/`_tie_stats`
        # thấy đúng: bậc chính bất biến, hoà chỉ còn khi CẢ HAI bằng nhau.
        out = [(pi, (float(sc), float(hop.get(pi, 0.0)))) for pi, sc in ranked]
        out.sort(key=lambda x: (-x[1][0], -x[1][1]))
        return out

    configs: list[tuple[str, dict]] = [("A_rank_matches", {})]
    for b in [float(x) for x in args.betas.split(",") if x.strip()]:
        configs.append((f"B_hop_s1_b{b:g}", {"steps": 1, "beta": b}))
        configs.append((f"C_hop_s2_b{b:g}", {"steps": 2, "beta": b}))
        configs.append((f"D_hop_s3_b{b:g}", {"steps": 3, "beta": b}))
        configs.append((f"E_tiebreak_s2_b{b:g}", {"steps": 2, "beta": b, "tiebreak": True}))

    if args.only.strip():
        keep = tuple(x.strip() for x in args.only.split(",") if x.strip())
        configs = [c for c in configs if c[0].startswith(keep)]
        if not any(n == "A_rank_matches" for n, _ in configs):
            configs.insert(0, ("A_rank_matches", {}))  # baseline luôn phải có
    logger.log(f"{len(configs)} cấu hình: " + ", ".join(n for n, _ in configs))

    results: dict[str, dict] = {}
    for name, kn in configs:
        r1 = mrr = 0.0
        n_tie = 0
        tie_sizes: list[int] = []
        miss = 0
        for qi, (src, query) in enumerate(queries, 1):
            if qi % 200 == 0:
                logger.log(f"  [{name}] {qi}/{len(queries)} ({time.time()-t0:.0f}s)")
            if kn.get("tiebreak"):
                ranked = _tiebreak(
                    ps.rank_matches(query, v, None, fuzzy_topk=0, top_n=n_par),
                    query, kn,
                )
            elif kn:
                ranked = retrieve(
                    mem, query, _idf_w(query),
                    beta=kn["beta"], steps=kn["steps"],
                    top_k=args.top_k, top_m=args.top_m, top_n=n_par,
                )
            else:
                ranked = ps.rank_matches(query, v, None, fuzzy_topk=0, top_n=n_par)
            er = _expected_rank(ranked, src, n_par)
            if er >= n_par:
                miss += 1
            r1 += 1.0 if er <= 1.5 else 0.0
            mrr += 1.0 / er
            is_tie, tsz = _tie_stats(ranked)
            if is_tie:
                n_tie += 1
            tie_sizes.append(tsz)
        m = len(queries)
        results[name] = {
            "r1": r1 / m, "mrr": mrr / m, "tie_rate": n_tie / m,
            "mean_tie": sum(tie_sizes) / m if m else 0.0, "miss": miss, "m": m,
        }
        r = results[name]
        logger.log(f"[{time.time()-t0:.0f}s] {name}: rank-1={100*r['r1']:.2f}% "
                   f"MRR={r['mrr']:.3f} tie={100*r['tie_rate']:.1f}% "
                   f"mean|tie|={r['mean_tie']:.2f} miss={miss}")

    base = results["A_rank_matches"]
    rows = []
    for name, r in results.items():
        if name == "A_rank_matches":
            continue
        d_r1 = 100 * (r["r1"] - base["r1"])
        d_tie = (
            100 * (base["tie_rate"] - r["tie_rate"]) / base["tie_rate"]
            if base["tie_rate"] > 0 else 0.0
        )
        rows.append({
            "config": name, **r, "d_r1_pp": d_r1, "d_tie_rel_pct": d_tie,
            "gate_pass": _gate_pass(d_r1, d_tie),
        })

    logger.log(f"\n=== P48 q_len={args.q_len} n={base['m']} ===")
    logger.log(f"  {'config':18s} {'rank-1':>8s} {'MRR':>7s} {'tie%':>7s} "
               f"{'Δr1':>8s} {'Δtie%rel':>9s}")
    logger.log(f"  {'A_rank_matches':18s} {100*base['r1']:7.2f}% {base['mrr']:7.3f} "
               f"{100*base['tie_rate']:6.1f}% {'—':>8s} {'—':>9s}")
    for row in rows:
        logger.log(f"  {row['config']:18s} {100*row['r1']:7.2f}% {row['mrr']:7.3f} "
                   f"{100*row['tie_rate']:6.1f}% {row['d_r1_pp']:+7.2f}pp "
                   f"{row['d_tie_rel_pct']:+8.1f}%")

    best = max(rows, key=lambda x: x["d_r1_pp"]) if rows else None
    # §13.15: từ khi phá-thế-hoà được WIRE mặc định, `A_rank_matches` ĐÃ có nó ⇒ E chồng
    # lên là no-op và Δ ≈ 0. Đó KHÔNG phải "Trụ D trượt" — phải phân biệt, nếu không mục
    # kết luận sẽ nói ngược hẳn với sự thật cho người đọc sau.
    already = base["tie_rate"] < 0.02
    ok = any(x["gate_pass"] for x in rows)
    logger.log(f"\n  G-hop: {'ĐẠT' if ok else 'TRƯỢT'}"
               + (f" — tốt nhất {best['config']} ({best['d_r1_pp']:+.2f}pp, "
                  f"tie {best['d_tie_rel_pct']:+.1f}%rel)" if best else ""))
    if already:
        logger.log(
            f"  ⚠️ Baseline A đã hoà {100*base['tie_rate']:.1f}% ⇒ **phá-thế-hoà ĐANG BẬT "
            f"trong rank_matches** (§13.15, `HOPFIELD_TIEBREAK_BETA`). E chồng lên là "
            f"no-op — Δ≈0 KHÔNG có nghĩa Trụ D trượt.\n"
            f"     Muốn đo lại lãi thật: đặt `ps.hopfield_tiebreak_beta = 0.0` trước khi "
            f"chạy để lấy baseline CHƯA phá hoà."
        )
    elif not ok:
        logger.log("  ⇒ định kiến §12.118/§0.W ('nhiều bước làm mờ') THẮNG; β không cứu."
                   " Đóng Trụ D, giữ rank_matches.")

    out = _ROOT / args.out_json
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "checkpoint": args.checkpoint, "q_len": args.q_len, "n": base["m"],
        "gates": {"r1_pp": GATE_R1_PP, "tie_rel_pct": GATE_TIE_REL},
        "top_k": args.top_k, "top_m": args.top_m,
        "baseline": base, "rows": rows, "gate_pass": ok,
        "seconds": round(time.time() - t0, 1),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.log(f"→ {out}")


if __name__ == "__main__":
    main()
