#!/usr/bin/env python
"""P47 (TK_045 §13.4 hàng 1) — CỔNG SINH-TỬ của tầng nơ-ron MỆNH ĐỀ.

Đo TRƯỚC khi code kiến trúc (luật đo-trước-code). Ba câu hỏi:

  M1 — ĐỘ HẠT: cắt emit của cha bằng tín hiệu biên ĐÃ HỌC (`P(EOS|w)` lift, §0.4.2 —
       KHÔNG danh sách dấu câu) ở nhiều ngưỡng θ → có θ nào cho mệnh đề 4–12 token không?

  M2 — TÁI TỔ HỢP KHÁC-CHA (kill switch): với mỗi mệnh đề C, ứng viên kế = mệnh đề C'
       mà mối nối (token nội dung CUỐI của C → token nội dung ĐẦU của C') **đã được
       chứng thực** trên W_skip ở d nhỏ. Bao nhiêu % ứng viên đến từ cha KHÁC?
       ≈ 0 ⇒ tầng mệnh đề chỉ phát lại cha = chép ⇒ ĐÓNG CẢ ĐỀ XUẤT §13.

  M3 — ENTROPY HIỆU DỤNG: H trên tập ứng viên (trọng số = sức mạnh mối nối), chia cho
       độ dài mệnh đề trung bình → bits/token so với **6.42 bits** (v4 §3.7).
       H_type ≥ 6 bits ⇒ ĐÓNG.

Luật 0: chỉ ĐỌC đồ thị (W_seq cho lift, W_skip cho mối nối). Không train, không hardcode
từ vựng / dấu câu. Mọi biên đến từ thống kê `→EOS` mà `ingest._with_eos` đã dạy.

Chạy:
  PYTHONPATH=src .venv/bin/python scripts/probe_clause_layer.py \\
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

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from nn_aivie.core.checkpoint_io import load_checkpoint  # noqa: E402
from nn_aivie.hca.clause_layer import (  # noqa: E402  — MỘT định nghĩa DUY NHẤT
    build_clause_layer,
    ensure_boundary_signal,
    join_candidates,
)
from nn_aivie.hca.w_skip import ensure_w_skip  # noqa: E402
from nn_aivie.util.progress import get_run_logger  # noqa: E402

# Cổng §13.4 (nêu TRƯỚC khi đo — chống chỉnh thước sau khi thấy số)
GATE_CROSS_MIN = 0.05        # cross-parent ratio < 5% ⇒ đóng (chỉ phát lại cha)
CLAUSE_LEN_LO, CLAUSE_LEN_HI = 4.0, 12.0  # dải "khoảng giữa" hữu ích

# ⚠️ SỬA CỔNG 2026-08-02 sau lần chạy đầu (ghi lại, KHÔNG xoá dấu vết — §13.6):
#   Cổng gốc của tôi là `H_type < 6.0 bits`. **Mis-specified.** Nó đo giá MỖI QUYẾT
#   ĐỊNH, trong khi luận điểm §13 là **khấu hao**: một quyết định 6.4-bit cam kết 9
#   token rẻ hơn hẳn chín quyết định 6.4-bit. Đại lượng đúng là **bits/token**.
#   Ngưỡng thay thế KHÔNG bịa hôm nay: `v4 §3.7` (2026-07-17) đã nêu đích
#   *"phân bố cần đi từ ~6 bits xuống ~2"* TRƯỚC ticket này.
GATE_BITS_PER_TOKEN_MAX = 2.0
GATE_H_TYPE_MAX = 6.0        # giữ để BÁO CÁO (cổng cũ), không dùng cho verdict


# ------------------------------------------------------- M2/M3 trên một mẫu
def measure_joins(layer, ws, d_join, sample, rng, logger):
    """Với mỗi mệnh đề mẫu: ứng viên kế = mệnh đề có token đầu nối được trên W_skip."""
    clauses = layer.clauses
    first_idx = layer.by_opener
    n_cl = len(clauses)
    if n_cl == 0:
        return {}
    idxs = rng.sample(range(n_cl), min(sample, n_cl))
    n_cand: list[int] = []
    n_cand_types: list[int] = []
    cross_ratio: list[float] = []
    h_type: list[float] = []
    h_clause: list[float] = []
    n_dead = 0
    for k, ci in enumerate(idxs):
        cl = clauses[ci]
        pi, last = cl.parent_idx, cl.last_content
        type_w = join_candidates(layer, ws, last, d_join=d_join)
        if not type_w:
            n_dead += 1
            n_cand.append(0)
            n_cand_types.append(0)
            continue
        # phân bố trên TOKEN MỞ mệnh đề kế (quyết định so được với 6.42 bits)
        openers = list(type_w.keys())
        tw = np.array([type_w[j] for j in openers], dtype=np.float64)
        q = tw / tw.sum()
        h_type.append(float(-(q * np.log2(q)).sum()))
        # Phân bố trên MỆNH ĐỀ (chia đều trong cùng token mở) — DẠNG GIẢI TÍCH.
        # Mỗi thành viên của token mở j có p = q_j/m_j ⇒ đóng góp m_j·p·log p = q_j·log(q_j/m_j)
        #   ⇒ H_clause = H_type + Σ_j q_j·log2(m_j)   (kỳ vọng log cỡ nhóm hoà)
        # Bản trước materialize list dài BẰNG SỐ ỨNG VIÊN mỗi mẫu (mini 1.8k, v12 ~138k
        # × 1500 mẫu) — cùng kết quả, nhưng ở v12 thì nó là nút cổ chai. Test khoá đẳng thức.
        m = np.array([len(first_idx[j]) for j in openers], dtype=np.float64)
        h_clause.append(float(h_type[-1] + (q * np.log2(m)).sum()))
        tot_cl = int(m.sum())
        # cross-parent: đếm phần BÙ — mỗi cha chỉ có vài mệnh đề, rẻ hơn quét mọi thành viên
        own = 0
        for ci_own in layer.by_parent.get(pi, ()):
            fc = clauses[ci_own].first_content
            if fc in type_w:
                own += 1
        cross = tot_cl - own
        n_cand.append(tot_cl)
        n_cand_types.append(len(type_w))
        cross_ratio.append(cross / max(1, tot_cl))
        if k and k % 500 == 0:
            logger.log(f"  … mối nối {k}/{len(idxs)}")

    def _m(a):
        return float(np.mean(a)) if a else 0.0

    return {
        "n_sample": len(idxs),
        "dead_rate": n_dead / max(1, len(idxs)),
        "mean_candidates": _m(n_cand),
        "median_candidates": float(np.median(n_cand)) if n_cand else 0.0,
        "mean_candidate_types": _m(n_cand_types),
        "cross_parent_ratio": _m(cross_ratio),
        "H_type_bits": _m(h_type),
        "H_clause_bits": _m(h_clause),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/network_v12_sent.pkl")
    ap.add_argument("--theta", type=float, default=2.0,
                    help="ngưỡng lift biên ĐƠN VỊ (2.0 = ar_eos_boundary_min_lift)")
    ap.add_argument("--pmi-thetas", default="none,-1.0,0.0,0.5,1.0,1.5",
                    help="ngưỡng PMI cắt biên MỆNH ĐỀ; 'none' = chỉ cắt câu")
    ap.add_argument("--dump", type=int, default=6,
                    help="số mệnh đề in ra để ĐỌC TAY mỗi ngưỡng")
    ap.add_argument("--d-join", type=int, default=4)
    ap.add_argument("--w-skip-dmax", type=int, default=8)
    ap.add_argument("--w-skip-min-count", type=int, default=1)
    ap.add_argument("--sample", type=int, default=1500)
    ap.add_argument("--max-parents", type=int, default=0, help="0 = hết")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-json", default="out/probe_clause_layer.json")
    args = ap.parse_args()

    logger = get_run_logger("probe_clause_layer")
    t0 = time.time()
    logger.log(f"P47 — cổng sinh-tử tầng mệnh đề · checkpoint={args.checkpoint}")
    logger.log(f"CỔNG: bits/token < {GATE_BITS_PER_TOKEN_MAX} (v4 §3.7) · "
               f"cross_parent ≥ {GATE_CROSS_MIN:.0%} · len ∈ [{CLAUSE_LEN_LO},{CLAUSE_LEN_HI}]"
               f"  | cổng-cũ H_type<{GATE_H_TYPE_MAX} MIS-SPECIFIED, chỉ báo cáo")

    gc.disable()
    state, _ = load_checkpoint(str(_ROOT / args.checkpoint))
    gc.enable()
    logger.log(f"load xong {time.time() - t0:.1f}s")

    v = state.vocab
    ps = state.parents
    if ps is None or not len(ps.parents):
        logger.log("⛔ checkpoint không có ParentStore — dừng.")
        return
    logger.log(f"vocab={len(v):,} · cha={len(ps.parents):,}")

    bound = ensure_boundary_signal(state, logger=logger)
    if bound.eos_base <= 0:
        logger.log("⛔ P(EOS) nền = 0 — checkpoint trước đợt EOS (§0.4.2). Probe vô nghĩa.")
        return
    lift = bound.lift
    logger.log(f"P(EOS) nền = {bound.eos_base:.6f}")
    top = np.argsort(-lift)[:12]
    logger.log("lift cao nhất: " + ", ".join(
        f"{v.token_of(int(t))!r}={lift[int(t)]:.1f}" for t in top))

    logger.log(f"dựng W_skip d_max={args.w_skip_dmax} min_count={args.w_skip_min_count} …")
    ws = ensure_w_skip(
        state, d_max=int(args.w_skip_dmax), min_count=int(args.w_skip_min_count),
        content_only=False,
    )
    logger.log(f"W_skip: {ws.n_edges:,} lượt cạnh · {ws.n_spans:,} span · "
               f"{time.time() - t0:.1f}s")

    rng = random.Random(args.seed)
    rows = []
    for spec in [x.strip() for x in args.pmi_thetas.split(",") if x.strip()]:
        use_pmi = spec.lower() not in {"none", "off"}
        pth = float(spec) if use_pmi else None
        logger.log(f"── PMI cut = {spec} (θ_eos={args.theta}) ──────────────")
        layer = build_clause_layer(
            state, pmi_theta=pth, eos_theta=float(args.theta),
            max_parents=int(args.max_parents), logger=logger,
        )
        clauses = layer.clauses
        if not clauses:
            logger.log("  (0 mệnh đề)")
            continue
        for ci in rng.sample(range(len(clauses)), min(args.dump, len(clauses))):
            surf_txt = "".join(
                v.token_of(t) for t in clauses[ci].tokens
            ).replace("Ġ", " ")
            logger.log(f"    | {surf_txt.strip()[:110]}")
        lens = np.array([len(c.tokens) for c in clauses], dtype=np.float64)
        m1 = {
            "pmi_cut": spec,
            "theta": args.theta,
            "n_clauses": len(clauses),
            "n_distinct_openers": len(layer.by_opener),
            "mean_len": float(lens.mean()),
            "median_len": float(np.median(lens)),
        }
        logger.log(f"  M1: {m1['n_clauses']:,} mệnh đề · dài TB {m1['mean_len']:.2f} "
                   f"(median {m1['median_len']:.0f}) · {m1['n_distinct_openers']:,} token mở")
        m23 = measure_joins(
            layer, ws, int(args.d_join), int(args.sample), rng, logger,
        )
        row = {**m1, **m23}
        row["bits_per_token"] = (
            row["H_type_bits"] / row["mean_len"] if row["mean_len"] > 0 else float("inf")
        )
        gate_len = CLAUSE_LEN_LO <= row["mean_len"] <= CLAUSE_LEN_HI
        gate_cross = row["cross_parent_ratio"] >= GATE_CROSS_MIN
        gate_bpt = row["bits_per_token"] < GATE_BITS_PER_TOKEN_MAX
        gate_h_old = row["H_type_bits"] < GATE_H_TYPE_MAX  # cổng cũ — chỉ báo cáo
        row["gate_len"], row["gate_cross"] = gate_len, gate_cross
        row["gate_bits_per_token"], row["gate_h_type_OLD"] = gate_bpt, gate_h_old
        row["gate_pass"] = bool(gate_len and gate_cross and gate_bpt)
        logger.log(
            f"  M2: ứng viên TB {row['mean_candidates']:.0f} "
            f"({row['mean_candidate_types']:.0f} token mở) · "
            f"KHÁC-CHA {row['cross_parent_ratio']:.1%} · chết {row['dead_rate']:.1%}"
        )
        logger.log(
            f"  M3: H_type {row['H_type_bits']:.2f} bits · H_clause "
            f"{row['H_clause_bits']:.2f} · **{row['bits_per_token']:.3f} bits/token** "
            f"(bigram = 6.42)"
        )
        logger.log(
            f"  CỔNG: len={'✓' if gate_len else '✗'} cross={'✓' if gate_cross else '✗'} "
            f"bits/tok={'✓' if gate_bpt else '✗'} ⇒ "
            f"{'ĐẠT' if row['gate_pass'] else 'TRƯỢT'}   "
            f"[cổng-cũ H_type={'✓' if gate_h_old else '✗'}]"
        )
        rows.append(row)

    payload = {
        "checkpoint": args.checkpoint,
        "gates": {
            "bits_per_token_max": GATE_BITS_PER_TOKEN_MAX,
            "cross_parent_min": GATE_CROSS_MIN,
            "clause_len": [CLAUSE_LEN_LO, CLAUSE_LEN_HI],
            "H_type_max_OLD_MISSPECIFIED": GATE_H_TYPE_MAX,
        },
        "d_join": args.d_join,
        "sample": args.sample,
        "rows": rows,
        "verdict": bool(any(r["gate_pass"] for r in rows)),
        "seconds": round(time.time() - t0, 1),
    }
    out = _ROOT / args.out_json
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.log(f"→ {out}")
    logger.log(
        "VERDICT: " + ("MỞ — có θ qua cả ba cổng ⇒ triển khai Trụ A"
                       if payload["verdict"]
                       else "ĐÓNG — không θ nào qua ⇒ dừng §13 Trụ A")
    )
    logger.log(f"tổng {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
