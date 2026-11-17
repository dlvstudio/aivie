#!/usr/bin/env python
"""P61 (TK_045 §13.25) — sinh cặp CHẤM MÙ so hai cổng: fold vs PMI-cross.

**Vì sao cần chấm mù.** P60 cho `pmi_cross` **chi phối** cổng fold trên v12 (pmi cao hơn
*và* chết ít hơn ở θ=1.25/1.5/1.75). Nhưng ở đó `pmi_cross` vừa là **cổng** vừa là **thước**
⇒ **tự tại** — đúng cái bẫy §13.10 đã mắc. Và `lạc` (cả hai bản) đã **nghỉ hưu** (§13.23:
đối chứng vàng cho thấy chúng đo trùng lặp bề mặt). ⇒ Arbiter còn lại **duy nhất đã được
kiểm** là **chấm mù**.

**Khác biệt phải để giám khảo tự thấy:** ở v12 cổng PMI cho đầu ra **NGẮN HƠN NHIỀU**
(θ=1.5 → 11.4 token vs fold 26.1). Ngắn có thể đọc thành "cụt" hoặc thành "gọn, đúng đề" —
metric không phán được, người đọc phán được. Rubric nhắc giám khảo **không** ưu ái độ dài.

Script CHỈ sinh + giấu nhãn + trộn; **không tự chấm** (nguyên tắc §12.111).

Chạy (mini):
  PYTHONPATH=src .venv/bin/python scripts/gen_gate_blind_pairs.py \\
      --checkpoint checkpoints/mini_sent_ep_qa.pkl --theta 1.5
Chạy (v12):
  … --checkpoint checkpoints/network_v12_sent.pkl --theta 1.25 \\
      --out-pairs out/gate_pairs_v12.txt --out-key out/gate_key_v12.json
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
from nn_aivie.eval.ruler import min_pairs_for  # noqa: E402
from nn_aivie.hca.clause_layer import (  # noqa: E402
    CoverageGate,
    build_clause_layer,
    compose_chain,
    content_folds,
    prompt_anchor_rescorer,
)
from nn_aivie.hca.metrics_pmi import PmiCrossIndex  # noqa: E402
from nn_aivie.hca.readout import decode_surface  # noqa: E402
from nn_aivie.hca.w_skip import ensure_w_skip  # noqa: E402
from nn_aivie.util.progress import get_run_logger  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/mini_sent_ep_qa.pkl")
    ap.add_argument("--theta", type=float, default=1.5, help="ngưỡng PMI của cổng B")
    ap.add_argument("--mode", default="gate", choices=("gate", "join", "anchor", "empty", "pivot", "len", "close"),
                    help="gate = fold vs cổng PMI (§13.25) · join = cùng cổng fold, "
                         "KHÁC bộ xếp mối nối (§13.26: S1 mù AUC .605 vs S3 .814) · "
                         "anchor = §13.31, cùng cổng fold, bộ xếp NEO-VÀO-ĐỀ có trọng số "
                         "độ hiếm `gidf` (xếp chứ KHÔNG chặn ⇒ không bao giờ cụt) · "
                         "empty = §13.37, cổng fold vs cổng fold + DỰ PHÒNG khi rỗng — "
                         "vòng duy nhất KHÔNG loại bên trống · "
                         "pivot = §13.39, cổng fold vs cổng fold + CHẶN BẢN LỀ ĐỔI VAI · "
                         "len = §13.41, CÙNG mọi thứ, chỉ khác max_clauses (5 vs --short) · "
                         "close = §13.42, len2 (mốc mới) vs DỪNG KHI ĐÓNG ĐƯỢC CÂU")
    ap.add_argument("--n-prompt", type=int, default=140)
    ap.add_argument("--pivot-max", type=float, default=0.05,
                    help="§13.39 ngưỡng chặn bản lề đổi vai (mode=pivot)")
    ap.add_argument("--max-clauses", type=int, default=5)
    ap.add_argument("--short", type=int, default=2,
                    help="§13.41 max_clauses của bên B (mode=len)")
    ap.add_argument("--pmi-cut", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-pairs", default="out/gate_pairs.txt")
    ap.add_argument("--out-key", default="out/gate_key.json")
    ap.add_argument("--force", action="store_true",
                    help="ghi đè khoá dù ĐÃ có verdict (mất khả năng tái lập)")
    args = ap.parse_args()

    # Khoá bị ghi đè = kết quả chấm cũ KHÔNG lật lại được. Đã xảy ra thật: khoá vòng một
    # bị một lần chạy `eval_ruler` sau đó ghi đè, và khi lật lại thì 4/87 cặp không khớp
    # (may là kết luận không đổi: 92.0% vs 91.6%). Chặn ở nguồn.
    _k = _ROOT / args.out_key
    _v = Path(str(_k).replace("_key", "_verdict"))
    if _k.exists() and _v.exists() and not args.force:
        raise SystemExit(
            f"⛔ {_k.name} đã có, và {_v.name} tồn tại ⇒ ghi đè sẽ làm verdict cũ KHÔNG lật "
            f"lại được.\n   Dùng tên file khác, hoặc --force nếu thật sự muốn bỏ kết quả cũ."
        )

    logger = get_run_logger("gen_gate_blind_pairs")
    t0 = time.time()
    _what = {"gate": f"cổng FOLD vs cổng PMI θ={args.theta}",
             "join": "mối nối S1 vs S3", "anchor": "mối nối S1 vs NEO-ĐỀ (§13.31)",
             "empty": "cổng fold vs fold+DỰ PHÒNG — đề bỏ trống TÍNH LÀ THUA (§13.37)",
             "pivot": f"cổng fold vs fold+CHẶN BẢN LỀ ĐỔI VAI ≥{args.pivot_max:g} (§13.39)",
             "len": f"max_clauses {args.max_clauses} vs {args.short} — DỪNG SỚM (§13.41)",
             "close": f"len{args.short} vs DỪNG-KHI-ĐÓNG-CÂU (trần {args.max_clauses}, §13.42)",
             }[args.mode]
    logger.log(f"cặp chấm mù: {_what} · {args.checkpoint}")
    logger.log(f"cần {min_pairs_for(15)} cặp cho +15pp @ power 80%")

    gc.disable()
    state, _ = load_checkpoint(str(_ROOT / args.checkpoint))
    gc.enable()
    layer = build_clause_layer(state, pmi_theta=args.pmi_cut, logger=logger)
    ws = ensure_w_skip(state, d_max=8, min_count=1, content_only=False)
    pmi = PmiCrossIndex(state)
    logger.log(f"tầng mệnh đề {len(layer):,} ({time.time()-t0:.0f}s)")

    rng = random.Random(args.seed)
    openers = [c for c in layer.clauses if c.opens_parent and len(c.tokens) >= 4]
    rng.shuffle(openers)
    prompts = openers[: args.n_prompt]

    def gen2(p, gate, *, pivot_max):
        ch, _ = compose_chain(
            state, layer, ws, p.tokens, max_clauses=args.max_clauses,
            gate=gate, rng=random.Random(args.seed), pivot_max=pivot_max,
        )
        return decode_surface(state, [int(t) for c in ch for t in c.tokens]).strip()

    def gen3(p, gate, *, max_clauses):
        ch, _ = compose_chain(
            state, layer, ws, p.tokens, max_clauses=max_clauses,
            gate=gate, rng=random.Random(args.seed),
        )
        txt = decode_surface(state, [int(t) for c in ch for t in c.tokens]).strip()
        return txt, len(ch)

    def gen4(p, gate, *, max_clauses, min_clauses):
        ch, _ = compose_chain(
            state, layer, ws, p.tokens, max_clauses=max_clauses,
            gate=gate, rng=random.Random(args.seed),
            stop_on_close=True, min_clauses=min_clauses,
        )
        txt = decode_surface(state, [int(t) for c in ch for t in c.tokens]).strip()
        return txt, len(ch)

    def gen(p, gate, join_rescore=None):
        ch, _ = compose_chain(
            state, layer, ws, p.tokens, max_clauses=args.max_clauses,
            gate=gate, rng=random.Random(args.seed), join_rescore=join_rescore,
        )
        return decode_surface(state, [int(t) for c in ch for t in c.tokens]).strip()

    rows = []
    for i, p in enumerate(prompts):
        if i and i % 40 == 0:
            logger.log(f"  {i}/{len(prompts)} ({time.time()-t0:.0f}s)")
        fold_gate = CoverageGate(
            theme_folds=frozenset(content_folds(state, p.tokens)), anchor_min_folds=1)
        n_a = n_b = None
        if args.mode == "close":
            # §13.42 — mốc so sánh là `len2`, KHÔNG phải len5: §13.41 đã cho len2 thắng
            # 78,7%, nên so với len5 nữa là so với thứ đã thua. Bên B giữ nguyên sàn
            # `min_clauses=2` rồi đi tiếp CHO TỚI KHI đóng được câu, trần max_clauses.
            a, n_a = gen3(p, fold_gate, max_clauses=args.short)
            b, n_b = gen4(p, fold_gate, max_clauses=args.max_clauses, min_clauses=args.short)
        elif args.mode == "len":
            # §13.41 — CÙNG cổng, CÙNG rng, CÙNG mọi cơ chế; khác đúng MỘT tham số.
            # Vì hai bên chọn y hệt nhau ở mỗi bước, bên ngắn LUÔN là tiền tố nghiêm ngặt
            # của bên dài ⇒ vòng này cô lập đúng câu hỏi "nối thêm có lợi không", không
            # lẫn với "nối khéo hơn không". Đó là chỗ §13.40 chưa tách được.
            a = gen(p, fold_gate)
            b, n_b = gen3(p, fold_gate, max_clauses=args.short)
            n_a = None  # len5 dùng gen() — không đếm ở đây; P75 không bắt buộc
        elif args.mode == "pivot":
            # §13.39 — CÙNG cổng fold; chỉ khác: bên B chặn ứng viên làm bản lề đổi vai
            # trong từ (`tối đa` ⇒ `đa số`). Bên B CÓ THỂ RỖNG vì chặn bớt ứng viên —
            # và ta KHÔNG giấu chỗ đó: giấu đi là lặp lại đúng sai lầm bảy vòng đầu
            # (§13.37), tức thu lợi mà không tính giá.
            a = gen(p, fold_gate)
            b = gen2(p, fold_gate, pivot_max=args.pivot_max)
        elif args.mode == "empty":
            # §13.37 — ĐỀ BỎ TRỐNG TÍNH LÀ THUA. Mọi vòng trước **loại** cặp có bên rỗng
            # (`if not a or not b`), nên 20,7% đề mà cổng không sinh được gì chưa bao giờ
            # được chấm. Bên B = cổng fold + DỰ PHÒNG: rỗng thì soạn lại KHÔNG cổng.
            # ⇒ hai bên chỉ khác nhau đúng ở những đề bị bỏ trống.
            a = gen(p, fold_gate)
            b = a if a else gen(p, None)
        elif args.mode in ("join", "anchor"):
            # CÙNG cổng fold (đã thắng 4 vòng chấm mù) — chỉ khác BỘ XẾP MỐI NỐI.
            if args.mode == "anchor":
                rs = prompt_anchor_rescorer(state, list(p.tokens))
            else:
                rs = lambda A, B: pmi.score(list(A.tokens), list(B.tokens))  # noqa: E731
            a = gen(p, fold_gate)
            b = gen(p, fold_gate, join_rescore=rs)
        else:
            a = gen(p, fold_gate)
            b = gen(p, CoverageGate(
                pmi_index=pmi, pmi_min=args.theta, theme_ids=tuple(p.tokens),
                anchor_min_folds=0))
        # --- lọc cặp. Hai chế độ `empty`/`pivot` KHÔNG loại bên rỗng: rỗng chính là cái
        # giá phải đem ra chấm (§13.37). Nhưng `a == b` thì luôn loại — cặp giống hệt
        # nhau không mang thông tin nào.
        if args.mode in ("empty", "pivot"):
            if a == b or not (a or b):
                continue
        elif not a or not b or a == b:
            continue
        rows.append((decode_surface(state, list(p.tokens)).strip(), a, b, n_a, n_b))

    pr = random.Random(20260802)
    pairs, key = [], {}
    for prompt, a, b, n_a, n_b in rows:
        pid = f"G{len(pairs):03d}"
        left_is_a = pr.random() < 0.5
        if args.mode == "join":
            na, nb = "join_S1", "join_S3"
        elif args.mode == "anchor":
            na, nb = "join_S1", "join_anchor"
        elif args.mode == "empty":
            na, nb = "fold", "fold_duphong"
        elif args.mode == "pivot":
            na, nb = "fold", "fold_chanbanle"
        elif args.mode == "len":
            na, nb = f"len{args.max_clauses}", f"len{args.short}"
        elif args.mode == "close":
            na, nb = f"len{args.short}", "dong_cau"
        else:
            na, nb = "fold", f"pmi{args.theta:g}"
        key[pid] = {"left": na if left_is_a else nb,
                    "right": nb if left_is_a else na,
                    "prompt": prompt}
        if n_a is not None or n_b is not None:
            # số mệnh đề theo NHÃN cấu hình (không theo X/Y) — báo cáo L.5
            key[pid]["n_clauses"] = {na: n_a, nb: n_b}
        pairs.append((pid, prompt, a if left_is_a else b, b if left_is_a else a))

    op = _ROOT / args.out_pairs
    op.parent.mkdir(parents=True, exist_ok=True)
    with open(op, "w", encoding="utf-8") as f:
        f.write("# CHẤM MÙ CÓ CẶP — chọn X / Y / hoà. Nhãn cấu hình KHÔNG có ở đây.\n")
        f.write(f"# n={len(pairs)} cặp · cần {min_pairs_for(15)} cho +15pp (power 80%).\n")
        f.write("# Tiêu chí: đúng ngữ cảnh · không salad · văn phong tự nhiên trôi chảy.\n")
        f.write("# ĐỘ DÀI KHÔNG phải tiêu chí — hai bên khác nhau nhiều về độ dài; đừng ưu\n")
        f.write("# ái bên dài, cũng đừng phạt bên ngắn nếu nó dừng gọn và đúng đề.\n")
        if args.mode in ("empty", "pivot"):
            # empty: §13.37 · pivot: §13.39 (B có thể rỗng vì chặn bản lề). Không giấu giá.
            f.write("#\n# ⚠️ VÒNG NÀY MỘT BÊN CÓ THỂ TRỐNG — hệ không sinh được gì. Trống\n")
            f.write("# KHÔNG tự động thua: nếu bên kia là salad lạc đề thì TRỐNG TỐT HƠN,\n")
            f.write("# vì thà im lặng còn hơn nói bậy. Nhưng trống cũng KHÔNG tự động hoà:\n")
            f.write("# nếu bên kia đọc được và bám đề thì trống là THUA thật.\n")
        f.write("\n")
        _E = "(TRỐNG — hệ không sinh được gì)"
        for pid, pr_, left, right in pairs:
            f.write(f"[{pid}] PROMPT: {pr_}\n   X: {left or _E}\n   Y: {right or _E}\n\n")
    (_ROOT / args.out_key).write_text(
        json.dumps(key, ensure_ascii=False, indent=1), encoding="utf-8")
    logger.log(f"\n→ {op} ({len(pairs)} cặp) · khoá giấu ở {args.out_key}")
    if len(pairs) < min_pairs_for(15):
        logger.log(f"⚠️ {len(pairs)} < {min_pairs_for(15)} — CHƯA đủ power. Tăng --n-prompt.")


if __name__ == "__main__":
    main()
