#!/usr/bin/env python
"""P77 (TK_045 §13.43) — sinh mẫu **KHÔNG LỌC** để chấm mù **TUYỆT ĐỐI**.

**Vì sao cần.** Mười một vòng chấm mù (§13.25–§13.42) đều đo **SO SÁNH**: A có hơn B không.
Hai thứ sống sót (cổng phủ fold, `max_clauses=2`) vì thế chỉ được chứng minh là **ít tệ hơn**,
**chưa** bao giờ được chứng minh là **tốt**. Giám khảo bốn vòng liên tiếp tự nói ra điều đó —
*"đa số cặp là chọn giữa hai đoạn salad"* — nhưng không vòng nào biến nó thành **con số**.

Vòng này đo tuyệt đối: từng đầu ra một, thang **dùng được / tạm / hỏng**, không so với gì.

**Hai điều làm khác mọi vòng trước:**

1. **KHÔNG LỌC.** Mọi vòng trước bỏ cặp `a == b` và (bảy vòng đầu) bỏ cả bên rỗng ⇒ mẫu đã
   thiên. Ở đây **giữ nguyên tất cả**, kể cả đề mà hệ **không sinh được gì** — rỗng là một
   loại đầu ra, và nó phải nằm trong mẫu số. Đây là bài học §13.35 (lỗi mẫu số) áp dụng
   ngược lại vào chính cách lấy mẫu.
2. Chạy ở **đúng cấu hình đang thắng**: cổng phủ fold + `max_clauses=2`. Không cơ chế thử
   nghiệm nào.

Chạy (v12 — cần máy có RAM):
  PYTHONPATH=src .venv/bin/python scripts/gen_absolute_samples.py \\
      --checkpoint checkpoints/network_v12_sent.pkl --n 200
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
from nn_aivie.hca.clause_layer import (  # noqa: E402
    CoverageGate,
    build_clause_layer,
    compose_chain,
    content_folds,
)
from nn_aivie.hca.readout import decode_surface  # noqa: E402
from nn_aivie.hca.w_skip import ensure_w_skip  # noqa: E402
from nn_aivie.util.progress import get_run_logger  # noqa: E402

EMPTY = "(TRỐNG — hệ không sinh được gì)"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/mini_sent_ep_qa.pkl")
    ap.add_argument("--n", type=int, default=200, help="số đề — KHÔNG lọc, mọi đề đều vào mẫu")
    ap.add_argument("--max-clauses", type=int, default=2, help="cấu hình đang thắng (§13.41)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="out/abs_samples.txt")
    ap.add_argument("--out-meta", default="out/abs_meta.json")
    ap.add_argument("--join-seq-max", type=float, default=None,
                    help="§13.44 — bật điều kiện hợp lệ mối nối; TRỘN CHUNG với bản hiện tại")
    ap.add_argument("--len-short", type=int, default=None,
                    help="§13.51 — so max_clauses hiện tại vs giá trị này, trên BỘ ĐỀ RỜI NHAU")
    ap.add_argument("--theme-gidf", type=float, default=None,
                    help="§13.48 — ngưỡng fold ĐẶC TRƯNG cho cổng phủ; TRỘN CHUNG với bản hiện tại")
    ap.add_argument("--out-key", default=None,
                    help="bắt buộc khi có --join-seq-max: khoá cấu hình của từng mẫu")
    args = ap.parse_args()

    logger = get_run_logger("gen_absolute_samples")
    t0 = time.time()
    logger.log(f"P77 mẫu tuyệt đối · {args.checkpoint} · n={args.n} · "
               f"max_clauses={args.max_clauses} · KHÔNG LỌC")

    gc.disable()
    state, _ = load_checkpoint(str(_ROOT / args.checkpoint))
    gc.enable()
    layer = build_clause_layer(state, pmi_theta=0.0, logger=logger)
    ws = ensure_w_skip(state, d_max=8, min_count=1, content_only=False)
    logger.log(f"tầng mệnh đề {len(layer):,} ({time.time()-t0:.0f}s)")

    rng = random.Random(args.seed)
    openers = [c for c in layer.clauses if c.opens_parent and len(c.tokens) >= 4]
    rng.shuffle(openers)
    prompts = openers[: args.n]

    # §13.44/§13.45 — khi so hai cấu hình trên thang TUYỆT ĐỐI, phải chấm CHUNG MỘT PHIÊN.
    # Hai phiên khác nhau áp hai mức chặt tay khác nhau, và chênh đó thừa sức nuốt hiệu ứng
    # thật (mốc gốc chỉ 0,5% — §13.43). Trộn chung + giấu nhãn là cách duy nhất để con số
    # mới so được với con số cũ.
    _n_mech = sum(x is not None for x in
                  (args.join_seq_max, args.theme_gidf, args.len_short))
    if _n_mech > 1:
        raise SystemExit("⛔ mỗi lượt chỉ thử MỘT cơ chế — nếu không thì không quy nhân được.")
    two = _n_mech == 1
    if two and not args.out_key:
        raise SystemExit("⛔ chế độ so sánh cần --out-key để giấu nhãn cấu hình.")

    rows, meta = [], []
    n_empty = 0

    # §13.51 — BỘ ĐỀ RỜI NHAU. §13.49 phát hiện chấm-cặp-tuyệt-đối HỞ NHÃN một phần: hai
    # cấu hình chạy trên cùng bộ đề nên mỗi đề xuất hiện đúng hai lần và giám khảo ghép cặp
    # được bằng mắt. Ở vòng len1-vs-len2 lỗ đó là TỐI ĐA — bản ngắn là TIỀN TỐ của bản dài.
    # Đổi sang hai bộ đề rời nhau: mất khả năng so cặp, nhưng bịt hẳn đường suy nhãn. Đánh
    # đổi này là CÓ Ý THỨC, đúng chỗ §13.49 đã ghi là phải chọn.
    if args.len_short is not None:
        half = args.n
        # Rời nhau theo MẶT CHỮ đã giải mã — không chỉ theo chỉ số clause. Corpus có nhiều
        # opener khác cha mà decode trùng (vd. «Giải nghĩa: (khẩu ngữ)»); cổng G-len1(2)
        # đòi 0 đề trùng giữa hai nhánh để giám khảo không ghép cặp bằng mắt.
        seen: set[str] = set()
        prompts, prompts_b = [], []
        for c in openers:
            s = decode_surface(state, list(c.tokens)).strip()
            if not s or s in seen:
                continue
            seen.add(s)
            if len(prompts) < half:
                prompts.append(c)
            elif len(prompts_b) < half:
                prompts_b.append(c)
            else:
                break
        if len(prompts) < half or len(prompts_b) < half:
            raise SystemExit(
                f"⛔ không đủ đề mặt-chữ rời nhau: cần {2*half}, "
                f"có {len(prompts)}+{len(prompts_b)} (unique={len(seen)})")
        for _cfg, _mc, _ps in ((f"len{args.max_clauses}", args.max_clauses, prompts),
                               (f"len{args.len_short}", args.len_short, prompts_b)):
            for p in _ps:
                gate = CoverageGate(
                    theme_folds=frozenset(content_folds(state, p.tokens)), anchor_min_folds=1)
                ch, st = compose_chain(
                    state, layer, ws, p.tokens, max_clauses=_mc,
                    gate=gate, rng=random.Random(args.seed))
                t = decode_surface(state, [int(x) for c in ch for x in c.tokens]).strip()
                if not t:
                    n_empty += 1
                rows.append((_cfg, decode_surface(state, list(p.tokens)).strip(), t or EMPTY))
                meta.append({"cfg": _cfg, "n_clauses": len(ch), "empty": not t,
                             "stop": st["stop"]})
        prompts = []          # bỏ qua vòng lặp chính

    for i, p in enumerate(prompts):
        if i and i % 50 == 0:
            logger.log(f"  {i}/{len(prompts)} ({time.time()-t0:.0f}s)")
        gate = CoverageGate(
            theme_folds=frozenset(content_folds(state, p.tokens)), anchor_min_folds=1)
        ch, st = compose_chain(
            state, layer, ws, p.tokens, max_clauses=args.max_clauses,
            gate=gate, rng=random.Random(args.seed),
        )
        txt = decode_surface(state, [int(t) for c in ch for t in c.tokens]).strip()
        pr_txt = decode_surface(state, list(p.tokens)).strip()
        cand = [("hien_tai", txt, ch, st)]
        if args.theme_gidf is not None:
            # §13.48 — cổng đòi ≥1 fold ĐẶC TRƯNG chung với đề. `reject_when_no_theme` là
            # bắt buộc: không có nó thì đề nào không còn fold đặc trưng sẽ làm cổng TẮT HẲN
            # và nhận mọi ứng viên — siết lại hoá ra mở toang, âm thầm (đo mini: 19,7% đề
            # rơi vào ca đó ở ngưỡng 7,0).
            g2 = CoverageGate(
                theme_folds=frozenset(content_folds(state, p.tokens, gidf_min=args.theme_gidf)),
                anchor_min_folds=1, gidf_min=args.theme_gidf, reject_when_no_theme=True)
            ch3, st3 = compose_chain(
                state, layer, ws, p.tokens, max_clauses=args.max_clauses,
                gate=g2, rng=random.Random(args.seed))
            t3 = decode_surface(state, [int(t) for c in ch3 for t in c.tokens]).strip()
            cand.append(("fold_dac_trung", t3, ch3, st3))
        if args.join_seq_max is not None:
            ch2, st2 = compose_chain(
                state, layer, ws, p.tokens, max_clauses=args.max_clauses,
                gate=gate, rng=random.Random(args.seed), join_seq_max=args.join_seq_max,
            )
            t2 = decode_surface(state, [int(t) for c in ch2 for t in c.tokens]).strip()
            cand.append(("mối_nối_kề", t2, ch2, st2))
        for cfg, t, c_, s_ in cand:
            if not t:
                n_empty += 1
            rows.append((cfg, pr_txt, t or EMPTY))
            meta.append({"cfg": cfg, "n_clauses": len(c_), "empty": not t, "stop": s_["stop"]})

    # trộn để nhãn không suy được từ thứ tự
    order = list(range(len(rows)))
    random.Random(20260803).shuffle(order)
    rows = [rows[i] for i in order]
    meta = [meta[i] for i in order]
    key = {}
    for j, (m, r) in enumerate(zip(meta, rows)):
        sid = f"A{j:03d}"
        m["id"] = sid
        key[sid] = m["cfg"]
        rows[j] = (sid, r[1], r[2])

    op = _ROOT / args.out
    op.parent.mkdir(parents=True, exist_ok=True)
    with open(op, "w", encoding="utf-8") as f:
        f.write("# CHẤM MÙ TUYỆT ĐỐI — mỗi mẫu chấm ĐỘC LẬP, không so với mẫu nào khác.\n")
        f.write(f"# n={len(rows)} mẫu · KHÔNG LỌC (mọi đề đều có mặt, kể cả đề hệ bỏ trống).\n")
        f.write("# Thang: dung_duoc / tam / hong. Xem rubric trong runbook P77.\n\n")
        for sid, pr, txt in rows:
            f.write(f"[{sid}] ĐỀ: {pr}\n   → {txt}\n\n")
    if two:
        (_ROOT / args.out_key).write_text(
            json.dumps(key, ensure_ascii=False, indent=1), encoding="utf-8")
    (_ROOT / args.out_meta).write_text(
        json.dumps({"checkpoint": args.checkpoint, "n": len(rows),
                    "max_clauses": args.max_clauses, "n_empty": n_empty,
                    "ty_le_trong": n_empty / len(rows) if rows else 0.0,
                    "menh_de_tb": sum(m["n_clauses"] for m in meta) / len(meta) if meta else 0,
                    "rows": meta}, ensure_ascii=False, indent=1), encoding="utf-8")
    logger.log(f"\n→ {args.out} ({len(rows)} mẫu · trống {n_empty} = "
               f"{n_empty/len(rows) if rows else 0:.1%}) · meta {args.out_meta}")


if __name__ == "__main__":
    main()
