#!/usr/bin/env python3
"""PROFILE một lần `generate()` — tìm nút thắt tốc độ. TK_045 §12.33.

    PYTHONPATH=src python -u scripts/prof_generate.py \
        --checkpoint checkpoints/network_v12_sent.pkl --max-len 6 \
        --log-file out/prof_generate_v12.log

## Vì sao script này tồn tại

Máy dev không load nổi v12, mà nút thắt chỉ lộ ra ở quy mô 430k cha: nhiều đoạn mã có chi phí
**tuyến tính theo SỐ CHA**, nên trên mini (5.664) chúng vô hình. Hai lần đã bắt được theo đúng
cách này — đừng đoán, cứ profile:

* §12.32 `rescore_parents_with_anchors` — 87% thời gian, `<setcomp>` gọi 170.844 lần / 2 bước.
  Vá xong: `generate()` trên mini nhanh 7,3×; nhưng v12 T1 no_agenda chỉ 3662s → 1995s (1,84×)
  ⇒ **còn nút thắt thứ hai**.
* §12.33 vòng `bi.succ[w0]` — độ dài tỉ lệ cỡ corpus (mini `succ['Ġlà']`=1.556 ⇒ v12 ~118.000
  mục/token), lại còn `bi.A[prev, pi]` tra ma trận thưa TRONG vòng lặp.

## Đọc kết quả

Nhìn cột `cumtime` (không phải `tottime`) để thấy nhánh nào ngốn nhất, rồi lần xuống hàm con.
Dấu hiệu của lớp lỗi này: `ncalls` **rất lớn** so với số token sinh ra, và hàm nằm trong
`hca/`. `max-len` nhỏ (4–8) là đủ — mục tiêu là TỈ LỆ, không phải chất lượng văn bản.
"""

from __future__ import annotations

import argparse
import cProfile
import gc
import io
import pstats
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from nn_aivie.core.checkpoint_io import load_checkpoint  # noqa: E402
from nn_aivie.hca.config import HCAConfig  # noqa: E402
from nn_aivie.hca.generate_hebbian import generate  # noqa: E402
from nn_aivie.hca.readout import decode_surface  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/network_v12_sent.pkl")
    ap.add_argument("--log-file", default="out/prof_generate.log")
    ap.add_argument("--prompt", default="Nhà Lý dời đô về Thăng Long năm")
    ap.add_argument("--max-len", type=int, default=6)
    ap.add_argument("--top", type=int, default=30)
    ap.add_argument("--mode", default="no_agenda", choices=["no_agenda", "handoff"])
    args = ap.parse_args()

    logf = (_ROOT / args.log_file).open("w", encoding="utf-8")
    t0 = time.time()

    def log(m=""):
        print(m, flush=True)
        logf.write(str(m) + "\n")
        logf.flush()

    log(f"[{time.time()-t0:.0f}s] load {args.checkpoint} …")
    gc.disable()
    state, _ = load_checkpoint(str(_ROOT / args.checkpoint))
    gc.enable()
    log(f"[{time.time()-t0:.0f}s] ✓ {len(state.parents):,} cha")

    cfg = HCAConfig(
        ar_matrix_core=True, ar_style_payoff=False, hybrid_temperature=0.0,
        ar_matrix_stack_weight=0.0, ar_no_agenda=True,
        ar_sent_agenda_handoff=(args.mode == "handoff"),
    )

    # Chạy nháp: mọi cache lười (bipartite, W_l2r, emit_membership, succ_arrays…) dựng ở đây,
    # để profile ĐO ĐÚNG chi phí MỖI TOKEN chứ không lẫn chi phí dựng một lần.
    t1 = time.time()
    generate(state, args.prompt, max_len=3, config=cfg, seed=42)
    log(f"[{time.time()-t0:.0f}s] warm-up {time.time()-t1:.1f}s (gồm dựng cache một lần)")

    # TÁCH chi phí CỐ ĐỊNH mỗi prompt khỏi chi phí MỖI TOKEN — bằng đại số, không cần thêm log.
    # Chạy hai độ dài rồi giải: t(n) = fixed + n·per_token. Đây là thứ phân định "nút thắt nằm ở
    # khâu chuẩn bị (1 lần/prompt) hay ở vòng sinh", mà profile phẳng dễ làm lẫn.
    # `max_len` của `generate` là số token MỚI (vào thẳng `max_steps`), KHÔNG phải tổng độ dài.
    plen = len(state.vocab.encode_prompt(args.prompt))
    pts = []
    for ml in (1, args.max_len):
        t1 = time.time()
        o_ = generate(state, args.prompt, max_len=ml, config=cfg, seed=42)
        pts.append((max(1, len(o_) - plen), time.time() - t1))
    (n1, t_1), (n2, t_2) = pts
    if n2 > n1:
        per = (t_2 - t_1) / (n2 - n1)
        fixed = t_1 - n1 * per
        log(f"\nTÁCH CHI PHÍ: {t_1:.1f}s/{n1} token · {t_2:.1f}s/{n2} token")
        log(f"  ⇒ cố định mỗi prompt ≈ **{fixed:.1f}s** · mỗi token ≈ **{per:.2f}s**")
        log("  cố định lớn ⇒ tối ưu khâu CHUẨN BỊ; mỗi-token lớn ⇒ tối ưu vòng sinh.")
    else:
        log(f"\nTÁCH CHI PHÍ: KHÔNG tách được — cả hai lần đều dừng ở {n1} token "
            f"(mạng tự EOS sớm). Chạy lại với `--prompt` sinh dài hơn.")

    pr = cProfile.Profile()
    t1 = time.time()
    pr.enable()
    out = generate(state, args.prompt, max_len=args.max_len, config=cfg, seed=42)
    pr.disable()
    dt = time.time() - t1
    n_tok = max(1, len(out) - plen)
    log(f"\n{dt:.1f}s cho {n_tok} token = **{dt/n_tok:.2f}s/token**  (chế độ {args.mode})")
    log(f"out: {decode_surface(state, out)[:200]!r}\n")

    s = io.StringIO()
    pstats.Stats(pr, stream=s).sort_stats("cumulative").print_stats(args.top)
    log("\n".join(s.getvalue().split("\n")[4:]))
    log("\nĐọc `cumtime`, tìm hàm trong `hca/` có `ncalls` rất lớn so với số token. Dán về.")
    logf.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
