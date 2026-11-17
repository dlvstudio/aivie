#!/usr/bin/env python3
"""DUMP `ask_form` — WH-slot học từ cha QA có nuốt nhầm nội dung không? TK_045 §12.44.

    PYTHONPATH=src python -u scripts/diag_wh_slot.py \
        --checkpoint checkpoints/network_v12_sent.pkl --log-file out/diag_wh_slot_v12.log

## Vì sao

`ask_form[t] = n_Q / (n_Q + n_A)` đếm trên cha `is_qa`: token nằm bên CÂU HỎI mà không bao giờ
nằm bên CÂU ĐÁP = từ chỉ hình thức hỏi. Không bảng từ, không nhìn mặt chữ ⇒ Luật 0 sạch — thay
cho luật `len(surface)≤2/3` của §12.42 (M4 tự nêu là lệch §0.0, và đo được: bắn nhầm 4/8 câu hỏi,
cấm cả `Ngô`/`Lê` là chính thực thể được hỏi).

Nhưng nó KHÔNG miễn nhiễm: đo mini θ=0.8 vẫn nuốt `mẹo` (từ nội dung chỉ xuất hiện ở câu hỏi
trong corpus đó). M4 nêu đúng rủi ro này (§12.43 M4e). Script này để **nhìn tận mắt** trước khi
tin — đừng bật knob rồi mới đọc.

## Đọc

Cột `n_Q`/`n_A` là số cha QA có token ở phần hỏi / phần đáp (đếm theo **fold**, vì `Ai` và `ai`
là hai token khác nhau — không gộp thì `ai` ra 0.73 và LỌT cổng, gộp mới thấy 1.17 là thật).
Nhìn danh sách `θ` xem có thực thể/nội dung nào bị nuốt không, rồi mới chọn ngưỡng.
"""

from __future__ import annotations

import argparse
import gc
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from nn_aivie.core.checkpoint_io import load_checkpoint  # noqa: E402
from nn_aivie.hca.matrix_core import _path_fold, wh_slot_ids  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/network_v12_sent.pkl")
    ap.add_argument("--log-file", default="out/diag_wh_slot.log")
    ap.add_argument("--min-ask", type=int, default=5)
    ap.add_argument("--top", type=int, default=30)
    ap.add_argument("--thetas", default="0.9,0.8,0.7,0.6")
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
    v, ps = state.vocab, state.parents
    log(f"[{time.time()-t0:.0f}s] ✓ {len(ps):,} cha")

    ask: dict[str, int] = {}
    ans: dict[str, int] = {}
    n_qa = 0
    for rec in ps.parents:
        if not (getattr(rec, "is_qa", False) and getattr(rec, "answer_ids", ())):
            continue
        n_qa += 1
        for f in {_path_fold(v, int(x)) for x in rec.match_ids}:
            ask[f] = ask.get(f, 0) + 1
        for f in {_path_fold(v, int(x)) for x in rec.answer_ids}:
            ans[f] = ans.get(f, 0) + 1
    log(f"\ncha is_qa = {n_qa:,}")
    if not n_qa:
        log("⚠️ KHÔNG có cha QA ⇒ `ask_form` không dùng được trên checkpoint này.")
        return 1

    rows = [
        (q / float(q + ans.get(f, 0)), f, q, ans.get(f, 0))
        for f, q in ask.items() if q >= args.min_ask and f
    ]
    rows.sort(reverse=True)
    log(f"\nTOP {args.top} `ask_form` cao nhất (min_ask={args.min_ask}) — soi xem có nội dung"
        " nào bị nuốt:")
    log(f"  {'ask_form':>9}{'n_Q':>7}{'n_A':>7}  fold")
    for r, f, q, a in rows[: args.top]:
        log(f"  {r:>9.2f}{q:>7}{a:>7}  {f!r}")

    log("\nSố token bị CẤM theo ngưỡng:")
    for th in (float(x) for x in args.thetas.split(",")):
        wh = wh_slot_ids(state, min_ask=args.min_ask, ask_form_min=th)
        sur = sorted({v.token_of(t).lstrip("ĠĊ") for t in wh})
        sur = [x for x in sur if x]
        log(f"  θ={th:<5} {len(wh):>5} id · {sur[:24]}")

    log("\nChọn θ NHỎ NHẤT mà danh sách vẫn KHÔNG có thực thể/nội dung. Dán bảng về.")
    logf.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
