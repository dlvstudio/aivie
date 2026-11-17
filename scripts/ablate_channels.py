#!/usr/bin/env python3
"""ABLATION β từng KÊNH của `matrix_core_logits` — TK_045 §12.23 (Tier-2 §12.5).

    PYTHONPATH=src python -u scripts/ablate_channels.py \
        --checkpoint checkpoints/network_v12_sent.pkl \
        --log-file out/ablate_channels.log

## Vì sao

`matrix_core_logits` cộng ~10 kênh, mỗi kênh một hằng số β chưa ai kiểm lại kể từ lúc đặt.
Khi output sai, không biết kênh nào đẩy; khi vá một kênh, không biết có che lỗi kênh khác không.
Ablation trả lời câu hỏi RẺ NHẤT và có giá trị kiến trúc cao nhất: **kênh nào thật sự gánh việc.**

Cách đo: tắt ĐÚNG MỘT kênh (β=0), chạy lại bộ đo rộng §12.22, so với BASE (đủ kênh).
Kênh tắt đi mà không đổi gì = **kênh chết** (bỏ được, giảm 1 siêu tham số + 1 lớp nhiễu khi chẩn).
Kênh tắt đi làm hỏng = kênh gánh việc, và hỏng KIỂU gì cho biết nó phụ trách gì.

Thước và bộ prompt DÙNG LẠI `eval_gen_broad.py` — không dựng thước thứ hai (§12.16/§12.18/§12.20:
ba lần liên tiếp lỗi nằm ở thước, nên càng ít thước càng ít chỗ hỏng).

## Đọc bảng

Mỗi dòng = một kênh bị tắt. Cột = chênh lệch so với BASE. `Δempty>0` = câm thêm;
`Δloop4>0`/`Δdigit⎵>0` = thoái hoá thêm; `Δcopy<0` = bớt bám bộ nhớ.
`|Δ| ≈ 0` ở mọi cột ⇒ ứng viên **kênh chết**, đề xuất bỏ (kèm số vào ticket).
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
from nn_aivie.hca.config import HCAConfig  # noqa: E402
from nn_aivie.hca.generate_hebbian import generate  # noqa: E402
from nn_aivie.hca.matrix_core import ensure_parent_bipartite  # noqa: E402
from nn_aivie.hca.readout import decode_surface  # noqa: E402
from nn_aivie.hca.sent_intra import clear_sent_intra_cache  # noqa: E402

from eval_gen_broad import MODES, PROMPTS, EmitIndex, _metrics  # noqa: E402

# Tên kênh → knob β. Giữ nguyên thứ tự "gần dữ liệu → xa dữ liệu" cho dễ đọc.
CHANNELS = {
    "bag(parent)": "ar_matrix_parent_weight",
    "succ+path": "ar_matrix_parent_succ_weight",
    "l2r": "ar_sent_l2r_weight",
    "h(long-ctx)": "ar_matrix_h_weight",
    "know": "ar_matrix_know_weight",
    "know_lr": "ar_matrix_know_lr_weight",
    "qk": "ar_matrix_qk_weight",
    "eos": "ar_matrix_eos_weight",
    "r2l_ffn": "ar_matrix_r2l_ffn_weight",
    "readout": "ar_matrix_readout_weight",
    "agenda": "ar_sent_agenda_weight",
}
KEYS = ("empty", "digit_split", "loop4", "copy", "n_facts", "stop_clean", "n_tok")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/network_v12_sent.pkl")
    ap.add_argument("--log-file", default="out/ablate_channels.log")
    ap.add_argument("--max-len", type=int, default=40)
    ap.add_argument("--mode", default="handoff", choices=list(MODES),
                    help="§12.21: handoff là chế độ sinh đa câu; no_agenda là chế độ probe cũ")
    ap.add_argument("--only", default="", help="chỉ chạy các kênh này (phẩy)")
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
    v = state.vocab
    idx = EmitIndex(ensure_parent_bipartite(state).emits)
    log(f"[{time.time()-t0:.0f}s] ✓ {len(state.parents):,} cha · chế độ={args.mode}")

    plist: list[list[int]] = []
    for ps_ in PROMPTS.values():
        plist.extend(list(v.encode_prompt(p)) for p in ps_)
    log(f"  {len(plist)} prompt (T2–T6 của eval_gen_broad)")

    def run(over: dict) -> dict[str, float]:
        acc = []
        clear_sent_intra_cache()  # §12.36: MỘT lần/lượt, không phải mỗi prompt (v12: ~150s/lần)
        for pids in plist:
            cfg = HCAConfig(ar_matrix_core=True, ar_style_payoff=False,
                            hybrid_temperature=0.0, ar_matrix_stack_weight=0.0,
                            **MODES[args.mode], **over)
            out = generate(state, decode_surface(state, pids), max_len=args.max_len,
                           config=cfg, seed=42)
            acc.append(_metrics(state, idx, list(pids), list(out), args.max_len))
        return {k: sum(x[k] for x in acc) / max(1, len(acc)) for k in KEYS}

    base = run({})
    log(f"[{time.time()-t0:.0f}s] BASE  " +
        "  ".join(f"{k}={base[k]:.2f}" for k in KEYS))

    want = [c.strip() for c in args.only.split(",") if c.strip()] or list(CHANNELS)
    log("\n" + "=" * 92)
    log(f"{'tắt kênh':<14}{'Δempty':>9}{'Δdigit⎵':>9}{'Δloop4':>8}{'Δcopy':>8}"
        f"{'Δn_facts':>10}{'Δstop':>8}{'Δn_tok':>9}")
    log("=" * 92)
    dead: list[str] = []
    for name in want:
        knob = CHANNELS.get(name)
        if knob is None or not hasattr(HCAConfig(), knob):
            log(f"{name:<14}  (không có knob `{knob}`, bỏ qua)")
            continue
        r = run({knob: 0.0})
        d = {k: r[k] - base[k] for k in KEYS}
        log(f"{name:<14}{d['empty']:>9.2f}{d['digit_split']:>9.2f}{d['loop4']:>8.2f}"
            f"{d['copy']:>8.2f}{d['n_facts']:>10.2f}{d['stop_clean']:>8.2f}"
            f"{d['n_tok']:>9.1f}   [{time.time()-t0:.0f}s]")
        if all(abs(d[k]) < 1e-9 for k in KEYS):
            dead.append(name)
    log("=" * 92)
    log(f"Ứng viên KÊNH CHẾT (mọi Δ = 0): {dead if dead else '(không có)'}")
    log("Δ=0 ở bộ prompt này KHÔNG chứng minh kênh vô dụng toàn cục — chỉ là ứng viên;")
    log("trước khi bỏ phải chạy thêm `--mode no_agenda` và bộ T1/T7 của eval_gen_broad.")
    log(f"\n[{time.time()-t0:.0f}s] → {args.log_file}")
    logf.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
