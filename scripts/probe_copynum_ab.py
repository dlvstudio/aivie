#!/usr/bin/env python3
"""A/B ar_matrix_copy_number trên v12 — TK_045 §12.5 v2 (lock_pi) + §12.12 (đóng số).

Load ckpt MỘT lần → OFF rồi ON (no_agenda). Cổng: ly → năm 1010 (verbatim);
không regress nq/thd/hung; bach kỳ vọng 1288 đủ (hết truncate fold).

Vòng chạy trước ra `10109` mà log KHÔNG cho biết đó là run sai hay chữ-số sinh THÊM
⇒ nay in **TRACE mỗi lần copy fire** (lock_pi · run · emit cha). Đừng đoán, đọc trace.

    .venv/bin/python -u scripts/probe_copynum_ab.py \\
        --checkpoint checkpoints/network_v12_sent.pkl \\
        --log-file out/probe_copynum_ab.log
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from nn_aivie.core.checkpoint_io import load_checkpoint  # noqa: E402
from nn_aivie.hca.config import HCAConfig  # noqa: E402
from nn_aivie.hca.generate_hebbian import generate  # noqa: E402
from nn_aivie.hca.readout import decode_surface  # noqa: E402
from nn_aivie.hca.sent_intra import clear_sent_intra_cache  # noqa: E402

# Cổng §12.5 + bach (truncate 1288→128)
PROMPTS = [
    ("ly", "Nhà Lý dời đô về"),
    ("bach", "Trận Bạch Đằng năm"),
    ("nq_def", "Ngô Quyền là vị vua sáng lập nhà Ngô, được"),
    ("thd", "Trần Hưng Đạo là"),
    ("hung", "Lê Lợi khởi nghĩa ở"),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/network_v12_sent.pkl")
    ap.add_argument("--log-file", default="out/probe_copynum_ab.log")
    ap.add_argument("--max-len", type=int, default=36)
    args = ap.parse_args()
    t0 = time.time()
    logf = open(_ROOT / args.log_file, "w", encoding="utf-8")

    def log(m=""):
        print(m, flush=True)
        logf.write(str(m) + "\n")
        logf.flush()

    state, _ = load_checkpoint(args.checkpoint)
    log(f"ckpt={args.checkpoint} max_len={args.max_len}")
    log("A/B ar_matrix_copy_number (no_agenda) — load 1 lần")

    # TRACE mỗi lần copy fire: run là gì, chép từ cha nào → phân biệt "run sai" vs "sinh thêm".
    from nn_aivie.hca import matrix_core as _MC

    _orig_run = _MC.digit_run_from_locked

    def _traced(bi, vocab, lock_pi, hist_tail, nxt):
        run = _orig_run(bi, vocab, lock_pi, hist_tail, nxt)
        if run:
            em = bi.emits[int(lock_pi)][:26]
            log(f"  [copy] w0={vocab.token_of(int(hist_tail))!r} pi={int(lock_pi)} "
                f"run={[vocab.token_of(int(t)) for t in run]} "
                f"emit={' '.join(vocab.token_of(int(t)) for t in em)!r}")
        return run

    _MC.digit_run_from_locked = _traced

    for copy_on in (False, True):
        clear_sent_intra_cache()
        tag = "COPY_ON" if copy_on else "COPY_OFF"
        log(f"\n########## {tag} ar_matrix_copy_number={copy_on} ##########")
        cfg = HCAConfig(
            ar_matrix_core=True,
            ar_no_agenda=True,
            ar_sent_agenda_handoff=False,
            ar_style_payoff=False,
            hybrid_temperature=0.0,
            ar_matrix_stack_weight=0.0,
            ar_matrix_copy_number=copy_on,
        )
        for name, prompt in PROMPTS:
            t1 = time.time()
            out = generate(state, prompt, max_len=args.max_len, config=cfg, seed=42)
            text = decode_surface(state, out)
            cont = text[len(prompt) :] if text.startswith(prompt) else text
            log(f"\n=== {name} gen ({time.time()-t1:.1f}s) ===")
            log(f"out:  {text[:360]}")
            log(f"cont: {cont[:220]!r}")

    log(f"\n[{time.time()-t0:.0f}s] → {args.log_file}")
    logf.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
