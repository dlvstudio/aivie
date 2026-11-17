#!/usr/bin/env python3
"""
TK_036 S3 — retrain kiểm chứng skip ledger trong pipeline train thật.

Chạy wiki ingest + paragraph pool từ checkpoint segseed (trước scale v10),
ghi checkpoint verify riêng, probe skip spans, eval discourse (common_noun).

  PYTHONPATH=src python -u scripts/retrain_tk036_verify.py
  PYTHONPATH=src python -u scripts/retrain_tk036_verify.py --max-sentences 3000 --skip-eval
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]


def _run(cmd: list[str], *, log_name: str) -> None:
    print(f"\n▶ {' '.join(cmd)}", flush=True)
    env = {**dict(**__import__("os").environ), "PYTHONPATH": "src"}
    rc = subprocess.call(cmd, cwd=_ROOT, env=env)
    if rc != 0:
        raise SystemExit(f"{log_name} failed (exit {rc})")


def main() -> int:
    ap = argparse.ArgumentParser(description="Retrain + verify TK_036 skip ledger S3")
    ap.add_argument(
        "--base",
        default="checkpoints/network_v7_history_segseed.pkl",
        help="checkpoint trước scale wiki v10",
    )
    ap.add_argument(
        "--histwiki-out",
        default="checkpoints/tk036_verify_histwiki.pkl",
    )
    ap.add_argument(
        "--paragraph-out",
        default="checkpoints/tk036_verify_paragraph.pkl",
    )
    ap.add_argument("--max-sentences", type=int, default=0, help="0 = toàn bộ wiki para")
    ap.add_argument("--max-units", type=int, default=25000)
    ap.add_argument("--skip-paragraph", action="store_true")
    ap.add_argument("--skip-probe", action="store_true")
    ap.add_argument("--skip-eval", action="store_true")
    args = ap.parse_args()

    hist_cmd = [
        sys.executable,
        "scripts/build_histwiki_incremental.py",
        "--checkpoint",
        args.base,
        "--out",
        args.histwiki_out,
    ]
    if args.max_sentences > 0:
        hist_cmd.extend(["--max-sentences", str(args.max_sentences)])
    _run(hist_cmd, log_name="build_histwiki_incremental")

    paragraph_ckpt = args.histwiki_out
    if not args.skip_paragraph:
        _run(
            [
                sys.executable,
                "scripts/build_paragraph_pool.py",
                "--checkpoint",
                args.histwiki_out,
                "--out",
                args.paragraph_out,
                "--max-units",
                str(args.max_units),
            ],
            log_name="build_paragraph_pool",
        )
        paragraph_ckpt = args.paragraph_out

    if not args.skip_probe:
        _run(
            [
                sys.executable,
                "scripts/probe_skip_spans.py",
                "--checkpoint",
                paragraph_ckpt,
                "--top",
                "30",
            ],
            log_name="probe_skip_spans",
        )

    if not args.skip_eval:
        _run(
            [
                sys.executable,
                "-u",
                "scripts/eval_discourse_prompts.py",
                "--checkpoint",
                paragraph_ckpt,
                "--n-history",
                "0",
                "--n-doc",
                "0",
                "--n-conv",
                "0",
                "--n-common",
                "10",
                "--max-modes",
                "2",
                "--out",
                "out/tk036_retrain_verify_eval.log",
            ],
            log_name="eval_discourse_prompts",
        )

    print(f"\n✓ TK_036 retrain verify xong → {paragraph_ckpt}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
