#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time
from pathlib import Path

from nn_aivie.core.checkpoint_io import load_checkpoint
from nn_aivie.hca.config import HCAConfig
from nn_aivie.hca.generate_hebbian import generate
from nn_aivie.hca.readout import decode_surface


def default_prompts() -> list[str]:
    return [
        "Hỏi: Việt Nam là gì Đáp:",
        "Hỏi: Hà Nội ở đâu Đáp:",
        "Hỏi: Nhà Lý là gì Đáp:",
        "Hỏi: Ai chữa bệnh Đáp:",
        "Hỏi: Nếu trời mưa thì Đáp:",
        "Người Minh dùng hỏa tiễn bắn voi",
        "phở là",
        "mùa hè em",
        "Hỏi: Thủ đô của Pháp là gì Đáp:",
        "Hỏi: Triều Lý là gì Đáp:",
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/network_v10.pkl")
    ap.add_argument("--max-len", type=int, default=16)
    ap.add_argument("--multi", action="store_true")
    ap.add_argument("--out", default="out/debug_prompts_batch_network_v10.log")
    ap.add_argument("--prompts-file", default="")
    args = ap.parse_args()

    prompts = default_prompts()
    if args.prompts_file:
        p = Path(args.prompts_file)
        if p.exists():
            lines = [x.strip() for x in p.read_text(encoding="utf-8").splitlines()]
            prompts = [x for x in lines if x]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    state, _ = load_checkpoint(args.checkpoint)
    load_sec = time.time() - t0

    cfg = HCAConfig()
    rows: list[str] = []
    rows.append(f"checkpoint={args.checkpoint}")
    rows.append(f"vocab={len(state.vocab)} parents={len(state.parents.parents)}")
    rows.append(f"load_sec={load_sec:.3f}")
    rows.append(f"max_len={args.max_len} multi={args.multi}")
    rows.append("")

    for i, prompt in enumerate(prompts, start=1):
        t = time.time()
        ids = generate(state, prompt, max_len=args.max_len, multi=args.multi, config=cfg)
        gen_sec = time.time() - t
        out_text = decode_surface(state, ids)
        rows.append(f"[{i}] prompt={prompt}")
        rows.append(f"[{i}] gen_sec={gen_sec:.3f} ids={len(ids)}")
        rows.append(f"[{i}] answer={out_text}")
        rows.append("")

    rows.append(f"total_sec={time.time() - t0:.3f}")
    out_path.write_text("\n".join(rows), encoding="utf-8")
    print(out_path)


if __name__ == "__main__":
    main()
