#!/usr/bin/env python3
from __future__ import annotations

import argparse
import faulthandler
import time
from pathlib import Path

from nn_aivie.core.checkpoint_io import load_checkpoint
from nn_aivie.hca.config import HCAConfig
from nn_aivie.hca.generate_hebbian import generate
from nn_aivie.hca.readout import decode_surface


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/network_v10.pkl")
    ap.add_argument("--prompt", default="Hỏi: Việt Nam là gì Đáp:")
    ap.add_argument("--max-len", type=int, default=8)
    ap.add_argument("--multi", action="store_true")
    ap.add_argument("--dump-every", type=int, default=30)
    ap.add_argument("--log", default="out/debug_single_prompt_network_v10.log")
    ap.add_argument("--stack", default="out/debug_single_prompt_network_v10.stack.log")
    args = ap.parse_args()

    log_path = Path(args.log)
    stack_path = Path(args.stack)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    stack_path.parent.mkdir(parents=True, exist_ok=True)

    with log_path.open("w", encoding="utf-8") as log, stack_path.open("w", encoding="utf-8") as sf:
        faulthandler.enable(sf)
        faulthandler.dump_traceback_later(max(1, args.dump_every), repeat=True, file=sf)

        t0 = time.time()
        log.write(f"start={t0}\n")
        log.flush()

        t = time.time()
        state, _ = load_checkpoint(args.checkpoint)
        log.write(
            f"load_sec={time.time() - t:.3f} vocab={len(state.vocab)} parents={len(state.parents.parents)}\n"
        )
        log.flush()

        t = time.time()
        ids = generate(state, args.prompt, max_len=args.max_len, multi=args.multi, config=HCAConfig())
        log.write(f"gen_sec={time.time() - t:.3f} ids={len(ids)}\n")
        log.write(f"out={decode_surface(state, ids)}\n")
        log.write(f"total_sec={time.time() - t0:.3f}\n")
        log.flush()

        faulthandler.cancel_dump_traceback_later()

    print(f"LOG={log_path}")
    print(f"STACK={stack_path}")


if __name__ == "__main__":
    main()
