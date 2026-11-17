#!/usr/bin/env python3
"""Dựng mẫu CHẤM MÙ cho bước 3 hub_spread — đo TUYỆT ĐỐI, không thước proxy.

Vì sao bỏ thước cũ: hai vòng `probe_hub_step3_v12.py` đều KHÔNG diễn giải được
vì ngưỡng `bậc-vào ≥3` bão hoà ở v12 (cột đối chứng `cooc` 88,1% rồi 83,9% —
gần trần). Bài học §13.43: chuỗi phép đo SO SÁNH chạy rất lâu mà không ai biết
mình đang ở đâu ⇒ phải có **mốc tuyệt đối** trước.

Script này KHÔNG chấm. Nó chỉ **in mẫu và giấu nhãn**:

* mỗi dòng = ``seed → hub → 5 thành viên``
* trộn hai nhánh `typed_first` (ứng viên) và `cooc` (ĐỐI CHỨNG), xáo thứ tự
* nhãn nhánh nằm riêng trong `--out-key`, **không** mở khi chấm

Cột đối chứng là bộ tự-kiểm của người chấm: nếu `cooc` cũng được chấm cao thì
thang chấm quá rộng rãi, không phải cơ chế tốt.

    PYTHONPATH=src python scripts/sample_hub_step3_blind.py \
      --checkpoint checkpoints/network_v12_subst_isg_pw.pkl
"""
from __future__ import annotations

import argparse
import gc
import json
import random
import resource
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from nn_aivie.core.checkpoint_io import load_checkpoint  # noqa: E402
from nn_aivie.hca.hub_spread import (  # noqa: E402
    _surf,
    build_hub_spread_index,
    reverse_peers,
    score_hubs_from,
)

T0 = time.monotonic()
_LOG: str | None = None


def log(msg: str) -> None:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1073741824
    line = f"[{time.monotonic() - T0:7.1f}s RSS {rss:5.2f}GB] {msg}"
    print(line, flush=True)
    if _LOG:
        with open(_LOG, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")


def main() -> int:
    global _LOG
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/network_v12_subst_isg_pw.pkl")
    ap.add_argument("--n", type=int, default=30, help="số seed MỖI nhánh")
    ap.add_argument("--seed", type=int, default=46)
    ap.add_argument("--n-members", type=int, default=5)
    ap.add_argument("--d-max", type=int, default=3)
    ap.add_argument("--min-count", type=int, default=2)
    ap.add_argument("--out", default="out/hub3_blind_samples.txt")
    ap.add_argument("--out-key", default="out/hub3_blind_key.json")
    ap.add_argument("--out-meta", default="out/hub3_blind_meta.json")
    ap.add_argument("--log", default="")
    args = ap.parse_args()
    _LOG = args.log or "out/hub3_blind.log"
    Path(_LOG).parent.mkdir(parents=True, exist_ok=True)

    gc.disable()
    log(f"nạp {args.checkpoint} …")
    state, _ = load_checkpoint(str(_ROOT / args.checkpoint))
    gc.enable()
    v = state.vocab
    log(f"nạp xong · vocab={len(v)} parents={len(state.parents)}")

    log("dựng hub_spread index …")
    idx = build_hub_spread_index(
        state, d_max=args.d_max, min_count=args.min_count)
    log(f"cạnh typed: {sum(len(d) for d in idx.via_pivot.values())}")

    # Seed ĐÓNG BĂNG — cùng rng cho mọi nhánh ⇒ so được từng cặp.
    rng = random.Random(args.seed)
    cand = sorted(idx.via_pivot.keys())
    rng.shuffle(cand)
    seeds = cand[: args.n]
    log(f"{len(seeds)} seed đóng băng (từ {len(cand)})")

    rows: list[dict] = []
    for ch in ("typed_first", "cooc"):
        for sid in seeds:
            hubs = score_hubs_from(
                state, idx, [sid], topk=1, gidf_min=0.0,
                ban_func=True, channel=ch)
            if not hubs:
                continue
            h = hubs[0]
            peers = reverse_peers(
                state, idx, [h.tid], exclude={sid},
                topk=args.n_members, channel=ch)
            rows.append({
                "nhanh": ch,
                "seed": _surf(v, sid),
                "hub": h.surface,
                "thanh_vien": [p.surface for p in peers],
                "hub_typed": bool(h.typed),
            })
    rng.shuffle(rows)

    key: dict[str, str] = {}
    meta: dict[str, dict] = {}
    lines: list[str] = [
        "# CHẤM MÙ — bước 3 hub_spread. Mỗi dòng: seed → hub → thành viên.",
        "# Chấm 3 mức, ghi vào file riêng, KHÔNG mở out-key trước khi chấm xong:",
        "#   dung = hub là LỚP thật của seed VÀ thành viên cùng lớp",
        "#   tam  = thành viên mạch lạc nhưng nhãn sai quan hệ",
        "#          (vd `mèo → chó → Shiba, Tạng` — đúng cụm, sai lớp)",
        "#   hong = vô nghĩa",
        "",
    ]
    for i, r in enumerate(rows):
        rid = f"H{i:03d}"
        key[rid] = r["nhanh"]
        meta[rid] = r
        tv = ", ".join(r["thanh_vien"]) or "(rỗng)"
        lines.append(f"[{rid}]  {r['seed']}  →  {r['hub']}  →  {tv}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    Path(args.out_key).write_text(
        json.dumps(key, ensure_ascii=False, indent=1), encoding="utf-8")
    Path(args.out_meta).write_text(
        json.dumps({"checkpoint": args.checkpoint, "n_seed": len(seeds),
                    "n_rows": len(rows), "rows": meta},
                   ensure_ascii=False, indent=1), encoding="utf-8")
    log(f"{len(rows)} dòng → {args.out} · nhãn giấu ở {args.out_key}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
