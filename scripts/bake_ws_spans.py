#!/usr/bin/env python3
"""Bake span TỪ (WS-CRF) ra sidecar — chạy MỘT LẦN, dùng lại mãi.

Vì sao cần: neo nhãn/chủ ngữ vào cụm ISG cho ra `tụ vĩ` (từ «lãnh tụ vĩ đại»),
`Bốn đài`, `Các tôn`. Đo trên `mini_sent_ep_qa.pkl`: **57%** cụm ISG không phải
đơn vị ngôn ngữ. Thay bằng span WS-CRF (TK_034, F1 0,9844) còn **15%** — và
15% đó gần như toàn `8 0` / `1 9 5 2` (BPE tách số), đã lọc trong script này.

Chi phí đo được: **40 ms/cha** ⇒ 430k cha ≈ 4,8 giờ nếu chạy tất. Hai đòn cắt
đã bật mặc định: chỉ cha CÓ pivot (`--only-pivot`), bỏ span toàn chữ số.

KHÔNG train lại gì cả. Span là **view dẫn xuất** từ `emit_ids` đã có — cùng họ
với W_skip / E_skip (TK_045 §0.W A.3: *"Cần train mới không? → KHÔNG"*).

    PYTHONPATH=src python scripts/bake_ws_spans.py \
      --checkpoint checkpoints/network_v12_subst_isg_pw.pkl
"""
from __future__ import annotations

import argparse
import gc
import resource
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from nn_aivie.core.checkpoint_io import load_checkpoint  # noqa: E402
from nn_aivie.hca.hub_spread import (  # noqa: E402
    _surf,
    load_ws_spans,
    save_ws_spans,
    ws_word_spans,
)
from nn_aivie.ws_crf.ws_model import WSCRFModel  # noqa: E402

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
    ap.add_argument("--ws-model", default="out/models/ws_crf_uts_full")
    ap.add_argument("--out", default="")
    ap.add_argument("--only-pivot", action="store_true", default=True)
    ap.add_argument("--all-parents", dest="only_pivot", action="store_false")
    ap.add_argument("--max-parents", type=int, default=0, help="0 = tất cả")
    ap.add_argument("--log", default="")
    args = ap.parse_args()
    ck = Path(args.checkpoint)
    out = args.out or str(ck.with_suffix("")) + ".ws_spans.json"
    _LOG = args.log or f"out/bake_ws_spans_{ck.stem}.log"
    Path(_LOG).parent.mkdir(parents=True, exist_ok=True)

    gc.disable()
    log(f"nạp {args.checkpoint} …")
    state, _ = load_checkpoint(str(_ROOT / args.checkpoint))
    gc.enable()
    log(f"nạp xong · parents={len(state.parents)} vocab={len(state.vocab)}")

    wsm = WSCRFModel.load(_ROOT / args.ws_model)
    log(f"WS-CRF nạp xong ({args.ws_model})")

    t = time.monotonic()
    spans = ws_word_spans(
        state, wsm,
        max_parents=args.max_parents,
        only_pivot_parents=args.only_pivot,
        progress=log,
    )
    dt = time.monotonic() - t
    log(f"{len(spans)} span · {dt/60:.1f} phút")

    save_ws_spans(out, spans)
    lai = load_ws_spans(out)
    assert lai == spans, "roundtrip sidecar SAI"
    mb = Path(out).stat().st_size / 1048576
    log(f"ghi {out} ({mb:.1f} MB) · roundtrip OK")

    v = state.vocab
    # 30 mẫu, rải đều — runbook bắt ĐỌC TAY vì thước tự động mù (xem dưới).
    ex = sorted(spans)[:: max(1, len(spans) // 30)][:30]
    for i, x in enumerate(ex):
        log(f"  mẫu[{i:02d}] {' '.join(_surf(v, t) for t in x)!r}")
    from nn_aivie.hca.context_positional import is_structural_token
    from nn_aivie.hca.hub_spread import _FUNC
    ban = sum(
        1 for x in spans
        if any(not any(c.isalnum() for c in _surf(v, t)) for t in x)
        or all(_surf(v, t).isdigit() for t in x)
        or any(is_structural_token(v, t) for t in x)
        or _surf(v, x[0]) in _FUNC or _surf(v, x[0]).lower() in _FUNC
    )
    log(f"span nghi rác (tự động): {ban}/{len(spans)} = "
        f"{ban/max(len(spans),1):.1%} (ISG: 57%)")
    log("⚠️ thước tự động MÙ với mảnh BPE tên ngoại lai (`René Lac os te`, "
        "`Bat ư`) — PHẢI đọc tay mẫu bên dưới, đừng chỉ tin %.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
