"""THƯỚC SINH ĐẦU-CUỐI — chấm PHÂN BỐ sau khi bơm, trên ca SẠCH (§78).

Thước v3 (`thuoc_sinh_dung_lop.py`) chấm THẲNG đề cử của chuỗi. Thước AR cũ
(`probe_hub_chain_gen_real`) chấm phân bố nhưng dựng bộ ca máy móc mà §60.8 đã
đo là 46% rác (`P` = `đoàn`·`bán`·`trở`, hạt = `[Giáo, Hà, Ngô, VA]`).

Thước này lấy CA SẠCH của v3 (lớp thật của chân lý v2) và chấm thứ hạng của
lớp đúng **trong phân bố cuối** sau khi đề cử đã được bơm — tức đo đúng thứ
người dùng nhận.

    PYTHONHASHSEED=0 PYTHONPATH=src python scripts/thuoc_sinh_dau_cuoi.py \
        checkpoints/classrich2.pkl 120 boost=4
"""
from __future__ import annotations

import gc
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
gc.disable()

import numpy as np  # noqa: E402

from nn_aivie.core.checkpoint_io import load_checkpoint  # noqa: E402
from nn_aivie.hca import generate_hebbian as G  # noqa: E402
from nn_aivie.hca import hub_spread as H  # noqa: E402
from nn_aivie.hca.config import HCAConfig  # noqa: E402

T0 = time.time()


def main() -> int:
    ck = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/classrich2.pkl"
    n_ca = next((int(a) for a in sys.argv[2:] if a.isdigit()), 120)
    cfg = HCAConfig()
    cfg.hub_chain_boost = 4.0
    for a in sys.argv[2:]:
        if a.startswith("boost="):
            cfg.hub_chain_boost = float(a[6:])
    chi_ct = "cuthe" in sys.argv
    state, _ = load_checkpoint(ck)
    v = state.vocab
    lop = json.load(open("out/chan_ly_v2_classrich2.json"))["lop"]
    _CT = {"chim", "thú", "cây", "cá", "loài", "hoa", "quả", "con", "giống",
           "kim loại", "hành tinh", "tỉnh", "thành phố", "sông", "núi"}
    print(f"\n### {ck} · boost={cfg.hub_chain_boost} · "
          f"{'lớp CỤ THỂ' if chi_ct else 'mọi lớp'}\n")

    ca: list[tuple[str, str, set[str]]] = []
    for c, ms in sorted(lop.items(), key=lambda kv: -len(kv[1])):
        if chi_ct and c.split()[0] not in _CT:
            continue
        ms = [m for m in ms if H.resolve_surface_ids(v, m)]
        if len(ms) < 3:
            continue
        for m in ms[:6]:
            ca.append((m, c, set(ms) - {m}))
    ca = ca[:n_ca]

    d1 = d5 = d10 = n = 0
    hang: list[int] = []
    for m, c, ban in ca:
        ids = [int(t) for t in v.encode_prompt(f"{m} là")]
        if not ids:
            continue
        # phân bố MỘT bước, đúng đường sinh thật
        tr: list = []
        try:
            G.generate_autoregressive(state, f"{m} là", max_len=1,
                                      config=cfg, seed=0, trace_out=tr)
        except Exception:  # noqa: BLE001
            continue
        if not tr or "top" not in tr[0]:
            continue
        n += 1
        from nn_aivie.core.tokenizer import MARK_SPACE as _MS
        top = [str(x.get("token", "")).replace(_MS, "").strip()
               for x in tr[0]["top"]]
        dung = {c, c.split()[0]} | ban
        if top[:1] and top[0] in dung:
            d1 += 1
        if any(x in dung for x in top[:5]):
            d5 += 1
        if any(x in dung for x in top[:10]):
            d10 += 1
        vt = next((i for i, x in enumerate(top, 1) if x in dung), None)
        hang.append(vt if vt else len(top) + 1)
    if not n:
        print("  KHÔNG ca nào chấm được — `trace_out` không trả `top`?")
        return 1
    print(f"  {n} ca · top-1 {100*d1/n:.1f}% · top-5 {100*d5/n:.1f}% · "
          f"top-10 {100*d10/n:.1f}%")
    print(f"  hạng trung vị của lớp đúng: {int(np.median(hang))}")
    print(f"\n[{time.time() - T0:.0f}s] xong")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
