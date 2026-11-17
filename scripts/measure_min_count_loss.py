"""Đo tầm với của lỗi `min_count` — nó cắt mất bao nhiêu CÂU PHÁT BIỂU LỚP?

`build_w_skip` cộng ``pw = power × priority`` vào ô, rồi `_aggregate_mats` lọc
``if w < min_count``. Tên là "count" nhưng so với TỔNG TRỌNG SỐ, nên với
``priority=0.9`` thì ``min_count=2`` thực ra là ``count ≥ 3``.

Script này KHÔNG dựng W_skip (quá nặng). Nó tính trực tiếp trọng số của đúng
loại cạnh ta cần — cặp ``(thành_viên, đầu_lớp)`` trong câu ``X là C`` — rồi đếm
bao nhiêu cặp rơi xuống dưới ngưỡng.

Chạy::

    PYTHONPATH=src python scripts/measure_min_count_loss.py \
        --checkpoint checkpoints/classrich.pkl
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

gc.disable()

from nn_aivie.core.checkpoint_io import load_checkpoint  # noqa: E402
from nn_aivie.core.tokenizer import MARK_SPACE  # noqa: E402
from nn_aivie.hca.hub_spread import _FUNC  # noqa: E402

_DET = frozenset({"một", "các", "những", "mọi", "mỗi", "cái", "con", "người"})
T0 = time.time()


def log(msg: str) -> None:
    print(f"[{time.time() - T0:9.1f}s] {msg}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", default="out/min_count_loss.json")
    args = ap.parse_args()

    log(f"nạp {args.checkpoint} …")
    state, _ = load_checkpoint(args.checkpoint)
    vocab, ps = state.vocab, state.parents
    n = len(vocab)

    def surf(t: int) -> str:
        return str(vocab.token_of(int(t))).replace(MARK_SPACE, "").strip()

    S = [surf(i) for i in range(n)]
    LA = {i for i in range(n) if S[i] == "là"}
    DET = {i for i in range(n) if S[i].lower() in _DET}
    FUNC = {i for i in range(n) if S[i].lower() in _FUNC or len(S[i]) <= 1}

    log("quét cặp (thành_viên → đầu_lớp) qua `là` …")
    w_pair: dict[tuple[str, str], float] = defaultdict(float)
    c_pair: dict[tuple[str, str], int] = defaultdict(int)
    for pi, rec in enumerate(ps.parents):
        if pi % 5000 == 0:
            log(f"  cha {pi}/{len(ps.parents)} · cặp={len(w_pair)}")
        pw = float(max(1, rec.power)) * float(getattr(rec, "priority", 1.0) or 1.0)
        seq = list(rec.emit_ids)
        for k, t in enumerate(seq):
            if t not in LA or k == 0 or k + 1 >= len(seq):
                continue
            mem = seq[k - 1]
            if mem in FUNC or mem in DET:
                continue
            j = k + 1
            while j < len(seq) and seq[j] in DET:
                j += 1
            if j >= len(seq) or seq[j] in FUNC:
                continue
            key = (S[mem], S[seq[j]])
            w_pair[key] += pw
            c_pair[key] += 1

    tot = len(w_pair)
    log(f"  tổng cặp riêng = {tot}")
    res = {"checkpoint": args.checkpoint, "n_cap": tot, "nguong": {}}
    for mc in (1, 2, 3):
        keep = sum(1 for w in w_pair.values() if w >= mc)
        # đối chứng: nếu so ĐÚNG số lần (count) thay vì trọng số
        keep_c = sum(1 for c in c_pair.values() if c >= mc)
        res["nguong"][str(mc)] = {
            "giu_theo_trong_so": keep,
            "giu_theo_so_lan": keep_c,
            "mat_do_loi": keep_c - keep,
            "pct_mat": round(100.0 * (keep_c - keep) / max(keep_c, 1), 1),
        }
        log(f"  min_count={mc}: trọng số giữ {keep} · số lần giữ {keep_c} "
            f"· LỖI CẮT MẤT {keep_c - keep} ({res['nguong'][str(mc)]['pct_mat']}%)")

    # ví dụ cặp bị cắt oan ở min_count=2
    victims = sorted(
        ((k, w_pair[k], c_pair[k]) for k in w_pair if c_pair[k] >= 2 and w_pair[k] < 2),
        key=lambda x: -x[2])[:40]
    res["nan_nhan_min_count_2"] = [
        {"cap": f"{a} là {b}", "trong_so": round(w, 3), "so_lan": c}
        for (a, b), w, c in victims
    ]
    log(f"  nạn nhân (count≥2 nhưng w<2) = {len(victims)} mẫu đầu")
    for v in res["nan_nhan_min_count_2"][:15]:
        log(f"    {v['cap']:40} count={v['so_lan']} w={v['trong_so']}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, ensure_ascii=False, indent=1))
    log(f"✓ → {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
