"""ĐỌC CÂU TRAIN sinh ra một cạnh — công cụ đã giải 5 nút thắt liên tiếp.

TK_046 bài học nặng số 3: đọc câu train giải được ca mà nhiều vòng đo không
giải được (§38 `thành` · §46 họ hàng rác · §54 `phổ biến` · §55 hai bộ lọc ·
§59 `bao gồm`). Script này làm việc đó có hệ thống: cho một cặp
`nguồn → nhãn`, in MỌI câu cha đã sinh ra cạnh đó, kèm pivot.

    PYTHONHASHSEED=0 PYTHONPATH=src python scripts/doc_cau_train_cua_canh.py \
        out/noise_big/toy_noise_150000.pkl chó:bao mèo:bao khỉ:động
"""
from __future__ import annotations

import gc
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
gc.disable()

from nn_aivie.core.checkpoint_io import load_checkpoint  # noqa: E402
from nn_aivie.core.tokenizer import MARK_SPACE  # noqa: E402
from nn_aivie.hca import hub_spread as H  # noqa: E402

T0 = time.time()


def main() -> int:
    ck = sys.argv[1]
    cap = [a.split(":", 1) for a in sys.argv[2:]]
    n_in = 4
    state, _ = load_checkpoint(ck)
    v, ps = state.vocab, state.parents
    S = [str(v.token_of(i)).replace(MARK_SPACE, "").strip() for i in range(len(v))]
    print(f"\n### {ck} · {len(ps.parents)} cha\n")

    for nguon, dich in cap:
        ids_n = set(H.resolve_surface_ids(v, nguon) or [])
        ids_d = set(H.resolve_surface_ids(v, dich) or [])
        print(f"── `{nguon}` → `{dich}` "
              f"({len(ids_n)} biến thể nguồn, {len(ids_d)} đích)")
        thay = 0
        for rec in ps.parents:
            seq = list(rec.emit_ids)
            if not (ids_n & set(seq) and ids_d & set(seq)):
                continue
            i = next(k for k, t in enumerate(seq) if t in ids_n)
            j = next(k for k, t in enumerate(seq) if t in ids_d)
            if not (0 < j - i <= 6):        # nguồn TRƯỚC đích, trong tầm pivot
                continue
            thay += 1
            if thay > n_in:
                continue
            txt = " ".join(S[t] for t in seq if S[t] and not S[t].startswith("<"))
            giua = " ".join(S[t] for t in seq[i + 1:j])
            print(f"   [{thay}] …{txt[:110]}")
            print(f"       nguồn→đích cách {j - i}, ở giữa: {giua!r}")
        print(f"   tổng {thay} câu\n")
    print(f"[{time.time() - T0:.0f}s] xong")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
