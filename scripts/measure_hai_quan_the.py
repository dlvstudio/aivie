"""ĐỘ GIAO của HAI QUẦN THỂ — láng giềng của THUỘC TÍNH vs thành viên của LỚP.

TK_046 §25 kết: `đốt → cháy, sống` đúng nghĩa nhưng `cháy`/`sống` không phải
thành viên lớp nào; ép vào vũ trụ lớp thì mất nghĩa. Nghi ngờ: hai quần thể gần
như RỜI NHAU trên corpus dạng này — và nếu đúng thì chuỗi ba bước không chạy
được ở đây vì lý do DỮ LIỆU, không phải cơ chế.

Trong ví dụ của tác giả hai quần thể TRÙNG nhau: `chó`, `mèo` vừa là láng giềng
của `nghịch ngợm` vừa là thành viên của `động vật`. Script này đo xem corpus
thật có tính chất đó không.

Đo trên MẪU RỘNG, không chọn tay — 40 ca chọn tay ở §20 có thể thiên lệch.

    PYTHONHASHSEED=0 PYTHONPATH=src python scripts/measure_hai_quan_the.py \
        --checkpoint checkpoints/classrich.pkl --n-prop 400
"""
from __future__ import annotations

import argparse
import gc
import json
import re
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
gc.disable()

from nn_aivie.core.checkpoint_io import load_checkpoint  # noqa: E402
from nn_aivie.core.tokenizer import MARK_SPACE  # noqa: E402
from nn_aivie.hca import hub_spread as H  # noqa: E402

_DET = frozenset({"một", "các", "những", "mọi", "mỗi"})
T0 = time.time()
RANK = "phieu"


def log(m: str) -> None:
    print(f"[{time.time() - T0:9.1f}s] {m}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--ws-spans", default="")
    ap.add_argument("--out", default="out/hai_quan_the.json")
    ap.add_argument("--n-prop", type=int, default=400)
    ap.add_argument("--rank", default="phieu")
    ap.add_argument("--k", type=int, default=10, help="lấy bao nhiêu láng giềng ①")
    args = ap.parse_args()

    global RANK
    RANK = args.rank
    log(f"nạp {args.checkpoint} …")
    state, _ = load_checkpoint(args.checkpoint)
    v, ps = state.vocab, state.parents
    n = len(v)
    raw = [str(v.token_of(i)) for i in range(n)]
    S = [r.replace(MARK_SPACE, "").strip() for r in raw]
    BAD = {i for i in range(n) if re.match(r"^<.*>$", S[i]) or S[i].endswith(":")}
    LA = {i for i in range(n) if S[i] == "là"}
    DET = {i for i in range(n) if S[i].lower() in _DET}
    FUNC = {i for i in range(n)
            if S[i].lower() in H._FUNC or len(S[i]) <= 1
            or MARK_SPACE not in raw[i]} | BAD

    log("đào lớp …")
    cls: dict[str, set[str]] = defaultdict(set)
    df: dict[str, int] = defaultdict(int)
    for rec in ps.parents:
        seq = list(rec.emit_ids)
        for s_ in {S[t] for t in seq if t not in FUNC}:
            df[s_] += 1
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
            h = S[seq[j]]
            if j + 1 < len(seq) and seq[j + 1] not in FUNC:
                h += " " + S[seq[j + 1]]
            cls[h].add(S[mem])

    log("dựng index …")
    wsp = H.load_ws_spans(args.ws_spans) if args.ws_spans else None
    idx = H.build_hub_spread_index(
        state, d_max=6, min_count=0.5, head_rule="trai_du_dac_thu",
        head_rule_trai="phai_du_dac_thu", label_indeg_q=0.8, ws_spans=wsp)

    # thuộc tính = token nội dung PHỔ BIẾN VỪA PHẢI (không hiếm, không hư từ)
    ung = sorted((c, s_) for s_, c in df.items() if 5 <= c <= 2000)
    ung.sort(key=lambda x: -x[0])
    props = [s_ for _c, s_ in ung[:args.n_prop]]
    log(f"  thuộc tính đem đo = {len(props)}")

    res: dict[str, dict] = {}
    for nguong in (2, 3, 5):
        big = {c: m for c, m in cls.items() if len(m) >= nguong}
        ALL = set().union(*big.values()) if big else set()
        m2c: dict[str, set[str]] = defaultdict(set)
        for c, ms in big.items():
            for m in ms:
                m2c[m].add(c)
        ty_all = len(ALL) / max(n, 1)

        giao: list[float] = []
        co_2 = 0
        co_bat_ky = 0
        n_co_hoi = [0]; n_dung1 = [0]; n_dung5 = [0]
        for P in props:
            ids = H.resolve_surface_ids(v, P)
            if not ids:
                continue
            s1 = H.score_hubs_from(state, idx, ids[:4], topk=args.k * 4,
                                   gidf_min=0.0, channel="typed_first")
            lang = [h.surface for h in s1 if h.surface != P][:args.k]
            if not lang:
                continue
            trong = [x for x in lang if x in ALL]
            giao.append(len(trong) / len(lang))
            if trong:
                co_bat_ky += 1
            votes: dict[str, int] = defaultdict(int)
            for x in trong:
                for c in m2c[x]:
                    votes[c] += 1
            chung = {c for c, z in votes.items() if z >= 2}
            if chung:
                co_2 += 1
                # CHUYỂN HOÁ: có cấu trúc rồi thì bước ② có chọn ĐÚNG lớp không?
                hat = [h for h in s1 if h.surface in trong][:4]
                if hat:
                    s2 = H.score_hubs_from(
                        state, idx, [h.tid for h in hat], topk=20,
                        gidf_min=0.0, channel="typed_first", rank=RANK)
                    cam = {h.surface for h in hat} | {P}
                    ok = [h for h in s2 if h.surface not in cam]
                    n_co_hoi[0] += 1
                    if ok and any(ok[0].surface in (c, c.split()[0]) for c in chung):
                        n_dung1[0] += 1
                    if any(any(h.surface in (c, c.split()[0]) for c in chung)
                           for h in ok[:5]):
                        n_dung5[0] += 1
        N = len(giao)
        res[f"lop>={nguong}"] = {
            "n_lop": len(big), "n_thanh_vien": len(ALL),
            "ty_le_vu_tru": round(100 * ty_all, 2),
            "n_thuoc_tinh": N,
            "giao_trung_binh_pct": round(100 * statistics.mean(giao), 1) if giao else 0,
            "giao_trung_vi_pct": round(100 * statistics.median(giao), 1) if giao else 0,
            "co_it_nhat_1_pct": round(100 * co_bat_ky / max(N, 1), 1),
            "co_2_CUNG_LOP_pct": round(100 * co_2 / max(N, 1), 1),
            "chuyen_hoa_co_hoi": n_co_hoi[0],
            "chuyen_hoa_top1_pct": round(100 * n_dung1[0] / max(n_co_hoi[0], 1), 1),
            "chuyen_hoa_top5_pct": round(100 * n_dung5[0] / max(n_co_hoi[0], 1), 1),
        }
        r = res[f"lop>={nguong}"]
        log(f"── lớp≥{nguong}: {len(big)} lớp · {len(ALL)} thành viên "
            f"({r['ty_le_vu_tru']}% vocab)")
        log(f"     giao trung bình = {r['giao_trung_binh_pct']}%  "
            f"(ngẫu nhiên = {r['ty_le_vu_tru']}%)")
        log(f"     có ≥1 láng giềng là thành viên lớp : {r['co_it_nhat_1_pct']}%")
        log(f"     có ≥2 láng giềng CÙNG MỘT lớp      : {r['co_2_CUNG_LOP_pct']}%  ← điều kiện CẦN của bước ②")
        log(f"     ⇒ CHUYỂN HOÁ ({r['chuyen_hoa_co_hoi']} cơ hội): ② chọn đúng lớp"
            f" top-1 = {r['chuyen_hoa_top1_pct']}%  ·  top-5 = {r['chuyen_hoa_top5_pct']}%")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, ensure_ascii=False, indent=1))
    log(f"✓ → {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
