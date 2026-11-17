"""CHẠY LẠI MỌI LUẬT ĐÃ KHAI TỬ — bài học TK_046 §28.5 số 22.

§20.4 khai tử `rank="phieu"` («tệ hơn») trên index cũ; đo lại trên cấu hình hiện
tại thì nó GẤP 27 LẦN. §20.5 và §21 cũng khai tử các luật `head_rule` /
`head_rule_trai` bằng cách quét MỘT vế với vế kia để mặc định — §24.1 đã cho
thấy phép quét đó bỏ sót tương tác (28,6% vs 68,4%).

⇒ Quét LƯỚI ĐẦY ĐỦ 4×4, chấm bằng CẢ HAI thước:
  • phủ cạnh lớp   — cấu trúc có vào được đồ thị không
  • chuyển hoá ②   — có cấu trúc rồi thì bước ② có chọn đúng lớp không

Thước thứ hai mới là thứ quyết định; §24 đã cho thấy phủ tăng mà chuyển hoá đứng.

    PYTHONHASHSEED=0 PYTHONPATH=src python scripts/grid_luat_hub.py \
        --checkpoint checkpoints/classrich.pkl \
        --ws-spans checkpoints/classrich.ws_spans.json --n-prop 200
"""
from __future__ import annotations

import argparse
import gc
import json
import re
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
PHAI = ("max_gidf", "trai_nhat", "trai_du_dac_thu", "dau_lop")
TRAI = ("max_gidf", "phai_nhat", "phai_du_dac_thu", "dau_ngu_doan",
        "phai_nhat_danh_tu")
T0 = time.time()


def log(m: str) -> None:
    print(f"[{time.time() - T0:9.1f}s] {m}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--ws-spans", default="")
    ap.add_argument("--out", default="out/grid_luat.json")
    ap.add_argument("--n-prop", type=int, default=200)
    ap.add_argument("--indeg-q", type=float, default=0.8)
    ap.add_argument("--rank", default="phieu")
    ap.add_argument("--chu-gia-q", type=float, default=0.0)
    ap.add_argument("--gidf-q", type=float, default=0.20)
    ap.add_argument("--chi-la", action="store_true")
    ap.add_argument("--pos-dict", default="", help="data/dict/vi_dict_raw.jsonl")
    ap.add_argument("--chi", default="", help="chỉ chạy các cặp `phai|trai`, phẩy ngăn")
    ap.add_argument("--chan-ly", default="",
                    help="JSON của build_chan_ly_v2 — chân lý v1 («token ngay "
                         "trước `là`») nhặt trạng từ/đuôi ghép/đại từ và chấm SAI "
                         "đúng ở chỗ cơ chế làm ĐÚNG (TK_046 §30).")
    args = ap.parse_args()

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

    log("chân lý: đào lớp …")
    cls: dict[str, set[str]] = defaultdict(set)
    pairs: set[tuple[str, str, str]] = set()
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
            full = h + (" " + S[seq[j + 1]]
                        if j + 1 < len(seq) and seq[j + 1] not in FUNC else "")
            cls[full].add(S[mem])
            pairs.add((S[mem], full, h))
    if args.chan_ly:
        cl2 = json.loads(Path(args.chan_ly).read_text())
        big = {c: set(m) for c, m in cl2["lop"].items()}
        want = {(a, c, c.split()[0])
                for c, ms in big.items() for a in ms}
        log(f"  CHÂN LÝ v2: {len(big)} lớp · {len(want)} cạnh")
    else:
        big = {c: m for c, m in cls.items() if len(m) >= 3}
        want = {(a, c, h) for (a, c, h) in pairs if c in big}
    m2c: dict[str, set[str]] = defaultdict(set)
    for c, ms in big.items():
        for m in ms:
            m2c[m].add(c)
    ALL = set().union(*big.values())
    ung = sorted((c, s_) for s_, c in df.items() if 5 <= c <= 2000)
    ung.sort(key=lambda x: -x[0])
    props = [s_ for _c, s_ in ung[:args.n_prop]]
    # NGỮ CẢNH CÂU cho mỗi thuộc tính — §43 cần nó; trước đây thước chỉ truyền
    # hạt giống nên tín hiệu ngữ cảnh không thể kích hoạt.
    ctx_of: dict[str, list[int]] = {}
    par_of: dict[str, list[int]] = {}
    can = set(props)
    for pi_, rec in enumerate(ps.parents):
        if not can:
            break
        seq = list(rec.emit_ids)
        sr = [S[t] for t in seq]
        for k0, w0 in enumerate(sr):
            if w0 in can and 3 <= k0:
                ctx_of[w0] = [int(t) for t in seq[max(0, k0 - 12):k0 + 12]]
                par_of[w0] = [pi_]
                can.discard(w0)
                break
    log(f"  ngữ cảnh câu cho {len(ctx_of)}/{len(props)} thuộc tính")
    log(f"  lớp={len(big)} · cạnh lớp={len(want)} · thuộc tính={len(props)}")

    wsp = H.load_ws_spans(args.ws_spans) if args.ws_spans else None
    pm = H.load_pos_map(args.pos_dict) if args.pos_dict else None
    if pm:
        log(f"  pos_map = {len(pm)} mục")
    res: dict[str, dict] = {}
    only = [tuple(x.split("|")) for x in args.chi.split(",")] if args.chi else None
    for rp in PHAI:
        for rt in TRAI:
            if only and (rp, rt) not in only:
                continue
            ten = f"{rp} | {rt}"
            t0 = time.time()
            idx = H.build_hub_spread_index(
                state, d_max=6, min_count=0.5, head_rule=rp, head_rule_trai=rt,
                label_indeg_q=args.indeg_q, label_gidf_q=args.gidf_q,
                chu_ngu_ket_hop_q=args.chu_gia_q, pos_map=pm, ws_spans=wsp)
            vp: dict[str, set[str]] = defaultdict(set)
            for src, dsts in idx.via_pivot.items():
                a = H._surf_ix(idx, v, src)
                for dst in dsts:
                    vp[a].add(H._surf_ix(idx, v, dst))
            phu = sum(1 for a, c, h in want
                      if c in vp.get(a, ()) or h in vp.get(a, ()))
            co = d1 = d5 = d1x = d1p = 0
            for P in props:
                ids = H.resolve_surface_ids(v, P)
                if not ids:
                    continue
                s1 = H.score_hubs_from(state, idx, ids[:4], topk=40,
                                       gidf_min=0.0, channel="typed_first")
                lang = [h for h in s1 if h.surface != P][:10]
                trong = [h for h in lang if h.surface in ALL]
                votes: dict[str, int] = defaultdict(int)
                for x in trong:
                    for c2 in m2c[x.surface]:
                        votes[c2] += 1
                chung = {c2 for c2, z in votes.items() if z >= 2}
                if not chung or not trong:
                    continue
                co += 1
                s2 = H.score_hubs_from(state, idx, [h.tid for h in trong[:4]],
                                       topk=20, gidf_min=0.0,
                                       channel="typed_first", rank=args.rank,
                                       loc_pos=bool(pm),
                                       chi_pivot_la=args.chi_la,
                                       ctx_ids=ctx_of.get(P),
                                       par_ids=par_of.get(P))
                cam = {h.surface for h in trong[:4]} | {P}
                ok = [h for h in s2 if h.surface not in cam]
                # TÁCH hai kiểu khớp — §39. `ok[0].surface in (c2, c2.split()[0])`
                # tính ĐÚNG cả khi cơ chế chỉ ra TOKEN ĐẦU của lớp đa âm tiết
                # (`thành` cho `thành phố`). Đó là chấm NƯƠNG, và đúng loại lỗi
                # bề-mặt↔id đã cắn ba lần (§38.6 số 42).
                if ok:
                    if any(ok[0].surface == c2 for c2 in chung):
                        d1 += 1; d1x += 1
                    elif any(ok[0].surface == c2.split()[0] and " " in c2
                             for c2 in chung):
                        d1 += 1; d1p += 1
                if any(any(h.surface in (c2, c2.split()[0]) for c2 in chung)
                       for h in ok[:5]):
                    d5 += 1
            res[ten] = {
                "phu_pct": round(100 * phu / max(len(want), 1), 1),
                "canh_typed": sum(len(d) for d in idx.via_pivot.values()),
                "co_hoi": co,
                "top1_pct": round(100 * d1 / max(co, 1), 1),
                "top1_KHOP_DU_pct": round(100 * d1x / max(co, 1), 1),
                "top1_chi_token_dau_pct": round(100 * d1p / max(co, 1), 1),
                "top5_pct": round(100 * d5 / max(co, 1), 1),
            }
            r = res[ten]
            log(f"  {ten:<40} phủ={r['phu_pct']:5.1f}%  ②top1={r['top1_pct']:5.1f}%"
                f"  (khớp ĐỦ {r['top1_KHOP_DU_pct']:.1f}% + chỉ token đầu "
                f"{r['top1_chi_token_dau_pct']:.1f}%)"
                f"  top5={r['top5_pct']:5.1f}%  (n={co})")
            del idx
            gc.collect()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, ensure_ascii=False, indent=1))
    log("═══ XẾP THEO ②top1 ═══")
    for k, d in sorted(res.items(), key=lambda kv: -kv[1]["top1_pct"])[:8]:
        log(f"  {k:<40} ②top1={d['top1_pct']:5.1f}%  top5={d['top5_pct']:5.1f}%"
            f"  phủ={d['phu_pct']:5.1f}%")
    log(f"✓ → {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
