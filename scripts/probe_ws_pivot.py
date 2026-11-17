#!/usr/bin/env python
"""P94 (TK_046 §13) — **TÁCH TỪ chặn được bao nhiêu phần bệnh bản lề?** (bước 0 của hướng nhãn)

Tác giả đề xuất **đánh nhãn dữ liệu train** (`[CN][DT]Con chó[/DT][/CN][TT]màu đỏ[/TT]`).
Trước khi train lại bất cứ thứ gì, đo **lợi ích rõ nhất, chắc nhất** của nhãn: **ranh giới từ**.

§13.39 đã đo **bệnh bản lề đồng tự** = **10,7%** mối nối SINH (THẬT 0,0%): bản lề là âm tiết
**cuối** của một từ trong A rồi bị đọc thành âm tiết **đầu** của một từ khác trong B
(`quốc gia` ⇒ `gia vị`). Thước đó dựng bằng **bigram đồng hiện** — cùng một đồ thị đang sinh
ra lỗi. Vòng này hỏi lại bằng **trọng tài ĐỘC LẬP**: WS-CRF `out/models/ws_crf_uts_full`,
F1 = **0,9844** trên 95.000 câu, không biết gì về checkpoint.

## Cờ WS — đúng chữ ký của bệnh, đọc bằng bộ tách từ

Với mối nối `(w_left, bản_lề, w_right)`:

| | |
|---|---|
| `dinh_trai` | tách A một mình ⇒ bản lề mang `I-W` (nó **nằm trong** một từ của A) |
| `dinh_phai` | tách `A(≤bản lề) + B(từ w_right)` ⇒ `w_right` mang `I-W` (dính **vào** bản lề) |

`co_WS = dinh_trai AND dinh_phai` — âm tiết bị buộc **cả hai phía**, đúng định nghĩa §13.39
nhưng suy ra từ **ngữ pháp**, không từ đồng hiện.

## Ba cột — thiếu cột nào thì cổng mù (bài học §0.5)

| cột | ý nghĩa |
|---|---|
| **SINH** | mối nối do `compose_chain` tạo |
| **THẬT** (đối chứng) | mối nối giữa hai mệnh đề **liên tiếp cùng một cha** — người viết thật |
| **báo nhầm** | cờ WS trên nhóm `pivot < 0,05`, tức nối mà thước cũ nói là **lành** |

Không có cột THẬT thì mọi tỷ lệ đều "trông cao". Không có cột báo nhầm thì một cổng chặn
100% bệnh **và** 80% nối lành vẫn trông như thắng.

## CỔNG G-ws (nêu TRƯỚC khi chạy)

1. **≥300** mối nối SINH đánh giá được (không bị bỏ vì lệch ranh giới âm tiết).
2. `co_WS`: **SINH − THẬT ≥ +8 điểm phần trăm**. Cờ phải **tách** đối chứng; WS gắn cờ cả nối
   của người viết thì cờ vô nghĩa.
3. **Độ nhạy**: trong nhóm `pivot ≥ 0,05` (bệnh §13.39 đã đếm), WS bắt được **≥ 50%**. Dưới
   ngưỡng đó, tách từ **không phải** công cụ đúng cho lỗi này.
4. **Chi phí**: báo nhầm (`co_WS` trên nhóm `pivot < 0,05`) **< 15%**.

## Dự đoán ghi TRƯỚC (Claude)

- (2) **đạt**, và tôi khá chắc — hai thước độc lập cùng chỉ một hiện tượng thì hiếm khi lệch.
- (3) đoán **60–80%**, không chắc: WS-CRF huấn luyện trên văn **liền mạch**; ở đây B là mệnh
  đề **ghép vào**, phân bố khác dữ liệu huấn luyện của nó.
- (4) là chỗ tôi **lo nhất**. Tiếng Việt có rất nhiều cặp âm tiết ghép được thành từ hợp lệ;
  WS có thể gắn cờ rộng. Nếu (4) trượt thì kết luận là *tách từ thấy bệnh nhưng không dùng làm
  cổng chặn được* — vẫn là kết quả có giá trị, và nó **giết** phần rẻ nhất của hướng nhãn.
- Bốn cổng là **VÀ**. Trượt bất kỳ cổng nào ⇒ ghi TRƯỢT, không diễn giải lại.

Chạy: `PYTHONPATH=src python scripts/probe_ws_pivot.py --n-prompt 400`
"""
from __future__ import annotations

import argparse
import gc
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from nn_aivie.core.checkpoint_io import load_checkpoint  # noqa: E402
from nn_aivie.hca.clause_layer import (  # noqa: E402
    CoverageGate,
    build_clause_layer,
    compose_chain,
    content_folds,
    ensure_boundary_signal,
)
from nn_aivie.hca.readout import decode_surface  # noqa: E402
from nn_aivie.hca.w_skip import ensure_w_skip  # noqa: E402
from nn_aivie.util.progress import get_run_logger  # noqa: E402
from nn_aivie.ws_crf.ws_model import WSCRFModel  # noqa: E402

GATE_MIN_N = 300
GATE_SEP_MIN = 0.08
GATE_RECALL_MIN = 0.50
GATE_FP_MAX = 0.15


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/mini_sent_ep_qa.pkl")
    ap.add_argument("--ws-model", default="out/models/ws_crf_uts_full")
    ap.add_argument("--n-prompt", type=int, default=400)
    ap.add_argument("--max-clauses", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="out/probe_ws_pivot.json")
    args = ap.parse_args()

    logger = get_run_logger("probe_ws_pivot")
    t0 = time.time()
    logger.log(f"P94 tách từ vs bệnh bản lề · {args.checkpoint} · WS {args.ws_model}")
    logger.log(f"cổng: n≥{GATE_MIN_N} · tách≥{GATE_SEP_MIN} · nhạy≥{GATE_RECALL_MIN} "
               f"· báo nhầm<{GATE_FP_MAX}")

    ws_model = WSCRFModel.load(_ROOT / args.ws_model)
    gc.disable()
    state, _ = load_checkpoint(str(_ROOT / args.checkpoint))
    gc.enable()
    layer = build_clause_layer(state, pmi_theta=0.0, logger=logger)
    ws = ensure_w_skip(state, d_max=8, min_count=1, content_only=False)
    bound = ensure_boundary_signal(state)
    logger.log(f"tầng mệnh đề {len(layer):,} ({time.time()-t0:.0f}s)")

    _tag_cache: dict[str, tuple[list[str], list[str]]] = {}

    def seg(text: str) -> tuple[list[str], list[str]]:
        """(âm tiết, nhãn B-W/I-W). Có nhớ đệm — cùng một mệnh đề A lặp lại nhiều lần."""
        hit = _tag_cache.get(text)
        if hit is not None:
            return hit
        syl = text.split()
        tags = ws_model.tag(syl) if syl else []
        if len(tags) != len(syl):          # phòng xa, không được im lặng
            tags = ["B-W"] * len(syl)
        _tag_cache[text] = (syl, tags)
        return syl, tags

    def prefix_syl(ids: list[int], k: int) -> tuple[str, int] | str:
        """(văn bản của ids[:k+1], chỉ số âm tiết của token k), hoặc **lý do** bỏ.

        Bỏ khi `k` không KẾT THÚC đúng một âm tiết — BPE cắt được giữa âm tiết, ép vào sẽ cho
        chỉ số âm tiết sai **âm thầm**. Lý do được đếm và in ra: tỷ lệ bỏ lệch giữa hai nhánh
        là đủ để làm hỏng phép so sánh, nên nó phải nhìn thấy được.
        """
        whole = decode_surface(state, ids)
        pre = decode_surface(state, ids[: k + 1])
        if not pre.strip():
            return "pre_rong"
        if not whole.startswith(pre):
            return "khong_la_tien_to"
        rest = whole[len(pre):]
        # Chỉ CHỮ/SỐ ngay sau mới nghĩa là âm tiết bị cắt đôi. Dấu câu dính liền (`gia` rồi
        # `,`) vẫn là ranh giới âm tiết — bản đầu loại oan cả nhóm này, 63% mối nối SINH.
        if rest[:1].isalnum():
            return "cat_giua_am_tiet"
        # bản lề là token NỘI DUNG cuối của `pre` ⇒ nó luôn là âm tiết cuối cùng
        return pre, len(pre.split()) - 1

    def ws_flag(a_ids: list[int], k: int, b_ids: list[int], j: int
                ) -> tuple[bool, bool] | str:
        """(dính TRÁI trong A, dính PHẢI sau khi nối), hoặc **lý do** bỏ."""
        got = prefix_syl(a_ids, k)
        if isinstance(got, str):
            return got
        pre, hi = got
        if hi < 0:
            return "hi_am"
        _syl_a, tags_a = seg(pre)
        if hi >= len(tags_a):
            return "hi_vuot_A"
        left = tags_a[hi] == "I-W"

        tail = decode_surface(state, b_ids[j:])
        if not tail.strip():
            return "duoi_B_rong"
        joined = pre + (tail if tail[:1].isspace() else " " + tail)
        _syl_j, tags_j = seg(joined)
        if hi + 1 >= len(tags_j):
            return "hi_vuot_noi"
        right = tags_j[hi + 1] == "I-W"
        return left, right

    def joins_of(clauses) -> list[tuple[int, int, int, list[int], int, list[int], int]]:
        """(w_left, bản_lề, w_right, A_ids, k, B_ids, j) — j là vị trí first_content trong B."""
        out = []
        for a, b in zip(clauses, clauses[1:]):
            ta, tb = [int(t) for t in a.tokens], [int(t) for t in b.tokens]
            if len(ta) < 2 or not tb:
                continue
            hinge = int(a.last_content)
            try:
                k = len(ta) - 1 - ta[::-1].index(hinge)
                j = tb.index(int(b.first_content))
            except ValueError:
                continue
            if k == 0:
                continue
            out.append((ta[k - 1], hinge, int(b.first_content), ta, k, tb, j))
        return out

    def collect(pairs) -> dict:
        why: Counter[str] = Counter()
        rec = []
        for wl, hg, wr, ta, k, tb, j in pairs:
            f = ws_flag(ta, k, tb, j)
            if isinstance(f, str):
                why[f] += 1
                continue
            piv = min(bound.p_before(wl, hg), bound.p_after(hg, wr))
            rec.append((f[0], f[1], f[0] and f[1], piv, wl, hg, wr))
        return {"rec": rec, "n_skip": sum(why.values()), "why": dict(why)}

    # --- SINH ---------------------------------------------------------------------------
    rng = random.Random(args.seed)
    openers = [c for c in layer.clauses if c.opens_parent and len(c.tokens) >= 4]
    rng.shuffle(openers)
    prompts = openers[: args.n_prompt]

    gen_pairs = []
    for i, p in enumerate(prompts):
        if i and i % 100 == 0:
            logger.log(f"  sinh {i}/{len(prompts)} ({time.time()-t0:.0f}s)")
        gate = CoverageGate(
            theme_folds=frozenset(content_folds(state, p.tokens)), anchor_min_folds=1)
        ch, _ = compose_chain(
            state, layer, ws, p.tokens, max_clauses=args.max_clauses,
            gate=gate, rng=random.Random(args.seed),
        )
        gen_pairs.extend(joins_of(ch))

    # --- THẬT: mệnh đề liên tiếp cùng một cha --------------------------------------------
    crng = random.Random(args.seed + 7)
    multi = [p_ for p_, ids in layer.by_parent.items() if len(ids) >= 2]
    crng.shuffle(multi)
    real_pairs = []
    for p_ in multi[: max(200, args.n_prompt)]:
        real_pairs.extend(joins_of([layer.clauses[j] for j in layer.by_parent[p_]]))

    G, R = collect(gen_pairs), collect(real_pairs)
    logger.log(f"SINH {len(G['rec']):,} (bỏ {G['n_skip']:,} {G['why']}) · "
               f"THẬT {len(R['rec']):,} (bỏ {R['n_skip']:,} {R['why']}) ({time.time()-t0:.0f}s)")

    def rate(rec, idx=2, sub=None) -> float:
        xs = [r for r in rec if sub is None or sub(r)]
        return sum(1 for r in xs if r[idx]) / len(xs) if xs else 0.0

    print(f"\n{'':<26}{'SINH':>10}{'THẬT':>12}")
    for lab, idx in (("dính TRÁI (I-W trong A)", 0), ("dính PHẢI (sau khi nối)", 1),
                     ("CỜ WS (cả hai phía)", 2)):
        print(f"{lab:<26}{rate(G['rec'], idx):>9.1%}{rate(R['rec'], idx):>12.1%}")
    print(f"{'số mối nối':<26}{len(G['rec']):>9,}{len(R['rec']):>12,}")
    print(f"{'bỏ (lệch âm tiết)':<26}{G['n_skip']:>9,}{R['n_skip']:>12,}")
    for w in sorted(set(G["why"]) | set(R["why"])):
        print(f"{'   · ' + w:<26}{G['why'].get(w, 0):>9,}{R['why'].get(w, 0):>12,}")

    sick = [r for r in G["rec"] if r[3] >= 0.05]
    well = [r for r in G["rec"] if r[3] < 0.05]
    recall = sum(1 for r in sick if r[2]) / len(sick) if sick else 0.0
    fp = sum(1 for r in well if r[2]) / len(well) if well else 0.0
    sep = rate(G["rec"]) - rate(R["rec"])

    print(f"\nĐối chiếu với thước §13.39 (pivot ≥ 0,05) trên nhóm SINH:")
    print(f"  nhóm BỆNH  n={len(sick):<6} WS bắt được   {recall:>6.1%}   (độ nhạy)")
    print(f"  nhóm LÀNH  n={len(well):<6} WS gắn cờ    {fp:>6.1%}   (báo nhầm)")

    ex_catch = [f"{decode_surface(state, [r[4], r[5]])!r} ⇒ "
                f"{decode_surface(state, [r[5], r[6]])!r}" for r in sick if r[2]][:10]
    ex_fp = [f"{decode_surface(state, [r[4], r[5]])!r} ⇒ "
             f"{decode_surface(state, [r[5], r[6]])!r}" for r in well if r[2]][:10]
    if ex_catch:
        print("\n  WS BẮT ĐƯỢC (bệnh):")
        for e in ex_catch:
            print(f"    {e}")
    if ex_fp:
        print("\n  WS BÁO NHẦM (thước cũ nói lành):")
        for e in ex_fp:
            print(f"    {e}")

    g1 = len(G["rec"]) >= GATE_MIN_N
    g2 = sep >= GATE_SEP_MIN
    g3 = recall >= GATE_RECALL_MIN
    g4 = fp < GATE_FP_MAX
    ok = g1 and g2 and g3 and g4
    print(f"\ncờ WS: SINH {rate(G['rec']):.1%} − THẬT {rate(R['rec']):.1%} = {sep:+.1%}")
    print(f"CỔNG G-ws: (1) {'✅' if g1 else '❌'} · (2) {'✅' if g2 else '❌'} · "
          f"(3) {'✅' if g3 else '❌'} · (4) {'✅' if g4 else '❌'} ⇒ "
          f"{'ĐẠT' if ok else 'TRƯỢT'}")

    op = _ROOT / args.out
    op.parent.mkdir(parents=True, exist_ok=True)
    op.write_text(json.dumps({
        "checkpoint": args.checkpoint, "ws_model": args.ws_model,
        "n_prompt": len(prompts),
        "sinh": {"n": len(G["rec"]), "n_bo": G["n_skip"], "ly_do_bo": G["why"],
                 "dinh_trai": rate(G["rec"], 0), "dinh_phai": rate(G["rec"], 1),
                 "co_ws": rate(G["rec"], 2)},
        "that": {"n": len(R["rec"]), "n_bo": R["n_skip"], "ly_do_bo": R["why"],
                 "dinh_trai": rate(R["rec"], 0), "dinh_phai": rate(R["rec"], 1),
                 "co_ws": rate(R["rec"], 2)},
        "tach": sep, "n_benh": len(sick), "do_nhay": recall,
        "n_lanh": len(well), "bao_nham": fp,
        "gate": {"min_n": GATE_MIN_N, "sep_min": GATE_SEP_MIN,
                 "recall_min": GATE_RECALL_MIN, "fp_max": GATE_FP_MAX},
        "gate_pass": bool(ok), "vi_du_bat_duoc": ex_catch, "vi_du_bao_nham": ex_fp,
        "seconds": round(time.time() - t0, 1),
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    logger.log(f"→ {args.out} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
