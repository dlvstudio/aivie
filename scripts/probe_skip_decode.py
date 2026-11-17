#!/usr/bin/env python3
"""GIẢI MÃ bằng W_skip làm CHỦ ĐẠO + softmax + vòng quay — TK_045 §12.105. **Chỉ đo.**

    PYTHONPATH=src python -u scripts/probe_skip_decode.py \
        --checkpoint checkpoints/mini_sent_ep_qa.pkl \
        --log-file out/skip_decode_mini.log

## Đặc tả (ý tác giả)

Với vị trí kế tiếp `t`, mỗi token lịch sử `h_{t-d}` đề cử qua `W_skip[d]`:

    Việt (d=2) → đẹp, vô, đánh          Nam (d=1) → đẹp, vô, xa
    ⇒ ứng viên {đẹp, vô, đánh, xa};  đẹp 2 phiếu · vô 2 · xa 1 · đánh 1

Điểm gia giảm theo trọng số (khoảng cách xa thì bớt), rồi **softmax** ra xác suất, rồi **vòng
quay** lấy ngẫu nhiên **đúng theo** phân phối ấy — không lấy argmax.

## Vì sao đây KHÁC những gì đã thử

§12.85 (M4) dùng W_skip làm **điểm cộng / cổng cứng** lên trên các kênh sẵn có, `ctx_max=2`, và
đường sinh vẫn **argmax** (`hybrid_temperature=0.0`). Ở đó W_skip không bao giờ được quyết.

Hai thứ mới ở đây:

1. **Sampling thay argmax.** Với argmax, `copy ≈ 1,0` gần như bị **cưỡng bức**: token điểm cao
   nhất luôn thuộc đoạn đã nhớ. Vòng quay là thứ phá vòng chép.
2. **W_skip là phân phối chủ đạo**, không phải phụ gia.

## Ba sửa cài đặt so với `w_skip.py`

* **Chuỗi ĐẦY ĐỦ, không `content_only`.** `w_skip.py` dựng với `content_only=True` ⇒ bỏ hết từ
  chức năng và dấu câu ⇒ sinh ra không có `là`/`của`/`.`. Văn đọc được cần chúng.
* **Cả cửa sổ**, không `ctx_max=2`.
* **Hai cách chấm** để biết phiếu hay độ lớn quan trọng hơn — §12.87 đã đo được rằng **xác suất
  thô thắng PPMI** cho việc trôi chảy, nên độ lớn có thể quan trọng:
  - `vote`  : đếm phiếu (đúng đặc tả) — `Σ_d λ^(d-1) · 1[b ∈ top-k của h_{t-d}]`
  - `mass`  : cộng xác suất — `Σ_d λ^(d-1) · P(b | h_{t-d}, d)`

## Thước

`scripts/probe_fluency_ruler.py` (§12.103) bắt **xáo trộn thô** rất mạnh (+3,64 nat/token) dù mù
với mối ghép tinh vi. Lỗi của sampling giống **xáo** hơn giống **ghép**, nên thước này dùng được
đúng chỗ. Kèm đọc tay — bắt buộc.

Mốc §12.103: corpus **−5,91** · văn sinh hiện tại **−3,97** · xáo **−9,55**.
Sinh bằng vòng quay mà tụt về phía −9,5 ⇒ văn què. Giữ được quanh −6 ⇒ đúng vùng văn thật.
"""

from __future__ import annotations

import argparse
import gc
import math
import random
import statistics as stx
import sys
import time
from collections import defaultdict
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from nn_aivie.core.checkpoint_io import load_checkpoint  # noqa: E402

PROMPTS = [
    "Việt Nam", "Nhà Lý dời đô về", "Ngô Quyền đánh bại quân",
    "Hoa Lư là kinh đô của", "Sông Hồng chảy qua", "Thủ đô Việt Nam là",
    "Trần Hưng Đạo", "Chiến thắng Bạch Đằng năm",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/mini_sent_ep_qa.pkl")
    ap.add_argument("--log-file", default="out/skip_decode.log")
    ap.add_argument("--d-max", type=int, default=8)
    ap.add_argument("--lam", type=float, default=0.85)
    ap.add_argument("--topk", type=int, default=40, help="số đề cử mỗi proposer (cho `vote`)")
    ap.add_argument("--min-count", type=int, default=2)
    ap.add_argument("--temp", type=float, default=1.0, help="nhiệt softmax")
    ap.add_argument("--max-len", type=int, default=30)
    ap.add_argument("--n-sample", type=int, default=3, help="số lần sinh mỗi prompt")
    ap.add_argument("--sweep", action="store_true", help="§12.105c quét λ/top-k/nhiệt")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    logf = (_ROOT / args.log_file).open("w", encoding="utf-8")
    t0 = time.time()

    def log(m=""):
        print(m, flush=True)
        logf.write(str(m) + "\n")
        logf.flush()

    log(f"[{time.time()-t0:.0f}s] load {args.checkpoint} …")
    gc.disable()
    state, _ = load_checkpoint(str(_ROOT / args.checkpoint))
    gc.enable()
    v, ps = state.vocab, state.parents
    log(f"[{time.time()-t0:.0f}s] ✓ {len(ps):,} cha")

    eos = getattr(v, "eos_id", None)
    if eos is None:
        eos = v._token_to_id.get("<EOS>")  # noqa: SLF001

    # ---- W_skip trên chuỗi ĐẦY ĐỦ (khác `w_skip.py`: không content_only) ------------------
    D = args.d_max
    cnt = [defaultdict(lambda: defaultdict(float)) for _ in range(D + 1)]
    ctot = [defaultdict(float) for _ in range(D + 1)]
    seqs = []
    for rec in ps.parents:
        e = [int(x) for x in (getattr(rec, "emit_ids", ()) or ())]
        if len(e) >= 4:
            seqs.append(e)
    for e in seqs:
        L = len(e)
        for i in range(L):
            for d in range(1, min(D, L - i - 1) + 1):
                cnt[d][e[i]][e[i + d]] += 1.0
                ctot[d][e[i]] += 1.0
    log(f"[{time.time()-t0:.0f}s] ✓ W_skip đầy đủ: {len(seqs):,} câu · "
        f"{sum(len(cnt[d]) for d in range(1, D+1)):,} hàng")

    top: list[dict[int, list[tuple[int, float]]]] = [dict() for _ in range(D + 1)]
    for d in range(1, D + 1):
        for a, row in cnt[d].items():
            it = [(b, w) for b, w in row.items() if w >= args.min_count]
            if not it:
                continue
            it.sort(key=lambda x: -x[1])
            top[d][a] = it[: args.topk]
    log(f"[{time.time()-t0:.0f}s] ✓ top-{args.topk} mỗi hàng")

    # ---- thước trôi chảy (bigram trên cùng corpus) — mốc §12.103 --------------------------
    bi: dict[int, dict[int, float]] = defaultdict(lambda: defaultdict(float))
    bctx: dict[int, float] = defaultdict(float)
    uni: dict[int, float] = defaultdict(float)
    utot = 0.0
    for e in seqs:
        for t_ in e:
            uni[t_] += 1.0
            utot += 1.0
        for a, b in zip(e, e[1:]):
            bi[a][b] += 1.0
            bctx[a] += 1.0

    def flu(seq: list[int]) -> float:
        if len(seq) < 2:
            return float("nan")
        tot = 0.0
        for x, y in zip(seq, seq[1:]):
            pu = max(uni.get(y, 0.0) / utot, 1.0 / utot)
            p = (bi.get(x, {}).get(y, 0.0) + pu) / (bctx.get(x, 0.0) + 1.0)
            tot += math.log(max(p, 1e-300))
        return tot / (len(seq) - 1)

    rng = random.Random(args.seed)

    def step(hist: list[int], mode: str) -> int | None:
        """Điểm ứng viên theo đặc tả tác giả, softmax, rồi VÒNG QUAY."""
        sc: dict[int, float] = defaultdict(float)
        for d in range(1, min(D, len(hist)) + 1):
            src = hist[-d]
            w = args.lam ** (d - 1)
            if mode == "vote":
                for b, _c in top[d].get(src, ()):
                    sc[b] += w                      # ĐẾM PHIẾU — đúng đặc tả
            else:
                row = cnt[d].get(src)
                if not row:
                    continue
                s_ = ctot[d].get(src, 0.0)
                if s_ <= 0:
                    continue
                for b, c in row.items():
                    if c >= args.min_count:
                        sc[b] += w * (c / s_)       # CỘNG XÁC SUẤT
        if not sc:
            return None
        ids = list(sc.keys())
        mx = max(sc.values())
        ex = [math.exp((sc[i] - mx) / max(args.temp, 1e-6)) for i in ids]
        tot = sum(ex)
        if tot <= 0:
            return None
        # VÒNG QUAY: xác suất ra token ĐÚNG BẰNG softmax (tích luỹ + một số ngẫu nhiên đều)
        r = rng.random() * tot
        acc = 0.0
        for i, w_ in zip(ids, ex):
            acc += w_
            if acc >= r:
                return i
        return ids[-1]

    def gen(prompt: str, mode: str) -> list[int]:
        hist = [int(t) for t in v.encode_prompt(prompt)]
        out: list[int] = []
        for _ in range(args.max_len):
            nx = step(hist + out, mode)
            if nx is None or (eos is not None and nx == eos):
                break
            out.append(nx)
        return out

    def sur(ids) -> str:
        return "".join(v.token_of(int(t)) for t in ids).replace("Ġ", " ").replace("Ċ", " ")

    if args.sweep:
        # §12.105c — QUÉT hiệu chỉnh. Điểm phiếu nằm trong dải 1..8; softmax nhiệt 1.0 trên dải
        # hẹp ấy ≈ ĐỀU trên ~300 ứng viên ⇒ salad. Ba tham số tác giả nêu ("gia giảm tuỳ trọng
        # số") đều có hướng dẫn từ số đã đo: λ nhỏ vì §12.82 cho d=1 là 82/18 còn d≥3 chỉ 55/45;
        # top-k nhỏ để bể ứng viên khỏi loãng; nhiệt thấp để phiếu thật sự phân biệt.
        log("\n" + "=" * 78)
        log("QUÉT HIỆU CHỈNH (chế độ `vote`) — logP/token, mốc: corpus −5.91 · xáo −9.55")
        log("=" * 78)
        log(f"  {'λ':>5}{'top-k':>7}{'nhiệt':>8}{'logP/token':>13}")
        # §12.105d SỬA TIÊU CHÍ: docstring nói "mức corpus là ĐÍCH, không phải cực đại", nhưng
        # bản đầu chọn max ⇒ ra λ=.85/k=40/T=.25 = −2.606, CAO HƠN cả bộ sinh chép (−3.97), tức
        # quay về chép. Chọn cấu hình GẦN MỨC CORPUS nhất.
        CORPUS = -5.909
        best = (None, 1e9, float("nan"))
        for lam_ in (0.85, 0.5, 0.25):
            for tk in (5, 12, 40):
                for tp in (0.1, 0.25, 0.6):
                    args.lam, args.topk, args.temp = lam_, tk, tp
                    top2 = [dict() for _ in range(D + 1)]
                    for d in range(1, D + 1):
                        for a, row in cnt[d].items():
                            it = [(b, w) for b, w in row.items() if w >= args.min_count]
                            if it:
                                it.sort(key=lambda x: -x[1])
                                top2[d][a] = it[:tk]
                    old, top[:] = top[:], top2
                    fl = []
                    for p_ in PROMPTS:
                        for _ in range(2):
                            o = gen(p_, "vote")
                            if len(o) >= 3:
                                fl.append(flu(o))
                    top[:] = old
                    mv = stx.mean(fl) if fl else float("nan")
                    log(f"  {lam_:>5}{tk:>7}{tp:>8}{mv:>13.3f}")
                    if mv == mv and abs(mv - CORPUS) < best[1]:
                        best = ((lam_, tk, tp), abs(mv - CORPUS), mv)
        if best[0]:
            log(f"\n  GẦN MỨC CORPUS NHẤT: λ={best[0][0]} · top-k={best[0][1]} · "
                f"nhiệt={best[0][2]} → {best[2]:.3f}  (corpus {CORPUS:.3f})")
            log("  (KHÔNG chọn cực đại: cực đại = chép. Xem §12.105d.)")
            args.lam, args.topk, args.temp = best[0]
            for d in range(1, D + 1):
                for a, row in cnt[d].items():
                    it = [(b, w) for b, w in row.items() if w >= args.min_count]
                    if it:
                        it.sort(key=lambda x: -x[1])
                        top[d][a] = it[: best[0][1]]
            args.lam, args.topk, args.temp = best[0]

    for mode in (("vote",) if args.sweep else ("vote", "mass")):
        log("\n" + "=" * 92)
        log(f"CHẾ ĐỘ `{mode}`  (λ={args.lam} · d≤{D} · top-k={args.topk} · nhiệt={args.temp})")
        log("=" * 92)
        fl = []
        for p_ in PROMPTS:
            for k in range(args.n_sample):
                o = gen(p_, mode)
                if len(o) >= 3:
                    fl.append(flu(o))
                mark = "" if k else f"{p_!r}"
                log(f"  {mark:<34} → {sur(o)[:96]!r}")
        if fl:
            log(f"\n  logP/token của văn sinh: **{stx.mean(fl):.3f}**  (n={len(fl)})")

    # §12.105e — ĐƯỜNG RÃ. Đọc tay thấy văn "bắt đầu đúng rồi rã trong ~5 token". Đo thay vì
    # đoán: logP tại TỪNG VỊ TRÍ của phần sinh, gộp qua nhiều mẫu. Nếu rã đơn điệu thì có một
    # cửa sổ dùng được, và độ dài cửa sổ ấy là con số cần biết.
    log("\n" + "=" * 78)
    log("ĐƯỜNG RÃ — logP tại từng vị trí của phần sinh (λ=%.2f · k=%d · nhiệt=%.2f)"
        % (args.lam, args.topk, args.temp))
    log("=" * 78)
    bypos: dict[int, list[float]] = defaultdict(list)
    for p_ in PROMPTS:
        for _ in range(6):
            o = gen(p_, "vote")
            if len(o) < 3:
                continue
            for i, (x, y) in enumerate(zip(o, o[1:])):
                pu = max(uni.get(y, 0.0) / utot, 1.0 / utot)
                pr = (bi.get(x, {}).get(y, 0.0) + pu) / (bctx.get(x, 0.0) + 1.0)
                bypos[i].append(math.log(max(pr, 1e-300)))
    log(f"  {'vị trí':>7}{'logP':>10}{'n':>7}")
    for i in range(min(14, max(bypos) + 1 if bypos else 0)):
        xs = bypos.get(i, [])
        if len(xs) >= 5:
            log(f"  {i+1:>7}{stx.mean(xs):>10.3f}{len(xs):>7}")
    log("  Rã đơn điệu ⇒ có CỬA SỔ dùng được; độ dài cửa sổ là con số cần biết.")

    log("\n" + "=" * 92)
    log("MỐC §12.103 để so:  corpus **−5,91**  ·  văn sinh HIỆN TẠI **−3,97**  ·  xáo **−9,55**")
    log("Tụt về phía −9,5 ⇒ văn què. Quanh −6 ⇒ đúng vùng văn thật. Cao hơn −4 ⇒ vẫn đang chép.")
    log("Nhưng ĐỌC TAY là cột phán — §12.103 đo được thước này mù với mối ghép tinh vi.")
    log(f"\n[{time.time()-t0:.0f}s] → {args.log_file}")
    logf.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
