#!/usr/bin/env python
"""P95 (TK_046 §14) — **NƠ-RON NHÃN TỪ LOẠI**: dựng, rồi đo nó có mang thông tin không.

Tác giả (2026-08-05) quyết: *"định nghĩa trước các nơ ron như vậy (theo nhãn) rồi chủ động kết
nối các nơ ron token/cụm vào nhãn — lấy sẵn từ bộ từ điển"*. Tôi đã nêu ý kiến ngược (§12: nên
để nhãn lại); tác giả giữ nguyên ⇒ làm đầy đủ.

## Đã có sẵn, không phải xây mới

TK_015 đã có: `vocab.pos_id(pos)` (nơ-ron hub từ-loại), `EdgeKind.POS_ANCHOR`, `EdgeKind.LEXATTR`
(từ → lớp-từ). `data/dict/vi_dict_raw.jsonl` có **36.764 mục**, mỗi mục một `part_of_speech`
tiếng Việt. Nhưng checkpoint mini có **0 cạnh** ba loại đó — từ điển **chưa từng được nạp**.

## Vấn đề CỐT LÕI phải đo trước khi ghi bất cứ cạnh nào

Vốn từ là **BPE ~mức âm tiết** (32.002), còn từ điển đánh nhãn ở **mức TỪ**. `động vật` là
*danh từ*, nhưng nó vỡ thành hai âm tiết, mà `động` còn nằm trong `động cơ`, `hoạt động`,
`chuyển động`, `bị động`. Nhãn vì thế **nhoè** khi hạ xuống âm tiết.

⇒ Trước khi dựng cạnh, đo: nhãn còn lại **bao nhiêu bit thông tin**? Nếu mỗi âm tiết đều rải
đều trên 3–4 từ loại thì nơ-ron nhãn **không phân biệt được gì**, và mọi thứ xây trên nó đều vô
nghĩa — biết ngay tại đây, không phải sau khi train.

## CỔNG G-pos (nêu TRƯỚC khi chạy)

1. **Phủ**: ≥ **60%** *lượt xuất hiện* token nội dung trong cha mini nhận được ≥1 nhãn.
   (Đếm theo **lượt**, không theo **loại** — khung dùng lượt.)
2. **Sạch**: phần của từ loại **trội nhất**, trung vị theo lượt, ≥ **0,70**. Dưới ngưỡng đó
   nhãn âm tiết là cháo, không phải vai.
3. **Không suy biến**: lớp từ loại **lớn nhất** chiếm < **60%** số lượt có nhãn. Một nhãn mà
   4/5 token đều mang thì không mang bit nào — đúng bài học §13.65 ("≥2 thực thể" bão hoà
   94,8% ⇒ bộ lọc không lọc gì).

## Dự đoán ghi TRƯỚC (Claude)

- (1) **đạt**, đoán 70–85%: từ điển tiếng Việt phủ tốt các âm tiết thông dụng.
- (2) là chỗ tôi **lo nhất**, đoán **0,55–0,70** ⇒ có thể **trượt**. Chính hiện tượng nhoè nêu
  trên. Nếu trượt, kết luận là *nhãn từ điển không hạ được xuống mức âm tiết*, và lối đi còn
  lại là đánh nhãn **lượt trong corpus** bằng WS-CRF chứ không phải tra từ điển theo loại.
- (3) borderline, đoán **45–60%**: "danh từ" là lớp lớn nhất trong mọi từ điển.
- Ba cổng là **VÀ**. Chỉ khi ĐẠT mới ghi cạnh vào checkpoint (`--write`).

Chạy:  `PYTHONPATH=src python scripts/build_pos_labels.py`
Ghi:   `PYTHONPATH=src python scripts/build_pos_labels.py --write`
"""
from __future__ import annotations

import argparse
import gc
import json
import re
import math
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from nn_aivie.core.checkpoint_io import load_checkpoint  # noqa: E402
from nn_aivie.core.edge_kinds import EdgeKind  # noqa: E402
from nn_aivie.hca.context_positional import is_structural_token  # noqa: E402
from nn_aivie.util.progress import get_run_logger  # noqa: E402

# nhãn phong cách được giữ: chỉ những mục thật sự là PHẠM VI DÙNG, bỏ chú giải ngữ nghĩa
_REGISTER = frozenset({
    "khẩu ngữ", "cũ", "phương ngữ", "hiếm", "văn chương", "thông tục", "trang trọng",
    "ít dùng", "từ cũ", "nói tắt", "kiêng", "mỉa mai",
})

_PUNCT = ".,!?;:…\"'()"

MIN_POS_WORDS = 20

GATE_COVER_MIN = 0.60
GATE_PURITY_MIN = 0.70
GATE_TOP_CLASS_MAX = 0.60


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/mini_sent_ep_qa.pkl")
    ap.add_argument("--dict", default="data/dict/vi_dict_raw.jsonl")
    ap.add_argument("--level", choices=("type", "word"), default="type",
                    help="type = tra từ điển theo LOẠI âm tiết · word = tách từ rồi tra TỪ")
    ap.add_argument("--ws-model", default="out/models/ws_crf_uts_full")
    ap.add_argument("--write", action="store_true",
                    help="ghi cạnh token→nhãn vào checkpoint mới (chỉ khi cổng ĐẠT)")
    ap.add_argument("--out-checkpoint", default="checkpoints/mini_sent_ep_qa_pos.pkl")
    ap.add_argument("--emit-map", default="",
                    help="xuất token → {tu_loai, phong_cach} (mức TỪ) cho probe sàng nhãn")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    if not args.out:
        args.out = f"out/pos_labels_mini_{args.level}.json"

    logger = get_run_logger("build_pos_labels")
    t0 = time.time()
    logger.log(f"P95 nơ-ron nhãn từ loại · mức {args.level.upper()} · "
               f"{args.checkpoint} · {args.dict}")
    logger.log(f"cổng: phủ≥{GATE_COVER_MIN} · sạch≥{GATE_PURITY_MIN} "
               f"· lớp lớn nhất<{GATE_TOP_CLASS_MAX}")

    gc.disable()
    state, _ = load_checkpoint(str(_ROOT / args.checkpoint))
    gc.enable()
    v, ps = state.vocab, state.parents
    n = len(v)

    # --- từ điển: từ → tập từ loại -------------------------------------------------------
    word_pos: dict[str, set[str]] = defaultdict(set)
    word_reg: dict[str, set[str]] = defaultdict(set)
    with open(_ROOT / args.dict, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                e = json.loads(line)
            except Exception:
                continue
            w, p = e.get("word"), e.get("part_of_speech")
            # ⚠️ 1.735/36.764 mục có part_of_speech = null (thành ngữ, tổ hợp). `str(None)`
            # biến chúng thành một nơ-ron nhãn giả tên "none" — bản đầu của script này dính
            # đúng lỗi đó và "none" chiếm 24,4% lớp trội. Bỏ hẳn, không đặt nhãn thay thế.
            if not isinstance(w, str) or not isinstance(p, str):
                continue
            w, p = w.strip().lower(), p.strip().lower()
            if w and p:
                word_pos[w].add(p)
            # nhãn PHONG CÁCH/PHẠM VI: ngoặc mở đầu định nghĩa — 34,8% mục có, chưa ai dùng
            m_ = e.get("meaning")
            if isinstance(w, str) and isinstance(m_, str):
                g = re.match(r"\s*\(([^)]{1,25})\)", m_)
                if g:
                    for part in g.group(1).lower().split(","):
                        part = part.strip()
                        if part in _REGISTER:
                            word_reg[w.strip().lower()].add(part)
    # bỏ lớp từ loại quá thưa: `ngã: động từ` (6 mục, lỗi định dạng), `danh từ chỉ phương
    # hướng` (1), `tiền tố`/`phụ tố` (1 mỗi loại). Chúng phủ 7/71.844 lượt = 0,01% nhưng vẫn
    # là NƠ-RON CẮM SẴN không nên tồn tại — nhãn phải là phạm trù, không phải hiện vật.
    _cnt: Counter[str] = Counter()
    for _w, _ps in word_pos.items():
        _cnt.update(_ps)
    _drop = {p_ for p_, c_ in _cnt.items() if c_ < MIN_POS_WORDS}
    if _drop:
        for _w in list(word_pos):
            word_pos[_w] -= _drop
            if not word_pos[_w]:
                del word_pos[_w]
        logger.log(f"bỏ {len(_drop)} lớp thưa (<{MIN_POS_WORDS} từ): {sorted(_drop)}")
    logger.log(f"từ điển: {len(word_pos):,} từ · "
               f"{len({p for s in word_pos.values() for p in s})} từ loại "
               f"({time.time()-t0:.0f}s)")

    # --- hạ nhãn xuống ÂM TIẾT/token -----------------------------------------------------
    # Mỗi từ đóng góp từ loại của nó cho MỌI token bề mặt của nó. Đây chính là chỗ nhoè, và
    # là thứ cổng (2) đang đo. Không làm mượt, không ưu tiên âm tiết đầu — bất kỳ ưu tiên nào
    # cũng là một lựa chọn thiết kế lén đưa tri thức vào, và sẽ che mất mức nhoè thật.
    tok_pos: dict[int, Counter[str]] = defaultdict(Counter)
    for w, poss in word_pos.items():
        try:
            ids = [t for t in v.encode_text(w) if 0 <= t < n and v.is_surface(t)
                   and not is_structural_token(v, t)]
        except Exception:
            continue
        for t in set(ids):
            for p in poss:
                tok_pos[t][p] += 1
    logger.log(f"token có nhãn: {len(tok_pos):,}/{n:,} ({time.time()-t0:.0f}s)")

    # --- đo theo LƯỢT xuất hiện trong cha mini --------------------------------------------
    purity: list[float] = []
    bits: list[float] = []
    cls_occ: Counter[str] = Counter()  # khối lượng PHÂN SỐ, xem `account`
    nlab: list[int] = []
    n_occ = 0
    cov_occ = 0
    tok_pos_occ: dict[int, Counter[str]] = defaultdict(Counter)
    tok_reg_occ: dict[int, Counter[str]] = defaultdict(Counter)
    n_by_src: Counter[str] = Counter()

    def account(cnt: Counter[str] | None, k: int = 1) -> None:
        """Gộp `k` lượt mang cùng bộ nhãn `cnt` vào thống kê.

        ⚠️ Khối lượng lớp chia theo **phân số**, KHÔNG lấy `max`. Ở mức TỪ mọi từ loại của
        một từ đều đồng hạng (đếm = 1), nên `max` phá vỡ theo thứ tự lặp của `set` — thứ tự
        đó phụ thuộc `PYTHONHASHSEED`, **ngẫu nhiên mỗi tiến trình**. Bản đầu dính đúng lỗi
        này: hai lần chạy cùng lệnh cho `danh từ 30,2%` rồi `62,9%`, và cổng (3) lật theo,
        trong khi cổng (1)(2) đứng yên nên trông vẫn như chạy đúng.
        """
        nonlocal cov_occ
        if not cnt:
            return
        cov_occ += k
        tot = sum(cnt.values())
        purity.extend([max(cnt.values()) / tot] * k)
        bits.extend([-sum((x / tot) * math.log2(x / tot) for x in cnt.values())] * k)
        nlab.extend([len(cnt)] * k)
        for p_, c_ in cnt.items():
            cls_occ[p_] += k * c_ / tot

    if args.level == "type":
        occ: Counter[int] = Counter()
        for rec in ps.parents:
            for t in rec.emit_ids:
                t = int(t)
                if 0 <= t < n and v.is_surface(t) and not is_structural_token(v, t):
                    occ[t] += 1
        n_occ = sum(occ.values())
        for t, c in occ.items():
            account(tok_pos.get(t), c)
    else:
        # --- mức TỪ: tách từ bằng WS-CRF rồi tra TỪ, không tra âm tiết -------------------
        # `động vật` tra nguyên từ ⇒ *danh từ* dứt khoát; tra `động` riêng ⇒ nhoè sang
        # `động cơ`, `hoạt động`, `bị động`. Đây là lối đi đã ghi TRƯỚC trong dự đoán, không
        # phải cổng dời sau khi thấy số.
        from nn_aivie.hca.readout import decode_surface  # noqa: PLC0415
        from nn_aivie.ws_crf.ws_model import WSCRFModel  # noqa: PLC0415
        wsm = WSCRFModel.load(_ROOT / args.ws_model)
        n_no_align = 0

        # --- LƯỢT 1: tách từ, thu (token, từ, có phải đầu đơn vị) --------------------------
        obs: list[tuple[int, str, bool]] = []
        for pi, rec in enumerate(ps.parents):
            if pi and pi % 1000 == 0:
                logger.log(f"  tách từ {pi:,}/{len(ps.parents):,} ({time.time()-t0:.0f}s)")
            ids = [int(t) for t in rec.emit_ids]
            whole = decode_surface(state, ids)
            syl = whole.split()
            if not syl:
                continue
            tags = wsm.tag(syl)
            if len(tags) != len(syl):
                continue
            word_of: list[str] = []
            cur: list[str] = []
            start: list[int] = []
            for s_, tg in zip(syl, tags):
                if tg == "I-W" and cur:
                    cur.append(s_)
                else:
                    cur = [s_]
                    start.append(len(word_of))
                word_of.append("")
                for j in range(start[-1], len(word_of)):
                    word_of[j] = " ".join(cur)
            first_syl = {st for st in start[:1]}
            for k, t in enumerate(ids):
                if not (0 <= t < n) or not v.is_surface(t) or is_structural_token(v, t):
                    continue
                n_occ += 1
                pre = decode_surface(state, ids[: k + 1])
                if not whole.startswith(pre) or whole[len(pre):][:1].isalnum():
                    n_no_align += 1
                    continue
                si = len(pre.split()) - 1
                if not (0 <= si < len(word_of)):
                    n_no_align += 1
                    continue
                obs.append((t, word_of[si], si in first_syl))

        # --- LƯỢT 2: quy tắc LỚP KÝ TỰ cho từ không có trong từ điển -----------------------
        # ⚠️ Viết hoa ĐẦU CÂU là bẫy: mọi từ đầu câu đều hoa. Nên tên riêng được quyết bằng
        # **thống kê viết hoa GIỮA câu** của chính dạng từ đó — vẫn thuần đếm, không danh
        # sách tên tiếng Việt nào. Chữ số thì hiển nhiên theo lớp ký tự.
        cap_mid: Counter[str] = Counter()
        tot_mid: Counter[str] = Counter()
        for _t, w_raw, is_first in obs:
            wl = w_raw.strip(_PUNCT).lower()
            if not wl or is_first:
                continue
            tot_mid[wl] += 1
            if w_raw.strip(_PUNCT)[:1].isupper():
                cap_mid[wl] += 1
        propn = {w_ for w_, c_ in tot_mid.items() if c_ >= 2 and cap_mid[w_] / c_ >= 0.5}
        logger.log(f"tên riêng suy từ viết hoa giữa câu: {len(propn):,} dạng từ")

        def _rule(w_raw: str) -> set[str] | None:
            wl = w_raw.strip(_PUNCT).lower()
            if not wl:
                return None
            if all(ch.isdigit() or ch in ".,/-" for ch in wl) and any(c.isdigit() for c in wl):
                return {"số"}
            if wl in propn:
                return {"tên riêng"}
            return None

        for t, w_raw, _is_first in obs:
            w_ = w_raw.strip(_PUNCT).lower()
            poss = word_pos.get(w_)          # TỪ ĐIỂN THẮNG — quy tắc chỉ lấp chỗ trống,
            src = "dict"                      # nên mọi thay đổi đo được là do PHỦ tăng
            if not poss:
                poss = _rule(w_raw)
                src = "rule"
            account(Counter({p_: 1 for p_ in poss}) if poss else None)
            if poss:
                for p_ in poss:
                    tok_pos_occ[t][p_] += 1.0 / len(poss)
                n_by_src[src] += 1
            regs = word_reg.get(w_)
            if regs:
                for r_ in regs:
                    tok_reg_occ[t][r_] += 1.0 / len(regs)
        logger.log(f"nguồn nhãn: từ điển {n_by_src['dict']:,} · quy tắc {n_by_src['rule']:,}")
        logger.log(f"lượt không căn được âm tiết: {n_no_align:,}")

    cover = cov_occ / max(n_occ, 1)
    med_pur = statistics.median(purity) if purity else 0.0
    mean_pur = statistics.mean(purity) if purity else 0.0
    med_bits = statistics.median(bits) if bits else 0.0
    top_cls, top_cls_n = (cls_occ.most_common(1)[0] if cls_occ else ("—", 0))
    top_share = top_cls_n / max(cov_occ, 1)

    print(f"\ntừ điển {len(word_pos):,} từ · token có nhãn {len(tok_pos):,}/{n:,}")
    print(f"lượt token nội dung trong cha: {n_occ:,}")
    print(f"\n{'phủ (theo lượt)':<32}{cover:>10.1%}")
    print(f"{'phần từ loại TRỘI — trung vị':<32}{med_pur:>10.1%}")
    print(f"{'phần từ loại TRỘI — trung bình':<32}{mean_pur:>10.1%}")
    print(f"{'entropy trung vị (bit)':<32}{med_bits:>10.2f}")
    print(f"{'số từ loại/token — trung vị':<32}"
          f"{statistics.median(nlab) if nlab else 0:>10.1f}")
    print(f"\nlớp trội theo lượt:")
    for p, c in cls_occ.most_common(8):
        print(f"  {p:<22}{c:>10,.0f}{c/max(cov_occ,1):>9.1%}")

    g1 = cover >= GATE_COVER_MIN
    g2 = med_pur >= GATE_PURITY_MIN
    g3 = top_share < GATE_TOP_CLASS_MAX
    ok = g1 and g2 and g3
    print(f"\nlớp lớn nhất: {top_cls} = {top_share:.1%}")
    print(f"CỔNG G-pos: (1) phủ {'✅' if g1 else '❌'} · (2) sạch {'✅' if g2 else '❌'} · "
          f"(3) không suy biến {'✅' if g3 else '❌'} ⇒ {'ĐẠT' if ok else 'TRƯỢT'}")

    res = {
        "checkpoint": args.checkpoint, "dict": args.dict, "muc": args.level,
        "n_word": len(word_pos), "n_pos_class": len({p for s in word_pos.values() for p in s}),
        "n_token_labeled": len(tok_pos), "n_vocab": n, "n_occ": n_occ,
        "phu": cover, "sach_trung_vi": med_pur, "sach_trung_binh": mean_pur,
        "entropy_bit": med_bits,
        "so_tu_loai_trung_vi": statistics.median(nlab) if nlab else 0,
        "lop_troi": {k: round(x, 1) for k, x in cls_occ.most_common(12)}, "lop_lon_nhat": top_cls,
        "phan_lop_lon_nhat": top_share,
        "gate": {"cover_min": GATE_COVER_MIN, "purity_min": GATE_PURITY_MIN,
                 "top_class_max": GATE_TOP_CLASS_MAX},
        "gate_pass": bool(ok), "seconds": round(time.time() - t0, 1),
    }
    op = _ROOT / args.out
    op.parent.mkdir(parents=True, exist_ok=True)
    op.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    logger.log(f"→ {args.out} ({time.time()-t0:.0f}s)")

    if args.emit_map:
        mp = _ROOT / args.emit_map
        mp.parent.mkdir(parents=True, exist_ok=True)
        mp.write_text(json.dumps({
            "muc": args.level,
            "tu_loai": {str(t): dict(c) for t, c in tok_pos_occ.items()},
            "phong_cach": {str(t): dict(c) for t, c in tok_reg_occ.items()},
        }, ensure_ascii=False), encoding="utf-8")
        logger.log(f"→ {args.emit_map} · từ loại {len(tok_pos_occ):,} token "
                   f"· phong cách {len(tok_reg_occ):,} token")

    if not args.write:
        return
    if not ok:
        logger.log("⛔ cổng TRƯỢT ⇒ KHÔNG ghi cạnh. Ghi một lớp nhãn nhoè vào checkpoint chỉ "
                   "làm mọi phép đo sau bị nhiễm mà không ai thấy.")
        return

    from nn_aivie.core.checkpoint_io import save_checkpoint  # noqa: PLC0415
    src = tok_pos_occ if args.level == "word" else tok_pos
    n_edge = 0
    for t, cnt in src.items():
        tot = sum(cnt.values())
        for p, c in cnt.items():
            state.adjacency.add(t, v.pos_id(p), c / tot, EdgeKind.LEXATTR)
            n_edge += 1
    logger.log(f"ghi {n_edge:,} cạnh token→nhãn từ {len(src):,} token "
               f"· {len(cls_occ)} nơ-ron nhãn (mức {args.level.upper()})")
    save_checkpoint(state, str(_ROOT / args.out_checkpoint))
    logger.log(f"→ {args.out_checkpoint} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
