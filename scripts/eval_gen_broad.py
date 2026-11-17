#!/usr/bin/env python3
"""ĐO RỘNG chất lượng sinh — TK_045 §12.22. Chạy được cả mini lẫn v12.

    PYTHONPATH=src python -u scripts/eval_gen_broad.py \
        --checkpoint checkpoints/network_v12_sent.pkl \
        --log-file out/eval_gen_broad.log

## Vì sao

§12.12–§12.21 ship 6 thay đổi, nhưng **toàn bộ bằng chứng nằm trên ~8 prompt tự chọn** và **đều
chạy `ar_no_agenda=True`**. Chưa ai biết tổng thể v12 tốt lên hay ta đang khớp quá vào 8 prompt.

## Thiết kế — liệt kê LOẠI ĐẦU VÀO trước, rồi mới soạn nhóm

Ba lần liên tiếp (§12.16 thước cắt-theo-dấu-chấm · §12.18 nhãn frame df=0 · §12.20 nhóm mù ca
GHÉP) khâu hỏng là thước chứ không phải code, luôn vì nhóm đo thiếu một loại đầu vào. Nên ở đây
loại đầu vào là thứ được liệt kê TRƯỚC:

  T1 verbatim   nối tiếp câu mạng đã thuộc (sinh từ chính emit của cha — luôn đúng miền)
  T2 paraphrase diễn đạt lại fact có thật (đảo trật tự / đổi từ mở đầu)
  T3 ghép       so sánh / đa thực thể / liệt kê — cần >1 trí nhớ
  T4 qa         khung `Hỏi: … Đáp:`
  T5 số         hỏi năm — đúng thứ §12.12–§12.14 vá
  T6 bịa        tiền đề hoán thực thể — mạng KHÔNG được khẳng định
  T7 dài        prompt có đoạn ngữ cảnh phía trước

× **2 chế độ**: `no_agenda` (một câu) và `handoff` (đa câu). §12.21 cho thấy handoff sinh khác
hẳn, mà mọi đo trước đều chạy no_agenda ⇒ bỏ chế độ này là bỏ nửa sự thật.

## Thước — đều KHÁCH QUAN, không cần LM chấm

  empty       không sinh gì thêm (từ chối / dừng ngay).  T6 muốn CAO, T1–T5 muốn ~0
  digit_split chuỗi kiểu `1 0 1 0` — chữ-số rời, đúng lỗi §12.12 đã vá.  muốn 0
  loop4       có 4-gram lặp ≥3 lần — thoái hoá.  muốn 0
  copy        LCS(cont, emit khớp nhất)/|cont| — mức verbatim
  n_facts     số cha KHÁC NHAU các câu output chép từ — bề rộng ghép (T3 muốn >1)
  stop_clean  dừng ở biên câu / EOS chứ không cụt vì hết `max_len`
  n_tok       độ dài phần sinh

Chỉ mục corpus dựng từ **emit của cha trong checkpoint**, KHÔNG parse lại file — đúng bài học
§12.16 (thước cắt theo `.` làm `5.000` vỡ thành `"000 lính…"`).

## Dùng để so BASELINE ↔ HEAD

Chạy 1 lần ở commit trước §12.12 và 1 lần ở HEAD với cùng `--checkpoint`, rồi so hai bảng.
Không có baseline thì bảng vẫn đọc được (mỗi thước có hướng mong muốn ghi ở trên).
"""

from __future__ import annotations

import argparse
import gc
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from nn_aivie.core.checkpoint_io import load_checkpoint  # noqa: E402
from nn_aivie.hca.config import HCAConfig  # noqa: E402
from nn_aivie.hca.generate_hebbian import generate  # noqa: E402
from nn_aivie.hca.matrix_core import _path_fold, ensure_parent_bipartite  # noqa: E402
from nn_aivie.hca.metrics_pmi import PmiCrossIndex  # noqa: E402
from nn_aivie.hca.readout import decode_surface  # noqa: E402
from nn_aivie.hca.sent_intra import clear_sent_intra_cache  # noqa: E402

# T2–T6 phụ thuộc MIỀN (sử Việt) — v12/mini đều có. Đổi corpus thì đổi khối này.
PROMPTS: dict[str, list[str]] = {
    "T2 paraphrase": [
        "Nhà Lý dời đô về",
        "Ngô Quyền đánh bại quân Nam Hán năm",
        "Lê Hoàn đánh bại quân Tống trên sông Bạch Đằng năm",
        "Trận Bạch Đằng năm",
        "Kinh đô đầu tiên của nhà nước Đại Cồ Việt là",
    ],
    "T3 ghép": [
        "Hỏi: so sánh nhà Lý và nhà Trần Đáp:",
        "Cả Hoa Lư và Thăng Long đều từng là",
        "Hỏi: Kể tên các trận thắng trên sông Bạch Đằng Đáp:",
        "Ngô Quyền và Lê Hoàn đều đánh bại quân xâm lược trên sông Bạch Đằng, còn Trần Hưng Đạo",
        "Hỏi: Nhà Lý và nhà Trần khác nhau thế nào Đáp:",
        "Hỏi: so sánh Ngô Quyền và Lê Hoàn Đáp:",
        "Cả Đinh Bộ Lĩnh và Lê Hoàn đều",
        "Hỏi: Kể tên các kinh đô của nước ta Đáp:",
        "Hoa Lư, Thăng Long và Tây Đô đều",
        "Hỏi: Trần Hưng Đạo và Ngô Quyền giống nhau ở điểm gì Đáp:",
    ],
    # §12.38: n=4 là QUÁ ÍT — trên v12, `degen` 1.00→0.75 và `n_tok` 1.0→8.8 đạt cổng CHỈ nhờ
    # MỘT ca (Lê Lợi), 3 ca kia vẫn `Đáp: ở/không/Kỳ`. Một ca không được phép kéo cả cột.
    "T4 qa": [
        "Hỏi: Nhà Lý dời đô về đâu? Đáp:",
        "Hỏi: Ai đánh bại quân Nam Hán trên sông Bạch Đằng? Đáp:",
        "Hỏi: Hoa Lư nằm ở tỉnh nào? Đáp:",
        "Hỏi: Lê Lợi khởi nghĩa ở đâu? Đáp:",
        "Hỏi: Ngô Quyền đánh bại quân Nam Hán năm nào? Đáp:",
        "Hỏi: Kinh đô đầu tiên của nước Đại Cồ Việt là gì? Đáp:",
        "Hỏi: Ai chỉ huy trận Bạch Đằng năm 1288? Đáp:",
        "Hỏi: Nhà Lý dời đô năm nào? Đáp:",
        "Hỏi: Đinh Bộ Lĩnh đặt tên nước là gì? Đáp:",
        "Hỏi: Lê Hoàn đánh bại quân nào trên sông Bạch Đằng? Đáp:",
    ],
    "T5 số": [
        "Nhà Lý dời đô về Thăng Long năm",
        "Chiến thắng Bạch Đằng năm",
        "Đinh Bộ Lĩnh chọn Hoa Lư làm kinh đô vào năm",
        "Trần Hưng Đạo tiêu diệt quân Nguyên Mông trên sông Bạch Đằng năm",
    ],
    "T6 bịa": [
        "Nhà Trần dời đô về Thăng Long năm",
        "Ngô Quyền đánh bại quân Tống năm",
        "Lê Lợi dời đô về Thăng Long năm",
        "Nhà Lý đánh bại quân Nguyên Mông năm",
        "Đinh Bộ Lĩnh đánh bại quân Nam Hán trên sông Bạch Đằng năm",
    ],
}
MODES = {
    "no_agenda": dict(ar_no_agenda=True, ar_sent_agenda_handoff=False),
    "handoff": dict(ar_no_agenda=True, ar_sent_agenda_handoff=True),
}


def _lcs(a: list[int], b: list[int]) -> int:
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b):
            cur.append(prev[j] + 1 if x == y else max(cur[j], prev[j + 1]))
        prev = cur
    return prev[-1]


class EmitIndex:
    """Chỉ mục ngược token→cha trên EMIT (bộ nhớ thật của mạng, không parse lại file)."""

    def __init__(self, emits, max_len: int = 64):
        self.emits = [list(map(int, e))[:max_len] for e in emits]
        self.inv: dict[int, list[int]] = defaultdict(list)
        for pi, e in enumerate(self.emits):
            for t in set(e):
                self.inv[t].append(pi)

    def best(self, seq: list[int], topk: int = 40) -> tuple[float, int]:
        """(LCS/|seq| lớn nhất, chỉ số cha) — chỉ chấm các cha chia sẻ nhiều token nhất."""
        if not seq:
            return 0.0, -1
        cnt: dict[int, int] = defaultdict(int)
        for t in set(seq):
            for pi in self.inv.get(t, ()):
                cnt[pi] += 1
        if not cnt:
            return 0.0, -1
        cands = sorted(cnt, key=lambda p: -cnt[p])[:topk]
        best, bpi = 0.0, -1
        for pi in cands:
            f = _lcs(seq, self.emits[pi]) / len(seq)
            if f > best:
                best, bpi = f, pi
        return best, bpi


def _is_digit(vocab, t: int) -> bool:
    try:
        s = vocab.token_of(int(t)).lstrip("ĠĊ")
    except Exception:
        return False
    return bool(s) and s.isdigit()


def _metrics(state, idx: EmitIndex, prompt_ids: list[int], out_ids: list[int],
             max_len: int, pmi: PmiCrossIndex | None = None) -> dict:
    v = state.vocab
    cont = [int(t) for t in out_ids[len(prompt_ids):]]
    m = {"n_tok": len(cont), "empty": int(not cont)}
    if not cont:
        return {**m, "digit_split": 0, "loop4": 0, "copy": float("nan"), "n_facts": 0,
                "stop_clean": 1, "degenerate": 1, "off_topic": float("nan"),
                "pmi_cross": 0.0}
    # `1 0 1 0`: hai chữ-số liền nhau mà cái sau CÓ `Ġ` (tách rời) — chữ ký lỗi §12.12
    # §12.107b: bản CHỈ xét cặp KỀ NHAU bị lách. Kênh phiếu §12.106 đánh vần số qua dấu câu
    # (`1 . 2 . 8` · `. 7 5 7 6`) ⇒ cột về 0.00 trong khi đọc tay thấy lỗi y nguyên. Nới sang
    # "hai chữ-số cách nhau ≤1 token, token chen giữa KHÔNG phải chữ-cái/chữ-số" để thước thấy
    # đúng hình dạng mới. Không nới thì mọi kết luận về số đều là thước bị lách, không phải vá.
    def _joins(k: int) -> bool:
        s = v.token_of(int(cont[k])).lstrip("Ġ")
        return bool(s) and any(ch.isalnum() for ch in s)

    split = any(
        _is_digit(v, cont[i]) and _is_digit(v, cont[i + g])
        and (g == 1 or not _joins(i + 1))
        and (g == 1 and v.token_of(int(cont[i + 1])).startswith("Ġ") or g == 2)
        for g in (1, 2)
        for i in range(len(cont) - g)
    )
    grams: dict[tuple, int] = defaultdict(int)
    for i in range(len(cont) - 3):
        grams[tuple(cont[i:i + 4])] += 1
    m["digit_split"] = int(split)
    m["loop4"] = int(any(c >= 3 for c in grams.values()))
    # §12.27: chuỗi RẤT NGẮN làm `copy` nói dối — 1 token thì LCS luôn khớp trọn một emit nào
    # đó ⇒ copy=1.00. Bảng v12 vì thế đọc như thể T4 QA hoàn hảo, trong khi đáp là 1 token rác
    # (`Đáp: ở`, `Đáp: không`). Dưới 4 token thì KHÔNG chấm copy.
    best, _pi = idx.best(cont) if len(cont) >= 4 else (float("nan"), -1)
    m["copy"] = round(best, 3)
    # cụt đến mức vô nghĩa — tách riêng vì `stop_clean` coi "EOS ngay" là dừng SẠCH
    m["degenerate"] = int(len(cont) <= 2)
    # §12.73 — LẠC ĐỀ. Mọi cột cũ đều mù với việc output nhảy sang tài liệu khác: v12 hm=3 chấm
    # "Long Hạo Thiên … Gạo hạt sen rửa sạch" là degen 0 · stop 1.00 · n_facts 3.45 ("bề rộng
    # cao"). Không có cột này thì KHÔNG quyết được bất kỳ thay đổi nào về độ dài.
    # Đo: mỗi CÂU trong phần sinh có chia fold nội dung nào với PROMPT không.
    ps_ = state.parents
    def _cf(ids):
        out_ = set()
        for t in ids:
            t = int(t)
            try:
                g = float(ps_._gidf(t))  # noqa: SLF001
            except Exception:
                continue
            f = _path_fold(v, t)
            if f and g >= 4.0:
                out_.add(f)
        return out_

    pf = _cf(prompt_ids)
    segs, cur = [], []
    for t in cont:
        cur.append(t)
        srf = v.token_of(int(t)).lstrip("ĠĊ")
        if srf in (".", "!", "?", "…") or int(t) == getattr(v, "eos_id", -1):
            if cur:
                segs.append(cur)
            cur = []
    if cur:
        segs.append(cur)
    later = [sg for sg in segs[1:] if len(sg) >= 4]
    if pf and later:
        m["off_topic"] = sum(1 for sg in later if not (_cf(sg) & pf)) / len(later)
    else:
        m["off_topic"] = float("nan")
    # n_facts: chia theo biên câu, mỗi câu khớp cha nào (≥0.5) — bề rộng ghép
    from nn_aivie.hca.sent_intra import _is_boundary

    facts, cur = set(), []
    for t in cont:
        cur.append(int(t))
        if _is_boundary(v, int(t)):
            if len(cur) >= 4:
                f, pi = idx.best(cur)
                if f >= 0.5 and pi >= 0:
                    facts.add(pi)
            cur = []
    if len(cur) >= 4:
        f, pi = idx.best(cur)
        if f >= 0.5 and pi >= 0:
            facts.add(pi)
    m["n_facts"] = len(facts)
    # `max_len` là số token MỚI, `out_ids` gồm cả prompt ⇒ phải so PHẦN SINH. Bản trước so
    # `len(out_ids) < max_len` nên với prompt dài (T7) nó luôn sai và chỉ còn dựa vào biên câu.
    m["stop_clean"] = int(len(cont) < max_len or _is_boundary(v, int(cont[-1])))
    # §12.117/P32 — PMI-cross prompt↔sinh (cạnh `lạc` fold). Cao = bám đề.
    m["pmi_cross"] = float(pmi.score(list(prompt_ids), cont)) if pmi is not None else 0.0
    return m


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/network_v12_sent.pkl")
    ap.add_argument("--log-file", default="out/eval_gen_broad.log")
    ap.add_argument("--max-len", type=int, default=48)
    ap.add_argument("--n-verbatim", type=int, default=20, help="số prompt T1 lấy từ emit cha")
    ap.add_argument("--json-out", default="")
    ap.add_argument("--samples-per-cell", type=int, default=4,
                    help="số mẫu in ra cho MỖI ô (chế độ × nhóm)")
    ap.add_argument("--extra", default="", help='JSON knob override, vd \'{"ar_matrix_and_form_min":0.9}\'')
    args = ap.parse_args()
    extra = json.loads(args.extra) if args.extra else {}

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
    v = state.vocab
    bi = ensure_parent_bipartite(state)
    idx = EmitIndex(bi.emits)
    log(f"[{time.time()-t0:.0f}s] ✓ {len(state.parents):,} cha · max_len={args.max_len}"
        f" · extra={extra or None}")
    log(f"[{time.time()-t0:.0f}s] dựng PMI-cross index …")
    pmi_idx = PmiCrossIndex(state)
    log(f"[{time.time()-t0:.0f}s] ✓ PMI-cross ({len(pmi_idx._c):,} cặp)")  # noqa: SLF001

    # T1 + T7 sinh TỪ CHECKPOINT ⇒ luôn đúng miền, không phụ thuộc corpus nào
    order = list(range(len(bi.emits)))
    random.Random(42).shuffle(order)
    t1, t7 = [], []
    for pi in order:
        e = [int(x) for x in bi.emits[pi]]
        if 10 <= len(e) <= 30 and len(t1) < args.n_verbatim:
            t1.append(e[: max(5, int(len(e) * 0.6))])
        elif 24 <= len(e) <= 60 and len(t7) < max(4, args.n_verbatim // 4):
            t7.append(e[: max(12, int(len(e) * 0.75))])
        if len(t1) >= args.n_verbatim and len(t7) >= max(4, args.n_verbatim // 4):
            break

    groups: dict[str, list[list[int]]] = {"T1 verbatim": t1}
    for name, ps_ in PROMPTS.items():
        groups[name] = [list(v.encode_prompt(p)) for p in ps_]
    groups["T7 dài"] = t7

    rows: dict[tuple[str, str], list[dict]] = {}
    # §12.38: bản trước lấy 60 mẫu ĐẦU TIÊN ⇒ no_agenda chạy trước nên chiếm hết (đo v12:
    # 47 no_agenda / 7 handoff) ⇒ không đọc tay được chế độ handoff, đúng thứ cần adjudicate.
    # Nay lấy tối đa `--samples-per-cell` cho MỖI ô (chế độ × nhóm).
    samples: dict[tuple[str, str], list[str]] = defaultdict(list)
    for mode, over in MODES.items():
        # §12.36: xoá MỘT lần mỗi chế độ, KHÔNG phải mỗi prompt. `ensure_sent_intra` đã tự
        # kiểm lam/d_max/content_only nên cache tự vô hiệu khi config đổi; xoá mỗi prompt chỉ
        # ép dựng lại `build_sent_intra_views` — trên v12 là ~150s MỖI PROMPT. Đo mini: 7 prompt
        # 95.5s → 2.1s (45×), output Y HỆT. Đây là lý do wall-clock v12 không nhúc nhích dù
        # profile cho thấy sinh đã nhanh 40× (§12.32–§12.34).
        clear_sent_intra_cache()
        for gname, plist in groups.items():
            acc = []
            for pids in plist:
                # §12.106: gộp bằng dict để `--extra` GHI ĐÈ được mọi knob. Bản cũ truyền
                # `hybrid_temperature=0.0` như đối số cứng ⇒ `--extra` đụng "multiple values"
                # ⇒ harness KHÔNG đo nổi sampling. Đúng lớp giới hạn dụng cụ đã cắn nhiều lần.
                _kw = dict(ar_matrix_core=True, ar_style_payoff=False,
                           hybrid_temperature=0.0, ar_matrix_stack_weight=0.0)
                _kw.update(over)
                _kw.update(extra)
                cfg = HCAConfig(**_kw)
                text_in = decode_surface(state, pids)
                out = generate(state, text_in, max_len=args.max_len, config=cfg, seed=42)
                acc.append(_metrics(state, idx, list(pids), list(out), args.max_len, pmi_idx))
                if len(samples[(mode, gname)]) < args.samples_per_cell:
                    samples[(mode, gname)].append(decode_surface(state, out)[:260])
            rows[(mode, gname)] = acc
            log(f"[{time.time()-t0:.0f}s]   xong {mode}·{gname} ({len(acc)})")

    def agg(a: list[dict], k: str) -> float:
        vals = [x[k] for x in a if x[k] == x[k]]  # bỏ NaN (copy không chấm khi cont quá ngắn)
        return sum(vals) / len(vals) if vals else float("nan")

    log("\n" + "=" * 112)
    log(f"{'chế độ':<10}{'nhóm':<16}{'n':>3}{'empty':>8}{'degen':>7}{'lạc':>7}{'pmi':>7}{'digit⎵':>8}{'loop4':>7}"
        f"{'copy':>7}{'n_facts':>9}{'stop':>7}{'n_tok':>7}")
    log("=" * 112)
    for mode in MODES:
        for gname in groups:
            a = rows[(mode, gname)]
            log(f"{mode:<10}{gname:<16}{len(a):>3}{agg(a,'empty'):>8.2f}"
                f"{agg(a,'degenerate'):>7.2f}{agg(a,'off_topic'):>7.2f}{agg(a,'pmi_cross'):>7.2f}"
                f"{agg(a,'digit_split'):>8.2f}"
                f"{agg(a,'loop4'):>7.2f}{agg(a,'copy'):>7.2f}"
                f"{agg(a,'n_facts'):>9.2f}{agg(a,'stop_clean'):>7.2f}{agg(a,'n_tok'):>7.1f}")
        log("-" * 112)
    log("Hướng mong muốn: empty↓ (trừ T6 bịa ↑) · degen↓ · **lạc↓** · **pmi↑** · digit⎵↓ · loop4↓ · stop↑")
    log("`lạc` = tỉ lệ CÂU (từ câu 2 trở đi) KHÔNG chia fold nội dung nào với prompt — đo TRÔI ĐỀ.")
    log("`pmi` = PMI-cross prompt↔sinh (§12.117/P32) — cao = bám đề; cạnh `lạc`, không thay cổng cũ.")
    log("  Cột này thêm ở §12.73: mọi cột cũ đều mù với việc output nhảy sang tài liệu khác.")
    log("`copy` KHÔNG chấm khi phần sinh <4 token (NaN) — 1 token luôn khớp trọn một emit.")

    # §12.38/§12.40: cổng T4 đếm CA đọc được, không lấy trung bình n_tok (1 ca dài kéo cột).
    log("\n── đếm ca (cổng §12.38) ──")
    for mode in MODES:
        for gname in ("T3 ghép", "T4 qa"):
            if gname not in groups:
                continue
            a = rows[(mode, gname)]
            n = len(a)
            readable = sum(1 for x in a if not x["degenerate"] and x["n_tok"] >= 4)
            multi = sum(1 for x in a if x["n_facts"] > 1)
            log(f"  {mode:<10}{gname:<12} readable(n_tok≥4&!degen)={readable}/{n}"
                f"  · n_facts>1={multi}/{n}")

    log("\n── mẫu đọc tay (mỗi ô tối đa "
        f"{args.samples_per_cell}) ──")
    for mode in MODES:
        for gname in groups:
            for txt in samples.get((mode, gname), ()):
                log(f"  [{mode}·{gname}] {txt!r}")

    if args.json_out:
        counts: dict[str, dict] = {}
        for mode in MODES:
            for gname in groups:
                a = rows[(mode, gname)]
                counts[f"{mode}|{gname}"] = {
                    "n": len(a),
                    "readable": sum(1 for x in a if not x["degenerate"] and x["n_tok"] >= 4),
                    "n_facts_gt1": sum(1 for x in a if x["n_facts"] > 1),
                }
        with (_ROOT / args.json_out).open("w", encoding="utf-8") as f:
            json.dump({
                "metrics": {f"{m}|{g}": rows[(m, g)] for m in MODES for g in groups},
                "samples": {f"{m}|{g}": samples.get((m, g), []) for m in MODES for g in groups},
                "counts": counts,
            }, f, ensure_ascii=False, indent=1)
        log(f"\n→ {args.json_out}")
    log(f"\n[{time.time()-t0:.0f}s] → {args.log_file}")
    logf.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
