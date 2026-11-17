#!/usr/bin/env python3
"""
SINH TỰ DO — mạng đồ thị KHÔNG bám agenda (TK_042, probe 2026-07-20).

    ⚠️ CHẠY M4 — checkpoint đồ thị 2GB → ~20GB RAM; máy dev 8GB KHÔNG load nổi (TK_040).
    PYTHONPATH=src python -u scripts/eval_free_gen.py --checkpoint checkpoints/network_v11.pkl

## Câu hỏi

"Chất lượng sinh của đồ thị nhìn tốt — vì nó bám agenda". Nhưng agenda = fact TRUY HỒI rồi
dội gần-nguyên-văn (§7.13: verbatim ~100%, 5-gram mới 0% = KHẢM corpus). Probe này TẮT dần
agenda để tách: chất lượng đó là **CHÉP** (agenda) hay **SINH thật** (dynamics W_seq/spreading)?

Cô lập sạch: cùng prompt khung `Hỏi:` (agenda LUÔN được lập), quét knob agenda ĐẦY → TẮT:
  * `ar_index_backbone`  = xương sống chép trật-tự-con-trong-cha (§7.13 thủ phạm verbatim)
  * `ar_agenda_boost`    = ưu tiên token agenda còn thiếu
  * `ar_spread_weight`   = trường spreading từ agenda
TẮT cả ba ⇒ chỉ còn base×micro (co-act × W_seq bigram + varlm) = dynamics THÔ của mạng.
+ `ar_temperature` cho "lạ lạ" (0 argmax → 0.7/1.0 lấy mẫu).

Đo mỗi lần sinh:
  * `copy%` = LCS với corpus (thước CHÉP thật — cao = khảm; reuse eval_discourse_prompts).
  * `rep4%` = tỉ lệ 4-gram nội bộ lặp (cao = loop thoái hoá).
  * toàn văn — ĐỌC TAY (copy/rep MÙ với "trôi chủ đề" & "bịa mạch lạc", §7.17d).

## Dự đoán ghi trước (đối chiếu — chống ảo tưởng)

Nếu agenda-TẮT vẫn mạch lạc + copy% tụt ⇒ mạng SINH thật, agenda chỉ chọn nội dung → tin
"đồ thị sinh tốt". Nếu agenda-TẮT sụp (loop/trôi/bịa) + copy% agenda-ĐẦY cao ⇒ chất lượng =
CHÉP, khớp §7.13/§0.5. Tiền lệ §7.12 (G6 "cai nghiện agenda: núm không mở van nào") nghiêng
về vế sau — nhưng đây đọc TAY trực tiếp, thước khác, nên vẫn đáng chạy.
"""

from __future__ import annotations

import argparse
import gc
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "scripts"))

from eval_discourse_prompts import build_corpus_index, _copy_frac  # noqa: E402 — reuse thước copy
from nn_aivie.core.checkpoint_io import load_checkpoint  # noqa: E402
from nn_aivie.hca.config import HCAConfig  # noqa: E402
from nn_aivie.hca.generate_hebbian import generate  # noqa: E402
from nn_aivie.hca.readout import decode_surface  # noqa: E402
from nn_aivie.data.quality import normalize_ws  # noqa: E402

# Khung Hỏi: → agenda LUÔN được lập ⇒ cô lập được đóng góp của agenda (khác prompt thường có
# thể không match cha). Chọn thực thể có nhiều fact trong corpus (agenda giàu để mất mát rõ).
QA_PROMPTS = [
    "Hỏi: Nhà Lý là gì",
    "Hỏi: Thủ đô của Việt Nam ở đâu",
    "Hỏi: Trần Hưng Đạo là ai",
    "Hỏi: Sông Hồng là gì",
    "Hỏi: Hà Nội là gì",
]
# Prompt THƯỜNG (không khung) — thử đường continuation tự do `_wseq_fallback` (hybrid_temperature).
FREE_PROMPTS = [
    "Việt Nam là",
    "Trong lịch sử,",
    "Nhà Lý được thành lập",
]


def rep4(text: str) -> float:
    """Tỉ lệ 4-gram lặp trong phần sinh — bắt loop thoái hoá không cần đọc tay."""
    toks = normalize_ws(text).split()
    if len(toks) < 5:
        return 0.0
    grams = [tuple(toks[i : i + 4]) for i in range(len(toks) - 3)]
    seen, rep = set(), 0
    for g in grams:
        if g in seen:
            rep += 1
        seen.add(g)
    return rep / len(grams)


def _gen_only(state, prompt: str, config: HCAConfig, *, multi: bool, max_len: int, seed: int) -> str:
    ids = generate(state, prompt, max_len=max_len, config=config, multi=multi, seed=seed)
    out = decode_surface(state, ids)
    return out[len(prompt):].strip() if out.startswith(prompt) else out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--checkpoint", default="checkpoints/network_v11.pkl")
    ap.add_argument("--max-len", type=int, default=48)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--corpus-limit", type=int, default=20000, help="số đơn vị corpus cho thước copy")
    args = ap.parse_args()

    t0 = time.time()

    def log(m):
        print(f"[{time.time()-t0:6.0f}s] {m}", flush=True)

    log(f"load {args.checkpoint} (gc.disable — nếu không có thể 73 phút, memory checkpoint-load-gc) …")
    gc.disable()
    state, _ = load_checkpoint(str(_ROOT / args.checkpoint))
    gc.enable()
    log("✓ loaded · dựng corpus index cho thước copy …")
    index = build_corpus_index(limit_units=args.corpus_limit)

    # (tên, config, prompts, multi) — agenda ĐẦY → TẮT dần; + tự do prompt thường.
    b = dict(ar_index_backbone=False, ar_agenda_boost=0.0, ar_spread_weight=0.0)  # agenda TẮT
    runs = [
        ("agenda ĐẦY (mặc định, khảm)", HCAConfig(), QA_PROMPTS, True),
        ("agenda TẮT · t=0 (argmax)", HCAConfig(**b), QA_PROMPTS, True),
        ("agenda TẮT · t=0.7", HCAConfig(**b, ar_temperature=0.7), QA_PROMPTS, True),
        ("agenda TẮT · t=1.0 (lạ)", HCAConfig(**b, ar_temperature=1.0), QA_PROMPTS, True),
        ("TỰ DO prompt thường · t=1.0", HCAConfig(hybrid_temperature=1.0), FREE_PROMPTS, False),
    ]

    summary = []
    for name, cfg, prompts, multi in runs:
        log(f"═══ {name} ═══")
        copies, reps = [], []
        for p in prompts:
            try:
                g = _gen_only(state, p, cfg, multi=multi, max_len=args.max_len, seed=args.seed)
            except Exception as e:  # cấu hình lạ có thể vấp — ghi, không chết cả run
                log(f"  ✗ {p!r}: {type(e).__name__}: {e}")
                continue
            c, r = _copy_frac(g, index), rep4(g)
            copies.append(c)
            reps.append(r)
            log(f"  {p}\n    copy={c:.2f} rep4={r:.2f} · {g}")
        if copies:
            summary.append((name, sum(copies) / len(copies), sum(reps) / len(reps)))

    log("═══════════ TÓM TẮT (copy thấp=novel · rep4 cao=loop) ═══════════")
    for name, c, r in summary:
        log(f"  copy {c:.2f} · rep4 {r:.2f}   {name}")
    log("→ ĐỌC TAY toàn văn ở trên: agenda-TẮT còn mạch lạc (⇒ sinh thật) hay sụp (⇒ chỉ chép)?")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
