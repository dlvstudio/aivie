#!/usr/bin/env python
"""P68 (TK_045 §13.34) — đầu ra sinh có **thiên về chữ HIẾM** hơn văn thật không?

**Vì sao.** §13.33 giám khảo mù thấy hai kiểu hỏng chứa **rác cực hiếm**: chỗ trống mẫu thư
`[chủ đề]` (**8 lần** trong toàn corpus) và mẩu `km 1661+100 QL1` (**1 lần** — hapax). P67 cho
thấy rò rỉ khuôn mẫu **không** phải vấn đề khối lượng dữ liệu: 0.80% đơn vị dính dấu, và phần
lớn số đó là bắt nhầm (ngoặc biên tập, danh sách lịch sử thật, `{` `}` trong bài lập trình).

⇒ Giả thuyết: đây là **khuếch đại chữ hiếm** ở khâu CHỌN, khớp với §13.31 (65.4% lựa argmax
trên bể đủ dựa vào token gặp ≤5 lần, p25 = hapax; văn thật chỉ 45.6%).

**Nhưng** một chuỗi hiếm nằm *trong* mệnh đề được chọn **chưa** chứng minh nó *gây ra* việc
chọn — mệnh đề được chọn cả cụm, rác có thể chỉ đi nhờ. Nên đo bằng so sánh phân bố:

    đầu ra sinh (v12c, cả hai cấu hình)   vs   câu THẬT lấy ngẫu nhiên từ chính corpus

Cả hai đều **ghép từ corpus**, nên nếu khâu chọn không thiên lệch thì hai phân bố phải trùng.
Lệch = bằng chứng thiên lệch, và **không** cần nạp checkpoint v12 (máy dev không nạp nổi).

Hạn: đếm theo **chữ tách trắng đã hạ chữ hoa**, không phải token BPE ⇒ so được **tương đối**
giữa hai phía, không so được với con số ≤5 của §13.31 (thước khác).

Chạy:
  PYTHONPATH=src python scripts/probe_rare_amplify.py
"""
from __future__ import annotations

import argparse
import json
import random
import re
import statistics
import sys
import unicodedata
from collections import Counter
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

_WORD = re.compile(r"[^\W_]+", re.UNICODE)
_SENT_SPLIT = re.compile(r"(?<=[.!?…])\s+|\n+")
_PAIR = re.compile(r"^\s{3}([XY]):\s*(.+)$")


def words(s: str) -> list[str]:
    return _WORD.findall(unicodedata.normalize("NFC", s).lower())


def stats(freqs: list[int]) -> dict:
    if not freqs:
        return {}
    freqs = sorted(freqs)
    n = len(freqs)
    return {
        "n_chu": n,
        "p25": freqs[n // 4],
        "trung_vi": statistics.median(freqs),
        "ty_le_le_5": sum(1 for f in freqs if f <= 5) / n,
        "ty_le_le_1": sum(1 for f in freqs if f <= 1) / n,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="config/train_v12_sent.yaml")
    ap.add_argument("--pairs", default="out/anchor_pairs_v12c.txt")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="out/probe_rare_amplify.json")
    args = ap.parse_args()

    import yaml

    man = yaml.safe_load((_ROOT / args.manifest).read_text(encoding="utf-8"))

    # --- 1. bảng tần suất chữ + kho câu thật ------------------------------------------
    df: Counter[str] = Counter()
    corpus_sents: list[str] = []
    rng = random.Random(args.seed)
    for d in man["datasets"]:
        p = _ROOT / d["path"]
        if not p.exists():
            continue
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    text = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(text, str):
                    continue
                df.update(words(text))
                for s in _SENT_SPLIT.split(text):
                    s = s.strip()
                    if len(words(s)) >= 4:
                        # reservoir thưa để không giữ cả corpus trong RAM (máy 8GB)
                        if rng.random() < 0.004:
                            corpus_sents.append(s)
    print(f"từ vựng {len(df):,} chữ · kho câu thật lấy mẫu {len(corpus_sents):,}")

    # --- 2. đầu ra sinh ---------------------------------------------------------------
    gen_lines: list[str] = []
    for ln in (_ROOT / args.pairs).read_text(encoding="utf-8").splitlines():
        m = _PAIR.match(ln)
        if m:
            gen_lines.append(m.group(2).strip())
    print(f"đầu ra sinh {len(gen_lines):,} dòng (cả hai cấu hình)")

    def collect(texts: list[str]) -> list[int]:
        out = []
        for t in texts:
            out.extend(df.get(w, 0) for w in words(t))
        return out

    gen = stats(collect(gen_lines))
    # so trên cùng cỡ mẫu để p25/trung vị không bị lệch do số lượng
    real_pool = corpus_sents[:]
    rng.shuffle(real_pool)
    real = stats(collect(real_pool[: max(1, len(gen_lines))]))

    print(f"\n{'':<22}{'SINH':>12}{'THẬT':>12}")
    for k in ("n_chu", "p25", "trung_vi", "ty_le_le_5", "ty_le_le_1"):
        a, b = gen.get(k, 0), real.get(k, 0)
        fmt = (lambda v: f"{v:.2%}") if k.startswith("ty_le") else (lambda v: f"{v:,.0f}")
        print(f"{k:<22}{fmt(a):>12}{fmt(b):>12}")

    verdict = ("SINH hiếm hơn THẬT" if gen.get("ty_le_le_5", 0) > real.get("ty_le_le_5", 0)
               else "KHÔNG thấy thiên lệch về chữ hiếm")
    print(f"\n⇒ {verdict}")

    out = {"gen": gen, "real": real, "verdict": verdict,
           "n_gen_dong": len(gen_lines), "n_real_mau": len(corpus_sents),
           "caveat": "đếm theo chữ tách trắng, KHÔNG phải token BPE — chỉ so tương đối"}
    op = _ROOT / args.out
    op.parent.mkdir(parents=True, exist_ok=True)
    op.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"→ {args.out}")


if __name__ == "__main__":
    main()
