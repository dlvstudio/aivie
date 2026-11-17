#!/usr/bin/env python3
"""Đánh giá dataset: «hài hước» → từ/cụm nào thực sự đặc trưng trong thân Đáp."""
from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path

DAP = re.compile(r"Đáp\s*:\s*", re.I)
COLON = re.compile(r"^(?:Viết|Kể|Hãy viết|Hãy kể)[^:]{0,80}:\s*", re.I)
OUT = Path("out/eval_humor_lexicon.log")


def split(t: str) -> tuple[str, str]:
    m = DAP.search(t)
    if m:
        return t[: m.start()], t[m.end() :]
    m2 = COLON.match(t)
    if m2:
        return t[: m2.end()], t[m2.end() :]
    return t, t


def toks(s: str) -> list[str]:
    s = re.sub(r"[\"'`‘’“”]", " ", s)
    return re.findall(r"[A-Za-zÀ-ỹ0-9]+(?:[’'\-][A-Za-zÀ-ỹ0-9]+)?", s)


def ngram_stats(bodies: list[str], n: int):
    tf, df = Counter(), Counter()
    for b in bodies:
        ws = [w.lower() for w in toks(b)]
        grams = [tuple(ws[i : i + n]) for i in range(len(ws) - n + 1)]
        tf.update(grams)
        df.update(set(grams))
    return tf, df


def main() -> int:
    lines: list[str] = []

    def log(m: str = "") -> None:
        print(m, flush=True)
        lines.append(m)

    humor: list[str] = []
    other: list[str] = []
    for p in (
        Path("data/text_corpus/styled_sft_1k.jsonl"),
        Path("data/text_corpus/styled_sft.jsonl"),
    ):
        if not p.is_file():
            continue
        for line in p.read_text(encoding="utf-8").splitlines():
            t = json.loads(line)["text"]
            lab, bod = split(t)
            if not bod.strip() or bod.strip() == lab.strip():
                continue
            is_h = ("hài" in lab.lower()) or ("hước" in lab.lower())
            (humor if is_h else other).append(bod.strip())

    N = len(humor)
    log(f"humor_bodies={N} other_style_bodies={len(other)}")
    log("(hầu hết styled corpus là hài → đánh giá nội tại trong humor, không A/B buồn)")

    uni_tf, uni_df = ngram_stats(humor, 1)
    bi_tf, bi_df = ngram_stats(humor, 2)
    tri_tf, tri_df = ngram_stats(humor, 3)

    STOP = {
        "và", "là", "của", "có", "một", "cho", "với", "như", "được", "không",
        "các", "này", "đó", "thì", "khi", "để", "trong", "về", "đã", "sẽ",
        "bị", "bằng", "hay", "rất", "cũng", "những", "người", "mình", "bạn",
        "lại", "còn", "nên", "mà", "ở", "đi", "làm", "ra", "vào", "lên",
        "xuống", "thấy", "biết", "nói", "đến", "từ", "sau", "trước", "rồi",
    }

    log("\n## 1) Unigram đặc trưng (df 5–60, bỏ function)")
    cands = []
    for (w,), df in uni_df.items():
        if df < 5 or df > 60 or w in STOP or len(w) <= 1:
            continue
        cands.append((df * math.log(1 + uni_tf[(w,)]), df, uni_tf[(w,)], w))
    cands.sort(reverse=True)
    for _, df, tf, w in cands[:35]:
        log(f"  df={df:3} tf={tf:4}  {w}")

    log("\n## 2) Bigram tái xuất hiện (df≥3), ưu tiên điệp âm")
    bc = []
    for g, df in bi_df.items():
        if df < 3:
            continue
        a, b = g
        if a in STOP and b in STOP:
            continue
        bonus = 1.6 if a[:1] == b[:1] else 1.0
        bc.append((df * math.log(1 + bi_tf[g]) * bonus, df, bi_tf[g], f"{a} {b}"))
    bc.sort(reverse=True)
    for _, df, tf, s in bc[:45]:
        log(f"  df={df:2} tf={tf:3}  {s}")

    log("\n## 3) Trigram (df≥2)")
    tc = []
    for g, df in tri_df.items():
        if df < 2:
            continue
        tc.append((df * math.log(1 + tri_tf[g]), df, tri_tf[g], " ".join(g)))
    tc.sort(reverse=True)
    for _, df, tf, s in tc[:30]:
        log(f"  df={df:2} tf={tf:3}  {s}")

    log("\n## 4) Bucket giọng (tỷ lệ thân chứa pattern)")
    patterns = {
        "chêm/giọng (phết, hơi lố, đấy chứ, thôi thì, nói thật)":
            r"(phết|hơi lố|đấy chứ|thôi thì|nói thật|công nhận|thật đấy)",
        "so sánh hài (hệt như, kiểu như, y như, giống hệt)":
            r"(hệt như|giống hệt|kiểu như|y như|như thể)",
        "đối lập xưa/nay":
            r"(thời nay|ngày xưa|bây giờ|giờ thì|ngày nay|thời buổi)",
        "hiện đại/meme (TikTok, Gen Z, deadline, check-in…)":
            r"(TikTok|Gen Z|deadline|check-?in|Unfriend|Threads|skincare)",
        "điệp âm/redup (ngơ ngác, tơi bời, lộng lẫy…)":
            r"(ngơ ngác|phũ phàng|tơi bời|xôn xao|nơm nớp|lộng lẫy|vĩnh viễn|"
            r"chen chúc|cuống quýt|ngậm ngùi|hừng hực|xịn mịn|rôm rốp|hoành tráng)",
        "tự trào/meta (nghe hơi, có lý, tưởng đang)":
            r"(nghe hơi|có lý|tưởng đang|tưởng là|chứ không phải)",
    }
    for name, pat in patterns.items():
        n = sum(1 for b in humor if re.search(pat, b, re.I))
        log(f"  {name}: {n}/{N} ({100 * n / max(N, 1):.1f}%)")

    log("\n## 5) Kết luận định hướng inventory")
    log("Nên ưu tiên (theo data):")
    log("  (A) cụm điệp âm/redup — dày, ổn định, mang sắc thái")
    log("  (B) frame so sánh: hệt như / kiểu như / y như")
    log("  (C) chêm thoại: phết, hơi lố, đấy chứ — HIẾM nhưng đúng giọng")
    log("  (D) neo hiện đại: TikTok/Gen Z/deadline — topic-hài đời thường")
    log("KHÔNG nên lấy làm 'hài': unigram topic đơn (lướt, mua, phút, kiệm=BPE)")
    log("KHÔNG đủ: chỉ «hài hước» neighbor CO_ACT (cốt/kịch)")

    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log(f"\n→ {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
