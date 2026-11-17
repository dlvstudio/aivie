#!/usr/bin/env python
"""P84 (TK_045 §13.53) — **"NGẮN THÌ RỖNG"**: đầu ra một mệnh đề có chỉ DỘI LẠI chữ của đề?

**Vì sao.** §13.52 đổi mặc định `max_clauses=1` vì mỗi mối nối đều làm hỏng đầu ra. Nhưng
giám khảo mô tả bản ngắn hỏng theo một kiểu riêng: *"DỘI LẠI chữ cuối của đề rồi tắt"* —
`học và` → `học hành` · `vùng xa` → `xa.` · `công cụ bằng đá` → `bóng đá Ý`. *"Dừng đúng chỗ
nhưng **nội dung rỗng**."*

Nếu đúng, thì 13,0% `dung+tam` của `len1` đang **che** một khuyết tật: hệ không nói bậy nữa,
nhưng cũng **không nói gì mới**.

**Đo trên dữ liệu ĐÃ CHẤM** (`abs4`, nhánh `len1`, 200 mẫu v12) — thuần văn bản, không cần
checkpoint, không sinh lại ⇒ không có chỗ cho tôi vô tình chọn mẫu.

**Định nghĩa DỘI:** chữ đầu tiên của phần sinh **xuất hiện trong 3 chữ cuối của đề**.
Đối chứng nội tại: các mẫu **không dội** trong cùng nhánh, cùng phiên chấm.

## DỰ ĐOÁN GHI TRƯỚC (ghi vào file TRƯỚC khi chạy — Claude)

1. Tỷ lệ dội trong nhánh `len1` (mẫu có chữ): **≥ 25%**.
2. Mẫu **dội** tệ hơn rõ rệt: `dung+tam` của nhóm dội **≤ 5%**, của nhóm không dội **≥ 20%**.
3. Nhánh `len2` cũng có dội nhưng **ít hơn** — vì nó còn cơ hội đi tiếp.

Nếu (2) **sai** — nhóm dội ngang hoặc hơn nhóm không dội — thì "dội = rỗng" là **ấn tượng của
giám khảo**, không phải khuyết tật, và §13.52 không cần sửa gì.

Chạy: `python scripts/probe_echo_empty.py`
"""
from __future__ import annotations

import json
import math
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_WORD = re.compile(r"[^\W_]+", re.UNICODE)


def words(s: str) -> list[str]:
    return _WORD.findall(unicodedata.normalize("NFC", s).lower())


def main() -> None:
    samples: dict[str, tuple[str, str]] = {}
    cur = None
    for line in (_ROOT / "out/abs4_samples_v12.txt").read_text(encoding="utf-8").splitlines():
        m = re.match(r"^\[(A\d+)\] ĐỀ: (.+)$", line)
        if m:
            cur = m.group(1)
            samples[cur] = (m.group(2).strip(), "")
            continue
        m = re.match(r"^\s+→ (.*)$", line)
        if m and cur:
            samples[cur] = (samples[cur][0], m.group(1).strip())

    grades = json.loads((_ROOT / "out/abs4_verdict_v12.json").read_text(encoding="utf-8"))["grades"]
    key = json.loads((_ROOT / "out/abs4_key_v12.json").read_text(encoding="utf-8"))

    def is_echo(prompt: str, out: str) -> bool | None:
        wo = words(out)
        wp = words(prompt)
        if not wo or not wp:
            return None
        return wo[0] in set(wp[-3:])

    rows = []
    for sid, (pr, out) in samples.items():
        if sid not in grades or "TRỐNG" in out:
            continue
        e = is_echo(pr, out)
        if e is None:
            continue
        rows.append((key[sid], e, grades[sid]))

    print(f"{'nhánh':<8}{'nhóm':<12}{'n':>5}{'dung+tam':>11}")
    stat = {}
    for cfg in sorted({r[0] for r in rows}):
        for e, lab in ((True, "DỘI"), (False, "không dội")):
            sub = [r for r in rows if r[0] == cfg and r[1] is e]
            if not sub:
                continue
            c = Counter(r[2] for r in sub)
            ok = c["dung_duoc"] + c["tam"]
            stat[(cfg, e)] = (ok, len(sub))
            print(f"{cfg:<8}{lab:<12}{len(sub):>5}{ok/len(sub):>10.1%}")

    for cfg in sorted({r[0] for r in rows}):
        n_e = sum(1 for r in rows if r[0] == cfg and r[1])
        n_t = sum(1 for r in rows if r[0] == cfg)
        print(f"\n{cfg}: tỷ lệ DỘI = {n_e}/{n_t} = {n_e/n_t:.1%}")
        if (cfg, True) in stat and (cfg, False) in stat:
            a, na = stat[(cfg, True)]
            b, nb = stat[(cfg, False)]
            p = (a + b) / (na + nb)
            se = math.sqrt(p * (1 - p) * (1 / na + 1 / nb))
            z = (b / nb - a / na) / se if se else 0.0
            print(f"  không-dội − dội = {b/nb - a/na:+.1%} · z = {z:.2f} ⇒ "
                  f"{'TÁCH' if abs(z) > 1.96 else 'KHÔNG tách khỏi nhiễu'}")

    print("\nví dụ DỘI (nhánh len1):")
    n = 0
    for sid, (pr, out) in samples.items():
        if key.get(sid) == "len1" and "TRỐNG" not in out and is_echo(pr, out) and n < 8:
            print(f"  {pr[-45:]!r}  →  {out[:45]!r}")
            n += 1


if __name__ == "__main__":
    sys.exit(main())
