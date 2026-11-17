# RUNBOOK TK_046 P98F — siết cầu cùng lớp trái (pattern) với donors

> Sau P98E ĐẠT hình thức: cùng token `nuôi` nhưng `mẹ nuôi` ≠ `Hồng nuôi`.
> Siết: seed và donors phải **cùng lớp trái** quanh bigram `(x, p)`.

## 0. Cổng (nêu TRƯỚC)

### Cầu (giữ P98E)

- bigram **d=1**; bề mặt `(x, p)` ⇒ cạnh **`p → x`**

### Siết pattern (mới)

Với mỗi cạnh `p→x`, ghi **trái** = token ngay trước `x` (bỏ punct; nếu không có → lấy lớp từ chính `x`).

`lop_trai`:

| lớp | ví dụ trái / entity |
|---|---|
| `nguoi` | `người`, `con`, kinship (`mẹ`,`gái`,…) |
| `dia_ly` | `Sông`, `Núi`, `biển`, `Đảo`, `vịnh`… |
| `khac` | còn lại |

Donor `b` chỉ đếm cho seed `a` qua `p` khi:

`lop_trai(p→a) ∩ lop_trai(p→b) ≠ ∅`

→ `Hồng nuôi` (trái `Sông` = `dia_ly`) **không** khớp `mẹ nuôi` / `gái nuôi` (`nguoi`).

### Ca / cổng G (như P98E)

| # | điều kiện |
|---|---|
| 1 | ≥10 ca dương điền được |
| 2 | Hit@1 ≥40% |
| 3 | FP âm ≤15% |
| 4 | creative (triple vắng) |

### Cổng phụ Q1 (chất lượng)

Trong examples dương: **không** còn ca `Hồng`/`Sông` qua cầu `nuôi` (hoặc mọi ca: `lop_trai(a)` giao donors ≠ ∅).

## 1. Lệnh

```bash
PYTHONPATH=src .venv/bin/python scripts/probe_analogy_exec_v4.py \
  --checkpoint checkpoints/mini_sent_ep_qa.pkl \
  --out out/probe_analogy_exec_v4.json
```

## 2. Kỳ vọng

- `Hồng là` / `Sông là` **biến** khỏi dương (cầu `nuôi`)
- Hit có thể giảm vs P98E 100%; FP thấp
- G ĐẠT nếu còn ≥10 ca sạch hơn

## 3. Kết quả (2026-08-07)

| | |
|---|---|
| bridge hit | 16/16 = **100%** · FP 0% |
| cổng G | **ĐẠT** |
| Q1 Hồng/Sông | **✅ chết** |

Artefacts: `out/probe_analogy_exec_v4.json` · `TK_046` §18.F

## 4. P98F+ — cộng nhiều cầu (cùng ngày)

`score(c) = Σ_p strength(p,c)` trên mọi cầu khớp `lop_trai` (không chỉ argmax 1 p).

| | |
|---|---|
| cổng G | vẫn **ĐẠT** (16/16, FP 0%) |
| ca ≥2 cầu | **1/16** (vd `anh`: `họ`+`thứ`) |

Nhiều cầu vẫn hiếm trên mini — trọng số đúng hướng; chất lượng nghĩa cần cầu cụm tiếp.
