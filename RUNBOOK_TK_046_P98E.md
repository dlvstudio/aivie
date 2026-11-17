# RUNBOOK TK_046 P98E — G-exec qua CẦU THUỘC TÍNH

## Cầu (đã sửa theo phản biện)

1. **Chỉ bigram d=1** (kề ngay) — không cửa sổ ±6.  
2. **Hướng thuộc_tính → thực_thể**: bigram bề mặt `(x, p)` ⇒ cạnh `p → x`.  
   - Đúng: `nghịch → khỉ` từ “khỉ nghịch”  
   - Sai: `mẹ → hôn` / đồng hiện xa như “ban hôn , Thái Tông”

## Ca sáng tạo

`(a,là,c)` VẮNG; ≥2 donors `b` cùng nhận `p→b` và có `(b,là,c)`.

## Cổng

| # | điều kiện |
|---|---|
| 1 | ≥10 ca dương |
| 2 | Hit@1 ≥40% |
| 3 | FP âm ≤15% |
| 4 | creative |

```bash
PYTHONPATH=src .venv/bin/python scripts/probe_analogy_exec_v3.py \
  --checkpoint checkpoints/mini_sent_ep_qa.pkl \
  --out out/probe_analogy_exec_v3.json
```
