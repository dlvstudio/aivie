# RUNBOOK TK_046 P98D — G-exec v2 (sáng tạo, không gold trigram)

> Thước P98C **hỏng**: gold = mọi `(a,f,c)` liên tiếp ở B → cắt cụm / tiếp nối cú pháp
> (`Thăng Long để`, `văn học Trung đại`). Không đo loại suy.
>
> Thước mới bám ví dụ tác giả: **fact `(a,f,c)` chưa từng có**, suy từ ≥2 bạn đồng lớp.

## 0. Cổng G-exec-v2 (nêu TRƯỚC)

### Xây ca (mini)

1. Chọn khung `f` = token `là` (surface; mọi id `là`/`Ġlà` nếu có).
2. **Dương (T1-style):** chủ thể `a` **không** có bộ ba `(a, là, c)` nào trong **toàn** corpus
   với một `c` cụ thể, nhưng **≥2** bạn `b` (cos `E` top) có `(b, là, c)`.
3. **Âm (T2-style):** `a` chỉ có **≤1** bạn mang `(b, là, c)` → **không** được điền `c`.

Hệ chỉ thấy đồng hiện / khung của các cha **không** chứa seed `a` làm chủ thể trước `là`
(cách ly: bỏ mọi cha có emit chứa `a` ngay trước `là` khi suy cho `a`).

### Cổng

| # | điều kiện | ý nghĩa |
|---|---|---|
| 1 | ≥30 ca dương điền được | đủ mẫu |
| 2 | Hit@1 dương **≥ 40%** | suy đúng `c` held-out (sáng tạo: triple chưa có cho `a`) |
| 3 | False-positive âm **≤ 15%** | không nhiễm loại (bàn→động vật) |
| 4 | Mọi hit dương: `(a,là,c)` **vắng** trên toàn corpus (kiểm lại) | không phải chép |

Đối chứng: cùng ca, điền bằng `embedding` (không min_support) và `ngau_nhien`.

## 1. Lệnh

```bash
PYTHONPATH=src .venv/bin/python scripts/probe_analogy_exec_v2.py \
  --checkpoint checkpoints/mini_sent_ep_qa.pkl \
  --out out/probe_analogy_exec_v2.json
```

## 2. Không làm

- Gold = trigram tiếp nối tùy ý (P98C).
- Cắm vào `rank_matches` / chép cha.
- Đổi mặc định generate.

## 3. Kết quả (2026-08-06)

| bộ | hit@1 dương | FP âm |
|---|---:|---:|
| `ngau_nhien` | 1,0% | 5,0% |
| `embedding` | 24,5% | 2,0% |
| **`hebbian_E`** | **35,5%** | **0%** |

Cổng (1)(3)(4) ✅ · (2) 35,5% < 40% ❌ → **TRƯỢT hẹp**. P98C rút. Không bake.
