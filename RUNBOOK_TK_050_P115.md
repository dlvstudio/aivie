# RUNBOOK TK_050 P115 — siết ep_da−chep về ≥ −2đ

> 2026-08-06. Sau P114 (decode thuc_the, ge2=43,3%). Cổng (3) TK_050 — tuỳ chọn.
> Đọc `TK_050_FRAME_GEN_PRODUCTIZE.md` §1 cổng (3) trước khi code.

## 1. Mục tiêu

Thu hẹp khoảng `ep_da_khe − chep_cau` (dùng+tạm) từ ~−3…−5đ xuống **≥ −2đ**,
*không* giảm `n_thay≥1` dưới 30% và *không* giảm `n_thay≥2` dưới 35% (giữ cổng 2).

Ứng viên (ghi TRƯỚC, đo sau):

| # | ý tưởng | vì sao |
|---|---|---|
| A | Cấm thay `Entity:` khác **niên đại**/triều đại so với đề | nhảy Lý↔Nguyễn = hỏng rõ |
| B | Cấm thay hub khác **lĩnh vực** (địa danh↔người↔sự kiện) | lệch nhân vật/lĩnh vực |
| C | Không Hebbian-CO (P108 đã đo hại) | đã loại |

## 2. Việc làm

1. Filter trong `_fill_da_khe_thuc_the`:
   - `--era-filter`: cấm ứng viên mang triều/niên đại lạ so với đề+khe gốc
   - `--soft-giao-single-only`: soft lop chỉ cho đơn vị 1-token (hub đa-token = hard giao)
   - `--hybrid-if-zero`: token hybrid chỉ khi 0 thay đơn vị
   - `--second-min-gain 2`: khe thứ 2 cần bam-gain ≥ 2
2. Sinh n=120 cấu hình P114 + 4 flag trên → `out/frame_p115_*`
3. Chấm mù. Cổng (3): `ep_da−chep ≥ −2đ` *và* ge1≥30%, ge2≥35%.

```bash
PYTHONPATH=src .venv/bin/python scripts/gen_three_arm.py \
  --checkpoint checkpoints/network_v12_subst.pkl \
  --label-map out/token_labels_v12.json \
  --arms chep_cau,ep_mot_khe_giao,ep_da_khe_giao \
  --max-thay 2 --slot-rank bam --min-gain 1 --no-co-graph \
  --decode-unit thuc_the \
  --era-filter --soft-giao-single-only --hybrid-if-zero --second-min-gain 2 \
  --n 120 \
  --out out/frame_p115_samples.txt \
  --out-key out/frame_p115_key.json \
  --out-meta out/frame_p115_meta.json
```

## 3. Giao nộp

Code filter + meta + verdict + cập nhật TK_050 §6.

## 4. Kết quả (2026-08-06) — cổng (3) TRƯỢT

Xem `TK_050_FRAME_GEN_PRODUCTIZE.md` §6. Ba biến thể đều Δ ≤ −6,7đ.
Trần sống vẫn **P112 token** (−3,3đ). Dừng siết.
