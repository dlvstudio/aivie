# RUNBOOK TK_051 P202 — nối có phép (sau P201 ĐẠT)

> 2026-08-06. P201 smoke: `n_src≥2=100%` · `ent_keep` compose ≥ chep/ep_da.
> Đọc `TK_051_MULTI_PARENT_COMPOSE.md` §3 cổng (3) — P202 chuẩn bị nối, P203 mới chấm.

## 1. Mục tiêu

Chỉ nối emit A+B khi chuyển `(cuối nội dung A → đầu nội dung B)` **đã thấy** trong
corpus (W_skip count hoặc bigram cha ≥ ngưỡng). Không đạt → fallback P112 (chỉ cha #1 + vá).

Cổng smoke P202 (trước chấm):

- Giữ `n_src≥2` ≥30% sau khi siết nối (không tụt về toàn fallback).
- `ent_keep` không thấp hơn P201 quá 5đ tuyệt đối trên cùng n=40.

## 2. Lệnh smoke (đã chạy)

```bash
PYTHONPATH=src .venv/bin/python scripts/gen_three_arm.py \
  --checkpoint checkpoints/network_v12_subst.pkl \
  --label-map out/token_labels_v12.json \
  --arms chep_cau,ep_da_khe_giao,compose_2cha \
  --max-thay 2 --slot-rank bam --min-gain 1 --no-co-graph \
  --join-d-max 8 --join-min-count 1 --top-n 24 \
  --n 40 \
  --out out/compose_p202_smoke.txt \
  --out-key out/compose_p202_smoke_key.json \
  --out-meta out/compose_p202_smoke_meta.json
```

Luật nối: cửa sổ 3 token keep cuối A × 3 đầu B trên W_skip d≤8; chọn B max `(ent_keep, join_score)`.

## 3. Kết quả smoke (2026-08-06) — **ĐẠT**

| | `n_src≥2` | `ent_keep` | joined |
|---|---:|---:|---:|
| P201 | 100% | 0,793 | 100% |
| **P202** | **100%** | **0,872** | 100% |

Cổng: (1) ✅ · (2) Δ=+0,080 ≥ −0,05 ✅. Việc kế: **P203** chấm mù.

## 4. Không làm

- Soft đa-khe / thuc_the.
- Chấm mù 400 trước khi smoke P202 đạt.
