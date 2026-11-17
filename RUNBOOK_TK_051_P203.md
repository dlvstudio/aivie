# RUNBOOK TK_051 P203 — chấm mù compose_2cha (cổng 3)

> Sau P202 smoke ĐẠT (`n_src≥2=100%` · `ent_keep` 0,872).
> Đọc `TK_051_MULTI_PARENT_COMPOSE.md` §3 cổng (3).

## 1. Cổng

`compose − chep ≥ −5đ` (dùng+tạm) **và** `n_src≥2` ≥30% trên n=120.

Đối chứng: `chep_cau`, `ep_da_khe_giao` (P112). Neo 40 như P107.

## 2. Sinh n=120

```bash
PYTHONPATH=src .venv/bin/python scripts/gen_three_arm.py \
  --checkpoint checkpoints/network_v12_subst.pkl \
  --label-map out/token_labels_v12.json \
  --arms chep_cau,ep_da_khe_giao,compose_2cha \
  --max-thay 2 --slot-rank bam --min-gain 1 --no-co-graph \
  --join-d-max 8 --join-min-count 1 --top-n 24 \
  --n 120 \
  --out out/compose_p203_samples.txt \
  --out-key out/compose_p203_key.json \
  --out-meta out/compose_p203_meta.json
```

Trộn neo seed=203 → `out/compose_p203_cham.txt` → chấm mù → `out/compose_p203_verdict.json`.

## 3. Kết quả (2026-08-06) — cổng (3) **ĐẠT**

| nhánh | dung | dùng+tạm | `n_src≥2` |
|---|---:|---:|---:|
| `chep_cau` | 36,7% | 54,2% | 0% |
| `ep_da_khe_giao` | 26,7% | 44,2% | 0% |
| **`compose_2cha`** | 18,3% | **55,0%** | **99,2%** |

`compose − chep = +0,8đ` · neo TRÔI −7,5%. Chi tiết `TK_051` §10.

## 4. Không làm

- Đổi luật nối giữa lúc chấm.
- Soft thuc_the / Hebbian-CO.
