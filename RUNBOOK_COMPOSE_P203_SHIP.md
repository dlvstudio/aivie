# RUNBOOK — compose_2cha P203 (đường ship research)

> Chốt 2026-08-06. Đọc `TK_051_MULTI_PARENT_COMPOSE.md` §11.
> Nhãn: **dual retrieve + bản lề W_skip** — không phải sinh sáng tạo tự do.
> Ship khung–khe 1-cha vẫn là `RUNBOOK_FRAME_P112_SHIP.md`.

## 0. Công thức

`join_d_max=8` · `join_min_count=1` · `top_n=24` · max 12 cand ·
chọn B theo `(ent_keep, join_score)` · fallback P112 (`bam` · min_gain=1 · max_thay=2) ·
ckpt `network_v12_subst.pkl`.

Thước đã đạt: `compose−chep = +0,8đ` · `n_src≥2 = 99,2%` · `ent_keep` 0,875.

## 1. Một lệnh sinh

```bash
PYTHONPATH=src .venv/bin/python scripts/gen_compose_2cha.py \
  --checkpoint checkpoints/network_v12_subst.pkl \
  --prompt "Trong các đợt tiến công"
```

## 2. Opt-in qua `aivie generate`

Mặc định `aivie generate` vẫn **`w_skip`**. `frame_da_khe` = P112 1-khung.

```bash
PYTHONPATH=src .venv/bin/python -m nn_aivie.cli generate \
  --checkpoint checkpoints/network_v12_subst.pkl \
  --mode compose_2cha \
  --prompt "Trong các đợt tiến công"
```

## 3. Không làm trên đường ship

- Không đổi mặc định `generation_mode` sang `compose_2cha`.
- Soft thuc_the / Hebbian-CO / săn `n_thay≥2`.
- Coi `n_thay≥2` = composition thành công (dùng `n_src≥2`).
