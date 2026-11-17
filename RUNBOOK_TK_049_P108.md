# RUNBOOK TK_049 P108 — Hebbian kế thừa tính chất + trần đa-khe

> Cho M4. Ngày giao: 2026-08-06.
> P107: `ep_da − chep = −7,5đ` (cổng 4 trượt hẹp) · đa-khe thắng 1-khe +9,2đ · mean ~28 thay.
> Hai đòn: (A) đông cứng suy luận paradigmatic → tính chất; (B) **trần** số khe thay.

## 1. Câu hỏi

Siết bể + Hebbian có kéo `ep_da − chep` từ −7,5đ lên **≥ −5đ** (cổng 4) mà vẫn giữ
`n_thay≥1 ≥ 30%` và `n_thay≥2 ≥ 25%` không?

Dự đoán ghi TRƯỚC: trần `max_thay=6` một mình thu hẹp ~3–8đ; Hebbian giúp chọn ứng viên
bám đề hơn qua `_co` kế thừa — hiệu cộng thêm 0–3đ. Kỳ vọng cổng 4 **đạt sát**.

## 2. Bake Hebbian

Từ `network_v12_subst.pkl` (không đè):

```bash
PYTHONPATH=src .venv/bin/python scripts/bake_hebbian_subst.py \
  --checkpoint checkpoints/network_v12_subst.pkl \
  --inherit-scale 0.45 \
  --max-inherit 8 \
  --min-w 0.05 \
  --out-checkpoint checkpoints/network_v12_subst_h.pkl \
  --out-meta out/bake_hebbian_v12_meta.json
```

Cổng bake: `n_inherited ≥ 10_000` (có thật sự viết cạnh) · hai lần chạy meta gần giống.

## 3. Sinh + chấm (cùng quy trình P107)

```bash
PYTHONPATH=src .venv/bin/python scripts/gen_three_arm.py \
  --checkpoint checkpoints/network_v12_subst_h.pkl \
  --label-map out/token_labels_v12.json \
  --arms chep_cau,ep_mot_khe_giao,ep_da_khe_giao \
  --max-thay 6 \
  --n 120 \
  --out out/frame_p108_samples.txt \
  --out-key out/frame_p108_key.json \
  --out-meta out/frame_p108_meta.json
```

Trộn neo seed=108 → `out/frame_p108_cham.txt` · chấm → `out/frame_p108_verdict.json`.

Cổng (4) lại: `ep_da − chep ≥ −5đ` *và* `n_thay≥1 ≥ 30%`. Báo thêm mean `n_thay` (kỳ vọng ≤6).

## 4. Giao nộp

Meta bake · samples/key/meta/cham/verdict · cập nhật `TK_049` §6.
