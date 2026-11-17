# RUNBOOK TK_049 P110 — siết gain≥2 + max_thay=2 (lấy ~1đ sát cổng 4)

> Cho M4. Ngày giao: 2026-08-06.
> P109: `ep_da−chep = −5,8đ` (cổng 4 cần ≥−5). Giữ cấu hình thắng: subst · bam · no-co-graph.
> Chỉ siết thêm: `--min-gain 2 --max-thay 2`.

## Dự đoán ghi TRƯỚC

- `n_thay≥1` tụt (P109: 41,7%) nhưng vẫn ≥30%.
- `ep_da − chep` vào **−5…0đ** ⇒ cổng 4 **đạt**. Nếu tụt về ≤−8đ ⇒ siết quá tay, quay `min-gain=1`.

```bash
PYTHONPATH=src .venv/bin/python scripts/gen_three_arm.py \
  --checkpoint checkpoints/network_v12_subst.pkl \
  --label-map out/token_labels_v12.json \
  --arms chep_cau,ep_mot_khe_giao,ep_da_khe_giao \
  --max-thay 2 --slot-rank bam --min-gain 2 --no-co-graph \
  --n 120 \
  --out out/frame_p110_samples.txt \
  --out-key out/frame_p110_key.json \
  --out-meta out/frame_p110_meta.json
```

Neo seed=110 → `out/frame_p110_cham.txt` · verdict `out/frame_p110_verdict.json`.
