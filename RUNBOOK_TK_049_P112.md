# RUNBOOK TK_049 P112 — cấu hình cổng (4) ĐẠT

> 2026-08-06. Kết quả: `ep_da−chep = −3,3đ` · `n_thay≥1 = 41,7%`.

```bash
PYTHONPATH=src .venv/bin/python scripts/gen_three_arm.py \
  --checkpoint checkpoints/network_v12_subst.pkl \
  --label-map out/token_labels_v12.json \
  --arms chep_cau,ep_mot_khe_giao,ep_da_khe_giao \
  --max-thay 2 --slot-rank bam --min-gain 1 --no-co-graph \
  --n 120 \
  --out out/frame_p112_samples.txt \
  --out-key out/frame_p112_key.json \
  --out-meta out/frame_p112_meta.json
```
