# RUNBOOK TK_049 P111 — min-gain=2 + max_thay=3 (đủ sáng tạo + giữ chất lượng)

> P110: Δ=−4,2đ ✅ nhưng n_thay≥1=25% ❌. Giữ gain=2, nới trần 3.

```bash
PYTHONPATH=src .venv/bin/python scripts/gen_three_arm.py \
  --checkpoint checkpoints/network_v12_subst.pkl \
  --label-map out/token_labels_v12.json \
  --arms chep_cau,ep_mot_khe_giao,ep_da_khe_giao \
  --max-thay 3 --slot-rank bam --min-gain 2 --no-co-graph \
  --n 120 \
  --out out/frame_p111_samples.txt \
  --out-key out/frame_p111_key.json \
  --out-meta out/frame_p111_meta.json
```

Neo seed=111. Cổng (4): Δ≥−5đ **và** n_thay≥1≥30%.
