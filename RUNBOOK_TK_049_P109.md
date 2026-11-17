# RUNBOOK TK_049 P109 — ablation: xếp khe theo bám đề, không CO-graph Hebbian

> Cho M4. Ngày giao: 2026-08-06.
> P108: Hebbian+max_thay=6 theo |cand| → `ep_da−chep = −14,2đ` (xấu hơn P107 −7,5đ).
> 1-khe lại khá hơn ⇒ nghi ngờ trần theo |cand| + CO-graph bão hoà.

## 1. Giả thuyết ghi TRƯỚC

Hai thay đổi độc lập so với P108:

| đòn | P108 | P109 |
|---|---|---|
| checkpoint | `subst_h` (Hebbian) | **`subst`** (chỉ SYNONYM P106) |
| CO-graph vào `_co` | bật | **`--no-co-graph`** |
| xếp khe | `|cand|` | **`--slot-rank bam`** (độ tăng bám đề; bỏ khe gain≤0) |
| trần | 6 | **3** |

Dự đoán: `ep_da − chep` về vùng **−5…−10đ** (gần P107 hoặc đạt cổng 4). Nếu vẫn ≤−14đ
⇒ bệnh không phải thứ tự/Hebbian mà là đa-khe giao nói chung trên substrate này.

## 2. Lệnh

```bash
PYTHONPATH=src .venv/bin/python scripts/gen_three_arm.py \
  --checkpoint checkpoints/network_v12_subst.pkl \
  --label-map out/token_labels_v12.json \
  --arms chep_cau,ep_mot_khe_giao,ep_da_khe_giao \
  --max-thay 3 --slot-rank bam --no-co-graph \
  --n 120 \
  --out out/frame_p109_samples.txt \
  --out-key out/frame_p109_key.json \
  --out-meta out/frame_p109_meta.json
```

Trộn neo seed=109 → chấm → `out/frame_p109_verdict.json`. Cổng (4) như P107.

Kiểm meta: mean `n_thay` ≤3 · `slot_rank=bam` · log có `+CO graph 0 (tắt)`.
