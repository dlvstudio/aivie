# RUNBOOK TK_050 P114 — decode thuc_the/entity hub

> 2026-08-06. Sau P113 (wire P112). Cổng (2) TK_050.
> Đọc `TK_050_FRAME_GEN_PRODUCTIZE.md` §1 cổng (2) trước khi chạy.

## 1. Mục tiêu

Điền khe ở mức **đơn vị thuc_the** (từ WS-CRF + ghép tên riêng), thay cả cụm entity
(`Lý Thái Tổ` → hub `Entity:…`), không chỉ âm tiết đơn.

Giữ cấu hình P112: `bam · min-gain=1 · max_thay=2 · no-co-graph · SYNONYM bake`.

| cổng | điều kiện |
|---|---|
| (2a) | `n_thay≥2` ≥ **35%** trên nhánh `ep_da_khe_giao` |
| (2b) | Giữ cổng (4): `ep_da−chep ≥ −5đ` **và** `n_thay≥1 ≥ 30%` |

## 2. Lệnh sinh mẫu

Checkpoint: `checkpoints/network_v12_subst.pkl` (bake P106 nếu thiếu).

```bash
PYTHONPATH=src .venv/bin/python scripts/gen_three_arm.py \
  --checkpoint checkpoints/network_v12_subst.pkl \
  --label-map out/token_labels_v12.json \
  --arms chep_cau,ep_mot_khe_giao,ep_da_khe_giao \
  --max-thay 2 --slot-rank bam --min-gain 1 --no-co-graph \
  --decode-unit thuc_the \
  --n 120 \
  --out out/frame_p114_samples.txt \
  --out-key out/frame_p114_key.json \
  --out-meta out/frame_p114_meta.json
```

Log kỳ vọng: `P114 decode thuc_the` · `chỉ mục đơn vị: bối cảnh … · lớp …`.

**Lưu ý:** bước dựng chỉ mục đơn vị quét ~430k cha + WS-CRF — **chậm** (~10–20 phút).

## 3. Thống kê nhanh từ meta

```bash
PYTHONPATH=src .venv/bin/python - <<'PY'
import json
from collections import Counter
m = json.loads(open("out/frame_p114_meta.json", encoding="utf-8").read())
rows = [r for r in m["rows"] if r.get("cfg") == "ep_da_khe_giao"]
c = Counter(int(r.get("n_thay", 0)) for r in rows)
n = len(rows)
ge1 = sum(v for k, v in c.items() if k >= 1) / max(n, 1)
ge2 = sum(v for k, v in c.items() if k >= 2) / max(n, 1)
print(f"ep_da n={n} · ge1={ge1:.1%} · ge2={ge2:.1%} · hist={dict(sorted(c.items()))}")
print(f"cổng 2a ge2≥35%: {'✅' if ge2 >= 0.35 else '❌'}")
PY
```

## 4. Chấm mù (cổng 4 / 2b)

Giống `RUNBOOK_TK_049_P107.md` §2–3: trộn neo 40 → `out/frame_p114_cham.txt` → chấm
dùng/tạm/hỏng tuyệt đối → `out/frame_p114_verdict.json`.

So `ep_da_khe_giao` vs `chep_cau` trên cùng bộ 120 đề.

## 5. Hồi quy P112 (token)

```bash
PYTHONPATH=src .venv/bin/python scripts/gen_frame_slot.py --verify-p112
```

Cổng: ≥95% khớp `n_thay` (mặc định `decode_unit=token`).

## 6. Giao nộp

- `src/nn_aivie/hca/frame_slot_gen.py` — `decode_unit=thuc_the`
- `--decode-unit` trên `gen_three_arm` / `gen_frame_slot`
- `out/frame_p114_*` + cập nhật `TK_050` §5
