# RUNBOOK — khung–khe P112 (đường ship)

> Chốt 2026-08-06. Đọc `TK_050_FRAME_GEN_PRODUCTIZE.md` §7.
> Nhãn: **truy hồi + khung có phép** — không phải sinh sáng tạo tự do.

## 0. Công thức

`decode_unit=token` · `bam` · `min_gain=1` · `max_thay=2` · `no_co_graph` ·
SYNONYM bake · cấm khe `số` · ckpt `network_v12_subst.pkl`.

Thước đã đạt: `ep_da−chep = −3,3đ` · `n_thay≥1 = 41,7%`.

## 1. Checkpoint subst (nếu thiếu)

Xem `RUNBOOK_TK_049_P106.md` — bake từ `network_v12_sent.pkl` → `network_v12_subst.pkl`.

## 2. Một lệnh sinh

```bash
PYTHONPATH=src .venv/bin/python scripts/gen_frame_slot.py \
  --checkpoint checkpoints/network_v12_subst.pkl \
  --prompt "Trong các đợt tiến công"
```

Defaults CLI = P112 (không cần truyền max-thay/min-gain trừ khi ablation).

Hồi quy vs P112 (opener-slice):

```bash
PYTHONPATH=src .venv/bin/python scripts/gen_frame_slot.py --verify-p112
```

Kỳ vọng: khớp `n_thay` ≥ 95% (đã đo 120/120).

## 3. Opt-in qua `aivie generate`

Mặc định `aivie generate` vẫn **`w_skip`** (không đổi hành vi eval cũ).

```bash
PYTHONPATH=src .venv/bin/python -m nn_aivie.cli generate \
  --checkpoint checkpoints/network_v12_subst.pkl \
  --mode frame_da_khe \
  --prompt "Trong các đợt tiến công"
```

Cần ckpt có cạnh SYNONYM (subst). Thiếu bake → lỗi rõ từ `frame_slot_gen`.

## 4. Không làm trên đường ship

- Không bật `decode_unit=thuc_the` làm mặc định (P114/P115).
- Không Hebbian-CO vào `_co` (P108).
- Không `min-gain≥2` mặc định (P110/111 hết sáng tạo đo được).
