# RUNBOOK TK_050 P113 — wire khung–đa-khe P112 vào đường sinh

> Cho M4. Ngày giao: 2026-08-06.
> Đọc `TK_050_FRAME_GEN_PRODUCTIZE.md` §1 cổng (1) trước khi code.

## 1. Mục tiêu

Một lối gọi (CLI hoặc flag `generate`) tái lập **đúng** cấu hình P112:

- checkpoint: `checkpoints/network_v12_subst.pkl` (bake từ `RUNBOOK_TK_049_P106.md` nếu thiếu)
- licence giao từ `SYNONYM` bake
- `--slot-rank bam --min-gain 1 --max_thay 2 --no-co-graph`
- cấm khe `số` qua `out/token_labels_v12.json`

## 2. Việc làm

1. Tách lõi `ep_da_khe` từ `scripts/gen_three_arm.py` → module import được
   (vd. `nn_aivie.hca.frame_slot_gen`) — không copy-paste luật.
2. CLI mỏng: `scripts/gen_frame_slot.py` (hoặc flag trên tool sinh hiện có) với defaults = P112.
3. Chạy n=120 cùng seed/cách lấy đề như P112 (hoặc so meta đã có): so dùng+tạm nhánh
   `ep_da` vs `out/frame_p112_verdict.json` sau khi chấm mù **hoặc** nếu cùng samples thì
   so bit-identical rate meta `n_thay` / text.

Cổng (1): lệch dùng+tạm ≤ **3đ** so với P112 `ep_da` 51,7% (tức ≥48,7% trên cùng thước),
*hoặc* nếu tái lập deterministic trên cùng parent set thì ≥95% trùng `n_thay` phân bố.

## 3. Giao nộp

Module + CLI + meta so sánh + cập nhật `TK_050` § kết quả P113.
