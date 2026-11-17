# RUNBOOK M4 — bake span TỪ (WS-CRF) ra sidecar

> Giao 2026-08-11. Chạy **một lần**, dùng lại mãi.
> Đọc `TK_046_ANALOGY_FRAME_TENSOR.md` §18.H · commit `ebbad25`, `fb5883d`.

## ⚠️ VÒNG 2 — BAKE LẠI sau khi vá bộ lọc span

Vòng 1 (`252b750`) đã chạy: 78 563 span · 1,5 phút · rác **0,0%** · bỏ 283 698 cha.
Ba cổng số đều đạt. **Nhưng đọc tay 12 mẫu thì con số 0,0% là sai** —
`'Hỏi: Tàu'` (mốc khung dính từ) · `'tại Gia Lý Học Viện'` (giới từ dẫn đầu) ·
`'René Lac os te'` (mảnh BPE tên ngoại lai). Thật ra ~35% không sạch.

Đã vá hai loại đầu (loại ba **không** vá — xem §5):

| lỗi | vì sao hai vá ĐẦU của tôi trượt | vá đúng |
|---|---|---|
| `Hỏi: Tàu` | `is_structural_token` không bắt (`Hỏi:` có `is_surface=True`), `_FUNC` cũng không | bỏ span có token kết bằng `:` — kiểm CẤU TRÚC, không phải danh sách từ |
| `tại Gia Lý Học Viện` | `tại` **không có** trong `_FUNC` (đã đo) | **TỈA** token dẫn đầu khi gidf thấp hơn max span ≥1,0 (tại 4,47 « Gia 6,06) — tỉa chứ không bỏ cả span |

Smoke mini (600 cha): rác tự động 0,0% → **4,7%**; mẫu `Chủ tịch`·`Việt Nam`·
`nghênh chiến`. Số **tăng** vì thước nay nghiêm hơn, không phải vì chất lượng tụt.

⇒ **Chạy lại y hệt lệnh §1.** Sidecar cũ sẽ bị ghi đè.

## 0. Vì sao

Năm vòng đo chuỗi ba bước đều tắc ở cùng một chỗ: **đơn vị neo sai**.

Cụm ISG là n-gram theo **tần suất**, không phải từ. Đo trên `mini_sent_ep_qa.pkl`
(4 234 cụm ≥2 token): **2 427/4 234 = 57% không phải đơn vị ngôn ngữ**
(dấu câu/EOS 28% · mở đầu bằng hư từ 15% · kết bằng hư từ 7% · số 7%) —
`là Nguyễn`(power 33) · `của phở` · `đá và` · `, tỉnh Nghệ An . <EOS>`.

Hệ quả: neo nhãn vào đó cho ra `tụ vĩ` (từ «lãnh **tụ** **vĩ** đại»), `Bốn đài`,
`Các tôn`, `tình` (từ «**tình** thế»). Mỗi vòng vá lại chọn một span sai KHÁC.

WS-CRF (TK_034, **F1 0,9844**) tách đúng cả năm câu đang hỏng:

```
Hồ Chí Minh | chính | là | lãnh tụ | vĩ đại
Bốn | đài truyền hình | lớn | nhất | tại | Hoa Kỳ | là | NBC CBS ABC | và | Fox
Các | tôn giáo | khác | chẳng hạn | như | Hồi giáo | và | Do Thái giáo
vũng | lầy | là | một | tình thế | khó khăn
```

**KHÔNG train lại gì.** Model WS-CRF đã train xong (95k câu, 2026-07-08) và
`frame_slot_gen` đã dùng nó trên đường ship. Span là **view dẫn xuất** từ
`emit_ids` — cùng họ W_skip/E_skip, TK_045 §0.W A.3: *"Cần train mới → KHÔNG"*.

## 1. Chạy

```bash
PYTHONPATH=src .venv/bin/python -m pytest -m "not slow" -q   # kỳ vọng 418 passed

PYTHONPATH=src .venv/bin/python scripts/bake_ws_spans.py \
  --checkpoint checkpoints/network_v12_subst_isg_pw.pkl
```

Ghi ra `checkpoints/network_v12_subst_isg_pw.ws_spans.json` +
`out/bake_ws_spans_network_v12_subst_isg_pw.log` (có tiến trình + RSS mỗi 20k cha).

Nếu muốn cả `network_v12_subst.pkl` thì chạy thêm một lần với `--checkpoint` đó.

## 2. Thời gian — con số M4 cần báo lại

Đo trên mini:

| chế độ | tốc độ |
|---|---|
| chạy TẤT cả cha | 40 ms/cha ⇒ 430k cha ≈ **4,8 giờ** |
| **chỉ cha CÓ pivot** (mặc định) | 0,6 s / 400 cha trên mini |

Lọc pivot là đòn cắt chính: chỉ cha chứa `là`/`như`/`gồm`/`thành`/`gọi` mới sinh
được cạnh, phần còn lại chạy WS-CRF là phí.

⚠️ **Tỉ lệ cha có pivot ở v12 chưa đo được** (máy dev không nạp nổi ckpt). Đó là
con số M4 báo về: log in `bỏ=<n>` mỗi 20k cha. Nếu tổng thời gian vượt **90 phút**
thì dừng và báo lại — đừng để chạy qua đêm rồi mới nói.

`--all-parents` để tắt lọc (đối chứng, **không** dùng cho bake thật).

## 3. Cổng — kiểm ngay trong log

| # | điều kiện | nếu trượt |
|---|---|---|
| 1 | roundtrip sidecar OK (script tự assert) | file hỏng ⇒ báo, đừng dùng |
| 2 | `span nghi rác (tự động)` ≤ **20%** | (mini vòng 2: 4,7%; ISG: 57%). >20% ⇒ ánh xạ token→âm tiết sai trên v12, báo kèm mẫu |
| 3 | số span ≥ 10 000 | quá ít ⇒ lọc pivot cắt nhầm, thử `--all-parents` trên 5k cha để so |

Script nay in **30 mẫu** rải đều — **đọc tay chúng**. Vòng 1 đã chứng minh cổng
số không đủ: nó báo 0,0% trong khi ~35% mẫu không sạch. Mini cho
`quan lại` · `hành chính` · `chiến lược` · `độc lập` · `Cửu Chân`. Nếu v12 cho
toàn mảnh vụn thì cổng (2) không bắt được, phải nói ra.

## 4. Giao nộp

Commit `out/bake_ws_spans_*.log` + **sidecar `.ws_spans.json` nếu ≤50MB**;
nếu lớn hơn thì ghi rõ đường dẫn local trong commit message (như quy ước
`RUNBOOK_TK_049_P106.md` §3 với ckpt).

Commit message: số span · thời gian · `span nghi rác` % · số cha bị lọc bỏ.
**Không** nhận xét chất lượng gì thêm.

## 5. Không làm

- Không train lại WS-CRF (đã có, F1 0,9844) — cũng không train lại mạng.
- Không chạy `--all-parents` cho bake thật (4,8 giờ, không cần).
- Không chấm mù ở vòng này — bake xong mới tới lượt đo.
- Không wire `hub_spread` vào `generate`.
- **Không vá mảnh BPE tên ngoại lai** (`René Lac os te`, `Bat ư`). Đó là hệ quả
  BPE cắt từ ngoại lai rồi WS-CRF gộp lại; ép sửa là thêm một vòng vá nữa.
  Ghi nhận trong báo cáo, đừng sửa.

## 6. Sau bake thì làm gì

Bake xong mới đo lại chuỗi ba bước với đơn vị mới
(`sample_hub_chain3_blind.py`, cổng G-chain3 cũ: ≥35% wire · <15% đóng).
Cổng vòng 2 hiện là **5,9%** — chưa biết đổi đơn vị có nhích được không, đó
chính là câu hỏi.

⚠️ WS-CRF **không** sửa kiểu hỏng 3: bước ① vẫn dùng đồng hiện d≤6 thay vì quan
hệ "mang thuộc tính" (`«Ireland» → tập, trung, canh, tác`). Đừng kỳ vọng nó tự
khỏi.
