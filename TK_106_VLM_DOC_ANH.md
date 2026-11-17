# TK_106 — Model của ta ĐỌC ẢNH (VLM, kiểu Gemma 26B/31B)

Mục tiêu: **một checkpoint duy nhất** — chat có thể đính kèm ảnh (ảnh chụp, ảnh chụp màn hình Excel/Word/email/code,
biểu đồ), ảnh và chữ đan xen trong cùng chuỗi, nhiều lượt. Không ghép model ngoài làm "não": phần hiểu + nói tiếng Việt
vẫn là thân 1.3B + expert của ta; chỉ "mắt" (SigLIP2) là bộ mã hoá ảnh mở (tự train mắt cần hàng tỷ cặp ảnh–chữ).

## 1. Kiến trúc (`scripts/vlm_1p3b.py`)

```
Ảnh ─► tách ô 384px (lưới theo tỉ lệ khung, ≤12 ô; ảnh nhiều ô + 1 ảnh toàn cảnh thu nhỏ)
    ─► SigLIP2 so400m-patch14-384 (428M, đóng băng; 27×27 mảnh/ô)
    ─► pixel-shuffle 3×3 (729 → 81 token/ô, 10 368 chiều) ─► MLP (LN → 4096 → 2048, 51M)
    ─► RMSNorm × emb_rms × gain  ⇒ token ảnh CÙNG CỠ token chữ (bài học TK_105: token mềm phình ×3,4 → hỏng)
Chuỗi: "Hỏi: " <ảnh> [toàn cảnh 81] (<hàng> [ô]…)… </ảnh> "\n" câu hỏi "\n" "Đáp: " …   (id −1 = chỗ token ảnh)
Thân 1.3B + expert code/văn phòng/học thuật (đóng băng) + EXPERT THỊ GIÁC (GatedFFN hid 4096, cổng khởi −2, lớp ra = 0)
```

| Ảnh | Lưới (≤12 ô) | Token ảnh |
|---|---|---|
| 384×384 / ảnh nhỏ | 1×1 | 83 |
| 390×844 (điện thoại) | 1×2 | 247 |
| 1440×900 | 3×2 | 571 |
| 1920×1080 | 4×2 (≤16 ô: 5×3) | 733 (1 301) |

Checkpoint: thêm mục `"vision"` (mắt bf16 + cấu hình + bộ chiếu + max_tiles) và `experts["vision"]`.
`hybrid_infer.Runner.prefill/generate(vis=…)` thay chỗ id âm bằng véc-tơ ảnh; UI (`serve_hybrid_ui.py` +
`hybrid_ui.html`): nút 📎 / dán / kéo-thả ảnh (≤4 ảnh/lượt, cạnh dài ≤2048px, PNG giữ nét chữ), nhớ đệm token ảnh theo
sha1, lượt sau vẫn dùng lại trạng thái hội thoại (không chạy lại mắt). Ảnh gốc chỉ giữ trong bộ nhớ trang (localStorage
chỉ lưu ảnh thu nhỏ để hiển thị).

Mắt tải riêng tháp ảnh bằng HTTP Range (`scripts/fetch_vision_tower.py`): 1,71GB thay vì cả file 4,54GB → bf16 0,86GB.

## 2. V0 trên M4 — KẾT QUẢ (2026-10-05)

- **smoke**: token ảnh rms 0,0242 = token chữ 0,0242 ✓; mắt 0,2–1,5s/ảnh (1–9 ô); prefill 744 token + sinh 40 token 4,4s.
  Chưa train → model "bịa" mô tả (xe đạp…) — đúng dự kiến.
- **overfit** (chỉ học bộ chiếu, 8 ảnh tự vẽ chữ + màu nền, 96 bước, ~10s/bước MPS): loss 2,92 → 0,004;
  **8/8 mô tả đúng nguyên văn** ("Ảnh có chữ "Toán" màu trắng trên nền xanh lá.") → gradient chảy đúng qua cả đường ống.
- **UI**: đóng gói 1 checkpoint (7,65GB) → gửi ảnh qua giao diện, trả lời đúng, ~19–21 tok/s, lượt 2 dùng lại trạng thái.
  (Checkpoint V0 đồ chơi đã xoá; cấu hình `hybrid-ui-vlm` trỏ sẵn tới `hybrid_1p3b_vlm_v1.pt`.)

## 3. V1 — dữ liệu (box; thầy Gemma-4-31B FP8 ĐỌC ĐƯỢC ẢNH: `Gemma4ForConditionalGeneration`, có `vision_config`)

| Loại | Nguồn | Kiểm |
|---|---|---|
| **Ảnh chụp màn hình tự dựng** (`scripts/gen_screens.py`) | Excel (bán hàng/lương/kho, ô chọn + công thức), hộp thư (email từ data/office), Word (bài SGK), VS Code (code_tb), biểu đồ cột SVG, app chat (multiturn); nhiều phông/cỡ/sáng-tối/độ rộng | **Đáp án chính xác** (đọc ô, công thức, tổng cột, người gửi, dòng N, giá trị cột…); câu cần hiểu (tóm tắt, soạn trả lời, giải thích code) → `teach`: thầy viết từ CHỮ GỐC |
| Mô tả ảnh tự nhiên | ảnh COCO / LLaVA-Pretrain → Gemma nhìn ảnh viết mô tả tiếng Việt | Gemma tự chấm |
| Hỏi–đáp ảnh Việt Nam | ViVQA, OpenViVQA, UIT-ViIC, KTVIC | đáp án sẵn |
| Biểu đồ/tài liệu | ChartQA, DocVQA (dịch) + biểu đồ tự vẽ | số liệu gốc |
| Chat nhiều lượt có ảnh | Gemma đóng vai người dùng hỏi về ảnh (như vòng thầy v5) | chấm 0–10 + sửa |

Trên box cần: `pip install playwright && playwright install --with-deps chromium` + phông Việt (`fonts-noto`).

## 4. V2 — train (1 H100)

| Giai đoạn | Học | Dữ liệu | Ước |
|---|---|---|---|
| 1 căn chỉnh | bộ chiếu | ~500K ảnh + mô tả | ~4h |
| 2 hỏi–đáp | bộ chiếu + expert thị giác (+ `--tower-last 6` tầng cuối mắt) | ~300K hội thoại có ảnh (màn hình, mô tả, VQA, biểu đồ) | ~15–20h |
| 3 vòng thầy | SFT + DPO trên bản sửa của Gemma (nhìn ảnh) | ~5K lượt | ~6–10h |

Đo: ViVQA, đọc đúng chữ/số trên ảnh màn hình giữ riêng (khớp chuỗi), số liệu biểu đồ, Gemma chấm hội thoại; kèm bộ
chuẩn chữ cũ (SGK, toán, văn phòng, chat 5 lượt) để chắc không quên.

Khi đổi thân (3B, hướng B) chỉ train lại bộ chiếu + expert thị giác (~1 ngày); dữ liệu V1 dùng lại nguyên.

## 5. V1 + V2 — KẾT QUẢ (2026-10-05/06, 1 H100)

**Dữ liệu V1** (`box_run_vlm_v1b.sh`, ~4h, phần lớn là thầy Gemma-31B sinh ~1,5K token/s):
44 193 ảnh màn hình tự dựng (+16 804 câu cần hiểu thầy điền) · 32 170 ảnh tự nhiên thầy nhìn (COCO 20K, KTVIC 2,8K,
OpenViVQA 10,2K) · 24 093 mẫu the_cauldron dịch (chartqa, textvqa, st_vqa, docvqa, scienceqa, ai2d, tabmwp, infographic,
screen2words) · 3 768 biểu đồ Việt → **align 69 203 · sft 128 496 (12 500 chỉ chữ)** · test: nat 1 620, screen 1 000,
chart 424, ovqa 1 070.

**Train V2** (thân + 3 expert đóng băng): giai đoạn 1 (bộ chiếu, 1 081 bước × 64, ~40 phút) val 2,71 → 1,68;
giai đoạn 2 (bộ chiếu + expert thị giác + 6 tầng cuối mắt, 4 015 bước × 32, ngân sách 6K token/lô, ~3,6h) val 1,27 → **0,72**.

**Đo (200 câu/bộ, thầy chấm theo đáp án chuẩn):**

| Bộ | Đúng | Một phần | Khớp giá trị | Ghi chú |
|---|---|---|---|---|
| Biểu đồ Việt (vichart) | **38%** | 8% | 13% | xu hướng/so sánh tốt, số cụ thể kém |
| OpenViVQA (ảnh VN có chữ) | 16% | 3% | 7% | đọc biển hiệu/số sai |
| Ảnh màn hình tự dựng | 7,5% | 8,5% | 12,9% | chép lại giống 0,205; đếm dòng Excel đúng 8/22 |
| Ảnh tự nhiên (mô tả chi tiết) | 0,5% | 19,5% | — | nắm đúng cảnh chính, sai chi tiết (màu, số lượng) → thầy chấm gắt |
| SGK chỉ chữ: có expert thị giác | 40,7% | 18,7% | | **không quên** |
| SGK chỉ chữ: v5 gốc | 39,3% | 19,3% | | |

Chẩn đoán: model **thấy bố cục + chữ to** (tiêu đề biểu đồ, số cột, biển STOP, cảnh chính) nhưng **chưa đọc được chữ/số
nhỏ** (giá trị cột, ô Excel, dòng code → bịa từ trí nhớ). Giả thuyết: (a) thiếu dữ liệu đọc chữ (~60K mẫu × 1 lượt, VLM
thật dùng hàng triệu), (b) nén 3×3 mảnh/token (42×42px) làm mất nét chữ nhỏ. → V3 chẩn đoán: +40K ảnh chữ tổng hợp
(`gen_ocr.py`, PIL) trộn 50/50, train tiếp 1 500 bước, đo lại.

**Bẫy đã gặp (đã vá vào kịch bản):**
- Container chỉ 1 792 luồng (`pids.max`): 25 Chrome cùng vLLM → vLLM sập "Resource temporarily unavailable", SSH chập chờn.
- HF tải 1 luồng ~1,5MB/s; tải 4 nguồn song song nhanh hơn hẳn (COCO 60 ảnh/s).
- 2 tiến trình dịch chồng nhau → trùng 576 mẫu → `build` khử trùng theo id.
- Hết ổ khi lưu bước 500 (ổ 120GB, checkpoint trợ lý 6,6GB + file thử) → chỉ lưu tầng mắt được học, ghi `.tmp` rồi đổi tên.
- vLLM 0,45 bộ nhớ không đủ chỗ KV cho Gemma-31B (cần ≥6,9GB) → đo tách 2 pha: sinh (6 tiến trình, tắt vLLM) rồi chấm.
- `--device` mặc định mps/cpu → trên box phải tự chọn cuda.

**M4:** `checkpoints/hf_hybrid/hybrid_1p3b_vlm_v1.pt` (7,85GB, 4 expert) · UI `hybrid-ui-vlm` cổng 8796 (~19 tok/s;
chờ 4–6s cho ảnh nhiều ô).

## 6. V3 chẩn đoán + V4 sửa kiến trúc — KẾT QUẢ (2026-10-06)

**V3** (+40K ảnh chữ tổng hợp `gen_ocr.py`, train tiếp 1 500 bước, kiến trúc V1): ảnh màn hình 7,5→9%, khớp giá trị 12,9→12,9%,
chép lại 0,205→0,178 — **thêm dữ liệu KHÔNG giúp** → nút thắt là kiến trúc (model nhận ra chủ đề rồi viết theo trí nhớ).

**V4** = gộp **2×2** (đệm 27→28 → 196 token/ô, thay 81) · bỏ ảnh toàn cảnh khi nhiều ô · ≤8 ô (1080p → 4×2 = 1 572 token) ·
**LoRA r64 trên wq/wk/wv/wo của 6 khối attention** (thân học CHÉP từ token ảnh) · dữ liệu V1 + 40K ảnh chữ (168 496 mẫu).
Giai đoạn 1: val 2,71→1,58 (V1 1,68) · giai đoạn 2 (3 500 bước × 32, ~4,2h): val 1,40→0,61.

| Bộ (200 câu) | V1 | **V4** |
|---|---|---|
| Ảnh màn hình · đúng | 7,5% | **38,5%** |
| · khớp giá trị | 12,9% | **41,9%** |
| · chép lại giống | 0,205 | **0,557** |
| OpenViVQA · đúng | 16% | **28,5%** |
| Ảnh tự nhiên · đúng / một phần | 0,5 / 19,5% | 1 / **65%** |
| Biểu đồ Việt · đúng | 38% | 32% |
| SGK chữ (150): LoRA + expert / chỉ expert / v5 gốc | 40,7 (V1) | 36,0 / 38,0 / 39,3 |

Theo loại ảnh màn hình (đúng/tổng, V1 → V4): email 2→19/29 · chat 0→10/18 · văn bản 1→6/15 · Excel 12→23/52 · biểu đồ 0→13/55 ·
code 0→6/31. Còn yếu: số nhỏ trên cột biểu đồ, giá trị ô Excel cụ thể, dòng N của code.

**Một checkpoint, không quên chữ:** `hybrid_1p3b_vlm_v4.pt` giữ LoRA RIÊNG (`vision.lora`, không gộp vào thân) + expert thị giác bọc
`Toggle` → UI chỉ bật cả hai khi hội thoại CÓ ẢNH; chat chữ chạy y hệt v5 (định luật Ôm, Điện Biên Phủ đúng như v5).
Bẫy: đặt `use_gate` PHẢI trước khi bọc Toggle (đặt lên lớp bọc → expert chạy không cổng → ra rác).
