# Báo cáo: model aivie 2,5B — mốc 20 tỷ token (TK_107)

*Ngày 12/10/2026 · số liệu đo trên box P2 (4×H100) và Mac M4 · log gốc: `out/box_results/p2/`*

## 1. Tóm tắt

- Thân model được **nuôi lớn từ 1.3B lên 2,49B tham số** (48 → 96 khối, 12 lớp attention, ngữ cảnh 2K → **8K**, RoPE 1e6), train tiếp **20 tỷ token** trên 4×H100 trong khoảng 5 ngày (pha A1 → A2 → B).
- So với 1.3B: **tốt hơn ở mọi nhóm dữ liệu**, nhất là code, toán, tiếng Anh; **đọc được ngữ cảnh dài 4K–8K** (1.3B hoàn toàn không).
- Còn yếu đúng như model nền chưa SFT: **bịa sự kiện, toán sai, không biết dừng**. SFT + 3 expert (P3) sẽ xử lý.
- Quyết định: **dừng ở 20 tỷ**, sang P3. Bản để kéo dài lên 50/100 tỷ đã lưu công khai trên HF.

## 2. Đo trên tập kiểm tra cố định (bits/token — càng thấp càng tốt)

Ngữ cảnh 2K, cùng cửa sổ dữ liệu cho mọi checkpoint:

| Nhóm dữ liệu | Thân vừa chèn khối | Sau A1 | Sau A2 | **Sau B (20 tỷ)** |
|---|---|---|---|---|
| Web tiếng Việt | 3,05 | 2,73 | 2,62 | **2,21** |
| Văn bản dài | 2,43 | 2,27 | 2,16 | **2,06** |
| Wikipedia Việt | 4,97 | 4,44 | 4,44 | **3,92** |
| Tiếng Anh giáo dục | 3,69 | 2,87 | 2,80 | **2,51** |
| Toán | 3,77 | 2,41 | 2,33 | **1,88** |
| Code | 3,69 | 1,76 | 1,70 | **0,98** |
| Dữ liệu pretrain cũ (ôn lại) | 2,67 | 2,65 | 2,64 | **2,61** |
| **Trung bình** | 3,47 | 2,73 | 2,67 | **2,31** |

Ngữ cảnh 8K: A2 2,60 → **B 2,22** (trung bình các nhóm).

## 3. Ngữ cảnh dài

| | 1.3B gốc (2K, RoPE 1e4) | **2,5B mới (8K, RoPE 1e6)** |
|---|---|---|
| Loss đoạn 4K–8K của văn bản 8K | 3,66 (vỡ ngoài 2K) | **1,54** |
| Loss đoạn 0–1K (văn bản 1K) | 2,23 | **1,62** |
| Tìm "kim" (thông tin giấu trong văn bản) @1K | 21/24 | 18/24 |
| @2K | 15/24 | 16/24 |
| @4K | **0/24** | **20/24** |
| @8K | **0/24** | **14/24** |

→ Model mới đọc được tài liệu dài 4K–8K. Ở 8K mới 14/24 — chưa chắc chắn, SFT có thêm 6.000 mẫu dài để luyện.

## 4. Thử model nền: 30 câu viết tiếp + 44 câu trắc nghiệm

Cả hai model đều là bản NỀN (chưa SFT), nhiệt độ 0,7, tối đa 120 token. Chi tiết: `probe_base_cpt_b.md`, `probe_base_1p3b.md`.

**Trắc nghiệm kiến thức: 2,5B 38/44 · 1.3B 36/44** (dễ 22/24 vs 20/24 · khó 16/20 vs 16/20).

### Tiến bộ rõ

| Loại | 2,5B mới | 1.3B cũ |
|---|---|---|
| Dịch ("I love my family") | "Tôi yêu gia đình tôi." rồi tự viết thêm các cặp dịch đúng | lặp một câu mãi |
| Tiếng Anh (quang hợp) | "convert carbon dioxide and water into sugars and oxygen" — đúng | sai kiến thức |
| Code Python / JS | đúng dạng code (vòng lặp, `append`, `return`) | chữ vô nghĩa, lẫn quảng cáo game |
| Truyện Kiều | viết tiếp đúng nguyên văn 8 câu | lạc sang bình luận thơ |
| Wiki (Mê Kông, Hồ Chí Minh, Hạ Long) | trôi chảy, nhiều số liệu đúng | khá trôi |
| Hướng dẫn / sức khoẻ / pháp luật | như văn bản thật (công thức phở, triệu chứng sốt xuất huyết đúng) | — |

### Còn yếu

| Lỗi | Ví dụ |
|---|---|
| Toán sai | 36 + 47 = **69** (đúng 83) · 2x + 6 = 0 → **−3/2** (đúng −3) |
| Bịa sự kiện | "Việt Nam thắng Thái Lan 3–0, hat-trick của Quang Hải, Thành Lương, Văn Toàn" |
| Logic code sai | `isPrime` trả `true` khi tìm được ước (ngược) |
| Nhớ chi tiết trong đoạn | đề nói con lớn nhất là Hoa 12 tuổi → model viết "Minh, 20 tuổi" (1.3B trả lời đúng) |
| Không biết dừng | "Thủ đô của Nhật Bản là gì?" → đúng Tokyo nhưng lặp cả chục lần |

Thử trong UI (M4, cổng 8797): "Giới thiệu ngắn về Vịnh Hạ Long" → viết như bài bách khoa, diện tích 1.553 km², 1.969 hòn đảo (khớp số liệu phổ biến), có đánh số mục và không dừng — đúng tính chất model nền.

## 5. Dữ liệu đọc ảnh (dựng lại)

Dữ liệu V1–V5 cũ đã mất cùng box TK_106 (không được sao lưu) → dựng lại trên box P2, **đẩy công khai** `donaldluong/nn-aivie-vlm-data` (~30GB, có README nguồn/giấy phép):

| Phần | Số lượng |
|---|---|
| Ảnh đời thường thầy Gemma nhìn (COCO, KTVIC, OpenViVQA) | 36.251 |
| Hỏi–đáp biểu đồ/tài liệu dịch sang tiếng Việt (the_cauldron) | 31.802 |
| Ảnh màn hình (Excel, email, văn bản, biểu đồ, chat) | 45.600 + 40.000 kiểu đọc-trước |
| **Ảnh code mới**: code cài lỗi, trình soạn + terminal báo lỗi đúng dòng, ảnh chụp màn hình | 20.000 train + 3.000 cho vòng thầy + 300 test |
| Trang báo giả (báo điện tử, báo in, ảnh chụp báo) | 40.600 |
| Ảnh chữ (đoạn văn, hoá đơn nhiều số) | 120.000 |
| **Tổng gộp cho train V5 + code** | **239.988 mẫu** |

## 6. Kế hoạch P3 (thứ tự mới)

| Bước | Việc | Ước tính (1×H100) |
|---|---|---|
| P3.1 | SFT: 125.330 mẫu (đủ mọi đợt SFT của 1.3B + 6.000 mẫu dài 8K), 2 epoch | ~14,5 giờ |
| P3.2 | 3 expert mới (code / văn phòng / học thuật), 12 điểm cắm | ~4–6 giờ |
| P3.5 | Chấm điểm so với v5 (eval_v1, easy20, văn phòng, SGK, toán, ngữ cảnh dài) | ~2–3 giờ |
| P3.4 | Train đọc ảnh trên thân mới | ~6–8 giờ |
| P3.3' | **Vòng thầy chung chữ + ảnh** (thầy chấm–sửa cả đọc ảnh, ưu tiên đọc code trong ảnh) | ~12–16 giờ |

**Trạng thái:** SFT đã chạy được ~600/7.676 bước (val loss 2,01 → 0,84) thì **box P3 offline** (sự cố máy chủ Vast <IP>) → mất ~2 giờ. Đã sửa: SFT **sao lưu HF mỗi 500 bước và tự chạy tiếp** khi box chết (thử trên Mac đạt). **Đang chờ thuê lại 1×H100** (ổ ≥250GB, máy chủ khác).

## 7. Checkpoint & dữ liệu trên HF

| Kho | Nội dung | Chế độ |
|---|---|---|
| `nn-aivie-hybrid-2p6b-ckpt-b` | `cpt_b.pt` — bản cuối mốc 20 tỷ (đã hạ lr), dùng cho SFT | riêng tư |
| `nn-aivie-hybrid-2p6b-stable` | `cpt_b_stable.pt` — bước 8582, trước hạ lr, kèm optimizer → để kéo dài train | công khai |
| `nn-aivie-p3-data` | dữ liệu SFT + expert + bộ chấm | công khai |
| `nn-aivie-vlm-data` | toàn bộ dữ liệu đọc ảnh | công khai |

Trên Mac: `checkpoints/hf_hybrid/hybrid_2p6b_b9536_bf16.pt` (4,98GB, bản nền mốc 20 tỷ để thử).
