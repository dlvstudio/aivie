# TK_062 — CHỐNG-DRIFT cho bàn giao span (causal-LM nối đuôi)

## Vấn đề
Đuôi do causal-LM nối (TK_061-v2) mượt nhưng LỆCH SỰ THẬT (sắt→"chất độc") vì LM
nhỏ không biết fact. Cần neo đuôi về đúng chủ đề/thực tế.

## Cơ chế (đã dựng trong `KnnLM.lm_extend`)
- **kNN-grounding** `knn_ground`: mỗi bước trộn `P=(1−g)·P_LM+g·P_kNN` (P_kNN từ
  datastore) — kéo về chuỗi THẬT trong corpus (kNN-LM Khandelwal). 1 forward/bước.
- **Topic anchor** `anchor`: thiên lệch `exp(a·cos(token, centroid))`, centroid =
  embedding các token NỘI-DUNG của prompt (đã lọc structural).

## Đo (base d=256)
| prompt | baseline | anchor a3 | g=0.4–0.5 |
|---|---|---|---|
| «sắt và» | "chất lỏng" (lệch) | **"kim loại/oxide"** ✓ | "kim loại" ✓ |
| «cây cối cần ánh mắt» | "trồng cây" (ổn) | thơ ca (lệch) ✗ | "cây trồng vườn" ✓ |
| «con hổ là **Xí**» | rác | rác | rác (đầu W_skip đã hỏng) |
| «bầu trời» (dài, tốt) | nối thêm ⇒ rác | — | — (server: cổng độ-dài chặn) |

## Kết luận (honest)
Anti-drift giúp **không nhất quán**: anchor tốt khi chủ đề RÕ (sắt→kim loại),
HẠI khi content mơ hồ (ánh mắt→thơ). kNN-ground g=0.4 an toàn hơn, lợi vừa phải.
Giới hạn gốc KHÔNG phải cơ chế mà là (a) LM nhỏ d=256; (b) đôi khi W_skip khởi
đầu đã hỏng ("con hổ là Xí") → đầu hỏng thì đuôi vô phương.

**Base lớn hơn (d=384) sẽ giúp nhất — nhưng KẸT trên phần cứng này** (MPS/bộ nhớ
hợp nhất, xem TK_061): train d=384 đứng im, cả khi RAM trống 13GB. d=256 là trần
thực tế của máy.

**Chốt production:** `lm_extend` mặc định `knn_ground=0.4, anchor=0.0` (grounding
nhẹ an toàn; anchor opt-in vì thất thường). Cổng độ-dài (chỉ nối khi W_skip cụt)
bảo vệ output tốt. `extend` vẫn opt-in (`/api/gen?extend=1`).

- [x] kNN-ground + anchor content-word; [x] đo; [x] config an toàn g=0.4.
- [ ] (chặn) base LM lớn hơn cần phần cứng khác (GPU rời/CUDA) — ngoài tầm máy này.
