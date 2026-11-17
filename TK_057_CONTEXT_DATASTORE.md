# TK_057 — CONTEXT DATASTORE: kho (h, next) cho kNN-LM

| Field | Value |
|-------|-------|
| **ID** | TK_057 | **Status** | 🟡 MỞ — nền cho TK_056. |
| **Liên quan** | TK_056 (dùng), TK_055 §190 (encoder), TK_042 G13. |

## Việc
- Quét corpus (sạch, TK_055 §193): mỗi vị trí `i` → key `h[i]` (encoder LEFT-context,
  causal-mask một forward/câu) + value `next_token = tokens[i+1]`.
- Lưu sidecar `<ckpt>.datastore.npz`: `keys (N,d) float16`, `vals (N,) int32`.
- Index ANN (FAISS/hnswlib nếu có; numpy brute-force cho POC nhỏ). Nén PQ nếu N lớn.
- N ước lượng: ~vài triệu vị trí (450k câu × ~10 token). POC: subset 100–300k.

## Cổng
- kNN theo `h` truy đúng "ngữ cảnh giống" (đọc mắt láng giềng hợp lý).
- Thời gian query < vài ms/bước (ANN); build offline một lần.
