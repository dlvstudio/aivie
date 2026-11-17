# TK_060 — EVAL PERPLEXITY: thước bits/token held-out

| Field | Value |
|-------|-------|
| **ID** | TK_060 | **Status** | 🟡 MỞ — nền đo cho TK_056/059. TK_042 §0.1 tiếc "chưa đo bits/byte". |

## Việc
Thước chuẩn để đo TIẾN THẬT (không chỉ mắt):
- Held-out (viwiki ngoài manifest / split câu): mỗi vị trí `(ctx, target)`.
- Đo `bits = -log2 P(target)` cho: W_skip vote đơn · kNN-LM (TK_056) · nội suy.
- Báo: bits/token TB, perplexity = 2^bits, lift@1 (P(target) so unigram), MRR.
- So baseline: unigram, bigram/W_seq, PPM (TK_031), W_skip.

## Cổng dùng cho TK_056
kNN-LM nội suy phải GIẢM bits/token so W_skip đơn — đó là bằng chứng phá tường đếm.
