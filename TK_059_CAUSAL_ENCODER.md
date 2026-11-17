# TK_059 — CAUSAL ENCODER: decoder nhỏ next-token, trộn logits

| Field | Value |
|-------|-------|
| **ID** | TK_059 | **Status** | 🟡 MỞ — xa hơn TK_056, rời Luật-0 nhất. |

## Việc
Encoder hiện masked (bidirectional). Train một DECODER nhỏ CAUSAL (next-token) trên
corpus sạch → logits trộn với W_skip vote: `P = λ·softmax(logits) + (1−λ)·P_Wskip`.
Gần transformer-LM thật nhất; footprint/không-diễn-giải cao hơn. Chỉ làm nếu TK_056
(kNN-LM) chưa đủ chất lượng.

## Bằng chứng thúc đẩy (từ TK_056)

Nới candidate set kNN-LM đã dựng xong nhưng ĐO ĐƯỢC vô ích vì encoder masked-
train cho `h` (ngữ cảnh 2-token) truy láng-giềng bị tần suất chi phối + nhiễu
chéo miền (top-kNN «sông và»→lọ/xe/bồn; «chó là»→một). Causal encoder ở ticket
này là điều kiện để nới candidate set (và kNN-LM nói chung) sinh lợi rõ.

## Kết quả (ĐẠT) — causal encoder giảm ~1 bit, phá tường hư-từ

**Train:** `scripts/train_ctx_encoder_causal.py` — next-token, mask nhân quả,
weight-tied logits, d=128/3 tầng/4 head, 8000 bước, corpus sạch (423k câu).
Held-out bits/token (LM trực tiếp): **40.4 → 6.78**. Lưu `ctx_encoder_causal.pt`
(KHÔNG đè ctx_encoder.pt — QA vẫn dùng masked bidirectional). Kiến trúc tương
thích `ctx_encoder._Enc` → nạp drop-in, `causal_last` chạy đúng.

**Datastore causal:** `dung_datastore.py --encoder ctx_encoder_causal.pt` (đọc
cfg d/tầng từ checkpoint) → 970.882 vị trí d=128 (`...datastore_causal.npz`, 248MB).

**Top-kNN (soi trực tiếp) — nội dung nổi lên:**
- «nước sôi ở»: masked 1/nước/nhiệt → causal **vùng/phía/nông/bãi/chân** (học "ở+địa điểm")
- «cây và»: masked **là·0.20** → causal là·0.04 + **hoa/vườn/lá**
- «chó là»: masked một·0.19 → causal **vật/động** ("động vật")
- Đỉnh hư-từ "một"/"là" SỤP; nhiễu chéo miền "lọ/bồn/xe/laser" BIẾN MẤT.

**kNN-LM bits/token held-out (2500 vị trí NGOÀI datastore):**

| | λ=0.3 | λ=0.5 |
|---|---|---|
| MASKED | 8.509 | 8.438 |
| **CAUSAL** | **7.596** | **7.388** |

Causal hơn masked **~1.05 bits** → perplexity **347 → 168 (≈½)**.

**Wire production:** `get_knn()` ưu tiên encoder+datastore causal (rơi về masked
nếu thiếu file). QA giữ masked bidirectional. Hook thêm `widen_k` (nới opt-in).

**Ghi chú:** sinh prompt 2-token phẳng («sông và»→"các") KHÔNG đổi vì bị argmax
W_skip-bigram át — thước đơn-token không phản ánh cải thiện 1-bit tổng thể. Lợi
ích ở ngữ cảnh DÀI + độ trôi chảy tổng thể; và là nền để nới/λ-cao sinh lợi.
