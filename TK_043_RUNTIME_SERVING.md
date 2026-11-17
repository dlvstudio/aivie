# TK_043 — RUNTIME & SINH NHANH: chạy được thật, sinh nhiều token/bước, bộ nhớ nhỏ

| Field | Value |
|-------|-------|
| **ID** | TK_043 |
| **Status** | 📋 SPEC (2026-07-19) — KV cache ĐÃ CODE (TK_042 §7.22); phần còn lại spec + điều kiện phân định, chưa chạy. |
| **Nguồn** | Câu hỏi tác giả 2026-07-19: *"có cách nào để mạng chạy runtime được không? hoặc tốc độ sinh cực nhanh, hoặc sinh được nhiều từ cùng lúc?"* |
| **Chốt thiết kế** | (1) **Tốc độ ≠ chất lượng** — ticket này KHÔNG cải thiện chất lượng sinh (đó là TK_042); nó làm cái đã có chạy NHANH + NHẸ, đầu ra GIỮ NGUYÊN. (2) **Kiến trúc hai tầng**: đồ thị = bộ nhớ (mmap/CSR, CPU), LM = tính toán (lượng tử hoá). (3) **Đúng-đắn TRƯỚC tốc độ**: mọi tối ưu phải chứng minh đầu ra không đổi (hoặc, với speculative, phân bố không đổi) TRƯỚC khi khoe tok/s. |
| **Liên quan** | TK_042 §7.22 (KV cache), §7.13/§0.5 (verbatim copy — ở đây thành ĐÒN BẨY), §7.15 (kNN-LM); **TK_040** (CSR nén đồ thị — phụ thuộc cứng cho runtime máy nhỏ); [[checkpoint-load-gc]] (gc.disable khi nạp) |
| **Nút chặn đã biết** | Đồ thị `network_v10.pkl` 2GB → ~20GB RAM dạng dict-Python → **máy Intel 8GB KHÔNG nạp nổi**. Runtime trên máy nhỏ **bị chặn bởi TK_040** (CSR) trước cả chuyện tăng tốc sinh. |

---

## 0. ĐO TRƯỚC (measure-first — BƯỚC 1 BẮT BUỘC trước khi tin bất kỳ tối ưu nào)

> Kỷ luật README Mục 0. Con số hiện có (đo thật, 2026-07):

| thứ | trạng thái đo được | nguồn |
|---|---|---|
| decode LM không cache | mỗi token tính LẠI O(ctx) — lặp 99% việc | `NeuralLM.probs` recompute cả cửa sổ |
| nạp đồ thị 2GB | ~60s (có `gc.disable`) / 73+ phút (không) | [[checkpoint-load-gc]] |
| RAM đồ thị full | ~20GB dict-Python | TK_040 M0 |
| tok/s LM hiện tại | **CHƯA ĐO** — cần probe_kv_cache chạy trên M4 | §2.1 |
| tỉ lệ accept khi đồ-thị-nháp | **CHƯA ĐO** — quyết định speculative có đáng không | §2.2 |

⚠️ **Hai ô "CHƯA ĐO" là bản lề.** tok/s quyết KV cache đáng bao nhiêu; tỉ lệ accept quyết
speculative decoding sống hay chết. KHÔNG code speculative trước khi đo accept (§2.2 M-accept).

---

## 1. KIẾN TRÚC MỤC TIÊU — hệ serving hai tầng

```
                       ┌──────────────────────────────────────────┐
   prompt ──────────►  │  DRAFTER (rẻ, nháp K token)               │
                       │  · đồ thị: verbatim continuation (lookup) │
                       │  · HOẶC LM nhỏ 11.4M (forward rẻ)         │
                       └───────────────┬──────────────────────────┘
                                       │ K token nháp
                       ┌───────────────▼──────────────────────────┐
                       │  VERIFIER (LM lớn v2, KV cache + int8)     │
                       │  1 forward pass verify CẢ K token         │
                       │  accept prefix đúng · sinh 1 token ở chỗ  │
                       │  sai · phân bố = LM lớn ĐƠN (chứng minh)   │
                       └──────────────────────────────────────────┘

  Bộ nhớ:  đồ thị = CSR mmap (TK_040) trên CPU  ·  LM = int8/int4 trên MPS/GPU
```

Ba trục tăng tốc **độc lập**, ghép được:
* **latency** (§2.1 KV cache) — mỗi token ra nhanh
* **nhiều token/bước** (§2.2 speculative) — 1 lần tính LM lớn → nhiều chữ
* **throughput** (§2.3 batch) — nhiều prompt song song
* **bộ nhớ** (§2.4 lượng tử + §2.5 CSR) — chạy máy nhỏ

---

## 2. CÁC THÀNH PHẦN (mỗi cái: cơ chế · điều kiện phân định GHI TRƯỚC · rủi ro)

### 2.1. KV cache — ✅ ĐÃ CODE (TK_042 §7.22)

**Cơ chế:** prefill prompt một lần, decode O(1)/token thay vì O(ctx). Chốt: RoPE xoay theo vị
trí TUYỆT ĐỐI (`rope(x,T,offset)`, token decode ở `past_len` không phải 0 — sai = lỗi im lặng
TK_041 §5.2). `NeuralLM.generate_cached`, `forward_cached`.

**M-speed (điều kiện phân định):** `probe_kv_cache.py` — (1) greedy có-cache == không-cache
TỪNG TOKEN (lệch ⇒ FAIL, sửa trước); (2) chỉ khi giống hệt mới đo tok/s. Kỳ vọng **5-20×**.
Chạy được NGAY (checkpoint `lm_v2_42m.pt` đã có). **Việc còn lại: chạy trên M4.**

**Rủi ro:** đã lường (RoPE offset). Nếu tok/s < 3× ⇒ nghi MPS overhead / batch=1 quá nhỏ —
đo lại với prompt dài hơn.

### 2.2. Speculative decoding — NGÔI SAO của ticket (đồ-thị-nháp / LM-verify)

**Vì sao hợp với TA:** ta ĐÃ có cặp draft/verify tự nhiên. Và điểm đẹp: **verbatim-copy —
nhược điểm CHẤT LƯỢNG (§7.13, §0.5: khảm 5-gram từ corpus) — thành ĐÒN BẨY TỐC ĐỘ ở đây.**
Đúng chỗ đồ thị chép đúng, drafter đoán đúng ⇒ LM lớn accept nhiều token/pass.

**Thuật toán (Leviathan 2023 / Chen 2023 — đầu ra = LM lớn ĐƠN, chứng minh được):**
```
lặp tới khi đủ n token:
  1. DRAFT: drafter đề xuất K token x_{t+1..t+K}
     · đồ thị: nếu prompt khớp một cha, lấy K token TIẾP của cha đó (lookup, ~0 chi phí)
     · hoặc LM nhỏ 11.4M: K bước forward rẻ
  2. VERIFY: LM lớn tính p(·|ctx) cho CẢ K vị trí trong 1 forward (KV cache, nối K token nháp)
  3. ACCEPT/REJECT:
     · greedy: accept token i nếu argmax p_i == x_i; dừng ở token sai ĐẦU TIÊN
     · sampling: accept prob min(1, p_i/q_i); reject → lấy mẫu từ (p−q)_+ rồi dừng
  4. sinh token ở chỗ reject bằng chính p ⇒ mỗi vòng nhả (số-token-accept + 1)
```
Tăng tốc ≈ **số token accept trung bình/vòng**. Copy regime: cao. Ngoài đó: thấp.

**M-accept (điều kiện phân định GHI TRƯỚC — đo TRƯỚC khi code đầy đủ):**
> Với 50 prompt, đo **số token drafter-đồ-thị đoán trùng argmax LM lớn** trước khi lệch.
> * TB ≥ **2.5 token/vòng** ⇒ speculative ĐÁNG — tăng tốc ~2×+ ở copy regime, code tiếp.
> * TB ≈ **1** ⇒ đồ thị nháp ít trùng LM (phân bố khác nhau — đồ thị chép, LM học) ⇒
>   **âm tính**: đổi drafter sang LM nhỏ 11.4M (cùng họ phân bố → accept cao hơn), đo lại.
> * cả hai drafter đều ≈1 ⇒ KHÉP speculative, chỉ giữ KV cache. Ghi thật.

Đây là phép đo RẺ (chỉ cần LM lớn + đồ thị/LM nhỏ, không cần cài đặt accept/reject đầy đủ) —
làm TRƯỚC, quyết định có xây tiếp không. Chống xây cả hệ speculative rồi mới biết accept=1.

**Rủi ro:** (1) drafter và verifier PHẢI cùng tokenizer (đều `bpe_v1` ✓). (2) đồ thị phân bố
khác LM ⇒ accept có thể thấp — đó CHÍNH LÀ cái M-accept đo. (3) cài đặt accept/reject sai ⇒
đầu ra lệch phân bố LM lớn — phải có test đối chứng (sample nhiều seed, so phân bố với LM đơn).

### 2.3. Batched / continuous batching — throughput

**Cơ chế:** decode nhiều prompt cùng lúc (batch chiều 0). `generate_cached` hiện batch=1;
mở rộng cache thành (B, nh, T, dk), pad-mask cho prompt khác độ dài. Continuous batching
(vLLM-style): prompt xong sớm rời batch, prompt mới chèn vào — giữ GPU luôn đầy.

**M-throughput:** tok/s tổng khi B=1 vs B=8 vs B=32. Kỳ vọng gần tuyến tính tới khi nghẽn
băng thông. Ưu tiên THẤP — chỉ cần khi phục vụ nhiều người; latency (2.1/2.2) cần trước.

**Rủi ro:** pad-mask sai ⇒ prompt này "nhìn" prompt kia (rò rỉ chéo batch) — test cách ly.

### 2.4. Lượng tử hoá LM (int8 → int4) — bộ nhớ + tốc độ

**Cơ chế:** v2 42M float32 ~168MB; int8 ~42MB, int4 ~21MB. Trọng số chiếm băng thông chính
lúc decode (batch nhỏ) ⇒ lượng tử = ít byte tải = nhanh hơn + nhẹ hơn. Weight-only int8 là
chuẩn, mất chất lượng ~không đáng kể; int4 (GPTQ/AWQ) mất nhẹ, cần hiệu chỉnh.

**M-quant (điều kiện phân định):** bits/token held-out int8 vs float32 — chênh < 0.02 ⇒ dùng
int8 mặc định. int4 chênh < 0.1 ⇒ dùng cho máy nhỏ. Chênh lớn ⇒ giữ float/int8.

**Rủi ro:** MPS hỗ trợ int lởm chởm — có thể phải CPU-int8 hoặc chờ. Đo trên phần cứng đích.

### 2.5. CSR nén đồ thị — PHỤ THUỘC CỨNG cho runtime máy nhỏ (= TK_040)

**Vì sao ở đây:** đồ thị 2GB → 20GB RAM dict-Python ⇒ máy 8GB không nạp. Runtime thật trên
máy nhỏ **bất khả** tới khi TK_040 xong (CSR: ~20GB → vài GB, mmap được ⇒ không nạp hết vào
RAM). Ticket này KHÔNG làm lại TK_040 — chỉ ghi nó là **cửa** cho §2.2 (drafter đồ thị) và
cho deploy. Trên M4/server RAM lớn thì bỏ qua được tạm.

**M-mem:** RSS khi phục vụ = (CSR đồ thị mmap) + (LM int8). Mục tiêu < 4GB để chạy máy phổ
thông. Đo sau khi TK_040 + §2.4.

### 2.6. Serving loop / API — chất keo

**Cơ chế:** một vòng phục vụ nhận prompt → encode (`vocab.encode_prompt`) → prefill → decode
(speculative nếu bật) → detokenize stream. Có sẵn `viz.server` (TK cũ) làm nền; thêm endpoint
generate streaming. KHÔNG phát minh — ghép các mảnh 2.1-2.4.

**M-e2e:** latency token-đầu (prefill) + tok/s ổn định + RSS, trên 1 prompt thật, phần cứng
đích. Đây là con số "chạy runtime được không" trả lời trực tiếp câu hỏi tác giả.

---

## 3. THỨ TỰ LÀM (theo phụ thuộc + đòn bẩy, không theo hứng)

| # | việc | phụ thuộc | trạng thái | đòn bẩy |
|---|---|---|---|---|
| 1 | **KV cache — CHẠY probe** (§2.1) | checkpoint v2 (có) | code xong, chờ M4 | latency 5-20×, nền mọi thứ |
| 2 | **M-accept đo đồ-thị-nháp** (§2.2) | LM lớn + đồ thị | chưa | QUYẾT speculative sống/chết |
| 3 | speculative decoding đầy đủ | #1 + #2 dương | chưa code | nhiều token/bước |
| 4 | lượng tử int8 (§2.4) | checkpoint v2 | chưa | bộ nhớ + tốc độ, độc lập |
| 5 | TK_040 CSR (§2.5) | — (ticket riêng) | SPEC | mở khoá máy nhỏ |
| 6 | batch (§2.3) | #1 | chưa | throughput (ưu tiên thấp) |
| 7 | serving loop (§2.6) | #1,#3,#4 | chưa | sản phẩm cuối |

**Đường tới hạn:** #1 → #2 → (nếu dương) #3. Đó là con đường trả lời "sinh nhiều token/bước".
#4 song song được ngay. #5 (TK_040) là cửa deploy máy nhỏ, làm khi cần chạy ngoài M4.

---

## 4. ĐIỀU KIỆN PHÂN ĐỊNH / ĐIỂM DỪNG (ghi trước — chống trôi mục tiêu)

* **KV cache** không đạt ≥3× ⇒ nghi đo/phần cứng, KHÔNG kết luận vô dụng tới khi đo prompt dài.
* **M-accept ≈ 1 cả hai drafter** ⇒ speculative KHÉP, giữ KV cache là đủ cho latency. Ghi âm
  tính, KHÔNG xây hệ accept/reject để "cứu".
* **int8 chênh > 0.02 bits** ⇒ không lượng tử mặc định; giữ float trên máy đủ RAM.
* **Xuyên suốt:** đầu ra/​phân bố phải chứng minh KHÔNG đổi TRƯỚC khi báo tốc độ. Một lần khoe
  tok/s trên đầu ra đã lệch = lặp lỗi im lặng TK_041 §5.2.

---

## 5. RỦI RO / ÂM TÍNH CÓ THỂ (ghi trước để không bất ngờ)

1. **Đồ thị nháp kém trùng LM** (phân bố khác họ) ⇒ speculative-đồ-thị chết; lùi về LM-nhỏ-nháp.
2. **LM lớn chưa tốt** (đang train, §7.20a) ⇒ sinh nhanh một mô hình dở vẫn là dở. Tốc độ là
   nhân tử, không phải chất lượng — nếu chất lượng âm, tốc độ vô nghĩa. Ưu tiên TK_042 trước.
3. **MPS/phần cứng** giới hạn int/kernel ⇒ vài tối ưu chỉ chạy trên GPU thật, không MPS.
4. **TK_040 chưa xong** ⇒ mọi thứ chạy được trên M4/server nhưng KHÔNG trên máy phổ thông.
5. Tối ưu chồng nhau có tương tác (cache + quant + spec) — đo TỪNG cái riêng trước khi ghép,
   đừng bật cả ba rồi mới thấy đầu ra lệch không biết vì cái nào.
