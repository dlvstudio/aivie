# TK_042 — LÕI SINH: từ máy truy hồi sang máy sinh thật (bức tường n-gram)

| Field | Value |
|-------|-------|
| **ID** | TK_042 |
| **Status** | 🟡 **G13 = việc tiếp, RẺ NHẤT (§7.15): kNN-LM — đồ thị KHÔNG cần học nói tiếng transformer, cho LM ĐỌC HỘ.** Đo (cùng ckpt/datastore/query): khoá `h` transformer **lift@1 22×** vs túi PPMI **10×** ⇒ `h` = biểu diễn ngữ cảnh ĐÃ HỌC, thứ §7.8d/§7.9 chốt là nút thắt. **Không train lại · không cần in-context learning (§7.14c) · không đi qua `micro` nên KHÔNG bị `ar_index_backbone` ghi đè (§7.13).** Đây là con đường THỨ BA mà §7.14d bỏ sót. ⚠️ Chưa đo bits/byte; `22×` vẫn yếu (kNN-LM thật dùng LM 250M+); đo trên mini. ‖ G12 (train có `[FACT]`) vẫn mở nhưng ĐẮT hơn (cần B5 + train lại + in-context learning mà 11.4M KHÔNG có). |
| **Nguồn** | Chỉ đạo tác giả 2026-07-16: *"train full cũng chưa để làm gì… nó vẫn khá là verbatim copy. Tôi muốn mạng sinh thật sự, giống mạng LLM, chuyển văn phong mượt mà theo ngữ cảnh, chạy theo phân bố xác suất hợp ngữ cảnh, lúc thì sinh 'chó', lúc thì 'cún', chứ không bám vào các đoạn văn đã train."* |
| **Liên quan** | v4 Trụ 1 (thước — **chưa xong**), Trụ 2 (order-2/R-slot **âm tính**), Trụ 4 (hợp nhất sinh), TK_031 (VarLM), TK_041 (BPE — mở khoá §6.2), `docs/study/03_ATTENTION_QKV_FROM_GRAPH.md` |
| **Kết luận ngắn** | Đường sinh hiện tại **không thể** đạt yêu cầu, và **không phải vì thiếu cơ chế** — vì nó là **mô hình đếm**, mà mô hình đếm không điều-kiện-hoá được ngữ cảnh dài. Đây là bức tường NLP đã đâm suốt 30 năm. Vượt nó cần **biểu diễn ngữ cảnh nén, HỌC được** — không phải thêm bảng đếm thứ chín. |

---

## 0. ĐO (2026-07-16) — và đọc cho đúng

### 0.1. Ba phép đo mới

**(a)** ~~Backbone KHÔNG phải thủ phạm~~ → 🔴 **RÚT LẠI HOÀN TOÀN 2026-07-17 (§7.13).**
**Backbone CHÍNH LÀ thủ phạm, và là thủ phạm DUY NHẤT.** Phép đo dưới đây dùng **3 prompt**,
không cái nào chạm chỗ backbone quyết định ⇒ **mẫu quá nhỏ**. Đo lại 8 tổ hợp cổng:
`backbone ON → micro-influence 0.0%` (cả 4) · `backbone off → 33–35%` (cả 4). FB gate và G4
verify **không liên quan**. **Toàn bộ TK_042 được xây trên phép bác bỏ SAI này.** Giữ nguyên
văn dưới đây làm bằng chứng lỗi phương pháp:

```
backbone BẬT: 'Thủ đô Việt Nam là Hà Nội. Thành phố nằm ở đồng bằng Bắc Bộ, ven sông Hồng…'
backbone TẮT: 'Thủ đô Việt Nam là Hà Nội. Thành phố nằm ở đồng bằng Bắc Bộ, ven sông Hồng…'
→ token TRÙNG: 74/75 = 99%
```

`ar_index_backbone` ép `scores[t] = max(cur_max) × 1.5` cho token kế **theo trật tự cha nguồn** —
nhìn code thì đó là "chép có chủ đích". **Nhưng tắt nó gần như không đổi gì**: 4 kênh còn lại
(W_seq/macro/FRAME_BRACKET/skip) **đã đồng thanh chỉ về đúng đoạn văn đó**. Backbone **thừa**,
không phải nguyên nhân. *(Chẩn đoán đầu của tôi sai; phép đo cứu.)*

**(b) Phân bố next-token KHÔNG phải delta** — nó **quá phẳng**:

| `H(next\|w0)`, 400 token tần cao nhất | `network_mini_bpe` (1 294 cha) | **`network_v10_histwiki` (151 430 cha)** |
|---|---:|---:|
| Phân nhánh TB | 153.7 | **566.7** successor/token |
| **Entropy TB** | 5.34 bits | **6.42 bits** |
| P(top-1) TB | 0.208 | **0.167** |
| Lựa chọn hữu hiệu (2^H) | 40.5 | **85.7** |
| Token gần-xác-định (H<1 bit) | 1.8% | **0.5%** |

**Entropy TĂNG theo scale** — thêm ~100× cha → mỗi bước từ 40 lên **86 lựa chọn ngang ngửa**.
Script: `scratchpad/branching_probe.py`.

**(c) `ar_temperature = 0.0` mặc định** → argmax → **luôn lấy mode**. Cờ lấy mẫu có sẵn, chưa bật.

### 0.2. ⚠️ ĐỌC CHO ĐÚNG — hai đại lượng KHÁC NHAU, không được trộn

| | Là gì | Giá trị |
|---|---|---|
| **6.42 bits** (§0.1b) | `H(next\|w0)` TB, mỗi `w0` **tính đều nhau** | "phân nhánh của một token điển hình" |
| **4.155 bits** (TK_031) | **cross-entropy** PPM trên held-out | "mô hình đoán văn THẬT giỏi cỡ nào" |

Chúng khác nhau: ngữ cảnh tần cao **dễ đoán hơn**, mà cross-entropy có **trọng số theo tần
suất**. → **KHÔNG được nói "khe 4.4 bits so với LLM"** (bản nháp đầu của ticket này nói vậy —
sai). Phép so đúng là **4.155 vs ~2 bits/token của LLM** — **mà chính thước đó đang rò (§4)**.

**Cái vững chắc từ §0.1b** (cùng phương pháp → xu hướng hợp lệ):
1. Entropy **tăng** theo scale (5.34 → 6.42) — **scale làm mô hình bigram KÉM tự tin hơn**.
2. Mode chỉ giữ **16.7%** khối lượng ⇒ **argmax là lựa chọn gần như tuỳ tiện**.
3. Phân bố **có** để lấy mẫu — nhưng nó **không biết gì** (86 lựa chọn ngang ngửa).

---

## 1. Bài toán

Tác giả muốn **bốn** thứ, và chúng **không cùng một bài toán** — trộn chúng là tự lừa:

| # | Yêu cầu | Bản chất |
|---|---|---|
| **R1** | Sinh theo **phân bố xác suất hợp ngữ cảnh** | mô hình hoá — **bài toán thật** |
| **R2** | **Không bám** đoạn văn đã train | hệ quả của R1 |
| **R3** | Văn phong **mượt mà theo ngữ cảnh** | hệ quả của R1 |
| **R4** | *"lúc chó, lúc cún"* | **register/style control** — **KHÁC** (xem §8.3) |

Ticket này nhắm **R1**. R2/R3 rơi ra từ R1. **R4 KHÔNG** — xem `docs/study/04`.

---

## 2. Chẩn đoán: LỜI NGUYỀN N-GRAM

**Vì sao agenda phải tồn tại:** nó là thứ **DUY NHẤT** cung cấp ngữ cảnh. Mô hình gốc chỉ điều
kiện trên **một token** (`w0`) → 86 lựa chọn ngang ngửa → bỏ agenda = **word salad** (đúng như
v4 mô tả). Giữ agenda = có ngữ cảnh — **nhưng ngữ cảnh đó LÀ một đoạn văn đã lưu**.

> **Truy hồi đang làm THAY việc của điều-kiện-hoá-ngữ-cảnh.**
> Đó là toàn bộ lý do đầu ra "khá là verbatim copy". Không phải bug ở decode; là **lỗ hổng
> ở mô hình**, và agenda là **miếng vá che nó**.

**Vì sao không chữa bằng cách đếm nhiều hơn:** thêm ngữ cảnh vào **bảng đếm** → đếm **thưa** dần
→ backoff → **về lại bigram**. Bằng chứng ngay trong repo: VarLM/PPM nhìn **4 token** thay vì 1
chỉ mua **0.21 bit** (4.369 → 4.155, TK_031).

Đây **không phải** khiếm khuyết của dự án này — là bức tường NLP đâm suốt 30 năm: Kneser-Ney
trên **hàng tỉ** token vẫn ~1.7 bits/char (tiếng Anh) vs neural ~1.1. **Bengio 2003 (NNLM) ra
đời CHÍNH XÁC vì bảng đếm không điều-kiện-hoá được ngữ cảnh dài.** Cách duy nhất đã biết để
vượt: **nén ngữ cảnh thành VECTOR** thay vì tra ô — đó là việc attention làm.

---

## 3. Vì sao TÁM cơ chế đều "trung tính" — một bệnh, không phải tám lần xui

| Cơ chế | Kết quả |
|---|---|
| Phân rã Re-Pair · Re-Pair đệ quy · order-2 · compose_episode | trung tính |
| R-slot vị trí · E-episode (extrinsic) · KN smoothing | **âm tính** |
| Theme neuron | +2pp |

**Cả tám đều là "đếm thêm thứ gì đó".** Chúng tinh chỉnh một phân bố cần đi từ ~6 bits xuống
~2 bits. 0.21 bit là **nhiễu** so với khe đó. Và **`chó`/`cún` không thể nổi** vì với entropy
6.42, mạng **không "chọn giữa chó và cún" — nó không biết gì cả**, agenda quyết hộ.

**Hệ quả trực tiếp cho v4:** *"drift TĂNG theo scale (v7 142k cha lạc nặng hơn)"* — 2026-07-09
ghi nhận mà không giải thích nổi — **nay có cơ chế**: scale → entropy/bước **tăng** → drift tăng.
**Scale HẠI vì mô hình không điều kiện hoá ngữ cảnh.** Đây cũng là lý do **B5 (train full v11)
sẽ KHÔNG sửa được gì** cho R1 (nó vẫn cần, nhưng vì lý do khác: guard TK_041).

---

## 4. ✅ G0 XONG (2026-07-16) — thước KHÔNG "rò", nó ĐO NHẦM ĐẠI LƯỢNG

> Đây là thứ đã giam v4 hai tuần. Không phải leakage.

```python
sum_bits += math.log2(rank)      # src/nn_aivie/eval/metrics.py:116
```

**`bits` = log2(THỨ HẠNG) của token đúng, KHÔNG phải cross-entropy.** *"4.155 bits"* nghĩa là
*"token đúng nằm ở hạng ~17.8 trung bình"* — **không liên quan gì đến lượng tin**.

### 4.1. Ba lỗi ĐỘC LẬP, cái thứ ba chí mạng với ticket này

| # | Lỗi | Hệ quả |
|:---:|---|---|
| 1 | Gọi `log2(rank)` là **"bits"** | Mời gọi **lỗi phạm trù**. v4 mắc (so với perplexity). **Tác giả ticket này cũng mắc** (chia cho bytes/token ra "0.83 bits/byte" = giỏi hơn GPT-2 tiếng Anh = bất khả) |
| 2 | `if not isfinite(scores[gold]): n_oov+=1; continue` | **BỎ QUA đúng những vị trí mô hình thất bại hoàn toàn** → thiên lệch chọn mẫu → số đẹp giả tạo |
| 3 | **Hạng ⊥ Calibration** | **MÙ với thứ TK_042/B cần đo** |

**Lỗi #3 giải thích một âm tính đã bị hiểu sai.** Log `out/tk031_kn_ab.log`:
```
λ-interp : bits=1.4305  top1=46.1%
KN d=0.5 : bits=2.6873  top1=28.4%   → v4 kết luận "KN âm +1.26 bits"
```
KN smoothing **sinh ra để sửa CALIBRATION** — nó đánh đổi top-1 lấy đuôi phân bố tốt hơn. Thước
hạng **chỉ thấy cái KN hy sinh, mù với cái KN sửa**. Mà **calibration CHÍNH LÀ thứ việc sinh
cần**: lấy mẫu là lấy từ *phân bố*, không phải từ *thứ hạng*. → v4 nghi thước là **đúng**, sai
nguyên nhân.

### 4.2. Thước mới — `scripts/eval_bits.py`

`bits/token = -mean log2 P(gold|ctx)` (cross-entropy THẬT, **MỌI vị trí**) · `bits/byte`.
Mô hình phải trả **phân bố hợp lệ không có số 0** (nội suy unigram add-1) — né điều đó chính là
lỗi #2.

⚠️ **`bits/byte` so được xuyên TOKENIZER, KHÔNG xuyên NGÔN NGỮ.** Tiếng Việt ~5.5 byte/token
(dấu UTF-8 tốn 2–3 byte/chữ) → byte tiếng Việt "rẻ" hơn → bits/byte thấp hơn **hệ thống**. So
với "GPT-2 tiếng Anh ~1.0" là **lỗi phạm trù thứ ba** của tác giả ticket trong cùng một phiên.
**Chưa có mốc ngoài** — muốn có, phải chạy một LM thật trên **CHÍNH held-out này**.

### 4.3. G1 — BASELINE THẬT (base-LM = bigram+unigram, KHÔNG agenda)

| Checkpoint | bits/token | **bits/byte** | bigram=0 | OOV | lam* |
|---|---:|---:|---:|---:|---:|
| `network_v10_histwiki` (151k cha, **legacy**) | 8.173 | **1.305** | 13.6% | **9.0%** | 0.85 |
| `network_mini_bpe` (1.3k cha, BPE) | 9.026 | 1.625 | 37.9% | **0.0%** | 0.50 |

*(Hai dòng KHÔNG so được với nhau — khác corpus train. So tokenizer công bằng cần B5 = train
full v11 BPE trên CÙNG manifest v10.)*

**Đọc được ngay:**
1. **`OOV = 9.0%` trên vocab legacy vs `0.0%` trên BPE** — gần **1/10 vị trí held-out** là token
   vocab cũ **chưa từng thấy** → không thể gán xác suất → thước cũ **bỏ qua chúng**. Đây là
   **bằng chứng định lượng cho giá trị TK_041**: vocab đóng ⇒ OOV bất khả.
2. **`lam* = 0.85`** (real) vs **0.50** (mini) — nhiều data thì bigram đáng tin hơn. Trên mini,
   nghiêng **một nửa** vào unigram là tối ưu ⇒ bigram gần như vô dụng.
3. **`8.173 bits/token` = 2^8.17 ≈ 288 lựa chọn hữu hiệu.** Đây là **BỨC TƯỜNG, đo bằng thước
   sạch** — con số G2 phải đánh bại.

**🎯 MỤC TIÊU G2 (nhị phân):** hạng-thấp phải **< 1.305 bits/byte** trên
`network_v10_histwiki` + cùng held-out. Thắng ⇒ khái quát hoá THẬT. Không ⇒ §7.4.

---

## 5. BA CON ĐƯỜNG — quyết định của TÁC GIẢ, không phải của AI

### A. Giữ thuần Hebbian-đếm

| ✅ Ưu | ❌ Nhược |
|---|---|
| Luật 0 sạch tuyệt đối | **Trần CỨNG ~6 bits** — R1 **bất khả**, vĩnh viễn |
| Mọi thứ giải thích được | Scale **hại** (drift tăng) |
| Không rủi ro, không train lại | Mọi cơ chế mới sẽ **tiếp tục trung tính** — dự đoán được |
| **Định vị trung thực vẫn có giá trị thật**: máy truy hồi + ghép fact có căn cứ, QA 86–100% | Không bao giờ là thứ tác giả mô tả |

→ Nếu chọn A: **đóng TK_042**, sửa README định vị dự án cho đúng, dừng hứa "sinh như LLM".

### B. Nút cổ chai HỌC ĐƯỢC bằng LUẬT CỤC BỘ ← **đề xuất**

> **"Không backprop" ≠ "không học".**

**Predictive coding** (Rao & Ballard 1999; Friston): mỗi tầng **dự đoán** đầu vào tầng dưới, cập
nhật synapse theo **sai số dự đoán CỤC BỘ**. Whittington & Bogacz (2017) chứng minh mạng
predictive-coding với **plasticity Hebbian cục bộ xấp xỉ được backprop**. Đây là thứ **vỏ não
thật sự làm**. (Họ hàng: Forward-Forward, Hinton 2022; target propagation.)

Cụ thể: một tầng dự đoán `e_next` từ `e_context` (pool có decay của token gần), hạng `r ≪ 32k`,
học bằng sai số cục bộ.

| ✅ Ưu | ❌ Nhược |
|---|---|
| **Nút cổ chai ÉP nén → khái quát hoá THẬT** — lần đầu dự án có | Vẫn **KHÔNG phải LLM**: ước ~3.5–4 bits (từ ~6) |
| **`cún` nhận khối lượng trong ngữ cảnh của `chó`** — hàng PPMI giống nhau, KHÔNG cần đồng hiện | Nới tinh thần "chỉ đếm" (dù không nới Luật 0 — vẫn học từ data mình) |
| **Trung thành sinh học HƠN hiện tại** (§6.1) | Cần thước sạch trước (§4) |
| Đã nửa-đặc-tả: `docs/study/03` (Q/K từ SVD đồ thị), treo từ 2026-07-02 | ⚠️ **Corpus 19.6M ĐỨNG NGAY RANH GIỚI sample-complexity cho r=64** (§7.1) → **50/50** |
| **TK_041 vừa mở khoá** (§6.2); **G2 ≈70% hạ tầng đã có** (§7.3) | Neo thực nghiệm `text8` (17M token): word2vec ra embedding **dùng được nhưng tầm thường** |

### C. Transformer nhỏ trên corpus

| ✅ Ưu | ❌ Nhược |
|---|---|
| **Đạt R1 thật** (~2–2.5 bits/byte, 20M token → ~5–20M params) | **Bỏ tiền đề dự án.** Backprop, không Hebbian |
| Công nghệ đã chín, rủi ro kỹ thuật thấp | **Đồ thị còn để làm gì?** ← câu hỏi thật |
| | Không còn là "đề án nghiên cứu Hebbian" |

---

## 6. Vì sao đề xuất B

### 6.1. Lập luận sinh học — B trung thành HƠN hiện tại

Nơ-ron vỏ não thật có **~10⁴ synapse, CỐ ĐỊNH**. Hebbian dưới **ngân sách synapse + chuẩn hoá**
= **quy tắc Oja** → hội tụ về **PCA**. **Não bị ÉP nén; đó là lý do nó khái quát hoá.**

> Đồ thị nn-aivie có **cạnh vô hạn** — nó mọc đến khi khớp dữ liệu **chính xác**. Không gì ép
> nó nén. Nên nó **NHỚ**, không **HIỂU**.
>
> **Dự án có Hebbian mà THIẾU đúng cái ràng buộc khiến Hebbian thông minh.**

Đây là chỗ mạng **phản bội sinh học** — mỉa mai thay, đúng chỗ nó tự nhận sinh học nhất. B **sửa**
điều đó; nó không phải nhượng bộ.

### 6.2. TK_041 vừa mở khoá — mối nối không lường trước

| | Trước TK_041 | Sau |
|---|---|---|
| Vocab | 126 516, **MỞ** | **32 002, ĐÓNG** |
| Ma trận bigram | 126k² = 16 tỉ ô | 32k² = **1 tỉ** (thưa → `scipy.svds` khả thi) |
| **53% hàng là hapax (df=1)** | → SVD trên hàng có **1 quan sát** = **nhiễu thuần** | hoà tan thành mảnh có df thật |

**Phân tích hạng thấp trên vocab CŨ là vô nghĩa** — quá nửa số hàng chỉ có một quan sát. Vocab
đóng 32k, mọi token có thống kê thật, **mới làm SVD/nút-cổ-chai có nghĩa**. Không thiết kế cho
điều này; nó rơi ra.

### 6.3. Đính chính một câu SAI trong v4

v4 §3 Trụ 3 viết: *"PPMI bậc-1: 'chó≈cún' chỉ nổi nếu đồng hiện"*. **Sai về kỹ thuật.** Tương
đồng **paradigmatic** đến từ **hàng vector giống nhau** (cùng hàng xóm), **KHÔNG** từ đồng hiện
trực tiếp — đó là toàn bộ cơ chế word2vec/GloVe (2013). **Chính T0 của dự án đã chứng minh:**
`cấy/trồng` rank 75→**3**; `chữa`→{bệnh, sẹo, thuốc, tật, AIDS, lây, mắc}. Hai từ đó **không**
đồng hiện với nhau.

→ **Cơ chế ĐÃ CHẠY. Cái thiếu: nó không được dùng để SINH.** E hiện là **phụ kiện tra cứu mờ**
(`hybrid_fuzzy_topk`, từng mặc định TẮT), không phải chất nền sinh. **Đó là cơ hội bị bỏ lỡ lớn
nhất của dự án.**

---

## 7. Lộ trình đo được (nếu chọn B)

### 7.0. Bài toán thật, nói bằng một hình ảnh: đây là NETFLIX

Đo (`network_mini_bpe`): `W_seq` là bảng **32 002 × 32 002 = 1 024 128 004 ô**, có số ở
**136 138 ô** → **ĐỘ ĐẦY 0.013%**, tức **99.99% bảng là SỐ KHÔNG**. Và mỗi ô **độc lập tuyệt
đối**: `W[chó][sủa]=50` **không nói gì** về `W[cún][sủa]`.

| | Netflix (2009) | nn-aivie |
|---|---|---|
| Bảng | người dùng × phim | token × token-kế |
| Rỗng | ~99% | **99.99%** |
| Câu hỏi | *"chưa xem phim này, có thích không?"* | *"chưa thấy 'cún sủa', xác suất bao nhiêu?"* |
| Tra bảng trả lời | **0 = không biết** | **0 = không biết** |

Lời giải Netflix = **phân tích hạng thấp**: `W ≈ A·Bᵀ` (`A,B`: 32 002×r). Với r=64 → **4.1M tham
số thay vì 1 tỉ ô** ⇒ **không đủ chỗ nhớ từng ô** ⇒ **buộc** phát hiện hàng nào giống hàng nào
⇒ **ĐIỀN VÀO các số không**:

```
hàng chó: {sủa:50, con:200, nuôi:30, nhà:80,…}  ← dày
hàng cún: {con:8, sủa:2}                        ← thưa (bản sao nhiễu của chó)
     ↓ SVD hạng r
a_cún ≈ a_chó  ⇒  P(cún | "con ___") > 0  ← DÙ "con cún" CHƯA TỪNG XUẤT HIỆN
```

**Không cần** `chó`/`cún` đồng hiện — chỉ cần **hàng xóm giống nhau** (§6.3). T0 đã chứng minh
cơ chế chạy trên chính data này: `cấy/trồng` rank 75→**3**.

### 7.1. ⚠️ RỦI RO ĐỊNH LƯỢNG — corpus đứng NGAY RANH GIỚI

Matrix completion cần ~`O(r · n · log n)` quan sát để phục hồi ma trận hạng `r`:

```
n = 32 002 (vocab BPE),  log n ≈ 10.4
  r =  16  →  ~5.3 triệu quan sát cần
  r =  32  →  ~10.7 triệu
  r =  64  →  ~21 triệu          ← corpus ta CÓ 19.6 triệu → NGAY RANH GIỚI
  r = 128  →  ~43 triệu          ← THIẾU HƠN 2 LẦN
```

**Corpus = 19.6M token = 19.6M quan sát bigram** (TK_041 §0.1).

→ **Sweep `r ∈ {16, 32, 64, 128}`** — KHÔNG phải {64,128,256,512} (bản nháp đầu ticket này ghi
sai; phép tính trên bác bỏ). Kỳ vọng: **r=32–64 là điểm ngọt**; r≥128 under-determined → overfit.

*(Cảnh báo phương pháp: bound trên là cho phục hồi CHÍNH XÁC ma trận hạng-r từ mẫu ngẫu nhiên
đều. Ta khác: ma trận không hạng thấp chính xác, mẫu theo Zipf (không đều), và ta chỉ cần đoán
tốt chứ không cần phục hồi. Zipf **giúp** — token tần cao được ước lượng tốt, đuôi là nhiễu. Nên
coi đây là **kiểm tra bậc-độ-lớn**, không phải định lý.)*

**Neo thực nghiệm sát nhất — `text8`:** corpus benchmark kinh điển **17 triệu token** (≈ ta).
word2vec trên text8 **chạy được**, cho embedding **dùng được nhưng tầm thường** (~30–40% analogy,
vs ~70% khi train 100 tỉ token). **Đó là kỳ vọng thực tế: THẬT, nhưng YẾU.** Đây là nguồn gốc
của dự đoán **G2 = 50/50** ở §9 — không phải khiêm tốn xã giao.

### 7.2. Phase

| Phase | Nội dung | Xong khi | Rủi ro |
|:---:|---|---|:---:|
| **G0** | **SỬA THƯỚC** (§4): held-out THẬT (khoá theo byte; kiểm rò bằng xáo/nhân bản), **bits-per-BYTE**, `scripts/eval_lm_ruler.py` | Số v10 hợp lý (**KHÔNG** < 1.0 bits/byte — hiện 0.83 = bất khả) | thấp — **BẮT BUỘC** |
| **G1** | **Đo lại bức tường** bằng thước sạch: bigram / PPM / (+agenda) / (−agenda) | Bảng 4 dòng = baseline THẬT | thấp |
| **G2** | **Thí nghiệm RẺ trước khi tin** (tiền lệ R-slot): PPMI **CÓ HƯỚNG** trên `W_seq` → SVD → chấm `⟨A[ctx],B[next]⟩`. **KHÔNG train lại.** Sweep **r ∈ {16,32,64,128}** (§7.1) | **hạng-r có THẮNG đếm-đầy-đủ trên held-out không?** Thắng ⇒ khái quát hoá THẬT | **trung bình** |
| **G3** | **Probe thay thế** — `P(cún \| "con ___ sủa") > 0` dù chưa từng thấy; đối chứng: cặp KHÔNG liên quan phải vẫn thấp | **nhị phân**: có khái quát hoá hay không | thấp |
| **G4** | Predictive-coding cục bộ (**chỉ nếu G2 dương**): `ŷ = A·e_ctx`; `ε = e_next − ŷ`; `ΔA ∝ ε ⊗ e_ctx` (quy tắc delta — **cục bộ, không chain rule**). Ngữ cảnh = pool có decay nhiều token | bits-per-byte < G1 | cao |
| **G5** | **Lấy mẫu** — `ar_temperature`/top-p/typical | **LÀM CUỐI** — lấy mẫu từ phân bố dốt = nhiễu (§8.4) | thấp |
| **G6** | **Cai nghiện agenda** — giảm dần trọng số; đo `copy`(LCS) tụt trong khi QA giữ | verbatim giảm, chất lượng giữ | cao |

### 7.3. Hạ tầng — B cần ít hơn tưởng

| Cần | Trạng thái |
|---|---|
| Ma trận đồng hiện | ✅ `W_seq` |
| PPMI + SVD thưa (`scipy.svds`) | ✅ `core/embedding.py` — **đang phí phạm làm phụ kiện tra cứu mờ** (§6.3) |
| Vocab ĐÓNG để SVD có nghĩa | ✅ **TK_041 vừa cho** (§6.2) |
| PPMI **CÓ HƯỚNG** (trái ≠ phải) | ❌ hiện đối xứng theo cửa sổ |
| Chấm điểm `⟨A[ctx], B[next]⟩` | ❌ |
| Thước sạch | ❌ **G0** |

→ **G2 ≈ 70% hạ tầng đã có.** Đó là lý do nó là thí nghiệm RẺ và phải làm trước G4.

### 7.5. ✅ G2 ĐÃ CHẠY (2026-07-16) — ÂM TÍNH, và nó phủ định CHÍNH §7.0

`scripts/probe_lowrank.py` trên `network_v10_histwiki` (151k cha), held-out essays, n=200:

| mô hình | bits/byte | Δ vs baseline |
|---|---:|---:|
| **ĐẾM đầy đủ (baseline)** | **1.288** | — |
| hạng r=16 | 1.889 | +0.601 ❌ |
| hạng r=32 | 1.858 | +0.570 ❌ |
| hạng r=64 | 1.838 | +0.550 ❌ |
| hạng r=128 | 1.803 | +0.515 ❌ |

Đơn điệu ~0.03/lần gấp đôi r ⇒ để đóng 0.515 cần **r ≈ 2 triệu**. Không phải "thiếu hạng".

#### 7.5a. Thí nghiệm CÀI SAI — SVD khớp 1 tỉ số không

Chẩn đoán trực tiếp (`network_mini_bpe`):
```
ô PPMI > 0 (quan sát THẬT) :       128 391
ô coi-là-0 (THIẾU dữ liệu) : 1 023 999 613  (99.99%)
trên ô QUAN SÁT: PPMI thật 3.523 → tái dựng 0.811   ← CHỈ GIỮ 23%
trên ô TRỐNG   : tái dựng 0.0002                     ← SVD khớp số 0 rất "ngoan"
```
SVD tiêu năng lực khớp **1.02 tỉ số không**, bóp quan sát thật xuống 23%. **Đó CHÍNH XÁC là bài
học Netflix mà §7.0 viện dẫn rồi không làm theo**: *"chưa đánh giá" ≠ "đánh giá 0 điểm"* —
Netflix thắng nhờ tối ưu **chỉ trên ô quan sát được** (weighted MF/ALS), không phải SVD thuần.
Đúng thuật toán phải là **SGNS/negative sampling** hoặc **GloVe** (chỉ fit nnz).

#### 7.5b. NHƯNG — kể cả cài ĐÚNG cũng vô ích. Trần lý thuyết:

| vị trí | tỉ lệ | bits/token |
|---|---:|---:|
| bigram **ĐÃ THẤY** | 86.4% | **6.422** |
| bigram **= 0** | 13.6% | **19.251** ← unigram backoff thảm hoạ |
| **tổng** | | **8.173** |

**NẾU hạng-thấp điền số 0 HOÀN HẢO** (giỏi bằng ô đã thấy — lạc quan phi thực tế):
→ **6.422 bits/token = 1.025 bits/byte**. Lợi tối đa **1.751 bits/token**.

> **Mà cần ~3 bits/token để sinh như LLM. ⇒ "điền số 0 bigram" KHÔNG THỂ là cách mạng, kể cả
> khi HOÀN HẢO. Bức tường LÀ CHÍNH BIGRAM, không phải các số 0 của nó.**

*(Nhất quán: 6.422 bits/token trên ô đã thấy ≈ entropy 6.42 bits đo ở §0.1b bằng phương pháp
hoàn toàn khác.)*

#### 7.5c. Phép loại suy Netflix đã lừa chính tác giả ticket

**Netflix = bài toán ĐIỀN MA TRẬN. LM = bài toán ĐIỀU KIỆN HOÁ THEO NGỮ CẢNH.** Điền bigram
hoàn hảo vẫn cho một **mô hình bigram** — hết thuốc ở 6.42. §7.0 đúng về *cơ chế* (hạng thấp
điền số 0) nhưng **sai về mục tiêu**.

→ **G2 (hạng-thấp trên BIGRAM) là thí nghiệm SAI ĐỀ.** Nội dung thật của B **chưa bao giờ là**
"phân tích ma trận bigram" — §6 đã viết `e_ctx` = **pool có decay NHIỀU token**. G2 rút gọn nó
về `ctx = w0` ⇒ thoái hoá thành bigram ⇒ chạm trần 6.42 bất kể hạng.

**Giá trị của G2: nó rẻ, và nó chặn ta xây G4 trên một tiền đề sai.** Đúng vai trò thiết kế.

### 7.6. Hệ quả — lộ trình phải VIẾT LẠI

**Điều đã CHỨNG MINH:** mọi mô hình chỉ nhìn **1 token ngữ cảnh** đều bị chặn ở **~6.4
bits/token**, bất kể ước lượng giỏi cỡ nào. Đó là trần thông tin của bigram trên corpus này.

**Suy ra: R1 chỉ đạt được bằng NGỮ CẢNH DÀI.** Và ngữ cảnh dài + bảng đếm = thưa (đã đo: 13.6%
bigram đã = 0; trigram sẽ tệ hơn nhiều). ⇒ **buộc phải NÉN ngữ cảnh thành vector** ⇒ **buộc phải
HỌC** một mô hình thật.

**⇒ B và C hội tụ về cùng một việc**, chỉ khác **luật học**:

| | B (predictive coding) | C (transformer) |
|---|---|---|
| Mô hình trên ngữ cảnh dài | **cần** | **cần** |
| Rẻ? | **KHÔNG** — G2 rẻ đã chết | không |
| Luật học | sai số **CỤC BỘ** (Rao-Ballard) | backprop |
| Đã chứng minh đạt ~2-3 bits? | ❌ chưa ai làm ở quy mô này | ✅ |

**Câu hỏi thật cho tác giả không còn là A/B/C mà là:** *muốn luật học CỤC BỘ (khoa học thú vị,
rủi ro cao, chưa ai chứng minh) hay BACKPROP (chắc chắn chạy)?* — và **cả hai đều là "train một
mô hình thật trên ngữ cảnh", không còn là "thêm cơ chế vào đồ thị"**.

### 7.7. G7 ("Bước 1") — ĐO CÁI GAP CHO RA SỐ THẬT · `scripts/train_neural_lm.py`

> Sau khi G2 chết (§7.5), đây là thí nghiệm rẻ TIẾP THEO và nó **quyết định thay cho mọi
> tranh luận A/B/C**. Code đã viết + smoke-test; **chờ chạy trên máy ≥16GB**.

#### 7.7a. ⚠️ PHÁT HIỆN PHẦN CỨNG — TK_040 M0 GHI SAI, và nó chặn cả B5

**Máy dev hiện tại: MacBook Pro Intel, 8 GB RAM** (đo 2026-07-16: `hw.memsize` = 8589934592).
**TK_040 M0 ghi *"swap trên MacBook 24GB"* — SAI.** Hệ quả nghiêm trọng, cần sửa TK_040:

| | |
|---|---|
| `network_v10.pkl` (1.85GB đĩa → **~20GB RAM**) | **KHÔNG load nổi trên máy 8GB.** Mà nó là checkpoint **MẶC ĐỊNH** (`core/paths.py`, README §0.4.1) |
| **TK_041 B5** (train full v11, 8h) | **sẽ đẻ ra checkpoint chính máy này không mở được** ⇒ B5 như đang viết là **vô nghĩa trên phần cứng hiện tại** |
| **TK_040 (CSR)** | không còn là "tối ưu cho vui" — nó **CHẶN việc dùng cơ bản** |

*(v10_histwiki 349MB → RSS ~3.3GB vẫn chạy được, có swap. Mọi số G0/G1/G2 đo trên nó.)*

**Torch trên máy này BẤT KHẢ:** Intel Mac ⇒ PyTorch **ngừng build wheel macOS x86 sau 2.2.2**
⇒ kẹt ở torch 2.2.2 ⇒ **không tương thích numpy 2.x** (`_ARRAY_API not found`). Không hạ numpy
(cả dự án chạy trên 2.2.6). Trên máy 24GB **Apple Silicon**: torch ≥2.3 + **MPS** → chạy tốt.

#### 7.7b. Thiết kế — MỘT script làm CẢ HAI mô hình

Bẫy lớn nhất của G7 là **so khập khiễng**. Số đếm `1.288 bits/byte` (§4.3) đến từ
`network_v10_histwiki`: **corpus khác**, **tokenizer khác** (legacy 113k). So nó với LM train
trên 99MB+BPE32k là so **ba biến cùng lúc**.

→ Script dựng **cả hai trên cùng một dòng dữ liệu**: cùng manifest, cùng `bpe_v1.json`, cùng
held-out (`eval_bits.load_heldout`, cùng seed 42), cùng công thức `-log2 P(gold)` trên **MỌI**
vị trí. **Khác biệt duy nhất còn lại = mô hình.** Baseline đếm dựng thẳng từ corpus đã
tokenize — **không cần load checkpoint 20GB**.

#### 7.7c. Chạy (máy ≥16GB, ưu tiên Apple Silicon)

```bash
pip install torch          # Apple Silicon: ≥2.3 (numpy 2 OK). Intel Mac: BẤT KHẢ (§7.7a)
PYTHONPATH=src python -u scripts/train_neural_lm.py --log-file out/g7_neural_lm.log
# mặc định: d=256 L=4 ctx=256 batch=32 steps=3000 → ~11M tham số, 19.6M token
```

#### 7.7d. Bug đã bắt bằng SMOKE TEST (đừng gỡ sanity check)

Bản đầu: **loss khởi đầu 42 nats = 60 bits/token**, trong khi trần lúc init chỉ là
`ln(32002) = 10.37 nats = 15 bits` (phân bố đều) — mô hình xuất phát từ chỗ **TỆ HƠN ngẫu
nhiên 4 lần**. Nguyên nhân: **tied weights + init mặc định** (`nn.Embedding` init N(0,1); head
dùng chung weight ⇒ logits ~ N(0,d) ⇒ std ~8 ⇒ logits cực đoan). Vá: init GPT-2 `N(0,0.02)`.
Sau vá: **loss init = 10.391 ≈ ln(V) chính xác** ✅ → 6.88 sau 200 step.

→ Giữ `sanity init` (so loss init với `ln(V)`) — nó **nổ được**, đã chứng minh.
→ Giữ guard `undertrained` — bản đầu in *"→ A là câu trả lời đúng"* ngay sau smoke test **12
   step**, đúng loại "kết luận từ số vô nghĩa" mà cả ticket này sinh ra để chống.

#### 7.7e. Đọc kết quả — NHỊ PHÂN, ghi trước

| Kết quả | Nghĩa | Làm gì |
|---|---|---|
| **nơ-ron < đếm − 0.1 bits/byte** | gap THẬT: đếm không lấy được ngữ cảnh dài | câu hỏi B (luật cục bộ) vs C (backprop) trở nên có nghĩa |
| **nơ-ron ≈ đếm** | 19.6M token **quá nhỏ** cho mô hình học | **A là câu trả lời đúng** — đóng TK_042, sửa định vị README, **không tiếc** |

Đối chiếu: 19.6M ≈ **text8** (17M). Kỳ vọng thực tế **~2.5–3 bits/token**, KHÔNG phải LLM (§8).

### 7.8. G8 — nn-aivie ĐÃ LÀ một kNN-LM, chạy ở **k=1** · `scripts/probe_knn_lm.py`

> Xuất phát từ câu hỏi tác giả 2026-07-17: *"có cách nào nén theo ngữ cảnh, cùng ngữ cảnh thì
> kéo lại gần nhau — lúc RUNTIME, chứ không train từ đầu?"*

#### 7.8a. Cái đó có tên — và dự án đã là một cái

**similarity-based smoothing** (Dagan/Lee/Pereira 1999) · **cache model** (Grave 2017) ·
**kNN-LM** (Khandelwal 2020). Ánh xạ vào repo:

| kNN-LM | nn-aivie |
|---|---|
| datastore | **`ParentStore`** (parents = chuỗi token đã lưu) |
| tìm k láng giềng | **`rank_matches`** |
| biểu diễn ngữ cảnh | túi token nội dung, trọng số IDF (`query_match_units`) |
| **k** | **1** (`best_match`) |
| trộn thành phân bố | ❌ **không** — dội nguyên `source_ids` của cha đó ra làm agenda |

> **"Verbatim copy" là TRIỆU CHỨNG TOÁN HỌC CỦA k=1.** Một kNN-LM lấy 1 láng giềng rồi chép nó
> thì **buộc phải** là máy chép. Không có bug nào — đó là hành vi ĐÚNG của thuật toán SAI.
> Dự án đã đi **nửa đường** tới một mô hình dựa-truy-hồi (họ RETRO/kNN-LM), và **nửa còn lại
> chưa bao giờ thử**.

#### 7.8b. Đo (train-free: chỉ dùng `E` d=96 + `parents` đã có)

`ctx = Σ decay^j · E[token_{i−1−j}]` (L=8, decay=0.7, chuẩn hoá L2); datastore 400k cặp;
`P ∝ Σ softmax(sim/τ)·1[next_i]`; nội suy unigram **giống baseline** → so công bằng.

**`network_v10_histwiki` (151k cha), cùng held-out, cùng công thức bits:**

| | bits/byte | Δ vs k=1 |
|---|---:|---:|
| **ĐẾM (bigram+unigram) — baseline §4.3** | **1.288** | — |
| kNN **k=1** *(= chế độ HIỆN TẠI của agenda)* | 1.907 | — |
| kNN k=16 | 1.761 | −0.146 |
| kNN k=64 | 1.678 | −0.229 |
| kNN **k=256** | **1.636** | **−0.270** · **vẫn THUA đếm +0.348** |

*(mini_bpe cùng xu hướng: k=1 → 2.260, k=64 → 1.909, Δ −0.352; baseline đếm 1.625.)*

#### 7.8c. RECALL@k — tách RETRIEVAL khỏi CALIBRATION (và nó lật dự đoán)

k=1 là mô hình **tự tin cực đoan**: `P(gold)`=1 nếu hàng xóm duy nhất đúng, =0 nếu sai ⇒ sai thì
trả giá khủng khiếp bằng bits. Nên bits giảm khi k↑ **có thể** chỉ là "bớt tự tin"
(calibration), không phải "tìm được nhiều hơn" (retrieval). Recall@k tách hai cái:

| | **recall@1** | **recall@256** |
|---|---:|---:|
| gold có nằm trong k hàng xóm? | **15.0%** | **57.3%** (**×3.8**) |

**Dự đoán của tác giả ticket (ghi trước khi đo): "chủ yếu là calibration" — SAI.** Trộn **tìm
được THẬT**.

#### 7.8d. …NHƯNG đọc kỹ thì âm tính này MẠNH HƠN, không yếu hơn

| | |
|---|---:|
| Token đúng **CÓ MẶT** trong 256 hàng xóm | **57.3%** |
| Mà bits vẫn | **1.636** (thua đếm **1.288**) |

> **THÔNG TIN CÓ Ở ĐÓ — MÔ HÌNH KHÔNG DÙNG ĐƯỢC.** 57.3% recall là trần trên rất khá: biết
> **chọn đúng** trong 256 ứng viên đó thì kết quả tốt hơn hẳn. Vấn đề: `softmax(sim/τ)` **trải
> khối lượng SAI** — nó xếp hạng hàng xóm bằng độ giống của *túi embedding*, mà độ giống đó
> **không tương quan đủ** với "hàng xóm nào có token kế ĐÚNG".

**Ba việc, tách bạch được:**

| Việc | Trạng thái |
|---|---|
| **Tìm ứng viên** (recall) | **57.3% — khá tốt** |
| **CHẤM ĐIỂM ứng viên** (xếp hạng) | **HỎNG — đây là chỗ mất bits** |
| Trộn thay vì chép | thật, −0.270 |

⇒ Âm tính KHÔNG phải *"kNN không tìm được"* mà là ***"kNN tìm được, nhưng không biết mình đã
tìm được"*** — định nghĩa của một hệ **có thông tin, thiếu HÀM CHẤM**.

> **⚠️ ĐÍNH CHÍNH (§7.8f-oracle, đo sau):** kết luận *"nút thắt là BIỂU DIỄN"* ở cuối mục này
> **chỉ đúng một nửa**. Oracle chứng minh hàm chấm bỏ phí **3.22 bits/token**, và chấm hoàn hảo
> **THẮNG đếm 1.23 bits** ⇒ biểu diễn hiện tại **KHÔNG** vô dụng. Nhưng nó vẫn dừng ở 6.943,
> còn cách R1 (~3.0) gần **4 bits** ⇒ **cả hai** (biểu diễn *và* hàm chấm) đều không đủ. Đọc
> §7.8f-oracle trước khi trích mục này.

Và **"chấm điểm ứng viên đúng" CHÍNH LÀ việc của attention**: cho mỗi token ngữ cảnh một trọng
số **phù hợp với dự đoán hiện tại**, thay vì trọng số vị trí CỐ ĐỊNH (`decay^j`). Đó là thứ
**phải HỌC**.

> **⇒ Nút thắt KHÔNG phải cách TRỘN. Nó là BIỂU DIỄN + HÀM CHẤM NGỮ CẢNH.**
>
> Núm **thứ nhất** của kNN (ba núm: biểu diễn · k · trộn) — núm quan trọng nhất — đang được vặn
> bằng **túi-embedding-trọng-số-vị-trí**, thứ rẻ nhất có thể. Chính bài kNN-LM ablate điểm này:
> nó thắng vì `h` là biểu diễn ngữ cảnh **ĐÃ HỌC**. **Mọi** mô hình truy-hồi thắng lớn (kNN-LM,
> RETRO) đều **BỔ TRỢ** cho một LM đã train, không **THAY** nó.

#### 7.8e. Điểm yếu của phép đo (khai trước)

* **Datastore chỉ 400k cặp / ~3M vị trí = ~13%.** kNN cải thiện theo `log(datastore)`; ước
  3 lần gấp đôi × ~0.03 ≈ **−0.09** — **không đủ** đóng khe 0.348. Nhưng đó là **PHỎNG ĐOÁN,
  không phải đo**. Muốn chắc: full datastore (~1.2GB — chật trên máy 8GB, xem §7.7a).
* `decay=0.7` ⇒ trọng số `1, 0.7, 0.49…` ⇒ ngữ cảnh **bị token cuối chi phối** ⇒ hiệu quả chỉ
  nhỉnh hơn bigram. Đây là *túi có trọng số vị trí*, **không phải attention**.
* Bug đã bắt lúc viết: `argpartition` trả top-kmax **KHÔNG sắp thứ tự** ⇒ `top[:, :k]` lấy k
  phần tử **tuỳ tiện** trong top-256, không phải k gần nhất. Bản sắp thứ tự cho số đúng
  (k=1: 2.359 → 2.260). **Lỗi im lặng thứ sáu của phiên** — kết quả vẫn "hợp lý", chỉ sai.

#### 7.8g. Liên hệ CLS — vì sao điều này củng cố §6.1

Tiêu đề bài kNN-LM: **"Generalization through Memorization"**. Phát hiện của họ: thêm kNN trên
**CHÍNH corpus mà LM đã train** vẫn giảm perplexity đáng kể ⇒ **LM không nhớ nổi dữ liệu của
chính nó**; bộ nhớ tường minh moi lại phần nó đánh rơi. Lợi ích lớn nhất rơi vào **sự kiện
hiếm, tên riêng, tri thức cụ thể**.

⇒ **kNN-LM = gắn HỒI HẢI MÃ vào VỎ NÃO.** nn-aivie là bài toán **ngược**: đã có hồi hải mã (QA
86–100%), thiếu vỏ não. Và kNN-LM chứng minh **hai cái KHÔNG thay thế nhau** — củng cố §6.1 và
củng cố rằng trần 6.42 bits là **thật**, không phải thiếu cơ chế.

#### 7.8f-oracle. TRẦN TRÊN CỦA HÀM CHẤM — đo 2026-07-17, và nó lật §7.8d… một nửa

`recall@256 = 57.3%` ⇒ token đúng CÓ MẶT trong 256 hàng xóm. Oracle = chấm **hoàn hảo** (dồn
hết khối lượng vào hàng xóm đúng nếu nó có mặt) ⇒ **trần trên của "sửa hàm chấm"**, tách khỏi
"sửa biểu diễn". `network_v10_histwiki`, cùng held-out:

| | bits/token | bits/byte |
|---|---:|---:|
| kNN k=256, chấm **cosine** (thực tế) | 10.163 | 1.623 |
| **ĐẾM (baseline)** | **8.173** | **1.288** |
| kNN k=256, chấm **HOÀN HẢO (oracle)** | **6.943** | **1.109** |
| *cần để sinh như LLM* | *~3.0* | — |

```
dư địa HÀM CHẤM : 10.163 → 6.943 = 3.22 bits/token   ← LỚN THẬT
oracle vs ĐẾM   :  8.173 → 6.943 = 1.23 bits/token   ← THẬT, oracle THẮNG đếm
oracle → LLM    :  6.943 → ~3.0  = 3.94 bits/token   ← CÒN NGUYÊN
```

**Lật được §7.8d một nửa:** không phải *"kNN vô vọng"* — hàng xóm **ĐỦ TỐT để thắng đếm** nếu
biết chấm; hàm chấm cosine đang bỏ phí **3.22 bits/token**.

**⚠️ NHƯNG KHÔNG lật được bức tường — và đây mới là điều phải đọc:**

> **Kể cả chấm HOÀN HẢO (bất khả — oracle nhìn trộm đáp án), ta dừng ở 6.943 bits/token, còn
> cách đích ~3.0 gần 4 BITS.** Oracle đưa ta từ *"thua đếm"* lên *"hơn đếm 1.23 bits"*, trong
> khi thứ cần là *"hơn đếm gấp nhiều lần"*. ⇒ **kNN + biểu diễn hiện tại KHÔNG đạt R1, kể cả
> ở trần lý thuyết của nó.**

**Hai lần tính nhẩm SAI của tác giả ticket (ghi lại — cùng họ lỗi phạm trù §0.2):**
1. Ước oracle ≈ 0.99 bits/byte "thắng đếm 0.3" → thật: **1.109**, thắng 0.179.
2. Rồi (sau khi thấy mini) lập luận ngược: *"42.7% trượt nuốt sạch cái lợi"* → **cũng sai**;
   oracle vẫn thắng đếm. Lỗi: lấy **unigram entropy TOÀN CỤC** (~11 bits) áp cho phân bố
   **lệch** — token hay gặp thì rẻ hơn nhiều.
   *(Dấu vết của 42.7% mù vẫn có thật: `λ*` của oracle = **0.5**, không phải 0.95 — kể cả chấm
   hoàn hảo vẫn nên tin unigram MỘT NỬA.)*

**Sổ nợ mới:** oracle@64 = 1.345 → oracle@256 = **1.109** ⇒ **chưa bão hoà**; k lớn hơn còn lợi.
Một **scorer HỌC ĐƯỢC** sẽ rơi giữa 1.623 (cosine) và 1.109 (oracle) — có vượt 1.288 không thì
**chưa ai biết**. Rẻ hơn G7 nhiều, nhưng §7.8f-oracle nói rõ: **kể cả thắng, nó KHÔNG cho R1.**

#### 7.8i. ĐỐI CHỨNG NGẪU NHIÊN — `recall` một mình VÔ NGHĨA (và nó hạ bệ oracle)

`recall@k` không có đối chứng thì không đọc được: với k lớn + Zipf, gold (nhất là token phổ
biến) lọt vào top-k **DO MAY**. Đo lại với đối chứng = k hàng xóm **ngẫu nhiên**:

| k | recall@k | **NGẪU NHIÊN** | **lift** |
|---:|---:|---:|---:|
| **1** | 15.0% | **0.2%** | **98×** |
| 16 | 30.1% | 2.7% | 11.2× |
| 64 | 43.3% | 10.6% | 4.1× |
| **256** | 57.3% | **26.5%** | **2.2×** |

**Hai kết luận ngược nhau, cả hai đều quan trọng:**

1. **Truy hồi là THẬT** — `lift@1 = 98×`. Biểu diễn thô (túi decay) **vẫn mang tín hiệu mạnh**.
   `ParentStore` + `E` (PPMI-SVD) đang **thực sự tìm đúng**. Đây là bằng chứng nửa "trí nhớ"
   của dự án **hoạt động**.
2. **Nhưng tín hiệu SUY THOÁI nhanh theo k**: 98× → 2.2×. **26.5%/57.3% ≈ gần MỘT NỬA
   recall@256 là rác Zipf.**

> **⇒ ORACLE (§7.8f-oracle) BỊ THỔI PHỒNG — kết luận *"hàng xóm đủ tốt, chỉ hàm chấm hỏng"*
> SẬP.** Oracle "chọn đúng trong 256 ứng viên", mà ~½ số lần ứng viên đúng đó **lọt vào do
> may**. Nó ăn gian bằng chính cái đuôi rác: nó không đo *truy hồi tốt cỡ nào*, nó đo *gold có
> phải token phổ biến không*.

#### 7.8h. ĐỌC TAY 4 ca — thứ 3 phép đo bits KHÔNG lật nổi

```
▶ ngữ cảnh: "…cẩn thận quan sát để tìm ra tất"        GOLD: 'cả'
  cosine top-5: "ra cho"→'nhà' · "tạo ra cho"→'nhân' · "khám phá ra"→'dòng' …
  hàng xóm ĐÚNG: HẠNG 255/256

▶ ngữ cảnh: "…Hãy"                                     GOLD: 'chắc'
  cosine top-5: "Hãy"→'xa' · "Hãy"→'mô' · "Hãy"→'chỉnh' …  (sim=1.000 — HOÀ TUYỆT ĐỐI)
  hàng xóm ĐÚNG: HẠNG 255 (sim=0.819) → KHÔNG phải ngữ cảnh "Hãy" ⇒ lọt vào DO MAY
```

Ca 1 chỉ thẳng vào gốc: `"tất cả"` là từ ghép **cố định**, token cuối quyết định gần như tuyệt
đối. Mà `decay=0.7, L=8` cho trọng số `1, 0.7, 0.49, 0.34…` **tổng ≈ 3.3** ⇒ `E[tất]` chỉ chiếm
**~30%** vector ⇒ **7 token còn lại dìm chết tín hiệu mạnh nhất**. **Biểu diễn tự bóp chết thứ
nó đang cầm** — không phải lỗi hàm chấm. Và `decay=0.7` là con số **chọn tuỳ tiện, không đo**.

Ca 2 (`"Hãy"`) là thứ phát hiện ra chuyện oracle ăn gian: hàng xóm "đúng" ở hạng 255 **không
liên quan gì** tới ngữ cảnh. → dẫn thẳng tới đối chứng ngẫu nhiên §7.8i.

**4 ca đọc tay lật được cái mà 3 phép đo bits không lật nổi** — đúng lớp phòng thủ đã bắt bug #3
của TK_041 (§5.2).

#### 7.9. G9 — SỬA BIỂU DIỄN (`scripts/probe_ctx_repr.py`), rẻ, runtime, không train

| decay | L | recall@1 | ngẫu | **lift@1** | recall@16 | **bits/byte** (k=16) |
|---:|---:|---:|---:|---:|---:|---:|
| 0.0 | 1 *(chỉ token cuối)* | 7.0% | 0.24% | 28.6× | 28.6% | 1.714 |
| **0.3** | 8 | **19.5%** | 0.24% | **80.1×** ★ | **38.3%** | **1.571** ★ |
| 0.5 | 8 | 18.7% | 0.24% | 76.8× | 36.2% | 1.604 |
| **0.7** | 8 *(G8 dùng — chọn BỪA)* | 15.0% | 0.24% | 61.4× | 30.1% | 1.702 |
| 0.9 | 8 | 9.6% | 0.24% | 39.2× | 22.5% | 1.815 |
| 0.5 | 2 | 14.2% | 0.24% | 58.2× | 35.3% | 1.613 |

**Ba kết luận:**

1. **Dự đoán *"chỉ token cuối sẽ thắng túi"* — SAI.** Token cuối một mình = **28.6×**; túi 8
   token `decay=0.3` = **80.1×**. ⇒ **Cộng vector CÓ lấy được ngữ cảnh dài** (kết luận đắt mà
   tác giả ticket dự đoán — **không xảy ra**).
2. **`decay=0.7` chọn bừa đang bỏ phí 30% lift** (61.4→80.1) **+0.131 bits/byte** — miễn phí,
   chỉ bằng vặn một tham số. *(Bài học: đừng để tham số tuỳ tiện nằm trong đường đo.)*
3. **NHƯNG 1.571 vẫn THUA đếm 1.288 (+0.283).** Biểu diễn tốt nhất **vặn tay được** vẫn không
   đủ.

> **⇒ §7.8d ĐÚNG NGAY TỪ ĐẦU: nút thắt là BIỂU DIỄN NGỮ CẢNH.** Và nay có bằng chứng **BA
> TẦNG** — bits · đối chứng ngẫu nhiên · đọc tay. Vặn tay biểu diễn cho **lợi thật nhưng có
> trần**; muốn vượt phải để **hàm HỌC** quyết token nào đáng chú ý ở **từng vị trí** — đó là
> attention, và không công thức viết tay nào thay được.
>
> `decay` là tham số **tuỳ tiện về bản chất**: không có giá trị nào đúng, vì câu trả lời đúng
> **khác nhau ở mỗi vị trí** (`tất` cần decay≈0; *"ảnh hưởng tích cực và tiêu cực của ___"* cần
> nhìn xa). **→ về G7.**

#### 7.8f. Giá trị + sổ nợ

**20 phút, tách được BA việc trước đó lẫn vào nhau** (tìm ứng viên / chấm điểm / trộn), và lật
được một dự đoán sai. Kết luận **thu hẹp về G7**, không mở nhánh mới: dư địa nằm ở **hàm chấm
ngữ cảnh học được** = attention = **phải học**.

**Sổ nợ — đo được, rẻ, chưa làm:**

* **Trần trên của hàm chấm:** `recall@256 = 57.3%` ⇒ nếu chấm HOÀN HẢO (luôn chọn đúng hàng xóm
  có token đúng) thì bits/byte = bao nhiêu? Đó là **trần của kNN với biểu diễn HIỆN TẠI**, và
  nó nói dư địa của "sửa hàm chấm" **tách khỏi** dư địa của "sửa biểu diễn". Nếu trần đó đã
  thấp hơn 1.288 nhiều ⇒ hàm chấm là đòn bẩy thật; nếu chỉ nhỉnh hơn ⇒ biểu diễn mới là gốc.
* **Datastore đầy** (~3M thay vì 400k) — xem §7.8e.
* `recall@k` bão hoà ở k nào (chưa bão hoà ở 256).

### 7.10. SPEC — G10: THAY KÊNH `W_seq` BẰNG LM (nếu G7 dương)

> ⚠️ **Toàn bộ mục này nằm SAU G7.** Nếu G7 cho `gap ≈ 0` ⇒ `P_LM ≈ P_đếm` ⇒ thay một kênh
> bằng cái y hệt nó **không đổi gì** ⇒ **A đúng**, xoá mục này. **Không code trước khi có số G7.**

#### 7.10a. Ý tưởng — phẫu thuật MỘT kênh, không viết lại đường sinh

`generate_autoregressive` hiện là (README §0.4.1):
```
scores = W_seq ⊙ macro ⊙ FRAME_BRACKET ⊙ skip     (4 kênh Hadamard)
         + agenda + index_backbone + EOS
```
**`W_seq` là kênh TRÔI CHẢY — và nó CHÍNH LÀ thứ 6.4 bits** (§0.1b). Nó là mắt xích yếu nhất.
Thay đúng nó:
```
scores = P_LM(next | 256 token) ⊙ macro ⊙ FRAME_BRACKET ⊙ skip
              ↑ ~3 bits?                    ↑ agenda vẫn lái NỘI DUNG
```

#### 7.10b. Vì sao cách này, không phải RAG nhét-ngữ-cảnh

| Phương án | Rủi ro chính |
|---|---|
| **1. Nhét `[FACT]` vào ngữ cảnh** (RAG ngây thơ) | LM train trên corpus TRẦN **không có lý do gì** chú ý khối `[FACT]` — với nó đó là text thường. **Sẽ không chạy.** |
| **2. Train CÓ truy hồi trong input** (RAG thật) | Cách đúng để (1) chạy. Nhưng **in-context learning nổi lên theo QUY MÔ** — LM **11M tham số** gần như chắc chắn KHÔNG có nó |
| **3. Trộn phân bố** `λ·P_LM + (1−λ)·P_đồ_thị` | Đã đo `P_đồ_thị`: tốt nhất **1.571**, thua đếm (§7.9). Trộn LM tốt với phân bố yếu ⇒ lợi có hạn. *(kNN-LM thắng vì datastore của nó từ **hidden state ĐÃ HỌC**, không phải túi embedding.)* |
| **4. Thay kênh `W_seq`** ← **chọn** | Xem §7.10c |

**Điểm mấu chốt của (4):** agenda lái nội dung **TỪ BÊN NGOÀI** (qua kênh macro) ⇒ **LM không
cần "hiểu" khối FACT** ⇒ **không phụ thuộc in-context learning**. Đó là lý do nó hợp với LM nhỏ,
trong khi (1)/(2) thì không.

#### 7.10c. Được gì

* **Giữ TẤT CẢ thứ đã hoạt động**: truy hồi (lift@1 **98×** §7.8i), FRAME_BRACKET, skip, EOS-lift,
  hub, theme, Luật 0 — **không đụng**.
* **Thay đúng mắt xích yếu nhất** (`W_seq` = 6.4 bits, đo được).
* **Trả lời trực tiếp "vì sao chép"**: hiện 4 kênh **đồng thanh** chỉ về đoạn đã lưu (đo: tắt
  backbone → **99% token y hệt**, §0.1a). Có LM thật, nó sẽ **BẤT ĐỒNG** với agenda ở chỗ nó
  biết ngữ cảnh — **và chỗ bất đồng đó CHÍNH LÀ nơi văn phong sinh ra**.
* Phẫu thuật **một kênh**.

#### 7.10d. ⚠️ Rủi ro ĐÃ BIẾT — Hadamard là phép AND

Nhân kênh: LM nói `X`, agenda nói `Y` ⇒ tích **dìm CẢ HAI** ⇒ chọn ra token `Z` tầm thường.
Bệnh kinh điển của kiến trúc nhân-kênh; có thể cho kết quả **TỆ HƠN cả hai thành phần**.

**Chặn:** dùng **nội suy log-space** thay nhân — `α·log P_LM + β·log P_agenda`, sweep α/β.
Biến "AND" thành "thoả hiệp". Rẻ.

#### 7.10e. Đo

| | |
|---|---|
| **Thước chính** | `scripts/eval_bits.py` — bits/byte, CÙNG held-out |
| **Thước sinh** | `copy` (LCS với corpus) — §0.4.2: `conv` 0.22 = sàn, `history_wiki_para` 0.57 = chép nửa. **`copy` PHẢI tụt** trong khi bits không xấu đi |
| **Guard** | QA/Para/Syn 86/88/67 + rank-1 76.3% **không được tụt** |
| **Đọc tay** | BẮT BUỘC — `copy` tụt mà văn thành word-salad thì là thua, không phải thắng (tiền lệ: `vb_seg` 100→4% "đạt acceptance NHƯNG đọc tay = verbatim-xáo-đầu-segment", v4 §6) |

#### 7.10f. Điều kiện mở

**G7 `gap ≥ 0.5` VÀ Δ < −0.1 bits/byte.** Không đủ hai ⇒ không mở G10.

### 7.11. 🔴 G10 ĐÃ LÀM → **TỰ GIẾT MÌNH**: kênh micro KHÔNG NỐI VỚI ĐẦU RA

> Phát hiện quan trọng nhất của TK_042. Nó phủ định §7.10, và **giải thích luôn 8 cơ chế
> trung tính** ở tầng sâu hơn mọi giả thuyết trước.

#### 7.11a. Đo (`network_mini_bpe`, LM `out/g7_best_model.pt`, vocab khớp 32 002)

| Thí nghiệm | Token trùng với baseline |
|---|---:|
| Tắt `ar_index_backbone` (§0.1a) | **99%** |
| **Thay `W_seq` → LM** (`ar_lm_weight=1.0`) | **100%** |
| **GIẾT HẲN micro** (mọi ứng viên điểm BẰNG NHAU) | **100%** (78/78) |

Đã xác minh LM **thật sự chạy**: `build_micro_dict` gọi 20/20 bước, `state.neural_lm=NeuralLM`,
`ar_lm_weight=1.0`. Nó **được gọi mà không đổi được gì**.

#### 7.11b. Kết luận — §7.10 SAI Ở TIỀN ĐỀ

> **`agenda`/`macro` quyết định TOÀN BỘ đầu ra. Kênh trôi chảy KHÔNG quyết định một token nào.**
>
> §7.10 cho rằng `W_seq` là *"mắt xích yếu nhất"*. Nó **KHÔNG PHẢI MẮT XÍCH** — nó **không nằm
> trong dây chuyền**. Thay `W_seq` (7.088 bits) bằng LM (5.594 bits, tốt hơn **1.494**) đổi
> **0 token** — không phải vì LM dở, mà vì **cái khe ta lắp nó vào không nối với đầu ra**.

**Ba cơ chế (agenda · backbone · micro) QUÁ THỪA** — gỡ cái nào cũng không đổi gì vì cả ba
cùng chỉ một hướng. **v4 đã viết đúng điều này từ 2026-07-01 mà chưa ai truy đến cùng:**
*"A3 bị index-backbone + agenda + spreading chi phối; micro chỉ phá thế hoà — mà các ca này
không có thế hoà."*

#### 7.11c. ⇒ LỜI GIẢI THẬT cho "8 cơ chế trung tính" (thay §3)

§3 nói: *"cả 8 đều là đếm thêm, mà đếm không điều-kiện-hoá ngữ cảnh dài"* — **đúng nhưng
CHƯA ĐỦ**. Lý do sâu hơn:

> **CẢ 8 CƠ CHẾ ĐỀU CẮM VÀO `micro`** (order-2, VarLM/PPM, KN smoothing, skip-bridge…). Mà
> `micro` **không nối với đầu ra**. Chúng trung tính **không phải vì yếu — mà vì bị NGẮT MẠCH.**

Và **"verbatim copy" ở tầng sâu nhất**: đầu ra **KHÔNG đến từ mô hình ngôn ngữ nào cả** — nó là
**agenda được ĐỌC RA**. Đó là lý do mọi cải tiến kênh micro sẽ đo ra trung tính **MÃI MÃI**.

#### 7.11d. Việc đúng: G6 (cai nghiện agenda) phải làm TRƯỚC G10

Hạ `ar_agenda_boost` / `ar_index_backbone` / `ar_macro_floor` **cho tới khi micro có tiếng
nói** (đo: giết micro ⇒ output PHẢI đổi), **rồi mới** hỏi LM có giúp không. Thứ tự cũ (G10
trước G6) **bất khả về mặt logic**.

⚠️ Cai nghiện agenda sẽ làm **chất lượng TỤT trước khi lên** — agenda đang là thứ duy nhất giữ
nội dung đúng (v4: bỏ agenda → word-salad). Phải có LM **đủ tốt** đỡ ngay khi buông. Đó mới là
G10 thật, và nó **khó hơn** §7.10 nhiều.

#### 7.11e. Bug tôi mắc khi làm G10 (lỗi im lặng thứ 8)

Chèn code LM vào **nhánh `else`** của `generate_autoregressive`. Nhưng
`ar_candidate_scoring=True` (mặc định) ⇒ đường THẬT chạy nhánh **sparse** ⇒ **code LM CHƯA BAO
GIỜ CHẠY**: `ar_lm_weight` 0.0/0.5/1.0 cho output **giống hệt nhau từng byte**. Không crash,
không test đỏ, vẫn ra chữ — **tính năng đơn giản là KHÔNG TỒN TẠI**.

**Bắt được vì thử `w=1.0`** (kỳ vọng phải đổi NHIỀU). Nếu chỉ thử `w=0.5` thì "đổi ít" rất dễ
tự lừa là *"có tác dụng nhẹ"*. → **Bài học: test cờ ở GIÁ TRỊ CỰC ĐOAN, nơi im lặng là bất
khả.** Vá: dời vào `ar_candidates.build_micro_dict` (kênh micro THẬT).

*(Mỉa mai: chính bug này dẫn tới phát hiện §7.11b — vì để chứng minh code đã chạy, tôi phải
giết hẳn micro, và lúc đó mới lộ ra micro vốn đã vô dụng.)*

### 7.12. G6 — CAI NGHIỆN AGENDA: **các núm KHÔNG mở được van nào**

> `scripts/probe_agenda_wean.py`. Thước: **micro-influence** = % token đổi khi giết micro.
> `0%` = micro vô dụng.

#### 7.12a. Sweep — mọi cấu hình đều 0%

| Cấu hình | micro-influence |
|---|---:|
| HIỆN TẠI (mặc định) | **0.0%** |
| `−backbone` | 34.6% ⚠ *(xem §7.12c — KHÔNG phải micro thắng)* |
| `−backbone −boost` | **0.0%** |
| `macro_floor` 0.3 / 1.0 / **3.0** | **0.0%** |
| `spread_weight` 0.2 | **0.0%** |
| **agenda TẮT HẲN** (`spread_weight=0`) | **0.0%** |

#### 7.12b. Phép thử QUYẾT ĐỊNH — micro ĐẢO NGƯỢC

"Giết micro → 0% đổi" có **HAI** cách giải thích, và §7.11 vội chọn một:

| | |
|---|---|
| **(a)** micro **bất lực** về số học | ← §7.11 khẳng định |
| **(b)** micro **ĐỒNG Ý** với macro nên bỏ đi cũng thế | ← chưa loại trừ |

*(Giả thuyết (b) rất hợp lý: `W_seq` học "Trần→Thủ" **TỪ CHÍNH câu đó**, mà macro lan từ agenda
**CHÍNH LÀ câu đó** ⇒ hai kênh tính từ **CÙNG một dữ liệu** ⇒ tất nhiên đồng ý.)*

**Phân biệt: cho micro tín hiệu NGƯỢC HẲN** (successor `W_seq` TỐT NHẤT → điểm THẤP NHẤT):

```
micro san PHẲNG  → đổi 0.0%
micro ĐẢO NGƯỢC  → đổi 0.0%     ← kể cả ở macro_floor=3.0
```

⇒ **(b) BỊ LOẠI. §7.11 ĐÚNG: micro bất lực về số học.** Nó **không đồng ý — nó không có quyền.**

#### 7.12c. Và các con số bác bỏ mọi cách giải thích "tầm thường"

| | |
|---|---|
| `\|ứng viên\|/bước` | **~10 100** (median 9 875) ⇒ **KHÔNG** phải "chỉ 1 ứng viên" |
| khe top1/top2 | **2.98×** ⇒ **KHÔNG** phải hoà |
| `macro_floor=3` | khe **2.98→1.62×** ⇒ núm **CÓ tác dụng** lên điểm… |
| …nhưng micro-influence | **vẫn 0%** ⇒ …mà **KHÔNG lật được winner** |

`assemble_ar_scores` **có** dùng `ar_macro_floor`/`ar_spread_weight` (đã đọc code). Không phải
nhánh chết. **Các núm hiện có KHÔNG phải van cai nghiện.**

#### 7.12d. ⚠️ ĐÍNH CHÍNH §0.1a — "tắt backbone → 99% y hệt" là do MẪU QUÁ NHỎ

§0.1a kết luận *"backbone KHÔNG phải thủ phạm"* từ **3 prompt** (74/75 = 99%). Thêm prompt
`'Khúc Thừa Dụ'` thì `−backbone` cho **word-salad**:
```
mặc định   : ' Dương Đình Nghệ.\nNgô Quyền độc lập năm 938.'
−backbone  : '.\n Dương Đình ngày NghệNgô Quyền. một độc lập năm 1 9 Hoàn 2'
```
⇒ **backbone CÓ tác dụng** — §0.1a chỉ không lấy trúng ca cần. *(Lỗi mẫu-quá-nhỏ của tác giả
ticket; tái lập được, không phải rò state.)*

Và một quan sát ngược đời **chưa giải thích được**: `−backbone` → salad, nhưng `−backbone
−boost` (cai **NHIỀU HƠN**) → **sạch, trùng mặc định 100%**. Tức `agenda_boost` **một mình,
không có backbone** mới là thứ gây salad — nó đẩy token agenda **sai thứ tự**. **Sổ nợ.**

#### 7.12e. ⇒ Kết luận G6: **KHÔNG cai được bằng cờ**

Không cấu hình nào trong không gian cờ hiện tại cho micro tiếng nói. **Cai nghiện agenda KHÔNG
phải việc chỉnh tham số — nó là việc VIẾT LẠI cách hợp kênh.** `micro` bị nhân với một macro
có dải động áp đảo, ở **mọi** giá trị floor đã thử.

**Việc đúng (chưa làm):** thay công thức hợp kênh — `α·log P_LM + β·log P_macro` (log-space,
sweep α/β) thay vì `(floor+micro)·(floor+macro)`. Đó là **G11**, và nó là thay-đổi-code, không
phải đổi-cờ. Chỉ đáng làm nếu tác giả chấp nhận đụng vào lõi `assemble_ar_scores`.

### 7.13. 🔴 G11 → **TÌM RA THỦ PHẠM: `ar_index_backbone`. VÀ NÓ LÀ THỨ §0.1a ĐÃ MIỄN TỘI.**

#### 7.13a. Đo — 8 tổ hợp cổng (`network_mini_bpe`)

| FB gate | G4 verify | **backbone** | micro-influence |
|:---:|:---:|:---:|---:|
| ON | ON | **ON** | **0.0%** |
| ON | ON | **off** | **34.7%** |
| ON | off | **ON** | **0.0%** |
| ON | off | **off** | **34.7%** |
| off | ON | **ON** | **0.0%** |
| off | ON | **off** | **32.7%** |
| off | off | **ON** | **0.0%** |
| off | off | **off** | **32.7%** |

> **`backbone ON` ⇒ 0.0% ở CẢ BỐN. `backbone off` ⇒ 33–35% ở CẢ BỐN.**
> **`ar_index_backbone` là thứ DUY NHẤT ghim đầu ra.** FB gate và G4 verify **không liên quan
> gì** — bật/tắt không đổi một phần trăm nào.

#### 7.13b. Code — hiển nhiên đến mức đau

```python
scores[t] = max(base, cur_max) * (1.0 + config.ar_index_margin)   # generate_hebbian
```
`t` = **token kế theo trật tự CHA NGUỒN**. Nó được gán bằng **max hiện tại × 1.5**.

> **Đây KHÔNG phải một kênh — nó là LỆNH GHI ĐÈ.** Không kênh nào thắng nổi một token được
> đặt bằng `max × 1.5`. `ar_index_backbone: bool = True` — **bật mặc định**.

#### 7.13c. ⇒ "VERBATIM COPY" KHÔNG PHẢI TRIỆU CHỨNG. NÓ LÀ **ĐẶC TẢ**.

`ar_index_backbone` làm **đúng những gì tên nó nói**: *"token kế tiếp theo trật tự nguồn → cho
vượt trội"* = **chép, theo đúng nghĩa đen**. Nó không hỏng. Nó đang chạy đúng thiết kế.

#### 7.13d. Mọi quan sát của TK_042 nay khớp — bằng MỘT nguyên nhân

| Quan sát | Lời giải |
|---|---|
| Giết micro → 0% đổi (§7.11) | backbone ghi đè |
| **Đảo ngược** micro → 0% đổi (§7.12b) | backbone ghi đè |
| Thay `W_seq`→LM (**tốt hơn 1.494 bits/token**) → **0 token** (§7.11a) | backbone ghi đè |
| `macro_floor=3` nén khe 2.98→1.62× mà không lật winner (§7.12c) | backbone ghi đè **SAU** đó |
| **8 cơ chế trung tính** (§3) | tất cả cắm vào `micro` — backbone ghi đè hết |
| **"verbatim copy"** | **§7.13c — đó là đặc tả** |

#### 7.13e. ⚠️ LỖI PHƯƠNG PHÁP NẶNG NHẤT CỦA TK_042 — của tác giả ticket

§0.1a mở đầu ticket bằng:
> *"Backbone KHÔNG phải thủ phạm — tắt nó → 99% token y hệt. **Chẩn đoán đầu của tôi sai; phép
> đo cứu.**"*

**Chẩn đoán đầu ĐÚNG. PHÉP ĐO mới sai** — 3 prompt, không cái nào chạm chỗ backbone quyết
định. **Tôi tự bác bỏ mình bằng một phép đo tồi, rồi xây TOÀN BỘ TK_042 lên trên phép bác bỏ
đó** (§2 "truy hồi làm thay điều-kiện-hoá", §3 "8 cơ chế đều là đếm", §7.10 "thay W_seq"…).

**Bài học — khác với 8 bug im lặng trước:** những cái kia là *code không chạy*. Cái này là
**code chạy, phép đo chạy, số đúng — mà KẾT LUẬN sai vì MẪU không phủ ca quyết định.** Một
A/B "99% giống nhau" trên 3 prompt **không** là bằng chứng miễn tội; nó chỉ là *"3 prompt này
không phân biệt được"*. → **A/B phải chọn mẫu ở nơi cơ chế ĐƯỢC KỲ VỌNG cắn, không phải nơi
tiện tay.**

#### 7.13f. Việc tiếp — và §7.6 phải viết lại

Câu hỏi **không còn** là "B hay C". Nó là: **gỡ `ar_index_backbone` thì còn lại gì?**

* `backbone off` ⇒ micro có **33–35%** tiếng nói ⇒ **G10 (kênh LM) nay CÓ THỂ đo được** —
  lần đầu.
* Nhưng `backbone off` ⇒ **word-salad** ở `'Khúc Thừa Dụ'` (§7.12d) ⇒ backbone đang **che**
  một lõi sinh rỗng. Gỡ nó = phơi ra sự thật, **chất lượng TỤT trước khi lên**.
* **G7 đã chứng minh LM đủ tốt để đỡ** (1.007 vs đếm 1.276, Δ −0.269) — nhưng **chưa ai thử
  `backbone off` + `LM on` cùng lúc**. Đó là thí nghiệm tiếp theo, và nay nó **có nghĩa**.

### 7.14. ĐỌC TAY CHÍNH LM — phép đo quan trọng nhất, và tôi bỏ sót nó suốt

> Rao giảng "đọc tay" cả ticket rồi **chưa bao giờ đọc tay chính cái LM**. Chỉ đo bits/byte
> (1.007) — mà bits **không nói** LM sinh ra cái gì.

#### 7.14a. LM MỘT MÌNH, không đồ thị (temp=0.8)

```
'Con chó' — 3 seed, 3 KẾT QUẢ KHÁC NHAU:
  seed0: 'Con chó quái hổ kia là nhảy lên chớ đi!"'
  seed1: 'Con chó sủa cùng những đứa con gái đã học xong, cậu ấy rất khỏe'
  seed2: 'Con chó là con chó, bò cổ, hay có con chó. Vì vậy để nuôi chó là mèo, chó yêu'
                                                                            ↑ tự đổi chó↔mèo
'Thủ đô Việt Nam là' → '…là tỉnh Nam Cộng hòa, tỉnh Nam.'      ← BỊA TRẮNG TRỢN
```

**Đó CHÍNH XÁC là R1/R2/R3/R4 (§1):** đa dạng theo seed · không chép · ngữ pháp đúng · tự đổi
`chó`↔`mèo`. **Và nó không biết gì cả.**

#### 7.14b. ⇒ Phân công được ĐO, không còn là lập luận (xác nhận §6.1 CLS)

| | Đồ thị (hồi hải mã) | LM (vỏ não) |
|---|:---:|:---:|
| Sự kiện đúng | ✅ QA 86–100% · lift@1 98× | ❌ **bịa** |
| Trôi chảy, văn phong | ❌ **chép** (§7.13c: đặc tả) | ✅ |
| **Đa dạng** (đổi seed → đổi văn) | ❌ | ✅ |

**Bù nhau CHÍNH XÁC.** Đây là bằng chứng thực nghiệm cho §6.1, không phải loại suy.

#### 7.14c. 🔴 RAG-bằng-prompt: ÂM TÍNH, dứt khoát

Nhét fact vào ngữ cảnh ngay trước câu hỏi:
```
A. không fact         : 'Thủ đô Việt Nam là tỉnh Nam Cộng hòa'      ← sai
B. fact ĐÚNG (Hà Nội) : 'Thủ đô Việt Nam là đô thị nông thôn'       ← KHÔNG nói "Hà Nội"
C. fact SAI  (Huế)    : 'Thủ đô Việt Nam là tỉnh của tỉnh Bến Tre'  ← cũng KHÔNG nói "Huế"
```
**B ≡ C ⇒ LM HOÀN TOÀN KHÔNG ĐỌC fact trong ngữ cảnh.**

⇒ **§7.10b ĐÚNG** *(lần này tác giả ticket đoán đúng TRƯỚC khi đo)*: **11.4M tham số không có
in-context learning** — nó nổi lên theo QUY MÔ.

#### 7.14d. Ba cách nối LM↔đồ thị — đo hai, cụt cả hai

| | Trạng thái |
|---|---|
| **(1)** Nhét `[FACT]` vào prompt | ❌ **ĐO: LM không đọc** (§7.14c, B≡C) |
| **(4)** Kênh `micro` (§7.10) | ❌ **ĐO: backbone ghi đè (§7.13); gỡ backbone thì macro đè (§7.12)** |
| **(2)** Train **CÓ truy hồi trong input** | **CHƯA THỬ — cái DUY NHẤT còn lại** |

#### 7.14e. ⇒ G12: LM đã train FULL — nhưng train SAI CÁCH cho việc ta cần

LM học *"đoán token kế trong văn bản TRẦN"*. Nó **chưa bao giờ được dạy** rằng có khối `[FACT]`
phía trước và phải bám vào. **Nên nó không bám — đúng như nó được train.**

```
Train HIỆN TẠI:  <văn bản>                             → đoán token kế
Train CẦN CÓ:    [FACT] <fact từ đồ thị> [/FACT] <văn bản>  → đoán token kế
```
Khi **MỌI** mẫu train đều có khối FACT chứa đúng thứ đoạn văn sắp nói, LM **buộc phải học** rằng
bám vào đó thì loss thấp hơn. Đó là cách RETRO/RAG hoạt động thật.

**Thứ tự BẮT BUỘC:**
1. **B5** — đồ thị BPE full (~8h). `[FACT]` phải lấy từ đồ thị THẬT, không phải mini 1 294 cha.
2. Dựng data có `[FACT]` — chạy truy hồi trên toàn corpus, **offline**, vài giờ.
3. **Train lại LM** — ~2.3h (đã biết throughput **30 341 token/giây** trên M4 Pro).
4. Đo: `eval_bits` + **đọc tay** + probe phản-sự-thật (fact SAI → LM có nói theo fact không?).

⚠️ **Chưa chắc:** **11.4M tham số có học nổi copy-from-context không — KHÔNG BIẾT.** TinyStories
cho thấy mô hình nhỏ học được *cấu trúc*; copy-from-context là **năng lực khác**. Nếu không học
nổi ⇒ phải tăng tham số ⇒ quay lại bài toán **data** (99MB là trần, không phải M4 — §8.1).

#### 7.14f. ⚠️ TRẦN CỦA MỌI SỐ TRONG §7.11–§7.13

**Toàn bộ chuỗi G6/G10/G11 đo trên `network_mini_bpe` — 1 294 cha.** Memory dự án ghi rõ:
*"ckpt nhỏ đánh lừa NGƯỢC DẤU"*. Và nó đã đánh lừa: `backbone OFF + boost 0 + spread 0` → verbatim
**sạch**, còn `backbone OFF + boost 0.6 + spread 1.0` → **salad** — cai **ÍT hơn** lại **TỆ hơn**.
Tác giả ticket đoán 3 lần, **sai cả 3**.

| Đứng vững (cấu trúc — không phụ thuộc checkpoint) | KHÔNG đứng vững (cần B5) |
|---|---|
| `ar_index_backbone` = thủ phạm (2 đo độc lập + đọc code) | mọi số **chất lượng** trên mini |
| "verbatim" là **ĐẶC TẢ** (§7.13c) | *"LM có cứu được không"* |
| §0.1a **miễn tội nhầm** backbone (§7.13e) | thứ tự cai nghiện đúng |
| **G7 dương tính** (corpus THẬT, không phải mini) | |
| **§7.14a–c** (LM sinh gì · RAG âm tính — không dùng đồ thị) | |

### 7.15. G13 — **kNN-LM: đồ thị KHÔNG cần học nói tiếng transformer; cho LM ĐỌC HỘ**

> Xuất phát từ câu hỏi tác giả 2026-07-17: *"làm sao để đồ thị nói được ngôn ngữ của
> transformer?"* — sau khi §7.14e chốt rằng đồ thị **không cắm vào chỗ FFN được** (khoá của nó
> là **id token**, không sống trong không gian ẩn `h ∈ R²⁵⁶`).

#### 7.15a. Ý — đảo ngược câu hỏi

Đừng dạy đồ thị nói. **Để LM phiên dịch hộ** — nó biến **cả hai đầu** về cùng một không gian:

```
mỗi vị trí trong mỗi nơ-ron cha
      ↓  cho LM chạy qua
  h ∈ R²⁵⁶   →   token thật sự đứng sau
      ↓
datastore (h, next) — CÙNG không gian với h lúc sinh
      ↓
lúc sinh:  P = λ·P_LM + (1−λ)·P_kNN(h)
```

**Không train gì.** Đồ thị giữ nguyên vai **kho sự thật nguyên văn**; LM làm **phiên dịch viên**.
Đây chính là **kNN-LM** (Khandelwal 2020) — công thức chuẩn trong văn liệu, không phải sáng chế.

#### 7.15b. Đo — `h` transformer vs túi PPMI làm KHOÁ TRUY HỒI

**CÙNG `network_mini_bpe` · CÙNG datastore 30k · CÙNG 400 query — chỉ khác KHOÁ:**

| Khoá | recall@1 | ngẫu nhiên | **lift@1** |
|---|---:|---:|---:|
| túi PPMI-SVD (d=96, decay 0.3) | 7.5% | 0.75% | **10×** |
| **`h` transformer (d=256)** | **11.0%** | 0.50% | **22×** |

⇒ **`h` tốt hơn túi PPMI ~2.2×.** Đó là **biểu diễn ngữ cảnh ĐÃ HỌC** — đúng thứ §7.8d/§7.9
kết luận là nút thắt. Lần đầu ta có nó.

⚠️ **Suýt lặp lại lỗi so-khập-khiễng của chính ticket này:** bản đầu so `22×` (mini, datastore
30k) với `98×` (§7.8i — **v10_histwiki, datastore 400k, baseline ngẫu 0.2%**). **KHÔNG so được**
— khác checkpoint, khác cỡ datastore, khác baseline. Bảng trên là bản đo lại **công bằng**, và
nó **lật kết quả**: trên cùng điều kiện, PPMI chỉ được **10×**, không phải 98×.

#### 7.15c. Vì sao G13 khác G12 — và rẻ hơn nhiều

| | **G12** (`[FACT]` trong input) | **G13** (kNN-LM) |
|---|---|---|
| Train lại LM | **CẦN** (~2.3h) + dựng data | **KHÔNG** |
| Cần B5 | **CẦN** (fact phải từ đồ thị thật) | nên có, nhưng chạy được ngay trên mini |
| Cần in-context learning | **CÓ** ← thứ 11.4M **KHÔNG có** (§7.14c đo được) | **KHÔNG** — trộn ở tầng phân bố, LM không cần "hiểu" |
| Có sẵn | LM ✓ | LM ✓ · parents ✓ · **chỉ cần encode** |

**G13 là con đường THỨ BA mà §7.14d bỏ sót** khi liệt kê (chỉ nêu: nhét-prompt ❌ · kênh-micro ❌
· train-có-fact). Nó tránh được **cả hai** bức tường đã đo:
* **không** cần in-context learning (§7.14c) — trộn ở tầng **phân bố**, không ở tầng ngữ cảnh
* **không** đi qua kênh `micro` ⇒ **không bị `ar_index_backbone` ghi đè** (§7.13)

#### 7.15d. Vì sao nó khớp với §7.14b (CLS)

Bài kNN-LM tên là ***"Generalization through Memorization"***: gắn **bộ nhớ tường minh** vào LM
giảm perplexity đáng kể, và **lợi NHẤT ở sự kiện hiếm / tên riêng / tri thức cụ thể** — tức
**đúng chỗ LM của ta đang bịa** (*"Thủ đô Việt Nam là tỉnh Nam Cộng hòa"*, §7.14a).

⇒ **kNN-LM là CÁI KHỚP NỐI** giữa "LM cho văn phong" và "đồ thị cho sự thật" — và nó là khớp
nối **đã được chứng minh**, không phải suy đoán của ticket này.

#### 7.15e. Ba cảnh báo

1. **`22×` là YẾU.** kNN-LM thật dùng LM **250M+ tham số / hàng chục tỉ token** — `h` của họ là
   biểu diễn chất lượng cao. `h` của ta (**11.4M / 19.7M token**) chỉ hơn túi-PPMI **2.2×**.
   Có thể **chưa đủ** để kNN-LM phát huy.
2. **Đo trên mini** (1 294 cha, datastore 30k) — checkpoint mà memory dự án cảnh báo **đánh lừa
   NGƯỢC DẤU** (§7.14f). Cần **B5**.
3. **Chưa đo bits/byte** — mới đo `lift@1` (chất lượng KHOÁ). Câu hỏi thật: `λ·P_LM +
   (1−λ)·P_kNN` có **thắng `P_LM` một mình (1.007)** không? **Chưa biết.**

#### 7.15f. Việc — rẻ, không train

1. Dựng datastore `(h, next)` từ parents bằng LM đã có.
2. `P = λ·P_LM + (1−λ)·P_kNN`, sweep λ · k · nhiệt độ khoảng cách.
3. Đo bằng **chính `eval_bits.py`**, cùng held-out. **Đích: < 1.007** (P_LM một mình).
4. **Probe phản-sự-thật:** datastore chứa *"Thủ đô = Hà Nội"* → LM có thôi bịa *"tỉnh Nam Cộng
   hòa"* không? Đây mới là phép đo **thật sự trả lời câu hỏi của dự án**.

### 7.16. "IDF pha loãng theo độ dài cha" — **KHÔNG TỒN TẠI** (và lộ ra thứ khác)

> Câu hỏi tác giả 2026-07-17: *"đoạn văn dài đang là một nơ-ron to — có nên chia nhỏ?"*
> Giả thuyết nền: cha dài ⇒ IDF **pha loãng** ⇒ truy hồi kém.

#### 7.16a. Đọc code — mẫu số là CÂU HỎI, không phải cha

```python
coverage = covered / (eff_idf_total or 1.0)   # eff_idf_total = IDF của QUERY
score = coverage                              # KHÔNG chuẩn hoá theo độ dài cha
```
Score đo *"cha này phủ bao nhiêu % **câu hỏi**"*. Thêm token vô can vào cha **không đụng** tử số
lẫn mẫu số.

#### 7.16b. Thí nghiệm có kiểm soát — CÙNG câu, nhúng vào cha dài dần

| cha | #token | score |
|---|---:|---:|
| chỉ câu đó | 9 | **0.0729** |
| + 1 câu | 18 | **0.0729** |
| + 5 câu | 57 | **0.0729** |
| + 15 câu | 151 | **0.0729** |

⇒ **Score BẤT BIẾN theo độ dài. KHÔNG có pha loãng.** ⇒ **TK_010 (*"không chia câu, không chia
đoạn"*) KHÔNG trả giá bằng độ chính xác truy hồi.**

#### 7.16c. Dự án đã đi qua câu hỏi này rồi — và đã bác bỏ

Comment ngay trong `rank_matches`:
> *"Trước: `score = coverage × (0.5 + 0.5 × specificity)` → fact-đúng-**DÀI** (khớp nhiều token
> nhưng loãng) bị thua fact-**SAI-NGẮN**. Giờ `score = coverage`"*

Từng **CÓ** hình phạt độ dài. Nó **gây hại**, đã bị gỡ. ⇒ Giả thuyết *"cha dài bị thiệt"* đã
được **thử, đo, bác bỏ** — trước câu hỏi hôm nay.

#### 7.16d. 🟡 SỔ NỢ — nhưng phép đo lộ ra thứ NGƯỢC LẠI, và nó đáng lo hơn

**Bốn cha đều `0.0729` — HOÀ TUYỆT ĐỐI.** Cha 9 token **đúng trọng tâm** và cha 151 token **tình
cờ chứa câu đó** được điểm **y hệt** ⇒ **winner do THỨ TỰ NGẪU NHIÊN trong list.**

> Vấn đề **không phải** cha dài *bị thiệt*. Là cha dài **KHÔNG BỊ PHẠT GÌ CẢ**, nên nó **ngang
> bằng** cha đúng trọng tâm. `rank_matches` **không có khái niệm "đơn vị nhỏ nhất phủ đủ câu
> hỏi"**.

Đây là điểm yếu THẬT của `rank_matches`, chưa ai ghi. **Đề xuất tác giả (tầng thứ bậc
token→câu→đoạn) chính là chỗ phá thế hoà này bằng CẤU TRÚC**, thay vì bằng hằng số phạt (thứ đã
thử và hại — §7.16c).

⚠️ Nhưng nó **rút ngắn cái bị chép, không dừng việc chép** (`backbone` vẫn copy — §7.13). Và
**cắt câu bằng gì** là vấn đề chưa có lời giải sạch: theo dấu chấm = **danh sách dấu câu quay
lại cửa sau** (vừa gỡ ở TK_041); theo độ dài cố định = cắt giữa mệnh đề; theo EOS-lift = EOS chỉ
đánh dấu **cuối đơn vị train**, không đánh dấu cuối câu **trong** đoạn.

#### 7.16e. So THẬT hai checkpoint (2026-07-18, M4, sau bake) — HOÀ RETRIEVAL, THUA CHẤT LƯỢNG

`scripts/probe_granularity.py` trên **đúng hai checkpoint chỉ khác granularity**:

```
network_v10_histwiki.pkl            151 430 cha  = CHỈ cha-CÂU
network_v10_histwiki_paragraph.pkl  176 430 cha  = + 25 000 cha-ĐOẠN
```

**Cha thắng retrieval GIỐNG HỆT NHAU cho CẢ 8/8 prompt** (23,6,6,8,6,8,7,6 token) — xác nhận
đúng §7.16d: cha-đoạn **không một lần nào** thắng `rank_matches` trước cha-câu ngắn hơn.

Nhưng **đầu ra sinh ra thì khác nhau rõ rệt**, và khác theo hướng **B (+ cha-đoạn) TỆ HƠN**:

| | A (chỉ cha-câu) | B (+ cha-đoạn) |
|---|---:|---:|
| đáp TB | **7.8 tok** | **33.4 tok** |
| copy TB (so cha thắng) | 34% | 4% |
| chạm trần `max_len=40` | 0/8 | **5/8** |

Đọc tay 6/8 prompt không-outlier — A cho câu trả lời **ngắn, ĐÚNG, dứt khoát**:

```
Khúc Thừa Dụ  → A: 'Dương Đình Nghệ'            (đúng — người kế vị)
Ngô Quyền     → A: 'độc lập năm 938'             (đúng — mốc sử nổi tiếng)
Lý Thái Tổ    → A: 'dời đô về Thăng Long'        (đúng — mốc sử nổi tiếng)
Hai Bà Trưng  → A: 'chống giặc Hán'              (đúng)
Trần Hưng Đạo → A: 'là Hưng Đạo Vương'           (đúng)
Lê Lợi        → A: 'khởi nghĩa Lam Sơn'          (đúng)
```

Cùng 6 prompt đó, B **trôi dài và lạc đề**, dù retrieval nhắm **đúng cha giống A**:

```
Hai Bà Trưng  → B: '. Khuôn viên cổ kính của Viện Đại học Đông Dương thì ch…'
                    (Hai Bà Trưng = năm 40; Đại học Đông Dương = 1906 — SAI THỜI ĐẠI)
Trần Hưng Đạo → B: 'từ Vạn Kiếp tới, Trần Quang Khải xuống thuyền chơi suố…'
                    (lạc sang giai thoại khác, không trả lời câu hỏi)
```

**Vì sao "hoà retrieval" mà "thua chất lượng" — cơ chế KHÁC với IDF pha loãng (§7.16a-d):**
`rank_matches` (đo *sau khi sinh*, để suy ra cha nào giống output nhất) hoà tuyệt đối. Nhưng
`generate()` **không đi theo `rank_matches`** — nó đi bằng agenda/spreading activation **trên
toàn đồ thị** (§0.1a, §7.13). Thêm 25 000 nơ-ron cha-đoạn (dù không thắng truy hồi) vẫn thêm
**node + cạnh adjacency** mà spreading có thể lạc vào giữa chừng. Đây là **ô THỨ TƯ**, không có
trong bảng runbook 3 hướng gốc: *cha-đoạn không thắng truy hồi, nhưng vẫn LÀM Ô NHIỄM đường đi
lúc sinh.*

⚠️ Một outlier giống hệt ở CẢ HAI nhánh (prompt `'Nhà Lý'` → `' ngự sử chứ , người ta đã muốn .
trốn tránh , bà cũng kh…'`, MỚI=20/20=100%): đọc không ra nghĩa — có vẻ là **câu chữ vô nghĩa**
(word-salad), không phải sinh thật có ý. Vì giống hệt ở cả hai checkpoint, đây là lỗi **độc lập
với câu hỏi granularity** — chưa điều tra, ghi lại để không lẫn vào kết luận trên.

**⚠️ SỬA — kết luận "củng cố TK_010" ở trên (viết lúc đầu) là OVERCLAIM, đã sửa 2026-07-18:**

Cái đo được **KHÔNG PHẢI** đúng đề xuất của tác giả. Đề xuất là **phân cấp thật**: token →
CHỈ nối vào nơ-ron câu → nơ-ron câu nối lên nơ-ron đoạn (token **không bao giờ** chạm đoạn trực
tiếp). Cái đo được (`histwiki_paragraph.pkl`) là xấp xỉ **rẻ hơn và khác về chất**: 25 000
nơ-ron đoạn thả **PHẲNG vào cùng một bể** với 151 000 nơ-ron câu, cạnh tranh ngang hàng — không
có tầng trên/dưới. Đây là hạn chế tôi tự nhận khi đề xuất đo (dùng checkpoint có sẵn cho rẻ),
không phải bản dựng đúng ý tưởng.

Thêm nữa: nhánh A (`histwiki.pkl`, chỉ cha-câu) **tự nó đã VI PHẠM TK_010** (TK_010 nói *"cả
episode là MỘT nơ-ron"* — A chia thành 151 430 câu). Checkpoint production thật khớp TK_010 là
`network_v10.pkl` (median cha 54 tok, không phải 19). A vs B **không phải** "giữ-nguyên-trạng
vs thêm-mới" — cả hai đều **khác** trạng thái hiện tại theo hướng chia nhỏ hơn TK_010 cho phép.
Nên kết luận đúng **không phải** "củng cố TK_010".

**Kết luận đúng, ba lớp:**

1. **Phiên bản ĐÃ ĐO** (đoạn thả phẳng cạnh câu) — bằng chứng rõ: **không nên**. Không thắng
   retrieval, có ô nhiễm đường sinh đo được (§7.16e trên).
2. **Phiên bản tác giả ĐỀ XUẤT** (phân cấp, token khoá vào câu) — **chưa bị bác bỏ**. Thật ra
   cơ chế gây hại vừa bắt được (đoạn cạnh tranh tự do trong bể phẳng) chính là thứ một cấu trúc
   phân cấp thật **sẽ chặn được**, vì token không bao giờ chạm trực tiếp nơ-ron đoạn. Đây là
   giả thuyết có cơ sở, KHÔNG phải kết quả đo.
3. **Dù kết quả (1) hay (2) thế nào, nó không đụng câu hỏi GỐC của ticket này** — *"mạng sinh
   thật, không bám verbatim"*. Nhánh TỐT NHẤT (A) vẫn **chép nguyên văn** (`copy TB 34%`,
   6/8 câu trả lời = LCS trùng khớp cha) — chỉ là chép đoạn NGẮN CHÍNH XÁC thay vì đoạn DÀI LAN
   MAN. `ar_index_backbone` (§7.13) vẫn ép chép bất kể nguồn là câu hay đoạn. Granularity ảnh
   hưởng ĐỘ CHÍNH XÁC của cái bị chép, không ảnh hưởng VIỆC CÓ CHÉP HAY KHÔNG.

⚠️ Cả hai checkpoint đều **tiền-BPE** (vocab 113 572/126 518, không phải BPE 32k) — cần lặp lại
phép so này trên `network_v11.pkl` (B5) để biết kết luận có đứng vững qua tokenizer mới không.

### 7.17. B5 xong — v11.pkl xác nhận khảm BẤT KỂ tokenizer + G15 kNN-LM phá được lỗi lặp

**B5 xong 2026-07-18 (M4, 8.91h):** `network_v11.pkl` — vocab BPE 32 002 (fingerprint
`1582dad40906e69f`), cùng corpus/manifest với `v10.pkl`, đã bake.

**§0.5 lặp lại trên BPE — kết luận ĐỨNG VỮNG:** `5-gram MỚI 0/216 = 0%`, đúng bằng không, y hệt
`v10.pkl` tiền-BPE (`0/276=0%`). **Tokenizer không phải nguyên nhân của "khảm"** — loại trừ được
một giả thuyết. So công bằng đầu tiên trong ticket (§7 dòng ~170 từng ghi *"chưa so được vì khác
corpus"* — giờ so được, cùng corpus cùng manifest): `v10` copy TB 5% vs `v11` copy TB 6% — gần
như không đổi.

⚠️ Outlier `'Nhà Lý'`/`'Trần Hưng Đạo'` đổi triệu chứng so với checkpoint histwiki (§7.16e): thay
vì chữ vô nghĩa, giờ dừng sau đúng 2 token ở dấu đóng ngoặc. Chưa điều tra — khác hẳn lỗi cũ.

#### 7.17a. G15 — kNN-LM datastore từ CORPUS THẬT (355 666 cặp, không phải câu viết tay)

Nối tiếp G13 (§7.15, 1 câu) và G14 (§7.16e outlier, 1+6 câu HARD viết tay) — lỗ hổng cuối cùng
trước khi kết luận kNN-LM có ích ở quy mô thật: dựng datastore từ `history_articles.jsonl` THẬT
(158 bài, cùng corpus đã train `v11.pkl`), test trên đúng 8 prompt lịch sử dùng xuyên suốt
ticket.

**Phát hiện mới, không có ở G13/G14:** ở `λ=1.0` (LM 11.4M thuần), **4/8 prompt lặp vòng thoái
hoá** — dấu hiệu kinh điển của LM nhỏ giải mã tham lam (argmax tự củng cố):
```
Ngô Quyền     → ', Ngô Quyền, Ngô Quyền, Ng…'
Nhà Lý        → ', nhà Lý, nhà Lý, nhà Lý, …'
Trần Hưng Đạo → ', Trần Hưng Đạo, Trần Hưng…'
Lê Lợi        → ', Lê Lợi, Lê Lợi, Lê Lợi, …'
```
Trộn với kNN (mọi λ<1.0), bản in cắt cụt 26 ký tự khiến tôi kết luận *"vòng lặp biến mất hoàn
toàn"* — **overclaim, đã sửa dưới đây sau khi đọc toàn văn (§7.17b).**

#### 7.17b. G15 v2 (toàn văn + 5-gram MỚI, 2026-07-18 M4) — sửa overclaim, số thật có đánh đổi

**`TỔNG 5-gram MỚI 433/1152 = 38%`** in ra trong log — **con số này KHÔNG dùng được**, vì trộn
lẫn hai loại "mới" khác hẳn nhau. Tách theo λ:

| λ | MỚI |
|---|---:|
| 1.0 (LM thuần) | **98.6%** |
| 0.5 | 34.7% |
| 0.2 | 12.2% |
| 0.0 (kNN thuần) | 8.0% |

`λ=1.0` chiếm gần hết tổng — nhưng đó là **giả tạo**: vòng lặp thoái hoá kiểu *"nhà thơ, nhà
thơ, nhà thơ…"* tạo 5-gram "mới" chỉ vì **corpus thật không ai lặp một cụm 10 lần liên tiếp**,
không phải vì có nội dung mới. Tính `λ=1.0` vào tổng là trộn "mới vì hỏng" với "mới vì sinh
thật" — §7.17a dùng con số `38%` này gián tiếp qua diễn giải là **sai**, phải sửa.

**Sửa lại tuyên bố "vòng lặp biến mất hoàn toàn"** — đọc toàn văn: ở `λ=0.5` vòng lặp **vẫn
còn**, chỉ **thô hơn** (lặp cả mệnh đề thay vì một token):
```
Ngô Quyền    λ=0.5: '…Đình Ninh Xá, xã Bắc An, Hải Phòng.⏎Đình Ninh Xá, xã Bắc An, tp Hải Phòng.⏎Đình Ninh Xá…'
Hai Bà Trưng λ=0.5: '…đường Nguyễn Chí Diểu, phường Thuận Hóa.⏎Đường Lê Đình Dương nay là đường Nguyễn Chí Diểu…'
```
Giảm mạnh so với `λ=1.0` (lặp từng token, sụp hoàn toàn) — nhưng **không** "biến mất hoàn toàn".

**Phát hiện thật, sau khi sửa: có ĐÁNH ĐỔI, không phải chiến thắng sạch.** λ cao → "mới" hơn về
thống kê nhưng dễ lặp hỏng; λ thấp → mạch lạc hơn hẳn nhưng gần chép nguyên (MỚI thấp). Đọc tay
`λ=0.0`, *"Lê Lợi"*: `'có cống số vàng cống theo lệ cho triều đình nhà Minh hay không.⏎Vua Lê
Thái Tổ sau khi lên ngôi năm 1428, chia trong nước làm năm đạo…'` — MỚI=0%, mọi 5-gram có sẵn
trong corpus, nhưng **ráp lại thành câu có nghĩa**, khác hẳn kiểu khảm-cụt-lủn (3-5 token rồi
dừng) thấy ở checkpoint đồ thị (§0.5, §7.16e). Không có điểm nào trong dải λ vừa **mới thật**
vừa **mạch lạc** cùng lúc — giới hạn thật của trộn tuyến tính đơn giản
`λ·P_LM + (1-λ)·P_kNN`, không phải bằng chứng đã giải quyết "sinh thật".

⚠️ Vẫn chưa đo bits/byte. `MỚI>0%` ở mức token cũng không chắc là "sinh thật" theo nghĩa ngữ
nghĩa — có thể chỉ là **ráp mới** hai mảnh cũ tại đúng ranh giới 5-gram (khảm ở hạt mịn hơn),
không phải nội dung mới về ý.

#### 7.17c. G15 v3 — LẤY MẪU thay argmax (2026-07-18 M4): biến thiên ✅ nhưng chất lượng SỤP ❌

Câu hỏi gốc TK_042 mô tả **lấy mẫu** (*"lúc chó lúc cún"*), không phải argmax. Chạy `--mode
sample --seeds 3 --sample-lams 0.5,0.3 --sample-temp 1.0`:

* **Biến thiên: THÀNH CÔNG tuyệt đối.** 3/3 seed cho văn bản khác nhau ở MỌI prompt/λ. Vòng lặp
  thoái hoá **biến mất hoàn toàn** — lần này THẬT (khác overclaim §7.17b, kia là argmax).
* **Chất lượng: SỤP.** Đọc tay phần lớn **vô nghĩa/sai sự thật**: *"Campuchia, Lào và Lào ngày
  nay là một vùng thuộc địa Ontario"* (Ontario = Canada); tên bịa *"alkaneban đã hạ cánh"*,
  *"Les Braque và Charles Burton Maria II"*; token rác thật `∩Es2 Jovianff`, `'w'w's`; số bịa
  `13017`, `2881 – 180`.

**`TỔNG 5-gram MỚI 1610/1728 = 93%` là BẪY — thước MỚI đã MÙ ở đây.** `5-gram MỚI` đo *"chưa từng
có trong corpus"*, KHÔNG phân biệt "mới vì sáng tạo mạch lạc" với "mới vì bịa". Ở đây 93% gần
như HOÀN TOÀN là loại thứ hai. Đây là lần thứ N trong ticket một thước đơn lẻ đánh lừa — bài học
lặp đi lặp lại.

**Nguyên nhân vật lý:** LM 11.4M ở 5.594 bits/token có phân bố RẤT PHẲNG. Lấy mẫu `temp=1.0` từ
phân bố phẳng ≈ bốc ngẫu nhiên. Argmax và sample-temp-1 là **hai đầu của CÙNG một bập bênh** của
LM quá yếu:

| chế độ | MỚI | mạch lạc | đúng sự thật |
|---|---|---|---|
| argmax λ=0.0 | 8% | ✅ như văn thật | ✅ nhưng gần chép nguyên |
| argmax λ=0.5 | 35% | ⚠️ lặp mệnh đề | ⚠️ |
| **sample temp=1.0 λ=0.5** | **93%** | ❌ word-salad | ❌ bịa |

Không có điểm nào vừa mới vừa đúng — vì bập bênh này là **triệu chứng LM 11.4M quá nhỏ**, KHÔNG
phải của cách giải mã. Đổi λ, đổi temp chỉ trượt DỌC bập bênh, không nhấc được nó lên.

**⇒ Kết luận: `5-gram MỚI` đã cạn đường.** Nó phân biệt được "khảm vs không-khảm" (§0.5, §7.16e
— vẫn đúng) nhưng KHÔNG phân biệt được "sinh thật vs bịa". Mọi bước tiếp theo (quét temp, top-k,
top-p, repetition penalty) đều sẽ đo ra "mới" MÃI MÃI đánh lừa cho tới khi có thước đo ĐỒNG THỜI
cả tính-mới lẫn tính-đúng. Thước đó **đã biết tên từ đầu ticket: bits/byte trên held-out** —
nó phạt CẢ lặp (dự đoán kém) LẪN bịa (dự đoán kém). Không trốn được nữa; đây là việc phải làm
tiếp, không phải tuỳ chọn.

⚠️ Cảnh báo trần cứng: nếu bits/byte cho thấy LM+kNN vẫn tệ ở mọi điểm vận hành, kết luận nhiều
khả năng là **LM 11.4M là trần** — cần LM lớn hơn (kNN-LM gốc Khandelwal dùng LM 250M+), không
phải chỉnh thêm núm. Ghi trước để không quét núm vô hạn.

#### 7.17d. G15 v4 — QUÉT temp + top-k (2026-07-18 M4): núm ĐÃ HẾT ĐƯỜNG, LM là trần

Theo hướng tác giả chọn: quét `temp∈{0.3,0.5,0.7}`, `top-k=40`, `λ=0.4`, đo thêm `rep4%` tự
động. Bảng tóm tắt:

| temp | MỚI% | rep4% | biến thiên |
|---|---|---|---|
| 0.3 | 36% | 8% | 100% |
| 0.5 | 56% | 1% | 100% |
| 0.7 | 74% | 0% | 100% |

**Top-k THẮNG ở đúng chỗ nhắm:** so temp=1.0 (§7.17c), cắt sạch rác đuôi (`∩Es2`, `alkaneban`,
`Ontario` biến mất) VÀ lặp thoái hoá gần hết (`rep4` 0-8%). Nếu chỉ nhìn 2 thước tự động →
trông như thắng.

**Nhưng đọc tay lật ngược:** văn giờ đúng ngữ pháp, mạch lạc, nhưng **SAI SỰ THẬT trơn tru** —
đúng "bịa mạch lạc" mà cả `MỚI%` lẫn `rep4%` đều mù:
```
Nhà Trần là  T0.3 → 'một nhà văn, ông Hồ bắt đầu đọc Tuyên ngôn'  (Trần 1225 → HCM 1945)
Hai Bà Trưng T0.5 → 'Năm 400, vua Lý Thánh Tông qua đời… thọ 71 tuổi'  (mất 1072, thọ 49)
Nhà Lý       T0.5 → 'Chiêu Hoàng. Lê Lợi sai quân đội Đại Việt…'  (trộn Lý ↔ Lê, cách 200 năm)
```
Ngay `T=0.5` (đẹp nhất theo 2 thước: rep 1%) nội dung vẫn sai.

**Mẫu hình nhất quán chỉ thẳng nguyên nhân:** token ĐẦU hầu như luôn đúng (Ngô Quyền→"Dương Thị
Vy", Nhà Lý→"Chiêu Hoàng", Lê Lợi→"cống… nhà Minh") — **kNN neo được điểm khởi đầu** — nhưng sau
vài token thì **TRÔI**. Chữ ký của LM không đủ năng lực GIỮ ngữ cảnh qua 40 token: kNN kéo về
đúng vùng, LM 11.4M không đủ mạnh bám lấy để đi tiếp cho đúng.

**⇒ KẾT LUẬN — điểm dừng vặn núm.** Đã quét HẾT núm giải mã có nghĩa: λ, temperature, top-k,
top-p, sample-vs-argmax. **Không núm nào nhấc được bập bênh mới-vs-đúng** (§7.17c) — vì nó là
giới hạn NĂNG LỰC MÔ HÌNH, không phải giới hạn giải mã. Cảnh báo trần cứng ở trên **đã thành
hiện thực**. Hai việc còn lại, KHÔNG phải vặn núm nữa:
1. **Đo bits/byte trên held-out** — chốt bằng con số thay vì đọc tay 8 prompt (LM thuần vs
   LM+kNN ở `T=0.5,top-k=40,λ=0.4`). Xác nhận kNN có cải thiện perplexity thật hay không.
2. **LM lớn hơn 11.4M** — nếu (1) xác nhận trần, đây là đòn bẩy DUY NHẤT còn lại. Khandelwal
   dùng 250M+. Đây là quyết định tài nguyên của tác giả, không phải việc đo tiếp.

#### 7.17e. G16 — bits/byte held-out (2026-07-18 M4): kNN DƯƠNG khiêm tốn + căng thẳng liều

Đo trên 16 bài held-out (tách khỏi 142 bài datastore, không leakage), 2500 token:

| λ | bits/byte | vs LM thuần |
|---|---|---|
| 1.0 (LM thuần) | 0.9529 | baseline |
| **0.7** | **0.9161** | **+3.9%** ← tốt nhất |
| 0.5 | 0.9460 | +0.7% |
| 0.4 | 0.9711 | **−1.9%** (tệ hơn) |
| 0.25 | 1.0296 | −8.1% |

**Đường cong chữ U rõ:** kNN cải thiện perplexity — THẬT, đo được bằng số — nhưng chỉ ở liều
NHẸ (λ=0.7, 30% kNN). Đẩy mạnh hơn thì HẠI; từ λ=0.4 xuống, density tệ hơn cả LM thuần.

**⚠️ CĂNG THẲNG liều — density-tốt ≠ sinh-tốt (nối ngược §7.17d):**
* density tốt nhất: λ=0.7 (kNN nhẹ)
* SINH có biến thiên (G15 v4): cần λ=0.4 — mà bits/byte cho thấy λ=0.4 **làm density TỆ HƠN LM
  thuần (−1.9%)**

Hai mục tiêu **kéo ngược nhau**: liều kNN đủ nhẹ để cải thiện perplexity thì quá nhẹ để lái
sinh; liều đủ nặng để lái sinh thì phá perplexity. Không có λ nào phục vụ cả hai. Đây là hệ quả
TRỰC TIẾP của LM 11.4M quá yếu: phân bố quá phẳng nên vừa CẦN kNN kéo (để sinh khác) vừa BỊ kNN
kéo lệch (khi kéo mạnh).

**⇒ Bức tranh khép lại (G13→G16):** kNN-LM cơ chế đúng, retrieval thật (§7.17a), cải thiện
perplexity đo được (+3.9%). Nhưng cả density (+3.9% khiêm tốn, Khandelwal đạt lớn hơn nhiều với
LM 250M+) lẫn sinh (trôi chủ đề §7.17d) đều **bị chặn bởi cùng một trần: LM 11.4M**. Mọi đường
đo đều hội tụ về cùng kết luận. Đòn bẩy còn lại DUY NHẤT là **LM lớn hơn** — không phải núm,
không phải thêm probe. Quyết định tài nguyên của tác giả.

⚠️ Caveat mẫu: 2500 token / 16 bài — đủ thấy U-curve rõ, nhưng con số chính xác (+3.9%) có sai
số; nếu cần chốt chặt hơn, tăng `--max-tokens`. Không đổi kết luận định tính.

### 7.18. G17 — "spreading học được + xếp chồng được" (yêu cầu tác giả 2026-07-18)

Tác giả: *"làm cho spreading activation học được và xếp chồng được thử xem"* — sau chuỗi thảo
luận bản chất attention/FFN (§7.17d-e): attention = định tuyến động học-đồng-thời, thứ DUY NHẤT
đồ thị không làm được bằng đếm; spreading activation = "attention thô sơ" (1 vòng, trọng số
đếm cố định, không xếp chồng).

**Nói thẳng đã thừa nhận trước:** làm spreading vừa học-được vừa xếp-chồng-được = phát minh lại
attention train bằng backprop. Vậy câu hỏi ĐO ĐƯỢC là: *khi đã cho phép backprop, HÌNH HỌC ĐẾM
của đồ thị còn đóng góp gì, hay backprop tự học hết?*

**Thiết kế** (`scripts/probe_graph_attention.py`): A/B hai transformer giống hệt từng batch
(cùng seed), khác ĐÚNG MỘT DÒNG — nhánh B: `logits_attn += α_l·(Ê_tᵢ·Ê_tⱼ)`, Ê = E của
`network_v11.pkl` chuẩn hoá hàng (đóng băng), α_l học được/tầng (init 1.0). Vì sao tích-trong-E
thay vì tra ma trận kề: PPMI-SVD ⇒ Ê·Êᵀ ≈ PPMI đã làm mượt — ĐÚNG hình học spreading dùng,
tính bằng một matmul (tra dict thưa 32k² = trăm triệu lookup Python, bất khả thi).

**Điều kiện phân định ghi trước:** B thắng rõ + α xa 0 ⇒ prior đồ thị CÓ ÍCH, đường mới nhấc
trần; Δ≈0/α→0 ⇒ backprop tự học hết, đồ thị không đóng góp ở tầng ĐỊNH TUYẾN (giá trị tầng BỘ
NHỚ vẫn nguyên — G16 +3.9%) — **âm tính trung thực, khép nhánh**. Probe d=128 L=2 4000 step:
tín hiệu, không SOTA; prior giúp NHIỀU NHẤT khi model/data nhỏ nên nếu thua ở đây thì gần chắc
thua ở bản to.

#### 7.18a. KẾT QUẢ (2026-07-18 M4): ÂM TÍNH TRUNG THỰC — hình học đồ thị THỪA với backprop

```
PLAIN best 9.8577 bits/tok · GRAPH best 9.8633 · Δ = −0.06% (hoà)
α cuối = [1.55, 1.06] — KHÔNG về 0, tầng 1 còn TĂNG
```

**Δ≈0 là trọng tài ⇒ đồ thị không đóng góp gì ở tầng định tuyến.** Hai chi tiết tinh tế phải
đọc đúng kẻo tự lừa:

1. **α tăng ≠ prior có ích.** Bảng điều kiện dự trù "α→0 = vô dụng" nhưng xảy ra ca thứ ba:
   α tăng mà Δ≈0. Mạng không cần DẬP bias — nó BÙ TRỪ quanh bias (Q/K tự co giãn), vì thông
   tin trong E **thừa** chứ không sai. Trọng tài là bits held-out, không phải α.
2. **Lợi thế sớm rồi tan** — chữ ký của prior-thừa: step 500–2000 GRAPH dẫn nhẹ (16.08→15.98,
   10.35→10.33), step 2500 hai đường chập một, cuối còn cắt xuống dưới. Prior giúp khởi động
   nhanh hơn chút, rồi backprop tự học được đúng thứ đó.

**Vì sao (nhìn lại thì hiển nhiên):** E xây bằng cách đếm CHÍNH corpus LM đang train. Backprop
truy cập cùng nguồn thông tin ⇒ học ra mọi thứ E biết, chỉ mất ~2000 step. Prior chỉ có giá
trị khi mang thông tin data train KHÔNG có; đây là cùng một data.

**Khép vòng G13→G17 đối xứng:**

| vai trò đồ thị | kết quả đo |
|---|---|
| ĐỊNH TUYẾN (E làm prior attention) — G17 | ❌ thừa, backprop tự học (Δ −0.06%) |
| BỘ NHỚ (datastore ngoài, không nén vào trọng số) — G16 | ✅ +3.9% bits/byte |

Giá trị độc nhất của đồ thị KHÔNG phải hình học (học lại được từ data) mà là **bộ nhớ ngoài
không-tham-số** — thứ LM không nén nổi vào trọng số. Đúng phân công CLS (hippocampus/neocortex,
§7 đầu ticket). Nhánh "dạy đồ thị nói tiếng attention" khép tại đây; đòn bẩy còn lại vẫn là
**LM lớn hơn** (§7.17e) + đồ thị giữ vai bộ nhớ (kNN-LM).

### 7.19. G18 — TẦNG NƠ-RON TRI THỨC (yêu cầu tác giả 2026-07-18, sau khi bác kNN-LM)

Tác giả: *"kNN-LM bản chất là RAG… tri thức nằm ngoài, không học. Tôi muốn: backprop OK, nhưng
phải có NƠ-RON MỚI mang tri thức, và sinh bản chất vẫn là LAN TRUYỀN KÍCH THÍCH."* — phê phán
đúng: kNN-LM và RAG cùng họ truy-hồi-lúc-suy-luận (khác điểm tiêm: ngữ cảnh vs phân bố);
datastore đóng băng, không gradient, không bao giờ thông minh lên.

**Thiết kế** (`scripts/probe_knowledge_neurons.py`) — ghép các mảnh LLM thật đã chứng minh:
FFN = key-value memory (Geva) · Memory Layers at Scale (Meta 2024: bảng triệu slot top-k thưa,
thắng FFN cùng FLOPs) · Net2Net (thêm nơ-ron giữa chừng) · ROME (fact định vị được vào nơ-ron):

```
Block cuối: FFN → MemLayer: N=4096 nơ-ron (key 64d, value 128d)
q = Wq·h → top-32 NỔ → h += Σ softmax·value        ← lan truyền kích thích, nghĩa đen
```

Đưa datastore VÀO TRONG mạng: slot = nơ-ron có danh tính, có gradient, CHỈ ĐƯỢC (in ra slot
nào nổ với câu hỏi nào), MỌC được. Sparse update ⇒ gradient chỉ chạm nơ-ron đang nổ — học cục
bộ, rất gần tinh thần Hebbian.

**Bài thi quyết định — MỌC-KHÔNG-QUÊN** (thứ FFN đóng cứng lẫn kNN-LM đều không có): sau train,
đưa 20 fact HƯ CẤU (vua/địa danh bịa — chắc chắn ngoài corpus) → mọc 20 slot (key = trung bình
q=Wq·h của chính mô hình trên fact — đúng không gian truy hồi G13-15 đã chứng minh), đóng băng
toàn mạng, train CHỈ slot mới → đo (i) bits trên fact (biết chưa) (ii) bits held-out cũ (quên
chưa). Đối chứng: FFN full-fine-tune cùng data (dự kiến quên). Rủi ro thật được đo: slot mới có
thể NỔ NHẦM trên ngữ cảnh cũ (giao thoa qua kích hoạt dù không đụng trọng số cũ).

**Điều kiện phân định ghi trước:** PASS = Δfact ≤ −2.0 bits/tok VÀ |Δquên| < 0.05 (nhánh mem);
FFN-FT làm nền quên; `mem_rand ≈ mem_graph` ⇒ gieo mầm đồ thị thừa (lặp G17) — ghi thật. bits
train chính MEM ≈ FFN là ĐỦ (Meta thấy lợi ở scale; tham số không cân bằng — so bits chỉ tham
khảo).

#### 7.19a. KẾT QUẢ G18 (2026-07-19 M4, 3000 + 6000 step): RỚT — nhưng chẩn đoán được ĐÚNG CHỖ SAI

| nhánh (6000 step) | Δfact (cần ≤−2.0) | Δquên (cần <0.05) | phán quyết |
|---|---:|---:|---|
| ffn full-FT | **−9.03** ✓ học được | **+13.02** ✗ quên thảm hoạ | đúng dự đoán — nền |
| mem_rand | −0.38 ✗ | +0.19 ✗ | RỚT |
| mem_graph | −0.37 ✗ | +0.17 ✗ | RỚT |

**FFN đối chứng diễn đúng kịch bản ngoạn mục:** 6000 step trả lời ĐÚNG 2/3 probe (*"ban hành
bộ luật" → " Hồng Nghĩa"*, *"giữ thành" → " Yên Phong suốt bốn tháng"*) — backprop nhét được
fact vào trọng số — nhưng held-out 9.5→22.5 bits: phá huỷ toàn bộ tri thức cũ.

**Nhánh mem: KEY chạy, VALUE chết.** `slot nổ: 20/32 là slot MỚI` khi hỏi fact — truy hồi
đúng. Nhưng sinh ra `'.<EOS><EOS>…'` — value chỉ học được phân bố biên của dòng fact.

**Chẩn đoán — SAI HẠT ĐỘ MỌC, nhìn ra là hiển nhiên:** 1 slot/fact = 1 vector value TĨNH phải
mã hoá ~25 dự đoán KHÁC NHAU THEO VỊ TRÍ ("dời đô về"→"thành", "thành"→"Phượng"…) — **bất khả
về biểu diễn**, không phải thiếu train. kNN-LM lưu THEO VỊ TRÍ — (h_t, next), 355k cặp/158 bài
— đúng hạt độ. 20 fact cần ~500 nơ-ron (1/vị trí), không phải 20.

Ghi chú phụ đắt giá: (1) `mem_rand ≈ mem_graph` giống nhau đến 4 chữ số — gieo mầm đồ thị thừa
**lần thứ ba** (G17, G18×2); (2) slot cũ nổ khi hỏi vua Việt = "Albion", "hợp chất Au(I)" —
key E64 không chọn lọc ngữ nghĩa.

**Lần hội tụ THỨ TƯ về cùng cấu trúc:** "nơ-ron tri thức mọc được" làm đúng hạt độ = **datastore
kNN-LM có gradient nằm TRONG mạng** (48 phép chiếu→attention; spreading học được→G17; cạnh học
bằng lỗi→predictive coding≡backprop; giờ nơ-ron mọc→datastore-vị-trí).

#### 7.19b. G18b — vá hạt độ: `--grow-mode position` — KẾT QUẢ: BÁC CHẨN ĐOÁN §7.19a

Mọc MỘT nơ-ron MỖI VỊ TRÍ token (330 slot, key = q của chính mô hình tại vị trí đó). Kết quả
(2026-07-19 M4): **Δfact −0.37 ≈ G18's −0.41 — KHÔNG đổi**, Δquên +0.10, probe vẫn `'.<EOS>'`
— **cùng chữ ký hỏng**. Nếu "sai hạt độ" là nguyên nhân, 330 slot phải học được. Nó không.
⇒ **chẩn đoán §7.19a SAI** (bị phép đo bác — ghi lại, tìm tiếp, đúng quy trình).

#### 7.19c. Soi lại probe → LỖI THIẾT KẾ ĐO (họ TK_041 §5.2) — G18c = VÒNG CHÓT

Ba pha của probe không cùng phân bố:

```
capture key : fact ĐỨNG MỘT MÌNH, vị trí 0   ┐ khớp nhau
eval        : fact ĐỨNG MỘT MÌNH, vị trí 0   ┘
train mọc   : cửa sổ 64-tok cắt từ DÒNG NỐI 20 fact, offset NGẪU NHIÊN   ← LỆCH PHA
```

Lúc train, `h` (⇒ `q`) tại cùng token KHÁC lúc capture (vị trí khác + ngữ cảnh trước khác) ⇒
nơ-ron "đúng" KHÔNG nổ lúc train ⇒ gradient rơi vào nơ-ron nổ nhầm, mỗi cái thấy hàng tá
target ⇒ học ra trung bình = `'.'`, EOS. **Giải thích được chữ ký hỏng Y HỆT ở cả hai chế độ
mọc** — lỗi tồn tại từ G18, làm nhiễu cả hai lần đo. Không crash, số vẫn ra, kết luận sai —
đúng họ lỗi im lặng TK_041 §5.2, lần này trong probe của chính tác giả script.

**G18c**: train mọc trên TỪNG FACT ĐỨNG RIÊNG từ vị trí 0 (khớp capture & eval; pad `y=-100`
bỏ qua). Áp cho CẢ nhánh ffn-FT (công bằng). Đây là sửa PHÉP ĐO, không phải núm kiến trúc.
**Kỷ luật ghi trước: VÒNG CHÓT của G18 — rớt nữa = khép nhánh dứt khoát**, phán quyết "bộ nhớ
tham số mọc được không giao hàng ở quy mô probe; kNN-LM phi-tham-số vẫn là câu trả lời thực
dụng". Điều kiện PASS giữ nguyên: Δfact ≤ −2.0 VÀ |Δquên| < 0.05.

#### 7.19d. KẾT QUẢ G18c (2026-07-19 M4, 6000 step): RỚT — KHÉP NHÁNH DỨT KHOÁT

| nhánh | Δfact | Δquên | |
|---|---:|---:|---|
| ffn full-FT | **−10.85** (fact → 0.057 bits!) | **+14.17** | học HOÀN HẢO, quên SẠCH |
| mem_rand | −0.39 | +0.094 | không học, không quên |
| mem_graph | −0.38 | +0.092 | không học, không quên |

**Đối chứng FFN là bằng chứng then chốt:** sau vá lỗi đo, FFN-FT thuộc lòng gần tuyệt đối —
probe trả lời ĐÚNG TỪNG CHỮ cả 3/3 (*'thành Phượng Nham năm 1361'*, *'Hồng Nghĩa năm 1477'*,
*'Yên Phong suốt bốn tháng'*). ⇒ **dữ liệu + quy trình train giờ CHUẨN, fact HỌC ĐƯỢC.**
Vậy mà nhánh memory đứng im: Δfact −0.38 ≈ G18 (−0.41) ≈ G18b (−0.37) — **ba vòng, ba biến
khác nhau (hạt độ, phân bố train), MỘT kết quả ⇒ thất bại là KIẾN TRÚC, không phải phép đo.**

**Tam giác đánh đổi hiện nguyên hình** (tổng kết cả chuỗi G13→G18):

| cách đưa fact mới vào | biết fact mới? | giữ tri thức cũ? |
|---|---|---|
| FFN full-FT (tham số, không bảo vệ) | ✅ 0.06 bits | ❌ +14.2 |
| memory mọc slot (tham số, bảo vệ bằng đóng băng) | ❌ −0.38 | ✅ +0.09 |
| kNN-LM datastore (phi-tham-số) | ✅ tức thì | ✅ không đụng |

Ở quy mô probe, hai góc THAM SỐ mỗi góc hỏng một vế; chỉ góc PHI-THAM-SỐ giao được cả hai.

Giả thuyết hậu-kiểm (GHI, KHÔNG test — nhánh đã khép): value bị NHOÈ qua top-k — key của các
vị trí trong cùng fact tương quan cao ⇒ mỗi vị trí kích hoạt ~32 slot phần lớn từ fact lân cận
⇒ mỗi value nhận gradient từ ~32 target xung đột ⇒ học ra trung bình ('.', `"`). Tín hiệu phụ
cùng hướng: ngay TRAIN CHÍNH memory đã thua FFN 0.3 bits (9.78 vs 9.48) — memory layer < FFN
toàn diện ở quy mô này. Meta thấy lợi ở HÀNG TỶ tham số, train-from-scratch CÙNG backbone —
không phải ghép lên backbone đóng băng 5M.

**⇒ PHÁN QUYẾT (đúng kỷ luật ghi trước):** nhánh "nơ-ron tri thức tham số mọc được" KHÉP.
Muốn nó sống cần scale lớn + train cùng backbone từ đầu = chính là quyết định "LM lớn hơn"
(§7.17e) chứ không phải một nhánh riêng. Trạng thái cuối của ticket không đổi: **kNN-LM
phi-tham-số làm bộ nhớ (✅ đo được +3.9%) + LM lớn hơn làm năng lực** — giờ có thêm tam giác
G18 làm bằng chứng định lượng vì sao phân công đó không phải lựa chọn tuỳ tiện mà là bắt buộc
ở quy mô hiện tại.

### 7.20. G19 — TRAIN LM LỚN HƠN: v2 42M + arm memory-from-scratch (phê duyệt 2026-07-19)

Đòn bẩy cuối cùng, tác giả duyệt cấu hình **B**: d=512, L=8, nh=8, ctx=512, **~42M** (3.7×
mốc 11.4M) + arm **B+mem** (~49M): MemLayer 16 384 slot thay FFN block 5/8, **train cùng
backbone từ đầu** — đúng điều kiện sống duy nhất §7.19d chỉ ra.

**Hiện đại hoá** (build_model_v2, một-nguồn-duy-nhất trong `neural_lm.py`): RoPE (thay bảng vị
trí học được — v1 đang dùng kiểu GPT-2 2019), RMSNorm, SwiGLU (hid=2/3·4d bội 64). Checkpoint
v2 = dict {cfg, state_dict}; `load()` đọc cfg tường minh vì `nh` KHÔNG suy được từ shape (qkv
gộp (3d,d) bất kể mấy head — đoán là lớp lỗi im lặng TK_041 §5.2).

**Trực giao head (câu hỏi tác giả):** ba tầng — (1) `orthogonal_` init ma trận gộp ⇒ head là
lát hàng ⇒ trực giao TỪNG ĐÔI tại t=0, miễn phí; (2) **Muon**: cơ chế ruột là Newton-Schulz
trực giao hoá momentum ⇒ update đẳng hướng ⇒ trọng số không sụp hạng trong train — đúng câu
trả lời thực nghiệm cho "giữ head khác nhau", trùng khớp optimizer đã chọn; (3) ràng buộc
cứng/phạt mềm: KHÔNG — nghiên cứu tỉa head (Michel/Voita 2019) cho thấy dư thừa một phần là
bình thường; thay ép bằng ĐO (chẩn đoán head-sim in cuối train). Muon lai chuẩn: ma trận ẩn
2D đi Muon (lr 0.02 phẳng), embedding/norm/bảng memory đi AdamW+OneCycle.

**Bức tường data ghi trước:** corpus 19.7M token; 11.4M cũ đã "vắt được data" (gap +0.811).
42M train ~4.2 epoch (ngưỡng Muennighoff), dropout + best-checkpoint. **Điều kiện phân định:**
(1) v2-42M PHẢI thắng rõ mốc **5.594 bits/token** — không thắng ⇒ bức tường là DATA, chốt bằng
số, hướng tiếp là thêm corpus; (2) arm mem vs arm thuần: mem ≈ thuần ⇒ memory chưa có lợi ở cỡ
này (ghi thật); mem thắng ⇒ mở lại bài thi mọc-không-quên trên nền này; (3) thắng 5.594 rồi ⇒
vá probe kNN (viết theo ruột v1) → chạy lại G16 λ-sweep + G15 v4 sinh — kỳ vọng thật: trôi chủ
đề giảm, căng thẳng liều λ (§7.17e) nới. Sửa kèm: bug save cuối-đè-best trong train script
(trước đây bản CUỐI ghi đè bản TỐT NHẤT đã lưu giữa chừng — phá logic early-stop).

#### 7.20a. KẾT QUẢ G19 lần 1 (2026-07-19 M4): HỎNG + BỊ NHIỄU — không kết luận, một lỗi của tôi

Held-out y hệt G7 (count baseline 1.276 xác nhận). Arm 1 (42M, 10k step, đo được **3.06h**):
**best 6.479 bits/token / 1.167 bpb** — **TỆ HƠN mốc 11.4M (5.594) 0.885 bits.** Sinh THẢM HOẠ,
lặp vô tận (*"con gái của con gái của con gái"*, *"Vương Trường Sinh, Vương Trường Sinh"*).
Arm 2 (mem 49M) dừng ở step 1500/10000 — dở dang, bỏ.

**KHÔNG kết luận "bức tường data" — hai nhiễu, một là lỗi tôi:**
1. **Thiếu train 3×:** G7 thấy 245M token (30k step), G19 chỉ 82M (10k step). 3.7× to trên 1/3
   data. Chẩn đoán của chính script: gap train/eval **0.292 < 0.5 = UNDERFIT** — đúng dòng
   cảnh báo "KHÔNG kết luận data quá nhỏ, train thêm". Chốt data-wall giờ = lặp đúng lỗi G7.
2. **LỖI CẤU HÌNH CỦA TÔI — Muon LR để PHẲNG.** Eval tố cáo: tốt nhất @5000 (6.48) rồi XẤU ĐI
   về 6.66 @10000, nảy dữ dội. Chữ ký "LR cao, không anneal, không lắng đáy". Comment tôi viết
   *"Muon giữ lr phẳng (chuẩn speedrun)"* SAI — speedrun nanoGPT anneal Muon LR về 0. Giải
   thích cả bits xoàng lẫn sinh lặp thoái hoá. **Đã vá:** anneal cosine (warmup 5% → 0).

**Điều DUY NHẤT sạch — trả lời đúng câu hỏi tác giả về trực giao head:** head-sim `mean|cos| ≈
0.005` cả 8 tầng (ngẫu nhiên d=512 ~0.044 ⇒ **trực giao hơn ngẫu nhiên 10×**). Orthogonal-init
+ Muon GIỮ head khác nhau rõ suốt train — độc lập với hai nhiễu trên. ✅

⇒ **Chạy lại arm 1 với Muon LR đã anneal, 30k step** (khớp token-budget G7, loại nhiễu 1). Nếu
VẪN > 5.594 rõ ⇒ khi đó mới cân data-wall. Nếu thắng ⇒ tiếp kế hoạch. Repo: 2 file .pt 365MB bị
commit làm ĐỨT fetch — đã gitignore `out/*.pt` + `git rm --cached`.

#### 7.20b. GỠ NÚT DATA — thêm viwiki (điều kiện tiên quyết của "LM lớn hơn")

Số học: 42M ở Chinchilla (~20 tok/tham số) cần ~840M token; corpus 19.7M thiếu **~40×**. "LM
lớn hơn" và "thêm data" KHOÁ NHAU — chính thiếu data làm G19 lần 1 nhiễu. `scripts/ingest_viwiki.py`
(HF `wikimedia/wikipedia 20231101.vi`, plain text, khử trùng + lọc stub + trần token) → kéo
corpus lên ~170M (~9×), cùng miền `histwiki`, giữ held-out `essays` riêng (không rò rỉ).
`config/train_v11.yaml` = v10 + viwiki. Rerun 30k dùng `--manifest config/train_v11.yaml`.

### 7.21. G20 — Mixture of Softmaxes ở tầng output (ý tưởng tác giả 2026-07-19, SPEC để dành)

Tác giả đề xuất "attention-hoá LM head" / "multi-head output" / "dynamic lexical memory". Đánh
giá đại số dứt khoát:

**Multi-head cộng logits SỤP về tuyến tính:** `Σᵢ h·W_Qi·(E·W_Ki)ᵀ = h·(Σᵢ W_Qi W_Kiᵀ)·Eᵀ =
h·M·Eᵀ`, M là MỘT ma trận d×d ⇒ 8 head = 1 head, hạng ≤ d (softmax-bottleneck, Yang 2018).
"Head ngữ pháp / head ngữ nghĩa" bị toán xoá sạch nếu trộn TRƯỚC softmax. RoPE-ở-output = lỗi
phạm trù (vocab không có vị trí chuỗi). Weight-tying "kìm hãm" cũng sai — ở data-ít nó là
regularizer PHẢI giữ.

**Phiên bản ĐÚNG toán của ý tưởng = MoS:** trộn SAU softmax, `P = Σₖ πₖ·softmax(tanh(h·Wₖ)·Eᵀ)`.
Vì softmax phi tuyến, `Σ softmax ≠ softmax(Σ)` ⇒ hạng log-P thoát trần d. GIỮ tied (chỉ thêm K
ma trận d×d + gating). Đã code: `build_model_v2(mos_k=K)` trả LOG-P (cờ `returns_logprob` ⇒
loss=nll, eval KHÔNG log_softmax lại — chống double-softmax = lỗi im lặng TK_041 §5.2).
`train_neural_lm.py --mos-k K`.

**⚠️ NHÁNH THỨ YẾU — điều kiện bật:** §7.20a đo nút thắt là DEPTH + DATA, KHÔNG phải output
head (sinh lặp G19 do thiếu-train + bug Muon, không do head phẳng). MoS cho GPT-2 chỉ +vài %
perplexity, trong khi ta thua 0.9 bits + thiếu data 40×. **CHỈ chạy G20 sau khi (1) viwiki gỡ
data (2) rerun 30k sạch, mà bits VẪN kẹt VÀ chẩn đoán chỉ vào output.** A/B như G17: cùng
backbone, chỉ bật `--mos-k 15` (Yang dùng 15), so bits/token. Chưa chạy thực (máy dev không
torch — chỉ kiểm cú pháp + smoke shape). Dynamic-K (`K=E(W_K1+g(h)W_K2)`) hoãn sau MoS — giải
đa-nghĩa-từ, không phải nút đo được.

### 7.22. KV CACHE — sinh nhanh (câu hỏi tác giả "chạy runtime / sinh nhanh", 2026-07-19)

Ba nghĩa "nhanh": latency (KV cache) · throughput (batch) · nhiều-token/bước (speculative
decoding — đồ-thị-nháp/LM-verify, hoặc Medusa). Làm cái nền trước: **KV cache** (§7.22, ĐÃ CODE).
Không cache: mỗi token tính lại O(ctx). Cache: prefill một lần, decode O(1)/token, 5-20× nhanh,
đầu ra KHÔNG đổi.

`NeuralLM.generate_cached` (chỉ v2). Attn/Block/LMv2 thêm đường `forward_cached(x, caches,
offset)`; `forward` train/eval GIỮ NGUYÊN chữ ký (gọi block bỏ cache). **Chốt kỹ thuật: RoPE
phải xoay theo VỊ TRÍ TUYỆT ĐỐI** — token decode ở `past_len`, không phải 0; `rope(x,T,offset)`
dùng `COS[offset:offset+T]`. offset=0 khớp shape nhưng xoay SAI ⇒ lỗi im lặng TK_041 §5.2.
Prefill `is_causal=True`, decode (có past) `is_causal=False` (query nhìn toàn bộ past+self).

`scripts/probe_kv_cache.py`: **đúng-đắn TRƯỚC tốc độ** — greedy có-cache PHẢI == không-cache
từng token (lệch ⇒ exit 1 + in vị trí lệch), rồi mới đo tok/s. Chưa chạy thực (máy dev không
torch — chỉ cú pháp). Bước tiếp (khi cần "nhiều token/bước"): **speculative decoding** dùng đồ
thị làm drafter — verbatim-copy (nhược điểm chất lượng §7.13) thành ĐÒN BẨY tốc độ vì drafter
đúng nhiều ⇒ LM verify K token/pass, đầu ra = LM đơn. Runtime trên máy nhỏ còn chặn bởi TK_040
(checkpoint đồ thị 2GB, Intel 8GB không nạp nổi).

Nếu **G2 âm tính** → corpus 19.6M token quá thưa cho nút cổ chai → **GHI ÂM TÍNH, DỪNG, quay về
A hoặc C**. **KHÔNG** đắp G4 lên một G2 âm tính, **KHÔNG** đổi thước cho đến khi nó dương.

*(Tiền lệ trong chính dự án: R-slot — "thí nghiệm rẻ chặn 1 lần train vô ích"; order-2, E-episode
extrinsic. Ba lần dự án đã làm đúng việc này. Đây là lần thứ tư.)*

---

## 8. Trần trung thực — ghi TRƯỚC khi đo

### 8.1. Dữ liệu
Corpus 99 MB ≈ **20M token**. Chinchilla-optimal cho 20M token ≈ mô hình **~1M tham số**.
**Sẽ KHÔNG có GPT-2 từ đây, bất kể kiến trúc.**

### 8.2. Sức mạnh mô hình
Bilinear hạng thấp về bản chất là **tuyến tính** — yếu hơn transformer nhiều. Ước thực tế:
**~6 → ~3.5–4 bits**. Tiến bộ thật, **không phải LLM**.

### 8.3. *"Lúc chó, lúc cún"* — R4 KHÔNG phải bài toán sinh
Nếu corpus có `chó` 5 000 lần và `cún` 50 lần, **một mô hình ĐÚNG phải sinh `chó` ~99%**. Đó là
khớp phân bố dữ liệu — **đúng, không phải lỗi**. "Lúc này lúc kia" là **điều khiển
register/văn phong** (`docs/study/04`), một bài toán **khác**. Trộn R4 vào R1 = **tự lừa**.

### 8.4. Lấy mẫu không phải câu trả lời
Bật `ar_temperature` **bây giờ** cho **nhiễu**, không cho văn phong: lấy mẫu từ phân bố
86-lựa-chọn-ngang-ngửa = word salad. **Phân bố phải giỏi TRƯỚC.**

---

## 9. Ticket này KHÔNG hứa

- ❌ **Không** hứa sinh như LLM. Trần 20M token + mô hình tuyến tính (§8.1, §8.2).
- ❌ **Không** hứa R4 (`chó`/`cún` luân phiên) — bài toán khác (§8.3).
- ❌ **Không** hứa G2 dương. **Dự đoán ghi trước: 50/50** — và con số đó ra từ **phép tính**, không
  phải khiêm tốn xã giao: corpus 19.6M quan sát vs ~21M cần cho r=64 (§7.1). Nếu âm tính → ghi
  âm tính, đó là kết quả hợp lệ (tiền lệ: order-2, R-slot, E-episode).
- ✅ **Có** hứa: **biết SỰ THẬT, đo được** — bức tường ở đâu, hạng thấp có vượt được không, và
  **quyết định A/B/C dựa trên số chứ không phải hy vọng**.

**Việc TIẾP THEO: tác giả chốt A / B / C.** Không viết code trước khi chốt — vì A nghĩa là đóng
ticket này và sửa định vị README, còn C nghĩa là viết lại dự án.

---

## 7.8. ✅ G7 ĐÃ CHẠY (2026-07-17) — DƯƠNG TÍNH 🎉 → **KẾT LUẬN B/C CÓ NGHĨA**

**Setup:** Transformer 11.4M params (d=256 L=4 h=4 ctx=256), Adam lr=3e-4, OneCycleLR, **30,000 steps**,
device=MPS (Apple Silicon arm64), corpus 19.7M token BPE 32k. Cùng manifest/tokenizer/held-out/
công thức với baseline đếm. Bug fix: MPS không hỗ trợ dropout trong `scaled_dot_product_attention`
→ `attn_dropout=0.0` trên MPS (FF dropout=0.1 giữ). 

**Sửa chữa từ run 3000 steps:** Run đầu (3000 steps) chỉ ~1.25 epoch ⇒ **UNDERFIT**: train loss 
6.938 ≈ eval loss 6.938 (không learnable gap). Extended to 30,000 steps (12 epochs) với mid-training 
eval every 2000 steps → nhặt được **best checkpoint @26000: 1.009 bits/byte**.

| Mô hình | bits/token | **bits/byte** | Δ vs baseline | Status |
|---|---:|---:|---:|---|
| **ĐẾM** (bigram+unigram, lam*=0.85) | 7.088 | **1.276** | — | baseline |
| **Transformer** @30000 (11.4M, MPS) | 5.594 | **1.007** | **-0.269** 🎯 | best final |
| Transformer @26000 (checkpoint) | 5.606 | **1.009** | -0.267 | ★ tốt nhất |

**Δ = −0.269 >> ngưỡng 0.1** ⇒ **HƠN ĐÁNG KỂ, 21% tốt hơn baseline.**

**Train/eval gap (final):** 4.783 bits/token (train) vs 5.594 bits/token (eval) = **+0.811 bits/token ≫ 0.5 threshold** ✓

#### 7.8-verify. Ba phép kiểm — kết luận này đứng vững (khác lần 3000 step)

| Kiểm | Kết quả |
|---|---|
| **Chẩn đoán FIT** (thay ngưỡng bịa `steps<500`) | `gap = +0.811` > 0.5 ⇒ **đã vắt được data**, KHÔNG underfit |
| **Overfit?** | **15/15** mốc eval đều `★ tốt nhất` ⇒ eval **CHƯA BAO GIỜ xấu đi** ⇒ không overfit |
| **Δ** | **−0.269** < −0.1 ✓ |

→ Guard đã làm đúng việc nó sinh ra để làm: lần 3000 step nó **đáng lẽ phải bắt** (và không
bắt — ngưỡng bịa); lần này chẩn đoán fit thật báo `✓ đã vắt được data`. **Số này tin được.**

#### 7.8-caveat. BA điều KHÔNG được suy ra từ G7 (ghi để không tự lừa)

1. **G7 KHÔNG chọn C.** Nó nói **B và C đều khả thi** — chứng minh *có gì đó để học*, KHÔNG nói
   *học bằng luật nào*. Câu hỏi **luật cục bộ (B) vs backprop (C)** vẫn **nguyên vẹn** (§7.6).
   *(Commit `612bd9f` ghi "A→C" — không chính xác.)*
2. **`1.007` KHÔNG phải sàn.** Eval **vẫn đang cải thiện** ở step 30 000 (mọi mốc đều ★; 26k→30k
   còn −0.002). Train thêm / mô hình lớn hơn sẽ xuống nữa. Đây là **cận TRÊN của dư địa**, không
   phải cận dưới.
3. **`5.594 bits/token` vẫn là LM YẾU** (≈ perplexity 48). Đối chiếu: GPT-2 small trên WebText
   ~4.6 bits/token; LLM hiện đại in-domain ~3–3.6. **Và held-out lệch miền** (train =
   wiki/hội-thoại/từ-điển, held-out = essays vi-alpaca) ⇒ số này còn bị phạt oan. Nó **THẬT**,
   nhưng **chưa đẹp** — và bits/byte **không** nói trực tiếp về độ trôi chảy khi sinh.

#### 7.8-unlock. ⇒ G10 MỞ

Điều kiện §7.10f ghi TRƯỚC khi chạy: `gap ≥ 0.5` **VÀ** `Δ < −0.1`.
**Đạt cả hai** (0.811 · −0.269) ⇒ **G10 (thay kênh `W_seq` bằng LM) MỞ.**

Ý nghĩa cụ thể cho G10: kênh `W_seq` đang là **7.088 bits/token**; LM cùng corpus/held-out là
**5.594** ⇒ **thay đúng mắt xích yếu nhất bằng thứ tốt hơn 1.494 bits/token**. Đó là lần đầu
TK_042 có một thay-đổi-kiến-trúc **được số hậu thuẫn trước khi làm**.

→ Mô hình **HỌC được ngữ cảnh**, không chỉ memorize. Phân tích:

- Eval trajectory: @2000: 1.298 → @8000: 1.093 (breakthrough) → @18000: 1.023 (best region) → @26000: 1.009 (plateau)
- Loss giảm smooth: 10.459 → 3.505 (step 9200) → 3.188 (step 26000) → 3.468 (final) — học được rồi overfit nhẹ
- Đếm **KHÔNG** capture ngữ cảnh dài; transformer **CÓ**

**Kết luận (sửa từ §7.8 cũ):**

> **→ TRANSFORMER HẬP LẠI ĐÁNG KỂ ⇒ GAP THẬT ⇒ B/C CÓ NGHĨA.**
> 
> A (19.7M token đủ cho đếm) là sai. Transformer học được pattern bigram miss.
> Corpus không nhỏ đến mức không học được; **19.7M là đủ để mô hình hội tụ** 
> với architecture này. Commit: SHA 15e3cc5.

**Điểm sang: B vs C?** Loại trừ A không đủ. Xem tiếp TK_042_v5 agenda.
