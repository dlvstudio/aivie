# TK_108 — Thí nghiệm 2: thay tầng attention của Gemma 4 12B bằng Mamba-2 (trên M4)

*Đêm 06→07/11/2026 · Mac M4 24 GB · MLX 0.32 · `mlx-community/gemma-4-12B-it-4bit` · script [scripts/exp2_mamba_layer.py](scripts/exp2_mamba_layer.py) · kết quả ở `out/exp2_mamba_layer/`*

## Câu hỏi

Nếu đổi tầng attention cửa sổ trượt (1024 token) của Gemma 4 12B sang Mamba thì:

1. Model mất bao nhiêu sức mạnh?
2. Model có chạy đúng với prompt và loại data không có lúc train không?
3. Model có chạy đúng với chuỗi dài hơn lúc train không?

## Tóm tắt

- **Thay 3/40 tầng cửa sổ trượt (P1, ~2,5 triệu token, 2,2 giờ trên M4):** Mamba lấy lại khoảng **70% phần mất đi** so với bỏ hẳn attention. Model vẫn trả lời gần giống hệt bản gốc, kể cả gọi tool.
- **Tổng quát tốt:**
  - Train chỉ bằng chat tiếng Việt (đã bỏ văn phòng, code, dịch), nhưng trên **code tiếng Anh** và **văn phòng** vẫn đạt mức tương đương.
  - Chuỗi **8K** (gấp 4 lần độ dài lúc train) không tệ đi.
- **Loại data quan trọng:** train bằng hội thoại tốt hơn hẳn train bằng web thô (KL 3 tầng 0,52 so với 0,91). Gemma bản chat vốn đoán rất kém trên văn bản thô.
- **Chưa đạt "không mất gì":**
  - Mới thay 3 tầng, đã có khoảng 15–20% từ đoán khác bản gốc.
  - Khi sinh chữ có vài lỗi nhỏ mà bản gốc không mắc.
  - Tầng 14 khó nhất.
- **P2 (so ở đầu ra cuối) trên M4: chưa chạy được.**
  - Ở bf16, gradient qua 48 tầng sai hướng.
  - Ở fp32, gradient đúng hướng nhưng train vẫn làm KL tăng, kể cả với lr rất nhỏ.
  - Cần làm lại trên GPU với batch lớn hơn (xem mục 5).

## 1. Cách làm

- **Mamba-2 khởi tạo từ chính attention của tầng đó:**
  - Q→C, K→B, V→x, O→out; giữ q_norm và k_norm.
  - Thêm phần mới: conv1d (khởi tạo là đồng nhất), dt, A, D, và chuẩn hoá đầu ra theo head.
  - Mức nhớ mỗi head từ 8 đến 2048 token.
  - Mỗi tầng 47 triệu tham số.
- **P1:** mỗi Mamba học bắt chước đầu ra attention tại tầng của nó, với đầu vào lấy từ model gốc. Loss là MSE sau `post_attention_layernorm`.
- **Tầng thay:** 1, 14, 30 (đầu, giữa, cuối; đều là tầng cửa sổ trượt). 1200 bước × 2048 token.
- **Đo trên data chưa thấy** (8 chuỗi × 2048, riêng 8K là 2 chuỗi × 8192):
  - chat tiếng Việt (multiturn, khác nguồn với data train);
  - văn phòng tiếng Việt;
  - code tiếng Anh;
  - hội thoại dài 8K.
- **Chỉ số:**
  - KL(gốc ‖ bản thay) trung bình mỗi token: càng thấp càng giống gốc.
  - Tỉ lệ từ đoán đầu bảng trùng với gốc.
- **So sánh:** gốc, **bỏ** (đầu ra attention của tầng = 0), **Mamba**, và **Mamba chỉ khởi tạo, chưa train**.

## 2. Kết quả chính (P1, train bằng chat tiếng Việt)

KL / tỉ lệ đoán trùng với gốc:

| Bộ đo | Bỏ 3 tầng | Mamba 3 tầng | Mamba tầng 1 | tầng 14 | tầng 30 |
|---|---|---|---|---|---|
| chat VI | 1,69 / 63% | **0,52 / 81%** | 0,30 / 88% | 0,45 / 84% | 0,08 / 90% |
| văn phòng VI (chưa thấy) | 2,09 / 61% | **0,63 / 80%** | 0,28 / 89% | 0,43 / 85% | 0,10 / 90% |
| code EN (chưa thấy) | 1,37 / 74% | **0,48 / 85%** | 0,22 / 92% | 0,34 / 88% | 0,08 / 92% |
| dài 8K (gấp 4 lần lúc train) | 1,31 / 69% | **0,39 / 85%** | 0,24 / 90% | 0,31 / 88% | 0,07 / 92% |

**Mamba chỉ khởi tạo, chưa train** (chat VI): 3 tầng có KL 1,42 và 65% đoán trùng, gần mức bỏ hẳn (1,69). Như vậy riêng việc chép Q/K/V/O sang chưa đủ; train P1 mới là phần quyết định.

**Train bằng web thô** (cùng cấu hình, đo trên các bộ hội thoại): 3 tầng có KL 0,91 / 0,92 / 0,74 / 0,69, kém rõ so với train bằng hội thoại. Lý do: Gemma bản chat đoán rất kém trên văn bản thô (perplexity hàng nghìn). Ollama cũng vậy: "The capital of France is" cho ra "...-". Nên dạy Mamba trên văn bản thô là dạy nó bắt chước một chế độ chạy bất thường của model.

## 3. Sinh chữ thật (6 ca trong bộ 28 ca, 3 tầng thay)

- Lập luận gần như **y hệt** bản gốc: cùng cấu trúc, cùng ý, cùng số liệu (119.101.600 / 19.101.600 đ). Lệnh gọi tool giống hệt: `get_weather("Đà Nẵng","2026-11-07")`.
- **Lỗi nhỏ mới xuất hiện:**
  - Chính tả: sửa "dành" thành "đạt", trong khi gốc sửa đúng là "giành".
  - Lãi suất: một dòng viết "119,101.6" (sai dấu), tổng cuối vẫn đúng.
  - COUNTIFS: lặp chữ "Condition Condition".
- **Tốc độ:** ~23,7 giây so với ~22 giây của gốc. Chậm hơn vì Mamba đang chạy bằng Python/MLX, chưa có kernel riêng.
- Ghi chú: phần suy nghĩ của model chiếm hết giới hạn 700 token, nên bài so này chủ yếu so cách lập luận. File: `out/exp2_mamba_layer/chat/gen_train.json`.

## 4. Đọc kết quả

1. **Tổng quát sang loại data chưa thấy: ĐÚNG.**
   - Train bằng chat tiếng Việt nhưng code tiếng Anh còn giữ tốt hơn chat tiếng Việt.
   - Mamba học "phép trộn thông tin giữa các token" chung, không học thuộc nội dung.
2. **Độ dài: ĐÚNG.**
   - Train ở 2K, chạy ở 8K vẫn tốt, thậm chí KL thấp hơn.
   - Lý do: Mamba chỉ thay phần nhìn gần; nhìn xa vẫn do 8 tầng toàn cục đảm nhận.
3. **Nhưng mất mát cộng dồn.**
   - Mỗi tầng còn sai một chút (KL 0,08–0,45), và 3 tầng cộng lại là 0,4–0,6.
   - Nếu thay cả 40 tầng chỉ bằng P1, model chắc chắn hỏng rõ.
   - Muốn thay nhiều tầng bắt buộc phải có **P2** để các tầng bù sai số cho nhau, cộng nhiều data hơn.
   - Thí nghiệm thay 8 tầng đang chạy để đo tốc độ cộng dồn (mục 6).
4. **Tầng không giống nhau.**
   - Tầng 30, ngay sau một tầng toàn cục, dễ thay (KL 0,08).
   - Tầng 14 khó: sai số theo từng tầng chỉ 0,005 mà KL đầu ra vẫn 0,45, tức là tầng nhạy.
   - Hệ quả: nên chọn thay các tầng dễ, giữ lại attention ở các tầng nhạy.

## 5. P2 trên M4 — chưa thành, và vì sao

| Lần | Cấu hình | KL kiểm tra (bước 0 → sau) | Ghi chú |
|---|---|---|---|
| a | bf16, L=1024 | — | 29,5 GB, tràn swap, 22 token/giây, bỏ |
| b | bf16, L=512, lr 5e-5 | KL train 0,30 → 1,7 | tăng |
| c | fp32, L=256×4, lr 2e-5 | 0,49 → 0,76 (bước 25) | tăng |
| d | fp32, L=256×4, lr 2e-6 | 0,49 → 0,48 → 0,64 (bước 50) | giảm rồi tăng |

**Những gì đã kiểm tra:**
- Ở **bf16**, đi theo hướng gradient làm KL **tăng**. Gradient qua 48 tầng bf16 nhiễu tới mức không còn là hướng giảm.
- Ở **fp32**, gradient đúng hướng trên một batch cố định: KL 0,279 → 0,239.
- Gradient của từng thành phần (attention trượt, attention toàn cục, MLP, đầu ra) đều khớp với sai phân.

**Vậy lỗi nằm ở cách tối ưu, không phải ở công thức.** Nghi vấn:
1. **Batch quá nhỏ.** 1024 token cho mỗi lần cập nhật, trong khi kế hoạch GPU là khoảng 0,5 triệu token.
2. **Adam dịch đồng loạt 47 triệu trọng số mỗi tầng** theo tín hiệu yếu từ đầu ra.
3. **Lệch độ dài.** Train ở 256 nhưng kiểm tra ở 512.

**Cách làm khi lên GPU:**
- Batch lớn hơn khoảng 100 lần, chuỗi 2K–8K.
- Kèm loss P1 (MSE từng tầng) làm "neo" cho Mamba khỏi lệch khỏi bản đã học.
- Hoặc chỉ mở cho P2 chỉnh các tham số nhỏ (dt, A, D, chuẩn hoá, out_proj).

## 6. Thay 8 tầng — model sụp, và vì sao

P1 800 bước bằng data hội thoại cho 8 tầng 1, 3, 7, 9, 14, 20, 26, 30. Thư mục: `out/exp2_mamba_layer/chat8/`.

**Thay cả 8 tầng cùng lúc (KL / đoán trùng):**

| Bộ đo | Bỏ 8 tầng | Mamba 8 tầng |
|---|---|---|
| chat VI | — | 4,98 / 32% |
| văn phòng VI | 5,85 / 23% | 5,28 / 31% |
| code EN | 3,09 / 53% | 1,41 / 71% |
| dài 8K | 4,78 / 27% | 4,32 / 39% |

**Chẩn đoán theo nhóm tầng** (chat VI, 2 chuỗi × 2048; [scripts/exp2_diag_subsets.py](scripts/exp2_diag_subsets.py)):

| Thay | KL | Đoán trùng |
|---|---|---|
| riêng tầng 1 / 3 / 7 / 9 | 0,29 / 0,10 / **0,97** / **2,26** | 87 / 93 / 72 / 55% |
| riêng tầng 14 / 20 / 26 / 30 | 0,39 / 0,45 / 0,21 / 0,08 | 84 / 81 / 86 / 90% |
| tầng 1, 3 → thêm 7 → thêm 9 | 0,29 → 0,95 → **4,47** | 88 → 72 → 32% |
| tầng 14, 20, 26, 30 (nửa sau) | **0,82** (thấp hơn tổng riêng lẻ ≈ 1,13) | 73% |
| tầng 1, 14, 30 | 0,62 | 78% |
| cả 8 tầng | 5,09 | 27% |

**Đọc kết quả:**
- **Thủ phạm là tầng 7 và 9**, nằm giữa hai tầng toàn cục 5 và 11.
  - Sai số theo từng tầng của chúng rất thấp (~0,008), nhưng đầu ra hỏng nặng.
  - Khi thay cùng nhau, sai số **khuếch đại**: 0,97 và 2,26 riêng lẻ, nhưng 4,47 khi đi chung.
  - Giả thuyết: các tầng đầu–giữa làm những việc cần độ chính xác trong cửa sổ, như chép hoặc tra lại đúng token. Đây là điểm yếu đã biết của Mamba.
- **Nửa sau model dễ thay,** và thay nhiều tầng còn ít hơn tổng từng tầng.
- **Sai số theo từng tầng KHÔNG dự đoán được độ nhạy.** Phải đo KL ở đầu ra cho từng tầng.
- **Hệ quả:** không thể đổi cả 40 tầng cửa sổ trượt một cách mù quáng.
  1. Quét độ nhạy cả 40 tầng.
  2. Chỉ đổi tầng dễ, giữ attention ở tầng nhạy, tức hybrid chọn lọc theo đo đạc.
  3. P2 có cứu được tầng 7 và 9 hay không thì chưa biết.

## 6b. Phương án A cho P2 trên M4 — vẫn không giảm, và lý do đã rõ

**Phương án A:**
- Chỉ chỉnh ~0,3 triệu tham số nhỏ (dt, A, D, conv, chuẩn hoá); khoá Q/K/V/O.
- 8 nghìn token mỗi lần cập nhật (32 × 256), fp32, neo P1 λ = 2.
- Khởi tạo từ bộ 3 tầng 1/14/30.

**Kết quả:** KL kiểm tra 0,492 → 0,533 → 0,525 → 0,586 (bước 0 / 5 / 10 / 15). KL trên chính batch train cũng tăng nhẹ (đường trung bình 1,15 → 1,24). Mình dừng ở bước 16 (`out/exp2_mamba_layer/kdA/`).

**Phép thử trực tiếp** ([scripts/exp2_kd_probe.py](scripts/exp2_kd_probe.py)): gradient fp32 gộp từ 8 batch × 512 token, đi một bước đã chuẩn hoá có độ dài η:

| η | KL trên chính 8 batch đó | KL trên bộ kiểm tra cố định |
|---|---|---|
| 0 | 1,22 | 0,492 |
| 0,01 | **0,72** | 0,508 |
| 0,03 | 1,05 | 0,754 |
| 0,1 | 1,03 | 0,553 |
| 0,3 | 1,11 | 0,704 |

**Kết luận:**
- **Mặt loss của P2 cực kỳ dốc và gồ ghề.** Độ lớn gradient khoảng 290. Một bước rất nhỏ (0,01) đã làm KL trên chính batch đó giảm 40%, nhưng **KL trên data khác lại tăng**, và kết quả không đơn điệu theo η.
- Như vậy gradient từ vài nghìn token chủ yếu **khớp với riêng batch đó** chứ không mang tín hiệu chung.
- P2 bắt buộc phải gộp gradient trên **rất nhiều token cho mỗi lần cập nhật** (hàng trăm nghìn), đúng như các bài đã công bố làm trên GPU.
- M4 chỉ đạt khoảng 50 token/giây khi lan ngược qua cả model, tức khoảng 3 giờ cho mỗi lần cập nhật 0,5 triệu token. Không khả thi.
- **Đây không phải bằng chứng P2 sai**, mà là bằng chứng P2 cần quy mô GPU. Kiểm chứng P2 vẫn còn mở: cần pilot GPU khoảng 20–40 USD, batch khoảng 0,5 triệu token, 3 tầng 1/14/30 trước, tiêu chí là KL kiểm tra giảm đều xuống dưới 0,49.

## 7. Kết luận cho kế hoạch

- **Về mặt kỹ thuật thì làm được:** khởi tạo từ attention, train P1, tổng quát sang data lạ và chuỗi dài đều ổn.
- **Lời khuyên hiện tại vẫn giữ:** dùng Gemma 4 nguyên bản kèm LoRA và RAG nếu cần sức mạnh. Đổi sang Mamba chỉ đáng làm nếu P2 trên GPU kéo được KL của nhiều tầng xuống gần 0.
- Bước rẻ tiếp theo:
  1. Xem kết quả thay 8 tầng.
  2. Nếu vẫn muốn đi tiếp: pilot GPU khoảng 20–40 USD, P2 batch lớn, khoảng 12 tầng dễ nhất.

## 8. Quét độ nhạy cả 40 tầng sliding (07/11, M4)

Mamba P1 cho mọi tầng sliding (chat8 cho 1,3,7,9,14,20,26,30 + sg1–sg4, 500 bước L=1024 mỗi nhóm 8 tầng). Đo KL so với gốc trên 2 chat_vi + 1 code_en × 2048 (`scripts/exp2_sensitivity.py`, kết quả `out/exp2_mamba_layer/sensitivity/`).

**Từng tầng (KL Mamba; trong ngoặc KL bỏ):**
- Rất dễ ≤0.10: 43 (0.26), 40, 37, 39, 30, 31, 45, 3, 25, 34 — gần như toàn nửa sau (30–45).
- Dễ 0.10–0.30: 27, 44, 2, 36, 26, 24, 4, 1, 22, 33, 38, 46 (bỏ 4.8!).
- Trung bình 0.30–0.50: 10, 21, 42, 32, 0 (bỏ 5.0!), 6, 14, 18, 19, 20, 16, 15.
- Khó 0.6–0.8: 12, 13 (bỏ 10–12 — tầng sống còn), 7, 28.
- Hỏng: 8 (1.51 > bỏ 0.66), 9 (1.67).

**Tổ hợp tham lam (thêm dần tầng dễ nhất):** 1→0.05 · 3→0.19 · 4→0.27 · 6→0.48 · 8→0.61 · 10→0.70 · 12→0.87 · 13→0.92 · 14→1.11 · 16→1.34.
- Gần CỘNG TÍNH (8 tầng: tổng đơn lẻ 0.59 vs đo 0.61) — khác hẳn chat8 (KL 4.98, khuếch đại do 7/9).
- Chọn đúng tầng: 8 tầng Mamba KL 0.61 thay vì 4.98 (tốt hơn 8×).

**Kết luận:** chỉ P1 (không P2) ta thay được ~4–6 tầng (8–12% số tầng) mà KL<0.5, ~10–13 tầng ở KL<1. Nửa sau mạng (sau tầng global 29) là chỗ đặt Mamba; giữ attention ở 7–9, 12–13, 28. P2 trên GPU là bước cần để đẩy lên 20–40 tầng.

### 8b. Bản lai 8 tầng sinh câu (`out/exp2_mamba_layer/hyb8/gen_hyb8.json`)
Tầng [43,40,37,39,30,31,45,3] ghép từ chat8/sg3/sg4. 6 ca thật, 300 token (vẫn trong pha thought).
- Nội dung/suy luận giữ gần nguyên: email, tóm tắt 3 ý, thành ngữ (giải nghĩa đúng), lãi kép 119.101.600 đúng, chọn SUMPRODUCT/COUNTIFS đúng, hiểu đề SQL đúng.
- Lỗi đặc trưng: MẤT DẤU ĐÓNG / chép chính xác — `"Huỷ)`, `A2:500` thay `A2:A500`, `'2026-10-01 AND` thiếu nháy, `AND_at_date` thay `created_at`, `to__city`, thiếu `**`/backtick đóng. → các tầng nửa sau làm việc khớp-cặp/chép chính xác (việc của attention); hại nhất với code/SQL/công thức.
- Tốc độ 12 s vs 10 s (Mamba chưa có kernel tối ưu).

## 9. Thay TẦNG GLOBAL bằng Mamba (07–08/11, M4)
Lý do: KV của 40 tầng sliding đã cố định (cửa sổ 1024) → thay chúng gần như không lợi; chi phí 256K nằm ở 8 tầng global. Mixer tổng quát hoá cho global (H=16, Hkv=1, D=512, K=V, trí nhớ 8→32768). P1: gla [5,11,17,23] L2048×500; glb [29,35,41,47] L1024×1000 (L2048 bị swap). `scripts/exp2_global.py`, kết quả `out/exp2_mamba_layer/gl/global_eval.json`.

**KL từng tầng (bỏ → Mamba, 2K / 8K):** 5: 0.75→0.35/0.27 · 11: 1.64→**16.0**/14.8 (hỏng) · 17: 0.99→0.50/0.50 · 23: 0.68→0.27/0.25 · 29: 0.63→0.14/0.16 · 35: 0.56→0.07/0.06 · 41: 0.50→0.19/0.19 · 47: **12.1**→0.58/0.47. 8K ≈ 2K dù train ≤2K.
Tổ hợp: 7 tầng (trừ 11) KL 1.53; thêm bỏ 11 → 2.69; cả 8 Mamba 12.9; bỏ cả 8 → 16.1.

**Kim đáy bể (mã 4 số trong ~7.600 token, 5 vị trí 4–49%):** gốc 5/5; Mamba TỪNG tầng global (5,17,23,29,35,41,47) đều 5/5; tầng 11: 0/5; **7 tầng cùng lúc (chỉ còn attention global ở 11): 0/5** — trả lời trôi chảy nhưng bịa ("Mã số là 123", "không có trong tài liệu").
→ Thay lẻ một tầng thì các tầng global còn lại vẫn tra cứu thay; Mamba P1 KHÔNG tự học được truy hồi chính xác tầm xa. Khớp với lỗi chép-chính-xác ở §8b.

**Tầng 11:** nmse_post 0.11 (không NaN, không phải BOS) mà KL 16 → mixer sinh hướng có hại ngay cả khi lỗi sau-norm nhỏ; giữ attention ở 11.

### 9b. Giữ bao nhiêu tầng global attention? (08/11)
Mamba P1 ở các tầng global còn lại; KL 2K · kim 5 ca ~7.600 token:
| giữ attention | KL 2K | kim |
|---|---|---|
| 11,47 | 0.98 | 0/5 |
| 11,29 | 1.38 | 0/5 |
| 11,23,47 | 0.92 | 0/5 |
| 11,29,47 | 0.86 | 0/5 |
| 5,11,29,47 | 0.77 | 0/5 |
| **11,23,35,47** | 0.83 | **5/5** |
| **11,17,29,41** | 1.00 | **5/5** |
- Cần ≥4/8 global attention VÀ rải đều (cách ~12 tầng); 5,11,29,47 (hở 11→29) hỏng dù KL thấp nhất → KL không dự báo truy hồi.
- Hỏng thì bịa trôi chảy ("Mã số là 101", "123456").
- Kết luận chỉ-P1: tiết kiệm ~50% KV ở ngữ cảnh dài (4/8 global → Mamba). Hơn nữa cần P2/train truy hồi trên GPU.
