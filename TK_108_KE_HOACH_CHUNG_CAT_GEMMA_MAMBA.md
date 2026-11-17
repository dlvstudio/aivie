# TK_108 — Phương án 2: chưng cất Gemma 4 sang kiến trúc lai Mamba · kết quả kiểm tra P0

*Ngày 06/11/2026 · kiểm tra trên M4 (24 GB) · tiếp theo [đánh giá Gemma 4 26B](TK_108_DANH_GIA_GEMMA4_THUC_TE.md)*

## Tóm tắt

| Mục kiểm tra P0 | Kết quả |
|---|---|
| Giấy phép | ✅ **Apache 2.0**, không gated. Được sửa kiến trúc, train tiếp và phát hành. |
| Kiến trúc 12B có hợp để thay bằng Mamba không | ✅ 48 tầng, gồm **40 tầng attention cửa sổ trượt** và 8 tầng toàn cục (tỉ lệ 5:1). Tầng cửa sổ trượt map sạch sang Mamba-2. |
| Chất lượng 12B gốc (đây là "trần" của bản lai) | ✅ **9,3/10** trên 22 ca code tiếng Việt, 21/22 dùng được. 26B-A4B được 9,8. |
| Chạy trên M4 (MLX / llama.cpp) | ✅ Khả thi: MLX đã có cả `gemma4` lẫn `mamba2`, ghép được thành một model. |
| **Lợi ích thực tế khi đổi sang Mamba** | ❌ **Gần như bằng 0.** Gemma 4 vốn đã tiết kiệm bộ nhớ ngữ cảnh: tầng cửa sổ trượt chỉ nhớ 1024 token, tầng toàn cục dùng chung K=V với 1 đầu. Bộ nhớ ngữ cảnh ở 32K token chỉ khoảng 0,6 GB. Đổi sang Mamba tiết kiệm khoảng 0,25 GB và gần như không nhanh hơn (chi tiết ở mục 3). |

**Kết luận:**
- Về kỹ thuật thì làm được. Nhưng **không đáng bỏ ra khoảng 2–3 nghìn USD** cho bản 12B: bỏ tiền ra mà khó đạt lại 9,3 điểm, trong khi gần như không nhanh hơn và không nhẹ hơn.
- **Khuyên:** muốn có trợ lý tiếng Việt thật thì làm **phương án 1**, tức LoRA tiếng Việt trên Gemma 4 12B hoặc 26B-A4B (mục 5).
- Nếu mục tiêu là có **model kiến trúc Mamba của riêng mình**, chọn hướng rẻ hơn: chưng cất Gemma vào chính bản aivie 2,5B (mục 6).

---

## 1. Gemma 4 12B gốc trên 22 ca (cùng bộ đề và cách chấm với c6l1, Qwen, Gemma 26B)

`ollama gemma4:12b`, nén 4-bit, nặng 7,7 GB. Cài đặt: `MAXNEW=6144`, `CTX=16384`. Không ca nào bị cắt (dài nhất 4913 token). Trung bình **~44 giây mỗi lượt, ~43 token/giây**.

| # | Ca | Điểm | Nhận xét |
|---|---|---|---|
| 01 | SĐT Việt Nam | 9 | Chạy thử 14 ca, tất cả đúng. Chỉ hơi dễ dãi khi chấp nhận dạng `+840…`. |
| 02 | pandas doanh thu | 10 | Đã chạy: ra Excel 2 sheet. Chỉ tạo file mẫu **khi chưa có** `sales.csv` (an toàn hơn bản 26B). |
| 03 | lỗi `[]` mặc định | 10 | Đã chạy: `['trà']`, `['cà phê']`. |
| 04 | KeyError 'email' | 10 | Giải thích đúng, có 3 cách sửa. |
| 05 | SQL top 5 Hà Nội | 9 | Đúng. Nhưng nói sai rằng `<= '2024-12-31'` "vẫn ổn" với cột có giờ phút (thực tế sẽ sót ngày 31/12). |
| 06 | debounce | 8 | Debounce chuẩn. Có dòng `fetch('/api/search?q=…encodeURIComponent')` thật nhưng để trong chú thích, còn đang chạy dữ liệu giả. |
| 07 | trang đăng nhập | 10 | Hợp điện thoại, chữ ô nhập 16px, có Quên mật khẩu. |
| 08 | FastAPI todo | 10 | Có Create, Update, Response, trả 404 khi sửa và xoá, xoá trả 204. |
| 09 | regex email bỏ trùng | 10 | Đã chạy: đúng và giữ thứ tự. |
| 10 | viết pytest | 10 | Có parametrize, ca biên, kiểm tra tính chất và lỗi. Đã chạy: **11/11 qua**. |
| 11 | LRUCache O(1) | 10 | Đã chạy: đúng chuỗi kiểu LeetCode. |
| 12 | bash sao lưu, giữ 7 bản | 9 | `ls -1tr … \| head -n -7 \| xargs -r rm` đúng trên Linux. Không kiểm tra lỗi của tar, biến không có nháy kép. |
| 13 | Excel | 10 | Đủ VLOOKUP, XLOOKUP, IFERROR, `SUMIF "SP0*"`. |
| 14 | Python sang JS | **3** | **Chạy là lỗi `ReferenceError: clean_word`.** Model còn để lại dòng "Sửa lỗi nhẹ ở đây" trong code, và nếu sửa tên biến thì mỗi từ lại bị đếm hai lần. Kết quả nó in ra là bịa, không phải output thật. |
| 15 | review SQLite | 9 | Bắt được SQL injection (có ví dụ), mật khẩu lưu thô và lỗi không đóng kết nối. Bản sửa dùng sha256 không có salt (có ghi chú nên dùng bcrypt). |
| 16 | hội thoại 3 lượt | 10 | Ba lượt nối nhau đúng. Đã chạy: **3/3 test qua**. |
| 17 | trò chơi đoán số | 9 | Đủ chức năng. Có in câu thừa "bạn có 5 giây để suy nghĩ" dù không có giới hạn giờ. |
| 18 | git lỡ commit `.env` | 10 | `rm --cached`, thêm `.gitignore`, rồi `commit --amend`. Có lưu ý trường hợp đã push và nên tạo `.env.example`. |
| 19 | asyncio + aiohttp | 10 | `Semaphore(3)`, `ClientTimeout(10)`, gather, tách lỗi timeout, HTTP và kết nối. |
| 20 | set và list | 10 | Dễ hiểu. Đã chạy ví dụ đo thời gian. |
| 21 | refactor giảm giá | 10 | Ba cách (dict lambda, bảng cấu hình, Strategy). Đã chạy: kết quả khớp code gốc. |
| 22 | React todo | 9 | Đúng hết. Chỉ dùng `onKeyPress` (đã lỗi thời). |

**Trung bình 9,3/10 · dùng được (≥7): 21/22 · hỏng: 1/22 (ca 14).**

So sánh với các bản khác: c6l1 được 1,3 · Qwen2.5-Coder 7B được 7,3 · Gemma 4 26B-A4B được 9,8.
- Bản 12B chỉ kém 26B một chút, nhưng nặng chỉ bằng một nửa (7,7 GB so với 16 GB) và chạy cùng tốc độ.
- Nếu làm chưng cất, đây là **mức trần**: bản lai giỏi lắm cũng chỉ đạt khoảng 9,3.

---

## 2. Kiến trúc Gemma 4 12B (đọc trực tiếp từ config và tensor)

- **Kích thước:** hidden 3840, MLP 15360, từ vựng 262 144 (dùng chung bảng nhúng). `final_logit_softcapping` 30. Một file `model.safetensors` nặng **23,9 GB** ở bf16.
- **Tầng toàn cục** nằm ở vị trí [5, 11, 17, 23, 29, 35, 41, 47]. 40 tầng còn lại là cửa sổ trượt 1024 token.
- **Tầng cửa sổ trượt:**
  - 16 đầu Q, 8 đầu KV, head 256. `q_proj [4096,3840]`, `k_proj`/`v_proj [2048,3840]`, `o_proj [3840,4096]`.
  - Có `q_norm`/`k_norm`. RoPE 1e4.
- **Tầng toàn cục:**
  - Head 512, **1 đầu KV, K = V** (không có `v_proj`). `q_proj [8192,3840]`, `k_proj [512,3840]`.
  - Rotary một phần 0,25, RoPE 1e6.
- **Thành phần khác:** có `layer_scalar` và 4 norm mỗi tầng. Model đầy đủ (`Gemma4UnifiedForConditionalGeneration`) có kèm phần ảnh và âm thanh; bản lai chỉ cần phần chữ.
- **Cách map một tầng cửa sổ trượt sang Mamba-2** (theo "Mamba in the Llama" / MOHAWK):

  | Thành phần Mamba-2 | Lấy từ Gemma | Chi tiết |
  |---|---|---|
  | C | Q | |
  | B | K | ngroups = 8, d_state = 256 |
  | x | V | lặp đôi 8 đầu thành 16 đầu, d_inner = 4096 |
  | out_proj | O | |
  | `dt`, `A`, `D`, `conv1d` | — | khởi tạo mới, rất ít tham số |

- **E4B không hợp để làm pilot**, vì có PLE (`hidden_size_per_layer_input` 256) và 18 tầng dùng chung KV, phức tạp hơn hẳn.
- **31B** có 60 tầng (50 cửa sổ trượt, 10 toàn cục), cùng kiểu với 12B.

## 3. Vì sao đổi sang Mamba gần như không được lợi (phát hiện chính của P0)

Mamba có lợi khi nó thay **attention toàn cục**, vì loại này có bộ nhớ ngữ cảnh và chi phí tăng theo độ dài. Nhưng ở Gemma 4 12B:

| Thành phần | Bộ nhớ ngữ cảnh (bf16) | Nếu đổi sang Mamba |
|---|---|---|
| 40 tầng cửa sổ trượt | Cố định 1024 × (2048 K + 2048 V) × 2 byte ≈ 8 MB mỗi tầng, tổng **~0,34 GB**, không tăng theo độ dài | Trạng thái Mamba 4096 × 256 × 2 byte ≈ 2 MB mỗi tầng, tổng ~0,08 GB, **tiết kiệm ~0,25 GB** |
| 8 tầng toàn cục | 8 × 512 × 2 byte = **8 KB mỗi token**: ~0,26 GB ở 32K, ~1 GB ở 128K | Giữ nguyên, vì đây là các tầng giúp nhớ xa |
| Trọng số | 7,7 GB (4-bit) đến 24 GB (bf16) | Gần như không đổi |

- **Về tốc độ:**
  - Khi sinh chữ, thời gian chủ yếu nằm ở việc đọc trọng số. Attention trong cửa sổ 1024 token đã rất rẻ.
  - Khi đọc đề, phần cửa sổ trượt vốn đã tăng tuyến tính theo độ dài.
  - Vì vậy Mamba **không đổi được bậc phức tạp**, chỉ bớt vài phần trăm.
- **Về chất lượng:** chưng cất luôn làm mất một ít. Các bài công bố mất khoảng 1–5 điểm benchmark khi đổi 50–75% số tầng. Bản lai sẽ ở khoảng **8–9 điểm** trong khi bản gốc đã 9,3.
- **Còn nếu đổi luôn 8 tầng toàn cục** để hưởng lợi khi ngữ cảnh rất dài: đây chính là các tầng giữ khả năng tìm lại thông tin ở xa. Theo kinh nghiệm ở TK_105/TK_107, khả năng nhớ xa sẽ sụt mạnh. Không nên làm.

→ **Gemma 4 vốn đã là một kiến trúc "lai tiết kiệm".** Phương án 2 trên 12B được cái danh "kiến trúc Mamba", nhưng đổi lại bằng tiền và một phần chất lượng.

## 4. Nếu vẫn muốn làm phương án 2 (để tham khảo, không khuyên)

| Pha | Việc | Data / tài nguyên | Cổng qua pha |
|---|---|---|---|
| P0 | Kiểm tra (**đã xong**, file này) | — | — |
| P1 | Căn từng tầng: Mamba-2 học bắt chước output attention (MSE) với 20 tầng cửa sổ trượt (50%) | 0,5–1 tỷ token, 1–2 GPU H100, ~1 ngày | MSE từng tầng < 5% |
| P2 | Chưng cất toàn model bằng KL trên logits. Tăng dần 50% lên 100% số tầng cửa sổ trượt; độ dài 2K, rồi 8K, rồi 16K | 3–8 tỷ token, 8×H100, ~5 ngày | Perplexity ≤ gốc +5% |
| P3 | SFT tiếng Việt với data cân lại, rồi vòng thầy/DPO | Data SFT C6 (đã có trên Mac) | — |
| P4 | Đo lại | 22 ca thực tế, HE/MBPP/C7/tool/office, VMLU, needle 8–32K | ≥ 90% điểm của bản gốc (≥ 8,4/10) |
| P5 | Chạy trên M4 | MLX: ghép `gemma4_text.py` với mixer Mamba-2 | ≥ 40 token/giây |

- **Chi phí:** pilot 50% khoảng 500–800 USD; bản 100% khoảng 2–3 nghìn USD (~1000 giờ H100).
- **Data cần dựng lại:** `data/cpt` (17 GB) và `data/cx3` (8 GB) đã mất cùng box. Script dựng lại vẫn còn (`scripts/build_cpt_data.py`, `scripts/build_code_expert_data.py`).

## 5. Khuyên: phương án 1, LoRA tiếng Việt trên Gemma 4 (giữ nguyên kiến trúc)

- **Nền:** Gemma 4 12B (cân bằng) hoặc 26B-A4B (tốt nhất). Cả hai đã đạt 9,3–9,8 điểm code mà chưa cần train gì.
- **Mục tiêu LoRA:**
  - Văn phong và danh tính "aivie".
  - Tool-use đúng định dạng của mình.
  - Office/Excel và kiến thức Việt Nam.
  - Trả lời gọn hơn, vì cả hai bản đều dài dòng.
- **Data:** SFT C6, data vòng thầy, tool, office, tất cả đã sao lưu trên Mac (`box_backup/data_1105/box_data_text.tgz`). Phải **cân lại** để tránh lỗi "thấy code là sửa" như c6l1.
- **Chi phí:**
  - QLoRA 12B trên 1 GPU A100/H100 trong vài giờ, khoảng **20–60 USD**.
  - Hoặc thử MLX-LoRA ngay trên M4 với bản 12B 4-bit (chậm nhưng không tốn đồng nào).
- **Cổng chấp nhận:** 22 ca không tụt (≥ 9,0), danh tính và tool đúng định dạng, VMLU không giảm.

## 6. Nếu cái cần là "model Mamba của riêng mình": chưng cất vào aivie 2,5B

- **Thầy:** Gemma 4 12B hoặc 26B sinh câu trả lời và logits (top-k) cho bộ đề đa dạng.
- **Trò:** chính bản hybrid Mamba 2,5B của mình (c6 base). Ở cỡ nhỏ này Mamba mới thật sự có lợi: nhanh và nhẹ trên M4.
- **Giới hạn:** "hiểu = cỡ model" (tường tri thức) vẫn còn, nên kỳ vọng thực tế là **5–7 điểm**, tức ngang Qwen-Coder 3B–7B, chứ không đạt 9.
- **Chi phí:** khoảng 100–300 USD GPU.

## Phụ lục — chạy lại

```bash
ollama pull gemma4:12b
MODEL=gemma4:12b OUT=out/real_eval_gemma4_12b MAXNEW=6144 CTX=16384 python3 scripts/real_use_eval_ollama.py data/eval_clean/real_code_22.json
```
