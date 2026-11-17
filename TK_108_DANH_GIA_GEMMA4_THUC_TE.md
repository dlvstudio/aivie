# TK_108 — Đánh giá thực tế Gemma 4 26B-A4B trên M4 (tập trung code)

*Ngày 06/11/2026 · Ollama 0.40 · `gemma4:26b-a4b` (MoE: 26 tỷ tham số, mỗi token chạy ~4 tỷ; nén 4-bit, 16 GB) · cùng bộ 22 ca, cùng cách chấm như [c6l1](TK_108_DANH_GIA_C6L1_THUC_TE.md) và [Qwen2.5-Coder](TK_108_DANH_GIA_QWEN_CODER_THUC_TE.md)*

## Tóm tắt

| | Gemma 4 26B-A4B |
|---|---|
| **Điểm trung bình** | **9,8 / 10** |
| Dùng được ngay (≥ 7 điểm) | **22 / 22** |
| Đúng một phần (4–6) | 0 |
| Hỏng (0–3) | 0 |
| Thời gian mỗi lượt trả lời | **~47 giây** (khoảng 25–35 giây suy nghĩ trước, rồi mới trả lời) |
| Tốc độ sinh chữ | ~30–65 token/giây |
| RAM cần | ~17–18 GB (M4 24 GB chạy được, nhưng nên tắt các app nặng) |

- **Đây là model tốt nhất trong các bản đã thử trên M4.** 22/22 ca đúng yêu cầu, và mọi ràng buộc chi tiết đều được xử lý.
- Ví dụ các ràng buộc làm đúng: "+84", "giữ 7 bản", "tối đa 3 request" (có semaphore), "404 khi xoá", "bỏ trùng mà giữ thứ tự", và review bắt được cả SQL injection lẫn mật khẩu lưu thô.
- Toàn bộ code mình chạy thử đều **đúng 100%**: 7 đoạn chạy thử, 2 bộ pytest (12/12 và 3/3 test qua).
- **Cái giá phải trả:** model chậm hơn vì luôn "suy nghĩ" trước khi trả lời, và câu trả lời dài (~2000 token mỗi lượt). Phải đặt giới hạn token cao (≥ 6144), nếu không câu trả lời bị cắt giữa chừng (xem mục 3).

---

## 1. Cách đo

- **Bộ đề:** `data/eval_clean/real_code_22.json`, gồm 22 ca tiếng Việt, trong đó có một hội thoại 3 lượt.
- **Cài đặt sinh chữ:** giống các bản khác, gồm nhiệt độ 0,7, top_p 0,9 và phạt lặp 1,1. Khác ở chỗ ngữ cảnh 16 384 và giới hạn **6144 token** cho 11 ca phải chạy lại; lý do ở mục 3.
- **Script:** `scripts/real_use_eval_ollama.py`. Biến môi trường mới `MAXNEW`/`CTX` dùng để nâng giới hạn; phần suy nghĩ của model được lưu ở trường `thinking`. Câu trả lời gốc nằm ở `out/real_eval_gemma4_26b/`.
- **Chấm điểm:**
  - Mình đọc từng câu và chấm 0–10 theo tiêu chí đúng yêu cầu, code chạy được, và giải thích không sai.
  - Code thuần tính toán thì chạy thử thật trên Mac, trong thư mục tạm, có giới hạn thời gian.
  - Code đụng mạng hoặc shell thì chỉ đọc, không chạy.

## 2. Bảng điểm

| # | Ca | Điểm | Nhận xét |
|---|---|---|---|
| 01 | SĐT Việt Nam | **10** | Regex `^(\+84\|0)(3\|5\|7\|8\|9)[0-9]{8}$` đúng. Đã chạy 15 trường hợp: `+84…`, thiếu số, thừa số, `+840…` và có khoảng trắng đều xử lý đúng. |
| 02 | pandas doanh thu | 9 | Đúng hết và tự đổi `Period` sang chuỗi để xuất Excel không lỗi. Chỉ có điều nó **tự ghi đè `sales.csv` bằng dữ liệu mẫu**; có cảnh báo trong chú thích, nhưng chạy y nguyên trên dữ liệu thật thì sẽ mất file. |
| 03 | lỗi `[]` mặc định | **10** | Giải thích đúng bản chất, sửa bằng `None` và có ví dụ. Đã chạy: `['trà']`, `['cà phê']`. |
| 04 | KeyError 'email' | **10** | Đưa ra 3 cách (`.get`, `if`, sửa dữ liệu) và lời khuyên đúng. |
| 05 | SQL top 5 Hà Nội | **10** | Đủ tổng tiền và số đơn, GROUP BY theo id kèm giải thích vì sao. Lọc theo khoảng ngày để dùng được index. |
| 06 | debounce tìm kiếm | 8 | Debounce chuẩn, trang HTML chạy được ngay. Nhưng để chạy thử được, nó **đổi sang API giả (jsonplaceholder)** thay vì `/api/search` như đề, và không có `encodeURIComponent`. |
| 07 | trang đăng nhập | **10** | Đẹp, hợp điện thoại (chữ ô nhập 16px để iPhone không tự phóng to), chữ tiếng Việt, có Quên mật khẩu. |
| 08 | FastAPI todo | **10** | Tách model Create/Update/Out, id tự tăng, PATCH trạng thái, **404 cho cả sửa lẫn xoá**, xoá trả 204. |
| 09 | regex email bỏ trùng | **10** | Dùng `dict.fromkeys`. Đã chạy: đúng. |
| 10 | viết pytest | **10** | Có parametrize, các ca biên, kiểm tra tính chất (tổng, độ dài, chênh lệch ≤ 1), số âm và thông báo lỗi. Đã chạy: **12/12 qua**. |
| 11 | LRUCache O(1) | **10** | Danh sách liên kết đôi tự viết, kèm node giả ở hai đầu. Đã chạy: đúng chuỗi kiểu LeetCode. |
| 12 | bash sao lưu, giữ 7 bản | **10** | `ls -t … \| tail -n +8 \| xargs rm -f` đúng, có kiểm tra mã thoát của tar, có `mkdir -p`, gợi ý cron. |
| 13 | Excel | **10** | Đủ VLOOKUP và XLOOKUP, IFERROR, `=B2*C2`, `SUMIF(A:A,"SP0*",D:D)`, có bảng tóm tắt. |
| 14 | Python sang JS | **10** | Không dùng thư viện ngoài, strip dấu câu bằng regex tương đương. Đã chạy: `[['mèo',3],['ăn',2],['cá',1]]`, khớp bản Python. |
| 15 | review SQLite | 9 | Bắt được SQL injection (có ví dụ tấn công), mật khẩu lưu thô (chuyển sang bcrypt) và lỗi không đóng kết nối. Một lỗi nhỏ: `with sqlite3.connect()` thực ra **không đóng** kết nối, chỉ commit. |
| 16 | hội thoại 3 lượt | **10** | Ba lượt nối tiếp nhau: hàm, rồi type hint và xử lý lỗi, rồi test với `tmp_path`. Đã chạy: **3/3 qua**. |
| 17 | trò chơi đoán số | **10** | Đủ input, bắt lỗi, giới hạn 1–100, đếm lượt đúng, highscore và chơi lại. |
| 18 | git lỡ commit `.env` | **10** | `reset --soft HEAD~1`, rồi `.gitignore`, `rm --cached`, commit lại. Đúng và an toàn, có thêm gợi ý `.env.example`. |
| 19 | asyncio + aiohttp | **10** | `Semaphore(3)`, `ClientTimeout(10)`, gather, in URL lỗi và tách theo loại lỗi (timeout, status, DNS). |
| 20 | set và list | **10** | Giải thích bằng ví dụ đời thường, bảng băm O(1) so với O(n), có đo thời gian trên 10 triệu phần tử. |
| 21 | refactor giảm giá | **10** | Dùng bảng quy tắc (ngưỡng, hệ số), thêm Strategy Pattern cho trường hợp phức tạp, có test thử. |
| 22 | React todo | **10** | Hooks và form submit, id bằng timestamp, toggle không sửa trực tiếp state, có xoá và số việc chưa xong. |

---

## 3. Lưu ý quan trọng khi dùng

1. **Phải để giới hạn token cao.**
   - Gemma 4 suy nghĩ trước khi trả lời, phần này tốn khoảng 500–2500 token, và Ollama tính chung vào `num_predict`.
   - Ở lần chạy đầu với giới hạn 2048, **11/22 ca bị cắt giữa code**. Trong UI hoặc API nên để `num_predict` từ 6144 trở lên và ngữ cảnh từ 16K trở lên.
2. **Chậm hơn Qwen-Coder.** Trung bình khoảng 47 giây mỗi lượt so với 5–20 giây. Nếu chỉ cần trả lời nhanh, có thể tắt chế độ suy nghĩ (`"think": false` trong API), nhưng chưa đo chất lượng khi tắt.
3. **Trả lời dài.** Câu trả lời thường có giải thích, bảng, hướng dẫn chạy và mẹo. Rất hợp để học và đọc hiểu, nhưng dài dòng nếu chỉ cần code.
4. **RAM:** model chiếm khoảng 16 GB. Trên M4 24 GB nên chạy một mình, không chạy cùng UI c6l1 hay các model khác.
5. **Vẫn nên đọc lại.** Có 3 điểm trừ nhỏ: ghi đè `sales.csv` mẫu (ca 02), dùng API giả thay vì endpoint thật (ca 06), và nói sai rằng `with` sẽ đóng kết nối sqlite (ca 15). Không có lỗi logic nào.

## 4. Kết luận

- **Gemma 4 26B-A4B dùng được làm trợ lý code tiếng Việt chất lượng cao ngay trên M4.** Mức này gần với trợ lý thương mại cho các việc thường ngày.
- Model vừa làm đúng loại việc được hỏi, vừa giữ được mọi ràng buộc chi tiết, vừa giải thích đúng.
- **Gợi ý cách dùng:**
  - Dùng **Gemma 4 26B-A4B** cho việc cần đúng: viết module, review, test, giải thích.
  - Dùng **Qwen2.5-Coder 7B** khi cần gợi ý nhanh.
- **Hướng phát triển dự án:**
  - Gemma 4 vốn là thầy của các vòng trước (bản 31B trên box). Kết quả này cho thấy chính dòng Gemma 4 là nền tốt để làm trợ lý tiếng Việt của mình.
  - Có thể ghép thêm RAG tài liệu và vòng tự chạy code. Hoặc fine-tune LoRA từ dữ liệu tiếng Việt đã có (vòng thầy, office, tool) lên bản 26B-A4B, thay vì tiếp tục nuôi model 2,5B từ đầu.

## Phụ lục — chạy lại

```bash
ollama pull gemma4:26b-a4b
MODEL=gemma4:26b-a4b OUT=out/real_eval_gemma4_26b MAXNEW=6144 CTX=16384 python3 scripts/real_use_eval_ollama.py data/eval_clean/real_code_22.json
```
