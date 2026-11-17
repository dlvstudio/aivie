# TK_108 — Đánh giá thực tế Qwen2.5-Coder trên M4 (tập trung code)

*Ngày 06/11/2026 · Ollama 0.40 · `qwen2.5-coder:7b` và `qwen2.5-coder:3b` (bản nén 4-bit mặc định của Ollama) · cùng bộ 22 ca, cùng cách chấm như [đánh giá c6l1](TK_108_DANH_GIA_C6L1_THUC_TE.md)*

## Tóm tắt

| | Qwen2.5-Coder 7B | Qwen2.5-Coder 3B |
|---|---|---|
| **Điểm trung bình** | **7,3 / 10** | **5,7 / 10** |
| Dùng được ngay (≥ 7 điểm) | **14 / 22** | 12 / 22 |
| Đúng một phần (4–6) | 8 / 22 | 3 / 22 |
| Hỏng (0–3) | **0 / 22** | 7 / 22 |
| Tốc độ trên M4 | ~46 token/giây | ~87 token/giây |
| Dung lượng | 4,7 GB | 1,9 GB |

- **Bản 7B dùng được làm trợ lý code hằng ngày** cho các việc phổ biến như pandas, SQL, FastAPI, React, debounce, sửa lỗi, review bảo mật, refactor và công thức Excel.
- **Không ca nào hỏng hoàn toàn.** Lỗi còn lại chủ yếu nằm ở **chi tiết ràng buộc**: regex sai thứ tự ưu tiên, bỏ giới hạn 3 request, xoay vòng 7 bản sao lưu sai, test có một assert sai.
- **Bản 3B** làm tốt các ca cơ bản nhưng hay sai logic hơn, vì vậy ít đáng tin hơn.
- Cả hai bản trả lời tiếng Việt rõ ràng và theo đúng loại việc được hỏi. Không bị lỗi "thấy code là sửa lỗi" như c6l1.

---

## 1. Cách đo

- **Bộ đề:** `data/eval_clean/real_code_22.json`, gồm 22 ca tiếng Việt, có một hội thoại 3 lượt.
- **Cài đặt sinh chữ** giống hệt lúc đo c6l1: nhiệt độ 0,7, top_p 0,9, phạt lặp 1,1, tối đa 2048 token, ngữ cảnh 8192.
- **Script:** `scripts/real_use_eval_ollama.py`. Câu trả lời gốc nằm ở `out/real_eval_qwen7b/` và `out/real_eval_qwen3b/`.
- **Chấm điểm:**
  - Mình đọc từng câu trả lời và chấm 0–10 theo tiêu chí đúng yêu cầu, code chạy được, và giải thích không sai.
  - Code thuần tính toán thì chạy thử thật trên Mac, trong thư mục tạm, có giới hạn thời gian (Python `-I`, pytest, pandas qua `.venv`, node).
  - Code đụng mạng, hệ thống hay shell thì chỉ đọc, không chạy.

## 2. Bảng điểm Qwen2.5-Coder 7B

| # | Ca | Điểm | Nhận xét |
|---|---|---|---|
| 01 | SĐT Việt Nam | 4 | Cấu trúc tốt, nhưng regex `^03\|05\|…\d{8}$` **sai thứ tự ưu tiên**. Đã chạy: `03123456789` (11 số) ra **hợp lệ**, `+84912345678` ra **không hợp lệ**. |
| 02 | pandas doanh thu | **10** | Đã chạy: doanh thu theo tháng và top 3 đúng, file Excel có đúng 2 sheet. |
| 03 | lỗi `[]` mặc định | 9 | Chẩn đoán đúng và sửa chuẩn (`None` rồi tạo list mới). Câu cuối giải thích hơi rối. |
| 04 | KeyError 'email' | 9 | Giải thích đúng, sửa bằng `if 'email' in user`. |
| 05 | SQL top 5 Hà Nội | 9 | Đủ tổng tiền, số đơn, GROUP BY theo id. Hàm `YEAR()` chỉ có ở MySQL. |
| 06 | debounce tìm kiếm | 9 | Debounce chuẩn, có input, fetch và render `<ul>`. Thiếu `encodeURIComponent`. |
| 07 | trang đăng nhập | 9 | Gọn, chạy tốt trên điện thoại, chữ tiếng Việt, có link Quên mật khẩu. |
| 08 | FastAPI todo | 8 | Đúng FastAPI và Pydantic, có 404 khi sửa. Xoá thì không trả 404. Ví dụ ghi gửi `completed` trong body nhưng code nhận qua query. |
| 09 | regex email bỏ trùng | 9 | Đã chạy: `['an.nguyen@gmail.com', 'sales@cty.vn']`, đúng. |
| 10 | viết pytest | 5 | Test có ý tốt, nhưng một assert **sai**: `tong < 0` mong có ValueError. Đã chạy pytest: **FAILED**. |
| 11 | LRUCache O(1) | 9 | Danh sách liên kết đôi cộng dict, đúng chuẩn. Đã chạy: kết quả đúng. |
| 12 | bash sao lưu, giữ 7 bản | 4 | Sao lưu đúng, nhưng **xoay vòng sai**: đếm trên đúng file vừa tạo nên không bao giờ xoá. Biến không đặt trong nháy. |
| 13 | Excel | 9 | Đủ cả 3 ý: VLOOKUP, `=B2*C2`, `SUMIFS(D:D,A:A,"SP0*")`. Không nhắc dấu `;` khi máy dùng định dạng Việt Nam. |
| 14 | Python sang JS | 6 | Logic đúng, nhưng **cần thư viện lodash** dù không ai yêu cầu. Đã chạy: lỗi `Cannot find module 'lodash'`. |
| 15 | review SQLite | 8 | Bắt được SQL injection và sửa bằng tham số `?`. Không nhắc lỗi mật khẩu lưu thô. |
| 16 | hội thoại 3 lượt | 6 | Lượt 1 và 2 tốt. Lượt 3 có một test thiếu `import json`, lại mong lỗi mà hàm đã tự bắt. Đã chạy: **1 qua, 1 hỏng**. |
| 17 | trò chơi đoán số | 5 | Đủ chức năng: input, bắt lỗi, highscore, chơi lại. Nhưng **chỉ đếm lượt khi đoán đúng**, nên lúc nào cũng ra "1 lần". |
| 18 | git lỡ commit `.env` | 4 | `git rm --cached` rồi commit mới làm **`.env` vẫn còn trong commit cũ** và sẽ lộ khi push. Đúng ra phải `reset --soft` hoặc `commit --amend`. Phần `.gitignore` đúng. |
| 19 | asyncio + aiohttp | 5 | Có gather, timeout, in URL lỗi, nhưng **không giới hạn 3 request** và tự nhận là chưa làm. |
| 20 | set và list | 7 | Có ví dụ đo thời gian với 1 triệu phần tử. Giải thích hơi nông, không nói tới bảng băm. |
| 21 | refactor giảm giá | 8 | Dùng dict cùng `.get`, gọn và đúng. Đã chạy: kết quả khớp bản gốc. Ngưỡng 1 triệu vẫn lặp lại hai lần. |
| 22 | React todo | 8 | Hooks đủ: thêm, gạch ngang, xoá, đếm việc. Nhãn ghi "Đã hoàn thành" nhưng lại hiện số việc **chưa** xong. Đổi trực tiếp object trong state. |

## 3. Bảng điểm Qwen2.5-Coder 3B (tóm tắt)

| # | Điểm | Nhận xét |
|---|---|---|
| 01 | 2 | `lstrip('+84')` xoá theo từng ký tự nên logic sai. Kết quả ghi trong ví dụ cũng sai. |
| 02 | 7 | Đúng, nhưng sheet top 3 ghi `index=False` nên **mất tên sản phẩm**. |
| 03 | 1 | Không sửa đúng lỗi, vẫn giữ `[]` mặc định. |
| 04 | 8 | Dùng `.get()`, đúng. |
| 05 | 8 | Đủ ý, nhưng GROUP BY theo tên thay vì id. |
| 06 | 9 | Đúng, có `encodeURIComponent`. |
| 07 | 8 | Đúng và đủ; input thiếu `box-sizing` nên có thể tràn ra ngoài khung. |
| 08 | 7 | FastAPI đúng; xoá không trả 404. |
| 09 | 9 | Đúng (`dict.fromkeys`). |
| 10 | 3 | Hai assert tính sai, ví dụ `chia_deu(15,3)` ghi ra `[4,4,3]`. |
| 11 | 8 | Code đúng; chú thích trong ví dụ sai. |
| 12 | 6 | Xoá file cũ hơn 7 **ngày** thay vì giữ 7 **bản**. |
| 13 | 4 | Sai cú pháp tên sheet; SUMIF cộng số lượng thay vì thành tiền. |
| 14 | 2 | Cần lodash; `\w` trong JS **cắt nát chữ có dấu** (đã chạy: "mèo" thành `'m','o'`). |
| 15 | 8 | Bắt được cả SQL injection lẫn lỗi mật khẩu lưu thô. |
| 16 | 7 | Ba lượt mạch lạc, có test dùng `tmp_path`. |
| 17 | 2 | Mở file `'a'` rồi `read()` nên **lỗi ngay khi chạy**. Lẫn chữ Hán "猜". |
| 18 | 7 | `git reset HEAD^` giữ được thay đổi, có `rm --cached` và `.gitignore`. |
| 19 | 2 | `asyncio.gather(..., limit=3)` là tham số không tồn tại, nên lỗi khi chạy. |
| 20 | 6 | Giải thích đúng về bảng băm, nhưng lẫn câu rác ("Khiển thị…"). |
| 21 | 8 | Giống bản 7B. |
| 22 | 3 | Bấm vào việc lại **xoá** việc; logic gạch ngang và phần đếm sai. |

---

## 4. Nhận xét

**Điểm mạnh của bản 7B:**
- **Làm đúng loại việc được hỏi.** Hỏi viết test thì viết test, hỏi review thì review, hỏi refactor thì refactor.
- **Dùng đúng công nghệ được yêu cầu:** FastAPI, React hooks, debounce, pandas, SUMIFS.
- **Code chạy được ngay trong đa số ca.** Trong 9 ca mình chạy thử, có 6 ca đúng hoàn toàn.
- **Giải thích đúng và không bịa.** Không có kiểu "pip install random" như c6l1.
- Tiếng Việt tự nhiên, trình bày theo bước và có hướng dẫn chạy.

**Điểm yếu của bản 7B:**
- **Hay sót một ràng buộc chi tiết:** "+84", "tối đa 3 request", "giữ 7 bản", 404 khi xoá. Người dùng cần đọc lại code chứ không nên chép mù.
- **Lỗi logic nhỏ mà khó thấy:** biến đếm lượt sai chỗ (ca 17), regex sai thứ tự ưu tiên (ca 01), một assert mong sai (ca 10).
- **Lời khuyên git cho `.env` không an toàn:** mật khẩu vẫn nằm trong lịch sử. Đây là loại lỗi nguy hiểm nhất trong cả bộ.
- Tự kéo thêm thư viện ngoài (lodash) khi không cần.

**Bản 3B:** nhanh gấp đôi và nhẹ hơn hai lần rưỡi. Các việc phổ biến làm tốt ngang bản 7B (ca 06, 09, 15, 18, 21). Khi logic phức tạp hơn thì sai nhiều hơn rõ, và có ca lỗi ngay khi chạy (17, 19). Hợp để gợi ý nhanh, không hợp để giao việc.

## 5. Gợi ý dùng

- **Dùng hằng ngày trên M4:** chọn `qwen2.5-coder:7b`. Chạy bằng `ollama run qwen2.5-coder:7b`, hoặc nối vào editor qua API `localhost:11434`.
- **Luôn chạy thử và đọc lại** phần ràng buộc chi tiết. Model này làm được khoảng 80–90% việc, còn 10–20% cuối thường là chỗ sai.
- **Nếu muốn nâng chất lượng:**
  - (a) Thêm vòng tự chạy code và sửa theo lỗi. Các lỗi ở ca 01, 10, 14, 16, 17 đều sẽ lộ ra ngay khi chạy.
  - (b) Fine-tune bằng dữ liệu tiếng Việt đã cân lại, rút từ đánh giá c6l1, để học các ràng buộc hay gặp.
  - (c) Khi có GPU thì thử bản 14B/32B.
- **Bộ đo này** (22 ca cùng script) dùng lại được để so khi đổi model hoặc sau khi fine-tune.

## Phụ lục — chạy lại

```bash
ollama pull qwen2.5-coder:7b
MODEL=qwen2.5-coder:7b OUT=out/real_eval_qwen7b python3 scripts/real_use_eval_ollama.py data/eval_clean/real_code_22.json
```
