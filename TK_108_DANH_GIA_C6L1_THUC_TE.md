# TK_108 — Đánh giá chất lượng thực tế của c6l1 (tập trung code)

*Ngày 06/11/2026 · checkpoint `box_backup/c6sft/c6l1_vlm_bf16.pt` (bản chính sau vòng thầy c6l1) · chạy trên M4, `--half`, ngữ cảnh 8192, UI cổng 8805*

## Tóm tắt

**c6l1 chưa dùng được làm trợ lý code thực tế.** Bộ thử gồm 22 ca dùng thật, viết bằng tiếng Việt như người dùng hay hỏi.

- **Điểm trung bình 1,3/10.**
- **0/22 ca dùng được ngay.** Chỉ 5/22 ca đúng một phần (4–6 điểm).
- **13/22 ca bị 0–1 điểm:** sai yêu cầu, code không chạy được, hoặc giải thích sai.

Điểm benchmark của c6l1 (HumanEval 26,8 · MBPP 39,3) **không phản ánh trải nghiệm thật**. Lý do nằm ở mục 4.

Hai điểm sáng:
- **Tốc độ** 25–30 token/giây trên M4. Không lỗi máy chủ, không trả về rỗng.
- **Tiếng Việt trôi chảy,** trình bày markdown và khối code gọn.

Lỗi lớn nhất gây ra bởi **dữ liệu, không phải do cách sinh chữ**. Mình đã chạy lại 6 ca ở nhiệt độ 0,2 (thay vì 0,7) và kết quả không khá hơn.

---

## 1. Cách đo

- **22 ca thực tế** gồm: hàm Python, pandas, sửa lỗi, đọc traceback, SQL, JS (debounce), HTML/CSS, FastAPI, regex, viết pytest, LRU cache, bash, công thức Excel, chuyển Python sang JS, review bảo mật, hội thoại 3 lượt, trò chơi CLI, git, asyncio, giải thích khái niệm, refactor, React.
- **Cấu hình sinh chữ** giống UI mặc định: nhiệt độ 0,7, top_p 0,9, phạt lặp 1,1, tối đa 2048 token (ca trò chơi 3072).
- **Chấm điểm:**
  - Mình đọc từng câu trả lời và chấm 0–10 theo tiêu chí đúng yêu cầu, code chạy được, và giải thích không sai.
  - Các đoạn code thuần tính toán thì chạy thử thật trên Mac (`python3 -I`/`node`, trong thư mục tạm, giới hạn 10 giây).
  - Code đụng mạng, file hệ thống hoặc shell thì chỉ đọc, không chạy.
- **Dữ liệu gốc:**
  - Câu trả lời đầy đủ ở `out/real_eval_c6l1/*.json`, bản nhiệt độ 0,2 ở `out/real_eval_c6l1/t02/`.
  - Bộ đề và script chạy được nêu ở phụ lục.

## 2. Bảng điểm

| # | Ca | Loại | Điểm | Nhận xét chính |
|---|---|---|---|---|
| 01 | SĐT Việt Nam | hàm Python | 3 | Chạy được với số đầu 0, nhưng **không xử lý `+84`**. Kết quả ghi trong ví dụ sai so với chính code. Đã chạy: `0912…` → True, `+84912…` → False. |
| 02 | pandas doanh thu | dữ liệu | 1 | Dùng cột `thang` không tồn tại. Cộng `so_luong` thay vì doanh thu. `.loc[...max()]` sai. Excel mở `mode='a'` sẽ lỗi khi file chưa có. |
| 03 | lỗi tham số mặc định `[]` | debug | **0** | Trả lại **y nguyên code lỗi** và bịa ra "toán tử or ở dòng 12". |
| 04 | KeyError 'email' | debug | 1 | Giải thích vô nghĩa ("hàm traceback chỉ chạy dòng 2–100"). Code trả về không đổi. |
| 05 | SQL top 5 khách Hà Nội 2024 | SQL | **6** | JOIN, lọc, GROUP BY, ORDER BY, LIMIT đúng. **Thiếu số đơn** (COUNT). Tự thêm `HAVING > 5 triệu` không ai yêu cầu. GROUP BY theo tên thay vì id. |
| 06 | debounce ô tìm kiếm | JS | **0** | **Không có hàm debounce.** Không có ô input. Hai đoạn code chép nhau. Chạm trần 2048 token. |
| 07 | trang đăng nhập | HTML/CSS | 4 | Form chạy được, có viewport. Tiêu đề lại là "Đăng nhập **thành công!**". **Thiếu link Quên mật khẩu.** Nhãn "Password" tiếng Anh. Placeholder lỗi ký tự "� dần!". |
| 08 | FastAPI todo | web | **0** | **Không dùng FastAPI.** Viết client `requests` gọi API không tồn tại. Đặt `@dataclass` lên method. |
| 09 | regex email bỏ trùng | xử lý chuỗi | 4 | Regex lấy đúng email, nhưng **không bỏ trùng** (đã chạy: ra 3 email, trùng 1). Giải thích nói "sắp xếp", "re.sub", "O(n²)" đều sai. |
| 10 | viết pytest | test | **0** | **Không viết test.** "Sửa" một hàm vốn đúng thành hàm sai. |
| 11 | LRUCache O(1) | thuật toán | 2 | Ý tưởng OrderedDict đúng, nhưng `put` xoá rồi **không ghi lại** khi khoá đã có (đã chạy: `get(1)` → -1). `keys().next()` lỗi. Cuối bài lặp `*****` tới hết 2048 token. |
| 12 | bash sao lưu, giữ 7 bản | bash | **0** | Chạy `git status` không liên quan. `tar` có `"$webroot/*"` trong nháy. Không xoay vòng 7 bản. |
| 13 | Excel đơn giá / thành tiền / SUMIF | Excel | 1 | Chỉ đưa một công thức INDEX/MATCH sai, tìm chuỗi "SP0". Thiếu cả 3 ý. Bịa ra kết quả "50000000". |
| 14 | Python sang JS | chuyển ngôn ngữ | 4 | Chạy được, nhưng **bỏ bước strip dấu câu và lowercase**. Trả về từ thay vì cặp (từ, số lần). Đã chạy: `['Mèo','ăn','cá.']`, sai so với bản Python. |
| 15 | review đăng nhập SQLite | review bảo mật | **0** | **Bỏ sót SQL injection** và mật khẩu lưu thô. Bịa ra lỗi "bare quote". Giữ nguyên f-string nguy hiểm. |
| 16 | hội thoại 3 lượt (JSON → type hint → test) | nhiều lượt | 1 | Lượt 1 dùng `sum(data.values())`, sai với danh sách. Lượt 2 **lỗi cú pháp** (`return` ngoài hàm). Lượt 3 không có test, dùng `tmp_path` chưa định nghĩa. |
| 17 | trò chơi đoán số CLI | ứng dụng nhỏ | **0** | Không có `input()`, máy tự đoán. Không lưu highscore. Bảo cài `pip install random`. |
| 18 | git: lỡ commit `.env` | công cụ | 2 | Bước 1 `reset --soft HEAD~1` đúng. Sau đó lại bảo `reset --hard`, **làm mất dữ liệu**. Không nhắc `.gitignore` hay `git rm --cached`. |
| 19 | asyncio + aiohttp, giới hạn 3 | async | **0** | Không semaphore, không gather, API aiohttp bịa ra. Có 11 URL dù đề nói 10. Bảo cài `pip install asyncio`. |
| 20 | set và list khi tìm `in` | giải thích | **0** | Giải thích **ngược và sai** ("set dùng đệ quy"). Không đo thời gian. |
| 21 | refactor giảm giá | refactor | **0** | Trả lại **y nguyên code** và bịa ra lỗi đã sửa. |
| 22 | React todo | React | **0** | Viết **HTML thuần, không có React**. `getElementById('todo-list')` trỏ tới phần tử không tồn tại. |

**Theo nhóm:**

| Nhóm | Ca | Điểm TB |
|---|---|---|
| Viết mới từ đề rõ (01, 02, 05, 09, 11, 13) | 6 | 2,8 |
| Front-end, web (06, 07, 08, 22) | 4 | 1,0 |
| Có sẵn code, cần làm việc khác ngoài sửa (10, 14, 15, 21) | 4 | 1,0 |
| Sửa lỗi, debug (03, 04) | 2 | 0,5 |
| Ứng dụng nhỏ, script (12, 17, 19) | 3 | 0 |
| Kiến thức, công cụ (18, 20) | 2 | 1,0 |
| Nhiều lượt (16) | 1 | 1,0 |

**Thử nhiệt độ 0,2 (6 ca: 01, 03, 08, 10, 11, 21):** không ca nào khá lên.
- Ca 01 bỏ hẳn code, chỉ liệt kê `assert` vô nghĩa.
- Các ca 03, 10, 21 vẫn trả y nguyên code và bịa ra "đã sửa lỗi".
- Ca 11 gần đúng hơn, nhưng lưu `[value, list]` làm hỏng giá trị.

Vì vậy lỗi không nằm ở cách sinh chữ.

---

## 3. Năm kiểu lỗi chính

### 3.1 "Thấy code là sửa lỗi". Đây là lỗi nặng nhất.
Khi đề có sẵn một khối code, model **gần như luôn trả lời theo khuôn sửa lỗi**. Câu trả lời mở đầu bằng "Lỗi nằm ở dòng…", "Lỗi chính…" hoặc "Lỗi trước đó…". Sau đó model bịa ra một lỗi, rồi trả lại code y nguyên hoặc tệ hơn, **dù đề yêu cầu viết test, refactor, review hay chuyển ngôn ngữ.**
- Gặp ở các ca 03, 04, 10, 15, 21. Ở nhiệt độ 0,2 lỗi này còn rõ hơn.
- Model còn "sửa" cả code vốn đúng (ca 10), nên thực tế còn tệ hơn là trả lời sai.

### 3.2 Bỏ qua công nghệ hoặc ràng buộc được yêu cầu
- FastAPI thành client `requests` (08). React thành HTML thuần (22). Debounce thành không có debounce (06).
- "Bỏ trùng" (09), "+84" (01), "số đơn" (05), "giữ 7 bản" (12), "tối đa 3 request" (19) đều bị bỏ.
- Model bắt được chủ đề chung nhưng **không theo đặc tả**.

### 3.3 Giải thích bịa đặt mà tự tin
Ví dụ: `pip install random` / `pip install asyncio`, "collections phải cài thư viện ngoài", "set dùng đệ quy", "bare quote", "O(n²)". Người mới học có thể tin những câu này, và đây là điểm **nguy hiểm**.

### 3.4 Mở đầu theo khuôn vòng thầy, lệch ngữ cảnh
Nhiều câu mở đầu như "Tôi đã thêm CSS…", "Tôi đã cập nhật…", "Lời giải triển khai…", "Lỗi trước đó nằm ở…", dù là lượt đầu tiên và chưa có gì để "thêm".
- Đây là dấu vết của mẫu **bản sửa lượt sau** (thầy sửa câu trả lời trước đó) bị học như câu trả lời lượt đầu.
- Trong `loop1.jsonl` có 377 mẫu "<done>Đã triển khai", 286 mẫu "<done>Đã sửa lỗi", 193 mẫu "Lỗi nằm ở".

### 3.5 Thoái hoá khi trả lời dài
- Lặp vô hạn ký tự (`*****…` ở ca 11), chép lại khối code (ca 06), chạm trần token.
- Khi bài dài hơn khoảng 600 token, chất lượng tụt rõ.

---

## 4. Vì sao benchmark cao mà dùng thật lại kém

| | HumanEval / MBPP | Dùng thực tế |
|---|---|---|
| Ngôn ngữ đề | tiếng Anh, ngắn | tiếng Việt, mô tả tự nhiên |
| Hình thức | có sẵn chữ ký hàm, docstring, chỉ cần viết thân hàm | tự thiết kế từ đầu: chọn thư viện, cấu trúc, nhiều file hoặc nhiều phần |
| Phạm vi | 1 hàm thuần, 5–20 dòng | API, front-end, script, nhiều ràng buộc cùng lúc |
| Loại việc | viết hàm | viết, sửa, test, review, refactor, giải thích, nhiều lượt |

Kỹ năng điền thân hàm theo docstring (HE 26,8) không chuyển được sang làm theo đặc tả tiếng Việt nhiều ràng buộc. Bộ C7 sát thực tế hơn, và kết quả ở đó cũng chỉ **4,8%**, khớp với bảng điểm trên.

## 5. Nguyên nhân gốc, có số liệu từ dữ liệu học

Mình rà `sft_b.jsonl` (215 nghìn mẫu, nền của c6l1). Trong đó có **44 275 mẫu có khối code trong đề**:

| Loại việc trên code có sẵn | Số mẫu | Tỉ lệ |
|---|---|---|
| Sửa lỗi / bug | 12 974 | **29,3%** |
| Chuyển ngôn ngữ | 1 521 | 3,4% |
| Refactor | 291 | 0,7% |
| Giải thích | 102 | 0,2% |
| Review / bảo mật | 87 | 0,2% |
| **Viết test** | **66** | **0,1%** |
| Khác (chủ yếu quỹ đạo tool/agent) | 29 234 | 66,0% |

Câu mở đầu phổ biến nhất trong SFT-B là "<done>Đã triển khai" (6 228), "<done>Đã sửa lỗi" (6 197), "Lỗi nằm ở" (4 296) và "Lời giải triển khai" (2 194).

Kết luận: số mẫu sửa lỗi nhiều gấp **khoảng 200 lần** số mẫu viết test hoặc review. Model học được quy tắc "có code thì nói Lỗi nằm ở…". Đây là lệch phân bố dữ liệu, không phải giới hạn kích thước model. Sửa lỗi này rẻ hơn nhiều so với tăng tham số.

Ngoài ra còn một nguyên nhân về kích thước:
- Model 2,5B tự pretrain, khoảng 20 tỷ token pretrain cộng 10 tỷ token học tiếp nặng code.
- Các model code dùng tốt cỡ 1,5–3B thường cần vài nghìn tỷ token.
- Vì vậy các kiểu lỗi 3.2, 3.3 và 3.5 sẽ khó hết hẳn chỉ bằng SFT.

---

## 6. Đề xuất, theo thứ tự đáng làm

1. **Cân lại dữ liệu code SFT** (rẻ, tác động lớn nhất):
   - Bổ sung 20–30 nghìn mẫu tiếng Việt cho các việc ngoài sửa lỗi: viết test, review bảo mật, refactor, giải thích, chuyển ngôn ngữ, viết từ đặc tả nhiều ràng buộc (FastAPI, React, bash, asyncio, pandas).
   - **Chạy kiểm tra từng mẫu** trong sandbox như C2/C3. Đây là phần thầy làm tốt.
   - Thêm **mẫu đối chứng**: đưa code đúng kèm yêu cầu viết test, đáp án là test chứ không phải "Lỗi nằm ở".
   - Hạ mẫu sửa lỗi xuống khoảng 10% số mẫu có code.
2. **Lọc khuôn mở đầu lệch ngữ cảnh** trong `build_c6_loop.py` và `build_sft_c6.py`:
   - Bỏ hoặc viết lại các bản sửa của thầy mở đầu bằng "Tôi đã thêm/cập nhật…", "Lỗi trước đó…" khi đó là lượt đầu.
   - Bỏ tiền tố `<done>` khỏi câu trả lời chat thường (chỉ giữ trong quỹ đạo tool).
3. **Đưa 22 ca này vào bộ đo hồi quy** (`out/real_eval_c6l1` làm mốc). Mở rộng lên khoảng 100 ca, có ca chạy thử tự động, để vòng sau đo được "dùng thật" thay vì chỉ HE/MBPP.
4. **Thầy chấm theo đặc tả**: vòng thầy sau cho thầy chấm nghiêm theo checklist ràng buộc của đề ("có dùng FastAPI không", "có bỏ trùng không"…), không chỉ chấm cảm quan.
5. **Chống thoái hoá câu dài**: thêm mẫu code dài 600–2000 token hoàn chỉnh. Khi sinh code, cân nhắc phạt lặp theo n-gram hoặc dừng khi lặp ký tự.
6. **Về lâu dài**: muốn có trợ lý code dùng được thật thì cần (a) thêm nhiều token pretrain code, hoặc (b) cho model đóng vai bộ điều phối và dùng tool. Hướng (b) gồm ghép RAG tài liệu, chạy code tự kiểm, rồi sửa theo kết quả chạy. Kết quả tool hiện tại (52,5%, sửa lỗi được 70%) cho thấy model làm tốt hơn khi được chạy code và nhìn kết quả.

## 7. Điểm mạnh nên giữ

- Tốc độ 25–30 token/giây trên M4 (`--half`). Ngữ cảnh 8K ổn định, không lỗi máy chủ.
- Tiếng Việt tự nhiên, bố cục câu trả lời rõ: khối code, các bước, lưu ý.
- SQL truy vấn cơ bản khá (ca 05). Các đoạn code ngắn thường chạy được (01, 09, 14), chỉ thiếu ràng buộc.
- Có thói quen hỏi lại hoặc đề nghị bước tiếp theo ở cuối câu trả lời.

---

## Phụ lục — chạy lại

- Bộ đề: `data/eval_clean/real_code_22.json`
- Script: `scripts/real_use_eval.py`. Script đặt nhiệt độ 0,7; muốn đo bản khác thì đổi `URL` và `OUT`.

```bash
# bật UI c6l1 (cổng 8805) rồi chạy cả 22 ca → out/real_eval_c6l1/<id>.json
python3 scripts/real_use_eval.py data/eval_clean/real_code_22.json
# chạy riêng vài ca, cho một máy chủ khác
URL=http://localhost:8804 OUT=out/real_eval_b python3 scripts/real_use_eval.py data/eval_clean/real_code_22.json 03_bugfix 10_pytest
```

Mỗi file kết quả lưu từng lượt hội thoại, gồm câu hỏi, câu trả lời, số token và thời gian.
