# Báo cáo C1 (mốc code) + sửa Excel — 15/10/2026

Đọc lúc 08:00 UTC (15:00 giờ VN). Box Vast 1×H100: GPU 64 °C, 100 %, 74,5 GB, 688 W (đang chạy thầy C2). Ổ `/`: **84 GB trống** / 250 GB (dùng 67 %).

## 1. C1 — đo code trên `ckpt/aivie_p3_r1_bf16.pt` — **XONG** (05:19 UTC)

Lúc 02:11 có "LỖI C1 mbpp/humaneval" (thiếu thư viện `datasets`); đã sửa và chạy lại từ 02:29, không còn lỗi.

| Bộ đo | n | pass@1 | Sau tự sửa | Mốc cũ (Mac, 40 bài HE) |
|---|---|---|---|---|
| HumanEval | 164 | **6,1 %** (10) | 6,1 % | v5 1.3B 17,5 % · P3 text 10 % |
| MBPP | 257 | **5,1 %** (13) | 5,1 % | — |
| Tool-use | 40 (20 fix, 20 implement) | **0 %** | — | v5 0/20 |

- Tool-use: **82 %** lượt không gọi công cụ; tổng chỉ 5 lần gọi công cụ hợp lệ trên 361 bước.
- Mốc thật của P3 trên 164 bài là 6,1 %, thấp hơn mốc 40 bài (10 %). 40 bài đầu dễ hơn: 9/10 bài qua đều nằm trong HumanEval/0–48.
- **Vòng tự sửa không mang lại gì** (+0 bài). Trong 85 bài HumanEval chạy đủ 4 vòng, 70 bài nộp lại y hệt code cũ. MBPP: 185/238.

### Kiểu lỗi (vòng cuối)

| Kiểu | HumanEval (154 trượt) | MBPP (244 trượt) |
|---|---|---|
| Qua test mẫu, trượt test ẩn | 71 | 6 |
| AssertionError (sai kết quả) | 40 | 61 |
| **NameError (đặt sai tên hàm)** | 7 | **77** |
| TypeError (sai chữ ký / kiểu) | 10 | 48 |
| SyntaxError / IndentationError | 17 | 32 |
| Khác (Index, Recursion, Module…) | 9 | 20 |

Nhận xét:
1. **MBPP: lỗi hàng đầu là đặt sai tên hàm** (77/244). Model không bám tên hàm có trong test của đề. Ví dụ: viết `assert_lowercase_underscore` trong khi test gọi `text_lowercase_underscore`; viết `assert_duplicate` trong khi cần `test_duplicate`. Chữ "assert/test" trong đề bị chép thành tên hàm. Chỉ cần sửa định dạng prompt/dữ liệu là gỡ được phần lớn nhóm lỗi này.
2. **HumanEval: 71 bài "qua test mẫu"**, chủ yếu vì đề không có ví dụ chạy được (test mẫu rỗng), nên vòng sửa dừng ngay sau vòng 0. Thực chất code vẫn sai thuật toán.
3. **Code chạy được nhưng sai logic**: kiểu "trông giống code", có `print` test kèm comment kết quả bịa.
4. **Vòng sửa ra lời văn thay cho code**: câu tiếng Việt bị đưa vào chỗ code nên gây SyntaxError.
5. **Tool-use**: model trò chuyện ("Bạn cần sửa regex thành…") thay vì phát `<tool>{...}</tool>`. Nó chưa từng học giao thức này, đúng như dự đoán. C3 sẽ xử lý.

### Ví dụ lời giải sai

**MBPP/16**: sai tên hàm và regex vô nghĩa
```python
def assert_lowercase_underscore(text):
    pattern = r'\b\w+\b'          # khớp mọi chữ, không kiểm "a_b"
    if re.match(pattern, text): return True
    ...
# test gọi text_lowercase_underscore → NameError
```

**HumanEval/0**: sai thuật toán, comment bịa kết quả, 4 vòng sửa nộp lại y hệt
```python
    return numbers[0] - numbers[1] <= threshold
print(has_close_elements([1.0, 2.0, 3.0], 0.5))  # True   (đề ghi False)
```

**HumanEval/32**: vòng sửa trả lời bằng văn xuôi, khối code thành câu tiếng Việt
```
Lỗi cú pháp ở dòng 2 của code bạn cung cấp. Bạn cần sửa thành `return math.pow(x, i)` ...
→ SyntaxError: invalid syntax
```

## 2. Sửa Excel (`box_run_office_fix.sh`) — **XONG** (05:47 UTC)

- **Dữ liệu:** `office_v2` có 3 911 dòng, sửa 201 lượt; `v5/sft` có 4 980 dòng, sửa 388 lượt. Công thức được đưa vào khối ```` ```excel ````.
- **Học expert văn phòng (vp):** 400 bước, 151 M tham số, 980 s. Val trước → sau:
  - offfix 0,619 → 0,608
  - fixfix 1,010 → 0,991
  - mt 0,967 → 0,915
  - calib giữ nguyên 0,873
- Cổng vẫn đóng nhỏ trên dữ liệu văn phòng (vp ≈ 0,03–0,08).
- Đã đẩy HF: `nn-aivie-2p6b-p3-snap/aivie_2p6b_p3_r1xl_bf16.pt`.

| Checkpoint | Excel (chấm chặt) | Công thức trong khối | easy20 |
|---|---|---|---|
| p3 (trước vòng thầy, Mac) | 17/27 | — | 17/20 |
| p3r1 | 9/27 | 19/27 | 16/20 |
| **p3r1xl** | **13/27** | **27/27** | **16/20** |

**Kết luận:**
- p3r1xl lấy lại được **một nửa** phần Excel bị mất (9 → 13; mốc gốc là 17).
- Định dạng đã sửa xong hoàn toàn: 27/27 bài có công thức trong khối.
- **easy20 không tụt** (16/20).
- Phần còn thiếu là **nội dung công thức**, không còn là định dạng.

**Đề xuất:** lấy **p3r1xl làm checkpoint chính**. Lý do: tốt hơn hoặc bằng p3r1 ở mọi chỉ số, và C4 đã cắm sẵn `p3r1xl_vp.pt` làm expert office.

Muốn lấy lại đủ 17/27 thì cần dữ liệu công thức đúng (thầy sinh rồi kiểm bằng tính thật trên bảng tính), không chỉ chuẩn hoá định dạng. Việc này để sau, không chặn C2.

## 3. C2 — đang chạy (bắt đầu 06:56 UTC)

- **Thử nhỏ:** vi 40/40 · debug 28/40 · predict 38/40 · sql 35/40.
- **Phần vi:** đã xong 14 000/30 000, giữ 100 %, khoảng 3,7 bài/s. Phần vi xong lúc khoảng **09:15 UTC (16:15 VN)**.
- **Còn lại:** debug 20K, predict 10K, sql 8K. Các phần này phải chạy code để kiểm, nên chậm hơn vi. **Ước tính C2 xong khoảng 13–16 h UTC (20–23 h VN)**. Đây là ước đoán vì chưa có tốc độ của debug/sql.
- **Lưu ý:** lúc 07:21 có tiến trình `trainGEN2 --device cpu --tag smokec4` chạy thử C4 trên CPU, dùng khoảng 22 lõi, giới hạn `timeout 2400`, nên tự dừng khoảng 08:01. Nó tranh CPU với sandbox kiểm test của C2. Nếu còn chạy thử kiểu này, nên làm sau khi C2 xong.
- **Không có "LỖI"** nào sau 02:11.

## 4. Đề xuất cho C4 (học expert code mới trên dữ liệu C2)

Dữ liệu sẵn có:
- `data/c2/oci_verified.jsonl`: 99 658 bài đã chạy test, qua 97,1 %, tiếng Anh.
- Khi C2 xong sẽ có thêm khoảng 60K bài vi / debug / predict / sql.

Đề xuất:
1. **Sửa định dạng trước khi học:** mỗi mẫu đưa **tên hàm và chữ ký bắt buộc** vào đề (giống cách MBPP đưa test) và chỉ trả **một khối code**, không kèm `print` test hay comment kết quả. Đây là cách trực tiếp nhất để gỡ nhóm NameError (77) và TypeError (48) của MBPP.
2. **Thêm mẫu "sửa theo traceback" vào cùng định dạng vòng tự sửa của bộ đo**: đầu vào là code cũ + traceback, đầu ra là **khối code mới, khác code cũ**. Phần debug 20K của C2 phục vụ đúng việc này. Hiện vòng sửa đóng góp 0 %.
3. **`trainGEN2` 2 giai đoạn, đúng quyết định 14/10**:
   - Thân, vp (`p3r1xl_vp.pt`), hoc và phần đọc ảnh đều đóng băng.
   - Giai đoạn A: chỉ bật expert code MỚI trên `--domain code:c2oci+c2vi+c2debug+c2predict+c2sql`.
   - Giai đoạn B: học cổng trên `--mix` có ôn lại khoảng 35 % chat/mt/office, để chặn quên tiếng Việt, theo kinh nghiệm `qlora-data-da-dang`.
   - Bản thử C4 đang dùng `--mix c2oci:0.5,mt:0.5`; nên thêm office và calib vào mix ôn.
4. **Đo lại bằng C1 đầy đủ (164/257/40) cộng Excel và easy20** sau C4, để chắc expert code không làm hỏng p3r1xl.
5. **Tool-use** vẫn sẽ là 0 cho đến khi có quỹ đạo C3. Đầu C3 thử Gemma và Qwen3-Coder trên khoảng 50 bài như đã định.
6. **Ổ đĩa:** còn 84 GB, đủ cho C2 (dữ liệu khoảng vài trăm MB) và vài checkpoint 7 GB. Nên dọn trước C4 nếu giữ nhiều bản expert.

*Không commit, không push, không chạy việc gì trên box.*
