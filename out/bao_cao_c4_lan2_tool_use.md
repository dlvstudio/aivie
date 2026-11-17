# Báo cáo C4 lần 2 — aivie biết dùng công cụ (tool-use 0% → 32,5%)

*TK_108 · 16/10/2026 · checkpoint `aivie_p3_c4b` (thân 2,5B P3 + expert code c4b, expert văn phòng/học thuật giữ nguyên)*

## 1. Tóm tắt

| Bộ đo | C1 (p3r1, gốc) | C4 sớm (c4e) | **C4 lần 2 (c4b)** |
|---|---|---|---|
| **Tool-use (40 bài dự án nhỏ)** | 0% | 0% | **32,5%** (13/40) |
| — sửa lỗi (fix, 20 bài) | 0% | 0% | **45%** (9/20) |
| — viết hàm (implement, 20 bài) | 0% | 0% | **20%** (4/20) |
| — lượt trả lời KHÔNG gọi công cụ | — | 99,8% | **2,3%** |
| HumanEval pass@1 | 6,1% | 7,3% | **11,0%** (sau tự sửa 12,2%) |
| MBPP pass@1 | 5,1% | 9,3% | **11,7%** |
| Excel (27 bài) | — | 13/27 | **13/27** (giữ nguyên) |
| easy20 (hội thoại chung) | — | 16/20 | **16/20** (giữ nguyên) |

**Ý chính:**
- Đây là lần đầu aivie **thật sự làm việc như một tác tử**: chạy test → đọc file → sửa → chạy lại test → báo xong.
- Trước C4 lần 2, model chỉ in ra một khối code trả lời và không bao giờ gọi công cụ (99,8% lượt không gọi).
- Học thêm code **không làm hỏng** phần văn phòng và hội thoại (Excel 13/27, easy20 16/20 y như trước).
- HumanEval và MBPP gần gấp đôi mốc C1 nhưng còn thấp (~11%) — năng lực viết code nền vẫn là điểm yếu.

## 2. Đã làm gì để có kết quả này

1. **C3 — thầy làm mẫu:** Gemma-4-31B giải 6.840 bài dự án nhỏ (sửa lỗi / viết hàm) bằng đúng giao thức công cụ của aivie
   (`<tool>{json}</tool>` → `<obs>` → … → `<done>`). Chỉ giữ quỹ đạo **test chạy lại QUA + mọi lượt đúng giao thức**:
   **5.910 quỹ đạo** (86%). Bài C3 dựng từ OpenCodeInstruct, **không trùng** 40 bài đo (dựng từ MBPP).
2. **C4 lần 2 — trò học theo mẫu:** expert code c4e học tiếp 1.500 bước (trainGEN, lr 1e-4); thân và các expert khác đóng băng.
   Trộn dữ liệu: quỹ đạo C3 45% · sách giáo khoa code 10% · ôn C2 (code đã kiểm) 30% · hội thoại + văn phòng 15%.
   Chỉ học các lượt của trợ lý (gọi công cụ, sửa code); kết quả công cụ trả về không tính vào mục tiêu học.

**Khung đo tool-use:** mỗi bài là một thư mục có `src.py` + `test_src.py`; model có tối đa **10 lượt** gọi công cụ
(`read_file`, `edit_file`, `write_file`, `run_tests`, `bash`, `done`…); chấm = chạy lại test sau cùng. Code chạy trong sandbox.

## 3. Ví dụ thành công

### Ví dụ 1 — sửa lỗi (`agent_fix_566`): gọn 5 bước, đúng quy trình

**Đề:** *"Trong dự án có `src.py` và bộ test `test_src.py`. Chạy test đang HỎNG vì `src.py` có lỗi. Hãy tìm và sửa lỗi…
Yêu cầu gốc: tính tổng các chữ số của một số nguyên không âm."*

```python
# src.py (có lỗi cài sẵn: dấu − thay vì +)
def sum_digits(n):
  if n == 0:
    return 0
  else:
    return n % 10 - sum_digits(int(n / 10))
```

| Bước | aivie làm | Kết quả trả về |
|---|---|---|
| 0 | `run_tests` | ❌ `assert sum_digits(345)==12` → AssertionError |
| 1 | `read_file src.py` | nội dung file có đánh số dòng |
| 2 | `edit_file`: `n % 10 - sum_digits(...)` → `n % 10 + sum_digits(...)` | Đã sửa src.py (1 chỗ) |
| 3 | `run_tests` | ✅ TẤT CẢ TEST QUA |
| 4 | `done` | *"Đã sửa lỗi tính toán trong hàm `sum_digits` … thay đổi phép trừ thành phép cộng…"* |

Model tự **chạy test trước** để thấy lỗi, **đọc file** rồi mới sửa, **chạy lại test** để xác nhận, và chỉ báo xong khi test qua.

### Ví dụ 2 — viết hàm (`agent_implement_600`): 4 bước

**Đề:** *"`src.py` có hàm `is_Even` chưa viết… viết thân hàm cho tất cả test qua."*

| Bước | aivie làm | Kết quả trả về |
|---|---|---|
| 0 | `read_file src.py` | `def is_Even(n): … raise NotImplementedError` |
| 1 | `edit_file`: `raise NotImplementedError` → `return n % 2 == 0` | Đã sửa src.py (1 chỗ) |
| 2 | `run_tests` | ✅ TẤT CẢ TEST QUA |
| 3 | `done` | *"Đã triển khai hàm `is_Even` … dùng toán tử chia lấy dư `%`…"* |

**9/13 bài thành công đi đúng mẫu "chuẩn"** như hai ví dụ trên (5–4 bước, có `done`). 4 bài còn lại qua test nhưng
đi đường vòng (ghi đè cả file bằng `write_file` nhiều lần, không gọi `done`) — xem mục 4.3.

### Để so sánh — cùng bài `sum_digits`, model C4 sớm (c4e) trả lời thế này

````
```python
import os

def get_sum_of_digits(n):
    """Returns the sum of the digits of a non-negative integer n. ..."""
    return sum(int(digit) for digit in str(n))
```
````

Không gọi công cụ nào, đổi luôn tên hàm, không đụng tới `src.py` → test vẫn hỏng. Đây là lý do C4 sớm đạt 0%.

## 4. Vì sao còn hỏng 27/40 bài

**Mọi bài hỏng đều do hết 10 lượt** (không bài nào báo `done` sai khi test còn hỏng — model không "nói dối" là đã xong).
Hết lượt vì model bị kẹt trong vòng lặp. Ba kiểu kẹt chính:

### 4.1 `edit_file` với đoạn `old` không khớp file — 18/27 bài hỏng, 88 lần

Ví dụ `agent_fix_620` (file dùng **TAB** để thụt lề, model viết **dấu cách**):

```
bước 2: edit_file old="            dp[i] = 1 - mxm"  →  Lỗi: không tìm thấy đoạn 'old' trong src.py.
        Dòng gần giống nhất: 'dp[i] = 1 - mxm' — copy y nguyên (đúng khoảng trắng thụt lề).
bước 3: edit_file old="        dp[i] = 1 - mxm"      →  (cùng lỗi)
bước 4: edit_file old="        dp[i] = 1 - mxm"      →  (cùng lỗi)
bước 5: read_file src.py                              →  (đọc lại — vẫn TAB)
bước 6: edit_file old="        dp[i] = 1 - mxm"      →  (cùng lỗi) … hết lượt
```

Khung công cụ đã gợi ý rõ cách sửa, nhưng model **không đổi cách làm** sau lỗi. Ngoài ra bản sửa định thay
(`1 + min(...)`) cũng sai logic (đúng phải là `1 + mxm`) — tức là kể cả `edit_file` chạy được thì bài này vẫn chưa chắc qua.

### 4.2 Lặp lại đúng một hành động không có tác dụng — 6 bài, 25 lần "old và new giống nhau"

Ví dụ `agent_implement_779` (đếm số lần xuất hiện của các list con, trả về dict với khoá là tuple):

```
bước 1: edit_file → viết  counts[item] = counts.get(item, 0) + 1   (sai: item là list, phải đổi thành tuple(item))
bước 2: run_tests → ❌ test hỏng
bước 3: edit_file old == new  →  Lỗi: old và new giống nhau
bước 4: run_tests → ❌
bước 5: edit_file old == new  →  Lỗi: old và new giống nhau
… lặp tới hết lượt
```

Model **thấy test hỏng nhưng không suy ra được nguyên nhân** (list không làm khoá dict được) và cứ gửi lại y nguyên bản cũ.

### 4.3 Ghi đè cả file liên tục bằng `write_file`, không chạy test / không báo xong

Ví dụ `agent_fix_644`, `agent_fix_740`: từ bước 6–9 model gọi `write_file` với **cùng một nội dung** 3–4 lần liên tiếp.
Hai bài này tình cờ **qua** vì nội dung ghi đè là đúng, nhưng model không gọi `run_tests` để biết điều đó và không `done`.

### 4.4 Một bài rơi vào "lải nhải" (`agent_implement_794`)

Sau 2 lượt đọc file, model sinh một `edit_file` có chuỗi `#$\n#$\n#$…` lặp vô tận → JSON hỏng → 8 lượt cuối
không gọi được công cụ nào.

## 5. Nhận định

- **Giao thức đã học xong:** 342/352 lượt (97%) là lời gọi công cụ hợp lệ; quy trình chuẩn (test → đọc → sửa → test → done)
  xuất hiện đúng ở các bài dễ.
- **Thứ còn thiếu là "phục hồi sau lỗi":** khi công cụ báo lỗi hoặc test vẫn hỏng, model **lặp lại** thay vì đổi cách.
  Điều này hợp lý: 5.910 quỹ đạo của thầy đều là quỹ đạo **thành công**, phần lớn đi thẳng; model ít thấy cảnh
  "làm sai → nhận lỗi → đổi cách".
- **Sửa lỗi (45%) dễ hơn viết hàm (20%)**: sửa chỉ cần đổi 1 dòng; viết hàm cần tự nghĩ ra lời giải — đúng với năng lực
  viết code còn thấp của model (HumanEval 11%).
- **Khoảng trắng/TAB** là một nguồn lỗi riêng (tokenizer bpe_v1 cắt khoảng trắng đầu dòng kém, xem đo tokenizer trên code).

## 6. Đề xuất bước tiếp

| # | Việc | Nhắm vào |
|---|---|---|
| 1 | **Thêm quỹ đạo "phục hồi"** vào C3: lấy từ nhật ký thầy những quỹ đạo có lượt lỗi rồi sửa được (edit không khớp → đọc lại → copy đúng; test hỏng → đọc traceback → đổi lời giải) | 4.1, 4.2 |
| 2 | **Bài có TAB / thụt lề lạ** trong C3 để model học copy y nguyên | 4.1 |
| 3 | **Khung công cụ chặn lặp:** gọi lại y hệt lời gọi trước thì trả lời ngay "bạn vừa làm đúng việc này — hãy đổi cách" | 4.2, 4.3 |
| 4 | **C5 — RL theo test** trên 23.000 bài C3 còn lại: thưởng khi test qua, phạt khi lặp/hết lượt — học thẳng hành vi phục hồi | tất cả |
| 5 | Tăng năng lực viết code nền (C2b đa ngôn ngữ đang chạy, C4 lần 3 tối nay) | implement 20% |

## 7. Tình trạng hiện tại

- Checkpoint `aivie_p3_c4b` đã đẩy lên HF: `nn-aivie-2p6b-p3-snap/aivie_2p6b_p3_c4b_bf16.pt` (16/10 14:25).
- Nhật ký đầy đủ 40 bài: `out/code_eval/agent_c4b.jsonl` trên box.
- Đêm nay: C2b phần B (thầy viết test + chạy thật 10 nhóm ngôn ngữ, lưu nhật ký đầy đủ) → C4 lần 3 (code đa ngôn ngữ)
  → đo lại toàn bộ + MultiPL-E 7 ngôn ngữ, kết quả ~05:00 sáng 17/10.
