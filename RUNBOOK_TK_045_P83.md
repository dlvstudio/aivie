# RUNBOOK TK_045 P83 — `len1` vs `len2`: **thí nghiệm CÓ GÁN** cho "mỗi mối nối đều làm hỏng" (§13.51)

Runbook tự đủ: mục 1 giao M4, mục 2 giao giám khảo.

## 0. Vì sao

§13.50 cắt lại 200 mẫu mốc gốc theo **số mệnh đề**:

| đầu ra có | n | `dung_duoc`+`tam` |
|---|---:|---:|
| 1 mệnh đề — **không mối nối nào** | 50 | **24,0%** |
| 2 mệnh đề — **một mối nối** | 91 | **6,6%** |

z = 2,96 (p < 0,01), và kiểm chéo trên phiên chấm khác cho 20,0% vs 6,6%.

**Nhưng đó là quan sát, không phải thí nghiệm.** Nhóm "1 mệnh đề" **không được gán ngẫu
nhiên** — đó là những đề mà chuỗi **tự chết** sau mệnh đề đầu, rất có thể khác về bản chất.
Vòng này **ép** mọi đề dừng ở 1 ⇒ không còn chọn mẫu.

## 1. Sinh mẫu — **giao M4**

```bash
PYTHONPATH=src .venv/bin/python scripts/gen_absolute_samples.py \
    --checkpoint checkpoints/network_v12_sent.pkl --n 200 --len-short 1 \
    --out out/abs4_samples_v12.txt --out-key out/abs4_key_v12.json \
    --out-meta out/abs4_meta_v12.json
```

- **`len2`** — `max_clauses=2`, cấu hình hiện tại, trên **200 đề đầu**.
- **`len1`** — `max_clauses=1`, trên **200 đề TIẾP THEO** (rời hẳn).

### 1.1 Vì sao BỘ ĐỀ RỜI NHAU — đây là thay đổi thiết kế có chủ ý

§13.49 phát hiện chấm-cặp-tuyệt-đối **hở nhãn một phần**: hai cấu hình chạy trên cùng bộ đề nên
mỗi đề xuất hiện **hai lần**, và giám khảo tự ghép cặp được (*"A043 bỏ trống còn A308 cùng đề
sinh dài"*). Ở vòng này lỗ đó là **tối đa** — bản `len1` là **tiền tố** của bản `len2`, ghép cặp
bằng mắt là chuyện hiển nhiên, và giám khảo sẽ đoán ra "có một bản bị cắt".

⇒ Đổi sang **hai bộ đề rời nhau**. Mất khả năng so **cặp** (yếu hơn về thống kê), nhưng **bịt
hẳn** đường suy nhãn. §13.49 đã ghi là phải chọn có ý thức giữa hai thứ này — đây là lựa chọn.

Smoke mini (`--n 120`): 240 mẫu, **0 đề trùng** giữa hai nhánh.

### 1.2 CỔNG G-len1 (nêu TRƯỚC)

| # | điều kiện | nếu trượt |
|---|---|---|
| 1 | **400 mẫu**, khoá chia đúng **200/200** | bug — dừng, báo |
| 2 | **0 đề trùng** giữa hai nhánh | máy dò bug của thiết kế rời — dừng, báo |
| 3 | tỷ lệ trống hai nhánh chênh **≤ 10pp** | hai bộ đề khác nhau về bản chất ⇒ so sánh bị nhiễu; **báo trước khi chấm** |
| 4 | mệnh đề TB của `len1` **≤ 1,0** | tham số không ăn |

### 1.3 Báo cáo M4

Commit ba file, kèm: trống **theo nhánh** · mệnh đề TB **theo nhánh** · số đề trùng (phải 0) ·
thời gian. **Không chấm.**

## 2. Giao chấm — phiên Claude RIÊNG

⛔ Cấm đọc: `out/abs4_key_v12.json`, mọi `out/*_key*`, `out/*verdict*`, `out/*meta*`,
`TK_045_MATRIX_CORE.md`, mọi `RUNBOOK_*`. File được đọc: **chỉ** `out/abs4_samples_v12.txt`.

### 2.1 Prompt giao giám khảo (copy nguyên khối)

> Trong `out/abs4_samples_v12.txt` có 400 mẫu (A000–A399), mỗi mẫu gồm một **ĐỀ** và phần
> **tiếp theo do máy sinh** (dòng `→`). Chấm **từng mẫu ĐỘC LẬP** — không so mẫu này với mẫu
> khác, không xếp hạng.
>
> Ba mức:
> - **`dung_duoc`** — phần sinh nối vào đề mà đọc **trôi**, **đúng ngữ cảnh**, **không có câu
>   vô nghĩa**. Tiêu chuẩn: gặp đoạn này trong bài viết thật, bạn **không khựng lại**. Không
>   cần hay, chỉ cần **không sai**.
> - **`tam`** — có ý đúng đề, đọc được phần nào, nhưng vấp: cụt giữa câu, một cụm lạc, ghép
>   gượng. Sửa vài chữ là dùng được.
> - **`hong`** — salad, lạc đề, tự mâu thuẫn, hoặc chuỗi số/mảnh vụn vô nghĩa.
>
> ⚠️ Mẫu `(TRỐNG — hệ không sinh được gì)` chấm là **`hong`**.
>
> ⚠️ **Đừng nới tay.** Phân vân giữa `dung_duoc` và `tam` ⇒ chọn **`tam`**. Phân vân giữa `tam`
> và `hong` ⇒ chọn **`hong`**.
>
> ⚠️ **Độ dài không phải tiêu chí.** Phần sinh rất ngắn là **thiết kế**, không phải lỗi — chỉ
> phạt nếu nó **đứt ngang giữa câu**. Một mẫu chỉ có một mệnh đề ngắn mà đúng đề và đọc trôi
> thì là `dung_duoc`, đừng hạ bậc vì nó ngắn.
>
> ⚠️ File có **hai loại mẫu trộn lẫn** do hai cấu hình sinh ra, **trên hai bộ đề khác nhau**
> (không đề nào xuất hiện hai lần). Bạn **không biết và không cần biết** mẫu nào thuộc loại
> nào. **Đừng đoán, đừng nhóm, đừng đổi thước giữa chừng.** Chấm tuần tự A000→A399 bằng **cùng
> một** thước.
>
> ⛔ Không mở `out/abs4_key_v12.json`, không mở `*_key*`/`*verdict*`/`*meta*` nào, không mở
> `TK_045_MATRIX_CORE.md` hay `RUNBOOK_*`.
>
> Ghi `out/abs4_verdict_v12.json`:
> `{"grades": {"A000": "hong", ...}, "notes": "6-10 câu"}` — đủ **400 mã**.
>
> Trong `notes`: (1) mẫu `dung_duoc` giống nhau ở điểm gì; (2) kiểu hỏng phổ biến nhất và tỷ
> trọng; (3) có mẫu nào **chỉ thiếu một chút** không, thiếu gì; (4) mẫu **ngắn** (một mệnh đề)
> và mẫu **dài hơn** có hỏng theo kiểu **khác nhau** không — mô tả hiện tượng, đừng đoán nhãn.

## 3. Lật khoá & kết luận

Đếm ba mức **theo từng cấu hình**, mẫu số 200 mỗi bên (gồm mẫu trống), **và** riêng nhóm không
trống.

### Dự đoán ghi TRƯỚC (Claude)

- **`len1` thắng ở CẢ HAI thước.** `dung_duoc + tam` toàn bộ **≥ 12%** (so **8,0%** của `len2`).
- Nhánh `len2` trên **bộ đề mới** phải ra **≈ 8%** (±4pp) — vừa là kiểm thước, vừa là kiểm
  **tổng quát hoá** sang bộ đề khác. Lệch nhiều ⇒ mốc cũ chỉ đúng cho một bộ đề, và **mọi**
  so sánh trước đó phải đọc lại.
- Nếu `len1` **không** hơn ⇒ **24% của §13.50 hoàn toàn là hiện vật chọn mẫu**. Bài học ghi
  trước: *nhóm con nào cũng có thể trông đẹp nếu nó được chọn bằng chính cơ chế đang xét.*

### Ý nghĩa từng kết cục — ghi trước

- **`len1` thắng** ⇒ khẳng định "mỗi mối nối đều làm hỏng". Đề xuất tác giả đặt
  `max_clauses=1` mặc định. Và phải nói thẳng giới hạn: hệ khi đó **không nối gì cả** — nó
  **chọn một mệnh đề có sẵn rồi dán vào đề**. Đó là **truy hồi + ghép một bước**; tham vọng
  "tái tổ hợp nhiều mệnh đề" của Trụ A **chưa có bằng chứng nào ủng hộ**, và §13 phải đổi mục
  tiêu cho trung thực.
- **Hoà** ⇒ mối nối **không** làm hỏng thêm; 24% là hiện vật chọn mẫu; giữ `max_clauses=2`.
- **`len2` thắng** ⇒ mối nối **có ích**, và toàn bộ cách đọc §13.50/§13.41/§13.42 phải viết
  lại — khi đó cái thắng của "ít mệnh đề" ở các vòng trước là do thứ khác, chưa biết là gì.
