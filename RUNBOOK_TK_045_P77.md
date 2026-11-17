# RUNBOOK TK_045 P77 — chấm mù **TUYỆT ĐỐI**: đầu ra có DÙNG ĐƯỢC không? (§13.43)

Runbook tự đủ, gồm **cả** phần giao M4 (mục 1) **và** phần giao giám khảo (mục 2).

## 0. Vì sao vòng này khác mười một vòng trước

Mười một vòng chấm mù đều đo **SO SÁNH**: A có hơn B không. Hai thứ sống sót — **cổng phủ
fold** và **`max_clauses=2`** — vì thế mới chỉ được chứng minh là **ít tệ hơn**, **chưa** bao
giờ được chứng minh là **tốt**.

Bốn giám khảo liên tiếp đã tự nói ra điều đó: *"mặt bằng chất lượng chung rất thấp — đa số cặp
là chọn giữa hai đoạn salad"*, *"hiếm khi có bên nào đọc trôi thật sự"*. Nhưng chưa vòng nào
biến câu đó thành **con số**.

⇒ Vòng này không có cấu hình B. Chỉ có **một** cấu hình — cái đang thắng — và câu hỏi duy
nhất: **bao nhiêu phần trăm đầu ra thật sự dùng được?**

Đây là **mốc gốc**. Mọi tiến bộ sau này phải đo lại chính con số này, nếu không thì "cải tiến"
chỉ là thu hẹp về phía ít sai nhất trong một vùng đều sai.

## 1. Sinh mẫu — **giao M4**

⚠️ Đừng chạy trên máy dev Intel 8GB (nạp v12 mất 83 phút rồi swap).

```bash
PYTHONPATH=src .venv/bin/python scripts/gen_absolute_samples.py \
    --checkpoint checkpoints/network_v12_sent.pkl --n 200 \
    --out out/abs_samples_v12.txt --out-meta out/abs_meta_v12.json
```

Cấu hình: **cổng phủ fold + `max_clauses=2`**, đúng cái đang thắng. Không cơ chế thử nghiệm nào.

⚠️ **KHÔNG LỌC.** Script giữ **mọi** đề, kể cả đề hệ **không sinh được gì** — rỗng là một loại
đầu ra và phải nằm trong mẫu số. Mọi vòng trước đều lọc (`a == b`, và bảy vòng đầu bỏ cả bên
rỗng); chính chỗ đó đã đẻ ra lỗi mẫu số §13.35. Đừng "dọn" mẫu.

### 1.1 CỔNG G-abs (nêu TRƯỚC)

| # | điều kiện | nếu trượt |
|---|---|---|
| 1 | **đúng 200 mẫu**, không mẫu nào bị bỏ | script có bug lọc — dừng, báo |
| 2 | tỷ lệ trống ∈ **15–40%** | §13.37 đo 27,8%; lệch nhiều ⇒ báo trước khi chấm |
| 3 | số mệnh đề trung bình ≤ **2,0** | `max_clauses=2` mà vượt ⇒ tham số không ăn |

### 1.2 Báo cáo M4

Commit `out/abs_samples_v12.txt` + `out/abs_meta_v12.json`, kèm: số mẫu · tỷ lệ trống · số
mệnh đề TB · thời gian. **Không chấm.**

## 2. Giao chấm — phiên Claude RIÊNG

⛔ Người chấm **không được** đọc `TK_045_MATRIX_CORE.md`, mọi `RUNBOOK_*`, mọi `out/*_key*`,
mọi `out/*verdict*`. File được đọc: **chỉ** `out/abs_samples_v12.txt`.

Vòng này **không có nhãn cấu hình để lộ** (chỉ một cấu hình), nên rủi ro không phải lộ nhãn mà
là **lộ kỳ vọng**: biết dự án mong gì thì giám khảo sẽ chấm rộng tay. Vì vậy vẫn cấm đọc ticket.

### 2.1 Prompt giao giám khảo (copy nguyên khối dưới)

> Trong `out/abs_samples_v12.txt` có 200 mẫu, mỗi mẫu gồm một **ĐỀ** (đoạn mở đầu) và phần
> **tiếp theo do máy sinh**. Chấm **từng mẫu ĐỘC LẬP** — không so mẫu này với mẫu khác, không
> xếp hạng.
>
> Với mỗi mẫu, cho **một** trong ba mức:
> - **`dung_duoc`** — phần sinh nối vào đề mà đọc **trôi**, **đúng ngữ cảnh**, và **không có
>   câu vô nghĩa**. Tiêu chuẩn: nếu thấy đoạn này trong một bài viết thật, bạn **không** khựng
>   lại. Không cần hay, chỉ cần **không sai**.
> - **`tam`** — có một ý đúng đề, đọc được phần nào, nhưng vấp: cụt giữa câu, một cụm lạc, hoặc
>   ghép hơi gượng. Sửa vài chữ là dùng được.
> - **`hong`** — salad, lạc đề, tự mâu thuẫn, hoặc chuỗi số/mảnh vụn vô nghĩa.
>
> ⚠️ Mẫu ghi `(TRỐNG — hệ không sinh được gì)` chấm là **`hong`**. Không sinh được cũng là
> không dùng được; đừng bỏ qua và đừng cho `tam`.
>
> ⚠️ **Đừng nới tay.** Đây là mốc gốc để đo tiến bộ về sau; chấm rộng bây giờ sẽ giấu mất mọi
> tiến bộ thật sau này. Nếu bạn phân vân giữa `dung_duoc` và `tam`, chọn **`tam`**. Nếu phân
> vân giữa `tam` và `hong`, chọn **`hong`**.
>
> ⚠️ **Độ dài không phải tiêu chí.** Phần sinh ngắn (1–2 mệnh đề) là thiết kế, không phải lỗi —
> chỉ phạt nếu nó **đứt ngang giữa câu**.
>
> ⛔ Không mở `TK_045_MATRIX_CORE.md`, không mở file `RUNBOOK_*` nào, không mở `out/*_key*` hay
> `out/*verdict*` nào.
>
> Ghi `out/abs_verdict_v12.json`:
> `{"grades": {"A000": "hong", "A001": "tam", ...}, "notes": "6-10 câu"}`
>
> `grades` phải có **đủ 200 mã**. Trong `notes`:
> 1. Trong các mẫu bạn cho `dung_duoc`, chúng **giống nhau ở điểm gì**? (nêu mã mẫu)
> 2. Kiểu hỏng **phổ biến nhất** là gì, và nó chiếm khoảng bao nhiêu phần các mẫu `hong`?
> 3. Có mẫu nào chỉ **thiếu một chút** là dùng được không — thiếu cái gì cụ thể?

## 3. Đọc kết quả

Đếm ba mức trên **mẫu số 200** (gồm cả mẫu trống).

### Dự đoán ghi TRƯỚC (Claude)

- `dung_duoc` **≤ 10%** · `hong` **≥ 60%** · trống ~28%.
- Nếu `dung_duoc` **> 25%** ⇒ tôi đã đánh giá hệ **quá thấp**, và mẫu salad tôi trích ở các
  vòng trước là mẫu thiên (chúng đến từ file cặp **đã lọc**).
- Nếu `dung_duoc` **< 3%** ⇒ hệ **chưa sinh được** theo bất kỳ nghĩa thực dụng nào, và mọi
  công việc tinh chỉnh cơ chế phải dừng cho tới khi có thay đổi **khác về chất**.

### Ý nghĩa từng kết cục — ghi trước

- **`dung_duoc` ≥ 25%** ⇒ có thể bắt đầu nói tới "sinh được" cho **một số loại đề**; việc kế là
  tìm xem loại đề nào, và giới hạn hệ vào đó.
- **`dung_duoc` 10–25%** ⇒ chưa dùng được, nhưng có **hạt nhân** — việc kế là đặc trưng hoá
  hạt nhân đó (câu hỏi 1 trong `notes`).
- **`dung_duoc` < 10%** ⇒ **chưa sinh được.** Kết luận thẳng, không vòng vo: hai cơ chế sống
  sót chỉ **thu hẹp cái sai**, chưa tạo ra cái đúng. Khi đó §13 phải chuyển từ "tinh chỉnh
  cách chọn" sang câu hỏi gốc — **vì sao một phép chọn đúng vẫn cho ra câu vô nghĩa**.
