# RUNBOOK TK_045 P76 — chấm mù vòng mười một: **DỪNG KHI ĐÓNG ĐƯỢC CÂU** (§13.42)

Runbook tự đủ. Người/agent chạy nó **không cần** đọc lịch sử chat.

## 0. Vì sao vòng này

Hệ sinh bằng cách **nối các mệnh đề có sẵn** thành chuỗi. Vòng trước cho một kết quả dứt khoát:
dừng ở **2** mệnh đề thắng dừng ở **5** với tỷ lệ **78,7%**. Lý do không phải "đi xa thì lạc" —
mỗi mối nối hỏng như nhau (~10%), nên chuỗi dài chỉ đơn giản có **nhiều cơ hội hỏng hơn**.

Nhưng dừng ở đúng 2 là **hằng số mù**: **37%** điểm dừng rơi vào **giữa câu**, và giám khảo
vòng trước đã phạt 8 ca đứt ngang (`dừng ở 'hoặc các'`, `'và nếu'`, `'một cách'`, `'bằng'`).

Vòng này thử: giữ sàn 2 mệnh đề, rồi **đi tiếp cho tới khi gặp mệnh đề đóng được câu** (trần
5). Câu hỏi: **đóng được câu có đáng cái giá của một mối nối tăng thêm không?**

Hai bên **giống hệt nhau ở phần đầu**; bên dài = bên ngắn cộng thêm đuôi.

## 1. Sinh cặp — giao M4, xem `RUNBOOK_TK_045_P47.md` PHẦN L

Artefact: `out/close_pairs_v12.txt` + khoá `out/close_key_v12.json`.

## 2. Giao chấm

**Một phiên Claude RIÊNG**, không phải phiên đã sinh cặp.

⛔ Người chấm **không được** đọc: `out/close_key_v12.json`, mọi `out/*_key*.json` vòng khác,
`TK_045_MATRIX_CORE.md`, và mọi file `RUNBOOK_*`.

### 2.1 Prompt giao giám khảo (copy nguyên khối dưới)

> Chấm mù các cặp trong `out/close_pairs_v12.txt`. Với **từng** cặp chọn **X / Y / hoà** theo:
> (a) đúng ngữ cảnh · (b) không salad · (c) văn phong tự nhiên trôi chảy.
>
> ⚠️ **Một bên luôn là phần đầu nguyên vẹn của bên kia** — bên dài = bên ngắn cộng thêm đuôi.
> Câu hỏi thật: **đoạn thêm đó làm đầu ra TỐT HƠN hay TỆ ĐI?**
>
> Vòng này có một trục cụ thể cần để ý: **bên ngắn có bị đứt ngang không**, và **đoạn thêm có
> đưa câu về chỗ trọn vẹn không**.
> - Bên ngắn dừng giữa câu (kết ở `và`, `hoặc`, `một cách`, `bằng`, dấu hai chấm…) là **lỗi
>   thật**, phải phạt.
> - Nhưng đoạn thêm mà **đóng được câu bằng cách lạc đề** thì **không** cứu được gì — trọn vẹn
>   về hình thức mà sai nội dung vẫn là hỏng.
> - Đoạn thêm vô thưởng vô phạt ⇒ **hoà**.
>
> ⚠️ **ĐỘ DÀI KHÔNG phải tiêu chí, theo CẢ HAI chiều.** Đừng thưởng bên dài vì "đầy đủ hơn",
> đừng thưởng bên ngắn vì "gọn".
>
> ⚠️ **X/Y là VỊ TRÍ đã trộn ngẫu nhiên từng cặp**, không phải nhãn cấu hình. Đừng suy luận
> thống kê ở mức X-vs-Y.
>
> ⛔ Không mở `out/close_key_v12.json`, không mở khoá vòng khác, không mở
> `TK_045_MATRIX_CORE.md` hay file `RUNBOOK_*` nào.
>
> Ghi `out/close_verdict_v12.json`:
> `{"verdicts": {"G000": "X", ...}, "notes": "5-8 câu"}`
>
> Trong `notes` trả lời **ba câu**, kèm mã cặp:
> 1. Đoạn thêm có **thường xuyên đóng được câu** không, hay nó chỉ kéo dài rồi lại treo lơ lửng?
> 2. Khi đoạn thêm đóng được câu, nó có **giữ đúng đề** không, hay đóng bằng cách nhảy chủ đề?
> 3. Bên ngắn bị **đứt ngang** ở bao nhiêu ca đáng kể, và đứt kiểu gì?

## 3. Lật khoá & kết luận

```bash
PYTHONPATH=src python scripts/score_blind_verdict.py \
  --key out/close_key_v12.json --verdict out/close_verdict_v12.json
```

Ghi TRƯỚC khi thấy phiếu:

- **`dong_cau` thắng, CI95 tách 50%** ⇒ **điểm dừng có căn cứ hơn hằng số**. Đề xuất tác giả
  bật `stop_on_close=True, min_clauses=2` mặc định. Đây sẽ là cơ chế **thứ ba** được arbiter
  xác nhận.
- **Hoà** ⇒ kết luận **sạch và hữu ích**: *đóng được câu không đáng giá bằng một mối nối tăng
  thêm.* Giữ `len2` làm điểm vận hành, và ghi rằng trần thật nằm ở **độ tin cậy mỗi mối nối**,
  không ở hình thức câu.
- **`len2` thắng** ⇒ đi thêm **luôn** lỗ, kể cả khi có lý do chính đáng. Khi đó hướng duy nhất
  còn lại là **nâng độ tin cậy mỗi mối nối lên trên ~90%**, hoặc **sinh 1–2 mệnh đề và dừng
  hẳn** — tức chấp nhận trần hiện tại và tối ưu trong nó.

⚠️ Mọi kết cục: **không** quét `min_clauses` 1/3/4 rồi chấm lại để tìm số đẹp. Muốn quét thì
nêu trước, bộ đề khác, ghi là vòng mới.
