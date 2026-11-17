# RUNBOOK TK_045 P74 — chấm mù vòng chín: **CHẶN BẢN LỀ ĐỔI VAI** (§13.39)

Runbook tự đủ. Người/agent chạy nó **không cần** đọc lịch sử chat.

## 0. Vì sao vòng này

Hệ nối hai mệnh đề qua **một âm tiết bản lề**. Bệnh: bản lề là âm tiết **cuối** của một từ ở
vế trước, rồi mối nối đọc nó thành âm tiết **đầu** của một từ khác — `quốc gia` ⇒ `gia vị`,
`tối đa` ⇒ `đa số`, `tư duy` ⇒ `duy nhất`. Câu vẫn trôi, nhưng **chủ đề trượt** ngay tại chỗ
nối.

§13.39 đo: **10,7%** mối nối sinh có chữ ký này, so với **0,0%** mối nối trong văn người viết.
Vòng này thử một bộ chặn nhắm đúng nó.

Hai bên **cùng** cổng phủ; chỉ khác: bên B **chặn** ứng viên làm bản lề đổi vai.

⚠️ Bên B **có thể trống**: chặn bớt ứng viên thì có đề tắc hẳn. Vòng này **không giấu** chỗ
đó — cái giá phải nằm trong phiếu.

## 1. Sinh cặp — giao M4, xem `RUNBOOK_TK_045_P47.md` PHẦN J

Artefact: `out/pivot_pairs_v12.txt` + khoá `out/pivot_key_v12.json`.

## 2. Giao chấm

**Một phiên Claude RIÊNG**, không phải phiên đã sinh cặp.

⛔ Người chấm **không được** đọc: `out/pivot_key_v12.json`, mọi `out/*_key*.json` vòng khác,
`TK_045_MATRIX_CORE.md`, và mọi file `RUNBOOK_*`.

### 2.1 Prompt giao giám khảo (copy nguyên khối dưới)

> Chấm mù các cặp trong `out/pivot_pairs_v12.txt`. Với **từng** cặp chọn **X / Y / hoà** theo:
> (a) đúng ngữ cảnh · (b) không salad · (c) văn phong tự nhiên trôi chảy.
>
> ⚠️ Một bên **có thể TRỐNG** — in là `(TRỐNG — hệ không sinh được gì)`. Đó là hành vi thật của
> một cấu hình, không phải lỗi dữ liệu. **Trống không tự động thua**: bên kia salad thì trống
> thắng. **Trống cũng không tự động hoà**: bên kia đọc được và bám đề thì trống là thua thật.
>
> ⚠️ **ĐỘ DÀI KHÔNG phải tiêu chí.** Một câu ngắn đúng đề tốt hơn năm câu dài lạc đề.
>
> ⚠️ **X/Y là VỊ TRÍ đã trộn ngẫu nhiên từng cặp**, không phải nhãn cấu hình. Đừng suy luận
> thống kê ở mức X-vs-Y.
>
> Hai bên thường **giống hệt nhau ở phần đầu rồi rẽ nhánh**. Chỗ **rẽ nhánh** thường là chỗ
> quyết định — đọc kỹ đúng chỗ đó.
>
> ⛔ Không mở `out/pivot_key_v12.json`, không mở khoá vòng khác, không mở
> `TK_045_MATRIX_CORE.md` hay file `RUNBOOK_*` nào.
>
> Ghi `out/pivot_verdict_v12.json`:
> `{"verdicts": {"G000": "X", ...}, "notes": "5-8 câu"}`
>
> Trong `notes` trả lời hai câu hỏi, kèm mã cặp làm ví dụ:
> 1. Có bên nào hay **cắt đôi tên riêng / cụm từ cố định** (kiểu `Lý Thái Tổ` bị đứt sau `Lý`,
>    `Việt Nam` bị đứt sau `Việt`) không?
> 2. Có bên nào hay **trượt chủ đề ngay tại chỗ nối** — bám đúng một âm tiết rồi nhảy sang cụm
>    khác hẳn nghĩa — không?
>
> Hai câu đó quan trọng hơn tỷ lệ thắng: chúng phân biệt được *bộ chặn có ăn đúng bệnh không*
> với *bộ chặn có phá hỏng thứ đang lành không*.

## 3. Lật khoá & kết luận

```bash
PYTHONPATH=src python scripts/score_blind_verdict.py \
  --key out/pivot_key_v12.json --verdict out/pivot_verdict_v12.json
```

Ghi TRƯỚC khi thấy phiếu:

- **`fold_chanbanle` thắng, CI95 tách 50%** ⇒ bệnh gốc **có lối chữa đầu tiên** sau chín vòng.
  Đề xuất tác giả bật `pivot_max=0.05` mặc định. Đây sẽ là cơ chế **thứ hai** trong toàn §13
  được arbiter xác nhận (sau cổng phủ fold).
- **Hoà** ⇒ chặn đúng chỗ nhưng **không đổi được chất lượng cảm nhận**. Giữ tắt. Ghi rõ: khuyết
  tật đo được ở mức 100× so với văn người mà chữa vẫn không ăn ⇒ nó **không phải** thứ quyết
  định chất lượng.
- **`fold` thắng** ⇒ bộ chặn **hại**. Đọc `notes` mục 1: nhiều khả năng nó cắt đôi tên riêng
  (`Lý Thái Tổ`, `Việt Nam`) — cái giá đã lường trước ở §13.39 nhưng chưa định lượng được.

⚠️ Mọi kết cục: **không** chỉnh `pivot_max` rồi chấm lại. Muốn quét ngưỡng thì phải nêu trước,
trên bộ đề khác, và ghi là vòng mới.
