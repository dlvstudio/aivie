# RUNBOOK TK_045 P75 — chấm mù vòng mười: **DỪNG SỚM** (§13.41)

Runbook tự đủ. Người/agent chạy nó **không cần** đọc lịch sử chat.

## 0. Vì sao vòng này

Hệ sinh bằng cách **nối các mệnh đề có sẵn** thành chuỗi, tối đa 5 mệnh đề. Chín vòng chấm mù
vừa qua đều hỏi *"nối mệnh đề nào thì tốt hơn"*. Vòng này hỏi một câu chưa ai hỏi: **nối thêm
có lợi không?**

Lý do có câu hỏi này: ở vòng chín, khi một cấu hình vô tình **cắt ngắn** chuỗi, bên bị cắt
thắng **84%** (CI95 [65,3%, 93,6%]); còn khi nó chỉ **đổi hướng nối**, nó là **đồng xu**
(51,3%). Nhưng đó là phân tích **hậu nghiệm**. Vòng này nêu trước để xác nhận hoặc bác.

Hai bên **giống hệt nhau về mọi thứ** — cùng cổng, cùng seed, cùng cơ chế — khác đúng **một
tham số**: bên A đi tối đa **5** mệnh đề, bên B dừng ở **2**. Vì vậy bên ngắn **luôn** là phần
đầu nguyên vẹn của bên dài; bên dài chỉ là bên ngắn **cộng thêm đuôi**.

## 1. Sinh cặp — giao M4, xem `RUNBOOK_TK_045_P47.md` PHẦN K

Artefact: `out/len_pairs_v12.txt` + khoá `out/len_key_v12.json`.

## 2. Giao chấm

**Một phiên Claude RIÊNG**, không phải phiên đã sinh cặp.

⛔ Người chấm **không được** đọc: `out/len_key_v12.json`, mọi `out/*_key*.json` vòng khác,
`TK_045_MATRIX_CORE.md`, và mọi file `RUNBOOK_*`.

### 2.1 Prompt giao giám khảo (copy nguyên khối dưới)

> Chấm mù các cặp trong `out/len_pairs_v12.txt`. Với **từng** cặp chọn **X / Y / hoà** theo:
> (a) đúng ngữ cảnh · (b) không salad · (c) văn phong tự nhiên trôi chảy.
>
> ⚠️ Trong vòng này, **một bên luôn là phần đầu nguyên vẹn của bên kia** — bên dài = bên ngắn
> cộng thêm đuôi. Câu hỏi thật là: **đoạn thêm đó làm đầu ra TỐT HƠN hay TỆ ĐI?**
> - Đoạn thêm bám đề, mở rộng ý, đọc trôi ⇒ **bên dài thắng**.
> - Đoạn thêm lạc đề, salad, hoặc kéo lê ⇒ **bên ngắn thắng**.
> - Đoạn thêm vô thưởng vô phạt ⇒ **hoà**.
>
> ⚠️ **ĐỘ DÀI KHÔNG phải tiêu chí, theo CẢ HAI chiều.** Đừng thưởng cho bên dài vì nó "đầy đủ
> hơn"; cũng đừng thưởng cho bên ngắn vì nó "gọn". Chỉ hỏi đoạn thêm có đáng có mặt không.
> Một bên ngắn **bị cụt giữa chừng** (đứt ngang, hụt nghĩa) thì đó là **lỗi của nó**, phải bị
> phạt — đừng nhầm "dừng gọn" với "đứt ngang".
>
> ⚠️ **X/Y là VỊ TRÍ đã trộn ngẫu nhiên từng cặp**, không phải nhãn cấu hình — bên ngắn không
> phải lúc nào cũng ở một phía. Đừng suy luận thống kê ở mức X-vs-Y.
>
> ⛔ Không mở `out/len_key_v12.json`, không mở khoá vòng khác, không mở
> `TK_045_MATRIX_CORE.md` hay file `RUNBOOK_*` nào.
>
> Ghi `out/len_verdict_v12.json`:
> `{"verdicts": {"G000": "X", ...}, "notes": "5-8 câu"}`
>
> Trong `notes` trả lời ba câu, kèm mã cặp:
> 1. Đoạn thêm hỏng theo **kiểu** gì khi nó hỏng?
> 2. Có ca nào đoạn thêm **thật sự cải thiện** không? Nêu mã cặp cụ thể; nếu không có, nói rõ.
> 3. Có ca nào bên ngắn **bị cụt** (đứt ngang, hụt nghĩa) không?

## 3. Lật khoá & kết luận

```bash
PYTHONPATH=src python scripts/score_blind_verdict.py \
  --key out/len_key_v12.json --verdict out/len_verdict_v12.json
```

Ghi TRƯỚC khi thấy phiếu:

- **`len2` thắng, CI95 tách 50%** ⇒ **trần chất lượng nằm ở ĐỘ DÀI CHUỖI.** Chín vòng vừa qua
  đều tối ưu *cách chọn* — nhắm sai chỗ. Nhưng **không** kết luận "đặt `max_clauses=2` là
  xong": đó là hằng số mù. Kết luận đúng là **cần một cơ chế DỪNG có căn cứ**, và `len2` trở
  thành **mốc so sánh mới** cho mọi vòng sau.
- **Hoà** ⇒ đoạn thêm không lợi cũng không hại. Giữ nguyên `max_clauses=5`; giả thuyết "nối
  thêm làm hại" **không đứng**, và con số 84% ở §13.40 là hiện vật của mẫu thiên.
- **`len5` thắng** ⇒ nối thêm **có lợi**, 84% ở §13.40 **bị bác thẳng**, và §13.40 phải ghi
  lại. Khi đó chỗ đáng nhìn là: vì sao cắt ngắn *bằng bộ chặn bản lề* lại thắng mà cắt ngắn
  *bằng tham số* lại thua — tức bộ chặn cắt **đúng chỗ**, và điểm dừng **có căn cứ** mới là
  thứ đáng làm.

⚠️ Mọi kết cục: **không** quét `--short` 3/4 rồi chấm lại để tìm số đẹp. Muốn quét thì nêu
trước, trên bộ đề khác, và ghi là vòng mới.
