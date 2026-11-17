# RUNBOOK TK_045 P71 — chấm mù vòng tám: **ĐỀ BỎ TRỐNG TÍNH LÀ THUA** (§13.37)

Runbook tự đủ. Người/agent chạy nó **không cần** đọc lịch sử chat.

## 0. Vì sao vòng này khác bảy vòng trước

Bảy vòng chấm mù trước đều **loại** cặp có bên rỗng (`if not a or not b: continue`). Nghĩa là
giám khảo **chưa bao giờ** nhìn thấy cái giá của việc hệ **không sinh được gì**.

§13.36 đo được cái giá đó không nhỏ: cổng phủ làm **20,7% số đề** trả về **rỗng**, và làm chuỗi
ngắn gần một nửa (2,62 vs 4,66 mệnh đề). Nhưng cùng cổng đó **thắng cả bốn vòng** chấm mù về
chất lượng. Hai sự thật không mâu thuẫn — chúng đo hai thứ khác nhau, và vòng này ghép chúng
lại làm một.

Hai bên trong vòng này **giống hệt nhau** ở mọi đề mà cổng sinh được. Mọi cặp đem chấm đều là
đề mà một bên **trống**.

## 1. Sinh cặp — giao M4, xem `RUNBOOK_TK_045_P47.md` PHẦN I

Artefact: `out/empty_pairs_v12.txt` + khoá `out/empty_key_v12.json`. ⚠️ Đừng chạy trên máy dev
Intel 8GB (nạp v12 mất 83 phút rồi swap).

## 2. Giao chấm

**Một phiên Claude RIÊNG**, không phải phiên đã sinh cặp.

⛔ Người chấm **không được** đọc: `out/empty_key_v12.json`, mọi `out/*_key*.json` vòng khác,
`TK_045_MATRIX_CORE.md`, và các runbook.

### 2.1 Prompt giao giám khảo (copy nguyên khối dưới)

> Chấm mù các cặp trong `out/empty_pairs_v12.txt`. Với **từng** cặp chọn **X / Y / hoà** theo:
> (a) đúng ngữ cảnh · (b) không salad · (c) văn phong tự nhiên trôi chảy.
>
> ⚠️ **Vòng này một bên thường TRỐNG** — in là `(TRỐNG — hệ không sinh được gì)`. Đây không
> phải lỗi dữ liệu; đó là hành vi thật của một trong hai cấu hình. Cách chấm:
> - **Trống KHÔNG tự động thua.** Nếu bên kia là salad, lạc đề, hay chắp vá vô nghĩa thì
>   **TRỐNG THẮNG** — thà im lặng còn hơn nói bậy.
> - **Trống cũng KHÔNG tự động hoà.** Nếu bên kia đọc được và bám đề thì **trống là THUA**.
> - Chọn "hoà" khi bên có chữ vừa đủ tệ để không hơn gì im lặng, mà cũng chưa tệ tới mức
>   im lặng đáng hơn.
>
> ⚠️ **ĐỘ DÀI KHÔNG phải tiêu chí.** Một câu ngắn đúng đề tốt hơn năm câu dài lạc đề.
>
> ⚠️ **X/Y là VỊ TRÍ đã trộn ngẫu nhiên từng cặp**, không phải nhãn cấu hình — bên trống
> không phải lúc nào cũng ở một phía. Đừng suy luận thống kê ở mức X-vs-Y.
>
> ⛔ Không mở `out/empty_key_v12.json`, không mở khoá vòng khác, không mở
> `TK_045_MATRIX_CORE.md` hay các file `RUNBOOK_*`.
>
> Ghi `out/empty_verdict_v12.json`:
> `{"verdicts": {"G000": "X", ...}, "notes": "5-8 câu"}`
>
> Trong `notes` trả lời thẳng một câu: **bên có chữ hỏng theo kiểu gì**, và **có ca nào bên có
> chữ thật sự dùng được không** (nêu mã cặp). Đó là thứ quyết định, không phải tỷ lệ thắng.

## 3. Lật khoá & kết luận

```bash
PYTHONPATH=src python scripts/score_blind_verdict.py \
  --key out/empty_key_v12.json --verdict out/empty_verdict_v12.json
```

Ghi TRƯỚC khi thấy phiếu — ba kết cục, ba nghĩa:

- **`fold_duphong` thắng, CI95 tách 50%** ⇒ **im lặng là khuyết tật thật**. Đề xuất tác giả bật
  dự phòng khi rỗng. Và §13.36 được đọc lại: cổng phủ **đang trả giá quá đắt**.
- **Hoà (CI phủ 50%)** ⇒ trống và salad **ngang nhau**. Giữ nguyên mặc định — nhưng ghi rõ:
  20,7% im lặng **không** phải khuyết tật đáng vá bằng cách nới cổng, vì nới ra chỉ đổi im lặng
  lấy chữ vô giá trị.
- **`fold` (bên trống) thắng** ⇒ **im lặng TỐT HƠN nói bậy**. Cổng phủ đúng ngay cả khi tính cả
  cái giá của nó. Kết luận mạnh nhất: hướng **"nới cổng cho chuỗi dài ra"** chết hẳn, và trần
  chất lượng nằm ở chỗ khác — không phải ở độ chặt của cổng.

⚠️ Dù kết cục nào: **không** chỉnh `anchor_min_folds` / `gidf_min` rồi chấm lại. Đó là dời cổng
sau khi thấy số — §13.32 đã ghi một lần vì sao không được làm vậy.
