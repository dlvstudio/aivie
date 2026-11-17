# RUNBOOK TK_045 P66 — chấm mù vòng bảy: mối nối `S1` vs **NEO-ĐỀ** (§13.31)

Runbook này tự đủ. Người/agent chạy nó **không cần** đọc lịch sử chat.

## 0. Vì sao vòng này

`join_rescore` hiện **tắt**. §13.30: bộ xếp `pmi_cross(A,B)` (`S3`) **thua** `S1` 59.7% dù AUC
ngoại tuyến 0.814 — vì nó tối ưu *mạch lạc cục bộ*, còn cái cần là *bám đề toàn cục*.

§13.31 đo tiếp và tìm ra chỗ hỏng thật: truy hồi **không** hỏng (98.2% mệnh đề đúng tới được),
**xếp hạng** mới hỏng — mệnh đề đúng nằm ở hạng trung vị **51/125**, tức thứ tự ngẫu nhiên.

Ứng viên vòng này: `prompt_anchor_rescorer` — xếp ứng viên theo **tổng độ hiếm `gidf` của các
fold trùng với ĐỀ**. Hai điểm mới so với mọi lần trước:

1. **Xếp, không chặn.** Mọi tiêu chí chặt trước đây đều làm **cổng** và chết đường (P57: 94%).
   Bể ứng viên trung vị ~1.550 mệnh đề ⇒ làm **bộ xếp** thì chặt là miễn phí, không có ngõ cụt.
2. **Trọng số hiếm, không đếm.** §13.20: đếm thì rác (`nước`, `1`, `bằng`) chi phối.

Đo trước khi chấm: sống ở **100%** mối nối. Tỷ lệ đổi đầu ra: **38.3%** trên mini, nhưng
**26.0%** trên v12 (§12.156/§13.32) — con số v12 mới là con số thật.

⚠️ **Cổng G-anchor mục (3) đã TRƯỢT** ở v12 (bỏ 74% > 70%). Vòng chấm này **không** trả lời
"có bật mặc định không" — cổng đã đóng câu đó, âm tính. Nó trả lời câu **khác**: *khi cơ chế
có chạm vào mối nối, nó có làm tốt hơn không?* §13.32 đã ghi trước ý nghĩa của cả ba kết cục;
đọc mục 3 dưới, đừng diễn giải lại sau khi thấy phiếu.

⚠️ Đây **chưa** là kết luận. Ba proxy ngoại tuyến liên tiếp đã đánh lừa. **Chấm mù là arbiter
duy nhất còn hiệu lực.**

## 1. Sinh cặp — ĐÃ XONG (M4, §12.156)

`out/anchor_pairs_v12c.txt` **104 cặp** (đủ power) + khoá `out/anchor_key_v12c.json`. Sinh từ
400 đề, `--mode anchor`, checkpoint `network_v12_sent.pkl`. Hai lượt phụ thiếu power:
`anchor_pairs_v12.txt` (43) và `_v12b.txt` (87) — **đừng** dùng.

Cần sinh lại thì xem `RUNBOOK_TK_045_P47.md` PHẦN G, và **giao M4**: máy dev Intel 8GB nạp
xong checkpoint v12 mất 83 phút rồi **swap**, 70 phút không qua nổi cột mốc 50.000 cha đầu của
`build_clause_layer`.

Hai bên **cùng** cổng fold; chỉ khác `join_rescore`. `A = join_S1` (đang chạy, không rescore),
`B = join_anchor`. Script tự trộn trái/phải và giấu nhãn; khoá ở `out/anchor_key_v12c.json`.
Script **từ chối** ghi đè khoá khi đã có verdict tương ứng — đừng `--force`.

Cần **≥88 cặp** cho +15pp @ power 80%. Nếu ít hơn, tăng `--n-prompt` và sinh vào **tên file
khác**.

## 2. Giao chấm

**Một phiên Claude RIÊNG**, không phải phiên đã sinh cặp.

⛔ Người chấm **không được** đọc: `out/anchor_key_v12c.json`, mọi `out/*_key*.json` vòng khác,
và `TK_045_MATRIX_CORE.md`.

### 2.1 Prompt giao giám khảo (copy nguyên khối dưới)

> Chấm mù các cặp trong `out/anchor_pairs_v12c.txt`. Với **từng** cặp chọn **X / Y / hoà** theo:
> (a) đúng ngữ cảnh · (b) không salad · (c) văn phong tự nhiên trôi chảy. So sánh **trực tiếp**
> trong cặp; chọn "hoà" khi thật sự không phân biệt được.
>
> ⚠️ **Độ dài KHÔNG phải tiêu chí.** Đừng ưu ái bên dài; đừng phạt bên ngắn nếu nó dừng gọn và
> đúng đề. **Một câu ngắn đúng đề tốt hơn ba câu dài lạc đề.**
>
> ⚠️ **X/Y là VỊ TRÍ đã trộn ngẫu nhiên**, không phải nhãn cấu hình. Đừng suy luận thống kê ở
> mức X-vs-Y (đã có vòng nhầm chỗ này và kết luận **ngược**).
>
> ⛔ Không mở `out/anchor_key_v12c.json`, không mở khoá của vòng khác, không mở
> `TK_045_MATRIX_CORE.md`.
>
> Ghi `out/anchor_verdict_v12c.json`:
> `{"verdicts": {"G000": "X", "G001": "hoà", ...}, "notes": "5-8 câu về KIỂU hỏng của mỗi bên"}`
>
> `notes` quan trọng ngang số đếm: §13.30 kết luận được là nhờ giám khảo tả được kiểu hỏng
> (*"nối bằng số/công thức, bẻ gãy ngữ vực"*), không phải nhờ tỷ lệ thắng.

## 3. Lật khoá & kết luận

```bash
PYTHONPATH=src python scripts/score_blind_verdict.py \
  --key out/anchor_key_v12c.json --verdict out/anchor_verdict_v12c.json
```

Script này là **nguồn duy nhất** để lật khoá: nó đếm theo **CẤU HÌNH** (không phải vị trí —
lỗi đó đã xảy ra hai lần), chuẩn hoá NFC chuỗi verdict, và luôn in thắng-thua **tách theo vị
trí** để phát hiện thiên lệch vị trí.

Đọc kết quả:

- **`join_anchor` thắng, CI95 tách khỏi 50%** ⇒ hướng "xếp theo neo-đề có trọng số hiếm"
  **đúng**, nhưng trần hiệu quả chỉ **26% đầu ra**. **Chưa** đủ để bật mặc định (cổng (3) đã
  trượt) — đủ để mở **PHẦN H**: vá trần bão hoà `gidf` (mục 4), với cổng "tỷ lệ đổi ở v12 phải
  TĂNG" nêu trước.
- **Hoà (CI phủ 50%)** ⇒ cơ chế **không** cải thiện chất lượng cảm nhận. Đóng hướng neo-đề,
  **không** vá `gidf`. **Không** tinh chỉnh `gidf_min` rồi chấm lại — đó là dời cổng sau khi
  thấy số.
- **`join_S1` thắng** ⇒ dương tính giả thứ tư, và là cái đắt nhất: **cả ba** proxy đo trước
  (100% sống · 38.3% đổi trên mini · 3/3 đọc tay tốt hơn) đều thuận mà vẫn thua. Ghi rõ rằng
  ba proxy đó **không** dự báo được chấm mù — dữ kiện về THƯỚC, đáng hơn cả cơ chế.

## 4. Nếu thắng — bước nâng cấp rõ nhất đã biết

`gidf` **bão hoà ở 9.64**: mọi token gặp trên ~50 lần đều cùng điểm ⇒ không phân biệt `đây`
(46 lần) với `thê` (2 lần); rác §13.20 nằm ở 4.0–4.2. Trần này nằm trong `ParentStore._gidf`.
Sửa nó là cách rẻ nhất để bộ xếp sắc hơn — **sau** khi chấm mù xác nhận hướng, không phải
trước.
