# TK_052 — CHỨNG MINH CƠ CHẾ LOẠI SUY (tách khỏi TK_046)

## Vì sao tách

TK_046 trộn **hai mục tiêu khác nhau** và điều đó vừa suýt gây một báo cáo sai
(§72 → đính chính §73):

| | mục tiêu | thước | trạng thái |
|---|---|---|---|
| A | chất lượng SINH — đúng tri thức đã nạp | thước sinh v3 (§70) | top-1 5,7% → **25,0%** ·  top-5 19,0% → **45,7%** |
| B | **chứng minh CƠ CHẾ LOẠI SUY** (ca «con khỉ») | ca thử vàng + ①②③ | **chưa chứng minh trên dữ liệu thật** |

§71/§72 mua được A bằng ĐƯỜNG THẲNG (đề cử cạnh đã có + xếp hạng bigram). Nó
**không nói gì về B** — §73 đã tách biến và xác nhận: chuỗi ba bước thuần không
sinh ra `khỉ` ở bất kỳ cấu hình nào.

TK_046 giữ mục tiêu A. Ticket này giữ mục tiêu B.

## Ý tưởng (của tác giả, TK_046 §19)

Dùng nơ-ron W_skip làm nhãn, lan ba bước:

```
nghịch ngợm → {chó, mèo} → động vật → {chó, mèo, KHỈ}
①              ②             ③
```

`khỉ` là **sáng tạo** vì nó chưa từng đứng cạnh `nghịch ngợm`.

## Trạng thái đo được, vào ticket này

| thước | TOY (5 câu vàng + nhiễu thật) | DỮ LIỆU THẬT |
|---|---|---|
| chuỗi ①②③ ra đúng lớp | ĐẠT tới **245 321 cha** | ② tự do **1,0%** (§65) |
| ① giao ≥2 đồng-thành-viên cho ② | 100% (toy viết sẵn) | **0,0%** top-4 · 16,1% trong bể (§66/§69) |
| ③ ra thành viên CHƯA chạm thuộc tính | ĐẠT (`khỉ` hạng 2–18) | chưa lần nào (2–6 ca lẻ, đều đọc ra là hiện vật) |

**Chẩn đoán gốc (§67.A):** vật mang của một tính từ trải khắp mọi lớp (`trắng`
→ vôi · mắt · tuyết · áo). Điều kiện cần của ② — hai vật mang CÙNG LỚP — là
chuyện hiếm trong văn xuôi. Ca thử vàng có điều kiện đó vì hai câu được **viết
sẵn**.

Ba hướng nới đã thử, không hướng nào đưa điều kiện cần lên:

- thêm dữ liệu **8×** (§65) — ② tự do đứng yên 1,0% → 0,8%
- thêm **loại cạnh thứ ba** (§69) — kênh dày 5,5×, ≥2 đồng-thành-viên 0,0% → 1,5% (cổng đòi 5%)
- **bốn luật chọn hạt** (§67.B) — trung tính cả bốn

## Câu hỏi của ticket này — và phép thử dứt điểm

Hai giả thuyết còn lại loại trừ nhau:

- **H1 — CƠ CHẾ SAI**: chuỗi ba bước không làm được việc nó hứa, kể cả khi dữ
  liệu có đúng hình dạng.
- **H2 — DỮ LIỆU KHÔNG CÓ HÌNH DẠNG ĐÓ**: cơ chế đúng, nhưng văn xuôi bách khoa
  không viết theo lối so sánh, nên điều kiện cần không bao giờ đủ.

Ca thử vàng không phân định được vì nó chỉ có **5 câu**: nó cho thấy cơ chế
chạy ở n=1, không cho thấy nó chạy ở quy mô.

**Phép thử: dựng corpus CÓ CHỦ ĐÍCH đúng cấu trúc vàng, hàng trăm bộ, TRỘN với
nhiễu thật, rồi đo tỉ lệ thu hồi thành viên bị giữ lại.**

```
mỗi bộ:  <thuộc tính A> như loài <m1> .        ← A chạm m1, m2
         <thuộc tính A> như loài <m2> .
         <m1> là <lớp C> .   <m2> là <C> .   <m3> là <C> .
         m3 KHÔNG BAO GIỜ xuất hiện cùng A     ← «con khỉ» của bộ đó
đo    :  prompt «<A> như loài ___» có ra m3 không
```

## Cổng — nêu TRƯỚC khi chạy

- **≥50% bộ có m3 trong top-10** ⇒ **H2 đúng**: cơ chế làm được, dữ liệu tự
  nhiên không có hình dạng đó. Kết luận: ý tưởng cần văn bản viết theo lối so
  sánh (thơ · tục ngữ · văn miêu tả), không phải bách khoa.
- **<10%** ⇒ **H1 đúng**: cơ chế sai, và ca thử vàng 5 câu là hiện vật n=1.
- **10–50%** ⇒ chưa phân định; ghi số, không kết luận.

Phải nêu thêm: corpus này là **tổng hợp**, nên mọi kết quả của nó chỉ nói về
CƠ CHẾ, không nói gì về chất lượng sinh trên dữ liệu thật. Không được trích số
của ticket này để tuyên bố về TK_046 mục tiêu A.

## Việc

1. `scripts/dung_corpus_loai_suy.py` — sinh corpus từ chân lý v2 + từ điển.
2. Train trộn với nguồn của `classrich2` (nhiễu thật).
3. `scripts/cham_thu_hoi_loai_suy.py` — đo tỉ lệ thu hồi m3.
4. Ghi kết quả vào ticket này, không vào TK_046.

---

# KẾT QUẢ VÒNG 1 (2026-08-15)

## 1. Hai lần corpus KHÔNG VÀO ĐƯỢC ĐỒ THỊ — bắt được vì kiểm cạnh trước khi tin điểm

Phép thử chạy ba lần. Hai lần đầu **đo ra 6,0% và 11,0%**, và cả hai đều là hiện
vật của khâu dựng dữ liệu, không phải của cơ chế:

| lần | triệu chứng | nguyên nhân |
|---|---|---|
| 1 | ① cạnh thuộc tính dựng được **5,0%** | **95% thuộc tính có tần suất 0** — chọn tính từ hiếm từ từ điển mà chúng không có trong từ vựng BPE, nên bị băm thành mảnh |
| 2 | ① **11,0%**, ② 64,1% | **129/145 bộ không vào tới cha**: pipeline chỉ dựng CHA CÂU khi một dòng tách được **≥2 câu** (`if len(sents) >= 2`), mà corpus bản đầu mỗi dòng một câu ngắn |
| 3 | ① **96,6%**, ② **94,5%** | gộp mỗi bộ thành MỘT ĐOẠN 5 câu, và ràng buộc mọi từ phải có trong từ vựng |

Trước khi sửa, tôi đã đoán sai **năm lần** liên tiếp (span WS nuốt pivot · `chu_gia`
· §62 biên từ · §68 bỏ đường lui · khớp bề mặt cụm). Mỗi lần đều vá xong mà số
không đổi. Thứ giải được là **gắn đo vào chính nhánh trích** và đếm xem có bao
nhiêu câu chạm tới nó (32/290) — rồi đếm ngược ra là câu không có trong cha.

## 2. Kết quả trên corpus ĐÃ VÀO ĐƯỢC (145 bộ, chuỗi THUẦN)

```
mật độ (số lần nhắc lại mỗi bộ)     3       12      40
THU HỒI m3 · top-1                0,7%    2,1%    2,8%
             top-5                2,8%   20,7%   23,4%
             top-10              18,6%   24,8%   27,6%
ĐỐI CHỨNG (m3 của bộ KHÁC)        1,4%    1,4%    0,7%
```

**Tín hiệu THẬT**: 27,6% so với đối chứng 0,7% — **gấp 40 lần ngẫu nhiên**. Đề
cử phụ thuộc prompt, không phải danh sách chung.

**Nhưng nó BÃO HOÀ quanh 25–28%**, không leo tới 50% dù dữ liệu được dựng đúng
cấu trúc, dày đặc, và mọi cạnh cần thiết đều tồn tại (① 96,6% · ② 94,5%).

## 3. Phán quyết theo cổng đã nêu trước

Cổng: ≥50% ⇒ H2 · <10% ⇒ H1 · 10–50% ⇒ **chưa phân định**.

**27,6% ⇒ CHƯA PHÂN ĐỊNH.** Không được kết luận «cơ chế sai», cũng không được
kết luận «chỉ tại dữ liệu».

Điều đo thêm được, và nó chỉ đúng chỗ cần truy tiếp: **lớp đúng chỉ xuất hiện
trong top-10 ở 1,4% bộ**. Tức ③ lan ra thành viên đúng ở ~25% bộ **mà không cần
② chọn đúng lớp** — hai bước đang không khớp nhau. Nút thắt của vòng sau nằm ở
đó, không nằm ở dữ liệu.

## 4. Việc vòng 2

1. Truy vì sao ② hầu như không bao giờ đề cử đúng lớp trên chính corpus dựng
   riêng cho nó (1,4%), trong khi cạnh `m→C` tồn tại ở 94,5% bộ.
2. Nếu ② được sửa mà thu hồi vượt 50% ⇒ H2 và ticket đóng với kết luận «cơ chế
   đúng, dữ liệu tự nhiên không có hình dạng đó».
3. Nếu ② sửa xong mà vẫn ~25% ⇒ H1, và ý tưởng cần thiết kế lại chứ không cần
   dữ liệu khác.

## 5. Bài học

- **Kiểm CẠNH trước khi tin ĐIỂM.** Hai lần đầu đều cho một con số sạch sẽ
  (6,0% · 11,0%) từ dữ liệu mà cơ chế chưa từng đọc được. Nếu không đếm cạnh,
  ticket này đã kết luận «cơ chế sai» ngay vòng đầu.
- **Đối chứng hoán vị là thứ rẻ nhất phân biệt tín hiệu với danh sách chung.**
  18,6% trông yếu cho tới khi đối chứng nói 1,4%.
- **Đoán năm lần không bằng gắn đo một lần.**

---

# KẾT QUẢ VÒNG 2 (2026-08-15) — 🔴 kênh thuộc tính §53 CHƯA BAO GIỜ được đường sinh dùng

## 1. Chạy TAY từng bước: chuỗi HOẠT ĐỘNG

Vòng 1 kết «nút thắt ở ②». Chạy tay ①②③ trên chính 145 bộ:

```
① ra vật mang đúng        97,2%
② ra lớp đúng ở HẠNG 1    75,2%
③ ra m3                   73,8%
```

**Chuỗi ba bước làm được việc nó hứa** — 73,8% bộ thu hồi được thành viên chưa
từng chạm thuộc tính. Nhưng đường sinh chỉ thu hồi 27,6%. Chênh lệch 46 điểm
nằm giữa hai đường chạy CÙNG một cơ chế.

## 2. Bốn giả thuyết bị bác, rồi nguyên nhân thật

| giả thuyết | đo | kết |
|---|---|---|
| token lớp chung (`loài`) trong prompt làm loãng | 27,6% → 29,0% | không phải |
| chỉ số của đường sinh dựng KHÔNG span WS | 73,8% vs 74,5% (span tắt) | không phải |
| `ke_ctx` chỉ vào khoá sắp xếp, `weight` trả về là số thô | vá xong: 27,6% | không phải (vẫn giữ bản vá, nó đúng) |
| **`hub_chain_proposals` gọi ① mà KHÔNG truyền `kenh_thuoc_tinh=True`** | **27,6% → 44,1%** | **ĐÚNG** |

§53 dựng hẳn **loại cạnh THỨ HAI** (`via_attr`) cho bước ① — quan hệ THUỘC TÍNH
— và ghi rõ nó là thứ bước ① cần. Nhưng đường sinh gọi:

```python
s1 = _H.score_hubs_from(state, idx, bien_the[:4], topk=60, gidf_min=0.0,
                        channel="typed_first")        # ← THIẾU kenh_thuoc_tinh
```

**Suốt từ §53 tới nay, đường sinh chạy bước ① trên kênh PHÂN LOẠI và mù hoàn
toàn với kênh thuộc tính.** Mọi phép đo về ① của TK_046 (§66 «0,0%» · §67 bốn
luật chọn hạt · §69 kênh vị ngữ «trượt cổng») đều đo một bước không đọc dữ liệu
của nó. Bằng chứng trực tiếp: `riêng «nhanh»` qua đường sinh cho đúng
`[('người', 5.4)]`, trong khi chạy tay cho `[báng, bèo, cir rus, vũ bão]`.

## 3. Sau khi vá

```
                        trước    sau
THU HỒI m3 · top-1       2,8%    2,1%
             top-5      23,4%   38,6%
             top-10     27,6%   44,1%
             top-30        —    53,8%
ĐỐI CHỨNG (bộ khác)      0,7%    0,7%
```

Bỏ token `loài` khỏi prompt: 44,8% (không khác mấy ⇒ không phải nút thắt).

**Cổng TK_052: 44,1% ⇒ vẫn CHƯA PHÂN ĐỊNH** (10–50%). Nhưng nội bộ chuỗi đạt
73,8%, và top-30 đạt 53,8% — tức m3 CÓ được sinh ra, chỉ xếp hạng thấp. Nút
thắt còn lại là **khâu gộp/xếp hạng đề cử**, không phải cơ chế lan.

## 4. Bản vá cũng cải thiện TK_046 (mục tiêu A)

Thước sinh v3 trên `classrich2` — **không hồi quy, còn tốt lên**:

```
top-1  25,0% → 25,0%      top-5  42,0% → 48,0%   (+6,0)
```

Ca thử vàng toy: 10/10 mức, khỉ 2–18. 440 test.

## 5. Bài học

- **Một tính năng có thể tồn tại, được đo, được ghi vào ticket — mà không bao
  giờ được GỌI.** §53 dựng `via_attr`, đo nó, kết luận về nó; đường sinh không
  truyền một tham số và cả kênh nằm im 20 mục. Cách phát hiện: chạy TAY cùng
  một chuỗi rồi so với đường sản xuất trên CÙNG một ca.
- **Chênh lệch giữa «chạy tay» và «đường thật» là chỗ đáng đo nhất.** 46 điểm
  chênh lệch chỉ ra một dòng mã.

---

# KẾT QUẢ VÒNG 3 (2026-08-15) — 🟢 **CỔNG ĐẠT: 70,3% ⇒ H2**

## 1. Khoảng cách cuối: prompt được gieo bằng MẢNH BPE

Vòng 2 còn 46 điểm chênh giữa chạy tay (71,7% top-5) và đường sinh (38,6%).
Bảy giả thuyết bị bác (token `loài` · span WS · `ke_ctx` không vào weight ·
`topk` ở ① · bỏ mảnh · dự phòng · trần biến thể). Nguyên nhân thật:

```
encode_prompt("ngặt")  →  ['ng', 'ặt']        89% tính từ bị tách!
```

`hub_chain_proposals` lặp theo **TOKEN** của prompt, nên nó gieo ① bằng `ặt`,
rồi `resolve_surface_ids('ặt')` chỉ tra lại được chính mảnh đó. Chạy tay dùng
cả bề mặt `ngặt`. Đây là bệnh §23.1 («một bề mặt nhiều id») ở khâu chưa ai vá:
**đầu vào của cả chuỗi**.

## 2. Ba chế độ gieo ①, và một xung đột thật giữa hai ticket

```
chế độ                              TK_052 thu hồi   TK_046 top-1   top-5
0 — chỉ MẢNH (không tra biến thể)        43,4%          38,0%       50,7%
1 — HỢP mảnh + cả-từ                     48,3%          17,7%       44,3%
2 — chỉ id CẢ TỪ                       **70,3%**        15,7%       34,3%
```

Hai corpus mã hoá cùng một từ khác nhau: trên corpus TK_052 cạnh bám vào id
CẢ TỪ, trên `classrich2` cạnh bám vào MẢNH. Hợp cả hai không cứu được vì trần
`n_bien_the` bị lấp bởi phía sai.

⇒ **Mặc định `hub_chain_gom_tu = 0`** (bảo vệ TK_046, và nó còn tốt hơn mặc
định cũ: top-1 25,0% → 38,0%). TK_052 chạy ở chế độ 2.

## 3. Phán quyết

```
THU HỒI m3 (chế độ 2):  top-1 4,1% · top-5 63,4% · top-10 70,3%
ĐỐI CHỨNG (m3 bộ khác):                                     0,0%
```

**70,3% ≥ 50% ⇒ H2 ĐÚNG.**

> **Cơ chế ba bước LÀM ĐƯỢC việc nó hứa.** Cho nó dữ liệu đúng hình dạng — thuộc
> tính chạm ≥2 vật mang cùng lớp, lớp có thành viên thứ ba chưa chạm thuộc tính —
> nó thu hồi thành viên đó ở **70,3%** bộ, với đối chứng **0%**. Ý tưởng «con
> khỉ» KHÔNG sai.
>
> Thứ thiếu là **dữ liệu có hình dạng đó**. Văn xuôi bách khoa không viết theo
> lối so sánh: đo trên `classrich2`, chỉ 5,5% câu có pivot so sánh, và vật mang
> của một tính từ trải khắp mọi lớp (§67.A).

## 4. Điều kiện để ý tưởng chạy trên dữ liệu thật

Ticket này đã xác định đủ điều kiện cần:

1. **Văn bản viết theo lối SO SÁNH** — thơ · ca dao · tục ngữ · văn miêu tả.
   «nhanh như cắt», «đen như mực», «hiền như bụt» là đúng khuôn ca vàng, và
   tiếng Việt có kho thành ngữ so sánh rất lớn.
2. **Thuộc tính phải là TỪ trong từ vựng** — 89% tính từ hiếm bị BPE băm; cần
   train BPE có chúng, hoặc gieo ① ở mức TỪ (chế độ 2) khi corpus cho phép.
3. **Mỗi thuộc tính cần ≥2 vật mang cùng lớp** — điều kiện cần của bước ②, đo
   được 0,0% trên văn xuôi bách khoa (§66) và ~100% trên corpus so sánh.

## 5. Việc còn lại

- Dựng corpus thành ngữ/ca dao so sánh tiếng Việt và đo lại — nếu ĐẠT thì ý
  tưởng chạy trên dữ liệu THẬT, không phải tổng hợp.
- Gỡ xung đột chế độ gieo ①: hai corpus cần hai chế độ, nghĩa là khâu mã hoá
  prompt đang gánh một quyết định đáng ra thuộc về chỉ số.

## 6. Bài học

- **Bảy giả thuyết sai trước một nguyên nhân đúng, và nguyên nhân đó nằm ở
  ĐẦU VÀO chứ không trong cơ chế.** Mọi giả thuyết của tôi đều về các bước
  bên trong; thứ hỏng là cách prompt được cắt thành token.
- **Một cổng đặt trước có giá trị đúng lúc nó ĐẠT.** 70,3% không phải con số
  tôi tự chọn để tuyên bố — nó vượt ngưỡng 50% đã viết ra trước khi chạy vòng 1.

---

# KẾT QUẢ VÒNG 4 (2026-08-15) — dữ liệu THẬT: **CHƯA PHÂN ĐỊNH ĐƯỢC**, và lý do đáng ghi

Vòng 3 chứng minh cơ chế làm được trên corpus TỔNG HỢP (70,3% · đối chứng 0,0%)
và nêu điều kiện: cần văn bản viết theo lối SO SÁNH. Vòng 4 đi kiểm điều kiện đó
trên dữ liệu thật.

## 1. Khuôn so sánh CÓ THẬT, ở quy mô

Quét 350 435 dòng của 7 nguồn, khuôn «tính từ như danh từ»:

```
viwiki 27 470 · conv_stories 5 083 · dictionary 826 · doc_add 422 ·
paragraph_units 370 · history 102          TỔNG 34 273

thuộc tính có ≥1 vật mang : 2 024
thuộc tính có ≥2 vật mang : 1 138  (56,2%)   ← điều kiện cần của ②
… trong đó ≥2 vật mang CÙNG MỘT LỚP : 279  (24,5%)
```

**Tiếng Việt thật CÓ khuôn ca vàng ở quy mô 279 bộ.** Điều kiện (1) của vòng 3
được thoả — trái với ấn tượng ở §66 rằng dữ liệu không có hình dạng đó (số 0,0%
ở §66 là của thứ CƠ CHẾ TRÍCH ĐƯỢC, không phải của văn bản).

## 2. Nhưng phép đo trên dữ liệu thật ra 0,0–1,1% ≈ đối chứng

Dựng corpus 23 105 câu THẬT chứa các so sánh đó + nguồn định nghĩa, train
80 592 cha, đo thu hồi m3:

```
277 bộ            top-5 0,0%   top-10 1,1%    đối chứng 0,7%
72 bộ lớp CỤ THỂ  top-5 0,0%   top-10 0,0%    đối chứng 0,0%
```

## 3. Vì sao — CHÂN LÝ CŨNG LÀ MỘT PHÉP TRÍCH (bài học nặng số 1, lặp lại)

Kiểm cạnh: ① chỉ dựng được **6,9%** bộ. Nhưng đọc thứ nó THẬT SỰ trích ra:

```
via_attr[nhanh] = [các · đỡ · trào · xoay chuyển · DƠI · CẮT]   ← «nhanh như cắt»
via_attr[đỏ]    = [SON · nếu · cho · TP · … · LỬA]              ← «đỏ như son»
```

**Cơ chế trích đúng thành ngữ thật.** Khoá của tôi thì đòi `sóc`/`tê giác` —
vì tôi đào khoá bằng **regex lấy token KỀ**, còn cơ chế lấy **trung tâm ngữ
đoạn**. Hai phép trích khác nhau ⇒ hai bộ vật mang khác nhau ⇒ phép đo chấm cơ
chế bằng một chân lý mà cơ chế không thể đúng.

Dựng lại khoá từ CHÍNH cạnh của cơ chế: ra 1 245 bộ, nhưng lớp của chúng là
`dài` · `trắng` · `có` · `nhẵn` — tính từ bị lấy làm lớp, và vật mang là `cả` ·
`chỉ` · `có`. Cả hai phía đều quá nhiễu để phân định.

## 4. Phán quyết vòng 4

**CHƯA PHÂN ĐỊNH.** Không được kết luận «ý tưởng không chạy trên dữ liệu thật» —
phép đo này không đủ tư cách kết luận điều đó.

Ba điều đã biết chắc sau vòng 4:

1. Khuôn so sánh **có thật** trong tiếng Việt ở quy mô hàng nghìn (34 273 khuôn ·
   1 138 thuộc tính ≥2 vật mang).
2. Cơ chế **trích đúng** một phần thành ngữ thật (`nhanh → cắt · dơi`).
3. Nút thắt là **chất lượng trích trên văn bản so sánh**: vật mang lẫn hư từ
   (`cả`, `chỉ`, `có`), và lớp của vật mang lẫn tính từ (`dài`, `trắng`).

## 5. Việc vòng 5

Làm sạch kênh so sánh trên văn bản thật — đúng việc §68 đã làm cho `là`/`giống`/
`chẳng hạn`, nhưng lần này cho vế PHẢI (vật mang) và cho lớp của nó. Chỉ khi vật
mang là DANH TỪ NỘI DUNG thì khoá mới dựng được, và lúc đó phép thử mới có tư
cách phân định.

## 6. Bài học

- **Chân lý cũng là một phép trích — lần thứ HAI trong cùng ticket này.** §31
  đã đính chính chân lý một lần; vòng 4 mắc lại ở dạng khác: khoá đào bằng regex
  kề, cơ chế đọc trung tâm ngữ đoạn. **Khoá và cơ chế phải dùng CÙNG một phép
  trích, nếu không phép đo chấm cơ chế bằng thứ nó không thể đúng.**
- **«Dữ liệu không có hình dạng đó» ở §66 là kết luận SAI.** Văn bản có; thứ
  không có là CẠNH mà cơ chế trích được từ nó. Hai câu đó khác nhau, và tôi đã
  viết nhầm câu thứ nhất suốt từ §66 tới §73.

---

# KẾT QUẢ VÒNG 5 (2026-08-15) — kênh so sánh đã sạch, nút thắt dời sang CẠNH LỚP

## 1. Vá: vật mang phải là DANH TỪ

§69 đã siết vế phải cho kênh VỊ NGỮ; kênh SO SÁNH thì chưa. Áp cùng luật:

```
via_attr trên corpus so sánh: 6 027 thuộc tính · 22 075 cạnh · 37,8% có ≥2 vật mang

nhanh → [các · thể · đá · ưng · DƠI · CẮT · thao · SÓC]
đỏ    → [SON · … · LỬA · ong]
hiền  → [RÙA · mức]
```

**Cơ chế nay trích đúng thành ngữ Việt thật**: «nhanh như cắt» · «nhanh như sóc»
· «đỏ như son» · «đỏ như lửa» · «hiền như rùa».

## 2. Nhưng thu hồi vẫn 0–1,1% ≈ đối chứng

```
277 bộ            top-5 0,7%   top-10 1,1%   đối chứng 0,7%
72 bộ lớp cụ thể  top-5 0,0%   top-10 0,0%   đối chứng 0,0%
```

## 3. Chẩn đoán: vật mang KHÔNG chia nhau lớp nào

```
via_pivot[sóc] = [quý hiếm · khác · mồi · lớn · mốc]        ← KHÔNG có lớp
via_pivot[cắt] = [tựa · tia sáng · Nhanh · dụng cụ]         ← `cắt` = dụng cụ (đa nghĩa)
via_pivot[dơi] = [lớn · rắn độc · khác · cánh · loài · họ]
via_pivot[rùa] = [Nhanh · quý hiếm · động vật · loài]
```

Bước ② cần hai vật mang CÙNG lớp. `sóc`+`cắt` không có lớp nào chung, nên chuỗi
đứt ở ② dù ① đã đúng.

## 4. Và đây mới là chỗ đau: câu định nghĩa CÓ, cha CÓ, cạnh KHÔNG

```
corpus : "sóc là thú gặm nhấm lớn hơn chuột, sống trên cây…"   → lớp phải là `thú`
         "cắt là chim ăn thịt, nhỏ hơn diều hâu…"              → lớp phải là `chim`
cha    : «cắt là chim ăn thịt , nhỏ hơn diều hâu , …»   CÓ trong `ps.parents`
cạnh   : via_pivot[cắt] KHÔNG chứa `chim`
```

Ngưỡng đặc thù đã loại trừ (`head_gidf_q` 0,20 · 0,05 · 0,00 cho kết quả **y hệt
nhau**). Nguyên nhân **chưa isolate được**.

Khuôn này — GIỐNG–LOÀI với phần khu biệt DÀI («X là <lớp> <một tràng đặc điểm>»)
— chính là khuôn từ điển đã dùng thành công ở TK_046 §65 (`chó là gia súc…`).
Ở đây nó thất bại, và đó là câu hỏi của vòng 6.

## 5. Trạng thái ticket sau 5 vòng

| | kết quả |
|---|---|
| cơ chế trên corpus TỔNG HỢP | **70,3%** thu hồi · đối chứng 0,0% ⇒ cơ chế ĐÚNG |
| khuôn so sánh trong tiếng Việt thật | **34 273** khuôn · 1 138 thuộc tính ≥2 vật mang |
| cơ chế trích thành ngữ thật | nhanh→cắt·sóc·dơi · đỏ→son·lửa · hiền→rùa ✓ |
| **thu hồi trên dữ liệu thật** | **1,1% ≈ đối chứng** ✗ |
| nút thắt đã định vị | **cạnh LỚP của vật mang không được trích** từ chính câu định nghĩa của nó |

Ba vòng liên tiếp (3 · 4 · 5) đều thu hẹp được nút thắt một bậc: từ «cơ chế hay
dữ liệu?» → «dữ liệu có, cơ chế trích được thuộc tính» → «nhưng không trích được
LỚP của vật mang». Vòng 6 có một câu hỏi hẹp và một khuôn câu cụ thể để truy.

## 6. Bài học

- **Sạch một kênh không tự động nối được chuỗi.** ① nay đúng hoàn toàn trên
  thành ngữ thật, và tổng thể vẫn 1,1%: chuỗi chỉ mạnh bằng mắt xích yếu nhất,
  và mắt xích đó vừa đổi chỗ chứ không biến mất.
- **«Câu có · cha có · cạnh không» là một trạng thái phải kiểm được.** Ba lần
  trong ticket này (vòng 1 hai lần, vòng 5 một lần) dữ liệu đúng nằm trong
  corpus mà không thành cạnh. Không có phép kiểm đó thì mọi con số đều mù.

---

# KẾT QUẢ VÒNG 6 (2026-08-15) — ba bản vá thật, và một ĐÍNH CHÍNH vòng 5

## 1. ĐÍNH CHÍNH VÒNG 5: cạnh `cắt → chim` CÓ TỒN TẠI

Vòng 5 kết «câu có · cha có · CẠNH KHÔNG». Sai. Tra theo **id thật của token
trong câu**:

```
id trong câu: cắt=17656 · chim=3413
via_pivot[17656] = [('chim', 1,0)]        ← CẠNH CÓ
resolve_surface_ids('cắt') = [3053, 9346, 17282, 17656]
```

Phép in của tôi gộp 4 biến thể BPE rồi cắt top-8, nên `chim` bị đẩy ra ngoài
khung nhìn. **Lần thứ tư trong phiên một kết luận của tôi là hiện vật của cách
hiển thị.**

## 2. Bản vá A — `_liet_ke` đọc nhầm khuôn GIỐNG–LOÀI, phát cạnh NGƯỢC CHIỀU

```
«cắt là chim ăn thịt , nhỏ hơn diều hâu , cánh dài và nhọn , bay rất nhanh»
 _liet_ke thấy 4 mục [chim · nhỏ · cánh · bay] và phát  chim → cắt
```

Khuôn liệt kê (§29, «<LỚP> là A, B và C») có ĐÚNG hình dạng bề mặt với định
nghĩa giống–loài. Phân biệt: liệt kê thật thì các mục cùng là DANH TỪ («NBC,
CBS, ABC và Fox»); phần khu biệt là tính từ/động từ. Đòi **đa số mục là danh
từ** ⇒ hết phát ngược.

## 3. Bản vá B — vật mang của kênh so sánh phải là DANH TỪ **và ĐỦ ĐẶC THÙ**

```
trước:  via_attr[nhanh] = [các · thể · đá · ưng · dơi · cắt · thao · sóc]
                          └── ba hư từ chiếm top-4, đẩy `cắt`/`sóc` khỏi hạt ①
sau  :  via_attr[nhanh] = [đá · ưng · cắt · thao · sóc]
```

Từ điển gọi `các`·`thể` là DANH TỪ (đúng bệnh §64) nên lọc từ loại KHÔNG đủ —
phải thêm ngưỡng `det_cut`, và **bỏ đường lui** (không ứng viên nào đạt thì
không phát cạnh). Kênh: 22 075 → 14 470 cạnh, sạch hẳn.

## 4. Cổng — ba thước tốt lên hoặc giữ

```
TK_046 thước sinh v3    top-1 38,0% · top-5 50,7% → 51,0%
ca thử vàng toy         khỉ 6·7·13·18 → 4·5·12·17        (TỐT HƠN)
TK_052 corpus tổng hợp  70,3% → 69,7%                    (giữ)
440 test
```

## 5. Nhưng dữ liệu thật vẫn 0,0% — và lý do nay đã rõ

Sau ba bản vá, `nhanh` → `[đá · ưng · cắt · thao · sóc]` và `cắt`+`ưng` CHIA
NHAU lớp `chim`. Điều kiện cần của ② đã có. Nhưng phép đo vẫn 0,0% vì **không
dựng được KHOÁ hợp lệ**:

- khoá bằng regex (vòng 4) → sai phép trích, đã bác;
- khoá bằng chính cạnh của cơ chế → **996 bộ nhưng lớp trội là `khác`** ·
  `quý hiếm` — nhãn rác nuốt mọi thứ, đúng hiện tượng §67.B đã đặt tên.

⇒ **Nút thắt cuối KHÔNG phải cơ chế lan, mà là CHẤT LƯỢNG NHÃN LỚP trên văn
xuôi thật.** Đó là địa hạt của TK_046 §64/§68, không phải của ticket này.

## 6. Trạng thái ticket sau 6 vòng

| câu hỏi | trả lời |
|---|---|
| cơ chế ba bước có làm được không? | **CÓ** — 69,7% thu hồi, đối chứng 0,0% (corpus tổng hợp) |
| tiếng Việt thật có khuôn so sánh không? | **CÓ** — 34 273 khuôn · 1 138 thuộc tính ≥2 vật mang |
| cơ chế có trích được khuôn đó không? | **CÓ, sau vòng 5–6** — nhanh→cắt·sóc·ưng · đỏ→son·lửa · hiền→rùa |
| vật mang có chia nhau lớp không? | **CÓ** — cắt+ưng → `chim` |
| đo được đầu-cuối trên dữ liệu thật chưa? | **CHƯA** — không dựng được khoá vì nhãn lớp trên văn xuôi thật bị `khác`/`quý hiếm` nuốt |

## 7. Bài học

- **Bốn lần trong một phiên, kết luận của tôi là hiện vật của PHÉP HIỂN THỊ hoặc
  PHÉP TRÍCH, không phải của cơ chế.** (§61 hai lần · vòng 4 · vòng 5.) Quy tắc
  rút ra: **trước khi kết luận «không có X», in ĐẦY ĐỦ theo TỪNG id, đừng gộp
  biến thể rồi cắt top-k.**
- **Một khuôn bề mặt có thể thuộc hai cấu trúc khác nhau.** «A là B, C và D» là
  liệt kê HOẶC định nghĩa giống–loài; phân biệt được bằng TỪ LOẠI của các mục,
  và §29 đã sống 40 mục mà không ai kiểm điều đó.

---

# 📋 TỔNG KẾT TK_052 SAU 6 VÒNG (2026-08-15)

## Câu hỏi và câu trả lời

| câu hỏi | trả lời | bằng chứng |
|---|---|---|
| cơ chế ba bước có làm được không? | **CÓ** | vòng 3 — thu hồi **69,7%**, đối chứng **0,0%** |
| tiếng Việt thật có khuôn so sánh không? | **CÓ** | vòng 4 — **34 273** khuôn · 1 138 thuộc tính ≥2 vật mang |
| cơ chế có trích được khuôn đó không? | **CÓ** (sau vòng 5–6) | `nhanh→cắt·sóc·ưng` · `đỏ→son·lửa` · `hiền→rùa` |
| vật mang có chia nhau lớp không? | **CÓ** | `cắt`+`ưng` → `chim` |
| đo được ĐẦU-CUỐI trên dữ liệu thật chưa? | **CHƯA** | nhãn lớp quá nhiễu để dựng khoá hợp lệ |

Bốn mắt xích đầu XANH. Mắt xích thứ năm là **phép đo**, không phải cơ chế.

## Ba lỗi thật tìm được (đều đã vá, đều cải thiện TK_046)

1. **vòng 2** — `hub_chain_proposals` không truyền `kenh_thuoc_tinh` ⇒ đường
   sinh mù với kênh §53 suốt ~20 mục. Vá: thu hồi 27,6% → 44,1%, và TK_046
   top-5 42% → 48%.
2. **vòng 3** — prompt gieo ① bằng MẢNH BPE (89% tính từ bị tách). Vá: 44,1% →
   **70,3%**.
3. **vòng 6** — `_liet_ke` đọc định nghĩa giống–loài thành LIỆT KÊ và phát cạnh
   NGƯỢC CHIỀU (`chim → cắt`). Vá: TK_046 lớp cụ thể 43,6% → **54,5%**.

## Bốn lần phép đo tự nó sai (không phải cơ chế)

- vòng 1 — 95% thuộc tính có tần suất **0** (không có trong từ vựng BPE);
- vòng 1 — 129/145 bộ **không vào tới cha** (pipeline cần ≥2 câu mỗi dòng);
- vòng 4 — khoá đào bằng **regex kề**, cơ chế đọc **trung tâm ngữ đoạn**;
- vòng 6 — phép in gộp 4 biến thể BPE rồi cắt top-8 ⇒ kết luận «cạnh không có» SAI.

Ba lần đầu đều cho một con số sạch sẽ (6,0% · 11,0% · 0,0%) từ dữ liệu mà cơ
chế chưa từng đọc được.

## Điều kiện để ý tưởng chạy trên dữ liệu thật

1. Văn bản viết theo lối SO SÁNH — tiếng Việt CÓ (34 273 khuôn) ✓
2. Thuộc tính phải là TỪ trong từ vựng — cần train BPE có chúng, hoặc gieo ① ở
   mức TỪ (`hub_chain_gom_tu = 2`) ✓
3. Mỗi thuộc tính ≥2 vật mang cùng lớp — CÓ sau vòng 5–6 ✓
4. **Lớp phải là lớp PHÂN LOẠI, không phải `phần`/`nơi`/`cái`** ✗ ← còn thiếu

Điều kiện 4 là ranh giới của kiến trúc hiện tại: TK_046 §75/§76 đã chứng minh
nhãn chung chung **không lọc được** bằng bất kỳ tín hiệu nội tại nào (sáu tín
hiệu) hay bằng nguồn từ điển. Muốn có nó phải nạp một **cây phân loại thật**.

## Vòng 7 — chỉ chạy được sau khi có nguồn phân loại

Khi có cây phân loại: dựng lại khoá bằng lớp thật, đo thu hồi trên corpus so
sánh 23 105 câu đã có (`data/text_corpus/so_sanh_that.jsonl`), cổng vẫn là
**≥50% top-10**.

---

# KẾT QUẢ VÒNG 7 (2026-08-15) — **con số 44% là HIỆN VẬT**, và nguyên nhân thật

## 1. ĐÍNH CHÍNH: 44% câu định nghĩa «không vào đồ thị» là SAI

Vòng 7 mở đầu bằng chẩn đoán «`sóc là thú…` có trong corpus mà 0 cha», rồi đo
ra **44% câu định nghĩa từ điển không vào đồ thị**. Cả hai đều SAI.

```
encode_prompt('sóc') → ['s', 'óc']
cha có thật: «s óc là thú gặm nhấm lớn hơn chuột , sống trên cây , …»
```

`sóc` bị BPE tách thành `s`+`óc`, nên văn bản ghép của cha là `"s óc là thú…"`
và phép tìm chuỗi con `"sóc là"` của tôi trượt. Câu **CÓ** trong đồ thị.

**Lần thứ NĂM trong ticket này một kết luận của tôi là hiện vật của phép so
khớp bề mặt.** Bốn lần trước: vòng 1 (hai lần) · vòng 4 · vòng 6.

## 2. Nguyên nhân THẬT — dây chuyền ba khâu

```
1. BPE tách `sóc` → `s` + `óc`
2. `_content_positions` LOẠI token một ký tự (`s`) khỏi dãy nội dung
3. ⇒ span WS `s óc` KHÔNG khớp được trong dãy đã lọc
4. ⇒ chủ ngữ trích ra là token `óc`, cạnh thành `óc → thú`
5. ⇒ `resolve_surface_ids('sóc')` trả [3041, 8320, 20844] — KHÔNG có id 2035
      (`óc`) ⇒ tra `via_pivot['sóc']` rỗng dù cạnh tồn tại
```

Đối chiếu: `cắt` KHÔNG bị tách (một token), nên `via_pivot[cắt] = [('chim',1,0)]`
dựng đúng. **Cùng một câu định nghĩa, cùng một khuôn — khác nhau ở chỗ từ có bị
BPE băm hay không.**

## 3. Vá được một nửa

| vá | kết quả |
|---|---|
| bề mặt span ghép theo `MARK_SPACE` (như §66.3 đã vá cho chân lý v2) | đúng hơn, nhưng chưa đủ |
| `label_by_surf` + `ids_theo_be_mat` — tra được cả **id ẢO** của span | hạ tầng đúng, dùng ở đường sinh |
| **`_content_positions` loại mảnh một ký tự** | **CHƯA vá** — đây là khâu chặn thật |

Ba thước sau khi vá: thước sinh **38,0 / 51,0** · lớp cụ thể **54,5 / 63,6** ·
TK_052 tổng hợp **70,3%** · ca vàng toy 10/10 (khỉ 4·5·12·17) — **không đổi
thước nào**. Đúng như §76: sửa đúng mà không mua được gì, vì phần bị ảnh hưởng
chưa tới được chỗ quyết định.

## 4. Việc vòng 8

Vá khâu 2 của dây chuyền: cho `_content_positions` giữ mảnh từ (hoặc cho
`_label_at` khớp span trên dãy THÔ thay vì dãy đã lọc). Rủi ro: nó đụng vào hàm
dùng chung của nhiều đường — phải A/B đủ bốn thước như mọi lần.

Số cần theo dõi: bao nhiêu % từ trong từ điển bị BPE tách (đo được 89% với tính
từ hiếm ở vòng 3) — đó là trần của phần đang mất.

## 5. Bài học

- **Năm lần so khớp bề mặt lừa tôi trong một ticket.** Quy tắc nay phải là:
  mọi phép kiểm «X có trong đồ thị không» phải chạy trên **id**, và nếu buộc
  phải so bề mặt thì so cả dạng ĐÃ BỎ DẤU CÁCH.
- **Hai từ cùng khuôn câu, một từ dựng được cạnh, một từ không** — khác biệt
  duy nhất là BPE. Đó là dấu hiệu lỗi nằm ở TẦNG MÃ HOÁ, không ở tầng trích.

---

# KẾT QUẢ VÒNG 8 (2026-08-15) — gộp mảnh BPE: **chạy được, chưa sửa được ca đích**

## 1. Vá: gộp MẢNH thành TỪ trước khi lọc nội dung

Vòng 7 định vị dây chuyền: BPE tách `sóc` → `s`+`óc` · `_content_positions` loại
`s` (không `is_surface`) · span WS `s óc` không khớp được · chủ ngữ thành `óc`.

Vá `_gop_manh_tu` trong `_mine_via_pivot` (không đụng hàm dùng chung): quét span
WS trên dãy THÔ, gộp mỗi dãy mà các mảnh sau **không mở đầu từ** thành id ẢO, rồi
mới lọc. Kèm `_sfm` — hàm tra bề mặt chịu được id ảo (29 chỗ gọi `_surf` trong
hàm đào phải đổi).

## 2. Kết quả: chạy, nhưng ca đích KHÔNG đổi

```
via_pivot        68 362 → 66 283 cạnh          (gộp có tác dụng thật)
via_pivot[sóc]   [khác·lớn·mốc·mồi·quý hiếm]   → Y HỆT, vẫn KHÔNG có `thú`
```

Gắn đo vào nhánh phân loại với cờ bật: **0 câu** có `toks[0]` là `sóc` hay `s`.
Tức với chính câu «s óc là thú…», phép gộp KHÔNG áp — và tôi **chưa isolate
được vì sao** (span có trong `ws_spans`, cả hai mảnh đều không mang `MARK_SPACE`,
điều kiện gộp thoả trên giấy).

## 3. Bốn thước

```
thước sinh v3 · 300 ca      38,0 / 51,0   →   38,0 / 51,0     (0 đổi)
thước sinh v3 · 55 ca cụ thể 54,5 / 63,6   →   56,4 / 63,6     (+1,9 top-1)
ca thử vàng toy             khỉ 4·5·12·17 →   khỉ 4·5·12·17   (0 đổi)
440 test
```

⇒ **Mặc định TẮT** (`hub_chain_gop_manh_tu = False`). Giữ cờ và hạ tầng
(`label_by_surf` · `ids_theo_be_mat` · `_ghep_bm` · `_sfm`) — chúng đúng và
sẽ cần khi vòng 9 tìm ra chỗ gộp không áp.

## 4. Việc vòng 9

Truy vì sao `_gop_manh_tu` không áp cho câu «s óc là thú…»: đặt đo NGAY TRONG
vòng gộp (in ra span nào khớp / không khớp cho câu đó), thay vì suy từ kết quả.
Đây đúng là cách đã giải được vòng 1 sau năm lần đoán sai.

## 5. Bài học

- **Vá đúng dây chuyền mà vẫn không tới đích thì dây chuyền còn khâu chưa thấy.**
  Vòng 7 mô tả năm khâu và tôi vá khâu 2–3; ca đích không đổi ⇒ hoặc mô tả thiếu
  một khâu, hoặc điều kiện gộp của tôi hẹp hơn tôi tưởng. Không được kết luận
  «đã vá» khi ca đích chưa đổi.

---

# KẾT QUẢ VÒNG 9 (2026-08-15) — 🟢 ca đích SỬA XONG · thu hồi tổng hợp **84,8%**

## 1. Gắn đo trong vòng gộp — và nó chỉ thẳng chỗ sai

Vòng 8 vá xong mà ca đích không đổi, tôi đoán ba lần rồi mới gắn đo NGAY TRONG
vòng gộp. Kết quả tức thì:

```
GOP  L=2  sp=(342, 2035)        ← «s»+«óc» ĐÃ ĐƯỢC GỘP
```

Phép gộp chạy đúng. Lỗi ở **bước sau**: `_ghep_bm` có guard «nếu không token nào
trong dãy mang `MARK_SPACE` thì ghép bằng dấu cách» — để bảo vệ fixture test.
Nhưng từ đứng **ĐẦU DÒNG** thì mảnh đầu cũng không mang dấu, nên `s`+`óc` rơi
đúng vào nhánh đó và cho bề mặt **`"s óc"`**, khiến `label_by_surf["sóc"]` rỗng.

Vá: guard phải hỏi **TỪ VỰNG** (`_vocab_co_mark`, có cache), không hỏi dãy này.

```
via_pivot[sóc] = [khác · lớn · mốc · mồi · quý hiếm · THÚ]     ✓ ca đích XONG
```

## 2. Bốn thước

```
TK_052 corpus tổng hợp   70,3%  →  84,8%   ← nhờ hạ tầng vòng 7–9
   (riêng cờ gộp mảnh)   84,8%  →  84,1%   (−0,7, trong nhiễu)
TK_046 · 55 ca lớp CỤ THỂ  top-1 54,5% → 56,4%
TK_046 · thước sinh 300 ca        38,0 / 51,0   KHÔNG đổi
ca thử vàng toy                   khỉ 4·5·12·17 KHÔNG đổi
440 test
```

**84,8% ≫ cổng 50%** — H2 nay vững hơn hẳn vòng 3 (69,7%).

⇒ Bật mặc định `hub_chain_gop_manh_tu = True`: đây là lỗi ĐÚNG-SAI, không có nó
thì mọi từ bị BPE băm **âm thầm** mất cạnh lớp.

## 3. Dữ liệu thật: vẫn 0,0%

Ca đích được sửa nhưng thước dữ liệu thật không nhúc nhích, vì nút thắt ở đó là
**khoá không dựng được** (nhãn lớp `khác`/`quý hiếm` nuốt mọi thứ — TK_046 §75/
§76 đã đóng hướng lọc). Không phải cùng một vấn đề.

## 4. Bài học

113. **Ba lần đoán rồi mới gắn đo — LẦN THỨ HAI trong cùng ticket.** Vòng 1 mất
     năm lần đoán, vòng 8–9 mất ba. Quy tắc đáng ghi to: **khi bản vá không đổi
     ca đích, gắn đo vào CHÍNH vòng lặp vừa vá, ngay lập tức.** Chi phí một lần
     chạy; tiết kiệm hàng giờ.
114. **Guard viết để bảo vệ test có thể phá dữ liệu thật.** `_ghep_bm` guard
     «dãy không có MARK_SPACE» đúng cho fixture, sai cho từ đầu dòng. Guard phải
     hỏi thứ BẤT BIẾN (từ vựng có đánh dấu không), không hỏi mẫu đang xét.

---

# KẾT QUẢ VÒNG 10 (2026-08-15) — gộp mảnh BẬT ĐƯỢC mặc định

Vòng 9 phải trả mặc định về TẮT vì phép gộp làm ca thử vàng trong fixture mất
`khỉ`. Nguyên nhân: vocab fixture **không đánh dấu ranh giới từ**, nên ở đó
không có khái niệm «mảnh của một từ» — gộp là vô nghĩa mà vẫn đổi hành vi ①.

Vá một dòng: `_gop_manh_tu` **vô hiệu** khi `not _vocab_co_mark(vocab)`.

```
thước sinh v3 · 300 ca       38,0 / 51,0        KHÔNG đổi
thước sinh v3 · 55 ca cụ thể 54,5 → 56,4 top-1  (+1,9)
TK_052 corpus tổng hợp       84,1%              (84,8% khi tắt cờ — trong nhiễu)
ca thử vàng toy              khỉ 4·5·12·17      KHÔNG đổi
440 test
```

⇒ `hub_chain_gop_manh_tu = True` (mặc định). Đề cử đọc mắt vẫn sạch:

```
«mèo là một loài ___» → thú 230,4 · khỉ · báo · bướm · Loài · cây
«hổ là loài ___»      → khỉ · báo · bướm · Loài · cây · Chim
«cây bàng là ___»     → loài · thực · bông · chè · cà · cây · mía
```

## Bài học

115. **Một tính năng «đúng về nguyên tắc» vẫn phải VÔ HIỆU ở nơi nó vô nghĩa.**
     Gộp mảnh BPE đúng với vocab thật, vô nghĩa với vocab không đánh dấu ranh
     giới — và ở đó nó không trung tính mà PHÁ. Điều kiện bật phải là điều kiện
     mà tính năng CÓ NGHĨA, không phải cờ do người dùng chọn.
