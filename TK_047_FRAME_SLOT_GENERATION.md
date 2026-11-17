# TK_047 — SINH THEO KHUNG–KHE: kiến trúc nơ-ron cho đường sinh

> Mở 2026-08-05, theo chỉ đạo của tác giả: *"sửa cái đáng chạy mà đang hỏng (sinh, 6%)… có thể
> sinh các nơ-ron nhãn định nghĩa trước, nơ-ron ngữ pháp, hoặc bất kỳ nơ-ron nào cần thiết cho
> việc sinh tốt hơn, sáng tạo và không chép agenda… nhưng cần có kiến trúc nơ-ron rõ ràng đã."*

## 0. Số học buộc phải đổi kiến trúc, không phải tinh chỉnh

§13.41 đo độ tin cậy **mỗi bước** ≈ **90%**, và **phẳng theo vị trí** ⇒ đó là tính chất của
**cơ chế**, không phải của dữ liệu hay của một linh kiện.

| độ tin cậy/bước | câu 15 bước |
|---:|---:|
| 90% | 20,6% |
| 95,5% | 50% |
| **98,5%** | **80%** |

Đo thật: AR **5,8%**, mệnh đề **7,0%** (§11 — hai đường **không tách nhau**). ⇒ Không có phép
tinh chỉnh cục bộ nào đưa 90% → 98,5%. **Sinh từng-token, tham lam, trái-sang-phải, không hồi
tố là kiến trúc không tới đích**, kể cả khi mọi linh kiện đều tốt.

Đường **chép** đạt **54,4%** không vì nó biết nhiều hơn, mà vì nó phát ra **một cấu trúc hoàn
chỉnh** thay vì **một token**. TK_045 đã thử đơn vị to hơn (mệnh đề) → vẫn **7,0%**, vì luật
ghép **vẫn cục bộ, tham lam, không hồi tố**.

⇒ Hỏng **không** nằm ở kích thước đơn vị. Nó nằm ở **thủ tục quyết định**.

## 1. Nguyên tắc kiến trúc

| | hiện tại | TK_047 |
|---|---|---|
| đơn vị quyết định | một token | **một khung câu** |
| thứ tự | trái → phải | **theo độ chắc chắn**, khe chắc nhất trước |
| ràng buộc | cục bộ (`W_seq`/`W_skip`) | **toàn cục** (khung + chủ đề) trước, cục bộ sau |
| hồi tố | không | khe điền sau **thắt** khe điền trước |
| nguồn của câu | một chỗ (chép) hoặc từng token (sinh) | **khung một chỗ, chất liệu nhiều chỗ** |

Vế cuối là điểm mấu chốt: câu `con khỉ là động vật` **sinh ra được** khi khung `[DT] là [DT]`
có trong corpus và `khỉ`/`động vật` học được từ **những cha khác**. Đó **đúng là ý tưởng loại
suy của tác giả**, phát biểu ở dạng **sinh** thay vì ở dạng viết-cạnh. Và nó **không phải chép**:
không cha nào chứa câu đó.

## 2. Bốn tầng nơ-ron

### Tầng N — nơ-ron NHÃN *(đã có, §14)*
8 nơ-ron từ loại, 7.113 cạnh `LEXATTR`, gán ở mức **TỪ** qua WS-CRF. Luật 0 sạch: nhãn nằm ở
**ingest**, không tra bảng lúc sinh.

### Tầng K — nơ-ron KHUNG CÂU  *(mới)*
Trừu tượng hoá một cha thành **chuỗi nhãn** của nó: `con chó màu đỏ` → `DT DT TT TT`. Chuỗi nào
lặp lại ≥2 lần trong corpus thì thành **một nơ-ron khung**. Thuần đếm ⇒ Luật 0 sạch.

Đây chính là **"nơ-ron ngữ pháp"** tác giả nêu — nhưng **học được**, không phải luật viết tay.
Không có quy tắc cú pháp nào được cắm vào; khung nào corpus dùng thì khung đó tồn tại.

### Tầng S — nơ-ron KHE  *(mới)*
Mỗi khung có các khe theo thứ tự. Cạnh `khe → token` là **đếm Hebbian**: token `t` đã điền khe
`s` của khung `k` bao nhiêu lần. Thêm cạnh **khe ↔ khe** trong cùng khung: hai khe cùng được
điền bởi cặp token nào (đây là chỗ ràng buộc **hồi tố** sống).

### Tầng C — nơ-ron CHỦ ĐỀ  *(một phần đã có)*
`content_folds` + `CoverageGate` đã có. Vai trò mới: ràng buộc **mọi khe cùng lúc**, không chỉ
khe kế tiếp. §13.41 đo hỏng **phẳng theo vị trí** ⇒ lỗi là **lệch chủ đề toàn cục**, đúng thứ
ràng buộc cục bộ không thấy.

## 3. Thủ tục sinh (kiến trúc, chưa phải ma trận)

```
1. đề  →  kích hoạt nơ-ron CHỦ ĐỀ (content folds của đề)
       →  kích hoạt nơ-ron KHUNG tương hợp với chuỗi nhãn của đề

2. CHỌN KHUNG   softmax trên hoạt độ khung
   ⇒ cam kết TOÀN CỤC, trước khi phát bất kỳ token nào
   ⇒ hình dạng câu bảo đảm là hình dạng CÓ THẬT trong corpus

3. ĐIỀN KHE — KHÔNG trái-sang-phải:
   lặp cho tới khi đầy:
     a. với mỗi khe trống, dựng phân bố ứng viên
        = (khe→token)  ×  (chủ đề)  ×  (khe↔khe với các khe ĐÃ điền)
     b. chọn khe có phân bố SẮC NHẤT (entropy thấp nhất)   ← quyết định chắc nhất trước
     c. điền nó (softmax), rồi truyền ràng buộc sang mọi khe còn trống

4. HIỆN THỰC HOÁ  khe → token bề mặt, dùng ranh giới TỪ (WS-CRF) để không vỡ từ
```

**Vì sao thủ tục này có thể phá tường 90%:**

| lỗi đã đo | cách kiến trúc này chặn |
|---|---|
| không biết trạng thái hoàn chỉnh cú pháp | khung chốt hình dạng **trước** |
| hỏng phẳng theo vị trí (lệch chủ đề) | chủ đề ràng buộc **mọi** khe cùng lúc |
| mọi mối nối làm hỏng (§13.52) | **không có mối nối** — một khung là một đơn vị |
| bệnh bản lề đồng tự (15,9%) | khe điền bằng **từ**, không chia sẻ âm tiết bản lề |
| quyết định rủi ro nhất đi trước | **đảo thứ tự**: chắc trước, rủi ro sau khi đã bị thắt |

Và **TK_046 đổ vào đúng chỗ**: `tach_lat` thắng `E` **1,60×** ở **xếp hạng ứng viên trong một
khung** — bước 3a **chính là** phép xếp hạng đó. Thế mạnh duy nhất đã chứng minh được đặt vào
đúng nơi cần nó.

## 4. Rủi ro — và phép đo giết được từng cái

| rủi ro | giết bằng | trạng thái |
|---|---|---|
| **kho khung quá thưa** (mỗi câu một khung ⇒ không khái quát) | P98, đo ngay | ⏳ |
| **sinh ra = chép trá hình** (khe chỉ điền lại đúng token cũ) | đo **tổ hợp mới**: số câu khung cho phép ÷ số câu đã thấy | ⏳ |
| nhãn chỉ phủ 73% lượt ⇒ khung khuyết | đo phủ theo khung | ⏳ |
| nút thắt **chọn câu đáp** (TK_045, 4 cơ chế cùng chết) | chưa có cách; **rủi ro lớn nhất còn lại** | ❗ |

⚠️ Rủi ro cuối là thật và tôi không có câu trả lời. Bốn cơ chế của TK_045 đều dừng ở **tiêu chí
chọn**. Kiến trúc này đổi *cái được chọn* (khung thay vì token) nhưng chưa chứng minh được là
đổi *chất lượng tiêu chí*.

## 5. P98 — CỔNG SUBSTRATE (nêu TRƯỚC, đo trước khi viết một dòng cơ chế nào)

Đo trên mini, ba mức hạt: **cả cha**, **mệnh đề**, **cửa sổ 4 nhãn**.

1. **Chia sẻ**: ≥ **30%** đơn vị có chuỗi nhãn trùng với ≥1 đơn vị khác.
2. **Tập trung**: top-200 khung phủ ≥ **20%** đơn vị.
3. **Tổ hợp mới**: trung vị của `log10(∏ số filler khác nhau mỗi khe) − log10(số lần khung
   được thấy)` ≥ **1,0** — khung phải cho phép **≥10×** số câu nó đã được thấy. Dưới ngưỡng
   đó, "sinh" chỉ là chép có trang trí.

**Dự đoán ghi TRƯỚC (Claude):**
- Mức **cả cha**: (1) **trượt** — câu dài ⇒ chuỗi nhãn gần như luôn duy nhất. Tôi đưa sẵn ba
  mức hạt vào **cùng một lần chạy** thay vì lặp lại vòng "đo → hỏng → đo lại".
- Mức **mệnh đề**: đây mới là câu hỏi thật, tôi đoán **borderline 25–45%**.
- Mức **cửa sổ 4**: (1)(2) đạt dễ nhưng cửa sổ **quá nhỏ để là khung câu** — nó ở đây làm
  **TRẦN**, không phải làm đáp án.
- (3) là chỗ tôi lo: nếu khung chia sẻ được đều là khung **ngắn và tầm thường** (`DT DT`), tổ
  hợp mới sẽ cao **một cách vô nghĩa**. Vì thế báo cáo (3) **kèm độ dài khung**, và không đọc
  (3) tách rời khỏi độ dài.
- Chạy **hai lần** trước khi đọc.

## 6. P98 — KẾT QUẢ: cổng TRƯỢT, nhưng nút thắt là **PHỦ NHÃN**, không phải thiếu substrate

Chạy hai lần, giống hệt nhau.

| mức hạt | đơn vị | khung | chia sẻ | top200 | dài TV | tổ hợp mới |
|---|---:|---:|---:|---:|---:|---:|
| `ca_cha` | 5.664 | 5.391 | **6,9%** | 8,4% | 8 | 1,20 |
| `menh_de` | 10.283 | 8.179 | **25,8%** | 18,5% | 5 | 1,31 |
| `cua_so_4` *(TRẦN)* | 100.307 | 1.676 | 99,6% | 82,8% | 4 | 2,38 |
| `menh_de_du_nhan` *(chẩn đoán)* | 652 | 347 | **57,2%** | **77,5%** | 4 | 1,20 |

🔴 **CỔNG G-frame TRƯỢT** ở cả `ca_cha` và `menh_de`.

**Dự đoán ghi trước đúng cả ba vế**: `ca_cha` trượt hẳn (6,9%); `menh_de` **borderline
25–45%** → ra **25,8%**, sát đáy khoảng, thiếu cổng **4,2 điểm**; `cua_so_4` đạt nhưng là trần.

### Vì sao trượt — và vì sao đó KHÔNG phải bản án cho kiến trúc

Nhãn phủ **73%** lượt; mệnh đề trung vị dài **5** ⇒ chỉ ~`0,73⁵ ≈ 21%` mệnh đề có **đủ nhãn**.
Một vị trí thiếu nhãn thành `?`, và hai mệnh đề **cùng ngữ pháp** biến thành **hai khung khác
nhau**. Số khung bị thổi lên, `chia sẻ` bị ép xuống.

Trên nhóm **đủ nhãn**: chia sẻ **57,2%**, top-200 **77,5%** — cả hai **vượt xa** cổng.

⚠️ Nhóm này **thiên lệch** và tôi không giấu: mệnh đề đủ nhãn thì **ngắn hơn** (khung trung vị
4 so với 5) và gồm **từ thông dụng hơn**. Một phần của bước nhảy là hiệu ứng chọn mẫu, không
phải chỉ do đủ nhãn. Nên đọc nó là **chỉ hướng**, không phải bằng chứng thay cổng.

### Việc kế — cụ thể, rẻ, đoán được

Nhìn vào token **không có nhãn** trong kho lịch sử này thì phần lớn là **tên riêng** (`Nguyễn`,
`Lý`, `Bạch Đằng`) và **số** — hai lớp mà **không cần từ điển nào** cũng gán được, chỉ cần
**lớp ký tự** (viết hoa → tên riêng; chữ số → số). Luật 0 **sạch**: đây là quy tắc ký tự, không
phải danh sách từ tiếng Việt, cùng loại với `is_structural_token` đang dùng.

**Dự đoán ghi TRƯỚC**: thêm hai nhãn đó đưa phủ **73% → 85–92%**, và `menh_de` chia sẻ
**25,8% → 40–55%** ⇒ **qua cổng**. Nếu phủ lên mà chia sẻ **không** lên, kết luận là chuỗi nhãn
**không phải** đơn vị khung đúng, và tôi sẽ bỏ tầng K chứ không nới cổng.

## 7. P98 vòng 2 — phủ nhãn **lên**, chia sẻ **XUỐNG**. Theo đăng ký trước: **BỎ TẦNG K**

Thêm hai nhãn bằng **quy tắc lớp ký tự** (Luật 0 sạch, không danh sách từ tiếng Việt nào):

- **số** — theo chữ số;
- **tên riêng** — ⚠️ viết hoa đầu câu là bẫy, nên quyết bằng **thống kê viết hoa GIỮA câu**
  của từng dạng từ (≥2 lượt giữa câu, ≥50% viết hoa) ⇒ 2.937 dạng từ. Vẫn thuần đếm.

Từ điển **thắng**, quy tắc chỉ lấp chỗ trống ⇒ mọi thay đổi đo được là do **phủ tăng**, không
do nhãn cũ bị xáo.

| | trước | sau |
|---|---:|---:|
| phủ (theo lượt) | 73,0% | **88,9%** ✅ *(đoán 85–92%)* |
| nguồn nhãn | từ điển 71.842 | + quy tắc **15.647** |
| nơ-ron nhãn | 8 | **10** |

**Kết quả (chạy hai lần, giống hệt):**

| mức hạt | đơn vị | khung | chia sẻ | top200 | tổ hợp mới |
|---|---:|---:|---:|---:|---:|
| `ca_cha` | 5.664 | 5.416 | 6,9% → **6,0%** | 8,4% → 7,9% | 1,20 → **0,90** |
| `menh_de` | 10.283 | 8.598 | 25,8% → **21,4%** | 18,5% → **14,9%** | 1,31 → **0,95** |
| `cua_so_4` *(trần)* | 100.307 | **3.187** | 99,6% → 99,2% | 82,8% → 61,0% | 2,38 → 2,15 |
| `menh_de_du_nhan` | 1.129 | 687 | 57,2% → **49,8%** | 77,5% → 56,9% | 1,20 → **0,90** |

🔴 **Mọi cột đều XUỐNG.** Dự đoán của tôi (chia sẻ 25,8% → 40–55%) **sai hoàn toàn**, và sai
theo hướng ngược.

### Vì sao — và đây là điều đáng học nhất của vòng này

`?` **đang làm ký tự đại diện**. Trước đó `Nguyễn` và `1258` **đều** là `?`, nên
`? ? N V` gộp câu-tên-riêng với câu-số vào **một** khung. Gán nhãn chính xác hơn **tách** chúng
thành `E E N V` và `M M N V`.

Bằng chứng trực tiếp: số khung ở `cua_so_4` **1.676 → 3.187**, gần **gấp đôi**, trong khi số
đơn vị không đổi.

⇒ **"Phủ nhiều hơn" và "chia sẻ nhiều hơn" kéo NGƯỢC CHIỀU nhau** khi phủ đến từ việc **thêm
lớp nhãn**. Nhãn mịn ⇒ khung có nghĩa ngữ pháp hơn nhưng **hiếm hơn**. Nhãn thô ⇒ chia sẻ nhiều
nhưng khung **không ràng buộc được gì**. Đây là căng thẳng **nội tại** của ý tưởng "khung =
chuỗi nhãn", không phải lỗi thi hành.

### Kết luận theo đúng đăng ký trước

§6 đã ghi: *"Nếu phủ lên mà chia sẻ **không** lên, kết luận là chuỗi nhãn **không phải** đơn vị
khung đúng, và tôi sẽ **bỏ tầng K** chứ không nới cổng."*

⇒ 🔴 **BỎ TẦNG K.** Chuỗi nhãn không phải đơn vị khung đúng.

Số nói thêm một điều: lặp lại tồn tại ở **~4 khe nhãn** (`cua_so_4`: 99,2%) và **chết ở 5+**
(`menh_de`: 21,4%). Nhưng khung 4 nhãn là **n-gram nhãn**, không phải khung câu — nó không mang
được cam kết toàn cục mà §0 đòi hỏi.

### Cái gì sống sót

| | |
|---|---|
| §0 — số học `0,9¹⁵` buộc bỏ sinh-từng-token tham lam | ✅ **vẫn đứng** (là số học, không phải giả thuyết) |
| §1 — phải tách LẬP KHUNG khỏi ĐIỀN KHE | ✅ vẫn đứng |
| §2 tầng K — **khung = chuỗi nhãn** | ❌ **chết** |
| §2 tầng N/S/C | chưa bị bác, nhưng **mất chỗ neo** khi K chết |

Tôi **chưa có** ứng viên thay thế đã đo cho "vật thể cam kết toàn cục". Một hướng chưa đo:
dùng **một cha CỤ THỂ làm khuôn** rồi **thay khe**, thay vì trừu tượng hoá thành chuỗi nhãn —
khuôn khi đó bảo đảm đúng ngữ pháp vì nó **là** một câu thật, và cái mới đến từ **phép thay**.
Đo được bằng đúng thước "tổ hợp mới" ở trên. Nhưng đó là **ý tưởng, chưa phải kết quả**.

## 8. P99 — KHUÔN CỤ THỂ + THAY KHE: **cổng ĐẠT**, lần đầu trong chuỗi

Thay "khung = chuỗi nhãn" (đã chết §7) bằng "khung = **một cha cụ thể**". Khuôn bảo đảm đúng
ngữ pháp vì nó **là** câu người viết; cái mới đến từ **phép thay khe**.

**Không cần cơ chế mới.** "Được phép thay" định nghĩa thuần đếm, **không tham số**: `c` thay
được `x` ở bối cảnh `(a, x, b)` nếu bộ ba `(a, c, b)` **có thật ở một cha khác**. Và bộ xếp
hạng ứng viên chính là `tach_lat` — nó xếp hạng *ứng viên trong một khung*, mà **một khe chính
là một khung**.

**Kết quả (chạy hai lần, giống hệt):**

| | |
|---|---:|
| cha có ≥1 vị trí thay được | **545/600** |
| khe nội dung thay được | 1.784/7.951 = **22,4%** |
| ứng viên/khe | trung vị **2** · trung bình 3,1 |
| **TỔ HỢP MỚI** (log10, trung vị) | **1,38** ⇒ **~24×** số câu đã thấy |

Xác nhận trên nửa **B** (suy phép thay trên nửa **A**):

| bộ thay | n | hẹp | chủ đề |
|---|---:|---:|---:|
| **`duoc_phep`** | 1.784 | **23,4%** | **80,8%** |
| `ngau_nhien_cung_loai` | 1.607 | 1,4% | 32,7% |
| **tỷ số** | | **17×** | **2,47×** |

✅ **CỔNG G-tmpl ĐẠT** cả ba. Đối chứng **không bão hoà** (32,7% < 50%) ⇒ thước chủ đề **phân
giải được**.

**Dự đoán ghi trước đúng cả hai vế**: (2) đoán 0,7–1,5 → **1,38**; (3) đoán 2–4× → **2,47×**.

### Ví dụ phép thay (KHÔNG phải phép đo chất lượng)

```
Sao Thiên Vương  → Sao Diêm Vương      Trần Nhân Tông → Trần Duệ Tông
Gạo Việt         → Phim Việt           rất cao        → rất sớm
thế giới         → thế lực             với vua        → với Pháp
Cửu Long         → Cửu Uyên  ✗         bắt ông        → bắt đầu  ✗
```

⚠️ Đây là **mắt tôi**, không phải chấm mù. Cổng đo **tính hợp lệ của phép thay**, **không** đo
**chất lượng đọc**. Trọng tài duy nhất vẫn là vòng **tuyệt đối mù** so **5,8% / 7,0% / 54,4%**.

### Một lỗi đã lộ ra ngay trong ví dụ — và nó có cách chữa sẵn

`bắt ông → bắt đầu` **vỡ ranh giới từ**: `bắt đầu` là **một từ**, phép thay tưởng mình đổi một
danh từ nhưng thực ra tạo ra một từ khác. Đây là **bệnh bản lề** (§13.39/§12) tái xuất ở tầng
thay khe.

Và lần này nó **có cổng chặn sẵn**: P94 đo cờ WS tách SINH khỏi THẬT **10,4×** (7,5% vs 0,7%).
Ở P94 cờ đó trượt vì bị đòi làm *bộ phát hiện bệnh* so với một chuẩn vàng rò; ở đây nó chỉ cần
làm **bộ lọc ứng viên** — việc dễ hơn nhiều, và là chỗ đúng cho nó. ⇒ **Tách từ tìm được nhà**.

### Trạng thái TK_047

| | |
|---|---|
| §0 số học buộc bỏ sinh-từng-token tham lam | ✅ |
| §1 tách LẬP KHUNG khỏi ĐIỀN KHE | ✅ |
| §2 tầng K (khung = chuỗi nhãn) | ❌ chết §7 |
| **§8 khung = cha cụ thể + thay khe** | ✅ **qua cổng substrate** |
| tầng N (nhãn) | dùng làm **đối chứng cùng-loại**, và lọc ứng viên |
| tầng S (khe) | = bộ ba `(a, ·, b)`, **đã có sẵn trong đếm** |
| tầng C (chủ đề) | thước "chủ đề" 80,8% **chính là** nó, đo được |

### Việc kế — thứ quyết định

Dựng **bộ sinh thật** theo §3 (chọn khuôn → xếp hạng ứng viên mỗi khe bằng `tach_lat` → lọc vỡ
từ bằng WS-CRF → điền theo độ chắc chắn), sinh mẫu, **chấm mù tuyệt đối**.

⚠️ Cổng của vòng đó phải nêu **trước** và phải có **hai vế**, không chỉ một:
1. **chất lượng** ≥ đường sinh hiện tại (5,8%/7,0%) — thanh thấp, nhưng là thanh đúng;
2. **KHÔNG phải chép**: tỷ lệ đầu ra **trùng nguyên văn** một cha phải **< 20%**. Thiếu vế này
   thì một bộ sinh "thay 0 khe" sẽ đạt 54,4% và trông như thắng lớn.

## 9. Bộ sinh KHUÔN + THAY KHE đã dựng — và **cổng (2) TRƯỢT trước cả khi chấm**

Thêm nhánh `khuon_thay` vào `gen_three_arm.py`. P99 chỉ đo **substrate**; bộ sinh cần thêm hai
thứ P99 không có:

1. **CHỦ ĐỀ của ĐỀ lái phép thay** — chỉ thay khi ứng viên **bám đề chặt hơn** token gốc
   (`|đồng hiện(c) ∩ chủ đề đề|`). Không tham số, không ngưỡng, không "thay k khe".
2. **Lọc VỠ TỪ bằng WS-CRF** — hoàn tác phép thay nào làm token dính vào láng giềng
   (`bắt ông → bắt đầu`). Đây là chỗ P94 tìm được nhà: làm **bộ lọc ứng viên**, không làm
   **bộ phát hiện bệnh**.
3. Điền **theo độ chắc chắn** (§3b): khe có mức tăng bám-đề **lớn nhất** đi trước.

Sinh 360 mẫu, 3 nhánh × 120 đề **rời nhau**: `khuon_thay` · `chep_cau` (mốc 54,4%, kiểm tái
lập thước) · `ar` (5,8%, thứ cần vượt).

| | |
|---|---:|
| có thay ≥1 khe | 65/120 = **54,2%** |
| số khe thay | trung vị **1** · trung bình 0,92 · max 5 |
| **TRÙNG NGUYÊN VĂN một cha** | **55/120 = 45,8%** |
| trống | **0/120** (`ar` 11,7%) |

🔴 **CỔNG G-gen vế (2) TRƯỢT**: ngưỡng nêu trước là **< 20%**, đo được **45,8%**. Ghi nhận,
**không dời**.

**Vì sao**: luật thay là **bảo thủ có chủ ý** — chỉ thay khi ứng viên bám đề **chặt hơn**. Ở
46% mẫu **không ứng viên nào** làm được thế, nên khuôn phát ra nguyên văn. Nới luật (ép thay
≥1 khe) sẽ hạ tỷ lệ trùng xuống, nhưng đó là **thay khi chủ đề nói đừng thay** — tức làm hỏng
chất lượng để đạt chỉ tiêu mới lạ. Đó là gaming chính cổng mình đặt, không làm.

### Phép tách trong CÙNG một vòng — cho biết cơ chế *hỏng* hay chỉ *thay quá ít*

Vòng chấm mù chia được nhánh `khuon_thay` làm hai nhóm bằng `meta` (giám khảo **không** thấy):

| nhóm | n | đọc ra sao |
|---|---:|---|
| **có thay** (≥1 khe) | 65 | phép thay có **làm hỏng** không |
| **trùng nguyên văn** | 55 | mốc chép nội bộ, phải ≈ `chep_cau` |

- nếu **có thay ≈ trùng nguyên văn** ⇒ phép thay **không làm hỏng**, vấn đề chỉ là **thay quá
  ít** — đó là bài toán **phủ**, sửa được (định nghĩa "được phép" chặt quá trên kho 5.664 cha);
- nếu **có thay ≪ trùng nguyên văn** ⇒ phép thay **làm hỏng**, và cơ chế chết như bốn cơ chế
  trước.

⏳ Đang chấm mù (phiên riêng, giám khảo bị cấm đọc `key`/`meta`/ticket/runbook/scripts).

## 10. Vì sao 45,8% trùng nguyên văn — **nút thắt CHỌN, y hệt TK_045**

Tách 55 mẫu trùng nguyên văn theo nguyên nhân:

| | |
|---|---:|
| **có ứng viên hợp lệ nhưng không cái nào bám đề chặt hơn** | **52/55 = 94,5%** |
| không có ứng viên nào | 3/55 = 5,5% |
| khe nội dung/khuôn (trung bình) | 13,3 — trong đó **4,1** khe *có* ứng viên |

⇒ Ràng buộc **không phải** độ thưa của phép cấp phép. Là **luật chọn**.

### Lỗi thiết kế: phép thay tối ưu đúng hàm mục tiêu truy hồi vừa tối ưu xong

`rank_matches` **đã chọn** cha bám đề nhất. Rồi phép thay hỏi *"ứng viên nào bám đề **chặt
hơn**?"* — theo cấu tạo thì thường **không có**. Vòng tự triệt tiêu.

### Ba biến thể luật thay — không cái nào qua cổng

| luật | trùng nguyên văn | ghi chú |
|---|---:|---|
| **chồng lấp chủ đề** (bản gốc) | **45,8%** | tự triệt tiêu với truy hồi |
| lấp gap, gap xét bằng **đồng hiện** | **100%** | `gap` luôn **rỗng** — §13.20 lần thứ n: token phổ biến đồng hiện với mọi thứ |
| lấp gap, gap = token đề **vắng mặt** | **79,2%** | ít thay hơn, và phép thay **xấu đi**: `Tết Nguyên Đán → Tết Liệu` |

Cổng nêu trước: **< 20%**. Xa nhất còn 45,8%. **Không cái nào đạt.**

### 🔴 Rủi ro §4 đã thành hiện thực

§4 ghi trước: *"Bốn cơ chế của TK_045 đều dừng ở **tiêu chí chọn**. Kiến trúc này đổi *cái được
chọn* (khung thay vì token) nhưng **chưa chứng minh được là đổi *chất lượng tiêu chí***."*

Nó **không** đổi. Cơ chế **thứ năm** chết ở **đúng chỗ** bốn cơ chế trước chết.

Và tôi **dừng chỉnh luật thay tại đây**. Ba biến thể là đủ để thấy đây không phải chuyện tinh
chỉnh; chỉnh tiếp chính là vòng "đo → hỏng → nới → đo lại" đã làm hỏng ba vòng đối chứng ở
TK_046 §16.

### Câu hỏi còn lại — và vòng chấm đang chạy trả lời đúng nó

Substrate **tốt** (P99: 24× tổ hợp mới, 2,47× đối chứng, cả ba cổng đạt). Luật chọn **hỏng**.
Vậy khi phép thay **có** xảy ra, nó **làm hỏng** hay **không**?

| kết quả chấm mù | đọc là |
|---|---|
| **có thay (65) ≈ trùng nguyên văn (55)** | phép thay **lành** ⇒ bài toán thu về **đúng một** thứ: *tiêu chí chọn* |
| **có thay ≪ trùng nguyên văn** | phép thay **làm hỏng** ⇒ substrate tốt cũng vô ích |

Đây là lần đầu nút thắt chọn bị cô lập khỏi mọi thứ khác — substrate đã qua cổng độc lập, nên
nếu phép thay lành thì **chỉ còn một biến** phải giải.

## 11. CHẤM MÙ 360/360 — **phép thay LÀM HỎNG**. Cơ chế thứ năm chết.

| nhánh | n | dùng được | tạm | hỏng | **dùng+tạm** |
|---|---:|---:|---:|---:|---:|
| `khuon_thay` | 120 | 28,3% | 33,3% | 38,3% | **61,7%** |
| `chep_cau` | 120 | 48,3% | 26,7% | 25,0% | **75,0%** |
| `ar` | 120 | 0,0% | 3,3% | 96,7% | **3,3%** |

### 🔴 Phép tách quyết định — trong CÙNG nhánh, CÙNG vòng, CÙNG giám khảo

| nhóm | n | **dùng+tạm** |
|---|---:|---:|
| **TRÙNG NGUYÊN VĂN** (không thay gì) | 55 | **81,8%** |
| **CÓ THAY** (≥1 khe) | 65 | **44,6%** |
| **chênh** | | **−37,2 điểm · z = −4,18 TÁCH** |

§9 ghi trước hai cách đọc. Kết quả rơi vào cách thứ hai: *"có thay ≪ trùng nguyên văn ⇒ phép
thay **làm hỏng** ⇒ substrate tốt cũng vô ích."*

⇒ **Mỗi phép thay trung bình phá hỏng câu.** Substrate qua cổng (P99: 24× tổ hợp mới, 2,47×
đối chứng) **không cứu được** điều đó. Đây là **cơ chế thứ năm** chết ở **tiêu chí chọn**.

### ⚠️ Hai điều làm yếu kết luận, phải ghi

**1. Phép tách KHÔNG ngẫu nhiên hoá.** Luật thay chỉ kích hoạt khi có ứng viên bám đề **chặt
hơn** — tức khi truy hồi tìm được khuôn **kém khớp**, còn chỗ để cải thiện. Vậy nhóm "có thay"
có thể **vốn đã khó hơn** từ đầu. Một phần trong 37,2 điểm là **chọn mẫu**, không phải **hư
hại**. Cách gỡ: **ép thay trên một nhóm bốc ngẫu nhiên** rồi so với nhóm chứng cùng phân bố —
đó là **thiết kế đo**, không phải chỉnh luật, nên không vi phạm điều tôi tự cấm ở §10.

**2. Thước KHÔNG tái lập vòng này.** `chep_cau` ra **75,0%**, so với **53,3%** (§13.60) và
**54,4%** (§11 TK_046) — lệch **~21 điểm**, z ≈ 3,4. `ar` thì ổn (5,8% → 3,3%, trong nhiễu).
Nguyên nhân **chưa rõ**: giám khảo khác phiên, bộ đề khác (n=120 nên biên lát khác n=150).

⇒ Mọi so sánh **trong vòng này** (ba nhánh, và phép tách trên) **hợp lệ** — cùng giám khảo,
cùng lô. Mọi so sánh **bắc cầu sang vòng khác** (`61,7%` vs `54,4%`, `3,3%` vs `5,8%`) **KHÔNG
an toàn** vòng này. Không được đọc `khuon_thay 61,7% > chep_cau 54,4%` — trong cùng vòng nó
**thua** chép, z = 2,22.

### Trạng thái TK_047

| | |
|---|---|
| §0 số học buộc bỏ sinh-từng-token tham lam | ✅ vẫn đứng |
| §1 tách LẬP KHUNG khỏi ĐIỀN KHE | ✅ vẫn đứng |
| §2 tầng K (khung = chuỗi nhãn) | ❌ chết §7 |
| §8 substrate khuôn + thay khe | ✅ qua cổng |
| **§11 phép thay ở đầu ra** | ❌ **làm hỏng, −37,2 điểm** |
| cổng (2) không-phải-chép | ❌ 45,8% ≫ 20% |

Cả hai vế cổng G-gen **trượt**. Kiến trúc khuôn–khe **không** vượt được đường chép, và phần
"sinh" của nó **trừ điểm**.

## 12. Vòng ÉP THAY — tách HƯ HẠI khỏi CHỌN MẪU *(dự đoán ghi TRƯỚC khi có số)*

§11 để lại một lỗ: phép tách `có thay` vs `trùng nguyên văn` **không ngẫu nhiên hoá**. Vòng này
bốc khuôn **ngẫu nhiên** vào nhánh ⇒ ba nhóm **cùng phân bố độ khó**.

| nhánh | làm gì |
|---|---|
| `chep_cau` | phát khuôn **nguyên văn** — đối chứng |
| `ep_ngau_nhien` | **ép thay đúng 1 khe**, ứng viên bốc **ĐỀU** trong các ứng viên hợp lệ |
| `ep_chu_de` | **ép thay đúng 1 khe**, ứng viên chọn theo **điểm bám đề** |

Hai hiệu số, hai câu hỏi **tách rời**:

- `ep_ngau_nhien` − `chep_cau` ⇒ **phép cấp phép (bộ ba) MỘT MÌNH** có đủ để thay không đau
  không? Đây là câu §11 chưa trả lời sạch.
- `ep_chu_de` − `ep_ngau_nhien` ⇒ **giá trị riêng của tiêu chí chủ đề**, đo trực tiếp lần đầu.

### Dự đoán ghi TRƯỚC (Claude)

- `ep_ngau_nhien` **thấp hơn hẳn** `chep_cau`, tụt **25–45 điểm**. Lý do: cấp phép bằng bộ ba
  là ràng buộc **yếu** — P99 đo thước **hẹp** chỉ **23,4%**, tức 3/4 phép thay hợp lệ **không**
  tái xuất hiện ở nửa corpus kia; và §13.31 đo 65,4% lựa chọn argmax dựa trên token gặp ≤5 lần.
- `ep_chu_de` − `ep_ngau_nhien` = **0–10 điểm**, tức **nhỏ**. Lý do: §15 và §16 đã cho thấy các
  tiêu chí chọn thêm vào **liên tục không mua gì** (nhãn từ loại 1,03×; tách hướng 0,97×).
- Nếu **cả hai** đúng ⇒ hướng khuôn–khe **đóng**: substrate tốt, cấp phép yếu, tiêu chí chọn
  không cứu được. Nếu `ep_ngau_nhien` ≈ `chep_cau` ⇒ §11 phần lớn là **chọn mẫu**, và hướng
  này **còn sống**.
- Thước trong vòng đọc bằng `chep_cau` của **chính vòng này**, **không** bắc cầu sang 75,0%
  hay 54,4% — §11 đã cho thấy bắc cầu không an toàn.

### KẾT QUẢ — cả hai dự đoán đúng, hướng khuôn–khe **ĐÓNG**

| nhánh | n | dùng được | tạm | hỏng | **dùng+tạm** |
|---|---:|---:|---:|---:|---:|
| `chep_cau` (nguyên văn) | 120 | 39,2% | 33,3% | 27,5% | **72,5%** |
| `ep_ngau_nhien` | 120 | 10,0% | 20,0% | 70,0% | **30,0%** |
| `ep_chu_de` | 120 | 6,7% | 23,3% | 70,0% | **30,0%** |

| so sánh | chênh | z | |
|---|---:|---:|---|
| `ep_ngau_nhien` − `chep_cau` | **−42,5đ** | **−6,59** | TÁCH |
| **`ep_chu_de` − `ep_ngau_nhien`** | **+0,0đ** | **0,00** | **KHÔNG TÁCH** |

**Dự đoán ghi trước đúng cả hai**: tụt 25–45đ → **42,5đ**; hiệu tiêu chí chủ đề 0–10đ →
**đúng 0,0đ**.

### 🔴 Hai kết luận, và cái thứ hai đổi cả chẩn đoán

**1. Phép thay LÀM HỎNG THẬT.** Có ngẫu nhiên hoá, hiệu ứng **không nhỏ đi** mà **to hơn**
(−37,2 → −42,5). §11 **không** phải hiện vật chọn mẫu.

**2. Tiêu chí chọn đáng giá ĐÚNG SỐ KHÔNG.** Chọn ứng viên **bám đề nhất** tốt **y hệt** bốc
**ngẫu nhiên**: 30,0% vs 30,0%, z = 0,00.

Vế (2) **bác chính chẩn đoán của tôi ở §10**. Nút thắt **không phải** "tiêu chí chọn kém" —
TK_045 kết luận vậy, tôi lặp lại vậy, và cả hai **sai**. Nút thắt là **BỂ ỨNG VIÊN**: hầu như
**mọi** ứng viên được cấp phép đều sai, nên chọn cái nào **không còn quan trọng**.

### Bài học đo lường — cùng họ với cổng-điểm-phần-trăm ở P94

P99 đã **báo trước** điều này và tôi **đọc sai**: thước **hẹp** chỉ **23,4%** — tức **76,6%**
phép thay "hợp lệ" **không hề tái xuất** ở nửa corpus kia. Tôi đọc nó là *"17× sàn ngẫu nhiên,
tốt"*.

⚠️ **Tỷ số lớn trên một sàn gần 0 thì vẫn gần 0 về tuyệt đối.** 17× của 1,4% là 23,4% — nghe
như thắng lớn, thực chất là **ba phần tư sai**. Cùng đúng họ với lỗi đặt cổng theo **điểm phần
trăm** trên nền 0,7% ở P94, chỉ đảo chiều. **Cổng phải có cả tỷ số VÀ mức tuyệt đối.**

### ⚠️ Thước vẫn chưa tái lập

`chep_cau`: **53,3% → 54,4% → 75,0% → 72,5%**. Hai vòng cũ nhất trí, hai vòng mới nhất trí,
giữa hai cụm lệch **~20 điểm** — **nguyên nhân chưa rõ**, và bộ đề khác nhau **không** giải
thích được (hai vòng mới dùng hai lát khác nhau mà vẫn trùng nhau). Mọi kết luận trên đều là
so sánh **trong cùng vòng**, nên không bị ảnh hưởng; nhưng đây là **nợ đo lường** phải trả.

## 13. TK_047 — tổng kết: ĐÓNG

| | |
|---|---|
| §0 số học `0,9¹⁵` buộc bỏ sinh-từng-token tham lam | ✅ **vẫn đứng** |
| §1 tách LẬP KHUNG khỏi ĐIỀN KHE | ✅ vẫn đứng |
| §2 tầng K — khung = chuỗi nhãn | ❌ chết §7 |
| §8 substrate khuôn + thay khe | ✅ qua cổng (24× tổ hợp mới) |
| §11 phép thay ở đầu ra | ❌ −37,2đ |
| **§12 phép thay có ngẫu nhiên hoá** | ❌ **−42,5đ, z = −6,59** |
| **§12 tiêu chí chọn** | ❌ **+0,0đ** — vô giá trị |

⇒ **Đóng hướng khuôn–khe**, theo đúng điều kiện tự đặt ở §12.

**Cái còn lại có giá trị**: nút thắt được định vị **chính xác hơn mọi lần trước**. Không phải
"thiếu dữ liệu", không phải "đơn vị sai", không phải "tiêu chí chọn kém" — mà là **phép cấp
phép bằng đồng hiện cục bộ tạo ra bể ứng viên gần như toàn sai**. Mọi cơ chế trong TK_045,
TK_046, TK_047 đều rút ứng viên từ **cùng một bể** đó. Đó là lời giải thích **duy nhất** phù
hợp với **cả sáu** kết quả âm tính, và nó nói rằng việc kế phải là **đổi cách sinh ứng viên**,
không phải đổi cách **chọn** trong số đó.

## 14. P100 — giao M4: đo độ chính xác cấp phép trên **v12**

§13 định vị nút thắt là **bể ứng viên**. Đại lượng đo trực tiếp bể đó là thước **hẹp** của P99:
suy phép thay trên nửa **A**, hỏi bộ ba `(a, c, b)` có **thật sự xuất hiện** ở nửa **B** không.
Trên mini: **23,4%**.

**Câu hỏi duy nhất:** 23,4% là tính chất của **DỮ LIỆU THƯA** hay của **ĐỒNG HIỆN CỤC BỘ**?
`v12` = **430.447 cha**, gấp **76×**. Đây là đại lượng duy nhất còn chặn đường và **chưa từng
được đo ở quy mô đó**.

**Cổng (4) mới, TUYỆT ĐỐI**: `hẹp ≥ 50%`. Thêm vào vì lỗi đọc số đã xảy ra thật — 23,4% là
"17× sàn ngẫu nhiên" và tôi đọc là tốt. **Tỷ số lớn trên sàn gần 0 vẫn gần 0 tuyệt đối.**
Từ nay cổng phải có **cả tỷ số VÀ mức tuyệt đối**.

| `hẹp` trên v12 | kết luận |
|---|---|
| **≥ 50%** | cấp phép **là chuyện dữ liệu** ⇒ hướng khuôn–khe **mở lại**, dựng lại bộ sinh trên v12 |
| 30–50% | cải thiện thật nhưng **chưa đủ**; ghi số, **không** dựng lại |
| **≈ 23% (đứng yên)** | 🔴 tính chất của **đồng hiện cục bộ** ⇒ hướng **đóng dứt điểm** |

**Dự đoán ghi TRƯỚC**: `hẹp` lên **35–55%**, nhưng **rất không chắc** — `v12` cũng làm `triA`
**rộng ra**, và ứng viên thêm vào là những cái **hiếm hơn**, có thể **kéo độ chính xác xuống**.
Hai lực ngược chiều; tôi không biết lực nào thắng. Đó chính là lý do đáng chạy. Nếu `hẹp`
**giảm** so với mini thì đó là bằng chứng mạnh nhất có thể cho việc cấp phép bằng bộ ba là ngõ
cụt, và ghi đóng ngay.

Runbook: [`RUNBOOK_TK_047_P100.md`](RUNBOOK_TK_047_P100.md).

## 15. P100 KẾT QUẢ (M4, v12) — cổng TRƯỢT, nhưng lực tôi lo **không thắng**

Chạy hai lần mỗi cấu hình, **giống hệt nhau**. Bước A (`token_labels_v12.json`) chạy được,
phủ **90,6%**, G-pos đạt.

| cấu hình | cha/nửa | **hẹp** | khe thay được | tổ hợp mới (log10) |
|---|---:|---:|---:|---:|
| `mini` | 2.832 | **23,4%** | 22,4% | 1,38 |
| `v12` (cap 100k) | 100.000 | **38,8%** | 60,2% | 6,64 |
| **`v12` (toàn bộ)** | 215.223 | **43,8%** | **68,1%** | **8,00** |

🔴 **CỔNG (4) TRƯỢT**: 43,8% < 50%. Rơi đúng dải tôi đăng ký trước là **30–50% ⇒ *cải thiện
thật nhưng chưa đủ; ghi số, KHÔNG dựng lại bộ sinh***.

### Dự đoán đúng, và **lực tôi lo đã không thắng**

Tôi đoán 35–55% → ra **43,8%**. Nhưng phần đáng chú ý hơn là phần tôi ghi *"rất không chắc"*:
v12 làm tập ứng viên **rộng ra** (khe thay được 22,4% → **68,1%**, tổ hợp mới 1,38 → **8,00**,
tức **10⁸** câu), và tôi lo ứng viên thêm vào **hiếm hơn** sẽ **kéo độ chính xác xuống**.

**Nó không kéo xuống.** Độ chính xác **tăng đơn điệu** dù bể ứng viên nở gấp bội. Đó là bằng
chứng thật rằng cấp phép bằng bộ ba **không** bão hoà theo hướng xấu.

### Xu hướng theo dữ liệu — chưa phẳng

| đoạn | Δ dữ liệu | Δ hẹp | độ dốc |
|---|---|---:|---:|
| mini → v12 100k | 1,55 decade | +15,4đ | 9,9 đ/decade |
| v12 100k → v12 toàn bộ | 0,33 decade | +5,0đ | **15,0 đ/decade** |

Độ dốc **không giảm** ở đoạn cuối. Ngoại suy thô từ đoạn cuối: đạt 50% cần **×2,6 dữ liệu**
≈ **1,1 triệu cha**.

⚠️ Đây là **ngoại suy ba điểm**, đoạn cuối chỉ dài 0,33 decade nên độ dốc 15,0 đ/decade **rất
nhiễu**. Không được đọc "chỉ cần ×2,6" như một lời hứa.

### 🔴 Hai nợ đo lường mới

**1. Thước chủ đề đã BÃO HOÀ ở quy mô v12.** Đối chứng ngẫu-nhiên-cùng-loại đạt **64,2%**
(mini: 32,7%), vượt ngưỡng bão hoà 50% mà probe tự cảnh báo. Tỷ số tụt 2,47× → **1,52×** —
nhưng **không phải** vì `duoc_phep` xấu đi (97,4%), mà vì **đối chứng tốt lên**. ⇒ Ở quy mô
v12, thước chủ đề **không phân giải được nữa**; §13.20 lần thứ n.

**2. Ngưỡng 50% chưa từng được kiểm.** Tôi đặt nó bằng lập luận *"phép thay chỉ có cơ hội
không-gây-hại khi ĐA SỐ được chứng thực"*. Nhưng hàm truyền **độ chính xác → chất lượng** chỉ
có **đúng một điểm đo**: 23,4% ⇒ **−42,5đ**. Không ai biết 43,8% cho ra bao nhiêu. Có thể vẫn
hỏng nặng; cũng có thể gần hoà.

⚠️ Phép đo giải quyết được (2) là **chạy lại vòng ép thay trên v12**. Nhưng điều đó **chính là
"dựng lại bộ sinh"** mà tiền-đăng-ký cấm ở dải 30–50%. Tôi **nêu mâu thuẫn này ra thay vì lặng
lẽ vượt qua nó**: nó là **phép đo hàm truyền**, không phải tuyên bố hướng đã mở lại — nhưng
ranh giới đó do tôi vẽ, nên **tác giả quyết**, không phải tôi.

### Trạng thái

Cấp phép bằng bộ ba **là chuyện dữ liệu ở mức đáng kể** — 23,4% → 43,8% và **chưa phẳng**. Đó
là kết quả **dương** đầu tiên cho câu hỏi "thêm dữ liệu có cứu được không" trong cả ba ticket.
Nhưng nó **chưa đủ** để mở lại hướng theo đúng cổng đã nêu, và **chưa ai biết** 43,8% đổi ra
bao nhiêu điểm chất lượng.

## 16. P101 — ĐO CHÍNH CÁI THƯỚC *(tiền-đăng-ký, ghi trước khi có số)*

### Tự đính chính: "thước tái lập 3 lần" là phát biểu SAI

Tôi đã viết câu đó nhiều lần trong TK_045/046/047. Kiểm lại: ba lần đó là **ba phiên chấm cùng
một bộ 200 mẫu** ở vòng tuyệt đối v12, nơi hệ đạt **0,5% / 8,5%** — tức **đáy thang**, chỗ đồng
thuận là chuyện dễ vì mọi thứ đều hỏng.

**Vùng giữa thang chưa từng được kiểm đồng thuận.** Và đúng ở đó xuất hiện chênh:

`chep_cau` qua bốn vòng: **53,3% → 54,4% → 75,0% → 72,5%** — hai cụm, lệch **~20 điểm**.

Mọi kết luận cơ chế trong ba ticket đều đứng trên thước này.

### Thiết kế

Chấm lại **ĐÚNG 240 mẫu đã chấm** của vòng `ar` bằng **một phiên độc lập**, giám khảo bị cấm
đọc điểm cũ. Điểm cũ trên chính tập này:

| nhánh | n | dùng+tạm (lần 1) |
|---|---:|---:|
| `ban_le` | 76 | 7,9% |
| **`chep_cau`** | 83 | **55,4%** |
| `ar` | 81 | 6,2% |

### CỔNG G-rater (nêu TRƯỚC)

1. **≥200 mẫu** được chấm lại.
2. **Đồng thuận từng mẫu ≥ 70%** và **kappa ≥ 0,50**. Dưới đó, thang đo **không phải công cụ
   tin được**, và mọi con số trong ba ticket phải đọc lại với sai số rất rộng.
3. **`chep_cau` lần 2 lệch < 10 điểm** so với **55,4%**.

### Hai kết cục, hai hệ quả

| `chep_cau` lần 2 | kết luận |
|---|---|
| **≈ 55%** | thước **ổn định**; chênh 20 điểm đến từ **bộ đề**, và so sánh trong-vòng vẫn là cách đọc đúng |
| **≈ 73%** | 🔴 **giám khảo trôi**; mọi con số lịch sử phải đọc lại, và thước cần neo (mẫu chuẩn cố định kèm mọi vòng) |

### Dự đoán ghi TRƯỚC (Claude)

- Tôi **thật sự không biết**. Hai cụm mỗi cụm hai vòng, mỗi vòng một phiên khác nhau — nếu là
  nhiễu giám khảo thuần tuý thì phải thấy **trộn lẫn**, không phải **hai cụm sạch**. Điều đó
  nghiêng về **bộ đề**. Nhưng bộ đề của hai vòng mới là **hai lát khác nhau** mà vẫn trùng
  nhau, và điều đó lại nghiêng về **giám khảo**. Hai mảnh bằng chứng **mâu thuẫn nhau**.
- Đồng thuận từng mẫu: đoán **70–80%**, kappa **0,5–0,65**. Ba mức có ranh giới `tam` mờ.
- Chỗ bất đồng nhiều nhất sẽ là ranh `tam`/`hong`, không phải `dung_duoc`/`hong`.

### ⚠️ Một sai lệch quy trình của tôi, ghi trước khi biết kết quả

Đề bài chấm lần 2 có thêm **một dòng không có ở lần 1**: *"câu chép lại nguyên văn câu đề rồi
không thêm gì ⇒ `hong`"*. Đó là **khác biệt do tôi tạo ra**, làm lần 2 **chặt hơn** ở đúng một
loại lỗi. Nó ảnh hưởng chủ yếu tới nhánh `ar` (hay nhại lại đề) và **gần như không** tới
`chep_cau` (luôn phát một cha KHÁC). Cột quyết định vì thế ít bị ảnh hưởng — nhưng tôi ghi ra
đây thay vì im lặng, và đồng thuận từng mẫu ở nhánh `ar` phải đọc kèm ghi chú này.

### KẾT QUẢ P101 — thước **ĐỒNG THUẬN TỐT** nhưng **KHÔNG HIỆU CHUẨN** ở vùng giữa

Chấm lại **đúng 240 mẫu cũ**, phiên độc lập, cấm đọc điểm cũ.

| nhánh | n | lần 1 | lần 2 | chênh |
|---|---:|---:|---:|---:|
| `ban_le` | 76 | 7,9% | 10,5% | +2,6đ |
| **`chep_cau`** | 83 | **55,4%** | **78,3%** | **+22,9đ** |
| `ar` | 81 | 6,2% | 0,0% | −6,2đ ⚠️ |

**Đồng thuận từng mẫu 83,3% · kappa 0,614** — cổng (2) **ĐẠT** (≥70%, ≥0,50).
**Cổng (3) TRƯỢT**: 22,9đ ≫ 10đ.

⚠️ `ar` −6,2đ **do lỗi của tôi**: đề bài lần 2 thêm dòng *"chép nguyên văn câu đề ⇒ hong"*, mà
`ar` hay nhại đề. Đã ghi trước khi biết số. `chep_cau` **không** bị dòng đó chạm tới.

### 🔴 Kết luận: chênh 20 điểm là **GIÁM KHẢO TRÔI**, không phải bộ đề

Cùng **đúng những mẫu đó**, chênh **tái hiện gần y nguyên** (+22,9 vs ~20 quan sát giữa các
vòng). Giả thuyết "bộ đề" **bị loại**.

Ma trận nhầm (hàng = lần 1):

| | `dung_duoc` | `tam` | `hong` |
|---|---:|---:|---:|
| `dung_duoc` | **20** | 0 | 0 |
| `tam` | **10 ↑** | 20 | 7 ↓ |
| `hong` | 2 ↑ | **21 ↑** | 160 |

Trôi **BẤT ĐỐI XỨNG**: 33 mẫu đi **lên**, 7 mẫu đi **xuống**. Đây là **nới tay có hệ thống**,
không phải nhiễu ngẫu nhiên.

### Vì sao "tái lập 3 lần" đánh lừa — và nó đánh lừa ở đâu

`dung_duoc` **ổn định tuyệt đối** (20/20 giữ nguyên) và `hong` gần như ổn định (160/183). Trôi
nằm **trọn vẹn ở hai ranh giới** `hong`↔`tam` và `tam`↔`dung_duoc`.

⇒ Ở **đáy thang** (hệ đạt 0,5%/8,5%) hầu như mọi mẫu đều là `hong` **sâu** ⇒ ba phiên nhất trí
⇒ trông như thước vững. Ở **vùng giữa** (đường chép, 55–78%) mật độ mẫu **ngay tại ranh giới**
là cao nhất ⇒ trôi **bung ra hết cỡ**. Thước **vững ở chỗ không cần chính xác** và **lỏng ở
chỗ cần**.

### Kết luận nào SỐNG, kết luận nào CHẾT

| kết luận | loại so sánh | |
|---|---|---|
| `ep_ngau_nhien` − `chep_cau` = **−42,5đ** | **trong vòng** | ✅ sống (và 42,5 > biên trôi 23) |
| `ep_chu_de` − `ep_ngau_nhien` = **+0,0đ** | trong vòng | ✅ sống |
| `khuon_thay` − `chep_cau` = −13,3đ | trong vòng | ✅ sống |
| `chep_cau` − `ar` = +48,6đ · `ar` ≈ `ban_le` | trong vòng | ✅ sống |
| **"76× dữ liệu → 1,9× chất lượng"** (v12 13,0% vs mini 7,0%) | **BẮC CẦU** | 🔴 **CHẾT** |
| mọi con số tuyệt đối trích dẫn xuyên vòng (53,3% / 54,4% / 61,7% / 72,5%) | bắc cầu | 🔴 **không so được** |

Cái chết thứ nhất đáng chú ý: tôi đã dùng "76× → 1,9×" để lập luận **tăng dữ liệu không cứu
được**. Lập luận đó **không còn cơ sở** — và nó đi cùng chiều với P100 (độ chính xác cấp phép
23,4% → 43,8%, chưa phẳng), tức bằng chứng về vai trò của dữ liệu **mạnh hơn** tôi đã nói.

### Cách chữa: thước phải có **NEO**

Từ nay mọi vòng chấm phải kèm **một bộ mẫu neo cố định** (~40 mẫu, cùng bộ, mọi vòng). Điểm neo
đo được **độ trôi của chính phiên đó**, và mọi số của vòng được hiệu chỉnh về nó. Không có neo
thì **không** so được xuyên vòng — và ba ticket vừa qua đã không có.

## 17. P102 — thang chấm **không kiểm sự thật** *(tiền-đăng-ký, ghi trước khi có số)*

Lộ ra khi tác giả hỏi ví dụ: mẫu A017 ra *"Năm **1071**, Lý Thường Kiệt đánh bại quân Tống trên
phòng tuyến sông Như Nguyệt"* — trận đó là **1077**. Câu trôi chảy, đúng chủ đề, **sai sự
thật**, và được chấm **`dung_duoc`**.

Với một cơ chế mà chế độ hoạt động **chính** là *thay tên–năm–địa danh*, đây **không phải lỗi
biên** — nó là **chế độ hỏng đặc trưng**. Thang hiện tại hỏi *trôi chảy + hợp đề*, **không** hỏi
*đúng không* ⇒ mọi con số `dung_duoc` là **trần trên**.

### Thiết kế — đối chứng nằm sẵn trong dữ liệu

Chỉ xét mẫu **đã được chấm `dung_duoc`** (tức hệ tự nhận là dùng được), gộp từ vòng `tmpl` và
vòng `ep`:

| nhóm | n | là gì |
|---|---:|---|
| **`NGUOI_VIET`** | 45 | `chep_cau` + khuôn phát **nguyên văn** ⇒ **câu người viết**, chưa qua phép thay |
| **`CO_THAY`** | 21 | đã qua **≥1 phép thay khe** |

⚠️ `NGUOI_VIET` là **sàn hoàn hảo**: đó là câu người thật viết trong corpus. Mọi tỷ lệ sai vượt
trên sàn đó là **do cơ chế tạo ra**. Không cần giả định gì thêm.

Người kiểm chứng bị cấm biết mẫu nào thuộc nhóm nào, và có mức **`khong_kiem_duoc`** để không
phải đoán bừa.

### CỔNG G-fact (nêu TRƯỚC)

1. **Đối chứng phải sạch**: `NGUOI_VIET` tỷ lệ `sai` **< 15%**. Vượt ngưỡng ⇒ người kiểm chứng
   **không tin được** (bịa lỗi cho câu người viết), phép đo **vô hiệu**, không đọc tiếp.
2. **`khong_kiem_duoc` < 40%** tổng thể. Cao hơn ⇒ chưa phân giải được, ghi rõ chứ không kết luận.
3. Đại lượng cần đọc: **`sai`(CO_THAY) − `sai`(NGUOI_VIET)** = phần sai sự thật **do phép thay
   sinh ra**.

### Dự đoán ghi TRƯỚC (Claude)

- `NGUOI_VIET` sai **5–12%**: câu bách khoa thật vẫn có lỗi, và người kiểm chứng cũng sai đôi
  chút. Nếu nó **vượt 15%** thì cổng (1) trượt và tôi **không** đọc số của nhóm kia.
- `CO_THAY` sai **35–60%**. Lý do: phép thay đổi đúng những khe mang **tên, năm, địa danh** —
  tức đúng chỗ sự thật nằm.
- Nếu chênh **≥ 25 điểm** ⇒ `dung_duoc` **phóng đại nghiêm trọng** với nhánh có thay, và mọi
  con số chất lượng của TK_047 phải đọc kèm khấu trừ này.
- n = 21 ở nhóm `CO_THAY` là **nhỏ** — sai số rộng. Ghi trước để không đọc quá con số.

## 18. Bộ mẫu NEO — đã dựng

`out/anchor_samples.txt` (40 mẫu) + `out/anchor_key.json`. Nhãn tham chiếu = **đồng thuận của
hai giám khảo độc lập** (P101), trải đều: 14 `dung_duoc` · 13 `tam` · 13 `hong`.

**Quy tắc từ nay:** mọi vòng chấm mù **phải trộn 40 mẫu neo này vào**, và báo cáo phải ghi
`độ trôi = điểm phiên trên neo − nhãn tham chiếu`. Không có con số đó thì vòng chấm **không so
được** với bất kỳ vòng nào khác — đúng lỗi đã làm hỏng mọi so sánh bắc cầu trong ba ticket.

### KẾT QUẢ P102 — thang chấm phóng đại **ÍT HƠN tôi lo**, và lỗi tập trung ở **NIÊN ĐẠI**

| nhóm | n | `dung` | `sai` | `?` | **SAI %** |
|---|---:|---:|---:|---:|---:|
| **`NGUOI_VIET`** (câu người viết) | 45 | 41 | 1 | 3 | **2,2%** |
| **`CO_THAY`** (đã thay ≥1 khe) | 21 | 15 | 3 | 3 | **14,3%** |
| **chênh** | | | | | **+12,1đ · z = 1,91 → KHÔNG TÁCH** |

✅ **Cổng (1) ĐẠT**: đối chứng `NGUOI_VIET` **2,2%** ≪ 15% — người kiểm chứng **không bịa lỗi**
cho câu người viết. Công cụ tin được.
✅ **Cổng (2) ĐẠT**: `khong_kiem_duoc` **9,1%** ≪ 40%.

### Dự đoán của tôi SAI, và sai theo hướng **bi quan**

Tôi đoán `NGUOI_VIET` 5–12% → thật **2,2%**; `CO_THAY` **35–60%** → thật **14,3%**. Và ngưỡng
tôi tự đặt — *"chênh ≥25đ ⇒ phóng đại nghiêm trọng"* — **không đạt** (12,1đ).

⇒ 🔴 **Tôi phải sửa điều đã nói với tác giả.** Tôi đã báo *"61,7% là trần trên, chất lượng dùng
được thật thấp hơn"* dựa trên **một** ví dụ (A017). Đo tử tế thì khấu trừ là **≈12 điểm trên
riêng nhóm có thay**, và nhóm đó chỉ chiếm 8/34 mẫu `dung_duoc` của `khuon_thay`. Các con số
chất lượng **đáng tin hơn** tôi đã nói. Một ví dụ nổi bật **không phải** một tỷ lệ.

Và z = 1,91 < 1,96 ⇒ ngay cả 12,1đ cũng **chưa tách được** ở n = 21. Tôi đã ghi trước rằng n
này nhỏ; đúng là nhỏ.

### Phát hiện dùng được: lỗi **chỉ** nằm ở NIÊN ĐẠI

Trong cả 66 mẫu, **tên người và địa danh đều khớp**. Bốn lỗi:

| | |
|---|---|
| T017 | `1071` — đúng là **1077** *(mẫu E071 kể cùng trận, ghi **đúng** 1077)* |
| E150 | `19/7/1864` — đúng là **19–20/8/1864** *(mẫu T150 cùng nội dung, ghi **đúng** tháng 8)* |
| T188 | hoán thuộc tính **Thành nhà Hồ ↔ Cổ Loa** |
| T222 | `người làm **con**` — nguyên văn `người làm **tôi**`, mất thế đối |

Hai lỗi đầu là **cặp**: cùng một sự kiện, một bản đúng một bản lệch. Đó là **bằng chứng trực
tiếp** rằng phép thay **làm hỏng một niên đại vốn tồn tại đúng ở chỗ khác trong corpus**.

⇒ Việc chữa **cụ thể và rẻ**: **cấm thay ở khe mang nhãn `số`** — nhãn đó **đã dựng sẵn** ở
§7. Nhắm đúng chỗ lỗi đo được, không phải phỏng đoán.

⚠️ Nhưng phải nói rõ tỷ lệ: sai-sự-thật đáng **≈12 điểm** trên một nhóm nhỏ, trong khi phép
thay làm hỏng **−42,5 điểm** ở thước chính. Cấm thay khe số **không cứu** được TK_047; nó chỉ
là một chỗ chữa đúng trong một cơ chế đã đóng.

## 19. Trạng thái thước sau P101 + P102

| | |
|---|---|
| đồng thuận từng mẫu | **83,3%**, kappa **0,614** — dùng được |
| **hiệu chuẩn vùng giữa** | 🔴 **trôi 23đ giữa hai phiên** ⇒ **cấm so bắc cầu** |
| kiểm sự thật | ✅ đo được, đối chứng sạch 2,2% |
| **phóng đại do bỏ qua sự thật** | **≈12đ** trên nhóm có thay, **chưa tách** (z=1,91) |
| **neo** | ✅ đã dựng, 40 mẫu, bắt buộc kèm mọi vòng từ nay |

Thước giờ **có công cụ đủ**: đồng thuận đo được, độ trôi đo được (nhờ neo), sự thật đo được.
Đó là thứ ba ticket vừa qua **không** có.

## 20. P103 KẾT QUẢ (M4, v12) — hư hại **THU HẸP 2/3**, TK_047 **MỞ LẠI**

**Độ trôi trên neo: −12,5đ** (phiên này 55,0% vs tham chiếu 67,5%) — lần đầu đo được độ trôi
của một phiên. Neo hoạt động.

| nhánh | n | dùng được | tạm | hỏng | **dùng+tạm** |
|---|---:|---:|---:|---:|---:|
| `chep_cau` | 120 | 30,0% | 20,8% | 49,2% | **50,8%** |
| `ep_ngau_nhien` | 120 | 11,7% | 25,0% | 63,3% | **36,7%** |
| `ep_chu_de` | 120 | 18,3% | 20,8% | 60,8% | **39,2%** |

| | v12 | mini | |
|---|---:|---:|---|
| `ep_ngau_nhien` − `chep_cau` | **−14,2đ** (z = −2,21 TÁCH) | −42,5đ | 🔴 **thu hẹp 2/3** |
| `ep_chu_de` − `ep_ngau_nhien` | +2,5đ (z = 0,40) | +0,0đ | vẫn **không tách** |

✅ Theo đúng tiền-đăng-ký: *"khoảng cách **thu hẹp rõ** ⇒ hàm truyền **có thật** ⇒ TK_047 **mở
lại**, bài toán thành bài toán **dữ liệu**."*

**Dự đoán của tôi lại sai theo hướng bi quan** — đoán −20 đến −35đ, thật **−14,2đ**. Hai vòng
liên tiếp (P102, P103) tôi ước tính **quá xấu**. Ghi lại như một thiên lệch có hệ thống của
chính tôi, không phải nhiễu.

### Hàm truyền — nay có HAI điểm

| độ chính xác cấp phép | hư hại |
|---:|---:|
| 23,4% *(mini)* | −42,5đ |
| **43,8%** *(v12)* | **−14,2đ** |

Độ dốc **+1,39đ trên mỗi điểm %** độ chính xác ⇒ **hư hại = 0 tại ≈ 54%**.

Ghép với độ dốc dữ liệu của P100 (15,0đ/decade):

| mục tiêu | dữ liệu cần |
|---|---|
| độ chính xác 50% | ×2,6 ⇒ ~**1,1 triệu cha** |
| **độ chính xác 54% (hư hại = 0)** | **×4,8 ⇒ ~2,1 triệu cha** |

Lần đầu trong ba ticket có một **lộ trình định lượng** thay vì một hướng đi mò.

### ⚠️ Ba điều làm yếu kết luận

**1. Giám khảo phiên này CHẶT hơn 12,5đ.** Chặt hơn ⇒ nhiều mẫu dồn về `hong` ⇒ **khoảng cách
giữa hai nhánh có thể bị NÉN**. Nghĩa là −14,2đ có thể là **ước lượng thấp** của hư hại thật.

Kiểm chéo bằng cột `dung_duoc` (ít phụ thuộc ranh `tam`/`hong` hơn):

| | `chep_cau` | `ep_ngau_nhien` | chênh |
|---|---:|---:|---:|
| mini | 39,2% | 10,0% | −29,2đ |
| **v12** | 30,0% | 11,7% | **−18,3đ** |

Vẫn thu hẹp, **37%** thay vì 67%. ⇒ Hướng **chắc chắn đúng**, **biên độ thì không**: thu hẹp
nằm đâu đó giữa **37% và 67%**. Không được trích riêng con số 2/3.

**2. Ngoại suy hai điểm.** Hàm truyền tuyến tính là **giả định**, không phải phép đo. "54%" và
"2,1 triệu cha" là **định hướng độ lớn**, không phải chỉ tiêu.

**3. Tiêu chí chọn vẫn vô giá trị** (+2,5đ, z = 0,40). Sáu vòng liên tiếp. Cải thiện đến **hoàn
toàn** từ *mật độ dữ liệu*, **không** từ cơ chế chọn nào.

### Trạng thái TK_047: MỞ LẠI, với điều kiện

Cơ chế khuôn–khe **không hỏng về nguyên lý** — nó **đói dữ liệu**. Nhưng nó vẫn **chưa thắng**:
trong cùng vòng, mọi nhánh có thay vẫn **thua** phát nguyên văn. Muốn hoà cần độ chính xác
~54%, tức **~5× dữ liệu hiện có**.
