# TK_048 — NƠ-RON LỚP THAY THẾ: trục *thay-thế-được*, và câu hỏi TOKEN vs CỤM

> Mở 2026-08-05. Nối tiếp TK_046 (loại suy) và TK_047 (khuôn–khe, đã đóng).

## 1. Chỗ trống — đo được, không phải đoán

| có gì | |
|---|---|
| `E` | **bạn cùng NGỮ CẢNH** (syntagmatic) — TK_046 §16 đo trực tiếp |
| nhãn từ loại | **8 lớp thô** — `khỉ` và `bàn` cùng là *danh từ*; giúp **1,03×** |
| 🔴 **thiếu** | biểu diễn *"hai đơn vị này **thay thế được** cho nhau"* (paradigmatic) |

Ý tưởng gốc của tác giả (`khỉ` ~ `chó` ~ `mèo` ⇒ suy ra `là động vật`) đòi **đúng tầng giữa**.
TK_046 đo được nó thắng `E` **1,60×** ở xếp hạng, **tái lập chính xác hai lần** — nhưng **chưa
bao giờ được đông cứng thành nơ-ron**: mỗi lần tính lại tại chỗ rồi vứt đi.

**Định nghĩa** (thuần đếm): bối cảnh của đơn vị `x` = tập `(trước, sau)`;
`sim(x,y) = |ctx(x) ∩ ctx(y)| / √(|ctx(x)|·|ctx(y)|)`, và **đòi ≥3 bối cảnh chung**.

## 2. P104 mức TOKEN — trục ĐÚNG, bằng chứng MỎNG

| đơn vị | **LỚP THAY THẾ** | **`E`** |
|---|---|---|
| **`Nguyễn`** | **Lý, Trần, Lê, Hồ, Tống, Triệu, Đinh, Minh** | Nguyễn, Hữu, Huệ, Phúc, Đặng |
| `vua` | tức, Hậu, tướng, Đường, Tống, Trần, Lê, ông | Tông, ngôi, Lý, Thánh, vương |
| `lớn` | thấp, nữa, **cao, nhỏ**, tư, ấy, biển | nhất, phần, nhiều, vùng, nhỏ |

Dòng `Nguyễn` là minh chứng sạch: trục mới trả về **các triều đại thay nhau được**; `E` trả về
**các mảnh của chính tên đó**. Hai trục **khác nhau về bản chất**, đúng như §16 dự đoán.

**Phủ: 394/2.680 = 14,7%** đơn vị có ≥1 láng giềng. Vùng bốc ngẫu nhiên thì mỏng và nhiễu
(`lý → tử`, `Giáp → Mông`).

### Hai lỗi của tôi

1. **Gom cụm THẨM THẤU.** Union-find trên cạnh top-8 hai chiều ⇒ **một cụm 2.061/2.680 (77%)**.
   Đó là **percolation** của đồ thị bậc ~6, không phải tính chất tương đồng. `A~B` và `B~C`
   **không** kéo theo `A~C` ở quan hệ thay-thế ⇒ nơ-ron này là **vùng láng giềng của từng đơn
   vị**, **không** phải một phân hoạch.
2. **Token hiếm thổi lên.** Bản đầu nhận **một** bối cảnh chung, và `1/√|ctx(y)|` rất lớn khi
   `y` hiếm ⇒ `vua → kali, xoáy, modem`. Mà "qua **nhiều** bối cảnh" vốn là **cốt lõi ý tưởng**
   — tôi cài sai ngay từ đầu.

## 3. 🔴 Quan sát của tác giả: **mọi thứ tới giờ là TOKEN ĐƠN LẺ**

> *"hiện chúng ta đang là token sang token, mà không phải là cụm… chúng ta đã có cơ chế để sinh
> nơ-ron cụm rồi, nhưng đã thực sự dùng đến chưa?"*

Kiểm tra **xác nhận, và còn mạnh hơn**:

| cơ chế nơ-ron cụm | có trong code | **trong checkpoint mini** |
|---|---|---|
| `entity_id()` — cụm tên riêng (TK_027) | ✅ | ❌ **0** |
| `concept_id()` · `sense_id()` · `label_id()` | ✅ | ❌ **0** |
| `FRAME_BRACKET` | ✅ | chỉ `Hỏi:`/`Đáp:` |

⇒ Toàn bộ TK_045/046/047 chạy trên **token**. Hạ tầng cụm nằm đó **chưa từng được dùng**.

## 4. P104 mức CỤM — **chính xác vọt lên, phủ SỤP ĐỔ**

Đơn vị = **TỪ** (WS-CRF) rồi ghép tiếp hai đơn vị liền nhau khi **PMI ≥ 3** (2.407 cặp được
ghép).

| | token | **cụm** |
|---|---:|---:|
| đơn vị có ≥3 bối cảnh | 2.680 | 3.146 |
| bối cảnh phân biệt | 50.010 | 27.197 |
| **phủ (có ≥1 láng giềng)** | **14,7%** | 🔴 **0,2%** (7/3.146) |

Nhưng **7 đơn vị sống sót thì hoàn hảo**, không một nhiễu:

```
Lịch sử lớp   → Toán lớp, Khoa học lớp, Địa lý lớp
Khoa học lớp  → Toán lớp, Lịch sử lớp, Địa lý lớp
Địa lý lớp    → Toán lớp, Lịch sử lớp, Khoa học lớp
```

So với mức token, nơi `vua → tức, Hậu, tướng` lẫn nhiễu **quá nửa**.

⇒ **Trực giác của tác giả ĐÚNG về chất lượng**: đơn vị cụm cho lớp **sạch hơn hẳn**. Nhưng trên
`mini` nó **đói dữ liệu tuyệt đối**: đơn vị nhiều hơn, mỗi đơn vị ít lượt hơn, bối cảnh gần như
duy nhất ⇒ "≥3 bối cảnh chung" hầu như không xảy ra.

⚠️ n = 7 **không kết luận được gì** về độ chính xác. Đây là **minh chứng khả năng**, không phải
phép đo.

## 5. Việc kế

Cả ba đường đo gần đây chỉ về **cùng một chỗ**:

| | mini | v12 |
|---|---|---|
| độ chính xác cấp phép (P100) | 23,4% | **43,8%**, chưa phẳng |
| phủ lớp thay thế — token | 14,7% | chưa đo |
| phủ lớp thay thế — **cụm** | **0,2%** | chưa đo |

Lớp thay thế mức cụm cần **≥3 bối cảnh chung**, mà số bối cảnh chung tăng **nhanh hơn tuyến
tính** theo mật độ dữ liệu ⇒ đây là ứng viên **hưởng lợi nhiều nhất** từ v12, và cũng là thứ
**không thể** kết luận trên mini.

⇒ Giao M4 chạy P104 trên v12 ở **cả hai mức đơn vị**, sau khi P103 xong.

## 6. P104 trên **v12** (M4) — phủ vọt lên; nhưng ưu thế của CỤM **không tái lập**

| | mini | **v12** |
|---|---:|---:|
| đơn vị — token | 2.680 | **23.273** |
| **phủ — token** | 14,7% | **49,2%** |
| đơn vị — cụm | 3.146 | **91.600** |
| **phủ — cụm** | 0,2% | **6,4%** (5.853) |
| cặp ghép PMI≥3 | 2.407 | 191.031 |

Phủ tăng mạnh ở **cả hai** mức — đúng dự đoán rằng đây là cơ chế **đói dữ liệu**.

### Mức TOKEN trên v12 — vùng láng giềng nay **thật sự tốt**

```
muốn → phải, cần, sẽ, dám, biết, nghĩ          ← lớp động từ tình thái
luôn → vẫn, sẽ, chỉ, đều, còn, đang            ← lớp phó từ
hiểu → biết, thấy, kiếm, đọc, tòi, nghe        ← lớp động từ nhận thức
tự   → bỗng, đột, đương, ngẫu, dĩ              ← tất cả đều là `X nhiên`
hơn  → nhất, khác, lắm, khoảng                 ← lớp so sánh
```

Đây là **chất lượng khác hẳn** mini, nơi `vua → tức, Hậu, tướng` lẫn nhiễu quá nửa.

### 🔴 Nhưng ưu thế của CỤM **KHÔNG tái lập**

Trên mini tôi báo 7 cụm **hoàn hảo** (`Lịch sử lớp → Toán lớp, Khoa học lớp…`) và suy ra *"đơn
vị cụm cho lớp sạch hơn hẳn"*. Trên v12, với **5.853** đơn vị cụm thay vì 7, vùng láng giềng
lại **nhiễu**:

```
vấn đề  → tập hợp con, vườn quốc gia, phiên bản miễn phí, nhà đàm phán
chơi    → kiếm sống, trả nợ, nhìn thấu, chia sẻ, an toàn hơn
bảo vệ  → khuyên dùng, bao nhiêu tiền, cấp phép, vài tháng
```

⇒ Suy luận của tôi từ **n = 7** là **may mắn cỡ mẫu nhỏ**, không phải quy luật. Tôi đã ghi
"n=7 không kết luận được gì" ngay lúc đó — và đúng là không kết luận được. **Ở quy mô đo được,
mức CỤM vẫn thua mức TOKEN cả về phủ (6,4% vs 49,2%) lẫn về độ sạch.**

Điều này **không** bác quan sát gốc của tác giả (mọi thứ đang là token đơn lẻ — vẫn đúng, và
hạ tầng nơ-ron cụm vẫn chưa dùng). Nó chỉ nói: **cách ghép cụm bằng PMI trên bigram chưa phải
cách đúng**. 191.031 cặp được ghép là quá nhiều và quá thô.

### ⚠️ Một lỗi của tôi trong probe

Hàng "từ dò" của bản v12 **KHÔNG đọc được**: tra cứu `{name(u): u}` bị **va chạm** — `ĠNguyễn`
(đầu từ) và `Nguyễn` (mảnh giữa từ) đều rút gọn về `"Nguyễn"`, dict giữ cái **đến sau**. Trên
mini tình cờ đúng; trên v12 nó chọn biến thể hiếm (**18** bối cảnh thay vì hàng trăm) ⇒ các ô
trống trong bảng v12 là **hiện vật**, không phải kết quả.

Đã sửa (khi trùng tên, giữ đơn vị **nhiều bối cảnh nhất**) và kiểm lại trên mini: `Nguyễn → Lý,
Trần, Lê, Hồ, Tống, Triệu, Đinh` trở lại đúng. **Bảng "từ dò" trên v12 cần chạy lại**; các con
số **phủ** và các **vùng bốc ngẫu nhiên** không bị ảnh hưởng vì chúng không dùng đường tra cứu đó.

## 7. Ghép cụm bằng **THỰC THỂ** thay vì PMI — sạch hơn hẳn, và tách được nguyên nhân

§6 kết luận *"ghép cụm bằng PMI trên bigram chưa phải cách đúng"*. Bản thay thế: ghép **chuỗi
TÊN RIÊNG liền nhau** thành một đơn vị. Ranh giới đến từ **lớp ký tự** (§7 TK_047 — thống kê
viết hoa GIỮA câu, **không** danh sách tên nào), nên nó là ranh giới **thật** của một thực thể,
không phải một ngưỡng thống kê.

Không dùng `ingest_entities` của TK_027 vì nó cần `statements_doc.jsonl` có field `title` — tức
**phụ thuộc corpus ngoài**. Quy tắc ký tự thì không phụ thuộc gì.

### Ba cách chia đơn vị, cùng trên mini

| đơn vị | số đơn vị | bối cảnh | **phủ** | số phép ghép |
|---|---:|---:|---:|---:|
| **token** (BPE ~âm tiết) | 2.680 | 50.010 | **14,7%** | — |
| **thực thể** (TỪ + chuỗi tên riêng) | 2.580 | 32.420 | **3,1%** | **318** |
| **cụm PMI** (TỪ + ghép PMI≥3) | 3.146 | 27.197 | **0,2%** | 2.407 |

⇒ Ghép thực thể **hơn ghép PMI 15×** về phủ, với **1/8 số phép ghép**. Và lớp thu được **sạch
hơn nhiều**:

```
khi   → lúc                    Bắc → bắc, nam, đông        (lớp phương hướng)
lúc   → khi                    Mạc → Đường                 (triều đại)
tên   → niên hiệu              Hải Vương → Hoả             (Sao Hải Vương / Sao Hoả)
```

So với cụm PMI trên v12: `vấn đề → tập hợp con, vườn quốc gia, phiên bản miễn phí`.

### 🔴 Nhưng phải tách đúng nguyên nhân — và nó **không** phải phép ghép

`thuc_the` chỉ ghép **318** chuỗi, tức gần như **không đụng gì** so với chia theo TỪ. Vậy mà
phủ vẫn tụt **14,7% → 3,1%**.

⇒ **Giá phải trả nằm ở việc chuyển từ ÂM TIẾT sang TỪ, không nằm ở phép ghép cụm.** Đơn vị TỪ
hiếm hơn đơn vị âm tiết, nên "≥3 bối cảnh chung" khó xảy ra hơn nhiều. Ghép thực thể **thêm**
độ chính xác mà **gần như không thêm** chi phí thưa; PMI thì thêm rất nhiều chi phí (2.407 phép
ghép ⇒ phủ 0,2%) mà không thêm độ chính xác.

Đây là chỗ tôi đọc sai ở §4: tôi quy cho "đơn vị cụm" cái giá vốn của **phân đoạn TỪ**.

### Việc kế

Chạy `--unit thuc_the` trên **v12**. Dự đoán ghi TRƯỚC: phủ **12–25%** (mini 3,1%; `cum` đã đi
0,2% → 6,4% khi lên v12, mà `thuc_the` xuất phát cao gấp 15× và thưa ít hơn). Nếu nó vượt phủ
của **token trên mini (14,7%)** *và* giữ được độ sạch, thì đơn vị-thực-thể là cách chia đúng —
và quan sát gốc của tác giả được xác nhận bằng số, không phải bằng ví dụ.

## 8. P105 trên v12 (M4) — **G-subst ĐẠT**, và tôi đoán thấp lần thứ ba

Mọi cấu hình chạy **hai lần, giống hệt nhau**.

| đơn vị | phủ (mini → **v12**) | xác nhận trên B | đối chứng |
|---|---:|---:|---:|
| `token` | 14,7% → **49,2%** | **90,6%** | 14,0% (6,5×) |
| **`thuc_the`** | 3,1% → **34,8%** | **84,5%** | 5,4% (15,6×) |
| `cum` (PMI) | 0,2% → **6,4%** | 63,6% | 1,0% |

✅ **Cổng (1)(2)(3) đều ĐẠT.** `thuc_the` hơn `cum` ở **cả** phủ (5,4×) lẫn độ chính xác
(84,5% vs 63,6%) ⇒ **ranh giới thật hơn ngưỡng thống kê**, §7 đúng.

Ví dụ `thuc_the` trên v12: `năm → tháng, thế kỷ, ngày` · `lớn → nhỏ, quan trọng, cao` ·
`vua → Hoàng đế, giáo hoàng`.

🔴 Dự đoán của tôi 12–25% → thật **34,8%**. **Lần thứ ba liên tiếp** đoán thấp (P102, P103,
P105). Đây là **thiên lệch có hệ thống của tôi**, không phải may — ghi lại để hiệu chỉnh, không
phải để bào chữa.

## 9. Nối TK_048 vào TK_047 — và **so sánh 84,5% vs 43,8% là SAI THƯỚC**

Tôi định đọc *"lớp thay thế 84,5% chính xác gấp đôi cấp phép bộ ba 43,8%"*. **Không được**:
hai thước hỏi hai câu khác nhau.

| thước | hỏi gì |
|---|---|
| lớp (§8) | `y` điền được **MỘT** bối cảnh nào đó của `x` ở B không |
| bộ ba (TK_047) | **ĐÚNG** bộ ba `(a, c, b)` có ở B không — **chặt hơn hẳn** |

Bắt lớp thay thế chịu **đúng thước của bộ ba**, trên mini, cùng bộ ca:

| cấp phép | **hẹp** | **chủ đề** | khe thay được | tổ hợp mới |
|---|---:|---:|---:|---:|
| **bộ ba** (TK_047) | **23,1%** | 80,9% | 22,4% | 24× |
| **lớp** (TK_048) | 🔴 **9,4%** | **96,9%** | 18,0% | 18× |
| **GIAO cả hai** | ✅ **48,1%** | ✅ **97,6%** | 🔴 3,7% | 🔴 3× |

⇒ Lớp thay thế **một mình thì TỆ HƠN** bộ ba ở khớp bối cảnh cục bộ (9,4% vs 23,1%) — đúng như
nghi ngờ, và nếu tôi đọc thẳng 84,5% vs 43,8% thì đã kết luận **ngược**.

Nhưng hai cơ chế **giỏi hai việc khác nhau**: bộ ba khớp **cục bộ**, lớp khớp **ngữ nghĩa**
(96,9%). **Giao** chúng lại thì độ chính xác **gấp đôi** (23,1% → **48,1%**) và chủ đề lên
**97,6%**.

⚠️ Đáng chú ý: **48,1% trên mini đã VƯỢT 43,8% của bộ ba trên v12.** Và hàm truyền của TK_047
§20 nói hư hại = 0 tại **≈54%**.

### 🔴 Nhưng ràng buộc chỉ ĐỔI CHỖ, chưa biến mất

| | trước | sau khi giao |
|---|---|---|
| độ chính xác | 🔴 23,1% | ✅ 48,1% |
| khe thay được | 22,4% | 🔴 **3,7%** |
| **tổ hợp mới** | 24× | 🔴 **3×** |

Cổng tổ hợp mới là **≥10×**; 3× nằm sát vùng *"chép có trang trí"*. Ta **đổi một cổng trượt lấy
một cổng trượt khác**. Đó là **tiến bộ chỉ khi** phủ/tổ hợp phục hồi được bằng dữ liệu — và có
cơ sở tin thế: `thuc_the` đã đi 3,1% → **34,8%** (11×) khi lên v12, còn mini thì cả hai thành
phần đều đói.

⚠️ Đối chứng cũng **tăng** ở nhánh giao (hẹp 10,8%, chủ đề 53,2%) vì bể ứng viên nhỏ và chọn
lọc ⇒ **tỷ số co lại**. Phải đọc **mức tuyệt đối trước**, đúng bài học TK_047.

### Việc kế

Giao M4 chạy `--licence giao` trên **v12**. Dự đoán ghi TRƯỚC — và lần này tôi **nâng** ước
tính vì đã sai thấp ba lần: **hẹp 60–75%**, **khe thay được 12–25%**, **tổ hợp mới 6–15×**.
Nếu hẹp vượt **54%** *và* tổ hợp mới ≥10× thì hàm truyền TK_047 §20 dự đoán phép thay
**không còn gây hại** — và đó là lần đầu cả hai cổng cùng đạt.

## 9b. KẾT QUẢ M4 — giao v12 **ĐẠT hai cổng quyết định**, nhưng ép-1-khe **không** mua chất lượng

| cấp phép (v12, full) | hẹp | khe thay được | tổ hợp mới |
|---|---:|---:|---:|
| `bo_ba` | 43,8% | 68,1% | rất lớn |
| `lop` | 9,8% | 96,4% | — |
| **`giao`** | **70,6%** | **30,1%** | **72×** |

✅ Hẹp ≥50% · tổ hợp ≥10× · hẹp >54% (mốc hư hại≈0 §20). Hai lần chạy giống hệt.

Ép thay **đúng 1 khe** (cùng thủ tục P103, thêm nhánh `ep_giao`):

| | dùng+tạm | vs `chep_cau` |
|---|---:|---:|
| `chep_cau` | 50,0% | — |
| `ep_bo_ba` | 42,5% | **−7,5đ** |
| `ep_giao` | 41,7% | **−8,3đ** |
| `ep_giao − ep_bo_ba` | — | **−0,8đ** |

⇒ Substrate giao **sống**. Thủ tục “chép một cha rồi đục **một** lỗ” **không** phải cơ chế sáng tạo
của tác giả — hẹp 70% không kéo chất lượng hơn bộ ba. Việc kế **không** còn là lọc decode thêm;
là **đông cứng lớp + điền nhiều khe**. → **TK_049**.

## 10. Việc kế (đóng TK_048 substrate → mở TK_049 gen)

Xong substrate (giao hẹp, G-subst). Gen đúng kiến trúc: bake `SYNONYM` + `Entity:` rồi
`ep_da_khe_giao` — xem `TK_049_FRAME_SLOT_ENTITY_GEN.md` + `RUNBOOK_TK_049_P106.md`.
P106 (2026-08-06): bake cổng 1–2 đạt · smoke cổng 3 đạt (`n_thay≥2` = 87,5%). Chất lượng = P107.

P107: `ep_da − chep = −7,5đ` (cổng 4 trượt hẹp) · `ep_da − ep_mot = +9,2đ` · không-chép 92,5%.
→ đa-khe thắng 1-khe; chưa sát trần chép. Hebbian = P108.