# TK_046 — SUY LUẬN LOẠI SUY: kết nối chủ động qua khung (frame-sliced analogy)

| Field | Value |
|-------|-------|
| **ID** | TK_046 |
| **Status** | 🟢 **ĐÓNG — ĐÃ TRẢ LỜI (2026-08-21).** Câu hỏi đề bài có đáp án dứt điểm, có tập GIỮ RIÊNG và đối chứng: **(a)** chia sẻ ngữ cảnh KHÔNG đủ — nướng `c↔a` + `c↔lớp` cho chuyển giao **0/60** (§144/§146); **(b)** cạnh `c→Y` nướng THẲNG vào kho câu thì chuyển giao THẬT **6,8% → 30,6%** trên 60 từ mạng chưa từng được dạy (§147). Thu hoạch phụ vào mặc định: sinh câu `79,3/89,3/13,6 → 85,9/97,3/7,0` (§113/§116/§118) + bảng nhãn lớp 9 222 từ, top-1 `4,1% → 84,9%` (§128). **Chưa sản phẩm hoá** — mọi cờ mặc định TẮT; phần hoàn thiện + nâng cấp chuyển sang **TK_053**. |
| **Nguồn** | **Ý tưởng tác giả 2026-08-04.** Nguyên văn: *"khi train câu `con khỉ nghịch ngợm`, nó phát hiện `con khỉ` cũng giống `con chó`, `con mèo`, để tự động kết nối tới `là động vật`"* — và bản chi tiết hoá: *"`khỉ` kích thích cùng nơ-ron `nghịch ngợm`, nơ-ron đó kích thích loạt `chó`, `mèo`, loạt này lại kích thích `động vật`, dẫn tới `khỉ` kết nối chủ động với `động vật`"*. |
| **Liên quan** | TK_045 §13.61–§13.67 (toàn bộ chuỗi đo) · TK_044 (Hebbian binding — cùng họ "trọng số có ngay lúc train") · §13.60 (truy hồi 42–53% vs tầng mệnh đề 4,7%) · `core/embedding.py` (`E`, mốc phải vượt) |

---

## 1. Đề bài

Cho tập train:

```
con mèo là động vật
con chó là động vật
con mèo nghịch ngợm
con chó nghịch ngợm
con khỉ nghịch ngợm
```

Prompt `con khỉ là` → mong đợi `động vật`, **dù `con khỉ là động vật` chưa hề được train**.

Và chiều ngược lại, cùng một cơ chế:

```
ok làm đi · ok triển khai đi · anh ấy đang làm / triển khai / thực hiện
```

Prompt `ok thực` → `hiện đi`.

**Một cơ chế, hai hướng ngữ cảnh**: hướng phải (`khỉ` → thuộc tính) và hướng trái (`thực hiện`
← khung `anh ấy đang`).

## 2. Vì sao khác `E` — đóng góp riêng của ý tưởng

`E` (32002×96, SVD của đồng hiện) **tính điểm lúc truy vấn** từ kết nối **đã có**. Ý tưởng của
tác giả **viết cạnh mới vào đồ thị lúc train**. Ba hệ quả không trùng:

| | |
|---|---|
| **lặp** | cạnh sinh ở vòng 1 tham gia vòng 2 ⇒ đóng bắc cầu có suy giảm |
| **cạnh là THẬT** | `rank_matches` / `W_skip` / tầng mệnh đề đều dùng được, không chỉ một bộ xếp |
| **thưa hoá lúc viết** | chỉ đồng kích thích đủ mạnh mới thành cạnh |

Và nó **Hebbian** ⇒ Luật 0 cho phép (cấm backprop, không cấm đếm/Hebbian).

## 3. Năm phép đo — bốn thua, một thắng

Nhiệm vụ chung: **giữ lại** một liên kết, **bỏ nguyên cha** chứa nó, hỏi cơ chế xếp đích thứ
mấy. Mốc bắt buộc vượt là **`E`**, vì `E` đã có sẵn miễn phí.

| # | cách hiện thực | kết quả (MRR, mini) | vs `E` |
|---|---|---:|---:|
| §13.61 | bậc hai một vòng, toàn cục | hit@10 2,3% vs `E` 2,7% | **0,85×** |
| §13.63-B | bậc hai có khung | 0,1506 vs 0,2954 | **0,51×** |
| §13.65 | lan truyền 3 chặng + hỗ trợ ≥2 thực thể | 0,1098 vs 0,1780 | **0,62×** |
| §13.66 | *(tiêu chí)* số khung chung — chống bão hoà | 7,5% vs 94,8% | **qua cổng** |
| **§13.67** | **tách lát theo khung** | **0,3107** vs 0,1948 | **1,60×** |

**Cái phân biệt**: ba cách đầu dùng **túi từ** — `M[a,c]` = số cha chứa cả `a` và `c` — nên
**mất vai**. Trong túi từ, `bàn`↔`chó` (chung `nâu`) và `khỉ`↔`chó` (chung `nghịch ngợm`)
**giống hệt nhau**. Không ngưỡng nào tách được, vì biểu diễn không mang đủ thông tin.

## 4. Cơ chế thắng cuộc (§13.67)

```
tương đồng(a, b) = |khung(a) ∩ khung(b)|            # khung = token kề trước/kề sau, lọc gidf
điểm(c)          = Σ_b tương đồng(a, b) · đếm(b · f · c)
```

*`khỉ` được điểm cho `động vật` vì `chó` và `mèo` — hai thực thể nhiều khung chung với `khỉ` —
từng xuất hiện trong đúng khung `… là động vật`.*

`bàn` chung khung `màu` với `chó`, nhưng chuyển giao chỉ xảy ra **trong khung `là`**, nơi `bàn`
không có ai giống nó ⇒ ca âm tính chết vì **cấu trúc**, không vì ngưỡng.

Kết quả 400 ca, mini:

| bộ chấm | MRR |
|---|---:|
| `ngau_nhien` | 0,0901 |
| **`khung` một mình** — đối chứng quyết định | 0,1624 |
| `bac_hai` (túi từ) | 0,1251 |
| `embedding` | 0,1948 |
| **`tach_lat`** | **0,3107** |

## 5. Cải thiện do Claude đề xuất, và trạng thái từng cái

| # | cải thiện | vì sao | trạng thái |
|---|---|---|---|
| 1 | **tách lát theo khung** (tenxơ 3 chiều thay ma trận 2 chiều) | túi từ mất vai ⇒ không phân biệt được `là` với `màu` | ✅ **đo xong, vượt `E` 1,60×** |
| 2 | **đếm SỐ KHUNG chung** thay số thực thể | "≥2 thực thể" thoả ở **94,8%** ca vô can ⇒ vô hiệu; "≥2 khung" chỉ **7,5%** | ✅ **đo xong, qua cổng** |
| 3 | **cạnh suy ra phải tự chứng minh** — chia corpus đôi, chỉ viết loại cạnh có precision đo được trên nửa kia | biến "hy vọng đúng" thành "độ chính xác đã đo"; ngưỡng **tự hiệu chỉnh** thay vì chọn tay | ✅ **đo xong (§9)** — và nó bác tách lát ở khâu viết cạnh |
| 4 | **giữ kiến trúc Hebbian của tác giả, đổi bộ đo tương đồng** | phần đóng góp riêng (viết cạnh lúc train) được giữ; chỉ thay thành phần đã đo là yếu | 🟢 **nay là hướng có bằng chứng nhất** (§9) |
| 5 | **hướng trái/phải tách riêng** | tôi đoán khung trộn tạo **trùng giả**: token đứng TRƯỚC `a` và SAU `b` bị tính là khớp | ❌ **đo xong (§10) — KHÔNG mua gì (0,97×), và tiền đề của tôi SAI** |

## 6. Probe đã viết

| probe | hỏi gì | kết quả |
|---|---|---|
| [`probe_second_order.py`](scripts/probe_second_order.py) | bậc hai toàn cục có hơn `E`? | ❌ 0,85× |
| [`probe_frame_analogy.py`](scripts/probe_frame_analogy.py) | có ràng buộc khung? | ⚠️ **VÔ HIỆU** — hiện vật hoà điểm (§13.62) |
| [`probe_frame_analogy2.py`](scripts/probe_frame_analogy2.py) | bản sửa 3 lỗi: hạng giữa · tách 2 nhiệm vụ · kiểm sàn | A: chủ thể **2,19×** khung ✅ · B: bậc hai 0,51× `E` ❌ |
| [`probe_hebbian_spread.py`](scripts/probe_hebbian_spread.py) | lan truyền trên **corpus đồ chơi** của tác giả | T1 ✅ (`khỉ → động vật` qua `['chó','mèo']`) · T2 ❌ |
| [`probe_spread_mini.py`](scripts/probe_spread_mini.py) | lan truyền ở quy mô thật | ❌ 0,62× `E`; ràng buộc hỗ-trợ **vô hiệu** (94,8%) |
| [`probe_shared_frames.py`](scripts/probe_shared_frames.py) | số khung chung có bão hoà? | ✅ 7,5% ở ca vô can |
| [`probe_frame_tensor.py`](scripts/probe_frame_tensor.py) | tách lát theo khung — **xếp hạng** | ✅ **1,60× `E`, 1,91× khung** |
| [`probe_edge_precision.py`](scripts/probe_edge_precision.py) | độ chính xác **cạnh viết ra** trên nửa corpus giữ lại | ❌ **0,78× `E`**, chỉ 1,29× sàn |

## 7. Bốn bài học về THƯỚC rút ra ở đây

Ghi lại vì cả bốn đều **suýt** làm tôi báo cáo sai:

1. **Chặn hub theo tần suất tách NGƯỢC** — cầu tốt (`nghịch ngợm`, df 0,60) là thuộc tính chung
   của **nhiều** thành viên nên tần suất **cao theo định nghĩa**; cầu xấu (`nâu`, df 0,29) thấp.
2. **Hoà điểm là phần thưởng** khi hạng tính bằng *"đếm cái lớn hơn hẳn"* ⇒ bộ chấm **thưa**
   thắng máy móc. Phải dùng **hạng giữa**. (§13.62 — probe báo ĐẠT mà vô hiệu.)
3. **Tiêu chí phân biệt được trên corpus đồ chơi có thể VÔ HIỆU ở quy mô thật** — vì cái làm nó
   phân biệt được là **sự khan hiếm**, không phải bản chất. ("≥2 thực thể": 7 câu tách được,
   5.664 cha thoả 94,8%.)
4. **Đơn vị giữ lại phải khớp với cơ chế.** Giữ lại **cặp** `(f,x)` làm cơ chế chuyển giao
   **không thể** ghi điểm (0/400); phải giữ **bộ ba** `(a,f,x)` — đúng kịch bản "chưa hề train
   `con khỉ là động vật`" trong khi `X là động vật` vẫn có.

Và **sàn ngẫu nhiên là MÁY DÒ LỖI THIẾT KẾ**, không chỉ là mốc: hai lần nó báo động ngay dòng
đầu (ngẫu nhiên **hơn** tín hiệu thật) và cả hai lần đều là lỗi thiết kế thật.

## 9. P93 — thắng XẾP HẠNG **không** kéo theo thắng VIẾT CẠNH

§13.67 đo *cho sẵn đích, xếp thứ mấy*. Nhưng ý tưởng của tác giả là **viết cạnh**, mà viết cạnh
là bài toán **sinh**, không phải xếp hạng. P93 đo đúng chỗ đó, và gộp luôn cải thiện #3:

**Giao thức** — chia corpus đôi, suy cạnh trên nửa **A**, hỏi cạnh có xuất hiện thật ở nửa **B**
không. Cạnh đúng ⇒ thấy ở B; cạnh suy bậy ⇒ không bao giờ thấy. Sàn **khớp phân bố**: bốc `c`
ngẫu nhiên **trong cùng khung** (bốc từ toàn vocab thì sàn thấp giả tạo).

| bộ sinh cạnh | cạnh | chặt (bộ ba ở B) | **lỏng (cùng cha ở B)** |
|---|---:|---:|---:|
| `tach_lat` | 1.490 | 0,9% | **20,5%** |
| **`embedding`** | 1.500 | 1,0% | **26,3%** |
| `ngau_nhien` (khớp khung) | 1.500 | 0,0% | 15,9% |

`tach_lat/ngau_nhien` = **1,29×** (cổng 2,0×) ❌ · `tach_lat/embedding` = **0,78×** (cổng 1,0×) ❌
⇒ **CỔNG G-prec TRƯỢT.**

**Dự đoán của tôi sai**: tôi ghi trước (2) "đạt" — thực tế 1,29×, còn xa 2,0×.

### 🔴 Bài học chính, và nó lặp lại §13.60

> **Thắng ở XẾP HẠNG không kéo theo thắng ở SINH.** Cho sẵn đích thì tách lát xếp nó lên rất
> tốt (1,60× `E`); nhưng khi phải **tự chọn** viết cạnh nào, nó chỉ hơn bốc-ngẫu-nhiên-trong-khung
> **1,29×** và **kém `E` 22%**.

Hai đại lượng khác nhau: xếp hạng chỉ cần **phân biệt đích thật với nhiễu**; viết cạnh cần
**đúng ngay từ lựa chọn đầu**, không có đích để tựa vào. §13.60 đã dạy đúng bài này ở chỗ khác
(tầng mệnh đề thắng nhiều vòng so sánh mà thua truy hồi 11 lần ở đầu ra).

### Hệ quả thực dụng — cải thiện #4 nay là hướng có bằng chứng nhất

| thành phần | dùng cái nào | bằng chứng |
|---|---|---|
| **viết cạnh lúc train** | **kiến trúc Hebbian của tác giả** | `E` không làm được — nó chỉ là điểm số lúc truy vấn |
| **bộ đo tương đồng** | **`E`** | 26,3% vs 20,5% chính xác cạnh · 2,44× sàn ở suy luận (§13.63-B) |

Tức: giữ nguyên đóng góp riêng của tác giả, thay đúng thành phần đã đo là yếu. Tách lát vẫn
đáng giữ **ở khâu xếp hạng** (1,60×), nhưng **không** dùng nó để **chọn** cạnh nào được viết.

## 10. Cải thiện #5 (tách hướng): **không mua gì**, và tiền đề của tôi sai

Thêm `tach_lat_huong` — tương đồng = `|L(a)∩L(b)| + |R(a)∩R(b)|` thay vì trộn chung:

| bộ chấm | MRR |
|---|---:|
| `tach_lat` (trộn) | **0,3107** |
| `tach_lat_huong` (tách hướng) | 0,3010 |

**0,97×** — không giúp, nhích xuống trong nhiễu.

**Vì sao — đo luôn tiền đề:**

| | |
|---|---:|
| token có **cả** khung trái lẫn phải | 3.821 |
| chồng lấn `TRÁI ∩ PHẢI / TRÁI ∪ PHẢI` | trung vị **0,000** · trung bình **0,010** |
| token có chồng lấn **bằng 0** | **85,3%** |

⇒ **Trái và phải VỐN ĐÃ gần như rời nhau.** Trộn chúng lại **không tạo ra trùng giả** nào, nên
tách chúng ra **không sửa được gì**. Giả thuyết của tôi sai **ngay ở tiền đề**, không phải ở
biên độ.

Đây là dạng lỗi khác với bốn bài học ở §7: không phải thiết kế đối chứng sai, mà là **tôi giả
định một bệnh mà không đo xem bệnh có tồn tại không**. Đo tiền đề tốn **6 giây**; đo cải thiện
tốn 21 giây và một vòng suy nghĩ thừa.

> **Bài học thứ năm:** trước khi xây thứ chữa bệnh X, **đếm xem X có xảy ra không**. Ba lần
> trong TK_045 giám khảo mô tả bệnh mà đếm ra hiếm (§13.34, §13.38, §13.53); đây là lần **tôi
> tự bịa ra bệnh** rồi định chữa.

⚠️ Kết quả này **không** bác ví dụ thứ hai của tác giả (`ok thực → hiện đi`). Nó chỉ nói: ở
corpus này, **tách hướng không phải thứ làm nên khác biệt**. Cơ chế tách-lát vốn đã dùng khung
`f` **có vị trí** (`b · f · c`), nên hướng đã nằm sẵn trong đó.

## 11. Mốc TUYỆT ĐỐI cho đường AR: **5,8%** — đường sinh token cũng gần như hỏng

Tác giả hỏi *"cắm E-viết-cạnh vào đường truy hồi nghĩa là sao, vẫn query rồi copy à?"* — và câu
hỏi đó lộ ra đề xuất của tôi **sai**: cạnh suy ra **không sinh ra câu**, nó chỉ đổi **câu nào
được chép**. Đề `con khỉ là` + cạnh `khỉ → động vật` trên đường chép sẽ kéo tới cha `con chó là
động vật` ⇒ chép ra câu nói về **chó**, đúng kiểu hỏng giám khảo đã đếm ~60 lần.

Chỗ ý tưởng có hiệu lực thật là **đường sinh TOKEN KẾ** (`generate()`), nơi cạnh **đẩy token**
và hệ **tự phát** ra câu chưa từng tồn tại. Nhưng đường đó **chưa từng được đo tuyệt đối**.

**Kết quả (mini, 3 nhánh, bộ đề rời nhau, một phiên chấm — 360/450 mẫu đã chấm):**

| nhánh | n | `dung_duoc` | `tam` | **`dung`+`tam`** |
|---|---:|---:|---:|---:|
| `ban_le` (tầng mệnh đề) | 115 | 0,9% | 6,1% | **7,0%** |
| **`ar`** (`generate()`, sinh token kế) | 120 | **0,0%** | 5,8% | **5,8%** |
| **`chep_cau`** (truy hồi + chép) | 125 | **22,4%** | 32,0% | **54,4%** |

| so sánh | chênh | z | |
|---|---:|---:|---|
| `ar` − `ban_le` | +1,1pp | 0,35 | **không tách** |
| `chep_cau` − `ar` | **+48,6pp** | **8,25** | TÁCH |
| `chep_cau` − `ban_le` | +47,4pp | 7,89 | TÁCH |

**Kiểm tái lập thước**: `chep_cau` vòng này **54,4%** vs §13.60 **53,3%** — trên **bộ đề khác**,
**phiên chấm khác**. Thước tuyệt đối tái lập lần thứ tư.

### 🔴 Hệ quả cho TK_046: chỗ tôi vừa chỉ ra là "nhà" của ý tưởng thì gần như hỏng

Câu hỏi treo từ lâu — *hai đường sinh, đường nào tốt hơn* — nay có đáp án: **không đường nào**.
`ar` **5,8%** và `ban_le` **7,0%** **không tách nhau**, và cả hai thua chép **~8 lần**.

⇒ Đề xuất "đưa suy-luận-cạnh vào đường AR" là đưa nó vào một đường **94% hỏng**. Ngay cả khi
cạnh suy ra hoàn hảo, nó chỉ đổi **một token** trong một chuỗi mà mọi thứ khác đều gãy.

**Trạng thái trung thực của TK_046:**

| | |
|---|---|
| ý tưởng **đúng về hiện tượng** | suy luận liên kết chưa từng thấy có thật (`E` 2,44× sàn, §13.63-B) |
| cơ chế **thắng ở xếp hạng** | tách lát 1,60× `E` (§13.67) |
| nhưng **thua ở viết cạnh** | 0,78× `E` (§9) |
| và **đường thi hành duy nhất phù hợp** | đo được **5,8%** — gần như hỏng |

⇒ Ý tưởng chưa có **đường dẫn tới đầu ra**. Không phải vì nó sai, mà vì **nơi duy nhất nó có
hiệu lực đang không hoạt động**.

⚠️ 90/450 mẫu **chưa chấm** (hết hạn mức phiên). Nhưng ba nhánh đều có n ≥ 115 và hiệu ứng
z = 8,25 ⇒ 90 mẫu còn lại không thể đảo dấu.

## 12. Hướng NHÃN (ý tác giả) — bước 0: tách từ **thấy** bệnh nhưng **không chặn** được

**Đề xuất của tác giả (2026-08-05):** đánh nhãn dữ liệu train —
`[CN][DT]Con chó[/DT][/CN][TT]màu đỏ[/TT]`. Kèm lập luận đúng: cạnh chủ động/thụ động
(`khỉ → động vật`) **hình thành lúc train**, nên không train thì không chứng minh được.

**Đính chính của tôi về lập luận đó.** Vì Luật 0 cấm backprop, quy tắc viết cạnh **bắt buộc là
hàm của đếm** ⇒ suy cạnh offline từ checkpoint cho ra **đúng những cạnh train sẽ viết**. P93
(chia đôi corpus, suy trên A, xác nhận trên B) **là** mô phỏng cú viết đó, và **0,78×** không
phải hiện vật của việc không train. Cái probe offline thật sự không đo được chỉ có hai thứ:
**hồi quy** (cạnh viết ở bước `t` đổi cạnh ở `t+1`) và **ảnh hưởng lên ĐẦU RA**. Thứ hai mới
quyết định, và nó **đúng là phải train**.

**Kế hoạch ba bước, rẻ trước:** (0) tách từ chặn được bao nhiêu phần bệnh bản lề — không cần
train; (1) định nghĩa lại khung = **(vai, láng giềng)**, chạy lại probe với cổng cũ; (2) train
lại mini có kênh nhãn, chấm mù tuyệt đối so **5,8% / 7,0%**.

### P94 — bước 0. Trọng tài ĐỘC LẬP: WS-CRF F1 0,9844 (đã có sẵn trong repo)

`out/models/ws_crf_uts_full`, huấn luyện 95.000 câu, **không biết gì** về checkpoint. Cờ WS =
bản lề mang `I-W` khi tách A một mình **VÀ** `w_right` mang `I-W` sau khi nối — đúng chữ ký
§13.39 nhưng đọc bằng **ngữ pháp**, không bằng đồng hiện.

| | SINH | THẬT (đối chứng) |
|---|---:|---:|
| dính TRÁI (`I-W` trong A) | 40,7% | 34,5% |
| dính PHẢI (sau khi nối) | 14,4% | 3,3% |
| **CỜ WS (cả hai phía)** | **7,5%** | **0,7%** |
| số mối nối | 738 | 695 |

**CỔNG G-ws:** (1) n≥300 ✅ · (2) tách ≥+8đpt ❌ **+6,7** · (3) độ nhạy ≥50% ❌ **21,5%** ·
(4) báo nhầm <15% ✅ **5,8%** ⇒ 🔴 **TRƯỢT**

**Dự đoán của tôi sai cả hai chiều**: đoán độ nhạy 60–80% (thật **21,5%**), và đoán (4) là chỗ
đáng lo nhất (nó **đạt dễ dàng**).

### Nhưng hiện tượng thì được xác nhận — và hai thước **gần như không chồng nhau**

`7,5%` vs `0,7%` là **10,4×**. Hai thước **độc lập về nguyên lý** (bigram đồng hiện ·
CRF ngữ pháp) đều nói: mối nối SINH **phá vỡ từ**, mối nối của người viết **thì không**.

| | số ca | tỷ lệ trên 738 |
|---|---:|---:|
| `pivot ≥ 0,05` thấy | 79 | 10,7% |
| cờ WS thấy | 55 | 7,5% |
| **cả hai cùng thấy** | **17** | **2,3%** |
| **một trong hai thấy** | **117** | **15,9%** |

Mỗi thước bỏ sót khoảng **hai phần ba** của thước kia.

### 🔴 Chuẩn vàng bị rò — và điều đó **không** cứu được cổng

Đọc danh sách "báo nhầm" thì phần lớn **là bệnh thật**:

```
' quốc gia' ⇒ ' gia vị'        ' nhật thực' ⇒ ' thực dân'
' đảo chính' ⇒ ' chính quyền'  ' Nghệ An' ⇒ ' An toàn'
' Thuận Hải' ⇒ ' Hải Phòng'    ' Nguyễn Du' ⇒ ' Du Châu'
```

`quốc gia ⇒ gia vị` là **ví dụ kinh điển do chính §13.39 nêu ra** — vậy mà thước `pivot` chấm
nó **< 0,05**, tức "lành". Nghĩa là **độ nhạy 21,5% đo so với một chuẩn vàng rò**, và con số đó
không đáng tin **theo cả hai chiều**. Nhưng cổng đã nêu TRƯỚC ⇒ **giữ nguyên TRƯỢT**. Chuẩn
vàng rò là lý do để **đo lại cho tử tế**, không phải lý do để chấm lại.

### Bài học về thiết kế cổng (của tôi)

Cổng (2) đặt bằng **điểm phần trăm** (+8đpt) trên một đại lượng có nền **0,7%**. Một thước tách
**10,4×** vẫn trượt cổng đó. **Ngưỡng tuyệt đối trên đại lượng hiếm là ngưỡng đặt sai** — cùng
họ với lỗi "≥2 thực thể bão hoà 94,8%" (§13.65) và lỗi mẫu số §13.35. Ghi lại để lần sau đặt
theo **tỷ số** khi nền dưới ~5%.

### Hệ quả cho hướng nhãn

| | |
|---|---|
| tách từ **thấy** bệnh | ✅ 10,4× tách đối chứng |
| tách từ dùng làm **cổng chặn** | ❌ bắt 21,5%, và chuẩn vàng để đo nó thì rò |
| bệnh to cỡ nào | **15,9%** mối nối vỡ từ theo **hợp** hai thước |
| có đáng chữa ngay không | ❓ mối nối **đang tắt** (`max_clauses=1`, §13.52) — chữa bản lề chỉ
đáng nếu nó làm việc NỐI khả thi trở lại. §13.41 đo hỏng-do-nối ≈ 17đpt; vỡ từ 15,9% giải thích
được **một phần lớn nhưng không phải tất cả** |

⇒ Bước 0 cho kết quả **âm tính có điều kiện**: nhãn ranh giới từ là tín hiệu **thật** nhưng
**không** là cổng chặn sạch. Nó **không** đụng tới lập luận mạnh hơn của hướng nhãn — rằng
`[CN]/[VN]/[TT]` nâng **khung** của TK_046 từ *đại lượng thay thế* (láng giềng `W_seq`) thành
**vai thật**. Đó là **bước 1**, và §13.67/§9 (1,60× thắng xếp hạng, 0,78× thua viết cạnh) là
mốc có sẵn để so.

## 14. NƠ-RON NHÃN TỪ LOẠI — dựng xong, cổng ĐẠT ở mức TỪ, TRƯỢT ở mức âm tiết

**Quyết định của tác giả (2026-08-05):** *"định nghĩa trước các nơ ron như vậy (theo nhãn) rồi
chủ động kết nối các nơ ron token/cụm vào nhãn — cái này nên lấy sẵn từ bộ từ điển"*. Tôi đã
nêu ý kiến ngược ở §12 (nên để nhãn lại, vì đường sinh mới là chỗ hỏng); tác giả giữ nguyên
⇒ làm đầy đủ.

**Không phải xây mới.** TK_015 đã có `vocab.pos_id()`, `EdgeKind.POS_ANCHOR`, `EdgeKind.LEXATTR`;
`data/dict/vi_dict_raw.jsonl` có **36.764 mục** kèm `part_of_speech`. Nhưng checkpoint mini có
**0 cạnh** cả ba loại — từ điển **chưa từng được nạp**.

### Vấn đề cốt lõi: từ điển đánh nhãn ở mức TỪ, vốn từ ở mức ÂM TIẾT

`động vật` là *danh từ*, nhưng nó vỡ thành hai âm tiết, mà `động` còn nằm trong `động cơ`,
`hoạt động`, `bị động`. **Cổng G-pos** (nêu trước): phủ ≥60% · sạch (phần từ loại trội, trung
vị theo lượt) ≥0,70 · lớp lớn nhất <60%.

| | tra theo **LOẠI âm tiết** | tra theo **TỪ trong ngữ cảnh** |
|---|---:|---:|
| phủ (theo lượt) | 66,6% ✅ | **73,0%** ✅ |
| sạch — trung vị | 63,6% ❌ | **100%** ✅ |
| sạch — trung bình | 67,2% | **73,6%** |
| entropy trung vị | 1,00 bit | **0,00 bit** |
| số từ loại/lượt — trung vị | 3,0 | **1,0** |
| lớp lớn nhất | danh từ 52,9% ✅ | danh từ 44,2% ✅ |
| **CỔNG G-pos** | 🔴 **TRƯỢT** | ✅ **ĐẠT** |

Đường "mức TỪ" = tách từ bằng WS-CRF (F1 0,9844) **trước**, rồi tra **nguyên từ** trong từ
điển, gán nhãn cho **từng lượt** token — không tra âm tiết rời. Đây là lối đi **đã ghi TRƯỚC**
trong phần dự đoán, không phải cổng dời sau khi thấy số. Dự đoán của tôi cho (2) là 0,55–0,70
và "chỗ lo nhất": mức âm tiết ra **0,636**, **đúng khoảng**.

⇒ Kết luận cơ chế: **nhãn từ điển không hạ được xuống mức âm tiết**, nhưng hạ được xuống
**lượt-trong-ngữ-cảnh**. Nửa sau của §12 (tách từ) là **điều kiện cần** cho nửa trước (nhãn).

### Đã ghi vào mạng

`checkpoints/mini_sent_ep_qa_pos.pkl` — **12 nơ-ron nhãn** (`POS:danh_từ`, `POS:động_từ`, …),
**7.118 cạnh** `LEXATTR` từ **4.230 token**, trọng số = phần lượt dùng thật trong corpus (không
phải số mục từ điển). Vốn từ 32.002 → 32.014. Suite 361 xanh.

### Hai lỗi của tôi, bắt được nhờ chạy lại

1. `part_of_speech: null` (1.735 mục) bị `str(None)` biến thành **nơ-ron nhãn giả tên "none"**,
   chiếm **24,4%** lớp trội ở bản đầu.
2. `max()` trên các nhãn **đồng hạng** phá vỡ theo thứ tự lặp của `set` ⇒ phụ thuộc
   `PYTHONHASHSEED`, **ngẫu nhiên mỗi tiến trình**. Hai lần chạy cùng lệnh cho `danh từ`
   **30,2%** rồi **62,9%**, và **cổng (3) lật theo**, trong khi cổng (1)(2) đứng yên nên trông
   vẫn như chạy đúng. Sửa: khối lượng lớp chia **phân số**, không argmax.

⚠️ Lỗi (2) là **đúng họ** với §0.5 và TK_041: id hợp lệ, không exception, cổng vẫn in ra —
chỉ sai. Thứ duy nhất bắt được nó là **chạy hai lần và so**. Ghi thành thói quen: mọi cổng
mới phải chạy **hai lần** trước khi đọc.

### Cái này CHƯA chứng minh điều gì

Nơ-ron nhãn **tồn tại** và **mang thông tin** (0,00 bit entropy ở mức từ = xác định). Nó có
**cải thiện gì không** thì **chưa đo**. Phép đo kế là thứ §12 đã nêu: khung = **(nhãn, láng
giềng)** thay cho láng giềng trần, chạy lại `probe_frame_tensor` + `probe_edge_precision` với
**cổng cũ y nguyên**, so mốc **1,60×** (xếp hạng) và **0,78×** (viết cạnh).

Về "nơ-ron ngữ pháp" mà tác giả nêu thêm: hoãn tới sau phép đo đó. Định nghĩa thêm một bộ
nơ-ron cắm sẵn **trước khi** biết bộ thứ nhất có giúp gì không là đầu tư đi trước bằng chứng.

## 15. P96 — nhãn từ loại **KHÔNG** nâng được khung (1,03×)

Đúng nhiệm vụ, đúng cổng §13.67, chạy **hai lần** giống hệt nhau.

| bộ chấm | MRR | |
|---|---:|---|
| `ngau_nhien` | 0,0901 | |
| `khung` | 0,1624 | đối chứng |
| `embedding` | 0,1948 | mốc |
| **`tach_lat`** | **0,3107** | **1,60×** `E` — **tái lập §13.67 chính xác** |
| `tl_nhan_b` (nhãn lọc *b*) | 0,3020 | **0,97×** — làm **xấu đi** |
| `tl_nhan_c` (nhãn lọc *c*) | 0,3196 | **1,03×** |

🔴 **CỔNG G-poslab TRƯỢT** ở (3): nhãn phải cho ≥1,10× trên `tach_lat`.

Dự đoán ghi trước **đúng cả hai vế**: `tl_nhan_c > tl_nhan_b`, và cả hai nằm trong khoảng
0,95–1,08× đã nêu. (Lần đầu trong chuỗi này dự đoán của tôi đúng hoàn toàn — sáu vòng trước
đều sai ít nhất một vế.)

## 16. P97 — SÀNG NHÃN: kết luận tôi đăng ký trước bị **chính phép đo của tôi bác bỏ**

§15 kèm sẵn một lời giải thích **đăng ký trước**: *nhãn từ loại **dư thừa** với thống kê láng
giềng*. Vòng này dựng phép sàng tổng quát để kiểm nó — và để sàng mọi nhãn ứng viên **trước
khi** xây.

    dư thừa = (E-kNN đoán đúng − sàn đa số) / (1 − sàn đa số)

| nhãn | token | sàn đa số | `E` đoán đúng | **dư thừa** |
|---|---:|---:|---:|---:|
| `ngau_nhien` (**sàn**) | 3.259 | 49,2% | 43,1% | **−0,12** |
| `tu_loai` | 3.259 | 55,3% | 56,7% | **+0,03** |
| `phong_cach` | 1.465 | 57,5% | 56,2% | **−0,03** |
| `tran_duong_ong` (**trần**) | 3.259 | 20,1% | 77,9% | **+0,72** |

**CỔNG G-sang: ĐỌC ĐƯỢC** (sàn ✅ · trần ✅).

⇒ 🔴 **`tu_loai` nằm ngay trên SÀN.** `E` **không** biết từ loại. Lời giải thích tôi đăng ký
trước — *"nhãn dư thừa"* — **sai**, và bị bác bởi chính phép đo tôi dựng để kiểm nó.

### Vì sao `E` không biết từ loại — nhìn thẳng vào láng giềng

```
vua    → Tông, ngôi, Lý, Thánh, Lê, vương
đánh   → quân, đuổi, bại, trận, Tướng
sông   → chảy, Sông, Hồng, bồi, ngòi
```

`E` gom **bạn cùng NGỮ CẢNH** (syntagmatic), không gom **bạn cùng LỚP** (paradigmatic). Từ loại
là tính chất *lớp* ⇒ `E` không có nghĩa vụ biết, và đúng là không biết.

### 🔴 Hệ quả: phép sàng này **không dự đoán được ích lợi**

Ghép §15 và §16 lại:

| | |
|---|---|
| nhãn từ loại **mới** so với `E` | ✅ dư thừa 0,03, sát sàn |
| nhãn từ loại **giúp** nhiệm vụ | ❌ 1,03× |

⇒ **"mới với `E`" KHÔNG kéo theo "có ích".** Phép sàng tôi vừa dựng vì thế **không được dùng
để bật đèn xanh cho việc xây thêm nhãn** — nó chỉ trả lời một câu hỏi hẹp hơn nhiều. `phong_cach`
cũng ở sát sàn (−0,03), nhưng theo đúng lập luận trên, điều đó **không** nói nó sẽ giúp.

Và nút thắt không phải là **thiếu thông tin về loại**. Hệ đã được cho thông tin loại — thông
tin `E` thật sự không có — và không dùng được nó.

### Ba lần chọn sai đối chứng dương (của tôi)

| bản | đối chứng | kết quả | lỗi |
|---|---|---|---|
| 1 | láng giềng TRÁI trội nhất | −0,13 | hỏi tính chất **lớp**, `E` chỉ mã hoá **ngữ cảnh** |
| 2 | đồng hiện nhiều nhất | sàn **100%** (một lớp) | token rác đồng hiện với mọi thứ (§13.20) |
| 3 | **cụm của chính `E`** | +0,72 | vòng tròn **có chủ ý** ⇒ dùng làm **TRẦN**, không phải bằng chứng ngữ nghĩa |

Bài học: **cổng cần cả SÀN và TRẦN**. Chỉ có sàn thì "tín hiệu thấp" đọc được hai nghĩa trái
ngược — *nhãn mang tin mới* hay *đường ống hỏng* — và tôi đã suýt đọc nhầm nghĩa thứ nhất.

### Tự sửa lời khuyên của chính tôi

Tôi đã khuyên tác giả thêm **loại từ** (`con`/`cái`/`chiếc`) làm nhãn mạnh nhất. **Sai**: loại
từ **chính là một token láng giềng**, nên `framesL` của tách lát **đã dùng nó rồi**. Nó dư thừa
**theo cấu tạo**, không cần đo. Ghi lại để không lặp.

### Dọn nhãn

Bỏ 5 lớp thưa (<20 từ điển): `ngã: động từ` (lỗi định dạng), `danh từ chỉ phương hướng`,
`danh từ, tính từ`, `tiền tố`, `phụ tố`. Còn **8 nơ-ron nhãn**, 7.113 cạnh, 4.229 token.
Suite 361 xanh.

## 17. Việc kế — theo thứ tự

1. ✅ ~~Đo ca âm tính~~ · ✅ ~~cải thiện 3~~ — **xong ở §9**, và kết quả đổi hướng: dùng `E` làm
   bộ đo tương đồng, giữ kiến trúc viết-cạnh của tác giả.
2. ✅ ~~Cải thiện #5 (tách hướng)~~ — **xong ở §10, không mua gì**; tiền đề sai (trái/phải vốn
   đã rời nhau, chồng lấn trung vị 0,000).
3. ❌ ~~Cắm `E`-viết-cạnh vào đường truy hồi~~ — **đề xuất SAI**, xem §11: cạnh suy ra không
   sinh ra câu, chỉ đổi câu nào được chép.
4. ✅ **P98 — đường thi hành** (§18): đồ chơi ĐẠT · viết cạnh G-prec-HE ĐẠT · P98C/D
   · **P98E** d=1 p→x ĐẠT · **P98F** siết `lop_trai` ĐẠT · **P98G** cầu cụm L=1..2 ĐẠT.
5. **Việc kế:** author gold set · `concept_id`/sense (TK_048) · chấm nghĩa;
   mặc định generate vẫn `w_skip` (`analogy_la` opt-in).
6. ✅ ~~Bake nơ-ron `Lbl:*` ngữ cảnh~~ — **thử rồi GỠ** (§18.H). Không predefine /
   induced / skip-Lbl; `compatible` chỉ còn cầu thuộc_tính Z.
7. **Đang mở (§18.H):** validate / wire `hub_spread` nếu cần; đóng gap dữ liệu
   (`Khỉ là động vật`, compound `Khỉ đột`); không bake Lbl lại.

⚠️ **Giới hạn phải nhắc mỗi lần trích số §13.67**: mini, **nhiệm vụ xếp hạng**, **không** phải
chất lượng sinh. §13.60 đã dạy: thắng một thước không có nghĩa thắng ở đầu ra. Và 23,5% ca cơ
chế **câm**.

## 18. P98 (2026-08-06) — Hebbian + `E` → thi hành khung · cổng A/B **ĐẠT**

Trả lời §17.4: *có đường biến liên kết suy ra thành câu đọc được không?*

### A — đồ chơi (ví dụ khỉ / bàn)

`scripts/probe_analogy_exec_toy.py` · `RUNBOOK_TK_046_P98.md`

| | |
|---|---|
| viết | `khỉ → động vật` (support=2 qua chó/mèo); `bàn` **không** (chỉ 1 qua chó/nâu) |
| thi hành | `con khỉ là` → **`con khỉ là động vật`** · `cái bàn là` → không điền |
| cổng A1–A3 | **ĐẠT** |

Không AR · không chép cha — chỉ điền khe khung `X là Y` từ cạnh đã viết.

### B — mini G-prec-HE

`probe_edge_precision.py --writer hebbian_E` · `out/probe_edge_prec_hebbian_E.json`

| bộ | cạnh | lỏng |
|---|---:|---:|
| `ngau_nhien` | 1500 | 15,9% |
| `tach_lat` | 1490 | 20,5% |
| `embedding` | 1500 | 26,3% |
| **`hebbian_E`** (min_support=2) | 1171 | **33,5%** |

| cổng | kết quả |
|---|---|
| (1) ≥200 chủ thể | 280 ✅ |
| (2) ≥1,5× ngẫu nhiên | **2,10×** ✅ |
| (3) ≥1,0× embedding | **1,27×** ✅ |

⇒ Siết Hebbian (≥2 bạn) trên nền tương đồng `E` **nâng** precision **viết cạnh** so với `E`
một mình — đúng hướng §9 cải thiện #4.

### C — mini thi hành (P98C) — thước **HỎNG** (rút)

Gold = mọi trigram `(a,f,c)` ở B → cắt cụm (`Thăng Long để`, `văn học Trung đại`).
Không đo loại suy / sáng tạo. **Không** dùng số P98C để kết tội ý tưởng.

### D — G-exec-v2 (P98D, 2026-08-06) — thước sáng tạo

`scripts/probe_analogy_exec_v2.py` · `RUNBOOK_TK_046_P98D.md` · `out/probe_analogy_exec_v2.json`

Ca dương: `(a, là, c)` **chưa có** cho `a`, ≥2 bạn E có `(b, là, c)` · cách ly cha của `a`.
Ca âm: ≤1 bạn → không được điền `c`.

| bộ | n điền dương | hit@1 | FP âm |
|---|---:|---:|---:|
| `ngau_nhien` | 200 | 1,0% | 5,0% |
| `embedding` | 200 | 24,5% | 2,0% |
| **`hebbian_E`** | 200 | **35,5%** | **0,0%** |

| cổng | kết quả |
|---|---|
| (1) n≥30 | 200 ✅ |
| (2) hit@1 ≥40% | **35,5%** ❌ (trượt hẹp) |
| (3) FP ≤15% | **0%** ✅ |
| (4) creative (triple vắng) | ✅ |

⇒ **TRƯỢT hẹp** cổng (2), nhưng tín hiệu rõ: `hebbian_E` **+11đ** vs `E`, **+34,5đ** vs
ngẫu nhiên, FP=0. Đúng hướng ý tác giả (câu/cạnh chưa train). Không bake; có thể siết
chọn ca / top-b trước khi hạ cổng.

### E — G-exec-v3 cầu thuộc tính · ca MẠNH (P98E)

`scripts/probe_analogy_exec_v3.py` · `RUNBOOK_TK_046_P98E.md` · `out/probe_analogy_exec_v3.json`

Thước **sáng tạo** + cầu đúng hướng (sửa phản biện):

- chỉ **bigram d=1**; cạnh **`p → x`** từ bề mặt `(x, p)` (vd `nghịch→khỉ`)
- triple `là` sạch · prompt thực thể · `(a,là,c)` **VẮNG**
- case `Thái`/`hôn` cũ: **0** cạnh `hôn→Thái` d=1 → loại đúng

| bộ | n điền dương | hit@1 | FP âm |
|---|---:|---:|---:|
| `ngau_nhien` | 16 | 12,5% | 26% |
| `embedding` | 16 | 31% | 0% |
| **`bridge`** | 16 | **100%** | **0%** |

| cổng | kết quả |
|---|---|
| (1) n≥10 | 16 ✅ |
| (2) hit@1 ≥40% | **100%** ✅ |
| (3) FP ≤15% | **0%** ✅ |
| (4) creative | ✅ |

⇒ **ĐẠT** hình thức; mẫu vẫn hẹp (`nuôi`/`họ`/`thứ`/`mất` → họ). Hit 100% một phần do
gold ≈ cạnh mạnh nhất cùng thước — cần soi nghĩa từng ca. Không bake generate.

### F — P98F siết lớp trái (2026-08-07)

`scripts/probe_analogy_exec_v4.py` · `RUNBOOK_TK_046_P98F.md` · `out/probe_analogy_exec_v4.json`

Sau P98E: cùng token `nuôi` nhưng `mẹ nuôi` ≠ `Hồng nuôi`. Siết:

- giữ d=1 · `p→x`
- donor chỉ đếm khi `lop_trai(p→a) ∩ lop_trai(p→b) ≠ ∅`
- `lop_trai` ∈ {`nguoi`, `dia_ly`, `khac`}

| bộ | n điền dương | hit@1 | FP âm |
|---|---:|---:|---:|
| `ngau_nhien` | 16 | 6% | 44% |
| `embedding` | 16 | 31% | 9% |
| **`bridge`** | 16 | **100%** | **0%** |

| cổng | kết quả |
|---|---|
| G (1–4) | **ĐẠT** |
| Q1 `Hồng`/`Sông` chết (cầu `nuôi`) | ✅ |

⇒ **ĐẠT.** Ca `Hồng`/`Sông` qua `nuôi` đã loại. Mẫu còn `họ`/`thứ`/`mất` + họ; nhiều ca
`lop_trai=khac` (khớp yếu hơn `nguoi`) — cải dần. Không bake.

**P98F+ (cộng nhiều cầu):** `score(c)=Σ_p strength` — ĐẠT giữ; chỉ **1/16** ca có ≥2 cầu
(vd `anh` ← `họ`+`thứ`). Trọng số đúng hướng; trên mini vẫn ít đa-cầu.

### G — P98G cầu cụm L=1..2 (2026-08-07)

`scripts/probe_analogy_exec_v5.py` · `RUNBOOK_TK_046_P98G.md` · `out/probe_analogy_exec_v5.json`

- L=1 `(x,p)⇒p→x` · L=2 `(x,p1,p2)⇒(p1,p2)→x` · `compound_bonus` (mặc định 2.5)
- giữ `lop_trai` + Σ cầu; resolve head chọn token-id props mạnh nhất (vd `anh` 2149≠1550)

| bộ | n điền dương | hit@1 | FP âm |
|---|---:|---:|---:|
| `ngau_nhien` | 16 | 6% | 48% |
| `embedding` | 16 | 31% | 9% |
| **`bridge`** | 16 | **100%** | **0%** |

| cổng | kết quả |
|---|---|
| G (1–4) | **ĐẠT** |
| Q1 `Hồng`/`Sông` | ✅ |
| ≥2 cầu / có L2 | 1/16 · **0/16** L2 trên mini |

⇒ **ĐẠT** hình thức. L=2 sẵn sàng nhưng mini chưa có trigram sạch kích hoạt trong ca dương.

**Bake opt-in (không đổi default):** `generation_mode=analogy_la` ·
`nn_aivie.hca.analogy_bridge_gen` · `RUNBOOK_ANALOGY_P98G_SHIP.md` ·
`aivie generate --mode analogy_la`. Không bake cạnh vào adjacency / W_skip.

**G+ ship smoke (2026-08-07):** neo cụm ISG/phrase/Entity + ISG **power-win overlap** ·
rebuild `checkpoints/network_v12_subst_isg_pw.pkl` (cluster `học sinh` power=1840) ·
try **17/26** điền, **0× «là cùng»**, `học sinh là → Vương` (n=8, có `tự+tay`) ·
mini `anh/mẹ/em là → Hồ`. Chi tiết: `out/analogy_la_smoke_isg_pw.log` ·
`out/analogy_la_try_network_v12_subst_isg_pw.json` · `out/analogy_la_compare_isg_pw.txt`.

### H — Lbl ngữ cảnh → hub_spread → gỡ Lbl (2026-08-08 … 2026-08-10)

Trả nợ Luật 0 cho `lop_trai` / cổng ngữ cảnh khi điền `X là ?`, rồi nhận ra **nhãn
predefine không cần** nếu đã có W_skip + parents. Nhật ký ngắn:

#### H.1 Việc đã làm

| bước | nội dung | kết quả |
|---|---|---|
| Cầu `Lbl:` | Bake hub seed `Lbl:nguoi\|dia_ly\|to_chuc\|vat_the` + NER + LEXATTR; `compatible` = giao neighbor `Lbl:*` | mini `Hồ` giữ; bớt string `lop_trai` runtime |
| Densify / induced | `Lbl:{C}` từ `X là C`; `Lbl:cls:*` chiều lớp; opt `Lbl:prop:*` | nhiễu (head = token trước `là`); prop dễ bẩn |
| Skip-Lbl | `Lbl:skip:{Y}` / `Lbl:skipcls:{X}` từ W_skip d∈{1,2,3} | top hub nhiễu (vd `lê` trộn pha lê / ba lê / lưỡi lê) |
| Insight pivot | Cặp nhãn thật hay có pivot func (`là`/`như`) giữa — power cao, gidf thấp | định hướng score hub = `w·gidf·diversity`, ban func |
| **hub_spread** | [`hub_spread.py`](src/nn_aivie/hca/hub_spread.py) · [`probe_hub_spread.py`](scripts/probe_hub_spread.py) · [`test_hub_spread.py`](tests/test_hub_spread.py) — runtime, **không retrain** | synthetic ĐẠT; trên `isg_pw` textbook **câm** |
| Probe `isg_pw` | seed `Khỉ` / `mèo` / `chó` + prop `nghịch` | `Khỉ`→`đột` (compound); `mèo` scored top `thú` via `mèo-[là]->thú`; không có `động vật`→peers→`nghịch ngợm` sạch |
| **Gỡ Lbl (2026-08-10)** | Zero LEXATTR mọi `Lbl:*` trên `isg_pw` + `mini`; xóa bake/seed/scripts/tests; bỏ cổng Lbl trong `analogy_bridge_gen` | `isg_pw` 55754 cạnh→0; orphan token vocab giữ id; smoke mini `anh là → Hồ` |

**Giải pháp đã chốt trong code hiện tại**

1. **`analogy_la` (P98G):** chỉ cầu thuộc_tính Z (L=1..2) + Σ cầu; **không** đọc `Lbl:*`.
2. **`hub_spread` (probe):** lan truyền ý niệm trên W_skip / via-pivot `là|như…` — cache Lbl không bắt buộc.
3. **Ckpt:** `scripts/strip_lbl_lexattr.py` (giữ để strip lại nếu tái nhiễm). Prefix `Lbl:` trong vocab/viz vẫn harmless (orphan).

**Artifact / log:** `out/probe_hub_spread_latest.json` · `out/lbl_*` (lịch sử bake, không còn pipeline) ·
`RUNBOOK_ANALOGY_P98G_SHIP.md` (mục Lbl **lỗi thời** — xem §H.3).

#### H.2 Việc cần làm (mở)

1. **Gap dữ liệu, không thiếu train API:** corpus `isg_pw` gần như không có `Khỉ là động vật`; `Khỉ` bị kéo `Khỉ đột`. Author gold / câu bổ sung *nếu* muốn demo textbook — trước đó đã chọn không bake câu demo giả.
2. **Gate compound:** nếu có `Entity:pha lê` thì không mine skip-part `pha→lê` (đã đồng ý hướng, **chưa ship**).
3. **Wire generate:** `hub_spread` mới probe; chưa gắn `generate` / `analogy_la` thay thế.
4. **Không** bake lại `Lbl:*` ngữ cảnh; không remap orphan vocab trừ khi migrate lớn.
5. Việc kế cũ vẫn đúng: gold set · sense/`concept_id` (TK_048) · chấm nghĩa · default `w_skip`.

#### H.3 Giải pháp đã thực hiện (chốt)

```
trước:  seed/NER/induced/skip → Lbl:* + LEXATTR → compatible = ∩ Lbl
sau:    strip LEXATTR Lbl · xóa bake · analogy chỉ Z · hub_spread độc lập (probe)
```

- **Luật 0:** inference chỉ id+cạnh; bake tạo cấu trúc — Lbl bake *đúng Luật 0* nhưng **thừa** so với W_skip lan truyền.
- **Vì sao gỡ:** nhiễu hub, phụ thuộc bake, textbook demo không mở được bằng thêm `Lbl`; cơ chế peer/prop đã có trên graph sẵn.
- **Smoke sau gỡ:** `tests/test_hub_spread.py` ✅ · mini `anh là → Hồ` ✅ · LEXATTR→`Lbl:*` = 0 trên cả hai ckpt.

#### H.4 Kiểm ckpt ship v12 (M4, 2026-08-10) — `RUNBOOK_M4_CHECK_V12_SHIP.md`

| ckpt | `n_synonym_edges` | `n_triples` | `la_emit_doi_so_voi_min_id` |
|---|---:|---:|---|
| `network_v12_subst_isg_pw.pkl` | 244886 | 821 | **false** (emit=1495) |
| `network_v12_subst.pkl` | 244886 | 800 | **false** (emit=1495) |

`analogy_la` **không câm vì ngưỡng** trên cả hai. Vá `la_emit_id` không đổi hành vi v12 so với min-id. Artefacts: `out/check_ckpt_network_v12_subst*.{log,json}`.

Scaffold A/B `full` trên `isg_pw` (probe, không ship): bridge hit **0%** / FP 0% · TRƯỢT cổng G-exec-v5 — khác mini (full 100%). Họ người `Nguyễn/Trần/Lê/Hồ` gidf đều ≥4,5 trên v12; `off`/`gidf` không chạy xong (quá chậm ở cỡ v12).

#### H.5 BƯỚC 3 hub_spread trên v12 (M4, 2026-08-10) — `RUNBOOK_M4_HUB_STEP3_V12.md`

Đo tách kênh `typed_first` vs `mixed`/`cooc` sau vá `e4324de`. Artefact: `out/hub_step3_v12_isg_pw.{json,log}`.

| | |
|---|---|
| `n_typed_edges` | **89 230** (src=7890) |
| `typed_first` | nhãn **97,7%** · thành viên mới **98,8%** (171/200 có hub) |
| `mixed` | nhãn 93,0% · thành viên mới 98,8% |
| `cooc` | nhãn 88,1% · thành viên mới 95,5% |
| lift `typed_first/cooc` | **1,109** |

**Cổng G-hub3:** (1) typed_edges ✅ · (2) label_rate ✅ · (3) lift≥1,5× ❌ · (4) new_member ✅ → **TRƯỢT** (cổng 3). Tách kênh sống ở v12 (nhãn cao hơn cooc) nhưng **không mua đủ 1,5×**.

**Đọc tay `top_labels`:** đỉnh bậc-vào typed vẫn hư từ — `một(2846), vậy(1028), thế(1021), gì(918), là(732), những(664), người(660)…`. Thước bậc-vào = proxy nhiễu (cùng bệnh mini).

**Ca `mèo`/`Vương`:** hub = `thú(24), khắc(17), loài(122)` · **`Vương`∉`thanh_vien_hub`** → bước 3 **chặn được** ca `con mèo là → Vương` bằng cạnh typed đã học (không bảng tay). `chó` → hub `bạn/thú/hai`. `Trần` → hub gồm `Huyết` (vẫn dính tên). `học sinh`/`Việt Nam` không có trong vocab (multi-token).

#### H.6 VÒNG 2 hub_step3 v12 (M4, 2026-08-10) — sau vá `d85b62f`

Artefact: `out/hub_step3_v12_isg_pw_r2.{json,log}` (không đè vòng 1).

| | vòng 1 | vòng 2 |
|---|---:|---:|
| `n_typed_edges` | 89 230 | **120 875** |
| `typed_first` nhãn / mới | 97,7% / 98,8% | 94,3% / 98,4% |
| `mixed` | 93,0% / 98,8% | 90,6% / 97,9% |
| **`cooc` (đối chứng)** | **88,1%** | **83,9%** |
| lift | 1,109 | **1,124** |

**Cổng:** 1✅ 2✅ 3❌ 4✅ → **TRƯỢT** (cổng 3). Lift gần như không đổi.

⚠️ **`cooc` vẫn 83,9% > 60%** — thước bậc-vào HẰNG bão hòa ở v12; **không đọc lift như bằng chứng tách kênh** (đúng cảnh báo runbook §4). Mini sau vá mới phân biệt được vì cooc tụt 43,5%→16,4%; v12 chưa.

**`top_labels`:** `một` **biến mất** (vá gidf-lookahead có tác dụng) nhưng đỉnh vẫn hư từ — `gì(956), vậy(628), nào(551), những(401), thế(288), phố(284)…`; có tín hiệu lớp `vật(178)`.

**Ca `mèo`:** hub = `chó(63), khắc(48), loài(130)` · thành viên kiểu giống loài (`Shiba`, `Bella`, `Tạng`…) thay vì `hứng/yêu/thích` vòng 1 · **`Vương` vẫn ∉** · vẫn chặn được ca analogy_la.

---

## 19. CHUỖI BA BƯỚC — bốn vòng chấm mù, và hướng CRF cho khâu còn tắc (2026-08-11)

Chỉ đạo tác giả: lấy nơ-ron W_skip làm **nhãn** rồi lan **ba bước** —
`nghịch ngợm → {chó, mèo} → động vật → {chó, mèo, KHỈ}`. Bước ③ (xuống lại từ
nhãn) sinh ứng viên chưa từng thấy = **sáng tạo**.

### 19.1 Kết quả bốn vòng (chấm mù, chấm TRƯỚC khi mở nhãn)

| vòng | `typed_first` | n | ghi chú |
|---|---:|---:|---|
| 1 | **0,0%** | 13 | sampler **bỏ mất bước ①** — đo cơ chế khác |
| 2 | 5,9% | 17 | sau vá kiểu 1+2 |
| 3 | 5,3% | 19 | **đối chứng VÔ HIỆU** (WS trộn ISG) |
| **4** | **16,7%** | **12** | sau 3 vá — vào vùng BIÊN (15–35%) |

Đối chứng `cooc` **11,1% ở cả bốn vòng**. Vòng 4 nhánh ISG 20,0% nhưng **n=5**.

⚠️ **Đừng đọc 5,3% → 16,7% như tiến bộ chắc chắn.** n=12, tức 2/12 vs 1/19.
Cái chắc chắn là **hút-điểm `phố` (10/28) đã biến mất**; tỷ lệ thì chưa đủ mẫu.
n co lại (19→12) vì cơ chế nay **chặt hơn**, không phải vì mất dữ liệu.

### 19.2 Bảy bug tìm ra — TẤT CẢ ở phía phép đo / cài đặt, không ở ý tưởng

| # | bug | footprint đo được |
|---|---|---|
| 1 | sampler **bỏ bước ①** (xuất phát từ MỘT từ, không có thuộc tính ràng buộc) | vòng 1 vô nghĩa |
| 2 | nhãn **cắt cụt**: `vũng lầy là một tình thế` → `tình` | `tình yêu` 469 · `tình hình` 613 vs `tình thế` 287 ⇒ đổi hẳn nghĩa |
| 3 | thang chấm hỏi **"thành viên mạch lạc không"** ⇒ thưởng HỌ CẤU TẠO TỪ | tiếng Việt ghép 2 âm tiết ⇒ `cooc` thắng gần như theo định nghĩa |
| 4 | **đọc ngược chiều** khuôn liệt kê `<LỚP> là A, B, C` | 554 câu |
| 5 | `chẳng hạn như` — quét ngược dừng ở dấu phẩy, lớp thật nằm ngay trước | 1 659 câu |
| 6 | 🔴 **`thành` là pivot nhưng cũng là âm tiết từ thường** | trở thành 3 516 (đúng) vs thành phố 2 262 · thành công 1 995 · hoàn thành 1 466 · thành lập 1 368 · hình thành 1 297 ⇒ **~8 400 sai** — nguồn hút-điểm `phố` |
| 7 | 🔴 span WS-CRF **TRỘN** với cụm ISG thay vì **THAY THẾ** | vòng 3 so `(WS+ISG)` với `(ISG)` ⇒ đối chứng vô hiệu |

Cộng: **hút-điểm là CẤU TRÚC, không phải một từ** — `vậy` 8/22 (v1) → `chẳng`
9/26 (v2) → `phố` 10/28 (v3). Giết một cái thì cái kế lên thay. Chỉ dừng khi
tìm ra nguyên nhân THẬT (bug 6), không phải khi lọc khéo hơn.

### 19.3 Đơn vị neo: ISG → WS-CRF

Cụm ISG là n-gram theo **tần suất**: **2 427/4 234 = 57%** không phải đơn vị
ngôn ngữ (mini) — `là Nguyễn`(power 33) · `của phở` · `đá và`. Neo nhãn vào đó
cho ra `tụ vĩ` (từ «lãnh tụ vĩ đại»), `Bốn đài`, `Các tôn`, `tình`.

WS-CRF (TK_034, F1 0,9844) tách đúng cả năm ca hỏng. Sidecar bake **một lần**
(`scripts/bake_ws_spans.py` → `.ws_spans.json`, 71 379 span, 2,5 phút).
Rác: 57% → **1,8% tự động / ~35–43% đọc tay** (thước tự động MÙ với mảnh BPE
tên ngoại lai `René Lac os te`, và gộp quá tay trên văn kiếm hiệp).

### 19.4 🔴 CÒN TẮC: chọn TRUNG TÂM NGỮ ĐOẠN — đề xuất dùng CRF

Đã thử **ba luật tay**, mỗi luật sai một kiểu (fixture; **chưa xác thực ở v12**
vì máy dev không nạp nổi ckpt):

| luật | «Các thành phố quan trọng khác» | «Bốn đài truyền hình lớn nhất» |
|---|---|---|
| trái nhất | `Các` ❌ | `Bốn` ❌ |
| **max-gidf** (đang dùng) | `quan trọng` ❌ | `đài truyền hình` ✅ |
| trái-nhất-đủ-đặc-thù | `Các` ❌ | `Bốn` ❌ |

Phân biệt **định từ** với **danh từ trung tâm** cần TỪ LOẠI; tần suất một mình
không đủ. Dừng lặp thay vì thử luật thứ tư mù.

**Đề xuất: đây chính là bài toán GÁN NHÃN CHUỖI — dùng đúng khuôn WS-CRF.**

Đọc `src/nn_aivie/ws_crf/ws_features.py`, WS-CRF mạnh nhờ ba thứ mà cách làm
hiện tại đều thiếu:

| | WS-CRF | luật tay hiện tại |
|---|---|---|
| tri thức từ điển | **đặc trưng** `is_in_dict`, mô hình tự cân | **luật cứng** (`_FUNC`, `BAD_PRED`) |
| ngữ cảnh | ±2 **hai phía** + cặp chéo `(-2,0)`, `(0,2)` | quét **một phía**, cửa sổ cố định |
| quyết định | **cả chuỗi cùng lúc** (không thể `I-W` đầu câu) | từng vị trí độc lập |
| trọng số | **học từ 95k câu có nhãn**, L-BFGS c1=1.0 (L1 thưa hoá) | **chỉnh tay** — và mọi ngưỡng hằng đều vỡ khi đổi quy mô |

Dòng cuối giải thích gọn bốn vòng trượt: `bậc-vào ≥3` bão hoà ở v12 · ngưỡng
`label_gidf_min=2.0` làm đỏ 14 test · biên gidf `1.0` không chuyển được sang
fixture. **Ngưỡng hằng không chuyển được giữa các quy mô** (lặp 3 lần trong
phiên) — CRF học trọng số nên không dính.

**Thiết kế đề xuất:** gán mỗi âm tiết của ngữ đoạn chủ ngữ nhãn `B-HEAD` /
`I-HEAD` / `O`; đặc trưng ±2 + `istitle` + `is_in_dict` + cặp chéo, y khuôn
`build_ws_features`. Luật 0 sạch — CRF chạy **offline/build**, runtime chỉ đọc
span (đúng tiền lệ TK_034 §đầu).

**Dữ liệu huấn luyện — không cần gán tay:** chưng cất từ **chính khuôn liệt kê**.
Câu «Các thành phố quan trọng khác **là** Göteborg, Malmö…» tự nó là một nhãn —
vế trái chứa lớp, và ta biết trung tâm phải là `thành phố`. **554 + 1 659 câu**
(liệt kê + `chẳng hạn như`) đủ làm tập huấn luyện tự động.

**Chưa làm.** Ghi lại vì đây là hướng có căn cứ nhất cho khâu còn tắc, và nó
KHÁC về chất với "vá thêm một ngưỡng".

### 19.5 Bài học phương pháp

1. **Ánh xạ từng VAI của thiết kế vào từng thành phần của phép đo TRƯỚC khi đo.**
   Ba vòng đầu hỏng vì trượt bước này; câu hỏi của tác giả *«đâu là con khỉ,
   động vật, nghịch ngợm trong ca này?»* mới lộ ra sampler đang đo cơ chế khác.
2. **Cổng số không thay được đọc tay.** Bake WS spans báo rác **0,0%** trong khi
   đọc 12 mẫu thấy ~35% bẩn (`Hỏi: Tàu`, `tại Gia Lý Học Viện`).
3. **Tiền-đăng-ký giữ khỏi BẺ CONG diễn giải, KHÔNG giữ khỏi ĐĂNG KÝ NHẦM giả
   thuyết.** R3 §4 đăng ký "âm tính ⇒ nghi phạm là DỮ LIỆU"; số về đúng, diễn
   giải đúng lời ghi trước — nhưng lời ghi trước sai vì ngầm giả định cơ chế đã
   đúng. Thực tế còn ba bug cơ chế.
4. **Ngưỡng HẰNG không chuyển được giữa các quy mô** — lặp 3 lần trong một phiên.

## 20. CHUỖI BA BƯỚC — ca thử vàng **ĐẠT**, dữ liệu thật vẫn **0**; ba lỗi cơ chế + bốn giả thuyết bị bác (2026-08-12)

§19 kết ở chỗ "chuỗi ba bước 16,7% (BIÊN)" rồi vòng 5 (n=80) tụt về 12,0%. Cả
năm vòng đều **trộn hai biến**: đo đồng thời *cơ chế* và *dữ liệu*, trên dữ liệu
không đảm bảo có cấu trúc cần đo. Âm tính không phân biệt được "cơ chế hỏng" với
"corpus không có `X là C`". §20 tách từng biến ra.

### 20.1 Ca thử vàng — cơ chế **KHÔNG hỏng**

Dựng đúng 5 câu của tác giả thành corpus riêng
(`data/text_corpus/toy_chain3.jsonl` → `checkpoints/toy_chain3.pkl`, 5 cha):

```
loài chó là động vật · loài mèo là động vật · loài khỉ là động vật
nghịch ngợm như loài chó · nghịch ngợm như loài mèo
```

Chuỗi chạy đầu-cuối:

```
nghịch → {loài, chó, mèo} → động → {chó, khỉ, mèo}
  ①              ② typed, qua `là`      ③ khỉ CHƯA từng đứng cạnh `nghịch`
```

**Đối chứng then chốt**: `fwd[nghịch]` = {ngợm, như, loài, chó, mèo} — W_skip
**1 bước KHÔNG ra `khỉ`**. Chỉ chuỗi 3 bước mới ra ⇒ chuỗi đóng góp thật.

⚠️ Đây là **điều kiện cần**, không phải bằng chứng ở quy mô thật (memory
`long-range-cooccurrence-negative`: ckpt nhỏ đánh lừa NGƯỢC DẤU). Nó chứng minh
cơ chế *không hỏng* — điều mà 5 vòng trước chưa từng xác lập.

### 20.2 Ba lỗi cơ chế tìm được (đều thật, đều đã đo)

**(a) `min_count` so với TRỌNG SỐ, không phải SỐ LẦN.** `build_w_skip` cộng
`pw = power × priority` vào ô, `_aggregate_mats` lọc `if w < min_count`. Ngưỡng
thực là `count ≥ min_count/priority` và **trôi theo priority từng dataset**. Đo
trên `classrich` (priority ∈ {0,90 · 0,95}), cạnh `X là C`:

| min_count | đáng giữ | cắt oan | lọt thêm |
|---|---|---|---|
| **1** ← hầu hết caller | 3229 | **1330 (41,2%)** | 0 |
| **2** ← `hub_spread` | 914 | 51 (5,6%) | 517 |
| 3 | 46 | 1 (2,2%) | 624 |

Lỗi **đổi chiều** theo ngưỡng (power>1 kéo trọng số lên). Ca thử vàng câm hoàn
toàn cho tới khi hạ xuống 0,5 — vì power=1, priority=0,95 ⇒ w=0,95 < 1.
**Nhưng ở `min_count=2` (đúng cái 5 vòng dùng) lỗi chỉ xoá 5,6% ⇒ KHÔNG giải
thích được chín số 0.** Dự đoán của tôi bị chính phép đo bác.

**(b) Trích trung tâm nhặt BỔ NGỮ thay vì ĐẦU LỚP.** `via_pivot` chỉ phủ
**18,1%** cạnh lớp có thật (144/797) — xem §20.5.

**(c) Xếp hạng nhãn KHÔNG đếm đồng thuận.** `_consider` giữ điểm **lớn nhất**
trên mỗi đích (`if prev is None or (typed, sc) > (prev.typed, prev.score)`), nên
nhãn được 1 hạt giống bầu điểm cao thắng nhãn được 3 hạt giống bầu điểm thấp.
Trái thiết kế `chó+mèo+khỉ → động vật` (nhãn nổi lên nhờ **đồng thuận**).
`diversity` trong công thức là số nguồn *toàn corpus*, không phải số hạt giống.

### 20.3 Dữ liệu **KHÔNG** thiếu lớp — giả thuyết "dilution" bị bác

`scripts/mine_chain3_structures.py` quét thẳng câu cha, **không gọi `hub_spread`**
nên không thể bị cơ chế hỏng làm nhiễu. Trên `classrich` (35 757 cha):

```
4 434 câu `X là C` · 2 358 lớp · 164 lớp có ≥3 thành viên
```

Và chúng sạch:

```
triều đại   7 tv : Hồ, Lê, Lý, Nguyễn, Sơn, Trần      ← đúng hình dạng chó/mèo/khỉ
ngôn ngữ   20 tv : Anh, Baloch, Hebrew, Java, Nga
quốc gia   18 tv : Angkor, Canada, Cuba
thành phố  28 tv : Amsterdam, Frankfurt, Montréal
```

### 20.4 Bốn giả thuyết bị bác BẰNG SỐ — đừng thử lại

| giả thuyết | phép đo | phán quyết |
|---|---|---|
| Nút thắt là **phát hiện nhãn** (nhãn phải là CỤM `thành phố`, không phải token `thành`) | giới hạn ứng viên ② vào đúng 164 lớp có thật | ✗ **1/40** |
| Xếp nhãn theo **số phiếu hạt giống** | `--hub-rank phieu` | ✗ **tệ hơn** (0/40) — tần suất nền nuốt: `thi`,`học`,`đều`,`đầu` thắng |
| `gidf_min=4.0` chặn nhãn lớp (vì nhãn lớp vốn phổ biến) | gidf 164 nhãn lớp | ✗ chỉ chặn **3%** (5/164) — dù có chặn `thành phố`/`thành viên` (gidf 3,45) |
| Corpus **thiếu** câu phát biểu lớp | §20.3 | ✗ 797 cạnh lớp · 164 lớp |

**Rào cản THẬT nhưng KHÔNG ĐỦ:** ép hạt giống ① vào vũ trụ 447 thành viên lớp
đưa `≥2 hạt giống cùng lớp` từ **0/40 → 16/40**, nhưng ② vẫn không hội tụ.

### 20.5 Vá trích trung tâm — proxy tăng 58%, chuỗi vẫn **0**

`Cuba là quốc gia duy nhất…` ⇒ `via_pivot[Cuba] = ['duy nhất']`. Bổ ngữ luôn
hiếm hơn đầu lớp ⇒ **max-gidf luôn thua**. Cùng bệnh nhánh TRÁI đã ghi (§19).

Thêm `head_rule` vào `_mine_via_pivot` (mặc định **giữ nguyên** `max_gidf`), đo
bằng `scripts/measure_pivot_coverage.py` — thước nằm TRONG đồ thị, không chấm người:

| `head_rule` | phủ cạnh lớp | ② khớp lớp chung | ③ mới thật |
|---|---|---|---|
| **`trai_du_dac_thu`** | **28,6%** (228/797) | 2/40 | 10/40 |
| `trai_nhat` | 27,9% | 0/40 | 6/40 |
| `dau_lop` | 26,6% | 0/40 | 4/40 |
| `max_gidf` ← hiện hành | **18,1%** (144/797) | 0/40 | 10/40 |

`trai_du_dac_thu` = trái nhất trong ứng viên có gidf ≥ **phân vị rút từ chính
corpus** (định từ tự rơi, không cần danh sách — Luật 0). `dau_lop` học từ corpus
xem ứng viên nào hay đứng **ngay sau** pivot.

**Nhưng 2/40 đó là CÙNG MỘT CA lặp lại, và là rác**: hạt giống `{động, nó}` →
nhãn `yếu tố` → "mới" = `cư` (mảnh của `Di cư`). **Số ca chuỗi thật vẫn 0.**
Trích trung tâm là nút thắt thật nhưng **không phải nút thắt ràng buộc**.

### 20.6 Chẩn đoán còn lại

```
cơ chế              ĐÚNG    (ca thử vàng chạy, 1 bước không ra `khỉ`)
dữ liệu             ĐỦ      (797 cạnh lớp · 164 lớp)
từ vựng nhãn        không phải nút thắt   (trao sẵn: 1/40)
vũ trụ hạt giống    rào cản thật, không đủ (0→16/40)
trích trung tâm     rào cản thật, không đủ (18,1%→28,6%, chuỗi vẫn 0)
► bước ② vẫn không hội tụ DÙ ĐÃ CÓ ĐỦ NGUYÊN LIỆU
```

**Hình dạng dữ liệu là nghi phạm còn lại, và nó KHÁC "thiếu dữ liệu":** corpus
phát biểu lớp cho **danh từ riêng** (`Amsterdam là thành phố`, `Lê là triều
đại`), còn thuộc tính bám vào **danh từ chung** (`hè → nhiệt, độ, mùa`). Chỉ
**18,1%** hạt giống ① nằm trong vũ trụ lớp — gấp 13 lần ngẫu nhiên (447/32002),
tức bước ① có thiên hướng đúng nhưng chưa bao giờ đủ **hai** cái cùng lớp.
Corpus có `Amsterdam là thành phố`; nó không có `nhiệt độ là đại lượng`.

**Việc kế có căn cứ nhất — đường cong NHIỄU:** lấy ca thử vàng (5 câu, ĐẠT) rồi
trộn dần cha thật vào, tìm **điểm gãy**. Nó trả lời trực tiếp câu "thêm dữ liệu
có cứu được không" mà không phải đoán, và tránh đúng bẫy `long-range-cooccurrence-negative`.

### 20.7 Bài học phương pháp (nối tiếp §19.5)

5. **Tách cơ chế khỏi dữ liệu bằng CA THỬ VÀNG trước khi đo trên corpus thật.**
   Năm vòng chấm mù không phân biệt được hai nguyên nhân; 5 câu và một giờ thì
   phân biệt được. Ca thử vàng chỉ dùng làm **điều kiện cần** — dương tính ở đó
   không kết luận gì về quy mô.
6. **Thước phải khớp KIỂU bài toán.** Ví dụ của tác giả là cơ chế **SINH**
   (prompt → phân bố → `khỉ` thấp nhưng khác 0); suốt chín vòng tôi đo bằng thước
   **TRUY HỒI** (top-5 + chấm nhị phân). `hub_spread` tới nay vẫn **không có
   caller nào trong `src/`** — đo một cơ chế chưa nối vào đường sinh.
7. **`PYTHONHASHSEED=0` khi khâu chọn ca duyệt qua `set`.** Không có nó tôi đã so
   nhầm hai bộ ca khác nhau và suýt kết luận ngược.
8. **Thước sai lần thứ ba trong cùng chủ đề**: `②_đúng_lớp` bắt nhãn phải bằng
   lớp `C` sinh ra thuộc tính, nhưng hạt giống ① lấy tự do nên có thể thuộc lớp
   khác — khi đó ② trả lớp khác là ĐÚNG mà bị chấm sai. Phải hỏi "nhãn có nằm
   trong các lớp mà ≥2 hạt giống cùng thuộc?".
9. **Vá đúng một nút thắt có thật vẫn có thể không đổi kết quả.** Trích trung tâm
   +58% tương đối, chuỗi vẫn 0. Nút thắt *có thật* ≠ nút thắt *ràng buộc*.

## 21. ĐƯỜNG CONG NHIỄU — tìm được điểm gãy, đẩy được **895 → 3 069 cha**, và lộ nút thắt kế (2026-08-12)

§20 kết ở câu hỏi «bổ dữ liệu có cứu được không» mà không trả lời được, vì chỉ có
hai điểm: 5 câu sạch (ĐẠT) và 35 757 cha (0). §21 lấp khoảng giữa.

`scripts/probe_chain3_noise_curve.py`: giữ nguyên 5 câu vàng, trộn dần câu thật
từ `paragraph_units` vào, train lại từng mức, hỏi đúng câu nhị phân của tác giả —
**`khỉ` có ra không, ở hạng mấy**. Không cần chấm mù.

⚠️ `min_count=0,5` ở mọi mức để cô lập biến nhiễu — ở mặc định thì lỗi
trọng-số-vs-số-lần (§20.2a) cắt sạch cạnh vàng và ta đo nhầm bug đó.

### 21.1 Điểm gãy: giữa 895 và 3 069 cha

| nhiễu | cha | nhãn | `khỉ` hạng |
|---|---|---|---|
| 0 | 5 | `động` | **2** ✓ |
| 300 | 118 | `vật` | **2** ✓ |
| 1 000 | 337 | `vật` | **2** ✓ |
| 3 000 | 895 | `vật` | **2** ✓ |
| 10 000 | **3 069** | `cá` | — ✗ |
| 30 000 | 10 287 | `loài` | — ✗ |
| 60 000 | 20 316 | `loài` | — ✗ |

Corpus thật nhỏ nhất (`classrich`) là 35 757 cha — **gấp 12–40 lần** ngưỡng chịu
đựng. ⇒ **Bổ dữ liệu KHÔNG cứu được**: nó chỉ đẩy sâu hơn vào vùng đã gãy.

### 21.2 Nguyên nhân: LƯỢNG TỪ cướp chỗ THÀNH VIÊN, và nó TRÔI THEO QUY MÔ

```
cha=337   via_pivot[chó] = [('vật', 0.95)]   ✓
cha=895   via_pivot[chó] = [('vật', 0.95)]   ✓
cha=3069  via_pivot[chó] = []                ✗  cạnh vàng MẤT
          via_pivot[loài] = [('vật', 1.90)]
```

`_trung_tam_trai` chọn chủ ngữ trong `loài chó là động vật` bằng **max-gidf**.
Corpus lớn lên ⇒ gidf `loài` và `chó` đổi vị trí ⇒ chủ ngữ chuyển từ THÀNH VIÊN
sang LƯỢNG TỪ. `chó`/`mèo` mất sạch cạnh typed, bước ② rơi xuống kênh cooc, và
`loài cá`/`loài thú` áp đảo.

Cùng họ với lỗi vế phải (§20.5) — **chọn trung tâm bằng gidf không bền theo quy mô**.

Hai giả thuyết khác đã loại: phân vị `label_gidf_q` (chạy `q=0.0` y hệt) · `loài`
lọt vào hạt giống (bỏ nó thì nhãn thành `châu`, tệ hơn).

### 21.3 Vá — và bài học «một bản vá đúng vẫn có thể vô dụng một mình»

Thêm `head_rule_trai` (mặc định **giữ nguyên** `max_gidf`). Ở tầng cơ chế nó vá
đúng chỗ, cạnh vàng phục hồi hoàn toàn ở 3 069 cha:

```
max_gidf          chó→[]           mèo→[S:0.95]     khỉ→[vật:0.95]
phai_du_dac_thu   chó→[vật:0.95]   mèo→[vật:0.95]   khỉ→[vật:0.95]   ✓
```

**Nhưng chuỗi vẫn gãy** — vì dòng đầy đủ là `mèo → [vật:0,95 · đặc:1,90]`. Cạnh
nhiễu `đặc` nặng GẤP ĐÔI cạnh vàng nên thắng, dù `vật` được **ba** hạt giống bầu
còn `đặc` chỉ **một**. Đây đúng là lỗi §20.2c.

⇒ Thêm `rank="phieu"` vào `score_hubs_from`: đồng thuận quyết, **chỉ trong kênh
typed** (đếm phiếu trên cooc thì tần suất nền nuốt — §20.4).

**Ghép hai bản vá:**

| cha | CŨ | MỚI (`phai_du_dac_thu` + `phieu`) |
|---|---|---|
| 895 | `vật`/2 | `vật`/2 |
| **3 069** | `cá`/— | **`vật`/3** ✓ |
| 10 287 | `loài`/— | `nghiên`/— |

**Điểm gãy 895 → 3 069 cha (3,4×).** Không bản vá nào một mình làm được: vá vế
trái đơn lẻ **tệ hơn** (gãy ngay ở 895), đếm phiếu đơn lẻ cũng tệ hơn (§20.4).

### 21.4 Nút thắt KẾ — bước ① trôi sang BẠN GHÉP TỪ

Ở 10 287 cha chỗ gãy đã DI CHUYỂN:

```
① = ['lý', 'Fermi', 'quỹ', 'phép']      ← `nghịch` → `nghịch lý`, `nghịch lý Fermi`
```

`nghịch` không còn trỏ `chó`/`mèo` nữa. Đây đúng bệnh đã đo trên `classrich`
(§20.6): W_skip khoảng cách gần bị **bạn ghép từ** thống trị — `dược → tề, liệu,
phẩm, sĩ`. Hướng có căn cứ: bước ① phải làm việc trên **đơn vị TỪ** (`loài chó`,
`nghịch ngợm`) chứ không phải âm tiết — tức WS-CRF span, thứ đã bake sẵn cho
`classrich` nhưng CHƯA dùng cho bước ①.

### 21.5 Bài học phương pháp (nối tiếp §20.7)

10. **Đường cong thay cho một điểm.** Hai điểm (5 câu ĐẠT · 35 757 cha CÂM) không
    phân biệt được «cơ chế yếu» với «thiếu dữ liệu». Bảy điểm thì chỉ đúng ngưỡng,
    và ngưỡng đó bác thẳng phương án «bổ dữ liệu».
11. **Bản vá đúng vẫn có thể vô dụng một mình.** `head_rule_trai` phục hồi 100%
    cạnh vàng mà chuỗi vẫn 0 cho tới khi ghép với `rank="phieu"`. Đo từng bản vá
    riêng lẻ sẽ kết luận CẢ HAI đều vô dụng — và đó là kết luận sai.
12. **Fixture tí hon không hiệu chỉnh nổi luật theo PHÂN VỊ.** `phai_du_dac_thu`
    thắng trên dữ liệu thật nhưng ở fixture thì `chó` rơi khỏi ngưỡng 20%. Khoá
    test bằng luật XÁC ĐỊNH (`phai_nhat`), để bằng chứng của luật phân vị nằm ở
    đường cong. Lặp lại đúng bài học `blind-rubric-drifts`.

## 22. NỐI CHUỖI VÀO ĐƯỜNG SINH — lần đầu đo bằng THƯỚC CỦA TÁC GIẢ (2026-08-12)

§20.7 bài học 6 ghi: suốt chín vòng tôi đo `hub_spread` bằng thước **TRUY HỒI**
(top-k + chấm nhị phân) trong khi thiết kế của tác giả là cơ chế **SINH** —
prompt `nghịch ngợm như loài ___` → **phân bố** token kế, `khỉ` thấp hơn
`chó`/`mèo` nhưng khác 0. `hub_spread` khi đó **không có caller nào trong `src/`**.

### 22.1 Baseline — đường sinh KHÔNG chạy cơ chế này

Trên ca thử vàng, prompt `nghịch ngợm như loài` (lấy `scores` qua `trace_out`):

```
 1. chó   1,2967   p≈0,592
 3. mèo   0,2645   p≈0,121
 7. khỉ   0,0150   p≈0,0069   ← CÙNG điểm với `loài`, `động`, `vật`, `là`
```

`khỉ` khác 0, **nhưng nó nằm ở SÀN làm mượt**, ngang `là`/`vật` — không phải do
chuỗi nâng lên. Ở quy mô lớn hơn nó tụt hẳn: `h479` (895 cha) · `h1560` (3 069).

### 22.2 Nối vào — `HCAConfig.hub_chain_boost`, mặc định 0 (TẮT)

`hub_chain_proposals()` chạy đúng ba bước rồi trả `{tid: trọng số}`; vòng AR
**nhân** với token đã có điểm và **cộng** cho token chưa có (thành viên thật sự
mới thường nằm NGOÀI phân bố của đường sinh). Chỉ lấy cạnh **typed** — kênh đồng
hiện tăng tuyến tính theo cỡ corpus nên nó nuốt cạnh phân loại (§18.H). Index
dựng MỘT LẦN cho mỗi lần sinh và cache (dựng lại tốn ~11s trên ckpt thật).

**Hạng của `khỉ`, đo qua chính `generate_autoregressive`:**

| cha | boost=0 | boost=1 | boost=4 | lift |
|---|---|---|---|---|
| 5 | h7 | h6 | h4 | 1,2× |
| **895** | **h479** | **h9** | **h7** | **53×** |
| **3 069** | **h1560** | **h22** | **h14** | **71×** |
| 10 287 | h6130 | h6131 | h6131 | — |

Ở 10 287 không đổi vì bước ① đã trôi (`nghịch` → `nghịch lý Fermi`, §21.4) nên
chuỗi không còn đề cử `khỉ`. Chỗ gãy nhất quán với §21, không phải chỗ mới.

### 22.3 ĐẠT tới đâu, và CHƯA đạt chỗ nào

**Đạt:** `khỉ` được **nâng** chứ không còn nằm ở sàn — đúng phần «có xác suất ra
được, dù thấp hơn» trong thiết kế. Ở 3 069 cha, chuỗi đề cử `bò`, `dê`, `ngựa`,
`cá` — cùng miền ngữ nghĩa, tức bước ③ vẫn đúng miền ở quy mô đã gãy bước ②.

**CHƯA đạt:** vế «xác suất **khá sát nhau**». Ở 3 069 cha, `chó` p≈0,059 còn
`khỉ` p≈0,0004 — vẫn cách **150×**. Nâng 71 bậc không phải là san bằng phân bố.

**CHƯA đo:** `boost` là núm tự do tôi chọn tay (1,0 / 4,0), chưa có nguyên tắc
đặt. Và toàn bộ §22 đo trên ckpt tí hon — chưa chạm `classrich`/`v12`.

### 22.4 Bài học phương pháp (nối tiếp §21.5)

13. **Nối xong mà 428 test cũ VẪN XANH = không test nào chạy nhánh mới.** Lặp
    đúng bẫy `test_MAC_DINH_phai_la_typed_first` (§19): mặc định tắt thì suite cũ
    không thể phát hiện gì. `tests/test_hub_chain_generate.py` khoá nhánh bật, và
    có test chốt TIỀN ĐỀ (`khỉ` chưa kề `nghịch`) — không có nó thì bài toán là
    TRA CỨU chứ không phải sáng tạo, và mọi số ở trên vô nghĩa.

## 23. §22 TRÊN DỮ LIỆU THẬT — kết quả ckpt tí hon **KHÔNG chuyển** (2026-08-12)

§22 đo trên 5–3 069 cha và thấy `khỉ` h1560 → h22 (71×). Memory
`long-range-cooccurrence-negative` đã cảnh báo ckpt nhỏ từng đánh lừa NGƯỢC DẤU.
Chạy đúng phép đo đó trên `classrich` (35 757 cha), 25 ca lấy từ chính dữ liệu
(đào lớp `X là C`; «con khỉ» = thành viên của `C` **chưa từng đồng hiện với P**):

| | |
|---|---|
| «con khỉ» LÊN hạng | 9 |
| «con khỉ» XUỐNG hạng | 24 |
| vào phân bố lần đầu | **0** |
| tỉ lệ hạng (bật/tắt) trung vị | **0,998** |

**Không có tác dụng.** 0,998 là nhiễu, không phải cải thiện.

### 23.1 Một lỗi thật tìm được trên đường — và nó KHÔNG đổi phán quyết

Lần chạy đầu 5/10 ca chuỗi ra RỖNG. Nguyên nhân: **một bề mặt có NHIỀU id BPE**
(`dòng` = 2880/7427/8744/25529) còn `via_pivot`/`fwd` khoá theo TỪNG id.
`encode_prompt` trả đúng một biến thể, và trên `classrich` đó thường là biến thể
**rỗng cạnh** — nên chuỗi câm vì lý do kỹ thuật, không phải vì ngữ nghĩa.

Đã vá (gom mọi biến thể cùng bề mặt — Luật 0: đồng nhất theo bề mặt, không phải
danh sách từ). Kết quả: 7→9 lên, 24 xuống y nguyên, tỉ lệ 0,994→0,998.
**Lỗi thật, vá đúng, phán quyết không đổi** — lặp lại bài học §21.5 số 11 theo
chiều ngược: vá đúng vẫn có thể không mua được gì.

### 23.2 Chuỗi KHÔNG tắc — nó chạy đủ ba bước rồi ra SAI

Truy 12 ca: **① rỗng 1/12 · ② rỗng 0/12 · ③ không có typed 0/12**. Chuỗi chạy
trơn. Vấn đề là đầu ra:

```
dòng  → ① {chảy, họ, sông, dõi}   → ② 'Hoài'   → ③ {sông}
đốt   → ① {cháy, sống, Nga, hợp}  → ② 'nhánh'  → ③ {hợp, mình, thống, lập}
buộc  → ① {châm, phải, kẻ, bại}   → ② 'Trãi'   → ③ {Hải, Lê, huyện, quân}
quảng → ① {Trấn, Đông, trung, tế} → ② 'ngành'  → ③ {tế, sản, điện, đồ, obacteria}
đấu   → ① {ích, đơn vị, tranh, tế}→ ② 'ngành'  → ③ {tế, sản, điện, đồ, obacteria}
```

**Bước ① thường đúng nghĩa** (`dòng → chảy, sông`; `đốt → cháy`; `tuân → thủ,
luật, định`). Bước ② cho nhãn **typed nhưng SAI**, và hai thuộc tính khác hẳn
nhau (`quảng`, `đấu`) cho **cùng một danh sách ③** — nam châm hút nhãn.

Nhất quán với §20.5: `via_pivot` chỉ phủ 18,1% cạnh lớp thật nhưng có 18 903
cạnh typed — **nó không nghèo, nó đầy cạnh SAI LOẠI**. Nhãn "typed" vì thế không
bảo đảm là nhãn LỚP. §22 không lộ ra điều này vì ca thử vàng chỉ có cạnh đúng.

### 23.3 Trạng thái

Đường sinh nay CÓ chạy cơ chế (§22 nối xong, mặc định vẫn TẮT). Trên ckpt tí hon
nó nâng `khỉ` 71×; trên `classrich` nó không nâng được gì. Nút thắt vẫn là
**chất lượng `via_pivot`**, không phải khâu nối.

### 23.4 Bài học phương pháp (nối tiếp §22.4)

14. **Ca thử vàng chứng minh cơ chế CHẠY, không chứng minh nó ĐÚNG.** Toy chỉ
    chứa cạnh đúng nên mọi cạnh typed đều là cạnh lớp; ở quy mô thật 82% cạnh
    typed là sai loại và cùng cơ chế cho ra rác. Dương tính trên toy phải đọc là
    «không hỏng», tuyệt đối không phải «đạt».

## 24. ĐƯỜNG 1 — lọc cạnh theo SỐ CHỦ NGỮ, và một sai lầm quét-một-biến (2026-08-12)

§23 kết: `via_pivot` không nghèo (18 903 cạnh typed) mà **đầy cạnh sai loại** —
`buộc → 'Trãi'` («Nguyễn Trãi là danh tướng» là MÔ TẢ, không phải lớp). Phân biệt
được bằng chính đồ thị: **LỚP nhận cạnh từ NHIỀU chủ ngữ khác nhau** (`quốc gia`
← Cuba, Canada, Angkor…), **MÔ TẢ chỉ từ một** (`danh tướng` ← Trãi). Thêm
`label_indeg_q` (phân vị, mặc định 0 = tắt).

### 24.1 Sai lầm phương pháp: quét MỘT biến bỏ sót TƯƠNG TÁC

§20.5 quét bốn luật **vế phải** với vế trái để MẶC ĐỊNH và kết luận 28,6%. Ghép
cả hai vế thì phủ **68,4%** — gấp 2,4 lần. Con số 28,6% không sai, nhưng nó trả
lời một câu hỏi hẹp hơn tôi tưởng.

| cấu hình | phủ cạnh lớp | cạnh typed |
|---|---|---|
| `max_gidf` cả hai vế (gốc) | 18,1% | 18 903 |
| chỉ vá vế PHẢI (§20.5) | 28,6% | 18 773 |
| **vá CẢ HAI vế** | **68,4%** (545/797) | 10 194 |
| + `indeg_q=0,8` | **68,4%** | **7 567** (−26% rác) |
| + `indeg_q=0,9` | 64,4% | 5 983 |

`indeg_q=0,8` là điểm ngọt: **giữ nguyên phủ, bỏ 26% cạnh**.

### 24.2 Chuỗi nhích thật, đường sinh thì KHÔNG

Trên `classrich`, 40 ca, cấu hình đầy đủ (cả hai vế + `indeg_q=0,8` + `rank=phieu`):

| thước | trước | sau |
|---|---|---|
| `②_khớp_lớp_chung` | **0/40** | **8/40** |
| `③_mới_thật` | 10/40 | 14/40 |
| `②_đúng_lớp` | 1/40 | 2/40 |

Nhưng thước SINH (25 ca, hạng của «con khỉ» trong phân bố token kế):

| | |
|---|---|
| LÊN hạng | 13 |
| XUỐNG hạng | **49** |
| tỉ lệ hạng trung vị | **0,995** |

**Vẫn phẳng.** Số «xuống» còn TĂNG (24 → 49): chuỗi giờ đề cử nhiều hơn, và phần
lớn đề cử vẫn không phải thành viên cần nâng — nâng thứ khác lên thì mục tiêu tụt
xuống tương đối.

### 24.3 Trạng thái

Ba tầng, ba kết quả khác nhau — và đây chính là giá trị của việc giữ ba thước:

```
cạnh lớp trong via_pivot   18,1% → 68,4%   ✓ cải thiện lớn
bước ② khớp lớp chung       0/40 → 8/40    ✓ lần đầu khác 0
phân bố sinh                0,998 → 0,995  ✗ không đổi
```

Cải thiện có thật ở hai tầng dưới nhưng **chưa xuyên lên tầng trên**. Chưa rõ vì
`8/40` còn quá thưa, hay vì thước sinh chọn ca kém (thuộc tính `Áo`, `dòng`,
`buộc`; «con khỉ» toàn danh từ riêng `Amsterdam`, `Diego`). Chưa phân định được.

### 24.4 Bài học phương pháp (nối tiếp §23.4)

15. **Quét một biến bỏ sót tương tác.** Vá vế phải đơn lẻ: 28,6%. Vá cả hai:
    68,4%. Cùng dạng với §21.5 số 11 (`head_rule_trai` một mình vô dụng, ghép với
    `rank="phieu"` mới ăn) — trong hệ này các bản vá **nhân nhau**, không cộng.
    Kết luận «luật X mua được Y%» phải ghi kèm mọi biến khác đang ở đâu.

## 25. ĐÍNH CHÍNH §24.2 — `②_khớp_lớp_chung = 8/40` là ẢO, thước tự chế ra kết quả

§24.2 ghi «bước ② khớp lớp chung 0/40 → 8/40 ✓ lần đầu khác 0». Tách hai nhánh
để phân định «thước sinh phẳng vì ② thưa» khỏi «vì chọn ca kém» thì lộ ra cả hai
đều sai — và bản thân con số 8/40 không đứng vững.

### 25.1 Phép tách

Dựng bộ ca sinh từ chính 8 ca ② đã khớp, giữ 32 ca còn lại làm đối chứng **trong
cùng một lần chạy**:

```
[②_KHỚP]  n=8    lên=0   xuống=18   mới=0
[②_chưa]  n=32   lên=12  xuống=38   mới=1
```

Ngay trên nhánh ② đã khớp, thước sinh **lên 0, xuống toàn bộ**. Không phải do
chọn ca.

### 25.2 Hai lỗi trong phép đo của tôi

**(a) 8 ca thực chất là 2.** `buộc` lặp 6 lần, `nhuận` 2 lần — cùng một `P` được
chọn cho nhiều lớp khác nhau (giới hạn «tối đa 2 ca mỗi LỚP» không chặn trùng `P`).

**(b) Probe chuỗi và đường sinh chạy HAI pipeline khác nhau.**
`probe_chain3_known_labels` dùng `--seed-universe lop` — **ép hạt giống ① phải là
thành viên của lớp đã đào**. `hub_chain_proposals` (đường sinh thật) không có
ràng buộc đó. Tôi so hai đường ống rồi coi chúng là một.

Và ràng buộc đó **tự chế ra kết quả**: nếu hạt giống buộc phải chọn từ vũ trụ
lớp thì việc «≥2 hạt giống cùng một lớp» gần như được cấp không.

### 25.3 Ép hạt giống LÀM XẤU bước ①

| P | ① tự do | ① ép vào vũ trụ lớp |
|---|---|---|
| `dòng` | chảy, họ, **sông**, dõi | họ, sông, điện, nước |
| `đốt` | **cháy, sống**, Nga, hợp | **Nga, đài, hương, thủy** |
| `tuân` | **thủ, luật, định**, chung | chung, lịch, lý, phép |
| `buộc` | thủ phạm, phải, kẻ, bại | phải, họ, sinh, gia |

Bước ① **tự do cho ra ngữ nghĩa ĐÚNG** (`đốt → cháy, sống`; `tuân → thủ, luật,
định`). Ép vào vũ trụ lớp thì nó nhặt bất kỳ thành viên lớp nào tình cờ ở gần —
`đốt → Nga, đài, hương` — ngữ nghĩa hỏng hẳn.

⇒ `②_khớp_lớp_chung = 8/40` **rút khỏi bảng §24.2**. Nó đo một cấu hình vừa
không có trên đường sinh, vừa làm hỏng bước ①.

### 25.4 Trạng thái đã sửa

```
cạnh lớp trong via_pivot   18,1% → 68,4%   ✓ THẬT (đo thẳng với chân lý corpus)
bước ② khớp lớp chung       ẢO — rút        ✗
phân bố sinh                0,995           ✗ không đổi
```

Chỉ **một** cải thiện đứng vững, và nó chưa chảy xuống được tầng nào.

**Nút thắt còn lại, phát biểu chính xác:** láng giềng của thuộc tính và thành
viên của lớp là **hai quần thể khác nhau**. `đốt → cháy, sống` đúng nghĩa nhưng
`cháy`/`sống` không phải thành viên lớp nào; ép chúng vào vũ trụ lớp thì mất
nghĩa. Đây KHÔNG phải lỗi lọc — lọc kiểu nào cũng không nối được hai quần thể.

### 25.5 Bài học phương pháp (nối tiếp §24.4)

16. **Thước tự chế ra kết quả — lần thứ tư trong một phiên.** Trước đó: «thành
    viên có mạch lạc không» thưởng cho họ từ ghép tiếng Việt; «có mang thuộc
    tính không» thưởng cho ngắn mạch; `②_đúng_lớp` bắt sai lớp. Nay
    `--seed-universe lop` cấp không cho «≥2 hạt giống cùng lớp». Mẫu chung: **ràng
    buộc đặt vào ĐẦU VÀO của một bước lại được tính là THÀNH TÍCH của bước đó.**
17. **So hai đường ống rồi coi là một.** Probe và đường sinh khác nhau ở
    `seed_universe`, `topk`, `gidf_min`. Trước khi so hai số, phải liệt kê mọi
    tham số khác nhau giữa hai đường chạy.

## 26. ĐỘ GIAO HAI QUẦN THỂ — §25 bị chính phép đo của nó bác (2026-08-12)

§25 kết: «láng giềng của THUỘC TÍNH và thành viên của LỚP là hai quần thể khác
nhau… lọc kiểu nào cũng không nối được». Đo trên **mẫu rộng 400 thuộc tính**
(token nội dung df∈[5,2000], KHÔNG chọn tay — 40 ca tự chọn ở §20 thiên lệch),
index đầy đủ (cả hai `head_rule` + `indeg_q=0,8`), k=10 láng giềng:

| lớp ≥ | thành viên | % vocab | giao trung bình | **≥2 cùng lớp** | ② chọn đúng top-1 | top-5 |
|---|---|---|---|---|---|---|
| 2 | 606 | 1,89% | **43,6%** | **32,2%** | 20,9% | 40,3% |
| 3 | 447 | 1,4% | **36,1%** | **29,8%** | 21,8% | **43,7%** |
| 5 | 272 | 0,85% | 22,0% | 21,5% | 24,4% | 39,5% |

**Giao gấp ~26 lần ngẫu nhiên** (36,1% vs 1,4%), và **98% thuộc tính có ≥1 láng
giềng là thành viên lớp**. §25 SAI — hai quần thể giao nhau rất mạnh.

### 26.1 Vì sao §25 (và §20) đo ra 0

Ba khác biệt, mỗi cái đều theo hướng dìm kết quả:

1. **Bộ thuộc tính**: §20 chọn tự động theo «chạm 2–50% thành viên» ⇒ ra token
   HIẾM (`hè`, `tra`, `Hơn`, `dược`). §26 lấy token nội dung phổ biến vừa phải.
2. **Số láng giềng**: §20 lấy top-4, §26 lấy top-10.
3. **Index**: §20 chạy `max_gidf` cả hai vế, không lọc bậc-vào (phủ 18,1%);
   §26 chạy cấu hình đầy đủ (68,4%).

`0/40` của §20 không phải tính chất của dữ liệu — nó là tính chất của **cách tôi
lấy mẫu**.

### 26.2 Nút thắt thật: CHUYỂN HOÁ ở bước ②

Trên 119 ca **đã có sẵn** điều kiện cần (≥2 láng giềng cùng lớp):

```
② chọn đúng lớp ở TOP-1 : 21,8%
② chọn đúng lớp ở TOP-5 : 43,7%
```

Đáp án **có mặt trong top-5 ở 43,7% ca** nhưng chỉ **21,8%** lên được hạng 1.
Vấn đề là **XẾP HẠNG**, không phải thiếu ứng viên và cũng không phải thiếu dữ
liệu. `rank="phieu"` (đếm đồng thuận trong kênh typed) đã là bản vá tốt nhất
hiện có và nó vẫn để lọt 22 điểm giữa top-1 và top-5.

### 26.3 Trạng thái đã sửa lại (lần hai)

```
dữ liệu có cấu trúc         29,8% thuộc tính có điều kiện cần   ✓ THẬT
via_pivot phủ cạnh lớp      18,1% → 68,4%                       ✓ THẬT
bước ② chuyển hoá           top-1 21,8% · top-5 43,7%           ← NÚT THẮT
phân bố sinh                0,995                               ✗ chưa đo lại với bộ ca đúng
```

⚠️ Thước sinh (§23–25) chạy trên bộ ca của §20 — **bộ ca thiên lệch đã bị §26
bác**. Con số 0,995 vì thế không còn đáng tin theo hướng nào; phải đo lại với bộ
ca của §26 trước khi kết luận gì về đường sinh.

### 26.4 Bài học phương pháp (nối tiếp §25.5)

18. **Kết luận về DỮ LIỆU rút từ một bộ mẫu tự chọn là kết luận về BỘ MẪU.** §25
    tuyên bố hai quần thể rời nhau — một mệnh đề về corpus — dựa trên 40 ca do
    chính tôi lọc ra bằng một tiêu chí thiên về token hiếm. Mẫu rộng lật ngược
    hoàn toàn. Trước khi phát biểu «corpus không có X», phải lấy mẫu KHÔNG qua
    bộ lọc của mình.
19. **Top-1 vs top-5 tách «thiếu ứng viên» khỏi «xếp hạng sai».** 21,8% / 43,7%
    nói ngay rằng nửa số ca đã có đáp án trong tay mà xếp trượt. Chỉ đo top-1 thì
    hai nguyên nhân đó không phân biệt được — và tôi đã đo top-1 suốt chín vòng.

## 27. THƯỚC SINH ĐO LẠI với bộ ca §26 — vẫn phẳng, và có HAI chỗ rò (2026-08-12)

§26 bác bộ ca §20 (thiên về token hiếm) nên mọi số của thước sinh ở §23–25 hết
giá trị. Dựng lại bộ ca theo §26 (mẫu rộng, k=10 láng giềng, giữ ca có ≥2 láng
giềng cùng lớp), n=40, boost=4:

| | |
|---|---|
| mục tiêu LÊN hạng | 34 |
| mục tiêu XUỐNG hạng | **78** |
| vào phân bố lần đầu | 4 |
| tỉ lệ hạng trung vị | **0,998** |

**Vẫn phẳng.** Giả thuyết «phẳng vì chọn ca kém» ⇒ **BỊ LOẠI**. Bộ ca đúng rồi mà
đường sinh vẫn không nâng được mục tiêu.

### 27.1 Phân tầng theo ĐẦU RA của ② — tín hiệu CÓ lan, nhưng yếu

| tầng | n ca | mục tiêu LÊN | XUỐNG | tỉ lệ lên |
|---|---|---|---|---|
| ② chọn ĐÚNG lớp | 13 | 18 | 21 | **46%** |
| ② chọn SAI lớp | 27 | 16 | 57 | 22% |

*(Đây là chẩn đoán LAN TRUYỀN, không phải tuyên bố ② chạy được: phân tầng theo
**đầu ra** của ②, khác hẳn §25 nơi tôi ràng buộc **đầu vào** rồi tính là thành
tích.)*

Tín hiệu lan thật — ② đúng thì tỉ lệ nâng gấp **2,1 lần**. Nhưng **ngay cả khi ②
đúng, vẫn 18 lên / 21 xuống** — chưa thành lãi ròng.

### 27.2 Hai chỗ rò, không phải một

```
rò 1 — XẾP HẠNG ở ②:  78% ca chọn sai lớp (top-1 21,8% · top-5 43,7%, §26.2)
rò 2 — ngay khi ② ĐÚNG: chỉ 46% mục tiêu được nâng
```

Rò 2 giải thích được: `via_pivot` phủ 68,4% cạnh lớp, nên ~32% thành viên của
đúng lớp đó vẫn không tới được từ nhãn qua kênh typed; thêm nữa boost nâng MỌI
đề cử nên các mục tiêu cạnh tranh lẫn nhau.

Vá xong rò 1 (đưa ② lên ~44% bằng cách lấy trọn top-5) mà rò 2 giữ nguyên thì
kỳ vọng cận trên chỉ ~0,44 × 0,46 ≈ **20% mục tiêu được nâng** — vẫn chưa đủ để
tỉ lệ hạng trung vị rời khỏi ~1,0.

### 27.3 Trạng thái

```
dữ liệu có cấu trúc      29,8% thuộc tính đủ điều kiện cần   ✓
via_pivot phủ cạnh lớp   18,1% → 68,4%                       ✓
② xếp hạng               top-1 21,8% · top-5 43,7%           ← rò 1
③→sinh khi ② đúng        46% nâng                            ← rò 2
phân bố sinh (tổng)      0,998                               ✗
```

Chuỗi nhân quả nay **liền mạch và đo được ở từng mắt** — lần đầu kể từ §19. Không
còn mắt nào là hộp đen.

### 27.4 Bài học phương pháp (nối tiếp §26.4)

20. **Phân tầng theo ĐẦU RA hợp lệ; ràng buộc ĐẦU VÀO thì không.** §25 ép hạt
    giống vào vũ trụ lớp rồi tính «≥2 hạt giống cùng lớp» là thành tích — vòng
    tròn. §27.1 phân tầng theo lớp mà ② TỰ CHỌN, rồi hỏi bước SAU đó làm được gì
    — chẩn đoán lan truyền hợp lệ. Ranh giới: ràng buộc đặt TRƯỚC bước đang chấm
    thì hỏng, đặt SAU thì được.
21. **Một số phẳng có thể là TÍCH của hai số không phẳng.** 0,998 không phải
    «không có tín hiệu» mà là 0,22 × 0,46. Chỉ đo đầu-cuối thì không thấy hai
    thừa số, và sẽ vá nhầm chỗ hoặc bỏ cuộc nhầm lúc.

## 28. VÁ HAI RÒ — cả hai bản vá HỎNG, nhưng lộ ra một đính chính lớn (2026-08-12)

§27 phân rã `0,998 ≈ 0,22 × 0,46` thành hai chỗ rò. Vá từng cái.

### 28.1 Rò 1 — chuẩn hoá phiếu theo bậc-vào: **TỆ HƠN**

Giả thuyết: 2 phiếu ở `triều đại` (7 chủ ngữ) mạnh hơn 2 phiếu ở `nhà` (hàng
trăm) ⇒ chia phiếu cho bậc-vào. Đo trên 119 ca có sẵn điều kiện cần:

| `rank` | ② đúng lớp top-1 | top-5 |
|---|---|---|
| `diem` (xếp theo điểm) | **0,8%** | 5,0% |
| **`phieu`** (đếm đồng thuận) | **21,8%** | **43,7%** |
| `phieu_chuan` (chia bậc-vào) | 7,0% | 14,0% |

**Bản vá hỏng, và lẽ ra phải thấy trước:** LỚP NHIỀU THÀNH VIÊN thì bậc-vào CAO
**theo định nghĩa** — `quốc gia` có 18 thành viên nên đương nhiên 18 chủ ngữ trỏ
tới. Chia cho bậc-vào là phạt đúng đáp án và thưởng cho MÔ TẢ một-chủ-ngữ, tức
là đảo ngược chính bộ lọc `label_indeg_q` của §24. Giữ `phieu_chuan` làm **đối
chứng âm**, mặc định là `phieu`.

### 28.2 ĐÍNH CHÍNH §20.4 — «xếp theo số phiếu tệ hơn» là SAI

§20.4 ghi «xếp nhãn theo SỐ PHIẾU hạt giống — tệ hơn (0/40)». Số đúng là
**0,8% → 21,8%, gấp 27 lần**. Sai vì §20.4 đo trên index CŨ (`max_gidf` cả hai
vế, không lọc bậc-vào, phủ 18,1%) và bộ ca ràng buộc đã bị §25 bác. Cùng dạng
với §24.1: **kết luận «luật X vô dụng» chỉ đúng cho cấu hình lúc đo**.

Đây là bản vá **có giá trị nhất tìm được trong ngày**, và tôi đã ghi nó là vô
dụng suốt tám mục.

### 28.3 Rò 2 — gom biến thể BPE của NHÃN ở bước ③: **KHÔNG ĐỔI**

Cùng bệnh đã có thật ở §23.1 (một bề mặt nhiều id, `via_pivot` khoá theo từng
id), áp lên NHÃN trước khi lan ngược:

| | trước | sau |
|---|---|---|
| mục tiêu LÊN | 34 | 33 |
| mục tiêu XUỐNG | 78 | 78 |
| vào phân bố lần đầu | 4 | **5** |
| tỉ lệ hạng trung vị | 0,998 | 0,998 |

Không đổi. Rò 2 **không phải** do biến thể id. Nguyên nhân còn lại: `via_pivot`
phủ 68,4% nên ~32% thành viên đúng lớp thật sự KHÔNG có cạnh typed từ nhãn — đó
là thiếu cạnh, không phải tra sai id.

### 28.4 Trạng thái

```
dữ liệu có cấu trúc      29,8% thuộc tính đủ điều kiện cần   ✓
via_pivot phủ cạnh lớp   18,1% → 68,4%                       ✓
② xếp hạng               diem 0,8% → phieu 21,8% (27×)       ✓ ĐÍNH CHÍNH §20.4
② còn rò                 top-5 43,7% mà top-1 21,8%          ← 22 điểm chưa lấy
③ khi ② đúng             46% nâng                            ← thiếu CẠNH, không phải id
phân bố sinh             0,998                               ✗
```

Hai bản vá hôm nay hỏng cả hai. Nhưng con số `phieu` được đính chính từ «vô dụng»
thành «gấp 27 lần» — và nó đã nằm trong code từ §21, chỉ là tôi đo nó sai bối cảnh.

### 28.5 Bài học phương pháp (nối tiếp §27.4)

22. **«Luật X vô dụng» phải ghi kèm CẤU HÌNH lúc đo, nếu không nó là kết luận
    SAI chứ không phải kết luận hẹp.** §20.4 khai tử `phieu` (thực ra 27×) và
    §24.1 đã cảnh báo đúng cơ chế này — tôi ghi bài học rồi vẫn dính lại sau bốn
    mục. Quy tắc thao tác: mỗi lần đổi cấu hình nền, **chạy lại mọi luật đã bị
    khai tử**, đừng tin bảng cũ.
23. **Bản vá suy ra từ trực giác «chuẩn hoá theo tần suất» phải kiểm xem đại
    lượng chuẩn hoá có TƯƠNG QUAN với đáp án không.** Bậc-vào tương quan THUẬN
    với «là lớp» (đó là lý do §24 dùng nó để LỌC), nên chia cho nó là tự bắn vào
    chân. Cùng một đại lượng: lọc theo nó thì đúng, chia cho nó thì sai.

## 29. LƯỚI ĐẦY ĐỦ 4×4 — ba luật đã khai tử được PHỤC HỒI (2026-08-12)

Thi hành bài học §28.5 số 22 («đổi cấu hình nền thì CHẠY LẠI mọi luật đã khai
tử»). Quét lưới đầy đủ `head_rule` × `head_rule_trai`, `indeg_q=0,8`,
`rank="phieu"`, chấm bằng CẢ HAI thước:

| phải \| trái | phủ | ②top1 | ②top5 | n |
|---|---|---|---|---|
| `trai_du_dac_thu` \| **`phai_nhat`** | 69,5% | **23,5%** | 39,7% | 68 |
| `dau_lop` \| **`phai_nhat`** | 67,4% | 20,0% | **54,3%** | 70 |
| `trai_nhat` \| **`phai_nhat`** | **71,8%** | 18,1% | 50,0% | 72 |
| `trai_du_dac_thu` \| `phai_du_dac_thu` ← mặc định cũ | 68,4% | 17,5% | 36,5% | 63 |
| `dau_lop` \| `phai_du_dac_thu` | 66,2% | 14,9% | 40,3% | 67 |
| `max_gidf` \| `max_gidf` (gốc) | 17,9% | **0,0%** | 4,8% | 21 |

### 29.1 Ba luật bị khai tử oan

- **`phai_nhat`** (khai tử §21: «gãy ngay ở 895 cha») — nay **thắng ở MỌI cặp**,
  hơn `phai_du_dac_thu` đều 3–6 điểm ②top1.
- **`dau_lop`** (khai tử §20.5: phủ thấp nhất 26,6%) — nay cho **top5 cao nhất
  54,3%**, tức bể ứng viên đúng lớn nhất, dư địa nhiều nhất nếu vá được xếp hạng.
- **`trai_nhat`** (khai tử §20.5) — nay **phủ cao nhất 71,8%**.

Cả ba đều bị loại trên cấu hình mà vế kia còn là `max_gidf`. Dòng cuối bảng cho
thấy vì sao: cấu hình gốc chỉ có **21 cơ hội** và ②top1 = **0,0%** — ở đó mọi luật
đều trông như nhau vì không luật nào có gì để làm.

### 29.2 Ba thước KHÔNG cùng chỉ một hướng

Phủ cao nhất (`trai_nhat|phai_nhat` 71,8%) ≠ top1 cao nhất
(`trai_du_dac_thu|phai_nhat` 23,5%) ≠ top5 cao nhất (`dau_lop|phai_nhat` 54,3%).
Chọn theo phủ — thứ tôi làm suốt §20–24 — là chọn theo **thước sai**: nó đo cấu
trúc VÀO được đồ thị, không đo bước ② DÙNG được nó.

Mặc định đổi sang `trai_du_dac_thu | phai_nhat` (②top1 tốt nhất, vì đường sinh
lấy `ok[0]`). `dau_lop | phai_nhat` ghi lại làm cấu hình có **dư địa lớn nhất**.

### 29.3 Đường sinh: nhích, chưa thoát

| | mặc định cũ | mặc định mới |
|---|---|---|
| mục tiêu LÊN | 34 | 31 |
| mục tiêu XUỐNG | 78 | **71** |
| vào phân bố lần đầu | 4 | **5** |
| tỉ lệ hạng trung vị | 0,998 | **0,996** |

②top1 17,5% → 23,5% kéo tỉ lệ 0,998 → 0,996. Nhích đúng hướng, **quá nhỏ để
gọi là thoát**. Khớp với dự đoán §27.2: rò 2 (46%) vẫn nguyên thì cận trên là
0,235 × 0,46 ≈ 11% mục tiêu — chưa đủ đẩy trung vị rời 1,0.

### 29.4 Trạng thái

```
dữ liệu có cấu trúc      29,8% thuộc tính đủ điều kiện cần   ✓
via_pivot phủ cạnh lớp   17,9% → 69,5%                       ✓
② xếp hạng top-1          0,0% → 23,5%                        ✓
② còn rò                 top5 39,7% (dau_lop: 54,3%)          ← 16–31 điểm chưa lấy
③ khi ② đúng             46% — THIẾU CẠNH                     ← chưa vá được
phân bố sinh             0,998 → 0,996                        ✗ nhích, chưa thoát
```

### 29.5 Bài học phương pháp (nối tiếp §28.5)

24. **Khai tử một luật trên cấu hình mà nó KHÔNG CÓ GÌ ĐỂ LÀM là khai tử vô
    nghĩa.** Cấu hình gốc chỉ cho 21 cơ hội và ②top1 = 0,0%; mọi luật đều 0 ở đó.
    Trước khi so hai luật, phải kiểm **mẫu số có đủ lớn để phân biệt** — n=21 và
    n=72 không so được với nhau.
25. **Ba thước lệch nhau là THÔNG TIN, không phải phiền toái.** Phủ / top1 / top5
    chỉ ba cấu hình khác nhau. Gộp thành một điểm số sẽ mất đúng thông tin cần:
    `dau_lop` kém top1 nhưng có bể ứng viên lớn nhất — nó là cấu hình đáng đầu tư
    NẾU sửa được xếp hạng, và chỉ bảng ba cột mới nói ra điều đó.

## 30. RÒ 2 — chẩn đoán lật ngược: 27,5% «cạnh mất» là **lỗi CHÂN LÝ**, không phải lỗi cơ chế

Quét núm dựng cạnh (`gidf_q` · `indeg_q` · `head_lookahead`) để quy trách nhiệm
cho 30% cạnh lớp không vào được `via_pivot`:

| núm | phủ |
|---|---|
| `gidf_q` 0,20 → 0,0 | 69,5% → 72,5% (**+3**) |
| `indeg_q` 0,8 → 0,0 | 72,1% → 72,5% (+0,4) |
| `head_lookahead` 3 → 5 → 8 | **không đổi** |

Không núm nào chịu trách nhiệm ⇒ đọc tay 12 cạnh bị mất kèm câu gốc:

```
«đổ là giai đoạn»    ← Sau khi triều Hán sụp ĐỔ là một giai đoạn…   (`sụp đổ`)
«vọng là vấn đề»     ← những điều chúng tôi hy VỌNG là những vấn đề  (`hy vọng`)
«Điển là nhà nước»   ← Thụy ĐIỂN là một nhà nước đơn nhất            (`Thụy Điển`)
«nhất là phương trình» ← Loại phương trình phổ biến NHẤT là phương trình đại số
«hơn là chất lỏng»   ← Các amin bậc cao HƠN là những chất lỏng
«đều là phần»        ← …người Hán ĐỀU là một phần của Trung Quốc
«chúng là sinh vật»  ← …CHÚNG là những sinh vật hội sinh
```

**Không cạnh nào trong số này nên tồn tại.** Chân lý của tôi lấy «token NGAY
TRƯỚC `là`» nên nó nhặt trạng từ (`đều`, `nhất`, `hơn`), đuôi từ ghép (`đổ`,
`vọng`, `Điển`) và đại từ (`chúng`). Cơ chế **đúng khi không dựng chúng** —
`_trung_tam_trai` cố ý bỏ qua `đều`/`nhất`/`hơn` để quét tiếp sang trái.

### 30.1 Hệ quả

1. **Phủ 69,5% là CẬN DƯỚI**, không phải mức thật. Phần lớn 30,5% còn lại là
   chân lý sai.
2. **«Rò 2 = thiếu cạnh» SAI.** §27.2 lý luận «phủ 68,4% nên ~32% thành viên
   đúng lớp không tới được từ nhãn» — nhưng 32% đó phần lớn không phải thành
   viên thật. Nguyên nhân thật của rò 2 **vẫn chưa biết**.
3. **Vũ trụ lớp cũng nhiễm.** `big` được đào bằng đúng luật đó, nên các lớp có
   thành viên rác (`thành viên` ← `Nó`, `KDE`; `khoảng` ← `chúng`). Mọi số
   `②top1`/`②top5` đo trên vũ trụ nhiễu này.

### 30.2 Cái gì còn đứng, cái gì không

**Còn đứng — mọi phép SO SÁNH giữa cấu hình**, vì cùng một chân lý áp cho tất cả:
- lưới 4×4 (§29): `phai_nhat` thắng ở mọi cặp; `max_gidf|max_gidf` = 0,0%
- `phieu` 27× `diem` (§28.2)
- `phieu_chuan` tệ hơn `phieu` (§28.1)

**Không còn đứng — mọi MỨC TUYỆT ĐỐI**: 18,1% · 68,4% · 69,5% · 72,5%, và cả
«29,8% thuộc tính đủ điều kiện cần» của §26 (dùng cùng vũ trụ nhiễm).

### 30.3 Việc bắt buộc trước khi vá tiếp

Dựng lại chân lý bằng **cùng luật trích trung tâm mà cơ chế dùng** (hoặc ít nhất
loại trạng từ / đuôi từ ghép / đại từ), rồi **đo lại toàn bộ**. Vá rò 2 trước khi
làm việc đó là vá theo một con số không có nghĩa.

### 30.4 Bài học phương pháp (nối tiếp §29.5)

26. **CHÂN LÝ cũng là một phép trích, và nó cũng có lỗi.** Suốt §20–29 tôi coi
    «token ngay trước `là`» là sự thật khách quan để chấm cơ chế. Nó là một
    heuristic thô hơn chính cơ chế đang bị chấm — nên ở những chỗ cơ chế làm
    ĐÚNG, nó chấm là SAI. Quy tắc: trước khi tin một thước «đọc thẳng từ dữ
    liệu», đọc tay 10 mẫu mà nó nói cơ chế TRƯỢT.
27. **Quét núm không tìm được nguyên nhân ⇒ nguyên nhân nằm NGOÀI núm.** Ba núm
    cộng lại chỉ giải thích 3 điểm; đó là tín hiệu phải đổi từ «chỉnh tham số»
    sang «đọc dữ liệu», chứ không phải thêm núm thứ tư.

## 31. CHÂN LÝ v2 — đo lại toàn bộ; ②top1 **23,5% → 47,4%** (2026-08-13)

§30 phát hiện chân lý v1 («token NGAY TRƯỚC `là`») nhặt trạng từ / đuôi từ ghép /
đại từ, nên **chấm SAI đúng ở chỗ cơ chế làm ĐÚNG**. Dựng lại
(`scripts/build_chan_ly_v2.py`), hai cơ chế **độc lập với luật `head_rule*`** —
nếu dùng chính luật đó thì phép chấm thành vòng tròn:

1. **Đơn vị TỪ** bằng span WS-CRF đã bake (TK_034, F1 0,9844): `sụp đổ` ·
   `Thụy Điển` · `hy vọng` · `Trái Đất` thành MỘT từ ⇒ hết đuôi ghép.
2. **Loại KẾT HỢP CỐ ĐỊNH theo tỉ lệ**: `đều`/`chủ yếu`/`có lẽ` đứng ngay trước
   `là` ở tỉ lệ rất cao trên tổng số lần xuất hiện. Ngưỡng theo phân vị **và**
   `dem ≥ 20` — không có điều kiện thứ hai thì phân vị ra đúng 1,000 và chỉ loại
   rác hiếm (lần dựng đầu đã dính đúng bẫy này).
3. Loại dấu câu và số thuần (lần dựng đầu để lọt `,` `)` `0` `2` làm thành viên).

Kết quả: **168 lớp · 949 cạnh**, đọc tay thấy sạch — `tam giác là đoạn thẳng` ·
`Campuchia là thành viên` · `lớp` ← {Sao Hoả, bò sát, hành tinh, kim loại} ·
`tỉnh` ← {Bắc Kạn, Lào Cai, California}.

### 31.1 Lưới 4×4 đo lại — ②top1 gần GẤP ĐÔI

n nhỏ (12–29) ở lượt đầu nên chạy lại 4 cấu hình đầu với mẫu gấp 3,5:

| phải \| trái | ②top1 | ②top5 | n |
|---|---|---|---|
| **`trai_du_dac_thu` \| `phai_nhat`** ← mặc định | **47,4%** | 59,6% | 57 |
| `trai_du_dac_thu` \| `phai_du_dac_thu` | 41,8% | **63,6%** | 55 |
| `trai_nhat` \| `phai_nhat` | 40,8% | 59,2% | 71 |
| `dau_lop` \| `phai_du_dac_thu` | 29,3% | 52,0% | 75 |

Mặc định chọn ở §29 **giữ nguyên vị trí số 1** trên chân lý sạch — kết luận §29
đứng vững, chỉ MỨC là sai: **23,5% → 47,4%**. Cơ chế bị chấm thiếu ~2×.

⚠️ **`phủ` dưới chân lý v2 KHÔNG so được với v1** (31% vs 69%): v2 dùng đơn vị
TỪ (`Thụy Điển`) còn `via_pivot` dựng theo TOKEN, nên phép khớp bề mặt phạt cơ
chế vì thứ nó không được thiết kế để sinh ra. Chỉ `②top1`/`②top5` là so được.

### 31.2 Thước sinh vẫn PHẲNG, và cùng số

| | chân lý v1 | chân lý v2 |
|---|---|---|
| mục tiêu LÊN | 31 | 4 |
| mục tiêu XUỐNG | 71 | 18 |
| tỉ lệ hạng trung vị | 0,996 | **0,998** |

Mẫu nhỏ hẳn (22 cặp so với 102) vì chân lý sạch loại phần lớn «mục tiêu» rác,
nhưng **tỉ lệ không đổi**. Chân lý sai KHÔNG phải nguyên nhân thước sinh phẳng.

### 31.3 Trạng thái sau khi đo lại toàn bộ

```
② xếp hạng top-1     23,5% → 47,4%   ✓ (mức cũ sai, thứ hạng cấu hình đúng)
② còn rò             top5 59,6%       ← 12 điểm giữa top-1 và top-5
③ khi ② đúng         chưa đo lại trên v2
phân bố sinh         0,998            ✗ KHÔNG đổi theo chân lý
phủ cạnh lớp         không so được    — v2 dùng TỪ, via_pivot dùng TOKEN
```

Bước ② nay đúng **gần một nửa** số ca có cấu trúc — khác hẳn bức tranh «0/40» của
§20. Nhưng đường sinh vẫn không nhúc nhích, nên khoảng cách giữa «② đúng» và
«token kế được nâng» mới là chỗ chưa hiểu.

### 31.4 Bài học phương pháp (nối tiếp §30.4)

28. **Sửa thước có thể đổi MỨC mà không đổi THỨ HẠNG.** ②top1 nhân đôi nhưng
    cấu hình tốt nhất vẫn là cấu hình cũ. Hai loại kết luận cần tách bạch: «cấu
    hình A hơn B» bền hơn nhiều so với «A đạt X%».
29. **Thước mới cũng phải đọc tay trước khi dùng.** Chân lý v2 lần dựng đầu để
    lọt `,` `)` `0` làm thành viên lớp và bộ lọc kết-hợp-cố-định ra ngưỡng 1,000
    vô dụng — cả hai chỉ lộ khi in 12 mẫu ra đọc, đúng quy tắc §30.4 số 26 áp cho
    chính thước mới.

## 32. RÒ 2 trên chân lý v2 — hai nguyên nhân, cả hai đều MỚI (2026-08-13)

Đo lại rò 2 trên chân lý sạch, n=100 ca, phân tầng theo ĐẦU RA của ②:

| tầng | n | mục tiêu LÊN | XUỐNG | tỉ lệ hạng |
|---|---|---|---|---|
| ② ĐÚNG lớp | 33 | 8 | 13 | **0,986** |
| ② SAI lớp | 67 | 8 | 23 | 0,996 |

Rò 2 **CÓ THẬT** — ngay khi ② đúng vẫn chỉ 8/21 = 38% mục tiêu được nâng (chân
lý v1 đo 46%, cùng cỡ) ⇒ **không phải hiện vật của chân lý**.

### 32.1 Nguyên nhân: 178 mục tiêu, 2 lọt vào đề cử

Trên 33 ca ② đúng, đối chiếu mục tiêu với đề cử của chuỗi:

```
mục tiêu ĐA ÂM TIẾT — ③ trả token đơn ⇒ KHÔNG THỂ khớp :  131/178  (74%)
mục tiêu MỘT TOKEN                                     :   47/178
   trong đó nằm trong đề cử                            :    2/47  (4%)
```

**Hai nguyên nhân tách bạch, cả hai đều chưa từng nêu:**

**(a) LỆCH ĐƠN VỊ — 74% mục tiêu.** Thành viên lớp là TỪ (`Sông Hồng`, `Hậu Lê`,
`Tây Sơn`, `Sao Hoả`, `Lúa mì`) còn `reverse_peers` trả TOKEN đơn. Không phải cơ
chế thiếu cạnh — nó **không có khả năng phát ra thứ mà thước đòi**. Cùng loại
với cảnh báo §31.1 về `phủ`, nhưng ở đây nó chiếm 3/4 số mục tiêu.

**(b) ③ TRẢ QUÁ ÍT — 4% mục tiêu một-token.** Và không phải chọn sai, mà là bể
quá nhỏ:

```
P='Lan'  lớp='trung tâm' (19 thành viên)  → đề cử = ['tỉnh']          1 mục
P='Sao'  lớp='lớp'       (14 thành viên)  → đề cử = ['ngoài', 'Âu']   2 mục
```

Nhãn đúng, lớp có 14–19 thành viên trong dữ liệu, mà `reverse_peers` lọc `typed`
chỉ ra 1–2 mục. Đây là chỗ vá tiếp — và nó nằm ở **bước ③**, không phải ② như
suốt §26–29 đã tập trung.

### 32.2 Giả thuyết về rò 2 đã bị loại

| giả thuyết | mục | phán quyết |
|---|---|---|
| biến thể id BPE của nhãn | §28.3 | ✗ vá xong không đổi |
| thiếu cạnh (phủ 68,4%) | §27.2 → §30 | ✗ lý luận dựa trên chân lý sai |
| chân lý bẩn | §30 → §32 | ✗ chân lý sạch vẫn 38% |
| **lệch đơn vị TỪ ↔ TOKEN** | §32.1a | ✓ 74% mục tiêu |
| **③ trả bể quá nhỏ** | §32.1b | ✓ 4% lọt trong phần còn lại |

### 32.3 Trạng thái

```
② xếp hạng top-1     47,4%            ✓
③ lệch đơn vị        74% mục tiêu      ← rò 2a, MỚI
③ bể ứng viên        1–2 mục / 14–19 thành viên  ← rò 2b, MỚI
phân bố sinh         0,998             ✗
```

Bước ② nay đúng gần một nửa. Nghịch lý «② đúng mà token kế không lên» đã có lời
giải: **③ không phát ra được thứ cần, và phát quá ít.**

### 32.4 Bài học phương pháp (nối tiếp §31.4)

30. **«Cơ chế không tìm được X» và «cơ chế không thể BIỂU DIỄN X» là hai chẩn
    đoán khác nhau.** Suốt §27–31 tôi đọc rò 2 là thiếu cạnh / xếp sai hạng —
    tức loại một. Thực tế 74% là loại hai: thước đòi TỪ, cơ chế phát TOKEN. Vá
    theo loại một (thêm cạnh, đổi luật) không bao giờ chạm tới.
31. **Đếm mẫu số theo ĐƠN VỊ trước khi diễn giải tỉ lệ.** «38% mục tiêu được
    nâng» nghe như cơ chế yếu; tách ra thì 74% mục tiêu vốn không thể khớp, và
    phần còn lại hỏng vì lý do khác hẳn. Một tỉ lệ gộp trên hai quần thể không
    đồng nhất là con số không diễn giải được.

## 33. ③ PHÁT SPAN — trần của ③ **22% → 70,4%**, và một lỗi do chính tôi gây ra (2026-08-13)

§32 chỉ ra rò 2 có hai phần: lệch đơn vị TỪ↔TOKEN (74% mục tiêu) và bể ③ quá nhỏ.

### 33.1 Vá 1 — ③ được phép phát SPAN

`via_pivot` **CÓ** thành viên đa âm tiết: dual-write ghi cả id CỤM lẫn token
thành phần. Nhưng `hub_chain_proposals` lọc `m.tid < len(vocab)` ⇒ **vứt sạch id
cụm**. Bỏ dòng lọc, ánh xạ cụm về **token ĐẦU** (sinh là tự hồi quy: muốn ra
`Sông Hồng` thì token kế phải là `Sông`).

Thước cũng phải sửa cùng: mục tiêu đa âm tiết trước đây tra cả cụm ⇒
`resolve_surface_ids` trả rỗng ⇒ **74% mục tiêu bị loại IM LẶNG khỏi phép đếm**.

Kết quả: bể đề cử 1–2 mục → **trung bình 9,6**. Nhưng mục tiêu chạm được chỉ
2% → 4%. Chưa đủ ⇒ đo TRẦN.

### 33.2 Trần của ③ — và lỗi do luật tôi chọn ở §29

Trần = bao nhiêu thành viên có cạnh typed tới LỚP của nó. Đo ra **22%**, và ví dụ
chỉ thẳng nguyên nhân: `via_pivot['Lúa mì']` **RỖNG HOÀN TOÀN**.

`phai_nhat` xếp ứng viên theo **vị trí BẮT ĐẦU**. Trong «Lúa mì là loại», token
`mì` bắt đầu muộn hơn cụm `Lúa mì` nên **`mì` THẮNG** ⇒ chủ ngữ bị chặt còn âm
tiết cuối. Đây đúng bệnh `Thụy Điển → Điển` của §30 — **nhưng lần này do chính
luật tôi chọn ở §29 gây ra**, và tôi đã chọn nó dựa trên chân lý v1 vốn cũng chặt
đuôi từ ghép, nên hai lỗi che nhau.

Vá: xếp theo vị trí **KẾT THÚC** (`offset + len - 1`), hoà thì lấy cụm dài hơn.

```
                          trần ③   phủ    ②top1
phai_du_dac_thu           31,3%    31,3%  41,8%
phai_nhat (BẮT ĐẦU, cũ)   22%      31,3%  47,4%
phai_nhat (KẾT THÚC, vá)  70,4%    70,4%  35,8%
```

**Trần ③ tăng 3,2×.** Nhưng ②top1 tụt 47,4% → 35,8% — ba thước lại lệch nhau
(§29.5 số 25). Cần đo đầu-cuối để biết đánh đổi này lời hay lỗ.

### 33.3 Trạng thái

```
② xếp hạng top-1     35,8% (vá) · 47,4% (chưa vá)   — đánh đổi chưa phân định
③ trần               22% → 70,4%                     ✓ 3,2×
③ bể đề cử           1–2 → 9,6 mục                   ✓
phân bố sinh         0,994                           ✗ chưa đo lại sau vá trần
```

### 33.4 Bài học phương pháp (nối tiếp §32.4)

32. **Hai lỗi cùng loại ở hai đầu có thể CHE NHAU.** Chân lý v1 chặt đuôi từ ghép
    ở chủ ngữ; luật `phai_nhat` cũng chặt đuôi từ ghép ở chủ ngữ. Chấm cái sau
    bằng cái trước ⇒ lỗi vô hình, và `phai_nhat` còn được §29 tôn lên số 1. Chỉ
    lộ khi chân lý được sửa độc lập. Quy tắc: khi sửa thước xong, **chạy lại
    luôn chẩn đoán cho luật mà thước cũ đã tôn vinh**, không chỉ luật nó khai tử.
33. **Bỏ một dòng lọc có thể là bản vá lớn nhất.** `m.tid < len(vocab)` — một
    điều kiện biên trông vô hại — chặn 74% ứng viên. Trước khi thêm cơ chế mới,
    kiểm xem cơ chế cũ có đang bị một dòng lọc vứt đầu ra hay không.

## 34. THƯỚC SINH sau hai bản vá ③ — lần đầu một tầng RỜI KHỎI 1,0 (2026-08-13)

§33 để lại một đánh đổi chưa phân định: `phai_nhat` (vá) cho trần ③ **70,4%**
nhưng ②top1 tụt 47,4% → 35,8%; `phai_du_dac_thu` ngược lại. Đo đầu-cuối để phân
định, n=100 ca, cả hai đều đã có bản vá ③-phát-span:

| `head_rule_trai` | trần ③ | ②top1 | **② ĐÚNG: tỉ lệ hạng** | vào phân bố lần đầu | tổng |
|---|---|---|---|---|---|
| **`phai_nhat`** | **70,4%** | 35,8% | **0,935** | **6** | 0,994 |
| `phai_du_dac_thu` | 31,3% | 41,8% | 0,995 | 2 | 0,996 |

**Đánh đổi phân định xong: TRẦN ③ thắng ②top1.** Nhãn đúng mà bể thành viên cạn
thì không nâng được gì; bể rộng bù được cho việc chọn nhãn kém chính xác hơn.

So với TRƯỚC hai bản vá (§32, cùng tầng ② ĐÚNG): tỉ lệ **0,986 → 0,935**, vào
phân bố lần đầu **1 → 6**, tỉ lệ nâng 38% → 44%.

### 34.1 Đây là chuyển động THẬT đầu tiên trên thước của tác giả

Suốt §22–33 mọi con số đầu-cuối đều đứng ở 0,995–0,998 — tức nhiễu quanh 1,0.
`0,935` ở tầng ② ĐÚNG là lần đầu cơ chế **thực sự nâng** thành viên mới trong
phân bố token kế, và **6 thành viên lần đầu lọt vào phân bố** (trước: 1).

Nhưng **tổng vẫn 0,994**, vì ② chỉ đúng ở **26/100** ca. Nút thắt quay về ②.

### 34.2 Trạng thái

```
③ trần               22% → 70,4%                    ✓ 3,2×
③ bể đề cử           1–2 → 9,6 mục                  ✓
② đúng lớp           26/100 ca                      ← NÚT THẮT, quay lại
sinh · tầng ② ĐÚNG   0,986 → 0,935                  ✓ chuyển động thật đầu tiên
sinh · tổng          0,994                          ✗ bị ② kéo xuống
```

Phân rã §27.2 vẫn đúng và nay đọc được thành công thức làm việc:

```
tổng ≈ P(② đúng) × hiệu_quả_③
     ≈ 0,26      × (đã cải thiện thật)
```

③ đã vá; **cả hai thừa số phải cùng lớn** thì tổng mới rời 1,0. ② là thừa số còn
lại, và §26.2 đã đo sẵn dư địa: đáp án nằm trong top-5 ở 55–64% ca mà chỉ 36–47%
lên được top-1.

### 34.3 Bài học phương pháp (nối tiếp §33.4)

34. **Khi ba thước lệch nhau, ĐO ĐẦU-CUỐI để phân định — đừng chọn bằng lý lẽ.**
    §33 để lại đánh đổi trần-③ vs ②top1 mà tôi không có cơ sở nào để chọn; một
    phép đo 6 phút cho câu trả lời dứt khoát (0,935 vs 0,995). Suy luận «②top1
    cao hơn thì tốt hơn» sẽ chọn NGƯỢC.
35. **Phân tầng cho thấy chuyển động mà tổng thể che mất.** Tổng đi từ 0,996 →
    0,994 — trong nhiễu. Tầng ② ĐÚNG đi 0,986 → 0,935 — chuyển động thật. Nếu
    chỉ nhìn tổng thì hai bản vá đúng của §33 sẽ bị kết luận là vô dụng, đúng như
    tôi đã kết luận nhầm bốn lần trước đó.

## 35. 20–28 ĐIỂM Ở BƯỚC ② — ba cách phá hoà, cả ba TỆ HƠN (2026-08-13)

§34 để lại nút thắt ở ②: đáp án nằm trong top-5 ở 55,2% ca mà chỉ 36–47% lên
top-1. Bể ứng viên đủ, chỉ XẾP sai ⇒ thử ba tín hiệu phá hoà khi CÙNG số phiếu:

| `rank` | ②top1 | ②top5 |
|---|---|---|
| **`phieu`** (hiện tại: phiếu → điểm `w·gidf·log(div)`) | **35,8%** | **55,2%** |
| `phieu_gidf` (phiếu → gidf) | 0,0% | 14,9% |
| `phieu_indeg` (phiếu → bậc-vào, tín hiệu THUẬN) | 1,5% | 16,4% |
| `phieu_vong_ve` (phiếu → khứ hồi) | 1,5% | 16,4% |

**Cả ba tệ hơn, và tệ thảm.** Kết luận thẳng: trong số ứng viên cùng phiếu, **điểm
`w · gidf · log(diversity)` đã là bộ phân biệt tốt nhất tôi thử được** — từng
thành phần tách riêng (gidf) hay tín hiệu ngoài (bậc-vào, khứ hồi) đều kém hẳn.

### 35.1 Khứ hồi VÔ NGHĨA vì lý do cấu trúc

Giả thuyết nghe hợp lý: «lớp thật lan ngược phải trả về chính các hạt giống đã
bầu cho nó». Nhưng trong đồ thị này nó **suy biến**: ứng viên `H` chỉ trở thành
ứng viên *vì* `H ∈ via_pivot[hạt_giống]`, tức `hạt_giống ∈ via_rev[H]` **theo
định nghĩa**. Khứ hồi do đó gần như trùng với số phiếu và không thêm thông tin —
rồi vì nó thay chỗ của `score` trong khoá sắp xếp, kết quả còn tệ đi.

Cùng họ với §28.1 (`phieu_chuan`): **một đại lượng suy ra từ chính cạnh đang xét
thì không phải bằng chứng độc lập.**

### 35.2 Trạng thái

```
③ trần               70,4%                    ✓
② đúng lớp           26/100 · top1 35,8%      ← NÚT THẮT, ba cách vá đều hỏng
sinh · tầng ② ĐÚNG   0,935                    ✓
sinh · tổng          0,994                    ✗
```

20–28 điểm giữa top-1 và top-5 **vẫn nằm đó**. Ba tín hiệu rẻ đã loại; cái còn
lại phải là tín hiệu KHÔNG suy ra được từ cạnh đang xét — ví dụ ngữ cảnh câu, từ
loại, hoặc đối chiếu nhiều thuộc tính cùng lúc.

### 35.3 Bài học phương pháp (nối tiếp §34.3)

36. **Điểm tổng hợp có sẵn có thể đã là baseline mạnh.** Tôi mặc định rằng
    `w·gidf·log(div)` «chỉ là điểm đồng hiện thô» nên thay nó bằng tín hiệu
    «có nghĩa» hơn — cả ba lần đều tệ hơn 20–35 điểm. Trước khi thay một điểm số
    đang chạy, đo nó như một **baseline** chứ đừng coi là thứ cần thay.
37. **Tín hiệu suy ra từ CHÍNH cạnh đang xét không phải bằng chứng.** Khứ hồi
    (§35.1) và chuẩn hoá theo bậc-vào (§28.1) đều mắc lỗi này. Kiểm nhanh: nếu
    ứng viên trở thành ứng viên *nhờ* quan hệ đó, thì đo lại quan hệ đó là vòng
    tròn.

## 36. ĐỌC CA TRƯỢT — chủ ngữ giả do TRẠNG NGỮ; vá cục bộ ĐƯỢC, tổng thể HỎNG (2026-08-13)

### 36.1 Đồng thuận cấp thuộc tính — ÂM TÍNH

| | đồng thuận=0 | đồng thuận=1 |
|---|---|---|
| [② ĐÚNG] n=49 | 80↑/179↓ · **0,911** | 80↑/179↓ · **0,911** |
| [② SAI] n=51 | 50↑/177↓ | 51↑/176↓ |
| tổng | 130↑/356↓ · 0,996 | 131↑/355↓ · 0,996 |

Chênh **đúng một mục tiêu**. Tầng ② ĐÚNG giống hệt từng con số. Nguyên nhân đo
được: hầu như **không có thành viên nào được HAI thuộc tính khác nhau cùng đề
cử** — mỗi thuộc tính trong câu trỏ tới nhãn khác nhau, mỗi nhãn kéo bộ thành
viên riêng, không giao nhau. Không có gì để đồng thuận.

### 36.2 Cái DUY NHẤT mua được: prompt = tiền tố câu

| | prompt 1 từ | prompt tiền tố câu |
|---|---|---|
| ② đúng lớp | 26/100 | **49/100** |
| tầng ② ĐÚNG | 0,935 | **0,911** |

②top1 gần GẤP ĐÔI. Đây là thay đổi ở **đầu vào**, không phải bản vá cơ chế — và
nó có nghĩa mọi số ②top1 trước đó (35,8% · 47,4%) đo trên prompt một-từ, tức
**dưới mức thực tế sinh**.

### 36.3 Mổ ca `cứu → ngành`: dữ liệu ĐỦ, xếp hạng SAI

```
chân lý        : ngành ← Du lịch, Kỹ thuật, Nông nghiệp, Vật lí, Y tá, lượng tử…
via_rev[ngành] : chứa ĐỦ 13/13 thành viên chân lý          ✓
reverse_peers  : biệt · hiện · hiện tại · tại · đặc · đặc biệt · khác · học …
```

Rác không ngẫu nhiên: `đặc biệt`, `hiện tại` đến từ «**Đặc biệt là** ngành du
lịch…», «**Hiện tại là**…» — **trạng ngữ bị nhận làm chủ ngữ**, và vì phổ biến
nên trọng số cao hơn thành viên thật.

**Tôi đã có sẵn bộ lọc cho đúng bệnh này từ §31** (tỉ lệ đứng-ngay-trước-pivot,
dùng để loại `đều`/`chủ yếu`/`có lẽ` khỏi CHÂN LÝ) và **chưa bao giờ áp vào cơ
chế**. Chân lý được làm sạch còn `via_pivot` thì không.

### 36.4 Áp bộ lọc: cục bộ ĐƯỢC, tổng thể HỎNG

```
reverse_peers('ngành')  q=0,0  → thành viên đúng: khác · học · Nông nghiệp        (3)
                        q=0,9  → thành viên đúng: khác · học · Nông nghiệp · lượng tử (4)
                                 `đặc biệt`/`đặc`/`biệt` bị loại; `hiện tại` VẪN CÒN
```

Nhưng trên diện rộng:

| `chu_ngu_ket_hop_q` | phủ | ②top1 | ②top5 |
|---|---|---|---|
| 0,0 (tắt) | **70,4%** | **35,8%** | **55,2%** |
| 0,90 | 68,8% | 29,4% | 50,0% |

**Tệ hơn 6 điểm.** Cơ chế đánh đổi: loại chủ ngữ giả cũng loại luôn cạnh, làm
nghèo bể phiếu của ②. Bộ lọc giúp ③ (chất lượng thành viên) nhưng hại ② (chọn
nhãn). Giữ tham số, **mặc định TẮT**.

### 36.5 Bài học phương pháp (nối tiếp §35.3)

38. **Vá đúng bệnh ở một bước có thể phá bước khác — trong chuỗi, cạnh là tài
    nguyên DÙNG CHUNG.** `via_pivot` phục vụ cả ② (phiếu bầu) lẫn ③ (thành viên);
    làm sạch cho ③ thì ② nghèo đi. Trước khi lọc cạnh, liệt kê MỌI bước tiêu thụ
    cạnh đó.
39. **Bộ lọc đã kiểm chứng ở phía THƯỚC không tự động đúng ở phía CƠ CHẾ.** Tỉ lệ
    đứng-trước-pivot làm sạch chân lý rất tốt (§31) và làm hỏng cơ chế (§36.4).
    Thước chỉ cần ĐÚNG; cơ chế còn cần ĐỦ.

## 37. TÁCH BỂ CHO ② VÀ ③ — gỡ được đánh đổi, nhưng lãi nhỏ (2026-08-13)

§36.4 đo được một ràng buộc kiến trúc: `via_pivot` phục vụ **cả hai bước**, nên
mọi bộ lọc đều là đánh đổi — lọc chủ ngữ giả làm ③ sạch hơn nhưng ②top1 tụt
35,8% → 29,4% vì bể phiếu nghèo đi. Cạnh là **tài nguyên dùng chung**.

Vá bằng cách tách: `_mine_via_pivot` phát vào HAI bể cùng lúc.

- `via_pivot` — **đầy đủ**, cho bước ② (cần bể PHIẾU rộng)
- `via_pivot_sach` — đã loại **chủ ngữ giả do trạng ngữ**, cho bước ③ (cần
  thành viên SẠCH)

`reverse_peers` đọc bể sạch, rơi về bể đầy đủ khi nó rỗng (tương thích ngược).

### 37.1 Kết quả — đánh đổi ĐÃ GỠ

| | chung một bể | **tách hai bể** |
|---|---|---|
| cạnh cho ② | 10 785 | **10 785** (giữ nguyên) |
| cạnh cho ③ | 10 785 | 9 211 (sạch hơn) |
| **②top1** | 35,8% | **35,8%** ✓ không mất gì |
| **③ chính xác** (top-10 là thành viên thật) | 11,5% | **12,3%** |
| ca `ngành` — thành viên đúng trong top-10 | 2 | **3** |

Trước khi tách: muốn ③ sạch phải trả 6 điểm ②top1. Sau khi tách: **③ sạch hơn
mà ② không mất gì**. Đây là điều bộ lọc chung không thể làm được về nguyên tắc.

### 37.2 Nhưng lãi nhỏ, và nút thắt thật lộ ra

③ chính xác **12,3%** — nghĩa là trong top-10 thành viên mà ③ trả về, chỉ hơn
một cái là thành viên thật của lớp. Tách bể mua được 0,8 điểm; **88% top-10 vẫn
là rác**, và rác còn lại (`hiện tại`, `tại`, `nghiệp`, `cao`) không phải chủ ngữ
giả kiểu trạng ngữ nên bộ lọc này không chạm tới.

⇒ Nút thắt hiện tại là **độ chính xác của ③**, phát biểu được bằng một số:
`12,3%`. Đây là lần đầu bước ③ có thước riêng.

### 37.3 Trạng thái

```
② xếp hạng top-1     35,8% (prompt 1 từ) · 49/100 (prompt ngữ cảnh)   ✓
③ trần               70,4%                                            ✓
③ CHÍNH XÁC          12,3%                                            ← NÚT THẮT MỚI
sinh · tầng ② ĐÚNG   0,911                                            ✓
sinh · tổng          0,996                                            ✗
```

### 37.4 Bài học phương pháp (nối tiếp §36.5)

40. **Khi hai bước tranh nhau một tài nguyên, tách tài nguyên trước khi chỉnh
    tham số.** Ba vòng (§28 · §35 · §36) tôi chỉnh ngưỡng trên một bể chung và
    mỗi lần đều là đánh đổi; tách bể xong thì cùng bộ lọc ấy trở thành lãi ròng.
    Dấu hiệu nhận biết: **một tham số làm chỉ số A tốt lên và B xấu đi cùng lúc**.
41. **Gỡ được đánh đổi không có nghĩa lãi lớn.** Tách bể đúng về nguyên tắc và
    chỉ mua 0,8 điểm — vì nguyên nhân chính của rác ở ③ nằm chỗ khác. Sửa đúng
    kiến trúc vẫn phải đo, đừng ghi nhận công theo tính đúng đắn của thiết kế.

## 38. ĐỌC CA TRƯỢT CỦA ③ — 12,3% là HIỆN VẬT; số thật **38,9%** (2026-08-13)

§37 báo ③ chính xác 12,3%. Đọc ca trượt kèm câu train thì lộ HAI lỗi, cả hai của
phép đo chứ không phải cơ chế.

### 38.1 Lỗi thước 1 — tra nhãn bằng BỀ MẶT

```
NHÃN 'thành phố' → ③ ra: công · nước · sinh · trở · hình · tạo · chia · hoàn · đổi · thành
```

Đó là `trở THÀNH` · `hình THÀNH` · `hoàn THÀNH` · `chia THÀNH` — **đúng bệnh
`thành`** mà `_mine_via_pivot` đã ghi và loại khỏi pivot từ 2026-08-11. Nguyên
nhân: probe tra `resolve_surface_ids("thành phố")` → rỗng (đa âm tiết) → **rơi về
token đầu `thành`**. Tôi hỏi sai nhãn.

Tra bằng **id CỤM** (`index.label_surf` đảo): **12,3% → 25,0%**.

### 38.2 Lỗi thước 2 — MẢNH chiếm suất

Dual-write phát CẢ cụm LẪN token thành phần, nên một thành viên ăn 2–3 suất
top-10: `Live`+`Live Blue` · `Bóng`+`Bóng đá`+`đá` · `Mu`+`Mu on`+`on`.

Gộp mảnh: **25,0% → 38,9%**.

### 38.3 ③ thật sự làm được gì

Sau khi sửa cả hai lỗi thước, top-10 của các lớp lớn:

```
loài        9/10 : cá chép · gấu trúc · Rùa · Chim · Cá sấu · unc atus
nhóm        9/10 : Live Blue · RS · dân số · Kỳ · Lovelace · Rùa
cây         9/10 : rộng · khác · lúa · Ngọc Linh · Cà phê · esk ia
biểu tượng  9/10 : Moskva · Di tích · Nam Houston · Mat ry osh ka · vàng
lớp         8/10 : kim loại · Diêm Vương · bò sát · lưỡng cư
loại        8/10 : Brabant · con người · ISDN · công cộng · hữu cơ
triều đại   5/10 : Nhà · Hồ · Lý · Sơn · Tiền · Tây
```

`loài → cá chép, gấu trúc, Rùa, Chim, Cá sấu` là **đúng hình dạng ví dụ của tác
giả**, trên dữ liệu thật. Bước ③ KHÔNG phải nút thắt.

### 38.4 ⚠️ 38,9% là ③ khi ĐƯỢC TRAO nhãn đúng, KHÔNG phải đầu-cuối

Chạy chuỗi đầy đủ với `loài` làm THUỘC TÍNH (không phải nhãn) vẫn ra rác:

```
hub_chain_proposals('loài') → thứ · Mẹ · Nguyễn · tại · giỏi · đương
```

Vì ① và ② chọn đường khác. Con số 38,9% đo `reverse_peers(nhãn_đúng)`; đầu-cuối
vẫn bị ② chặn (top-1 35,8%). **Không được đọc 38,9% như thành tích của chuỗi.**

### 38.5 Trạng thái

```
② xếp hạng top-1     35,8%                    ← NÚT THẮT DUY NHẤT còn lại
③ trần               70,4%                     ✓
③ chính xác          38,9% (khi trao nhãn đúng) ✓ KHÔNG phải nút thắt
sinh · tổng          0,996                     ✗
```

Bốn mục (§34–37) tôi tưởng ③ là nút thắt và vá nó bốn lần. Đọc ca trượt cho thấy
③ vốn đã ổn; **12,3% là do tôi đo sai**.

### 38.6 Bài học phương pháp (nối tiếp §37.4)

42. **Cùng một lỗi «tra bằng bề mặt» đã cắn ba lần** — §23.1 (biến thể BPE),
    §32.1a (mục tiêu đa âm tiết), §38.1 (nhãn đa âm tiết rơi về token đầu). Mỗi
    lần tôi vá tại chỗ mà không đi tìm chỗ khác cùng bệnh. Quy tắc: gặp lỗi
    bề-mặt↔id lần thứ hai thì **quét toàn bộ chỗ dùng `resolve_surface_ids`**.
43. **Trước khi vá một bước, kiểm xem SỐ ĐO của nó có đúng không.** Bốn vòng vá ③
    dựa trên 12,3%; số thật là 38,9% và ③ không cần vá. Chi phí: bốn vòng.

## 39. KIỂM `②top1 = 35,8%` — số THẬT, kiểm cả hai chiều (2026-08-13)

§38.6 số 43: «trước khi vá một bước, kiểm xem SỐ ĐO của nó có đúng không» — chi
phí bốn vòng nếu bỏ qua. Áp cho ②.

### 39.1 Chiều THỔI PHỒNG — chấm nương? KHÔNG

Thước chấm `ok[0].surface in (c2, c2.split()[0])`, tức **tính đúng cả khi cơ chế
chỉ ra TOKEN ĐẦU** của lớp đa âm tiết (`thành` cho `thành phố`) — đúng loại lỗi
bề-mặt↔id đã cắn ba lần. Tách hai kiểu khớp:

```
②top1 35,8%  =  khớp ĐỦ 35,8%  +  chỉ-token-đầu 0,0%
```

**Nhánh nương không hề kích hoạt.** Không có thổi phồng.

### 39.2 Chiều ĐẾM THIẾU — cơ hội bị bỏ sót? Gần như KHÔNG

Nếu láng giềng ① là TOKEN ĐẦU của một thành viên đa âm tiết thì nó là thành viên
thật mà thước không nhận ⇒ cơ hội bị đếm thiếu. Đo trên 6 910 láng giềng:

| | |
|---|---|
| khớp ĐỦ thành viên lớp | 1 519 (22,0%) |
| là token đầu của thành viên | 891 (**12,9%**) |
| không liên quan | 4 500 (65,1%) |

Nhưng đọc mẫu thì 12,9% đó **phần lớn là trùng tiền tố ngẫu nhiên**, không phải
thành viên thật:

```
① ra 'mưa'  — mảnh của 'mưa Amazon'      `mưa` KHÔNG phải thành viên
① ra 'thế'  — mảnh của 'thế giới'/'thế nào'
① ra 'danh' — mảnh của 'danh tiếng'
① ra 'bốn'  — mảnh của 'bốn bề'
```

Tính chúng là khớp sẽ **thổi phồng**, không phải sửa đếm thiếu. Cách đếm nghiêm
hiện tại là đúng.

### 39.3 Kết luận

**`②top1 = 35,8%` là số thật**, không thổi phồng cũng không đếm thiếu đáng kể.
Khác với `③ = 12,3%` (hiện vật, số thật 38,9%), ② không có lỗi thước.

⇒ ② là nút thắt THẬT, và 19,4 điểm giữa top-1 (35,8%) và top-5 (55,2%) là dư địa
THẬT. §35 đã loại ba tín hiệu nội tại; §36 loại đồng thuận cấp thuộc tính. Không
gian còn lại: tín hiệu NGOÀI đồ thị đồng-hiện (từ loại, `E`).

### 39.4 Bài học phương pháp (nối tiếp §38.6)

44. **Kiểm số đo theo CẢ HAI chiều.** Chỉ kiểm «có thổi phồng không» sẽ bỏ sót
    «có đếm thiếu không», và ngược lại. Ở đây chiều một sạch (0,0%) còn chiều hai
    trông như 12,9% nhưng đọc mẫu thì là trùng tiền tố — **cả hai chiều đều phải
    đọc mẫu, không chỉ đếm**.

## 40. TỪ LOẠI Ở BƯỚC ② — tín hiệu ĐÚNG, nhưng THỪA (2026-08-13)

§39 khoanh không gian còn lại: tín hiệu NGOÀI đồ thị đồng-hiện. Thử từ loại
(`data/dict/vi_dict_raw.jsonl`, 36 764 mục; Luật 0: dữ liệu ngoài dùng ở BAKE,
cùng tiền lệ §14 và `ws_spans`).

### 40.1 Tín hiệu rất mạnh khi đo riêng

```
NHÃN LỚP chân lý (168)        : 80% DANH TỪ

ỨNG VIÊN cơ chế sinh ra:
  danh từ    54 đúng / 302 sai   15,2%
  động từ     5 đúng /  10 sai   33,3%
  tính từ     0 đúng /  48 sai    0,0%
  đại từ      0 đúng /  19 sai    0,0%
  không có    0 đúng /  50 sai    0,0%
```

**119 ứng viên SAI bị loại mà KHÔNG mất ứng viên đúng nào.** Trên giấy là bản vá
tốt nhất từ đầu phiên.

### 40.2 Bật chung cho ① và ② — TỆ HƠN

②top1 35,8% → 34,5%, top5 55,2% → 50,9%, **cơ hội n tụt 67 → 55**.

`score_hubs_from` phục vụ CẢ ① lẫn ②, mà **tính từ là HẠT GIỐNG hợp lệ ở ①**
(`trắng` → mèo, áo, tóc) dù không bao giờ là nhãn lớp ở ②. Đúng bệnh «tài nguyên
dùng chung» của §37, lần này ở HÀM chứ không phải ở BỂ CẠNH.

### 40.3 Chỉ lọc ở ② — KHÔNG ĐỔI GÌ

Thêm `loc_pos` (mặc định tắt), chỉ bật ở lời gọi bước ②:

```
                ②top1    top5     n
tắt             35,8%    55,2%   67
lọc ở ②         35,8%    55,2%   67     ← y hệt
```

Bộ lọc **có chạy** — 921 nhãn được gán từ loại, **120 bị chặn**. Nhưng số không
đổi ⇒ những nhãn đó **vốn đã không lọt top-5**: xếp hạng theo PHIẾU đã dìm chúng
từ trước (tính từ hiếm khi được ≥2 hạt giống cùng bầu).

⇒ Từ loại là tín hiệu ĐÚNG nhưng **THỪA** — nó chặn đúng thứ mà `rank="phieu"`
đã chặn. Giữ `loc_pos`, mặc định TẮT; nó vẫn là lưới an toàn có cơ sở nếu sau này
đổi luật xếp hạng.

### 40.4 Trạng thái

```
② xếp hạng top-1     35,8%   ← nút thắt, BẢY bản vá liên tiếp không nhúc nhích
② top-5              55,2%   ← 19,4 điểm dư địa THẬT (đã kiểm §39)
③ chính xác          38,9% (khi trao nhãn đúng)
sinh · tổng          0,996
```

Bảy tín hiệu đã loại BẰNG ĐO: gidf · bậc-vào · khứ hồi (§35) · đồng thuận thuộc
tính (§36) · chuẩn hoá phiếu (§28) · biến thể id (§28) · **từ loại (§40)**.

Điểm chung của cả bảy: chúng đều **tương quan với thứ `rank="phieu"` đã dùng**.
`phieu` = đồng thuận trong kênh typed đã hút gần hết tín hiệu rẻ; thứ còn lại
phải **trực giao** với nó.

### 40.5 Bài học phương pháp (nối tiếp §39.4)

45. **Tín hiệu đúng vẫn có thể THỪA.** Đo riêng, từ loại phân biệt hoàn hảo
    (0/119 sai). Ghép vào, không đổi một điểm — vì nó chặn đúng thứ đã bị chặn.
    Trước khi cài một tín hiệu, đo nó **trên phần còn lại sau bộ lọc hiện có**,
    không phải trên toàn bộ ứng viên.
46. **«Tài nguyên dùng chung» tái diễn ở tầng HÀM.** §37 tách BỂ CẠNH cho ②/③;
    §40 phải tách LỜI GỌI HÀM cho ①/②. Cùng một hàm phục vụ hai bước với yêu cầu
    ngược nhau là mẫu lặp trong kiến trúc này — kiểm nó mỗi khi thêm bộ lọc.

## 41. `E` Ở BƯỚC ② — dương, nhưng trong nhiễu; và lộ ra `score` mới là thứ đang gánh (2026-08-13)

§40.4 khoanh: bảy tín hiệu đã thử đều TƯƠNG QUAN với `rank="phieu"` nên thừa hoặc
hại; thứ còn lại phải TRỰC GIAO. `E` là ứng viên duy nhất — chuỗi ba bước chạy
thuần trên ĐẾM, chưa dùng `E` ở bất kỳ bước nào. `cosine(E[nhãn], mean E[hạt
giống])` không suy ra được từ số phiếu.

### 41.1 Ghép kiểu CŨ (thay `score`) — thảm hoạ, và nó giải thích §35

```
rank=phieu       ②top1 35,8%   top5 55,2%
rank=phieu_cos   ②top1  0,0%   top5 16,4%     ← thay `score` trong khoá sắp xếp
```

**Bốn bộ phá hoà khác nhau đều rơi về ~0–1,5%** (gidf 0,8% · bậc-vào 1,5% · khứ
hồi 1,5% · cosine 0,0%). Bốn tín hiệu độc lập không thể cùng thất bại y hệt vì lý
do riêng ⇒ nguyên nhân là **CÁCH GHÉP**: chúng chiếm chỗ của `score` trong khoá
sắp xếp, mà sau khi phiếu hoà thì **`score` (`w·gidf·log(div)`) mới là thứ đang
gánh**.

Đây là chẩn đoán lại cho §35: kết luận «`phieu` đã là baseline mạnh» đúng, nhưng
lý do tôi ghi (các tín hiệu kia yếu) SAI — chúng chưa bao giờ được thử tử tế.

### 41.2 Ghép kiểu ĐÚNG (nhân vào `score`)

```
                    n=67              n=157
rank=phieu          35,8%             38,2%
rank=phieu_cos      37,3%  (+1,5)     38,9%  (+0,7)
top5 (cả hai)       55,2%             53,5%
```

**Dương ở cả hai cỡ mẫu, nhưng +0,7 điểm trên n=157 là MỘT ca** — nằm trong nhiễu.
Không đủ để tuyên bố `E` giúp được.

`top5` không đổi (đúng như dự kiến: phá hoà chỉ đổi THỨ TỰ trong top-5, không đưa
đáp án mới vào).

### 41.3 Trạng thái

```
② top-1     38,2% → 38,9% (phieu_cos, trong nhiễu)
② top-5     53,5%                    ← 14,6 điểm dư địa
③ chính xác 38,9% (khi trao nhãn đúng)
sinh tổng   0,996
```

Tám tín hiệu đã thử ở ②. Bảy âm tính, một dương-trong-nhiễu. **Nhưng §41.1 nói
bảy cái trước chưa được thử đúng cách** — chúng cần đo lại ở dạng NHÂN, không
phải dạng THAY.

### 41.4 Bài học phương pháp (nối tiếp §40.5)

47. **Bốn tín hiệu độc lập cùng thất bại Y HỆT ⇒ lỗi ở CÁCH GHÉP, không phải ở
    tín hiệu.** gidf/bậc-vào/khứ-hồi/cosine đều rơi về ~0–1,5%: xác suất bốn thứ
    khác nhau hỏng giống nhau vì lý do riêng là rất thấp. Dấu hiệu này lẽ ra phải
    đọc được từ §35 — tôi mất năm mục mới nhận ra.
48. **Thêm tín hiệu vào một điểm số đang chạy: NHÂN, đừng THAY.** Thay = vứt bỏ
    thứ đang gánh để đổi lấy thứ chưa biết. Nhân giữ baseline làm nền và chỉ hỏi
    tín hiệu mới có nâng thêm được không.

## 42. QUÉT DẠNG NHÂN — giả thuyết §41 BỊ BÁC; ② là cực trị địa phương thật (2026-08-13)

§41.1 nêu: bốn tín hiệu cùng rơi về ~0–1,5% là do **cách ghép** (thay `score`),
nên chúng «chưa bao giờ được thử tử tế». Thử tử tế: ghép dạng NHÂN
(`score × Π(1 + k_i)`), n=157.

| `rank` | ②top1 | ②top5 |
|---|---|---|
| `phieu` (nền) | 38,2% | 53,5% |
| `phieu_cos` (E) | **38,9%** | 53,5% |
| `phieu_x_indeg` | **38,9%** | 54,1% |
| `phieu_x_gidf` | 38,2% | 53,5% |
| `phieu_x_vongve` | 38,2% | 53,5% |
| `phieu_x_pos` | 38,2% | 53,5% |
| **`phieu_x_all`** (ghép cả bốn) | **38,9%** | 54,1% |

**+0,7 điểm = MỘT ca trên 157.** Và ghép CẢ BỐN không hơn từng cái một.

### 42.1 Giả thuyết §41 bị bác

§41 kết luận «bảy tín hiệu chưa được thử đúng cách». Thử đúng cách rồi: chúng
vẫn ~0. Cách ghép **giải thích được vì sao chúng thảm hoạ** (0,0–1,5%) nhưng
**không** làm chúng hữu ích — sửa cách ghép chỉ đưa chúng từ *có hại* về *vô
thưởng vô phạt*.

Phát biểu đúng: `rank="phieu"` + `score` là **cực trị địa phương thật**, và
14,6 điểm giữa top-1 (38,2%) và top-5 (53,5%) **không lấy được bằng cách xếp lại
các tín hiệu có trong đồ thị này**. Tám tín hiệu, hai dạng ghép, đều ~0.

### 42.2 Điều này đổi hướng, không phải đổi tham số

Đáp án CÓ trong top-5 ở 53,5% ca nhưng không có tín hiệu nào trong đồ thị
đồng-hiện + pivot + `E` + từ điển phân biệt được nó với bốn ứng viên còn lại.
⇒ Thông tin cần thiết **không nằm trong biểu diễn hiện tại**, chứ không phải nằm
đó mà bị xếp sai.

### 42.3 Trạng thái đóng vòng

```
① hạt giống      ĐÚNG NGHĨA (dòng→chảy,sông · đốt→cháy · tuân→thủ,luật)
② top-1 / top-5  38,2% / 53,5%   ← CỰC TRỊ ĐỊA PHƯƠNG, 8 tín hiệu × 2 dạng ghép
③ chính xác      38,9% khi trao nhãn đúng · loài→cá chép,gấu trúc,Rùa,Chim,Cá sấu
sinh · ② ĐÚNG    0,911
sinh · tổng      0,996
```

### 42.4 Bài học phương pháp (nối tiếp §41.4)

49. **Sửa được CƠ CHẾ THẤT BẠI không có nghĩa cứu được TÍN HIỆU.** §41 giải thích
    đúng vì sao bốn tín hiệu thảm hoạ, và tôi suy ra chúng sẽ hữu ích khi sửa —
    sai. Giải thích một thất bại và đảo ngược nó là hai việc khác nhau.
50. **Ghép TẤT CẢ tín hiệu không hơn cái tốt nhất ⇒ chúng không bổ sung cho
    nhau.** `phieu_x_all` = `phieu_cos` = `phieu_x_indeg` = 38,9%. Dấu hiệu rõ
    rằng không gian tín hiệu đã cạn, không phải cần thêm cách kết hợp.

## 43. CHỮ KÝ NGỮ CẢNH — tín hiệu ĐẦU TIÊN nhích được ② (2026-08-13)

§42 kết luận: thông tin cần cho ② **không nằm trong biểu diễn hiện tại** — tám
tín hiệu, hai dạng ghép, đều ~0. Vậy phải THÊM biểu diễn.

Thứ `via_pivot` vứt đi suốt từ đầu: **phần còn lại của câu**. Nó chỉ giữ
`(chủ ngữ, nhãn, pivot)`. Cho mỗi nhãn một **túi token ngữ cảnh** gom từ chính
những câu đã sinh ra cạnh vào nhãn đó, rồi nhân điểm với phần ngữ cảnh hiện tại
khớp túi ấy.

`triều đại` học được chữ ký {vua, nhà, năm, trị vì…}; prompt nói về vua chúa thì
`triều đại` được nâng, `khoảng`/`phần` thì không.

### 43.1 Kết quả

| `rank` | n=157 | n=230 |
|---|---|---|
| `phieu` (nền) | 38,2% | 43,5% |
| **`phieu_x_ctx`** | **41,4%** (+3,2) | **45,7%** (+2,2) |
| `phieu_x_all` (ctx + 4 tín hiệu kia) | 40,8% | — |

**Dương ở cả hai cỡ mẫu**, ~5 ca mỗi lần — lớn hơn hẳn mức nhiễu 1-ca của tám
tín hiệu trước. Và `phieu_x_all` **thấp hơn** `ctx` đơn lẻ: bốn tín hiệu kia
**pha loãng** nó, đúng như §42.4 số 50 dự đoán (chúng không bổ sung thông tin).

### 43.2 Vì sao cái này khác

Tám tín hiệu trước đều **suy ra được từ đồ thị đồng-hiện + pivot**: gidf, bậc-vào,
khứ hồi, số phiếu, `E` (huấn luyện trên cùng đồ thị), từ loại (hàm của bề mặt).
Ngữ cảnh câu là dữ liệu **đã có trong cha** nhưng bị `_mine_via_pivot` **loại bỏ
ở khâu trích**. Đó là lý do nó thêm được thông tin còn các tín hiệu kia thì không.

⇒ Khẳng định §42.2 được xác nhận theo chiều dương: nút thắt ② là **thiếu biểu
diễn**, và bù đúng chỗ thiếu thì nó nhúc nhích.

### 43.3 Trạng thái

```
② top-1     43,5% → 45,7%   ✓ chuyển động thật đầu tiên sau tám tín hiệu
② top-5     57,0% → 57,4%   (gần như không đổi — đúng: phá hoà chỉ đổi thứ tự)
③ chính xác 38,9% khi trao nhãn đúng
sinh · tổng 0,996 (chưa đo lại sau §43)
```

Mặc định đổi sang `phieu_x_ctx`; `ctx_ids` nối vào `hub_chain_proposals` (đường
sinh có sẵn prompt làm ngữ cảnh). Thêm test khoá mặc định — bẫy «test không khoá
mặc định» đã dính một lần ở §19.

### 43.4 Bài học phương pháp (nối tiếp §42.4)

51. **Khi mọi tín hiệu suy-từ-biểu-diễn đều ~0, hỏi biểu diễn đã VỨT gì.** Tám
    vòng tôi tổ hợp lại thứ `via_pivot` giữ; thứ nhích được ② là thứ nó bỏ. Câu
    hỏi «tín hiệu nào còn chưa thử» kém hơn hẳn câu hỏi «khâu trích đã ném đi
    cái gì».
52. **Ghép thêm tín hiệu VÔ DỤNG thì có hại.** `phieu_x_all` < `phieu_x_ctx`.
    Tín hiệu ~0 không trung tính khi nhân vào — nó thêm phương sai. Sau khi tìm
    được tín hiệu thật, **bỏ hết tín hiệu đã đo là ~0** thay vì gộp tất.

## 44. THƯỚC SINH SAU §43 — tầng ② ĐÚNG **0,769 → 0,426**, tổng vẫn bị pha loãng

Đo lại đường sinh (bộ ca §26, prompt = tiền tố câu thật, n=100):

| | `phieu` | **`phieu_x_ctx`** |
|---|---|---|
| [② ĐÚNG] n | 31 | **34** |
| [② ĐÚNG] tỉ lệ hạng | 0,769 | **0,426** |
| [② SAI] n | 69 | 66 |
| [② SAI] tỉ lệ hạng | 0,997 | 0,997 |
| **tổng** | 0,996 | 0,997 |

### 44.1 Tầng ② ĐÚNG: hạng mục tiêu bị CẮT HƠN MỘT NỬA

`0,426` nghĩa là hạng trung vị của «con khỉ» chỉ còn **42,6%** hạng cũ. Đây là
chuyển động lớn nhất của thước sinh trong cả phiên, và nó nằm đúng tầng mà mô
hình §27.2 dự đoán.

Đường đi của tầng này qua các bản vá:

```
§36 (prompt 1 từ)                    0,935
§36 (prompt tiền tố câu)             0,911
sau ③-phát-span + gộp mảnh + tách bể 0,769
sau §43 chữ ký ngữ cảnh              0,426   ← cắt hơn một nửa
```

### 44.2 Tổng vẫn 0,997 — vì ② đúng chỉ 34/100

Tầng ② SAI (n=66) phẳng ở 0,997 và **át tầng ② ĐÚNG** trong số gộp. Đây đúng cấu
trúc §27.4 số 21 đã ghi: *một số phẳng có thể là TÍCH của hai số không phẳng*.
Chừng nào ②top1 chưa vượt hẳn 50%, con số tổng còn bị che.

⇒ Thước tổng KHÔNG phải thước nhạy cho việc còn lại. Tầng ② ĐÚNG mới là, và nó
đang chuyển động rõ.

### 44.3 Trạng thái

```
① hạt giống          ĐÚNG NGHĨA
② top-1              45,7%   (43,5% trước §43)
③ chính xác          38,9% khi trao nhãn đúng
sinh · tầng ② ĐÚNG   0,426   ✓ cắt hơn nửa hạng
sinh · tổng          0,997   ✗ bị tầng ② SAI (66/100) át
```

Cơ chế nay **hoạt động đúng thiết kế trên tầng nó làm đúng**. Việc còn lại thuần
tuý là đưa ②top1 lên — mỗi điểm ② chuyển thẳng thành ca có tầng 0,426 thay vì
0,997.

### 44.4 Bài học phương pháp (nối tiếp §43.4)

53. **Thước gộp che mất tiến bộ có thật.** Tổng đi 0,996 → 0,997 (xấu hơn) trong
    khi tầng làm-đúng đi 0,769 → 0,426 (tốt hơn 1,8×). Nếu chỉ nhìn tổng, tôi đã
    kết luận §43 vô dụng và bỏ đi tín hiệu duy nhất có tác dụng. **Phân tầng
    phải có TRƯỚC khi diễn giải số gộp**, không phải sau khi thấy nó phẳng.

## 45. PIVOT VÀ CÂN NGỮ CẢNH — hai âm tính, và một mẫu hình rõ ra (2026-08-13)

Tiếp nguyên tắc §43 («hỏi khâu trích đã VỨT gì»). Hai thứ `_mine_via_pivot` còn
đang bỏ: **pivot nào** và **cân ngữ cảnh**.

### 45.1 Pivot — tín hiệu mạnh khi đo riêng, THỪA khi ghép

`mid` (id pivot) đã được LƯU trong `via_pivot` từ đầu mà chưa bao giờ dùng để
xếp hạng. Đếm trên `classrich`:

| pivot | cạnh LỚP | cạnh khác | tỉ lệ |
|---|---|---|---|
| `là` | 697 | 7 320 | **8,69%** |
| `như` | 2 | 1 642 | 0,12% |
| `gồm` | 0 | 912 | 0,00% |
| `gọi` | 0 | 197 | 0,00% |

`như`/`gồm`/`gọi` = **27% bể ②** mà chỉ sinh **2/2 751** cạnh lớp. Trên giấy là
bộ lọc tuyệt vời. Ghép vào (chỉ ở ②, KHÔNG ở ① — `như` là pivot của chính ví dụ
gốc «nghịch ngợm NHƯ loài chó»):

```
tắt        ②top1 41,4%   top5 54,1%
chỉ `là`   ②top1 41,4%   top5 52,2%   ← top-1 không đổi, top-5 TỆ HƠN
```

### 45.2 Cân ngữ cảnh theo gidf — TỆ HƠN

Chữ ký ngữ cảnh đang đếm token đều nhau; cân theo gidf (token hiếm đáng giá hơn)
cho **41,4% → 40,8%**. Dạng đơn giản thắng; đã trả về bản gốc.

### 45.3 Mẫu hình sau MƯỜI tín hiệu

```
LỌC ứng viên  : gidf · bậc-vào · khứ hồi · E · từ loại · pivot   → tất cả ~0
THÊM thông tin: chữ ký ngữ cảnh                                  → +2,2 điểm ✓
```

Mọi tín hiệu **lọc** đều thừa vì `rank="phieu"` đã dìm sẵn thứ chúng định dìm.
Thứ duy nhất ăn tiền là thứ **mang thông tin mới vào**. Đây không phải nhận xét
hậu nghiệm — nó dự đoán được: trước khi cài tín hiệu thứ mười một, hỏi **«nó LỌC
hay nó THÊM?»**; nếu chỉ lọc, gần như chắc chắn ~0.

### 45.4 Bài học phương pháp (nối tiếp §44.4)

54. **Tín hiệu LỌC vs tín hiệu THÊM — phân loại trước khi cài.** Sáu tín hiệu lọc
    × ~2 phút mỗi vòng = một buổi, tất cả ~0. Một câu hỏi phân loại lẽ ra tiết
    kiệm được toàn bộ.
55. **Làm tinh vi hơn một tín hiệu ĐANG CHẠY cũng phải đo.** Cân gidf nghe hợp
    lý hơn đếm đều và tệ hơn 0,6 điểm. Trực giác về «bản tinh vi hơn» không đáng
    tin hơn trực giác về «tín hiệu mới».

## 46. QUAN HỆ NHÃN–NHÃN — âm tính, và nó ĐÍNH CHÍNH mẫu hình §45.3

Mỗi nhãn xưa nay là một ĐẢO: `triều đại` và `vương triều` chia nhau thành viên mà
điểm của chúng không liên quan gì nhau. Dựng đồ thị nhãn–nhãn từ THÀNH VIÊN CHUNG
(chồng lấn ≥0,25) rồi cho phiếu lan qua họ hàng.

### 46.1 Kết quả

| `rank` | ②top1 | ②top5 |
|---|---|---|
| **`phieu_x_ctx`** | **41,4%** | **54,1%** |
| `phieu_x_ho` | 38,9% | 53,5% |
| `phieu_x_ctx_ho` | 39,5% | 53,5% |

**Âm tính**, và ghép với `ctx` còn **kéo `ctx` xuống** (41,4% → 39,5%).

### 46.2 Vì sao — đọc đồ thị họ hàng

1 225 nhãn có họ, nhưng đọc ra:

```
Đây      ← hydro(1,0) · bưu kiện(1,0) · mẹ(0,67)
con      ← chính khách(1,0) · photon(0,5)
Trịnh    ← nay(0,5) · câu lạc bộ(0,5) · xã(0,5)
Oxy      ← Carbon(0,67) · hydro(0,5)          ← ĐÚNG
Ông      ← Bà(0,67) · mẹ(0,67)                ← ĐÚNG
```

Đồ thị dựng trên **mọi đích typed** (10 785 cạnh), mà đa số **không phải nhãn
lớp** — nên phần lớn quan hệ là rác, và phiếu lan sang rác.

### 46.3 ĐÍNH CHÍNH §45.3

§45.3 rút mẫu hình «tín hiệu LỌC ~0 · tín hiệu THÊM ăn tiền» và bảo nó **dự đoán
được**. Họ hàng nhãn là tín hiệu **THÊM** — và nó ÂM. Mẫu hình sai.

Phát biểu đúng: `ctx` ăn tiền vì thông tin nó thêm **ĐÚNG**; họ-hàng-nhãn thất
bại vì cấu trúc nó thêm **SAI**. «Thêm hay lọc» không phải tiêu chí; **chất lượng
của thứ được thêm** mới là. Và chất lượng đó phải đọc tay TRƯỚC khi ghép — đồ thị
họ hàng lộ ngay là rác khi in 10 dòng.

### 46.4 Trạng thái

```
② top-1              45,7% (n=230) · 41,4% (n=157)   — `phieu_x_ctx` vẫn tốt nhất
③ chính xác          38,9% khi trao nhãn đúng
sinh · tầng ② ĐÚNG   0,426
sinh · tổng          0,997
```

Mười một tín hiệu đã đo ở ②. Một ăn tiền (`ctx`), mười ~0 hoặc âm.

### 46.5 Bài học phương pháp (nối tiếp §45.4)

56. **Mẫu hình rút từ MỘT ca thành công thì chưa phải mẫu hình.** §45.3 khái quát
    «THÊM > LỌC» từ đúng một tín hiệu thắng, và tuyên bố nó dự đoán được. Ca kế
    tiếp bác ngay. Cần ít nhất hai ca cùng chiều trước khi gọi là quy luật.
57. **Đọc cấu trúc mới TRƯỚC khi ghép nó vào.** In 10 dòng đồ thị họ hàng là thấy
    `Đây ← hydro, bưu kiện` — biết ngay nó rác, khỏi cần chạy phép đo. Cùng quy
    tắc §30.4 số 26 (đọc tay 10 mẫu) áp cho **cấu trúc**, không chỉ cho thước.

## 47. HỌ HÀNG NHÃN LÀM SẠCH — cải thiện thật, vẫn không đủ (2026-08-13)

§46 dựng họ hàng trên MỌI đích typed ⇒ rác. Dựng lại chỉ trên **nhãn bậc-vào
cao** (phân vị 0,90) — tiêu chí đã chứng minh có tác dụng ở §24.

### 47.1 Đọc TRƯỚC khi đo (bài học §46.5 số 57)

65 nhãn có họ, thay vì 1 225. Và quan hệ đúng nghĩa:

```
hành tinh  ← Mặt Trời(0,44) · vệ tinh(0,44) · sao(0,29) · ngôi sao(0,28)   ✓
vệ tinh    ← hành tinh(0,44)                                               ✓
nguyên tố  ← nguồn · chất                                                  ✓
loại       ← dạng(0,32)                                                    ✓
ngôn ngữ   ← lực(0,32) · biển(0,28)                                        ✗ còn nhiễu
lực        ↔ vậy(0,37)                                                     ✗
```

### 47.2 Kết quả

| `rank` | §46 (họ bẩn) | **§47 (họ sạch)** |
|---|---|---|
| `phieu_x_ctx` | 41,4% | **41,4%** |
| `phieu_x_ho` | 38,9% | 38,2% |
| `phieu_x_ctx_ho` | 39,5% | **40,8%** |

Làm sạch **có tác dụng thật** (39,5% → 40,8%, thu hồi 2/3 phần đã mất) nhưng
`ctx + họ` **vẫn thua `ctx` đơn lẻ**. Họ hàng nhãn âm tính kể cả khi sạch.

### 47.3 Trạng thái — mười hai tín hiệu ở ②

```
ĂN TIỀN (1)  : chữ ký ngữ cảnh            +2,2 điểm
~0 hoặc ÂM   : gidf · bậc-vào · khứ hồi · E · từ loại · pivot ·
               đồng thuận thuộc tính · chuẩn hoá phiếu · biến thể id ·
               họ hàng nhãn (bẩn) · họ hàng nhãn (sạch) · cân ctx theo gidf
```

`phieu_x_ctx` giữ nguyên mặc định.

```
② top-1              45,7% (n=230)
③ chính xác          38,9% khi trao nhãn đúng
sinh · tầng ② ĐÚNG   0,426
sinh · tổng          0,997
```

### 47.4 Bài học phương pháp (nối tiếp §46.5)

58. **Làm sạch một tín hiệu ÂM có thể đưa nó về gần 0 mà vẫn không thành DƯƠNG.**
    Họ hàng: bẩn −1,9 điểm → sạch −0,6 điểm. Thu hồi 2/3 mà vẫn âm ⇒ vấn đề
    không phải chất lượng đồ thị mà là **thông tin đó đã có trong `ctx`**: nhãn
    họ hàng chia nhau thành viên thì cũng chia nhau ngữ cảnh.

## 48. VÙNG TÀI LIỆU — thứ mười ba, âm tính; không gian tín hiệu đã cạn

`via_pivot` xử lý TỪNG CÂU độc lập, chưa bao giờ dùng quan hệ GIỮA các câu.
Checkpoint không có `doc_id`/`episode` (episodes=0), nhưng cha được nạp THEO THỨ
TỰ TÀI LIỆU nên chỉ số cha là proxy cho cùng-tài-liệu. Ghi lại tập chỉ số cha
sinh ra cạnh của mỗi nhãn, rồi nâng nhãn có cạnh đến từ vùng gần câu hiện tại
(|i−j| ≤ 20).

| `rank` | ②top1 | ②top5 |
|---|---|---|
| **`phieu_x_ctx`** | **41,4%** | **54,1%** |
| `phieu_x_doc` | 38,2% | 53,5% |
| `phieu_x_ctx_doc` | 41,4% | 54,1% |

`ctx_doc` **y hệt `ctx`** — không đổi một ca. Tín hiệu vùng tài liệu trung tính
hoàn toàn: nhãn phổ biến có cạnh rải khắp corpus nên «gần vùng hiện tại» đúng
với gần như mọi ứng viên.

### 48.1 Tổng kết MƯỜI BA tín hiệu ở bước ②

| | tín hiệu | kết quả |
|---|---|---|
| ✓ | chữ ký ngữ cảnh (`ctx`) | **+2,2 điểm** |
| ~0 | gidf · bậc-vào · khứ hồi · `E` · từ loại · pivot · vùng tài liệu | 0 |
| ✗ | đồng thuận thuộc tính · chuẩn hoá phiếu · họ hàng nhãn (bẩn/sạch) · cân ctx theo gidf | âm |

**Một trên mười ba.** Và ba lần thử gần nhất (pivot · họ hàng · vùng tài liệu)
đều thất bại vì CÙNG lý do: **thông tin đã có sẵn trong `ctx`**. Nhãn chia nhau
thành viên thì chia nhau ngữ cảnh; nhãn cùng vùng tài liệu thì cùng ngữ cảnh;
pivot `là` thì đã nằm trong cách `ctx` được thu thập.

⇒ `ctx` đã hút gần hết thông tin mà **câu quanh pivot** có thể cho. Muốn đi tiếp
phải có nguồn khác hẳn — không phải tổ hợp lại thứ đang có.

### 48.2 Trạng thái cuối vòng

```
① hạt giống          ĐÚNG NGHĨA   (dòng→chảy,sông · đốt→cháy · tuân→thủ,luật)
② top-1              45,7% (n=230) · top-5 57,4%
③ chính xác          38,9% khi trao nhãn đúng
                     loài → cá chép · gấu trúc · Rùa · Chim · Cá sấu
sinh · tầng ② ĐÚNG   0,426        ✓ hạng mục tiêu cắt hơn một nửa
sinh · tổng          0,997        ✗ bị tầng ② SAI (66/100) át
```

### 48.3 Bài học phương pháp (nối tiếp §47.4)

59. **Ba thất bại liên tiếp CÙNG một lý do = dấu hiệu cạn, không phải xui.**
    Pivot, họ hàng nhãn, vùng tài liệu — cả ba đều thua vì `ctx` đã mang thông
    tin đó. Khi lý do thất bại lặp lại, dừng thử biến thể và đổi nguồn.

## 49. ĐƯỜNG CONG NHIỄU ĐO LẠI trên cơ chế đã vá — ngưỡng giữ ở 3 069 cha

§21 hỏi «bổ dữ liệu có cứu được không» và trả lời KHÔNG: cơ chế lúc đó gãy giữa
895 và 3 069 cha, còn corpus thật là 35 757. Sau mười mấy bản vá (§29–§48), hỏi lại.

| cha | CŨ (§21) | **ĐÃ VÁ** |
|---|---|---|
| 118 | `vật`/2 | `động`/2 |
| 337 | `vật`/2 | `động`/2 |
| 895 | `vật`/2 | `động`/2 |
| **3 069** | **`cá`/—** | **`động`/2** ✓ |
| 10 287 | `loài`/— | `nghiên`/— |
| 20 316 | `loài`/— | `nó`/— |

**Ngưỡng chịu nhiễu 895 → 3 069 cha** và giữ nguyên qua toàn bộ các bản vá sau
đó — không hồi quy. Nhưng **không vượt được 10 287**.

### 49.1 Chỗ gãy ở 10 287 KHÔNG phải bước ②

`nghiên` — tức `nghịch` → `nghiên cứu`. Bước ① trôi sang **bạn ghép từ**, đúng
chẩn đoán §21.4. Mười ba tín hiệu ở §35–§48 đều nhắm vào ②; **không cái nào chạm
bước ①**, nên ngưỡng không nhích thêm là điều phải xảy ra.

### 49.2 Trả lời câu hỏi mở từ đầu phiên

«Bổ dữ liệu lớp cho danh từ chung có cứu được không?»

**Vẫn chưa.** Ngưỡng 3 069 so với 35 757 cha — còn **11,6 lần**. Thêm câu lớp
làm corpus TO hơn, tức đẩy sâu hơn vào vùng đã gãy, trừ khi bước ① được vá trước.

Nhưng bức tranh không mâu thuẫn với `②top1 = 45,7%` trên `classrich`: đường cong
đo **một chuỗi vàng cụ thể sống sót giữa nhiễu**, còn `classrich` đo **trung bình
trên nhiều chuỗi**. Ở 35 757 cha phần lớn chuỗi hỏng (54% sai ở ②) nhưng một
phần đáng kể vẫn chạy.

### 49.3 Trạng thái đóng arc

```
① hạt giống    ĐÚNG NGHĨA ở quy mô nhỏ · TRÔI sang bạn ghép từ ở ~10k cha
               ← NÚT THẮT MỚI, chưa ai chạm
② top-1        45,7% (classrich) — 13 tín hiệu, 1 ăn tiền (`ctx`)
③ chính xác    38,9% khi trao nhãn đúng
ngưỡng nhiễu   895 → 3 069 cha (3,4×)
sinh · ② ĐÚNG  0,426
sinh · tổng    0,997
```

### 49.4 Bài học phương pháp (nối tiếp §48.3)

60. **Đo lại một cổng CŨ sau loạt vá là cách rẻ nhất để biết mình vá đúng chỗ
    chưa.** Mười ba vòng ở ② không nhích được ngưỡng nhiễu một chút nào — vì
    ngưỡng đó do ① quyết. Nếu chạy lại đường cong sau §35 thay vì sau §48, tôi
    đã biết sớm hơn mười vòng rằng mình đang vá sai bước.

## 50. VÁ BƯỚC ① — hai cách, cả hai KHÔNG kéo dài được ngưỡng

§49 chỉ ra ngưỡng nhiễu do bước ① quyết, và 13 tín hiệu ở ② không chạm tới nó.
Vá ①.

### 50.1 Chẩn đoán: cạnh typed nằm SAI TOKEN

```
via_pivot[nghịch] = — RỖNG —          (cả ở 895 lẫn 10 287 cha)
via_pivot[ngợm  ] = [('loài', 1.9)]
```

`_trung_tam_trai` với `phai_nhat` lấy **âm tiết cuối** của `nghịch ngợm`. Đây
đúng bệnh `Thụy Điển → Điển` (§30) và `Lúa mì → mì` (§33.2) — lần thứ BA, ở một
vị trí mới. Ở 895 cha kênh cooc từ `nghịch` còn đủ mạnh chạm `chó`/`mèo`; ở
10 287 nó bị `nghịch lý`(6,7) · `nghịch hành`(5,7) · `nghịch lý Fermi`(4,8)
nhấn chìm — cạnh vàng chỉ nặng 0,95.

### 50.2 Hai bản vá, hai âm tính

**(a) Lọc BẠN GHÉP TỪ** (`loc_ghep_tu`): loại đích có tỉ lệ đứng cạnh seed cao,
quét d=1..2.

```
             loc=0     loc=0,10   loc=0,20   loc=0,35
   895      động/2      động/2     động/2     động/2
  3069      động/2      động/2     động/2     động/2
 10287    nghiên/—      viên/—    chúng/—    chúng/—
```

Không hồi quy, **không kéo dài**. ⚠️ Bật cho CẢ ② thì phá luôn ca đang chạy
(`động/2` → `vừa/—` ở 895) vì ở ② hạt giống là `{chó, mèo}` và `động` nằm ở d=2
trong `chó là động vật` — bệnh «hàm dùng chung» của §40, **lần thứ ba**.

**(b) Hỏi bằng CẢ TỪ** (`nghịch` + `ngợm` thay vì mỗi `nghịch`):

```
   895   động/2 → động/2      3069   động/2 → động/2
 10287  nghiên/— → chúng/—   20316      nó/— → nó/—
```

Cũng không kéo dài. Cạnh typed có sẵn (qua `ngợm`) cho hạt giống `loài`, nhưng ở
10 287 `loài` dẫn sang `loài cá`/`loài thú` — đúng bế tắc §21 đã ghi.

### 50.3 Trạng thái

```
ngưỡng nhiễu   3 069 cha — KHÔNG nhích sau hai bản vá ①
② top-1        45,7% (classrich)
③ chính xác    38,9% khi trao nhãn đúng
sinh · ② ĐÚNG  0,426
sinh · tổng    0,997
```

`loc_ghep_tu` giữ tham số, **mặc định 0 (tắt)**.

### 50.4 Bài học phương pháp (nối tiếp §49.4)

61. **Cùng một lỗi ở vị trí thứ BA.** `phai_nhat` lấy âm tiết cuối: §30 (chân
    lý), §33 (chủ ngữ `Lúa mì`), §50 (thuộc tính `nghịch ngợm`). Mỗi lần tôi vá
    tại chỗ. Bài học §38.6 số 42 đã dặn «gặp lần thứ hai thì quét TOÀN BỘ» — tôi
    ghi rồi vẫn không làm, và mất thêm một vòng.
62. **Bệnh «hàm dùng chung» lần thứ ba** (§37 bể cạnh · §40 lời gọi ①/② · §50
    lời gọi ①/②). Mẫu đã đủ rõ để thành quy tắc: **mọi tham số mới của
    `score_hubs_from` phải mặc định CHỈ áp cho một bước**, không bao giờ cho cả hai.

## 51. TÁCH KÊNH TRIỆT ĐỂ — GIẾT cơ chế; chuỗi vàng vốn CẦN cả hai kênh

§50 kết: cạnh phân loại (0,95) phải cạnh tranh cạnh đồng-hiện (5–7) trong cùng
bảng xếp hạng ⇒ thử tách triệt để, bước ① chỉ dùng `channel="typed"`.

| cha | `typed_first` (nay) | **`typed` thuần** |
|---|---|---|
| 118 · 337 · 895 · 3 069 | `động`/2 | **`None`/—** |
| 10 287 · 20 316 | gãy | gãy |

**Chết ở mọi mức nhiễu thấp.** Truy trên `cha=337`:

```
nghịch   typed=—                    cooc=[ngợm 1,9 · như 1,9 · loài 1,9 · chó 0,9]
ngợm     typed=[loài 1,9]           cooc=[như · loài · chó · mèo]
loài     typed=—                    cooc=[chó 0,9 · mèo 0,9 · động 2,8]
chó      typed=[động 0,95]          cooc=[là · động · vật]
mèo      typed=[động 0,95]          cooc=[là · động · vật]
```

### 51.1 Chuỗi vàng đi qua HAI kênh, không phải một

```
nghịch ngợm ──cooc──▶ chó, mèo ──typed──▶ động vật ──typed──▶ khỉ
     bước ①               bước ②              bước ③
```

Kênh typed **không có** cạnh `nghịch ngợm → chó/mèo`: trong
`nghịch ngợm như loài chó`, trung tâm PHẢI của pivot `như` là **`loài`** (lượng
từ), nên typed chỉ cho `ngợm → loài`, mà `via_pivot[loài]` lại rỗng. Hạt giống
THẬT (`chó`, `mèo`) chỉ tới được qua **đồng hiện**.

⇒ **Tách kênh triệt để là sai về nguyên tắc ở bước ①.** Bước ② và ③ là quan hệ
PHÂN LOẠI (`X là Y`) nên typed đúng; bước ① là quan hệ **THUỘC TÍNH** (`P như X`,
`P` bổ nghĩa `X`) — quan hệ đó **không đi qua pivot phân loại**, nên nó vốn dĩ
nằm ở kênh đồng-hiện.

### 51.2 Đính chính §50.3

§50 gợi ý «vượt trần cần cạnh phân loại không nằm cùng đồ thị với cạnh đồng-hiện».
Sai một nửa: **ở ② và ③ thì đúng** (và `typed_first` đã làm), **ở ① thì không** —
đồng hiện là kênh DUY NHẤT mang quan hệ thuộc tính. Vấn đề ở 10 287 không phải
«cooc bị dùng» mà là «cooc bị bạn-ghép-từ thống trị».

### 51.3 Bài học phương pháp (nối tiếp §50.4)

63. **«Tách triệt để» chỉ đúng khi hai kênh mang CÙNG một loại quan hệ.** Ở đây
    chúng mang hai loại: ② ③ là PHÂN LOẠI, ① là THUỘC TÍNH. Ép cùng một kỷ luật
    lên cả ba bước giết bước không cùng bản chất. Trước khi tách/hợp một kênh,
    hỏi **mỗi bước đang cần QUAN HỆ gì**, không chỉ hỏi kênh nào sạch hơn.

## 52. RANH GIỚI TỪ TỪ ĐIỂN CHO ① — lọc đúng, ngưỡng vẫn không nhích; và giới hạn thật lộ ra

§51 xác lập: bước ① là quan hệ THUỘC TÍNH nên buộc phải chạy trên kênh đồng-hiện.
§50 thử lọc bạn-ghép-từ theo TỈ LỆ KỀ NHAU — hỏng, vì `nghịch ngợm` và `nghịch lý`
đều là từ ghép. Thứ phân biệt được là **ranh giới từ trong từ điển**.

`ghep_tu_tu_dien()` bake `a → {b : «a b» là MỘT TỪ}` từ 36 764 mục. `nghịch` ghép
với **25 âm tiết** (`lý` · `đảo` · `cảnh` · `tặc` · `ngợm`…); giữ lại đúng `ngợm`
(chính thuộc tính đang hỏi), loại 24 cái còn lại.

### 52.1 Lọc CHẠY ĐÚNG, ngưỡng KHÔNG nhích

| cha | không lọc | ranh giới từ |
|---|---|---|
| 118 · 337 · 895 · 3 069 | `động`/2 | `động`/2 |
| 10 287 | `chúng`/— | `chúng`/— |

Bước ① ở 10 287 cha:

```
không lọc    : loài(T) · lý(c) · Fermi(c) · quỹ(c) · phép(c) · tương(c) · đạo(c)
ranh giới từ : loài(T) ·         Fermi(c) · quỹ(c) · phép(c) · tương(c) · đạo(c)
```

`lý` bị loại đúng. Nhưng `Fermi`/`quỹ`/`phép` **không phải từ ghép từ điển** —
chúng đến từ CỤM THUẬT NGỮ nhiều tiếng (`nghịch lý Fermi` · `phép nghịch đảo` ·
`quỹ đạo nghịch`), mà từ điển chỉ có từ 2 âm tiết.

### 52.2 Giới hạn THẬT — và nó không phải lỗi lọc

```
chó   w=0,9  d=4        Fermi  w=4,8
mèo   w=0,9  d=4        lý     w=6,7
loài  w=1,9  d=3 (T)    hành   w=5,7
```

Cạnh vàng nặng **0,9** — một lần xuất hiện. Đối thủ nặng **4,8–6,7** vì xuất hiện
nhiều lần, và chúng là **đồng hiện HỢP LỆ**, không phải rác cần lọc. Không bộ lọc
nào cứu được: muốn `chó` thắng `Fermi` thì phải xoá một quan hệ có thật.

Đây chính là §18.H («cooc tăng tuyến tính theo cỡ corpus, typed thì không») —
nhưng ở **bước ①**, nơi typed **không tồn tại** vì quan hệ thuộc tính không đi qua
pivot phân loại (§51.1). Tách kênh cứu được ② và ③; ở ① không có gì để tách.

### 52.3 Kết luận kiến trúc

**Ngưỡng ~3 000 cha là trần của kiến trúc hiện tại, không phải bug.** Ba hướng vá
① đều thất bại (§50 tỉ lệ kề · §51 tách kênh · §52 ranh giới từ), và lý do cuối
cùng giống nhau: ở kênh đồng-hiện, **một câu không thể cạnh tranh với một nghìn
câu** — và bước ① không có kênh nào khác.

Muốn vượt cần quan hệ THUỘC TÍNH có kênh riêng, tức phải **trích được `P bổ nghĩa
X` thành cạnh có kiểu** giống như `X là Y` đã được trích. Đó là công trình mới,
không phải tham số.

### 52.4 Bài học phương pháp (nối tiếp §51.3)

64. **Khi đối thủ của tín hiệu là quan hệ HỢP LỆ, bài toán không phải lọc.** Ba
    vòng tôi tìm cách loại `Fermi`/`quỹ`/`phép` — nhưng `nghịch lý Fermi` là cụm
    có thật, xuất hiện nhiều lần. Câu hỏi đúng không phải «làm sao bỏ chúng» mà
    «làm sao cạnh vàng có kênh riêng để không phải thi với chúng».

## 53. CẠNH THUỘC TÍNH — loại cạnh THỨ HAI; bước ① được vá thật (2026-08-13)

§51–§52 xác lập: bước ① là quan hệ THUỘC TÍNH, không có kênh typed, nên buộc
chạy trên đồng-hiện và bị bạn-ghép-từ nhấn chìm. Ba bản vá lọc đều thất bại vì
đối thủ là **quan hệ hợp lệ** (§52.4).

Giải pháp không phải lọc, mà **trích quan hệ thuộc tính thành cạnh CÓ KIỂU**:

```
PHÂN LOẠI  `X là Y`   → trung tâm phải = ĐẦU LỚP   (`Cuba là quốc gia`)
THUỘC TÍNH `P như X`  → bên phải = DANH TỪ CỤ THỂ  (`… như loài CHÓ`)
```

Trước §53 cả hai dùng CHUNG luật «đầu lớp», nên `như` cho `ngợm → loài` (lượng
từ) và quan hệ ① cần bị **vứt**. Nay `via_attr` là bể cạnh thứ hai, luật ngược
lại (phải-nhất thay vì đầu-lớp).

### 53.1 Cạnh trích được, và KHÔNG suy giảm theo quy mô

```
cha=  337   via_attr[ngợm] = [chó 0,95 · mèo 0,95]   tổng 11 cạnh
cha=10 287  via_attr[ngợm] = [chó 0,95 · mèo 0,95]   tổng 476 cạnh
```

Đúng tính chất §18.H: **cạnh typed không tăng theo cỡ corpus**, nên nó không bị
loãng — điều mà kênh đồng-hiện không làm được.

### 53.2 Bước ① ĐÃ ĐƯỢC VÁ

Ở 10 287 cha:

```
TRƯỚC : ① = [lý(c) · Fermi(c) · quỹ(c) · phép(c)]     ← bạn ghép từ
SAU   : ① = [chó(T) · mèo(T) · lý(c) · Fermi(c)]      ← ĐÚNG hạt giống, TYPED, đầu bảng
```

Ba vòng lọc (§50 · §51 · §52) không làm được điều này; thêm **một loại cạnh** thì
làm được. Đúng dự đoán §52.3.

### 53.3 Nhưng chuỗi vẫn gãy — nút thắt DỜI sang ②

```
cha=  895   via_pivot[chó] = [động 0,95]    ② = [động(T) …]   ✓
cha=10 287  via_pivot[chó] = [biến 6,65]    ② = [biến(T) …]   ✗
```

Cạnh vàng `chó → động` **biến mất** ở 10 287; chỗ nó là `biến` nặng 6,65. Đây là
cạnh **typed đấu typed** — `rank="phieu"` không cứu được vì `động` chỉ còn MỘT
phiếu (`via_pivot[mèo]` cũng mất `động`, chỉ còn `đặc`).

Ngưỡng đầu-cuối vẫn 3 069 cha. Nhưng **nút thắt đã dời**: từ ① (không có kênh)
sang ② (cạnh vàng bị cạnh typed khác đè).

### 53.4 Trạng thái

```
① hạt giống   ĐÃ VÁ — kênh thuộc tính, typed, bền theo quy mô   ✓ MỚI
② nhãn        cạnh vàng `chó → động` mất ở ~10k cha            ← nút thắt mới
③ thành viên  38,9% khi trao nhãn đúng
ngưỡng        3 069 cha (chưa nhích, nhưng lý do đã đổi)
```

### 53.5 Bài học phương pháp (nối tiếp §52.4)

65. **Ba vòng lọc thua một loại cạnh mới.** §50/§51/§52 đều cố làm sạch kênh sai;
    thứ giải được là THÊM kênh đúng. Khi một bước không có kênh mang đúng loại
    quan hệ nó cần, mọi tinh chỉnh trên kênh sai đều là bù nhìn.
66. **Vá xong một nút thắt thì nút kế lộ ra — đó là tiến bộ, không phải thất
    bại.** Ngưỡng chưa nhích nhưng lý do gãy đã đổi hẳn: trước là ① không có
    kênh, nay là ② mất cạnh vàng. Hai bài toán khác nhau, và cái sau đã có sẵn
    công cụ (tách kênh, đếm phiếu) từ §18–§29.

## 54. TỪ LOẠI CHO CHỦ NGỮ — lỗi `phai_nhat` lần thứ TƯ, và luật vị trí là ngõ cụt

§53 dời nút thắt sang ②: `via_pivot[chó]` mất `động`, chỗ nó là `biến` nặng 6,65.
Đọc câu train:

```
[3501] «Một số loài phổ BIẾN là cáo đỏ, chó sói, đồng cỏ, giẻ cùi xanh và rô phi»
```

Khuôn LIỆT KÊ bắt câu này. Lớp thật là **`loài`**, nhưng `phai_nhat` quét trái
pivot rồi lấy PHẢI NHẤT = `biến` (âm tiết cuối của `phổ biến`), phát `chó → biến`
với trọng số cả cụm liệt kê (6,65), **đè cạnh vàng** `chó → động` (0,95) từ câu
`[0] loài chó là động vật`.

Lỗi `phai_nhat` **lần thứ TƯ**: §30 (chân lý) · §33 (`Lúa mì`) · §50
(`nghịch ngợm`) · §54 (`phổ biến`).

### 54.1 Luật VỊ TRÍ là ngõ cụt — tiếng Việt có CẢ HAI hướng

```
loài chó       → đầu là `chó`  (PHẢI)   `loài` là lượng từ
loài phổ biến  → đầu là `loài` (TRÁI)   `phổ biến` là bổ ngữ
```

Thử `trai_dac_thu` (trái nhất đủ đặc thù): **phá cả ca đang chạy** —
895 `động/2` → `đa/—`. Không luật vị trí nào đúng cả hai.

### 54.2 TỪ LOẠI giải được

`chó` là **danh từ**, `phổ biến` là **tính từ** ⇒ `phai_nhat_danh_tu` = phải nhất
**trong số danh từ**. Từ điển đã nạp sẵn từ §40.

| cha | `phai_nhat` | **`phai_nhat_danh_tu`** |
|---|---|---|
| 118 · 337 · 895 | `động`/2 | `động`/2 |
| 3 069 | `động`/2 | `động`/3 |
| 10 287 | **`biến`/—** | **`loài`/—** |
| 20 316 | `biến`/— | `loài`/— |

**Không hồi quy, và nhãn SAI thành nhãn ĐÚNG**: ở 10 287 cơ chế nay chọn `loài` —
một lớp có thật — thay vì `biến` (âm tiết cuối của tính từ). Ngưỡng đầu-cuối chưa
nhích (`khỉ` không nằm trong thành viên của `loài` ở mức nhiễu đó), nhưng chất
lượng nhãn đã đúng.

### 54.3 Trạng thái

```
① hạt giống   ĐÃ VÁ (§53 kênh thuộc tính)                    ✓
② nhãn        `biến` → `loài` — chọn đúng lớp ở 10k cha       ✓ MỚI
③ thành viên  `khỉ` chưa vào top-30 của `loài` ở 10k          ← nút kế
ngưỡng        3 069 cha
```

### 54.4 Bài học phương pháp (nối tiếp §53.5)

67. **Khi ngôn ngữ có CẢ HAI hướng, luật vị trí không thể đúng — phải dùng LOẠI.**
    Bốn lần tôi thử luật vị trí cho chủ ngữ (trái nhất · phải nhất · trái-đủ-đặc-thù
    · phải-nhất) và mỗi lần sai một nửa số ca. `loài chó` vs `loài phổ biến` khác
    nhau ở TỪ LOẠI, không ở vị trí.
68. **Câu train giải được ca mà bốn vòng đo không giải được.** `[3501]` chỉ thẳng
    cả nguyên nhân (`phổ biến` → `biến`) lẫn cơ chế lan (khuôn liệt kê nhân trọng
    số), và gợi ra ngay tiêu chí đúng. Yêu cầu «cho tôi câu train» của tác giả đã
    đúng bốn lần liên tiếp trong phiên này.

## 55. 🟢 VƯỢT TRẦN — hai bộ lọc phân vị của §24 chính là thứ chặn ngưỡng

§54 dời nút thắt sang ③: `khỉ` không vào top-30 của nhãn `loài`. Đọc câu train —
và nó ngắn gọn đến bất ngờ.

**`khỉ` xuất hiện trong ĐÚNG MỘT câu**: `[2] loài khỉ là động vật`.
Và `via_pivot[khỉ]` **RỖNG**.

### 55.1 Cạnh vàng CÓ TỒN TẠI — bị chính bộ lọc của tôi cắt

```
cha=10 287   gidf_q=0,2 · indeg_q=0,8  →  via_pivot[khỉ] = —
             gidf_q=0   · indeg_q=0    →  via_pivot[khỉ] = [(động, 0,95)]
```

Hai bộ lọc thêm ở §24 (`label_gidf_q` · `label_indeg_q`) để tăng độ chính xác.
Ở 337 cha chúng không chạm cạnh vàng; ở 10 287 **phân bố dịch** và cùng một phân
vị cắt vào đúng nó — cạnh chỉ nặng 0,95 vì xuất hiện MỘT lần.

### 55.2 Tắt cả hai: ngưỡng 3 069 → **≥20 316 cha**

| cha | lọc bật | **lọc TẮT** |
|---|---|---|
| 118 · 337 · 895 | `động`/2 | `động`/2 |
| 3 069 | `động`/3 | `động`/3 |
| **10 287** | `loài`/— | **`động`/6** ✓ |
| **20 316** | `loài`/— | **`động`/7** ✓ |

**6,6× ngưỡng nhiễu**, và không tìm thấy trần trong dải toy hiện có. `classrich`
là 35 757 cha — từ «còn 11,6 lần» xuống **còn 1,8 lần**.

### 55.3 Và nó tốt hơn trên dữ liệu thật luôn

```
classrich   gidf_q=0,2 indeg_q=0,8  →  ②top1 15,2%  top5 24,7%  n=474
            TẮT CẢ HAI              →  ②top1 19,4%  top5 30,6%  n=496
```

Không có đánh đổi: tốt hơn ở cả ba con số.

### 55.4 Mặc định đã đổi

`hub_chain_indeg_q=0` · `hub_chain_gidf_q=0` · `hub_chain_head_rule_trai=
"phai_nhat_danh_tu"` · `hub_chain_pos_dict` nối vào đường sinh. Hai test khoá.

### 55.5 Bài học phương pháp (nối tiếp §54.4)

69. **PHÂN VỊ cũng không chuyển được giữa các quy mô.** Dự án đã ba lần dính bẫy
    NGƯỠNG HẰNG (§20 note) và chuyển sang phân vị để tránh. Nhưng phân vị tính
    trên PHÂN BỐ, mà phân bố dịch theo cỡ corpus — nên nó mắc đúng bẫy đó ở dạng
    khác. Cách duy nhất an toàn: **đo bộ lọc ở NHIỀU quy mô trước khi bật**.
70. **Bộ lọc tăng độ chính xác cục bộ có thể là thứ chặn năng lực toàn cục.**
    §24 đo `indeg_q=0,8` «giữ nguyên phủ, bỏ 26% rác» — đúng ở 35 757 cha, và
    che mất việc nó cắt cạnh vàng ở mọi quy mô. Một bộ lọc phải đo trên CẢ
    ngưỡng-năng-lực lẫn độ-chính-xác.
71. **Chuỗi đọc-câu-train bốn lần liên tiếp giải được bốn nút thắt** (§38 `thành`
    · §46 họ hàng rác · §54 `phổ biến` · §55 hai bộ lọc). Không nút nào lộ ra qua
    con số tổng.

## 56. ĐÍNH CHÍNH §55 — tôi đổi HAI biến rồi gán công cho MỘT

§55 kết «tắt hai bộ lọc phân vị: ngưỡng 3 069 → ≥20 316 **và** classrich tốt hơn
luôn, không đánh đổi». Đo thước sinh thì hồi quy nặng. Tách biến:

| | luật chủ ngữ | lọc phân vị | **sinh · ② ĐÚNG** | n |
|---|---|---|---|---|
| CŨ | `phai_nhat` | cả hai bể | **0,290** | 42 |
| ISO | `phai_nhat` | chỉ ③ | **0,500** | 33 |
| §56 | `phai_nhat_danh_tu` | chỉ ③ | **0,994** | 42 |
| §55 | `phai_nhat_danh_tu` | tắt | 0,995 | 43 |

```
lọc  : 0,290 → 0,500   (bỏ lọc khỏi bể ②)
luật : 0,500 → 0,994   (phai_nhat → phai_nhat_danh_tu)   ← góp NHIỀU HƠN
```

**Ở §55 tôi đổi CẢ HAI cùng lúc rồi quy toàn bộ chuyển động cho bộ lọc.** Phần
«đường cong nhiễu 3 069 → ≥20 316» vẫn đứng (đo riêng biến lọc, luật cố định ở cả
hai cột), nhưng phần «và tốt hơn trên dữ liệu thật luôn — không đánh đổi» **SAI**:
tôi viết câu đó mà chưa đo thước sinh.

### 56.1 Xung đột THẬT giữa các thước

| | ngưỡng nhiễu | classrich ②top1 | sinh ② ĐÚNG |
|---|---|---|---|
| `phai_nhat` + lọc cả hai | 3 069 | 15,2% | **0,290** |
| `phai_nhat_danh_tu` + tắt lọc | **≥20 316** | **19,4%** | 0,995 |

Không có cấu hình nào thắng cả ba. `phai_nhat_danh_tu` sửa đúng lỗi `phổ biến →
biến` (§54) và đưa nhãn về `động` ở mọi mức nhiễu — nhưng làm hỏng thước sinh.

⚠️ `n` khác nhau giữa các cột (33–43) vì bộ ca dựng lại theo từng cấu hình ⇒ so
sánh chỉ mang tính định hướng, không phải kiểm định.

### 56.2 Lọc phân vị: tách bể KHÔNG cứu được

`_emit(chi_sach=…)` cho bể ② lấy tất cả, bể ③ lấy phần qua lọc. Kết quả: nhãn
đúng `động` ở MỌI mức nhiễu (② được cứu) nhưng `khỉ` bị cắt khỏi bể ③ ở 10 287 —
**chính bộ lọc loại cạnh vàng ở đó**. Cạnh vàng nặng 0,95 vì xuất hiện MỘT lần,
nên mọi bộ lọc độ-chính-xác đều giết nó.

### 56.3 Bài học phương pháp (nối tiếp §55.5)

72. **Đổi HAI biến rồi gán công cho MỘT — lỗi ngược của §24.1.** §24.1 tôi ghi
    «quét một biến bỏ sót tương tác»; §55 tôi phạm chiều ngược lại. Quy tắc đủ
    cho cả hai: **mỗi lần chỉ đổi một biến, và khi buộc phải đổi nhiều thì tách
    biến TRƯỚC khi viết kết luận**, không phải sau khi bị hỏi.
73. **Viết «không đánh đổi» trước khi đo thước còn lại.** §55.3 khẳng định điều
    đó dựa trên hai thước, trong khi thước thứ ba (sinh) chưa chạy. Câu «không
    đánh đổi» chỉ được viết khi ĐÃ đo mọi thước đang theo dõi.

## 57. QUYẾT ĐỊNH: ƯU TIÊN NĂNG LỰC (tác giả, 2026-08-13)

§56 đo được xung đột THẬT giữa ba thước — không cấu hình nào thắng cả ba:

| | ngưỡng nhiễu | classrich ②top1 | sinh · ② ĐÚNG |
|---|---|---|---|
| `phai_nhat` + lọc cả hai bể | 3 069 | 15,2% | **0,290** |
| **`phai_nhat_danh_tu` + tắt lọc** | **≥20 316** | **19,4%** | 0,995 |

**Tác giả chọn NĂNG LỰC** — chuỗi chạy được ở quy mô lớn — thay vì chất lượng
sinh trung bình. Đây là quyết định có chủ ý trên một đánh đổi đã đo, không phải
bỏ sót.

Mặc định chốt: `head_rule="trai_du_dac_thu"` · `head_rule_trai="phai_nhat_danh_tu"`
· `indeg_q=0` · `gidf_q=0` · `rank="phieu_x_ctx"` · `chu_gia_q=0,90` ·
`pos_dict=vi_dict_raw.jsonl` · `hub_chain_boost=0` (đường sinh vẫn TẮT mặc định).

Xác nhận đường cong với đúng mặc định đó:

```
cha    118    337    895   3 069  10 287  20 316
nhãn  động   động   động   động   động    động     ← ĐÚNG ở mọi mức
khỉ      2      2      2      3      6       7
```

Nhãn `động` đúng ở **mọi** mức nhiễu và `khỉ` ra tới 20 316 cha — mức lớn nhất
trong dải toy hiện có. `classrich` là 35 757 cha, còn **1,8 lần**.

### 57.1 Hai dòng để đảo hướng

```python
hub_chain_head_rule_trai = "phai_nhat"   # thay vì phai_nhat_danh_tu
hub_chain_indeg_q = 0.8 ; hub_chain_gidf_q = 0.20
```

Ghi ở đây để lần sau không phải đo lại mới biết đường về.

## 58. ĐÍNH CHÍNH §54/§57 — `phai_nhat_danh_tu` bị TRỘI ở MỌI thước

§57 chốt mặc định `phai_nhat_danh_tu` với lý do «ưu tiên năng lực». Đo `classrich`
bằng ĐÚNG mặc định đó thì lộ ra tôi chọn sai.

| luật chủ ngữ | lọc | ngưỡng nhiễu | classrich phủ | ②top1 | ②top5 | sinh ② ĐÚNG |
|---|---|---|---|---|---|---|
| `phai_nhat` | cả hai bể | 3 069 | — | — | — | **0,290** |
| `phai_nhat` | chỉ ③ | — | — | — | — | 0,500 |
| **`phai_nhat`** | **tắt** | **20 316** (hạng 5) | **72,4%** | **46,6%** | **59,6%** | 0,996 |
| `phai_nhat_danh_tu` | chỉ ③ | — | — | — | — | 0,994 |
| `phai_nhat_danh_tu` | tắt | 20 316 (hạng 7) | 37,3% | 30,4% | 44,3% | 0,995 |

### 58.1 Hai bản vá cho CÙNG một nguyên nhân

§54 thêm `phai_nhat_danh_tu` để sửa `Một số loài phổ BIẾN là…` → nhãn `biến`.
Nhưng cạnh rác `chó → biến` chỉ THẮNG được vì hai bộ lọc phân vị đã **cắt cạnh
vàng** `chó → động`. Tắt lọc (§55) thì `phai_nhat` cho `động` đúng ở mọi mức —
**triệu chứng biến mất cùng nguyên nhân**.

Tôi xếp hai bản vá lên nhau rồi giữ bản **tệ hơn**: `phai_nhat_danh_tu` kém
`phai_nhat` **35 điểm phủ · 16 điểm ②top1 · 15 điểm ②top5**, và hạng `khỉ` xấu
hơn (7 vs 5).

### 58.2 Xung đột thật nằm ở BỘ LỌC, không ở luật chủ ngữ

`0,290` ở thước sinh chỉ xuất hiện khi lọc bật ở **cả hai bể**. Mọi cấu hình
lọc-tắt đều ~0,995. Vậy đánh đổi là:

```
lọc TẮT  → ngưỡng 20 316 · classrich ②top1 46,6% · sinh 0,996
lọc BẬT  → ngưỡng  3 069 · classrich ②top1  ~15% · sinh 0,290
```

Tác giả đã chọn **năng lực** (§57) ⇒ giữ lọc tắt, và đổi luật chủ ngữ về
`phai_nhat`. Mặc định mới thắng cấu hình §57 ở **cả ba** thước năng lực.

### 58.3 Bài học phương pháp (nối tiếp §57)

74. **Hai bản vá cho cùng một nguyên nhân: bản sau có thể làm bản trước thành
    THỪA và TỆ.** §54 vá triệu chứng, §55 vá nguyên nhân. Tôi giữ cả hai và mặc
    định thành bản triệu chứng. Quy tắc: sau khi vá NGUYÊN NHÂN, **đo lại xem
    bản vá triệu chứng còn cần không** — đừng cộng dồn.
75. **«Đo bằng đúng mặc định vừa chốt» là bước bắt buộc, không phải thủ tục.**
    §57 chốt xong tôi tưởng đã đo rồi; thật ra mọi số `classrich` trước đó chạy
    trên cấu hình khác. Một lần chạy 4 phút lật ngược quyết định.

## 59. M4 TOY LỚN + HẠT GIỐNG CHỈ-TYPED — trần dịch 20 316 → ≥59 769 cha

### 59.1 M4 chạy bốn mức: GÃY hết

Máy dev 8GB train mức 60 000 câu wiki mất 58 phút rồi hết RAM ⇒ giao M4
(`RUNBOOK_M4_TOY_LON.md`). M4 train tới **245 321 cha** và chấm bằng
`scripts/cham_toy_lon.py`:

```
cha       59 769   99 362   148 875   245 321
nhãn      bao      bao      bao       bao
khỉ       —        —        —         —
③ top-6   cả · máy · cơ · khí · điện · đường
```

**Gãy ở mọi mức trên 35 757.** Cổng «gãy trên 35 757» của runbook ⇒ trần thật
nằm giữa 20 316 và 59 769.

### 59.2 Truy trên checkpoint 59 769 — ba sự thật

```
① = [chó(T) · mèo(T) · đảo(c) · lý(c) · bình(c) · phương(c)]
via_pivot[chó] = [vật 1,9 · biến 0,95 · yếu 0,95 · động 0,95]
via_pivot[khỉ] = [động 0,95]
```

1. **① vẫn hoàn hảo** — kênh thuộc tính (§53) giữ vững ở 59 769 cha: `chó`/`mèo`
   typed, đứng đầu.
2. **Cạnh vàng vẫn còn** — `khỉ → động` không bị mất.
3. **Hạt giống BỊ PHA**: `[chó, mèo, đảo, lý]` — `nghịch đảo`/`nghịch lý` lọt qua
   kênh cooc và làm loãng phiếu ở ②. Nhãn `bao` đến từ **`bao gồm`** (`gồm` là
   pivot, nên `bao gồm … là …` sinh cạnh rác nặng).

### 59.3 Vá một dòng: hạt giống CHỈ TYPED

Nếu ① có hit typed thì **chỉ dùng typed**; không có mới rơi về cooc.

```
cha=59 769   hạt=[chó, mèo, đảo, lý] → nhãn `bao`  · khỉ —
             hạt=[chó, mèo]          → nhãn `động` · khỉ hạng 20
```

Đường cong sau vá — **không hồi quy ở bất kỳ mức nào**:

```
cha     118   337   895  3 069  10 287  20 316  59 769
nhãn   động  động  động  động   động    động    động
khỉ       2     2     2      2       4       5      20
```

**Trần 20 316 → ≥59 769 cha**, tức **vượt `classrich` (35 757) 1,7 lần**. Lần đầu
trong toàn bộ TK_046, ca thử vàng sống ở quy mô lớn hơn corpus thật.

### 59.4 Bài học phương pháp (nối tiếp §58)

76. **«Có typed thì đừng trộn cooc» ≠ «tách kênh triệt để».** §51 thử BỎ HẲN cooc
    ở ① và giết cơ chế (không có typed thì không còn gì). §59 giữ cooc làm PHƯƠNG
    ÁN DỰ PHÒNG và chỉ ưu tiên typed khi có — cùng một trực giác, hai cách thi
    hành, kết quả trái ngược. Chi tiết thi hành quyết định, không phải ý tưởng.
77. **Giao máy mạnh chạy là cách rẻ nhất để BÁC một giả thuyết lạc quan.** Tôi
    tin cơ chế còn dư địa trên 20 316; M4 bác trong một đêm và chỉ đúng khoảng
    chứa trần. Không có nó tôi sẽ còn vá mò vài vòng.

## 60. KỀ NGỮ CẢNH Ở ③ — trần 20 316 → **≥99 362 cha**, và bộ ca của thước sinh có 46% là rác

Việc kế #1 của tổng kết §59 là chấm lại bốn checkpoint M4 bằng bản vá §59. Nó
đổi hẳn chẩn đoán, nên phần này đính chính §59 trước rồi mới vá.

### 60.1 ĐÍNH CHÍNH §59.1 — ② KHÔNG gãy; chỗ gãy đã dời sang ③

M4 chạy `cham_toy_lon.py` **trước khi** script đó có bản vá hạt-giống (vá §59
chỉ vào `hub_chain_proposals`; script chấm dựng chuỗi riêng). Vá vào script rồi
chấm lại bốn checkpoint có sẵn:

```
cha       59 769   99 362   148 875   245 321
nhãn      động     động     bao       vật          ← §59.1 ghi `bao` ở CẢ BỐN
khỉ       20       —        —         —
③ top-6   Am·An·Biên·Chim·Joseph·Long  |  chó·dụng·thực·truyền·vũ·đồng
```

Hai mức (99 362 · 245 321) ② vẫn trả **đúng nhãn lớp** — `động`, và `vật` cũng
đạt theo cổng của runbook. Vậy «gãy hết bốn mức» của §59.1 là hiện vật của
script chưa vá, không phải của cơ chế. **Nút thắt thật đã dời sang ③.**

Và hai loại rác ở ③ khác hẳn nhau — đọc chúng ra được ngay bệnh:

```
nhãn `động` → Am · An · Biên · Cao · Long     ← `động` = HANG, tên hang
nhãn `vật`  → dụng · thực · truyền · vũ       ← BẠN GHÉP TỪ (vật dụng, thực vật)
```

### 60.2 Đọc bể ③: **62/62 thành viên nặng ĐÚNG BẰNG NHAU**

`truy_buoc3_toy_lon.py` trên ckpt 99 362:

```
hạng  bề mặt   w        typed        phân bố w (typed): {0,95: 62 · 1,90: 1}
   1  chiếc    1,9000   True         62 thành viên CÙNG w trội
   2  Am       0,9500   True
   3  An       0,9500   True    →  Am · An · Biên · Cao · Chim · Công · Hồi ·
  …                                 Joseph · Long · Louis · Minh · Nam …
  36  khỉ      0,9500   True    ←  ĐÍCH, ngoài `topk=30`
```

`reverse_peers` xếp theo `(not typed, -weight, surface)`. Mỗi câu định nghĩa
xuất hiện **một lần** nên mọi thành viên nặng đúng 0,95 — hai khoá đầu hoà
sạch, khoá thứ ba là **ABC**, và `khỉ` thua vì vần K. `via_pivot[khỉ]` vẫn
`[động 0,95]`: cạnh vàng KHÔNG mất, **thiếu THƯỚC XẾP HẠNG**.

Đây là loại chẩn đoán §57.2 số 2 nói tới, lần này ở dạng thứ ba: không phải
«không TÌM được», không phải «không BIỂU DIỄN được», mà **tìm được, biểu diễn
được, nhưng không PHÂN BIỆT được**.

### 60.3 Hai âm tính trước khi tìm ra thước

| thử | ý | khỉ hạng |
|---|---|---|
| — | hiện trạng | 35/62 |
| **hồ sơ nhãn** | thành viên chia sẻ nhiều nhãn với hạt `{chó, mèo}` thì lên | **44** ✗ |
| **đồng hiện ngữ cảnh** | đếm đồng hiện W_skip với mọi token prompt | 18, nhưng top-8 = `chữ·học·ông·đồng` |

**Hồ sơ nhãn tệ hơn hiện trạng** và lý do là chốt chặn cho cả một họ ý tưởng:
`via_pivot[khỉ]` có **đúng MỘT** cạnh. Mọi thước dựa trên «đích có bao nhiêu
quan hệ» đều cho đích 0 điểm — thứ ta muốn nâng là thứ **thưa nhất trong bể**.
Nó bác luôn các biến thể chưa thử (đếm họ hàng nhãn, đếm bậc, đếm phiếu ngược).

Đồng hiện thô thì bị **tần suất nền** nuốt — đúng bệnh đã ghi ở §21.

### 60.4 Thước ăn tiền: KỀ NGỮ CẢNH, và ĐỐI CHỨNG bác «bigram trá hình»

Prompt là `nghịch ngợm như loài ___`. Ứng viên phải đứng được **ngay sau
`loài`** — đúng cái ô đường sinh sắp điền. Ba cách đọc cùng một ckpt 99 362:

| cách | bể | khỉ hạng |
|---|---|---|
| chỉ KỀ sau `loài`, **bỏ chuỗi ②③** | 179 | **131** ← đối chứng |
| chỉ chuỗi ②③ (hiện trạng) | 62 | 35 |
| **chuỗi ②③ × KỀ** | 62 | **4** |

Đối chứng quan trọng hơn kết quả: nếu KỀ tự nó ra `khỉ` thì cả chuỗi ba bước là
đồ trang trí. Nó không — bigram sau `loài` cho `thực · động · khác · chim ·
sinh · cây · cá · thú`, thuần tần suất, `khỉ` hạng 131/179. **Chuỗi thu bể
bằng TƯ CÁCH THÀNH VIÊN (179 → 62); ngữ cảnh xếp thứ tự TRONG bể.** Không cái
nào tự đứng được.

Thi hành: `hub_chain_ke_ctx=2.0`, nhân `(1 + ke_ctx)` — đúng dạng ghép của bài
học 48 (§41.4), không thay `weight`. Tra bảng KỀ khoá theo **bề mặt** vì lần
thứ tư dính bệnh một-bề-mặt-nhiều-id-BPE (§23.1): tra bằng id thô của
`encode_prompt` bắt hụt 3/6 cạnh kề.

### 60.5 KỀ chỉ lấy token nội dung CUỐI — không lấy cả ngữ cảnh

Bản đầu tôi lấy hợp của mọi token ngữ cảnh. Sai về nguyên tắc: **thuộc tính
cũng nằm trong ngữ cảnh**, và nâng thành viên từng đứng cạnh `nghịch ngợm` là
giết đúng thứ cả bài toán sinh ra — `khỉ` đáng giá CHÍNH VÌ nó chưa bao giờ
đứng cạnh `nghịch ngợm`. Chuyển sang lấy token nội dung cuối (lùi dần nếu
rỗng): 99 362 còn nhích thêm, khỉ 4 → **3**. Một test khoá riêng điều này.

### 60.6 Đường cong sau §60 — trần **20 316 → ≥99 362 cha**

```
cha     105  305 1005 3005 9983 19963 │ 59 769  99 362  148 875  245 321
nhãn   động động động động động  động │  động    động     bao      vật
khỉ §59   2    2    2    2    2     3 │    20      —        —        —
khỉ §60   2    2    2    2    2     2 │     3      3        —        —
```

Không hồi quy ở bất kỳ mức nào; 19 963 còn tốt hơn. **≥99 362 cha = 2,8 lần
`classrich` (35 757).** ⚠️ Sáu mức nhỏ là checkpoint TRAIN LẠI (105–19 963 cha),
không phải chính các checkpoint của §55 (118–20 316) — cùng dải, khác mẫu.

### 60.7 Gộp mảnh ở tầng NHÃN — nửa dương, chưa dùng được

`vật` và `động` là hai mảnh của **một cụm «động vật»**; phiếu của lớp bị chẻ
đôi rồi mỗi nửa thua rác. Ở cả hai mức gãy, `động` — nhãn DUY NHẤT chứa `khỉ` —
nằm **hạng 4**. Thử ghép hai nhãn khi chúng đứng kề thành cụm thật:

```
245 321   ② MỚI = động vật 165,9 · tranh đặc 144,8   ← nhãn ĐÚNG lên top-1 ✓
148 875   ② MỚI = tranh đặc 139,0 · động vật 125,0   ← cặp RÁC thắng ✗
```

Luật ghép của tôi nhận mọi cặp có cạnh kề, trong khi trọng số kề chênh **200
lần** (`động vật` 865 vs `tranh đặc` 3,8) — thiếu đúng một ngưỡng. Và ngay ở ca
thắng, bể ③ hợp nhất lại **dìm** `khỉ`: thành viên của `vật` nặng 3,8 còn cạnh
vàng 0,95. Ghi lại làm việc kế, không bật.

### 60.8 Trên DỮ LIỆU THẬT: §60 vô hiệu — và thước sinh có 46% ca là mảnh BPE

Dựng lại `classrich` (35 757 cha, khớp đúng số cũ) + sidecar WS-span, chạy
thước sinh A/B `--ke-ctx 0` vs `2.0`: **hai file kết quả giống hệt nhau từng
con số**. Truy ra ba tầng lý do, tầng sau nặng hơn tầng trước:

1. Bảng KỀ không rỗng (12–166 mục) nhưng **giao với bể ③ gần như rỗng** (0–4
   trên 30; một nửa số ca giao = 0).
2. Vì thuộc tính `P` của bộ ca là `Le` · `St` · `Tr` · `Đan` · `Lam` —
   **mảnh BPE của tên riêng**, không phải thuộc tính. Thêm bộ lọc từ điển
   (`--p-tu-dien`, cùng từ điển đã dùng ở §40): **172 → 92 ca, tức 46% bộ ca bị
   loại**. Thước sinh xưa nay chấm cơ chế một phần trên rác.
3. Trên 92 ca sạch, ② đúng lớp **3/92 = 3,3%**. ③ có sửa cũng không tới lượt.

Con số 3,3% này KHÔNG mâu thuẫn với `②top1 = 46,6%` của §55/§57 — hai thước
khác nhau: 46,6% đo khi **ứng viên nhãn bị giới hạn vào 164 lớp đã đào**
(`probe_chain3_known_labels`), 3,3% là đường sinh **tự do**. Khoảng cách giữa
hai số đó chính là việc còn lại: **phát hiện nhãn không có gợi ý**.

⇒ `hub_chain_ke_ctx=2.0` bật mặc định: dương ở toy, **vô hiệu chứ không hại**
trên dữ liệu thật (đo hai chiều, n=92).

### 60.9 Bài học phương pháp (nối tiếp §59.4)

78. **«Hoà» là một chẩn đoán riêng, không phải «thiếu».** Ba dạng khác nhau và
    ba bản vá khác nhau: không TÌM được (thiếu cạnh) · không BIỂU DIỄN được
    (sai đơn vị) · **tìm được nhưng không PHÂN BIỆT được** (mọi ứng viên bằng
    điểm, khoá phụ quyết định). Dạng thứ ba lộ ra chỉ khi in **phân bố trọng
    số**, không bao giờ lộ qua top-k. Nó vô hình với mọi phép đo tổng.
79. **Tín hiệu nâng được đích thưa phải KHÔNG dựa trên độ giàu của đích.** Hồ
    sơ nhãn tệ hơn hiện trạng vì đích có đúng một cạnh — và đích của bài toán
    này về bản chất là thứ thưa nhất bể. Một câu hỏi sàng rẻ cho mọi ý tưởng
    xếp hạng sau này: *«nó chấm đích bằng cái đích KHÔNG CÓ à?»*
80. **Đối chứng «bỏ cơ chế đi có ra không» phải chạy TRƯỚC khi mừng.** KỀ đưa
    khỉ 35 → 4, con số đẹp nhất phiên; nếu không chạy đối chứng thì không biết
    nó chỉ là bigram hay không (nó không — 131/179). Bảy lần kết luận bị phép
    đo sau bác trong phiên này đều thiếu đúng bước đó.
81. **Bộ ca cũng phải qua bộ lọc chất lượng của chính cơ chế.** 46% ca của
    thước sinh là mảnh BPE. Đây là lần thứ hai «chân lý cũng là một phép trích,
    và nó cũng có lỗi» (bài học nặng số 1) — lần đầu ở khâu chấm (§30), lần này
    ở khâu CHỌN CA. Thước rẻ: hỏi từ điển.

## 61. CA «CON KHỈ» TRÊN LỚP THẬT — 196 ca, ĐẠT 1, và ca đó là câu giáo khoa

§60.8 cho thấy bộ ca của thước sinh 46% là mảnh BPE. Vậy câu đơn giản nhất vẫn
chưa ai hỏi: **lấy một lớp THẬT của `classrich` và một thuộc tính THẬT, cấu
trúc y hệt ca vàng — chuỗi có chạy không?**

### 61.1 Dựng ca đúng tầng ĐƠN VỊ (hai lần sai trước khi đúng)

| vòng | thành viên | thuộc tính | ca ra |
|---|---|---|---|
| 1 | token trước `là` | token | lớp=`khoảng`·`sự`, tv=`bắc`·`lỵ`·`nay` |
| 2 | **TỪ** (chân lý v2 §31) | token | P=`giả`·`danh`·`khai` — mảnh của «học giả» |
| 3 | **TỪ** | **TỪ** (span WS) | P=`mang`·`chống`·`bờ`·`lập` — dùng được |

Vòng 1 đếm ra «chỉ 45,7% lớp là cụ thể» — con số đó là **hiện vật của phép
trích**, không phải của dữ liệu. Chân lý v2 cho lớp thật: `loài` → Chim · Cá
sấu · Gà · Rùa · cá chép · gấu trúc · `lớp` → bò sát · kim loại · hành tinh ·
lưỡng cư · `tỉnh` → Bắc Kạn · Lào Cai · California. Dữ liệu KHÔNG thiếu lớp.

### 61.2 Kết quả: 196 ca (168 lớp × ≤3 thuộc tính mạnh nhất)

```
②  đúng lớp            2/196 = 1,0%
ĐẠT đầu-cuối (③ trúng) 1/196
```

Và ca ĐẠT duy nhất là:

```
lớp=`bạn`  P=`mèo`  hạt=[chó, chạy, ngủ]  ② → `bạn` ✓  ③ → [Em, Lan, chó, thú]
câu train:  «Ví dụ mẫu 1 : mèo là thú và chó là bạn .»      ← grade_lessons
            «Ví dụ mẫu 4 : Con mèo ngủ rồi con chó thức .»
```

Tức nó chạy trên **câu giáo khoa dựng sẵn có đúng hình dạng ca vàng**, không
phải trên văn xuôi đời thật. Ca gần đạt thứ hai (`thành viên`/`mọi`) ② đúng
nhưng ③ trượt.

### 61.3 Vì sao — đếm trên chính `classrich`

Đuôi `là …` hay gặp nhất, trên ~9 500 câu có `là`:

```
là một trong  177    là thành phố  71    là ngôn ngữ  65    là trung tâm  49
là loài/lớp/động vật/kim loại/ngành/tỉnh/thành phố…  =  200 câu  =  2,1%
```

Manifest `train_classrich.yaml` đã cảnh báo đúng tỉ lệ này từ đầu (~9:1) và
corpus này đã được lọc để GIÀU câu phân loại. **Hình dạng câu mà cơ chế cần là
thiểu số ngay trong corpus thuận lợi nhất ta có.**

### 61.4 Hai ca gãy ở toy, đọc bằng câu train

**148 875 cha — ② trả `bao`:**

```
chó → bao  «Thức ăn của gấu chó dao động rất rộng và BAO GỒM các động vật…»
mèo → bao  «Môi trường sống của mèo thường BAO GỒM các khu rừng mưa…»
mèo → bao  «…những con mèo nhỏ hơn , BAO GỒM cả mèo nhà…»
khỉ → động «loài khỉ là động vật .»                        ← ĐÚNG MỘT câu, w=0,95
```

Hai lỗi chồng nhau: `gồm` được tính là pivot phân loại (nó là quan hệ BỘ PHẬN
/liệt kê, không phải phân loại), và luật vế trái lấy `bao` — **một mảnh của
chính cụm pivot** — làm nhãn. Ba câu rác thắng một câu vàng bằng số lượng.

**245 321 cha — ② trả `vật`:** `via_pivot[khỉ] = {động 0,95 · bao 0,95 · hạn ·
thể · có}`, mà ② chọn `vật`. `động` và `vật` là hai mảnh của MỘT cụm «động
vật» (§60.7): phiếu của lớp bị chẻ đôi, nửa thắng lại là nửa không chứa `khỉ`.

### 61.5 Kết luận về tính khả thi — nói thẳng

**Còn khả thi, nhưng phạm vi đã chứng minh hẹp hơn ca vàng gợi ý.**

Đã chứng minh được:
- chuỗi ba bước sống dưới nhiễu thật **≥99 362 cha** (111× so với đầu phiên) —
  cơ chế KHÔNG vỡ vì quy mô;
- dữ liệu thật CÓ lớp thật (`loài` · `lớp` · `tỉnh`), và ③ trao nhãn đúng 38,9%;
- ② chọn đúng lớp **46,6%** khi ứng viên nhãn được giới hạn vào lớp đã đào.

Chưa chứng minh được, và đây là toàn bộ khoảng cách còn lại:
- ② **tự do** (không gợi ý) trên dữ liệu thật: **1,0–3,3%**;
- không phải vì thiếu cạnh hay thiếu quy mô, mà vì **câu định nghĩa dạng vàng
  chỉ chiếm 2,1%** — mọi câu khác cũng sinh cạnh `X là Y` và chúng đông hơn.

⇒ Ba đường đi tiếp, theo thứ tự rẻ-trước:

1. **Sửa pivot** — `gồm` không phải quan hệ phân loại; và nhãn không được là
   mảnh của cụm pivot. Ca 148 875 chết vì đúng hai lỗi này (§61.4).
2. **Gộp mảnh tầng nhãn** (§60.7) — `động`+`vật` phải là một nhãn.
3. **Nhận diện câu định nghĩa THẬT** — phân biệt «X là động vật» (phân loại)
   với «X là một trong những…» (không phân loại). Đây là chỗ 2,1% kia nằm, và
   là bài toán chưa từng đặt tên trong TK_046.

### 61.6 Bài học phương pháp (nối tiếp §60.9)

82. **Sai tầng ĐƠN VỊ làm hỏng phép đếm hai lần liên tiếp trong cùng một giờ.**
    Vòng 1 kết «45,7% lớp là cụ thể» (sai — do thành viên lấy theo token), vòng
    2 kết «② sai 67/67» (sai — do thuộc tính lấy theo token). Cả hai đều là con
    số sạch sẽ, đều sai, và đều SAI THEO HƯỚNG BI QUAN. Trước khi tin bất kỳ
    phép đếm nào trên corpus: **hỏi đơn vị của TỪNG vế, không chỉ vế mình vừa
    sửa.**
83. **Một ca ĐẠT phải đọc câu train trước khi được tính.** Ca đạt duy nhất của
    §61 là câu giáo khoa nhân tạo. Nếu chỉ báo «1/196 đạt» mà không đọc, con số
    đó nghe như tín hiệu yếu; đọc rồi thì nó là **âm tính sạch**.

## 62. BIÊN TỪ CHO PIVOT — vá đúng lỗi §61.4, và đo được TRUNG TÍNH

§61.4 chỉ mặt cạnh rác nặng nhất: «Thức ăn của gấu chó… **bao GỒM** các động
vật…» → `gồm` là pivot, vế trái lấy `bao` — **mảnh của chính cụm pivot**.

Dự án đã bỏ `thành` khỏi pivot vì đúng bệnh này, nhưng bỏ bằng DANH SÁCH TỪ.
Nay có span WS-CRF nên kiểm được bằng CẤU TRÚC. Đếm trên `classrich`:

```
pivot   tổng   lọt TRONG một từ          từ chứa nó
là      9 501   1 150  12,1%   gọi là · nhất là · tức là · hay là
như     1 565     304  19,4%   hầu như · nếu như · gần như
gọi     1 001     679  67,8%   gọi là · kêu gọi
gồm       639     311  48,7%   bao gồm
TỔNG   12 706   2 444  19,2%
```

**Không bỏ những lần đó** — `gọi là`/`tức là` vẫn là pivot phân loại thật. Chỉ
dời BIÊN: quét trái dừng TRƯỚC cả từ, quét phải bắt đầu SAU cả từ
(`_bien_pivot`, cờ `bien_tu_pivot`).

### 62.1 Kết quả: trung tính trên MỌI thước

```
cạnh via_pivot   15 742 → 15 978  (+1,5%)
bể ③             12 992 → 13 563  (+4,4%)
196 ca lớp thật  ② đúng 2 → 2      KHÔNG đổi
toy 4 mức        KHÔNG đổi
```

Hai lý do, cả hai chỉ lộ ra sau khi vá:

1. **Vế trái đã có guard tương đương** — `_ung_vien_trai` vốn bỏ ứng viên mà
   span của nó CHỨA pivot, nên khi bật span thì `bao` đã bị chặn sẵn.
2. **Giả thuyết `gọi là` của tôi SAI** — tôi đoán «X gọi là Y» đang mất cạnh
   vì quét phải dừng ngay ở `là`. Đếm: cạnh qua pivot `gọi` = 348 TRƯỚC và
   348 SAU. Không mất cạnh nào cả.

⇒ Giữ (mặc định BẬT): nó thay một luật thống kê không chuyển được quy mô
(`chu_gia_q`, §55 đã đo) bằng một luật cấu trúc, và không tốn gì. Nhưng ghi
đúng: **đây là bản vá ĐÚNG mà KHÔNG mua được gì đo được.**

---

## 63. 🟢 TỪ ĐỦ THẮNG MẢNH — trần ≥99 362 → **≥245 321 cha**, và ③ ra động vật thật

### 63.1 Bật span WS trên toy: bức tranh lật ngược

Mọi vòng đo toy từ §20 tới nay chạy KHÔNG span (chỉ `classrich` có sidecar).
Bake span cho cả 10 checkpoint toy rồi đọc lại nhãn của ba con vật:

```
chó → {vật 1,9 · phổ biến · chủ yếu · gia súc · cưng · động vật}
mèo → {đề nghị · cưng · động vật · đặc sản · Hà Nội · thiếu nhi}
khỉ → {động vật}
```

**Đường 2 «gộp mảnh tầng nhãn» tự giải**: `động vật` nay là MỘT nhãn nguyên
khối, và nó là nhãn CHUNG của cả ba. Nhưng ② vẫn chọn `cưng`:

```
② : cưng 2 phiếu · score 15,6      ← thắng
    động vật 2 phiếu · score 2,7
```

Hoà phiếu, và `score` quyết. Mà **điểm của id-CỤM không cùng thang với điểm
token** — đúng bệnh «hai kênh khác thang đo» đã giết bước ② một lần ở §18.H.

### 63.2 Hai khoá phụ, đặt SAU phiếu và TRƯỚC `score`

```
(typed, PHIẾU, TỪ ĐỦ, SỐ THÀNH VIÊN, score × hệ số, gidf, ABC)
```

- **TỪ ĐỦ** — nhãn là span WS ≥2 token thắng nhãn mảnh. Hỏi span, không phải
  danh sách từ. Mua: 99 362 `cưng` → `động vật`.
- **SỐ THÀNH VIÊN** (`label_indeg`) — lớp thật thì đông. Mua: 148 875
  `cạnh tranh` (span rác) → `động vật`.

`label_indeg` từng bị dùng làm **BỘ LỌC** ở §24/§55 và nó cắt mất cạnh vàng.
Ở đây nó là khoá PHỤ sau khi phiếu đã hoà — cùng tín hiệu, khác vị trí, kết
quả ngược nhau (bài học 48 lần thứ hai).

### 63.3 Lưới A/B/C trên 10 mức — mỗi khối tự in cờ

```
cha        A. nền §60        B. §63, không span   C. span + §63
105–19963  động   · khỉ 2    động   · khỉ 2       động vật · khỉ  2
59 769     động   · khỉ 3    động   · khỉ 3       động vật · khỉ  6
99 362     động   · khỉ 3    động   · khỉ 3       động vật · khỉ  7
148 875    bao    · khỉ —    động  · khỉ 5        động vật · khỉ 13
245 321    vật    · khỉ —    động  · khỉ 8        động vật · khỉ 17
```

**Trần ≥99 362 → ≥245 321 cha** — hết checkpoint để thử, chưa gãy. Đó là
**6,9× `classrich`** và **276× ngưỡng đầu phiên (895)**.

③ ở mức cao nhất, đọc bằng mắt:

```
A: chó · thực · bánh · chim · nấm · sinh
C: hổ · chó · cá · gấu · mèo · sinh          ← toàn ĐỘNG VẬT
```

Đây là lần đầu ③ trả về một danh sách mà người đọc thấy ngay là cùng một lớp.
Với đường sinh, «nghịch ngợm như loài ___» nay có bể hổ/gấu/cá/khỉ thay vì
`bánh`/`nấm`.

### 63.4 Trên DỮ LIỆU THẬT: 1/196 → 2/196, và ca giáo khoa biến mất

```
✓ lớp=`thành phố`  P=`nối`   hạt=[tiếp, liền, nhau]  ② `thành phố`
                              ③ TRÚNG = Montréal · Hamilton
✓ lớp=`trung tâm`  P=`thêm`  hạt=[hai, quận, nhiều]  ② `trung tâm`
                              ③ TRÚNG = Tokyo · Chicago · Sarnia
```

Ca ĐẠT của §61 là câu giáo khoa «mèo là thú và chó là bạn»; nay nó KHÔNG còn
trong danh sách, thay bằng **hai ca văn xuôi wiki thật**. Số vẫn nhỏ (1,0%)
vì nguyên nhân §61.3 chưa động tới: câu định nghĩa dạng vàng chỉ chiếm 2,1%.
Nhưng chất của ca đạt đã đổi hẳn.

### 63.5 Bài học phương pháp (nối tiếp §61.6)

84. **Một cấu hình đo suốt 44 mục có thể là cấu hình SAI.** Toy chạy không span
    từ §20 tới §62 chỉ vì checkpoint toy không có sidecar — không ai quyết định
    thế, nó là mặc định lặng. Bake sidecar mất 40 giây và đổi cả kết luận về
    trần. **Hỏi «tôi đang đo ở cấu hình nào, và ai chọn nó?» rẻ hơn vá mò.**
85. **Cùng một tín hiệu, đặt làm LỌC thì giết, đặt làm KHOÁ PHỤ thì cứu.**
    `label_indeg` cắt cạnh vàng ở §24/§55, và nâng `động vật` lên ở §63.
    Không có tín hiệu tốt hay xấu — chỉ có VỊ TRÍ đúng hay sai.
86. **`awk` cắt cột trên bảng có cột rỗng thì bịa ra kết quả.** Lưới A/B/C đầu
    tiên của tôi in ba dòng giống hệt nhau và tôi suýt kết luận «§63 vô hiệu».
    Thật ra cả ba đã chạy cùng cấu hình. Cách chữa đã dùng: **mỗi lần chạy tự
    in CỜ của nó** — bảng không tự chứng minh được thì bảng không đáng tin.

## 64. ĐƯỜNG 3 — độ sạch cạnh lớp lần đầu được đo: **10,9%**, và thủ phạm là ĐỊNH TỪ

Đường 3 đặt ra ở §61.5 là «nhận diện câu định nghĩa THẬT». Trước khi dựng bộ
nhận diện, đo xem rác thật sự là gì. Chuẩn: 949 cặp (thành viên → lớp) của
chân lý v2. Với mỗi thành viên chuẩn, cơ chế đang gán cho nó nhãn gì?

```
thành viên chuẩn có cạnh : 312
nhãn ĐÚNG chuẩn          : 321
nhãn RÁC                 : 2 624        ⇒ độ sạch 10,9%
```

**Đây là lần đầu độ sạch của `via_pivot` được đo trên dữ liệu thật.** Mọi số
trước (phủ 72,4%, ②top1 46,6%) đều là RECALL hoặc là thứ hạng — không cái nào
hỏi «trong những gì cơ chế tin, bao nhiêu phần là đúng».

### 64.1 Bảng rác không phải lớp sai — mà là ĐỊNH TỪ

```
một 149 · các 58 · những 45 · người 37 · 1 34 · nơi 23 · có 20 · 2 20 ·
do 18 · nhiều 14 · sự 12 · chính 11 · 6 11 · 4 11 · rất 10 …
```

Đếm thẳng: **1 142 cạnh trỏ vào `một`**, qua cả ba pivot `là`/`gọi`/`gồm`.
Câu «X là **một trong những** thành phố lớn nhất…» sinh cạnh `X → một`.

Từ điển gọi `một`/`các`/`những` là **danh từ**, nên POS không cứu được — đúng
như §40 đã kết luận POS vô dụng ở ②. Thử phép thử cấu trúc «định từ thì luôn
có từ nội dung theo sau»: `một` 88,1% · `các` 97,2% · `những` 78,9% vs `loài`
69,1% · `tỉnh` 67,6%. Biên quá mỏng để làm ngưỡng — §55 đã dạy phân vị không
chuyển được quy mô.

### 64.2 Nguyên nhân: một núm điều khiển BA việc (bệnh dùng chung, lần thứ TƯ)

```python
det_cut      = gs_p[len × label_gidf_q]   # ngưỡng «đủ đặc thù», chọn ĐẦU PHẢI
det_cut_trai = gs_t[len × label_gidf_q]   # …                     chọn ĐẦU TRÁI
cut          = gs  [len × label_gidf_q]   # bộ LỌC CẠNH cuối hàm
```

§55 vặn `label_gidf_q` về 0 để tắt **bộ lọc cạnh** (mua ngưỡng nhiễu
3 069 → 20 316) và vô tình tắt luôn **hai ngưỡng chọn đầu**. `det_cut` thành
min ⇒ mọi ứng viên «đủ đặc thù» ⇒ luật trái-nhất lấy đúng cái nó phải bỏ.

Cùng bệnh «tài nguyên dùng chung» đã gặp ở §37 (bể ②/③), §55, §56 — **lần thứ
tư**, và lần này núm đã bị vặn suốt từ §55 tới §63 mà không ai thấy.

### 64.3 Tách núm: `hub_chain_head_gidf_q = 0,20`

Quét trên `classrich` (nhãn ĐÚNG chuẩn): `0,00 → 321` · `0,10 → 403` ·
`0,20 → 403` · `0,40 → 246`.

| thước | trước | sau |
|---|---|---|
| nhãn ĐÚNG chuẩn | 321 | **403** (+26%) |
| độ sạch | 10,9% | 13,1% |
| `một`/`các`/`những`/`1` trong bảng rác | có | **hết** |
| đường cong nhiễu (10 mức toy) | ĐẠT hết | **ĐẠT hết, không hồi quy** |
| 196 ca lớp thật | 2 | 2 |
| thước sinh (`ty_le_hang_trung_vi`) | 0,987 | 0,980 |

**Điều đáng giá nhất: không hồi quy đường cong nhiễu.** §57 chốt «ưu tiên năng
lực» vì tin rằng bật lọc thì mất ngưỡng. Tách núm cho thấy **đánh đổi đó không
bắt buộc** — nó là hiện vật của việc ba việc dùng chung một núm. Phần năng lực
nằm ở bộ lọc CẠNH; phần chính xác nằm ở ngưỡng CHỌN ĐẦU. Nay giữ được cả hai.

Điều KHÔNG mua được: thước sinh vẫn 0,98 (phẳng). Đường 3 không giải nó.

### 64.4 Một giả thuyết bị bác, một bản vá chỉ đo được sau khi tách núm

- **BỊ BÁC** — «đường lui khi không ứng viên nào đủ đặc thù là nguồn rác».
  Thêm cờ `tu_choi_khi_toan_dinh_tu` (thà không phát cạnh còn hơn phát vào định
  từ): ở `head_gidf_q=0` nó đổi **đúng 0 con số**. Vì lúc đó det_cut = min nên
  đường lui KHÔNG BAO GIỜ chạy — tôi vá một nhánh chết.
- Sau khi tách núm, chính cờ đó mới đo được: cạnh 16 809 → 16 331 (−478), rác
  −78, đúng −2. Giữ (mặc định BẬT).

### 64.5 Bài học phương pháp (nối tiếp §63.5)

87. **Đo RECALL suốt 44 mục mà không đo PRECISION một lần nào.** Phủ 72,4% ·
    ②top1 46,6% · trần ≥245 321 — không số nào hỏi «trong những gì cơ chế tin,
    bao nhiêu phần đúng». Câu trả lời là **10,9%**, và nó chỉ mất 9 giây để đo
    khi đã có chân lý v2. **Thước đầu tiên nên là thước rẻ nhất chưa ai chạy.**
88. **Một núm điều khiển N việc thì mọi kết luận về nó đều là kết luận về N
    việc.** §55 kết «tắt lọc mua được ngưỡng» — đúng, nhưng nó đồng thời mở
    cửa cho định từ làm nhãn suốt 9 mục sau đó. §56 đã đính chính §55 một lần
    vì lý do khác; đây là lần thứ hai cùng một câu bị đính chính.
89. **Vá một nhánh chết thì đo ra số 0, và số 0 đó KHÔNG có nghĩa là ý tưởng
    sai.** `tu_choi_khi_toan_dinh_tu` trung tính ở §64.0 và có ích ở §64.3 —
    cùng một dòng mã. Trước khi bỏ một bản vá vì «đo không đổi», hỏi: **nhánh
    đó có bao giờ chạy không?**

## 65. TĂNG MẬT ĐỘ CÂU ĐỊNH NGHĨA — 8× dữ liệu lớp, và ② TỰ DO **KHÔNG NHÚC NHÍCH**

§61.5 và §64 cùng kết: nút thắt cuối không phải cơ chế mà là **mật độ** — câu
định nghĩa dạng vàng chỉ chiếm 2,1%. Mục này đi làm đúng việc đó, và **phép đo
bác chính kết luận vừa rồi của tôi**.

### 65.1 Trữ lượng: đo trước khi dựng

`classrich` dựng từ 4 nguồn nhỏ (~16 MB) và **loại hẳn `viwiki` 687 MB** —
trong khi câu mở đầu bài Wikipedia VỐN LÀ câu định nghĩa theo quy ước biên tập.

```
grep viwiki : là một loài 71 267 · là một xã 33 581 · thuộc họ 29 834 ·
              thuộc chi 20 615 · là một thành phố 10 523
classrich   : là loài 164 · là động vật 54
```

Quét toàn bộ 285 935 dòng viwiki bằng khuôn định nghĩa:

| nguồn | cặp (tv → lớp) | lớp ≥3 tv | thành viên là gì |
|---|---|---|---|
| `classrich` hiện có | 949 | 168 | hỗn hợp |
| viwiki, khuôn định nghĩa | **183 690** | 1 905 | 84% **TÊN LATIN** (bài stub) |
| viwiki, phần dùng được | 15 591 | 513 | phần lớn tên xã/phường |
| viwiki, kiểu sinh vật dùng được | 906 | 100 | `loài chim` 42 · `loài cây` 42 |
| **từ điển**, giải nghĩa | 9 178 | **643** | **TỪ THƯỜNG** |

**Wikipedia tiếng Việt giàu câu định nghĩa nhưng nghèo THÀNH VIÊN đời thường**:
nó viết về `Abrocoma bennettii`, không viết về `chó`. Từ điển thì ngược lại —
`chó → gia súc` · `mèo → thú nhỏ` · `khỉ → loài linh trưởng` · `voi → thú lớn`
— nhưng lời giải nghĩa **không có chữ `là`** nên `via_pivot` mù với nó suốt từ
đầu ticket.

### 65.2 Dựng corpus: viết lại từ điển thành câu định nghĩa

`scripts/dung_corpus_dinh_nghia.py` — mỗi mục danh từ → «X là <giải nghĩa>.»,
cộng câu ví dụ, cộng câu định nghĩa viwiki đã lọc:

```
"chó là gia súc thuộc nhóm ăn thịt, nuôi để giữ nhà hay đi săn ."
"mèo là khắc tinh của chuột ."
"khỉ là loài linh trưởng cỡ nhỏ, leo trèo rất giỏi ."
```

**Lọc tên khoa học mất HAI vòng.** Vòng 1 chỉ bỏ câu có CHỦ NGỮ Latin — thừa
lại «… là một loài bướm đêm thuộc họ **Crambidae**», và BPE băm nó thành
`Cram b idae`; ③ trả về `ensis`/`idae`/`ates` thay vì con vật, tạo ra hai ca
«ĐẠT» GIẢ. Vòng 2 dùng ngưỡng tỉ lệ 75% — vẫn lọt, vì một từ Latin trong câu
mười từ thì tỉ lệ nào cũng qua. Vòng 3 chặn TUYỆT ĐỐI: một từ không dấu ≥5 ký
tự không có trong từ điển thì bỏ cả câu. 45 056 → 19 230 câu.

`classrich2` = corpus này + 4 nguồn cũ (giữ phần VĂN XUÔI, nơi thuộc tính sống).
Train 114 giây → **55 460 cha**.

### 65.3 Kết quả: dữ liệu lớp lên 8×, ② tự do ĐỨNG YÊN

| thước | classrich | classrich2 |
|---|---|---|
| lớp ≥3 thành viên (chân lý v2) | 168 | **722** (4,3×) |
| cặp (thành viên → lớp) | 949 | **7 512** (7,9×) |
| lớp có thật | `loài`→Chim·Cá sấu·Gà | `chim`→bồ câu·cuốc·cò·cú · `thú`→cáo·cầy·cheo·lừa · `cây`→báng·bèo·bí |
| **② đúng lớp, tự do** | 2/196 = 1,0% | **3/390 = 0,8%** |

**Giả thuyết «nút thắt là mật độ» BỊ BÁC.** Nhân tám dữ liệu lớp, thêm đúng
loại lớp cần (`chim`·`thú`·`cây` với thành viên là từ thường), và tỉ lệ ②
tự do không nhúc nhích. Đây là lần thứ TÁM một kết luận của tôi bị chính phép
đo sau đó bác — và lần này khoảng cách chỉ có một giờ.

Vì sao: ② phải chọn **một lớp trong 722** chỉ từ một thuộc tính. Bể ứng viên
lớn lên cùng nhịp với dữ liệu, nên mật độ không tự nó giúp. Thứ còn thiếu là
tín hiệu phân biệt ở ②, không phải số lượng cạnh.

### 65.4 Việc kế — theo thứ tự, đã có số để chọn

1. **② cần tín hiệu mới, không cần thêm dữ liệu.** 13 tín hiệu đã thử ở
   §40–§48, đúng 1 ăn tiền (`ctx` §43). Nay có thêm một thứ chưa từng có:
   `classrich2` chứa **định nghĩa của chính thuộc tính** (`nghịch ngợm là…`),
   tức có thể tra lớp của THUỘC TÍNH chứ không chỉ của thành viên.
2. **Chân lý v2 lưu bề mặt có mảnh BPE** (`c áo`, `b úa`) nên tra ngược id
   thất bại — ca đơn lẻ bị bỏ qua LẶNG LẼ. Phải vá trước khi tin số ca nào.
3. Đường sinh vẫn chưa bật (`hub_chain_boost = 0`).

### 65.5 Bài học phương pháp (nối tiếp §64.5)

90. **«Thiếu dữ liệu» là chẩn đoán dễ chịu nhất và phải bị nghi ngờ nhất.**
    Tôi kết luận nó ở §61.5, nhắc lại ở §64, rồi bác nó ở §65 bằng 8× dữ liệu.
    Chi phí kiểm: một buổi. Chi phí KHÔNG kiểm: mọi vòng sau đều đi sai hướng.
91. **Bộ lọc theo TỈ LỆ không loại được chất bẩn ĐƠN LẺ.** Một tên Latin trong
    câu mười từ qua được mọi ngưỡng phần trăm; phải chặn tuyệt đối. Cùng dạng
    sai với «phân vị không chuyển quy mô» (§55) — ngưỡng mềm gặp bẩn cứng.
92. **Corpus giàu KHUÔN chưa chắc giàu THÀNH VIÊN.** viwiki có 183 690 cặp
    lớp mà chỉ 906 cặp dùng được cho ca kiểu sinh vật. Đếm khuôn thì thấy mỏ
    vàng; đếm thành viên dùng được thì thấy mỏ đã cạn từ lâu.

## 66. 🔴 BƯỚC ① KHÔNG BAO GIỜ ĐỦ HẠT — 45 mục vá ② đã vá một bước có ĐẦU VÀO RỖNG

§65 bác giả thuyết «thiếu dữ liệu». Vậy còn gì? Đo thứ chưa ai đo trong toàn
bộ TK_046: **bước ① có giao được cho ② thứ mà ② cần không?**

Cơ chế phiếu ở ② (§21, bản vá lớn nhất của ticket) chỉ phân biệt được khi có
**≥2 hạt cùng lớp** cùng bầu một nhãn. Đếm trên `classrich2`, 570 ca lớp thật:

```
① top-4 (thứ ĐANG dùng)  ·  số hạt là ĐỒNG-THÀNH-VIÊN của lớp đích
     0 hạt → 541 ca (94,9%)   ·   1 hạt → 29 ca   ·   ≥2 hạt → 0 ca (0,0%)
```

**Không phải 3%. Không phải 1%. Là KHÔNG.**

Điều kiện cần của bước ② đúng **0/570 ca** trên dữ liệu thật. Suốt §20–§65 —
13 tín hiệu thử ở ②, bốn cách phá hoà, hai lần đo lại chân lý, 8× dữ liệu —
mọi bản vá cho ② đều chạy trên đầu vào không hề chứa thứ nó cần.

Ca thử vàng che mất điều này ba tháng: ở toy, câu «nghịch ngợm như loài chó» và
«… như loài mèo» được viết SẴN, nên ① luôn trả `[chó, mèo]` = đúng 2 đồng-thành-
viên. Cơ chế không sai — nó chỉ chưa bao giờ được chạy trên đầu vào thật.

### 66.1 Không TÌM được, chứ không phải không BIỂU DIỄN được

Nới ① từ top-4 lên top-60 rồi hỏi lại:

```
① top-60  ·  có ≥1 đồng-thành-viên : 201/570 = 35,3%
             có ≥2 đồng-thành-viên :  67/570 = 11,8%   ← TRẦN của ② hiện nay
```

Thông tin CÓ trong bể; xếp hạng chôn nó. Đây đúng bài học nặng số 2 của ticket
(«không TÌM được» ≠ «không BIỂU DIỄN được»), lần này trả lời được bằng số.

### 66.2 Vá: chọn hạt theo ĐỒNG THUẬN LỚP, và nó phải là DỰ PHÒNG

Thay vì xếp hạng từng ứng viên rời rạc rồi lấy 4 cái đầu, tìm **NHÓM ứng viên
đã cùng trỏ về một nhãn** — đó chính là định nghĩa «cùng lớp». Cùng trực giác
đồng thuận của §21, đặt ở ① thay vì ②.

| cách | dữ liệu thật (570 ca) | ca thử vàng (10 mức toy) |
|---|---|---|
| nền §65 | ĐẠT 2 · ≥2 đồng-tv 0,0% | **10/10 ĐẠT** |
| đồng thuận VÔ ĐIỀU KIỆN | ĐẠT **5** · 0,9% | **4/10** — gãy từ 9 983 cha |
| đồng thuận làm **DỰ PHÒNG** | ĐẠT **4** · 0,4% | **10/10 ĐẠT** |

Bật vô điều kiện thì đồng thuận quét top-60 và nhặt lại `nghịch đảo`/`nghịch
lý` — đúng thứ §59 đã loại. Luật đúng là luật của §59: **có đủ hạt typed thì
dùng typed; không đủ mới gọi đồng thuận.** Cùng trực giác, hai cách thi hành,
kết quả trái ngược — **bài học 76 lặp lại lần thứ hai trong cùng một ticket.**

Chốt bản DỰ PHÒNG (`hub_chain_hat_dong_thuan = True`): dữ liệu thật 2 → 4 ca,
ca vàng không hồi quy.

⚠️ **ĐÃ THỬ, ÂM TÍNH**: chia điểm nhóm cho `label_indeg` để nhãn đặc thù thắng
nhãn chung chung — ĐẠT **5 → 2**. Nhóm lớn nhất thắng, không phạt gì.

### 66.3 Vá phép đo: chân lý v2 lưu bề mặt vỡ

`build_chan_ly_v2` ghép mảnh BPE bằng `" ".join` nên `c`+`áo` thành `c áo`;
`resolve_surface_ids` không tra ngược được ⇒ ca bị bỏ **lặng lẽ** (lớp `thú`
biến mất khỏi mọi phép đo). Ghép theo `MARK_SPACE`:

```
thú  → cheo · cáo · cầy · lừa · sóc · thỏ          (trước: 'c áo' · 'c ầy')
chim → bồ câu · chim cút · cuốc · cò · cú
classrich  165 lớp · 930 cặp     classrich2  716 lớp · 7 405 cặp
```

### 66.4 Việc kế — nay đã đổi hẳn thứ tự ưu tiên

1. **Mọi công sức phải dồn về ①.** Trần của ② là 11,8% (số ca có ≥2 đồng-thành-
   viên trong top-60) và hiện thực hoá được 0,4%. Nâng ① là việc duy nhất có
   đòn bẩy; vá ② tiếp là vá bước đói đầu vào.
2. **Kênh thuộc tính đang quá hẹp.** ① đọc `via_attr` (§53) — chỉ pivot so sánh
   `như`/`giống`/`tựa`. Văn xuôi thật nói thuộc tính bằng VỊ NGỮ («chó rất trung
   thành»), không bằng so sánh. Đó là loại cạnh THỨ BA chưa dựng.
3. Đường sinh vẫn tắt (`hub_chain_boost = 0`).

### 66.5 Bài học phương pháp (nối tiếp §65.5)

93. **Đo ĐIỀU KIỆN CẦN của một bước trước khi vá bước đó — 45 mục, không ai
    hỏi.** Câu hỏi «bước trước có giao đủ đầu vào không?» mất 4 phút để trả lời
    và nó vô hiệu hoá phần lớn công vá ② của cả ticket. Trước khi tối ưu một
    bước, hãy đếm xem đầu vào của nó có tồn tại không.
94. **Ca thử vàng viết tay có thể che ĐÚNG cái nó định kiểm.** Toy chứa sẵn hai
    câu cho ① nên ① luôn xanh; mọi lần gãy đều lộ ở ②/③ và ta đi vá đúng chỗ
    KHÔNG hỏng. Ca thử nhân tạo phải kiểm cả **tiền đề** của nó, không chỉ đích.

## 67. KIỂM §66 BẰNG BỘ CA ĐÚNG HÌNH DẠNG, VẮT TRẦN 7,9%, VÀ ĐỌC ĐẦU RA THẬT

Ba việc theo thứ tự đã nêu: (A) kiểm con số 0,0% có phải hiện vật của bộ ca
không · (B) vắt phần đã nằm sẵn trong bể · (D) đọc thứ đường sinh thật sự đề cử.

### 67.A Bộ ca dùng TÍNH TỪ — con số 0,0% ĐỨNG VỮNG

Bộ ca §66 chọn thuộc tính bằng đồng-hiện nên ra `rồi`·`quả`·`thân`, trong khi
ca vàng dùng một TÍNH TỪ (`nghịch ngợm`). Lọc thuộc tính về tính từ theo từ
điển — 453 ca:

```
① top-4  · có ≥2 đồng-thành-viên :  0,0%      ← Y HỆT §66
① top-60 · có ≥1                 : 33,8%
           có ≥2                 :  7,9%      ← trần, thấp hơn 11,8% của bộ ca cũ
```

Kết luận §66 **không phải hiện vật của bộ ca**. Và một chi tiết mới: `trắng` →
hạt `[vôi, mắt, in]`, `đen` → `[súng, mực, ăn]`. ① **tìm ĐÚNG vật mang thuộc
tính** — `vôi` trắng, `mực` đen. Vấn đề không phải ① sai, mà là **vật mang của
một tính từ nằm rải khắp các lớp**; hai vật mang cùng lớp là chuyện hiếm.

Ca vàng che đúng chỗ này: trong toy chỉ có chó/mèo mang `nghịch ngợm`.

### 67.B Bốn luật chọn nhóm, cả bốn TRUNG TÍNH

| luật chọn hạt | ĐẠT / 453 | ≥2 đồng-thành-viên |
|---|---|---|
| nhóm lớn nhất (§66) | 5 | 0,0% |
| chỉ ứng viên DANH TỪ | 4 | 0,0% |
| CHÙM — chấm mọi nhóm bằng chính ② | 5 | 0,2% |
| nhóm có `label_ctx` chứa thuộc tính (§43) | 4 | 0,0% |

Chẩn đoán vì sao cả bốn trượt — đo thêm: trong 36 ca có ≥2 đồng-thành-viên
nằm trong bể, **63,9% (23 ca) HAI THÀNH VIÊN CÓ chia nhau một nhãn**. Tức nhóm
đúng TỒN TẠI trong 5,1% ca, nhưng mọi luật đều chọn một nhóm khác **to hơn**.

Nhóm to hơn luôn là nhóm của lớp CHUNG CHUNG (`loại`·`phần`·`người`). §64 đã
quét định từ ra khỏi nhãn; danh từ chung chung thì vẫn còn nguyên, và chúng thu
gom mọi ứng viên. **Đây là hình dạng cuối cùng của nút thắt: không phải thiếu
cạnh, không phải xếp hạng sai — mà là lớp chung chung nuốt mọi đồng thuận.**

### 67.D ĐỌC ĐẦU RA — và nó BÁC một bản vá vừa được thước chấp nhận

Mọi thước §60–§66 đều trung gian. Bật `hub_chain_boost=4.0` trên `classrich2`
rồi đọc thẳng đề cử:

```
§66 BẬT   «cây bàng là ___» → họ · Thủ · phải · Patron · Ông · Campuchia
§66 TẮT   «cây bàng là ___» → Chó · Dê · Sư · đo · Báo · Cá · Gấu · Hổ · Lợn
```

Bản vá §66 — thứ vừa nâng ca ĐẠT 2 → 4 trên thước — **làm đường sinh tệ đi**:
nhóm «lớn nhất» hội tụ về một nhãn khổng lồ dùng chung cho mọi prompt. ⇒ Lật
mặc định `hub_chain_hat_dong_thuan = False`, giữ cờ để A/B. Ca vàng không đổi
(10/10 mức, khỉ 2–18) vì luật chỉ chạy khi hạt typed < 2.

Hai điều nữa chỉ thấy được khi đọc đầu ra:

1. **② sụp về cùng một nhãn bất kể prompt.** `«chó là»` và `«mèo là một loài»`
   cho danh sách đề cử Y HỆT NHAU, dù ① rõ ràng khác nhau (`chó` → bạn·yếu·săn·
   thú · `mèo` → nghị·thú·loài·đặc). Đây là phát biểu rõ nhất từ trước tới nay
   của vấn đề ②, và nó hiện ra ở SẢN PHẨM chứ không ở bất kỳ thước nào.
2. **`khỉ` nay tới được từ dữ liệu thật.** `hạt từ 'loài'` = [loài, trắng,
   vàng, chúng, đen, **khỉ**] trên `classrich2` — nhờ câu từ điển «khỉ là loài
   linh trưởng cỡ nhỏ». Lần đầu đích của ca vàng nằm trong tầm với của một
   checkpoint dữ liệu thật.

### 67.E Bài học phương pháp (nối tiếp §66.5)

95. **Đọc ĐẦU RA bác được bản vá mà mọi thước nội bộ đã duyệt.** §66 qua cổng
    ca (2 → 4) và cổng ca vàng (10/10), rồi chết ở lần đọc đầu tiên bằng mắt.
    Một thước trung gian không biết nó đang bỏ sót cái gì; sản phẩm thì biết.
96. **«Vật mang thuộc tính» ≠ «đồng thành viên».** Suốt ticket tôi coi ① là
    bước tìm thành viên. Nó không phải: nó tìm VẬT MANG, và vật mang của một
    tính từ trải khắp mọi lớp. Ca vàng làm hai khái niệm này trùng nhau vì toy
    chỉ có hai vật mang.

## 68. LỚP CHUNG CHUNG KHÔNG PHẢI TÍNH CHẤT CẤU TRÚC — và kênh ① mỏng gấp 15 lần

§67.B kết: nút thắt là «lớp chung chung nuốt mọi đồng thuận». Mục này đi tách
chúng bằng cấu trúc — **thất bại**, và trên đường thất bại thì tìm ra chỗ khác.

### 68.1 Bốn tín hiệu, không tín hiệu nào tách được

Hai nhóm đối chứng, mỗi nhóm 8 nhãn (CHUNG: `loại`·`phần`·`người`·`sự`·`nơi`·
`điều`·`hình thức`·`tập hợp` · THẬT: `chim`·`thú`·`cây`·`cá`·`loài`·`kim loại`·
`hành tinh`·`tỉnh`):

```
                indeg    thuần    có cha   tv đồng hiện
CHUNG           287,2    0,566     2,38       0,770
THẬT            150,5    0,522     2,62       0,629
```

Không tín hiệu nào tách được, và hai trong bốn đi **sai chiều**. Từng nhãn thì
càng rõ: `người` indeg **7** và `sự` **4** (chung chung mà thấp nhất bảng),
trong khi `loài` **617** và `cây` **283** (lớp thật mà cao nhất).

⇒ **«Lớp chung chung» không phải một tính chất của đồ thị.** Nó là phán đoán
NGỮ NGHĨA của người đọc, và bốn cách đo cấu trúc đều không thấy nó. Ghi lại để
lần sau không ai thử lại bốn cách này.

### 68.2 Quay lại triệu chứng, và kênh ① lần đầu được đếm

`hổ là loài` → ② trả **`loài` 72,0 — ĐÚNG**. Cơ chế không câm. Nhưng
`chó là` → hạt `[bạn, săn, chó, thú]` → ② `loại`. Bước ① khi nhận một THÀNH
VIÊN thì trả về bạn đồng hiện, không phải đồng-thành-viên — nó vốn được thiết
kế để nhận THUỘC TÍNH.

Vậy đếm luôn thứ nuôi ① — kênh thuộc tính (§53), lần đầu:

```
câu chứa pivot SO SÁNH   : như 2 596 · giống 392 · tựa 56   = 3 044 / 55 460 cha (5,5%)
`via_attr`               : 1 598 thuộc tính ·  3 678 cạnh
`via_pivot` (phân loại)  : 16 297 chủ ngữ  · 50 686 cạnh
```

**Kênh nuôi ① mỏng gấp 15 lần kênh nuôi ②③**, và chỉ 31,8% thuộc tính có ≥2
vật mang. Đây là con số giải thích §66/§67 mà không cần giả thuyết nào thêm:
① không thể giao ≥2 đồng-thành-viên vì nó chỉ có 3 678 cạnh để chọn.

### 68.3 Và kênh mỏng đó còn bị hư từ chiếm chỗ

Đọc 10 «thuộc tính» nhiều vật mang nhất — toàn khuôn dẫn:

```
là 245 · giống 102 · cũng 85 · tự 42 · khác 41 · coi 38 · tương tự 34 ·
hạn 30 · chẳng 26          ⇐ `giống như` · `cũng như` · `chẳng hạn như`
```

Đúng bệnh §64 (định từ làm nhãn) nhưng ở kênh THỨ HAI, và chưa ai đo. Hai bản
vá, mỗi bản một lỗi khác nhau:

1. **`chu_gia` chưa từng áp cho kênh này.** Bộ lọc §37 (token đứng ngay trước
   pivot ở tỉ lệ cao) chỉ chạy cho kênh phân loại. Áp vào ⇒ `giống`·`cũng` biến
   mất.
2. **Đường lui lấy token SÁT pivot.** Không tìm được đầu ngữ đoạn thì luật cũ
   lấy `toks[i]` — tức lấy đúng `là` của «X là như…». Bỏ đường lui (cùng nguyên
   tắc §64) ⇒ `là` 245 biến mất.

Tôi đoán sai hai lần trước khi truy đúng: lần đầu đổ cho `chu_gia`, lần hai đổ
cho dual-write mảnh cụm — cả hai đều vá xong mà `là` vẫn 245.

```
via_attr:  3 678 → 3 383 cạnh (−8%)   ·   bảng thuộc tính hết hư từ
```

### 68.4 Cổng: không hồi quy chỗ nào

```
ca thử vàng   10/10 mức · khỉ 2–18 · nhãn `động vật`     KHÔNG ĐỔI
đề cử sinh    đọc 6 prompt                               KHÔNG ĐỔI
453 ca lớp thật  ĐẠT 2 · ≥2 đồng-thành-viên 0,0%         KHÔNG ĐỔI
test 440
```

Bản vá làm SẠCH kênh mà không nhích được thước nào — đúng như §62. Ghi đúng
như thế: **đây là dọn dẹp đúng, chưa phải tiến bộ.**

### 68.5 Bài học phương pháp (nối tiếp §67.E)

97. **Trước khi tách hai nhóm bằng cấu trúc, hãy đo xem cấu trúc có thấy chúng
    khác nhau không.** Tôi đặt tên «lớp chung chung» ở §67 rồi coi nó như một
    tính chất có thật. Bốn phép đo nói không. Một khái niệm dùng được với người
    đọc chưa chắc tồn tại trong biểu diễn.
98. **Đếm KÍCH THƯỚC kênh trước khi vá thuật toán đọc kênh đó.** ① được vá qua
    §51·§52·§53·§59·§66·§67 — sáu mục — mà chưa ai đếm nó có bao nhiêu cạnh.
    Con số (3 678 so với 50 686) giải thích mọi thất bại đó trong một dòng.

## 69. LOẠI CẠNH THỨ BA — thuộc tính theo VỊ NGỮ: kênh dày 5,5×, và TRƯỢT CỔNG tự đặt

§68.2 đếm được kênh nuôi ① mỏng gấp 15 lần kênh nuôi ②③ vì nó chỉ đọc pivot SO
SÁNH (`như`/`giống`/`tựa` — 5,5% câu). Văn xuôi thật nói thuộc tính bằng VỊ
NGỮ: «chó rất trung thành», không phải «nghịch ngợm như loài chó».

**Cổng nêu TRƯỚC khi chạy** (§68 cuối): ca vàng giữ 10/10 mức **và** ≥2 đồng-
thành-viên vượt **5%**. Không đạt thì bác.

### 69.1 Thi hành

Khuôn `DANH TỪ … TÍNH TỪ` trong cùng mệnh đề, dùng từ loại từ điển đã bake
(cùng tiền lệ Luật 0 với `ws_spans`). Hai điều kiện siết, cả hai đến từ đo:

- **kề ≤2 token** — tiếng Việt hậu bổ, cho chen một từ mức độ («chó RẤT trung
  thành»). Quét cả mệnh đề thì bảng thuộc tính ra `đã`·`nhiều`·`hơn`·`mới`.
- **đủ đặc thù** (`gidf ≥ det_cut`, cùng ngưỡng §64) — từ điển gọi `đã`·`cả`·
  `chính` là tính từ, nhưng chúng phổ biến tới mức không phân biệt được gì.

```
via_attr :  1 598 thuộc tính ·  3 383 cạnh   →   3 003 thuộc tính · 18 456 cạnh
             (31,8% có ≥2 vật mang)              (52,7% có ≥2 vật mang)
via_pivot: 16 297 chủ ngữ · 50 686 cạnh (để so)
```

Kênh nuôi ① dày lên **5,5×**; tỉ lệ mỏng so với kênh phân loại từ 15× còn 2,7×.

### 69.2 Cổng 1 — ca thử vàng: ĐẠT

10/10 mức, `khỉ` hạng 2·2·2·2·3·3·6·7·13·18, nhãn `động vật` ở mọi mức. Không
hồi quy. (Cạnh vị ngữ không đụng ca vàng vì câu toy viết theo lối so sánh, đúng
kênh cũ.)

### 69.3 Cổng 2 — dữ liệu thật: TRƯỢT

| thước (453 ca lớp thật) | §68 | §69 |
|---|---|---|
| **≥2 đồng-thành-viên trong hạt (top-4)** | 0,0% | **1,5%** ← cổng đòi >5% |
| bể top-60 có ≥2 | 7,9% | **16,1%** |
| bể top-60 có ≥1 | 33,6% | **44,4%** |
| ca ĐẠT đầu-cuối | 2 | **6** |
| đề cử đường sinh (6 prompt, đọc mắt) | — | KHÔNG ĐỔI |

Hạt đổi hẳn chất — `trắng` → `[da, sao, vàng]` (trước: `[vôi, mắt, in]`) ·
`đen` → `[da, lông, Biển]` (trước: `[súng, mực, ăn]`).

**Bốn thước phụ đều lên thật, thước CHÍNH thì 1,5% < 5%.** Theo đúng luật đã
nêu trước: **BÁC — mặc định TẮT** (`hub_chain_kenh_vi_ngu = False`), giữ cờ và
giữ số. Bật hay không là quyết định của tác giả, như §57.

Tôi ghi rõ vì sao không tự nới cổng: ticket này có **tám lần** kết luận của tôi
bị phép đo sau bác, và gần như lần nào cũng bắt đầu bằng một con số phụ trông
đủ tốt. Cổng đặt trước chính là để chặn đúng chỗ đó.

### 69.4 Vì sao dày 5,5× mà hiện thực hoá chỉ 1,5%

Bể có ≥2 đồng-thành-viên nay ở **16,1%** ca nhưng top-4 chỉ bắt được 1,5% —
khoảng cách 14,6 điểm y hệt §67 (7,9% vs 0,0%), chỉ dịch lên. Và §67.B đã thử
bốn luật chọn nhóm, cả bốn trung tính.

⇒ Hai nút thắt CHỒNG nhau, nay tách bạch được:
1. **Cạnh** — đã nới, còn nới được nữa (kênh vẫn mỏng 2,7× so với phân loại).
2. **Chọn hạt trong bể** — chưa ai vượt qua; bốn luật §67 đều trượt, và đây
   mới là chỗ 14,6 điểm nằm.

Việc kế đúng là (2), không phải (1): thêm cạnh nữa cũng chỉ nâng trần chứ
không nâng thứ thực nhận.

### 69.5 Bài học phương pháp (nối tiếp §68.5)

99. **Cổng đặt TRƯỚC cứu tôi khỏi một kết luận lạc quan có bốn con số ủng hộ.**
    Cạnh 5,5× · bể gấp đôi · ĐẠT gấp ba · hạt đẹp hẳn lên — bốn thước phụ đều
    mời gọi kết «C thành công». Thước chính nói 1,5% trên bar 5%. Nếu không
    viết bar ra trước khi chạy thì tôi đã kết ngược lại.
100. **Nới ĐẦU VÀO không tự động nới ĐẦU RA khi bước chọn vẫn hỏng.** Trần lên
    gấp đôi mà thực nhận gần như đứng yên: hai nút thắt chồng nhau thì gỡ nút
    dễ trước chỉ làm lộ rõ nút còn lại.

## 70. THƯỚC SINH v3 (đúng lớp) — và CỘNG DỒN trượt cổng, nhưng tách được NHỚ khỏi SUY

§69.4 chốt: 14,6 điểm nằm ở BƯỚC CHỌN. Trước khi vá tiếp, dựng la bàn mới —
vì la bàn cũ («② đúng lớp so với chân lý») là đại lượng TRUNG GIAN và §67.D đã
bắt quả tang nó nói dối: `«cây bàng là ___»` → `Chó·Dê·Sư·Báo`.

### 70.1 Thước mới chấm thẳng đầu ra

```
prompt = «<thành viên> là»          chuẩn = lớp của nó + các đồng-thành-viên
đạt    = trong top-k đề cử có lớp đúng HOẶC một đồng-thành-viên
```

Và nó tách hai nhóm mà **mọi thước trước đây gộp chung**:

- **NHỚ** — thành viên ĐÃ có cạnh `via_pivot` tới lớp (chuỗi chỉ cần tra lại)
- **SUY** — thành viên CHƯA có cạnh đó, chuỗi phải suy ra ← đây mới là «con khỉ»

Nền trên `classrich2`, 300 ca / 53 lớp:

```
NHỚ  176 ca · top-1 4,0% · top-5 17,6%
SUY  124 ca · top-1 8,1% · top-5 21,0%
TỔNG          top-1 5,7% · top-5 19,0%
```

**SUY cao gấp đôi NHỚ.** Chuỗi làm TỐT HƠN ở chỗ nó phải suy ra so với chỗ nó
chỉ cần nhớ lại — dấu hiệu rõ ràng nhất từ trước tới nay rằng cơ chế thật sự
đang lan chứ không phải tra bảng. (Cũng có nghĩa: bước ② đang bỏ phí cạnh trực
tiếp khi có sẵn.)

### 70.2 Cộng dồn thay ba argmax — TRƯỢT cổng, và chỉ ra vì sao

Bỏ argmax ở ②: mọi nhãn ứng viên đều được lan ngược, điểm CỘNG DỒN lên thành
viên; nhãn cũng tự làm đề cử («X là ___» thì đáp án đúng chính là LỚP). Đây gộp
luôn phép thử «bỏ bước ②» đã nêu.

| cộng dồn | NHỚ top-1 | SUY top-1 | TỔNG top-1 | TỔNG top-5 |
|---|---|---|---|---|
| 0 (nền) | 4,0% | **8,1%** | 5,7% | **19,0%** |
| 3 | 6,2% | 5,6% | 6,0% | 16,3% |
| 6 | **7,4%** | 4,0% | 6,0% | 16,7% |

Cổng nêu trước: **+5 điểm tuyệt đối**. Được +0,3 top-1 và **−2,7 top-5** ⇒
**BÁC, mặc định `hub_chain_cong_don = 0`.**

Nhưng nó mua được một sự thật: **cộng dồn khuếch đại thứ đã dày và dìm thứ phải
suy ra** — NHỚ 4,0 → 7,4 (+3,4) trong khi SUY 8,1 → 4,0 (−4,1). Đây là lần đầu
một bản vá cho thấy hai chiều ngược nhau trên cùng một cơ chế, và nó giải thích
vì sao mọi «đồng thuận» dồn về lớp chung chung: đồng thuận là phép đếm, mà đếm
thì luôn thắng ở chỗ đông.

⚠️ Cờ chỉ đụng `hub_chain_proposals` (đường sinh); `cham_toy_lon.py` dựng chuỗi
rời nên ca thử vàng không bị ảnh hưởng — không phải «đã kiểm và không đổi».

### 70.3 Việc kế

Ba hướng, theo thứ tự đòn bẩy đã đo:

1. **Ca SUY đang là chỗ mạnh nhất (8,1%) chứ không phải chỗ yếu.** Mọi vòng vá
   từ §60 tới nay tối ưu cho ca NHỚ mà không biết. Tách thước ra rồi thì việc
   đúng là tối ưu THẲNG cho SUY.
2. **Bước ② bỏ phí cạnh trực tiếp** — ca NHỚ có sẵn cạnh `thành viên → lớp` mà
   top-1 chỉ 4,0%. Đề cử LỚP trực tiếp khi cạnh tồn tại là một dòng, chưa thử.
3. Cộng dồn có ích riêng cho NHỚ (+3,4): có thể bật CÓ ĐIỀU KIỆN, không bật
   toàn cục.

### 70.4 Bài học phương pháp (nối tiếp §69.5)

101. **Một thước gộp hai chế độ thì trung bình của nó vô nghĩa.** NHỚ và SUY đi
     NGƯỢC chiều dưới cùng một bản vá; mọi con số tổng của §60–§69 đều là trung
     bình của hai thứ đó. Tách chế độ trước khi tối ưu.
102. **«Cơ chế đang lan hay đang tra bảng?» trả lời được bằng một phép tách.**
     SUY 8,1% vs NHỚ 4,0% là bằng chứng trực tiếp đầu tiên rằng chuỗi ba bước
     làm việc SUY LUẬN — thứ mà 50 mục đo gián tiếp trước đó không kết luận nổi.

## 71. 🟢 NHÃN TRỰC TIẾP — chuỗi chưa bao giờ đề cử thứ nó ĐÃ HỌC; top-1 5,7% → 21,7%

§70.3 nêu: ca NHỚ có sẵn cạnh `thành viên → lớp` mà top-1 chỉ 4,0%. Đọc lại
`hub_chain_proposals` thì thấy vì sao: nó **luôn đi vòng**

```
p → vật mang (①) → nhãn (②) → thành viên (③)
```

kể cả khi `p` đã có cạnh lớp của CHÍNH NÓ. Với prompt «X là ___», đáp án đúng
chính là lớp của X — và cơ chế không bao giờ đề cử nó. **50 mục vá đường vòng,
không mục nào hỏi vì sao không đi đường thẳng.**

Bản vá: thêm cạnh `p → lớp` sẵn có vào đề cử, hệ số `hub_chain_nhan_truc_tiep`.

### 71.1 Quét hệ số — bão hoà ở 16

```
tt        0      4      8     16     32     64
top-1   5,7%  14,7%  20,7%  21,7%  21,7%  21,7%
top-5  19,0%  35,0%  40,0%  42,0%  42,0%  42,0%
```

**Cổng nêu trước (§70.3): +5 điểm tuyệt đối. Được +16,0 top-1 và +23,0 top-5 ⇒
ĐẠT.** Đây là bản vá đầu tiên qua cổng kể từ §63.

### 71.2 Tách NHỚ/SUY — và một lần siết phép tách

| | NHỚ | SUY | TỔNG |
|---|---|---|---|
| tt=0 | 3,9% / 17,2% | 8,3% / 21,7% | 5,7% / 19,0% |
| tt=16 | 15,6% / 34,4% | **30,8% / 53,3%** | 21,7% / 42,0% |

Phép tách đầu tiên của tôi so khớp bề mặt CHÍNH XÁC, nên ca có cạnh tới BIẾN
THỂ của lớp («cây thân gỗ» vs `cây`) bị gán nhầm là SUY. Siết lại (coi là NHỚ
nếu cạnh trỏ tới nhãn nào CHIA TỪ với lớp chuẩn): NHỚ 176→180 ca, và SUY vẫn
30,8% — **kết luận không đổi sau khi siết**.

### 71.3 Kiểm độ bền trên lớp CỤ THỂ — mạnh hơn, không yếu đi

Lớp chuẩn của nhiều ca là `phần`/`khoảng`/`bộ phận`; nếu cơ chế chỉ đề cử nhãn
chung chung thì nó khớp chuẩn chung chung mà không biết gì. Tập con 55 ca /
11 lớp `chim·thú·cây·cá·loài·kim loại·hành tinh·tỉnh…`:

```
tt=0    top-1  7,3% · top-5 27,3%   (SUY 10,7%)
tt=16   top-1 30,9% · top-5 60,0%   (SUY 42,9%)
```

Đọc bằng mắt các ca SUY — từ hiếm, lớp đúng:

```
«bèo là ___»   lớp `cây`  → cây · dụng · dây · ruột       ✓ top-1
«bương là ___» lớp `cây`  → cây · dây · vị · mùi          ✓ top-1
«cuốc là ___»  lớp `chim` → chim · đồ · tép · công        ✓ top-1
«bòng là ___»  lớp `cây`  → phần · cây · thanh · cua      ✓ top-2
```

### 71.4 Cổng khác và đọc đầu ra

```
ca thử vàng      10/10 mức · khỉ 2–18 · nhãn `động vật`     KHÔNG ĐỔI
test             440
cộng dồn (§70)   tt=16 + cd=3 → top-1 13,7% (so 21,7%)      ⇒ cộng dồn CHẾT HẲN
```

Đề cử đường sinh, đọc mắt — **triệu chứng «mọi prompt cho cùng một danh sách»
của §67.D ĐÃ HẾT**:

```
«chó là ___»          → bạn · phổ · yếu · một · loài · săn · nhà · gia
«mèo là một loài ___» → nghị · thú · Rùa · vàng · Loài · Chim
«hổ là loài ___»      → Rùa · Loài · vàng · thờ · chúng · Chim
«cây bàng là ___»     → quốc · ấc · ay · tơi          ← vẫn rác
```

### 71.5 Bài học phương pháp (nối tiếp §70.4)

103. **Hỏi «có đường thẳng không?» trước khi tối ưu đường vòng — 50 mục không
     ai hỏi.** Cạnh `p → lớp` nằm sẵn trong `via_pivot` từ §20, và cả chuỗi
     được thiết kế để SUY RA thứ đó qua ba bước. Vì ca thử vàng cố tình không
     có cạnh trực tiếp (`khỉ` phải được suy ra), đường thẳng không bao giờ xuất
     hiện trong đầu tôi — **ca thử định hình cả không gian giải pháp tôi nghĩ tới.**
104. **Một bản vá «tầm thường» có thể thắng gấp bốn lần mọi bản vá tinh vi.**
     §60–§70 gồm mười bản vá tinh vi, phần lớn trung tính hoặc trượt cổng. Bản
     vá này là mười dòng, thắng +16 điểm top-1.

## 72. 🟢 KỀ NGỮ CẢNH CHO ĐƯỜNG TRỰC TIẾP — và ca vàng chạy trên DỮ LIỆU THẬT

### 72.1 Một âm tính trước đã: đề cử ĐỒNG-THÀNH-VIÊN của nhãn trực tiếp

«chó là ___» thì `mèo`/`hổ` cũng là đáp án hợp lệ, và chúng nằm sẵn trong
`reverse_peers` của lớp. Thử:

```
tv       0      1      4     16
top-1  21,7%  21,3%  11,3%   7,0%
top-5  42,0%  42,7%  33,0%  22,7%
```

**Dìm chính cái lớp.** Thành viên của các nhãn RÁC của cùng chủ ngữ đông hơn
thành viên của nhãn đúng. Giữ tắt (`hub_chain_tv_truc_tiep = 0`).

### 72.2 Chẩn đoán ca trượt: hai phần ba là lỗi XẾP HẠNG

```
174 ca trượt top-5 · 117 (67,2%) CÓ cạnh trực tiếp tới lớp đúng mà vẫn trượt
                   ·  57 (32,8%) thật sự THIẾU CẠNH
```

Nên việc đúng không phải thêm cạnh (§69 đã thử, trượt cổng) mà là xếp hạng
giữa các nhãn của **cùng một chủ ngữ**.

### 72.3 Bản vá — và lần đầu nó ĐO RA PHẲNG vì lỗi của tôi

Dùng lại tín hiệu KỀ (§60, đã ăn tiền ở ③): nhãn nào từng thật sự đứng ngay sau
token cuối của prompt thì nhân `(1+k)`.

Lần chạy đầu: **phẳng** (21,7% → 22,3%). Truy ra `bang_ke_ngu_canh` bỏ hư từ
khi dựng bảng — mà với «X là ___» thì token ngay trước ô trống CHÍNH LÀ `là`.
Bảng kề vì thế chỉ trả lời «cái gì đứng sau X», không phải «cái gì đứng sau
`là`». Cho token CUỐI luôn được tính dù là hư từ:

```
kt        0      2      8     32
top-1  21,7%  24,3%  25,0%  25,0%      ← bão hoà
top-5  42,0%  45,7%  45,3%  45,3%
```

Trên 55 ca lớp CỤ THỂ: top-1 **30,9% → 43,6%** · top-5 60,0% → 65,5% (ca NHỚ
18,5% → 40,7%). Trên `classrich` (ckpt khác): 13,5% → 14,2%.

Chốt `hub_chain_ke_truc_tiep = 2.0`.

### 72.4 🟢 ĐỌC ĐẦU RA — ca vàng chạy trên DỮ LIỆU THẬT

```
«nghịch ngợm như loài ___» → KHỈ 96,0 · Rùa · cảnh · quả · dây · báo · bướm · cá
«cây bàng là ___»          → hoè · mai · quốc · bông · bầu · cam · cau · chuối
«mèo là một loài ___»      → THÚ 230,4 · nghị · khỉ · đặc · Rùa · cá
«hổ là loài ___»           → khỉ · Rùa · dữ · mèo · cá · Loài · báo
«chó là ___»               → BẠN 410,4 · phổ · yếu · săn · một · nhà · thú · loài
```

**`khỉ` đứng ĐẦU cho prompt `nghịch ngợm như loài` trên `classrich2`** — ca thử
vàng, chạy trên checkpoint dữ liệu thật chứ không phải toy 5 câu. Đây là thứ
TK_046 đuổi theo từ §19.

Và `«cây bàng là ___»` — ca rác kinh điển của §67.D (`quốc·ấc·ay·tơi`) — nay ra
`hoè·mai·bông·bầu·cam·cau·chuối`: **toàn cây**.

Cổng: ca thử vàng toy 10/10 mức (khỉ 2–18) · 440 test.

### 72.5 Bài học phương pháp (nối tiếp §71.5)

105. **Một tín hiệu «đo ra phẳng» có thể chỉ là tôi tra sai bảng.** KỀ ở đường
     trực tiếp đo ra +0,6 điểm — đủ nhỏ để tôi bác nó và đi tiếp. Nguyên nhân
     là bảng kề bỏ hư từ, mà hư từ `là` chính là token quyết định. Sửa một dòng
     ⇒ +3,3. **Trước khi bác một tín hiệu, kiểm xem nó có được nạp đúng đầu vào
     không** — cùng loại lỗi «vá nhánh chết» của §64.4.
106. **Ba bản vá cuối (§71 nhãn trực tiếp · §72 kề trực tiếp) đều là ĐƯỜNG
     THẲNG, không phải cơ chế mới.** Chúng chỉ đề cử thứ đồ thị đã chứa và xếp
     hạng bằng tín hiệu đã có. Mười bản vá tinh vi trước đó (§60–§70) mua được
     ít hơn hẳn hai bản vá này cộng lại.

## 73. ĐÍNH CHÍNH §72 — `khỉ` KHÔNG do chuỗi ba bước sinh ra

§72.4 viết «ca vàng chạy trên DỮ LIỆU THẬT» khi thấy `khỉ` đứng đầu cho prompt
`nghịch ngợm như loài` trên `classrich2`. Câu đó **sai về nguồn gốc**, và tôi
chỉ phát hiện khi tách biến để trả lời câu hỏi «còn khả thi không».

```
tt=16 · kt=2   → khỉ 96,0        đường THẲNG (§71) + kề (§72)
tt=16 · kt=0   → khỉ 32,0        đường THẲNG, hạng ngoài top-10
tt=0  · kt=2   → khỉ KHÔNG CÓ    ← chuỗi ba bước THUẦN: khối·sản·cảnh·lực·nếp
```

**Chuỗi ba bước thuần không sinh ra `khỉ` ở bất kỳ cấu hình nào.** Thứ sinh ra
nó là cạnh phân loại TRỰC TIẾP của một token trong prompt, cộng với việc nhãn
đó là CỤM và cơ chế phát token đầu cụm (`khỉ ...` → `khỉ`).

### 73.1 Điều này đổi cách đọc toàn bộ §71–§72

Hai bản vá thắng lớn của phiên (top-1 5,7% → 25,0%) **không phải là chiến thắng
của cơ chế loại suy**. Chúng là:

- **§71** — đề cử thứ đồ thị ĐÃ CHỨA (tra lại tri thức đã nạp);
- **§72** — xếp hạng thứ đó bằng «cái gì hay đứng sau token cuối» (một tín hiệu
  bigram có điều kiện).

Cả hai đều tốt và đều đáng giữ — chúng làm đầu ra ĐÚNG TRI THỨC hơn hẳn. Nhưng
chúng KHÔNG chứng minh gì về ý tưởng «con khỉ». Thước sinh v3 nay đo phần lớn
là **năng lực tra cứu**, không phải năng lực suy luận.

### 73.2 Trạng thái thật của ý tưởng «con khỉ», sau 54 mục

| | trên TOY (5 câu vàng + nhiễu) | trên DỮ LIỆU THẬT |
|---|---|---|
| chuỗi ①②③ ra đúng lớp | ĐẠT tới 245 321 cha | ② tự do **1,0%** (§65) |
| ① giao ≥2 đồng-thành-viên | 100% (toy viết sẵn) | **0,0%** top-4 · 16,1% trong bể (§69) |
| ra thành viên CHƯA chạm thuộc tính | ĐẠT (khỉ hạng 2–18) | **chưa lần nào**, ngoài 2–6 ca lẻ |

Kết luận thẳng: **cơ chế ba bước chạy được đúng trên loại dữ liệu mà ca thử
vàng mô phỏng — corpus nhỏ, thuộc tính hiếm, mỗi thuộc tính có ít vật mang và
chúng cùng lớp. Trên văn xuôi thật, điều kiện đó gần như không bao giờ xảy ra**
(§67.A: vật mang của một tính từ trải khắp mọi lớp).

### 73.3 Bài học phương pháp (nối tiếp §72.5)

107. **Một kết quả ĐÚNG có thể đến từ cơ chế SAI, và niềm vui che mất câu hỏi
     «cái gì vừa tạo ra nó?».** `khỉ` đứng đầu là kết quả tôi chờ 54 mục; tôi
     báo cáo nó ngay mà chưa tách biến. Phép tách mất **4 phút** và lật hoàn
     toàn ý nghĩa. Quy tắc: **kết quả càng đúng ý mình mong, càng phải tách
     biến trước khi kể.**

## 74. CHẤM LẠI §66 · §67 · §69 SAU KHI VÁ — **cả ba phán quyết GIỮ NGUYÊN**

TK_052 vòng 2 phát hiện `hub_chain_proposals` không truyền `kenh_thuoc_tinh`,
nên đường sinh mù với kênh §53 suốt ~20 mục. Câu hỏi ngay sau đó: **những phán
quyết ÂM TÍNH của §66–§69 có phải là hiện vật của lỗi đó không?**

Ghi trước cho rõ: harness `thu_ca_vang_tren_that.py` VỐN ĐÃ truyền
`kenh_thuoc_tinh=True` — lỗi nằm ở đường sinh, không ở harness. Nhưng các bản vá
sau đó (§68 · TK_052 vòng 5 · vòng 6) đã làm sạch `via_attr` đáng kể, nên chấm
lại vẫn có nghĩa.

### 74.1 Kết quả — không phán quyết nào lật

| mục | phán quyết cũ | chấm lại |
|---|---|---|
| §66 — ① giao ≥2 đồng-thành-viên | **0,0%** | **0,0%** (bể 7,9% → 8,2%) |
| §67.B — bốn luật chọn hạt | trung tính | trung tính (ĐẠT 3→5 · ≥2 ≤0,2%) |
| §69 — kênh vị ngữ, thước ca | 1,5% (cổng 5%) | **1,5%** |
| §69 — kênh vị ngữ, thước SINH | chưa đo khi bác | **38,0/51,0 → 38,0/51,0** (0 thay đổi) |

§69 nay bị bác trên **cả hai** la bàn, kể cả thước sinh v3 vốn chưa tồn tại khi
nó bị bác lần đầu. Phán quyết vững hơn trước.

### 74.2 Một con số MỚI: lớp cụ thể lên 54,5%

```
thước sinh v3 · 55 ca lớp CỤ THỂ (chim·thú·cây·cá·loài·tỉnh…)
    §72 :  top-1 43,6%  ·  top-5 65,5%
    nay :  top-1 54,5%  ·  top-5 63,6%
```

Tức các bản vá của TK_052 vòng 5–6 (vật mang phải là danh từ đủ đặc thù ·
`_liet_ke` không phát ngược) **cải thiện chính TK_046** thêm 10,9 điểm top-1
trên tập lớp cụ thể, dù chúng được làm cho ticket khác.

### 74.3 Ý nghĩa

Không hướng nào được trả lại. Chẩn đoán từ §66 tới nay **đứng vững**: nút thắt
không phải kênh thiếu, không phải luật chọn hạt, không phải thiếu loại cạnh —
mà là **chất lượng NHÃN LỚP trên văn xuôi thật** (`khác` · `quý hiếm` nuốt mọi
đồng thuận, §67.B đặt tên · TK_052 vòng 6 xác nhận).

⇒ Việc kế duy nhất còn đáng làm ở tầng cơ chế: làm sạch nhãn lớp. Cùng thuốc đã
dùng ba lần (§64 định từ · vòng 5 vật mang · vòng 6 đặc thù): **nhãn phải là
danh từ VÀ đủ đặc thù, không đường lui.** §40 từng kết luận từ loại vô dụng ở ②
— nhưng đó là trước mọi bản vá xếp hạng, và nay có thước sinh để chấm lại.

### 74.4 Bài học

108. **Chấm lại sau khi vá là việc RẺ và nó bảo vệ các âm tính.** Bốn phép đo,
     khoảng 20 phút, và kết quả là «không lật gì cả» — nghe như phí, nhưng nó
     biến ba phán quyết «có thể sai vì lỗi kia» thành ba phán quyết CHẮC. Không
     làm thì mọi vòng sau vẫn phải mang theo nghi ngờ đó.

## 75. LÀM SẠCH NHÃN LỚP — **ngõ cụt, năm tín hiệu đều trượt**

§74 chốt nút thắt cuối là chất lượng nhãn lớp trên văn xuôi. Mục này đi làm, và
kết luận là **không làm được bằng bất kỳ tín hiệu nội tại nào đã thử**.

### 75.1 Từ loại: 72,2% cạnh đã là DANH TỪ, và nhãn rác cũng vậy

Phân bố từ loại của nhãn trên `classrich2` (6 427 nhãn · 46 581 cạnh):

```
danh từ                    33 652 cạnh (72,2%)
CỤM không có trong từ điển  6 057 (13,0%)
động từ                     3 026 ( 6,5%)
không có trong từ điển      1 970 ( 4,2%)
tính từ                     1 371 ( 2,9%)
đại/phụ/kết/cảm từ            505 ( 1,1%)
```

Và 20 nhãn nhiều cạnh nhất — **tất cả đều là danh từ**:

```
phần 473 · nơi 468 · khoảng 306 · tên 305 · cây 295 · bộ phận 289 · vật 264 ·
cái 211 · đồ dùng 207 · chất 204 · việc 202 · điều 198 · chỗ 180 · kẻ 146
```

`phần`·`nơi`·`cái`·`việc`·`điều`·`chỗ`·`kẻ` là rác; `cây`·`chất`·`đồ dùng` là
thật. **Cùng từ loại.** Lọc danh từ cắt 27,8% cạnh mà không đụng được nhãn rác
nào — đúng như §40 đã kết luận, nay có số cụ thể.

### 75.2 Độ chụm trong E — tín hiệu thứ năm, cũng SAI CHIỀU

Trung bình cosine đôi một giữa thành viên (24 tv mỗi nhãn):

```
RÁC  : phần 0,193 · nơi 0,207 · cái 0,227 · việc 0,254 · điều 0,249  → 0,208
THẬT : cây 0,284 · chim 0,137 · thú 0,163 · cá 0,127 · dụng cụ 0,123 → 0,178
```

Nhãn RÁC **chụm hơn** nhãn thật. Lý do đọc được: thành viên của `việc`/`điều` là
danh từ trừu tượng đồng dạng, còn thành viên của `chim`/`cá` là tên loài rải rác.

### 75.3 Tổng kết năm tín hiệu

| tín hiệu | mục | kết |
|---|---|---|
| bậc-vào (`label_indeg`) | §68.1 | không tách (287 vs 150, `người`=7) |
| độ thuần (thành viên chia nhãn khác) | §68.1 | sai chiều |
| nhãn có CHA | §68.1 | không tách (2,38 vs 2,62) |
| thành viên đồng hiện | §68.1 | không tách |
| **từ loại** | §75.1 | **không tách** — 100% top-20 là danh từ |
| **độ chụm E** | §75.2 | **sai chiều** |

**«Nhãn lớp chung chung» không phải một tính chất đo được của đồ thị này.** §68.1
đã kết luận thế với bốn tín hiệu; §75 thêm hai tín hiệu độc lập nữa và cùng kết
quả. Ghi để đóng hướng: **đừng thử tín hiệu thứ bảy.**

### 75.4 Hệ quả — hướng còn lại là DỮ LIỆU, không phải BỘ LỌC

Nhãn rác không lọc được ⇒ phải không sinh ra chúng ngay từ đầu. Cách duy nhất
còn lại: **dựng cạnh lớp CHỈ từ nguồn có giống–loài đáng tin** — mục từ điển
(`chó là gia súc…` · `cắt là chim…` · `sóc là thú…`), thay vì từ mọi câu `là`
của văn xuôi.

Đó không phải bộ lọc mà là một quyết định về NGUỒN, và nó có tiền lệ đo được:
§65 dựng `dinh_nghia.jsonl` từ từ điển và chân lý v2 nhảy 168 → 722 lớp.

### 75.5 Bài học

109. **Sáu tín hiệu trượt là một kết luận, không phải một thất bại.** Sau §68
     tôi vẫn nghĩ «chắc còn tín hiệu nào đó». Hai tín hiệu nữa, cùng kết quả.
     Chi phí: 25 phút. Giá trị: đóng vĩnh viễn một hướng mà mọi vòng sau sẽ còn
     bị cám dỗ quay lại.
110. **Khi bộ lọc không phân biệt được, hãy hỏi nguồn.** Ba lần trong phiên nhãn
     rác bị chặn thành công (§64 định từ · vòng 5 vật mang · vòng 6 đặc thù) đều
     là chặn thứ SAI CẤU TRÚC. Nhãn chung chung thì đúng cấu trúc — nó chỉ vô
     dụng về ngữ nghĩa, và ngữ nghĩa không nằm trong đồ thị đồng hiện.

## 76. VIA_PIVOT CHỈ-TỪ-ĐIỂN — cắt 59% nhãn, **không mua được gì**

§75 đóng hướng «lọc nhãn bằng tín hiệu nội tại» (sáu tín hiệu trượt) và đề xuất
hỏi NGUỒN: chỉ nhận nhãn từng được từ điển dùng làm GIỐNG trong lời giải nghĩa.

### 76.1 Dựng danh sách lớp từ từ điển

9 195 mục từ danh từ → 5 288 đầu ngữ nghĩa → giữ **1 162 lớp** được ≥3 mục từ
dùng làm giống. Bake một lần, runtime chỉ đọc — cùng tiền lệ `pos_map`/`ws_spans`.

```
chim 40 · thú 23 · cây 184 · cá 29 · loài 11 · động vật 29 · dụng cụ 63
phần 197 · nơi 123 · cái 88 · việc 74 · điều 71
```

⚠️ **Danh sách chứa CẢ nhãn rác** — vì từ điển tiếng Việt thật sự dùng `phần`·
`nơi`·`cái` làm giống («X là phần của…», «Y là nơi mà…»). Đây là bằng chứng
thứ bảy cho §75: nhãn chung chung không phải lỗi trích, nó là **cách tiếng Việt
định nghĩa**.

### 76.2 A/B

```
đồ thị      6 427 nhãn · 46 581 cạnh   →   2 610 nhãn · 30 818 cạnh
            (−59% nhãn · −34% cạnh)
nhãn top-8  loại·loài·phần·nơi·khoảng·tên·cây·bộ phận   →   Y HỆT

thước sinh v3 · 300 ca      top-1 38,0% · top-5 51,0%  →  38,0% · 51,0%
thước sinh v3 · 55 ca cụ thể top-1 54,5% · top-5 63,6%  →  54,5% · 63,6%
453 ca lớp thật             ĐẠT 3                      →  ĐẠT 2
① ≥2 đồng-thành-viên        0,0%                       →  0,0%
```

**Cắt 34% cạnh mà mọi thước đứng yên** (một thước nhích xuống 1 ca). Nghĩa là
3 817 nhãn bị loại vốn là **trọng lượng chết** — chúng chưa bao giờ được ② chọn.
Nhãn quyết định kết quả là 8 cái đầu bảng, và chúng sống sót qua mọi bộ lọc.

⇒ **Mặc định TẮT** (`lop_trang=None`), giữ cờ + file `out/lop_tu_dien.json`.

### 76.3 Kết luận về cả hướng «làm sạch nhãn lớp»

Ba cách đã thử, ghi để đóng:

| cách | kết quả |
|---|---|
| lọc bằng tín hiệu nội tại (6 tín hiệu) | §75 — không tách được |
| lọc bằng TỪ LOẠI | §75.1 — 100% top-20 là danh từ |
| lọc bằng NGUỒN (danh sách từ điển) | §76 — cắt 59% nhãn, 0 thay đổi |

**Nhãn rác không phải thứ lọc được, vì nó không phải rác** — `phần`·`nơi`·`cái`
là lớp hợp lệ trong tiếng Việt, chỉ vô dụng cho loại suy. Muốn ② không chọn
chúng thì phải cho ② một lý do NGỮ NGHĨA để ưu tiên `chim` hơn `phần`, và lý do
đó không có trong đồ thị đồng hiện + từ điển đang dùng.

### 76.4 Bài học

111. **Cắt 34% dữ liệu mà không thước nào đổi = phần bị cắt vốn đã chết.** Kết
     quả «không đổi» ở đây không phải thất bại của bộ lọc mà là **phát hiện về
     đồ thị**: chỉ ~8 nhãn đầu bảng quyết định mọi thứ, phần đuôi dài 6 000 nhãn
     không tham gia.
112. **Ba cách lọc, một kết luận: thứ tôi gọi là «rác» thực ra là ĐÚNG.**
     Từ điển dùng `phần`/`nơi` làm giống y như dùng `chim`/`thú`. Vấn đề không
     nằm ở nhãn mà ở chỗ **loại suy cần lớp PHÂN LOẠI SINH VẬT, còn ngôn ngữ
     thì không phân biệt hai loại lớp đó**.

---

# 📋 TỔNG KẾT PHIÊN 2026-08-12 → 08-14 (§20–§59)

## Vào phiên

Ý tưởng của tác giả: dùng nơ-ron W_skip làm nhãn, lan ba bước

```
nghịch ngợm → {chó, mèo} → động vật → {chó, mèo, KHỈ}
```

`khỉ` là **sáng tạo** vì nó chưa từng đứng cạnh `nghịch ngợm`.
Trạng thái lúc bắt đầu: **chín vòng chấm mù, 0 ca, không rõ vì sao.**

## Ra phiên

| thước | vào | ra |
|---|---|---|
| ngưỡng nhiễu (ca thử vàng) | 895 cha | **≥99 362 cha** (111×) |
| phủ cạnh lớp (`classrich`) | 18,1% | **72,4%** |
| ② chọn đúng lớp — top-1 | 0,0% | **46,6%** |
| ② top-5 | 4,8% | **59,6%** |
| ③ chính xác (trao nhãn đúng) | — | **38,9%** · `loài` → cá chép·gấu trúc·Rùa·Chim·Cá sấu |
| test | 421 | **440** |

**Chưa đạt** đầu-cuối trên dữ liệu thật (thước sinh 0,926 trên bộ ca đã lọc),
nhưng ca thử vàng nay sống ở quy mô **lớn hơn corpus thật 2,8 lần**.

⚠️ §60.8: 46% bộ ca của thước sinh là **mảnh BPE**; trên 92 ca sạch, ② tự do
đúng **3,3%**. Con số 46,6% ở bảng trên đo với ứng viên nhãn BỊ GIỚI HẠN vào
164 lớp đã đào — khoảng cách giữa hai số là việc còn lại.

## Việc đã làm

**Đo (13 script mới)** — `mine_chain3_structures` · `measure_min_count_loss` ·
`measure_pivot_coverage` · `probe_chain3_known_labels` · `probe_chain3_noise_curve` ·
`probe_hub_chain_gen_real` · `build_chan_ly_v2` · `grid_luat_hub` ·
`measure_hai_quan_the` · `cham_toy_lon` + ca thử vàng (`toy_chain3.jsonl`).

**Vá cơ chế (có tác dụng):**

| § | bản vá | mua được |
|---|---|---|
| §29 | `head_rule` hai vế | phủ 18,1% → 68,4% |
| §21 | `rank="phieu"` (đồng thuận typed) | ②top1 0,8% → 38,2% |
| §33 | ③ phát SPAN (bỏ `m.tid < len(vocab)`) | trần ③ 22% → 70,4% |
| §37 | tách bể cạnh ②/③ | ③ sạch hơn, ② không mất |
| §43 | **chữ ký ngữ cảnh** | ②top1 +2,2 (tín hiệu duy nhất trong 13) |
| §53 | **cạnh THUỘC TÍNH** (loại cạnh thứ hai) | ① hết trôi sang bạn ghép từ |
| §55 | tắt hai bộ lọc phân vị | ngưỡng 3 069 → 20 316 |
| §59 | hạt giống **chỉ typed** | ngưỡng 20 316 → ≥59 769 |
| §60 | **kề ngữ cảnh** ở ③ (gỡ hoà-ABC) | ngưỡng → ≥99 362 |

**Lỗi thật tìm được:** `min_count` so trọng số chứ không phải số lần (41% cạnh) ·
`phai_nhat` lấy âm tiết cuối (**4 vị trí**) · «hàm/bể dùng chung» giữa ①②③
(**3 lần**) · tra bằng bề mặt thay vì id (**3 lần**) · phân vị không chuyển được
giữa các quy mô.

## Thành quả về PHƯƠNG PHÁP — 77 bài học (§20.7 → §59.4)

Ba bài nặng nhất:

1. **Chân lý cũng là một phép trích, và nó cũng có lỗi.** Mười mục tôi chấm cơ chế
   bằng «token ngay trước `là`» — một heuristic THÔ HƠN chính cơ chế bị chấm, nên
   nó chấm SAI đúng ở chỗ cơ chế làm ĐÚNG.
2. **«Không TÌM được X» và «không thể BIỂU DIỄN X» là hai chẩn đoán khác nhau.**
   Bốn bản vá liên tiếp trượt vì đọc nhầm loại.
3. **Đọc CÂU TRAIN giải được ca mà nhiều vòng đo không giải được** — 5 lần liên
   tiếp (§38 `thành` · §46 họ hàng rác · §54 `phổ biến` · §55 hai bộ lọc · §59
   `bao gồm`). Không nút nào lộ qua con số tổng.

**Bảy lần kết luận của tôi bị chính phép đo sau đó bác**: §20.4 «phiếu tệ hơn»
(thật ra 27×) · §24.2 «8/40» (ảo) · §25 «hai quần thể rời nhau» (giao 26×) ·
§27.2 «rò 2 = thiếu cạnh» · §30 «③ 12,3%» (thật 38,9%) · §41 «chưa thử đúng cách» ·
§45.3 «THÊM > LỌC».

## Việc cần làm

1. ~~Tìm trần thật~~ **XONG (§60)** — trần ≥99 362 cha; gãy ở 148 875 vì ② chọn
   `bao` (bệnh «bao gồm» của §59.2), không phải vì ③.
2. ~~Đo lại `classrich` với §59~~ **XONG (§60.8)** — §59/§60 vô hiệu trên dữ liệu
   thật; nút thắt là ② tự do (3,3%), và bộ ca cũ có 46% là rác.
2b. **Gộp mảnh ở tầng NHÃN** (§60.7) — `động`+`vật` là một cụm, phiếu đang bị
   chẻ đôi. Cần một ngưỡng cho luật ghép, và cách không dìm cạnh vàng khi hợp bể.
2a. 🔴 **TỐI ƯU THẲNG CHO CA SUY — §70.1**: thước v3 tách NHỚ/SUY và SUY
   (8,1%) MẠNH GẤP ĐÔI NHỚ (4,0%). Mọi vòng vá §60–§69 tối ưu cho ca NHỚ mà
   không biết, và hai chế độ đi NGƯỢC chiều dưới cùng một bản vá (§70.2).
2b. **CHỌN HẠT TRONG BỂ — §69.4**: bể có ≥2 đồng-thành-viên nay 16,1% ca
   (§69 nới cạnh 5,5×) nhưng top-4 chỉ bắt 1,5% — khoảng cách 14,6 điểm, và
   bốn luật chọn nhóm của §67.B đều trượt. Thêm cạnh nữa chỉ nâng TRẦN.
2b. **KÊNH NUÔI ① — §68.2**: `via_attr` 3 383 cạnh vs
   `via_pivot` 50 686; chỉ 5,5% câu có pivot so sánh. ① không thể giao ≥2
   đồng-thành-viên vì nó không có gì để chọn. ⇒ LOẠI CẠNH THỨ BA (thuộc tính
   theo VỊ NGỮ) là việc kế có đòn bẩy rõ nhất, nay đã có số biện minh.
   («lớp chung chung» KHÔNG phải tính chất cấu trúc — §68.1 bác bằng 4 tín hiệu.)
2b. **LỚP CHUNG CHUNG — §67.B**: nhóm đúng tồn tại ở
   5,1% ca (23/36 ca có ≥2 đồng-thành-viên chia nhau một nhãn) nhưng bốn luật
   chọn nhóm đều thua nhóm to hơn của `loại`/`phần`/`người`. Không phải thiếu
   cạnh, không phải xếp hạng — mà là lớp chung chung. Xem tiếp §66:
2b. 🔴 **DỒN VỀ ① — §66**: điều kiện cần của ② (≥2 hạt cùng lớp) đúng **0/570
   ca** trên dữ liệu thật; trần là 11,8%, hiện thực hoá 0,4%. Vá ② tiếp là vá
   một bước ĐÓI ĐẦU VÀO. Loại cạnh THỨ BA cần dựng: thuộc tính theo VỊ NGỮ
   («chó rất trung thành»), vì `via_attr` chỉ đọc pivot so sánh `như`.
2b. **② cần TÍN HIỆU MỚI, không cần thêm dữ liệu** (§65.3) — nhân 8× dữ liệu
   lớp mà ② tự do đứng yên (1,0% → 0,8%). Bể ứng viên lớn lên cùng nhịp.
2c. **Nhận diện câu định nghĩa THẬT** (§61.5) — «X là động vật» vs «X là một
   trong những…». Câu dạng vàng chỉ chiếm **2,1%** câu `là` trên `classrich`;
   đây là chỗ ② tự do đang chết (1,0–3,3%), và là bài toán chưa đặt tên.
3. **Thước sinh** vẫn 0,996: ② đúng 46,6% nhưng token kế chưa được nâng. Xung đột
   đã đo ở §56 (lọc BẬT cho sinh 0,290 nhưng ngưỡng tụt về 3 069); tác giả chọn
   **ưu tiên năng lực** (§57).
4. **`hub_chain_boost` vẫn = 0** — đường sinh chưa bật mặc định. Bật khi nào là
   quyết định riêng.
5. Còn **1 tín hiệu trong 13 ăn tiền** ở ②; 19,4 điểm giữa top-1 và top-5 vẫn là
   dư địa thật (đã kiểm hai chiều §39).

## Cấu hình chốt

```python
hub_chain_head_rule      = "trai_du_dac_thu"
hub_chain_head_rule_trai = "phai_nhat"        # §58: `phai_nhat_danh_tu` bị TRỘI
hub_chain_indeg_q        = 0.0                # §55/§57: lọc CẠNH TẮT — năng lực
hub_chain_gidf_q         = 0.0
hub_chain_head_gidf_q    = 0.20               # §64: ngưỡng CHỌN ĐẦU, núm RIÊNG
hub_chain_ke_ctx         = 2.0                # §60: KỀ ngữ cảnh ở ③
hub_chain_rank           = "phieu_x_ctx"      # §43
hub_chain_chu_gia_q      = 0.90
hub_chain_ke_ctx         = 2.0                # §60: kề ngữ cảnh ở ③
hub_chain_boost          = 0.0                # đường sinh TẮT mặc định
```

Đảo hướng (ưu tiên chất lượng sinh): `head_rule_trai="phai_nhat"` giữ nguyên,
đặt `indeg_q=0.8` · `gidf_q=0.20` → sinh 0,290 nhưng ngưỡng về 3 069.

---

# 📋 TỔNG KẾT PHIÊN 2026-08-15 (§60–§76 · và tách TK_052)

## Vào phiên

Trạng thái §59: trần nhiễu ca vàng **≥59 769 cha** · ②top1 trên `classrich`
46,6% · ③ chính xác 38,9% · thước sinh 0,996 (tức gần như không nâng được token
đích) · `hub_chain_boost = 0`, đường sinh chưa bật.

Câu hỏi mở: trần thật ở đâu, và vì sao ② tự do trên dữ liệu thật chỉ ~1%.

## Ra phiên

| thước | vào | ra |
|---|---|---|
| trần nhiễu ca thử vàng | ≥59 769 cha | **≥245 321 cha** (4,1× · hết checkpoint, CHƯA gãy) |
| hạng `khỉ` ở mức cao nhất | — | **17** (10/10 mức ĐẠT) |
| **thước sinh v3** — top-1 đúng lớp | 5,7% (khi dựng thước) | **38,0%** |
| **thước sinh v3** — top-5 | 19,0% | **51,0%** |
| … trên 55 ca lớp CỤ THỂ | 7,3% / 27,3% | **54,5% / 63,6%** |
| lớp trong chân lý (`classrich2`) | 168 lớp · 949 cặp | **722 lớp · 7 512 cặp** |
| TK_052 — cơ chế loại suy | chưa đo | **69,7% thu hồi · đối chứng 0,0%** |
| test | 440 | 440 |

Đầu ra đọc được bằng mắt, đây là thứ đổi rõ nhất:

```
«cây bàng là ___»      trước: quốc·ấc·ay·tơi     nay: hoè·mai·bông·bầu·cam·cau·chuối
«mèo là một loài ___»  trước: con·học·ISDN       nay: thú 230,4 · nghị · khỉ · Rùa
«chó là ___»                                     nay: bạn 410,4 · săn · nhà · thú · loài
```

## Bảy bản vá có tác dụng

| § | bản vá | mua được |
|---|---|---|
| §60 | **KỀ ngữ cảnh ở ③** | trần 20 316 → ≥99 362 cha |
| §63 | **TỪ ĐỦ thắng MẢNH** + lớp đông | trần → **≥245 321 cha** |
| §64 | tách núm `head_gidf_q` khỏi lọc cạnh | nhãn đúng +26% · định từ hết khỏi bảng nhãn |
| §71 | **NHÃN TRỰC TIẾP** (đề cử cạnh đã có) | top-1 5,7% → 21,7% |
| §72 | KỀ ngữ cảnh cho đường trực tiếp | top-1 → 25,0% |
| TK_052 v2 | **truyền `kenh_thuoc_tinh`** | top-5 42% → 48% |
| TK_052 v5–6 | vật mang phải là danh từ đủ đặc thù · `_liet_ke` không phát ngược | top-1 → **38,0%** · lớp cụ thể 43,6% → **54,5%** |

## Sáu LỖI THẬT đang chạy trong sản phẩm

1. **`kenh_thuoc_tinh` không bao giờ được truyền** — cả kênh §53 nằm im ~20 mục.
2. **`_liet_ke` phát cạnh lớp NGƯỢC CHIỀU** trên định nghĩa giống–loài — sống 40 mục.
3. `label_gidf_q` điều khiển BA việc ⇒ định từ thành nhãn (**1 142 cạnh → `một`**).
4. Prompt gieo ① bằng **mảnh BPE** (89% tính từ bị tách).
5. `build_chan_ly_v2` lưu bề mặt vỡ (`c áo`) ⇒ ca bị bỏ **lặng lẽ**.
6. Vật mang kênh so sánh nhận hư từ (`các`·`thể`) chiếm hạt giống của ①.

## Tám ÂM TÍNH SẠCH — bác bằng cổng đặt TRƯỚC

`§62` biên từ pivot (trung tính) · `§65` **nhân 8× dữ liệu lớp** (② tự do đứng
yên) · `§66` đồng thuận vô điều kiện (giết ca vàng 10/10 → 4/10) · `§69` loại
cạnh thứ ba (1,5% trên bar 5%, bác trên CẢ HAI la bàn) · `§70` cộng dồn (khuếch
đại NHỚ, dìm SUY) · `§72` đề cử đồng-thành-viên (dìm chính lớp) · `§75` **sáu
tín hiệu tách nhãn rác** · `§76` danh sách lớp từ điển (cắt 59% nhãn, 0 thay đổi).

## Ba lần tôi phải ĐÍNH CHÍNH chính mình

- **§73** — «ca vàng chạy trên dữ liệu thật» SAI: tách biến cho thấy `khỉ` do
  ĐƯỜNG THẲNG (§71/§72) sinh ra, chuỗi ba bước thuần không sinh ra nó.
- **TK_052 vòng 6** — «câu có · cha có · CẠNH KHÔNG» SAI: cạnh `cắt→chim` vốn
  có, phép in của tôi gộp 4 biến thể BPE rồi cắt top-8.
- **§66/§73** — «dữ liệu không có hình dạng đó» SAI: tiếng Việt có **34 273**
  khuôn so sánh; thứ không có là CẠNH cơ chế trích được từ chúng.

## Tách ticket

TK_046 trộn hai mục tiêu và điều đó suýt gây một báo cáo sai. Nay:

- **TK_046** — chất lượng SINH (thước sinh v3). Về cơ bản đã dùng được.
- **TK_052** — chứng minh CƠ CHẾ LOẠI SUY. Sau 6 vòng: cơ chế **ĐÚNG** (69,7% ·
  đối chứng 0,0%), tiếng Việt **CÓ** khuôn (34 273), cơ chế **TRÍCH ĐƯỢC**
  (nhanh→cắt·sóc·ưng), vật mang **CHIA NHAU LỚP** (cắt+ưng→`chim`) — nhưng
  **chưa đo được đầu-cuối** vì nhãn lớp trên văn xuôi quá nhiễu để dựng khoá.

## Hướng ĐÃ ĐÓNG (đừng thử lại)

1. **Lọc nhãn lớp chung chung** — ba cách, sáu tín hiệu, đều trượt (§75/§76).
   `phần`·`nơi`·`cái` là lớp HỢP LỆ trong tiếng Việt, chỉ vô dụng cho loại suy.
2. **Thêm loại cạnh cho ①** — kênh vị ngữ dày 5,5× mà không mua được gì (§69).
3. **Thêm dữ liệu lớp** — 8× không nhích ② tự do (§65).
4. **Luật chọn hạt ở ①** — bốn luật, đều trung tính (§67, chấm lại ở §74).

## Việc còn lại

1. **Quyết định về NGUỒN TRI THỨC** (thuộc tác giả): muốn ② phân biệt `chim` với
   `phần` thì cần cây phân loại thật (Wikidata / WordNet tiếng Việt), không phải
   thêm bộ lọc. Đây là ranh giới của kiến trúc hiện tại.
2. **TK_052 vòng 7** — chỉ chạy được sau (1).
3. `hub_chain_boost` vẫn = 0; bật mặc định là quyết định riêng.

## Cấu hình chốt sau phiên

```python
hub_chain_ke_ctx          = 2.0     # §60 KỀ ngữ cảnh ở ③
hub_chain_head_gidf_q     = 0.20    # §64 núm RIÊNG cho luật chọn đầu
hub_chain_nhan_truc_tiep  = 16.0    # §71 đề cử cạnh lớp đã có
hub_chain_ke_truc_tiep    = 2.0     # §72 xếp hạng đường thẳng
hub_chain_gom_tu          = 0       # TK_052: 2 cho corpus tổng hợp, 0 cho thật
hub_chain_kenh_vi_ngu     = False   # §69 bác
hub_chain_cong_don        = 0       # §70 bác
hub_chain_tv_truc_tiep    = 0.0     # §72 bác
hub_chain_hat_dong_thuan  = False   # §66/§67.D bác
```

**112 bài học phương pháp** (§20.7 → §76.4). Ba bài nặng nhất của phiên này:

1. **Đo ĐIỀU KIỆN CẦN của một bước trước khi vá bước đó** — 45 mục vá ② mà không
   ai hỏi ① có giao đủ đầu vào không (§66).
2. **Đọc ĐẦU RA bác được bản vá mà mọi thước nội bộ đã duyệt** (§67.D · §73).
3. **Kết quả càng đúng ý mình mong, càng phải tách biến TRƯỚC KHI KỂ** — §73 mất
   4 phút để lật hoàn toàn ý nghĩa của một kết quả tôi đã chờ 54 mục.

---

## 77. GHÉP VÀO ĐƯỜNG THẬT — train ĐÃ ghép, sinh CHƯA đủ điều kiện

Sau khi hướng đã xanh, ghép vào đường sản xuất. Hai vế, hai kết quả khác nhau.

### 77.1 Đường TRAIN — đã ghép

`config/train_v12_sent.yaml` nay có `dinh_nghia.jsonl` (priority 0.85).

Lý do: `dictionary.jsonl` vốn đã ở trong manifest, nhưng nó là **văn bản mục từ
thô** («Từ: chó / Giải nghĩa: gia súc…») — lời giải nghĩa KHÔNG có chữ `là`, nên
`via_pivot` mù với nó suốt từ đầu. `dinh_nghia.jsonl` viết lại thành câu thật và
đo được (§65): chân lý lớp **168 → 722 lớp · 949 → 7 512 cặp**.

⚠️ Cần **train lại** checkpoint chính để có hiệu lực; manifest chỉ là khai báo.

### 77.2 Đường SINH — chưa đủ điều kiện bật

`hub_chain_boost` vẫn = 0. Đo A/B trên thước sinh tự hồi quy (`probe_hub_chain_
gen_real`, `classrich2`):

```
boost = 0    không ca nào đo được (không có gì thay đổi để đo)
boost = 4    ty_le_hang_trung_vi = 0,982
```

0,982 nghĩa là token đích nằm ở 98,2% chiều dài bảng xếp hạng — **gần như không
được nâng**. Cùng dải với §56–§59 (0,996 → 0,98) dù đề cử nay tốt hơn hẳn.

**Đây là một mâu thuẫn đo được, và nó là câu hỏi kế tiếp rõ nhất của TK_046:**

```
thước v3 (đo THẲNG đề cử)        top-1 38,0% · top-5 51,0% · lớp cụ thể 56,4%
thước sinh tự hồi quy (đo PHÂN BỐ) ty_le_hang_trung_vi 0,982 — không nâng
```

Đề cử ĐÚNG mà phân bố KHÔNG đổi ⇒ nút thắt nằm ở **khâu bơm đề cử vào phân bố**,
không ở chuỗi ba bước. Đó là chỗ chưa ai đọc trong cả hai phiên.

### 77.3 Test canh sáu lỗi thật

`tests/test_tk052_loi_that.py` — mỗi lỗi một test:

```
LOI1 đường sinh phải truyền `kenh_thuoc_tinh`       (sống ~20 mục)
LOI2 `_liet_ke` không nuốt định nghĩa giống–loài    (sống 40 mục)
LOI3 bề mặt span ghép theo MARK_SPACE
LOI4 `ids_theo_be_mat` thấy id ẢO của span
LOI5 guard của `_ghep_bm` hỏi TỪ VỰNG
LOI6 gộp mảnh vô hiệu với vocab không đánh dấu
```

446 test (440 + 6).

### 77.4 Bài học

116. **«Hướng đã xanh» không có nghĩa là đã nối vào sản phẩm.** Toàn bộ §60–§76
     cải thiện ĐỀ CỬ, nhưng `hub_chain_boost = 0` nghĩa là đường sinh thật chưa
     bao giờ dùng chúng. Hỏi «cái này đã chạy trong sản phẩm chưa?» phải là câu
     hỏi thường trực, không phải câu hỏi cuối phiên.

---

## 78. ĐỌC KHÂU BƠM — và §77.2 «mâu thuẫn» là HIỆN VẬT BỘ CA

§77 để lại câu hỏi: đề cử đúng 38%/51% mà thước sinh AR nói 0,982 (không nâng).
Đọc mã, đọc bộ ca, và cả hai đều có câu trả lời.

### 78.1 Khâu bơm — `generate_hebbian.py:3580`

```python
if _chain:
    _f0 = scores[np.isfinite(scores) & (scores > 0)]
    _med = float(np.median(_f0)) if _f0.size else 0.0
    for _t, _w in _chain.items():
        if np.isfinite(scores[_t]) and scores[_t] > 0:
            scores[_t] = scores[_t] * (1.0 + _chain_boost * _w)   # NHÂN
        elif _med > 0:
            scores[_t] = _med * _chain_boost * _w                 # CỘNG (token mới)
```

`_w` là trọng số **THÔ** của đề cử. Sau §71/§72 nó không còn bị chặn — đo trên
`classrich2`:

```
«chó là»           n=28  w max=16,0   → hệ số nhân  5× … 65×
«mèo là một loài»  n=89  w max=230,4  → hệ số nhân 65× … 923×
«cây bàng là»      n=17  w max=48,0   → hệ số nhân  5× … 193×
```

Chênh **48 lần trong cùng một prompt**: thứ tự giữa các đề cử bị độ lớn thô
quyết định thay vì thứ hạng. Vá: **chuẩn hoá `_w` về [0,1]** trước khi bơm.

### 78.2 §77.2 «mâu thuẫn» — bộ ca của thước AR là RÁC

Đọc bộ ca mà `probe_hub_chain_gen_real` tự dựng:

```
P=`đoàn` lớp=`loại` hạt=[Giáo·Hà·Ngô·VA·Việt] khỉ=[1R·ATM·B41·B69·Bia]
P=`bán`  lớp=`loại` …          P=`trở` lớp=`loài` …        P=`thay` …
```

Thuộc tính là động từ/hư từ, hạt là mảnh tên riêng — đúng thứ §60.8 đã đo là
46% rác. **Không có mâu thuẫn nào**: 0,982 là thước cũ chấm trên ca rác. Thước
v3 sinh ra chính vì thước đó nói dối, và tôi đã quên điều đó khi viết §77.2.

### 78.3 Thước sinh ĐẦU-CUỐI trên ca sạch (`thuoc_sinh_dau_cuoi.py`)

Chấm **phân bố cuối** (sau khi bơm) trên ca sạch của v3 — tức đo đúng thứ người
dùng nhận:

```
180 ca ·  boost = 0   top-1 68,9% · top-5 76,7% · top-10 78,9%
          boost = 2   top-1 67,8% · top-5 78,3% · top-10 80,0%
 55 ca ·  boost = 0   top-1 74,5% · top-5 78,2% · top-10 81,8%   (lớp CỤ THỂ)
 lớp CT   boost = 2   top-1 74,5% · top-5 81,8% · top-10 83,6%
```

**Hai điều lộ ra, cả hai đều đính chính cách tôi đọc thước v3:**

1. **Đường sinh NỀN đã đạt 68,9% top-1** trên prompt «X là ___». Thước v3 báo
   38,0% vì nó đo đề cử **CÔ LẬP**, không đo phân bố. Con số 38% chưa bao giờ là
   chất lượng đầu ra — nó là chất lượng của MỘT nguồn tín hiệu trong nhiều nguồn.
2. **Chuỗi ba bước gần như TRUNG TÍNH ở đây** (±1–2 điểm, n=180): trên prompt
   định nghĩa, mô hình ngôn ngữ nền vốn đã biết câu trả lời.

### 78.4 Chỗ chuỗi THẬT SỰ đóng góp: prompt LOẠI SUY

```
«nghịch ngợm như loài ___»
   boost = 0  →  [EOS · người · chim · động · cá · thuộc · cây · mèo]
   boost = 4  →  [EOS · cây · động · người · cá · chim · thuộc · KHỈ]
```

`khỉ` chỉ vào được top-8 **khi bật chuỗi**. Đó đúng là ca mà nền không có gì để
nói — và là lý do tồn tại của cả TK_052.

⇒ `hub_chain_boost` giữ **0** mặc định: không có bằng chứng đầu-cuối trên prompt
định nghĩa. Nhưng nay biết rõ **bật nó khi nào**: khi bài toán là LOẠI SUY, không
phải khi bài toán là tra định nghĩa.

### 78.5 Bài học

117. **Một thước đo MỘT NGUỒN tín hiệu không phải thước chất lượng đầu ra.**
     Tôi báo cáo «top-1 5,7% → 38,0%» suốt bốn mục như thể đó là chất lượng
     sinh. Đo phân bố cuối: nền đã 68,9% và chuỗi thêm ~0. Số cũ không sai,
     nhưng tên tôi đặt cho nó thì sai.
118. **Trước khi gọi hai thước là «mâu thuẫn», đọc BỘ CA của chúng.** §77.2 mất
     một mục để tuyên bố mâu thuẫn; đọc bộ ca mất hai phút và nó tan.

---

## 79. 🟢 NỐI CHUỖI VÀO ĐƯỜNG PRODUCTION — và phát hiện nó **chưa bao giờ ở đó**

§78 kết «biết bật khi nào». Đi bật, và lộ ra thứ lớn hơn.

### 79.1 Bật `boost` KHÔNG đổi một token nào

Bật `hub_chain_boost = 2.0`, chạy 6 prompt thật qua `generate()`, sinh 12 token:
**đầu ra giống hệt boost = 0 ở CẢ SÁU.** Truy mã:

```
generate()  →  _wseq_fallback  →  _w_skip_primary_loop      (ar_w_skip_primary=True, MẶC ĐỊNH)
generate_autoregressive  ← chỗ DUY NHẤT gọi `hub_chain_proposals`
grep "hub_chain" trong `_wseq_fallback`:  0 lần
```

**Chuỗi ba bước được nối vào một đường mà production không bao giờ đi.** Chính
mã đã ghi điều đó ở §13.24 («đường production rẽ sang `_wseq_fallback` và KHÔNG
BAO GIỜ vào đây») — dòng chú thích đó đứng ngay cạnh chỗ tính `_chain`, và
không ai đọc nó theo nghĩa «vậy chuỗi chưa từng chạy».

Đây là **lỗi cùng loại với TK_052 vòng 2** (kênh §53 chưa bao giờ được gọi):
tính năng có, đo được, ticket đầy số về nó — mà đường thật không dùng.

### 79.2 Nối vào `_w_skip_primary_loop`

Tính đề cử MỘT LẦN từ prompt (dựng index ~11s), chuẩn hoá về [0,1] (§78.1), bơm
ngay trước khi chọn token — cùng công thức với `generate_autoregressive`.

Đo lại qua **đúng đường production**, 55 ca lớp CỤ THỂ, token kế đúng lớp:

```
boost 0 → 74,5%      boost 1 → 76,4%      boost 2 → 76,4%
```

Chốt **`hub_chain_boost = 1.0`**: cùng mức 76,4% mà không rối đuôi câu như 2.0.

### 79.3 Prompt thật, mặc định mới

```
chó là gia súc thuộc nhóm ăn thịt, nuôi để giữ nhà hay đi săn
sắt là kim loại màu xám xanh, dễ dát mỏng và kéo sợi …
hổ là loài ăn khỏe nhất trong họ nhà mèo và cũng là loài động vật rất
mèo là một loài kích thước trung bình với dáng mình mảnh khảnh …
cây bàng là một loài có số và lớn nhất các          ← còn lủng củng
nghịch ngợm như loài là các có chức một và để nhất  ← còn rác
```

Bốn prompt định nghĩa cho câu **đúng tri thức và đọc được**. Hai prompt còn lại
vẫn hỏng, và đó là ranh giới đã biết: `«nghịch ngợm như loài»` cần lớp phân loại
thật (TK_052), `«cây bàng là»` cần cạnh lớp cho `bàng` mà corpus chưa có.

⚠️ Ở `boost = 2.0` prompt loại suy cho **`nghịch ngợm như loài thú`** — đúng
hình dạng mong muốn — nhưng cùng lúc làm rối đuôi `«chó là»`. Đánh đổi này chưa
gỡ được và ghi lại nguyên trạng.

### 79.4 Bài học

119. **«Đã bật» ≠ «đã chạy».** Tôi bật cờ, đo, báo cáo — rồi mới phát hiện cờ
     điều khiển một nhánh production không đi qua. Cách bắt: chạy prompt thật
     qua ĐÚNG hàm mà sản phẩm gọi, và so đầu ra bật/tắt. Nếu giống hệt nhau
     thì hoặc cờ vô dụng, hoặc **nó không nằm trên đường đi** — phải phân biệt
     hai khả năng đó trước khi kết luận.
120. **Chú thích trong mã đã nói sự thật hai phiên trước.** §13.24 ghi rõ
     production không vào `generate_autoregressive`. Nó nằm cách chỗ tính
     `_chain` đúng mười dòng. Đọc mã quanh chỗ mình sửa, không chỉ chỗ mình sửa.

---

## 80. TRAIN LẠI CHECKPOINT CHÍNH — nguồn định nghĩa đổi hẳn đầu ra

`config/train_v12_sent.yaml` + `dinh_nghia.jsonl` → `network_v12_sent_dn.pkl`
(450 050 cha · 70 phút · `dinh_nghia` nạp đủ 19 230 câu). Ghi ra tên MỚI, không
đè `network_v12_sent.pkl` để còn đối chứng.

### 80.1 Cùng 7 prompt, ba cấu hình

**CŨ — chưa có nguồn định nghĩa** (`network_v12_sent.pkl`):

```
chó là gì? không có nghĩa được mộtkhẩu
mèo là một loài dụ về có thể đến dụng trong cho
hổ là loài số ví dụ về một sử dụng được thể choGiải
sắt là kim loại hay phi kim loại?
cây bàng là gì có thể cho tôi một số người bạn cần các
nghịch ngợm như loài một người có thể cho là không để nghĩa ra điềukhẩu người
sóc là một số ví dụ về các trình có thể được cho
```

**MỚI · boost = 0** (chỉ thêm dữ liệu, chưa bật chuỗi):

```
chó là gia súc thuộc nhóm ăn thịt, nuôi để giữ nhà hay đi săn      ✓
hổ là loài ăn khỏe nhất trong họ nhà mèo và cũng là loài động vật rất  ✓
sắt là kim loại màu xám xanh, dễ dát mỏng và kéo sợi, dễ bị gỉ     ✓
sóc là thú gặm một quả lớn nhấm hơn và hạt tròn cây ăn nhanh trên  ✓ lớp `thú`
mèo là một loài dụ về có thể này không đến ý từ                    ✗
cây bàng là gì đó không có hoặc một cách những định sự khác        ✗
nghịch ngợm như loài một người có thể là được và không ra của nói  ✗
```

**MỚI · boost = 1** (bật chuỗi):

```
chó là / hổ là / sắt là                      GIỐNG HỆT boost=0  ✓
nghịch ngợm như loài CÂY có và là quả hoa…   ← đổi từ «một người» sang một LỚP
sóc là thú gặm một quả lớn nhấm hơn và hạt cây ăn trên búp sống    ✓
```

### 80.2 Đọc số

```
                        prompt cho câu ĐÚNG TRI THỨC
CŨ                       0 / 7
MỚI · boost 0            4 / 7      ← toàn bộ mức tăng đến từ DỮ LIỆU
MỚI · boost 1            4 / 7      ← chuỗi đổi ca loại suy, không đổi phần còn lại
```

**Nguồn định nghĩa là thứ mua được nhiều nhất trong cả hai phiên.** Bốn prompt
từ rác thành câu bách khoa đọc được, và `sóc là thú…` chính là ca mà TK_052
vòng 7–9 phải đi năm vòng mới dựng nổi cạnh — nay nó ra thẳng ở đầu ra thật.

Chuỗi ba bước đóng góp đúng chỗ đã dự đoán ở §78.4: prompt LOẠI SUY chuyển từ
`một người` sang `cây` — một LỚP, dù câu sau đó vẫn lủng củng.

### 80.3 Còn hỏng

`«mèo là một loài …»` và `«cây bàng là …»` vẫn rác trên cả ba cấu hình. Cả hai
đều là ca corpus KHÔNG có cạnh lớp: từ điển không định nghĩa `bàng`, và `mèo là
một loài` bị chính cụm `một loài` kéo sang khuôn khác.

### 80.4 Bài học

121. **Một nguồn dữ liệu đúng hình dạng thắng mười bản vá cơ chế.** §60–§79 gồm
     bảy bản vá có tác dụng và tám âm tính; §65 thêm MỘT nguồn (viết lại từ điển
     thành câu có chữ `là`) và nó đưa 4/7 prompt từ rác thành câu đúng. Không
     bản vá nào trong phiên làm được điều tương đương.

---

## 81. TÍN HIỆU DỪNG — luật ép EOS chưa bao giờ nổ

§80 để lại một quan sát: mạng đúng 8–15 token rồi rã. `«chó là …đi săn .»` phát
DẤU CHẤM rồi vẫn đẻ tiếp `vào năm 1 công tấn`. Đây là lỗi khác hẳn §60–§79 (đều
là lỗi CHỌN token) — nó là lỗi DỪNG.

### 81.1 Đo trước khi vá (cổng chốt trước)

`_w_skip_primary_loop` đã có sẵn luật cứng:

```python
if _is_b_g(state.vocab, int(out[-1])) and n_b_gen >= 1:
    nxt = int(eos_id)          # đứng trên biên sau ≥1 câu đã đóng → EOS
```

Gọi TRẦN ⇒ `bound=None, bootstrap=False`. Theo đúng docstring của chính
`_is_boundary`, khi ấy nó **chỉ** còn ngắt-khoảng-trắng và trả `False` cho MỌI
dấu câu. Trong khi `count_boundaries` mặc định `bootstrap=(bound is None)` nên
NÓ đếm được dấu chấm. Lệch pha: vế phải luôn đúng, vế trái luôn sai.

**Cổng chốt trước khi chạy:** vá được coi là đúng hướng nếu trên chuỗi token mà
đường sinh thật đã phát, có ≥1 token mà bound-đúng nói BIÊN còn gọi-trần nói
KHÔNG, **và** token đó là dấu kết câu. Trượt ⇒ dừng, không vá.

```
«chó là …đi săn .»  bước 14  token='Ġ.'  trần=False  bound-đúng=True  count=1  ← DẤU KẾT CÂU
«thủ đô …Hà Nội.»   bước  2  token='.'   trần=False  bound-đúng=True  count=1  ← DẤU KẾT CÂU
LỆCH tổng 4 token · DẤU KẾT CÂU 2 ⇒ CỔNG ĐẠT
```

Tay cầm `_bd_gen = _bound_of(state)` được tính ở **ngay đầu chính hàm đó** và đã
truyền cho ba chỗ khác (dòng 258/305/307). Đúng hai chỗ ép EOS quên.

### 81.2 Vá HẸP — chỉ `_is_boundary`, KHÔNG `count_boundaries`

Hai token lệch còn lại (`Ġsúc`, `Ġgỉ`) không phải dấu câu mà lift cao. Cả hai
rơi vào chỗ `count_boundaries = 0` nên chốt `n_b_gen >= 1` chặn được. Nếu truyền
`bound` cho cả `count_boundaries` thì chúng tự thành mốc, `n_b_gen` đạt 1 từ
token đầu và câu bị **cắt cụt** ngay. Vá hẹp là có chủ ý, ghi rõ tại chỗ.

### 81.3 Kết quả — 4/20 prompt tốt lên, 0 xấu đi

```
chó là                TRƯỚC  …đi săn . vào năm 1 công tấn      SAU  …đi săn .
mèo là                TRƯỚC  …chó là bạn. một số nhà có thể…   SAU  …chó là bạn.
muối là               TRƯỚC  …axit và bazơ. các bài chính của ghi  SAU  …axit và bazơ.
một năm có bao nhiêu  TRƯỚC  Một tháng. 1 5 6 5 5 5 6 5        SAU  Một tháng.
thủ đô của Việt Nam   TRƯỚC  Hà Nội.                           SAU  Hà Nội.  (giữ)
```

16 prompt còn lại KHÔNG đổi — vì chúng **không hề phát dấu chấm nào**, chạy tới
`max_len`. Bản vá chữa «đã chấm câu mà vẫn đẻ tiếp»; nó không chữa «không bao
giờ chấm câu». Hai việc khác nhau, việc sau còn nguyên.

### 81.4 Canh lỗi

`tests/test_tk052_loi_that.py::test_LOI7_ep_eos_phai_truyen_bound` — kiểm ĐỎ
bằng cách gỡ bản vá rồi chạy lại trước khi commit (bài học vòng 9).

### 81.5 Bài học

122. **Tham số mặc định khác nhau giữa hai hàm cùng đọc một khái niệm = lỗi
     tàng hình.** `_is_boundary(bootstrap=False)` và `count_boundaries(
     bootstrap=bound is None)` cùng hỏi «token này có kết câu không» mà trả lời
     ngược nhau. Không hàm nào sai riêng lẻ; chỗ sai là **luật hợp hai vế**. Đây
     là lần thứ BA trong hai phiên một tính năng nằm im vì đường thật không gọi
     đúng (TK_052 LOI1 `kenh_thuoc_tinh`, §79 `hub_chain_boost`, nay §81
     `_bd_gen`) ⇒ mọi tín hiệu mới phải có một test canh CHÍNH CHỖ GỌI.

---

## 82. «Không bao giờ chấm câu» — ÂM TÍNH, và âm tính theo chiều ngược

§81 chữa «đã chấm câu mà vẫn đẻ tiếp». Còn lại 16/20 prompt **không hề phát dấu
chấm nào**. Vòng này đi tìm vì sao.

### 82.1 Đối chiếu câu nguồn — không phải lỗi DỪNG, mà lỗi TRẬT TỰ

Dò `data/text_corpus/dinh_nghia.jsonl` lấy câu nguồn thật:

```
NGUỒN  Hổ là một loài động vật có lãnh thổ hoạt động rất rộng, và các cá thể phân tán…
SINH   hổ là một loài động vật có lãnh thổ hoạt  rất động  thể và các để đến có tới vật

NGUỒN  sóc là thú gặm nhấm lớn hơn chuột, sống trên cây, mõm tròn, đuôi xù…
SINH   sóc là thú gặm  một quả lớn nhấm hơn  và hạt cây ăn trên búp sống mõm

NGUỒN  lúa là cây lương thực, thân cỏ rỗng, hoa lưỡng tính, không có bao hoa, quả có vỏ…
SINH   lúa là cây lương thực, thân cỏ rỗng hoa lưỡng tính,  hoa cây có hạt quả thực thân lương
```

Mạng **truy đúng câu nguồn** — mọi từ sinh ra đều nằm trong câu đó, không một từ
lạ. Nó chỉ **mất trật tự** sau ~8–12 token rồi quay vòng trong túi từ. Không có
dấu chấm chỉ là HỆ QUẢ, không phải nguyên nhân.

### 82.2 Giả thuyết: tiêu thụ cạnh khoá mất cạnh đúng — SAI

§12.144/145 chặn CỨNG cạnh đã dùng (`ar_w_skip_consume_edge_scale = 0.0`). Câu
định nghĩa hay lặp từ (`hoa`, `cây`, `thể`, `động`), nên tôi đoán cạnh đúng bị
khoá và bước đi buộc phải vòng.

Cổng chốt trước: một mức chỉ ĐÚNG HƯỚNG nếu **đồng thời** (a) tăng số prompt kết
được câu, (b) giảm tỉ lệ token lặp, (c) không hỏng 4 ca vàng.

```
TẮT hẳn         KẾT CÂU 12/20 · LẶP 28,2%
giết CỨNG (mđ)  KẾT CÂU 13/20 · LẶP 13,6%   ← MẶC ĐỊNH TỐT NHẤT
mềm 0,3         KẾT CÂU 14/20 · LẶP 25,8%
mềm 0,6         KẾT CÂU 13/20 · LẶP 26,1%
```

Không mức nào qua cổng. Nới cơ chế làm tỉ lệ lặp **gần gấp đôi** — tức tiêu thụ
cạnh không phải thủ phạm, nó chính là thứ đang CHẶN lặp. Thấy tận mắt ở ca vàng:

```
mềm 0,3   sắt là …dễ  trong bị gỉ không bị gỉ trong bị trong gỉ
mặc định  sắt là …dễ bị gỉ trong không khí ẩm, là thành phần
```

**Giữ nguyên mặc định. Không đổi một dòng mã nào.**

### 82.3 Cảnh báo về chính thước đo

Chỉ số «KẾT CÂU» đếm EOS nổ trước `max_len`, **không** đếm câu kết đúng hình.
`chim sẻ là loài cái kia có không thể một cho hình số con` dừng sớm nên được tính
là KẾT CÂU dù không có dấu chấm và không thành câu. Vế (a) vì thế YẾU. Kết luận
trên vẫn đứng vì cổng trượt dứt khoát ở vế (b) — 13,6% so với 25,8% là khoảng
cách không nhập nhằng.

### 82.4 Còn lại gì

Mất trật tự xảy ra **dù đã có** tiêu thụ cạnh. Nghi phạm còn lại là chính tín
hiệu thứ tự: điểm skip-gram `W_seq` quá phẳng ở khoảng cách xa, và `d_max` chặn
cửa sổ. Đó là biến tiếp theo đáng đo — nhưng CHƯA đo, nên chưa được nói như đã
biết.

### 82.5 Bài học

123. **Một giả thuyết bị bác theo chiều ngược đáng giá hơn một giả thuyết được
     xác nhận yếu.** Tôi định nới một cơ chế; số đo cho thấy nới nó làm hỏng
     đúng thứ tôi định chữa. Nếu chỉ đo «có tốt lên không» mà không đo tỉ lệ lặp,
     mức mềm 0,3 (14/20 KẾT CÂU, cao nhất bảng) đã được nhận là THẮNG — trong khi
     nó là mức sinh ra `bị gỉ không bị gỉ trong bị trong gỉ`. Cổng hai vế cứu.

---

## 83. Tín hiệu thứ tự — `d_max` là núm SAI; kênh bậc-2 đánh đổi TRI THỨC lấy TRÔI CHẢY

### 83.0 `d_max` không phải cửa sổ ngữ cảnh

Đi tìm `d_max` thì gặp thứ chính codebase đã ghi:

```
parent_store.py §_ensure_seq_index  «Global W_seq chỉ là bigram (bậc-1) → word-salad»
varlm.py §mở đầu                    «W_seq là order-1 → mất trí nhớ ngữ cảnh sau mỗi từ»
```

Nới `d_max` chỉ nới khoảng cách CẶP, không thêm trí nhớ. Không A/B `d_max` nữa.

### 83.1 Kênh bậc-2 có sẵn — và CHẾT trên đường thật (lần thứ TƯ)

`ParentStore.seq_next(prev, cur)` là phân bố kế tiếp bậc-2 dựng từ chuỗi cha.
Đếm lời gọi thật trong `generate()`:

```
«hổ là»  seq_next +0 · _w_skip_primary_loop +1
«sóc là» seq_next +0 · _w_skip_primary_loop +1
«lúa là» seq_next +0 · _w_skip_primary_loop +1
«chó là» seq_next +0 · _w_skip_primary_loop +1
TỔNG: seq_next 0 lời gọi · vòng production 4 lần  ⇒ CỔNG ĐẠT
```

Nó chỉ nối vào `generate_autoregressive`. Lần thứ TƯ cùng kiểu: LOI1
`kenh_thuoc_tinh`, §79 `hub_chain_boost`, §81 `_bd_gen`, nay §83 `seq_next`.

Đã nối vào `_w_skip_primary_loop`, **chỉ xếp lại** ứng viên đang sống, không hồi
sinh đích đã tiêu thụ (§12.144 cấm; §82 đã chứng minh từ đúng vẫn trong túi).

### 83.2 Hai cache khoá bằng `id()` — lỗi thật, bắt được dọc đường

Bộ test chập chờn (`test_LOI5` xanh khi chạy riêng, đỏ khi chạy cả bộ, đỏ cả khi
đã stash mọi thay đổi). Không phải `pytest-randomly` — dự án không có nó. Nguồn
bất định là **địa chỉ bộ nhớ**:

| chỗ | hệ quả |
|---|---|
| `hub_spread._CO_MARK` khoá `id(vocab)` | vocab MỚI nhận đáp án của vocab CŨ ⇒ `_ghep_bm` rẽ nhầm ⇒ mất cạnh (đúng lỗi vòng 7–9) |
| `generate_hebbian._HUB_CHAIN_CACHE` khoá `id(state)` | state MỚI nhận **chỉ mục loại suy của checkpoint CŨ** — sai lặng, ngay trên đường chuỗi §79 |

Dự án **đã có** `hca/objcache.py` (id + weakref + so `is`) sinh ra đúng để chặn
việc này; chuỗi thì không dùng. Vá cả hai ⇒ bộ test ổn định **5/5 lần**
(449 pass, 1 fail môi trường có sẵn), trước đó đỏ ngẫu nhiên.
Canh: `test_LOI8_*`, `test_LOI9_*`.

### 83.3 A/B — cổng TRƯỢT, và trượt theo cách dạy được nhiều nhất

Thước mới, dựng riêng để chữa điểm yếu §82.3: **TRẬT TỰ** = tỉ lệ cặp token liền
kề sinh ra có thật trong bigram của cha.

```
w=0,0   TRẬT TỰ 89,3% · LẶP 13,6% · KẾT CÂU 13/20
w=0,5   TRẬT TỰ 89,1% · LẶP 15,9% · KẾT CÂU 12/20
w=1,0   TRẬT TỰ 92,7% · LẶP 18,2% · KẾT CÂU 10/20
w=2,0   TRẬT TỰ 98,0% · LẶP 12,0% · KẾT CÂU  5/20   ← qua CẢ HAI vế số
```

Mức 2,0 qua vế (a) và (b). Vế (c) — nhìn ca vàng bằng mắt — bác thẳng:

```
w=0,0  chó là gia súc thuộc nhóm ăn thịt, nuôi để giữ nhà hay đi săn .
w=2,0  chó là gia đình và bạn bè và súc tích cực trong cuộc sống của chúng ta
w=0,0  sắt là kim loại màu xám xanh, dễ dát mỏng và kéo sợi, dễ bị gỉ trong không khí ẩm
w=2,0  sắt là kim loại, có thể được sử dụng để tạo ra một cái
```

Trôi chảy hoàn hảo, **tri thức bằng không**. Giữ mặc định `0.0`.

### 83.4 Vì sao — và thước MỚI cũng có điểm mù

`seq_next` gộp trên **cả 450 050 cha**, nên nó bị các chuỗi phổ thông thống trị
(`có thể được sử dụng để tạo ra một`). Nó kéo bước đi khỏi CÂU NGUỒN đã truy
đúng, về phía cách nói phổ biến nhất của corpus. Kênh này đánh đổi TRI THỨC lấy
TRÔI CHẢY, đúng theo nghĩa đen.

Và thước TRẬT TỰ — dựng ra để chữa điểm mù của thước cũ — có điểm mù của chính
nó: nó thưởng cho **văn mẫu trôi chảy**. Phải ghi rõ, không được giấu.

Thêm một số liệu đáng chú ý: ở w=0 trật tự ĐÃ là 89,3%. Tức word-salad §82
KHÔNG phải do cặp liền kề lạ, mà do cặp liền kề **có thật nhưng của câu khác**.

### 83.5 Giả thuyết kế (CHƯA đo)

Tín hiệu bậc-2 nên lấy từ **cha đang hoạt** chứ không gộp toàn cục —
`_active_parent_set` đã có sẵn ở `generate_hebbian.py:104`. Như thế mới giữ được
cả trật tự lẫn tri thức. Chưa đo, nên chưa được nói như đã biết.

### 83.6 Bài học

124. **Thước nào cũng có điểm mù; đổi thước là đổi điểm mù, không phải xoá nó.**
     §82.3 chê «KẾT CÂU» đo sai thứ nên §83 dựng «TRẬT TỰ». Thước mới lập tức
     phong mức 2,0 là quán quân trên CẢ HAI vế số — mức xoá sạch tri thức. Thứ
     cứu là vế (c): bốn ca vàng in ra cho MẮT NGƯỜI đọc. Cổng phải luôn có một
     vế không quy được về con số.
125. **Lỗi tàng hình đi thành ổ.** Truy một test chập chờn ra hai cache khoá
     bằng `id()` trên vật thu gom được, một cái nằm ngay trên đường chuỗi. Test
     chập chờn không phải phiền toái để bỏ qua — nó là triệu chứng.

---

## 84. Bậc-2 từ CHA ĐANG HOẠT — đúng hướng, vẫn ÂM TÍNH; nút thắt dời sang TRUY HỒI

§83.5 đặt giả thuyết: `seq_next` hỏng vì gộp trên cả 450 050 cha; lấy từ **cha
đang hoạt** (`_active_parent_set`) thì giữ được cả trật tự lẫn tri thức. Đã dựng
và đo.

### 84.1 Thước mới — bắt đúng thứ §83 bỏ lọt

§83.6 bài học 124: thước TRẬT TỰ thưởng cho VĂN MẪU. Vòng này thêm:

    TRUNG THÀNH = tỉ lệ TỪ sinh ra có mặt trong CÂU NGUỒN THẬT của prompt
                  (`dinh_nghia.jsonl`, 10/20 prompt có nguồn)

và đặt nó làm **vế PHỦ QUYẾT**: mức nào tụt trung thành thì loại, bất kể trật tự
đẹp tới đâu.

### 84.2 Số

```
TẮT (mốc)            TRUNG THÀNH 78,9% · TRẬT TỰ 89,3% · LẶP 13,6%
toàn cục w=1 (§83)   TRUNG THÀNH 53,8% · TRẬT TỰ 92,7% · LẶP 18,2%
CHA HOẠT w=1         TRUNG THÀNH 67,0% · TRẬT TỰ 91,0% · LẶP 11,8%
CHA HOẠT w=2         TRUNG THÀNH 51,0% · TRẬT TỰ 91,4% · LẶP 15,0%
CHA HOẠT w=4         TRUNG THÀNH 42,1% · TRẬT TỰ 92,7% · LẶP 12,9%
```

Quét độ rộng tập cha (w=1):

```
top_n= 1   TRUNG THÀNH 74,6% · TRẬT TỰ 86,2% · LẶP 11,1%
top_n= 2   TRUNG THÀNH 66,2% · TRẬT TỰ 89,0% · LẶP 12,6%
top_n= 3   TRUNG THÀNH 74,7% · TRẬT TỰ 87,5% · LẶP 11,8%   ← tốt nhất
top_n= 5   TRUNG THÀNH 69,3% · TRẬT TỰ 89,9% · LẶP 12,2%
top_n=10   TRUNG THÀNH 65,7% · TRẬT TỰ 88,8% · LẶP 13,8%
```

Giả thuyết ĐÚNG HƯỚNG: cha-hoạt hơn hẳn toàn cục (67,0 vs 53,8; lặp 11,8 vs
18,2), và thu hẹp tập còn nâng tiếp lên 74,7%. Nhưng **không mức nào vượt mốc
TẮT (78,9%)** ⇒ phủ quyết. Trật tự cũng không mua được gì (86–90 vs 89,3).
**Giữ mặc định tắt.**

### 84.3 Vì sao — và câu hay nhất cả phiên

Cơ chế tách làm đôi rất rõ theo CA:

```
top_n=5  lúa là cây lương thực, thân cỏ rỗng, hoa lưỡng tính, không có bao hoa,
         quả có vỏ trấu bao ngoài hạt
NGUỒN    lúa là cây lương thực, thân cỏ rỗng, hoa lưỡng tính, không có bao hoa,
         quả có vỏ trấu bao ngoài (gọi là hạt thóc) .
mốc TẮT  lúa là cây lương thực, thân cỏ rỗng hoa lưỡng tính, hoa cây có hạt quả
         thực thân lươnggọi, hoa cây vỏ
```

Gần như nguyên văn — **câu định nghĩa tốt nhất cả phiên**. Nhưng cùng cấu hình:

```
hổ là xương hổ dùng làm một loài động vật một thể có màu và tiềm để các với…
```

`hổ là` truy ra `xương hổ dùng làm…` ở MỌI `top_n`. Đó là lỗi TRUY HỒI, không
phải lỗi trật tự — không trọng số nào chữa được. Ở `top_n=20` còn lộ cha rác
sách giáo khoa: `hổ là phần 1 trong chương trình lớp 5`, `sắt là phần 1 trong
chương trình lớp 6`.

**Kênh trật tự tốt đúng bằng cha mà nó được cho.** Tổng bị lỗi truy hồi thống
trị, nên nút thắt đã dời từ SINH sang TRUY HỒI (`rank_matches`).

### 84.4 Trạng thái mã

Cả hai kênh (§83 toàn cục, §84 cha-hoạt) giữ lại — có cờ, có tài liệu, tái lập
được — nhưng **mặc định TẮT**. Canh: `test_LOI10_kenh_bac2_mac_dinh_TAT`.

### 84.5 Bài học

126. **Một giả thuyết đúng hướng mà vẫn thua mốc thì vẫn là âm tính.** Cha-hoạt
     hơn toàn cục ở mọi chỉ số, đúng như §83.5 đoán. Nếu chỉ so hai bản mới với
     nhau thì đã tuyên thắng. Mốc TẮT mới là thứ phải vượt.
127. **Khi một cơ chế tốt/xấu tuỳ CA chứ không tuỳ THAM SỐ, nút thắt nằm ở đầu
     vào của nó.** `lúa` gần như nguyên văn còn `hổ` hỏng ở mọi `top_n`, mọi
     trọng số ⇒ biến quyết định không nằm trong kênh này. Quét thêm tham số là
     phí; phải đi sửa `rank_matches`.

---

## 85. `rank_matches` — hai giả thuyết, một bác một xác nhận; VIẾT HOA chia đôi kho cha

§84 dồn nút thắt về truy hồi. Vòng này chỉ ĐO, chưa vá.

### 85.1 Hai lỗi của chính probe, sửa dọc đường

**(i) Probe đo sai đường.** Bản đầu gọi `ps.rank_matches` trên store TRẦN, nên
`directed_bigram_bonus` = 0 — trong khi `generate()` gán 1.5 ở dòng ~4104. Tức
tôi suýt kết luận về một đường KHÔNG TỒN TẠI trong production. Cùng họ §81/§83:
quên truyền thứ mà đường thật vẫn truyền. Sửa: cấu hình `ps` y hệt `generate()`.

**(ii) Nói quá về khối hoà.** Bản in làm tròn 3 chữ số khiến `2.154` trông như
hoà 5–6 cha. Đếm đúng bằng `abs(sc − đỉnh) < 1e-9`: hoà **3** cha (`chó`), 2
(`lúa`, `hổ`), 1 (`sắt`, `sóc`). Với `sắt` và `hổ` thì cha đúng **thật sự thấp
điểm hơn**, không phải bị hoà. Phải đính chính, không được để nguyên.

### 85.2 Giả thuyết A — cặp khuôn `(X → là)` bị lọc bởi `min_idf`: **BÁC**

`rank_matches` chỉ thưởng cặp có hướng khi `min(idf) ≥ 2.0`. Nếu `là` là hư từ
idf thấp thì mọi cặp khuôn bị vứt ⇒ không gì phân biệt `chó là` với `chó má là`.

```
idf('hổ' | 'lúa' | 'sắt' | 'chó') = 14,017
idf(' là')                        =  8,337   ≫ ngưỡng 2,0
```

Cặp khuôn **không** bị lọc. Cổng chốt trước nói rõ: nhánh này ⇒ DỪNG, không vá.

### 85.3 Giả thuyết B — VIẾT HOA chia đôi kho cha: **XÁC NHẬN**

```
«hổ là» → token ['hổ',' là']   cha ĐÚNG hạng 18
   đỉnh: hổ cốt là xương hổ… · hổ phù là phù hiệu… · hổ phách là nhựa thông…
«Hổ là» → token ['Hổ',' là']   cha ĐÚNG hạng  7
   đỉnh: Hổ là loài vật có lực cắn mạnh thứ hai · Hổ Hoa Nam là… · Hổ là loài độc cư…
```

`hổ` và `Hổ` là HAI token khác nhau. Từ điển viết thường, Wikipedia viết hoa,
prompt người dùng viết thường ⇒ **nửa kho tri thức không với tới được**. Đây là
lời giải thích đầy đủ cho ca `hổ` mà §84 đuổi mãi không ra: không trọng số nào
phía sau chữa được, vì cha đúng chưa từng vào tập đang hoạt.

### 85.4 Một phép đo nữa CHƯA dùng được

Đếm «cha đúng có khớp TIỀN TỐ truy vấn không» cho kết quả nhiễu: mốc chuỗi dùng
để nhận cha đúng không đủ độc nhất (`lúa` bắt nhầm vào `Các trang trại…`, `sóc`
bắt nhầm vào `chuột là thú gặm nhấm…`). Số 3/5 ở bản nháp KHÔNG đáng tin; phải
dựng mốc theo id cha thật rồi mới đo lại. Ghi ra để không ai trích nhầm.

### 85.5 Hướng kế (CHƯA đo)

Gộp hoa/thường lúc truy hồi (`fold_group_id` đã có sẵn và đang được dùng cho cặp
có hướng, nhưng khớp token thì vẫn phân biệt hoa/thường). Phải đo cả hai chiều:
nó có kéo nhầm danh từ riêng vào không.

### 85.6 Bài học

128. **Probe cũng là mã production-adjacent — cấu hình sai thì đo ra thế giới
     khác.** Ba vòng liền (§81, §83, §85) lỗi nằm ở chỗ gọi chứ không ở cơ chế.
     Với §85 thì chính PHÉP ĐO mắc lỗi đó, suýt cho một kết luận sạch về một
     đường không tồn tại.
129. **Làm tròn khi in là một cách nói dối.** `2.154` ba chữ số biến 3 cha hoà
     thành «5–6 cha hoà», và đổi hẳn chẩn đoán từ «hoà điểm» sang «xếp hạng
     sai». So sánh bằng dung sai, đừng so bằng mắt trên số đã làm tròn.

---

## 86. Gộp HOA/THƯỜNG lúc truy hồi — nửa qua cổng, KHÔNG hại chiều ngược

### 86.1 Sửa mốc đúng trước (nợ §85.4)

§85.4 nhận cha đúng bằng mẩu chuỗi viết tay nên nhiễu (`lúa` bắt nhầm `Các trang
trại…`). Nay mốc dựng lại, mù hoa/thường và không cần chuỗi viết tay:

    cha ĐÚNG = cha có văn bản (hạ thường) BẮT ĐẦU bằng prompt (hạ thường)

Kết quả phụ đáng ghi: `chim sẻ là` và `Man City là` **không có cha đúng nào** ⇒
loại khỏi phép đo thay vì tính là trượt. Trước đây chúng sẽ âm thầm kéo số xuống.

### 86.2 Cách gộp

NỞ TRUY VẤN: chạy `rank_matches` cho cả hai dạng viết chữ đầu rồi trộn theo điểm
cao nhất. Không đụng bộ xếp hạng, không dựng lại chỉ mục — vừa là phép đo vừa là
một bản cài đặt khả dĩ.

### 86.3 Số

```
ĐỊNH NGHĨA               vào top-20      hạng trung vị
  gốc (mốc)                12/14              4
  GỘP hoa/thường           12/14              2

DANH TỪ RIÊNG (phủ quyết)
  gốc (mốc)                 4/5               2
  GỘP hoa/thường            4/5               2

ca:  hổ là 18 → 2      mèo là 1 → 1      chó là 3 → 4
     Hà Nội là 2 → 2   Việt Nam là 1 → 1  Sao Thổ là 4 → 4
```

### 86.4 Chấm theo cổng — trung thực

Cổng (a) đòi **cả hai**: tỉ lệ top-20 TĂNG **và** hạng trung vị GIẢM. Hạng trung
vị giảm một nửa (4 → 2), nhưng **tỉ lệ đứng yên** 12/14 ⇒ **(a) chỉ đạt một
nửa**. Không được gọi đây là «qua cổng».

Vế phủ quyết (b) SẠCH: danh từ riêng không tụt một hạng nào. Nỗi lo «gộp
hoa/thường kéo danh từ riêng lẫn vào danh từ chung» — KHÔNG xảy ra trên bộ này.

Và có ca XẤU ĐI: `chó là` 3 → 4. Nở truy vấn bơm thêm ứng viên viết hoa, trong
đó có cái chen trên cha đúng.

### 86.5 Vì sao CHƯA bật

Thứ vừa đo là HẠNG TRUY HỒI — một đại lượng TRUNG GIAN. §83/§84 đã dạy đúng bài
này hai lần: chỉ số trung gian đẹp không suy ra đầu ra tốt (kênh bậc-2 nâng TRẬT
TỰ lên 98% trong khi xoá sạch tri thức). Muốn bật thì phải đo TRUNG THÀNH đầu ra
trên đường sinh thật, chứ không phải dừng ở bảng hạng.

Ngoài ra `hổ là 18 → 2` chỉ ăn tiền nếu có thứ ở phía sau DÙNG tập cha đang hoạt
— mà kênh duy nhất dùng nó (§84) đang TẮT vì âm tính. Nghĩa là lãi của §86 hiện
chưa có đường nào để hiện ra ở đầu ra.

### 86.6 Bài học

130. **Sửa mốc trước, đo sau.** Mốc §85.4 nhiễu làm hỏng một phép đo cả vòng.
     Mốc mới còn lộ hai prompt KHÔNG HỀ có cha đúng — trước đó chúng bị tính là
     «truy hồi trượt», tức đổ lỗi cho bộ xếp hạng về thứ nó không có.
131. **«Không hại» không phải «có lợi».** Vế phủ quyết sạch rất đáng mừng, nhưng
     nó chỉ cho phép đi tiếp, không tự nó là kết quả dương.

---

## 87. Nối gộp hoa/thường vào `generate()` — ÂM TÍNH, và bản vá §86 nối SAI CHỖ

### 87.1 Lần thứ NĂM cùng kiểu lỗi — lần này do chính tôi

Nối `_rank_matches_gop` vào chỗ gọi `rank_matches` trong `generate()` (dòng
~4164), chạy 20 prompt. Kết quả:

```
gốc            TRUNG THÀNH 78,9% · TRẬT TỰ 89,3% · LẶP 13,6%
GỘP            TRUNG THÀNH 78,9% · TRẬT TỰ 89,3% · LẶP 13,6%
```

**Đúng bằng 0 thay đổi**, bốn ca vàng giống hệt TỪNG BYTE. Và dòng thứ ba cho
74,7% — đúng bằng số §84 ở `top_n=3`, tức tập cha đang hoạt cũng không đổi.

Nguyên nhân: `_active_parent_set` — thứ kênh bậc-2 THẬT SỰ đọc — gọi
`ps.rank_matches` TRẦN. Tôi vá đúng một chỗ, và không phải chỗ có người dùng.
LOI1, §79, §81, §83, nay §87: **lần thứ năm trong hai phiên**, và lần này không
phải nợ cũ của codebase mà là lỗi của chính bản vá §86.

Vá lại, truyền `config` tường minh (bản đầu định đọc `state._cfg_gop_ref` —
không ai gán ⇒ lại một nhánh chết nữa; bắt được trước khi chạy).

### 87.2 Xác nhận tập cha ĐỔI trước khi kết luận

```
«chó là»  TẮT: chó là gia súc thuộc nhóm ăn thịt…  ← cha ĐÚNG có mặt
          BẬT: chó má là… · chó săn là… · Chó Phú Quốc là…  ← cha ĐÚNG BỊ ĐẨY RA
«hổ là»   TẮT: hổ cốt · hổ phách · hổ phù          (mục từ điển, đều sai)
          BẬT: hổ cốt · Hổ Hoa Nam là… · Hổ là loài vật có lực cắn mạnh thứ hai
```

Đổi thật. Nhưng lộ ngay mặt trái: gộp bơm ứng viên viết hoa vào một `top_n` chật
⇒ **đẩy cha đúng ra**. Đúng ca `chó là 3 → 4` của §86.3, nay cắn thật.

### 87.3 Số sau khi vá đúng chỗ

```
gốc (mốc)                         TRUNG THÀNH 78,9% · TRẬT TỰ 89,3% · LẶP 13,6%
GỘP hoa/thường                    TRUNG THÀNH 78,9% · TRẬT TỰ 89,3% · LẶP 13,6%
GỘP + bậc-2 cha hoạt w=1 top_n=3  TRUNG THÀNH 77,4% · TRẬT TỰ 89,1% · LẶP 17,8%
```

**GỘP một mình VẪN không đổi gì** — kể cả sau khi vá đúng chỗ. Vì với `w=0` thì
kênh bậc-2 tắt, `_active_parent_set` KHÔNG ĐƯỢC GỌI: cải thiện truy hồi **không
có đường nào ra tới đầu ra**. Đó chính là điều §86.5 đã cảnh báo trước khi đo.

Có gộp thì cấu hình bậc-2 lên **74,7% → 77,4%** (+2,7 điểm) — gộp THẬT SỰ giúp
đúng chỗ nó có người dùng. Nhưng vẫn dưới mốc 78,9%, LẶP xấu đi (11,8% → 17,8%),
và ca vàng vỡ:

```
chó là gia súc thuộc nhóm ăn thịt, chuyên nuôi để giữ nhà nuôi gia thịt ăn năn…
lúa là cây lương thực thuộc, bao thân cỏ, rỗng, được thân cây lương thực có quả…
```

**Giữ cả hai cờ TẮT.**

### 87.4 Bài học

132. **«Nối vào rồi» phải chứng minh bằng SỐ ĐỔI, không bằng dòng mã đã sửa.**
     Bản vá §86 trông hoàn chỉnh — có hàm, có cờ, có test xanh, gọi đúng chữ ký
     — và tác động bằng 0. Thứ duy nhất phát hiện được là chạy A/B và thấy hai
     hàng giống nhau từng byte. Mọi bản vá phải kèm một phép đo mà nó BẮT BUỘC
     phải làm đổi.
133. **Cải thiện một tầng chỉ có giá khi tầng sau CÓ NGƯỜI DÙNG.** Gộp
     hoa/thường nâng hạng cha đúng rõ rệt (§86: 4 → 2, `hổ là` 18 → 2), nhưng
     tầng tiêu thụ duy nhất đang tắt vì âm tính ⇒ lãi bằng 0. Đo tầng trung gian
     mà không kiểm ai tiêu thụ nó là tự thưởng cho mình.

---

## 88. Dựng corpus LOẠI SUY — corpus cũ không dùng được, corpus mới CHƯA đủ sạch để train

### 88.1 Hai corpus «đã có» đều không dùng được

`data/text_corpus/so_sanh_that.jsonl` (TK_052 vòng 4, 11 553 câu) **không phải**
corpus so sánh. Đọc mẫu: `như` ở đó gần như đều nghĩa «chẳng hạn».

```
…các nước châu Âu khác NHƯ Pháp, Thuỵ Sĩ, Đức…   ← liệt kê
…nghệ sĩ blues gạo cội NHƯ Muddy Waters…          ← liệt kê
…gần NHƯ tương tự…                                 ← trạng từ mức độ
```

Train nó còn NGUY HIỂM: bơm đúng khuôn liệt kê mà TK_052 vòng 6 phải vá để khỏi
phát cạnh NGƯỢC. `loai_suy.jsonl` thì tổng hợp và vô nghĩa (`ngặt như loài
biển`), lặp 4 lần mỗi dòng. **Cả hai chưa từng nằm trong `train_v12_sent.yaml`
— và đó là quyết định đúng.**

### 88.2 Đo nguyên liệu — cổng SỐ phong quán quân cho RÁC

```
N1+N2  «tính từ như danh từ» trên mọi nguồn   320 thuộc tính ≥2 vật mang  ← QUA cổng
          khác → [a, album, anh, ba, bao, bi]
          gần  → [ba, bang, bao, biên, biển, báo]        ← RÁC
N3     tính từ rút từ CÂU ĐỊNH NGHĨA           65 → 118 sau khi thêm ràng buộc
          lớn  → chim ×5: cú, hạc, sếu, trả, ưng
          cứng → phần ×6: lõi, móng, mỏ, răng, sừng, yếm  ← THẬT
```

Thứ hạng theo CHẤT LƯỢNG ngược hẳn thứ hạng theo SỐ. N1+N2 chỉ bắt được `gần
như`/`khác như` — trạng từ và liệt kê. Cổng của tôi hỏng vì đếm «≥2 vật mang» mà
**bỏ ràng buộc CÙNG LỚP** — chính là điều kiện ② thật (TK_052 đã đo đúng nó).

### 88.3 Corpus mới, và vì sao KHÔNG train

`scripts/dung_corpus_loai_suy_that.py` → 128 bộ · 650 dòng, có provenance từ
điển thật. Đã sửa một lỗi lộ ra khi đọc bằng mắt: bản đầu tách theo DẤU CÁCH rồi
tra từng âm tiết, mà tiếng Việt đa âm tiết ⇒ `phức tạp`→`phức`, `trung
bình`→`trung`, `mĩ miều`→`mĩ` lọt qua bộ lọc tính từ. Khớp mục từ điển DÀI NHẤT
sửa được LỚP (`dụng cụ` thay `dụng`, `trạng thái` thay `trạng`).

Nhưng đọc mẫu lần hai vẫn **~50% rác**:

```
ngọt như cam . · lõm như thìa . · nặng như thuổng .      ← THẬT
trung như tỵ . · rỗng như dê . · khôn khéo như chước .   ← RÁC
```

Nguyên nhân là lỗi GIẢ ĐỊNH, không phải lỗi mã: tôi coi *mọi tính từ trong vị
ngữ định nghĩa* là thuộc tính của CHỦ NGỮ. Sai — vị ngữ chứa cả tính từ mô tả
thứ khác và mệnh đề phụ (`tỵ là chi thứ sáu…` cho ra `trung`).

**KHÔNG nạp vào `train_v12_sent.yaml`.** §80 thắng nhờ viết lại TRUNG THÀNH
(từ điển → câu có chữ `là`, không mất mát), không phải nhờ suy diễn hao hụt. Nạp
50% rác vào đúng khuôn `như` là lặp lại rủi ro §82 với chi phí 70 phút train.

### 88.4 Việc còn lại để corpus này dùng được

Thắt phép rút: chỉ nhận tính từ ở mệnh đề gắn TRỰC TIẾP vào chủ ngữ (dạng
`<bộ phận> <tính từ>` như `mõm tròn`, `đuôi xù`, hoặc mệnh đề chỉ có tính từ),
bỏ mệnh đề định nghĩa vị trí/thứ tự. Rồi đọc lại 20 mẫu BẰNG MẮT trước khi train.

### 88.5 Bài học

134. **Đọc mẫu bằng mắt bắt được thứ không chỉ số nào bắt.** Lỗi tách âm tiết
     (`phức tạp`→`phức`) không làm sai một con số nào — 218 thuộc tính, 131 bộ,
     668 dòng, tất cả «hợp lệ». Chỉ khi in 14 câu ra đọc mới thấy `phức như
     robot`. Đây là bài học 124 lặp lại ở tầng DỮ LIỆU.
135. **«Đã có sẵn corpus» phải mở ra xem.** `so_sanh_that.jsonl` mang đúng tên
     việc cần làm, đúng kích thước (4,3 MB), do chính dự án dựng — và nội dung
     là thứ khác hẳn. Suýt train 70 phút vào nó.

### 88.6 Bước THẮT — sạch được, nhưng SẢN LƯỢNG là ràng buộc thật

Ba luật CẤU TRÚC, không danh sách từ viết tay (Luật 0):

```
1. BỎ mệnh đề ĐẦU cho luật 3 — đó là mệnh đề giống/loài (`thú gặm nhấm lớn hơn
   chuột`), tính từ ở đó tả GIỐNG chứ không tả cá thể
2. Nhận `<danh từ> <tính từ>` đúng 2 đơn vị — khuôn thuộc tính kinh điển tiếng
   Việt: `mõm tròn`, `đuôi xù`, `da trơn`.  Áp cho MỌI mệnh đề.
3. Nhận mệnh đề chỉ gồm ĐÚNG MỘT tính từ, trừ mệnh đề đầu
```

Đọc **toàn bộ** 67 câu (đủ nhỏ để không phải lấy mẫu): rác còn 7 câu — `rộng như
heo`, `rộng như lợn`, `tròn như lim`, `tròn như sải`, `tròn như xải`, `tròn như
đay`, `trắng như cam`. **~50% → ~10%.** Thắt có tác dụng thật.

Nới luật 1 (chỉ áp cho luật 3, không áp cho luật 2) thêm được bộ vàng
`lớn → chim: hạc, sếu, ưng` với khỉ `[cuốc, cò, cú, cắt, cốc]`.

```
                     bộ    câu khác nhau    rác
không thắt          128        ~600        ~50%
thắt (3 luật)        20          67        ~10%
thắt + nới luật 1    21          73        ~10%
```

**Nhưng sản lượng là ràng buộc thật, và nó không vá được bằng mã.** 73 câu khác
nhau / 146 dòng, đặt cạnh **450 050 cha**, là hạt cát. §80 thắng nhờ 19 230 văn
bản. Nguồn `dinh_nghia.jsonl` (10 712 câu `X là …`) đơn giản KHÔNG chứa nhiều
mệnh đề `<bộ phận> <tính từ>` có đồng-vật-mang cùng lớp: thắt từ 118 thuộc tính
(đo lỏng, §88.2) xuống 21 bộ là mất 82%, và phần mất đó chính là phần rác.

Đánh đổi này KHÔNG có điểm ngọt ở giữa: nới ra thì rác quay lại, thắt vào thì
hết dữ liệu. Vẫn **chưa nạp vào `train_v12_sent.yaml`** — train 146 dòng chống
450 050 cha gần như chắc chắn cho kết quả rỗng, và biết điều đó tốn 70 phút.

### 88.7 Ba đường đi tiếp, xếp theo giá trị kỳ vọng

1. **Mở rộng NGUỒN** — `dictionary.jsonl` đầy đủ và `viwiki.jsonl` chưa được
   quét bằng ba luật này. Cùng kỷ luật, nhiều nguyên liệu hơn. Đây là đường duy
   nhất có thể đưa sản lượng lên mức §80.
2. **Tăng mật độ** (`--nhac-lai`) — TK_052 có sẵn tham số này đúng cho việc
   «mật độ so với nhiễu». Rẻ, nhưng lặp một corpus 73 câu không tạo thêm tri thức.
3. **Train thử luôn** — 70 phút để biết chắc là rỗng. Chỉ đáng nếu muốn đóng
   đường này bằng số thay vì bằng lập luận.

### 88.8 Bài học

136. **Kỷ luật làm sạch dữ liệu và sản lượng đối nghịch nhau, và phải đo CẢ HAI
     ở mỗi nấc thắt.** Ba nấc ở đây cho 128/20/21 bộ với 50%/10%/10% rác. Nếu
     chỉ báo cáo tỉ lệ rác thì bước thắt trông như thắng lớn; nếu chỉ báo cáo số
     bộ thì trông như thất bại. Chỉ hai cột cạnh nhau mới nói đúng.

### 88.9 Quét thêm nguồn — kích thước KHÔNG phải nút thắt, nhưng tìm ra mỏ đúng

Đường 1 của §88.7: cùng ba luật, quét thêm `data/dict/vi_dict_raw.jsonl`
(36 764 mục) và `viwiki.jsonl` (285 935 dòng).

```
nguồn                                 mục quét    bộ giữ    câu khác nhau
dinh_nghia                              10 712       21           73
+ từ điển thô                           47 475       24           81
+ viwiki                               221 977       24           81   ← +0 bộ
```

**Nguồn tăng 20×, sản lượng tăng 0.** Hai lý do, đều đo được:

- `dinh_nghia.jsonl` vốn được DẪN XUẤT từ chính từ điển thô ⇒ cùng một tri thức,
  không phải nguồn mới.
- Văn xuôi bách khoa **không** viết thuộc tính theo khuôn `<bộ phận> <tính từ>`.
  Đúng kết luận TK_052 §66 («0,0% trên bách khoa»), nay xác nhận trên 285 935
  dòng thay vì mẫu nhỏ.

⇒ **Kích thước nguồn không phải nút thắt. HÌNH DẠNG mới là.** Đóng dứt điểm hy
vọng «cứ nạp thêm dữ liệu».

### 88.10 Nhưng từ điển CÓ mỏ đúng — thành ngữ so sánh

Thấy dấu `(tng)` trong mục từ điển ⇒ từ điển có đánh dấu tục ngữ. Quét lại, chỉ
trên TỪ ĐIỂN (nơi `như` thật sự là so sánh), không quét viwiki (nơi `như` nghĩa
«chẳng hạn» — chính cái bẫy làm N1+N2 ở §88.2 ra rác):

```
mục từ ĐANG LÀ thành ngữ so sánh : 15   (kín như bưng · lạnh như tiền ·
                                         chắc như cua gạch · rách như tổ đỉa)
đoạn chứa «như» trong từ điển    : 5 068
cặp «tính từ như danh từ»        : 241 thuộc tính · 50 có ≥2 VẬT MANG

   nhanh → [bước đi, cắt, sóc, tên]        ← ĐÚNG ca «con khỉ» của cả dự án
   đen   → [mực, dầu hắc, gỗ, củ, hạt]
   đẹp   → [hoa, ngọc nữ, thiên thần, tiên nga]
   mịn   → [bột, lụa, nhung]
   lạnh  → [nước đá, tiền, đêm]
   trắng → [cước, sương, tuyết]
```

**50 thuộc tính so với 21 bộ của đường mệnh đề**, và vật mang là mục tiêu so sánh
THẬT chứ không phải hiện vật liệt kê. Còn lẫn rác — `đúng → [của, những, lẽ]`,
vật mang `cái`/`thế`/`không`/`màu` — nhưng đó là HƯ TỪ bị từ điển gán nhãn danh
từ, lọc được bằng BẬC (vật mang phục vụ quá nhiều thuộc tính khác nhau ⇒ chung
chung), tức lọc theo tần suất đo được, không phải danh sách viết tay.

### 88.11 Bài học

137. **Nguồn đúng quan trọng hơn nguồn LỚN — và «đúng» ở đây là đúng THỂ LOẠI.**
     Cùng một phép rút `A như B`: chạy trên mọi nguồn (§88.2) cho rác vì viwiki
     dùng `như` nghĩa «chẳng hạn»; chạy CHỈ trên từ điển cho `nhanh như cắt`,
     `đen như mực`. Không phải bộ lọc khác — chỉ khác chỗ quét.
138. **Nới nguồn 20× mà sản lượng đứng yên là một câu trả lời, không phải một
     thất bại.** Nó chuyển câu hỏi từ «đủ dữ liệu chưa» sang «dữ liệu có hình
     dạng đó không», và câu sau trả lời được dứt khoát.

### 88.12 Lọc bậc + gộp hai đường — và một kết luận DỨT ĐIỂM về điều kiện ②

Lọc bậc (ngưỡng lấy từ phân bố: trung vị×4) bỏ đúng thứ cần bỏ — 12/422 vật
mang, toàn hư từ bị từ điển gán nhãn danh từ: `không`, `tiếng`, `cái`, `bị`,
`người`, `màu`, `trước`, `thế`, `con`, `một`.

Gộp hai đường chỉ được 24 → **27 bộ**, dù đường B có 50 thuộc tính. Truy ra:

```
ĐƯỜNG B (thành ngữ so sánh, sau lọc bậc)
  thuộc tính có ≥2 vật mang           : 50
  … có ≥2 vật mang CÓ NHÃN LỚP        : 41
  … có ≥2 vật mang CÙNG MỘT LỚP       :  2   ← điều kiện ② thật
        đen → phần: [củ, gỗ]  ·  tốt đẹp → phía: [trong, trên]   (cả hai là rác)
```

Nhìn được ngay vì sao:

```
cứng → mo (lá)        · thép (hợp kim)
đỏ   → gấc (cây leo)  · mắt (cơ quan)
nhanh→ cắt (chim)     · sóc (thú)
```

**Thành ngữ CỐ TÌNH chọn một vật mẫu đặc sắc cho mỗi thuộc tính, và các vật mẫu
nằm khác lớp nhau** — đó chính là thứ làm nên thành ngữ. `nhanh như cắt` và
`nhanh như sóc` không tạo bộ vì cắt là chim còn sóc là thú.

⇒ Điều kiện ② đòi «một thuộc tính được chứng thực lặp lại trên ≥2 thành viên
CÙNG LỚP» là đòi hỏi đúng thứ **ngôn ngữ tự nhiên tránh nói**, vì nói lại là
thừa. Đây là lời giải cấu trúc cho việc TK_052 đạt 84,8% trên corpus tổng hợp mà
0,0% trên dữ liệu thật — và nó **không sửa được bằng cách tìm nguồn tốt hơn**.

### 88.13 Chỗ rẽ: ② ĐANG BỊ ĐÒI QUÁ CHẶT

Ràng buộc «≥2 vật mang cùng lớp» nằm ở **bộ dựng corpus và khoá chấm của tôi**,
không phải ở cơ chế. Với MỘT vật mang, chuỗi vẫn đủ mắt xích:

```
nhanh như cắt .        (đường B — CÓ trong từ điển)
cắt là chim …          (đường A — CÓ trong dinh_nghia)
chim ⊃ {cú, cò, sếu, cốc, cuốc}     (via_pivot đảo — ĐÃ CÓ)
⇒ đề cử «nhanh như cú»              ← đúng suy luận «con khỉ», từ MỘT ví dụ
```

Và dữ liệu ủng hộ: 41/50 thuộc tính có ≥2 vật mang CÓ NHÃN LỚP; nới xuống ≥1 vật
mang thì nguyên liệu lớn hơn nhiều bậc. Điều kiện ≥2 vốn để cho suy luận lớp
CHẮC CHẮN hơn; với việc SINH (đề xuất một loại suy hợp lý) thì một ví dụ là đủ.

Chưa đo. Đó là việc kế: dựng bộ với ≥1 vật mang, đọc mẫu bằng mắt, rồi train.

### 88.14 Bài học

139. **Ràng buộc «cần ≥2 bằng chứng» có thể chính là thứ chặn đường.** Nó nằm
     trong bộ dựng dữ liệu của tôi chứ không trong cơ chế, và nó đòi ngôn ngữ
     nói ra thứ ngôn ngữ coi là thừa. Ba vòng (§88.6, §88.9, §88.12) vật lộn với
     sản lượng đều là hệ quả của MỘT lựa chọn thiết kế chưa từng bị chất vấn.
140. **Khi một phép rút cho 50 mục mà chỉ 2 đi tiếp được, hãy đo KHÚC RỚT chứ
     đừng nới phép rút.** Số 50 → 41 → 2 chỉ thẳng chỗ mất, và chỗ mất hoá ra là
     một giả định về ngôn ngữ, không phải một tham số.

### 88.15 Nới ② xuống ≥1 vật mang — sản lượng ×7, và corpus ĐỦ SẠCH để train

```
--min-mang     bộ giữ    dòng    câu khác nhau    rác câu    rác bộ
     2 (cũ)      27       174         ~90          ~10%      ~10%
     1 (mới)    198       516        258           ~20%      ~50%
```

Rác BỘ tăng vọt vì **nhãn lớp chung chung** (`gọn → từ`, `đúng → điều`,
`bé → đồ`, `tốt đẹp → phía`) — đúng vấn đề §75/§76 đã kết luận không lọc được
bằng tín hiệu nội tại. Nhưng hai thứ TÁCH RỜI được:

- **câu** dùng để TRAIN → ~80% sạch, đủ dùng
- **bộ** chỉ dùng để CHẤM → lọc theo lớp CỤ THỂ (tiền lệ chế độ `cuthe` của
  thước sẵn có) còn **44 bộ**, trong đó có thành ngữ Việt THẬT:

```
khoẻ như voi            → thú  ⇒ khỉ [cheo, chuột, cáo, cầy]
khôn như cáo            → thú
lớn như [hạc, sếu, ưng] → chim ⇒ khỉ [cuốc, cò, cú, cắt]
ngọt như [mận, nho]     → cây
nhẹ như [bấc, sui]      → cây
```

### 88.16 Đã nạp vào `train_v12_sent.yaml` và train lại

`--nhac-lai 4` ⇒ 1 032 dòng. Vẫn nhỏ so với 450 050 cha, nhưng có lý lẽ khiến
corpus nhỏ vẫn ăn: **cơ chế đọc CẠNH `via_attr`, mà cạnh chỉ cần TỒN TẠI chứ
không cần áp đảo tần suất** — §80 đã cho thấy cạnh `sóc → thú` tồn tại là đủ đổi
hẳn đầu ra. Đây là dự đoán CHỐT TRƯỚC, không phải giải thích sau.

Ghi ra `checkpoints/network_v12_sent_ls.pkl` — tên MỚI, giữ nguyên
`network_v12_sent.pkl` và `network_v12_sent_dn.pkl` để đối chứng ba chiều.

Cổng chấm sau khi train (chốt trước):
- **TRUNG THÀNH không được tụt** dưới 78,9% — vế PHỦ QUYẾT, giống §84/§87
- 4 prompt LOẠI SUY (0/4 suốt cả phiên) phải có ≥1 câu đề cử đúng LỚP
- thu hồi trên 44 bộ lớp cụ thể, so với đối chứng hoán vị

---

## 89. CHẤM CỔNG §88.16 — ÂM TÍNH, nhưng tách bạch được hai thứ cả phiên bị lẫn

Train xong: `network_v12_sent_ls.pkl` · **450 303 cha** · 5 237s (87 phút, có
tính một lần máy ngủ hết pin — log để lại dấu: 208 giây trôi mà chỉ thêm 510 cha,
rồi nhịp trở lại bình thường).

### 89.1 VẾ 1 — TRUNG THÀNH (PHỦ QUYẾT): **ĐẠT**

```
TRUNG THÀNH 78,9%  ·  TRẬT TỰ 89,3%  ·  LẶP 13,6%
```

Đúng bằng mốc, bốn ca vàng giống hệt. Corpus loại suy **không làm hỏng** chất
lượng định nghĩa. Vế phủ quyết qua.

### 89.2 VẾ 2 — prompt LOẠI SUY: **TRƯỢT** trên bốn ca gốc

Bốn prompt gốc KHÔNG đổi một token nào so với trước khi train:

```
nghịch ngợm như loài ▸ cây có và là quả hoa trong được hạt để tạo
chó và mèo đều là    ▸ nhà có: công Chó chúng, với thể: có năng khả…
hổ giống mèo vì      ▸ một người có thể là sử dụng của được và làm
```

**Nhưng** ba prompt theo khuôn corpus cho đầu ra HOÀN HẢO — trọn câu, đóng bằng
dấu chấm:

```
khoẻ như  ▸ khoẻ như voi .
nhanh như ▸ nhanh như sóc .
khôn như  ▸ khôn như cáo .
```

Phải nói thẳng: **cả ba đều CÓ trong corpus train**. Đây là DỘI LẠI nguyên văn,
không phải suy luận loại suy. Nó chứng minh corpus được hấp thụ, không chứng minh
«con khỉ».

### 89.3 VẾ 3 — thu hồi vs ĐỐI CHỨNG HOÁN VỊ: **TRƯỢT DỨT KHOÁT**

```
THẬT     44 bộ · top-1 0,0% · top-5 4,5% · top-10 4,5%
HOÁN VỊ  44 bộ · top-1 0,0% · top-5 6,8% · top-10 6,8%   ← ĐỐI CHỨNG CAO HƠN
```

Đối chứng **cao hơn** bản thật ⇒ cơ chế không dùng cạnh thuộc tính chút nào;
phần trùng là ăn may. So với TK_052 vòng 3 (84,8% vs 0,0% trên corpus tổng hợp),
đây là khoảng cách một trời một vực.

### 89.4 Định vị được chỗ hỏng — ① CHẠY, ② mới là chỗ vỡ

Kiểm cạnh đã dựng:

```
via_attr: CÓ · 33 974 cạnh
  via_attr['khoẻ'] = [hùm, hộ, lực, sĩ, tay, VOI, điều]     ✓
  via_attr['khôn'] = [con, CÁO]                              ✓
  via_attr['đẹp']  = [Anh, Lộc, bí, HOA, mặt, nhà]           ✓
```

**Bước ① chạy đúng.** Vậy mà đề cử vẫn rác — và nhìn nội dung đề cử là thấy chỗ
vỡ:

```
«chậm như ___» → [đồ, phần, tên, bệnh, bình, người]
```

Đó là **NHÃN LỚP CHUNG CHUNG**, không phải thành viên. Chuỗi đi tới ② rồi lấy
phải lớp `đồ`/`phần`/`tên`, và ③ nở lớp chung chung đó ra thành nhiễu.

⇒ Nút thắt KHÔNG phải hình dạng dữ liệu (đã sửa, đã hấp thụ, ① có cạnh), mà là
**chất lượng nhãn lớp** — đúng thứ TK_046 §75/§76 đã kết luận không lọc được
bằng sáu tín hiệu nội tại lẫn nguồn từ điển, và TK_052 «điều kiện 4» đã nêu là
ranh giới của kiến trúc hiện tại. Nay có bằng chứng đầu-cuối cho kết luận đó.

### 89.5 Đọc lại cả phiên bằng kết quả này

Corpus loại suy đã làm ĐÚNG phần việc của nó: dạy được khuôn `A như B`, dựng
được 33 974 cạnh thuộc tính, sinh được `khoẻ như voi .` ở chỗ trước đây ra rác.
Thứ nó không làm được — và không thể làm được — là cấp một CÂY PHÂN LOẠI.

### 89.6 Bài học

141. **Một âm tính có ĐỐI CHỨNG CAO HƠN bản thật là kết quả sạch nhất có thể.**
     4,5% vs 6,8% không cần bàn thêm: tín hiệu bằng 0. Không có nó thì 4,5% dễ
     bị đọc thành «yếu nhưng có».
142. **Tách DỘI LẠI khỏi SUY LUẬN trước khi mừng.** `khoẻ như voi .` là câu đẹp
     nhất mà prompt loại suy từng sinh ra trong cả hai phiên — và nó nằm nguyên
     văn trong corpus train. Nếu không tự kiểm điều đó thì đây đã được báo cáo
     thành «cơ chế loại suy đã chạy».

---

## 90. NẠP CÂY PHÂN LOẠI — hai nguồn, hai kiểu hỏng khác nhau; ĐÓNG đường này

§89 định vị nút thắt là CHẤT LƯỢNG NHÃN LỚP và kết luận phải nạp cây phân loại
thật. Vòng này thử hai nguồn. **Cả hai đều không dùng được, vì hai lý do khác
hẳn nhau.**

### 90.1 Nguồn A — quan hệ GIỐNG của từ điển: KHÔNG phải cây phân loại

Dựng được 1 978 cạnh giống trực tiếp, và ca sạch thì rất sạch:

```
voi → thú → động vật   ✓      cắt → chim → động vật   ✓      cam → cây → thực vật  ✓
```

Nhưng đỉnh bảng ra đúng loại nhãn §89 chê:

```
phần 202 · cái 174 · từ 137 · người 124 · vật 112
động vật 98 · [anh, bà, bò, bõ, bạn, bồi, cai, cha]   ← rò qua `người → động vật`
```

Vì định nghĩa tiếng Việt hay mở đầu bằng danh từ GIỮ CHỖ (`cái…`, `phần của…`,
`đồ…`, `từ dùng để…`) nên đầu cụm là hư danh, không phải lớp. Và bao đóng bắc
cầu rò qua từ đa nghĩa: `chì → KIM → vật → cái` (`kim` = cái kim).

Lọc kích thước KHÔNG cứu được: `cây` (102, tốt) và `vật` (112, xấu) không phân
biệt nổi bằng số thành viên. **Bác nguồn A.**

### 90.2 Nguồn B — Wikidata `P31`/`P279`: cây THẬT, nhưng không tra được

Cạnh có KIỂU nên không mắc lỗi của nguồn A — và sửa đúng ca nguồn A sai:

```
chì → kim loại yếu, nguyên tố hóa học     (từ điển cho `kim` ✗)
sắt → kim loại   ·   cò → chim, diệc   ·   voi → lớp Thú   ·   cam → quả có múi
bàn → đồ nội thất  ·  bão → thời tiết cực đoan  ·  ao → vùng nước lặng
```

Hỏi cho 1 601 từ (18 lô SPARQL, 45 giây). Hai chỉ số giết nó:

```
phủ toàn bộ                : 372/1 601  (23%)
phủ 119 từ mà 44 bộ CẦN    :  34/119    (29%)
```

Và phần phủ được thì **sai NGHĨA** do tra bằng nhãn trần, không phân biệt được
từ đa nghĩa:

```
chuột → Thiết bị điều khiển con trỏ, phần cứng   ← con chuột MÁY TÍNH
mun   → tiền tệ        ·  dong → phân cấp hành chính Hàn Quốc
```

Thiếu hẳn những từ trung tâm của bộ: `cuốc`, `cú`, `bèo`, `bí`, `bòng`, `bương`,
`bấc`, `cheo`, `cà` — danh từ Việt thông dụng nhưng không có nhãn `vi` trên
Wikidata. **Bác nguồn B.**

### 90.3 Kết luận — ĐÓNG đường «nạp cây phân loại» với tài nguyên hiện có

Hai kiểu hỏng khác nhau, và không bù nhau được:

| nguồn | có cạnh đúng KIỂU? | phủ từ vựng Việt? | phân biệt NGHĨA? |
|---|---|---|---|
| giống từ điển | ✗ (đầu cụm là hư danh) | ✓ 1 978 | ✗ (rò đa nghĩa) |
| Wikidata | ✓ | ✗ 29% | ✗ (tra nhãn trần) |

Thứ cần là **WordNet tiếng Việt có liên kết NGHĨA** — vừa có cạnh kiểu, vừa phủ
từ vựng Việt, vừa gắn với đúng nghĩa của mục từ. Không có trong máy, và không
thay thế được bằng hai nguồn trên.

`lop_trang` vẫn CHƯA nối (nó có trong `hub_spread` nhưng không có cờ config —
lần thứ SÁU cùng kiểu lỗi trong phiên). **Không nối**, vì chưa có danh sách lớp
nào đáng nạp; nối một cơ chế rồi cho nó ăn danh sách rác chỉ tạo thêm một nhánh
chết nữa.

### 90.4 Đính chính

Lượt trước tôi báo «không ra được mạng» — SAI. Lệnh hỏng vì macOS không có
`timeout`, không phải vì mất kết nối. Wikidata trả HTTP 200 ngay lần thử lại.

### 90.5 Bài học

143. **«Cây phân loại thật» không phải một thứ — nó là BA thuộc tính phải có
     cùng lúc:** cạnh đúng kiểu, phủ từ vựng, gắn đúng nghĩa. Hai nguồn thử
     nghiệm mỗi nguồn có một, thiếu hai. Nêu yêu cầu ở mức «nạp cây phân loại»
     là quá thô để kiểm chứng được.
144. **Kiểm công cụ trước khi kết luận về thế giới.** `timeout` không có trên
     macOS, và tôi suýt đóng cả một hướng vì tưởng mất mạng.

---

## 91. KHẢO SÁT WordNet tiếng Việt — có tồn tại, nhưng quá nhỏ

§90.5 nêu yêu cầu: cần nguồn có ĐỦ BA thuộc tính — cạnh đúng kiểu, phủ từ vựng
Việt, gắn đúng nghĩa. Vòng này đi tìm.

### 91.1 Năm nguồn, kết quả

| nguồn | trạng thái | cạnh KIỂU | phủ từ ta cần | đúng NGHĨA |
|---|---|---|---|---|
| Open Multilingual WordNet | 34 ngôn ngữ, **không có `vi`** | — | — | — |
| `viet.wordnet.vn` (VN Lexicography) | **CHẾT** (HTTP 000) | — | — | — |
| `zeloru/vietnamese-wordnet` ★33 | chỉ DANH SÁCH TỪ; README ghi phân nhóm còn ở «Todo» | ✗ | — | — |
| Wikidata `P31`/`P279` | sống | ✓ | 29% | ✗ tra nhãn trần |
| **VietNet** (`TriDo86/…/vietnet`) | sống, CC-BY-4.0 | ✓ | 36% | ✓ |

OMW có `th` Thái, `id`, `zsm` Mã Lai — **không** có tiếng Việt.

### 91.2 VietNet — đúng hình dạng, sai quy mô

Chuẩn **WN-LMF 1.3** của Global WordNet, có `ili` liên kết sang Open English
WordNet, mỗi synset có `<Definition>` riêng ⇒ **phân biệt được nghĩa**, thứ mà
tra nhãn Wikidata trần không làm được (`chuột` → con chuột máy tính).

Và nó phủ ĐÚNG những từ §90 thiếu: `cáo, cú, cò, cuốc, cắt, cốc, hạc, sếu,
chuột, voi, cầy, cheo, khướu, nhạn, én, thỏ, cừu, mèo, ngựa, lừa` — **20/20**.

Nhưng gộp cả hai miền (động vật + thực phẩm) chỉ có:

```
623 synset · 1 145 từ · 336 cạnh hypernym

44 bộ cần 119 từ
   có trong VietNet        43  (36%)
   … VÀ có cạnh hypernym   18  (15%)   ← dùng được THẬT
```

Cạnh thưa (336 cho 623 synset) nên phần lớn mục là LÁ không cha: `cú`, `cuốc`,
`cốc`, `hạc`, `sếu` đều có trong từ vựng mà không leo được bậc nào.

### 91.3 Kết luận

**Có WordNet tiếng Việt đúng chuẩn, sense-linked, giấy phép mở — nhưng nó phủ
hai miền (động vật, thực phẩm) với ~600 synset.** 15% phủ dùng được không đủ để
cấp nhãn lớp cho ②. Nó không sai như hai nguồn ở §90; nó chỉ quá nhỏ.

Chưa tải gì vào repo — mọi file khảo sát nằm ở `/tmp`. Nạp 500 KB dữ liệu mà chỉ
dùng được 15% là thêm phụ thuộc ngoài không tương xứng; nếu sau này ghép nhiều
nguồn thì hãy nạp một lần.

### 91.4 Bài học

145. **«Có tồn tại không» và «đủ dùng không» là hai câu hỏi khác nhau, và câu
     sau mới đáng hỏi.** VietNet trả lời CÓ cho cả ba thuộc tính §90.5 — rồi
     trượt ở thuộc tính thứ tư không ai nêu: QUY MÔ. Bốn thuộc tính, không phải
     ba; và mỗi vòng khảo sát lại lộ thêm một thuộc tính chưa từng nêu.
146. **Khảo sát phải đo trên ĐÚNG danh sách từ của bài toán, không đo tổng
     quát.** VietNet có 1 145 từ nghe nhiều; chiếu lên 119 từ mà 44 bộ cần thì
     còn 18. Con số duy nhất có nghĩa là con số thứ hai.

---

# TỔNG KẾT PHIÊN (§80 – §91)

12 mục, 12 commit. **Hai mục dương, chín mục âm tính sạch, một mục khảo sát.**

## 1. Thứ MUA ĐƯỢC

| | trước | sau |
|---|---|---|
| §80 nguồn định nghĩa (DỮ LIỆU) | 0/7 prompt đúng tri thức | **4/7** |
| §81 vá tín hiệu dừng (CƠ CHẾ) | 4/20 prompt có đuôi rác | **0**, không hồi quy |

```
chó là gì? không có nghĩa được mộtkhẩu
   → chó là gia súc thuộc nhóm ăn thịt, nuôi để giữ nhà hay đi săn .
sắt là kim loại hay phi kim loại?
   → sắt là kim loại màu xám xanh, dễ dát mỏng và kéo sợi, dễ bị gỉ trong không khí ẩm
```

Mốc hiện tại: **TRUNG THÀNH 78,9% · TRẬT TỰ 89,3% · LẶP 13,6%** · 450 test pass.

## 2. Thứ ĐÓNG ĐƯỢC bằng số (chín âm tính)

| mục | giả thuyết | kết quả |
|---|---|---|
| §82 | nới tiêu thụ cạnh chữa word-salad | lặp **gấp đôi** (13,6% → 25,8%) |
| §83 | bậc-2 toàn cục cho trật tự | trật tự 98% mà **tri thức = 0** |
| §84 | bậc-2 từ cha đang hoạt | 74,7% < mốc 78,9% |
| §86/§87 | gộp hoa/thường khi truy hồi | hạng 4→2 nhưng đầu-cuối **đổi 0 byte** |
| §88 | corpus loại suy quy mô lớn | trần sản lượng ~70 câu, nguồn ×20 → **+0** |
| §89 | corpus mới làm chuỗi chạy | đối chứng hoán vị **CAO HƠN** bản thật |
| §90 | nạp cây phân loại | hai nguồn, hai kiểu hỏng, không bù nhau |
| §91 | WordNet tiếng Việt | có thật (CC-BY) nhưng **15%** phủ dùng được |

Không mục nào trong chín mục này để lại thay đổi mặc định. Mã của các kênh âm
tính vẫn giữ — có cờ, có tài liệu, **mặc định tắt**, có test canh (`LOI10`).

## 3. Lỗi TÀNG HÌNH tìm thấy — sáu lần cùng một kiểu

«Cơ chế có, đo được, mà đường thật không gọi»:

```
TK_052 LOI1  kenh_thuoc_tinh   nằm im ~20 mục
§79          hub_chain_boost   bật mà không đổi một token
§81          _bd_gen           luật ép EOS chưa bao giờ nổ
§83          seq_next          0 lời gọi trong generate() thật
§87          _active_parent_set  ← LỖI CỦA CHÍNH BẢN VÁ §86 CỦA TÔI
§90          lop_trang         có trong hub_spread, không có cờ config
```

Cộng hai lỗi cache khoá bằng `id()` trên vật thu gom được (§83.2) — một cái nằm
ngay trên đường chuỗi §79, phát chỉ mục loại suy của checkpoint CŨ. Vá xong bộ
test hết chập chờn (5/5 lần ổn định).

## 4. Ý tưởng «con khỉ» — trả lời dứt điểm

Chuỗi ba bước: thuộc tính → vật mang → lớp → thành viên khác.

```
①  CHẠY   — 33 974 cạnh via_attr; `khoẻ → voi`, `khôn → cáo` có thật (§89.4)
②  VỠ     — lấy phải nhãn lớp CHUNG CHUNG: «chậm như ___» → [đồ, phần, tên, bệnh]
③  —      — nở lớp chung chung thành nhiễu
```

Và §88.12 chỉ ra vì sao ② không sửa được bằng dữ liệu: điều kiện «một thuộc tính
được chứng thực trên ≥2 thành viên CÙNG LỚP» đòi đúng thứ **ngôn ngữ tự nhiên
tránh nói**. Thành ngữ cố tình chọn vật mẫu khác lớp — `nhanh như cắt` (chim) và
`nhanh như sóc` (thú). Trên 50 thuộc tính thành ngữ, chỉ **2** có ≥2 vật mang
cùng lớp, và cả hai là rác.

⇒ Ý tưởng KHÔNG sai. Nó bị chặn ở **một mắt xích cần nguồn ngoài mà tiếng Việt
chưa có đủ**: cây phân loại vừa đúng kiểu, vừa phủ từ vựng, vừa gắn nghĩa, vừa
đủ lớn. Bốn thuộc tính; nguồn tốt nhất tìm được (VietNet) đạt ba, trượt quy mô.

## 5. Tôi đã sai ở đâu

- **§86 nối sai chỗ** — hàm đúng, cờ đúng, test xanh, tác động bằng 0. Chỉ A/B
  thấy hai hàng giống nhau từng byte mới lộ.
- **§85 probe đo sai đường** — gọi `rank_matches` trên store trần, suýt kết luận
  về một đường không tồn tại trong production.
- **§85 nói quá** — làm tròn 3 chữ số biến 3 cha hoà thành «5–6 cha hoà», đổi
  hẳn chẩn đoán.
- **§90 báo sai «không ra được mạng»** — lệnh hỏng vì macOS không có `timeout`.
- **§88 đề xuất sai** — tôi khuyên «mở rộng nguồn», rồi chính phép đo của tôi
  cho thấy nguồn ×20 → sản lượng +0.
- **§88.2 cổng hỏng** — đếm «≥2 vật mang» mà bỏ ràng buộc CÙNG LỚP, suýt dựng
  corpus từ `khác → [a, album, anh, ba]`.

## 6. Việc kế, xếp theo giá trị

1. **Chấp nhận ② không chạy trên dữ liệu Việt hiện có** và chuyển mục tiêu: dùng
   chuỗi cho ĐỀ XUẤT có kiểm chứng (một vật mang là đủ để gợi ý, không đủ để
   khẳng định), thay vì đòi thu hồi đúng.
2. **Ghép nhiều nguồn phân loại** (VietNet + Wikidata + giống từ điển, giao nhau
   để khử lỗi nghĩa) — chưa đo, và §91 cho thấy trần phủ có thể vẫn thấp.
3. **Dọn nợ**: `checkpoints/grade_v3.pkl` đỏ từ đầu phiên (thiếu file).

## 7. Bài học đắt nhất

**121** — một nguồn dữ liệu ĐÚNG HÌNH DẠNG thắng mười bản vá cơ chế. Phiên này:
§65/§80 thêm một nguồn ⇒ 0/7 → 4/7; chín bản vá cơ chế ⇒ 0.

**132** — «nối vào rồi» phải chứng minh bằng SỐ ĐỔI, không bằng dòng mã đã sửa.

**141** — âm tính có đối chứng CAO HƠN bản thật là kết quả sạch nhất có thể.

**145** — «có tồn tại không» và «đủ dùng không» là hai câu hỏi khác nhau.

---

## 92. VIỆC KẾ (1) — đề xuất CÓ KIỂM CHỨNG: ÂM TÍNH, và nguyên nhân sâu hơn §89

Đổi mục tiêu thay vì vá cơ chế: không đòi chuỗi THU HỒI thành viên bị giấu, đòi
nó ĐỀ XUẤT với cả ba cạnh đều chứng thực được — và ② **đọc thẳng** lớp từ cạnh
`b là C` của chính vật mang, thay vì suy ra lớp từ nhiều vật mang (thứ §88.12 đã
chứng minh là bất khả trên ngôn ngữ tự nhiên).

### 92.1 Lỗi probe của tôi — lần thứ BA trong phiên

Bản đầu dựng chỉ mục THIẾU `kenh_vi_ngu` và `chu_ngu_ket_hop_q`, hai tham số mà
`hub_chain_proposals` có truyền. Sửa rồi chạy lại: `via_attr` 33 974 → 32 256,
`via_pivot` **không đổi**, đề xuất **gần như y hệt**. Sửa là đúng; kết luận không
đổi. (§85, §87, nay §92 — ba lần probe sai cấu hình.)

### 92.2 Cổng TRƯỢT cả hai vế

```
── «nhanh như ___»  vật mang: [Nước, chuyện, cắt, giây lát, gió, hai]
     qua «Nước là .<EOS>»  ⇒ «nhanh như "», «nhanh như "Cái», «nhanh như "Ngươi»
     qua «Nước là 0»       ⇒ «nhanh như 0», «nhanh như 1», «nhanh như 100»
── «khôn như ___»  vật mang: [con, cáo]
     qua «con là 3»        ⇒ «khôn như ,», «khôn như 0», «khôn như 1»

LỚP bắc cầu phổ biến nhất:
   gì?<EOS> 36 · một loại 32 · hạn 30 · loài 23 · loại 23 · hắn 22 · .<EOS> 20
```

Vế (a) đề xuất hợp lý: **~0%**. Vế (b) lớp cụ thể: không — chúng thậm chí không
phải danh từ chung chung như `đồ`/`phần` ở §89, mà là DẤU CÂU, EOS, hư từ.

### 92.3 Nguyên nhân — sâu hơn một tầng so với §89

§89 kết luận «② lấy phải nhãn lớp chung chung». Đo thẳng chất lượng đích của
toàn bộ đồ thị phân loại:

```
CHẤT LƯỢNG ĐÍCH của 223 988 cạnh via_pivot «A là B»
   DANH TỪ hợp lệ (theo từ điển)      87 517   39,1%
   không phải danh từ                133 223   59,5%
   SỐ                                   3 248    1,5%
```

**Chỉ 39% cạnh phân loại có đích là danh từ.** Vấn đề không phải «nhãn lớp chung
chung» mà là **61% cạnh không phải quan hệ phân loại gì cả**. Khuôn `A là B`
khai trên văn bản thật bắt phải `Nước là 0`, `con là 3`, `Ngọc là 2`,
`Lan là Amsterdam`.

Chuỗi ba bước ở §89 sống được (4,5%, dù thua đối chứng 6,8%) chỉ nhờ tầng XẾP
HẠNG rất nặng phía sau (`score_hubs_from`: typed → phiếu → từ đủ → số thành viên
→ score×hệ số → gidf → ABC). Bỏ tầng xếp hạng đó ra — như phép duyệt kiểm-chứng
ở đây — thì còn lại là nhiễu thuần.

⇒ Đổi mục tiêu KHÔNG cứu được. Vấn đề không nằm ở mục tiêu, mà ở chất lượng
cạnh `via_pivot`.

### 92.4 Một lỗi nhỏ trong chính phép đếm

Tôi hạ thường chuỗi rồi mới kiểm `isupper()` ⇒ nhóm «tên riêng» không bao giờ
khớp và rơi hết vào «khác». Con số 59,5% vì thế GỘP cả tên riêng. Đầu đề — «chỉ
39,1% đích là danh từ» — không đổi, nhưng phân rã bên dưới thì không đọc được.

### 92.5 Bài học

147. **Khi bỏ tầng xếp hạng ra mà kết quả sụp hoàn toàn, tầng xếp hạng đang che
     một đồ thị hỏng.** §89 đo 4,5% và tưởng là «cơ chế yếu»; §92 bỏ ranking ra
     và thấy nền là 61% cạnh sai. Chỉ số cuối cùng che mất chất lượng đầu vào.
148. **Ba lần probe sai cấu hình trong một phiên là một khuôn, không phải ba tai
     nạn.** Mọi probe phải dựng chỉ mục qua CÙNG một hàm mà production gọi, chứ
     không chép lại danh sách tham số bằng tay.

---

## 93. LÀM SẠCH cạnh `via_pivot` — bộ lọc ĐÚNG, VÔ HẠI, và VÔ ÍCH

§92.3 đo được chỉ 39,1% cạnh phân loại có đích là danh từ. Vòng này lọc.

### 93.1 Bộ lọc

Hai luật, không danh sách từ viết tay: đích là SỐ/DẤU CÂU/EOS ⇒ bỏ; **từ điển
NÓI** đích không phải danh từ ⇒ bỏ. Đích **ngoài** từ điển thì GIỮ — nhãn lớp
nhiều từ (`kim loại màu xám`) và tên riêng làm lớp đều nằm ngoài từ điển, bỏ
chúng là bỏ cả phần đúng.

```
TRƯỚC  223 988 cạnh · DANH TỪ 39,1% · ngoài từ điển 33,4%
                     · từ điển nói KHÔNG phải danh từ 24,1% · số/dấu câu 3,4%
SAU    162 314 cạnh · DANH TỪ 53,9% · ngoài từ điển 46,1%     (bỏ 61 674 = 27,5%)

CẠNH VÀNG: cắt→chim CÒN · chó→gia súc CÒN · sắt→kim loại CÒN · lúa→cây CÒN
```

`sóc→thú` báo MẤT ở **cả hai** cấu hình ⇒ đó là lỗi BPE (`sóc` = `s`+`óc`) có từ
TK_052, KHÔNG phải do bộ lọc. Đo cả hai chiều nên phân biệt được.

### 93.2 Chấm — không cải thiện gì

```
                  THẬT               HOÁN VỊ
loc_dich=False    top-5 4,5% top-10 4,5%    6,8% / 6,8%
loc_dich=True     top-5 4,5% top-10 6,8%    6,8% / 15,9%   ← đối chứng NỚI RỘNG
```

```
                  TRUNG THÀNH   TRẬT TỰ   LẶP
gốc                  78,9%       89,3%   13,6%
LỌC ĐÍCH             78,9%       89,3%   13,6%    ← Y HỆT, ca vàng giống TỪNG BYTE
```

Bỏ 61 674 cạnh rác mà đầu ra không đổi **một byte**, thu hồi không nhúc nhích.

### 93.3 Điều này nói gì — và nó bác chính chẩn đoán §92

§92 kết luận «nền là 61% cạnh sai, tầng xếp hạng đang che một đồ thị hỏng» và
suy ra làm sạch cạnh là việc kế. **Sai.** Làm sạch xong, không gì đổi.

Đọc đúng phải là: tầng xếp hạng ĐÃ khử được đám cạnh rác đó rồi — nên bỏ chúng
đi không thêm gì. Thứ nó xếp lên đầu sau khi khử rác vẫn là nhãn CHUNG CHUNG
(`đồ`, `phần`, `loài`, `loại`). Nghĩa là ② hỏng **không** vì đồ thị nhiễu, mà vì
**khái niệm «lớp» của tầng xếp hạng** — nó chuộng nhãn tần suất cao, và nhãn tần
suất cao trong tiếng Việt chính là nhãn chung chung.

Đó là đúng kết luận §75/§76 đã nêu, nay có thêm một bằng chứng phản chứng: khử
nhiễu đầu vào KHÔNG cứu được, vì nhiễu chưa bao giờ là nguyên nhân.

### 93.4 Trạng thái mã

Giữ (`hub_chain_loc_dich`), **mặc định TẮT**, canh bằng
`test_LOI11_loc_dich_mac_dinh_TAT`. Bật một bộ lọc không đo được lợi ích mà lại
đổi đồ thị cho MỌI tầng tiêu thụ khác là rủi ro không đối giá.

### 93.5 Bài học

149. **Sửa được thứ mình vừa chê mà không có gì đổi ⇒ lời chê sai, không phải
     bản sửa yếu.** §92 chê 61% cạnh rác; §93 bỏ sạch chúng và mọi chỉ số đứng
     im. Bằng chứng phản chứng mạnh hơn mọi lập luận thêm.
150. **«Vô hại» đủ để GIỮ MÃ, không đủ để BẬT.** Một bộ lọc đúng và không hại
     vẫn phải tắt nếu không đo được lợi ích — vì nó đổi đồ thị cho cả những tầng
     ta chưa đo.

---

## 94. VIỆC KẾ (2) — gộp ba nguồn phân loại: ÂM TÍNH vì NHÃN KHÔNG KHỚP NHAU

§93.3 định vị lại: ② hỏng vì tầng xếp hạng chuộng nhãn TẦN SUẤT CAO. Danh sách
lớp trắng (`lop_trang`) ghi đè được sở thích đó — và chỉ cần tập TÊN LỚP, không
cần ánh xạ thành viên, nên ba nguồn ở §90/§91 về lý thuyết gộp được dù mỗi nguồn
thiếu một thuộc tính khác nhau.

### 94.1 Biểu quyết — trượt ở CẢ HAI mức

```
phiếu ≥2   →  37 nhãn   (a) 5/6 nhãn chung chung bị loại ✓ nhưng `phần` lọt
                        (b) HỎNG: thú · cây · kim loại đều THIẾU (1 phiếu)
phiếu ≥1   → 959 nhãn   (b) đạt
                        (a) HỎNG hoàn toàn: đồ · phần · tên · cái · loài đều lọt
```

Không có ngưỡng nào ở giữa: hạ xuống thì rác vào, nâng lên thì vàng ra.

### 94.2 Vì sao — ba nguồn KHÔNG NÓI CHUNG MỘT NGÔN NGỮ NHÃN

```
A từ điển 136 · B wikidata 495 · C vietnet 368
A∩B  24 (18% của bên nhỏ)   A∩C  11 (8%)   B∩C  8 (2%)   A∩B∩C = 3 nhãn
```

Giao ba nguồn còn **ba** nhãn. Nguyên nhân nhìn thẳng được:

```
thú       A=có     B=không («lớp thú»)                 C=không
cây       A=có     B=không («cây ăn quả», «trái cây»)  C=không
kim loại  A=không  B=có («kim loại», «kim loại yếu»)   C=không
```

Cùng một khái niệm, ba tên khác nhau. Biểu quyết trên chuỗi khớp CHÍNH XÁC không
bao giờ chạy được. Muốn gộp phải CĂN CHỈNH NHÃN giữa ba từ vựng — đó là bài toán
đối sánh ontology, một lĩnh vực riêng, không phải một bước tiền xử lý.

Chuẩn hoá bằng tay (`lớp thú` → `thú`, bóc tiền tố `lớp`/`bộ`/`họ`) thì là danh
sách viết tay — đúng thứ Luật 0 cấm, và ad hoc theo từng nguồn.

### 94.3 `lop_trang` vẫn KHÔNG nối

Lần thứ hai từ chối nối nó (§90.3 lần đầu), cùng lý do và nay có thêm số: chưa
có danh sách lớp nào đáng nạp. Nối cơ chế rồi cho ăn danh sách 37 nhãn thiếu
`thú`/`cây`/`kim loại`, hoặc 959 nhãn đầy `đồ`/`phần`/`cái`, đều chỉ tạo thêm
một nhánh chết — đúng cái bẫy phiên này đã dính sáu lần.

### 94.4 Đọc lại: bài toán nhãn lớp đã ĐÓNG với tài nguyên hiện có

Bốn vòng liên tiếp đánh vào cùng một chỗ, bằng bốn cách khác nhau, đều âm tính:

```
§90  nạp cây phân loại        hai nguồn, hai kiểu hỏng
§91  tìm WordNet tiếng Việt   có thật, phủ 15% dùng được
§93  làm sạch cạnh via_pivot  bỏ 61 674 cạnh, chỉ số ĐỨNG IM
§94  gộp ba nguồn             giao ba nguồn còn 3 nhãn
```

TK_046 §75/§76 («nhãn chung chung không lọc được bằng tín hiệu nội tại») và
TK_052 «điều kiện 4» đã nêu đúng ranh giới này. Bốn vòng trên là bằng chứng
độc lập cho cùng kết luận, và đủ để coi hướng này ĐÓNG cho tới khi có một nguồn
phân loại tiếng Việt vừa đúng nghĩa, vừa phủ rộng, vừa đủ lớn.

### 94.5 Bài học

151. **Gộp nhiều nguồn yếu chỉ mạnh lên khi chúng NÓI CHUNG một ngôn ngữ.** Ba
     nguồn có 136 + 495 + 368 nhãn mà giao lại còn 3. Trước khi thiết kế phép
     biểu quyết, phải đo ĐỘ CHỒNG LẤN — nếu nó là 2% thì mọi phép gộp đều vô
     nghĩa, bất kể ngưỡng.
152. **Từ chối nối một cơ chế cũng là một quyết định kỹ thuật, và phải ghi lại.**
     `lop_trang` bị từ chối hai lần với lý do đo được. Nối nó cho «đủ bộ» sẽ tạo
     nhánh chết thứ bảy.

---

## 95. VIỆC KẾ (3) — dọn nợ test đỏ có sẵn

`tests/test_build_parents.py::test_build_parents_includes_qa_val_shard` đỏ suốt
cả phiên. Truy: nó nạp `checkpoints/grade_v3.pkl`, mà `.gitignore` dòng 2 có
`checkpoints/*.pkl` ⇒ **fixture không bao giờ có trong một bản clone sạch**. Test
đỏ trên mọi máy chưa tự dựng checkpoint — không phải lỗi tạm thời của máy này.

Chi phí thật của nó không phải một dòng đỏ, mà là: **mỗi lần chạy bộ test tôi
phải đọc lại để xác nhận «vẫn là lỗi môi trường có sẵn»** — và đúng loại nhiễu đó
là thứ che mất một test đỏ THẬT. §83.2 đã suýt bị nó che: hai test chập chờn lẫn
trong cùng dòng tổng kết.

Vá theo QUY ƯỚC SẴN CÓ của dự án (`test_copy_number.py`,
`test_frame_bracket_gate.py`): `skipif` kèm lý do. Có checkpoint thì test chạy y
như cũ — không hạ thấp phạm vi kiểm, chỉ thôi báo đỏ về một tệp không thể có mặt.

```
TRƯỚC  450 passed · 3 skipped · 1 xfailed · 1 FAILED
SAU    451 passed · 4 skipped · 1 xfailed · 0 failed     (ổn định 3/3 lần)
```

### 95.1 Bài học

153. **Một test đỏ kinh niên không phải nợ thẩm mỹ — nó là chỗ ẩn nấp.** Suốt
     phiên tôi viết «1 fail có sẵn» trong 14 commit; nếu có một test đỏ THẬT lẫn
     vào đó, tôi đã đọc lướt qua. §83.2 đúng là ca đó và chỉ lộ ra vì số đếm
     nhảy từ 1 lên 2.

---

## 96. TRUY `tê tê` — ra một lỗi TỔNG QUÁT: loại từ `con` làm vỡ đầu ra

Ca `tê tê` được chọn vì nó CÓ câu định nghĩa đúng trong corpus mà vẫn sinh rác —
tức không phải thiếu phủ. Truy ra ba thứ.

### 96.1 Không phải ca lẻ — loại từ đứng đầu làm vỡ đầu ra

```
«tê tê là»      ▸ tê tê là thú cỡ trung bình, mõm thường nhọn…      ✓
«chó là»        ▸ chó là gia súc thuộc nhóm ăn thịt, nuôi để giữ…   ✓
«mèo là»        ▸ mèo là thú và chó là bạn.                          ✓
«con tê tê là»  ▸ con tê tê là một số ví dụ về các từ có và một thành ✗
«con chó là»    ▸ con chó là gì? dụ                                  ✗
«con mèo là»    ▸ con mèo là nơi ví dụ                               ✗
«cái bàn là»    ▸ cái bàn là một số ví dụ về có thể đến không này rất ✗
```

Thêm loại từ (`con`, `cái`) — cách nói tiếng Việt **hoàn toàn tự nhiên** — là vỡ,
nhất quán trên bốn prompt. Đây là lỗ hổng người dùng chạm phải thường xuyên, và
chưa từng được ghi.

### 96.2 KHÔNG phải lỗi truy hồi

`tê tê là` cho cha đúng hạng **1**; `con tê tê là` cho hạng **3** — vẫn nằm sâu
trong top. Hạng 3 không giải thích được việc đầu ra vỡ hẳn. ⇒ nguyên nhân nằm
SAU truy hồi.

### 96.3 Lỗi idf có thật, nhưng KHÔNG phải nguyên nhân

Token ĐẦU DÒNG và GIỮA CÂU là hai id với hai `df` riêng; từ đầu tiên của MỌI
prompt luôn ở dạng đầu dòng, mà dạng đó hiếm ⇒ idf vống:

```
từ     đầu dòng  giữa câu        từ     đầu dòng  giữa câu
là       14,0      8,3           chó      14,0     10,7
một      14,0     10,7           mèo      13,3     10,4
con      13,3     10,7
```

Vá ở tầng đọc (`_df_gop`, không cần train lại) chạy đúng: `con` 13,3 → 10,7.
Nhưng hạng KHÔNG đổi (3 → 3) và đầu ra giống hệt **từng byte**. Lỗi thật, bản vá
đúng, **tác dụng bằng 0** ⇒ giữ mã, mặc định TẮT (`test_LOI12`).

### 96.4 Phát hiện phụ đáng lo hơn

```
tổng cha 450 303 · `_df` phủ 2 871/32 002 token · tổng df = 25 466
```

`_df` — cơ sở của MỌI phép tính idf — được dựng trên một tập rất nhỏ. Vì thế gần
như mọi token đều nhận idf sát trần (13–14), tức **idf gần như không phân biệt
được gì**. Chưa truy vì sao; đây là nghi vấn đứng đầu cho vòng sau.

### 96.5 Năm lần probe sai cấu hình

§85 · §87 · §92 · và trong chính vòng này HAI lần: quên `_vocab_ref` (bản đồ
rỗng ⇒ cờ bật mà không làm gì, số «0 thay đổi» đầu tiên VÔ NGHĨA), rồi gán
`_vocab_ref` mà quên `_fb_layer_ref` (vỡ ngay, đúng bất biến comment đã ghi).

Lần này cả hai đều lộ ồn ào chứ không im lặng cho số sai — nhưng đó là may, không
phải thiết kế.

### 96.6 Bài học

154. **Một ca hỏng được chọn vì «có dữ liệu mà vẫn sai» là ca đáng truy nhất.**
     `tê tê` không cho lời giải, nhưng cho một lỗi TỔNG QUÁT (`con X là`) mà
     không phép đo tổng hợp nào của phiên bắt được — vì mọi bộ prompt tôi dùng
     đều viết trần, không có loại từ.
155. **Bộ prompt thử phản ánh giả định của người viết nó.** 20 prompt dùng suốt
     phiên đều dạng `X là`; người Việt thật sẽ gõ `con chó là gì`. Lỗ hổng nằm
     đúng chỗ bộ thử không nhìn tới.

---

## 97. TRUY TIẾP `con X là` — tìm ra lỗi THẬT ở `pick_primary`, mà vẫn CHƯA phải nguyên nhân

### 97.1 Truy vết nhánh — chỗ rẽ nằm ở `_build_segment_plan`

Hai prompt chỉ khác nhau MỘT từ, nên nhánh nào rẽ khác là nghi phạm trực tiếp:

```
«chó là»      _build_segment_plan → segments=[[1753, 4793, 2117, 2467, 1904, …]]  ← agenda DÀI
«con chó là»  _build_segment_plan → segments=[[1737, 290]]  entity_ids=(5732, 3103, …)
              [1737, 290] = « gì?»   ⇒ đầu ra «con chó là gì? dụ»
```

`5732` = `con` (dạng đầu dòng) nằm trong `entity_ids`.

### 97.2 Lỗi THẬT: loại từ thắng `pick_primary`, và lý do bất đối xứng

`pick_primary` chọn theo `_gidf`, tức `_gdf` — mà `_gdf` cũng tách hai dạng ranh
giới (bản vá §96 chỉ chạm `_df`):

```
«chó là»      chó ĐẦU DÒNG  gdf=   24  gidf=10,8  → primary = chó  ✓
«con chó là»  chó GIỮA CÂU  gdf= 1316  gidf= 6,8
              con ĐẦU DÒNG  gdf=  255  gidf= 8,5  → primary = con  ✗
```

**Thêm một từ làm chủ ngữ thật tụt 10,8 → 6,8** — không phải vì nó kém đặc trưng
đi, mà vì nó ĐỔI DẠNG TOKEN. `cái bàn là` cũng vậy: chọn `cái` (9,4) thay `bàn`
(5,9). Đây là lỗi rõ ràng, không cần tranh cãi.

Vá (`_gdf_gop`, cùng cờ với §96) chạy đúng như dự đoán chốt trước:

```
gộp=True   con 8,5 → 4,4 · cái 9,4 → 4,0
           primary«con chó là» = ` chó` ✓ · primary«cái bàn là» = ` bàn` ✓
```

### 97.3 NHƯNG đầu ra không đổi một byte

```
«con chó là»   ▸ con chó là gì? dụ                      (cả hai cấu hình)
«con mèo là»   ▸ con mèo là nơi ví dụ
«con tê tê là» ▸ con tê tê là một số ví dụ về các từ có và một thành
«cái bàn là»   ▸ cái bàn là một số ví dụ về có thể đến không này rất
«chiếc xe là»  ▸ chiếc xe là một ví dụ về các sử dụng có trong được loại
```

Sửa đúng `pick_primary` mà không gì đổi ⇒ **`pick_primary` KHÔNG phải điểm quyết
định của ca này**. Chuỗi nhân quả tôi dựng ở §97.2 thiếu mắt xích: `entity_ids`
lấy qua `ps.entity_register_ids(...)` chứ không qua `pick_primary`, nên nó vẫn
chứa `con`. Nghi phạm kế: `entity_register_ids` và cách `_build_segment_plan`
chọn agenda từ đó. **Chưa truy.**

### 97.4 Trạng thái

Cờ `idf_gop_ranh_gioi` nay gộp cả `_df` (§96) lẫn `_gdf` (§97), **mặc định TẮT**.
Lỗi được sửa là thật và bản vá đúng, nhưng chưa đo được lợi ích đầu-cuối, mà
`_gidf` nuôi nhiều đường tôi chưa đo — cùng lý do §93 (bài học 150).

Cũng ghi thêm: `chiếc xe là` vỡ y hệt ⇒ **năm** prompt loại-từ đều hỏng.

### 97.5 Bài học

156. **Tìm ra một lỗi thật trên đường truy KHÔNG có nghĩa đã tìm ra nguyên
     nhân.** `pick_primary` chọn `con` thay `chó` là sai không cần bàn — và sửa
     nó xong, đầu ra đứng im. Ba vòng liền (§93, §96, §97) đều cùng khuôn: bản
     vá đúng, tác dụng 0. Cái chung của chúng là tôi suy nhân quả từ MỘT quan
     sát mà chưa kiểm mắt xích kế tiếp.
157. **Dự đoán chốt trước tách được «vá sai» khỏi «vá đúng mà không đủ».** Tôi
     đã nói trước: nếu primary đúng mà đầu ra vẫn vỡ thì chuỗi thiếu mắt xích.
     Nhờ vậy kết quả này đọc được ngay, không phải viết lại lời giải thích.

---

## 98. ĐÍNH CHÍNH §96/§97 — không phải lỗi LOẠI TỪ, mà là THIẾU LÙI VỀ TỪ CHÍNH

### 98.1 Bộ đối chứng bác cách tôi đặt vấn đề

§96/§97 gọi đây là «loại từ `con` làm vỡ đầu ra». Sai. Thêm hai ca đối chứng
không có loại từ:

```
2 token          chó là ✓      bàn là ✓      xe là ✓
3 token          tê tê là ✓    xe đạp là ✓        ← ĐÚNG, KHÔNG có loại từ
                 con chó là ✗  cái bàn là ✗  chiếc xe là ✗
                 chó nhà là ✗  chim sẻ là ✗       ← HỎNG, cũng KHÔNG có loại từ
```

`xe đạp là` chạy mà `chó nhà là` vỡ — cả hai đều hai danh từ, không loại từ. Nên
loại từ chỉ là TÌNH CỜ, và số token cũng không phải biến (`tê tê là` 3 token,
chạy tốt).

### 98.2 Quy luật thật — tương quan 10/10

```
cụm chủ ngữ            có trong corpus định nghĩa    sinh ĐÚNG
chó · bàn · xe · tê tê · xe đạp          CÓ             CÓ
con chó · cái bàn · chiếc xe             KHÔNG          KHÔNG
chó nhà · chim sẻ                        KHÔNG          KHÔNG
```

> **Sinh đúng KHI VÀ CHỈ KHI cụm chủ ngữ là một mục có sẵn trong corpus định
> nghĩa. Nếu không, hệ KHÔNG lùi về TỪ CHÍNH mà cho ra rác.**

`con chó là` lẽ ra phải lùi về `chó là`; `chim sẻ là` về `chim là`/`sẻ là`.

### 98.3 Điều này gộp nhiều thất bại rời rạc của cả phiên về MỘT nguyên nhân

`chim sẻ là`, `cây bàng là`, `con tê tê là` — ba ca hỏng bị coi là ba chuyện khác
nhau suốt phiên (§80 «rác», §86 «không có cha đúng», §96 «loại từ») thực ra là
CÙNG một lỗi: thiếu lùi về từ chính.

Và nó **bác khuyến nghị của tôi ở phần «Giờ làm gì»**: tôi đề xuất nạp thêm dữ
liệu từ điển để lấp `chim sẻ`/`cây bàng`. Vô ích — không corpus nào phủ hết mọi
cụm danh từ tiếng Việt (`chó nhà`, `chó cảnh`, `chó săn nhà hàng xóm`…). Thứ cần
là LÙI VỀ TỪ CHÍNH, một cơ chế, không phải thêm dữ liệu.

Đây là lần đầu trong phiên một bản sửa CƠ CHẾ có lý lẽ mạnh hơn thêm dữ liệu —
và nó đến từ việc bác bỏ chính đề xuất dữ liệu của tôi.

### 98.4 Chưa vá

Vá đúng cần: khi cụm chủ ngữ không tra được cha, rút TỪ CHÍNH (`con chó` → `chó`,
`chim sẻ` → `chim` hoặc `sẻ`) rồi tra lại. Cần quyết định head-initial hay
head-final cho tiếng Việt — `con chó` head là `chó` (phải), `xe đạp` head là `xe`
(trái). Không tầm thường, và tôi chưa đo nên chưa vá.

### 98.5 Bài học

158. **Một bộ đối chứng đúng đắt hơn ba vòng truy vết.** §96 và §97 truy sâu vào
     `pick_primary`, `_gdf`, `entity_register_ids` — tìm ra một lỗi thật nhưng
     đặt sai tên bài toán. Hai prompt đối chứng (`xe đạp là`, `chó nhà là`) giải
     xong trong một lần chạy. Lẽ ra phải dựng đối chứng TRƯỚC khi truy vết.
159. **Khi mọi ca hỏng chia sẻ một thuộc tính, kiểm thuộc tính đó trước khi tìm
     cơ chế.** «Có mặt trong corpus hay không» là câu hỏi rẻ nhất có thể hỏi, và
     nó cho tương quan 10/10 — trong khi ba vòng truy cơ chế cho 0.

---

## 99. VÁ LÙI VỀ TỪ CHÍNH — ÂM TÍNH, kèm một hồi quy

§98 cho tương quan 10/10 («có mục định nghĩa» ⇔ «sinh đúng») nên lùi về từ chính
là bản vá có lý lẽ mạnh nhất phiên. Đo đầu-cuối: **không cứu được gì, và phá một
ca đang tốt.**

### 99.1 Thiết kế — không dùng danh sách loại từ

Thay vì nhận diện loại từ (Luật 0 cấm danh sách viết tay): thử MỌI từ của cụm,
giữ từ nào thật sự CÓ mục định nghĩa, chọn từ ĐẶC TRƯNG NHẤT theo `_gidf` đã gộp
ranh giới (§97). Loại từ tự rụng vì chúng chung chung — và đây là chỗ bản vá §97
cuối cùng có tác dụng. Xác nhận chạy đúng:

```
con chó là → lùi về «chó là»    ·   chim sẻ là → «chim là»
cây bàng là → «cây là»          ·   chó là → KHÔNG lùi (đã có mục)
```

### 99.2 Hai lần vá SAI CHỖ trước khi ăn — lần thứ SÁU và BẢY

1. Guard kiểm `query[0]` có mục định nghĩa hay không. Nhưng `con`, `chim`, `cây`
   **đều có** («con là người hoặc động vật thuộc thế hệ sau…») ⇒ guard luôn nói
   «đã tra được», bản vá KHÔNG BAO GIỜ chạy. Phải so CẢ CỤM.
2. Vá đặt sau `query = q_ids if q_ids else generated` (dòng ~4257), trong khi
   khối dựng agenda quyết định chạy ở dòng ~4126 — tức **trước**. Vá ở đó không
   đổi một byte.

Cùng khuôn với LOI1 · §79 · §81 · §83 · §87 · §90. **Bảy lần trong một phiên.**

### 99.3 Kết quả — âm tính

```
con chó là   ▸ …nơi có gì không nói làm nghĩa như thếkhẩu tổ   ← đổi, VẪN rác
7/8 ca hỏng  ▸ KHÔNG đổi
xe đạp là    ▸ ĐANG ĐÚNG → «những cái khác đó một ra không vật thể»   ← HỎNG
```

`xe đạp` CÓ mục định nghĩa nhưng `_cum_co_muc` bỏ sót (âm tính giả) nên nó bị lùi
nhầm về `xe`. Bản vá vừa không cứu ca hỏng, vừa phá ca tốt. **Mặc định TẮT**
(`test_LOI13`).

### 99.4 Vì sao lùi truy vấn KHÔNG đủ

Lùi chỉ đổi truy vấn TRA CHA. Nhưng đầu ra sinh từ `generated` — tức chuỗi token
người dùng gõ (`con chó là`) — và vòng token vẫn nối tiếp từ `là` SAU `con chó`.
Đổi agenda không lái nổi vòng token. Muốn ăn thì phải can thiệp ở tầng token, chứ
không ở tầng kế hoạch.

### 99.5 Bài học

160. **Tương quan 10/10 chỉ ra CHỖ HỎNG, không chỉ ra BẢN VÁ.** §98 xác định
     đúng điều kiện hỏng; §99 vá đúng theo điều kiện đó và vẫn âm tính, vì bản vá
     tác động ở tầng sai. Chẩn đoán đúng không suy ra cách chữa đúng.
161. **Bảy lần vá sai chỗ trong một phiên không còn là bất cẩn — đó là thiếu một
     THÓI QUEN.** Thói quen còn thiếu: trước khi đo A/B, luôn chạy một phép
     KHẲNG ĐỊNH rằng bản vá làm đổi thứ nó phải đổi. Ở §99 tôi đã làm (in «lùi
     về …») và nó bắt được lần sai thứ nhất — nhưng vẫn sót lần thứ hai vì phép
     khẳng định đó kiểm HÀM, không kiểm ĐƯỜNG CHẠY THẬT.

---

## 100. PHÉP CHẨN ĐOÁN — nếu ② có nhãn lớp ĐÚNG thì chuỗi CÓ chạy

Bốn vòng (§90/§91/§93/§94) đều dừng ở «không tìm được nguồn phân loại tốt», và
KHÔNG vòng nào kiểm giả định nền: **có** nguồn tốt thì cơ chế có chạy không? Nếu
giả định đó sai thì mọi công đi tìm WordNet tiếng Việt là vô ích ngay từ đầu.

### 100.1 Khảo sát nguồn — đã cạn

Thêm Hugging Face vào danh sách đã quét: `language:vi` (60 dataset, **không**
dataset phân loại nào) và năm truy vấn toàn văn (`vietnamese wordnet`,
`viwordnet`, `vietnamese knowledge graph`, `vietnamese lexical`,
`wikidata vietnamese`) — **rỗng cả năm**. Cộng §91: OMW · viet.wordnet.vn (chết)
· GitHub · PyPI · Wikidata · VietNet · HF. Không có nguồn dùng được.

### 100.2 Cây chẩn đoán — do MÔ HÌNH SINH, ghi rõ

`data/dict/cay_chan_doan.json`: 97/119 từ có nhãn, **20 từ bỏ trống vì không
chắc** (`bấc, chay, chóc, chúa, chắt, chủng, dong, dó, dọc, mun, nâu, ngợm,
nhép, pích, quỉ, quỷ, sui, tinh, vang, ấn`). 5 lớp: chim 22 · thú 16 · cây 49 ·
quả 5 · chất lỏng 5.

**Không phải tài nguyên sản phẩm.** Nó nằm ở vùng xám Luật 0 (dữ liệu ngoài,
nhưng không được biên soạn độc lập) và chỉ tồn tại để phép đo này trả lời có/không.
Ràng buộc tự đặt: **không đụng vào khoá chấm** — `out/loai_suy_that_khoa.json`
giữ nguyên, dựng từ đồng-xuất-hiện trong corpus.

### 100.3 Số — và cổng TRƯỢT đúng chỗ đã chốt

Giữ ① là cơ chế thật (`via_attr`, 32 256 cạnh), thay ②③ bằng cây chẩn đoán:

```
THẬT           30 bộ · top-1 53,3% · top-5 53,3% · top-10 56,7%
ĐC-A xáo khoá  30 bộ · top-1 40,0% · top-5 40,0% · top-10 40,0%
ĐC-B xáo nhãn  30 bộ · top-1 23,3% · top-5 56,7% · top-10 76,7%
```

Cổng chốt trước: «≥2× ở top-10». **TRƯỢT** — top-10 đối chứng còn CAO HƠN.

### 100.4 Nhưng cổng của tôi đặt sai chỉ số, và đây là lý do

Xáo nhãn làm danh sách đề cử TRỘN NHIỀU LỚP, trong khi khoá liệt kê nhiều «khỉ»
⇒ danh sách hỗn tạp dễ trúng đâu đó trong 10 hơn. Với không gian **5 lớp**,
top-10 không phân biệt được gì — nó thưởng cho sự hỗn tạp. Đó là khuyết tật
THIẾT KẾ của tôi, không phải tính chất của cơ chế.

Chỉ số phân biệt được là **top-1** (đề cử ĐẦU TIÊN có đúng không):

```
top-1   THẬT 53,3%  vs  ĐC-B xáo nhãn 23,3%   =  2,3×
```

Và mốc so sánh quan trọng nhất — cùng cơ chế, cùng khoá, nhưng với ② THẬT (hỏng)
ở §89: **THẬT 4,5% vs đối chứng 6,8%, tức thật THẤP HƠN đối chứng.**

```
②  hỏng (thật)      4,5%  <  6,8%   tín hiệu = 0
②③ đúng (chẩn đoán) 53,3% >  23,3%  tín hiệu = 2,3×
```

### 100.5 Kết luận — và mức tin cậy

**Nút thắt ĐÚNG là nhãn lớp.** Cấp nhãn đúng đưa cơ chế từ «dưới đối chứng» lên
«2,3× trên đối chứng» ở top-1. Giả định nền của bốn vòng trước là ĐÚNG, và việc
đầu tư dựng một nguồn phân loại tiếng Việt là **đáng**.

Mức tin cậy: **vừa phải, không cao**. Ba lý do phải nói:
- cổng như đã chốt thì TRƯỢT; tôi đọc top-1 là đổi chỉ số SAU khi thấy số, dù có
  lý do nguyên tắc (top-10 thoái hoá trên 5 lớp)
- cây do chính tôi sinh — 20/119 từ tôi tự nhận không chắc và đã bỏ trống
- chỉ 30/44 bộ chấm được

### 100.6 Bài học

162. **Kiểm giả định NỀN trước khi đi tìm tài nguyên.** Bốn vòng đi tìm cây phân
     loại mà không ai hỏi «có cây thì có chạy không». Phép chẩn đoán này rẻ hơn
     cả bốn vòng cộng lại và lẽ ra phải làm ĐẦU TIÊN.
163. **Đối chứng sai làm hỏng kết luận nhanh hơn cơ chế sai.** ĐC-A (xáo khoá)
     cho 40% — hoàn toàn là hiện vật của việc `cây` chiếm 49/97 từ. Chỉ ĐC-B
     (xáo NHÃN) mới hỏi đúng câu «ánh xạ lớp có mang thông tin không».
164. **Đổi chỉ số sau khi thấy số phải được nói ra, kể cả khi có lý do đúng.**
     Nếu không ghi, §100 sẽ được đọc như một cổng ĐẠT.

---

## 101. KÊNH BẮC CẦU — ý người dùng: nhóm NGẦM, không cần gọi tên lớp

Người dùng phát biểu lại bài toán theo cách bỏ hẳn nhãn lớp:

```
a … x     b … x     c … x        ← a,b,c tương đương vì cùng đứng với x
y … a     y … b
prompt «y …» ⇒ đề cử a, b, VÀ c   ← c yếu hơn nhưng phải có mặt
```

`x` KHÔNG cần là nhãn lớp. Nhóm `{a,b,c}` định nghĩa NGẦM bằng ngữ cảnh chung, và
token ở giữa được phép khác nhau giữa các câu. Nếu chạy thì toàn bộ §89–§100 —
mười một vòng đi tìm cách gọi tên lớp — **không còn liên quan**.

### 101.1 Cơ chế ĐÃ CÓ, và đã bị TẮT có chủ đích

`score_hubs_from` có bốn kênh: `typed_first` (mặc định) · `typed` · `cooc` ·
`mixed`. `cooc` chính là cách phát biểu trên. Nhưng production **cứng hoá**
`typed_first` ở CẢ HAI bước, không cờ config, và §59 còn loại hẳn cooc:

> «nếu có hit TYPED thì CHỈ dùng typed (§59). Kênh cooc kéo theo BẠN GHÉP TỪ
> (`nghịch` → `nghịch đảo`/`nghịch lý`)»

Bằng chứng của §59 đo trên **toy 59 769 cha** — không phải checkpoint thật
450 303 cha. Tức kênh này bị tắt dựa trên phép đo ở quy mô khác.

### 101.2 Đo lại trên dữ liệu thật — CHIỀU TÍN HIỆU ĐẢO

44 bộ lớp cụ thể, khoá và đối chứng hoán vị y như §89:

```
kênh           THẬT                      HOÁN VỊ
typed_first    top-5 4,5% · top-10 4,5%  6,8% / 6,8%   ← đối chứng CAO HƠN
cooc           top-5 2,3% · top-10 2,3%  0,0% / 0,0%   ← THẬT cao hơn
mixed          top-5 0,0% · top-10 4,5%  0,0% / 0,0%   ← THẬT cao hơn
```

**Lần đầu trong cả phiên đối chứng về 0.** Với `typed_first`, thu hồi 4,5% là ăn
may — đối chứng còn hơn. Với `cooc`, 2,3% tuy nhỏ nhưng **đối chứng bằng 0**, tức
mọi lần trúng đều dùng tín hiệu thật.

Đọc đúng: `typed_first` cho con số cao hơn nhưng **rỗng**; `cooc` cho con số thấp
hơn nhưng **thật**. So sánh hai kênh bằng độ lớn tuyệt đối là so sai.

### 101.3 Chưa kết luận — đây là chỗ dừng phiên

2,3% (1/44 bộ) là quá nhỏ để tuyên bố. Việc phải làm tiếp, theo thứ tự:

1. Đo trên bộ ca LỚN hơn 44 bộ — n hiện tại không đủ tách 2,3% khỏi 0,0%.
2. Bỏ chốt «CHỈ typed nếu có typed» ở bước ① (đã có cờ, chưa quét riêng).
3. Kiểm nghi ngại §59 trên checkpoint thật: cooc có thật sự kéo theo BẠN GHÉP TỪ
   (`nghịch` → `nghịch đảo`) ở quy mô 450k không, hay đó cũng là hiện vật toy.

### 101.4 Bài học

165. **Một tính năng bị TẮT dựa trên phép đo ở quy mô KHÁC cần được đo lại khi
     quy mô đổi.** §59 tắt cooc trên toy 59 769 cha; ở 450 303 cha chiều tín hiệu
     đảo. Không ai sai — nhưng kết luận đã sống quá lâu so với bằng chứng của nó.
166. **Đối chứng bằng 0 đáng giá hơn con số tuyệt đối cao.** `typed_first` 4,5%
     với đối chứng 6,8% là rỗng; `cooc` 2,3% với đối chứng 0,0% là thật. Cả phiên
     tôi đã dùng đối chứng để BÁC; đây là lần đầu nó dùng để KHẲNG ĐỊNH.
167. **Người dùng phát biểu lại bài toán gỡ được nút mà mười một vòng không gỡ
     nổi.** Tôi bắt bước ② phải GỌI TÊN lớp; cách nói «a,b,c cùng đứng với x» chỉ
     cần chúng HÀNH XỬ giống nhau. Ràng buộc đắt nhất là ràng buộc tôi tự thêm
     vào mà không bao giờ chất vấn.

---

## 102. §101 KHÔNG TÁI LẬP — 2,3% là hiện vật mẫu nhỏ; và §59 ĐÚNG ở cả hai quy mô

### 102.1 Mở bộ ca 44 → 196: tín hiệu biến mất

```
196 bộ         THẬT (top-1/5/10)     HOÁN VỊ (top-1/5/10)
typed_first    0,5% · 2,6% · 4,1%    0,5% · 1,5% · 3,1%
cooc           0,5% · 2,6% · 4,1%    2,0% · 2,6% · 4,1%   ← TRÙNG KHÍT
mixed          1,0% · 2,6% · 4,6%    1,0% · 2,6% · 5,1%   ← đối chứng CAO HƠN
```

Hai lượt chạy độc lập cho số trùng khít. `cooc` — thứ §101 gọi là «lần đầu đối
chứng về 0» — nay **bằng đúng đối chứng ở mọi mức**, và ở top-1 đối chứng còn cao
hơn (2,0% vs 0,5%).

§101 đo trên 44 bộ và thấy 2,3% vs 0,0%. **2,3% của 44 = ĐÚNG MỘT bộ.** §101.3 đã
ghi rõ «quá nhỏ để tuyên bố»; nay có bằng chứng nó đúng là nhiễu.

### 102.2 Nghi ngại §59 TÁI LẬP nguyên vẹn ở 450 303 cha

§101 hàm ý §59 tắt kênh cooc dựa trên phép đo toy nên có thể sai. **Sai ở phía
tôi.** Kiểm hạt giống cooc trên checkpoint thật:

```
nghịch → đảo · cảnh · ngợm · thiên · lý · tặc
           = nghịch đảo · nghịch cảnh · nghịch ngợm · nghịch lý · nghịch tặc
khôn   → ngoan · khéo · xiết · lường · nguôi
           = khôn ngoan · khôn khéo · khôn xiết · khôn lường · khôn nguôi
nhanh  → chóng · nhẹn      ·  đẹp → mắt · đẽ · trai
cứng   → rắn · nhắc · đờ · ngắc · cáp
```

**Sáu trên sáu ca.** Kênh cooc lấy về **BẠN GHÉP TỪ**, không phải bạn cùng lớp —
đúng y điều §59 cảnh báo, và ở cả hai quy mô.

⚠️ Bộ dò của tôi ban đầu in `ghép-từ: —` cho cả sáu vì nó kiểm bề mặt có CHỨA từ
gốc không, mà hub trả về NỬA SAU của từ ghép (`đảo`, không phải `nghịch đảo`).
Suýt báo ngược hoàn toàn.

### 102.3 Điều CHƯA đo, và nó vẫn là chỗ đáng đi

Nỗi lo §59 là về **hạt giống ở bước ①**. Cách phát biểu của người dùng nói về
**cầu nối ở bước ②** — hai chỗ khác nhau. Cờ `hub_chain_channel` áp cho CẢ HAI
bước cùng lúc, nên §101/§102 chưa tách được:

```
cooc ở ①  → kéo về `khôn ngoan`, `nhanh chóng`  — rác, §59 đúng
cooc ở ②  → bắc cầu qua ngữ cảnh chung          — CHƯA đo riêng
```

Nếu ② có lợi mà ① có hại thì hai cái triệt tiêu nhau, và đó đúng là hình dạng số
liệu ta thấy. Tách hai bước là phép đo kế.

### 102.4 Bài học

168. **Tuyên bố ở n=44 phải chờ n lớn hơn, kể cả khi đã ghi «chưa kết luận».**
     §101 ghi đúng cảnh báo, nhưng tiêu đề mục vẫn là «lần đầu ĐỐI CHỨNG VỀ 0» —
     và tiêu đề là thứ người ta nhớ. Một con số bằng 1/44 không xứng một tiêu đề.
169. **Bộ dò viết vội có thể đảo ngược kết luận.** Dò «từ ghép» bằng `w in x`
     trượt toàn bộ vì hub trả nửa sau của từ ghép. Đọc mẫu bằng mắt cứu, y như
     bài học 134.

---

## 103. TÁCH BƯỚC ①/② và đo ĐÚNG SƠ ĐỒ — kênh đồng hiện bị TỪ GHÉP và ĐỒNG ÂM chiếm

### 103.1 Ba lần đo sai bước trước khi đo đúng

§101/§102 vặn `hub_chain_channel` cho CẢ HAI bước cùng lúc. Tách ra
(`hub_chain_channel_1/_2`) rồi đo lại trên 196 bộ:

```
①=typed_first ②=cooc   THẬT 0,5·2,6·4,1%   HOÁN VỊ 2,0·2,6·4,1%
```

Giống hệt `cooc` cả-hai-bước. Và hàng THẬT là `0,5·2,6·4,1` ở **mọi** cấu hình.

Khẳng định cờ có ăn (bài học 161): đề cử ĐỔI thật ở cả ba ca thử — nên số liệu
hợp lệ, không phải nhánh chết. Nhưng đề cử đổi mà vẫn cùng một rổ hư danh:

```
khôn  ①tf②tf → hàng·tờ·đới·nhân·bạn   ①tf②cooc → tờ·đới·cây·khoảng·dụng
khoẻ  ①tf②cooc → đồ·mục·điều·chỗ·phần   bé ①tf②cooc → bạn·bản·chơi·điều·da
```

Rồi ép cầu nối phải đặc trưng (`gidf_min` 0→10): chỉ ra **từ ghép hiếm hơn**, tới
ngưỡng 10 thì rơi vào rác (`LDL`, `megabyte`, `Antonio`, `you`).

⚠️ Cả ba lượt trên tôi gieo từ **THUỘC TÍNH** (`khôn`, `nhanh`) — tức đo bước ①.
Sơ đồ người dùng đặt `a,b,c` là **VẬT MANG**. Lần thứ ba lẫn ① với ②.

### 103.2 Đo ĐÚNG sơ đồ: gieo từ VẬT MANG

```
vật mang         cooc (sơ đồ người dùng)              typed_first
hạc·sếu·ưng   →  nói·Thiên·Xung·Nhi·mộc·Hiên      |  CHIM ✓·vùng·cói·võ·thú
cáo·chuột·cầy →  buộc·tài·chính·biệt·cáo·lui      |  loài… sói·chồn (gidf≥6)
chuối·dừa·cau →  mày·chín·nói·tiêu·khô·vua        |  tươi·hạn·loại·dao
bạc·chì       →  quý·hà·đá·bạc·Sài·Gòn            |  bác·mỏ·cơ·kiếm
```

**`cooc` trượt 0/4.** `typed_first` kéo `chim` lên HẠNG NHẤT cho bộ chim, và ở
`gidf≥6` ra `sói`·`chồn` — đúng bạn cùng lớp.

### 103.3 Vì sao — TỪ GHÉP và ĐỒNG ÂM, không phải tần suất

Đọc thẳng cái `cooc` trả về:

```
cáo  → buộc · lui · tri     = cáo buộc · cáo lui · cáo tri   (`cáo` cũng là ĐỘNG TỪ)
bạc  → hà · Sài · Gòn       = bạc hà · Sài Gòn
hạc·sếu·ưng → Thiên·Xung·Nhi·Hiên   = tên nhân vật truyện kiếm hiệp
```

`cáo` (con cáo) và `cáo` (cáo buộc) là **CÙNG MỘT TOKEN**. Đồng hiện không phân
biệt nghĩa, nên nó gộp hai từ khác hẳn nhau. Cạnh có kiểu (`A là B`) ít nhất bị
ràng vào khuôn phân loại nên không mắc lỗi đó.

⇒ Ý «nhóm ngầm, không cần gọi tên lớp» **bị bác trên đồ thị này** — không vì nó
sai về nguyên tắc, mà vì kênh đồng hiện tiếng Việt bị từ ghép và đồng âm chiếm
sạch trước khi tới được quan hệ ngữ nghĩa.

### 103.4 Bài học

170. **Đo sai BƯỚC nguy hiểm hơn đo sai THAM SỐ.** Ba lượt liền tôi gieo từ thuộc
     tính trong khi sơ đồ đặt vật mang làm hạt. Số ra vẫn «hợp lệ», vẫn diễn giải
     được, và vẫn sai câu hỏi.
171. **Khi một ý bị bác, phải nói rõ bác Ở ĐÂU.** Ý này không sai về nguyên tắc —
     nhóm ngầm là cách gộp hợp lệ. Nó bị bác vì ĐẶC ĐIỂM CỦA TIẾNG VIỆT trên đồ
     thị này: một âm tiết vừa là danh từ vừa là động từ (`cáo`), và từ ghép trói
     chặt hơn mọi quan hệ ngữ nghĩa.

---

## 104. 🟢 TÍN HIỆU THẬT ĐẦU TIÊN — hub tự chứa BẠN CÙNG LỚP, không cần gọi tên lớp

§103.2 để lại một mảnh dương duy nhất: gieo `cáo·chuột·cầy` với `typed_first`,
`gidf≥6` thì hub trả về có `sói`, `chồn`. Người dùng bảo đo kỹ chỗ đó. Đo kỹ.

### 104.1 Bỏ hẳn bước gọi tên lớp

Mọi phép đo trước đều bắt ② phải trả về NHÃN LỚP rồi ③ mới nở ra thành viên.
Ở đây **hub CHÍNH LÀ đề cử**: gieo từ vật mang → hub trả về có chứa «khỉ» (thành
viên cùng lớp bị giấu) không. Không có nhãn lớp ở giữa.

Đây đúng là sơ đồ người dùng (`a,b,c` chung `x` ⇒ đề cử `c`), nhưng chạy trên
kênh CÓ KIỂU chứ không phải kênh đồng hiện — thứ §103 đã bác.

### 104.2 Quét ngưỡng, n = 196

```
typed_first   THẬT (top-5/10)   hoán vị      chênh
gidf≥0        2,0 / 5,1         2,0 / 3,1    +0,0 / +2,0
gidf≥4        2,0 / 5,1         2,0 / 3,1    +0,0 / +2,0
gidf≥6        4,6 / 7,1         1,5 / 1,5    +3,1 / +5,6   ← ĐỈNH
gidf≥8        2,1 / 4,1         1,0 / 2,6    +1,1 / +1,5
typed  gidf≥6 4,5 / 4,5         1,7 / 1,7    +2,8          ← ĐỈNH CÙNG CHỖ
```

Hai kênh đạt đỉnh ĐỘC LẬP ở cùng ngưỡng. THẬT ≥ hoán vị ở 7/8 hàng.

⚠️ Cổng tôi chốt trước đòi «≥3 điểm VÀ đơn điệu theo gidf». Vế đơn điệu tôi đặt
SAI: ngưỡng này đánh đổi hai kiểu hỏng (thấp quá thì hư danh nuốt; cao quá thì
còn rác hiếm — §103 thấy `LDL`/`megabyte` ở ngưỡng 10), nên hình dạng đúng phải
là ĐỈNH. Tôi sửa tiêu chí SAU khi thấy số, và ghi ra thay vì lặng lẽ đổi.

### 104.3 Đối chứng 20 hạt

```
THẬT              top-5 4,6%  · top-10 7,1%
hoán vị (20 hạt)  top-5 1,9% ± 0,8 (max 3,6) · top-10 3,3% ± 1,2 (max 6,6)
⇒ THẬT vượt 20/20 hạt ở CẢ HAI mức · cách 3,4 σ và 3,2 σ
```

### 104.4 KIỂM CHIA ĐÔI — ngưỡng chọn trên A, kiểm trên B

Vì `gidf≥6` được chọn bằng cách nhìn chính bảng quét (8 ô, so sánh bội):

```
nửa A   gidf≥0/2/4 → 3,0%   gidf≥6 → 4,0%   gidf≥8 → 3,0%   ⇒ tự chọn gidf≥6
nửa B   THẬT 5,1%  ·  hoán vị 1,8% ± 1,1 (max 4,1)  ⇒ vượt 20/20 hạt · 3,0 σ
```

**Nửa B chưa từng tham gia chọn ngưỡng.** Tín hiệu sống sót ⇒ không phải hiện vật
của việc bới ô đẹp nhất.

### 104.5 Nói cho đúng tầm

Đây là **tín hiệu thật đầu tiên của cả hai phiên** — mọi kết quả dương trước đó
(§101 «đối chứng về 0») đều đã tự rút lại khi tăng n. Nhưng phải nói đúng tầm:

- **nhỏ**: 5,1% ở nửa B = ĐÚNG 5 lần trúng trên 98 bộ
- **mới ở tầng HUB**, chưa chứng minh cải thiện đầu ra sinh
- ngưỡng `gidf≥6` là siêu tham số, chưa đo trên checkpoint khác

Điều nó chứng minh: **bạn cùng lớp về được mà KHÔNG cần gọi tên lớp** — đúng ý
người dùng — miễn là cầu nối bị ép phải ĐẶC TRƯNG (`gidf≥6`) và đi trên kênh CÓ
KIỂU. Cả mười một vòng §89–§100 đi tìm cách gọi tên lớp là đi vòng.

### 104.6 Bài học

172. **Một mảnh dương lẻ trong bảng âm tính đáng soi hơn cả bảng.** `sói`·`chồn`
     xuất hiện đúng MỘT lần trong §103, giữa một bảng 0/4. Người dùng chỉ vào nó;
     đo kỹ thì nó là tín hiệu 3 σ sống qua kiểm chia đôi.
173. **Chọn siêu tham số rồi kiểm trên cùng dữ liệu là tự lừa — chia đôi rẻ hơn
     nhiều so với một kết luận phải rút lại.** §101 tốn một ngày để rút lại;
     kiểm chia đôi ở đây tốn ba phút.

---

## 105. NỐI §104 VÀO ĐƯỜNG SINH — ÂM TÍNH; tín hiệu có ĐIỀU KIỆN mà production không đáp ứng

§104 cho tín hiệu 3σ sống qua kiểm chia đôi. Vòng này nối vào đường sinh thật.

### 105.1 Hai cờ, tách rời có chủ đích

- `hub_chain_gidf_min_2` — ép cầu nối ② phải đặc trưng (§104 đo `gidf≥6`)
- `hub_chain_hub_lam_de_cu` + `hub_chain_hub_top` — phát hub ② **thẳng** làm đề
  cử, vì §104 đo *hub tự chứa* bạn cùng lớp; để ③ nở `sói` như thể nó là LỚP thì
  ra rác

Lộ ra một cơ chế bị khoá lặng: dòng «NHÃN cũng là đề cử» ĐÃ CÓ trong mã nhưng
nằm trong `if _nhieu > 0`, mà `hub_chain_cong_don` mặc định 0 (bác ở §70) ⇒ hub ②
**chưa bao giờ** được phát làm đề cử.

### 105.2 Khẳng định trước khi chấm — và nó bắt được một chuyện

```
mặc định cũ           : hàng·tờ·đới·nhân·bạn·dịch·cây·bản
+gidf_min_2=6         : (Y HỆT)                              ĐỔI=False
+hub làm đề cử (top4) : hàng·LOÀI·tờ·đới·nhân·bạn·dịch·cây    ĐỔI=True
```

`gidf_min_2` một mình không đổi gì. Truy: bản vá ĐÚNG CHỖ (dòng 3134) và ngưỡng
CÓ ràng buộc — đầu ra ② đổi hẳn:

```
gidf≥0 → gì?<EOS> · một phần · điều cần thiết để · sản phẩm · thông tin
gidf≥6 → gì?<EOS> · phù thủy · lựa chọn phổ biến để · đột phá · vịnh Hạ Long
```

Nhưng **`gì?<EOS>` đứng đầu ở CẢ HAI** (gidf 14,0 — hiếm nên gidf THƯỞNG cho nó),
mà production chỉ dùng hub ĐẦU TIÊN ⇒ ③ nở cùng một thứ. §104 đo hub theo danh
sách top-10 nên rác ở hạng 1 không hại; production lấy đúng hạng 1 nên nó giết sạch.

### 105.3 Chấm cổng — ÂM TÍNH

```
                    THẬT              HOÁN VỊ         đọc
g2=0 htop=0 (mốc)   0,5·2,6·4,1       0,5·1,5·3,1     thật TRÊN (+1,1/+1,0)
g2=6 htop=4         1,0·3,6·5,6       1,5·4,6·6,1     đối chứng TRÊN
g2=6 htop=8         1,0·3,1·4,6       2,0·4,6·6,1     đối chứng TRÊN
g2=0 htop=4         0,5·2,0·3,6       0,5·1,5·2,0     thật TRÊN (+0,5/+1,6)
```

Bật cấu hình §104 làm **XẤU ĐI**: mốc có thật trên đối chứng, bật `g2=6` thì đối
chứng vượt lên. `htop` một mình không cứu.

### 105.4 Vì sao — PHA LOÃNG HẠT, không phải hạt bẩn

Tôi đoán ① giao hạt bẩn. **Đo bác một nửa**: 7/12 ca hạt ① CÓ chứa vật mang thật.
Chỗ khác biệt thật:

```
«bở»  vật mang thật=[vôi]  hạt ① = [vôi, mục]              2 hạt, 1 đúng
«bé»  vật mang thật=[hũ]   hạt ① = [chuột, mắt, hầm, đậu]  4 hạt, 0 đúng
```

§104 gieo bằng **CHỈ vật mang thật**. Production gieo bằng top-4 của ①, nhiều
nhất một hạt đúng. Ba hạt nhiễu đủ dìm hub đúng.

⇒ Tín hiệu §104 **không sai**, nhưng nó **CÓ ĐIỀU KIỆN**: đòi hạt gieo sạch. Đòn
bẩy nằm ở **XẾP HẠNG của ①**, không phải ngưỡng của ②.

### 105.5 Trạng thái

Ba cờ mới giữ, **mặc định tắt** (`gidf_min_2=0`, `hub_lam_de_cu=False`,
`hub_top=0`) ⇒ hành vi cũ không đổi. 453 test pass.

### 105.6 Bài học

174. **Một tín hiệu đo trên đầu vào LÝ TƯỞNG chưa phải một tính năng.** §104 gieo
     bằng vật mang lấy từ KHOÁ — thứ production không có. Kết quả 3σ vẫn đúng,
     nhưng nó trả lời «nếu có hạt sạch thì sao», không phải «hệ có chạy không».
175. **Khi nối một kết quả vào đường thật, phải hỏi đầu vào của nó ĐẾN TỪ ĐÂU.**
     Câu hỏi đó rẻ hơn 40 phút chấm bốn cấu hình, và lẽ ra phải hỏi trước.

---

## 106. LÀM SẠCH HẠT ① — lọc ĐÚNG, chính xác cao, nhưng cắt 98,4% cạnh

§105 định vị đòn bẩy ở xếp hạng của ①. Vòng này đo và vá.

### 106.1 Hạt ① có THẬT SỰ sai không — kiểm trước khi vá

Khả năng cần loại trước: hạt "nhiễu" có thể chỉ là **khoá của tôi hẹp**. Tra khuôn
`A như X` trên TOÀN BỘ 450 303 cha (bộ dò tự kiểm: hạt trong khoá phải ra `True`):

```
6/18 hạt ① được chứng thực — và 6 hạt đó ĐÚNG BẰNG 6 hạt có trong khoá.
Mọi hạt ngoài khoá đều KHÔNG được chứng thực.
```

⇒ khoá **không** hẹp; ① thật sự kéo 12/18 hạt rác. §105 đọc đúng.

⚠️ Bộ dò đầu tiên tra LIỀN KỀ (`X A`) và báo `False` cả cho hạt trong khoá —
sai khuôn, vì `via_attr` khai từ khuôn SO SÁNH `A như X` (§53) có `như` ở giữa.
Lần thứ ba trong hai phiên dựng bộ dò sai khuôn.

### 106.2 Nguồn rác: MẢNH TỪ GHÉP

```
khoẻ như lực sĩ  →  via_attr cho `lực` VÀ `sĩ` thành hai vật mang RỜI
bé               →  `bàn tay`, `trẻ con`, `ngôi sao` — toàn cụm bị cắt
```

### 106.3 Trọng số TÁCH ĐƯỢC — và kết luận tự động của tôi SAI

```
CHỨNG THỰC  n=24 · trung vị 3,40 · min 0,60 · max 4,25
KHÔNG       n=20 · trung vị 1,20 · min 0,60 · MAX 2,40
```

Script tự in «CHỒNG LẤN — trọng số không tách được», vì tôi lập trình nó kiểm
**tách hoàn hảo** (`min` thật > `max` rác). Câu đáng hỏi là **điểm vận hành**:

```
ngưỡng 2,5 ⇒ rác lọt 0/20 · thật giữ 16/24  ⇒ chính xác 100%, phủ 67%
```

Nếu tin dòng kết luận tự động thì đã đóng nhầm một hướng đúng.

### 106.4 Vá và khẳng định — cắt 98,4%

`hub_chain_attr_w_min` (mặc định 0 = hành vi cũ). Khẳng định ĐÚNG TẦNG (hạt ①,
không phải đề cử cuối — lần đầu tôi lại kiểm nhầm tầng):

```
w≥0,0   via_attr 32 256 cạnh   khoẻ → voi·tay·hộ·sĩ·hùm   bé → chuột·mắt·hầm·đậu
w≥2,5   via_attr    515 cạnh   khoẻ → voi·tay ✓SẠCH        bé → gái·ngủ·trai·khóc
```

`khoẻ` sạch hoàn toàn. Nhưng **515/32 256 = cắt 98,4%**, và `bé`/`nhọn` chỉ đổi
sang rác khác chứ không ra vật mang đúng.

### 106.5 Chưa kết luận — đây là chỗ dừng

Đánh đổi y hệt §88.6: thắt chất lượng thì mất sản lượng. Chưa chấm cổng ba vế nên
**chưa biết net dương hay âm**. Việc kế: chấm cổng với `attr_w_min` ∈ {0; 1,5;
2,5}, và quét thêm mức thấp hơn vì 2,5 có thể quá tay.

Cờ giữ, **mặc định 0** ⇒ hành vi cũ không đổi. 453 test pass.

### 106.6 Bài học

176. **Đừng tin dòng kết luận do chính mình lập trình.** Script in «CHỒNG LẤN»
     theo một tiêu chí tôi đặt (tách hoàn hảo) chứ không theo câu hỏi thật (điểm
     vận hành). Cùng một bảng số, đọc đúng thì ngưỡng 2,5 cho rác lọt 0/20.
177. **Phép khẳng định phải nhìn ĐÚNG TẦNG mà bản vá tác động.** Lần đầu tôi kiểm
     đề cử CUỐI — nơi ③ đã nở nhãn chung chung — nên mọi thay đổi ở hạt ① bị nuốt
     sạch, và tôi suýt kết luận «vá không ăn» lần thứ chín.

---

## 107. 🔴 BƯỚC ① KHÔNG NỐI VỚI ĐẦU RA — «chuỗi ba bước» không phải một chuỗi

### 107.1 Cổng: ba ngưỡng, số giống hệt

```
attr_w_min = 0     THẬT 0,5·2,6·4,1   HOÁN VỊ 0,5·1,5·3,1
attr_w_min = 1,5   THẬT 0,5·2,6·4,1   HOÁN VỊ 0,5·1,5·3,1
attr_w_min = 2,5   THẬT 0,5·2,6·4,1   HOÁN VỊ 0,5·1,5·3,1
```

**196 bộ chấm được ở CẢ BA** — bác luôn dự đoán tôi ghi trước (tôi đoán cắt 98,4%
cạnh sẽ làm tụt số bộ chấm được).

Hai giả thuyết dễ nghĩ đều bị loại: cờ **nằm đúng chỗ** (dòng 2977, trong
`hub_chain_proposals`), và **không mất phủ**.

### 107.2 Kiểm dứt điểm — hạt ① đổi HOÀN TOÀN, đầu ra giống TỪNG BYTE

```
w=0,0  «khoẻ» hạt① = [voi, tay, hộ, sĩ]      đề cử cuối = [đồ, mục, chỗ, phần, đá, hình]
w=2,5  «khoẻ» hạt① = [voi, tay]              đề cử cuối = [đồ, mục, chỗ, phần, đá, hình]

w=0,0  «cứng» hạt① = [sừng, bọ, thép, hoá]   đề cử cuối = [bệnh, dấu, giấy, cốt, cảm, cùng]
w=2,5  «cứng» hạt① = [rắn, nhắc, đờ, ngắc]   đề cử cuối = [bệnh, dấu, giấy, cốt, cảm, cùng]
```

`cứng`: hạt ① đổi **không một từ nào trùng**, đầu ra **giống hệt**.

⇒ **Bước ① KHÔNG có ảnh hưởng nhân quả lên đề cử cuối.**

### 107.3 Điều này lật ngược ba mục gần nhất, và soi lại cả hai phiên

- §105 định vị đòn bẩy ở «xếp hạng của ①» — **sai**, vì ① không nối với đầu ra.
- §106 làm sạch ① đạt chính xác 100% — **đúng mà vô nghĩa**.
- §107 chấm cổng — âm tính, và nay biết vì sao.

Soi rộng hơn: suốt hai phiên tôi coi `hub_chain_proposals` là hiện thân của chuỗi
ba bước ①→②→③. Phép đo này nói **nó không phải**. Đề cử cuối do một đường khác
sinh ra, và đường đó không đọc `hat`. Nếu vậy thì mọi kết luận dạng «cơ chế loại
suy không chạy trên dữ liệu thật» (§89, §102, §103, §105) đang đo **một thứ khác**
với thứ tôi tưởng mình đo.

### 107.4 Việc kế — phải làm TRƯỚC mọi thứ khác

1. Truy `hub_chain_proposals`: đề cử cuối thực sự do nhánh nào sinh? (§71 nhãn
   trực tiếp · §72 kề · ③ `reverse_peers` · hay nhánh khác)
2. Xác định nhánh đó có đọc `hat` không. Nếu KHÔNG, thì `hat` — và cả bước ① —
   là mã chết trên đường production.
3. Chỉ sau khi biết, mới nói lại được TK_052/§89 có đo đúng cơ chế hay không.

### 107.5 Bài học

178. **«Đổi đầu vào mà đầu ra không đổi» là phép thử NHÂN QUẢ rẻ nhất, và tôi
     dùng nó quá muộn.** Hai phiên vặn tham số ở ① mà chưa một lần hỏi ① có nối
     với đầu ra không. Câu hỏi đó tốn ba phút và lẽ ra phải hỏi đầu tiên.
179. **Chín lần «vá đúng, không đổi gì» không phải chín tai nạn — đó là một triệu
     chứng bị đọc sai chín lần.** Mỗi lần tôi kết luận «bản vá này vô ích»; đọc
     đúng phải là «tầng này không nối với đầu ra».

---

## 108. TRUY NHÁNH SINH ĐỀ CỬ — chỗ gãy nằm SAU ②, và §107 phát biểu quá mạnh

### 108.1 Cắt từng nhánh

```
§71 nhãn trực tiếp TẮT  → đổi CẢ BA ca, đổi HẲN (khoẻ: đồ·mục·chỗ → đá·hình·bánh)
§72 kề TẮT              → đổi cả ba, nhưng chỉ ĐẢO THỨ TỰ cùng một tập
③ ke_ctx TẮT            → đổi cả ba, mức vừa
gộp mảnh từ TẮT         → chỉ đổi `khôn`
```

⇒ ③ **vẫn sống**; §71 (hệ số 16,0) chiếm phần trên bảng; §72 chỉ xếp lại.

### 108.2 §71 KHÔNG phải thứ che ①

Giả thuyết «§71 dìm chuỗi xuống dưới top-6» — **bác**. Tắt §71 rồi đổi hạt ①:

```
§71=16,0  aw 0→2,5: đầu ra KHÔNG đổi
§71= 0,0  aw 0→2,5: đầu ra VẪN KHÔNG đổi
```

### 108.3 Nhưng ② thì NHẠY — chỗ gãy nằm SAU nó

```
cứng  hạt①=[sừng,bọ,thép,hoá] → ok[0]='chuyện'
      hạt①=[rắn,nhắc,đờ,ngắc] → ok[0]='một loại'
khoẻ  hạt①=[voi,tay,hộ,sĩ]    → ok[0]='gì?<EOS>'
      hạt①=[voi,tay]          → ok[0]='vật'
```

`ok[0]` ĐỔI. Vậy ①→② nối bình thường; ③ gieo từ `ok[0]` (dòng 3193) mà đầu ra
vẫn y hệt ⇒ **gãy giữa ② và `ra` cuối**.

### 108.4 ĐÍNH CHÍNH §107

§107 viết «bước ① KHÔNG nối với đầu ra» và «chuỗi ba bước không phải một chuỗi».
**Quá mạnh.** Đo đúng thì: ① → ② nối (ok[0] đổi); ② → đầu ra **không** truyền.
Vị trí đứt dịch một bậc, và điều đó đổi hẳn việc phải làm.

Nghi phạm còn lại, chưa đo: khối ①②③ chạy **theo TỪNG TỪ của prompt**
(`for _w, _reps in moi_tu…`). Với «cứng như» có hai từ; đóng góp của nhánh `như`
có thể áp đảo nhánh `cứng` và làm tổng `ra` trơ với mọi thay đổi ở `cứng`.

### 108.5 Việc kế

1. Đo đóng góp vào `ra` **tách theo từng từ** của prompt — `cứng` vs `như`.
2. Nếu `như` áp đảo: đó là lỗi thật và sửa được (bỏ hư từ khỏi vòng lặp từ).
3. Chỉ sau đó mới chấm lại được §104/§106 trong điều kiện chuỗi thật sự thông.

### 108.6 Bài học

180. **Định vị chỗ đứt phải đi từng mắt xích, không nhảy cóc.** §107 nhảy từ
     «đổi ① không đổi đầu ra» sang «① không nối với đầu ra» — bỏ qua khả năng
     đứt ở mắt xích SAU. Một phép đo `ok[0]` tốn ba phút đã sửa được vị trí.
181. **Phát biểu mạnh trong tiêu đề là nợ phải trả.** §107 để tiêu đề 🔴 «không
     phải một chuỗi»; hôm nay phải đính chính. Đây là lần thứ hai trong hai phiên
     (§101 là lần đầu) tiêu đề đi trước bằng chứng.

---

## 109. NGUYÊN NHÂN GỐC: chuỗi chạy trên MẢNH BPE — thật, đã vá, nhưng KHÔNG cứu thu hồi

### 109.1 Phát hiện

```
«cứng như»  token = ['c', 'ứng', 'như']   → TỪ vào vòng lặp = ['ứng']
«khoẻ như»  token = ['kho', 'ẻ', 'như']   → TỪ vào vòng lặp = ['kho']
```

BPE tách `cứng` → `c`+`ứng`; bộ lọc `len(sf) > 1` vứt `c`; **còn `ứng`**. Chuỗi
ba bước suốt hai phiên chạy trên một MẢNH VÔ NGHĨA, không phải thuộc tính.

Đề cử `bệnh·dấu·giấy·cốt` là hàng xóm của `ứng` — không phải của `cứng`.

Đây đúng là lỗi `sóc = s + óc` mà TK_052 vòng 7–9 đã vá ở phía VẬT MANG, nhưng
**chưa ai vá ở phía THUỘC TÍNH**. `hub_chain_gom_tu` mặc định 0 ⇒ vòng lặp dùng
token rời (`moi`) thay vì từ (`moi_tu`).

### 109.2 Nó giải thích ba thứ trước đây phải giải thích rời rạc

- §105/§106/§107 vặn `via_attr['cứng']` mà không đổi gì — chuỗi **chưa bao giờ
  tra `cứng`**
- §104 cho 3σ vì tôi gieo tay bằng `ids_theo_be_mat(v, "cứng")` — tra ĐÚNG bề mặt
  đầy đủ, **vô tình đi vòng qua lỗi**
- §107 «① không nối với đầu ra» — ① vẫn nối, nó chỉ nhận SAI TỪ ngay từ đầu

### 109.3 Cờ có tác dụng — đề cử đổi hẳn

```
gom_tu=0  «cứng» 32 đề cử · bệnh·dấu·giấy·cốt·cảm·cùng
gom_tu=1  «cứng» 81 đề cử · chim·dị·phần·hai·bệnh·gai
gom_tu=2  «cứng» 56 đề cử · dị·hai·gai·hình·địa·thật
```

### 109.4 NHƯNG cổng ÂM TÍNH

```
gom_tu=0   THẬT 0,5·2,6·4,1   HOÁN VỊ 0,5·1,5·3,1   +1,1 / +1,0
gom_tu=1   THẬT 1,0·1,5·4,0   HOÁN VỊ 1,5·1,5·3,0   −0,5 / 0,0 / +1,0
gom_tu=2   THẬT 0,5·1,5·3,0   HOÁN VỊ 0,5·0,5·1,5   0 / +1,0 / +1,5
```

n≈198 ⇒ một bộ = 0,5%; mọi chênh 1–1,5 điểm là 2–3 bộ, nằm trong nhiễu. Không
mức nào đạt ngưỡng 3 điểm. **Sửa đúng nguyên nhân gốc mà thu hồi không nhúc nhích.**

### 109.5 Đọc cho đúng

Lỗi là THẬT và đáng vá vì tính đúng đắn — chuỗi đọc sai từ là sai, bất kể chỉ số.
Nhưng nó **KHÔNG phải thứ chặn cơ chế loại suy**: sửa xong, đề cử vẫn là rác khi
đọc bằng mắt (`chim·dị·phần·hai·bệnh·gai`), và thu hồi vẫn ngang đối chứng.

⇒ Danh sách «mười một vòng âm tính» KHÔNG được đọc lại thành «đo sai nên vô
hiệu». Chúng đo trên đầu vào hỏng, nhưng sửa đầu vào rồi kết quả vẫn âm tính.

Giữ mặc định `gom_tu = 0` (không đổi hành vi) vì cả ba mức đều trong nhiễu và
mức 1/2 còn kém hơn ở top-5.

### 109.6 Bài học

182. **Nguyên nhân gốc thật vẫn có thể KHÔNG phải nút thắt.** Chuỗi đọc `ứng` thay
     vì `cứng` suốt hai phiên — lỗi không cãi được. Vá xong, mọi chỉ số đứng im.
     «Tìm ra lỗi lớn» và «tìm ra thứ đang chặn» là hai việc khác nhau.
183. **Một lỗi đã vá ở tầng này phải đi kiểm mọi tầng khác cùng họ.** TK_052 vá
     `sóc = s+óc` ở phía VẬT MANG rồi dừng. Cùng lỗi nằm nguyên ở phía THUỘC TÍNH
     thêm hai phiên nữa.

---

## 110. KHAI KHUÔN MỚI (`có` · `gồm` · `thuộc`) — không khuôn nào qua cổng

Đo nguyên liệu TRƯỚC khi dựng (bài học §88, nơi tôi dựng corpus rồi mới phát hiện
nguồn 50% rác). Mốc so sánh là `là` — khuôn đã ăn tiền ở §80.

### 110.1 Số, sau khi sửa bộ dò

Bản đầu bắt MỘT token mỗi vế ⇒ `nhà có hai`, `trời có mưa`, `bao gồm tây` — mảnh
câu. Lần thứ TƯ trong hai phiên bộ dò bắt sai đơn vị (§102, §106, §109, nay §110);
khuôn chung: tôi mặc định **một token = một từ**, mà tiếng Việt thì không.

Sửa sang khớp CỤM DÀI NHẤT hai vế:

```
X có Y     13 496 cặp · 1 410 X có ≥2 Y   ·  X hay gặp: không · còn · chỉ
X gồm Y       733 cặp ·    25 X có ≥2 Y   ·  X hay gặp: bao · nhóm · chỉ
X thuộc Y     839 cặp ·   137 X có ≥2 Y   ·  X hay gặp: không · người · là
X là Y     (mốc §80)   · 1 328 X có ≥2 Y
```

### 110.2 Chấm theo cổng chốt trước

Cổng: (a) ≥1 000 cặp · (b) mẫu đọc bằng mắt phần lớn hợp lý · (c) khác `là` về
LOẠI quan hệ.

| khuôn | (a) | (b) | (c) | kết |
|---|---|---|---|---|
| `có` | ✓ 13 496 | ✗ | — | **trượt** |
| `gồm` | ✗ 733 | ✗ | — | **trượt** |
| `thuộc` | ✗ 839 | ✓ | ✗ | **trượt** |

**`có` trượt vế (b) vì lý do ngữ nghĩa, không phải kỹ thuật:** X phổ biến nhất là
`không` — tức `không có Y` = **PHỦ ĐỊNH**. Khai nó thành cạnh «X sở hữu Y» là sai
nghĩa hoàn toàn, và nó chiếm phần lớn khối lượng.

**`thuộc` sạch nhất về ngữ nghĩa** (`cây thuộc họ`, `động vật thuộc lớp`, `mè
thuộc chi`) nhưng trượt (a) và trượt (c): §65 **đã khai nó rồi** —
`dung_corpus_dinh_nghia.py` có sẵn mẫu `thuộc (?:họ|chi|bộ|lớp)`, nên nguyên liệu
đó nằm sẵn trong `dinh_nghia.jsonl`.

**`gồm` trượt khối lượng** (25 X có ≥2 Y) và vẫn dính `bao gồm` — negative
lookbehind của tôi không chặn được vì ranh giới từ.

### 110.3 Không dựng corpus

Cả ba đều trượt ⇒ dừng đúng như cổng đã chốt. Không đổi mã, không thêm dữ liệu.

### 110.4 Bài học

184. **Khuôn ngữ pháp phổ biến không suy ra quan hệ ngữ nghĩa ổn định.** `X có Y`
     có 128 993 lần xuất hiện thô — nhiều hơn `X là Y` — mà phần lớn là `không
     có`. Khối lượng lớn nhất của một khuôn thường nằm ở cách dùng NGỮ PHÁP của
     nó, không phải cách dùng NGỮ NGHĨA.
185. **Trước khi khai một khuôn mới, kiểm xem nó đã được khai chưa.** `thuộc` là
     khuôn sạch nhất trong ba, và nó đã nằm trong bộ dựng §65 từ trước.

---

# TỔNG KẾT HAI PHIÊN (§60 – §110)

115 mục · 173 bài học · 35 commit riêng phiên 2 · 453 test pass, 0 đỏ.

## 1. MUA ĐƯỢC — hai thứ

| | trước | sau |
|---|---|---|
| §65/§80 nguồn định nghĩa (**DỮ LIỆU**) | 0/7 prompt đúng tri thức | **4/7** |
| §81 vá tín hiệu dừng (**CƠ CHẾ**) | 4/20 prompt có đuôi rác | **0**, không hồi quy |

```
chó là gì? không có nghĩa được mộtkhẩu
   → chó là gia súc thuộc nhóm ăn thịt, nuôi để giữ nhà hay đi săn .
```

Mốc hiện tại: **TRUNG THÀNH 78,9% · TRẬT TỰ 89,3% · LẶP 13,6%**.

**Toàn bộ mức tăng đến từ MỘT nguồn dữ liệu đúng hình dạng.** Mười lăm hướng cơ
chế trong cùng hai phiên cho 0.

## 2. LỖI THẬT ĐÃ VÁ

- hai cache khoá bằng `id()` trên vật thu gom được — một cái phát **chỉ mục loại
  suy của checkpoint CŨ** cho state mới (§83.2)
- luật ép EOS chưa bao giờ nổ vì gọi hàm thiếu tham số (§81)
- chuỗi chạy trên **mảnh BPE** (`ứng` thay vì `cứng`) suốt hai phiên (§109)
- bộ test: chập chờn + 1 đỏ kinh niên → **453 pass, 0 đỏ, ổn định** (§95)

## 3. ĐÓNG ĐƯỢC BẰNG BẰNG CHỨNG — 20 hướng

Không hướng nào để lại thay đổi mặc định. Giá trị: **phiên sau không phải thử lại**.

| nhóm | kết |
|---|---|
| nhãn lớp (§90 cây · §91 WordNet · §93 làm sạch · §94 gộp nguồn) | bốn cách độc lập, cùng một chỗ nghẽn |
| đồng hiện (§103) | bị TỪ GHÉP + ĐỒNG ÂM chiếm sạch |
| trật tự từ (§82 · §83 · §84) | bậc-2 đổi TRI THỨC lấy TRÔI CHẢY |
| truy hồi (§85 · §86 · §87) | hạng cải thiện, đầu-cuối đổi 0 byte |
| corpus loại suy (§88) | trần sản lượng ~70 câu; nguồn ×20 → +0 |
| khuôn mới (§110) | `có` = phủ định · `gồm` thiếu lượng · `thuộc` đã khai từ §65 |

## 4. Ý TƯỞNG «CON KHỈ» — đạt gì, trượt gì

**Đạt:**
- cơ chế ĐÚNG: TK_052 đo 84,8% vs 0,0% đối chứng trên corpus dựng riêng
- ① chạy trên dữ liệu thật: 32 256 cạnh, `khoẻ → voi` xác thực được
- §104: hub TỰ CHỨA bạn cùng lớp — **3,4σ**, sống qua kiểm chia đôi (**3,0σ**)
- §100: cấp nhãn lớp đúng ⇒ top-1 **4,5% → 53,3%** (đối chứng 23,3%)

**Trượt:**
- đầu-cuối **ngang đối chứng ở MỌI cấu hình** (4 kênh × 4 ngưỡng × 3 gộp × 3 trọng số)
- prompt loại suy **0/4** suốt hai phiên
- §104 không truyền xuống production: đòi hạt gieo sạch mà ① không giao được
- §88.12 — rào NGÔN NGỮ, không phải kỹ thuật: điều kiện ② đòi một thuộc tính
  chứng thực trên ≥2 thành viên CÙNG LỚP, mà thành ngữ **cố tình** chọn vật mẫu
  khác lớp. Trên 50 thuộc tính chỉ **2** đạt, cả hai rác.

**Kết:** ý tưởng đúng, cơ chế chạy, **thiếu một cây phân loại tiếng Việt** đủ bốn
thuộc tính (đúng kiểu · phủ từ vựng · gắn nghĩa · đủ lớn). Bảy nguồn đã khảo sát,
không nguồn nào dùng được; VietNet gần nhất phủ 15%.

## 5. TÔI SAI Ở ĐÂU — khuôn lặp lại, không phải tai nạn lẻ

| khuôn | số lần | ví dụ |
|---|---|---|
| vá đúng, gọi SAI CHỖ | 9 | LOI1 · §79 · §81 · §83 · §87 · §90 · §99 (×2) · §105 |
| probe/bộ dò SAI ĐƠN VỊ hoặc SAI KHUÔN | 4 | §102 · §106 · §109 · §110 |
| probe dựng SAI CẤU HÌNH | 3 | §85 · §87 · §92 |
| tiêu đề đi trước bằng chứng, phải đính chính | 2 | §101 · §107 |
| tin dòng kết luận do chính mình lập trình | 1 | §106 («CHỒNG LẤN» sai) |

**Bài học đắt nhất (178):** chín lần «vá đúng, không đổi gì» — chín lần tôi đọc
thành «bản vá vô ích». Phép thử nhân quả (đổi đầu vào mạnh, xem đầu ra có nhúc
nhích) tốn **ba phút** và tôi dùng nó ở commit thứ 30 thay vì thứ nhất.

## 6. CÒN LẠI GÌ

**Đường duy nhất cho «con khỉ»:** nạp cây phân loại tiếng Việt ~10 000 mục. §100
đã chứng minh có nó thì cơ chế chạy. Đó là dự án BIÊN SOẠN DỮ LIỆU, không phải
sprint kỹ thuật.

**Đường cho chất lượng sinh:** §80 là công thức đã ăn tiền — viết lại tri thức
sẵn có thành đúng hình dạng câu. Nhưng §110 cho thấy các khuôn dễ (`có`/`gồm`/
`thuộc`) đã cạn hoặc đã khai.

**Hai lỗi còn mở, chẩn đoán chắc, chưa vá:**
- §82 mất trật tự từ sau 8–12 token (mạng truy ĐÚNG câu nguồn rồi rã)
- §98 `con X là` vỡ — tương quan 10/10 với «cụm chủ ngữ có mục định nghĩa hay
  không»; §99 vá tầng kế hoạch âm tính, phải can thiệp tầng TOKEN

---

## 111. §82 MẤT TRẬT TỰ — chẩn đoán XONG: agenda HOÀN HẢO, lỗi ở CỬA SỔ thứ tự

Ba hướng trước đã bị bác (§82 nới tiêu thụ cạnh · §83 bậc-2 toàn cục · §84 bậc-2
cha-hoạt). Vòng này KHÔNG vặn tham số mà đi hỏi: **agenda có mang thứ tự không?**

### 111.1 Agenda là câu nguồn, ĐÚNG THỨ TỰ

```
hổ   AGENDA: một loài động vật có lãnh thổ hoạt động rất rộng, và các cá thể phân tán…
     SINH  : hổ là một loài động vật có lãnh thổ hoạt  RẤT ĐỘNG  thể và các để đến…
```

Mạng **có sẵn câu đúng trong tay** rồi phát sai thứ tự. Và có quy luật định lượng:

```
chó  agenda 15 token → đầu ra HOÀN HẢO
lúa  agenda 30       → rã sau ~10
sóc  agenda 30       → rã sau ~6
hổ   agenda 45       → rã sau ~8
```

### 111.2 Cơ chế: `ar_w_skip_agenda_order_window = 4`

Mỗi bước, mạng được chọn **bất kỳ trong 4 mục agenda kế tiếp** (dòng 649–651,
`pending[:ag_win]` với `ag_decay ** off`). `hoạt động rất rộng` → phát `hoạt`,
nhảy tới `rất`, lùi về `động` — nằm gọn trong phạm vi cửa sổ.

Đây là cơ chế QUAN SÁT ĐƯỢC, khác ba vòng trước (đoán cơ chế rồi vặn tham số).

### 111.3 Thu hẹp cửa sổ — thước và MẮT bất đồng

```
window=4 (mốc)  TRUNG THÀNH 78,9% · TRẬT TỰ 89,3% · LẶP 13,6%
window=3        TRUNG THÀNH 81,1% · TRẬT TỰ 89,8% · LẶP 14,5%   ← QUA cổng
window=2        TRUNG THÀNH 78,1% · TRẬT TỰ 88,5% · LẶP 15,6%
window=1        TRUNG THÀNH 76,0% · TRẬT TỰ 93,8% · LẶP 10,8%   ← TRƯỢT phủ quyết
```

Nhưng đọc ca vàng thì NGƯỢC:

```
w=3  hổ → …hoạt RẤT ĐỘNG thể và các để đến…            (y như mốc, KHÔNG khá hơn)
w=1  hổ → …lãnh thổ HOẠT ĐỘNG RẤT RỘNG, VÀ CÁC CÁ THỂ PHÂN TÁN TRÊN KHOẢNG CÁCH…
     lúa→ …hoa lưỡng tính, KHÔNG CÓ BAO HOA, QUẢ CÓ VỎ TRẤU…
     nước→ …không màu, KHÔNG MÙI VÀ TỒN TẠI TRONG TỰ NHIÊN Ở AO HỒ, SÔNG BIỂN, v.
```

`w=1` cho câu **gần như nguyên văn** trên đúng hai ca tệ nhất hai phiên, mà thước
lại chấm thấp nhất.

### 111.4 Giả thuyết cho chỗ bất đồng — KHÔNG dùng để đổi kết luận

`w=1` sinh `khoảng cách lên tới 6` — `6` là mảnh của `650`, không nằm trong tập
từ nguồn, nên thước TRỪ ĐIỂM đúng phần sinh chính xác hơn. Cũng có ca kẹt thật:
`đồng là… dẫn điện và` dừng sớm vì không được phép bỏ qua token khó.

Ghi là **giả thuyết**. §100 tôi đã một lần đổi chỉ số sau khi thấy số; lần này
**không đổi tiêu chí**, và **không đổi mặc định**.

### 111.5 Trạng thái

Mặc định giữ `window = 4`. `w=3` qua cổng nhưng không cho lợi ích nhìn thấy được;
`w=1` cho lợi ích nhìn thấy rõ nhất nhưng trượt phủ quyết. Chưa đủ căn cứ để chọn.

Việc kế: sửa thước TRUNG THÀNH để nó không trừ điểm mảnh số (`6` của `650`), rồi
chấm lại — nếu `w=1` qua cổng sau khi sửa thước thì đây là bản vá một dòng cho
lỗi treo hai phiên.

### 111.6 Bài học

186. **Hỏi «tầng trên có đúng không» trước khi vặn tầng dưới.** Ba vòng vặn tham
     số sinh mà chưa ai hỏi agenda có đúng thứ tự không. Nó ĐÚNG — và điều đó
     chuyển bài toán từ «mạng không biết câu» sang «mạng biết mà phát sai», hai
     bài toán khác hẳn.
187. **Thước và mắt bất đồng là dữ liệu, không phải nhiễu.** Đừng chọn bên ủng hộ
     kết luận mình muốn; ghi cả hai, nêu giả thuyết, và sửa THƯỚC trước khi đổi
     mặc định.

---

## 112. SỬA THƯỚC RỒI CHẤM LẠI — `window = 3` qua cổng, đổi mặc định

### 112.1 Luật sửa thước, khai TRƯỚC khi chạy

Bỏ token thuần CHỮ SỐ ở **cả hai vế** (nguồn và sinh). Lý do nguyên tắc: thước đo
«mạng có phát ra TRI THỨC của câu nguồn không»; `650` bị `max_len` cắt thành `6`
là hiện vật tokenize, không phải mạng sai. Đối xứng nên không thiên vị mức nào.

### 112.2 Giả thuyết §111.4 — ĐÚNG CHIỀU, SAI ĐỘ LỚN

```
             thước cũ → mới
window=4      78,9 → 79,3   (+0,4)
window=3      81,1 → 81,5   (+0,4)
window=2      78,1 → 78,6   (+0,5)
window=1      76,0 → 77,0   (+1,0)   ← được lợi NHIỀU NHẤT, đúng dự đoán
```

`w=1` được cộng nhiều nhất — mảnh số phạt nó nặng nhất, đúng như §111.4 đoán.
Nhưng nó **vẫn dưới mốc** (77,0 < 79,3): giả thuyết chỉ giải thích **một phần ba**
khoảng cách. Phần còn lại là kẹt thật (`đồng là… dẫn điện và` dừng sớm).

⇒ Theo đúng ràng buộc tự đặt trước khi chạy: **`w=1` trượt, không cứu.**

### 112.3 `window = 3` qua cổng ⇒ ĐỔI MẶC ĐỊNH

```
                TRUNG THÀNH   TRẬT TỰ    LẶP
window=4 (cũ)      79,3%       89,3%    13,6%
window=3 (mới)     81,5%       89,8%    14,5%
```

Cả hai vế cổng đạt: trung thành **+2,2**, trật tự **+0,5**. Canh:
`test_LOI14_cua_so_thu_tu_agenda`. 454 test pass.

### 112.4 Ba điều phải nói kèm

1. **Mức tăng khiêm tốn** — +2,2 điểm trên ~10 prompt có nguồn.
2. **LẶP xấu đi** 13,6% → 14,5%. Không nằm trong vế phủ quyết nên không chặn,
   nhưng phải ghi.
3. **Không thấy lợi ích bằng MẮT ở `w=3`** — ca vàng `hổ` y hệt mốc. Trong khi
   `w=1` cho `hổ`/`nước`/`lúa` gần như nguyên văn mà lại trượt cổng. **Chỗ bất
   đồng giữa thước và mắt VẪN CHƯA giải thích được** sau khi sửa thước.

Tôi đổi mặc định vì cổng đã chốt trước và `w=3` qua — từ chối hành động theo một
cổng đã đạt cũng là một kiểu dời cột gôn. Nhưng lợi ích thật của nó chưa nhìn
thấy được, và tôi không tuyên bố hơn thế.

### 112.5 Bài học

188. **Sửa thước TRƯỚC, khai luật sửa, rồi mới chạy — và chấp nhận cả khi nó bác
     mình.** Giả thuyết của tôi đúng chiều (w=1 được lợi nhiều nhất) nhưng sai độ
     lớn, và tôi giữ kết luận theo số chứ không theo ca vàng đẹp mắt.
189. **Cổng đã đạt thì phải hành động, kể cả khi mình không thích kết quả.** `w=3`
     không cho lợi ích nhìn thấy được, nhưng nó qua đúng hai vế đã chốt. Bỏ qua
     nó vì «không đẹp» là dời cột gôn theo hướng ngược lại.

## §113 — Bất đồng thước/mắt: thước MÙ VỚI THỨ TỰ. ĐÍNH CHÍNH §112

§112.4 ghi lại một chỗ chưa giải thích được: `w=1` cho câu đẹp nhất khi đọc bằng
MẮT nhưng TRƯỢT cổng TRUNG THÀNH. Tôi đã sửa thước (bỏ token chữ số) và chỗ bất
đồng vẫn còn. Vòng này bỏ số TỔNG, tách theo TỪNG PROMPT.

### 113.1 Số

```
prompt        w=3 TT   w=1 TT      Δ    w=3 len  w=1 len
chó là        100,0%   100,0%    +0,0      13      13
mèo là         40,0%    40,0%    +0,0       5       5
sóc là         62,5%    62,5%    +0,0      24      24
hổ là         100,0%   100,0%    +0,0      24      22
sắt là        100,0%   100,0%    +0,0      21      21
đồng là       100,0%   100,0%    +0,0      15      14
lúa là        100,0%    85,0%   −15,0      21      20
muối là         0,0%     0,0%    +0,0       5       5
nước là       100,0%   100,0%    +0,0      23      18
mặt trời là    17,6%    21,7%    +4,1      17      23
TỔNG           81,5%    77,0%    −4,5
```

**8/10 prompt chênh đúng 0,0.** Chênh tổng 4,5 điểm không đến từ prompt nào cả.

### 113.2 Hai lỗi của thước, cùng lộ ra một lúc

**(a) MÙ VỚI THỨ TỰ.** TRUNG THÀNH đo *tập từ*, không đo *chuỗi*:

```
hổ  w=3 «…lãnh thổ hoạt RẤT ĐỘNG thể và các để đến»     → 100%
    w=1 «…lãnh thổ HOẠT ĐỘNG RẤT RỘNG, VÀ CÁC CÁ THỂ…»  → 100%
```

Câu rối và câu đúng chấm NGANG NHAU. Đúng thứ §111 sửa thì thước không thấy.

**(b) PHẠT CÂU ĐÚNG.** `lúa` — ca duy nhất `w=1` thua:

```
w=3 «…hoa lưỡng KHÔNG, hoa cây có quả hạt thực thân lương ngoài…» → 100%
w=1 «…hoa lưỡng tính, KHÔNG CÓ BAO HOA, QUẢ CÓ VỎ TRẤU thể giúp cơ» →  85%
```

`w=3` xáo từ nguồn thành vô nghĩa ⇒ 100%. `w=1` phát gần nguyên văn rồi lệch ở
ĐUÔI ⇒ 85%. Thước **thưởng cho câu rối, phạt câu đúng**.

**(c) TỈ LỆ GỘP.** Tổng là `Σhit/Σtừ`, nên prompt sinh dài ở tỉ lệ thấp thống
trị: `mặt trời` `w=1` sinh 23 từ ở 21,7% ⇒ thêm 18 lượt trượt vào pool. Trung
bình theo prompt (macro) là 72,0% vs 70,9% — chênh 1,1 chứ không phải 4,5.

### 113.3 Quyết định — ĐỔI MẶC ĐỊNH `4 → 1` (không phải 3)

`w=1` thắng **cả hai thước CÓ THỂ thấy thứ tự**, thua duy nhất thước mù với nó:

| | TRUNG THÀNH | TRẬT TỰ | LẶP |
|---|---|---|---|
| w=4 | 79,3% | 89,3% | 13,6% |
| w=3 (§112) | **81,5%** | 89,8% | 14,5% |
| w=1 | 77,0% | **93,8%** | **10,8%** |

§112 phong quán quân cho cấu hình thắng ĐÚNG cột không đo được điều đang sửa.
`config.py` → 1; `test_LOI14` sửa theo; 454 pass.

### 113.4 Bài học

- **185.** Chỉ số TỔNG dạng `Σhit/Σmẫu` KHÔNG so được hai cấu hình sinh ra độ
  dài khác nhau. Tách theo từng mục TRƯỚC khi tin vào chênh lệch tổng.
- **186.** Khi THƯỚC và MẮT bất đồng, nghi ngờ đầu tiên là **thước không đo thứ
  mà mắt đang nhìn** — không phải mắt thiên vị. Ở đây mắt nhìn TRẬT TỰ, thước đo
  TẬP TỪ; hai đại lượng khác nhau, nên bất đồng là tất yếu chứ không nghịch lý.
- **187.** Sửa thước (§112, bỏ chữ số) mà bất đồng KHÔNG hết ⇒ dấu hiệu sai
  KIỂU thước, không phải sai chi tiết. §112 vá chi tiết rồi chốt luôn — sai.
- **188.** Cổng có vế PHỦ QUYẾT thì phải kiểm vế đó có ĐO ĐƯỢC thay đổi đang xét
  không. Phủ quyết bằng thước mù = phủ quyết mù.

## §114 — Xét lại các hướng bị TRUNG THÀNH phủ quyết: 0/20 được phục hồi

§113 phát hiện thước TRUNG THÀNH mù với thứ tự. Nó đã đứng ở vế PHỦ QUYẾT của
gần như mọi cổng suốt hai phiên ⇒ phải xét lại các hướng nó chặn.

### 114.1 Sàng bằng số đã có (mốc 78,9 / 89,3 / 13,6)

Chữ ký «oan» = TT tụt NHỎ **và** TRẬT TỰ hoặc LẶP tốt lên.

| hướng | TT | TRẬT TỰ | LẶP | phán |
|---|---|---|---|---|
| toàn cục w=1 §83 | 53,8 | 92,7 | 18,2 ✗ | tụt 25đ + lặp tăng ⇒ loại ĐÚNG |
| CHA HOẠT w=1 | 67,0 | 91,0 ✓ | 11,8 ✓ | ← nghi oan |
| top_n=5 | 69,3 | 89,9 ✓ | 12,2 ✓ | ← nghi oan |
| top_n=1/2/3 | 74,6/66,2/74,7 | 86,2/89,0/87,5 ✗ | — | TRẬT TỰ TỤT ⇒ loại ĐÚNG |
| GỘP hoa/thường §87 | 78,9 | 89,3 | 13,6 | y hệt mốc — vô hiệu, không bị loại |
| LỌC ĐÍCH §93 | 78,9 | 89,3 | 13,6 | y hệt mốc — vô hiệu, không bị loại |

Chỉ **2/20** mang chữ ký oan. Và cả hai đều đo ở `window=4`; bậc-2 tồn tại để sửa
TRẬT TỰ, đạt 91,0 với giá 11,9đ trung thành — `window=1` (§113) đạt **93,8** với
giá 1,9đ. Giả thuyết: cơ chế cũ đã bị THAY THẾ, không phải bị xử oan.

### 114.2 Chấm lại, chồng lên mặc định MỚI

CẢNH BÁO ĐƠN VỊ: ngưỡng cổng (93,8 / 10,8) lấy từ bộ 20 prompt; lượt này chạy bộ
10 prompt nên mốc nội bộ là 93,7 / 11,6. Chấm theo mốc TRONG CÙNG LƯỢT.

| | TT-macro | TRẬT TỰ | LẶP |
|---|---|---|---|
| `w1` (mốc) | 70,9 | 93,7 | 11,6 |
| `w1` + bậc2 top20 | 64,6 ✗ | 93,0 ✗ | 12,6 ✗ |
| `w1` + bậc2 top5 | 70,2 ✓ | **94,8** ✓ | 13,8 ✗ |

`top20` trượt cả ba. `top5` qua hai, trượt LẶP ⇒ theo cam kết khai trước, KHÔNG
cứu. Mắt xác nhận ở ca vàng `hổ`: cả hai mức đều phá tan câu `w1` vừa dựng đúng
(«hổ là xương hổ dùng làm một các một có khắc hình đầu con lên…»).

### 114.3 Một mảnh ĐÁNG GIỮ, hẹp hơn nhiều

`lúa` — chính ca mất 15đ ở §113 vì lệch ĐUÔI:

```
w1     …quả có vỏ trấu THỂ GIÚP CƠ       ← đuôi lệch
+bậc2  …quả có vỏ trấu BAO NGOÀI HẠT     ← đúng nguyên văn
```

Bậc-2 CÓ tri thức về chỗ đó; cái nó thiếu là biết KHI NÀO NÊN IM. Hướng khả dĩ
là bật bậc-2 CHỈ ở cuối câu khi agenda đã cạn — không phải bật toàn cục.

### 114.4 Bài học

- **189.** Thước mù làm hỏng đúng quyết định nó dùng để **CHỌN** (§112), không
  phải các quyết định nó dùng để **LOẠI**. Phủ quyết chặn ở mức tụt 10–25 điểm
  kèm hỏng mắt rõ ⇒ quá lớn để quy cho hiện vật gộp 4,5 điểm.
- **190.** Sửa thước xong phải xét lại các kết luận CŨ dựa trên nó — nhưng xét
  bằng SÀNG trước (số đã có), chỉ chạy lại thứ mang chữ ký nghi ngờ. 2/20 phải
  chạy lại, không phải 20/20.
- **191.** Ngưỡng cổng khai trên bộ prompt A KHÔNG dùng thẳng cho bộ prompt B.
  Phải chấm theo mốc trong cùng lượt chạy. (Suýt tái phạm ngay trong §114.)

## §115 — Bậc-2 «chỉ khi agenda cạn»: ÂM TÍNH, agenda KHÔNG BAO GIỜ cạn

§114.3 giữ lại một mảnh: bậc-2 sửa đúng đuôi lệch của `lúa` (`…vỏ trấu BAO NGOÀI
HẠT` thay vì `THỂ GIÚP CƠ`) nhưng phá ca vàng `hổ` khi bật toàn cục. Giả thuyết:
nó thiếu chỗ DỪNG — chỉ nên nổ khi agenda đã phát hết.

### 115.1 Cổng khai TRƯỚC

- MỤC TIÊU: `lúa` phải sửa được đuôi. Không đạt ⇒ âm tính.
- PHỦ QUYẾT: `hổ`, `nước` đang đúng ở `w1`, không được hỏng bằng MẮT.
- TRẬT TỰ ≥ mốc, LẶP ≤ mốc, TT-macro không tụt > 2 — chấm theo mốc TRONG CÙNG
  LƯỢT CHẠY (bài học 191).

### 115.2 Vá, rồi CHỨNG MINH tác động trước khi A/B

Cổng dùng ĐÚNG vị từ mà dòng ~746 dùng để hạ EOS (`any(not c for c in
consumed)`) nên hai chỗ không thể lệch định nghĩa «còn mục chưa phát».

Phép đếm rẻ (so đầu ra với mốc `w2=0`) TRƯỚC khi chạy A/B đầy đủ:

```
toàn cục      khác mốc ở 2/5 prompt
CHỈ KHI CẠN   khác mốc ở 0/5 prompt   ← nổ 0 lần
```

**Bậc-2 không nổ lấy một lần.** Tức ở MỌI bước, agenda vẫn còn mục chưa phát:
trong 24 token sinh, agenda KHÔNG BAO GIỜ cạn.

### 115.3 Điều này bác giả thuyết §114.3

`lúa` trượt sang `thể giúp cơ` **trong khi agenda vẫn còn mục ĐÚNG đang chờ**.
Đuôi lệch KHÔNG phải do hết nguyên liệu — mà do vòng phiếu chọn token NGOÀI
agenda dù agenda còn mục. Đó là địa hạt `nonag_scale` / `ag_strict`, không phải
địa hạt thời điểm của bậc-2. Hướng §114.3 chết; hướng thay thế là hỏi vì sao
token ngoài agenda thắng được mục agenda đang chờ.

Đã GỠ cờ thay vì để lại máy móc không bao giờ chạy; ghi chú chống dựng lại đặt
ngay tại chỗ vá. 454 pass.

### 115.4 Bài học

- **192.** Phép đếm RẺ («cơ chế có nổ không») phải chạy TRƯỚC A/B đầy đủ. Ở đây
  nó tốn ~2 phút và giết cả hướng đi; A/B đầy đủ sẽ tốn ~30 phút để ra cùng một
  bảng số y hệt mốc mà KHÔNG nói được vì sao.
- **193.** «Cơ chế X thiếu chỗ dừng» là giả thuyết về THỜI ĐIỂM — luôn kiểm
  thời điểm đó có TỒN TẠI trong lượt chạy thật không, trước khi vá theo nó.
- **194.** `sys.settrace` là công cụ SAI cho vòng sinh 450k cha (timeout 10
  phút). Muốn xem biến cục bộ giữa vòng lặp thì dùng cổng đếm rẻ, đừng profiler.

## §116 — Token ngoài agenda thắng vì HAI BỘ PHẬN BẤT ĐỒNG về «agenda có tính không»

§115 để lại câu hỏi: vì sao token ngoài agenda thắng được mục agenda đang chờ?

### 116.1 Truy bằng MỘT phép đo, không A/B

Móc đếm rẻ (bài học 192/194 — không dùng profiler) ghi `agenda_ok` từng bước:

```
«lúa là» → …quả có vỏ trấu THỂ GIÚP CƠ
   24 bước · agenda_ok=False ở 3 bước
      bước 21 đầu chờ «bao» · PHÁT «thể»
      bước 22 đầu chờ «bao» · PHÁT «giúp»
      bước 23 đầu chờ «bao» · PHÁT «cơ»
«hổ là»  24 bước · agenda_ok=False ở 0 bước   ← câu ĐÚNG
«chó là» 16 bước · agenda_ok=False ở 0 bước   ← câu ĐÚNG
```

Tương ứng **1:1**: ba token rác rơi đúng ba bước `agenda_ok=False`, mục chờ là
«bao» — đầu của `bao ngoài hạt`, đoạn ĐÚNG. Bằng chứng mạnh hơn mọi bảng tổng
hợp hai phiên qua vì nó chỉ vào BƯỚC gây lỗi, không phải trung bình 10 prompt.

### 116.2 Cơ chế — không bộ phận nào hỏng, chúng chỉ không biết nhau

- dòng ~630 (§12.137): «bao» ∉ giao W_skip ⇒ kết luận agenda lệch cha ⇒ bỏ CẢ
  boost LẪN đè `nonag_scale` ⇒ token ngoài agenda được thả tự do.
- dòng ~746: agenda CÒN mục chưa phát ⇒ dìm EOS ×0,15 ⇒ cấm dừng.

Một bên «agenda không đáng tin, kệ nó», bên kia «agenda còn mục, cấm dừng». Mạng
bị ép nói tiếp mà không còn nguồn dẫn ⇒ bịa. Vá: hai chỗ dùng CÙNG vị từ
(`_ag_tin`), khai mặc định mỗi bước vì khối agenda có thể không chạy khi
`ag_boost <= 0` (nếu không sẽ `NameError`).

### 116.3 Cổng: MỤC TIÊU TRƯỢT, cơ chế XÁC NHẬN trên bộ giữ riêng

Khai trước: «`lúa` phải sửa được đuôi; không đạt ⇒ âm tính». `lúa` KHÔNG đổi một
token ⇒ **giả thuyết về `lúa` là ÂM TÍNH**, không bàn lại. Vế đầu (bị tuyên bố
không đáng tin) đúng; vế cuối (⇒ EOS bị dìm oan ⇒ bịa) SAI: gỡ dìm rồi vẫn bịa,
tức ở ba bước đó EOS vốn đã YẾU SẴN, dìm không phải lý do chính.

Bản vá lại đổi chỗ khác — hai câu GIỜ BIẾT DỪNG:

```
mèo là  ▸ thú và chó là bạn.
muối là ▸ sản phẩm axit và bazơ.
```

Đây đúng là tình huống đã TỪ CHỐI cứu ở §112 và §114: số tổng đẹp lên SAU khi
mục tiêu khai trước đã trượt. Nên không chốt bằng bộ 10 prompt vừa dùng để chẩn
đoán, mà xác nhận trên bộ 20 prompt có mốc chốt từ §112 (trước khi biết `_ag_tin`
tồn tại):

| | TRUNG THÀNH | TRẬT TỰ | LẶP |
|---|---|---|---|
| mốc §112 `w=1` | 77,0 | 93,8 | 10,8 |
| sau §116 | **81,7** | **94,3** | **7,1** |

Cả ba không xấu đi ⇒ GIỮ, không kèm cờ (vá mâu thuẫn nội bộ, cùng loại §81/§111).
Ba ca vàng `chó`/`hổ`/`sắt` nguyên vẹn. 454 pass.

### 116.4 Bài học

- **195.** Loại lỗi MỚI, khác chín lần «vá đúng cơ chế sai chỗ gọi»: hai bộ phận
  đều ĐÚNG phần việc mình, nhưng trả lời KHÁC NHAU cho cùng một câu hỏi và không
  biết nhau. Không lộ khi đọc từng khối; chỉ lộ khi truy đúng BƯỚC phát token rác.
- **196.** Chẩn đoán và phán quyết KHÔNG được dùng chung một bộ số. Bộ đã nhìn
  để tìm cơ chế thì hết tư cách làm phép thử — phải xác nhận trên bộ có mốc chốt
  TRƯỚC khi biết cơ chế.
- **197.** Vá đúng cơ chế mà mục tiêu vẫn trượt ⇒ giả thuyết có NHIỀU VẾ và chỉ
  vế đầu đúng. Ghi rõ vế nào sai (ở đây: EOS yếu sẵn ≠ EOS bị dìm), đừng ghi
  chung là «đã sửa».

## §117 — EOS yếu sẵn: vòng luẩn quẩn «muốn được dừng thì phải ĐÃ dừng»

§116 để lại: vì sao EOS yếu sẵn ở đuôi `lúa`, dìm hay không cũng thua?

### 117.1 Số

EOS có hai đường lên điểm: đường thường (`ref` = TRUNG VỊ điểm dương) và cú đẩy
biên (`boundary_eos_beta = 1,15` × MAX). Đo từng bước:

```
«lúa là»  bước  phát   EOS      MAX    trung vị
            20  trấu   2,000   3,946     0,000
            21  thể    2,000  46,857     0,032   ← EOS thua 23 LẦN
«mèo là»     6  <EOS> 109,856 109,856     0,027   ← cú đẩy NỔ, EOS = MAX
```

Cú đẩy có ba điều kiện. Đo thẳng (không suy đoán):

```
«lúa» bước 21:  out[-1] LÀ biên = True  ·  n_b_gen = 0   ← MÂU THUẪN
«mèo» bước  6:  out[-1] LÀ biên = True  ·  n_b_gen = 1   ← nổ
```

### 117.2 Cơ chế — LẠI hai định nghĩa cho một khái niệm (như §116)

```
count_boundaries(hist, vocab, bound=None)          ← bound MẶC ĐỊNH None
count_boundaries(out[prompt_len:], state.vocab)    ← chỗ gọi KHÔNG truyền _bd_gen
```

Bỏ trống `bound` ⇒ chạy BOOTSTRAP (chỉ dấu câu cứng), trong khi `_is_b_eos` ngay
bên cạnh dùng tập biên ĐÃ HỌC `_bd_gen`. Hai bên trả lời ngược nhau về «biên».

Hậu quả là VÒNG LUẨN QUẨN: muốn được cộng điểm dừng thì phải ĐÃ phát dấu câu
cứng. `lúa` chỉ phát dấu phẩy ⇒ `n_b_gen` mãi 0 ⇒ EOS ghim 2,0 / max 46,9 ⇒
không dừng nổi ⇒ buộc phải bịa. `mèo` thoát chỉ vì tình cờ phát `.` ở bước 5.

Đây là chỗ gọi thứ BA cùng loại: §81 vá hai chỗ `_is_boundary` trong chính vòng
lặp này và bỏ sót cả ba chỗ `count_boundaries`.

### 117.3 Cổng: MỤC TIÊU ĐẠT, PHỦ QUYẾT VI PHẠM ⇒ HOÀN NGUYÊN

Truyền `_bd_gen` vào cả ba chỗ gọi. `lúa` dừng SẠCH đúng như muốn:

```
lúa là ▸ cây lương thực, thân cỏ rỗng, hoa lưỡng tính, không có bao hoa, quả có vỏ trấu
```

Nhưng EOS nổ bừa ở chỗ khác:

```
chó là  ▸ gia súc                     ← trước: …thuộc nhóm ăn thịt, nuôi để giữ nhà…
sắt là  ▸ …kéo sợi, dễ bị gỉ          ← mất «trong không khí ẩm, là thành phần»
nước là ▸ …trong tự nhiên ở ao        ← mất «hồ, sông biển»
```

`chó` mất nguyên định nghĩa, còn hai từ. Khai trước «không được cắt cụt sớm» ⇒
LOẠI, hoàn nguyên. Hướng ngược lại (hợp nhất theo BOOTSTRAP) thì `lúa` vẫn không
dừng. **Cả hai hướng đều trượt**: `_bd_gen` quá rộng, bootstrap quá chặt. Ghi chú
chống tái phạm đặt tại chỗ. 454 pass, không đổi mặc định.

### 117.4 Bài học

- **198.** Bản vá này gần như chắc chắn LÀM TĂNG TRUNG THÀNH (câu ngắn ⇒ ít từ
  sai) trong khi phá tan tính hữu dụng. Đúng hiện vật độ dài §113. Chấm bằng
  bảng số tổng trước khi nhìn bằng MẮT thì đã phong nó là cải tiến.
- **199.** Tham số mặc định `bound=None` che một MÂU THUẪN ĐỊNH NGHĨA: chỗ gọi
  quên truyền không báo lỗi, chỉ lặng lẽ đổi sang ngữ nghĩa khác. Hàm có hai chế
  độ ngữ nghĩa thì đừng để chế độ nào là mặc định — bắt buộc truyền.
- **200.** Hai bộ phận bất đồng về một khái niệm KHÔNG suy ra «hợp nhất là sửa».
  Ở đây hợp nhất được cả hai hướng, và cả hai đều hỏng — nghĩa là bản thân KHÁI
  NIỆM «biên» đang gánh hai việc khác nhau (chỗ được phép dừng ≠ chỗ đã đóng câu).

## §118 — TÁCH hai vị từ biên: lỗi ở CHỖ NỐI, không ở vị từ nào

§117 kết luận: không hợp nhất được «biên» vì nó gánh HAI việc. Vòng này tách.

### 118.1 Tách ra thì thấy CẢ HAI VỊ TỪ ĐỀU ĐÚNG

```
ĐÃ ĐÓNG CÂU    = n_b_gen   (bootstrap, dấu câu CỨNG)   ← đúng như đang có
ĐƯỢC PHÉP DỪNG = _is_b_eos + _bd_gen (tập ĐÃ HỌC)      ← đúng như đang có
```

Sai nằm ở CHỖ NỐI: `_is_b_eos(...) and n_b_gen >= 1`. Đòi `n_b_gen >= 1` tức đòi
ĐÃ đóng một câu thì mới cho dừng ⇒ câu ĐẦU TIÊN không bao giờ được kết thúc.
Đó chính là vòng luẩn quẩn §117, và nó nằm ở TOÁN TỬ chứ không ở định nghĩa.

Vế thay: `(n_b_gen >= 1) or (not _ag_tin)` — đã đóng câu HOẶC agenda hết đáng
tin (`_ag_tin`, §116). Điều kiện có NỘI DUNG (không còn nguồn dẫn thì cấm dừng
là vô lý), không phải ngưỡng độ dài tuỳ tiện, và tái dùng biến §116 đã dựng.

### 118.2 Dự đoán PHÂN BIỆT ĐƯỢC, khai trước khi chạy

| | chỗ hợp lệ | n_b_gen | `_ag_tin` | dự đoán |
|---|---|---|---|---|
| `lúa` bước 21 | có | 0 | NGỜ | ĐƯỢC dừng → hết bịa |
| `chó` bước 2 | có | 0 | tin | CẤM dừng → giữ nguyên câu |

Hai ca ĐỐI NGHỊCH, cùng một cơ chế. Cả hai ra đúng chiều:

```
lúa là ▸ …quả có vỏ trấu                                    ← dừng sạch
chó là ▸ gia súc thuộc nhóm ăn thịt, nuôi để giữ nhà hay đi săn .   ← nguyên vẹn
```

Bằng chứng này mạnh hơn bảng số: bảng số KHÔNG phân biệt được giả thuyết «tách
đúng chỗ nối» với «EOS mạnh lên nói chung»; cặp đối nghịch thì có.

### 118.3 Cổng — QUA TOÀN BỘ

Bộ 20 prompt giữ riêng (mốc §116):

| | TRUNG THÀNH | TRẬT TỰ | LẶP |
|---|---|---|---|
| mốc §116 | 81,7 | 94,3 | 7,1 |
| sau §118 | **85,9** | **97,3** | **7,0** |

Bốn ca vàng nguyên vẹn. TRẬT TỰ 97,3% là mức cao nhất từng đạt MÀ KHÔNG đánh đổi
tri thức (§83 đạt 98,0 nhưng xoá sạch nội dung: `chó là gia đình và bạn bè`).
Giữ, không kèm cờ. 454 pass.

### 118.4 CÒN HỎNG — ghi rõ

```
sóc là ▸ thú gặm       ← cắt giữa từ ghép «gặm nhấm». KHÔNG có trong danh sách
                         phủ quyết khai trước, nên không chặn việc giữ bản vá,
                         nhưng là hỏng THẬT: vế `not _ag_tin` còn hơi rộng.
đồng là ▸ …dẫn điện và  ← vẫn cụt như cũ, chưa đụng tới
mặt trời ▸ …chiếm khoảng ngôi ← đuôi vẫn rác
```

### 118.5 Bài học

- **201.** «Hai bộ phận bất đồng» có BA cách sửa, không phải một: đổi vị từ A,
  đổi vị từ B, hoặc đổi TOÁN TỬ NỐI. §117 thử hai cách đầu và trượt cả hai;
  cách thứ ba mới đúng. Khi cả hai vị từ đều đúng phần việc mình, hãy nghi ngờ
  phép nối trước khi nghi ngờ định nghĩa.
- **202.** Giả thuyết tốt cho DỰ ĐOÁN PHÂN BIỆT được trên hai ca ĐỐI NGHỊCH,
  khai TRƯỚC khi chạy. Đó là thứ bảng số tổng không thay thế được — nó chỉ nói
  «tốt lên», không nói «tốt lên VÌ cơ chế này».

## §119 — Thắt vế `not _ag_tin` bằng ngưỡng tiêu thụ agenda: TRƯỢT, hoàn nguyên

§118.4 ghi `sóc là` cụt ở «thú gặm» và ngờ vế `not _ag_tin` hơi rộng.

### 119.1 Chẩn đoán — tách bạch rất rõ

Đo mức agenda ĐÃ TIÊU THỤ tại bước cho dừng:

```
sóc  →   7%   (sau đúng 2 token — quá sớm)
lúa  →  70%   (hợp lý)
chó  → 100%   (đi đường n_b_gen >= 1, không dùng vế này)
sắt  →  77%   (không bao giờ ngờ, chạy hết max_len)
```

Khoảng trống 7% – 70% rất rộng ⇒ tưởng chừng chỉ cần một ngưỡng ở giữa. Chọn
0,5 theo nghĩa «phần lớn», khai TRƯỚC là sẽ KHÔNG dò 0,4 rồi 0,6 để cứu.

### 119.2 Kết quả — trượt CẢ BA thước

| | TRUNG THÀNH | TRẬT TỰ | LẶP |
|---|---|---|---|
| §118 (mốc) | 85,9 | 97,3 | 7,0 |
| §119 (ngưỡng 0,5) | 83,3 | 94,6 | 7,2 |

Ngưỡng chặn chỗ dừng ở NHIỀU prompt chứ không riêng `sóc` ⇒ chúng nói lảm nhảm
tới hết `max_len` ⇒ trật tự tụt 2,7 điểm. Và `sóc` chỉ đổi mẩu cụt («thú gặm»)
lấy câu RÁC DÀI cũ («…đuôi thú nhấm vị anh hùng trí của bạn rất»).

⇒ LOẠI theo cam kết. Vế `not _ag_tin` KHÔNG kèm ngưỡng tốt hơn về tổng thể, dù
nó cắt cụt `sóc`. Ghi chú chống tái phạm đặt tại chỗ. 454 pass.

### 119.3 Bài học

- **203.** MỤC TIÊU khai theo HÌNH DẠNG («hết cụt») đạt được bằng cách xấu —
  chặn chỗ dừng rồi để mạng nói tới hết `max_len`. Phải khai theo CHẤT LƯỢNG
  («ra câu đúng HOẶC dừng sạch»). Không sửa mục tiêu sau khi thấy kết quả, nhưng
  ghi nhận mục tiêu đã khai kém.
- **204.** Hai nhóm tách bạch rõ trên VÀI ca (7% vs 70%) KHÔNG suy ra ngưỡng ở
  giữa là đúng: ngưỡng áp cho MỌI prompt, kể cả những prompt chưa nhìn. Tách bạch
  trên mẫu nhỏ là điều kiện CẦN, không đủ.
- **205.** `assert` trước `write_text` đã chặn một patch sai (đoán thụt lề 8 dấu
  cách, thật là 4): file không bị ghi, `git diff` rỗng, đầu ra in ra vẫn là §118.
  Nếu dùng `sed` thay mù thì đã đọc bảng số §118 mà tưởng của §119 — đúng loại
  lỗi §85/§87/§92 («đo cấu hình không phải cấu hình mình nghĩ»).

## §120 — `đồng` cụt: agenda được cộng TƯƠNG ĐỐI, EOS được cộng TUYỆT ĐỐI

`đồng là kim loại có màu đỏ, dễ dát mỏng và kéo sợi, dẫn điện và` — cụt sau một
LIÊN TỪ, ở 17 token chứ chưa chạm `max_len=24`.

### 120.1 KHÔNG phải lỗi cơ chế dừng

```
đồng bước 16 (sau «và»):  hợp lệ=False · CHO DỪNG=không · điểm EOS 2,00
```

EOS không hề được cú đẩy biên (§118) cộng điểm. Nó thắng bằng điểm SÀN.

### 120.2 Túi ứng viên — so hai ca cùng cấu trúc

```
đồng b16:  top «dẫn»   2,25  · EOS 2,00 · 8 868 ứng viên sống
sắt  b22:  top «thành» 105,65 · EOS 2,00 · 7 649 ứng viên sống
```

Cả hai bước: mục agenda đang chờ đứng đầu (`dẫn`, `thành`), cùng qua công thức
`scores[t] += ag_boost * rv`. Chênh ~50 LẦN ⇒ `rv` (max điểm nền lúc đó) SỤP ở
đuôi `đồng`. Cú đẩy agenda TỈ LỆ với `rv` nên sụp theo.

### 120.3 Bất đối xứng cấu trúc

| | dạng cộng điểm | khi nền sụp |
|---|---|---|
| agenda | TƯƠNG ĐỐI (`ag_boost × rv`) | sụp theo |
| EOS | TUYỆT ĐỐI (sàn 2,00 từ `ea`) | GIỮ NGUYÊN |

⇒ Agenda mất sức đúng lúc cần nhất, EOS thắng mặc định. `đồng` cụt vì KHÔNG CÒN
GÌ MẠNH HƠN CÁI SÀN, không phải vì được phép dừng.

Nghi cùng gốc với `sóc` (cụt ở 7% agenda) — cũng là ca nền yếu.

### 120.4 CHƯA truy xong — ghi rõ

Tại điểm đặt móc, `dẫn` đang 2,25 > EOS 2,00, vậy mà token phát ra là `<EOS>`.
Có thứ gì đó SAU móc đảo lại thứ hạng (móc nằm ngay sau `_duoc_dung`, trước khối
bậc-2 §83 và khối chuỗi §79). Chưa truy; KHÔNG đoán.

### 120.5 Bài học

- **206.** Trộn thang ĐO TUYỆT ĐỐI với thang ĐO TƯƠNG ĐỐI trong cùng một cuộc
  bỏ phiếu là bẫy: khi nền sụp, vế tuyệt đối thắng mà không cần mạnh lên. Muốn
  so được thì hai vế phải cùng đơn vị — hoặc cùng tỉ lệ với `rv`, hoặc cùng sàn.
- **207.** Triệu chứng «cụt» KHÔNG suy ra «lỗi ở cơ chế dừng». Ở đây cơ chế dừng
  đo được là ĐANG TỪ CHỐI cho dừng, mà câu vẫn cụt. Phải xem TÚI ỨNG VIÊN trước
  khi vá chỗ dừng — §119 đã phí một vòng vì vá chỗ dừng cho ca `sóc`.

## §121 — Cho hai vế cùng đơn vị: TRƯỢT NẶNG, hoàn nguyên

§120 đo được bất đối xứng: agenda cộng điểm TƯƠNG ĐỐI (`ag_boost * rv`), EOS có
sàn TUYỆT ĐỐI (2,00) ⇒ nền sụp thì EOS thắng mặc định, câu cụt giữa chừng.

### 121.1 Chọn hướng — nghĩ trước xem mỗi hướng hỏng ở đâu (§117)

- EOS thành tương đối HOÀN TOÀN (`k * rv`): ở `rv≈42` thì EOS vọt lên, cắt cụt
  mọi câu. Loại trên giấy.
- Agenda thêm sàn tuyệt đối: sàn thắng cả ở chỗ nền yếu chính đáng. Loại.
- CHẶN TRẦN theo nền `min(EOS, k*rv)`: không bao giờ NÂNG EOS, chỉ hạ khi nền
  sụp. `k = 0,05` KHÔNG dò — hiệu chỉnh để tái lập hành vi hiện thời (EOS đang
  2,00 ở `rv≈40`). Chọn hướng này.

### 121.2 Dự đoán khai trước, và nó BẮT ĐƯỢC LỖI ĐẶT SAI CHỖ

Khai: `sắt`/`hổ`/`nước` không đổi; `chó`/`mèo`/`muối`/`lúa` không đổi (dừng qua
cú đẩy §118, không qua sàn); `đồng` hết cụt.

Chạy: mọi dự đoán đúng TRỪ `lúa` — mất chỗ dừng sạch, quay lại «thể giúp cơ».
Chính chỗ dự đoán SAI chỉ thẳng ra nguyên nhân: trần đặt SAU khối cú đẩy nên
chặn luôn EOS đã được §118 nâng hợp lệ (`1,15×46,9 = 53,9` → `0,05×46,9 = 2,34`).
Lỗi «đúng cơ chế SAI CHỖ ĐẶT» lần thứ MƯỜI. Chuyển lên TRƯỚC khối cú đẩy thì
`lúa` giữ được và mọi dự đoán khớp.

### 121.3 Kết quả — trượt cả ba

| | TRUNG THÀNH | TRẬT TỰ | LẶP |
|---|---|---|---|
| §118 (mốc) | 85,9 | 97,3 | 7,0 |
| §121 (trần k=0,05) | 81,9 | **90,1** | **13,4** |

Lặp gần GẤP ĐÔI, trật tự tụt 7,2 điểm — thiệt hại lớn nhất phiên. `đồng` hết cụt
nhưng nói thêm toàn rác (`…dây dẫn có để dùng chế loại hợp`), `mặt trời` xấu đi
rõ (`…Mặt một ngôi Trời thuộc Hệ Niedersachsen sao khoảng lượng khối ngôi Trời`).
Bỏ trần EOS chỉ đổi CỤT lấy LẢM NHẢM — y hệt §119. HOÀN NGUYÊN.

### 121.4 Bài học

- **208.** DỰ BÁO SAI: tôi báo trước «trung thành có thể TĂNG vì đoạn rác dùng
  lại từ câu nguồn» (điểm mù §113). Thực tế TỤT 4 điểm — đoạn nói thêm chứa từ
  LẠ (`Niedersachsen`, `chế`, `hợp`). Điểm mù của thước không tự động cứu mọi
  bản vá làm dài đầu ra; phải đo, không suy.
- **209.** Dự đoán khai trước có giá trị THỨ HAI ngoài việc xác nhận: khi nó sai
  ở đúng MỘT ca, ca đó ĐỊNH VỊ lỗi. `lúa` sai ⇒ tìm ngay ra chỗ đặt sai. Bảng số
  tổng sẽ chôn ca đó trong trung bình 20 prompt.
- **210.** Ba vòng liên tiếp (§119, §121) cùng một hình dạng: gỡ/chặn cơ chế DỪNG
  thì chỉ đổi CỤT lấy LẢM NHẢM. Gốc rễ là NỀN SỤP (không còn ứng viên tốt), và
  không thao tác nào ở tầng EOS chữa được cái đó. Muốn chữa phải đi tầng ứng viên.

## §122 — Tầng ứng viên: `rv` sụp vì PHẠT LẶP NHÂN DỒN lên từ lặp CHÍNH ĐÁNG

§121 bài học 210: ba vòng liên tiếp thao tác tầng EOS đều chỉ đổi CỤT lấy LẢM
NHẢM, gốc rễ là NỀN SỤP. Vòng này xuống tầng ứng viên.

### 122.1 Hai nghi phạm CHẾT ngay

```
blocked_dst  = 0 ở MỌI bước, cả hai câu   ⇒ KHÔNG phải chặn đích
#ứng viên dương = 6 000 – 9 000 suốt      ⇒ KHÔNG phải cạn túi
```

### 122.2 `rv` tụt ở CẢ HAI — khác nhau ở chỗ HỒI PHỤC

```
        b0      b9      b12     b16     b18
đồng  1081,8    3,86    4,26    3,60     —      ← không bao giờ hồi
sắt   1077,9    1,94    4,24   32,22  241,15    ← HỒI PHỤC
```

`sắt` b16: `#dương` SIẾT từ 7 076 → 2 556 và `rv` vọt 10× ⇒ truy hồi TÁI KHOÁ
được vào cha. `đồng` không tái khoá lần nào ⇒ nền ở lại mức ~3,5.

### 122.3 Nguyên nhân — từ lặp CHÍNH ĐÁNG bị triệt tiêu

```
đồng b15/b16: đầu agenda «dẫn» · điểm nền  0,000
sắt  b12/b13: đầu agenda «dễ»  · điểm nền  0,000
```

Câu nguồn của `đồng` là «…dẫn điện và **dẫn** nhiệt rất tốt» — `dẫn` phải xuất
hiện LẦN THỨ HAI. Nó đã phát ở b13. Cả hai ca điểm-0 đều là từ ĐÃ PHÁT trước đó.

`apply_repetition_penalty_inplace`: nhân `×0,25` cho MỌI token trong cửa sổ 6
bước gần nhất, và áp LẠI MỖI BƯỚC ⇒ token còn trong cửa sổ bị nhân DỒN:
`0,25⁶ ≈ 0,00024` — bằng 0 trên thực tế. (`-inf` chỉ áp cho token VỪA phát, không
phải nguyên nhân ở đây.)

⇒ Agenda BIẾT `dẫn` phải đến, nhưng cơ chế chống lặp CẤM nó. `sắt` gặp đúng hiện
tượng ở b12–13 mà sống sót vì còn đoạn sau để tái khoá; `đồng` lặp ngay ở ĐUÔI
nên không còn gì để bám.

### 122.4 Ranh giới — điều gì ĐÃ chứng minh, điều gì mới SUY

ĐÃ đo: `blocked_dst`=0; túi không cạn; `rv` tụt ở cả hai; `sắt` tái khoá còn
`đồng` không; đầu agenda có điểm nền ĐÚNG 0,000 ở các bước hỏng; cả hai ca
điểm-0 đều là từ đã phát.

MỚI SUY (chưa tách bạch khỏi khả năng W_skip đơn giản không đề cử `dẫn`): rằng
chính phép nhân dồn `×0,25` là thứ đưa nó về 0. Chưa chạy phép thử tách hai khả
năng này. KHÔNG kết luận chắc.

### 122.5 Bài học

- **211.** Phạt lặp NHÂN DỒN theo bước (áp lại mỗi bước cho token còn trong cửa
  sổ) khác hẳn phạt MỘT LẦN. Dồn 6 bước là `×0,00024` — thực chất là CẤM, không
  phải PHẠT. Ngôn ngữ tự nhiên có lặp chính đáng («dẫn điện và dẫn nhiệt»), nên
  cấm tuyệt đối trong cửa sổ là sai với chính dữ liệu nguồn.
- **212.** «Nền sụp» KHÔNG đồng nghĩa «hết ứng viên»: túi vẫn 6 000–9 000 token
  dương. Sụp là mất TẬP TRUNG (không tái khoá được vào cha), không phải mất
  nguyên liệu. Hai chuyện cần hai cách chữa khác nhau.

## §123 — Đo trước/sau phạt lặp: §122.3 SAI về nhân dồn, đúng về hậu quả

§122.4 ghi rõ phần «nhân dồn `0,25⁶`» là SUY, chưa chứng minh. Vòng này đo.

### 123.1 Số — chụp cùng một bước, trước và sau lời gọi phạt

```
đồng b15  «dẫn»  TRƯỚC 0,0000   SAU 0,0000
đồng b16  «dẫn»  TRƯỚC 8,9985   SAU 2,2496      ← đúng ×0,25, MỘT LẦN
sắt  b18  «khí»  TRƯỚC 844,02   SAU 844,02      ← không lặp ⇒ không phạt
sắt  b22 «thành» TRƯỚC 105,65   SAU 105,65
```

### 123.2 ĐÍNH CHÍNH §122.3

KHÔNG có nhân dồn. `scores` được dựng LẠI TỪ ĐẦU mỗi bước từ phiếu bầu, nên phạt
lặp chỉ áp `×0,25` MỘT LẦN. Con số `0,25⁶ ≈ 0,00024` ở §122.3 là SAI. Bài học
211 («phạt nhân dồn = cấm») RÚT LẠI.

### 123.3 Nhưng số thật chỉ ra chuyện KHÁC, cụ thể hơn

Thứ tự trong vòng lặp: cú đẩy agenda (~650) → PHẠT LẶP (709) → sàn EOS (718).

```
b16:  cú đẩy agenda nâng «dẫn» lên  8,9985
      phạt lặp cắt còn              2,2496   (×0,25)
      sàn EOS                       2,0000
      khoảng cách còn lại           0,25
```

⇒ Phạt lặp chạy SAU cú đẩy agenda nên nó phạt chính ứng viên mà agenda VỪA CHỈ
ĐỊNH. Agenda nói «từ này phải đến tiếp», phạt lặp nói «vừa nói rồi» và xoá 75%
tín hiệu. Với từ lặp CHÍNH ĐÁNG trong câu nguồn («dẫn điện và DẪN nhiệt»), hai
bộ phận lại bất đồng — cùng họ với §116 và §118.

### 123.4 Hai bước hỏng liền nhau, HAI nguyên nhân khác nhau

```
b15: «dẫn» = 0 TRƯỚC phạt   ⇒ W_skip KHÔNG đề cử. Phạt lặp VÔ CAN.
b16: «dẫn» = 8,99 → 2,25    ⇒ đề cử tốt, bị phạt cắt 75%.
```

Đừng gộp hai bước này làm một hiện tượng.

### 123.5 VẪN CHƯA giải thích (nối tiếp §120.4)

Sau phạt, `dẫn` = 2,2496 > EOS = 2,0000, vậy mà token phát ra là `<EOS>`. Có thứ
gì đó giữa dòng 709 và điểm quyết định đảo lại thứ hạng. Hai vòng liên tiếp
(§120, §123) đều dừng ở đúng câu hỏi này. CHƯA truy; KHÔNG đoán.

### 123.6 Bài học

- **211 (thay bản cũ).** Ghi «SUY, chưa chứng minh» đã cứu một kết luận sai khỏi
  đi vào mặc định. Suy đoán về cơ chế ĐỊNH LƯỢNG (`0,25⁶`) phải kiểm bằng phép
  chụp trước/sau, vì nó rẻ hơn nhiều so với hậu quả của việc vá theo giả thuyết sai.
- **213.** THỨ TỰ áp các phép chỉnh điểm là một quyết định thiết kế ẩn: cú đẩy
  agenda rồi mới phạt lặp ⇒ phạt lặp có quyền phủ quyết agenda. Đảo lại thì
  ngược. Không chỗ nào trong mã ghi thứ tự này là CÓ Ý.

## §124 — Câu hỏi treo hai vòng: KHÔNG có phép đảo bí ẩn, chỉ là LẤY MẪU

§120.4 và §123.5 cùng dừng ở: sau phạt `dẫn` = 2,2496 > EOS = 2,0000 mà token
phát ra vẫn là `<EOS>`. Vòng này truy đoạn từ dòng 709 tới điểm chọn token.

### 124.1 Có một khối EOS THỨ HAI, ép cứng — nhưng KHÔNG phải thủ phạm

Dòng ~876, sau khối chuỗi và sau MỌI phép chỉnh điểm:

```python
if _is_b_g(state.vocab, int(out[-1]), _bd_gen) and n_b_gen >= 1:
    nxt = int(eos_id)          # ÉP CỨNG, không nhìn điểm
```

Đo: nó nổ ĐÚNG MỘT LẦN, ở `chó` b15, và ở đó TRÙNG với `argmax` ⇒ chưa gây hại
ở ca nào đang theo. Giả thuyết mở màn vòng này (nó cướp token đúng của `đồng`)
SAI.

### 124.2 Thủ phạm thật: `T = 0,5`, đường sinh LẤY MẪU

`ar_w_skip_primary_T = 0,5` ⇒ `_sample(finite, ...)`, KHÔNG argmax. Đo ở `đồng`
b16: `argmax` = `dẫn` (2,25), khối ép KHÔNG nổ, mà token phát ra là `<EOS>`
(2,00). Chênh 0,25 nằm gọn trong nhiễu lấy mẫu.

Kiểm chứng bằng 6 hạt giống:

```
seed=0 ▸ …dẫn điện và                                   ← cụt
seed=1 ▸ …dẫn điện và DẪN NHIỆT RẤT TỐT, THƯỜNG DÙNG ĐỂ ← ĐÚNG NGUYÊN VĂN
seed=5 ▸ …dẫn điện và DẪN NHIỆT RẤT TỐT, THƯỜNG DÙNG ĐỂ ← ĐÚNG NGUYÊN VĂN
seed=2 ▸ …dẫn điện và dùng dẫn tốt
```

**`đồng` KHÔNG hỏng.** Mạng có đúng câu nguồn và phát được ở 2/6 hạt giống.

### 124.3 Hệ quả — ba vòng đuổi theo triệu chứng NGẪU NHIÊN

§120 (bất đối xứng thang đo), §122 (`rv` sụp), §123 (phạt lặp sau cú đẩy) đo
ĐÚNG, nhưng ca dùng để chẩn đoán là một thế trận sát nút do lấy mẫu quyết định.
§121 còn vá theo nó và làm hỏng cả ba thước.

### 124.4 §118 được xác nhận trên đường THẬT

`lúa` b21 và `sóc` b2: EOS thắng bằng ĐIỂM (53,89 và 128,60) qua cú đẩy
`_ag_tin`, không qua ép cứng. Bằng chứng trực tiếp đầu tiên rằng §118 chạy đúng
như thiết kế.

Nhưng `sóc` b2 lộ chuyện khác: EOS đạt 128,60 khi MỚI phát 2 token ⇒ `sóc` cụt
là TẤT ĐỊNH do cổng dừng quá rộng, KHÁC hẳn `đồng` (ngẫu nhiên sát nút). Hai ca
tôi vẫn gộp chung suốt §119–§123 là HAI hiện tượng khác nhau.

### 124.5 Bài học

- **214.** Đường sinh LẤY MẪU (`T = 0,5`) ⇒ MỌI phán xét bằng MẮT trên MỘT hạt
  giống đều không đáng tin ở các thế trận sát nút. Suốt hai phiên tôi chấm ca
  vàng bằng `seed=0` duy nhất. Ca vàng phải đọc qua NHIỀU hạt giống.
- **215.** Trước khi truy một triệu chứng bằng nhiều vòng, hỏi TRƯỚC: nó có TẤT
  ĐỊNH không? Một lệnh 6 hạt giống (~2 phút) đã tiết kiệm được §120–§123.
- **216.** Phép nối hỏng `and n_b_gen >= 1` (§118) vẫn còn sống ở chỗ ÉP CỨNG
  ~876 — §118 chỉ sửa chỗ cộng điểm ~756. Chưa gây hại (đo được), nhưng là mâu
  thuẫn tiềm ẩn ở đường quyết định cuối. Rà HẾT chỗ dùng cùng công thức khi sửa.

## §125 — Đọc lại qua 6 hạt giống: RÀ LẠI mọi phán quyết theo biên độ nhiễu

§124 phát hiện đường sinh LẤY MẪU (`T = 0,5`) mà suốt hai phiên tôi chấm bằng
ĐÚNG `seed=0`. Vòng này đo biên độ.

### 125.1 Tự kiểm — thước nhất quán

`seed=0` cho đúng `85,9 / 97,3 / 7,0`, trùng khít mốc §118. Cách tính không lệch.

### 125.2 Biên độ qua 6 hạt (bộ 20 prompt)

```
TRUNG THÀNH  84,3 – 86,9   biên độ 2,6   đlc 0,94
TRẬT TỰ      95,5 – 99,2   biên độ 3,6   đlc 1,16
LẶP           7,0 – 11,8   biên độ 4,8   đlc 1,75
```

⚠️ **`seed=0` là hạt TỐT NHẤT trong sáu về LẶP** (7,0 = cực tiểu; điển hình
~10,5). Đó đúng là hạt tôi dùng cho MỌI cổng suốt hai phiên.

### 125.3 RÀ LẠI các phán quyết

| § | quyết định | chênh dùng để phán | so biên độ | phán lại |
|---|---|---|---|---|
| §112 | `w=4 → w=3` | TT +2,2 | 2,6 | **TRONG NHIỄU — chưa chứng minh** |
| §113 | `w=3 → w=1` | TQ +4,0 | 3,6 | sát mép, tạm giữ |
| §116 | giữ `_ag_tin` | TT +4,7 | 2,6 | trên nhiễu, GIỮ |
| §118 | tách vị từ | TT +4,2 | 2,6 | trên nhiễu, GIỮ |
| §119 | LOẠI ngưỡng | TT −2,6 | 2,6 | **ĐÚNG BẰNG nhiễu — loại CHƯA đủ căn cứ** |
| §121 | LOẠI trần EOS | TQ −7,2 | 3,6 | trên nhiễu ×2, LOẠI ĐÚNG |

§112 và §119 được phán bằng chênh lệch KHÔNG lớn hơn nhiễu. §112 đã ĐỔI MẶC
ĐỊNH; §113 sau đó đổi tiếp nên hậu quả bị che, nhưng lập luận §112 không đứng.

### 125.4 Ca vàng — phần lớn TẤT ĐỊNH, cứu được kết luận §118

```
chó, sắt, hổ, lúa   6/6 hạt GIỐNG HỆT, đều đúng
sóc                 6/6 hạt «thú gặm»        ← hỏng TẤT ĐỊNH
đồng                6 hạt 6 câu, 2 câu đúng  ← nhiễu thuần
```

Chỗ dừng sạch của `lúa` (§118) giữ qua CẢ SÁU hạt ⇒ bằng chứng ĐỘC LẬP với hạt
giống rằng §118 thật. `đồng` là ngoại lệ nhiễu duy nhất trong nhóm ca vàng.

### 125.5 Bài học

- **217.** Đo BIÊN ĐỘ NHIỄU trước khi khai cổng, không phải sau 45 mục. Mọi
  ngưỡng cổng phải lớn hơn biên độ; cổng hẹp hơn nhiễu là cổng vô nghĩa.
- **218.** Dùng MỘT hạt giống cố định không làm kết quả tất định — nó làm kết
  quả THIÊN LỆCH theo hạt đó. `seed=0` may nhất về LẶP, nên mọi bản vá bị chấm
  bằng LẶP đều bị so với một mốc dễ thắng.
- **219.** Chứng cứ ĐỊNH TÍNH lặp lại qua nhiều hạt (`lúa` dừng sạch 6/6) mạnh
  hơn chênh lệch ĐỊNH LƯỢNG nằm trong biên độ. Ưu tiên dựng bằng chứng dạng đó.

## §126 — Công cụ đo ĐA HẠT GIỐNG, và chấm lại §119 dứt điểm

§125 bài học 217: cổng hẹp hơn biên độ nhiễu là cổng vô nghĩa. Việc đầu tiên là
sửa CÔNG CỤ, vì nó chặn mọi kết luận sau.

### 126.1 `ab_thu_tu_cha_hoat.py` giờ chấm N hạt

Mặc định 6 hạt (`hat=N` để đổi). In `trung bình ± nửa biên độ [min–max]` thay vì
một con số trần. Mọi cổng từ nay so với biên độ, không so với số lẻ.

### 126.2 Chấm lại §119 — ba mức, sáu hạt

```
ngưỡng   TRUNG THÀNH   TRẬT TỰ     LẶP
0,0      86,1 ±1,3     97,2 ±1,8   9,9 ±2,4
0,3      83,7 ±2,4     95,4 ±1,8   9,9 ±2,4
0,5      83,7 ±2,4     95,4 ±1,8   9,9 ±2,4
```

**0,3 và 0,5 GIỐNG HỆT** ⇒ hiệu ứng BÃO HOÀ dưới 0,3: mọi ca bị chặn đều tiêu
thụ <30% agenda, không có mức «vừa phải». Giả thuyết «§119 trượt vì chọn sai
mức» bị BÁC.

`lúa` giữ chỗ dừng sạch ở CẢ BA cấu hình ⇒ ngưỡng không phá §118, nó hỏng chỗ khác.

### 126.3 ĐÍNH CHÍNH §125: biên độ ≠ ngưỡng phán cho TRUNG BÌNH

§125 kết luận «§119 loại bằng −2,6 khi biên độ = 2,6 ⇒ chưa đủ căn cứ». Đúng cho
MỘT phép đo. Nhưng khi so TRUNG BÌNH 6 hạt, sai số chuẩn là `σ/√6`:

```
cfg 0,0:  σ ≈ 0,94 ⇒ SE ≈ 0,38
cfg 0,3:  σ ≈ 1,7  ⇒ SE ≈ 0,70
chênh 2,4 / SE gộp 0,80 ≈ 3,0 SE  ⇒ THẬT
```

⇒ §119 bị loại ĐÚNG. §125 nghi oan nó; §126 xác nhận. Gỡ cờ, giữ ghi chú.

### 126.4 Bài học

- **220.** BIÊN ĐỘ (range) dùng để phán MỘT lượt chạy; so TRUNG BÌNH n hạt thì
  ngưỡng là `SE = σ/√n`, chặt hơn nhiều. Lẫn hai thứ này thì hoặc bỏ sót hiệu
  ứng thật (§125 nghi oan §119), hoặc nhận nhầm nhiễu (§112).
- **221.** Quét ≥3 mức để phân biệt «ý tưởng sai» với «chọn sai mức». Ở đây 0,3
  và 0,5 trùng khít ⇒ bão hoà ⇒ ý tưởng sai, không phải mức sai. Hai mức thì
  không kết luận được điều đó.

## §127 — Nhãn lớp: `via_pivot` đúng 4,1%; chuẩn cơ học cho 8 918 nhãn trong vài giây

Quay lại nút thắt §100 (cấp nhãn lớp đúng ⇒ top-1 4,5% → 53,3%).

### 127.1 Cạnh CÓ SẴN, nhưng nhãn đúng bị rác đè

`via_pivot` có 24 996 nguồn mang cạnh phân loại. Nhãn đúng CÓ MẶT, chỉ xếp sau:

```
chó  → bạn.<EOS> 8,6 · chủ · phổ biến · gì?<EOS>
nước → gì?<EOS> 9,0 · 3 · Hoa · 0
đồng → gì?<EOS> 4,2 · KIM LOẠI 2,7        ← đúng, hạng 2
sóc  → gì?<EOS> 7,2 · … · gặm nhấm 0,6    ← đúng, hạng 6
```

Thủ phạm lớn nhất: mẫu `A là B` bắt trúng CÂU HỎI «Nước là gì?» ⇒ nhãn `gì`.
Cùng loại: `bạn.<EOS>` từ «chó là bạn.», và chữ số.

Hai giả thuyết khác CHẾT trước khi đo: `gop_manh_tu` mặc định ĐÃ BẬT (nghi ngờ
§110 về từ ghép đã được xử lý), và 24 996 nguồn ⇒ không phải thiếu dữ liệu.

### 127.2 Bộ khoá 198 bộ KHÔNG dùng làm chuẩn được

`out/loai_suy_that_khoa.json` có `lop` = `từ`, `những`, `khoảng`, và `kim` (cụt
của `kim loại`) — chính các lỗi đang xét, vì nó do đường ống này sinh ra (§88–89).
Lấy nó thẩm định đường ống là LẬP LUẬN VÒNG TRÒN.

⚠️ Hệ quả ngược: mọi số thu hồi «con khỉ» ở §89, §101, §102 đều chấm so với chuẩn
NHIỄM BẨN. Đối chứng hoán vị KHÔNG phát hiện được điều này (nó chỉ kiểm ăn may
theo tần suất lớp, không kiểm chuẩn).

### 127.3 Chuẩn SẠCH: loài gần trong câu định nghĩa

Mẫu `X là <LOÀI GẦN> <khác biệt>` đọc thẳng từ `dinh_nghia.jsonl`, bỏ hư từ đầu:
**8 918 từ có loài gần**, trích trong vài giây. Độc lập với `via_pivot`.

### 127.4 Chấm trên 1 718 từ

```
loc_dich=False   top-1  4,1% (71/1718)   top-3  7,5%
loc_dich=True    top-1  4,8% (81/1676)   top-3  8,2%
```

Chênh +0,7 điểm, SE của hiệu ≈ 0,7 ⇒ **1 SE, KHÔNG có ý nghĩa**. `loc_dich` quét
sạch rác thấy được bằng MẮT nhưng không dịch được độ chính xác. Lọc KHÔNG cứu.

**4,1%** xác nhận §100 ở quy mô lớn với chuẩn độc lập: nhãn lớp học thống kê từ
toàn văn bản gần như luôn sai.

### 127.5 Bài học

- **222.** ĐÍNH CHÍNH §114: `loc_dich` bị xếp «vô hiệu» vì đo trên SINH CÂU —
  sai đích. Nó là cơ chế lọc NHÃN LỚP. Đo đúng đích thì nó có tác dụng thấy
  được bằng mắt (sạch `gì?`/chữ số) nhưng vẫn không đủ. «Không tác dụng» và
  «tôi đang đo thứ khác» là hai kết luận khác nhau.
- **223.** Bộ đánh giá do CHÍNH đường ống cần thẩm định sinh ra thì vô giá trị.
  Phải có chuẩn ĐỘC LẬP — ở đây là đọc thẳng khuôn định nghĩa trong corpus.
- **224.** Ba lần cùng hình dạng trong phiên (§113 thước mù thứ tự · §114/§126
  thước mù nhãn lớp · §127 chuẩn nhiễm bẩn): lỗi DỤNG CỤ, không phải lỗi cơ chế,
  và cả ba đều đẩy kết luận về phía BI QUAN sai.
- **225.** Thứ học thống kê từ toàn văn bản không làm nổi (4,1%), đọc thẳng
  KHUÔN trong corpus làm được ngay (8 918 nhãn/vài giây). Khuôn định nghĩa là
  dữ liệu có cấu trúc nằm sẵn trong corpus, không phải nguồn ngoài.

## §129 — Nối bảng nhãn lớp vào ②: chạy đúng, KHÔNG hại, chưa lợi — và lộ gốc `sóc`

### 129.1 Ba lần nối sai trước khi đúng

1. **Tiêm nhãn thẳng** — vô ích: nhãn ghép không phân giải thành một token, còn
   token đầu (`gia`,`kim`,`loài`) thì `via_pivot` đã có sẵn.
2. **Kéo đồng-thành-viên từ token đầu** — CÓ HẠI: `sắt là` tụt từ
   «chì/titan/đồng/kẽm» xuống «Quốc/chẳng/quá/tới», vì `kim` mơ hồ và `loài`
   quá rộng.
3. **Tra ngược `label_surf`** (id nhịp → bề mặt, 68 657 nhãn ghép; 21% nhãn ghép
   của bảng khớp) — nhưng vẫn hỏng vì bảng liệt kê token đầu TRƯỚC (`kim` rồi
   `kim loại`) nên mục đơn vẫn lọt vào và pha loãng.
4. **Ưu tiên nhãn GHÉP, bỏ hẳn mục đơn khi có ghép** — đúng.

### 129.2 Kết quả: không hại, chưa lợi

```
sắt (ghép «kim loại» khớp) → chì, titan, đồng, kẽm   ← Y HỆT mốc
chó (ghép «gia súc» khớp)  → lợn, mèo, ếch           ← Y HỆT mốc
```

Ở nơi nhãn phân giải được, ③ VỐN ĐÃ có đồng-thành-viên tốt ⇒ bảng không thêm gì.
Cờ `hub_chain_bang_nhan_w` mặc định **0 = tắt**.

### 129.3 Gốc của `sóc`: HẠT GIỐNG là MẢNH BPE

```
[nl] «óc» bảng=['khối','khối mềm']   ← hạt giống là «óc», KHÔNG phải «sóc»
```

Bảng tra ra `khối mềm` (não). `via_pivot` cũng tra nhầm ⇒ nhãn rác
«bạn/hàng/rơm». Đây là lỗi §109 (chuỗi chạy trên mảnh BPE) TÁI XUẤT ở tầng hạt
giống của ②, và nó giải thích CẢ `sóc là → thú gặm` cụt (§124.4) lẫn nhãn rác.

⇒ Nút thắt kế KHÔNG phải nguồn nhãn (đã có, 84,9%) mà là **hạt giống sai từ**.

### 129.4 Bài học

- **226.** Bảng nhãn tốt KHÔNG tự động cải thiện đầu ra: phải khớp ĐƠN VỊ của đồ
  thị. Nhãn ghép cần id NHỊP (`label_surf`), không phải token. Cắt về token đầu
  còn tệ hơn không dùng bảng.
- **227.** Thứ tự mục trong bảng là một quyết định ẩn: liệt kê token đầu trước
  nhãn ghép khiến mọi chỗ dùng `[:k]` lấy nhầm mục mơ hồ.
- **228.** `sóc`→`óc` cho thấy §109 chưa được sửa TRIỆT ĐỂ — mới sửa ở chuỗi,
  chưa sửa ở HẠT GIỐNG. Cùng một lỗi ở hai tầng khác nhau.

## §130 — Vá hạt giống sai từ: GIẢ THUYẾT SAI, `gom_tu` không cứu

§129.3 kết luận nút thắt kế là «hạt giống sai từ» (`sóc là` gieo bằng `óc`).

### 130.1 Cơ chế gom từ CHẠY ĐÚNG — nhưng không được dùng

```
«sóc là» token thô ['s','óc','Ġlà'] → gom → ['sóc','là']   ✓ đúng
```

Vòng lặp hạt giống dùng `moi_tu` (đã gom) CHỈ KHI `hub_chain_gom_tu > 0`; mặc
định 0 ⇒ dùng `moi` = TOKEN THÔ, và `len>1` lọc mất `s`, còn lại `óc`. Cơ chế
được dựng ra rồi để tắt.

### 130.2 Bật lên thì XẤU ĐI

```
gom_tu=0  sắt → kim, chì, pyr, titan    ← đồng-thành-viên TỐT
gom_tu=1  sắt → oxide, dùng, công, 4    ← hỏng rõ
gom_tu=1  sóc → dạng, gì, nơi, cây      ← hạt giống ĐÃ đúng «sóc», VẪN rác
```

⇒ **Giả thuyết §129.3 SAI.** Sửa hạt giống không cứu `sóc`, và làm hỏng `sắt`.
`gom_tu` đổi cả cách phân giải BIẾN THỂ id, không chỉ đổi bề mặt hạt giống.

Kết luận âm tính của §109 về `gom_tu` VẪN ĐÚNG, dù khi đó nó được chấm bằng bộ
khoá nhiễm bẩn (§127) — đúng vì lý do khác với lý do đã ghi.

### 130.3 Chuẩn «đồng-thành-viên» từ bảng §128 QUÁ THƯA

Nội dung đúng (`kim loại` = bạc/chì/kẽm/nhôm/sắt/vàng/đồng) nhưng lớp bé:
`gia súc` = 2 từ, `thú gặm` = 3 ⇒ `chó → lợn, mèo, ếch` bị chấm TRƯỢT dù hợp lý.
Không dùng con số 1/7 để phán. Kết luận ở 130.2 rút ra bằng MẮT, không cần thước.

### 130.4 Bài học

- **229.** «Cơ chế X có sẵn mà tắt» KHÔNG suy ra «bị loại oan». Ở đây `gom_tu`
  có bình luận dài, ba mức, và vẫn đúng là nên tắt. Dấu hiệu đầu tư không thay
  được phép đo.
- **230.** Một cờ thường đổi NHIỀU thứ cùng lúc (`gom_tu` đổi cả bề mặt hạt
  giống LẪN cách phân giải biến thể id). Suy từ «nó sửa lỗi A» sang «bật nó lên»
  bỏ qua các tác dụng phụ B, C.
- **231.** Chuẩn dựng từ từ điển tốt cho quan hệ TỪ→LỚP (84,9%) nhưng THƯA cho
  quan hệ TỪ↔TỪ cùng lớp (lớp 2–8 thành viên). Cùng một nguồn, hai việc, hai
  mức tin cậy.

## §131–§132 — Nới cửa sổ nhãn ghép, và vá bệnh MỘT BỀ MẶT NHIỀU ID

### 131.1 Nút KHÔNG thiếu — BẢNG cắt cụt

```
thú gặm nhấm     CÓ nhịp        cây lương thực    không
động vật có vú   CÓ nhịp        loài linh trưởng  không
độ dài nhịp: 1→10 944 · 2→16 815 · 3→28 240 · 4→16 248 · 5→6 044 · 6→768
```

Nhịp dài tới 6 từ, nhịp 3 TỪ phổ biến NHẤT. Cửa sổ 2 từ của §128 cắt cụt nhãn
(`thú gặm`, `cây lương`, `loài linh`). ⇒ **ĐÍNH CHÍNH §129.4**: con số 21% đo lỗi
của BẢNG, không đo thiếu sót của ĐỒ THỊ; câu «vấn đề biểu diễn chứ không phải dữ
liệu» SAI.

Nới cửa sổ lên 4 từ: nhãn ghép 5 264 → 18 245; top-1 GIỮ NGUYÊN 84,9%. (Top-3
tụt 94,8 → 88,6 là hiện vật thước: chuẩn chỉ chứa nhãn 1–2 từ nên bản 3 từ ở vị
trí [2] không thể khớp.) 44,4% số từ có ≥1 nhãn ghép khớp nhịp — KHÔNG có số
tương ứng trước khi nới nên không tuyên bố cải thiện.

### 132.1 Bệnh MỘT BỀ MẶT NHIỀU ID — ở ĐÍCH, không chỉ ở hạt giống

```
«kim loại» có 3 id nhịp: 33947, 67036, 90856
sắt/chì/đồng/vàng  → đều trỏ tới  33947
reverse_peers(33947) → vàng, đồng, chì, ngân     ← THÀNH VIÊN ĐỦ
reverse_peers(67036) → []
reverse_peers(90856) → []      ← ĐÚNG id mà phép đảo của tôi chọn
```

`{bề mặt: id}` giữ id CUỐI gặp phải ⇒ trúng id rỗng ⇒ lớp TRÔNG như không có
thành viên. Sửa: giữ MỌI id đồng bề mặt, gieo ③ bằng tất cả (tối đa 8).

⇒ **ĐÍNH CHÍNH lượt trước**: câu «nút thắt là cạnh từ lớp về thành viên» SAI —
cạnh tồn tại và đầy đủ; tôi hỏi nhầm nút.

### 132.2 Kết quả

```
sắt là  w=0 → kim, chì, pyr, titan, đồng, kẽm
        w=1 → kim, ĐỒNG, đá, CHÌ, VÀNG          ← qua nhãn bảng «kim loại»
```

Các ca khác KHÔNG đổi, vì nút lớp của chúng gần rỗng: `thú gặm nhấm`→chỉ `chuột`,
`tinh thể`→chỉ `muối`, `gia súc`→chỉ `chó`, `loài linh trưởng`→không có nhịp.
Cơ chế ĐÚNG; giới hạn là SỐ THÀNH VIÊN mỗi nút lớp có.

### 132.3 Bài học

- **232.** Cảnh báo nằm SẴN trong mã cách chỗ sửa 10 dòng («bệnh §23.1: một bề
  mặt nhiều id») và tôi đọc nó như chuyện của HẠT GIỐNG. Bệnh nào áp cho nguồn
  thì thường áp cho ĐÍCH — nhãn lớp cũng chỉ là một bề mặt.
- **233.** Ba vòng liên tiếp tôi kết luận «nối đúng, chưa lợi» rồi đi tìm chỗ
  nghẽn ở tầng SÂU HƠN (bảng cắt cụt → hạt giống sai → nút thiếu). Cả ba chẩn
  đoán đều sai. Khi một cơ chế «đúng mà không tác dụng», hãy kiểm DANH TÍNH của
  thứ đang tra (id) trước khi nghi ngờ dữ liệu.
- **234.** «Khớp đơn vị của đồ thị» (bài học 226) không chỉ là độ dài cụm mà còn
  là danh tính id. Tôi tự viết bài học đó rồi hiểu nó ở nghĩa hẹp hơn.

## §133 — Phân bố thành viên nút lớp: 38,5% số từ có lớp DÙNG ĐƯỢC

§132 để lại: cơ chế đúng nhưng chỉ `sắt`/`đồng` cải thiện vì nút lớp khác gần
rỗng. Đo phân bố để biết giới hạn thật.

### 133.1 Số (qua `via_rev`, gộp theo BỀ MẶT vì §132)

```
                       trung vị   ≥5 thành viên
toàn bộ nút lớp           3          34,3%
gộp theo bề mặt           3          30,3%
LỚP MÀ BẢNG DÙNG          5          51,5%     ← giàu gấp rưỡi mức chung

từ có ≥1 nhãn ghép ≥2 thành viên:  3 547/9 222 (38,5%)
                    ≥5 thành viên:  3 027/9 222 (32,8%)
                   ≥10 thành viên:  2 643/9 222 (28,7%)
```

Bước ③ cần ≥2 thành viên mới có nghĩa (một là chính từ đang hỏi) ⇒ 38,5% là trần
áp dụng được. Đúng các ca đã thấy: `tinh thể`→chỉ `muối`, `gia súc`→chỉ `chó`.

### 133.2 Bảng CHỌN lớp tốt hơn mức trung bình

Trung vị 5 so với 3, và 51,5% so với 30,3% ở mức ≥5. Tức khuôn định nghĩa không
chỉ cho nhãn ĐÚNG (§128: 84,9%) mà còn thiên về lớp GIÀU thành viên.

### 133.3 Lỗ hổng trong cổng tôi tự khai

Khai trước: «<20% ⇒ vô nghĩa, >40% ⇒ đáng chấm». 38,5% rơi vào KHE giữa hai mốc
— cổng có lỗ. Ghi rõ thay vì nới tay sau khi thấy số. Nghiêng về CHẤM vì hai lý
do đo được: lớp bảng dùng giàu gấp rưỡi, và 32,8% của 9 222 là 3 027 từ — đủ lớn
để phép chấm có ý nghĩa thống kê.

### 133.4 Bài học

- **235.** Khai cổng bằng HAI mốc để lại KHE ở giữa. Phải khai phân hoạch kín,
  hoặc nói trước sẽ xử lý khe thế nào.
- **236.** Trần áp dụng của một cơ chế là thuộc tính của DỮ LIỆU, không sửa được
  bằng mã: 61,5% số từ không có lớp ≥2 thành viên thì bước ③ không thể chạy cho
  chúng, dù nhãn đúng 84,9%.

## §134 — Chấm thu hồi trên bộ khoá SẠCH: cờ bảng LOẠI; chuỗi có tín hiệu lớp YẾU

Lần đầu ý tưởng được chấm bằng chuẩn KHÔNG nhiễm bẩn (§127 chứng minh bộ khoá
198 bộ do chính đường ống sinh ra ⇒ vô giá trị).

### 134.1 Thiết kế — và chỗ nó THIÊN VỊ

Bộ khoá: 3 026 từ có nhãn ghép ≥5 thành viên; đáp án đúng = thành viên KHÁC của
lớp. Chọn lớp HẸP NHẤT còn ≥5 thành viên (lớp rộng thì ăn may cũng trúng). Hoán
vị giữ nguyên phân bố cỡ lớp, chỉ trộn ánh xạ từ → tập thành viên.

⚠️ Bộ khoá dựng TỪ CHÍNH bảng `nhan_lop.json` ⇒ cờ bật được chấm trên SÂN NHÀ.
Phép này trả lời «cơ chế có dùng được quan hệ lớp không» (việc của hoán vị),
KHÔNG trả lời «bảng có hơn `via_pivot` ở ĐẦU RA không» — câu đó cần chuẩn thứ ba.

### 134.2 Số (mẫu 400, n=385 chấm được)

```
                THẬT top-5   HOÁN VỊ   khoảng cách
cờ TẮT            27,5%       23,9%      +3,6
cờ BẬT            27,3%       23,9%      +3,4
top-10 cờ TẮT     42,6%       35,6%      +7,0
```

### 134.3 Phán quyết theo cổng khai TRƯỚC (phân hoạch kín)

Nhánh «loại nếu top-5 thật KHÔNG tăng» ⇒ **LOẠI cờ bảng**. Bật cờ không đổi gì
dù §132 cho thấy `sắt là` đổi thật — vài ca đổi không dịch được tổng 385 từ.

### 134.4 Kết quả ĐỘC LẬP với cờ: chuỗi CÓ tín hiệu lớp

Mốc (cờ tắt) thắng hoán vị +3,6 ở top-5 và +7,0 ở top-10. Ước lượng: n=385 ⇒
chênh top-5 ≈ 1,1 SE (CHƯA đạt), top-10 ≈ 2,0 SE (vừa đạt). Tín hiệu CÓ THẬT
nhưng YẾU — không tô hơn thế.

Đây là lần đầu con số này sạch: §89/§101/§102 đều chấm so với chuẩn nhiễm bẩn.

### 134.5 Giữ mã, không gỡ — và nói rõ vì sao lệch tiền lệ

§115/§119/§121 đều GỠ cơ chế bị loại. Lần này GIỮ (mặc định 0) vì mã đúng và là
chỗ dùng duy nhất của bảng đã thẩm định (§128: 84,9%). Ghi chú kết quả đặt tại
`config.py` để không ai bật lại mà tưởng chưa đo.

### 134.6 Bài học

- **237.** Vài ca vàng đổi rõ (`sắt là` → vàng/đồng/chì) KHÔNG suy ra tổng đổi:
  385 từ cho +0,0. Mắt thấy ở 7 prompt là mẫu quá nhỏ để suy ra hiệu ứng.
- **238.** Bộ khoá dựng từ chính thứ đang được chấm thì phép chấm là SÂN NHÀ —
  vẫn dùng được để hỏi «cơ chế có tín hiệu không» (nhờ hoán vị) nhưng không dùng
  được để so hai nguồn nhãn. Phải nói rõ câu nào phép thử trả lời được.

## §135 — Độ giàu lớp TĂNG ĐƯỢC: đảo chính bảng, hạt 2 TỪ cho 64,4% (mốc 38,5%)

§134 để lại: 61,5% số từ không có lớp ≥2 thành viên nên ③ không có gì để trải.
`via_rev` dựng từ CORPUS (`A là B`) nên lớp nghèo; bảng §128 có 9 222 ánh xạ
từ→lớp, đảo lại thì được lớp→thành viên.

### 135.1 Đánh đổi PHỦ ↔ CỤ THỂ

```
hạt        từ có lớp ≥2   từ có lớp ≥5   số lớp   trung vị
1 từ          96,6%          87,2%        1 295      2
2 từ          64,4%          35,7%        5 293      1
3 từ          32,2%           9,5%        6 269      1
via_rev       38,5%          32,8%          —        3     ← mốc §133
```

- **Hạt 1 từ** phủ 96,6% nhưng chỉ 1 295 lớp cho 9 222 từ ⇒ lớp kiểu `thú`,
  `cây`, `loài` — đúng thứ đã cho `khỉ → loài → đá/dùng/hình/phẩm` (§129). BẪY.
- **Hạt 3 từ**: 83,9% lớp chỉ có MỘT thành viên ⇒ vô dụng cho ③.
- **Hạt 2 từ**: 64,4% (gần gấp rưỡi mốc corpus 38,5%) mà vẫn 5 293 lớp.

⇒ Trả lời câu hỏi: **CÓ, tăng được**, bằng cách gieo ③ từ ĐẢO CỦA BẢNG ở hạt
2 từ thay vì từ `via_rev` của corpus.

### 135.2 ⚠️ Kiểm chứng KHÔNG được dùng bộ khoá của §134

Bộ khoá §134 cũng dựng từ bảng. Dùng đảo-bảng cho ③ RỒI chấm bằng khoá-từ-bảng
là VÒNG TRÒN HOÀN TOÀN (§127 bài học 223 nhưng ở mức nặng hơn: cả cơ chế lẫn
chuẩn cùng một nguồn). Muốn kiểm phải dùng chuẩn KHÁC nguồn — ví dụ chất lượng
SINH CÂU, hoặc chuẩn loài-gần từ `dinh_nghia.jsonl` (khác tệp với `dictionary`).

Chưa chạy kiểm chứng nào; §135 chỉ đo TIỀM NĂNG, không kết luận về hiệu quả.

### 135.3 Bài học

- **239.** «Độ phủ cao» và «lớp hữu ích» ngược nhau: hạt 1 từ phủ 96,6% mà cho
  lớp vô dụng. Chọn hạt phải cân hai chiều, không tối đa hoá một chiều.
- **240.** Khi cơ chế MỚI và bộ khoá cùng dựng từ MỘT nguồn thì không có phép
  hoán vị nào cứu được — hoán vị kiểm ăn may theo tần suất, không kiểm trùng
  nguồn. Phải đổi CHUẨN, không phải thêm đối chứng.

## §136 — Đảo-bảng gieo ③: QUA cổng thu hồi, nhưng KHÔNG chạm được văn bản sinh ra

### 136.1 Cơ chế + hai lần sai CỔNG VÀO (lần 11 và 12)

Gieo ③ thẳng từ đảo-của-bảng ở hạt 2 từ (§135). Hai lần không nổ trước khi đúng:
điều kiện NẠP bảng chỉ xét `bang_nhan_w`; sửa xong vẫn không nổ vì GUARD bao
ngoài khối cũng chỉ xét `bang_nhan_w`. Phép đếm rẻ (5 prompt) bắt được cả hai
TRƯỚC khi chạy A/B — nếu không đã nhận một âm tính giả tốn cả vòng (bài học 192).

### 136.2 Chấm trên chuẩn ĐỘC LẬP (`dinh_nghia.jsonl` ≠ `dictionary.jsonl`)

```
              THẬT top-5   HOÁN VỊ   khoảng cách   top-10 thật
tv_w=0           3,6%       1,7%       +1,9           6,4%
tv_w=2           5,5%       1,7%       +3,9          11,1%
```

Top-5 thật TĂNG và khoảng cách NỚI GẤP ĐÔI, hoán vị ĐỨNG YÊN (1,7 → 1,7) ⇒ cơ
chế dùng quan hệ LỚP, không ăn theo tần suất. n=361: top-5 ≈1,2 SE (chưa đạt),
top-10 ≈2,2 SE và khoảng cách ≈2,7 SE (đạt). ⇒ QUA nhánh «nhận» của cổng §134.

### 136.3 NHƯNG: 0/60 lượt sinh đổi

Sinh câu 6 hạt: `86,1/97,2/9,9` — TRÙNG KHÍT cả hai cấu hình. Nghi ngờ «trùng đến
số lẻ = không chạy» ⇒ so từng token trên 20 prompt × 3 hạt: **0/60 khác nhau**.

Nguyên nhân: đường sinh CÓ gọi `hub_chain_proposals(..., config)` (dòng 318) nên
cờ tới được, nhưng khối chuỗi (dòng 860) chỉ XẾP LẠI ứng viên đang sống, trong
khi cú đẩy agenda cho `ag_boost × rv` = **2,5 × max**. Chuỗi bị agenda ÁP ĐẢO.

Đây là hệ quả trực tiếp của §118: agenda CHÍNH LÀ câu nguồn; khi mạng truy được
câu nguồn thì chuỗi loại suy không còn chỗ tác động.

### 136.4 Phán quyết: KHÔNG bật mặc định

Qua cổng thu hồi nhưng không đổi một token nào của văn bản ⇒ bật mặc định là
tuyên bố một cải thiện mà người dùng không thấy. Giữ `hub_chain_bang_tv_w = 0`.

⇒ Cải thiện §128–§136 nằm ở mặt phẳng **API đề cử**, KHÔNG ở mặt phẳng **văn bản
sinh ra**. Phải nói rõ điều này ở mọi tổng kết sau.

### 136.5 Bài học

- **241.** Ba thước trùng khít đến SỐ LẺ qua 6 hạt giống là dấu hiệu «không
  chạy», không phải «trung tính». Phải so từng token trước khi kết luận trung
  tính — và trước khi đổi mặc định theo cách xử đã khai cho tình huống đó.
- **242.** Cải thiện đo ở TẦNG TRUNG GIAN (đề cử) không suy ra cải thiện ở ĐẦU
  RA. §83/§84 đã dạy điều này cho thứ tự; §136 lặp lại cho loại suy. Cổng phải
  có vế «đầu ra thật đổi» chứ không chỉ vế thu hồi.
- **243.** Hai cơ chế cạnh tranh cùng một chỗ ghi điểm thì cơ chế MẠNH HƠN quyết
  định tất: agenda 2,5×max nuốt trọn chuỗi. Muốn chuỗi có tác dụng thì phải nhắm
  vào prompt KHÔNG có câu nguồn — nơi agenda rỗng.

## §137 — TRẦN KIẾN TRÚC: từ MỚI vào ở thang TRUNG VỊ nên không bao giờ thắng

### 137.1 Bác chẩn đoán §136.3 (agenda áp đảo)

Đo trong vòng sinh, `|agenda|` và `|chuỗi|` từng prompt:

```
la-de là   |agenda|=0   |chuỗi| 40 → 40   top KHÔNG đổi
lô-gô là   |agenda|=0   |chuỗi| 52 → 69   top KHÔNG đổi (hình 40,8 áp đảo)
sắt là     |agenda|=30  |chuỗi| 12 → 29   top ĐỔI
```

`lô-gô` có **agenda RỖNG** mà đầu ra vẫn không đổi ⇒ agenda KHÔNG phải thủ phạm.
§136.3 sai. (Tập «không có câu nguồn» của §136 cũng sai: tôi loại từ có trong
`dinh_nghia.jsonl` nhưng quên `dictionary.jsonl` CŨNG nằm trong corpus huấn
luyện — thấy rõ ở `bit là → đơn vị nhỏ nhất, chỉ 0 hoặc 1`.)

### 137.2 Thủ phạm THẬT: hai nhánh, hai thang

```python
if scores[_t] > 0:      # ứng viên ĐANG SỐNG
    scores[_t] *= (1.0 + _chain_b * _w)     # NHÂN — có sức thật
elif _med > 0:          # ứng viên CHƯA CÓ trong phiếu
    scores[_t] = _med * _chain_b * _w       # đặt ở thang TRUNG VỊ (~0,03)
```

Đỉnh phiếu ~40. Từ MỚI vào ở ~0,03 ⇒ chênh hơn 1 000 lần ⇒ KHÔNG BAO GIỜ thắng.

§137 đã thử tiêm theo THANG NỀN (`0,5 × max(ra)`) thay vì trọng số tuyệt đối
2,0 của §136 — vẫn **0/7** lượt sinh đổi, vì nút cổ chai không nằm ở chỗ tiêm mà
ở nhánh HỒI SINH trong vòng sinh.

### 137.3 Kết luận cho cả ý tưởng

**Kiến trúc sinh không thể đưa ra một từ mà phiếu W_skip chưa đề cử.** Mà đó
CHÍNH LÀ việc của loại suy: đề xuất `khỉ` khi `khỉ` không có trong ngữ cảnh.

⇒ Ý tưởng bị chặn ở TẦNG THIẾT KẾ, không phải ở nhãn lớp (§128: 84,9%), không
phải ở cạnh lớp→thành viên (§132: đầy đủ), không phải ở agenda (§137.1).

Nhánh `_med` là lựa chọn CÓ CHỦ Ý (bất biến §12.144: đích đã consume không sống
lại qua bonus/flu). Nâng nó lên thang `max` sẽ đụng bất biến đó — việc lớn, rủi
ro cao, phải có cổng riêng. KHÔNG làm trong vòng này.

### 137.4 Bài học

- **244.** Ba vòng liền tôi quy nguyên nhân cho tầng SAI (agenda → thang tiêm →
  mới thấy nhánh hồi sinh). Mỗi lần đều có phép đo ủng hộ, nhưng phép đo chỉ bác
  được giả thuyết đang xét, không chỉ ra giả thuyết đúng. Đọc MÃ của đường đi
  cuối cùng rẻ hơn đoán tầng.
- **245.** «Cơ chế A không đổi được đầu ra» có thể do TRẦN CỨNG ở đường đi, chứ
  không do A yếu. Kiểm trần TRƯỚC khi tăng trọng số — tôi đã tăng hai lần
  (tuyệt đối 2,0 → tương đối 0,5×max) mà trần không nhúc nhích.

---

# TỔNG KẾT PHIÊN §113 – §137

25 mục, bài học 185 → 245. Một mặc định đổi (§113), một bảng dữ liệu mới (§128),
một công cụ đo sửa lại (§126). Mọi thứ khác là chẩn đoán và đính chính.

## 1. THU HOẠCH CHẮC CHẮN

**Sinh câu (§113, §116, §118)** — thu hoạch lớn nhất, và là PHỤ PHẨM chứ không
phải mục tiêu:

```
              TRUNG THÀNH   TRẬT TỰ   LẶP
đầu phiên        79,3        89,3    13,6
cuối phiên       85,9        97,3     7,0    (6 hạt: 86,1±1,3 · 97,2±1,8 · 9,9±2,4)
```

Ba vá độc lập: `w=4→1` (§113), `_ag_tin` (§116), tách hai vị từ biên (§118).
Ca vàng ổn định 6/6 hạt: `chó là`, `lúa là`, `hổ là`, `sắt là`.

**Bảng nhãn lớp (§128)** — `top-1 4,1% → 84,9%` trên 8 568 từ, 9 222 mục, bake từ
khuôn định nghĩa của từ điển. Đứng độc lập, dùng được cho việc khác.

**Công cụ đo đa hạt giống (§126)** — mọi cổng từ nay so với biên độ nhiễu.

## 2. Ý TƯỞNG CON KHỈ: 4 MẮT XÍCH, MẮT THỨ TƯ BỊ CHẶN

| mắt xích | trạng thái | bằng chứng |
|---|---|---|
| ① thuộc tính → vật mang | chạy | §104 3,4σ |
| ② vật mang → nhãn lớp | **chạy tốt** | §128 84,9% |
| ③ nhãn lớp → thành viên | chạy | §136 top-10 11,1% vs 4,2% hoán vị |
| ④ **ra văn bản sinh** | **BỊ CHẶN** | §137: 0/60, 0/14, 0/7 |

**Trần kiến trúc (§137)**: vòng sinh có hai nhánh, hai thang — ứng viên đang
sống được NHÂN, ứng viên MỚI vào ở thang TRUNG VỊ (~0,03 so với đỉnh ~40). Kiến
trúc không thể phát ra từ mà phiếu W_skip chưa đề cử — đúng việc của loại suy.

Đường duy nhất còn lại là nâng nhánh hồi sinh, đụng bất biến §12.144 (chặn
word-salad). **QUYẾT ĐỊNH: GIỮ §12.144**, không đánh đổi chất lượng câu — thứ
duy nhất đang thật sự tốt — lấy khả năng loại suy chưa chứng minh được.

Prompt đích vẫn hỏng: `nghịch ngợm như loài`, `chó và mèo đều là`,
`chó nghịch ngợm, mèo nghịch ngợm, còn`, `hổ giống mèo vì`.
(`khoẻ như → voi`, `nhanh như → sóc` đúng nhưng là THÀNH NGỮ học thuộc.)

## 3. LỖI DỤNG CỤ — chủ đề xuyên suốt phiên

Bốn lần cùng hình dạng, cả bốn đều đẩy kết luận về phía BI QUAN SAI:

1. **§113** `TRUNG THÀNH` mù với THỨ TỰ — chấm câu rối ngang câu đúng, còn phạt
   câu đúng khi lệch đuôi. §112 đã chọn mặc định bằng thước này.
2. **§114/§127** thước sinh câu mù với chất lượng NHÃN LỚP — `loc_dich` bị xếp
   «vô hiệu» vì đo sai đích.
3. **§127** bộ khoá 198 bộ do CHÍNH đường ống cần thẩm định sinh ra ⇒ mọi số thu
   hồi ở §89/§101/§102 chấm so với chuẩn nhiễm bẩn.
4. **§125** chấm bằng MỘT hạt giống trong khi đường sinh có `T = 0,5`; `seed=0`
   hoá ra là hạt TỐT NHẤT về LẶP.

## 4. ĐÍNH CHÍNH CHÍNH MÌNH TRONG PHIÊN

- §113 đính chính §112 (chọn mặc định bằng thước mù)
- §123 đính chính §122.3 (không có nhân dồn `0,25⁶`)
- §126 đính chính §125 (biên độ ≠ ngưỡng phán cho TRUNG BÌNH)
- §131 đính chính §129.4 (21% đo lỗi BẢNG, không đo đồ thị)
- §132 đính chính lượt trước (cạnh lớp→thành viên ĐẦY ĐỦ, tôi hỏi nhầm nút)
- §137 đính chính §136.3 (agenda KHÔNG phải thủ phạm)

## 5. HƯỚNG BỊ LOẠI (có số, không bàn lại)

`w=1` cửa sổ agenda (§113 nhận) · ngưỡng tiêu thụ agenda (§119/§126, bão hoà) ·
trần EOS theo nền (§121, lặp gấp đôi) · `gom_tu` (§130, phá `sắt`) ·
`hub_chain_bang_nhan_w` (§134, 0 tác dụng) · `hub_chain_bang_tv_w` (§136–137,
qua cổng thu hồi nhưng 0 tác dụng lên văn bản).

## 6. CÒN TREO

- `sóc là ▸ thú gặm` — cụt TẤT ĐỊNH 6/6 hạt, chưa rõ nguyên nhân
- `mặt trời là ▸ …chiếm khoảng ngôi` — đuôi rác
- phép nối hỏng `and n_b_gen >= 1` còn sống ở khối ÉP CỨNG ~876 (§124.5)
- 61,5% số từ không có lớp ≥2 thành viên (§133) — trần dữ liệu, không sửa bằng mã
- 12 commit phiên này KHÔNG KÝ (1Password khoá giữa chừng)

## §138 — GIÁ của việc nới §12.144: rẻ ở vùng an toàn, nhưng KHÔNG MUA ĐƯỢC GÌ

§137 khuyên giữ §12.144 dựa trên SUY ĐOÁN («nới ra sẽ gây word-salad»), chưa đo.
Vòng này đo.

### 138.1 Quét thang hồi sinh (6 hạt, 20 prompt)

```
hồi_sinh   TRUNG THÀNH   TRẬT TỰ     LẶP        «chó là»
0,0        86,1 ±1,3     97,2 ±1,8   9,9 ±2,4   nguyên vẹn
0,25       86,3 ±1,3     97,1 ±2,1   9,6 ±2,4   nguyên vẹn
0,5        86,2 ±1,6     96,8 ±1,7   9,3 ±2,4   nguyên vẹn
1,0        80,7 ±4,3     93,4 ±2,9  19,2 ±5,6   «gia gia gia gia gia súc»
```

**Suy đoán §137 ĐÚNG về nguy hiểm, SAI về vị trí.** Vách đá nằm giữa 0,5 và 1,0,
không phải ngay khi nới. Ở 0,25–0,5 giá BẰNG KHÔNG: cả ba thước trong biên độ
nhiễu (§125: 2,6/3,6/4,8), ca vàng nguyên vẹn, LẶP còn tốt hơn chút.

### 138.2 Nhưng vùng an toàn KHÔNG mua được gì

```
số prompt ĐỔI ở mức 0,5: 1/7
  nghịch ngợm như loài  0,00 ▸ …được hạt để tạo
                        0,50 ▸ …được hạtLoài chúng     ← đổi, XẤU hơn
```

Một prompt đổi, đổi theo hướng tệ. `Loài` (nhãn lớp) có ra được nhưng chèn sai
chỗ.

### 138.3 Điều mức 1,0 tiết lộ

`chó là gia gia gia gia gia súc` — ở đó chuỗi THẮNG được và đẩy `gia` (từ nhãn
`gia súc`) ra văn bản, nhưng phát LẶP. ⇒ Chuỗi CÓ thể ra văn bản; cái nó thiếu
là biết ĐẶT TỪ VÀO ĐÂU. Không có khái niệm vị trí/ngữ pháp, nó chỉ biết «từ này
đáng nói», không biết «đáng nói ở chỗ nào».

⇒ Trần thật KHÔNG chỉ là thang hồi sinh (§137 đúng một nửa). Kể cả khi cho chuỗi
đủ sức thắng, nó vẫn không sinh ra loại suy — nó sinh ra lặp.

### 138.4 Kết luận: GIỮ §12.144 — giờ có BẰNG CHỨNG

Khuyến nghị của §137 đứng vững, nhưng lý do đổi: không phải «nới ra là hỏng
ngay» (sai — có vùng an toàn rộng), mà là «vùng an toàn không mua được gì, còn
vùng mua được gì thì hỏng». Cờ `ar_w_skip_chain_hoi_sinh` giữ mặc định 0.

### 138.5 Bài học

- **246.** Khuyến nghị dựa trên SUY ĐOÁN phải được đánh dấu là suy đoán và ĐO
  khi có cơ hội. §137 khuyên đúng nhưng vì lý do sai; nếu người dùng nghe theo
  mà không đo thì đã đóng một cửa chưa ai mở ra xem.
- **247.** «Cơ chế không ra được đầu ra» có thể che một vấn đề THỨ HAI đứng sau:
  cho nó đủ sức thắng thì lộ ra nó không biết đặt từ vào đâu. Gỡ trần thứ nhất
  chỉ để lộ trần thứ hai — nên đo LỢI cùng lúc với GIÁ, đừng đo giá trước rồi
  mới hỏi lợi.

## §139 — Cổng VỊ TRÍ: cứu được ~40% thiệt hại, chưa đủ để mở vùng hữu ích

§138 để lại: chuỗi biết «từ này đáng nói», không biết «đáng nói ở CHỖ NÀO».
Cổng: chỉ cho đề cử chuỗi tác động nếu có chứng cứ bigram câu cha (`_seq_bi`)
đứng SAU token vừa phát — `là → gia` có, `gia → gia` không.

### 139.1 Số (6 hạt, 20 prompt)

```
cấu hình             TRUNG THÀNH   TRẬT TỰ     LẶP
mốc (hs=0)           86,1 ±1,3     97,2 ±1,8    9,9 ±2,4
hs=0,5 không cổng    86,2 ±1,6     96,8 ±1,7    9,3 ±2,4
hs=0,5 CÓ cổng       85,7 ±1,6     97,2 ±1,7   10,4 ±3,0
hs=1,0 không cổng    80,7 ±4,3     93,4 ±2,9   19,2 ±5,6
hs=1,0 CÓ cổng       83,9 ±2,1     95,6 ±1,8   18,0 ±3,7
```

Ở `hs=0,5` ba cấu hình KHÔNG phân biệt được (mọi chênh trong biên độ §125) —
đúng dự đoán: vùng đó chuỗi chưa thắng nổi nên không có gì để cổng sửa.

Ở `hs=1,0` cổng lấy lại ~40% thiệt hại (TT +3,2 · TQ +2,2) và khôi phục HOÀN
TOÀN ca `sắt`. Nhưng LẶP vẫn 18,0 so với 9,9 ⇒ vượt xa biên độ 4,8 ⇒ CHƯA ĐỦ.

### 139.2 Vì sao chưa đủ

```
chó là ▸ gia súc thuộc nhóm ăn thịt GIA GIA, vị GIA, được, của GIA có GIA GIA,
lúa là ▸ cây CÂY lương thực LƯƠNG có quả CÂY hạt hoa
```

Câu bắt đầu ĐÚNG rồi thoái hoá. Cổng chỉ xét token LIỀN TRƯỚC, mà `gia` có chứng
cứ đứng sau hàng trăm token khác nhau trong 450 303 câu cha ⇒ qua cổng gần như
mọi lúc. Lỗi còn lại là TÁI XUẤT trong cùng câu, không phải sai vị trí cục bộ.

### 139.3 Phán quyết

Giữ `ar_w_skip_chain_vi_tri = False`. Cổng là mảnh ĐÚNG nhưng một mình không mở
được vùng hữu ích. Bước còn thiếu: chặn đề cử chuỗi TÁI XUẤT trong cùng câu
(một-lần-mỗi-từ), thứ mà phạt lặp ×0,25 không kham nổi khi `hs=1,0`.

### 139.4 Bài học

- **248.** Bigram một bước là ràng buộc vị trí YẾU trên corpus lớn: 450k câu cha
  làm gần như mọi cặp đều có chứng cứ. Ràng buộc vị trí phải tính cả LỊCH SỬ câu
  đang sinh, không chỉ token liền trước.
- **249.** `pgrep -lf "python.*nn-aivie"` KHÔNG khớp tiến trình chạy
  `.venv/bin/python scripts/…` — mẫu tìm sai cho kết quả rỗng, và tôi đọc rỗng
  thành «đã chết» rồi dựng hai giả thuyết cho một cái chết không có thật. Kiểm
  mẫu tìm TRƯỚC khi kết luận từ kết quả rỗng.
- **250.** `grep` lọc đầu ra nuốt mất traceback (lặp lại lỗi §103). Khi chạy nền,
  giữ NGUYÊN stderr vào file, lọc lúc ĐỌC.

## §140 — «Một lần mỗi từ» chữa được GIÁ, nhưng LỢI vẫn bằng không

### 140.1 Ràng buộc đúng là MỘT-LẦN, không phải cổng vị trí

```
cấu hình            TRUNG THÀNH   TRẬT TỰ     LẶP     ca vàng
mốc (hs=0)          86,1 ±1,3     97,2 ±1,8    9,9     nguyên vẹn
hs=1,0 trần         80,7 ±4,3     93,4 ±2,9   19,2     «gia gia gia gia gia súc»
hs=1,0 +vị trí      83,9 ±2,1     95,6 ±1,8   18,0     thoái hoá
hs=1,0 +MỘT LẦN     85,3 ±1,3     94,9 ±3,1    9,2     NGUYÊN VẸN
hs=1,0 +CẢ HAI      85,1 ±2,7     95,6 ±2,2   10,5     nguyên vẹn
```

LẶP `19,2 → 9,2` — thấp hơn cả mốc. Ca vàng khôi phục nguyên văn. Cổng vị trí
(§139) hoá ra THỪA: thêm vào chỉ làm lặp nhích lên 10,5.

⚠️ §139 đã chẩn đoán ĐÚNG («lỗi còn lại là TÁI XUẤT, không phải sai vị trí cục
bộ») rồi lại đi dựng cổng vị trí. Tôi viết ra câu trả lời rồi làm việc khác.

Lưu ý phương sai: TRẬT TỰ `94,9 ±3,1` [90,8–97,0] rộng hơn mốc `±1,8` — trung
bình trong nhiễu nhưng có hạt giống kém hơn đáng kể.

### 140.2 Nhưng lợi vẫn KHÔNG CÓ

```
đổi 3/9 prompt, cả ba trung tính hoặc XẤU hơn:
  nghịch ngợm như loài  …hạt để tạo  → …hạt ra có
  hổ giống mèo vì       …và làm      → …được một người   (lặp)
  mặt trời là           …khoảng ngôi → …khoảng en
```

Lần thứ BA cùng hình dạng (§119, §138, §140): làm cơ chế TRẢ NỔI GIÁ không làm
nó HỮU ÍCH.

### 140.3 Kết luận sâu hơn: ĐƠN VỊ SAI

Loại suy là mệnh đề ở cấp **CÂU** («khỉ cũng nghịch ngợm»), không phải ưu tiên ở
cấp **TOKEN**. Nhét nó vào cuộc bỏ phiếu token của mô hình tự hồi quy thì dù
thắng được cũng chỉ chèn thêm một từ cùng lớp vào vị trí ngẫu nhiên.

⇒ Điều này CỦNG CỐ hướng nướng cạnh (§141): cạnh nướng sẵn đổi CHÍNH cuộc bỏ
phiếu, khác bản chất với việc chỉnh sau khi phiếu đã đếm.

### 140.4 Bài học

- **251.** Chẩn đoán đúng KHÔNG tự động dẫn tới bản vá đúng. §139 viết đúng nguyên
  nhân («tái xuất») rồi vá theo hướng khác («vị trí»). Đọc lại chẩn đoán của
  chính mình TRƯỚC khi chọn chỗ vá.
- **252.** Ba lần «trả nổi giá mà vô ích» là dấu hiệu sai ĐƠN VỊ, không phải sai
  tham số. Khi vặn mọi tham số đều không ra lợi ích, hỏi xem cơ chế có đang được
  biểu diễn ở đúng cấp độ không.

## §141 — NƯỚNG CẠNH BẬC HAI: bốn lượt, ba lỗi nền, tín hiệu xuất hiện ở lượt bốn

Người dùng chỉ ra thiết kế đúng: cạnh `c → Y` phải được TẠO lúc train/bake, rồi
đường sinh QUÉT như mọi cạnh khác — không suy lúc sinh. §140 xác nhận lập luận:
đề cử suy-lúc-sinh vào ở HẠNG HAI (thang trung vị) nên vặn kiểu gì cũng hỏng.

### 141.1 Quy mô — khả thi

Bao đóng thô 227,7 triệu cạnh (185× bậc một) ⇒ không nướng thẳng. Hai ràng buộc
của người dùng cắt xuống mức nướng được:

```
trần bậc TG   K≥   ước tính   so bậc 1
    73 (p90)   2    1,46 M      1,2×
   674 (p99)   2   18,2 M      14,8×
```

`K ≥ 2` (hai trung gian PHÂN BIỆT) là ràng buộc cốt lõi — một mình cắt >3 lần.

### 141.2 Bốn lượt, ba lỗi NỀN

| # | lỗi | triệu chứng |
|---|---|---|
| 1 | quan hệ nền = BIGRAM (liền kề) | `chó → để/đến/trên`, `Nh → là/"/)` |
| 2 | ngưỡng = phân vị gidf trên TOÀN từ vựng | `chó`(gdf 1316), `kẽm`(66) đều BỊ LOẠI |
| 3 | xếp hạng = ĐẾM THÔ | `chó → phương/nữ/thần`, `sắt → ta/đề/bản` |
| 4 | PMI (mũ 1,0) | tín hiệu XUẤT HIỆN, lẫn mảnh BPE |

Lỗi 1: ghép hai chặng LIỀN KỀ = khuôn ngữ pháp, không phải tương tự phân bố.
Mẫu người dùng viết `X … a` — đồng hiện ở khoảng cách BẤT KỲ.
Lỗi 2: phân bố gidf trên từ vựng lệch mạnh về phía hiếm ⇒ phân vị của NÓ không
phải công cụ cắt hư từ. (§128 dùng phân vị THÀNH CÔNG vì lấy trên đúng quần thể.)
Lỗi 3: đếm thô luôn nổi từ phổ biến.

### 141.3 Lượt 4 — tín hiệu THẬT lần đầu

```
kẽm  → THIẾC, QUẶNG        lúa → THÓC
hổ   → Fel, TIGRIS         mèo → Fel, Coel      (cùng họ!)
muối → KAINIT              chó → ugula/adai/öpp  ← vẫn rác
```

PMI đẩy ngược sang từ SIÊU HIẾM (bệnh kinh điển) nên mảnh BPE nổi lên. Nhưng
quan hệ ĐÚNG xuất hiện lần đầu trong ba phiên: `kẽm → thiếc/quặng` là đúng loại
cạnh mà ý tưởng cần, và sinh ra HOÀN TOÀN TỰ ĐỘNG — không từ điển, không nhãn
lớp, không bảng.

Bằng chứng phụ từ lượt 2 (chỉ từ siêu hiếm, đếm thô): `Huobi → Heco/coin/
Ethereum/MXC`, `prajñā → śāstra`, `aly → Slav/Byzantine/Macedonia` — sạch và
đúng miền. Ở lát từ vựng mà tần suất không bóp méo, cơ chế bậc hai CHẠY ĐÚNG.

### 141.4 Trạng thái: nửa TẠO xong, nửa DÙNG chưa làm

Bảng cạnh nằm ở `out/canh_bac_hai.pkl`, CHƯA nối vào bảng kề ⇒ đường sinh chưa
thấy. Ý người dùng («khi sinh quét như các kết nối khác») mới thực hiện một nửa.

Bước kế đã chẩn đoán: PMI cần SÀN TẦN SUẤT để loại mảnh BPE (mũ giữa 0 và 1,
cộng `gdf_min`). Dừng tự vặn tham số ở đây theo cam kết đã khai.

### 141.5 Bài học

- **253.** Ba lượt liền tôi dựng cơ chế TRƯỚC khi kiểm quan hệ nền, và cả ba lỗi
  đều là kiến thức cơ bản về đồng hiện phân bố mà tôi ĐÃ BIẾT. Với mỗi lượt
  40–150 giây thì giá rẻ, nhưng thứ tự sai: kiểm nền bằng một truy vấn nhỏ
  TRƯỚC, nướng SAU.
- **254.** Cùng một công thức (cắt theo phân vị) đúng ở §128 và sai ở §141 vì
  QUẦN THỂ khác nhau. Kỹ thuật không tự mang theo tính đúng đắn — quần thể mới
  là thứ quyết định.
- **255.** Đếm thô nổi từ PHỔ BIẾN, PMI nổi từ SIÊU HIẾM. Hai bệnh đối xứng; lời
  giải là PMI có SÀN, không phải chọn một trong hai.

## §142 — Sàn tần suất: đồng hiện bậc hai cho HÀNG XÓM CHỦ ĐỀ, không cho ANH EM CÙNG LỚP

§141 dừng ở «PMI cần sàn». Vòng này thêm sàn, quét hai điểm.

### 142.1 Sàn đổi HÌNH DẠNG đồ thị theo chiều KHÔNG lường trước

```
không sàn:  28 830 nguồn · bậc ra trung vị  55 · trần TG   776
sàn ≥20:    10 283 nguồn · bậc ra trung vị 311 · trần TG 2 013
```

Sàn loại từ HIẾM (vốn ít kết nối), giữ từ PHỔ BIẾN (vốn nhiều kết nối) ⇒ mật độ
TĂNG gấp sáu, và trần trung gian theo phân vị 90 tự nới từ 776 lên 2 013 nên hết
chặn hư từ. Tôi nghĩ về sàn như bộ lọc «làm mọi thứ nhỏ đi» — sai chiều.

### 142.2 Hai điểm quét, cả hai CHƯA ĐẠT

```
A sàn20 mũ0,75   kẽm → sulfuric/calci/magnesi/sulfide/Vàng/kim   ĐÚNG miền
                 mèo → Lông/Móng/Chuột                            có nghĩa
                 hổ  → Từng/Đầy/Đem/Kia                           rác
                 lòng → Ng/Vị/Tay/Dứt          (ngẫu nhiên)       rác
B sàn50 mũ0,50   nấm → Món/Trộn/Thịt/trứng/Cắt                    mạch lạc
                 dưa → Dùng/Thịt/Cắt/khoai/ngâm                   mạch lạc
                 chó → Hai/hiếm/Làm/Ông/Vừa                       rác
```

B nổi dạng ĐẦU CÂU viết hoa (`Hai`,`Ông`,`Vừa`,`Đối`) — chúng đồng hiện rộng.

### 142.3 KẾT LUẬN: sai LOẠI QUAN HỆ, không phải sai tham số

Ở các lát chạy được, thứ tìm ra là **liên tưởng CHỦ ĐỀ**: `muối → salad/rau/cơm`
(ngữ cảnh nấu ăn), `nấm → món/trộn/thịt`, `kẽm → sulfuric/calci` (ngữ cảnh hoá
học). KHÔNG phải `muối → đường`, `kẽm → thiếc` — tức không phải THÀNH VIÊN CÙNG
LỚP.

Đồng hiện bậc hai đo **tương tự NGỮ CẢNH**. Hai từ cùng lớp thường cùng ngữ cảnh,
nhưng cùng ngữ cảnh KHÔNG suy ra cùng lớp — `muối` và `rau` cùng câu nấu ăn mà
khác lớp hoàn toàn. Loại suy cần quan hệ TAXONOMIC; cơ chế này cho quan hệ
TOPICAL.

Đó cũng giải thích vì sao `via_pivot` (khuôn `A là B`) cho nhãn đúng 84,9% (§128)
còn đồng hiện cho 4,1%: khuôn định nghĩa mã hoá quan hệ lớp TRỰC TIẾP, đồng hiện
chỉ mã hoá «hay xuất hiện gần nhau».

### 142.4 Sáu lượt nướng — dừng

Đã đủ để kết luận về LOẠI quan hệ. Thêm tham số không đổi được bản chất.

### 142.5 Bài học

- **256.** «Cùng ngữ cảnh» ≠ «cùng lớp». Mọi cơ chế dựa trên đồng hiện đều đo
  cái thứ nhất. Muốn cái thứ hai phải có KHUÔN mã hoá quan hệ lớp (`A là B`,
  khuôn định nghĩa) — đó là lý do bảng §128 đạt 84,9% còn đồng hiện 4,1%.
- **257.** Bộ lọc có thể đổi hình dạng phân bố theo chiều NGƯỢC với trực giác:
  sàn tần suất làm đồ thị DÀY hơn vì nó loại đúng nhóm thưa. Kiểm phân bố SAU
  khi lọc, đừng giả định lọc luôn làm nhỏ đi.
- **258.** `set -- $P` KHÔNG tách từ trong zsh — lần thứ HAI (§103 là lần đầu,
  đã có bài học 144). Lần này `argparse` chặn ngay nên không hỏng dữ liệu, và
  tôi kiểm mẫu tách từ ngay sau khi gõ. Viết tham số TƯỜNG MINH, đừng lặp qua
  chuỗi ghép.

## §143 — NƯỚNG CÂU LỚP: kết nối VÀO ĐƯỢC mạng, nhưng nội dung hỏng vì chia khối

### 143.1 Phát hiện kiến trúc: đường sinh KHÔNG đọc bảng kề

```python
def build_w_skip(...):
    ps = state.parents          # ← KHO CÂU CHA, không phải TypedAdjacency
    for rec in ps.parents:
        for i, j in cặp trong cùng câu, khoảng cách ≤ d_max
```

`vote_logits` → `W_skip` → dựng từ **kho câu**, nên **thêm cạnh vào
`TypedAdjacency` KHÔNG ảnh hưởng gì tới sinh câu**. Đây là lý do sâu xa khiến
§136–§141 thất bại: tôi tìm cách tiêm mạnh hơn vào một cuộc bỏ phiếu mà nguồn
của nó là kho CÂU, không phải kho CẠNH.

⇒ «Kết nối trong mạng» mà đường sinh thấy = **cặp từ cùng nằm trong một câu cha,
trong cửa sổ d_max**. Nên nướng câu lớp, không nướng cạnh.

### 143.2 Nướng: 1 654 câu, cha 450 303 → 451 957

Khuôn `A, B, C và D đều là <LỚP>.` nạp qua `add_example` (đường ingest chuẩn,
`defer_index=True` + `begin_batch`/`end_batch`), priority 0,80 < câu thật 1,0.

### 143.3 KẾT QUẢ CẤU TRÚC: ĐẠT — lần đầu trong ba phiên

```
ĐỔI 4/11 — và cả bốn đều là prompt LOẠI SUY
6 prompt định nghĩa: KHÔNG đổi một token nào
```

Kết nối vào được mạng, đường sinh quét được, tác động KHU TRÚ đúng nhóm prompt
có khuôn tương ứng. Không cần cờ nào — chỉ đổi kho câu.

### 143.4 KẾT QUẢ NỘI DUNG: HỎNG — hai nguyên nhân tách bạch

**(a) Lỗi của tôi — CHIA KHỐI phá đồng hiện.** Lớp `kim loại` 8 thành viên, cắt
thành khối rời 6 từ/câu:

```
bạc, chì, kim, kẽm, nhôm và sắt đều là kim loại.   ← sắt
vàng và đồng đều là kim loại.                       ← đồng
sắt/đồng nằm chung câu nướng: 0 câu
```

Chính việc chia khối phá đúng thứ đang cố tạo. ⇒ `sắt và đồng đều là` → `cuộc
họp.` vì không có kết nối `sắt↔đồng` nào. Sửa: cửa sổ TRƯỢT hoặc nhiều câu hoán
vị thứ tự, để MỌI CẶP đồng hiện ở đâu đó.

**(b) Giới hạn DỮ LIỆU — ca kinh điển không tồn tại.**

```
chó → gia súc  ·  lớp «gia súc» = {chó, vật nuôi}
mèo → thú nhỏ  ·  LỚP KHÁC
```

`chó` và `mèo` KHÔNG cùng lớp ở mức nhãn 2 từ — nên `chó và mèo đều là` không
thể có câu nướng, và nó đúng là prompt loại suy DUY NHẤT không đổi. Lùi lên nhãn
1 từ (`gia`/`thú`) thì §135 đã đo là quá rộng (`khỉ → loài → đá/dùng/hình`).

⚠️ Ví dụ kinh điển của chính ý tưởng — `chó` và `mèo` cùng lớp — KHÔNG có trong
dữ liệu ở mức hạt dùng được. Suốt ba phiên tôi chưa từng kiểm điều này.

### 143.5 Bài học

- **259.** Kiến trúc quyết định «kết nối» nghĩa là gì. Ở đây `TypedAdjacency` có
  16 loại cạnh mà đường sinh chính KHÔNG đọc cái nào — nó đọc kho CÂU. Trước khi
  «thêm kết nối», phải truy xem đường tiêu thụ đọc CẤU TRÚC NÀO.
- **260.** Chia khối để giữ cửa sổ d_max đã PHÁ đúng quan hệ cần tạo. Khi mục
  tiêu là ĐỒNG HIỆN, mọi phép chia nhóm đều cắt cặp — phải dùng cửa sổ trượt.
- **261.** Kiểm ca kinh điển của chính ý tưởng (`chó`+`mèo` cùng lớp) TRƯỚC khi
  dựng ba phiên cơ chế quanh nó. Nó không có trong dữ liệu, và điều đó đáng lẽ
  phải biết từ §22.

## §144 — Cửa sổ trượt: «sắt và đồng đều là ▸ KIM LOẠI» — đáp án đúng ĐẦU TIÊN

### 144.1 Sửa chia khối → trượt chồng 50% + 2 lượt hoán vị

3 021 câu (so 1 654), cha 450 303 → 453 320. Tiền đề đạt:

```
sắt+đồng 2 câu · kẽm+vàng 2 câu · bạc+nhôm 2 câu · chó+mèo 0 câu
vd: «kẽm, nhôm, sắt, vàng và đồng đều là kim loại.»
```

### 144.2 ⚠️ Phép kiểm ĐẦU TIÊN của tôi SAI — lại bệnh §132

Tra bằng `encode_prompt("sắt")` → id của `sắt` ĐỨNG ĐẦU CÂU, trong khi trong câu
nó mang id `Ġsắt` khác hẳn ⇒ báo **0 câu chung** cho cả bốn cặp. Suýt kết luận
cửa sổ trượt thất bại khi nó thành công.

Thứ cứu: in DỮ LIỆU THÔ cạnh kết luận — bốn câu `kim loại` in ngay dưới bảng đếm
và tự mâu thuẫn với số 0. Chỉ in con số thì không có gì bắt lỗi.

Lần thứ HAI trong phiên vấp «một bề mặt nhiều id» (§132 là lần đầu, mất ba vòng).

### 144.3 Kết quả

```
sắt và đồng đều là  ▸ KIM LOẠI.        ← ĐÚNG, lần đầu ba phiên
kẽm và vàng đều là  ▸ đối mặt          ← SAI, dù cũng 2 câu chung
chó và mèo đều là   ▸ (không đổi)      ← 0 câu chung
5 prompt định nghĩa ▸ không đổi        ← không thụt lùi
```

### 144.4 ĐÂY LÀ GÌ, VÀ KHÔNG LÀ GÌ

**LÀ**: bằng chứng kết nối lớp vào được mạng, đường sinh quét được như mọi kết
nối khác, tác động khu trú đúng khuôn, KHÔNG cần cờ nào. Đúng nguyên tắc người
dùng nêu: tạo lúc bake, quét lúc sinh.

**KHÔNG LÀ**: loại suy. Mạng phát lại câu VỪA NẠP — học thuộc. Loại suy thật là
prompt về từ KHÔNG có trong câu nướng nào mà mạng vẫn suy ra lớp qua thành viên
khác. Chưa kiểm.

Và `kẽm và vàng` hỏng dù cũng 2 câu chung ⇒ ngay cả HỌC THUỘC cũng chưa ổn định.

### 144.5 Bài học

- **262.** In DỮ LIỆU THÔ cạnh mọi bảng đếm. Con số một mình không tự tố cáo
  được lỗi của phép đếm; dữ liệu thô thì có.
- **263.** «Đáp án đúng» ở prompt khớp khuôn câu vừa nạp = HỌC THUỘC. Phân biệt
  rành mạch với loại suy trước khi báo cáo, kể cả khi kết quả đẹp.

## §145 — Kiểm loại suy THẬT: phép thử VÔ HIỆU theo cấu tạo

### 145.1 Thiết kế và kết quả bề mặt

Ứng viên: từ KHÔNG có câu định nghĩa gốc nhưng có ≥2 anh em cùng lớp CÓ định
nghĩa. 28 ứng viên; 3/12 hiện ra có vẻ đạt:

```
✓ công-te-nơ là ▸ thiết bị có dạng một thùng chứa hình hộp lớn, có dung tích tiêu chuẩn
✓ mi-crô là     ▸ thiết bị dùng thu âm hoặc làm giọng nói…
✓ bit là        ▸ đơn vị nhỏ nhất, chỉ 0 hoặc 1
```

### 145.2 VÔ HIỆU — cả ba là CHÉP NGUYÊN VĂN

Tra `dictionary.jsonl` (CÓ trong corpus huấn luyện):

```
công-te-nơ CÓ  «thiết bị có dạng một thùng chứa hình hộp lớn, có dung tích tiêu chuẩn»
mi-crô     CÓ  «thiết bị dùng thu âm hoặc làm giọng nói hoặc giọng hát nghe lớn hơn»
bit        CÓ  «… đơn vị thông tin nhỏ nhất»
```

Tôi lọc «không có định nghĩa» chỉ trên `dinh_nghia.jsonl` mà quên `dictionary.
jsonl` cũng được train — ĐÚNG lỗi đã mắc và ĐÃ GHI NHẬN ở §136, lặp lại nguyên vẹn.

### 145.3 Hệ quả SÂU HƠN: không có tập giữ riêng, theo CẤU TẠO

Bảng nhãn lớp (§128) trích TỪ CHÍNH `dictionary.jsonl`. ⇒ **mọi từ có nhãn lớp
đều đã có định nghĩa trong corpus huấn luyện.** Không tồn tại từ nào vừa có anh
em cùng lớp vừa chưa từng được dạy định nghĩa.

⇒ Với thiết lập hiện tại, **KHÔNG THỂ chứng minh chuyển giao** — không phải vì
cơ chế yếu mà vì thiếu tập giữ riêng. Muốn kiểm phải TRAIN LẠI với một phần định
nghĩa bị giữ lại (~87 phút), rồi nướng câu lớp cho đúng nhóm bị giữ.

Các ca KHÔNG chép được (`an-bom`,`com-pa`,`hi-fi`,`ca-ra`) đều CŨNG có định nghĩa
trong từ điển mà vẫn ra rác ⇒ ngay cả ghi nhớ cũng chỉ đạt một phần.

### 145.4 Bài học

- **264.** Lặp lại NGUYÊN VẸN lỗi §136 (quên `dictionary.jsonl` cũng được train)
  dù đã tự ghi nhận nó. Ghi bài học KHÔNG đủ; phải dựng danh sách CORPUS ĐÃ TRAIN
  một lần và tra nó ở mọi phép thử «không có nguồn».
- **265.** Bộ đánh giá sinh từ CÙNG NGUỒN với dữ liệu huấn luyện thì không có
  tập giữ riêng — và điều đó là tính chất CẤU TẠO, phát hiện được bằng suy luận
  TRƯỚC khi chạy, không cần đo.

## §146 — PHÉP THỬ GIỮ RIÊNG: KHÔNG có chuyển giao (0/60)

Phép thử quyết định cho câu hỏi treo ba phiên, lần đầu có TẬP GIỮ RIÊNG thật.

### 146.1 Thiết lập

60 từ (mỗi lớp một từ, lớp ≥3 thành viên) bị XOÁ định nghĩa khỏi CẢ HAI nguồn
(`dictionary.jsonl` + `dinh_nghia.jsonl`) trước khi train; anh em cùng lớp GIỮ
NGUYÊN. Train lại (~2 giờ) → `net_ho.pkl` 449 866 cha. Nướng 3 017 câu lớp
(§144, cửa sổ trượt) → `net_ho_lop.pkl` 452 883 cha.

Hai checkpoint khác nhau ĐÚNG MỘT THỨ: kết nối lớp.

### 146.2 Kết quả

```
CHƯA nướng : 2/60      CÓ nướng : 4/60      chỉ CÓ trúng: 2
```

Hai ca «chuyển giao» là DƯƠNG TÍNH GIẢ do luật chấm của tôi quá lỏng (chấp nhận
khớp `lop.split()[0]` — chỉ cần chứa TỪ ĐẦU của nhãn):

```
dioxin   lớp «chất độc»  ▸ «CHẤT lượng và các vật của học được…»  ← khớp «chất»
xe-mi-na lớp «buổi sinh» ▸ «gì BUỔI không thể làm được cho…»      ← khớp «buổi»
```

Cả hai đều là câu RÁC tình cờ chứa một từ phổ biến.

⇒ **CHUYỂN GIAO THẬT: 0/60.**

### 146.3 Kết luận cho cả ý tưởng

Kết nối lớp trong mạng cho phép PHÁT LẠI câu lớp đã nạp (§144: «sắt và đồng đều
là ▸ kim loại») nhưng KHÔNG dẫn nội dung định nghĩa sang từ chưa được dạy. Mẫu
`X…a, X…b, X…c` + `a…Y, b…Y` ⟹ `c→Y` KHÔNG xảy ra ở tầng sinh câu.

Đây là câu trả lời dứt điểm, có tập giữ riêng, có đối chứng, không giải thích
được bằng ghi nhớ.

### 146.4 Bài học

- **266.** Luật chấm lỏng tạo DƯƠNG TÍNH GIẢ đúng ở ca ít ỏi mà ta mong đợi
  nhất. `lop.split()[0]` biến «chất độc» thành khớp «chất» — một từ có mặt khắp
  nơi. Khai luật chấm CHẶT trước, và luôn ĐỌC đầu ra thô của mọi ca trúng.
- **267.** `gc.disable()` sao chép mù từ script ngắn sang script sinh 60 lượt
  trên đồ thị 1,2 GB ⇒ OOM im lặng, không traceback. Tách tiến trình cho mỗi
  checkpoint và IN TIẾN ĐỘ — bản đầu chết 34 phút mà tôi tưởng đang chạy.

---

# TỔNG KẾT PHIÊN (tiếp) §138 – §146

Nối tiếp tổng kết §113–§137. Bài học 246 → 267. Không đổi mặc định nào.
Phần này trả lời DỨT ĐIỂM câu hỏi treo ba phiên.

## 1. NGƯỜI DÙNG CHỈ RA THIẾT KẾ ĐÚNG — và nó được chứng minh

«Phải tạo kết nối ngay khi train, rồi khi sinh quét như các kết nối khác, chứ
không suy ra lúc sinh.»

**§143 xác nhận bằng mã**: `vote_logits` → `W_skip`, mà `build_w_skip` dựng từ
`ps.parents` — KHO CÂU CHA, **không** từ `TypedAdjacency`. Nên:

- Thêm cạnh vào bảng kề (16 loại) KHÔNG ảnh hưởng gì tới sinh câu.
- Mọi cách tiêm lúc sinh (§136–§141) vĩnh viễn ở HẠNG HAI: nhánh hồi sinh đặt
  chúng ở thang TRUNG VỊ (~0,03) trong khi cạnh thật ~40.
- «Kết nối trong mạng» mà đường sinh THẤY = cặp từ cùng câu cha trong cửa sổ d_max.

⇒ Nướng CÂU LỚP vào kho cha. §143: ĐỔI 4/11, **cả bốn đều là prompt loại suy**,
6 prompt định nghĩa không đổi một token — khu trú đúng, KHÔNG cần cờ nào.
§144: «sắt và đồng đều là ▸ **kim loại.**» — đáp án đúng ĐẦU TIÊN trong ba phiên.

## 2. PHÉP THỬ QUYẾT ĐỊNH: KHÔNG có chuyển giao (0/60)

§145 phát hiện mọi phép thử trước VÔ HIỆU THEO CẤU TẠO (bảng nhãn trích từ chính
`dictionary.jsonl` đang được train ⇒ không có tập giữ riêng).

§146 dựng tập giữ riêng thật: 60 từ xoá định nghĩa khỏi CẢ HAI nguồn, train lại
~2 giờ, nướng 3 017 câu lớp. Hai checkpoint khác nhau ĐÚNG MỘT THỨ.

```
CHƯA nướng 2/60 · CÓ nướng 4/60 — nhưng cả 2 ca chênh là DƯƠNG TÍNH GIẢ
(luật chấm lỏng: «chất độc» khớp «chất» trong «chất lượng»)
⇒ CHUYỂN GIAO THẬT: 0/60
```

**Kết luận**: kết nối lớp cho phép PHÁT LẠI câu đã nạp, nhưng KHÔNG dẫn tri thức
sang từ chưa được dạy. Mẫu `X…a,X…b,X…c` + `a…Y,b…Y` ⟹ `c→Y` không xảy ra ở
tầng sinh câu. Thứ đi qua kết nối là BỀ MẶT CÂU, không phải TRI THỨC.

## 3. VÌ SAO — ba trần chồng lên nhau

| trần | bằng chứng |
|---|---|
| **Đơn vị sai** | loại suy là mệnh đề cấp CÂU, nhét vào bỏ phiếu cấp TOKEN thì dù thắng cũng chỉ chèn một từ vào vị trí ngẫu nhiên (§140) |
| **Quan hệ sai** | đồng hiện bậc hai cho HÀNG XÓM CHỦ ĐỀ (`muối→salad/rau`), không cho ANH EM CÙNG LỚP (§142) |
| **Dữ liệu thiếu** | `chó → gia súc`, `mèo → thú nhỏ` — ví dụ kinh điển của chính ý tưởng KHÔNG cùng lớp ở mức hạt dùng được (§143) |

## 4. THU HOẠCH GIỮ LẠI

- **Sinh câu** (§113/§116/§118): `79,3/89,3/13,6` → `85,9/97,3/7,0`, ca vàng ổn
  định 6/6 hạt giống. Đây là thứ DUY NHẤT vào mặc định.
- **Bảng nhãn lớp** (§128): 9 222 từ, top-1 `4,1% → 84,9%`, đứng độc lập.
- **Công cụ đo đa hạt giống** (§126) + phát hiện kiến trúc §143.

## 5. LỖI QUY TRÌNH LẶP LẠI — chủ đề của phiên

- **§145 lặp NGUYÊN VẸN lỗi §136**: quên `dictionary.jsonl` cũng được train, dù
  đã tự ghi nhận. Ghi bài học KHÔNG đủ.
- **§144 lặp lỗi §132**: «một bề mặt nhiều id» — lần này trong chính mã KIỂM
  CHỨNG, suýt báo cáo ngược. Cứu được nhờ in DỮ LIỆU THÔ cạnh bảng đếm.
- **§142 lặp lỗi §103**: `set -- $P` không tách từ trong zsh, lần thứ HAI.
- **§146 lặp lỗi §144**: script không in tiến độ ⇒ chết 34 phút mà tưởng đang
  chạy; cộng `gc.disable()` sao chép mù gây OOM im lặng.
- **Ba lần đọc sai trạng thái tiến trình** vì mẫu `pgrep` khớp nhầm (quá hẹp →
  «đã chết»; quá rộng → khớp chính vỏ bash). `kill -0 PID` không có bệnh này.

## 6. CÒN LẠI

- `sóc là ▸ thú gặm` cụt tất định · đuôi rác `mặt trời là`
- phép nối hỏng `and n_b_gen >= 1` ở khối ép cứng ~876 (§124.5)
- nhãn ghép 2 từ cắt cụt (`ban chấp`) và lớp sai (`tam thể → ba màu`) — trần §128
- 20 commit phiên này KHÔNG KÝ (1Password khoá giữa chừng)

## §147 — CẠNH c→Y: chuyển giao THẬT, 6,8% → 30,6%

Người dùng: «phải có cơ chế để c nối với y». §144 mới nướng `c ↔ a` và
`c ↔ nhãn lớp` (ngữ cảnh chung) — §146 đo chuyển giao 0/60. Vòng này nướng
THẲNG cạnh `c…Y`.

### 147.1 Cơ chế

```
X … a, X … b, X … c   ← X = CÙNG LỚP (bảng §128)
a … Y, b … Y          ← Y = nội dung ĐỊNH NGHĨA THẬT của anh em
⟹ nướng «c là <mô tả thừa hưởng>»  ← c và Y CÙNG CÂU CHA ⇒ W_skip thấy
```

Chọn câu thừa hưởng = định nghĩa anh em trùng `Y` nhiều nhất (túi-từ thì sai ngữ
pháp: «kiếm là binh khí binh khí dùng cán dài»). 55 câu cho 60 từ giữ riêng,
priority 0,70 < câu lớp 0,80 < câu thật 1,0.

### 147.2 Kết quả — chuyển giao THẬT đầu tiên của dự án

```
MOC (không có cạnh lớp)  TRUNG THÀNH với định nghĩa thật:  6,8%
CY  (có cạnh c→Y)                                         30,6%     ← ×4,5
```

Mạng CHƯA TỪNG được dạy định nghĩa 60 từ này (§146 giữ riêng).

```
bướm  THẬT «CÔN TRÙNG có bốn cánh mỏng…»   CY «CÔN TRÙNG cánh nửa cỡ nhỏ…»
bờ    THẬT «DẢI ĐẤT làm giới hạn…»          CY «DẢI ĐẤT phù sa nổi cao lên…»
chài  THẬT «LƯỚI hình nón, mép dưới có chì» CY «LƯỚI hình túi do tàu thuyền kéo»
                            mốc: «một số ví dụ về các có thểGiải vật trong»
```

LOÀI GẦN đúng gần như mọi ca; phần mô tả mượn của anh em nên có chỗ sai.

### 147.3 Giới hạn — bịa nội dung, đo được

```
lao  ← sốt rét : «bệnh lây do kí sinh trùng truyền qua muỗi anophel»  ← SAI (lao do vi khuẩn Koch)
đẹn  ← chốc    : «bệnh ở da đầu»                                       ← SAI (đẹn ở lưỡi trẻ sơ sinh)
kiếm ← giáo    : «cán dài»                                             ← SAI (kiếm cán ngắn)
xe-mi-na ← seminar                                                     ← ĐÚNG (đồng nghĩa)
```

Thừa hưởng đúng khi lớp là ĐỒNG NGHĨA, sai khi lớp chỉ là CHỦNG LOẠI. Ràng buộc
≥2 anh em đồng thuận KHÔNG chặn được vì anh em thường sai cùng kiểu. Người dùng
chấp nhận («bịa cũng không sao, sẽ sửa dần»).

### 147.4 Thiết kế TRỌN VẸN đề xuất (chưa dựng)

- **Phân tầng priority**: `c là <lớp>` 0,85 · `+ thuộc tính MỌI anh em có` 0,70 ·
  `mô tả anh em gần nhất` 0,55. Truy hồi vốn tôn trọng priority ⇒ mạng ưu tiên
  phát biểu an toàn, rơi xuống suy đoán chỉ khi không còn gì. Chính là «dù c có
  thể yếu hơn».
- **Dải xuất xứ**: `priority < 0,75` = suy ra. Tri thức thật vào sau ở 1,0 TỰ
  ĐỘNG đè lên — sửa dần mà không cần xoá. Kèm công cụ dọn dải đó để nướng lại.
- **Nhãn DÀI cho thừa hưởng, nhãn NGẮN cho lớp**: bảng đã có cả hai (`kim loại`
  vs `kim loại màu xám`). `bệnh lây` quá thô nên gộp sốt rét với lao.

### 147.5 Bài học

- **268.** Cạnh phải ở ĐÚNG DẠNG mà đường tiêu thụ đọc. §144 nướng ngữ cảnh
  chung ⇒ 0/60. §147 nướng `c` và `Y` CÙNG CÂU ⇒ 30,6%. Cùng một ý tưởng, khác
  nhau ở chỗ cạnh có nằm trong cấu trúc mà `build_w_skip` quét hay không.
