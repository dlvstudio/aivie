# TK_053 — NÂNG CẤP ĐƯỜNG SINH: mượn phát minh LLM, biến đổi cho kiến trúc thưa

| Field | Value |
|-------|-------|
| **ID** | TK_053 |
| **Status** | 🟡 **MỞ** — kế hoạch, chưa dựng. |
| **Nguồn** | Yêu cầu tác giả: *«nghĩ như chuyên gia lâu năm, có thể mượn phát minh bên LLM miễn sao biến đổi được»* |
| **Kế thừa** | TK_046 §113–§147 (đo đạc + bảng nhãn lớp + bộ giữ riêng 60 từ + thước đa hạt giống) |

## 0. Kiến trúc này KHÔNG phải transformer — mượn gì phải hợp dạng

Bộ nhớ tình tiết THƯA (kho câu cha) + bỏ phiếu `W_skip` theo khoảng cách nhảy.
Không gradient; đếm/Hebbian. Retrieval-augmented THEO CẤU TẠO (agenda = câu nguồn
đã truy). §143: `build_w_skip` đọc `ps.parents`, KHÔNG đọc `TypedAdjacency`.

⇒ Mượn được cái nào phải diễn dịch sang «câu cha + phiếu», không phải sang
«trọng số + gradient».

## 1. DỪNG THEO ENTROPY  ← ưu tiên 1

**Bằng chứng**: §117–§118 tốn 4 mục cho EOS; trần cuối là vòng luẩn quẩn «muốn
được dừng thì phải ĐÃ dừng» (`n_b_gen >= 1`). §139/§140 vá bằng luật biên → chỉ
cứu 40%. `sóc là ▸ thú gặm` cụt TẤT ĐỊNH 6/6 hạt; đuôi `mặt trời` rác.

**Mượn**: LLM không dùng luật biên — dừng khi phân bố ứng viên TÁN RA.

**Diễn dịch**: tính entropy của `scores` mỗi bước (đã có sẵn vector). Entropy vọt
= hết nguyên liệu = dừng. Thay `n_b_gen`/`_bd_gen`/cú đẩy biên bằng MỘT đại lượng
đo được.

**Cổng**: `sóc` hết cụt HOẶC đuôi `mặt trời` sạch; 6 hạt giống, ba thước không
xấu quá biên độ (§125: TT 2,6 · TQ 3,6 · LẶP 4,8). Phụ: gỡ được phép nối hỏng
`and n_b_gen >= 1` còn sống ở khối ÉP CỨNG ~876 (§124.5).

## 2. SOẠN–THẨM (speculative decoding)  ← ưu tiên 2

**Bằng chứng**: §147 đạt `6,8% → 30,6%` nhưng BỊA: `lao ← muỗi anophel`,
`kiếm ← cán dài`. Tác giả chấp nhận bịa nhưng muốn «sửa dần».

**Mượn**: draft model đề xuất — verifier chấp nhận/bác từng đoạn.

**Diễn dịch**: chuỗi loại suy = BẢN NHÁP; kho câu cha = BỘ THẨM. Sinh xong một
cụm, tra cụm đó có được CHỨNG THỰC trong câu cha nào không. `bờ → dải đất` giữ
(có chứng thực); `lao → muỗi` cắt (không câu nào nối `lao` với `muỗi`).

**Cổng**: trên 60 từ giữ riêng — TRUNG THÀNH với định nghĩa thật ≥ 30,6% VÀ tỉ lệ
từ BỊA (có trong sinh, không có trong định nghĩa thật, không chứng thực) GIẢM.

## 3. HIỆU CHỈNH THANG ĐIỂM (layer-norm cho phiếu)  ← ưu tiên 3

**Bằng chứng**: §120 đo trực tiếp — agenda cộng TƯƠNG ĐỐI (`2,5 × rv`), EOS sàn
TUYỆT ĐỐI (2,00), chuỗi hồi sinh ở TRUNG VỊ. Ba thang trộn trong một cuộc bỏ
phiếu ⇒ vặn tham số nào cũng chỉ dịch chỗ vỡ (§121 lặp 19,2%, §138–§140).

**Diễn dịch**: chuẩn hoá MỌI nguồn đóng góp về cùng thang trước khi cộng.

**Cổng**: ba thước không xấu đi, VÀ ít nhất một trong các hướng đã bị loại
(§119/§121/§136) trở nên trung tính thay vì có hại — dấu hiệu thang đã đúng.

## 4. AGENDA MỀM (attention thay argmax)  ← ưu tiên 4

**Bằng chứng**: §120 đo `rv` sụp 1 081 → 3,5 khi agenda cạn ⇒ `đồng là ▸ dẫn
điện và` cụt. Agenda hiện là MỘT câu cha (chọn cứng).

**Diễn dịch**: agenda = HỖN HỢP top-k câu cha theo điểm khớp. Câu thứ nhất cạn
thì câu thứ hai đỡ.

**Rủi ro đã biết**: §83 trộn nguồn làm mất tri thức (`chó là gia đình và bạn bè`).
⇒ cổng phải có vế PHỦ QUYẾT bằng TRUNG THÀNH.

## 5. SINH HAI THÌ (chain-of-thought rút gọn)  ← ưu tiên 5

**Bằng chứng**: bảng nhãn lớp `84,9%` (§128) nhưng đường sinh KHÔNG dùng; §136
nối vào ② thì 0/60 lượt sinh đổi.

**Diễn dịch**: thì 1 sinh NHÃN LỚP (độ tin cao), thì 2 sinh MÔ TẢ có điều kiện
trên nhãn đó. Hiện hai việc chen nhau trong một cuộc bỏ phiếu.

## 6. HOÀN THIỆN §147 (đã thiết kế, chưa dựng)

- **Phân tầng priority**: `c là <lớp>` 0,85 · `+ thuộc tính MỌI anh em có` 0,70 ·
  `mô tả anh em gần nhất` 0,55. Truy hồi vốn tôn trọng priority.
- **Dải xuất xứ** `< 0,75` = suy ra ⇒ tri thức thật vào sau ở 1,0 TỰ ĐỘNG đè lên.
  Kèm công cụ dọn dải đó để nướng lại sau khi cải thiện bảng.
- **Nhãn DÀI cho thừa hưởng** (`kim loại màu xám`), nhãn NGẮN cho lớp. `bệnh lây`
  quá thô nên gộp sốt rét với lao.

## 7. KỶ LUẬT ĐO — bắt buộc, rút từ TK_046

- Khai cổng TRƯỚC, phân hoạch KÍN (§133 để lại khe 20–40%).
- Chấm ĐA HẠT GIỐNG (§125: `seed=0` là hạt may nhất về LẶP).
- Phép đếm RẺ «cơ chế có nổ không» TRƯỚC A/B (§115, §136 — bắt được 2 âm tính giả).
- In DỮ LIỆU THÔ cạnh mọi bảng đếm (§144 — phép đếm tự mâu thuẫn, chỉ dữ liệu thô
  bắt được).
- Kiểm CORPUS ĐÃ TRAIN trước mọi phép thử «không có nguồn» (§136 và §145 mắc
  cùng lỗi hai lần).

---

## §149 — BAO ĐÓNG LOẠI SUY TRÊN `W_skip`: cơ chế ĐÚNG, chất lượng KHÔNG TÁCH ĐƯỢC

Tác giả đính chính hướng đi: cạnh loại suy **phải suy ra từ dữ liệu gốc lúc
train**, không phải nhét câu tổng hợp vào corpus (tôi đã làm sai ở §148 — đó là
dạy thuộc, không phải suy ra). Đặc tả:

```
a…Y ⇒ Y↔a       X…a ⇒ X↔a
b…Y ⇒ Y↔b       X…b ⇒ X↔b       ⟹ phát hiện c↔Y
                X…c ⇒ X↔c
```

### 149.1 Diễn dịch sang phép toán ma trận

`build_w_skip` cho `mats[d]` = ma trận thưa `a→b` theo khoảng cách nhảy. Đặc tả
trên chính là:

```
S = Bn · Bnᵀ    ← tương tự: c và a cùng nối tới nhiều X  (B nhị phân, cosine)
E = S · W       ← c thừa hưởng cạnh THẬT của anh em
E = E − E∩W     ← chỉ giữ cạnh MỚI (đó mới là ngoại suy)
```

### 149.2 Bốn lần sai TRỤC trước khi chạy được

| lượt | sai | triệu chứng |
|---|---|---|
| §141-1 | quan hệ nền = bigram liền kề | `chó → để/đến/trên` |
| §141-2 | ngưỡng lấy phân vị trên TOÀN từ vựng | `chó`(gdf 1316), `kẽm`(66) đều bị loại |
| §141-3 | xếp hạng = đếm thô | `sắt → ta/đề/bản` |
| §149-1 | cắt HÀNG thay vì cắt CỘT | `sắt`,`chó`,`lúa` → *không có cạnh nào* |

Cắt hub phải áp cho **cột** (ngữ cảnh `X` phải mang thông tin), KHÔNG cho hàng
(từ được sinh cạnh). Lỗi này chỉ lộ khi tra một từ cụ thể — bảng tổng không thấy.

### 149.3 Chi phí — khả thi

```
W (d=1..4)         4 917 443 nnz
S thô              399 841 936  ← không giữ nổi
S top-10/hàng          265 036  ← top-k thay ngưỡng chung
E = S·W              8 341 921
E bỏ cạnh cũ, top-15   399 429  (8% của W)
```

### 149.4 CHẤT LƯỢNG: tách đôi theo THỂ LOẠI VĂN BẢN

```
S (tương tự) — ĐÚNG:
  sắt → mangan, nhôm, oxide, magie      kẽm → phosphor, mangan, oxide
  lúa → ruộng, cấy, đòng, gặt           bướm → bươm, nhộng
E (cạnh sinh ra):
  sắt → Cu, Ore, cobalt, Hg, CaO, SiO   kẽm → Cu, Fe, Hg, cobalt, silic   ✓
  chó → khẩu, hiếm, cũ, phương, dụ      lúa → hiếm, cũ, Khẩu, phương, dụ  ✗ (GIỐNG HỆT nhau)
  hổ  → rụp, systemic, thá, 川          mèo → ví, dụ, quỉ, seminar        ✗
```

Sạch cho từ **kỹ thuật** (ngữ cảnh hẹp, đặc trưng), rác cho từ **đời thường**
(`conv_stories` — truyện, ngữ cảnh tản mát). Cùng cơ chế, chất lượng phụ thuộc
THỂ LOẠI của văn bản chứa từ đó.

### 149.5 Ngưỡng tương tự KHÔNG cứu được — đo, không đoán

```
ca TỐT: sắt 0,196 · kẽm 0,243 · đồng 0,265 · bướm 0,321
ca XẤU: chó 0,175 · mèo 0,138 · hổ 0,187 · lúa 0,253
```

`lúa` (rác) **cao hơn** `kẽm` và `sắt` (sạch) ⇒ **chồng lấn hoàn toàn**, không
ngưỡng nào tách được. Và cả tám đều dưới trung vị từ vựng (0,408) — nhóm điểm
cao là token hiếm ngữ cảnh bé, tức ngưỡng cao chỉ giữ đúng nhóm vô dụng.

⇒ **KHÔNG cắm vào `build_w_skip`** theo đúng cách xử đã khai trước khi đo. Cắm
bây giờ sẽ đưa `hổ → rụp, systemic, 川` thành cạnh thật, phá ca vàng đang tốt.

### 149.6 Điều ĐÃ chắc, đáng giữ

Bước `S` **hoạt động**: `sắt → mangan/nhôm/oxide/magie`, `lúa → ruộng/cấy/đòng/
gặt`, `bướm → nhộng` — quan hệ CÙNG LỚP rút thuần tuý từ đồ thị đã học, không từ
điển, không nhãn, không dữ liệu thêm. Đây là vế `X…a, X…b, X…c` của đặc tả và nó
ĐÚNG. Chỗ hỏng nằm ở bước thừa hưởng `E`, và ở việc không phân biệt được từ nào
đáng thừa hưởng.

### 149.7 NGUYÊN TẮC — thuật toán KHÔNG phân biệt nguồn

Tác giả chốt: *«thuật toán không phân biệt nguồn, đối xử với mọi loại data train
như nhau, chỉ có chiến thuật train là khác nhau (sft / thường) — train sao nó
chạy vậy»*.

⇒ Đề xuất trước đó của tôi («chỉ sinh cạnh từ nguồn bách khoa, bỏ conv_stories»)
**BỊ LOẠI**: nó vá triệu chứng bằng cách phân biệt dữ liệu, thay vì sửa thuật
toán. Mọi cơ chế trong TK_053 phải chạy đồng nhất trên mọi cha, bất kể nguồn.

### 149.8 Hướng còn lại — đều KHÔNG phân biệt nguồn

- **Láng giềng TƯƠNG HỖ (mutual top-k)**: chỉ nhận `c ~ a` khi `a` nằm trong
  top-k của `c` **VÀ** `c` nằm trong top-k của `a`. Triệu chứng đo được — `chó`
  và `lúa` ra danh sách GIỐNG HỆT nhau — đúng dấu hiệu bị hub kéo: hub nằm trong
  top-k của mọi từ nhưng không có từ nào trong top-k của nó. Tương hỗ loại thẳng
  nhóm đó. Thuần thuật toán, không đụng tới nguồn.
- **PPMI trên ngữ cảnh trước khi cosine**: hiện `B` chỉ nhị phân hoá. PPMI hạ
  trọng số ngữ cảnh phổ biến ngay ở khâu tạo vector, thay vì sửa ở khâu sau.
- **Đồng thuận thay vì tổng**: `E` hiện là tổng có trọng số qua mọi anh em. Đòi
  ≥2 anh em CÙNG đề xuất một đích thì cạnh mới được sinh — cùng tinh thần `K≥2`
  của đặc tả gốc, áp ở bước thừa hưởng thay vì bước tương tự.
- **Đa nghĩa**: `kiếm → tìm, ra, có, thể` — danh từ *kiếm* lẫn động từ *kiếm*
  cùng một token. Cổng tương tự không tách được; cần tách nghĩa trước.

## §150 — Láng giềng TƯƠNG HỖ: thất bại NGƯỢC HƯỚNG, và đánh giá cả tuyến

### 150.1 Số

```
S top-10: 265 036 → TƯƠNG HỖ: 132 678 (50% sống sót)

TƯƠNG HỖ           sắt → (không có)   kẽm → (không có)   ← MẤT 2 ca ĐÚNG
                   chó → Munchkin, fu, mun, cau, meo
                   mèo → Corgi, Đốm, structive
TƯƠNG HỖ + ĐT≥2    sắt → (không có)   kẽm → (không có)
```

Triệu chứng nhắm tới ĐÃ HẾT — `chó` và `lúa` không còn ra danh sách giống hệt.
Nhưng giá phải trả là **hai ca duy nhất đang đúng bị xoá sạch**.

Lý do: anh em của `sắt` (`mangan`,`oxide`,`magie`) là từ kỹ thuật hiếm; top-10
của CHÚNG bị chi phối bởi thứ khác nên quan hệ không tương hỗ. Ràng buộc hai
chiều giết đúng nhóm có tín hiệu sạch nhất.

### 150.2 Năm lần thất bại ở CÙNG MỘT BƯỚC

| lượt | công thức bước thừa hưởng | kết quả |
|---|---|---|
| §141-1 | ghép 2 chặng bigram | hư từ |
| §141-2 | + ngưỡng phân vị gidf | mất hết từ thường dùng |
| §141-3 | + đếm thô | từ phổ biến thắng |
| §149 | S·W, cắt cột, chuẩn hoá đích | sạch cho kỹ thuật, rác cho đời thường |
| §150 | + tương hỗ / đồng thuận | mất luôn ca kỹ thuật |

Bước **TƯƠNG TỰ** (`S`) lần nào cũng cho tín hiệu THẬT: `sắt → mangan/nhôm/
oxide/magie`, `lúa → ruộng/cấy/đòng/gặt`, `bướm → nhộng`. Bước **THỪA HƯỞNG**
(`S·W`) hỏng ở cả năm công thức.

### 150.3 Đánh giá thẳng

Năm công thức khác nhau cùng hỏng ở một bước ⇒ không còn giống chuyện chỉnh
tham số. Giả thuyết đáng ngờ nhất: **`W_skip` không mang thông tin đủ để thừa
hưởng**. Nó là đồng hiện theo khoảng cách nhảy — mã hoá «hay đứng gần nhau»,
không mã hoá «là thuộc tính của». `sắt` và `nhôm` giống nhau thì suy được; nhưng
«cạnh của `nhôm`» gồm cả `sản xuất`, `tấm`, `công ty` — không phải thuộc tính
để thừa hưởng.

⇒ Cùng kết luận §142 ở tầng khác: đồng hiện cho **hàng xóm chủ đề**, không cho
**thuộc tính phân loại**. Bao đóng bậc hai kế thừa đúng giới hạn đó.

### 150.4 Khuyến nghị

DỪNG tuyến bao đóng trên `W_skip`. Không thử công thức thứ sáu. Phần đáng giữ là
bước `S` — quan hệ cùng lớp rút thuần tuý từ đồ thị, không từ điển — có thể dùng
cho việc khác (gợi ý từ gần nghĩa, mở rộng truy vấn), nhưng KHÔNG dùng để thừa
hưởng nội dung.

Các hướng TK_053 §1–§5 (entropy, soạn–thẩm, thang điểm, agenda mềm, hai thì)
KHÔNG phụ thuộc tuyến này và vẫn đứng nguyên.

## §151 — DỪNG THEO ĐỘ TIN: ĐẠT, vào mặc định

### 151.1 Đo trước khi cắm — tách bạch 4,6 lần

`p_top1` = xác suất đỉnh của phân bố điểm mỗi bước:

```
sinh bình thường  0,38 – 0,96
hết nguyên liệu   0,029 – 0,082      ← khoảng trống 4,6×, KHÔNG chồng lấn
```

Ba ca có nhãn biết trước:

- `mặt trời là` phát rác `ngôi` ở bước có `p_top1 = 0,082` ⇒ tín hiệu «hết
  nguyên liệu» có SẴN một bước TRƯỚC khi mạng phát bậy.
- `chó là`/`lúa là` thấp nhất `0,38` ⇒ không bị cắt oan.
- `sóc là` lúc cụt có `p_top1 = 0,076` ⇒ **đính chính §124.4**: `sóc` cụt vì
  mạng THẬT SỰ không có gì để nói tiếp — THIẾU DỮ LIỆU, không phải lỗi luật
  dừng. §119–§130 vá sai chỗ suốt bốn mục.

### 151.2 Kết quả (6 hạt, 20 prompt)

```
độ_tin   TRUNG THÀNH   TRẬT TỰ     LẶP
0,00     86,1 ±1,3     97,2 ±1,8   9,9 ±2,4
0,10     88,3 ±0,3     99,4 ±0,0   9,7 ±0,4
0,15     88,3 ±0,3     99,3 ±0,0   9,9 ±0,4      ← MẶC ĐỊNH
0,25     88,0 ±0,0     99,3 ±0,0   9,4 ±0,3
```

Chênh `+2,2` với SE gộp ≈ 0,39 ⇒ **5,6 SE**. Ca vàng nguyên vẹn; `mặt trời là`
hết đuôi rác (`…chiếm khoảng` thay vì `…chiếm khoảng ngôi`).

### 151.3 PHƯƠNG SAI SỤP — hệ quả không lường trước

`±1,3 → ±0,3` · `±1,8 → ±0,0` · `±2,4 → ±0,4`. `TRẬT TỰ 99,3 ±0,0` nghĩa là sáu
hạt giống cho CÙNG một con số. Lý do: cơ chế cắt đúng vùng mạng không chắc chắn
— vùng mà lấy mẫu (`T = 0,5`, §124) quyết định kết quả. Cắt đi thì đầu ra gần
như tất định.

⇒ Ngoài chất lượng, nó còn làm hệ **có thể tái lập** — mọi phép đo sau này chặt
hơn hẳn.

### 151.4 Bài học

- **269.** Thay LUẬT bằng ĐẠI LƯỢNG. §116–§118 tốn 4 mục vá `n_b_gen`/`_bd_gen`/
  cú đẩy biên — toàn luật thủ công về «biên là gì» — và vẫn còn đuôi rác. Một
  câu hỏi đo được («phân bố còn đỉnh rõ không») giải quyết gọn hơn.
- **270.** Đo TRƯỚC khi cắm, trên ca có NHÃN BIẾT TRƯỚC (dừng đúng / cụt sớm /
  đuôi rác), cho thấy ngay khoảng trống 4,6× — và cũng lật lại chẩn đoán `sóc`
  đã sai suốt bốn mục.

## §152 — Chấm lại cửa sổ agenda bằng thước đã chặt: §113 ĐÚNG, §112 SAI

§125 ghi hai quyết định chưa chứng minh được vì nằm trong nhiễu. §151 làm phương
sai sụp (`±1,3 → ±0,3`) nên phân định được.

```
window   TRUNG THÀNH   TRẬT TỰ     LẶP     «lúa là»
4        84,2 ±0,5     95,3 ±1,0   8,1     …thân cỏ rỗng hoa lưỡng tính, hoa   ✗ vỡ
3        83,5 ±1,2     96,2 ±0,6   6,9     …hoa lưỡng không, hoa               ✗ vỡ
2        83,5 ±1,2     95,9 ±0,5   7,7     …thân cỏ rỗng hoa, bao hoa          ✗ vỡ
1        88,3 ±0,3     99,3 ±0,0   9,9     …không có bao hoa, quả có vỏ trấu   ✓ ĐÚNG
```

`w=1` hơn mức nhì **+4,1 TT** và **+3,1 TQ** — với `±0,3` thì đó là ~10 sigma.
⇒ §113 (đổi 3→1) ĐÚNG; §112 (chọn 3) SAI, đúng như §125 đã ngờ.

**Giá trị thứ hai của §151**: nó biến thước thành công cụ PHÂN ĐỊNH ĐƯỢC. Cùng
phép quét, cùng bộ prompt — trước nằm trong nhiễu, giờ chênh 10 sigma. Mọi quyết
định tồn đọng vì «trong biên độ» đều có thể chấm lại.

## §153 — Hạ ngưỡng độ tin 0,15 → 0,05: thước KHÔNG phân định được

### 153.1 Bộ đánh giá của tôi che mất tác dụng phụ

§151 vào mặc định ở 0,15 sau khi qua cổng (mục tiêu + phủ quyết + 3 thước × 6
hạt). Nhưng bộ 20 prompt có **19/20 là khuôn `X là`** và KHÔNG có `muối`, nên
cổng không thấy:

```
muối là            0,00 ▸ sản phẩm axit và bazơ.   →  0,15 ▸ sản phẩm axit   ✗ MẤT
vì sao trời mưa?   0,00 ▸ (rác)                    →  0,15 ▸ (RỖNG)
ai là người · sắt và đồng đều là · nghịch ngợm như loài  →  (RỖNG)
```

Cùng lỗi §113 ở tầng khác: ở đó THƯỚC mù với thứ tự; ở đây BỘ PROMPT mù với thể
loại. Tôi ghi bài học về thước nhưng không nghĩ tới THÀNH PHẦN bộ đánh giá.

### 153.2 Thước không phân định được giữa hai mức

```
độ_tin   TRUNG THÀNH   TRẬT TỰ     LẶP
0,00     86,1 ±1,3     97,2 ±1,8   9,9
0,05     86,8 ±0,7     99,2 ±0,3   9,4
0,10     88,3 ±0,3     99,4 ±0,0   9,7
0,15     88,3 ±0,3     99,3 ±0,0   9,9
```

`0,10/0,15` hơn `+1,5 TT` — nhưng TRUNG THÀNH là **TỈ LỆ** (đúng/tổng) nên **cắt
sâu hơn thì tự tăng**. Hiện vật độ dài §113. ⇒ `+1,5` KHÔNG chứng minh tốt hơn;
thước này về cấu tạo không so được hai mức khác nhau ở độ sâu cắt.

### 153.3 Phân định bằng ca đọc được

```
0,05  muối là ▸ sản phẩm axit VÀ BAZƠ.   ĐÚNG   ·  mặt trời ▸ …khoảng NGÔI  rác lộ liễu
0,15  muối là ▸ sản phẩm axit             SAI    ·  mặt trời ▸ …khoảng        sạch
```

Chọn **0,05**: `sản phẩm axit` là phát biểu SAI đọc như hoàn chỉnh (muối là sản
phẩm của axit VÀ bazơ) — cùng dạng nguy hiểm với `bộ binh`↔`kị binh` tác giả đã
chỉ ra. `…khoảng ngôi` thì hỏng lộ liễu, người đọc biết ngay.

`0,05` vẫn giữ phần lớn lợi ích cấu trúc: TRẬT TỰ `97,2 → 99,2`, phương sai
`±1,8 → ±0,3`.

### 153.4 Bài học

- **271.** Cổng khai trước vẫn có thể MÙ nếu BỘ ĐÁNH GIÁ thiên lệch. 19/20 prompt
  cùng một khuôn ⇒ cổng không thể phát hiện hỏng ở khuôn khác. Kiểm THÀNH PHẦN bộ
  đánh giá, không chỉ kiểm thước.
- **272.** Thước dạng TỈ LỆ không so được hai cấu hình khác nhau ở ĐỘ DÀI đầu ra
  — nó luôn thưởng cấu hình cắt sâu hơn. Khi hai mức chỉ khác chỗ cắt, phải phân
  định bằng ca đọc được, không bằng con số.
- **273.** «Sai mà đọc như hoàn chỉnh» nguy hiểm hơn «hỏng mà trông hỏng». Ưu
  tiên giữ nội dung đúng hơn là làm sạch bề mặt.

## §154 — CÒN LẠI (chưa làm)

- Sửa BỘ ĐÁNH GIÁ: thêm prompt hỏi / mở / loại suy vào bộ chấm chuẩn.
- `mùa xuân ▸ RỖNG` ở MỌI ngưỡng kể cả 0,0 — lỗi có sẵn, chưa truy.
- Phép nối hỏng `and n_b_gen >= 1` còn sống ở khối ÉP CỨNG ~876 (§124.5).
- TK_053 §2 soạn–thẩm · §4 agenda mềm · §5 sinh hai thì — chưa làm.
  §3 hiệu chỉnh thang điểm: ĐÃ ĐO, bất đối xứng còn nhưng §151 đã chặn phần lớn
  tác hại; KHÔNG có ca hỏng đo được để sửa ⇒ hoãn (tránh lặp §119/§138/§140:
  «chạy đúng, không lợi ích»).

## §155 — SỬA BỘ ĐÁNH GIÁ: thêm nhóm prompt, thước PHỦ, đếm RỖNG

§153 phát hiện cổng vẫn mù vì bộ đánh giá thiên lệch. Sửa ba chỗ:

### 155.1 Ba sửa đổi

- **Bộ prompt theo NHÓM**: 30 prompt / 4 nhóm — ĐỊNH NGHĨA 15 · HỎI 4 · LOẠI SUY
  7 · MỞ 4. Bản cũ 19/20 cùng khuôn `X là`. Thêm `muối là`, `khỉ là`, `bướm là`
  — những ca ĐÃ bắt được lỗi thật mà bộ cũ không có.
- **Thước PHỦ (recall)**: `(từ nguồn có mặt trong sinh)/(tổng từ nguồn)`.
  `TRUNG THÀNH` là TỈ LỆ nên cắt sâu hơn thì TỰ TĂNG; `PHỦ` đi NGƯỢC CHIỀU nên
  cắt là TỤT. Hai thước ngược chiều mới bắt được mất nội dung.
- **Đếm RỖNG**: prompt không sinh token nào. Bản cũ loại chúng khỏi mẫu số nên
  chúng vô hình.

### 155.2 Tự kiểm — bộ mới thấy đúng thứ bộ cũ mù

```
độ_tin   TRUNG THÀNH   PHỦ        TRẬT TỰ    LẶP   RỖNG
0,00     80,6 ±3,5     34,5 ±1,5  95,8 ±1,9  9,7   1,5/30
0,05     84,5 ±1,5     34,2 ±1,4  98,3 ±1,0  8,3   7,7/30
0,10     86,3 ±1,4     34,0 ±1,4  98,9 ±0,4  7,9   8,7/30
0,15     86,3 ±1,1     33,7 ±1,4  98,9 ±0,5  8,0   9,7/30
```

`TRUNG THÀNH` TĂNG theo ngưỡng · `PHỦ` GIẢM đơn điệu · `RỖNG` TĂNG mạnh — đúng
hành vi dự kiến ⇒ bộ mới hoạt động.

### 155.3 Điều bản cũ GIẤU HOÀN TOÀN

Ở `0,15`: **9,7/30 prompt hoá RỖNG** — gần một phần ba. Ở `0,05`: 7,7/30, so với
1,5/30 khi tắt. Tôi tưởng chỉ 4 prompt bị ảnh hưởng (số nhìn thấy bằng mắt);
thực tế thêm ~6 prompt nữa.

**Nhưng đánh đổi vẫn chấp nhận được ở 0,05**: `PHỦ` chỉ tụt `34,5 → 34,2`, nằm
gọn trong nhiễu `±1,4` ⇒ nội dung THẬT gần như không mất; prompt hoá rỗng vốn
đang sinh RÁC. Đổi lại `+3,9 TT` và `+2,5 TQ`.

⇒ Giữ mặc định `0,05`. Khác biệt: giờ cái giá HIỆN RÕ trên bảng thay vì ẩn.

### 155.4 Bài học

- **274.** Bộ đánh giá cần thước NGƯỢC CHIỀU nhau. Một thước tỉ lệ đơn độc luôn
  có hướng thiên lệch; thước thứ hai ngược chiều biến thiên lệch đó thành đánh
  đổi NHÌN THẤY ĐƯỢC.
- **275.** Prompt không sinh gì bị LOẠI KHỎI MẪU SỐ của mọi thước tỉ lệ ⇒ vô
  hình. Phải đếm riêng. Đây là cách một thay đổi làm rỗng 1/3 bộ prompt mà vẫn
  «cải thiện» mọi thước.

## §156 — Chặn EOS ở BƯỚC ĐẦU: RỖNG 7,7 → 0,0, nội dung không mất

### 156.1 Chẩn đoán `mùa xuân ▸ rỗng`

`mùa xuân` sinh đúng 1 token = EOS (rỗng) ở MỌI ngưỡng kể cả 0,0. Nguyên nhân:
`xuân` hay đứng CUỐI CÂU trong corpus (`...mùa xuân.`) ⇒ phiếu bước 0 chọn EOS.
Thêm `là`/`rất` thì chạy bình thường (12 token). Không phải lỗi §151 — mạng thật
sự coi `mùa xuân` là cụm đã trọn.

### 156.2 Bất biến: mỗi lượt sinh phải ra ≥1 token

Chặn EOS khi `out` chỉ có prompt (chưa sinh token nào); cũng không dừng theo độ
tin ở bước đầu. Không cờ.

### 156.3 Kết quả (30 prompt, 6 hạt)

```
              TRUNG THÀNH   PHỦ        TRẬT TỰ    LẶP   RỖNG
trước (§155)  84,5 ±1,5     34,2 ±1,4  98,3 ±1,0  8,3   7,7/30
sau  (§156)   83,6 ±1,3     34,2 ±1,4  97,5 ±0,8  8,1   0,0/30
```

- `RỖNG 7,7 → 0,0`: mục tiêu đạt hoàn toàn.
- `PHỦ` GIỮ NGUYÊN 34,2: nội dung THẬT không mất — prompt hoá rỗng giờ sinh chữ
  nhưng không lấy đi phần phủ câu nguồn.
- `TRUNG THÀNH −0,9`: hiện vật TỈ LỆ — 7,7 prompt vào lại mẫu số mang chữ chưa
  hoàn hảo. `PHỦ` không tụt ⇒ là loãng mẫu số, KHÔNG phải mất nội dung.
- Ca vàng nguyên vẹn; `mùa xuân ▸ phần ấm áp ấm` (ấm = đúng nghĩa mùa xuân).

### 156.4 Bài học

- **276.** Bộ đánh giá §155 tự chứng minh giá trị NGAY: chỉ có TRUNG THÀNH thì
  `−0,9` trông như hại; `PHỦ` + `RỖNG` cạnh nhau đọc đúng thành cải thiện. Thước
  ngược chiều biến «hại giả» thành «lợi thật nhìn thấy được».
- **277.** Ca hỏng TẤT ĐỊNH + bản vá nhắm ĐÚNG cơ chế đo được (EOS thắng bước 0)
  ⇒ khác §119/§138/§140 («chạy đúng, vô ích»). Điều kiện để vá đáng làm: có ca
  hỏng cụ thể, không phải «hợp lý về nguyên tắc».

## §157 — GỠ khối ép cứng EOS: mã chết, đã xoá

§124.5 ghi khối ÉP CỨNG (`nxt = int(eos_id)` khi ở biên + `n_b_gen >= 1`) còn
mang phép nối hỏng và chưa vá. Kiểm bằng A/B trước khi động.

### 157.1 A/B — mã chết

Chạy 20 prompt hai lần (có / bỏ nhánh ép, cùng seed):

```
đầu ra KHÁC:                              0/20
số lần ép EOS mà argmax lẽ ra chọn khác: 0
```

Nhánh chỉ từng gán EOS ở đúng lúc EOS đã là argmax thắng sẵn ⇒ bỏ không đổi gì.

### 157.2 Vì sao chết

§118 (cộng điểm EOS) + §151 (dừng theo độ tin) + §156 (chặn EOS bước đầu) đã
chuyển toàn bộ việc dừng sang tầng ĐIỂM SỐ. Tới lúc chọn token, nếu ở biên câu
trọn thì EOS đã thắng sẵn trong `scores` — ép cứng thành thừa.

### 157.3 Xoá

Gỡ cả khối `if boundary_eos_beta > 0 and out ...` bao quanh chọn token, còn lại
`nxt = argmax/sample`. Kéo theo: phép nối hỏng `and n_b_gen >= 1` (§124.5, món
nợ từ 33 mục trước) BIẾN MẤT, lời gọi `count_boundaries` thừa + import `_is_b_g`
tại đó cũng biến mất. LOI7 cập nhật: chỉ còn 1 chỗ gọi `_is_boundary` (§118),
vẫn canh nó truyền `_bd_gen`. 454 pass.

### 157.4 Bài học

- **278.** Cơ chế cũ không tự biến mất khi cơ chế mới thay nó — nó thành MÃ CHẾT
  mang theo nợ (phép nối hỏng). Sau mỗi thay đổi lớn ở một tầng (§151 đổi cách
  dừng), rà lại các cơ chế CŨ cùng tầng: chúng có thể đã thừa. A/B «bỏ đi có đổi
  gì không» là phép kiểm dứt điểm.

## §158 — TẠO THỤ ĐỘNG cạnh c→Y trong ĐỒ THỊ (tồn tại lúc bake, chưa tham gia sinh)

Tác giả làm rõ yêu cầu THẬT: không cần cạnh MẠNH để sinh ngay, mà cần cạnh
`c↔Y` **tồn tại vật lý** trong đồ thị, tạo lúc train, YẾU cũng được — «sau này
sẽ có thuật toán khác củng cố». §149–150 tôi hiểu sai: cứ đòi chất lượng cao để
tham gia sinh, rồi kết luận «không cắm được». Sai bài toán.

### 158.1 §143 hoá ra là điều kiện THUẬN LỢI

`build_w_skip` (đường sinh) đọc `ps.parents`, KHÔNG đọc `TypedAdjacency`. ⇒ cạnh
thêm vào bảng kề TỒN TẠI mà KHÔNG nhiễu sinh — đúng «còn yếu nên không tham
gia». Thêm `EdgeKind.ANALOGY` với `INFERENCE_PRIORITY = 0.0` (khuôn `SEQUENTIAL`/
`SENSE`: ngoài mọi nhóm readout).

### 158.2 Nướng — phép toán §149, nạp vào adjacency

```
S = Bn·Bnᵀ (cosine, cắt cột hub, ngưỡng ≥0,05)   c ~ a qua ngữ cảnh chung
E = S·W    (thừa hưởng cạnh anh em ⇒ c→Y)
E -= E∩W   (chỉ cạnh MỚI = ngoại suy)
top-15/hàng, weight 0,05 ⇒ adjacency kind=ANALOGY
```

`scripts/nuong_canh_analogy.py`: **398 832 cạnh** trong 73s (compute 3s + lưu 70s).

### 158.3 HAI BẤT BIẾN xác nhận

- **Persist qua reload**: 398 832 cạnh ANALOGY còn nguyên sau nạp lại từ đĩa,
  truy được: `sắt → natri/silic/Cu/magnesi`, `kẽm → nhôm/silic/Fe`.
- **Sinh KHÔNG đổi**: 5 ca vàng ra y hệt baseline (`muối là ▸ sản phẩm axit và
  bazơ.`, `khỉ là ▸ loài linh trưởng cỡ nhỏ`). Cạnh không tham gia.

⇒ ĐẠT yêu cầu: cạnh `c→Y` tồn tại, thụ động, sẵn cho thuật toán củng cố sau.

### 158.4 Chất lượng cạnh — lẫn lộn, CHẤP NHẬN ĐƯỢC vì thụ động

Tốt cho từ kỹ thuật (`sắt`,`kẽm` → hoá học), rác cho từ đời thường
(`chó`,`lúa` → `dụ/khẩu/cũ`) — đúng §149. Vì cạnh YẾU + chưa tham gia, rác không
hại; thuật toán củng cố sau sẽ TỈA (trọng số thấp → cắt) và NÂNG (đúng → tăng).

### 158.5 Bài học

- **279.** Xác định BÀI TOÁN trước khi tối ưu. §149–150 tôi tối ưu «cạnh đủ tốt
  để SINH» suốt 5 lượt; yêu cầu thật là «cạnh TỒN TẠI để CỦNG CỐ SAU». Bài toán
  thứ hai giải trong 1 lượt vì không đòi chất lượng.
- **280.** `bake_derived=True` khi save mine lại TOÀN BỘ chỉ mục (28 phút chưa
  xong). Chỉ thêm cạnh adjacency thì KHÔNG cần — W_skip/seq đọc `ps.parents`,
  không phụ thuộc adjacency. Lưu thẳng: 70s.
- **281.** `print()` ra FILE bị đệm khối ⇒ tôi tưởng compute treo (thực ra xong
  trong 3s, kẹt buffer) và GIẾT NHẦM 2 tiến trình đang chạy tốt. `PYTHONUNBUFFERED=1`
  từ ĐẦU khi chạy nền, luôn.

---

# TỔNG KẾT TK_053 (§151 – §158)

## Trạng thái

Đường sinh — bốn cơ chế vào mặc định, mỗi cơ chế có ca hỏng cụ thể + đo chứng minh:

```
              TRUNG THÀNH   TRẬT TỰ     LẶP        RỖNG
đầu           79,3          89,3        13,6        —
cuối          86,8 ±0,7     99,2 ±0,3   9,4 ±0,5    0/30
```

- §151 DỪNG THEO ĐỘ TIN (0,05) — thay 4 mục luật biên thủ công; phương sai sụp
  `±1,8 → ±0,0` ⇒ hệ TÁI LẬP ĐƯỢC; nhờ đó §152 đóng câu hỏi treo từ §125.
- §156 chặn EOS bước đầu — `RỖNG 7,7 → 0,0`.
- §157 GỠ khối ép cứng EOS (mã chết) — trả nợ phép nối hỏng 33 mục tuổi.
- §155 SỬA BỘ ĐÁNH GIÁ — 30 prompt/4 nhóm + thước PHỦ (ngược chiều) + đếm RỖNG.

Cạnh loại suy — §158 ĐẠT yêu cầu THỤ ĐỘNG:
- **398 832 cạnh `c→Y`** trong đồ thị (`EdgeKind.ANALOGY`, w=0,05, priority=0,0).
- Suy ra từ bao đóng tương tự trên `W_skip` (`S=Bn·Bnᵀ`, `E=S·W`, bỏ cạnh cũ).
- Persist qua reload · sinh KHÔNG đổi · mọi cạnh là NGOẠI SUY (`c` chưa có cạnh
  trực tiếp). Ví dụ sạch: `sắt→natri/silic/Cu/magnesi`, `natri→hydro/soda/Kali`.
- Chất lượng lẫn lộn: tốt cho từ kỹ thuật, rác cho đa nghĩa (`đồng→SWAT`).

## §159 — ĐỀ XUẤT CỦNG CỐ cạnh c→Y (đặc biệt khi là TRI THỨC THẬT)

Vấn đề: 398 832 cạnh yếu, chất lượng lẫn. Cần thuật toán NÂNG cạnh đúng, TỈA
cạnh nhiễu. «Tri thức thật» = quan hệ c→Y được XÁC NHẬN bởi bằng chứng NGOÀI
phép suy loại suy. Bốn nguồn xác nhận, xếp theo sức phân biệt tri-thức-thật:

### A. XÁC NHẬN HEBBIAN TRỄ — mạnh nhất cho «tri thức thật»

Cạnh ANALOGY là một DỰ ĐOÁN. Khi train/đọc tiếp mà `c` và `Y` THẬT SỰ đồng hiện
trong một câu thật ⇒ thực tế xác nhận dự đoán ⇒ củng cố. Đây đúng cách não hợp
nhất: giả thuyết được trải nghiệm xác nhận thì mạnh lên.

Cài: lúc ingest câu mới, nếu câu chứa cả `c` lẫn `Y` VÀ đã có cạnh `ANALOGY
c→Y` ⇒ THĂNG cạnh: tăng trọng số, hoặc DI CƯ sang `EdgeKind.REINFORCED`
(priority 0,50 — tham gia sinh). Cạnh không bao giờ được xác nhận ⇒ ở lại 0,05.

⇒ Phân biệt tri-thức-thật (được thực tế xác nhận) khỏi loại-suy-suông (chỉ hợp
lý). KHÔNG cần nguồn ngoài — chính dòng dữ liệu train làm trọng tài.

### B. ĐỐI CHIẾU BẢNG NHÃN LỚP (§128, 84,9%) — xác nhận từ tri thức curated

Bảng `nhan_lop.json` map từ→lớp độ tin cao. Nếu `c→Y` mà `Y` NẰM TRONG định
nghĩa/lớp thật của `c` ⇒ xác nhận bởi nguồn ĐỘC LẬP (từ điển) ⇒ củng cố.
Ví dụ `sắt→kim loại` thăng vì bảng nói `sắt` LÀ `kim loại`.

⇒ Neo loại suy vào tri thức curated. Bù cho A ở những cạnh đúng mà data train
chưa kịp xác nhận.

### C. ĐỒNG THUẬN ANH EM — tín hiệu tin cậy nội tại

`E = S·W` hiện CỘNG qua mọi anh em. Đếm SỐ ANH EM PHÂN BIỆT cùng đề xuất một
`Y`: nhiều (≥K) ⇒ thuộc tính LỚP thật; một ⇒ trùng hợp. Nâng cạnh đồng-thuận
cao, tỉa cạnh một-anh-em. (Đúng điều kiện `K≥2` đặc tả gốc, áp ở bước củng cố.)

### D. NHẤT QUÁN TAM GIÁC — tỉa cạnh treo

Cạnh `c→Y` đáng tin hơn nếu nằm trong TAM GIÁC nhất quán: `c→Y`, `Y→d`, `c→d`
cùng tồn tại ⇒ cạnh cắm trong vùng mạch lạc. Cạnh treo (không tam giác) ⇒ nhiễu,
tỉa. Thuần đồ thị, không nguồn ngoài.

### Tiền đề chung — TÁCH NGHĨA trước (nếu muốn chất lượng cao)

`đồng→SWAT` sinh từ đa nghĩa token (`đồng` kim loại / cánh đồng / tiền). Gom cụm
ngữ cảnh các lần xuất hiện của `c` thành các NGHĨA, tính tương tự PER NGHĨA. Đây
là tiền đề cho chất lượng, không phải củng cố — nhưng A/B/C/D đều chạy tốt hơn
sau khi tách nghĩa.

### Khuyến nghị thứ tự

1. **A (Hebbian trễ)** — cơ chế chính, phân biệt tri-thức-thật, không nguồn ngoài.
2. **B (đối chiếu bảng)** — bù cho A, neo vào curated, rẻ (bảng đã có).
3. **C + D** — lọc nhiễu nội tại, chạy một lần trên đồ thị hiện tại.
4. Tách nghĩa — nếu chất lượng A/B/C/D chưa đủ.

A và B cùng nhau chính là định nghĩa «tri thức thật»: **được thực tế xác nhận
(A) HOẶC được tri thức curated xác nhận (B)** — khác với loại suy chỉ hợp lý.
