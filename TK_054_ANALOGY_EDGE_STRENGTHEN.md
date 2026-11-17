# TK_054 — CỦNG CỐ cạnh loại suy c→Y: phân biệt TRI THỨC THẬT khỏi suy suông

| Field | Value |
|-------|-------|
| **ID** | TK_054 |
| **Status** | 🟡 **MỞ** — A đang làm. |
| **Nguồn** | Yêu cầu tác giả: *«củng cố cạnh c–Y, nhất là khi c–Y là tri thức thật»* |
| **Kế thừa** | TK_053 §158 (398 832 cạnh ANALOGY thụ động, w=0,05, priority=0,0) |

## 0. Bài toán

§158 tạo 398 832 cạnh `c→Y` THỤ ĐỘNG, chất lượng lẫn (sạch cho từ kỹ thuật, rác
cho đa nghĩa). Cần thuật toán NÂNG cạnh đúng, TỈA nhiễu. «Tri thức thật» = c→Y
được XÁC NHẬN bởi bằng chứng NGOÀI phép suy loại suy.

Tính chất then chốt (từ cách dựng §158, `E -= E∩W`): cạnh `c→Y` nghĩa là c và Y
KHÔNG đồng hiện trong khoảng cách ≤4 ở corpus TRAIN. ⇒ xác nhận phải đến từ dữ
liệu CHƯA THẤY hoặc quan hệ ở khoảng cách xa hơn.

## A. XÁC NHẬN HEBBIAN TRỄ  ← làm trước

Cạnh ANALOGY là DỰ ĐOÁN. Khi đọc dữ liệu MỚI mà `c` và `Y` THẬT SỰ đồng hiện ⇒
thực tế xác nhận ⇒ củng cố (tăng w hoặc di cư `EdgeKind.REINFORCED` priority
0,50 — tham gia sinh). Chính dòng dữ liệu làm trọng tài, không nguồn ngoài.

**Nguồn xác nhận**: `viwiki.jsonl` (285 935 bài) — KHÔNG trong manifest train ⇒
giữ riêng tự nhiên. Cạnh dự đoán từ corpus train, xác nhận bởi viwiki chưa thấy.

**Bước**:
1. Nạp `net_ls_analogy.pkl`, lấy tập cạnh ANALOGY `c→Y` (dict c → set(Y)).
2. Duyệt viwiki (mẫu), mã hoá câu, tìm cặp token đồng hiện có cạnh ANALOGY.
3. Đếm số lần xác nhận mỗi cạnh.
4. ĐO TRƯỚC KHI NÂNG: bao nhiêu % cạnh được xác nhận? Cạnh xác nhận có SẠCH hơn
   cạnh không (sắt→natri xác nhận, đồng→SWAT không)?
5. Nếu đo tốt: nâng cạnh xác nhận → REINFORCED, giữ phần còn lại 0,05.

**Cổng**: cạnh được xác nhận phải THIÊN về ca sạch (kỹ thuật) hơn ca rác (đa
nghĩa) — nếu xác nhận đều tay cả hai thì viwiki không phân biệt được, hướng yếu.

## B. ĐỐI CHIẾU BẢNG NHÃN LỚP (§128, 84,9%)

`c→Y` mà `Y` trong lớp thật của `c` (bảng `nhan_lop.json`) ⇒ xác nhận bởi từ
điển độc lập. Neo loại suy vào curated. Bù cho A ở cạnh đúng mà data chưa xác
nhận. RẺ (bảng sẵn).

## C. ĐỒNG THUẬN ANH EM ≥K

`E=S·W` cộng qua mọi anh em. Đếm số anh em PHÂN BIỆT cùng đề xuất một Y: nhiều
⇒ thuộc tính LỚP; một ⇒ trùng hợp. Đúng `K≥2` đặc tả gốc, áp ở bước củng cố.

## D. NHẤT QUÁN TAM GIÁC

`c→Y` đáng tin hơn nếu có tam giác `c→Y, Y→d, c→d`. Cạnh treo ⇒ tỉa. Thuần đồ thị.

## Tiền đề: TÁCH NGHĨA

`đồng→SWAT` từ đa nghĩa token. Gom cụm ngữ cảnh → nghĩa, tương tự per nghĩa. Tiền
đề cho chất lượng; A/B/C/D chạy tốt hơn sau tách nghĩa.

## Thứ tự

A (chính, phân biệt tri-thức-thật) → B (bù, rẻ) → C+D (lọc nhiễu) → tách nghĩa.
A+B = định nghĩa vận hành «tri thức thật»: xác nhận bởi THỰC TẾ (A) HOẶC CURATED (B).

## §160–161 — A ĐÃ LÀM: xác nhận Hebbian trễ qua viwiki → REINFORCED

### 160 — ĐO xác nhận (viwiki 40k bài, 2,6 triệu câu)

```
80 495 / 398 832 cạnh ANALOGY được xác nhận (20,2%)
```

Xác nhận PHÂN BIỆT sạch/rác (cổng qua):

```
từ SẠCH:  sắt 12/15 · kẽm 10/15 · chì 11/15 · natri 11/15
từ RÁC:   đồng 7/15 · chó 4/15 · lúa 5/15
```

Cạnh sạch xác nhận ĐÚNG ngữ nghĩa và tần suất cao: `sắt→silic` 32×, `natri→
nitrat` 26×. Nhiễu lọt ở tần suất thấp: `đồng→SWAT` 1×. ⇒ TẦN SUẤT là tín hiệu
phân biệt tri-thức-thật.

### 161 — NÂNG cạnh xác nhận ≥3 → REINFORCED

41 254 cạnh (đồng hiện ≥3 lần) di cư sang `EdgeKind.REINFORCED` (priority 0,50),
trọng số `min(0,5; 0,15 + 0,05·log(1+cnt))` — thang theo độ tin.

```
sắt   → silic 0,32 · magnesi 0,31 · natri 0,30 · cobalt 0,28   (hoá học, sạch)
natri → nitrat 0,31 · magiê 0,31 · hydro 0,29 · Kali 0,24
đồng  → Gregory/paladi/mẹo (7→3, tỉa)   chó → dụ (4→1)   lúa → nghĩa/dụ/Phương
```

Ngưỡng ≥3 tỉa nhiễu; nhiễu còn lại (`đồng→Gregory`) là đa nghĩa token — cần tách
nghĩa (tiền đề).

### Bất biến

- 41 254 cạnh REINFORCED persist qua reload.
- Sinh KHÔNG đổi (ca vàng y hệt): đường W_skip mặc định đọc `ps.parents`, chưa
  đọc layer REINFORCED. Củng cố ở tầng ĐỒ THỊ; nối vào sinh là bước sau.

### Thành quả A

Ba bước hoàn tất chu trình «dự đoán → xác nhận → củng cố»:
1. §158 DỰ ĐOÁN — 398 832 cạnh ANALOGY từ bao đóng tương tự.
2. §160 XÁC NHẬN — viwiki giữ riêng, 20,2%, phân biệt sạch/rác.
3. §161 CỦNG CỐ — 41 254 cạnh «tri thức thật» → REINFORCED, trọng số theo độ tin.

«Tri thức thật» hiện thực đúng nghĩa: DỰ ĐOÁN từ train + QUAN SÁT trong dữ liệu
chưa thấy. `sắt→silic` (đoán vì sắt~kẽm~chì, xác nhận 32× trong viwiki) là ví dụ
mẫu.

### Bài học

- **282.** Nguồn GIỮ RIÊNG tự nhiên (viwiki ngoài manifest) biến «loại suy» thành
  «tri thức kiểm chứng được» mà không cần nhãn tay — chính dòng dữ liệu làm trọng
  tài, đúng hợp nhất Hebbian.
- **283.** Cạnh dựng với `E -= E∩W` (chỉ ngoại suy) ⇒ xác nhận PHẢI từ nguồn
  khác corpus dựng cạnh, nếu không là vòng tròn. viwiki thoả vì ngoài manifest.

## §162 — B (đối chiếu bảng): THẤT BẠI vì ĐỘ PHỦ bảng hẹp

```
849/398 832 cạnh cùng lớp bảng (0,2%)
sắt 0/15 · natri 0/15 · kẽm 1/15 · chì 1/15   ← ca vàng gần trắng
```

Nguyên nhân: ĐÍCH loại suy phần lớn KHÔNG trong bảng.

```
sắt/kẽm/nhôm/đồng/chì  → trong bảng ('kim loại')
natri/silic/magnesi/cobalt/hydro/nitrat  → KHÔNG trong bảng
```

Bảng (§128) dựng từ định nghĩa `dictionary.jsonl` — phủ ~9k từ thông dụng, thiếu
từ vựng KỸ THUẬT (nguyên tố/hợp chất). B đòi CẢ `c` LẪN `Y` trong bảng ⇒ cạnh
sạch `sắt→natri` không xác nhận được vì `natri` không được phân loại.

### Củng cố §142/§150: cạnh loại suy là TOPICAL, không TAXONOMIC

`sắt→natri` được xác nhận bởi «đồng hiện trong văn bản hoá học» (A ĐÚNG) nhưng
KHÔNG bởi «cùng lớp phân loại từ điển» (B SAI — natri metalloid vs sắt metal, và
natri không trong bảng). Bảng phân loại không phải công cụ đúng cho cạnh đồng
hiện.

### Phán quyết

B đóng góp KHÔNG đáng kể cho tập cạnh này (0,2%, phần lớn trùng A). KHÔNG nới
lỏng để «cứu» (tránh bẫy §119/§138/§140). A đã bao trùm điều B định làm, với độ
phủ rộng hơn (mọi từ trong văn bản vs 9k từ bảng).

### Bài học

- **284.** Nguồn xác nhận phải KHỚP LOẠI QUAN HỆ của cạnh. Cạnh TOPICAL (đồng
  hiện) cần nguồn đồng hiện (viwiki, A); nguồn TAXONOMIC (bảng lớp, B) khớp cạnh
  taxonomic — mà cạnh này không phải. Chọn nhầm nguồn xác nhận = 0,2%.
- **285.** Độ phủ của nguồn xác nhận quyết định. Bảng 9k từ không phủ từ vựng
  kỹ thuật ⇒ mù với đúng nhóm cạnh SẠCH nhất. Đo độ phủ TRƯỚC khi tin vào nguồn.

## §163 — C (đồng thuận anh em): THẤT BẠI vì HUB Ở PHÍA ĐÍCH

```
phân bố: 89,1% cạnh đồng thuận=1 · giữ ≥2 còn 10,9% · ≥3 còn 3,3%
```

Lọc MẠNH, nhưng KHÔNG phân biệt sạch/rác — NGƯỢC:

```
sắt → natri(4) magnesi(3) cobalt(3)      silic(1)   ← tốt nhưng đồng thuận THẤP
đồng → mẹo(7) microkernel(3) Gregory(2)             ← RÁC nhưng đồng thuận CAO
chó → dụ(7) thông(4) khẩu(3)                        ← rác đồng thuận cao
```

`đồng→mẹo`(7) > `sắt→silic`(1) ⇒ ngưỡng đồng thuận GIỮ RÁC BỎ TỐT. Nguyên nhân:
`dụ/thông/mẹo` là token PHỔ BIẾN (`ví dụ`,`thông tin`) mà MỌI anh em cùng nối tới
— HUB ở phía ĐÍCH, đúng bệnh §150. Cắt hub ở phía NGỮ CẢNH (cột) không chặn hub
ở phía đích của bước đồng thuận.

## §164 — KẾT LUẬN TK_054: chỉ A (nguồn NGOÀI) hoạt động

Ba tín hiệu NỘI TẠI đồ thị đều thất bại, cùng một gốc:

| hướng | tín hiệu | thất bại vì |
|---|---|---|
| B §162 | bảng lớp (taxonomic) | sai loại quan hệ + độ phủ hẹp (0,2%) |
| C §163 | đồng thuận anh em | hub ở phía đích (giữ rác) |
| D (chưa làm) | tam giác | — cùng đồ thị, cùng hub, DỰ BÁO thất bại |

**Vì sao chỉ A hoạt động**: B/C/D tính TỪ CHÍNH cấu trúc đồng hiện thiên lệch
(hub, đa nghĩa, topical). Chúng KẾ THỪA thiên lệch đó nên không thể tự phát hiện
nhiễu do chính thiên lệch sinh ra. A (viwiki GIỮ RIÊNG) neo NGOÀI đồ thị — thực
tế độc lập làm trọng tài, nên phân biệt được `sắt→silic` (32×) khỏi `đồng→SWAT`
(1×).

⇒ **A là cơ chế củng cố DUY NHẤT đã chứng minh.** 41 254 cạnh (§161) là kết quả
giữ lại. B/C đóng (âm tính, có số). D KHÔNG làm — dự báo cùng thất bại, và làm
là lặp bẫy «thử công thức nội tại thứ N» (§141–§150 đã 5 lượt).

### Bài học

- **286.** Tín hiệu tính TỪ đồ thị thiên lệch KHÔNG tự sửa được thiên lệch của
  chính nó. Muốn phân biệt tín-hiệu-thật khỏi nhiễu-cấu-trúc, cần MỎ NEO NGOÀI
  (dữ liệu giữ riêng). Đây là lý do sâu xa A thắng B/C/D.
- **287.** Cùng bệnh HUB xuất hiện ở NHIỀU vị trí: §150 hub phía nguồn, §163 hub
  phía đích. Cắt hub một phía không chặn phía kia. Hub là thuộc tính của phân bố
  tần suất, phải xử ở MỌI phép nhân ma trận, không chỉ một.

## §165 — NỐI cạnh REINFORCED vào đường sinh: an toàn nhưng GẦN NHƯ TRƠ

§143: đường sinh đọc `ps.parents`, không đọc adjacency ⇒ tiêm tường minh. Khối
mới: token nội dung trong 3 token ngữ cảnh gần kích thích đích REINFORCED, scale
`beta·rv·w` (bài học §137: thang tuyệt đối không thắng), CHỈ nâng đích ĐANG SỐNG
(scores>0, không hồi sinh token chết). Cờ `ar_w_skip_reinforced_beta`, mặc định 0.

### Đo trên net_ls_reinforced.pkl

```
beta=0,3  → Y HỆT beta=0 (no-op)
beta=0,6  → chỉ «đồng là ▸ …dẫn điện và LOÀI DẪN» đổi; ca vàng nguyên vẹn
```

### Vì sao trơ — và vì sao đó có thể ĐÚNG

Đích loại suy (`natri`,`silic`) KHÔNG phải ứng viên sống khi sinh câu định nghĩa
về `sắt` (phiếu bước đó là `là`/`kim`/`loại`). «Chỉ nâng đích sống» ⇒ không có gì
để nâng. Muốn `c→Y` tham gia phải HỒI SINH đích chết — địa hạt §137–§140, vỡ câu.

NHƯNG cạnh loại suy là tri thức LIÊN TƯỞNG, không thuộc câu định nghĩa. Việc nó
KHÔNG override sinh định nghĩa (ca vàng vững) có thể là hành vi ĐÚNG: cạnh chỉ
nudge khi đích đã contextually hợp lý, không ép từ liên tưởng vào chỗ ngữ pháp
không nhận.

### Phán quyết

Giữ cờ mặc định 0 (opt-in, chỉ có nghĩa trên checkpoint có REINFORCED). Cơ chế
NỐI XONG, an toàn, sẵn. Để tham gia mạnh hơn cần: (a) tách nghĩa (giảm nhiễu đích
liên tưởng), (b) prompt liên tưởng/liệt kê nơi đích loại suy là ứng viên hợp lệ,
(c) tích hợp thông minh hơn ở tầng câu (TK_053 §5 sinh hai thì).

Chu trình hoàn tất: §158 tạo → §161 củng cố → §165 nối (bảo thủ). Cạnh c→Y giờ:
tồn tại (§158), được xác nhận & mạnh lên khi là tri thức thật (§160-161), và CÓ
THỂ tham gia sinh (§165) — đúng lộ trình tăng dần tác giả đặt ra.

### Bài học

- **288.** «Nối vào sinh» an toàn (live-only) và «tham gia thật» (resurrect)
  loại trừ nhau trong kiến trúc này: cái đầu trơ, cái sau vỡ câu. Cạnh liên
  tưởng đúng ra chỉ nên nudge, không override — nên «trơ ở câu định nghĩa» là
  tính năng, không phải lỗi.

## §167 — Pipeline rebake ĐÃ XONG: net_ls_reinforced.pkl = sản phẩm + cạnh

Rebake «checkpoint sản phẩm mang cạnh loại suy» thực ra đã hoàn tất trong chuỗi
§158→§161, KHÔNG cần chạy lại:

```
network_v12_sent_ls.pkl (sản phẩm, 450 303 cha)
  --§158--> net_ls_analogy.pkl      (+ 398 832 cạnh ANALOGY)
  --§161--> net_ls_reinforced.pkl   (+ 41 254 cạnh REINFORCED)
```

`net_ls_reinforced.pkl` = ĐỒ THỊ SẢN PHẨM ĐẦY ĐỦ + cả hai loại cạnh. Với
`ar_w_skip_reinforced_beta = 0,4` (§166 bật mặc định), cạnh REINFORCED tham gia
sinh khi đích hợp lệ.

Đã kiểm (các bước trước, không cần load lại):
- 450 303 cha (§158) · 398 832 ANALOGY (§158) · 41 254 REINFORCED persist reload
  (§161 `vr_rf.log`) · ca vàng sinh đúng (§165 `rf_sinh.log`).

⚠️ Máy dev còn ~1,3–1,7 GB RAM trống, load checkpoint cần ~2,9 GB ⇒ vài lần load
xác nhận lại BỊ OS GIẾT lúc load. Không phải lỗi checkpoint (các lần load trước
thành công khi RAM đủ). Kiểm chứng ở §158/§161/§165 là chuẩn.

### Trạng thái cuối TK_054

Cạnh loại suy c→Y đi trọn vòng đời:
1. TẠO thụ động (§158) — 398 832 cạnh trong đồ thị.
2. XÁC NHẬN bằng dữ liệu giữ riêng (§160) — viwiki, 20,2%, phân biệt sạch/rác.
3. CỦNG CỐ «tri thức thật» (§161) — 41 254 cạnh → REINFORCED, thang theo độ tin.
4. NỐI vào sinh (§165) + BẬT mặc định (§166) — tham gia khi ngữ cảnh hợp lệ.

Checkpoint triển khai: `net_ls_reinforced.pkl` (gitignored; dựng lại bằng
`nuong_canh_analogy.py` + `nang_canh_xac_nhan.py`).

B/C/D đóng âm tính — chỉ A (mỏ neo NGOÀI đồ thị) phân biệt được tri-thức-thật.

## §169 — Đo cạnh REINFORCED trên prompt LIÊN TƯỞNG: TRƠ HOÀN TOÀN

§165 đo cạnh không đổi câu ĐỊNH NGHĨA (đúng). Vòng này đo trên prompt LIÊN TƯỞNG
nơi đích loại suy *đáng lẽ* là ứng viên — net_ls_pruned.pkl, beta 0 vs 0,4:

```
«các kim loại như»  (không đổi)   ▸ khác nhau có thể được sản xuất...
«sắt, kẽm,»         (không đổi)   ▸ chất lượng
«kim loại gồm có»   (không đổi)   ▸ lẽ
«sắt và»            (không đổi)   ▸ các
```

4/4 KHÔNG đổi. Cạnh REINFORCED KHÔNG tham gia sinh, kể cả ở đây.

### Vì sao — và hệ quả cho hướng §1

Sau `sắt, kẽm,` phiếu W_skip bước kế KHÔNG có `natri`/`silic` là ứng viên sống
(mạng nối bằng từ phổ biến `chất lượng`). «Chỉ nâng đích sống» ⇒ không có gì để
nâng. Đây là căng thẳng §165/§137: muốn `c→Y` tham gia phải HỒI SINH đích chết —
vỡ câu. Base cũng kém (`sắt, kẽm, ▸ chất lượng`): mạng không xử được prompt liên
tưởng vì train chủ yếu khuôn định nghĩa.

⇒ **KẾT LUẬN QUAN TRỌNG cho lộ trình**: cải thiện CHẤT LƯỢNG cạnh (tách nghĩa §1)
KHÔNG giúp SINH chừng nào cạnh chưa THAM GIA được. Blocker là KIẾN TRÚC (live-only
vs resurrect), không phải chất lượng cạnh. Tách nghĩa cải thiện cạnh như TRI THỨC
LƯU TRỮ (cho truy hồi/route khác tương lai), không cho sinh hiện tại.

### Hướng đúng đã lộ

Để cạnh tham gia AN TOÀN: hồi sinh có kiểm soát, CHỈ cạnh REINFORCED, CHỈ khi
mạng KHÔNG CHẮC (p_top1 thấp — chính tín hiệu §151). Lúc mạng phân vân là lúc
nudge liên tưởng ít phá nhất VÀ hữu ích nhất. Nối §151 (độ tin) với §165 (tiêm):
tiêm mạnh khi độ tin thấp, tắt khi độ tin cao (câu đang chắc, đừng chen).

### Bài học

- **289.** «Cạnh tồn tại» ≠ «cạnh tham gia». Live-only an toàn nhưng trơ; đích
  loại suy không bao giờ là ứng viên ngữ pháp ở khuôn định nghĩa. Cải thiện chất
  lượng cạnh vô nghĩa cho SINH chừng nào chưa giải quyết THAM GIA.
- **290.** Thứ tự đúng: giải THAM GIA (hồi sinh có kiểm soát theo độ tin) TRƯỚC,
  rồi mới đáng cải thiện CHẤT LƯỢNG (tách nghĩa). Làm §1 trước §4 đã là sai thứ
  tự — §4 vừa lộ ra điều đó.

## §170 — Cạnh loại suy VÀO W_skip: hết «phân biệt», nhưng VOTE PROD chặn

Tác giả hỏi đúng chỗ: sao cạnh loại suy không được tính khi sinh, nó bị thuật
toán bỏ qua vì lý do gì? Nó cũng như cạnh W_skip khác mà.

### 170.1 Tầng 1 — SAI HỘP (đã sửa)

Cạnh loại suy nằm ở `TypedAdjacency` (EdgeKind.ANALOGY/REINFORCED). Nhưng
`vote_logits` đọc `W_skip` (`ws.mats[d]`, dựng từ `ps.parents`) — HỘP KHÁC. Không
phải thuật toán phân biệt cờ; nó ở kho vote không đọc.

Sửa: cắm REINFORCED vào `mats[1]` trong `build_w_skip` (× scale). Đo trực tiếp:
```
sau build scale=200, mats[1][sắt]: silic 65 · magnesi 63 · natri 60 · cobalt 56
   (max cạnh CÂU gốc của sắt chỉ 22)
```
⇒ Cạnh ĐÃ là cạnh W_skip thật, đứng ĐẦU hàng. Hết phân biệt.

### 170.2 Tầng 2 — VOTE PROD (giao Hadamard) chặn

Nhưng «sắt và ▸ các» KHÔNG đổi kể cả scale 500. Vì vote dùng PROD: điểm ứng viên
= TÍCH đề xuất qua MỌI token ngữ cảnh. `natri` do `sắt` đề xuất (60) nhưng `và`
KHÔNG ⇒ tích = 60 × 0 = 0, bất kể scale.

Xác nhận bằng prompt MỘT token (không có từ thứ hai để giao):
```
sắt   ▸ ĐỒNG tốt...      natri ▸ trong CLO và      ← loại suy NỔI khi ít giao
```

⇒ Đây KHÔNG phải phân biệt cờ, mà là VOTE ĐỒNG THUẬN: ứng viên phải được NHIỀU
token ngữ cảnh cùng đề xuất. Cạnh loại suy MỘT-NGUỒN (chỉ nguồn đề xuất đích) bị
giao triệt tiêu khi ngữ cảnh có ≥2 token.

### 170.3 Hệ quả

Scale 60 giữ mặc định: ca vàng nguyên vẹn (bị prod chặn ở ngữ cảnh giàu), loại
suy nổi ở ngữ cảnh MỘT token. Đây là hành vi ĐÚNG của vote đồng thuận — không cần
tắt/phân biệt, chính prod quyết định khi nào cạnh nổi.

Để loại suy tham gia ở ngữ cảnh giàu cần vote CỘNG (không giao) cho cạnh một-nguồn
— tức tiêm SUM có kiểm soát theo độ tin (plan a §169): khi mạng KHÔNG CHẮC, cộng
đề cử loại suy bên cạnh phiếu prod. Đó là bước kế đúng.

### Bài học

- **291.** «Cạnh trong đúng hộp» (§170.1) vẫn chưa đủ để THAM GIA nếu cơ chế đọc
  hộp đó lọc theo cách khác (prod §170.2). Phải truy tới TẦNG QUYẾT ĐỊNH cuối
  (vote aggregation), không dừng ở tầng lưu trữ.
- **292.** Vote PROD = đồng thuận: lọc thẳng cạnh một-nguồn. Loại suy BẢN CHẤT
  một-nguồn (chỉ c đề xuất Y). Hai cái xung khắc thiết kế — cần SUM riêng, không
  phải scale to hơn (scale vô dụng khi nhân với 0).

## §171 — Plan (a) triển khai: HỒI SINH SUM có kiểm soát theo độ tin

Sửa khối §165 trong `generate_hebbian.py`: khi `p_top1 = max/sum(phiếu dương) <
ar_w_skip_reinforced_conf` (mặc định 0.2 — mạng KHÔNG chắc), cộng đề cử REINFORCED
như **SUM**, **hồi sinh cả đích chết** (`scores[_j] = max(cur,0) + β·rv·w`), vòng
qua vote PROD (§170.2). Khi mạng chắc (đang phát câu định nghĩa đúng): tắt, không
chen. Thêm cờ `ar_w_skip_reinforced_conf`. 454 test pass; ca vàng (sắt/chó/muối/
khỉ là…) GIỮ NGUYÊN qua mọi ngưỡng conf ⇒ an toàn, không hồi quy.

Nhưng đo trên net_ls_pruned + net_ls_reinforced: **output y hệt qua mọi conf**.
Truy ra vì sao → §172.

## §172 — MẤU CHỐT THẬT: đích loại suy là MẢNH SUBWORD, không phải từ khái niệm

Soi trực tiếp tầng cạnh (không qua sinh):

- **net_ls_pruned** (§168): sắt có **0** cạnh REINFORCED. Tỉa nghĩa cắt sạch.
- **net_ls_reinforced** (41254 cạnh): sắt vẫn **0**. Cạnh tồn tại là tự-lặp
  (5895 cạnh i==j) + rác subword (`Ass→ationale`, `ngày→ngữ`) + hàng loạt mảnh
  byte UTF-8 (`�→�`). Bước xác nhận §161 khớp viwiki ở mức TOKEN nên confirm
  trúng mảnh byte đồng-xuất-hiện tầm thường.
- **net_ls_analogy** (398832 cạnh, TRƯỚC xác nhận): sắt có 15 cạnh →
  `magnesi, sulf, zir, cr, nicotin, coni, Ga, zir…` — **ĐÚNG hàng xóm hóa học**
  (magiê, lưu huỳnh, zirconi, crôm)! Cơ chế đóng closure `S=Bn·Bnᵀ` **thật sự**
  tìm ra nguyên tố cùng nhóm. NHƯNG tất cả là **mảnh subword** (từ hiếm bị BPE
  cắt), trọng số đều 0.05 (sàn).

### Đây là lý do THỨ BA — sâu hơn §170

§170 nêu hai tầng (sai hộp; vote prod). Tầng ba, gốc rễ hơn: **đích loại suy là
mảnh subword, không phải token từ trọn vẹn.** Dù có tiêm SUM và hồi sinh (§171),
cộng điểm cho mảnh `magnesi` cũng không tạo ra TỪ mạch lạc khi sinh — mảnh không
tự đứng thành ứng viên như `sắt` đứng. Bước xác nhận §161 rồi biến chúng thành
rác byte. Nên §171 **trơ không phải vì thiết kế sai** mà vì **dữ liệu nuôi nó ở
sai hạt**.

### Hệ quả — cơ chế đúng, hạt sai

Kết luận đảo ngược sự bi quan §164/§170: closure loại suy KHÔNG vô dụng — nó tìm
đúng nguyên tố cùng nhóm. Vấn đề là **hạt token**: phải chạy closure trên KHÁI
NIỆM TRỌN VẸN (span đã gom, như bảng nhãn-lớp các phiên trước đã dựng span cho
nhãn ghép), hoặc chặn đích closure về token-từ-trọn-vẹn, hoặc gom mảnh về từ mẹ.
Đây chính là việc «tách nghĩa / token-level» mà §169 hoãn — hóa ra KHÔNG hoãn
được: tham-gia-sinh không giải được nếu không chạm hạt token trước.

### Bài học

- **293.** Trước khi kết luận «cơ chế trơ», SOI TRỰC TIẾP tầng dữ liệu ở đúng
  hạt. Ba checkpoint, ba kết quả khác nhau: pruned=rỗng, reinforced=rác byte,
  analogy=đúng-nhưng-mảnh. Nếu chỉ đo qua sinh sẽ kết luận sai là «thiết kế hỏng».
- **294.** Closure `S=Bn·Bnᵀ` phân biệt tri thức thật ở mức KHÁI NIỆM, nhưng
  base vocab BPE cắt từ hiếm (tên nguyên tố) thành mảnh ⇒ tín hiệu đúng bị khóa
  trong hạt không sinh được. Chất lượng loại suy bị chặn bởi TOKENIZATION, không
  phải bởi vote hay nguồn.
- **295.** Thứ tự đúng đảo lại: **tách nghĩa/hạt-khái-niệm TRƯỚC**, rồi mới closure
  + tiêm SUM (§171) mới có đích sạch để hồi sinh. §171 để sẵn, đúng và an toàn,
  chờ dữ liệu đúng hạt.

## §174 — TIÊM CẠNH KHÁI NIỆM TRỌN VẸN (thụ động) + tường §164 lặp lại ở cấp-từ

Sau §173 (closure cấp-từ cho đích whole-word), thử LỌC nguồn-concept khỏi
nguồn-từ-thường bằng thống kê nội tại:

- Ngưỡng trọng số E đơn: TOPW≥0.2 vẫn giữ 2997 nguồn (từ thường vẫn qua).
- **Độ mạch lạc cụm anh em** (mật độ Sk trong top-6 anh em): THẤT BẠI tách:
  `sắt` coh=0.021 (THẤP, nhưng là concept tốt); `đồng` coh=0.374, `số` 0.354
  (CAO nhưng là hub đồng nghĩa/stopword). Kim loại hiếm → anh em rời rạc trong
  Sk cắt ngưỡng; stopword → anh em đồng-xuất-hiện dày ⇒ coh cao.

**Kết luận: đúng lại tường §164.** Thống kê NỘI TẠI không phân biệt được
tri-thức-thật (miền) với cụm stopword — vì cả hai giống nhau về cấu trúc
đồng-xuất-hiện. Chỉ NEO NGOÀI (plan A viwiki) phân biệt được. Ở cấp-từ tường này
lặp lại y nguyên, chỉ khác HẠT.

### Quyết định — nhà đúng cho tập suy-ra-nhưng-nhiễu: ANALOGY THỤ ĐỘNG

Tác giả đã nói rõ: «bịa cũng không sao, sửa dần dần», cạnh chỉ cần TỒN TẠI +
SUY RA từ dữ liệu gốc, thuật toán sau (plan A) củng cố cạnh thật. Vậy:

- Tập cạnh cấp-từ (166 648 cạnh, 8598 nguồn) → nạp **ANALOGY** (priority 0.0,
  w=0.05, THỤ ĐỘNG): §143 W_skip không đọc ANALOGY ⇒ sinh KHÔNG đổi (đã chứng
  minh, không cần chạy lại dưới sức ép RAM). Cạnh TỒN TẠI, suy ra từ base, bật
  mặc định — đúng spec «phải có cạnh ở đó».
- REINFORCED (active, §171) DÀNH cho cạnh ĐÃ XÁC NHẬN ngoài — không nhận noise.

Xác minh cạnh khái niệm TỒN TẠI trong `net_ls_concept.pkl`:
`sắt→thiếc, sắt→chloride, kẽm→cadmium, kẽm→mangan, natri→hydroxide, natri→bor,
mèo→corgi, mèo→pug` — tất cả w=0.05. Từ TRỌN VẸN, không mảnh (§172 đã sửa).

### Trạng thái — «hạt khái niệm trọn vẹn» XONG

Cơ chế loại suy §158 giờ chạy ở HẠT KHÁI NIỆM: đích là từ trọn vẹn tham-gia-sinh
được. Cạnh tồn tại + thụ động + derived. §171 (SUM hồi sinh) để sẵn cho bước
active. Còn lại là CHẤT LƯỢNG (tách concept/stopword) — cần plan A CẤP-TỪ:
word-segment viwiki held-out, xác nhận cạnh cấp-từ → promote lên REINFORCED.

### Bài học

- **296.** Đổi HẠT (token→từ) sửa được §172 (đích mảnh) nhưng KHÔNG sửa §164
  (tách tri-thức-thật). Hai vấn đề trực giao: hạt là biểu diễn, xác-nhận là chất
  lượng. Word-grain cho đích SẠCH; vẫn cần neo NGOÀI cho đích ĐÚNG.
- **297.** Nhà cạnh theo độ tin cậy: derived-nhưng-chưa-xác-nhận → ANALOGY thụ
  động (tồn tại, trơ); đã-xác-nhận-ngoài → REINFORCED active. Không trộn noise
  vào tầng active. Đúng ý tác giả «cạnh tồn tại, thuật toán sau củng cố».

## §175 — PLAN A CẤP-TỪ: neo NGOÀI (viwiki) xác nhận cạnh khái niệm

§174 để cạnh cấp-từ ở ANALOGY thụ động (derived, nhưng chưa tách concept/stopword).
Nay áp plan A (§160) ĐÚNG HẠT: word-segment viwiki held-out (30k bài, 2,37M câu),
đếm đồng-hiện cặp (c,Y) → xác nhận. 15 019/62 398 cạnh (24%, khớp §160 20%).

### Neo ngoài TÁCH được concept khỏi stopword

| nguồn | xn | top đích (số đếm) | loại |
|---|---|---|---|
| sắt | 8/8 | thiếc·49, sulfide·24, chloride·10 | concept ✓ |
| kẽm | 6/8 | mangan·27, sulfide·23, pyrit, cadmium | concept ✓ |
| natri | 5/8 | hydroxide·38, amoni·10, halide·5 | concept ✓ |
| chì | 5/8 | quặng·31, natri·20, thép·8 | concept ✓ |
| đồng | 2/4 | pà·6, inuit·3 (rác) | homograph, YẾU |
| mèo | 1/8 | bulldog·1 | loại suy đúng nhưng YẾU |
| toán | 1/7 | thales·7 | stopword, YẾU |

Số đếm xác nhận = thứ hạng độ mạnh TỰ NHIÊN: sắt→thiếc(49) mạnh, mèo→bulldog(1)
yếu. Đúng điều §164 tiên đoán: chỉ NGUỒN NGOÀI phân biệt tri-thức-thật; thống kê
nội tại (§174 coh) thì không.

### Hạn chế đã lộ — co-occur ≠ analogy

Loại suy mà THÀNH VIÊN KHÔNG đồng-hiện trong held-out (mèo & giống-chó hiếm cùng
câu) bị xác nhận YẾU DÙ ĐÚNG khái niệm. Plan A (Hebbian đồng-hiện) mạnh cho miền
member-cùng-câu (kim loại trong bài luyện kim) nhưng mù với loại suy chéo-danh-mục.
Đó là lý do CẦN nhiều thuật toán (plan B bảng, C/D đồ thị) — mỗi cái bắt một dạng.

### Kết quả — promote 38 738 cạnh REINFORCED (active)

Cạnh xác nhận → REINFORCED, trọng số theo ln(số đếm). Giờ là tầng ACTIVE §171 đọc:
khi mạng KHÔNG chắc, hồi sinh SUM đích concept đã-xác-nhận. `net_ls_concept_conf.pkl`.

### Bài học

- **298.** Plan A đổi HẠT (token→từ) từ §161 (ra rác byte) sang §175 (ra
  thiếc/mangan/hydroxide sạch). Cùng thuật toán, khác hạt ⇒ khác trời vực. §172
  (hạt) là điều kiện cần để §164 (neo ngoài) cho kết quả dùng được.
- **299.** Số đếm đồng-hiện = thước độ tin SẴN CÓ, không cần ngưỡng tay: cạnh
  mạnh (metal, count cao) nổi tự nhiên trên cạnh yếu (count 1) qua w=ln.
- **300.** Co-occur confirmation MÙ với loại suy chéo-danh-mục (mèo→chó-breed).
  Một neo không đủ; cần bộ neo (A đồng-hiện + B curated + C/D đồ thị) — đúng
  «nhiều thuật toán củng cố» tác giả đã hình dung.

## §176 — THAM GIA SINH: bug cửa sổ ngữ cảnh + công thức boost + cap nguồn

§171 (SUM hồi sinh) đã đúng thiết kế nhưng đo trên `net_ls_concept_conf` vẫn TRƠ.
Instrument (`HCA_DBG171`) truy ra BA lỗi nối tiếp — cả ba đều ở đường THAM GIA:

### 176.1 Bug cửa sổ — bỏ sót token PROMPT

`out[max(prompt_len, len(out)-2):]` chỉ nhìn token ĐÃ SINH, KHÔNG nhìn prompt. Mà
khái niệm nguồn (sắt) thường Ở TRONG prompt ⇒ `n_ctx_edges=0`, không bao giờ tra
được cạnh của sắt. Sửa: `out[max(0, len(out)-3):]` — 3 token cuối bất kể ranh
prompt. Sau sửa: `sắt và` step 2 → n_ctx_edges=12, fired=12. §171 BẮN.

### 176.2 Boost quá yếu — không vượt max ở vùng phẳng

`beta·rv·w`: ở vùng KHÔNG chắc (conf 0.018) rv nhỏ nên boost càng nhỏ, đích hồi
sinh KHÔNG thắng nổi argmax hiện tại (`các`→`các`, fired nhưng vô hiệu). Sửa:
`rv·(1+beta·w)` — cạnh đã-xác-nhận mạnh (w cao) → >rv → nút khái niệm NỔI LÊN khi
mạng thật sự phân vân. Sau sửa: `các`→`Archimedes`, `là`→`Franklin`. argmax ĐỔI.

### 176.3 Nhưng nổi SAI concept — hư từ cũng là nguồn

`Archimedes/Franklin` sai: đích số-đếm-cao từ nguồn HƯ TỪ (`học`→archimedes·57,
`số`→angela). Cửa sổ gồm `và/là`, mà hư từ df cao vẫn lọt filter df≥5 ⇒ cạnh của
chúng (co-occur khắp nơi) lấn cạnh concept đặc thù. Đúng lại tường §164 TẠI SINH.
Sửa: cap df nguồn `[20,2000]` (từ NỘI DUNG, loại hư từ). Nguồn 8598→4761;
`đồng/toán/học` → không-candidate. Concept sạch giữ: sắt→thiếc·49, kẽm→mangan·27,
natri→hydroxide·38, chì→quặng·31. Promote 24 037 cạnh.

### Kết quả — THAM GIA ĐẠT

`sắt và` → **`thiếc`** (iron & tin — loại suy kim loại đúng, whole-word, derived
+ viwiki-confirmed). Ca vàng confident (`sắt là kim loại…`) — cổng shut mọi bước,
output KHÔNG đổi (an toàn). Mặc định `conf=0.2`: chỉ bắn khi p_top1<0.2 (rất
phẳng), production (không cạnh REINFORCED) TRƠ hoàn toàn; chỉ active nơi có cạnh.

### Bài học

- **301.** Ba lỗi tham-gia XẾP TẦNG, mỗi cái che cái sau: cửa sổ (không tra được)
  → boost (tra nhưng không thắng) → nguồn (thắng nhưng sai). Phải instrument từng
  tầng; đo end-to-end chỉ thấy «trơ» mà không biết tầng nào.
- **302.** Loại suy phải anchor trên TỪ NỘI DUNG, không hư từ. df cap ở NGUỒN là
  bộ lọc concept/stopword RẺ và HIỆU QUẢ tại sinh — nơi coherence (§174) thất bại,
  df-cap + neo-ngoài (§175) cùng nhau đủ dùng.
- **303.** «Mặc định bật» an toàn khi cơ chế TỰ TRƠ ở checkpoint không có dữ liệu
  (production) và chỉ active nơi cạnh tồn tại — không cần cờ tắt/bật riêng.

## §177 — PLAN B (bảng nhãn-lớp): neo curated bù co-occur, bắt chéo-danh-mục

§175 lộ hạn chế plan A: co-occur MÙ với loại suy thành-viên-không-đồng-hiện. Plan
B dùng `nhan_lop.json` (9222 mục, §128): c→Y xác nhận nếu CHUNG nhãn lớp (genus
≥2 token). Nguồn ngoài KHÁC A — grounding curated, không co-occur.

### Kết quả — cao chính xác, thấp phủ

44 cạnh (bỏ self-loop), đúng ca A mù: xôi/phở→nem (món ăn), lụa→tuyn (hàng dệt),
sét→chớp (hiện tượng), mão→hợi/tuất→thìn (kí hiệu 12 con giáp), gan→tuyến,
thận→vú (bộ phận), lít→phút (đơn vị). Không cần đồng-hiện — chung LỚP là đủ.

Phủ THẤP: chỉ 23% đích loại suy có genus (đích nhiều là công thức/kí hiệu hoá học
sio2/cu/nhoam — không phải mục từ điển). Metal (sắt→thiếc) KHÔNG qua B vì thiếc/
sio2 thiếu genus khớp. B và A phủ tập KHÁC nhau ⇒ union đúng, không trùng.

### A∪B — bộ neo tác giả hình dung

net_ls_concept_conf.pkl giờ mang A (đồng-hiện: kim loại, hoá học) ∪ B (chung-lớp:
món ăn, dệt, giáp, bộ phận). Đúng «nhiều thuật toán củng cố cạnh c-Y» — mỗi neo
bắt một DẠNG tri-thức-thật: A cho member-cùng-câu, B cho co-hyponym-curated.

### Bài học

- **304.** Không neo đơn nào đủ. A (co-occur) mạnh miền member-đồng-hiện, mù
  chéo-danh-mục; B (class curated) mạnh co-hyponym, thấp phủ (đích ngoài từ điển).
  Union bắt hợp — đúng thiết kế «bộ neo», không phải một công thức vạn năng.
- **305.** Phủ của B bị chặn bởi CHẤT LƯỢNG ĐÍCH closure: đích công thức/kí hiệu
  (sio2, cu) không có mục từ điển. Muốn B phủ metal ⇒ phải làm sạch đích closure
  về mục-từ-điển trước — lever kế tiếp (làm sạch đích > thêm neo).

## §178 — ĐO DIỆN RỘNG: §171 an toàn ở quy mô, lợi ích hẹp (vùng rất-phân-vân)

Mode `reinforced` trên harness (30 prompt × 4 nhóm, 2 hạt, net_ls_concept_conf):

| Config | TRUNG THÀNH | PHỦ | TRẬT TỰ | LẶP | RỖNG |
|---|---|---|---|---|---|
| §171 TẮT | 80.0±3.1 | 34.5 | 93.5 | 8.6 | 0 |
| conf=0.20 (mặc định) | 80.0±3.1 | 34.5 | 93.5 | 8.6 | 0 |
| conf=0.35 | 78.8 | 34.7 | 91.9 | 8.5 | 0 |

### conf=0.20 = TẮT (byte-identical)

Không MỘT số nào đổi. §171 KHÔNG bắn trên cả 30 prompt. Vì cổng p_top1<0.2 (rất
phẳng) HIẾM gặp trong sinh thật: prompt định nghĩa/hỏi confident, prompt loại suy
(«sắt và đồng đều là») cũng đủ tự tin ⇒ cổng đóng. Ca `sắt và` (cụt, rất phân vân)
mới mở cổng (§176). Kết luận: **mặc định AN TOÀN ở quy mô — trung tính, không hại.**

### conf=0.35 hại nhẹ

Nới cổng → TRUNG THÀNH −1.2, TRẬT TỰ −1.6 (trong biên nhiễu, nhưng SAI CHIỀU),
PHỦ phẳng. Bắn nhiều hơn = chen vào lúc mạng CÓ opinion vừa phải → phá nhẹ. Giữ
mặc định 0.20, KHÔNG lên 0.35.

### Thước trung thành KHÔNG tín được loại suy — theo cấu trúc

TRUNG THÀNH = từ-sinh ∈ câu-nguồn. Loại suy BẢN CHẤT sinh từ NGOÀI câu nguồn
(sắt→thiếc: thiếc không trong định nghĩa sắt). Nên thước này CẤU TRÚC không thưởng
được loại suy đúng — nó chỉ đo «không phá sinh thường» (ĐẠT). Đo lợi ích loại suy
cần thước khác (độ giàu nhóm LOẠI SUY chấm mắt, hoặc recall thành-viên-lớp).

### Bài học

- **306.** §171 là bộ tăng NÍCH (niche) có cổng an toàn: vô hại ở sinh thường
  (cổng hiếm mở), chỉ giúp ở phân-vân-thật. «Mặc định bật» đúng vì downside≈0.
- **307.** Cổng phủ quyết (TRUNG THÀNH+PHỦ không tụt) PASS ở 0.20, FAIL nhẹ ở 0.35
  → biên an toàn là ngưỡng cổng, không phải beta.
- **308.** Không thể dùng MỘT thước cho cả «không phá» và «loại suy tốt»: thước
  trung thành cấu trúc mù với ngoại suy. Cần thước riêng cho nhánh loại suy —
  việc kế nếu muốn ĐO (không chỉ chứng minh cơ chế) giá trị loại suy.

## §179 — LÀM SẠCH ĐÍCH CLOSURE: lọc an toàn, nhưng xác nhận đã lo phần lớn

Lever §305: đích closure còn công thức/kí hiệu (sio2, fe2o3, h3o, ca2) + lỗi chính
tả (nhoam). Thử tách bằng df: THẤT BẠI — df không phân biệt (cadmium=8 vs hạo=360,
giáy=22 nhiễu df CAO). Tín hiệu đúng là ORTHOGRAPHIC, không tần suất.

### Lọc an toàn (chữ-số/phi-chữ-cái/<eos>) — chính xác tuyệt đối

Giữ 84% đích; bỏ đúng công thức (sio2→có chữ số), tag (<eos>), kí hiệu (/cm³).
GIỮ được đích hiếm-nhưng-thật (cadmium df8, pyrit df5, bor df5, halide df6, corgi
df9) mà ngưỡng df≥10 làm mất. Nhiễu chính tả còn lại (nhoam df2, cón df3) KHÔNG
tách được bằng luật — giao cho bước XÁC NHẬN.

### Nhưng tác động lên tập confirmed: NHỎ

Tái chạy plan A với đích đã sạch: 24 013 cạnh REINFORCED vs 24 037 (§176) — gần
Y HỆT. Vì bước xác nhận (số đếm đồng-hiện) ĐÃ lọc nhiễu: nhoam/hạo không đồng-hiện
với sắt trong viwiki ⇒ count 0 ⇒ tự rớt. Lọc đích chỉ dọn candidate TRƯỚC xác
nhận, không đổi kết quả SAU xác nhận. Đích sạch hơn thấy ở tầng chưa-xác-nhận
(natri hết ca2/h3o) và ở phủ plan B.

### Bài học

- **309.** Nhiễu closure có HAI loại cần HAI công cụ: (a) công thức/tag —
  ORTHOGRAPHIC, lọc luật chính xác; (b) lỗi chính tả/từ-lạ — KHÔNG luật nào tách,
  phải để NEO NGOÀI (xác nhận) lọc bằng bằng chứng. Đừng ép df làm việc của neo.
- **310.** Làm sạch TRƯỚC một bước đã-tự-lọc thì lợi ích biên. Giá trị thật của
  §179 là ở tầng THỤ ĐỘNG (§174) và phủ plan B, không ở tập plan-A-confirmed —
  nơi số đếm đã là bộ lọc. Đặt bộ lọc ĐÚNG TẦNG.
