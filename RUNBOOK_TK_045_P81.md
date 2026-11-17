# RUNBOOK TK_045 P81 — cổng đòi **FOLD ĐẶC TRƯNG**: đo tuyệt đối, trộn chung (§13.48)

Runbook tự đủ: mục 1 giao M4, mục 2 giao giám khảo.

## 0. Vì sao

§13.43 (mốc gốc, đã **tái lập** ở §13.46): **0,5% dùng được**, 91% hỏng. §13.46: bệnh chính là
*"bám vài chữ cuối của đề rồi **nhảy sang miền văn bản hoàn toàn khác**"* — ~3/4 mẫu hỏng có
chữ. §13.47 đo miền và tìm ra chỗ hổng:

| ngưỡng fold | SINH | THẬT (hai cha liên tiếp cùng episode) | NGẪU NHIÊN | SINH ở đâu |
|---|---:|---:|---:|---:|
| 4,0 (production) | 0,0468 | 0,0723 | 0,0104 | 59% |
| **7,0 (đặc trưng)** | **0,0055** | **0,0562** | 0,0013 | **8%** |

**96,2%** mối nối sinh **không chia sẻ một fold đặc trưng nào** (văn thật 81,2%). Cái "trùng
fold" mà cổng đang ép là **trùng RÁC** (`năm`, chữ số — §13.20). Nhưng miền **là** đại lượng
đếm được: văn thật gấp **10×** bản sinh, **43×** sàn ngẫu nhiên.

⇒ Ứng viên: cổng phủ đòi **≥1 fold ĐẶC TRƯNG** chung với đề, thay ngưỡng 4,0.

## 1. Sinh mẫu — **giao M4**

```bash
PYTHONPATH=src .venv/bin/python scripts/gen_absolute_samples.py \
    --checkpoint checkpoints/network_v12_sent.pkl --n 200 --theme-gidf 5.0 \
    --out out/abs3_samples_v12.txt --out-key out/abs3_key_v12.json \
    --out-meta out/abs3_meta_v12.json
```

Sinh **cả hai** cấu hình trên **cùng 200 đề** (400 mẫu), trộn, giấu nhãn. Chấm **một** phiên —
§13.46 đã chứng minh thước tuyệt đối **tái lập được** giữa hai phiên, nhưng chỉ khi rubric
giống hệt; trộn chung là cách chắc chắn nhất.

### 1.1 Vì sao ngưỡng 5,0 — quy tắc chọn nêu TRƯỚC, **không nhìn chất lượng**

Quét trên mini (600 đề), chọn **ngưỡng CAO NHẤT mà hệ vẫn sinh cho ≥50% số đề**:

| `gidf_min` | 4,0 | 4,5 | **5,0** | 5,5 | 6,0 | 7,0 |
|---|---:|---:|---:|---:|---:|---:|
| trống | 20,7% | 29,0% | **45,3%** | 64,0% | 82,2% | 94,8% |

⇒ **5,0**. Quy tắc chỉ dùng **tỷ lệ trống**, một đại lượng **không phải** biến kết quả — nên
đây không phải chọn tham số theo kết quả.

⚠️ §13.32: **mini không định cỡ được**. Tỷ lệ trống ở v12 sẽ khác; cổng dưới đây tính cả điều đó.

### 1.2 CỔNG G-abs3 (nêu TRƯỚC)

| # | điều kiện | nếu trượt |
|---|---|---|
| 1 | **400 mẫu**, khoá chia đúng **200/200** | script có bug — dừng, báo |
| 2 | trống của `fold_dac_trung` ∈ **25–75%** | ngoài dải ⇒ v12 khác mini về bản chất, **báo trước khi chấm**, đừng tự đổi ngưỡng |
| 3 | trống của `hien_tai` ≈ **29,5%** (±5) | phải khớp §13.43, nếu không thì mẫu đã lệch |

### 1.3 Báo cáo M4

Commit ba file, kèm: trống **theo từng nhánh** · mệnh đề TB theo nhánh · thời gian.
**Không chấm.**

## 2. Giao chấm — phiên Claude RIÊNG

⛔ Cấm đọc: `out/abs3_key_v12.json`, mọi `out/*_key*`, `out/*verdict*`, `out/*meta*`,
`TK_045_MATRIX_CORE.md`, mọi `RUNBOOK_*`. File được đọc: **chỉ** `out/abs3_samples_v12.txt`.

### 2.1 Prompt giao giám khảo (copy nguyên khối)

> Trong `out/abs3_samples_v12.txt` có 400 mẫu (A000–A399), mỗi mẫu gồm một **ĐỀ** và phần
> **tiếp theo do máy sinh** (dòng `→`). Chấm **từng mẫu ĐỘC LẬP** — không so mẫu này với mẫu
> khác, không xếp hạng.
>
> Ba mức:
> - **`dung_duoc`** — phần sinh nối vào đề mà đọc **trôi**, **đúng ngữ cảnh**, **không có câu
>   vô nghĩa**. Tiêu chuẩn: gặp đoạn này trong bài viết thật, bạn **không khựng lại**. Không
>   cần hay, chỉ cần **không sai**.
> - **`tam`** — có ý đúng đề, đọc được phần nào, nhưng vấp: cụt giữa câu, một cụm lạc, ghép
>   gượng. Sửa vài chữ là dùng được.
> - **`hong`** — salad, lạc đề, tự mâu thuẫn, hoặc chuỗi số/mảnh vụn vô nghĩa.
>
> ⚠️ Mẫu `(TRỐNG — hệ không sinh được gì)` chấm là **`hong`**.
>
> ⚠️ **Đừng nới tay.** Phân vân giữa `dung_duoc` và `tam` ⇒ chọn **`tam`**. Phân vân giữa `tam`
> và `hong` ⇒ chọn **`hong`**.
>
> ⚠️ **Độ dài không phải tiêu chí.** Phần sinh rất ngắn là **thiết kế**; chỉ phạt nếu nó **đứt
> ngang giữa câu**.
>
> ⚠️ **QUAN TRỌNG NHẤT:** file có **hai loại mẫu trộn lẫn** do hai cấu hình sinh ra. Bạn
> **không biết và không cần biết** mẫu nào thuộc loại nào. **Đừng đoán, đừng nhóm chúng lại,
> đừng đổi thước giữa chừng.** Chấm tuần tự A000→A399 bằng **cùng một** thước. Giữ thước đều
> là việc quan trọng nhất của vòng này.
>
> ⛔ Không mở `out/abs3_key_v12.json`, không mở `*_key*`/`*verdict*`/`*meta*` nào, không mở
> `TK_045_MATRIX_CORE.md` hay `RUNBOOK_*`.
>
> Ghi `out/abs3_verdict_v12.json`:
> `{"grades": {"A000": "hong", ...}, "notes": "6-10 câu"}` — đủ **400 mã**.
>
> Trong `notes`: (1) mẫu `dung_duoc` giống nhau ở điểm gì; (2) kiểu hỏng phổ biến nhất và tỷ
> trọng; (3) có mẫu nào **chỉ thiếu một chút** không, thiếu gì; (4) **có cảm giác đầu ra rơi
> vào hai nhóm chất lượng khác nhau không** — nếu có thì khác nhau ở điểm gì (đừng đoán nhãn,
> chỉ mô tả).

## 3. Lật khoá & kết luận

Đếm ba mức **theo từng cấu hình**, trên mẫu số 200 mỗi bên (gồm mẫu trống), **và** riêng nhóm
không trống.

### Dự đoán ghi TRƯỚC (Claude)

- `hien_tai`: `dung_duoc` **0,5%**, `tam` **8,5%** — phải **tái lập** §13.43/§13.46. Nếu lệch
  nhiều ⇒ thước đã trôi, và **mọi so sánh của vòng này vô hiệu**.
- `fold_dac_trung`: `dung_duoc` **1–5%** toàn bộ; và riêng **nhóm không trống**,
  `dung_duoc + tam` **≥ 20%** (so 12,8% của `hien_tai`).
- Đây là dự đoán **yếu**: mười cơ chế trước đều hoà hoặc thua. Nhưng đây là cơ chế đầu tiên
  nhắm vào đại lượng **đã đo được là tách 10×** giữa văn thật và bản sinh.

### Ý nghĩa từng kết cục — ghi trước

- **Hơn ở CẢ HAI thước** ⇒ chẩn §13.47 đúng: hệ mù miền, và fold đặc trưng là tín hiệu miền
  dùng được. Đề xuất tác giả nâng `gidf_min` của `CoverageGate` (kèm `reject_when_no_theme`).
- **Chỉ hơn ở nhóm không trống** ⇒ đúng hướng nhưng quá đắt. Việc kế là **hạ ngưỡng** — nêu
  trước, bộ đề khác, vòng mới. **Không** quét ngưỡng rồi chấm lại trên chính bộ đề này.
- **Không hơn ở cả hai** ⇒ tín hiệu miền **có tách được** (§13.47) nhưng **dùng làm cổng thì
  không ăn**. Khi đó cả hai họ — siết mối nối (§13.46) và siết miền — đều đóng, và §13 phải
  hỏi câu khác hẳn: *vì sao chọn đúng miền, đúng bề mặt, mà câu vẫn vô nghĩa.*
