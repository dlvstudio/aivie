# RUNBOOK TK_045 P79 — mối nối **KỀ NHAU**: đo trên thang TUYỆT ĐỐI, trộn chung một phiên (§13.45)

Runbook tự đủ: mục 1 giao M4, mục 2 giao giám khảo.

## 0. Vì sao

§13.43 (mốc gốc): cấu hình đang thắng cho **0,5% dùng được**, **91% hỏng**. §13.44 định vị
nguyên nhân cấu trúc:

- **46,6%** mối nối sinh đặt cạnh nhau hai token **chưa bao giờ đứng kề nhau** trong corpus
  (văn thật: 5,2%). `W_skip` chỉ chứng thực **đồng hiện trong cửa sổ 8**, không chứng thực
  **kề nhau** — mà chỗ ghép bề mặt thì chỉ "kề nhau" mới có nghĩa.
- Tầng mệnh đề **CẮT** ở `pmi ≤ 0` nhưng **NỐI** ở pmi trung vị **+0,57** ⇒ **bộ cắt và bộ nối
  dùng hai tiêu chí mâu thuẫn nhau**; nó nối ở đúng loại điểm mà chính nó không coi là biên.
  Đó là lý do cấu trúc khiến `quốc gia` + `gia vị` trông hợp lệ.

Ứng viên `join_seq_max=0.0`: mối nối chỉ hợp lệ khi chuyển tiếp bề mặt **được chứng thực kề
nhau** *và* nằm trong **dải biên** — tức bắt bộ nối dùng **cùng tiêu chí** với bộ cắt. Thuần
đếm trên `W_seq` ⇒ Luật 0 sạch.

Đo trước trên mini: ghép chưa-từng-thấy **46,3% → 0,0%**, nhưng **trống 20,7% → 56,5%** và
mệnh đề TB 1,41 → 0,58. **Cái giá lớn và đã biết trước** — §13.37 nói im lặng hơn nói bậy,
nhưng "hơn bao nhiêu" thì phải để giám khảo phán.

## 1. Sinh mẫu — **giao M4**

```bash
PYTHONPATH=src .venv/bin/python scripts/gen_absolute_samples.py \
    --checkpoint checkpoints/network_v12_sent.pkl --n 200 --join-seq-max 0.0 \
    --out out/abs2_samples_v12.txt --out-key out/abs2_key_v12.json \
    --out-meta out/abs2_meta_v12.json
```

Script sinh **cả hai** cấu hình trên **cùng 200 đề** (400 mẫu), **trộn** và **giấu nhãn**.

⚠️ **Vì sao phải trộn chung, không chấm hai phiên:** mốc gốc chỉ **0,5%**. Hai phiên giám khảo
khác nhau áp hai mức chặt tay khác nhau, và chênh đó **thừa sức nuốt** một hiệu ứng cỡ vài
phần trăm. Trộn chung là cách duy nhất để con số mới **so được** với con số cũ.

⚠️ **KHÔNG LỌC** — mẫu trống vẫn vào, và vào **cả hai** phía.

### 1.1 CỔNG G-abs2 (nêu TRƯỚC)

| # | điều kiện | nếu trượt |
|---|---|---|
| 1 | **400 mẫu**, khoá chia đúng **200/200** | script có bug — dừng, báo |
| 2 | trống của nhánh `mối_nối_kề` ∈ **40–75%** | mini đo 56,5%; lệch nhiều ⇒ báo trước khi chấm |
| 3 | tỷ lệ ghép **chưa từng thấy** của nhánh mới = **0%** | cơ chế không ăn — dừng, báo |

### 1.2 Báo cáo M4

Commit ba file, kèm: số mẫu · trống **theo từng nhánh** · mệnh đề TB theo nhánh · thời gian.
**Không chấm.**

## 2. Giao chấm — phiên Claude RIÊNG

⛔ Cấm đọc: `out/abs2_key_v12.json`, mọi `out/*_key*`, `out/*verdict*`, `out/*meta*`,
`TK_045_MATRIX_CORE.md`, mọi `RUNBOOK_*`.

File được đọc: **chỉ** `out/abs2_samples_v12.txt`.

### 2.1 Prompt giao giám khảo (copy nguyên khối)

> Trong `out/abs2_samples_v12.txt` có 400 mẫu, mỗi mẫu gồm một **ĐỀ** và phần **tiếp theo do
> máy sinh** (dòng `→`). Chấm **từng mẫu ĐỘC LẬP** — không so mẫu này với mẫu khác.
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
> ⚠️ Trong file có **hai loại mẫu trộn lẫn** do hai cấu hình sinh ra, nhưng bạn **không biết
> và không cần biết** mẫu nào thuộc loại nào. **Đừng đoán**, đừng cố nhóm chúng lại — cứ chấm
> tuần tự từng mẫu bằng **cùng một** thước từ đầu tới cuối. Giữ thước đều là việc quan trọng
> nhất của vòng này.
>
> ⛔ Không mở `out/abs2_key_v12.json`, không mở file `*_key*`/`*verdict*`/`*meta*` nào, không
> mở `TK_045_MATRIX_CORE.md` hay `RUNBOOK_*`.
>
> Ghi `out/abs2_verdict_v12.json`:
> `{"grades": {"A000": "hong", ...}, "notes": "6-10 câu"}` — đủ **400 mã** A000–A399.
>
> Trong `notes`: (1) mẫu `dung_duoc` giống nhau ở điểm gì; (2) kiểu hỏng phổ biến nhất và tỷ
> trọng; (3) có mẫu nào **chỉ thiếu một chút** không, thiếu gì.

## 3. Lật khoá & kết luận

Đếm ba mức **theo từng cấu hình** trên mẫu số 200 mỗi bên (gồm mẫu trống).

### Dự đoán ghi TRƯỚC (Claude)

- `hien_tai`: `dung_duoc` ≈ **0,5–2%** (khớp mốc §13.43).
- `mối_nối_kề`: `dung_duoc` **3–10%**, `hong` **≥80%** (vì trống ~56% đã tự vào `hong`).
- Nếu tính riêng **mẫu KHÔNG trống**, `mối_nối_kề` phải **hơn rõ**: `dung_duoc + tam` ≥ **35%**
  so với ~12,8% của `hien_tai` (§13.43). Đây là chỗ hiệu ứng phải hiện ra nếu H đúng.

### Ý nghĩa từng kết cục — ghi trước

- **`mối_nối_kề` cao hơn ở CẢ HAI thước** (toàn bộ, và riêng mẫu không trống) ⇒ chẩn §13.44
  **đúng**: sai lầm nằm ở **đại lượng** làm mối nối. Đề xuất tác giả bật `join_seq_max` mặc
  định, và §13 chuyển trọng tâm sang **giảm tỷ lệ trống** thay vì sửa cách chọn.
- **Chỉ hơn ở mẫu KHÔNG trống, thua ở toàn bộ** ⇒ cơ chế **đúng hướng nhưng quá đắt**. Việc kế
  là nới `join_seq_max` (dải biên rộng hơn) — **nêu trước**, vòng mới, bộ đề khác.
- **Không hơn ở cả hai** ⇒ chẩn §13.44 **sai**: ghép "chưa từng thấy kề nhau" **không** phải
  nguyên nhân chính của 91% hỏng, dù nó chiếm 46,6%. Khi đó phải quay lại hỏi *vì sao một phép
  chọn hợp lệ vẫn cho ra câu vô nghĩa* — và mọi ứng viên dạng "siết điều kiện mối nối" đều
  chết theo.
