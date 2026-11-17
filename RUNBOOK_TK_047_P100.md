# RUNBOOK TK_047 P100 — đo ĐỘ CHÍNH XÁC CẤP PHÉP trên **v12**

> Cho M4. M4 chỉ thấy repo, không thấy hội thoại ⇒ mọi thứ cần biết đều ở đây.
> Ngày giao: 2026-08-05.

## 1. Vì sao chạy cái này

TK_047 (§13) đóng hướng **khuôn + thay khe** trên `mini` (5.664 cha). Kết quả then chốt:

| | |
|---|---:|
| ép thay 1 khe vs phát nguyên văn | **−42,5 điểm** (30,0% vs 72,5%), z = −6,59 |
| chọn ứng viên **bám đề nhất** vs bốc **ngẫu nhiên** | **+0,0 điểm**, z = 0,00 |

Vế thứ hai đổi cả chẩn đoán: nút thắt **không phải** *tiêu chí chọn kém* mà là **BỂ ỨNG VIÊN
gần như toàn sai** — chọn cái nào cũng vậy vì cái nào cũng hỏng.

Đại lượng đo trực tiếp bể đó là **độ chính xác cấp phép (thước "hẹp")**: suy phép thay trên nửa
corpus **A**, hỏi bộ ba `(a, c, b)` có **thật sự xuất hiện** ở nửa **B** không.

Trên `mini`: **23,4%** ⇒ ba phần tư phép thay "hợp lệ" **chưa từng tồn tại** ở đâu khác.

**Câu hỏi duy nhất của P100:** con số đó là tính chất của **DỮ LIỆU THƯA** hay của **ĐỒNG HIỆN
CỤC BỘ**? `v12` có **430.447 cha — gấp 76 lần** `mini`. Nếu 23,4% chủ yếu do thưa thì nó phải
lên rõ rệt. Nếu nó **đứng yên ở quy mô 76×**, hướng đồng-hiện-cục-bộ **đóng dứt điểm**.

## 2. Cổng — đã nêu TRƯỚC, ĐỪNG SỬA

Cổng nằm sẵn trong `scripts/probe_template_slot.py` (hằng số đầu file). **Không chỉnh chúng.**

| | | |
|---|---|---|
| (1) | ≥300 cha có khe thay được | `GATE_MIN_PARENT` |
| (2) | tổ hợp mới trung vị ≥ 1,0 (log10) | `GATE_NOVEL_MIN` |
| (3) | thước **chủ đề** ≥ 2,0× đối chứng | `GATE_VS_RANDOM` |
| **(4)** | **thước HẸP ≥ 50% TUYỆT ĐỐI** | `GATE_NARROW_MIN` ← **cột quyết định** |

⚠️ Cổng (4) mới thêm, và lý do nó tồn tại là một **lỗi đọc số đã xảy ra thật**: trên mini,
hẹp 23,4% = **17× sàn ngẫu nhiên**, và tỷ số đó bị đọc là "tốt". **Tỷ số lớn trên sàn gần 0
thì vẫn gần 0 về tuyệt đối.** Vì thế P100 phải đọc **mức tuyệt đối** trước, tỷ số sau.

## 3. Lệnh

Chạy từ gốc repo. Dùng `PYTHONPATH=src` (không có `.venv`).

### Bước A *(tuỳ chọn, ~60–90 phút)* — bản đồ nhãn từ loại cho v12

Chỉ dùng cho **đối chứng cùng-từ-loại** (cổng 3). **Cổng (4) không cần nó.** Nếu bước này lỗi
hoặc quá lâu, **bỏ qua** — probe tự chạy tiếp với đối chứng yếu hơn và **in cảnh báo rõ**.

```bash
PYTHONPATH=src python scripts/build_pos_labels.py \
  --checkpoint checkpoints/<v12>.pkl \
  --level word --emit-map out/token_labels_v12.json
```

### Bước B — phép đo chính

```bash
PYTHONPATH=src python scripts/probe_template_slot.py \
  --checkpoint checkpoints/<v12>.pkl \
  --map out/token_labels_v12.json \
  --n-parent 600 \
  --max-index-parent 100000 \
  --out out/probe_template_slot_v12.json
```

**Chạy HAI LẦN**, ghi ra `_r1.json` / `_r2.json`, và **so hai bản trước khi đọc**. Đây là quy
tắc bắt buộc của dự án: một lỗi `PYTHONHASHSEED` đã từng làm một cổng **lật** giữa hai lần chạy
cùng lệnh, trong khi các cổng khác đứng yên nên bảng trông vẫn đúng. Hai bản khác nhau ⇒ **báo
lại, đừng đọc**.

### Ghi chú vận hành

- `--max-index-parent 100000` giới hạn **100.000 cha mỗi nửa** (≈35× mini). Bắt đầu bằng con
  số này. Nếu RAM còn thoải mái, chạy thêm một lượt với `--max-index-parent 0` (toàn bộ) và
  **báo cả hai** — độ chính xác có thể phụ thuộc quy mô, và đó chính là thứ đang hỏi.
- Rủi ro RAM lớn nhất là `coB` (token → tập token đồng hiện). Nếu OOM: giảm
  `--max-index-parent` xuống 50000 rồi 25000, và **báo lại mốc chạy được**.
- Checkpoint nạp có `gc.disable()` sẵn (nhanh ~1300×). Đừng bỏ.
- Tiến trình ghi ra `out/logs/probe_template_slot_*.log`.

## 4. Đọc kết quả

Bảng in ra có dạng:

```
bộ thay                       n      hẹp    chủ đề
duoc_phep                  ....   XX.X%     YY.Y%
ngau_nhien_cung_loai       ....    Z.Z%      W.W%

ĐỘ CHÍNH XÁC TUYỆT ĐỐI (hẹp)  = XX.X%   (mini: 23,4% · cổng ≥50%)   ← CỘT QUYẾT ĐỊNH
```

| `hẹp` trên v12 | kết luận |
|---|---|
| **≥ 50%** | cấp phép **là chuyện dữ liệu**; hướng khuôn–khe **mở lại**, và bước kế là dựng lại bộ sinh trên v12 |
| **30–50%** | cải thiện thật nhưng **chưa đủ**; ghi số, **không** dựng lại bộ sinh |
| **≈ 23% (đứng yên)** | 🔴 tính chất của **đồng hiện cục bộ**, không phải của dữ liệu ⇒ hướng **đóng dứt điểm** |

## 5. Dự đoán ghi TRƯỚC

*(Của Claude, ghi trước khi chạy — để đối chiếu, không phải để tin.)*

- `hẹp` lên **35–55%**: dữ liệu dày 76× làm bộ ba được chứng thực nhiều hơn hẳn. Nhưng **rất
  không chắc**, vì `v12` cũng làm `triA` **rộng ra** — nhiều ứng viên được cấp phép hơn, và
  ứng viên thêm vào là những cái **hiếm hơn**, tức có thể **kéo độ chính xác xuống**. Hai lực
  ngược chiều, tôi không biết lực nào thắng. Đó là lý do phép đo này đáng chạy.
- Cổng (1)(2)(3) **đạt** dễ trên v12.
- Nếu `hẹp` **giảm** so với mini thì đó là bằng chứng **mạnh nhất có thể** cho việc cấp phép
  bằng bộ ba là ngõ cụt, và tôi sẽ ghi đóng ngay.

## 6. Giao nộp

Commit vào nhánh hiện tại:

- `out/probe_template_slot_v12_r1.json`, `_r2.json` (và bản `--max-index-parent 0` nếu chạy)
- `out/logs/probe_template_slot_*.log`
- `out/token_labels_v12.json` nếu bước A chạy được

Trong commit message ghi rõ: **`hẹp` tuyệt đối**, `chủ đề`, tỷ số vs đối chứng, `--max-index-parent`
đã dùng, hai lần chạy **có giống nhau không**, và bước A **có chạy được không**. Nếu số lệch
giữa hai lần chạy, **nói rõ điều đó thay vì chọn một bản**.
