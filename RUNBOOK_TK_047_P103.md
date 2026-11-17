# RUNBOOK TK_047 P103 — vòng ÉP THAY trên **v12**: đo HÀM TRUYỀN độ-chính-xác → chất-lượng

> Cho M4. M4 chỉ thấy repo, không thấy hội thoại ⇒ mọi thứ cần biết đều ở đây.
> Ngày giao: 2026-08-05.

## 1. Câu hỏi

Trên `mini`, ép thay 1 khe **tụt −42,5 điểm** (30,0% vs 72,5%) — đo ở độ chính xác cấp phép
**23,4%**. P100 đo lại trên `v12`: độ chính xác lên **43,8%** và **chưa phẳng**.

**Hàm truyền *độ chính xác → chất lượng* mới có ĐÚNG MỘT điểm.** P103 đo điểm thứ hai.

| kết quả trên v12 | kết luận |
|---|---|
| khoảng cách **thu hẹp rõ** | hàm truyền **có thật** ⇒ TK_047 **mở lại**, bài toán thành bài toán **dữ liệu** |
| khoảng cách **giữ ~−42đ** | 🔴 độ chính xác **không** chuyển thành chất lượng ⇒ **đóng dứt điểm** |

Cả hai kết cục đều dứt khoát. Đừng tìm cách làm cho số "đẹp" — kết cục âm cũng có giá trị đầy đủ.

## 2. Ba thay đổi bắt buộc so với vòng mini

**(a) BỘ MẪU NEO.** `out/anchor_samples.txt` (40 mẫu) + `out/anchor_key.json` (nhãn tham chiếu
= đồng thuận hai giám khảo độc lập). **Phải trộn vào file chấm.** P101 đo được giám khảo trôi
**23 điểm** ở vùng giữa thang giữa hai phiên ⇒ **không có neo thì vòng này không so được với
vòng mini**, và toàn bộ phép đo mất ý nghĩa.

**(b) CẤM THAY KHE `số`.** Đã cài sẵn trong `gen_three_arm.py` qua `--label-map`. P102 đo: 100%
lỗi sự thật nằm ở **niên đại**; tên người và địa danh khớp hết.

**(c) Bản đồ nhãn v12**: `out/token_labels_v12.json` — M4 đã dựng ở P100. Dùng lại.

## 3. Lệnh

Chạy từ gốc repo, `PYTHONPATH=src` (không có `.venv`).

### Bước 1 — sinh mẫu

```bash
PYTHONPATH=src python scripts/gen_three_arm.py \
  --checkpoint checkpoints/network_v12_sent.pkl \
  --label-map out/token_labels_v12.json \
  --arms chep_cau,ep_ngau_nhien,ep_chu_de \
  --n 120 \
  --out out/ep_v12_samples.txt \
  --out-key out/ep_v12_key.json \
  --out-meta out/ep_v12_meta.json
```

Kiểm log có dòng `cấm thay khe 'số': N token` với **N > 0**. Nếu N = 0 ⇒ bản đồ nhãn không nạp
được, **dừng lại và báo**, đừng chạy tiếp.

### Bước 2 — trộn NEO vào file chấm

```bash
PYTHONPATH=src python - <<'PY'
import re, random, pathlib
main = pathlib.Path("out/ep_v12_samples.txt").read_text(encoding="utf-8")
anch = pathlib.Path("out/anchor_samples.txt").read_text(encoding="utf-8")
pat = r"(\[[A-Z]?\w+\] ĐỀ: .*?\n   → .*?\n)"
blocks = re.findall(pat, main, re.S) + re.findall(pat, anch, re.S)
random.Random(103).shuffle(blocks)
with open("out/ep_v12_cham.txt", "w", encoding="utf-8") as f:
    f.write("# CHẤM MÙ TUYỆT ĐỐI — mỗi mẫu chấm ĐỘC LẬP, không so với mẫu nào khác.\n")
    f.write(f"# n={len(blocks)} mẫu · KHÔNG LỌC · thang: dung_duoc / tam / hong.\n\n")
    for b in blocks:
        f.write(b + "\n")
print("→ out/ep_v12_cham.txt", len(blocks), "mẫu")
PY
```

### Bước 3 — chấm mù

Mở **một phiên riêng**. Giám khảo **chỉ được đọc** `out/ep_v12_cham.txt`. **Cấm** mọi file có
`key`/`meta`/`verdict`/`anchor` trong tên, mọi `TK_*.md`, `RUNBOOK_*.md`, `scripts/`, `src/`.

Đề bài chấm: **sao chép NGUYÊN VĂN** phần "Thang chấm TUYỆT ĐỐI" trong `TK_047 §16` — **đừng
sửa một chữ nào**. P101 cho thấy tôi thêm **một dòng** vào đề bài lần 2 và nhánh `ar` tụt
**6,2 điểm** vì đúng dòng đó. Đề bài là **một phần của dụng cụ đo**.

Bắt giám khảo ghi file **sau mỗi lô 40 mẫu** (gộp rồi ghi đè) ra `out/ep_v12_verdict.json`.
Một phiên trước đã đọc hết 360 mẫu rồi hết hạn mức trước khi kịp ghi ⇒ mất trắng.

### Bước 4 — tính ĐỘ TRÔI rồi mới đọc kết quả

```bash
PYTHONPATH=src python - <<'PY'
import json, collections, math
v = json.load(open("out/ep_v12_verdict.json"))["grades"]
ak = json.load(open("out/anchor_key.json"))
mk = json.load(open("out/ep_v12_key.json"))
ok = lambda ids: sum(1 for i in ids if v.get(i) in ("dung_duoc", "tam")) / max(len(ids), 1)
aid = [i for i in ak if i in v]
print(f"NEO n={len(aid)} · phiên này {ok(aid):.1%} · tham chiếu "
      f"{sum(1 for i in aid if ak[i] in ('dung_duoc','tam'))/len(aid):.1%}")
print(f"ĐỘ TRÔI = {ok(aid) - sum(1 for i in aid if ak[i] in ('dung_duoc','tam'))/len(aid):+.1%}")
for arm in ("chep_cau", "ep_ngau_nhien", "ep_chu_de"):
    ids = [i for i, a in mk.items() if a == arm and i in v]
    c = collections.Counter(v[i] for i in ids)
    print(f"{arm:<16} n={len(ids):<4} dung_duoc {c['dung_duoc']/len(ids):>6.1%} "
          f"· dùng+tạm {ok(ids):>6.1%}")
PY
```

⚠️ **Đại lượng cần đọc là HIỆU giữa các nhánh trong CÙNG vòng này**, không phải mức tuyệt đối.
Mức tuyệt đối chỉ so được sau khi trừ độ trôi trên neo. Trên `mini` hiệu là:
`ep_ngau_nhien − chep_cau = −42,5đ` và `ep_chu_de − ep_ngau_nhien = +0,0đ`.

## 4. Ghi chú vận hành

- `v12` **không nạp được trên máy dev** (Intel 8GB, bị swap). Đó là lý do việc này giao M4.
- Nạp checkpoint có `gc.disable()` sẵn (nhanh ~1300×). Đừng bỏ.
- Bước 1 xây chỉ mục bộ ba trên toàn bộ 430k cha — **rủi ro RAM lớn nhất**. Nếu OOM, báo lại
  kèm mốc; **đừng** tự giảm quy mô rồi im lặng, vì quy mô chính là biến đang đo.
- Tiến trình ghi ra `out/logs/gen_three_arm_*.log`.

## 5. Dự đoán ghi TRƯỚC (Claude)

- `ep_ngau_nhien − chep_cau` trên v12: **−20 đến −35đ** (mini: −42,5đ). Tức **có thu hẹp** nhưng
  **chưa đóng**. Lý do: độ chính xác lên 23,4% → 43,8%, chưa tới đa số.
- `ep_chu_de − ep_ngau_nhien`: vẫn **0–8đ**, tức vẫn ~vô giá trị. §12 đo đúng 0,0đ và không có
  gì trên v12 làm tiêu chí chọn tốt lên.
- **Độ trôi trên neo**: đây là lần đầu đo được. Tôi đoán **|trôi| ≤ 15đ**, nhưng thật sự không
  biết — P101 đo được 23đ giữa hai phiên.
- Nếu hiệu **giữ nguyên ~−42đ** dù độ chính xác gần gấp đôi ⇒ đóng dứt điểm, và đó là kết quả
  **có giá trị đầy đủ**, không phải thất bại của vòng đo.

## 6. Giao nộp

Commit: `out/ep_v12_samples.txt`, `out/ep_v12_key.json`, `out/ep_v12_meta.json`,
`out/ep_v12_cham.txt`, `out/ep_v12_verdict.json`, `out/logs/*`.

Commit message **phải ghi**: độ trôi trên neo · ba mức của từng nhánh · hai hiệu số · số token
bị cấm thay ở khe `số` · và **đề bài chấm có bị sửa chữ nào không**.
