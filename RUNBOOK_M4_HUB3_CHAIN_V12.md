# RUNBOOK M4 — chuỗi BA BƯỚC đầy đủ trên v12 (chấm mù)

> Giao 2026-08-10, sau vòng chấm mù `3643f16`.
> **Thay** `RUNBOOK_M4_HUB3_BLIND_V12.md` — vòng đó đo THIẾU bước ①.
> M4 chỉ CHẠY và COMMIT. **KHÔNG chấm, KHÔNG mở `--out-key` / `--out-meta`.**

## ⚠️ VÒNG 2 — chạy lại sau khi vá kiểu 1 + kiểu 2

Vòng 1 (`8b416b9`) đã chạy: 22 dòng · chấm mù cho `typed_first` **0,0%** ·
`cooc` 11,1%. Đọc mẫu ra **bốn kiểu hỏng**, hai kiểu nặng nhất đã vá:

| kiểu | vòng 1 đo được | đã vá |
|---|---|---|
| **2 · ĐỌC NGƯỢC CHIỀU** | «Bốn đài truyền hình lớn nhất tại Hoa Kỳ **là** NBC, CBS, ABC và Fox» là câu PHÂN LOẠI thật, nhưng khuôn `A là B` đọc ngược ⇒ sinh cạnh thành-viên→thành-viên (`NBC → Fox`). Khuôn này có **554 lần** trong corpus | `_liet_ke`: nhận danh sách ≥3 mục sau pivot, phát `thành_viên → LỚP` |
| **1 · hút-điểm hồi chỉ** | nhãn `vậy` hút **8/22 dòng**, cả 8 cho y hệt `[nói, chuyện, giản, nhiều, lớn]`; nguồn = truyện mạng («thì ra là vậy», 54 lần) | lọc nhãn theo **PHÂN VỊ** gidf (không ngưỡng hằng — đã dính bẫy đó với `bậc-vào ≥3`) |

**Chưa vá (cố ý):**
- **kiểu 3** — bước ① dùng đồng hiện d≤6, không phải quan hệ "mang thuộc tính"
  (`«Ireland» → tập, trung, canh, tác`). Cần quan hệ có kiểu cho ①.
- **kiểu 4** — bước ① nuốt nửa cụm (`Siri → sử, dụng` từ `sử dụng`).

⇒ Nếu vòng 2 vẫn ~0% thì **hai kiểu còn lại không phải nguyên nhân chính**, và
theo cam kết đã ghi: **đóng hẳn**, không vá tiếp.

**Giới hạn đã biết, đừng báo như lỗi:** nhãn trích ra mang tên `Bốn đài` /
`Ba loài` thay vì `đài truyền hình` / `loài thú` — số từ + loại từ lọt vào cụm.
Nhóm THÀNH VIÊN vẫn đúng (`Bốn đài → ABC, CBS, Fox, NBC`), chỉ tên nút xấu và
không hợp nhất được với câu khác nói `đài truyền hình`.

## 0. Vì sao phải đo lại

Vòng trước cho `typed_first` 10,7% vs `cooc` 73,3% → tưởng đóng được hướng typed.
Đọc kỹ ca `lầy` thì lộ ra **ba lỗi, đều của Claude**, và kết luận đó không đứng:

| # | lỗi | bằng chứng |
|---|---|---|
| 1 | **Bỏ mất bước ①.** Sampler xuất phát từ MỘT từ đơn lẻ, không có thuộc tính ràng buộc | thiết kế tác giả có 3 bước; sampler chỉ gọi `score_hubs_from` + `reverse_peers` |
| 2 | **Nhãn bị cắt cụt.** `vũng lầy là một tình thế` → nhãn `tình` thay `tình thế` | `tình yêu` 469 / `tình hình` 613 vs `tình thế` 287 trong corpus ⇒ cắt đôi là ĐỔI NGHĨA |
| 3 | **Thang chấm thưởng nhầm.** Hỏi "thành viên có mạch lạc không" ⇒ thưởng HỌ CẤU TẠO TỪ, mà tiếng Việt ghép 2 âm tiết nên `cooc` thắng gần như theo định nghĩa | `lội → bơi, lặn, cá, lụt` được chấm `tam` dù `lội` không phải lớp của `lầy` |

Đã vá 1+2 (commit kèm). Vòng này sửa nốt 3 bằng **câu hỏi chấm khác hẳn**.

## 1. Chạy

```bash
PYTHONPATH=src .venv/bin/python -m pytest -m "not slow" -q   # kỳ vọng 417 passed

PYTHONPATH=src .venv/bin/python scripts/sample_hub_chain3_blind.py \
  --checkpoint checkpoints/network_v12_subst_isg_pw.pkl \
  --n 30 \
  --out out/hub3_chain_v12_r2_samples.txt \
  --out-key out/hub3_chain_v12_r2_key.json \
  --out-meta out/hub3_chain_v12_r2_meta.json
```

`--d-max` mặc định **6** (không phải 3): trong «nghịch ngợm như loài chó» thì
`chó` ở **d=4**, để 3 là cắt mất nó — đã đo trên ví dụ tác giả.

Mỗi dòng:
```
[C012]  thuộc tính «nghịch»  ·  hạt giống: loài, chó, mèo
        ·  nhãn: động vật (2 phiếu)  ·  MỚI: khỉ
```

## 2. Câu hỏi chấm — KHÁC vòng trước

Vòng trước hỏi *"hub có phải nhãn không"*. Vòng này hỏi đúng thứ thiết kế hứa:

> **Các từ ở `MỚI` có thật sự mang THUỘC TÍNH đầu dòng không?**

`dung` = phần lớn từ MỚI mang thuộc tính · `tam` = ít nhất một từ mang, còn lại
không sai trắng trợn · `hong` = không từ nào mang.

**Không chấm nhãn giữa dòng** — nó chỉ để tham khảo. `khỉ` có mang tính
«nghịch ngợm» hay không mới là câu hỏi, chứ không phải `động vật` có đẹp không.

## 3. Cổng G-chain3 — ghi TRƯỚC

| # | điều kiện trên `typed_first` | kết luận |
|---|---|---|
| 1 | `dung + tam` ≥ **35%** | chuỗi ba bước có thực chất ⇒ wire vào `generate` |
| 2 | 15–35% | biên — soi ca cụ thể, chưa mở rộng |
| 3 | < **15%** | đóng hướng, ghi âm tính §18.H — lần này **đúng cơ chế** mới đóng được |

**Tự-kiểm:** `cooc` là đối chứng trộn lẫn trong bộ mẫu. Nếu `cooc` ≥ `typed_first`
thì tách kênh không mua gì ở chuỗi đầy đủ — phải nói rõ, đừng đọc thành thắng.

## 4. Đối chứng đã có

**Ví dụ tác giả, chạy trọn ba bước ở nhiễu 2000** (test
`test_CHUOI_BA_BUOC_day_du_ra_duoc_khi`):

```
① «nghịch»  → loài, chó, mèo
② hạt giống → động vật  (2/3 phiếu, cụm ISG NGUYÊN KHỐI)
③ nhãn      → khỉ       ← chưa từng đứng cạnh «nghịch»
```

Cơ chế chạy đúng thiết kế trên fixture. **Câu hỏi cho v12: có sống ở corpus thật không?**

Mini (`mini_sent_ep_qa.pkl`, 19 dòng) chỉ để kiểm định dạng — chất lượng kém, ví dụ:
```
[C002] thuộc tính «vua» · hạt giống: Tông, Lê, Trần, Lý · nhãn: Tông · MỚI: Thái, Thánh, Nhân
```
Không phải chuẩn vàng.

## 5. Không làm

- Không chấm, không mở `key.json` / `meta.json` (meta **cũng** chứa nhãn nhánh).
- Không ghi nhận xét chất lượng vào commit message — nó mồi cho người chấm.
- Không chỉnh `--seed`, `--n`, `--d-max` cho "mẫu đẹp hơn".
- Không wire `hub_spread` vào `generate` ở vòng này.

## 6. Giao nộp

Commit `out/hub3_chain_v12_{samples.txt,key.json,meta.json}` + `out/hub3_chain.log`.
Commit message: **chỉ** số dòng, số thuộc tính, `n_typed_edges`, số nhãn CỤM, thời gian.
