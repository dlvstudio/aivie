# RUNBOOK M4 — chuỗi BA BƯỚC vòng 4

> Giao 2026-08-11, sau vòng 3 `b1e1245`. **Thay** `RUNBOOK_M4_HUB3_CHAIN_V12_R3.md`.
> M4 chỉ CHẠY và COMMIT. **KHÔNG chấm, KHÔNG mở `--out-key` / `--out-meta`.**

## 0. Vì sao chạy lại — ba bug, và một phép đo VÔ HIỆU

Vòng 3 cho WS **5,3%** · noWS 7,1%, và tôi kết luận *"WS-CRF không mua gì,
nghi phạm là dữ liệu"*. **Cả hai đều sai.** Soi 28 ca trượt, tái lập trên bàn:

| # | bug | bằng chứng |
|---|---|---|
| **1** | **`thành` là pivot nhưng cũng là ÂM TIẾT từ thường** | trở thành 3 516 (đúng) vs thành phố 2 262 · thành công 1 995 · hoàn thành 1 466 · thành lập 1 368 · hình thành 1 297 ⇒ **~8 400 sai / 3 500 đúng**. «nền kinh tế **thành** phố số 5 phục hồi» bị đọc như triple phân loại — **nguồn hút-điểm `phố`, 10/28 dòng** |
| **2** | **nhãn nuốt cả pivot** | «Các thành phố quan trọng khác **LÀ** Göteborg, Malmö…» → lớp = `trọng khác là` |
| **3** | 🔴 **span WS-CRF bị TRỘN với cụm ISG thay vì THAY THẾ** | ISG vẫn cạnh tranh và thắng bằng span rác `phố quan` (cắt ngang `thành phố`/`quan trọng`) |

Bug 3 nghĩa là **vòng 3 đã so `(WS+ISG)` với `(ISG)`** — không phải `(WS)` với
`(ISG)`. **Phép đo đối chứng vòng 3 vô hiệu.** Vòng 4 là lần đầu so đúng.

Cả ba đã vá (`55c6dd2`, 421 test).

## 1. Chạy

```bash
PYTHONPATH=src .venv/bin/python -m pytest -m "not slow" -q   # kỳ vọng 421 passed

# (A) với đơn vị WS-CRF
PYTHONPATH=src .venv/bin/python scripts/sample_hub_chain3_blind.py \
  --checkpoint checkpoints/network_v12_subst_isg_pw.pkl --n 30 \
  --out out/hub3_chain_v12_r4_samples.txt \
  --out-key out/hub3_chain_v12_r4_key.json \
  --out-meta out/hub3_chain_v12_r4_meta.json

# (B) ĐỐI CHỨNG — đơn vị ISG
PYTHONPATH=src .venv/bin/python scripts/sample_hub_chain3_blind.py \
  --checkpoint checkpoints/network_v12_subst_isg_pw.pkl --n 30 --ws-spans none \
  --out out/hub3_chain_v12_r4_noWS_samples.txt \
  --out-key out/hub3_chain_v12_r4_noWS_key.json \
  --out-meta out/hub3_chain_v12_r4_noWS_meta.json
```

**KHÔNG bake lại sidecar.** Sidecar chứa span TỪ của WS-CRF, không phụ thuộc
tập pivot ngoài việc chọn cha nào được tách. Bỏ `thành` chỉ làm bake mới HẸP
hơn ⇒ sidecar hiện tại (`1ea917e`, 71 379 span) là **tập bao**, vẫn đúng.

(A) phải in `sidecar WS-CRF: 71379 span TỪ`; (B) phải in
`⚠️ CHẠY KHÔNG CÓ span WS-CRF`. **Không thấy đúng dòng thì dừng.**

## 2. Câu hỏi chấm — giữ nguyên

> **Các từ ở `MỚI` có thật sự mang THUỘC TÍNH đầu dòng không?**

`dung` = phần lớn mang · `tam` = ít nhất một mang, còn lại không sai trắng trợn
· `hong` = không từ nào mang. **Không chấm nhãn giữa dòng.**

## 3. Cổng G-chain3 — giữ nguyên, KHÔNG nới

| # | `typed_first` (bộ A) | kết luận |
|---|---|---|
| 1 | `dung+tam` ≥ **35%** | có thực chất ⇒ wire vào `generate` |
| 2 | 15–35% | biên |
| 3 | < **15%** | đóng hướng, ghi âm tính §18.H |

Mốc: vòng 1 **0,0%** · vòng 2 **5,9%** · vòng 3 **5,3%** (WS) / 7,1% (ISG, so nhầm).

**So A với B** là câu hỏi thứ hai, lần đầu trả lời được: đơn vị WS-CRF có mua gì
không. Chênh nhỏ ở n≈20 là nhiễu — đừng đọc thành kết luận.

## 4. ⚠️ Đã biết chưa giải — ĐỪNG báo như phát hiện mới

**Chọn trung tâm ngữ đoạn chưa giải được.** Ba luật đã thử, mỗi luật sai một kiểu
(đo trên fixture, **chưa xác thực được ở v12** vì máy dev không nạp nổi ckpt):

| luật | «Các thành phố quan trọng khác» | «Bốn đài truyền hình lớn nhất» |
|---|---|---|
| trái nhất | `Các` ❌ | `Bốn` ❌ |
| **max-gidf** (đang dùng) | `quan trọng` ❌ | `đài truyền hình` ✅ |
| trái-nhất-đủ-đặc-thù | `Các` ❌ | `Bốn` ❌ |

Nên nhãn ra `quan trọng` thay `thành phố` là **đã biết**. Phân biệt ĐỊNH TỪ với
DANH TỪ TRUNG TÂM cần từ loại; tần suất một mình không đủ. Ghi nhận nếu thấy,
**đừng sửa**.

## 5. 🔴 KHÔNG đăng ký lại giả thuyết "dữ liệu là thủ phạm"

Runbook R3 §4 ghi trước: *"vòng 3 âm tính ⇒ nghi phạm là DỮ LIỆU (76% corpus là
truyện mạng)"*. Số về đúng vậy và tôi đã diễn giải theo đúng lời ghi trước —
**nhưng lời ghi trước ấy sai**, vì nó ngầm giả định cơ chế đã đúng. Thực tế còn
ba bug cơ chế.

⇒ Vòng 4 **không** đăng ký trước giả thuyết nào về nguyên nhân. Có số rồi mới
soi ca trượt, như vòng 2 và 3 đã làm. Bài học: tiền-đăng-ký giữ khỏi bẻ cong
diễn giải, **không** giữ khỏi đăng ký nhầm giả thuyết.

## 6. Không làm

- Không chấm, không mở `key.json` / `meta.json` (meta **cũng** chứa nhãn nhánh).
- Không nhận xét chất lượng trong commit message.
- Không bake lại sidecar · không chỉnh `--seed`/`--n`/`--d-max`/ngưỡng cổng.
- Không sửa luật chọn trung tâm ngữ đoạn (§4).
- Không wire `hub_spread` vào `generate`.

## 7. Giao nộp

Commit 6 file `out/hub3_chain_v12_r4*` + `out/hub3_chain.log`.
Commit message: **chỉ** số dòng mỗi bộ · số span sidecar đã nạp · `n_typed_edges`
mỗi bộ · số nhãn CỤM · thời gian.
