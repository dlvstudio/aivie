# RUNBOOK M4 — chuỗi BA BƯỚC vòng 3, với đơn vị WS-CRF

> Giao 2026-08-11, sau bake sidecar `1ea917e`.
> **Thay** `RUNBOOK_M4_HUB3_CHAIN_V12.md` (vòng 2).
> M4 chỉ CHẠY và COMMIT. **KHÔNG chấm, KHÔNG mở `--out-key` / `--out-meta`.**

## 0. Đổi gì so với vòng 2

Vòng 2 (`8b416b9`) cho `typed_first` **5,9%** · `cooc` 11,1% — dưới cổng 15%.
Rà 24 ca trượt + truy câu train ra **bốn khiếm khuyết**, và một **chặn gốc**:

| | đã vá | commit |
|---|---|---|
| A · chủ ngữ lấy token kề pivot (13 700 câu: `chính là`, `cũng là`…) | ✅ trung tâm ngữ đoạn | `3758b9d` |
| B · `chẳng hạn như` — quét ngược dừng ở dấu phẩy (1 659 câu) | ✅ vượt tối đa một dấu phẩy | `3758b9d` |
| C · số từ thành nút lớp (`Bốn`) | ✅ chọn theo gidf | `3758b9d` |
| D · boilerplate từ điển «Ví dụ:» / «Giải nghĩa:» (52 486 lần) | ✅ bỏ token trước `:` | `3758b9d` |
| **CHẶN GỐC · cụm ISG 57% không phải đơn vị ngôn ngữ** | ✅ **đổi sang span WS-CRF** | `ebbad25`, `fb5883d` |

Sidecar đã bake xong (`1ea917e`): **71 379 span** · 2,5 phút · rác tự động 1,8%.

## 1. Chạy

```bash
PYTHONPATH=src .venv/bin/python -m pytest -m "not slow" -q   # kỳ vọng 418 passed

PYTHONPATH=src .venv/bin/python scripts/sample_hub_chain3_blind.py \
  --checkpoint checkpoints/network_v12_subst_isg_pw.pkl \
  --n 30 \
  --out out/hub3_chain_v12_r3_samples.txt \
  --out-key out/hub3_chain_v12_r3_key.json \
  --out-meta out/hub3_chain_v12_r3_meta.json
```

`--ws-spans` mặc định **auto** → tự nạp `network_v12_subst_isg_pw.ws_spans.json`.
Log phải in `sidecar WS-CRF: <n> span TỪ`. **Nếu không thấy dòng đó thì dừng** —
nghĩa là đang đo mà không dùng sidecar, y hệt vòng 2.

### Đối chứng đơn vị (chạy thêm, rẻ)

```bash
PYTHONPATH=src .venv/bin/python scripts/sample_hub_chain3_blind.py \
  --checkpoint checkpoints/network_v12_subst_isg_pw.pkl \
  --n 30 --ws-spans none \
  --out out/hub3_chain_v12_r3_noWS_samples.txt \
  --out-key out/hub3_chain_v12_r3_noWS_key.json \
  --out-meta out/hub3_chain_v12_r3_noWS_meta.json
```

Cùng code, chỉ khác đơn vị neo (ISG thay WS-CRF). Đây là cách duy nhất tách
"vá A–D có tác dụng" khỏi "đổi đơn vị có tác dụng".

## 2. Câu hỏi chấm — giữ nguyên vòng 2

> **Các từ ở `MỚI` có thật sự mang THUỘC TÍNH đầu dòng không?**

`dung` = phần lớn mang · `tam` = ít nhất một mang, còn lại không sai trắng trợn
· `hong` = không từ nào mang. **Không chấm nhãn giữa dòng.**

## 3. Cổng G-chain3 — giữ nguyên, không nới

| # | `typed_first` | kết luận |
|---|---|---|
| 1 | `dung+tam` ≥ **35%** | có thực chất ⇒ wire vào `generate` |
| 2 | 15–35% | biên — soi ca cụ thể |
| 3 | < **15%** | đóng hướng, ghi âm tính §18.H |

Mốc: vòng 1 **0,0%** → vòng 2 **5,9%**.

**Tự-kiểm:** `cooc` trộn trong bộ mẫu. `cooc` ≥ `typed_first` ⇒ tách kênh không
mua gì — phải nói rõ, đừng đọc thành thắng.

## 4. 🔴 Điều đã biết trước, ghi ra để không tự lừa

**76% corpus v12 là truyện mạng.** Cỡ các nguồn trong `config/train_v12_sent.yaml`:

| nguồn | cỡ |
|---|---:|
| **conv_stories** (kiếm hiệp/ngôn tình dịch) | **70,6 MB** |
| dictionary | 8,8 MB |
| doc_add_articles · paragraph_units | 13,7 MB |
| history_articles | 1,6 MB |
| grade_lessons + subject_articles + qa | **0,2 MB** |

Nguồn đó đã chi phối cả ba vòng trước: `là vậy` (54 lần, hút 8/22 dòng vòng 1),
`chẳng hạn như` (1 567/1 659 lần, hút 9/26 dòng vòng 2), và các span gộp quá tay
trong bake (`Kim Công sơn mạch`, `Thái Cổ truyền`, `ngộ Hỏa chi`).

⇒ **Nếu vòng 3 vẫn ~0%, nghi phạm số một KHÔNG còn là cơ chế mà là DỮ LIỆU.**
Khi đó việc đúng là đổi tỉ lệ nguồn, **không** vá cơ chế lần thứ sáu. Ghi trước
điều này để lúc có số không ai (kể cả Claude) diễn giải chệch đi.

## 5. Không làm

- Không chấm, không mở `key.json` / `meta.json` (meta **cũng** chứa nhãn nhánh).
- Không nhận xét chất lượng trong commit message — nó mồi cho người chấm.
- Không chỉnh `--seed`, `--n`, `--d-max`, ngưỡng cổng cho "số đẹp hơn".
- Không bake lại sidecar (đã có `1ea917e`).
- Không wire `hub_spread` vào `generate`.

## 6. Giao nộp

Commit 6 file `out/hub3_chain_v12_r3*` + `out/hub3_chain.log`.
Commit message: **chỉ** số dòng mỗi bộ · số span sidecar đã nạp · `n_typed_edges`
· số nhãn CỤM · thời gian.
