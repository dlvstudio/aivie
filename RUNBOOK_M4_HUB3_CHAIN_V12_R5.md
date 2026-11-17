# RUNBOOK M4 — chuỗi BA BƯỚC vòng 5: TĂNG MẪU (n=80)

> Giao 2026-08-11, sau vòng 4 `d04a2d5`. **Không đổi code, không đổi cổng.**
> M4 chỉ CHẠY và COMMIT. **KHÔNG chấm, KHÔNG mở `--out-key` / `--out-meta`.**

## 0. Vì sao chỉ tăng mẫu

Vòng 4 cho `typed_first` **16,7%** — lần đầu vượt ngưỡng 15%, vào vùng BIÊN.
Nhưng đó là **2/12**. Vòng 3 là **1/19** (5,3%). Ở n cỡ này khoảng tin cậy nuốt
trọn khoảng cách — **chưa kết luận được gì**.

| vòng | typed_first | n | tam/n |
|---|---:|---:|---|
| 3 | 5,3% | 19 | 1/19 |
| 4 | **16,7%** | **12** | **2/12** |
| cooc (đối chứng) | 11,1% | 9 | 1/9 — **cả bốn vòng** |

Cái ĐÃ chắc: hút-điểm `phố` (10/28 vòng 3) **biến mất** sau khi bỏ pivot `thành`.
Cái CHƯA chắc: tỷ lệ. Vòng này chỉ để biết 16,7% có thật không.

**Không vá gì thêm trước khi có số này.** Quyết định đầu tư vào bộ chọn trung
tâm bằng CRF (§19.4) phải dựa trên mẫu đủ, không phải 2/12.

## 1. Chạy — y hệt vòng 4, chỉ đổi `--n 30` → `--n 80`

```bash
PYTHONPATH=src .venv/bin/python -m pytest -m "not slow" -q   # kỳ vọng 421 passed

# (A) đơn vị WS-CRF
PYTHONPATH=src .venv/bin/python scripts/sample_hub_chain3_blind.py \
  --checkpoint checkpoints/network_v12_subst_isg_pw.pkl --n 80 \
  --out out/hub3_chain_v12_r5_samples.txt \
  --out-key out/hub3_chain_v12_r5_key.json \
  --out-meta out/hub3_chain_v12_r5_meta.json

# (B) ĐỐI CHỨNG — đơn vị ISG
PYTHONPATH=src .venv/bin/python scripts/sample_hub_chain3_blind.py \
  --checkpoint checkpoints/network_v12_subst_isg_pw.pkl --n 80 --ws-spans none \
  --out out/hub3_chain_v12_r5_noWS_samples.txt \
  --out-key out/hub3_chain_v12_r5_noWS_key.json \
  --out-meta out/hub3_chain_v12_r5_noWS_meta.json
```

`--seed` giữ **46** (mặc định) ⇒ 30 thuộc tính của vòng 4 nằm trong 80 của vòng
này, **so được từng dòng**. Không đổi seed.

(A) phải in `sidecar WS-CRF: 71379 span TỪ`; (B) phải in
`⚠️ CHẠY KHÔNG CÓ span WS-CRF`. Không thấy đúng dòng thì dừng.

Không bake lại sidecar. Thời gian dự kiến ~2 phút/bộ (vòng 4: 1m44s cho n=30;
khâu đắt là dựng index, không phải số thuộc tính).

## 2. Chấm + cổng — KHÔNG đổi

Câu hỏi: **các từ ở `MỚI` có mang THUỘC TÍNH đầu dòng không?**
Cổng: ≥35% wire · 15–35% biên · <15% đóng. **Không nới.**

## 3. Không làm

- Không chấm, không mở `key.json` / `meta.json`.
- Không đổi `--seed`, `--d-max`, ngưỡng cổng, hay bất kỳ tham số nào khác.
- Không sửa luật chọn trung tâm ngữ đoạn (đã biết chưa giải — TK_046 §19.4).
- Không bake lại sidecar · không wire `hub_spread` vào `generate`.

## 4. Giao nộp

Commit 6 file `out/hub3_chain_v12_r5*` + `out/hub3_chain.log`.
Commit message: **chỉ** số dòng mỗi bộ · số span sidecar · `n_typed_edges` mỗi
bộ · thời gian. **Không** nhận xét chất lượng.
