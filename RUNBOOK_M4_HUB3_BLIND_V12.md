# RUNBOOK M4 — dựng mẫu CHẤM MÙ bước 3 trên v12 (đo TUYỆT ĐỐI)

> Giao 2026-08-10, sau vòng 2 (`126f2e8`). Thay `RUNBOOK_M4_HUB_STEP3_V12.md`
> cho câu hỏi "cơ chế này có dùng được không".
> **M4 chỉ CHẠY và COMMIT. KHÔNG chấm, KHÔNG mở `--out-key`.**

## 0. Vì sao bỏ thước cũ

Hai vòng `probe_hub_step3_v12.py` đều **không diễn giải được**:

| | vòng 1 | vòng 2 |
|---|---:|---:|
| typed_first | 97,7% | 94,3% |
| **cooc (đối chứng)** | **88,1%** | **83,9%** |
| lift | 1,11 | 1,12 |

Cột đối chứng gần trần ⇒ thước không phân biệt được gì ⇒ lift vô nghĩa. Ngưỡng
`bậc-vào ≥3` chỉnh trên mini (2 231 cạnh) không chuyển được sang v12 (120 875).

Lỗi thiết kế là của Claude, không phải M4 chạy sai — M4 đã làm đúng khi **báo
bão hoà thay vì đọc lift**.

Bài học §13.43 áp dụng nguyên văn: *chuỗi phép đo SO SÁNH có thể chạy rất lâu mà
không ai biết mình đang ở đâu — phải có mốc TUYỆT ĐỐI trước.* Nên vòng này bỏ
thước proxy, chuyển sang **đọc tay mẫu**.

## 1. Chạy (một lệnh)

```bash
PYTHONPATH=src .venv/bin/python -m pytest -m "not slow" -q    # kỳ vọng 410 passed

PYTHONPATH=src .venv/bin/python scripts/sample_hub_step3_blind.py \
  --checkpoint checkpoints/network_v12_subst_isg_pw.pkl \
  --n 30 \
  --out out/hub3_blind_v12_samples.txt \
  --out-key out/hub3_blind_v12_key.json \
  --out-meta out/hub3_blind_v12_meta.json
```

Sinh ~60 dòng (30 seed × 2 nhánh, đã xáo). Mỗi dòng:

```
[H017]  mèo  →  chó  →  Bella, Shiba, Tạng, giống, Pháp
```

## 2. 🔴 Việc M4 KHÔNG làm

- **Không chấm.** Claude chấm mù ở phiên riêng rồi mới mở nhãn.
- **Không mở / không đọc / không tóm tắt `--out-key`** trong commit message.
  Nhãn nhánh lộ ra là hỏng cả phép đo.
- Không sửa `--seed` (46) hay `--n` để "mẫu đẹp hơn".
- Không wire `hub_spread` vào `generate`.

## 3. Cổng G-hub3-abs — ghi TRƯỚC khi có số

Chấm 3 mức mỗi dòng: `dung` (hub là LỚP thật của seed **và** thành viên cùng
lớp) · `tam` (thành viên mạch lạc nhưng nhãn sai quan hệ) · `hong`.

| # | điều kiện trên nhánh `typed_first` | kết luận |
|---|---|---|
| 1 | `dung + tam` ≥ **40%** | cơ chế có thực chất ⇒ đáng vá tiếp (lọc câu hỏi khỏi mining, sửa quan hệ nhãn) |
| 2 | 15% ≤ `dung+tam` < 40% | biên — cần ca cụ thể để quyết, không mở rộng |
| 3 | `dung + tam` < **15%** | **ĐÓNG hướng bước 3**, ghi âm tính vào TK_046 §18.H |

**Tự-kiểm thang chấm:** nếu `cooc` (đối chứng) đạt trong vòng **10 điểm** so với
`typed_first` thì hoặc thang chấm quá rộng rãi, hoặc tách kênh không mua gì —
phải nói rõ điều nào, không được đọc như thắng lợi.

## 4. Vì sao có cột đối chứng trong bộ mẫu

`cooc` = đồng hiện thuần, không dùng kênh typed. Nó là **bộ tự-kiểm của người
chấm**: đọc mù thì không biết dòng nào thuộc nhánh nào, nên không thể vô thức
chấm rộng tay cho nhánh mình muốn thắng. Đây là chỗ vòng 1–2 thiếu.

## 5. Giao nộp

Commit ba file `out/hub3_blind_v12_{samples.txt,key.json,meta.json}` +
`out/hub3_blind.log`.

Commit message ghi **duy nhất**: số dòng, số seed, `n_typed_edges`, thời gian
chạy. **Không** ghi bất kỳ nhận xét chất lượng nào — nó sẽ mồi cho người chấm.

## 6. Đối chứng mini (đã chạy, để M4 biết đầu ra trông thế nào)

12 seed × 2 nhánh trên `mini_sent_ep_qa.pkl` cho các dòng kiểu:

```
[H009]  động  →  vật  →  thực, sinh, phi, hiện, hóa
[H002]  gái   →  Nguyễn → nhà, Nguyễn, chúa, Nhạc, họ
[H005]  quan  →  Thích  → Đỗ
```

Chỉ để kiểm định dạng đầu ra — **không phải** chuẩn chất lượng.
