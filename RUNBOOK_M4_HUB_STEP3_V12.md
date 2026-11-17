# RUNBOOK M4 — đo BƯỚC 3 (lan ngược từ nhãn) trên v12

> Giao 2026-08-10. Máy dev (Intel 8GB) không nạp nổi ckpt v12 (~16GB RAM).
> Đọc `TK_046_ANALOGY_FRAME_TENSOR.md` §18.H · commit `e4324de`, `d85b62f`.

## ⚠️ ĐÂY LÀ VÒNG 2 — CHẠY LẠI, không phải việc đã xong

Vòng 1 (`24f79e3`) đã chạy: `n_typed=89230` · lift **1,11** · cổng 3 ❌ ·
`mèo ↛ Vương`. Cảm ơn — **số đó lộ ra hai khiếm khuyết đã vá** (`d85b62f`):

| # | vòng 1 đo được | đã vá |
|---|---|---|
| 1 | 11/15 nhãn mạnh nhất là **hư từ** (`một` 2846, `vậy` 1028, `thế` 1021) | `_mine_via_pivot` lấy token LIỀN KỀ sau pivot ⇒ `X là MỘT động vật` cho nhãn `một`. Nay lấy **trung tâm ngữ đoạn** = gidf cao nhất trong 3 token kế |
| 2 | `mèo`→hub `thú` **đúng**, nhưng thành viên `['hứng','yêu','thích',…]` | `reverse_peers` (bước 3) còn trộn kênh bằng hệ số hằng `×4.0`. Nay tách kênh như `score_hubs_from` |

⇒ **Chạy lại y hệt lệnh §1.** Cả hai vá nằm trong code, không cần đổi tham số.
Mini sau vá: lift 1,44 → **2,64**, cả 4 cổng đạt (§4).

Khiếm khuyết #3 **chưa vá** — ngưỡng bậc-vào là HẰNG, có thể vẫn bão hoà ở v12.
Xem cảnh báo §4 (đọc cột `cooc` TRƯỚC khi đọc lift).

## 0. Đề bài (một dòng)

Chỉ đạo tác giả: lấy nơ-ron W_skip làm **nhãn** rồi lan **ba bước**:

```
nghịch ngợm → {chó, mèo} → động vật → {chó, mèo, KHỈ}
                            ^nhãn      ^bước 3 = ứng viên CHƯA từng thấy = sáng tạo
```

Bước 1–2 đã có trong `analogy_la`. **Bước 3 chưa wire.** `hub_spread` từng bị ghi
là "synthetic ĐẠT, trên `isg_pw` CÂM" (§18.H) — nay đã tìm ra nguyên nhân và vá:

> `score_hubs_from` **cộng hai kênh khác thang đo** vào một điểm (`boost=8.0`).
> Kênh typed (`A·là/như·B`) đếm câu định nghĩa ⇒ **không đổi** theo cỡ corpus.
> Kênh cooc (W_skip) ⇒ **tăng tuyến tính**. Đo trên ví dụ tác giả: thêm 20 000
> câu đời thường thì `xóm` lên 9 529 còn `động` giữ nguyên 3,0 — hệ số nhân hằng
> nào cũng thua. Vá: xếp hạng theo **(kênh, điểm)**, `channel="typed_first"`.

**Câu hỏi cho M4: vá đó có sống ở quy mô v12 không?**

## 1. Chạy

```bash
PYTHONPATH=src .venv/bin/python -m pytest -m "not slow" -q     # kỳ vọng 410 passed

PYTHONPATH=src .venv/bin/python scripts/probe_hub_step3_v12.py \
  --checkpoint checkpoints/network_v12_subst_isg_pw.pkl \
  --out out/hub_step3_v12_isg_pw_r2.json
```

Nếu quá chậm: hạ `--n-seeds` (mặc định 200). **Khâu đắt là dựng index**
(`_mine_via_pivot` quét toàn bộ 430k cha), không phải phần chấm — hạ `n-seeds`
không cứu được khâu đó. Trên mini: index 2,5s, chấm 0,1s/cột.

## 2. Thước — nằm TRONG đồ thị, không chấm cảm tính

| thước | định nghĩa |
|---|---|
| **hub có phải NHÃN** | **bậc-vào trên cạnh typed** ≥3 — tức có ≥3 chủ thể khác nhau nói `X là <hub>`. `động vật` cao; `xóm` ≈0. Không ai phán xét. |
| **có sáng tạo** | thành viên lan ngược từ nhãn mà cặp (thành viên, seed) **chưa từng đồng hiện** |

Ba cột đối chứng cùng bộ seed đóng băng (`--seed 46`): **`typed_first`** (mới) ·
`mixed` (hành vi cũ) · `cooc` (đồng hiện thuần).

## 3. Cổng G-hub3 — ghi TRƯỚC khi có số

| # | điều kiện | nếu trượt |
|---|---|---|
| 1 | `n_typed_edges` ≥ 1 000 | kênh typed không tồn tại ở v12 ⇒ **dừng**, mọi cột dưới vô nghĩa |
| 2 | `typed_first.ty_le_nhan` ≥ 30% | hub đầu bảng phần lớn không phải nhãn ⇒ hướng chết |
| 3 | lift `typed_first / cooc` ≥ 1,5× | tách kênh không mua gì ở quy mô thật |
| 4 | `ty_le_co_thanh_vien_moi` ≥ 20% | bước 3 không sinh ứng viên mới ⇒ không có sáng tạo để nói |

Script tự in `CỔNG {...}` và ghi vào JSON — **không tự chỉnh ngưỡng để cho đạt**.

## 4. Mốc đối chứng đã có trên mini (`out/hub_step3_mini_sent_ep_qa.json`)

**Vòng 2 (2026-08-10, sau khi vá `_mine_via_pivot` + `reverse_peers`):**

| cột | tỉ lệ nhãn | có thành viên mới |
|---|---:|---:|
| `typed_first` | **43,2%** | 88,5% |
| `mixed` (cũ) | 31,1% | 91,9% |
| `cooc` | **16,4%** | 97,4% |

lift = **2,64** ⇒ **cả 4 cổng ĐẠT trên mini**.

Vòng 1 (trước vá) để đối chiếu: 62,6% / 56,8% / **43,5%**, lift **1,44**, cổng (3)
trượt. Điểm mấu chốt không phải cột đầu tăng — mà **cột ĐỐI CHỨNG tụt 43,5% →
16,4%**: trước đây thước bão hoà nên không phân biệt được gì.

⚠️ **Vẫn phải soi trên v12 — hai rủi ro chưa loại:**

1. **Ngưỡng `bậc-vào ≥3` là HẰNG, có thể lại bão hoà ở v12.** Vòng 1 trên v12 cho
   `cooc` = 88,1% (gần trần) vì v12 có 89 230 cạnh typed so với 2 231 ở mini.
   Vá mining làm đổi số cạnh, nhưng **không đảm bảo hết bão hoà**. → Đọc cột
   `cooc` TRƯỚC: nếu nó > 60% thì thước lại hỏng, **báo lại** thay vì đọc lift.
2. **Đọc tay `top_labels`.** Mini vòng 1: `một(96) phố(50) là(44) người(41)` — hư
   từ. Vòng 2: `di(17) phố(14) tinh(12) phần(11) tâm(11) Thăng(10)` — đã ra từ nội
   dung. Trên v12 kiểm lại: nếu `một`/`những`/`vậy` còn ở đỉnh thì vá mining chưa
   ăn ở quy mô thật.

⚠️ `mèo`/`chó` không có hub nào trên mini (không có cạnh typed đi ra) — v12 vòng 1
thì CÓ (`mèo` → `thú`). Kiểm lại ở vòng 2.

## 5. Ca cụ thể — hub có chặn được `con mèo là Vương` không?

M4 đã báo (`7e37626`): `analogy_la` trên v12 cho `con mèo là → Vương`,
`nhà Trần là → Huyết` — gán **tên người** cho con vật và triều đại. Cơ chế chặn
lớp (`lop_trai`/`Lbl`) đã bị gỡ ở §18.H.

Script in mục `cases` cho `mèo, chó, Trần, học sinh, Việt Nam`. Đọc:

- hub của `mèo` là gì, bậc-vào bao nhiêu, và **`Vương` có nằm trong
  `thanh_vien_hub` không**.
- Nếu **không** → bằng chứng bước 3 chặn được ca này bằng **cạnh đã học**, không
  bằng bảng viết tay. Đó là kết quả đáng giá nhất của cả runbook.
- Nếu **có** → bước 3 không phân biệt được, ghi lại nguyên văn.

## 6. Giao nộp

**Vòng 2: ghi ra file KHÁC** để so được với vòng 1 —
`--out out/hub_step3_v12_isg_pw_r2.json` (đừng đè file vòng 1).

Commit `out/hub_step3_v12_isg_pw_r2.{json,log}` + một mục vào `TK_046` §18.H ghi:
`n_typed_edges` · ba cột (tỉ lệ nhãn / thành viên mới) · lift · phán cổng 1–4 ·
nhận xét đọc tay `top_labels` · kết luận ca `mèo`/`Vương`.

Commit message ghi gọn: `n_typed_edges` · lift · pass/fail từng cổng.

## 7. Không làm

- Không wire `hub_spread` vào `generate` ở vòng này — chỉ ĐO.
- Không chỉnh ngưỡng trong `probe_hub_step3_v12.py` cho cổng đạt; nếu thấy ngưỡng
  sai thì **báo lại kèm số**, đừng sửa rồi chạy tiếp.
- Không bake lại `Lbl:*` (đã gỡ, §18.H).
- Không đổi mặc định `generation_mode` (vẫn `w_skip`).
