# RUNBOOK TK_048 P105 — LỚP THAY THẾ trên **v12**: ba cách chia đơn vị + thước xác nhận

> Cho M4. M4 chỉ thấy repo, không thấy hội thoại ⇒ mọi thứ cần biết đều ở đây.
> Ngày giao: 2026-08-05. Chạy được **ngay**, không phụ thuộc việc nào khác.

## 1. Vì sao

Tác giả chỉ ra: mọi phân tích của TK_045/046/047 chạy trên **token đơn lẻ**, trong khi hạ tầng
nơ-ron **cụm** (`entity_id`, `concept_id`) có trong code mà **checkpoint có 0 cái**.

TK_048 đo ba cách chia đơn vị trên `mini`, và **tách được nguyên nhân**:

| đơn vị | phủ (mini) | phép ghép | |
|---|---:|---:|---|
| `token` — BPE ~âm tiết | **14,7%** | — | |
| `thuc_the` — TỪ + chuỗi **tên riêng** | 3,1% | **318** | ranh giới **thật** (lớp ký tự) |
| `cum` — TỪ + ghép **PMI≥3** | 0,2% | 2.407 | ngưỡng thống kê, **nhiễu** |

⇒ Giá phải trả nằm ở **chuyển từ ÂM TIẾT sang TỪ**, **không** ở phép ghép: `thuc_the` chỉ ghép
318 chuỗi mà phủ đã tụt 14,7% → 3,1%. Ghép thực thể **thêm độ chính xác, gần như không thêm chi
phí**; ghép PMI thì ngược lại.

**Và cơ chế này ĐÓI DỮ LIỆU**: M4 đã đo ở P104 rằng lên v12 phủ `token` **14,7% → 49,2%** và
`cum` **0,2% → 6,4%**. `thuc_the` **chưa được đo trên v12** — đó là việc này.

## 2. Con số quan trọng nhất đã có: **độ chính xác 89%**

Thước xác nhận (chia đôi corpus, suy lớp trên **A**, kiểm trên **B**) trên mini:

| đơn vị | lớp | đối chứng ngẫu nhiên |
|---|---:|---:|
| `token` | **89,2%** | 4,4% |
| `thuc_the` | **88,6%** | 0,0% |

⚠️ So với **cấp phép bộ ba** của TK_047: **23,4%** (mini) / **43,8%** (v12). Lớp thay thế
**chính xác gấp 2–4 lần**. Đó đúng là khác biệt "mẫu hình nhất quán" vs "một mẩu bằng chứng"
mà TK_048 §1 nêu — nay **đo được**, không còn là lập luận.

⇒ Vấn đề của cơ chế này **không phải độ chính xác**, mà là **PHỦ**. Đó là thứ P105 đo.

## 3. Lệnh

Chạy từ gốc repo, `PYTHONPATH=src` (không có `.venv`). **Ba lệnh, cùng checkpoint v12.**

```bash
for U in token thuc_the cum; do
  PYTHONPATH=src python scripts/probe_subst_class.py \
    --unit "$U" \
    --checkpoint checkpoints/network_v12_sent.pkl \
    --label-map out/token_labels_v12.json \
    --holdout \
    --out "out/subst_v12_${U}_holdout.json"
done
```

Rồi chạy lại **không** `--holdout` (dùng toàn bộ dữ liệu) để lấy con số **phủ** thật:

```bash
for U in token thuc_the cum; do
  PYTHONPATH=src python scripts/probe_subst_class.py \
    --unit "$U" \
    --checkpoint checkpoints/network_v12_sent.pkl \
    --label-map out/token_labels_v12.json \
    --out "out/subst_v12_${U}.json"
done
```

**Chạy hai lần mỗi cấu hình và so** trước khi đọc. Một lỗi `PYTHONHASHSEED` đã từng làm một
cổng **lật** giữa hai lần chạy cùng lệnh, trong khi các cổng khác đứng yên nên bảng trông vẫn
đúng.

### Ghi chú vận hành

- `--unit thuc_the` **bắt buộc** có `out/token_labels_v12.json` chứa nhãn `tên riêng` (M4 đã
  dựng ở P100). Nếu script báo `⛔ … không có nhãn 'tên riêng'`, **dừng và báo lại**.
- `--unit token` **không** cần bản đồ nhãn.
- `cum` và `thuc_the` phải tách từ bằng WS-CRF trên toàn bộ 430k cha — **chậm nhất**, và là chỗ
  dễ OOM nhất. Nếu OOM, **báo lại kèm mốc**; đừng tự giảm quy mô rồi im lặng, vì quy mô chính
  là biến đang đo.
- Nạp checkpoint có `gc.disable()` sẵn. Đừng bỏ.

## 4. CỔNG G-subst (nêu TRƯỚC — ĐỪNG SỬA)

1. **Phủ `thuc_the` trên v12 ≥ 12%.** Dưới ngưỡng đó, đơn vị-thực-thể **không đủ dùng** dù có
   sạch, và kết luận là **giữ mức token**.
2. **Không tụt độ chính xác**: thước xác nhận `lop` ≥ **70% TUYỆT ĐỐI** *và* ≥ **5×** đối chứng,
   cho **cả** `token` và `thuc_the`. (Mini: 89,2% / 88,6%.)
   ⚠️ Đọc **mức tuyệt đối TRƯỚC**, tỷ số sau. Bài học TK_047: *tỷ số lớn trên sàn gần 0 vẫn là
   gần 0*.
3. `thuc_the` phải **hơn `cum`** ở **cả** phủ lẫn độ chính xác. Nếu không, ranh-giới-thật không
   hơn ngưỡng-thống-kê và §7 sai.

## 5. Dự đoán ghi TRƯỚC (Claude)

- Phủ `thuc_the`: **12–25%**. Cơ sở: mini 3,1%; `cum` đã đi 0,2% → 6,4% khi lên v12, mà
  `thuc_the` xuất phát **cao gấp 15×** và thưa ít hơn.
- Độ chính xác giữ **≥85%** ở cả hai mức — nó đã cao trên mini với dữ liệu ít hơn nhiều.
- `thuc_the` **hơn** `cum` ở cả hai cột. Nếu **không**, tôi sai ở §7 và phải ghi lại.
- ⚠️ **Tôi đã sai theo hướng BI QUAN hai vòng liên tiếp** (P102, P103). Nếu phủ vượt **25%**,
  đó là lần thứ ba, và tôi ghi nó thành thiên lệch có hệ thống chứ không phải may.

## 6. Giao nộp

Commit `out/subst_v12_*.json` và `out/logs/probe_subst_class_*.log`.

Commit message **phải ghi**: phủ của **cả ba** mức đơn vị · thước xác nhận (**mức tuyệt đối**
và tỷ số) của cả ba · hai lần chạy **có giống nhau không** · và **vài vùng láng giềng mẫu** của
`thuc_the` để đọc bằng mắt.
