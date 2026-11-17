# RUNBOOK M4 — toy lớn: tìm TRẦN của ngưỡng nhiễu (chuỗi ba bước)

**Bối cảnh (M4 chỉ thấy repo, không thấy chat):** TK_046 §20–§58 vừa đưa chuỗi ba
bước từ «0 ca trên dữ liệu thật» lên chỗ chạy được tới **20 316 cha** trên ca thử
vàng chó/mèo/khỉ. `classrich` là **35 757 cha**. Câu chưa trả lời được: **cơ chế
có sống nổi ở quy mô VƯỢT corpus thật không, và trần ở đâu?**

Máy dev (Intel 8GB) train mức 60 000 câu wiki mất **58 phút** và không đủ RAM chạy
tiếp — nên giao M4.

## Việc

### 1. Train toy ở ba mức (phần đắt)

```bash
PYTHONHASHSEED=0 PYTHONPATH=src python scripts/probe_chain3_noise_curve.py \
  --pool data/text_corpus/viwiki.jsonl \
  --muc 60000,100000,150000,250000 \
  --min-count 0.5 --d-max 6 \
  --tmp out/noise_big --out out/noise_150k.json
```

- `out/noise_big/toy_noise_60000.pkl` **ĐÃ CÓ** (59 769 cha) — script tự bỏ qua
  nếu file tồn tại? **KHÔNG** — nó train đè. Muốn tiết kiệm thì bỏ `60000` khỏi
  `--muc`.
- ⚠️ Số mà script này tự in ra là của chuỗi **CŨ** (chưa có kênh thuộc tính §53,
  `rank="phieu_x_ctx"` §43, luật vế trái §58). **Bỏ qua chúng.** Phần cần giữ là
  checkpoint.

### 2. Chấm lại bằng ĐÚNG mặc định hiện tại

```bash
PYTHONHASHSEED=0 PYTHONPATH=src python scripts/cham_toy_lon.py out/noise_big
```

In ra bảng `cha · nhãn · khỉ hạng · ③ top-6`.

## Cần biết gì để đọc kết quả

Ca thử vàng là 5 câu (`data/text_corpus/toy_chain3.jsonl`) trộn với nhiễu thật:

```
loài chó là động vật · loài mèo là động vật · loài khỉ là động vật
nghịch ngợm như loài chó · nghịch ngợm như loài mèo
```

Chuỗi phải cho: `nghịch ngợm → {chó, mèo} → động → {chó, khỉ, mèo}`.
**`khỉ` chưa bao giờ đứng cạnh `nghịch ngợm`** — nó ra được là do suy luận, đó là
điểm của cả bài toán.

**ĐẠT** = cột `nhãn` là `động` (hoặc `vật`) VÀ `khỉ hạng` khác `—`.

Số đã có trên máy dev:

```
cha      118    337    895   3 069  10 287  20 316
nhãn    động   động   động   động   động    động
khỉ        2      2      2      3      4       5
```

## Cổng (nêu TRƯỚC khi chạy)

- **≥35 757 cha mà vẫn ĐẠT** ⇒ cơ chế chịu được quy mô `classrich`; nút thắt còn
  lại KHÔNG phải nhiễu. Ghi vào TK_046 §59.
- **Gãy trong khoảng 20 316–35 757** ⇒ trần nằm ngay dưới corpus thật; ưu tiên kế
  là nâng trần, không phải vá bước nào.
- **Gãy trên 35 757** ⇒ ghi mức gãy; đó là trần thật đầu tiên đo được của kiến trúc.

## Cấu hình mặc định đang chốt (TK_046 §57–§58)

```
head_rule=trai_du_dac_thu · head_rule_trai=phai_nhat
indeg_q=0 · gidf_q=0          ← lọc TẮT, quyết định «ưu tiên năng lực» của tác giả
rank=phieu_x_ctx · chu_gia_q=0,90 · hub_chain_boost=0 (đường sinh TẮT)
```

`scripts/cham_toy_lon.py` đọc thẳng `HCAConfig()` nên nó luôn dùng mặc định hiện
tại — không cần truyền tham số.

## Lưu ý vận hành

- Train mức 150 000–250 000 câu wiki có thể ngốn **>16 GB**; nếu OOM thì hạ
  `--muc` xuống và báo lại mức tối đa chạy được.
- `gc.disable()` đã bật sẵn trong các script (nạp checkpoint nhanh 1300×).
- Mọi tiến trình dài ghi log ra `out/*.log` — kiểm bằng `tail -f`.
