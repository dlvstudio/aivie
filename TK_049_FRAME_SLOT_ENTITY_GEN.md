# TK_049 — SINH KHUNG–ĐA-KHE TRÊN ĐƠN VỊ THỰC THỂ + LỚP THAY THẾ ĐÔNG CỨNG

> Mở 2026-08-06. Nối TK_047 (khung–khe, mở lại §20) + TK_048 (lớp thay thế; giao hẹp 70,6%
> trên v12 nhưng ép-1-khe không mua chất lượng).
>
> Chỉ đạo gốc của tác giả (TK_047): *"sửa cái đáng chạy mà đang hỏng (sinh)… nơ-ron nhãn /
> ngữ pháp / bất kỳ nơ-ron cần thiết… kiến trúc rõ… sáng tạo và không chép agenda."*
> Quan sát TK_048: *"đang token sang token… đã có cơ chế nơ-ron cụm — đã dùng chưa?"* → **chưa**.

## 0. Chẩn đoán đóng vòng đo (không đoán)

| Phát hiện | Hệ quả |
|---|---|
| Ép-1-khe hại vì **bể** sai, không vì ranker (TK_047 §12) | Đừng thêm tiêu chí chọn |
| Dữ liệu thu hẹp hại (−42→−14đ) | Hàm truyền có thật |
| Giao hẹp **70,6%** · tổ hợp **72×** (TK_048 §9b) | Substrate cấp phép **đủ** |
| `ep_giao − ep_bo_ba = −0,8đ` | Lọc lớp trên thủ tục chép+1-lỗ **vô giá trị** |
| `entity_id` / `concept_id` = 0 trong ckpt | Ý cụm của tác giả **chưa bao giờ vào graph** |
| Production = W_skip token | Đường sinh vẫn **không** phải khung–khe |

**Công thức đúng (TK_047 §1), chưa từng chạy:** *khung một chỗ, chất liệu nhiều chỗ* — điền
**nhiều khe** từ lớp thay thế trên đơn vị **trên token**, cạnh **đông cứng lúc bake**, không
tính lại rồi vứt mỗi lần probe.

## 1. Kiến trúc (nêu TRƯỚC khi code)

### 1.1 Train / bake (P106)

1. Phân đoạn cha → đơn vị **từ** (WS-CRF) + ghép chuỗi **tên riêng** (`thuc_the`, TK_048 §7).
2. Với mỗi đơn vị đa-token: spawn `Entity:…` (hoặc `Concept:…`) + `CO_ACTIVATION` tới token thành viên.
3. Dựng lớp thay thế (cùng định nghĩa TK_048: ≥3 bối cảnh chung, chuẩn hoá tần suất, top-k).
4. Ghi cạnh **`SYNONYM`** (kênh paradigmatic đã có, priority 0 — đọc tường minh) hai chiều giữa
   đại diện đơn vị (token đơn hoặc entity hub), trọng số = sim.
5. (P107) Hebbian: nếu A~B (`SYNONYM`) và A có cạnh tới P (tính chất) thì viết B→P — **chưa**
   làm ở P106; cổng riêng.

Checkpoint ra: `checkpoints/network_v12_subst.pkl` (không đè v12 gốc).

### 1.2 Decode (P106)

Thủ tục **không** phải W_skip primary và **không** phải ép đúng 1 khe:

1. `rank_matches(đề)` → cha khuôn (khác cha đề).
2. Khung = chuỗi emit của cha (bề mặt); ứng viên mỗi vị trí nội dung =
   `tri(a,b) ∩ hàng xóm SYNONYM(x)` (− cấm `số`) — tức **giao**, đọc từ graph đã bake.
3. Điền **mọi** khe có ≥1 ứng viên hợp lệ, theo độ chắc chắn giảm dần; WS-CRF hoàn tác nếu vỡ từ.
4. Meta bắt buộc: `n_thay`, `n_khe_de_xuat`, `trung_nguyen_van`.

Nhánh đo (cùng bộ đề rời, neo 40 mẫu như P103):

| nhánh | ý nghĩa |
|---|---|
| `chep_cau` | trần — phát nguyên văn |
| `ep_mot_khe_giao` | P103-style, 1 khe, licence giao (đối chứng thủ tục cũ) |
| `ep_da_khe_giao` | **thủ tục mới** — điền mọi khe giao được |

### 1.3 CỔNG G-frame (nêu TRƯỚC)

| # | điều kiện | nếu trượt |
|---|---|---|
| 1 | Bake: ≥1 cạnh `SYNONYM` trên ≥20% token nội dung có ctx≥3 | lớp không vào graph — dừng |
| 2 | Bake: ≥100 `Entity:` nếu `--unit thuc_the` | cụm vẫn ngủ — báo, không giả vờ |
| 3 | Smoke `ep_da_khe_giao`: tỷ lệ mẫu `n_thay≥2` ≥ **25%** | vẫn là chép+1-lỗ — sửa decode |
| 4 | Vòng chấm (P107): `ep_da_khe_giao − chep_cau` ≥ **−5đ** *và* tỷ lệ không-chép (`n_thay≥1`) ≥ **30%** | cơ chế chưa thắng thủ tục |

⚠️ Cổng (4) **không** lấy hep probe làm proxy. Hep đã 70% mà ép-1-khe vẫn −8đ.

### 1.4 Dự đoán ghi TRƯỚC

- Cổng (1)(2)(3) **đạt** trên v12 (phủ token lớp 49%; thuc_the 34%).
- `ep_da_khe_giao` sẽ **thấp hơn** `chep_cau` lúc đầu (−10…−20đ) vì nhiều khe = nhiều chỗ hỏng —
  nhưng tỷ lệ `n_thay≥2` phải cao; nếu thấp thì chưa đúng kiến trúc.
- Chỉ khi cổng (4) đạt mới nói “hướng khung–đa-khe sống”. Nếu (3) đạt mà (4) trượt nặng
  (<−20đ) ⇒ substrate vẫn chưa đủ cho đa-khe; quay lại Hebbian P107 trước khi scale.

## 2. Việc không làm

- Không đè `network_v12_sent.pkl`.
- Không thêm ranker chủ đề vào bể (đã =0đ).
- Không scale data mù săn hep 54% trên `bo_ba` token.
- Không coi W_skip primary là đường TK_049.

## 3. Runbook

`RUNBOOK_TK_049_P106.md` — bake + smoke đa-khe. Vòng chấm đầy đủ = P107 (sau khi cổng 1–3 đạt).

## 4. KẾT QUẢ P106 (2026-08-06) — cổng bake + smoke **đạt**; chất lượng chưa chấm

### Bake → `checkpoints/network_v12_subst.pkl` (local, gitignore)

| thước | giá trị | cổng |
|---|---:|---|
| `ty_le_token_co_synonym` | **59,0%** | ✅ ≥20% |
| `n_entity` | **222.309** | ✅ ≥100 |
| cạnh `SYNONYM` (hướng) | 244.886 | — |
| vocab 32.002 → 247.609 | (+Entity) | — |

Meta: `out/bake_subst_v12_meta.json`.

### Smoke `--arms chep_cau,ep_mot_khe_giao,ep_da_khe_giao` · n=40 · ckpt subst

| nhánh | `n_thay≥1` | `n_thay≥2` | trung nguyên văn | mean `n_thay` |
|---|---:|---:|---:|---:|
| `chep_cau` | 0% | 0% | — | 0 |
| `ep_mot_khe_giao` | 95% | 0% | 5% | 0,95 |
| **`ep_da_khe_giao`** | **95%** | **87,5%** ✅ | 5% | 53,9 |

Cổng (3) đạt mạnh. Mean ~54 khe/mẫu = thủ tục **đúng là đa-khe**, không còn chép+1-lỗ.
⚠️ Một số mẫu ~440 thay — bể giao rộng; P107 phải đo chất lượng (cổng 4) trước khi scale.

Artefacts: `out/frame_da_khe_smoke{,_key,_meta}.{txt,json}`.

### Việc kế

P107: chấm mù đủ (neo ~40 như P103) trên ba nhánh; cổng (4) = `ep_da_khe − chep ≥ −5đ`
*và* không-chép ≥30%. Nếu (3) đạt mà (4) trượt nặng → Hebbian property edges trước khi scale.

## 5. KẾT QUẢ P107 (2026-08-06) — đa-khe **thắng 1-khe**, cổng (4) **trượt hẹp**

Checkpoint: `network_v12_subst.pkl`. n=120/nhánh + 40 neo. Thang tuyệt đối chuẩn (không sửa chữ).

| | n | dung_duoc | dùng+tạm |
|---|---:|---:|---:|
| `chep_cau` | 120 | 39,2% | **53,3%** |
| `ep_mot_khe_giao` | 120 | 25,0% | **36,7%** |
| **`ep_da_khe_giao`** | 120 | 20,8% | **45,8%** |

| Đại lượng | giá trị |
|---|---:|
| ĐỘ TRÔI (neo) | **−7,5đ** (60,0% vs tham chiếu 67,5%) |
| `ep_mot − chep` | **−16,7đ** |
| **`ep_da − chep`** | **−7,5đ** |
| `ep_da − ep_mot` | **+9,2đ** |
| meta `ep_da` `n_thay≥1` / `≥2` | **92,5%** / **75,8%** |

Cổng (4): hiệu ≥−5đ ❌ (−7,5) · không-chép ≥30% ✅ → **TRƯỢT hẹp**.

Đọc theo §1.4: hiệu nằm vùng **−5…−20đ** = sống một phần. Đa-khe **rõ ràng hơn** thủ tục
1-khe (+9,2đ) dù mean ~28 thay/mẫu — không phải “càng thay càng chết”. Việc kế: Hebbian
property edges (**P108**) + siết bể (topk/min_share hoặc trần `n_thay`) trước scale.

Artefacts: `out/frame_da_khe_{samples,key,meta,cham,verdict}.*` · `RUNBOOK_TK_049_P107.md`.

## 6. P108 — Hebbian kế thừa tính chất + trần `max_thay=6` (tiền-đăng-ký)

P107 cổng 4 trượt hẹp (−7,5đ). Hai đòn nêu TRƯỚC khi có số:

1. **Bake Hebbian** trên `network_v12_subst.pkl`: A~B (`SYNONYM`) ∧ A→P (`CO`) ⇒ B→P
   (scale 0,45, max 8 đích/hướng). Ckpt: `network_v12_subst_h.pkl`.
2. **Trần đa-khe** `--max-thay 6` — P107 mean ~28 thay; siết để đỡ salad.
3. Decode đọc thêm top-16 CO graph vào `_co` (để Hebbian ảnh hưởng xếp ứng viên).

Cổng (4) lại trên n=120 + neo. Dự đoán: đạt sát (−5…0đ). Runbook: `RUNBOOK_TK_049_P108.md`.

## 7. KẾT QUẢ P108 (2026-08-06) — trần+Hebbian **không** đóng cổng 4; đa-khe **xấu hơn** P107

Bake Hebbian: `n_inherited=678.644` ✅ · mean `n_thay` 27,8 → **3,57** (max_thay=6) · `n_thay≥2` vẫn 75%.

| | n | dung_duoc | dùng+tạm | vs P107 |
|---|---:|---:|---:|---:|
| `chep_cau` | 120 | 45,0% | **55,8%** | +2,5đ |
| `ep_mot_khe_giao` | 120 | 30,0% | **48,3%** | **+11,6đ** |
| `ep_da_khe_giao` | 120 | 29,2% | **41,7%** | **−4,1đ** |

| Đại lượng | P107 | P108 |
|---|---:|---:|
| ĐỘ TRÔI | −7,5đ | **−7,5đ** |
| `ep_da − chep` | **−7,5đ** | **−14,2đ** ❌ |
| `ep_da − ep_mot` | +9,2đ | **−6,7đ** |

Cổng (4) **TRƯỢT** (xa hơn). Đọc: Hebbian+CO-graph **có vẻ giúp 1-khe** (−16,7→−7,5đ), nhưng
trần theo *số ứng viên* (chắc chắn ≠ đúng) + đa-khe **làm hại**. Việc kế: ablation
`max_thay` không Hebbian-CO · đổi thứ tự khe sang **độ tăng bám đề** (không theo |cand|) ·
hoặc `max_thay∈{2,3}`.

Artefacts: `out/frame_p108_*` · `out/bake_hebbian_v12_meta.json` · `RUNBOOK_TK_049_P108.md`.

## 8. P109 — xếp khe theo bám đề + ablation không CO-graph (tiền-đăng-ký)

P108 làm hại đa-khe. Chạy trên `subst` (không Hebbian), `--slot-rank bam --max-thay 3 --no-co-graph`.
Dự đoán: `ep_da−chep` về −5…−10đ. Runbook: `RUNBOOK_TK_049_P109.md`.

## 9. KẾT QUẢ P109 (2026-08-06) — **gần cổng 4 nhất**; vẫn trượt hẹp (−5,8đ)

| | n | dung_duoc | dùng+tạm | vs P107 | vs P108 |
|---|---:|---:|---:|---:|---:|
| `chep_cau` | 120 | 41,7% | **55,8%** | +2,5 | = |
| `ep_mot_khe_giao` | 120 | 26,7% | **44,2%** | +7,5 | −4,1 |
| **`ep_da_khe_giao`** | 120 | 36,7% | **50,0%** | **+4,2** | **+8,3** |

| Đại lượng | P107 | P108 | **P109** |
|---|---:|---:|---:|
| ĐỘ TRÔI | −7,5đ | −7,5đ | **−5,0đ** |
| `ep_da − chep` | −7,5đ | −14,2đ | **−5,8đ** |
| `ep_da − ep_mot` | +9,2đ | −6,7đ | **+5,8đ** |
| mean `n_thay` / `≥1` | 27,8 / 92% | 3,57 / 92% | **0,91 / 41,7%** |

Cổng (4): hiệu ≥−5đ ❌ (−5,8) · không-chép ≥30% ✅ → **TRƯỢT sát**.

Đọc: bỏ Hebbian-CO + xếp theo **độ tăng bám đề** + trần 3 **khôi phục** hướng P107 và
vượt nó. Đa-khe lại thắng 1-khe. Việc kế hẹp: siết thêm chọn ứng viên (hoặc `max_thay=2` /
chỉ thay khi gain≥2) để lấy ~1đ còn thiếu — **không** quay lại Hebbian-CO kiểu P108.

Artefacts: `out/frame_p109_*` · `RUNBOOK_TK_049_P109.md`.

## 10. P110 — min-gain=2 + max_thay=2

| | dùng+tạm | vs chep |
|---|---:|---:|
| `chep_cau` | 54,2% | — |
| `ep_mot` | 52,5% | −1,7đ |
| `ep_da` | **50,0%** | **−4,2đ** ✅ phần chất lượng |

ĐỘ TRÔI −7,5đ · `n_thay≥1` = **25,0%** ❌ (≥30%) · mean 0,39.

Cổng (4) **TRƯỢT** vì hết sáng tạo (siết quá). Chất lượng đã đủ (≥−5đ). Việc kế: giữ
`min-gain=2`, nới `max_thay=3` (P111).

## 11. KẾT QUẢ P111 — nới trần **không** tăng tỷ lệ thay; xác nhận nút thắt = `min-gain=2`

`ep_da−chep = −4,2đ` (như P110) · `n_thay≥1 = 25%` (y hệt). Max_thay 2→3 vô hiệu khi chỉ
25% mẫu có khe gain≥2. P112: `min-gain=1` + `max_thay=2` (giữa P109 và P110).

## 12. KẾT QUẢ P112 (2026-08-06) — cổng (4) **ĐẠT**

Cấu hình: `subst` · `--slot-rank bam --min-gain 1 --max-thay 2 --no-co-graph`.

| | dùng+tạm |
|---|---:|
| `chep_cau` | **55,0%** |
| `ep_mot_khe_giao` | 46,7% |
| **`ep_da_khe_giao`** | **51,7%** |

| Đại lượng | giá trị |
|---|---:|
| ĐỘ TRÔI | **−5,0đ** |
| **`ep_da − chep`** | **−3,3đ** ✅ (≥−5) |
| `ep_da − ep_mot` | **+5,0đ** |
| `n_thay≥1` / `≥2` | **41,7%** ✅ / 23,3% |

Cổng (4) **ĐẠT**. Chuỗi: P107 −7,5 → P109 −5,8 → P110/111 −4,2 (hết sáng tạo) → **P112 −3,3**.
Công thức sống: **bam + min-gain=1 + max_thay=2**, không Hebbian-CO.

Artefacts: `out/frame_p112_*`.

## 13. Việc kế sau cổng (4) — tiền-đăng-ký TK_050

P112 chứng minh **thủ tục** khung–đa-khe sống trên probe. Chưa phải sản phẩm sinh.

### Khoảng trống còn lại (đo được / đã biết)

| # | khoảng trống | bằng chứng |
|---|---|---|
| A | Production vẫn **W_skip** primary | `generate_hebbian.py`; TK_049 §0 |
| B | Decode vẫn **token**, dù bake có 222k `Entity:` | P106 spawn; gen chỉ đọc SYNONYM token |
| C | `n_thay≥2` chỉ **23,3%** (P112) | đa-khe thật còn mỏng — nhiều mẫu chỉ 1 thay |
| D | Cách chep còn **−3,3đ** | trần chép chưa đóng |
| E | Ckpt `subst` **local/gitignore** | M4 khác máy phải bake lại từ runbook |

### Lộ trình đề xuất (thứ tự cứng)

**P113 — đóng đường probe vào API sinh (wire)**  
Nhánh opt-in trong `generate` / CLI: `mode=frame_da_khe` dùng đúng cấu hình P112
(`bam · min-gain=1 · max_thay=2 · SYNONYM bake · cấm số`). Cổng: cùng bộ đề P112, lệch
dùng+tạm vs probe **≤ 3đ** (không hồi quy).

**P114 — đơn vị thuc_the lúc decode**  
Điền khe theo **từ/entity hub** (thay cả cụm `Lý Thái Tổ`), không chỉ âm tiết. Cổng:
`n_thay≥2` ≥ **35%** *và* `ep_da−chep ≥ −5đ` (giữ cổng 4).

**P115 — thu hẹp −3,3đ**  
Chỉ sau P113 ổn định. Ứng viên: lọc niên đại chặt hơn · cấm nhảy `Entity` khác lĩnh vực ·
không đụng lại Hebbian-CO kiểu P108 (đã đo hại).

**Không làm:** scale data mù săn hep; quay W_skip primary; coi hep probe = chất lượng.

Chi tiết vận hành: `RUNBOOK_TK_050_P113.md` + `TK_050_FRAME_GEN_PRODUCTIZE.md`.
