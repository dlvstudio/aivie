# TK_026 — WIRE nơ-ron NGHĨA (từ điển) vào retrieval — bước "DÙNG" còn thiếu của TK_016

| Field | Value |
|-------|-------|
| **ID** | TK_026 |
| **Status** | 🟡 BUILT + ĐO — net-ÂM trên bộ 6 prompt, **default OFF**. Probe cô-lập DƯƠNG (sense tách sạch) nhưng inject vào rerank bị **va-chạm fold** "Lý"→"lý". Ghi nợ + đường thật. |
| **Nguồn** | Tác giả: "có từ điển rồi mà, sao train từ điển chưa có tác dụng?" → chẩn: từ điển nạp rồi (36.764 sense + bind chạy) nhưng **đường sinh KHÔNG đọc sense** (lỗ wiring TK_016). |

## 1. Chẩn đoán "vì sao từ điển chưa có tác dụng" (bằng grep)

- `ingest_dictionary` + `bind_sense_resonance` ĐÃ chạy (`build_dict_checkpoint.py`) → 36.764 `Sense:word#k`
  + cạnh SENSE + co-act(sense↔định-nghĩa↔fact-corpus). **100% sense có cạnh co-act** (probe: median deg 17).
- Nhưng grep `hca/` + retrieval `parent_store`: **không hàm nào đọc `Sense:`/`EdgeKind.SENSE`** (chỉ thân
  `bind_sense_resonance`). Sinh gieo trên **token trần + cụm ISG** → đi thẳng nơ-ron cha, KHÔNG qua sense.
- → **Lỗ WIRING** (không phải data): chuỗi TK_016 D1′ *ingest → bind → DÙNG*; bước "DÙNG" chưa dựng.

## 2. Probe cô-lập (measure-first, trước khi wire)

Cộng-hưởng fact với hồ-sơ sense (co-act của định-nghĩa, curated):

| Fact | mây#1 (khí tượng) | mây#2 (song mây) |
| :--- | :---: | :---: |
| "Mây bao gồm giọt nước và băng lơ lửng" | **4.0** | 1.0 |
| "Đỉnh Mây Bạc thuộc rừng Cúc Phương" | 0.0 | 0.0 |

→ **Tách sạch tuyệt đối** cho từ CÓ nghĩa từ-điển (khác hẳn co-act corpus P2 nhiễu). Hồ sơ sạch vì =
từ-vựng ĐỊNH-NGHĨA. Ca "lý thuyết": nghĩa lý-thuyết bắt fact vật-lý rõ (2-3) vs dynasty yếu (0-1) —
tách được kiểu **cổng-âm**; NHƯNG `Sense:triều#1/2` (thuỷ-triều/triều-đình) = 0 cho mọi fact nhà Lý →
KHÔNG route dương được dynasty (tên riêng "Lý" vô-nghĩa-từ-điển).

## 3. Code (wire vào bồi-fact rerank)

- `parent_store.query_sense_profile(q_content, co_layer, vocab)`: TRƯỜNG nghĩa = gộp hồ-sơ co-act các
  `Sense:*` mà **lemma-đầu fold trùng token query** (chỉ mục `_sense_by_fold`). Lọc punct + hư-từ (gidf<3.0)
  + bỏ token query. Nguồn = định-nghĩa curated (sạch).
- `generate_hebbian._rank_extra_fact_candidates`: field ưu tiên **sense** (TK_026), fallback co-act (P2);
  cổng resonance≥frac×max rồi xếp stimulus. Cờ `knowledge_sense_resonance`.

## 4. ĐO (network_v4_full, `out/tk026_sense.log`) — net-ÂM

- ✅ "Mây là gì" → giữ "Mây bao gồm giọt nước" (= baseline, không hỏng như P2).
- ❌ **"Nhà Lý có gì" → "Tham khảo về vật lý . Sinh lý học người và động vật"** — TỆ HƠN baseline.
- ❌ "Triều Lý đã làm gì" → rác (productivity text).
- Guard 71/62/50 — KHÔNG đổi.

**Nguyên nhân (va-chạm fold, xác nhận đo):** query "Lý"(id 243) **fold về "lý"** → `_sense_by_fold["lý"]`
gọi MỌI nghĩa "lý*" (lý-thuyết, **vật lý, sinh lý**) → fact các nghĩa đó cộng hưởng field → đẩy lên.
Va-chạm xảy ra ở **tầng fold/token, TRƯỚC sense** — sense-machinery không cứu được.

## 5. Kết luận trung thực (3 cơ chế, chống ảo tưởng)

| Cơ chế | "Lý thuyết" | "Mây" | Net vs baseline |
| :--- | :---: | :---: | :---: |
| **baseline** (stimulus thuần) | leak | ✅ có định nghĩa | — (TỐT NHẤT) |
| P2 co-act (TK_025) | ✅ hết | ❌ mất định nghĩa | wash/âm |
| **TK_026 sense** | ❌ tệ hơn (vật lý) | ✅ giữ | **âm** |

→ **Cả hai resonance đều KHÔNG thắng baseline** — thất bại ĐỐI XỨNG. Gốc chung: đồng-tự "Lý" chỉ
tách được ở tầng NGHĨA, mà "Lý"(triều, tên riêng) **case-fold va vào "lý"(common)** — artifact
tokenize, upstream của sense. Từ điển CÓ tín hiệu sạch (probe §2) nhưng inject qua fold thì nhiễm.

**Default:** cả `knowledge_sense_resonance` lẫn `knowledge_coact_resonance` = **False** (baseline).
Cơ chế + probe + doc giữ làm nợ đo-được.

## 6. Đường thật cho "Lý" (ghi, chưa làm)

1. **Sense THỰC-THỂ tên-riêng**: tạo nơ-ron nghĩa cho "Lý-triều" (gom fact triều-đại), tách khỏi
   "lý"-common — từ điển không có, phải sinh từ corpus (cluster fact quanh Nhà/triều Lý). Việc xây.
2. **Sửa va-chạm fold ở query**: khi entity là tên-riêng-viết-hoa, KHÔNG fold sang common-noun sense
   (dùng cap_class như thuộc-tính, Luật 0) — nhưng "Mây" cũng viết hoa → cần phân biệt tinh hơn.
3. **Wire sense cho từ CÓ-nghĩa-từ-điển** (không tên riêng): Mây/phở/… net-dương ở nhóm này; chỉ bật
   khi entity không phải tên-riêng-fold-va-chạm. Đây là phần từ-điển "có tác dụng" thật, khả thi.
