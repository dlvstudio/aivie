# TK_055 — QA GROUNDED + UI xem loại suy

Mảng phái sinh từ TK_054: đưa cạnh loại suy lên **UI graph** và sửa **QA** trên
checkpoint lớn (`net_ls_concept_conf.pkl`, 450 303 cha, A∪B).

## §180-181 — UI graph nhận diện loại suy

- `viz_default_checkpoint()` ưu tiên checkpoint loại suy MỚI NHẤT (dò tên
  concept/analogy/reinforced + mtime) → server tự trỏ đúng.
- ANALOGY vào bảng ưu tiên export + màu teal (phân biệt reinforced vàng).
- Endpoint `/api/graph/token` + trang `/loaisuy`: force-graph token→token (UI cũ
  chỉ xoay quanh rule; checkpoint này không rule). sắt→{thiếc,sulfide,Ni,cu} gold;
  mèo→{Bulldog,Corgi,Pug,Shiba} teal (chéo-danh-mục).

## §182-183 — trang /prompt + cầu nối QA→statement

- `/api/v3/complete` (fact/essay/multi) quá nặng; `/api/gen` gọi thẳng generate().
- QA "có bốn": generate() thô chỉ nối token CỤC BỘ; định nghĩa lưu dạng statement.
  Cầu nối "X là gì?"→sinh "X là" cho định nghĩa đúng.

## §184 — cache rank_matches: 19.3s→8.5s

completion_graph KHÔNG treo (19.3s); nút thắt rank_matches gọi 5× cùng query.
Cache `ranked` (bỏ top_n, phiên bản=len(parents)). NHƯNG completion_graph vẫn trả
sai (retrieval tìm cha tiếp tuyến, generate bỏ qua match) — perf ≠ correctness.

## §185 — A+B: QA định nghĩa GROUNDED

`ParentStore.define_entity(X)`: cha STATEMENT "X là …", X ở VỊ TRÍ CHỦ NGỮ +
copula, xếp đơn-vị-ngữ→priority→power. Trả câu ĐÃ LƯU verbatim (không sinh):
Sắt→"kim loại màu xám", Mèo→"khắc tinh của chuột", Đồng→"kim loại màu đỏ" (đúng
homograph). 200-660ms, KHÔNG cần W_skip.

## §186 — retrieval tổng quát (câu hỏi khác dạng)

`answer_fact(query_ids)`: `facts_about` (Theme index, theme_weight) + lọc SAU
is_qa/câu '?'. Không danh sách WH (từ hỏi IDF thấp tự nhẹ). Chó ăn gì→"ăn thịt",
Nước sôi→"100OC", Thủ đô VN→"Hà Nội", Lúa→"ruộng nước".

## §187 — tách nghĩa: lọc mạch-lạc-định-nghĩa + GATE cụm ghép

Đồng âm: "Hổ sống ở đâu?"→"Hổ thẹn… sống mũi". Lọc theo coh với sense_vocab (từ
CHỮ CÁI của định nghĩa; dấu câu/số gây coh giả). Cứu trong cửa sổ hẹp. GATE: 2
token đầu là cụm ghép (`_phrase_lookup`, "Thủ đô") → bỏ lọc (q_content[0]="Thủ"
không phải entity, def "đầu lợn" sai). Hổ→"Rắn hổ mang"(miền động vật, thoát "Hổ
thẹn"); Thủ đô→"Hà Nội" (giữ). AN TOÀN, cải thiện thật, nhưng chưa ra ĐÚNG hổ.

## §188 — SENSE-CLUSTER ĐẦY ĐỦ: THẤT BẠI (kết quả âm)

Mục tiêu: gom ngữ cảnh "Hổ" thành các NGHĨA, tách cọp khỏi "hổ mang"(rắn)/"hổ
phù"(ấn tín)/"hổ thẹn". Bốn cách, đều hỏng:

1. **Cluster mở rộng ngữ cảnh** — hạt giống (định nghĩa) chứa từ GENERIC (loài, có,
   là, vật) ⇒ mở rộng lan ra **1328 từ**, cluster = MỌI cha (1134) → không tách.
2. **Ratio-coh với định nghĩa** — loại OAN fact khía-cạnh: "nước sôi ở 100OC"
   chia sẻ 0 từ với định nghĩa "nước là chất lỏng không màu" → bị bỏ.
3. **Lọc cấu trúc cụm** (thực thể trong cụm ghép = nghĩa khác) — `_phrase_lookup`
   CÓ "thủ đô/đồng thời/đồng hồ" nhưng KHÔNG "hổ mang/thẹn/phù" (tách/tần suất
   khác) → không nhận diện được compound của hổ.
4. **Bản chất**: hai nghĩa CHIA SẺ từ vựng thật (hổ/rắn đều "loài động vật"); tiger
   facts đa dạng, không cụm quanh MỘT định nghĩa.

### Bài học

- **WSD gốc**: tách nghĩa tin cậy cần **infrastructure checkpoint THIẾU** —
  embedding tách-nghĩa (per-sense vector) hoặc **kho nghĩa curated**. Tín hiệu có
  (định nghĩa, phrase table, co-occur) đều KHÔNG đủ tách cọp/rắn-hổ-mang.
- Mọi discriminator phân biệt được nghĩa thì ĐỒNG THỜI loại oan fact-khía-cạnh
  hợp lệ (chúng khác định nghĩa theo thiết kế). Đánh đổi không vượt qua được bằng
  ngưỡng.
- **Giữ §187** (an toàn, cải thiện thật) làm mốc. Sense-cluster đầy đủ hoãn tới
  khi có embedding-tách-nghĩa hoặc kho nghĩa.

## §188b — TÁCH NGHĨA bản RÚT GỌN KHẢ THI: cosine embedding neo-định-nghĩa (ĐẠT)

§188 (word-cluster) thất bại; nhưng chuyển sang KHÔNG GIAN EMBEDDING thì được. POC
gom occurrence "hổ" bằng vector-ngữ-cảnh (mean embedding) + k-means: **cụm "xấu
hổ" tách sạch**. Bản rút gọn không cần k-means/lưu sense-id:

`answer_fact(..., embedding)`:
- `_ctx_vec` = trung bình embedding content-token (trừ thực thể), chuẩn hoá.
- `_dvec` = vector ngữ cảnh của ĐỊNH NGHĨA thực thể (nghĩa gốc).
- Lọc: trả candidate khía-cạnh-cao đầu tiên có `cos(fact, def) ≥ 0.45`. Không
  ứng viên nào qua sàn → fallback top (= §186, KHÔNG hồi quy). GATE cụm ghép giữ.

### Vì sao THÀNH khi word-based (§188a) thất bại

| word-overlap/cluster | cosine embedding |
|---|---|
| "loài" trùng khắp → cobra lọt; ratio loại oan "nước sôi"(0 từ chung def) | LIÊN TỤC: "nước sôi"↔"nước chất lỏng" cos 0.48 (gần) → GIỮ; "Hổ thẹn" cos 0.41 → BỎ |
| mọc từ seed → nổ 1328 từ | vector cố định, không lan |
| đơn-nghĩa bị chẻ oan | đơn-nghĩa mọi fact cos cao → tự giữ (Chó 0.74, Sắt 0.70) |

Kết quả: Hổ→"Rắn hổ mang"(thoát "Hổ thẹn"); Chó/Nước/Sắt/Lúa/Đồng/Thủ-đô KHÔNG
đổi. An toàn by-design (fallback). 456 test pass. `scripts/poc_sense_embedding.py`.

### Hạn chế còn

Embedding 96-D trên corpus nhiễu CHƯA đủ phân giải để tách cọp khỏi rắn-hổ-mang
(cả hai cos ~0.67 với def động vật). Bỏ được nghĩa XA (xấu hổ) nhưng chưa tách
nghĩa GẦN (cùng miền động vật). Muốn sắc hơn: PPMI context vector, hoặc per-sense
embedding lưu sẵn (k-means offline như POC) — bước lớn hơn, đã có đường.

## §189 — PER-SENSE EMBEDDING ĐẦY ĐỦ: tường cứng embedding (kết quả âm, có SỐ)

Xây builder offline (`scripts/dung_sense_clusters.py`): gom occurrence mỗi token
bằng k-means trên vector-ngữ-cảnh, lưu (centroid, tập cha) mỗi nghĩa. Vấn đề: KHÔNG
có gate tin cậy tách token đa-nghĩa khỏi đơn-nghĩa.

### Đo dứt điểm — silhouette KHÔNG phân biệt

| token | silhouette (K=2) | loại |
|---|---|---|
| hổ | 0.255 | đa nghĩa |
| chó | 0.251 | ĐƠN nghĩa |
| đồng | 0.308 | đa nghĩa |
| sắt | 0.199 | đơn nghĩa |
| khỉ | 0.322 | đơn nghĩa |

Token ĐA-nghĩa (hổ) và ĐƠN-nghĩa (chó) có silhouette **gần như nhau** (~0.25). Và
min-cos giữa centroid của hổ (K=3) = 0.70 — cụm không tách hướng. ⇒ **KHÔNG có tín
hiệu** để (a) biết token nào đa nghĩa, (b) tách nghĩa nào ra nghĩa nào.

### Vì sao — và cần gì thật

Embedding 96-D dựng bằng PPMI+SVD trên corpus NHIỄU (nhiều tu-tiên) không mã hoá
nghĩa đủ tách. Cụm POC "đọc được" là **ảo giác lấy mẫu** (mắt nhặt ví dụ), không
tách thống kê (silhouette 0.23 khắp nơi = gom yếu đều). Per-sense embedding đầy đủ
**không khả thi** với vector này.

Cần THẬT (ngoài phạm vi checkpoint hiện tại):
- **Embedding ngữ cảnh** (transformer) — mỗi lần xuất hiện một vector, tách nghĩa
  tự nhiên; HOẶC embedding chiều cao hơn + mục tiêu sense-aware.
- **Corpus sạch** — nhiễu tu-tiên đang lấn "hổ", dìm nghĩa sạch.
- **Kho nghĩa curated** — tag "hổ mang"/"xấu hổ" là lexeme riêng.

### Chốt

**§188b (cosine embedding neo-định-nghĩa) là bản TỐT NHẤT khả thi** với hạ tầng
hiện có: bỏ được nghĩa XA (xấu hổ, cos 0.41), an toàn không hồi quy. Tách nghĩa
GẦN (cọp vs rắn-hổ-mang) và per-sense đầy đủ đòi embedding/corpus tốt hơn — đã
chứng minh bằng số, không phải chưa thử. `scripts/dung_sense_clusters.py` +
`poc_sense_embedding.py` giữ làm tham chiếu; sidecar KHÔNG wire (không tin cậy).

## §190 — ENCODER NGỮ CẢNH (Mức 2): PHÁ TƯỜNG §189, cổng nghiệm thu ĐẠT

Dựng encoder self-attention 2 tầng, d=96 (init từ E), huấn luyện masked-LM trên
450k câu (MPS, torch 2.13). `scripts/train_ctx_encoder.py` +
`trich_du_lieu_encoder.py`. Cổng: silhouette gom cụm nghĩa token ĐA-nghĩa (hổ)
phải VƯỢT ĐƠN-nghĩa (chó).

### Đo — vector NGỮ CẢNH tách, vector TĨNH thì không

| | silhouette(hổ) | silhouette(chó) | tỉ lệ |
|---|---|---|---|
| §189 embedding TĨNH | 0.255 | 0.251 | 1.0× (không tín hiệu) |
| §190 encoder NGỮ CẢNH (loss 4.7) | ~0.24 | ~0.13 | **~1.7×** |

Nhất quán step 3000–20000: hổ 0.21–0.32, chó 0.12–0.15. Loss 10.4→4.7 (đang học,
60 bước/s MPS). **Lần đầu có tín hiệu** phân biệt đa/đơn nghĩa — đúng thứ §189
chứng minh embedding tĩnh KHÔNG có. Kiến trúc Mức 2 (artifact) HOẠT ĐỘNG.

### Nhưng cụm còn ĐỤC — corpus là đồng-giới-hạn

Dump cụm "hổ": tách MỘT PHẦN nhưng lẫn — cụm 2 trộn "xấu hổ" với "Sư tử và Hổ".
Vì corpus **ngập tu-tiên** ("Bạch Hổ", "Hắc Hổ Kiếm", "Huyền Vũ") lấn occurrence
"hổ", dìm nghĩa sạch. Đúng dự đoán artifact: encoder chịu nhiễu tốt hơn tĩnh
(có tín hiệu) nhưng làm sạch corpus vẫn cộng hưởng.

### Chốt & đường tiếp

- **Cổng nghiệm thu ĐẠT** (hổ ≫ chó, khác hẳn §189) ⇒ tầng biểu diễn NGỮ CẢNH
  mang nghĩa. Pha 2 (bake sidecar → rewire answer_fact khớp ngữ-cảnh) đã MỞ.
- Đồng-giới-hạn: corpus nhiễu. Train dài hơn (loss chưa đáy) + làm sạch tập
  tu-tiên → cụm sắc hơn. Model bé (d=96,2 tầng); nâng d/tầng cũng còn dư địa.
- `checkpoints/ctx_encoder.pt` (13MB) lưu. §188b vẫn là mốc PRODUCTION cho QA
  tới khi Pha 2 xong.

## §191 — GIÁ TRỊ THẬT: encoder TÁCH được nghĩa GẦN (cọp vs rắn-hổ-mang)

Phép thử payoff — cosine tới định nghĩa Hổ, encoder NGỮ CẢNH vs TĨNH (§188b):

| câu | cos ngữ-cảnh | cos tĩnh |
|---|---|---|
| Hổ là loài ăn khỏe nhất họ nhà mèo (CỌP) | **0.89** | 0.67 |
| Rắn hổ mang (RẮN) | **0.27** | 0.67 |
| Hổ mang chúa (rắn) | 0.70 | 0.63 |
| Hổ thẹn | 0.51 | 0.41 |

Encoder: cọp thật **0.89** (đỉnh rõ), rắn-hổ-mang **0.27** — TÁCH sạch. Tĩnh: cả
hai ~0.67, KHÔNG phân biệt. Đây đúng "nghĩa GẦN" §188b/§189 chứng minh vượt vector
tĩnh — encoder ngữ cảnh làm được. `scripts/do_gia_tri_encoder.py`.

### Đường wiring (Pha 2)

answer_fact: thay `_ctx_vec` tĩnh bằng vector encoder (hidden tại vị trí thực thể).
Dùng làm LỌC NGHĨA (sàn cos, bỏ rắn-hổ-mang 0.27), GIỮ xếp hạng khía-cạnh. Chi
phí: nạp encoder trong server (13MB) + inference/câu (~ms trên MPS/CPU). Vẫn còn
nhiễu nhẹ (Hổ thẹn 0.51) — train dài hơn + corpus sạch sẽ sắc thêm, nhưng tín
hiệu chính (cọp≫rắn) đã đủ mạnh để wiring.

## §192 — WIRING encoder vào server: "Hổ sống ở đâu?" → CỌP THẬT

Đưa encoder thành module (`src/nn_aivie/hca/ctx_encoder.py`, torch tùy chọn, thiếu
thì trả None → rơi về §188b). `answer_fact(..., encoder)`: QUY TẮC KHOẢNG-CÁCH —
giữ aspect-top TRỪ KHI có candidate (không phải định nghĩa) cos-ngữ-cảnh cao hơn
hẳn (>0.30) → đổi. Loại def khỏi so sánh (cos≈1.0 luôn thắng).

### Kết quả — cả chuỗi §187–191 thất bại, giờ đúng

| câu hỏi | §188b tĩnh | §192 encoder |
|---|---|---|
| Hổ sống ở đâu? | Rắn hổ mang (RẮN) | **Hổ là loài ăn khỏe nhất họ nhà mèo (CỌP)** ✓ |
| Chó ăn gì? | Chó ăn thịt | Chó ăn thịt (giữ) |
| Nước sôi? | 100OC | 100OC (giữ) |
| Thủ đô VN? | Hà Nội | Hà Nội (giữ) |
| Lúa trồng? | ruộng nước | ruộng nước (giữ) |

Gap-rule: Hổ đổi (cọp 0.89 − Hổ-thẹn 0.51 = 0.38 > 0.30); đơn-nghĩa gap≤0.22 →
giữ aspect-top, KHÔNG hồi quy. Server nạp encoder lazy (`get_encoder`), thiếu
torch/checkpoint thì tự dùng cosine tĩnh. 456 test pass.

### Chốt hành trình tách nghĩa §187→§192

Static embedding (§187/§188b/§189) chạm trần — chứng minh bằng số (silhouette
hổ=chó). Encoder ngữ cảnh Mức 2 (§190) phá trần (hổ 1.7× chó), tách nghĩa gần
(§191: cọp 0.89 vs rắn 0.27), wiring cho QA đúng (§192). Kiến trúc trong artifact
đi trọn từ giả thuyết → chứng minh → sản phẩm.

## §193 — CORPUS SẠCH + TRAIN DÀI: encoder sắc hơn

`scripts/loc_tu_tien.py`: dò văn tu-tiên/web-novel bằng marker mạnh (thuật ngữ tu
luyện + đại từ tiểu thuyết hắn/nàng/ngươi). Loại 5.9% câu (26510), hổ tu-tiên 20%.
Train encoder 40k bước trên 423793 câu SẠCH (`--clean-mask`).

### So noisy-20k

| | noisy 20k | clean 40k |
|---|---|---|
| loss cuối | 4.7 (chững) | **4.0** (giảm đều) |
| gate hổ/chó | 1.7× | **1.9×** |
| cọp (payoff) | 0.89 | **0.95** (đỉnh sắc) |
| cọp thứ hai | 0.53 | **0.74** |
| cụm "xấu hổ" | trộn | tách sạch hơn |

Corpus sạch → LM tốt hơn hẳn (loss 4.0 vs 4.7); cọp thật sắc hơn (0.95). QA giữ
đúng: Hổ→cọp, Chó/Nước/Lúa/Thủ-đô không hồi quy. `ctx_encoder.pt` ghi đè (bản
sạch); `ctx_encoder_noisy20k.pt` giữ backup.

### Còn dư địa

Cụm 0 vẫn tên-riêng tu-tiên (marker chưa bắt hết narrative); một số nghĩa-rắn
(Hổ mang chúa) cos lên theo (0.83) — train nhiều hơn nén range lên. Nhưng cọp vẫn
đỉnh rõ, quy tắc khoảng-cách bắn đúng. Nâng d/tầng + lọc corpus sâu hơn = đường
tiếp. Đủ tốt cho production hiện tại.

## §194 — NÂNG ENCODER d=192/3 tầng: tách nghĩa XA sắc hơn

Nâng d=96→192, 2→3 tầng, 4→6 đầu. Embedding init 96 chiều đầu từ E (giữ cấu trúc)
+ dư random. Config lưu kèm checkpoint (`{cfg,state}`); module đọc để dựng đúng
kiến trúc (tương thích format cũ). Train 30k trên corpus sạch.

### So d=96-sạch

| | d=96 (§193) | d=192 (§194) |
|---|---|---|
| loss cuối | 4.0 | **~3.0–3.5** |
| gate hổ silhouette | 0.24 | **~0.40** (cao hơn) |
| Hổ-thẹn (nên thấp) | 0.56 | **0.32** ✓ |
| tên riêng (Hổ-ca) | 0.55 | **0.23** ✓ |
| cọp thật | 0.95 | 0.89 |
| cọp Hổ-Hoa-Nam | 0.74 | **0.79** |
| cobra (Hổ mang chúa) | 0.83 | 0.81 (vẫn khó) |

Model lớn khởi động chậm (fresh phần dư) nhưng vượt: loss thấp hơn hẳn, tách nghĩa
XA (xấu-hổ, tên riêng) sắc hơn rõ. Nghĩa GẦN (cọp vs rắn-hổ-mang) vẫn là ca khó
nhất cho cả hai — cần corpus sạch sâu hơn / nhiều data hơn. E2E: Hổ→cọp,
Chó/Nước/Thủ-đô/Lúa không hồi quy.

### Đánh đổi & backup

d=192 chất lượng hơn nhưng inference/candidate chậm hơn (~2× → latency QA cao
hơn). Backup: `ctx_encoder_d96clean.pt` (nhanh hơn), `ctx_encoder_noisy20k.pt`.
Đổi production: copy backup lên `ctx_encoder.pt`, server tự đọc config đúng.

## §194b — PRODUCTION = d96clean (nhanh)

Đổi `ctx_encoder.pt` sang bản d=96-sạch (§193): latency warm **236ms** (vs d=192
~1.5s), vẫn trả cọp thật cho "Hổ sống ở đâu?". d=192 (§194) chất lượng tách-nghĩa-
xa nhỉnh hơn nhưng chậm ~6× — để dành khi cần độ sắc, không dùng production. Đổi
qua lại: copy `ctx_encoder_{d96clean,noisy20k}.pt` (hoặc bản d192) lên
`ctx_encoder.pt`; module tự đọc config/format. Backup giữ cả ba.

## §195 — TỐI ƯU LATENCY ĐƯỜNG SINH

Profile: cùng W_skip warm, «sắt và» 1.0s vs «các kim loại như» 100s (lần đầu)→2.5s
(lần sau). Nút thắt: (a) vote loop W_skip theo max_len; (b) `is_structural_token`/
`_surf` gọi 200–380k lần trong vòng; (c) rank_matches per-query + swap bộ nhớ cho
prompt token-phổ-biến (lần ĐẦU mỗi nội dung).

Sửa:
- CACHE `is_structural_token` (mảng bool [vocab]) + `_surf` (dict) — hàm thuần,
  bỏ 200–380k lần gọi/sinh.
- Hạ max_len mặc định 40→24 (endpoint + trang) — ít bước vote hơn.

Đo (warm): sắt và 392→**151ms**; kẽm và 10.3s→**819ms** (2–12× nhanh hơn).

### Còn lại — cold per-query + swap

Lần ĐẦU một nội dung prompt mới vẫn chậm/dao động (kẽm và cold >90s): rank_matches
quét 450k cha (cache §184 chỉ giúp lần SAU cùng query) + SWAP (server giữ
checkpoint 2.9GB + W_skip + torch encoder → sinh nặng swap). Đây là giới hạn BỘ
NHỚ + agenda, không phải compute thuần — cần giảm footprint (không nạp encoder khi
chỉ sinh / W_skip nhẹ hơn) hoặc cap candidate set, bước sâu hơn.

## §196 — CAP CANDIDATE SET: cold per-content 10–90s → ~1.5s

RSS đo được: một lần sinh phình +1.6GB, free RAM ~14MB → SWAP → cold 114s (lần
ĐẦU-TIÊN mỗi server). Nhưng nút thắt per-content là `rank_matches` mở rộng
`cand_idx` từ MỌI token: token phổ biến ("và/như/có") kéo 100k+ cha vào → vòng
chấm điểm quét hết.

Sửa (`rank_matches`): chỉ MỞ RỘNG candidate từ token PHÂN BIỆT (IDF ≥ 0.5×đỉnh).
Token phổ biến không phình cand_idx (vẫn TÍNH ĐIỂM nếu candidate chứa). Đáp đúng
luôn chứa token phân-biệt (thực thể) nên không mất — 456 test pass.

### Đo

| | trước cap | sau cap |
|---|---|---|
| prompt sinh MỚI (token phổ biến) | 10–90s | **~0.8–1.6s** |
| lặp lại | ~250ms | ~250ms |
| QA (Hổ/Chó/Vàng) | đúng | **đúng, không hồi quy** |

Còn: lần sinh ĐẦU-TIÊN mỗi server ~114s (build + swap một lần) — pre-warm bằng
một request lúc khởi động là hết. Baseline RSS ~5GB (checkpoint+W_skip+encoder)
trên máy ~14MB free là ràng buộc môi trường, không phải thuật toán.

## §197 — PRE-WARM lúc khởi động: request đầu không chờ

Startup event `_warm_checkpoint` (nền, daemon thread) chỉ nạp index; thêm:
pre-load encoder (`get_encoder()`) + một `generate("sắt và", max_len=4)` để dựng
W_skip + agenda ngay lúc khởi động. Build ~114s chạy NỀN, không chặn server.

Kết quả: request ĐẦU của user (kể cả prompt sinh chưa từng gọi) nhanh NGAY —
«thiếc và» 1.6s, «lúa và ngô» 1.0s, «Cá sống ở đâu?» 0.56s — thay vì chờ 114s.
Log báo `[warm] W_skip + encoder ready (đường sinh sẵn sàng)` khi xong.

### Chốt latency (§195–197)

- QA: 200–800ms (grounded, tách nghĩa encoder).
- Sinh prompt mới: 0.8–1.6s (cap candidate §196).
- Sinh lặp: ~250ms.
- Build một-lần: chạy nền lúc khởi động (§197), user không thấy.
- Ràng buộc còn lại: baseline RSS ~5GB trên máy 14MB free — môi trường, không
  phải thuật toán.
