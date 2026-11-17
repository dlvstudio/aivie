# TK_056 — kNN-LM SINH: encoder ĐỌC HỘ, phá tường mô-hình-đếm

| Field | Value |
|-------|-------|
| **ID** | TK_056 |
| **Status** | 🟡 MỞ — hiện thực hoá TK_042 G13 bằng encoder §190–194 (TK_055). |
| **Nguồn** | Mục tiêu tác giả: chất lượng sinh ngang LLM. Hội tụ: chẩn đoán TK_042 (cần biểu diễn ngữ cảnh HỌC được) + encoder ngữ cảnh đã xây (TK_055 §190). |
| **Liên quan** | TK_042 §7.15 (G13 kNN-LM, lift@1 22× với `h` transformer), TK_055 §190–194 (encoder), TK_057 (datastore), TK_060 (thước perplexity). |

## Bức tường (TK_042, vẫn đúng)
Sinh hiện tại = mô hình ĐẾM (W_skip). Không điều-kiện-hoá ngữ cảnh dài → phân bố
next-token phẳng (entropy 6.42 bits). Cần biểu diễn ngữ cảnh NÉN, HỌC được — không
phải bảng đếm thứ chín.

## Cơ chế (kNN-LM, Khandelwal 2020, diễn dịch)
1. **Datastore** (TK_057): mỗi vị trí train → `(h = encoder LEFT-context, next_token)`.
2. Mỗi bước sinh/eval: `h` ngữ cảnh hiện tại → **k-NN** trong datastore → phân bố
   next-token của láng giềng (softmax theo khoảng cách).
3. **Nội suy**: `P = λ·P_kNN + (1−λ)·P_Wskip`. Encoder chỉ ĐỌC HỘ; đồ thị vẫn là bộ nhớ.

## Cổng nghiệm thu (TK_060)
- **bits/token held-out GIẢM** so W_skip đơn (thước chính, TK_042 tiếc chưa đo).
- Entropy next-token giảm; lift@1 của `h` > túi PPMI (TK_042: 22× vs 10×).
- Văn mượt hơn khi đọc mắt; không hồi quy QA/loại suy.

## Rủi ro
- Encoder masked (bidirectional) ≠ causal → dùng LEFT-context (causal-mask hoặc
  prefix). Chuẩn hơn = TK_059 (causal encoder).
- Train/infer mismatch; datastore lớn → TK_057 lo nén/ANN.

## Luật 0
Encoder train gradient (rời Luật-0-thuần) — nhưng đúng con đường TK_042 G13 tự vạch
("mượn `h`, không train lại đồ thị"). Dự án chuyển sang HYBRID: đồ thị + encoder nhỏ.

## POC (validated) — encoder h > túi-tĩnh > đếm

`scripts/poc_knn_lm.py` (nhẹ: enc_data.npz + ctx_encoder.pt, không cần checkpoint).
Datastore 445k vị trí (LEFT-context h, causal-mask), held-out 17.8k vị trí:

| mô hình | bits/token ↓ |
|---|---|
| unigram | 10.041 |
| bigram (đếm) | 10.012 |
| kNN túi-TĨNH (PPMI-bag) | 9.564 |
| **kNN-LM(encoder h) λ=0.5** | **8.618** |

kNN(h) hơn đếm **1.42 bits**, hơn túi-tĩnh **0.95 bits** — encoder `h` là biểu diễn
ngữ cảnh TỐT HƠN, đúng TK_042 (h 22× vs PPMI 10×). **Luận điểm phá-tường ĐẠT.**

Hạn chế: perplexity ~393 còn cao (LM thật ~10–50) vì encoder d=96 nông, masked
(causal mismatch), datastore cùng miền. Nâng: causal encoder (TK_059) + d/tầng lớn
+ datastore đầy đủ (TK_057) → bits xuống sâu. Nhưng CƠ CHẾ đã đúng.

### Bước tiếp
1. TK_057: bake datastore đầy đủ (sidecar ANN).
2. Wire vào `generate()`: `P = λ·P_kNN + (1−λ)·P_Wskip` mỗi bước.
3. Đo văn sinh + perplexity thật; TK_059 nếu cần sắc hơn.

## Tích hợp SINH (ĐẠT) — datastore đầy đủ + blend vào generate()

**TK_057 datastore:** `scripts/dung_datastore.py` bake 45k câu sạch → 970.882 vị
trí `(h causal-mask, next)`, sidecar `checkpoints/net_ls_concept_conf.datastore.npz`
(186MB, keys float16 L2-chuẩn hoá trên MPS).

**Blend:** `src/nn_aivie/hca/knn_lm.py` — mỗi bước vote, `KnnLM.blend` TRỘN
`P=(1−λ)·P_Wskip+λ·P_kNN` CHỈ trên tập ứng-viên-SỐNG (tôn trọng mask, không hồi
sinh token chết). Hook `set_knn_hook` trong `generate_hebbian.py` (dòng ~986, sau
mọi mask/penalty). Encoder thêm `causal_last` (mask nhân quả, vị trí cuối).

**Wire server:** `/api/gen?knn=1&knn_lambda=0.35` (mặc định BẬT); `get_knn()` lazy;
pre-warm nạp datastore + warm 1 lượt lúc khởi động.

**A/B (checkpoint thật 450k cha), λ-sweep:**

| prompt | đếm (λ=0) | λ=0.35 | λ=0.85 |
|---|---|---|---|
| sắt và | thiếc | thiếc | sulfide |
| đồng và | …dây dẫn là. **một cách** (rác) | …dây dẫn là. (**cắt sạch**) | word-salad |
| hổ và | câu tốt | =câu tốt | word-salad |
| vàng là kim loại | định nghĩa dài tốt | = | cụt |
| chó và mèo là | tốt | = | tốt |

**Chốt:** λ=0.35 KHÔNG bao giờ làm xấu output tốt (định nghĩa có cha mạnh → vote
áp đảo, blend không lật), đôi khi CẮT rác đuôi, giúp ở prompt phẳng/loại-suy.
λ≥0.85 kNN át cấu trúc W_skip → word-salad. **Default production λ=0.35.**

Hạn chế còn lại: lợi ở λ an toàn còn KÍN đáo vì blend chỉ xáo lại tập ứng-viên
W_skip (chỗ đếm chắc thì đếm áp đảo). Bứt phá lớn cần: (a) nới tập ứng viên cho
kNN đưa token MỚI; (b) encoder CAUSAL-train (TK_059) — encoder hiện masked-train,
causal-mask lúc suy là mismatch làm tín hiệu kNN yếu hơn tiềm năng.

## Nới candidate set (dựng + ĐO — vô ích ở encoder hiện tại → TK_059)

`KnnLM.blend` nới tập ứng viên = SỐNG ∪ {top-k kNN surface W_skip chưa bầu};
token nới chỉ mang khối kNN (nội suy công bằng), tôn trọng `-inf` cứng + chống
lặp gần. Có `widen_gain` khuếch đại khối nới.

**Chẩn đoán (instrument + soi trực tiếp):**
- Nới FIRING: 22 bước blend thêm 10 token nới — nhưng **THẮNG argmax = 0/22**
  (token nới λ·P=0.35·P không thắng nổi vote W_skip tự tin).
- `widen_gain` 1→6: «núi và» luôn "rừng" (W_skip tốt sẵn), «sông/cây và» luôn
  "các", «mặt trời và» nguồn/các/dễ — gain cao chỉ nổi NHIỄU, không ra nội dung.
- Soi top-kNN trực tiếp: «chó là»→"một"(.19)/thú/loại/vật, «hổ là»→"một"(.19),
  «sông và»→sông/một/lọ/xe/bồn (nhiễu chéo miền), «mặt trời và»→như/ví/màu/lặn.

**Kết luận:** cơ chế nới ĐÚNG và AN TOÀN, nhưng vô ích vì encoder masked-train +
ngữ cảnh 2-token cho `h` bị TẦN SUẤT chi phối ("một"/"là" — thứ W_skip đã có) và
NHIỄU chéo miền. kNN & W_skip đồng ý ở token an toàn nhàm chán ⇒ nới không thêm
được nội dung. **Đây là tường ENCODER, không phải lỗi nới.** Mặc định `widen_k=0`
(TẮT); bật khi TK_059 (causal encoder) làm `h` sắc → nới mới sinh lợi.

## Revisit nới trên prompt DÀI (với causal encoder) — vẫn 0 thắng → ĐÓNG hướng nới

Sau khi có causal encoder (kNN-LM bits 7.39, láng-giềng nội-dung), thử lại nới
trên ngữ cảnh ≥5 token (`scripts/thu_noi_dai.py`), instrument đếm nới-thắng:

- W_skip MẠNH («hổ là loài thú…», «người nông dân…»): 4 chế độ (off/reorder/nới8/
  nới12-gain3) **y hệt** — chuỗi verbatim W_skip áp đảo argmax.
- W_skip YẾU («cây cối cần ánh»→off "nắng" TỐT, nới ra "mắt" XẤU; «dòng sông chảy»
  →off "và", nới "Long"/"lợ"): base đã hỏng, nới không cứu, đôi khi tệ hơn.
- **Thống kê: nới thêm 3 token / toàn bộ prompt · THẮNG argmax = 0.**

**Kết luận cuối:** nới KHÔNG sinh lợi dù encoder đã tốt gấp 2× — vì blend LỆ THUỘC
tập ứng viên W_skip: nơi W_skip mạnh thì áp đảo, nơi yếu thì chính base đã hỏng và
kNN-top hoặc đã sống hoặc bị -inf (hiếm khi được thêm). `widen_k=0` giữ TẮT vĩnh
viễn cho kiến trúc này.

**Đòn bẩy thật (mở hướng mới):** để KHAI THÁC phân bố causal tốt gấp 2×, kNN/causal-
LM phải là NGƯỜI LÁI generation hạng nhất — sinh trực tiếp từ `P=λ·P_kNN+(1−λ)·
P_causalLM` (encoder giờ CÓ logits LM, bits 6.78), thay vì chỉ xáo quanh vote
W_skip. Đây là TK_056-v2 / ticket mới, không phải tinh chỉnh nới.
