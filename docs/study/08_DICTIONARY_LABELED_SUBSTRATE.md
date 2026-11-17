# 08 — TỪ ĐIỂN LÀM SUBSTRATE CÓ NHÃN: NEO POS/NGHĨA, ĐỒNG-TRÁI NGHĨA, KHUNG XA

> Nguồn: trao đổi tác giả ↔ Claude, 2026-07-05. Tác giả đặt bài: *"nạp training data từ
> bộ từ điển tiếng Việt thật — mỗi danh từ/động từ/từ đều có nhãn (POS, từ đơn/đa nghĩa,
> đồng nghĩa, trái nghĩa, định nghĩa, ví dụ) → tạo ra nhiều điểm NEO cho train; dùng nhiều
> nơ-ron hơn để neo thay vì dùng fact; và link ngang giữa các nơ-ron KHÔNG kề nhau trong
> câu (`nếu→thì`, `bởi vì→nên`, `Hỏi:→Đáp:`) cho ngữ pháp + cấu trúc câu."* — think outside
> the box, hoàn toàn dựa vào mạng + data train, KHÔNG hardcode rule.

## 0. Vì sao đây là đòn bẩy, không phải thêm một feature

Chẩn đoán v4 §2 (W2): **substrate cấu trúc RỖNG** — toàn bộ tín hiệu trật tự/chuyển tiếp/
ngữ pháp của máy sinh chỉ có `W_seq` bigram (Markov bậc-1) + `ar_index_backbone` ép-verbatim
([`generate_hebbian.py`](../../src/nn_aivie/hca/generate_hebbian.py) §A3). Hệ quả đúng như
quan sát: sinh **neo vào 1 fact nổi trội**, **ghép fact không mượt** (cầu nối chỉ 1 dấu phẩy),
**đa nghĩa nuốt nghĩa** (T0 âm tính: "trị" bị "giá trị" nuốt), **synonym trượt** (50/17/33%).

Từ điển KHÔNG phải nguồn tri-thức-nội-dung (như wiki) — nó là **nguồn NHÃN CẤU TRÚC**: mỗi
mục cấp sẵn POS, lớp từ, quan hệ đồng/trái nghĩa, và **ví dụ = khung ngữ pháp chuẩn**. Đây
chính là substrate mà W2 bảo là rỗng, nạp vào bằng đúng pipeline train hiện có.

## 1. LUẬT 0 — SẠCH TUYỆT ĐỐI (đính chính hiểu nhầm sớm)

Đưa đoạn giải thích một từ vào = **văn bản train** như mọi đoạn wiki/QA khác, chảy qua
`ParentStore.add_example` → token + co-occurrence + W_seq + nơ-ron cha. Nhãn "động từ",
"Từ đơn", "đồng nghĩa: ủi" trở thành **nơ-ron nối bằng cạnh HỌC ĐƯỢC** — KHÔNG một dòng
`if token=="là"`, KHÔNG một trọng số ưu tiên nào cho bất kỳ từ nào.

**Kỷ luật duy nhất còn giữ** = Luật 0.0 (TK_012) áp lại, KHÔNG phải luật mới: lúc **sinh**,
mạng đọc neo "động từ" như MỘT NƠ-RON (kích hoạt lan qua nó, đọc cạnh) — không decode nhãn
POS ra string rồi `if pos==VERB: boost`. Cái đầu graph-native (được); cái sau rơi vào lớp
re-parse token-string mà TK_012 đang dọn. **Ranh giới này không liên quan tới việc nạp từ
điển** — nạp là sạch. Mục 0 README vốn cho phép tường minh "lesson, curriculum, corpus".

## 2. Hai bức tường "cần pretrain" SỤP xuống thành "cần độ phủ từ điển"

| Bức tường (v4 §4 ghi "trần cứng, cần data/pretrain") | Cơ chế cũ | Với từ điển |
| :--- | :--- | :--- |
| **Synonym** (chữa≈trị) — PPMI bậc-1 mù vì không đồng hiện | phải mine từ co-occurrence | **cạnh đồng-nghĩa TƯỜNG MINH** trong mục từ → hạ xuống trần **độ phủ từ điển** (chỉ là nạp thêm data, Luật 0 sạch) |
| **Đa nghĩa một-vector** (T0: "trị" bị "giá trị" nuốt) | 1 token = 1 vector, không tách nghĩa | từ điển cho **inventory nghĩa + ví dụ RIÊNG mỗi nghĩa** → tách "là#danh-từ" ≠ "là#copula" thành **nơ-ron nghĩa riêng**, mỗi cái một hồ sơ co-occurrence riêng |

**Calibrate trung thực (chống ảo tưởng):** từ điển cấp *substrate* để phân biệt; **định
tuyến một lần-xuất-hiện về đúng sense-neuron lúc runtime** vẫn là phần khó (q·k type-level
không làm per-occurrence — doc 03 §4). Khác biệt cốt lõi: trước đây *không có gì để định
tuyến tới*; giờ có sense-neuron + ví dụ có nhãn để spreading/attention bám vào. Wall hạ
mạnh, **chưa biến mất**. Độ phủ synonym = độ phủ từ điển (hữu hạn).

## 3. Bốn trụ (D1–D4) — từ điển là hạt giống mở khoá 3 ý còn lại của tác giả

### D1 — Từ điển làm TẦNG NEO NGUYÊN THUỶ (ý ④; rẻ nhất, đòn bẩy cao nhất)

Nạp từ điển; mỗi mục sinh ra chùm neo. Nhãn POS + lớp từ + nghĩa + ví dụ = **nhiều điểm
neo/từ** — đúng ý ③ ("dùng nhiều nơ-ron để neo thay vì fact"). "động từ"/"danh từ" được
**hàng nghìn từ** nối tới → thành **hub khổng lồ** = neo mạnh nhất khi sinh.

Một nhãn POS gỡ cùng lúc các tường "cần POS" mà nhật ký lặp lại:
- **Anaphora** (v4 §7.3): P2 chết vì không tách được đại-từ khỏi danh-từ đồng-hiện
  (`Hồ Chí Minh→chủ/nãi`). POS = **inventory đại-từ SẠCH** → wire lại `entity_anaphor_pref`
  gate `pos==PRONOUN`.
- **q·k nhiễu function-token** (doc 07: "nhiễu ở Đáp:/,"): tách **head trợ-từ/liên-từ** theo
  POS → q·k nội-dung không bị hub function-token nhấn chìm.

### D2 — Attention Q/K CÓ HƯỚNG (ý ②; hoàn thành doc 03)

Build thật doc 03: `M_seq (PPMI có hướng) ≈ UΣVᵀ → Wq=U√Σ, Wk=V√Σ → q·k ≈ PPMI(i→j)`.
Multi-head theo loại cạnh (cú pháp W_seq / chủ đề co-act / fact→fact). Từ điển làm D2 chạy
được nơi doc 07 thất bại: **gate/tách head theo POS** cho function-token. Cắm vào A3 thay
`ar_index_backbone` cứng: token bắc-cầu off-agenda ("Đáp:"→"Việc") có q·k>0 nếu cùng khuôn.

### D3 — LINK NGANG TẦM XA (ý ⑦; nguyên thuỷ MỚI, quan trọng cho ngữ pháp)

ISG hiện **chỉ** promote bigram **liền kề** ([`seg_graph.py::_register_bigram`](../../src/nn_aivie/core/seg_graph.py)).
Ý tác giả — `nếu→thì`, `bởi vì→nên`, `Hỏi:→Đáp:` — là cạnh **KHÔNG kề**, học được bằng:
- quét trong cùng cha/episode các cặp (token_mở, token_sau) không liền kề → PMI có hướng;
- **gate bằng POS (D1)**: chỉ mở cạnh xa khi đầu là **liên-từ/tiểu-từ/nhãn-frame** → chống
  bùng nổ O(n²);
- vật chất hoá thành cạnh **`FRAME_BRACKET`** (loại cạnh mới) → một **head diễn ngôn** ở D2.

Từ điển cấp data trực tiếp: các nghĩa 5/6 của "là" ("hễ… là…", "chắc là chắc") + ví dụ
"Thời gian là thứ không chờ ai" = **khung `X là Y` chuẩn**. Khi sinh: gặp "nếu" → neo "thì"
nổi ở H_mid → giữ khung ngữ pháp XUYÊN segment. Bigram + index-backbone về nguyên lý không
làm được việc này.

### D4 — SINH = ĐỌC TỪ NGĂN XẾP NEO (ý ③; doc 05)

Viết lại A3: thay `ar_index_backbone` ép-verbatim bằng **readout hợp nhất từ stack neo đa
tầng** (đồng hồ đa tầng doc 05 §5): `H_slow` (POS-frame/topic/persona) · `H_mid` (t1 +
neo-câu + khung D3) · `H_fast` (token: W_seq + q·k). agenda-câu chỉ còn **một** neo trong
nhiều, không độc quyền khoá cả câu → hết "neo vào 1 fact nổi trội".

## 4. Thiết kế nạp (cho đúng ví dụ "là" tác giả đưa)

Đường ingest mới, tái dùng `add_example` + thêm loại cạnh. Sơ đồ neo cho một mục:

```
lemma "là"
 ├─ HAS_ATTR → {Từ đơn, Từ đa nghĩa, Từ thuần việt, Từ toàn dân}     (hub loại-từ)
 ├─ sense "là#1" ─POS→ danh từ  ─DEF→ (câu định nghĩa, parent) ─EX→ [ví dụ, parent]
 ├─ sense "là#2" ─POS→ động từ  ─SYN→ "sà"  ─ANT→ "bay lên"
 ├─ sense "là#3" ─POS→ động từ  ─SYN→ "ủi"  ─ANT→ "vò"
 ├─ sense "là#4" ─POS→ động từ(copula) ─EX→ "Thời gian là thứ không chờ ai"  ← khung X-là-Y
 └─ sense "là#5..#9" ─POS→ trợ từ                                    ← function-token (tách head D2)
```

Định nghĩa + ví dụ đi qua ingest thường (co-occurrence/W_seq) → **miễn phí làm dày khung
ngữ pháp**. Cạnh `SYN/ANT/POS/HAS_ATTR/DEF/EX` = loại cạnh mới (`EdgeKind`), serialize như
các layer hiện có. Sense-neuron = một loại nơ-ron mới (`kind="sense"`), cùng loài với
cluster/theme/hub/t1 (giữ tính phân dạng của mạng).

**Độ hạt (quyết định đo trước):** tách tới sense (`là#1..#9` — mạnh cho đa nghĩa nhưng nặng
vocab) HAY dừng ở POS (nhẹ). Nghiêng **tách sense** vì đó chính là đòn phá tường đa-nghĩa —
nhưng probe C0 trên 10–20 từ trước khi cam kết.

## 5. Lộ trình đo được (thí nghiệm rẻ chặn công lớn)

> Điều kiện tiên quyết (không thương lượng): **bước 0 = mở held-out generation lớn + metric
> nghiêm** (v4 Trụ 1 mở rộng). Nếu không, nạp từ điển xong sẽ lại đọc ra "trung tính 100/100"
> mà không thấy nó cứu synonym/đa-nghĩa ở đâu — bay mù (W1).

| Phase | Nội dung | Đo bằng (đọc tay + metric) | Chi phí |
| :--- | :--- | :--- | :--- |
| **C0 — probe rẻ** | Nạp tay ~15 mục (gồm "là","chân","đông" đa nghĩa) vào `_t3`; đếm neo hút được; đọc tay sense-neuron có tách nghĩa không | ≥N cha/sense đúng miền; đồng-nghĩa nối đúng | 1 buổi |
| **D1** | Parser từ điển → `ingest_dictionary.py`; POS/attr/sense/syn/ant neurons; serialize | roundtrip; thống kê membership hub POS | vừa |
| **D1-anaphor** | Wire lại pronoun-gate bằng `pos==PRONOUN` | eval_anaphor (lật được inanimate "Trái Đất→nó"?) | nhỏ |
| **D1-syn** | fuzzy match đọc thêm cạnh SYN tường minh | Synonym probe (kỳ vọng >50/33%) | nhỏ |
| **D2** | Q/K SVD có hướng, head tách theo POS (A0 doc 03) | held-out bigram rank; đọc tay 10 khuôn U/V | vừa |
| **D3** | Cạnh `FRAME_BRACKET` PMI-xa gate POS | đọc tay cặp `nếu→thì`/`Hỏi:→Đáp:`; coverage | vừa |
| **D4** | A3 multi-anchor thay `ar_index_backbone` | A/B: cov/ord + `verbatim_rate` giảm mà chất lượng giữ | lớn, cuối |

**Acceptance C0 (điều kiện mở D1):** ≥70% mục gieo hút được ≥3 cha/ví dụ đúng nghĩa khi đọc
tay; ≥1 từ đa nghĩa tách được ≥2 sense-neuron có hồ sơ hàng-xóm khác nhau rõ; mục "chết"
(0 attach) ghi rõ = tín hiệu thiếu độ phủ, KHÔNG tự lừa bằng attach mờ.

## 6. Rủi ro ghi trước

1. **Định tuyến sense per-occurrence** vẫn khó (§2) — substrate có, routing runtime là việc
   riêng (spreading/q·k ngữ cảnh). Không hứa giải đa-nghĩa trọn vẹn ở D1.
2. **Vocab phình** nếu tách sense cho mọi từ — cân bằng: chỉ tách sense cho từ đa nghĩa
   (từ điển đã đánh dấu "Từ đa nghĩa"); từ đơn-nghĩa giữ 1 neo.
3. **Chất lượng parser từ điển** — mục semi-structured, crawl có thể nhiễu (HTML, quảng cáo).
   Cần quality gate như `data/quality.py`.
4. **"Trung tính" nếu thiếu bước 0** — mọi số sẽ bão hoà trên eval n=7/8/6 hiện tại. Bắt buộc
   mở held-out trước (W1).
5. **Nhãn ≠ hiểu** (doc 06 §6 rủi ro 4) — neo POS chỉ hữu ích nếu D4 đọc nó khi sinh; nạp
   suông = nhãn trang trí. Metric phải chứng minh ablate neo → sinh tệ đi.

## 7. Liên kết chéo

- **doc 03** (Q/K heads) = D2; POS gate cho head function-token là mảnh doc 07 thiếu.
- **doc 05** (anchor neurons) = D4; POS/sense/attr = các `AnchorNeuron` kind mới, TỰ NHIÊN
  có sẵn thay vì phải mine invariance.
- **doc 06** (primed concept) — sense-neuron mồi bằng định nghĩa/ví dụ = đúng "② đấu dây bằng
  resonance"; curriculum lớp 3→ĐH = nạp từ điển theo tầng.
- **doc 07** (spreading vs directed) — agenda vẫn gánh "NÓI GÌ"; từ điển + D2/D3 lo "CÁCH nói"
  + đúng POS/nghĩa. Từ điển KHÔNG chữa "fact sai" (đó là lỗi retrieval/data).
- **TK_013** (ISG) = nền cho D3 (mở rộng từ bigram-kề sang bracket-xa); **TK_012** (Luật 0.0)
  = ranh giới inference-đọc-neuron cho D1.

## 8. Trần cứng — nói thẳng

- Q/K + từ điển cải thiện **CÁCH nói** (chuyển tiếp, ngữ pháp, đúng nghĩa) — **KHÔNG** chữa
  "fact sai": đó là lỗi retrieval + hub-guard + data.
- Synonym/đa-nghĩa hạ từ trần-kiến-trúc xuống **trần-độ-phủ-từ-điển + routing runtime** —
  tốt hơn nhiều nhưng không phải zero.
- Hệ vẫn **tái tổ hợp khuôn mẫu, không phát minh** (v4 §4.2). Định vị: lõi sinh chuẩn văn
  phạm + màng lọc ngữ cảnh, không phải LLM tổng quát.
</content>
</invoke>
