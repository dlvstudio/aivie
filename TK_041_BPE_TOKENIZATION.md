# TK_041 — BPE TOKENIZATION: đổi ĐÁY của tháp, không mất một byte nào

| Field | Value |
|-------|-------|
| **ID** | TK_041 |
| **Status** | 🟢 **B0–B4 XONG (2026-07-16)** — tokenizer + vocab + 3 đường nạp + train mini + verify **17/17 PASS**. Còn: B5 (train full v11 ~8h), B6 (retune hằng), B7 (số học). |
| **Nguồn** | Chỉ đạo tác giả 2026-07-16: *"tách token giống như mạng LLM, dùng tokenization như mạng LLM, với BPE, như vậy chúng ta sẽ không bỏ qua bất kỳ cái gì, kể cả dấu cách"* + *"cải tiến để training tốt cho cả các số (phục vụ các phép toán học sau này)"* |
| **Chốt thiết kế (tác giả)** | (1) BPE **tự train trên corpus nhà** — KHÔNG mượn tokenizer LLM (Luật 0 sạch). (2) **Đáy = token BPE; mọi tầng trên GIỮ NGUYÊN** — ghép cụm/nơ-ron cụm/ISG/cha y như hiện tại. Khác duy nhất: token không còn là "text tách theo dấu cách rồi FOLD_STRIP". |
| **Liên quan** | README §0.0 (nợ `FOLD_STRIP` + `align._PUNCT_CHARS`), §0.4.2 (dấu câu là token + EOS + BA đường nạp), TK_012 §9.5 (sổ nợ tokenizer), TK_013 (ISG), TK_040 (RAM/CSR), TK_032/034 (word-seg VI) |
| **Phá huỷ** | **MỌI checkpoint hiện có vô giá trị** (`network_v10.pkl` 65k cha / 7.7h train + sidecar VarLM). Đây là cái giá đã biết trước, ghi ở TK_012 §9.5: *"gỡ = đổi vocab = mọi checkpoint vô giá trị"*. Ticket này **chính là** lúc trả món nợ đó. |

---

## 0. ĐO TRƯỚC (B0 — đã chạy 2026-07-16, mẫu 16.3 MB corpus thật)

> Kỷ luật measure-first (README Mục 0). BPE tự-train pure-Python, pre-token digit-split,
> 6 file `data/text_corpus/*` (grade_lessons, qa_supervised, history_articles,
> subject_articles, doc_add_articles, paragraph_units), 28 999 đơn vị.
> Script: `scratchpad/bpe_probe.py` + `scratchpad/cmp_current.py` → chuyển vào
> `scripts/probe_bpe.py` ở B1.

### 0.1. Kết quả — NGƯỢC với nỗi sợ chính

**Mẫu 16.3 MB (probe ban đầu):**

| | HIỆN TẠI (`tokenize_punct`) | **BPE 8k** | **BPE 16k** | **BPE 32k** |
|:---|---:|---:|---:|---:|
| Token/corpus | 3 008 443 | 3 477 609 | 3 350 333 | **3 284 000** |
| Δ so với hiện tại | — | +15.6% | +11.4% | **+9.2%** |
| Fertility (tok/từ-ws) | 1.11 | 1.28 | 1.24 | **1.21** |
| Vocab | **59 480 và MỞ** | 8 000 | 16 000 | **32 000 — ĐÓNG** |
| Train merge table | — | 32s | 87s | **180s** |

**TOÀN corpus 99.2 MB / 65 160 đơn vị (B1 thật, `checkpoints/bpe_v1.json`, fingerprint
`1582dad40906e69f`) — số CHỐT, tốt hơn probe:**

| | HIỆN TẠI (`tokenize_punct`) | **BPE 32k** | Δ |
|:---|---:|---:|---:|
| Token/corpus | 18 211 343 | **19 631 500** | **+7.8%** |
| Vocab | **126 516 và MỞ** | **32 002 và ĐÓNG** | **−74.7%** |
| Fertility (tok/từ-ws) | — | **1.195** | — |
| bytes/token | — | **5.05** | — |
| Train merge table | — | **6.6 phút** | — |
| **Roundtrip byte-exact** | **bất khả** | **OK — 65 160/65 160 đơn vị** | — |

→ Chuỗi dài thêm **+7.8%** (không phải ×2–3), vocab **giảm 4 lần** và thành hữu hạn. Giá
RAM CO_ACTIVATION (hog #1 theo TK_040 M0) ≈ +8%, **không** chặn. TK_040/CSR vẫn cần cho lý
do riêng của nó, nhưng KHÔNG phải điều kiện chặn ticket này.

**Kết luận B0 — nỗi sợ "chuỗi nổ 2–3× → CO_ACT/RAM chết" là SAI.** Tiếng Việt là ngôn ngữ
âm tiết, âm tiết ngắn và lặp cực nhiều → BPE phủ gần trọn âm tiết ngay ở vocab 8k. Ở 32k,
chuỗi chỉ dài thêm **+9.2%** so với tokenizer hiện tại. RAM CO_ACTIVATION (hog #1 theo
TK_040 M0) tăng ~9%, **không** ×3. TK_040 (CSR) vẫn cần cho lý do cũ của nó, **không phải
điều kiện chặn ticket này**.

Ví dụ tách (vocab 32k — đọc tay sạch, âm tiết nguyên vẹn, `Ġ` = có khoảng trắng đứng trước):

```
'Hà Nội là thủ đô của Việt Nam .'
  → ['Hà','ĠNội','Ġlà','Ġthủ','Ġđô','Ġcủa','ĠViệt','ĠNam','Ġ.']
'Nhà Trần đánh thắng quân Nguyên năm 1288 .'
  → ['Nhà','ĠTrần','Ġđánh','Ġthắng','Ġquân','ĠNguyên','Ġnăm','Ġ1','2','8','8','Ġ.']
```

### 0.2. Ba phát hiện định hình thiết kế

**(a) 53% vocab hiện tại là nơ-ron ma.** Hapax (token xuất hiện **đúng 1 lần**):
**31 539 / 59 480 = 53.0% vocab**, nhưng chỉ **1.05% lượt token**. Tức: **quá nửa số
token-neuron trong mạng là vật thể một-lần-dùng, df=1, không bao giờ được củng cố Hebbian**,
mà vẫn ngồi trong vocab kéo `_gdf`/`_gidf` lệch và phình mọi ma trận. BPE hoà tan chúng
thành mảnh **tái dùng được** — mảnh của từ hiếm là mảnh của từ khác.

**(b) Số học đang BẤT KHẢ THI về mặt kiến trúc.** Token chứa chữ số hiện tại:
**8 590 loại = 14.4% vocab**, cho vỏn vẹn 75 110 lượt (2.5%). Nghĩa là `1288` là **một
nơ-ron**, `1287` là **một nơ-ron khác hoàn toàn không liên quan**, `357` lại một nơ-ron nữa.
Mạng **không có đường nào** để biết 1287 và 1288 có họ hàng — chứ đừng nói cộng chúng.
Không phải "mạng học số kém"; là **mạng không có cơ quan để học số**. Digit-split biến
14.4% vocab đó thành **10 nơ-ron chữ số** dùng chung, đồng-hiện dày đặc → có nền để học.
Chi tiết §3.

**(c) NFC/NFD: 28 999/28 999 đơn vị (100%) chứa dấu tổ hợp được.** Tiếng Việt có `ế` viết
kiểu NFC (1 code point) hay NFD (`e`+`◌̂`+`◌́`, 3 code point) — **nhìn giống hệt nhau, BPE
tách hoàn toàn khác nhau**. Hiện tại `.split()` không quan tâm nên bug ẩn. BPE **phải** chốt
NFC trước mọi thứ, nếu không merge table sẽ học hai họ merge song song cho cùng một chữ.

---

## 1. Nguyên lý: BPE **là** Re-Pair — ta chỉ dịch ĐÁY xuống

Điểm mấu chốt để không hiểu nhầm ticket này là "thay kiến trúc":

> **BPE và Re-Pair/ISG (TK_013) là CÙNG một thuật toán** — gộp cặp liền kề xuất hiện nhiều
> nhất thành đơn vị mới, lặp lại. Mạng **đã chạy thuật toán này rồi** (`_repair_grammar`,
> `seg_graph` promote bigram→`ClusterNeuron`). Khác biệt duy nhất: mạng chạy nó **BÊN TRÊN**
> mức từ (`Hà`+`Nội` → cluster), LLM chạy nó **BÊN DƯỚI** mức từ (`H`+`à` → `Hà`).

Nên đây **không** phải cách mạng kiến trúc — là **nối dài cái tháp đã có xuống một tầng**,
và **thay cái đáy giả bằng đáy thật**:

```
HIỆN TẠI:  [text.split() + FOLD_STRIP]  →  từ  →  ISG cluster  →  cha
              ^^^^^^^^^^^^^^^^^^^^^^^
              đáy GIẢ: giả định "dấu cách = ranh giới đơn vị" (sai với tiếng Việt),
              vứt khoảng trắng/xuống dòng, vocab MỞ vô hạn, số = atom vô nghĩa

TK_041:    [byte → NFC → pre-token → BPE 32k]  →  subword  →  ISG cluster  →  cha
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
              đáy THẬT: vô-tổn-hao (decode(encode(x)) == x từng byte), vocab ĐÓNG,
              chữ số tách rời, khoảng trắng/xuống dòng LÀ token
                                              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                                              tầng này KHÔNG ĐỔI MỘT DÒNG NÀO
```

**Hệ quả đẹp (không phải mục tiêu, nhưng là lợi ích thật):** tầng ISG bây giờ **học lại từ
vựng** thay vì được cho sẵn. `Hà`+`ĠNội` đủ power → `ClusterNeuron` — y hệt hôm nay. Nhưng
`Ġthu`+`ỷ` (từ hiếm bị tách) cũng promote được → **từ hiếm không còn là nơ-ron ma df=1, mà
là cụm học được từ mảnh dùng chung**. Đây là câu trả lời kiến trúc cho phát hiện (a).

**Trần trung thực:** ticket này **không** phá trần synonym (v4 §4.1) — `chó≈cún` vẫn cần
đồng-hiện. Nó **không** tự làm mạng biết cộng (§3.4). Nó gỡ **hai** thứ: đáy giả, và
món nợ `FOLD_STRIP` (README §0.0).

---

## 2. Thiết kế tokenizer

### 2.1. Đường ống (thứ tự BẮT BUỘC, mỗi bước vô-tổn-hao)

```
text (str)
  │  ① unicodedata.normalize("NFC", text)        ← đo 0.2(c): 100% đơn vị cần
  │  ② special-token split (nguyên tử, không bao giờ tách)
  │  ③ pre-token regex (biên BPE không được vượt)
  │  ④ BPE merge (bảng cố định, train offline)
  ▼
list[int]  — id trong không gian phân vùng §2.4
```

### 2.2. ② Special token — nguyên tử, BPE không đụng tới

Frame protocol là **được phép tường minh** (README Mục 0: *"Frame protocol (`Hỏi:`/`Đáp:`,
`<EOS>`, `Var:`…)"*). Chúng là **giao thức**, không phải từ vựng → tách trước BPE:

| Token | Vì sao nguyên tử |
|---|---|
| `<EOS>` | mốc kết đơn vị (§0.4.2 — LIFT ≥ 2.0 phụ thuộc nó) |
| `Hỏi:` `Đáp:` | `align.tokenize_punct` đã phải đặc cách chúng rồi (`if word in (HOI_LABEL, DAP_LABEL)`) — BPE tách `:` ra là **mất mốc `Đáp:` và bỏ qua lặng lẽ mọi cặp QA** (bug nền đã vấp ở TK_010, xem comment `align.py:40-42`) |

`<UNK>` **chết** — byte-fallback (§2.3) khiến UNK không thể xảy ra. Giữ id để tương thích
pickle cũ, nhưng count phải = 0 (thêm mục verify).

### 2.3. ③ Pre-token regex — nơi cưỡng chế MỌI luật bất khả xâm phạm

BPE chỉ được merge **trong** một pre-token, không bao giờ **xuyên** biên. Nên mọi luật cứng
đặt ở đây, và merge table **không thể** vi phạm dù data có nói gì:

```python
PAT = re.compile(r"\d|[^\W\d_]+|\s+|[^\s\w]", re.UNICODE)
```

| Nhánh | Ý nghĩa | Luật được cưỡng chế |
|---|---|---|
| `\d` | **một** chữ số | chữ số **không bao giờ** merge với nhau, cũng không với chữ cái (§3) |
| `[^\W\d_]+` | chuỗi chữ cái (có dấu) | biên chữ/số/dấu không bị nhoè |
| `\s+` | khoảng trắng | **được giữ**, không vứt |
| `[^\s\w]` | 1 ký tự dấu | mỗi dấu là đơn vị riêng |

**Khoảng trắng (đúng yêu cầu "không bỏ qua kể cả dấu cách"):**
- 1 space trước từ → gộp vào marker `Ġ` (`Ġthủ`) — kiểu GPT-2, hiệu quả nhất
- `\n` → token `Ċ` riêng · tab/space-lặp → token riêng

→ **Đây là chỗ ta ngừng đốt dữ liệu.** Hiện tại `pipeline.py:32` ghi thẳng
*"`\n` → khoảng trắng lúc tokenize"* — tức **cấu trúc đoạn văn bị huỷ ngay lúc nạp**, rồi
TK_035 phải dựng lại "paragraph unit" bằng cơ chế khác ở tầng trên. Sau ticket này, xuống
dòng là một token mạng **học** được.

**Byte-fallback:** ký tự không có trong alphabet (emoji, chữ Hán trong wiki, ký tự lạ) → rã
thành byte 0–255 (đã có sẵn 256 id). **Không còn OOV, không còn `<UNK>`.**

### 2.4. Không gian id phân vùng — và cái chết của `is_surface_token` string-op

Hiện tại structural node là **chuỗi có prefix** trong cùng vocab, và `is_surface_token` làm
`startswith` trên 6 prefix — README §0.0 xếp loại (B) *"inference re-parse token-string"*.
Đổi vocab là dịp phân vùng id **một lần cho xong**:

| Dải id | Nội dung | Grow? |
|---|---|---|
| `[0, 256)` | byte fallback | cố định |
| `[256, 256+S)` | special: `<EOS>`, `<UNK>`, `Hỏi:`, `Đáp:`, `Ċ`… | cố định |
| `[BPE_LO, BPE_HI)` = 32k | **subword — dải BỀ MẶT duy nhất** | **ĐÓNG** (từ merge table) |
| `[STRUCT_LO, ∞)` | `Entity:` `Concept:` `Sense:` `POS:` `R_-k` `Var:` `Rule:` `QFocus:` `Lbl:` | mở |

**Lợi ích trực tiếp cho Luật 0.0:**

| Hôm nay | Sau TK_041 |
|---|---|
| `is_surface_token(t)` = `startswith` 6 prefix | `BPE_LO <= id < BPE_HI` — **so int** |
| `vocab.surface_mask()` = build mảng bool 34k phần tử, cache thủ công, rebuild khi vocab lớn (comment `vocab.py:116`: 0.47s/prompt) | **slice tĩnh** — vocab bề mặt ĐÓNG, không rebuild bao giờ |
| `w0_tok.endswith(":")` — **so-string cuối cùng còn sót** (README §0.0, §0.4.2 sổ nợ) | `w0_id == DAP_ID` — **chết hẳn** ✅ |

→ Ticket này **đóng** mục (B) của bảng tuân thủ Luật 0.0, không thêm nợ mới.

### 2.5. `fold_group_id` / `cap_class` — viết lại trên bề mặt BPE

`FOLD_STRIP = ".,;:!?…\"'()[]"` **xoá** (cùng `align._PUNCT_CHARS`) — cả hai là tri thức
chính tả viết tay, đúng thứ §0.4.2 vừa dọn ở tầng inference nhưng chưa dọn được ở tokenizer.

| | Hôm nay | Sau |
|---|---|---|
| `fold_of` | `token.strip(FOLD_STRIP).lower()` + vá `or token.lower()` (vá cho bug `fold_of('.')==fold_of('?')==''`) | `surface_of(id).lstrip("Ġ").lower()` — **không strip dấu**, không cần vá |
| `cap_class` | trên token thô | trên bề mặt đã bỏ marker `Ġ` |
| `case_equiv_ids` | Trái Đất ≡ Trái đất | **giữ nguyên cơ chế**, chỉ đổi nguồn fold |

Sau khi bỏ `FOLD_STRIP`, `Ġthủ` / `Thủ` / `thủ` cùng fold-group; `.` giữ fold `.` riêng —
**bug fold rỗng biến mất theo cấu trúc**, không phải nhờ vá.

### 2.6. Detokenize — MỘT hàm duy nhất, và cái bug ai cũng sống chung

Hiện có **~20 chỗ** `" ".join(tok(t) for t in ids)` (`generate_hebbian.py:1074`,
`readout.py:92`, `export_graph_v3.py` ×5, `qa.py`, `chat.py`, `frame_bracket_ref.py`…). Mỗi
chỗ **bịa ra** khoảng trắng vì tokenizer đã vứt mất khoảng trắng thật → mọi output của dự án
đều mang lỗi hiển thị `Hà Nội .` (space thừa trước dấu chấm).

→ **`Vocabulary.detokenize(ids) -> str`** là detokenizer **DUY NHẤT**:
`"".join(surface_of(i) for i in ids if is_surface(i)).replace("Ġ", " ")`.
Mọi `" ".join` bề mặt **xoá**. Kèm test cấm tái phát (grep-based, kiểu test Luật 0 sẵn có).

**Bất biến vô-tổn-hao (test nền của cả ticket):**
```python
for text in corpus_sample:          # ≥ 10k đơn vị thật, gồm \n, số, emoji, chữ Hán
    assert detokenize(encode(text)) == unicodedata.normalize("NFC", text)   # BYTE-EXACT
```
Đây là thứ tokenizer hiện tại **không thể** pass — và là định nghĩa vận hành của
*"không bỏ qua bất kỳ cái gì"*.

---

## 3. SỐ HỌC — làm cho được, và nói thẳng cái gì nó KHÔNG cho

### 3.1. Bệnh (đo 0.2(b)): mạng không có cơ quan để học số

`1288` = 1 nơ-ron · `1287` = 1 nơ-ron khác · `357` = 1 nơ-ron khác. 8 590 loại cho 75 110
lượt → **df trung bình 8.7**, phần lớn hapax. Không có cạnh nào nối chúng. Mọi phép toán
đều là fact rời rạc phải học thuộc riêng lẻ → **bất khả thi tổ hợp**.

### 3.2. Chữa tầng token: digit-split **một chữ số** (đã nằm trong regex §2.3)

`\d` (không phải `\d+`) → mỗi chữ số một token. `1288` → `Ġ1`,`2`,`8`,`8`.

| | Trước | Sau |
|---|---|---|
| Vocab số | 8 590 loại (14.4% vocab) | **10** |
| df mỗi nơ-ron số | ~8.7 | ~7 500 |
| `1287` vs `1288` | 2 atom vô can | chia 3/4 nơ-ron |
| Giá chuỗi | — | +75k token = **+0.1%** tổng |

**Vì sao một chữ số, không gộp 1–3 chữ số như GPT-4o/Llama3:** literature đo thẳng
(Singh & Strouse 2024, *Tokenization counts*) — **single-digit tốt nhất cho số học**, và
nếu gộp thì phải-gộp-từ-phải-sang-trái mới không phá cột. Ta không cần đánh đổi độ dài
(+0.1%) nên chọn thẳng phương án tốt nhất, khỏi có câu hỏi grouping.

**Regex cưỡng chế, không phải merge table hy vọng:** chữ số không thể merge với chữ số
(mỗi cái là pre-token riêng), không thể merge với chữ cái (`\d` vs `[^\W\d_]+` khác nhánh).
Data có nói gì cũng không lách được. → `Ġnăm1288` **không thể tồn tại**.

### 3.3. Chữa tầng data (Luật 0 sạch — data, không rule runtime)

Digit-split cho **nền**; số học cần thêm **curriculum**. Cả ba đều là thay đổi DATA:

| # | Việc | Vì sao |
|---|---|---|
| **N1** | Chuẩn hoá phân cách số trong `normalize_corpus.py`: `1.000` (VI) / `1,000` (EN) / `1 000` → chốt **một** dạng | Hiện `1.000` là 1 token; sau digit-split nó thành `1`,`.`,`0`,`0`,`0` — dấu `.` **giữa số** đọc như dấu câu → nhiễu mốc EOS (§0.4.2 LIFT). Phải quyết, không được để lẫn |
| **N2** | Mở curriculum số học: `grade_lessons.jsonl` đã có `1 cộng 1 bằng 2` (grade 3) — sinh có hệ thống 1–4 chữ số, cộng/trừ/nhân, **có cả bước trung gian** | Digit-split chỉ có tác dụng nếu chữ số **đồng-hiện dày** ở mọi vị trí. Corpus hiện tại: 2.5% lượt token |
| **N3** | **(thí nghiệm, đo rồi mới giữ)** dạng đáp **đảo chữ số** hoặc scratchpad thẳng cột | Nhớ số dư (carry) chạy **phải→trái**, mà mạng sinh **trái→phải**. Đảo chữ số làm carry cùng chiều sinh → cạnh bigram học được. Đây là DATA format, không phải rule runtime → Luật 0 sạch |

### 3.4. ⚠️ Trần trung thực — ghi TRƯỚC khi đo, chống ảo tưởng sức mạnh

> **Digit-split là ĐIỀU KIỆN CẦN, KHÔNG ĐỦ.** Nó gỡ rào chắn "mỗi số là một atom vô can".
> Nó **không** trao cho mạng **giá trị vị trí** (place value): đồ thị này là máy đồng-hiện —
> nó không có cơ quan biết `8` ở cột trăm khác `8` ở cột đơn vị.

Dự đoán thẳng, ghi vào ticket trước khi chạy: **N2 một mình sẽ ra kết quả ÂM TÍNH hoặc chỉ
học thuộc bảng** — đúng bài học đã lặp lại 3 lần trong dự án này (Trụ 2 order-2 trung tính,
R-slot âm tính, E-episode extrinsic âm tính, memory `long-range-cooccurrence-negative`).
Nếu N3 (đảo chữ số) cũng không lật, thì kết luận đúng **không phải** "cần cơ chế mới" mà là:
**số học cần biểu diễn vị trí, thứ mà co-occurrence thuần không có** — ghi âm tính, đừng
đắp cơ chế lên.

Cái ticket này **cam kết** cho số học: **cái rào chắn kiến trúc bị gỡ, và câu hỏi trở nên
ĐO ĐƯỢC** (§7 `arith`). Không cam kết mạng biết cộng.

---

## 4. Blast radius — file & việc

### 4.1. Mới

| File | Vai trò |
|---|---|
| `src/nn_aivie/core/tokenizer.py` | `BPETokenizer`: `train()` / `encode()` / `decode()`; NFC + special + pre-token + merge; byte-fallback |
| `scripts/train_bpe.py` | Train merge table từ manifest → artifact |
| `checkpoints/bpe_v1.json` | **Merge table — artifact GHIM, versioned, bất biến.** Đổi = đổi vocab = chết checkpoint. Hash vào `checkpoint_io` để load phát hiện lệch |
| `scripts/probe_bpe.py` | B0 probe (từ `scratchpad/bpe_probe.py`) — fertility/vocab/roundtrip, chạy lại được |
| `data/eval/arith.jsonl` | Eval số học held-out (§7) |
| `tests/test_tokenizer.py` | Roundtrip vô-tổn-hao, digit-split, special nguyên tử, NFC, byte-fallback |

### 4.2. Sửa

| File | Việc |
|---|---|
| `core/vocab.py` | Phân vùng id §2.4; `is_surface_token` → so int; `surface_mask` → slice tĩnh; `fold_of`/`cap_class` trên bề mặt BPE; **`FOLD_STRIP` XOÁ**; `detokenize()` |
| `teaching/align.py` | `tokenize` / `tokenize_punct` / **`_PUNCT_CHARS` XOÁ** → gọi tokenizer. ⚠️ `align_example` khớp `ConstSlot` bằng `tokens.index(const)` — const nhiều token sau BPE (§8 R3) |
| `training/ingest.py` · `training/pipeline.py` · `scripts/build_varlm.py` | **BA ĐƯỜNG NẠP** — §4.3 |
| `training/ingest_dictionary.py` · `ingest_entities.py` | `tokenize_punct` → tokenizer |
| `hca/generate_hebbian.py` | `prompt.strip().split()` ×2 (`:1236`, `:1610`) → encode; `w0_tok.endswith(":")` (`:1411`) → `== DAP_ID`; `" ".join` (`:875`, `:1074`) → `detokenize` |
| `hca/readout.py:92` · `viz/export_graph_v3.py` (×5, +`:239` split) · `core/frame_bracket_ref.py` · `data/qa.py` · `data/chat.py` · `eval/metrics.py` · `viz/server/main.py` | → `detokenize` / encode |
| `core/varlm.py` | `max_ctx` 4 → **5** (fertility 1.21 × 4 ≈ 4.8) — A/B bằng bits |
| `hca/config.py` | `max_len` 48→58, `K_max` 16→20 (×1.21); ngưỡng promote ISG (§8 R2) |
| `training/normalize_corpus.py` | N1 phân cách số; **ngừng huỷ `\n`** |
| `scripts/verify_checkpoint.py` | Mục mới §7.2; sửa mục `vocab có token '.'` (`:84`) — `.` giờ là `Ġ.` |

### 4.3. ⚠️ BA ĐƯỜNG NẠP — đã vấp 2 lần, không vấp lần 3

README §0.4.2 ghi bằng máu: đợt EOS **sót `pipeline.py` → suýt bake 7.7 giờ ra checkpoint
không mốc kết**; **sót `build_varlm.py` → verify FAIL**. Ba đường tự tokenize độc lập:

| Lệnh | Đường | Sau TK_041 |
|---|---|---|
| `train.py --manifest` | `pipeline.run_manifest` → `ingest_text_file` (`pipeline.py:60`: `tok = tokenize_punct if spec.split_punct else lambda s: s.split()`) | **cả hai nhánh chết** — một tokenizer |
| `train_v3.py --extra-dir` | `ingest_statements`/`ingest_qa` (`ingest.py:56,100`) | nt |
| `build_varlm.py` | tự tokenize → `lm.add_sequence` | nt |

**Chốt kiến trúc:** cờ `split_punct` **XOÁ** — nó tồn tại vì có hai tokenizer đối chọi. Sau
ticket này **chỉ có một đường tokenize trong toàn repo**; `state.tokenizer` là nguồn duy
nhất, không đường nạp nào được tự `.split()`. Test grep cấm `.split()` trên đường nạp.

---

## 5. Lộ trình

| Phase | Nội dung | Xong khi |
|:---:|---|---|
| **B0** ✅ | Đo fertility/vocab/hapax/số/NFC | §0 |
| **B1** ✅ | `core/tokenizer.py` + `train_bpe.py` + `bpe_v1.json` + `tests/test_tokenizer.py` (40 test) | **roundtrip byte-exact 65 160/65 160 đơn vị toàn corpus**; fertility 1.195 |
| **B2** ✅ | `vocab.py`: phân vùng id, `is_surface` so-int, fold/cap, `detokenize`/`encode_text`/`encode_prompt`, `FOLD_STRIP`→legacy-only; 17 call-site `is_surface_token`→`is_surface`; `" ".join`→`detokenize`; `endswith(":")`→so-id | `tests/test_vocab_bpe.py` (16); **184 test gốc còn nguyên** |
| **B3** ✅ | BA đường nạp → `encode_unit`; `normalize_ws_keep_newlines` (ngừng đốt `\n`); `train.py --bpe`; checkpoint MANG THEO tokenizer | `\n` có gdf>0 trong mạng (verify #17) |
| **B4** ✅ | **train mini** → bake → `build_varlm` → `verify_checkpoint` | **17/17 PASS** (10 cũ + 7 mới §7.2) |
| **B5** | **train full v11** (~8h) → bake → varlm → verify → §7 guard | so baseline §7.1 |
| **B6** | Retune hằng đo được: `varlm.max_ctx` 4/5/6, `max_len`, `K_max`, ngưỡng ISG | A/B từng cái, giữ nếu dương |
| **B7** | Số học: N1 phân cách + N2 curriculum + eval `arith` → N3 (đảo chữ số) **chỉ nếu** N2 chạm trần | §3.4 — ghi âm tính nếu âm tính |

### 5.1. Kết quả B1–B4 (2026-07-16)

**Mini BPE (`checkpoints/network_mini_bpe.pkl`, 1 294 cha, vocab 32 002):** verify **17/17**.
Đáng chú ý — **mốc kết SỐNG SÓT qua việc đổi tokenizer**: `'.'` P(→EOS)=0.054 **lift 18.1**
(ngưỡng 2.0), và `'.'` gdf=681 **gidf=1.641** (tự chìm bằng thống kê, đối chiếu `Việt` 4.503).
FRAME_BRACKET 44 899 cạnh / endpoints 17 063 (không kẹt). VarLM 11.6M token, cluster-aware.

**Đọc tay đầu ra (mini — chỉ chứng minh ĐƯỜNG ỐNG, KHÔNG phải số chất lượng, xem cảnh báo
`train_mini.yaml`):**

```
'Hỏi: Thủ đô Việt Nam là gì ? Đáp:'
  → 'Hỏi: Thủ đô Việt Nam là gì ? Đáp: Thủ đô Việt Nam là Hà Nội. Thành phố nằm ở đồng
     bằng Bắc Bộ, ven sông Hồng, là trung tâm chính trị và văn hóa của cả nước.'
'Nhà Trần là'
  → 'Nhà Trần là chính sách đoàn kết nội bộ của những người lãnh đạo.\nĐó là chưa kể tới
     Trần Thủ Độ, Trần Khánh Dư, Trần Quốc Toản....'
```
`Hà Nội.` — **không còn space thừa trước dấu chấm** (§2.6); `\n` **được sinh ra** (trước đây
bị `normalize_ws` đốt lúc nạp nên bất khả).

### 5.2. Ba bug do chính test/verify bắt — ghi lại vì cả ba đều IM LẶNG

| # | Bug | Vì sao nguy hiểm | Bắt bởi |
|:---:|---|---|---|
| **1** | Marker `Ġ`/`Ċ` trong text GỐC bị nhầm là marker do tokenizer sinh → decode trả về dấu cách → **mất chữ** | roundtrip hỏng nhưng chỉ trên text hiếm | `test_marker_chars_in_raw_text_survive` — vá: symbol mang XUẤT XỨ (`pre_tokenize` trả `list[list[str]]`, không phải `list[str]`) |
| **2** | Chữ số hiếm rớt khỏi alphabet (suy từ tần suất + `min_char_count`) → `52` → `['5','<0x32>']` | **roundtrip VẪN ĐÚNG** ⇒ trốn thoát hoàn toàn im lặng; mà `2` thôi không còn là nơ-ron dùng chung = **tái phát đúng bệnh §3.1** ở tầng alphabet | `test_all_ten_digits_are_neurons_even_if_absent_from_corpus` — vá: ghim cứng `DIGITS` |
| **3** | Đường SINH còn `prompt.strip().split()` → `id_of("Thủ")` trúng biến thể KHÔNG-Ġ → prompt `'Hỏi: Thủ đô Việt Nam'` thành `'Hỏi:ThủđôViệtNam'` = chuỗi **chưa từng tồn tại lúc train**; từ lạ (`'gì?'`) còn **ĐẺ id mới** → phá tính ĐÓNG | không crash, không test nào đỏ — sinh vẫn ra chữ, chỉ là trật hết cạnh | **ĐỌC TAY** đầu ra mini. Không phép đo tự động nào bắt; đúng lý do R6 tồn tại. Vá: `vocab.encode_prompt` |
| **4** | **Prompt không chuẩn hoá như train.** Train (`pipeline`) chạy `normalize_ws_keep_newlines` (kết thúc bằng `.strip()`); `encode_prompt` thì không → `'  Hỏi:'` = `['Ġ','Ġ','Hỏi:']` ≠ `['Hỏi:']` → **một dấu cách thừa của người dùng đẩy prompt ra ngoài phân bố train** | Nghịch lý: chính tính **vô-tổn-hao** gây ra — nó giữ trung thành cả rác đầu vào. Trước TK_041 `.strip()` THỰC SỰ thừa (`"  a b  ".split()` vốn bỏ trắng), nên dễ tưởng gỡ được | **TÁC GIẢ hỏi** *"tại sao phải strip, tại sao phải split mà không dùng tokenize?"* (2026-07-16) — không test nào, không đo nào, không cả đọc-tay-đầu-ra bắt được: cần người nhìn vào DÒNG CODE và hỏi "tại sao". Vá: `encode_prompt` chuẩn hoá y hệt train |

**Bài học:** cả bốn đều là lỗi *im lặng* — không exception, không test đỏ, không số xấu. Đó
là đặc trưng của lỗi tầng tokenizer: id vẫn hợp lệ, chỉ là trỏ sai. Bốn lớp phòng thủ, mỗi
lớp bắt được cái mà lớp khác mù:

| Lớp | Bắt được | MÙ với |
|---|---|---|
| Test roundtrip | #1 | #2 (roundtrip vẫn đúng!), #3, #4 |
| Test bất-biến-riêng | #2 | #3, #4 |
| **Đọc tay đầu ra** | #3 | #4 (prompt sạch thì không lộ) |
| **Người đọc code hỏi "tại sao"** | #4 | — |

→ §7.2 phải là verify **exit 1**, không phải mắt người đọc log. Và không lớp nào thay được
lớp nào. Bất biến **TRAIN ≡ PROMPT** nay khoá bằng test
(`test_prompt_encodes_identically_to_train`, `test_prompt_robust_to_stray_whitespace`).

**Sổ nợ nhỏ (đã đo, chưa sửa):** `Đáp:` là special nguyên tử nên nó CẮT chuỗi → dấu cách
đứng trước nó thành token `Ġ` độc lập thay vì tiền tố `ĠĐáp:`. Vô hại (train và prompt
GIỐNG NHAU, roundtrip đúng), giá = 1 token thừa mỗi nhãn frame.

**Không gộp B4 và B5.** Đường nạp có 3 nhánh và lịch sử đã chứng minh nó sót.

---

## 6. Thước đo — vì sao bits-per-**BYTE** thành bắt buộc

⚠️ **Số/token không so được xuyên tokenizer.** Perplexity, next-token-acc, `n_tokens` của
VarLM — mọi thứ chia cho token — **vô nghĩa** khi mẫu số đổi. Một tokenizer tách vụn hơn tự
động "ppl đẹp hơn" mà không giỏi hơn tí nào.

→ **bits-per-byte là thước DUY NHẤT so được** giữa v10 và v11. v4 §3 Trụ 1 mục 5 đã muốn nó
("hiện chỉ smoke"); ticket này làm nó **bắt buộc**. Việc B1: nâng `scripts/eval_lm_ruler.py`
lên bits-per-byte thật trên held-out **cố định** (khoá theo byte, không theo token).

So được xuyên tokenizer / không:

| So được ✅ | KHÔNG so được ❌ |
|---|---|
| bits-per-byte | perplexity, bits-per-token |
| QA/Para/Syn accuracy (task-level) | next-token accuracy |
| retrieval rank-1 / MRR | `n_tokens`, vocab size |
| essay cov/ord/match, `copy` (LCS) | `verbatim_rate` (đơn vị đổi) |

---

## 7. Acceptance

### 7.1. Guard — KHÔNG được tụt (task-level, so được)

| Bộ | Baseline v10 | Nguồn |
|---|---|---|
| QA / Para / Syn | **86 / 88 / 67** | README §0.4.1 |
| Retrieval rank-1 / MRR (`grade_lessons_test.pkl`) | **76.3% / 0.853** | TK_036 |
| `copy` (LCS) `conv` / `history_wiki_para` | **0.22 / 0.57** | README §0.4.2 |
| bits-per-byte | *đo lại trên v10 ở B1* | — |

**Kỳ vọng trung thực:** guard **đi ngang** là **THẮNG**. Ticket này đổi đáy, không thêm năng
lực ngữ nghĩa — không có lý do gì QA tăng. Nếu số **tăng**, phải nghi thước trước khi mừng.
Rủi ro thật là **tụt** (§8 R1).

### 7.2. Mục verify mới (`scripts/verify_checkpoint.py`, exit 1 nếu trượt)

1. `decode(encode(x)) == NFC(x)` byte-exact trên 1 000 đơn vị corpus
2. Hash `bpe_v1.json` khớp hash ghi trong checkpoint
3. Vocab bề mặt **đóng**: không id bề mặt nào ≥ `BPE_HI`
4. `<UNK>` count = **0**
5. Không token nào vừa chứa chữ số vừa chứa non-digit
6. `Ċ` (`\n`) và `Ġ`-token có mặt với df > 0 — chứng minh khoảng trắng **thật sự** được nạp
7. `<EOS>` LIFT ≥ 2.0 vẫn đứng (§0.4.2 — mốc kết sống sót qua đổi tokenizer)

### 7.3. Số học (§3, **đo và ghi kể cả âm tính**)

`data/eval/arith.jsonl` held-out — cộng/trừ 1–4 chữ số, chưa từng có trong train.
Baseline v10 dự kiến ≈ **0%** (không có cơ quan). Báo cáo: exact-match, và **digit-level
accuracy theo vị trí cột** (chỗ nào carry gãy).

---

## 8. Rủi ro

| # | Rủi ro | Vì sao thật | Xử lý |
|:---:|---|---|---|
| **R1** | **IDF loãng → retrieval tụt** | Toàn bộ `rank_matches` đứng trên `_gdf`/`_gidf` **mức từ**. Fertility 1.21 nghĩa là ~79% từ nguyên vẹn — nhưng **từ bị tách chính là từ hiếm = từ IDF CAO = thực thể phân biệt** (`Cao Bằng`). Tách `Cao`+`ĠBằng` → IDF rải ra 2 mảnh phổ biến hơn → đúng lớp bug README §0.5.8/§0.5.10 đã đau | (a) ISG promote mảnh về lại cụm → IDF ở **match-unit**, không ở subword thô (`query_match_units` đã ưu tiên phrase/cluster); (b) `Entity:` neuron (TK_027) đã key theo **cụm title, không fold**; (c) **đo bằng rank-1 76.3% — guard cứng, không thương lượng** |
| **R2** | Ngưỡng promote ISG lệch | Ngưỡng power bigram tune trên bigram **từ**. Giờ có thêm bigram **trong-từ** (mảnh của từ hiếm) — phân bố đếm đổi | B6 A/B ngưỡng. Dấu hiệu tốt: `ở→đâu` power=19 (TK_013) phải còn đó |
| **R3** | `align_example` gãy | `tokens.index(const_slot.const)` giả định const = **1 token**. `là` sau BPE = `Ġlà` (1 token, may) nhưng const dài thì không | Khớp theo **chuỗi id** thay `index`. Teach-pack v2 đã xoá phần lớn (TK_019) → mặt tiếp xúc nhỏ; `g1_passive`/`g2_imitation` chỉ dùng `tokenize` |
| **R4** | RAM +9% trên nền đã swap | TK_040 M0: full v10 ~20–22GB / máy 24GB. +9% CO_ACT có thể đẩy qua mép | B4 mini đo RSS trước. Nếu B5 swap → **TK_040 M1 (CSR) thành chặn**, làm trước B5. **Không** vì thế mà bỏ digit-split (+0.1%) |
| **R5** | Mọi checkpoint chết | 65k cha, 7.7h, + 20 checkpoint lịch sử (`grade_v3*`) mà test/guard đang trỏ vào | Chấp nhận — chính là món nợ TK_012 §9.5. Giữ v10 để so bits-per-byte. Test `@pytest.mark.slow` trỏ checkpoint cũ: skip hoặc trỏ v11 |
| **R6** | Đọc tay tiếng Việt xấu đi ở từ hiếm | Fertility trung bình đẹp (1.21) **che** đuôi phân bố — từ hiếm có thể tách kiểu `Ġngh`+`ĩa` | B1: dump 200 từ hiếm nhất + **đọc tay**. Fertility trung bình **không** đủ làm bằng chứng |

---

## 9. Chốt lại — ticket này KHÔNG làm gì

Chống ảo tưởng sức mạnh (README Mục 0), nói trước:

- ❌ **Không** phá trần synonym (`chó≈cún`) — vẫn cần đồng-hiện (v4 §4.1)
- ❌ **Không** cho mạng hiểu LỖI CHÍNH TẢ. BPE cho **containment**, KHÔNG cho **robustness** —
  xem §9.1
- ❌ **Không** làm mạng biết cộng — chỉ gỡ rào và làm câu hỏi đo được (§3.4)
- ❌ **Không** đổi bất kỳ tầng nào trên đáy — ISG/cụm/cha/hub/theme **y nguyên** (chỉ đạo tác giả)
- ❌ **Không** kỳ vọng guard tăng — đi ngang là thắng (§7.1)
- ✅ **Có** làm: đáy vô-tổn-hao (`decode(encode(x))==x`), vocab đóng 126k→32k, hoà tan 53%
  nơ-ron ma hapax, mở đường số học, **giữ khoảng trắng + xuống dòng**, và **trả xong món nợ
  Luật 0.0** (`FOLD_STRIP`, `_PUNCT_CHARS`, `endswith(":")`, `surface_mask` rebuild)

### 9.1. Lỗi chính tả — CONTAINMENT, không phải ROBUSTNESS (đo 2026-07-16)

> Ghi rõ để người sau không tưởng tầng này đã lo xong chính tả.

**Cơ chế:** encode LUÔN bắt đầu từ ký tự rồi gộp dần. Từ đúng lên tới đỉnh, từ sai kẹt ở đáy:

```
'Lập' :  ['L','ậ','p']  →merge #170→ ['L','ập'] →merge #3714→ ['Lập']   ✅
'Lậb' :  ['L','ậ','b']  →  không merge nào khớp  →  ['L','ậ','b']       ⛔ tụt đáy
'Lập trình' → ['Lập','Ġtrình']  ·  'Lậb tringf' → ['L','ậ','b','Ġtr','ing','f']
                                    → CHUNG: **0 nơ-ron**
```

**Được (containment):**

| | `Lậb tringf Vietj Nôi thu do…` |
|---|---|
| **BPE** | vocab 32 002 → **32 002** (mọc **0**); mọi mảnh là nơ-ron THẬT đã train |
| Cũ | vocab 8 → **20**: mỗi typo đẻ một **nơ-ron ma** df=0, KHÔNG cạnh nào, chết vĩnh viễn |

→ Đây là gốc của "53% vocab cũ là hapax" (§0.2a): mỗi lỗi chính tả trong 99MB để lại một xác.

**KHÔNG được (robustness):** `Lậb` chia **0 nơ-ron** với `Lập` — mạng không hề thấy chúng gần nhau.

**Mạng thật (đo trên `network_mini_bpe`):**

| Prompt | Kết quả |
|---|---|
| `Thủ đô Việt Nam là gì` | ✅ *"…là Hà Nội. Thành phố nằm ở đồng bằng Bắc Bộ…"* |
| `Thủ đô Việt Nam là **gìf**` | ✅ **vẫn đúng y hệt** |
| `**Thu do Viet Nam la gi**` (mất dấu) | ❌ **lạc hẳn** → trả lời về Đà Nẵng |

⚠️ **Cái cứu ca #2 KHÔNG phải BPE mà là IDF:** `gìf` hỏng nhưng `Thủ`/`đô`/`Việt`/`Nam` còn
nguyên và chúng mới là neo IDF cao. Khi MỌI token đổi (mất dấu) → không còn neo → lạc. Mảnh
`L`/`ậ`/`ing` có df rất cao → IDF thấp → không neo được → suy biến thành *"không tín hiệu"*
chứ không phải *"tín hiệu sai"*. Công của **IDF + phạt distinct-ratio** (§0.5.9/0.5.10),
không phải của tokenizer.

| Loại lỗi | Sống? | Vì sao |
|---|:---:|---|
| Typo lẻ 1 từ trong câu | ✅ | token khác giữ neo IDF |
| Typo trúng ĐÚNG từ khoá duy nhất | ❌ | mất neo |
| Sai HỆ THỐNG (mất dấu, telex) | ❌ | mọi token đổi, chia 0 nơ-ron |

**Trần thật:** hiểu typo cần tương đồng **mặt chữ** (edit-distance), mà `hybrid_fuzzy_topk` đi
qua E — vốn dựa **đồng-hiện**, còn `Lậb` chưa từng đồng hiện với gì. Cùng họ trần với synonym
(v4 §4.1). TK_041 KHÔNG đụng tới. Với dự án này là đánh đổi đúng: corpus train là văn bản chuẩn.
