# TK_012 — MẠNG THUẦN NƠ-RON: tách content/label ra content-cache

| Field | Value |
|-------|-------|
| **ID** | TK_012 |
| **Status** | ✅ Done (2026-07-03) — 3 phase triển khai; 296 test pass; guard đo lại (xem §7) |
| **Nguồn** | Luật 0.0 (README, 2026-07-03): mạng nơ-ron KHÔNG chứa text/label/content |

## 0. Kết quả triển khai (2026-07-03)

- **Phase 1 — tách nhãn:** thêm `core/content_cache.py` (`ContentCache`: parent tag + episode key/title/source) và `core/checkpoint_io.py` (`load_checkpoint`/`save_checkpoint` = graph thuần `.pkl` + sidecar `.content.pkl`, auto-migrate checkpoint cũ). Gỡ `tag` khỏi `ParentRecord`, gỡ `key/title/source` khỏi `AbstractNeuron`; serialize graph KHÔNG còn nhãn. Build scripts (`build_episodes`, `build_sft_episodes`, `build_t1_unified`), `eval/v4.eval_essay`, `viz/export_graph_v3.completion_graph` + `viz/server` nhận `cache` tường minh. **Inference & training không import/không đọc cache** — test `test_training_blind_to_cache` + `test_inference_blind_to_cache` chứng minh graph/output bất biến với cache.
- **Phase 2 — thuộc-tính-token:** `Vocabulary` precompute `is_punct/cap_class/fold/fold_group_id` khi token vào vocab; hot-path (`_content_ids`, `_fold`, `_ensure_case_equiv`, `discourse`, `generate_hebbian`) đọc thuộc tính thay vì `.strip()/.lower()/_is_punct` runtime.
- **Phase 3 — phrase_assoc khóa id:** `_phrase_assoc` + `_ep_bg_df` khóa cặp `(fold_group_id, fold_group_id)` (int) thay string; serialize thuần id; checkpoint cũ (khóa string) convert lazy khi có vocab.

## 1. Tổng quan

Đưa mạng về **thuần nơ-ron**: đồ thị chỉ có id + cạnh (kiểu/trọng số) + kiểu nơ-ron + chuỗi
token-id cấu trúc. Mọi text/label/content (câu/đoạn/episode, tag, title, source) chuyển sang
**content-cache riêng** (`neuron_id → {...}`) chỉ phục vụ UI/tracing/debug. Inference & eval
CHỈ đọc mạng thuần — không decode string rồi thao tác chuỗi.

## 2. Hiện trạng chi tiết (rà soát 2026-07-03)

**Lõi đồ thị đã GẦN THUẦN** — cha/cụm/hub/theme lưu **chuỗi token-id** (cấu trúc phân rã),
KHÔNG lưu raw string. Ba nhóm vi phạm còn lại:

| Nhóm | Vi phạm | Vị trí | Mức |
|---|---|---|---|
| **A — nhãn trong nơ-ron** | `ParentRecord.tag`; `AbstractNeuron.key/title/source` | `parent_store.py:161,199-202` (+serialize) | Rõ ràng — dễ tách |
| **B — inference re-parse token-string** | dấu câu (`_is_punct`, `==". "`) 27 · viết hoa (`.isupper()`) 3 · case-fold (`_fold`) 18 = ~48 chỗ | `parent_store.py`, `hca/discourse.py`, `hca/generate_hebbian.py` | Đọc TOKEN (ngoại lệ được phép) nhưng nên thành thuộc-tính-token |
| **C — dict khóa surface-string** | `_phrase_assoc[("mì","kẹp")]` (bigram bề mặt) | `parent_store.py` (build_phrase_assoc/related_phrases) | Vượt token → phải khóa id |

Ai đọc nhãn hiện tại: **viz** (hợp lệ — UI/debug); **eval** `matched_key = rec.tag/ep.key`
(`eval/v4.py:356,372,390` — đo "khớp đúng bài nguồn"; là METRIC lúc eval, không phải inference
quyết đầu ra, nhưng `tag` vẫn KHÔNG được lưu trong graph — chuyển sang cache).

## 3. Cách hiểu đầu bài (lời tác giả)

Mạng chỉ có kết nối/trọng số/kiểu — thuần nơ-ron; text/content/label để cache riêng
(id→content) chỉ render/trace/debug. Đảm bảo không AI agent/dev nào inference/train dựa trên
content thay vì trên mạng. Token là ngoại lệ (có surface).

## 4. Ý nghĩa

Siết Luật 0 lên một bậc: không chỉ cấm hardcode từ vựng mà cấm cả *đọc content lúc suy luận*.
Ép mọi năng lực đi qua đồ thị thật → không thể "ăn gian" bằng string-matching. Bonus: mạng
tách khỏi text → dễ swap ngôn ngữ, dễ nén/serialize, dễ chứng minh Luật 0.

## 5. Đề xuất giải pháp (3 phase, mỗi phase đo guard đầy đủ TRƯỚC/SAU)

### Phase 1 — Content-cache + tách NHÃN (nhóm A) — rẻ, làm trước
- `core/content_cache.py`: `ContentCache{ neuron_key → {text, tag, title, source, kind} }`.
  Serialize thành **file .pkl RIÊNG** đi kèm checkpoint (vd `*_t3.content.pkl`), KHÔNG nằm
  trong GraphState pickle chính.
- Gỡ `tag` khỏi `ParentRecord`; `key/title/source` khỏi `AbstractNeuron` → cache. Build script
  ghi cache song song. Inference không đọc cache. **Eval**: `match_ok` đọc cache qua API
  tường minh `cache.tag_of(parent_idx)` (eval = đo, được phép đọc cache; đánh dấu rõ "debug read").
- Viz đọc cache cho label. **Acceptance**: guard 3 checkpoint không đổi (nhãn không tham gia
  inference nên PHẢI không đổi — nếu đổi là có đường inference lén đọc nhãn → bắt được bug).

### Phase 2 — Thuộc-tính-token thay re-parse (nhóm B)
- `Vocabulary` build sẵn cho mỗi token: `is_punct`, `cap_class` (upper/lower/other),
  `fold_group_id` (id nhóm case-equiv). Lưu trong vocab (tầng token — ngoại lệ được phép).
- Thay `_is_punct(token_of(t))` → `vocab.is_punct(t)`; `.isupper()` → `vocab.cap_class(t)`;
  `_fold` so sánh → so `fold_group_id`. ~48 call-site, cơ học.
- **Acceptance**: metric BẰNG HỆT trước (thuần refactor đọc-cùng-thông-tin-nguồn-khác).

### Phase 3 — phrase_assoc khóa id (nhóm C)
- Khóa `_phrase_assoc` bằng `(token_id, token_id)` hoặc **phrase-neuron-id** thay surface tuple.
- fold-group cho case (từ Phase 2). Rebuild assoc trên `_t3`.
- **Acceptance**: `related_phrases` trả cặp giống Phase-0 (chỉ đổi khóa nội bộ).

## 6. Đề xuất thay đổi cụ thể

| File | Phase | Thay đổi |
|---|---|---|
| `core/content_cache.py` | 1 | **Mới** — ContentCache + load/save sidecar |
| `core/parent_store.py` | 1,3 | gỡ `tag`; `_phrase_assoc` khóa id |
| `core/parent_store.py` (AbstractNeuron) | 1 | gỡ `key/title/source` → cache |
| `core/state.py` | 1 | load/save content-cache sidecar cạnh .pkl |
| `core/vocab.py` | 2 | thuộc-tính token: is_punct/cap_class/fold_group |
| `hca/*.py`, `parent_store` | 2 | thay ~48 re-parse → thuộc-tính-token |
| `scripts/build_*.py` | 1 | ghi cache song song |
| `eval/v4.py` | 1 | đọc tag/key qua cache API (debug read tường minh) |
| `viz/export_graph_v3.py` | 1 | label từ cache |
| tests | 1-3 | cache roundtrip; assert inference KHÔNG chạm cache (mock cache = rỗng → metric giữ) |

**Acceptance tổng:** guard QA/Para/Syn/comp/essay/anaphor **không đổi** qua cả 3 phase (nhãn
+ refactor đọc-nguồn-khác không được đổi output); test "inference-blind-to-cache" (dựng state
với cache RỖNG → mọi metric inference giữ nguyên — chứng minh Luật 0.0 tuân thủ bằng máy);
pytest xanh.

## 7. Rủi ro & không làm

- **Rủi ro:** nếu Phase-1 làm metric ĐỔI → có đường inference đang lén đọc nhãn (bug ẩn) →
  đó chính là giá trị của ticket, GHI + vá. Serialization đổi format → cần script migrate
  checkpoint cũ (hoặc rebuild).
- **Không làm:** đổi tầng token-surface (vocab giữ id↔surface — ngoại lệ); đổi logic inference
  (chỉ đổi NGUỒN đọc, không đổi thuật toán); gộp cache vào pickle chính (phải tách file).

## 8. Cập nhật 2026-07-09

- Auto-migrate checkpoint cũ (nhãn nhúng trong graph) **vẫn** trong `checkpoint_io.load_checkpoint`.
- Test `test_load_checkpoint_migrates_legacy_labels` **đã xóa** (TK_019 §7) — không còn duy trì
  regression cho format pickle cũ; ưu tiên sidecar `.content.pkl` chuẩn.

---

## 9. DẤU CÂU LÀ TOKEN BÌNH THƯỜNG (2026-07-15) — gỡ `is_punct`/`PUNCT_TOKENS`

> Chỉ đạo tác giả: *"chúng ta sẽ không phân biệt text hay dấu câu, dấu câu cũng là token
> như bình thường — bỏ tất cả chỗ nào đang hard code dấu câu"*.

### 9.1 Rà soát Luật 0 toàn repo — repo SẠCH hơn README nói

| nhóm | kết quả |
| :--- | :--- |
| Từ vựng tiếng Việt hardcode | **0** — không WH-list, không danh sách động từ, không phân loại câu hỏi |
| Danh sách từ | chỉ `PUNCT_TOKENS` + `_FRAME_TOKENS` (frame protocol — Luật 0 **cho phép**) |
| So string lúc suy luận | **1**: `w0_tok.endswith(":")` (at_boundary) |
| Thuộc tính chữ | chỉ `vocab._cap_class_of` — **tính sẵn lúc build**, đúng Luật 0.0 |
| Regex trên bề mặt | **0** trong `hca/` + `core/` |

→ README §0.0 **stale 2 chỗ**: (a) "~48 chỗ inference re-parse token-string" — TK_012 đã trả
gần hết, nay còn **đúng 1**; (b) mục (C) `phrase_assoc` khóa surface-string — Phase 3 đã đổi
sang fold-group-id. **Nên gạch cả hai khỏi README.**

### 9.2 Gốc: một VÒNG LUẨN QUẨN, và `is_punct` che một BUG

```
_is_match_token():  if vocab.is_punct(tid): return False   ← dấu câu bị loại khỏi thống kê
        ↓
_gdf['.'] không bao giờ được cộng  →  _gidf('.') = 7.168 = CAO NHẤT corpus
        ↓                                (gấp 6× 'Việt' = 1.166)
~24 filter is_punct rải khắp inference tồn tại để CHE hậu quả đó
```

Đếm dấu câu vào `_gdf` → `.` có df≈n → gidf≈**1.097** → thấp nhất → **tự chìm bằng thống kê**,
không cần danh sách nào nói hộ. KHÔNG cần train lại: `slot_chain`+`match_span` có trong
checkpoint, `_gdf` vốn đã dựng lại lúc load.

**Bug `FOLD_STRIP` mà `is_punct` che** (thật, nhưng **KHÔNG** phải nguyên nhân cú tụt — xem §9.3):

```python
fold_of('.') == fold_of(',') == fold_of('?') == fold_of(':') == ''   # CÙNG fold-group!
```

`FOLD_STRIP` strip sạch dấu câu → mọi dấu câu fold về chuỗi RỖNG → **với mạng chúng là MỘT
token**. Bug ẩn nhiều tháng vì `is_punct` lọc dấu câu **trước khi** tới `fold`. Gỡ filter thì
fold `''` tràn vào `_query_content_folds` → `_longest_suffix_cut` khớp trúng **mọi** dấu câu
trong source → cắt hết câu → **agenda RỖNG** (test `test_completion_graph_autoregressive_mode`).

**Hai hằng KHÔNG độc lập — chúng giữ nhau sống.** `is_punct` che bug của `FOLD_STRIP`;
`FOLD_STRIP` làm `is_punct` trông cần thiết.

Vá: `fold_of()` trả `folded or token.lower()` — token strip xong còn rỗng thì giữ mặt chữ.
Mỗi dấu câu có fold-group riêng, mà `Việt.`≡`việt` và `(học)`≡`học` vẫn nguyên.

### 9.3 Đo — và BA CHẾ ĐỘ dễ lẫn (đọc kỹ trước khi trích số)

`match_ids` được **serialize**, nên nó đóng băng luật `_is_match_token` LÚC TRAIN. Sinh ra ba
trạng thái khác nhau, và **rất dễ đo nhầm chế độ**:

| chế độ | query có dấu câu | `match_ids` của cha có dấu câu | rank-1 | MRR |
| :--- | :---: | :---: | --: | --: |
| baseline (còn `is_punct`) | không | không | 61.00% | 0.741 |
| **ckpt CŨ + code mới, chưa resync** | **có** | không ← *bất đối xứng* | **65.00%** | **0.782** |
| **sau TRAIN LẠI** (= có resync) | có | **có** | **59.50%** | **0.678** |

*(lessons, query=prefix 50%, n=200 mỗi arm)*

**ĐÍNH CHÍNH 2 điều từng ghi sai ở bản trước của mục này:**

1. **"+4.0pp" là ARTEFACT.** Nó là hàng giữa — query mang dấu câu còn `match_ids` của cha thì
   không, vì chúng serialize từ luật cũ. Trạng thái đó **biến mất ngay khi train lại**. Từng
   dùng nó để biện minh cho việc land — sai.
2. **"Vá `FOLD_STRIP` biến hồi quy thành cải thiện" — SAI.** Đo trực tiếp trong chế độ đối
   xứng: `fold_of` **có bug** và **đã vá** cho kết quả **Y HỆT** (59.50 / 0.678). Bug fold vẫn
   có thật và vá vẫn đúng (agenda rỗng là hậu quả thật), nhưng nó **không** phải nguyên nhân
   cú tụt.

**Giá thật của việc gỡ hardcode:** ≈ **−1.5pp rank-1 / −0.06 MRR** trên self-retrieval
(43k wiki: −1.5pp/−0.012). **Chỉ đạo tác giả: GIỮ, không revert** — hardcode vi phạm Luật 0,
*"che kết quả yếu bằng heuristic → ảo tưởng sức mạnh"* (README Mục 0). −1.5pp với kiến trúc
sạch hơn +0pp nhờ nạng. Cải tiến dần.

**Nhưng thước self-retrieval có thể sai:** eval discourse trên `network_v10` train lại
(2026-07-16) cho **văn tốt hơn rõ** — dấu câu quay lại đúng chỗ, `multi%` 50→90%. Xem §9.7.

### 9.4 Đã gỡ

`vocab.PUNCT_TOKENS`, `vocab.is_punct()`, `vocab._is_punct` · `parent_store._is_punct()`
(code chết) · `_is_match_token`, `_content_ids` · ~20 call site ở `generate_hebbian` /
`skip_span` / `frame_bracket` / `g2_imitation` / `consolidate` / `ingest_dictionary` /
`ingest_entities` / `export_embedding` · `_strip_leading_punct_ids` (chống `. .` — tri thức
chính tả viết tay) · `verify_v3` miễn-trừ G4 (A/B 7/7 giống hệt; grounding của `.` = 227 vs
vmin 0.04 → miễn trừ thừa suốt).

**KHÔNG còn một dòng code nào hỏi "token này có phải dấu câu không".** 184 test pass.

### 9.5 `at_boundary` — mốc ranh giới: từ danh sách → HỌC ĐƯỢC (đã xong 2026-07-15/16)

`at_boundary` (TK_009) cần biết *"w0 có kết thúc đơn vị không"* → nếu có thì tin trật tự NGUỒN
vô điều kiện (bug gốc: `"Việc đọc sách"→"đọc Việc sách"`). Đây là ngữ nghĩa **KHÁC** "lọc nội
dung", nên gỡ `is_punct` làm nó mất mốc.

Thử 2 tín hiệu dẫn xuất để thay, **cả hai thất bại** — không tách nổi ranh giới khỏi nội dung:

- **Entropy hàng W_seq**: `.`=5.93 nhưng `là`=5.45 (không phải ranh giới, hạng 2), `?`=3.08 <
  `học`=4.39. Entropy đo "đi được nhiều nơi", không đo "kết thúc đơn vị".
- **Tỉ lệ đứng cuối slot_chain**: `.`=0.152, `?`=0.018, nhưng `,`=`:`=`Đáp:`=**0.000** — ngang
  hệt từ nội dung. Dấu phẩy/hai chấm là ranh giới GIỮA câu, không bao giờ kết thúc chuỗi.

→ **Đồ thị không mã hoá "kết thúc đơn vị" vì DỮ LIỆU TRAIN không đánh dấu nó.** Chẩn đoán của
tác giả đúng; hai phép đo trên là bằng chứng.

**Lối ra (ý tác giả): EOS ở CUỐI mỗi đơn vị train** (`_with_eos`). EOS vào `ids` như token
BÌNH THƯỜNG (không đặc cách): W_seq học `. → EOS`, REINFORCED học `đáp → EOS`; nó KHÔNG vào
`match_ids` vì `is_surface_token(EOS)`=False → không nhiễu retrieval; `decode_surface` lọc khỏi
output.

**Tín hiệu = LIFT, không phải P tuyệt đối** (`ar_eos_boundary_min_lift = 2.0`):

```
lift(w0) = P(EOS | w0) / P(EOS) nền        # nền = khối lượng ·→EOS / tổng W_seq, memo 1 lần/state
```

Vì **P phụ thuộc ĐỘ DÀI ĐƠN VỊ** — chốt ngưỡng P tuyệt đối là bẫy đã vấp (`0.25` hiệu chỉnh
trên corpus 3 câu → **trượt** trên mẻ thật ở `P=0.057`):

| corpus | `P(.→EOS)` | nền | **lift** |
| :--- | --: | --: | --: |
| đơn vị 1 câu | **1.000** | 0.167 | **6.0** |
| đơn vị 6 câu | **0.167** | 0.024 | **7.0** |

`P` đổi **6×**, lift gần như **không đổi**. Test `test_boundary_survives_long_units` khoá.

**Lùi an toàn:** checkpoint train trước 2026-07-15 không có cạnh →EOS → `_ends_unit` trả False
→ `at_boundary` lùi về chỉ nhận mốc qua nhãn frame = đúng hành vi cũ, không tệ hơn.

**SỔ NỢ:** thước hiện tại (184 test + self-retrieval + eval discourse) **KHÔNG phủ** ca TK_009
gốc (`"Việc đọc sách"→"đọc Việc sách"`). `at_boundary` vừa đổi cơ chế mà không có test cho ca
đã sinh ra nó — cần kiểm bằng tay hoặc thêm test.

### 9.6 Còn nợ — 2 hằng (tokenizer)

| nợ | file | gỡ được khi |
| :--- | :--- | :--- |
| `FOLD_STRIP` | `core/vocab.py` | tokenizer tách hết `"'()[]` → co còn `.lower()` |
| `_PUNCT_CHARS` | `teaching/align.py` | **gốc** — đổi tokenize = đổi vocab = **mọi checkpoint vô giá trị** |
| `w0_tok.endswith(":")` | `hca/generate_hebbian.py` | chỗ so-string **cuối cùng** trong inference |
| `resync_match_ids` vô điều kiện | `core/state.py` | gate theo "match_ids đã có dấu câu chưa" — nghi là nguyên nhân load chậm |

`FOLD_STRIP` giờ **đúng vai** — *"ký tự strip khi fold `Việt.`≡`việt`"*, KHÔNG phải *"dấu câu là
gì"*. Nó tồn tại vì `_PUNCT_CHARS=",.;:!?"` tách CHƯA HẾT: `tokenize_punct('(học)')` trả
`['(học)']` nguyên cục → bỏ `FOLD_STRIP` là `(học)` ≠ `học`. Ba danh sách không khớp nhau
(`_PUNCT_CHARS` thiếu `…` → token `…` **chưa từng tồn tại** → nhánh `is_punct('…')` chết từ đầu).

Cả ba gỡ được **cùng một lượt** ở đợt train đổi tokenizer.

### 9.7 Kết quả sau train lại (`network_v10`, 2026-07-16) — **verify 10/10 PASS**

**Verify** (`scripts/verify_checkpoint.py`, xem `VERIFY_CHECKPOINT.md`): 8/10 lần đầu → **10/10**
sau khi sửa 2 lỗi CODE (checkpoint train đúng, KHÔNG cần train lại):

```
'.' gdf=38,976  gidf=1.513          (đối chiếu 'Việt' 3.781, 'là' 1.894) — dấu câu TỰ CHÌM
EOS cuối source_ids: 2000/2000      EOS trong match_ids: 0/2000
P(EOS) nền = 0.00358
'.' P(→EOS)=0.057  →  lift = 16.0   (ngưỡng 2.0)  ends_unit=True; nội dung: KHÔNG nhận nhầm
varlm học mốc kết: 119,093 ctx      (trước: 0)    varlm khớp vocab: 0 id ngoài vocab
```

2 lỗi CODE đã sửa: (1) **ngưỡng P tuyệt đối** → lift (§9.5); (2) **`build_varlm` là đường nạp
THỨ BA** — tự tokenize, không qua `ingest.py` lẫn `pipeline.py` → sidecar không thấy EOS. Sót
**ngay trong lúc viết `VERIFY_CHECKPOINT.md` §3 cảnh báo về chính chuyện này**.

**LIFT chứng minh được thiết kế:** cùng `P=0.057` đã làm ngưỡng tuyệt đối `0.25` trượt, nay cho
**lift=16.0**. `P` phụ thuộc độ dài đơn vị; lift thì không.

**Eval discourse (nets-ON = đường mặc định)** — dấu câu **quay lại và đặt đúng chỗ** (trước đây
agenda bị lọc sạch nên A3 không thể sinh):

```
CŨ (còn is_punct): 'phở là món ăn rất phổ biến ở Việt Nam  nhưng phở Hà Nội có những cách…'
MỚI:               'phở là món ăn rất phổ biến ở Việt Nam , nhưng phở Hà Nội có những cách … .'
```

`multi%` 50→90% (`common_noun`), 60→100% (`statements_doc`) — nhưng **phần lớn là tautology**:
`multi` = `sents>=2 or ("," in out …)`, tức nó **đo "có dấu câu không"**.

### 9.7.1 PHÁT HIỆN LỚN: EOS/lift/varlm là NO-OP trên đường mặc định

Rebuild sidecar (**119 093 ctx EOS**) + `at_boundary` bật (**lift=16.0**) → eval summary **GIỐNG
HỆT** bản trước (chỉ `statements_doc` lệch 1 char). Không phải chúng vô dụng — **đường sinh mặc
định không gọi tới chúng**:

```python
# generate() — eval chạy KHÔNG có --no-verbatim-nets
tail = _conv_pool_tail(...)
if tail:
    return _emit_tail(..., verbatim_only=not config.ar_tail_as_agenda_opener)  # False → True
# _emit_tail:1023
if verbatim_only or not config.ar_tail_as_agenda:
    return generated + tail        # ← RETURN, KHÔNG vào generate_autoregressive
```

**`at_boundary` và `ar_varlm_weight` đều sống BÊN TRONG `generate_autoregressive`.** Lưới verbatim
return trước → cả hai là no-op. Xác nhận lại điều đã đo 15-07, nay trên checkpoint mới.

### 9.7.2 Eval `--no-verbatim-nets` — đọc HẾT 10 ca khác nhau

| bucket | nets-ON copy | nets-OFF copy |
| :--- | --: | --: |
| `common_noun` | 0.51 | 0.51 |
| `conv` | 0.22 | 0.21 |
| `frame` | 0.35 | **0.61** ↑ |
| `history_wiki_para` | 0.57 | **0.61** ↑ |
| `statements_doc` | 0.49 | **0.56** ↑ |

`copy` **TĂNG** khi tắt lưới — ngược trực giác. Lý do: ON sinh `"Lịch sử : Tiêu đề : Chúa Trịnh…"`
(scaffolding lạc, LCS thấp); OFF nối tiếp đúng mạch văn Wikipedia (LCS cao). → **`copy` không tách
được "chép mạch lạc" khỏi "sinh tốt trùng nguồn"** — thước lại mù thêm một chiều (xem §9.8.4).

**Chấm tay 10 ca khác nhau (19 prompt×mode; varlm=0):** **OFF hơn 5 · hoà 2 · OFF KÉM 3.**

| ca | phán |
| :--- | :--- |
| `'8 ngày qua đã xuất hiện ba'` → OFF: *"ba **vùng thả dù** : "Natasha" (phía tây bắc Điện Biên Phủ)…"* | **OFF thắng rõ** — nối đúng cú pháp + nghĩa |
| `'Tuy nhiên trong một số trường hợp'` → OFF: *"**không chia đoạn văn có thể gây khó khăn**…"* | OFF thắng — đúng chủ đề; ON dội `Hội thoại :` lạc |
| `'Tây Sơn vẫn còn'` → OFF: *"**kiểm soát Bắc Bộ** . Sau khi Mạc Kính Điển chết (1580)…"* | OFF hơn — nhưng **lệch fact**: Tây Sơn (TK18) → nhà Mạc (TK16) |
| `'những khái niệm chính'` → OFF: *"chính **vì thế mà** người đời sau…"* | **splice** — bám chữ `chính` rồi nối bừa |
| `'Chương 277'` → OFF: *"nhìn chằm **Mạn** chằm"* | **OFF HỎNG** — A3 chèn token phá từ láy |
| `'Hỏi: Nhà Lý là gì Đáp:'` → ON: đáp trọn vẹn (Lý Công Uẩn, 974, 1009, 1010); OFF: mảnh giữa đoạn `: "Thân Vệ là người khoan thứ…"` | **ON THẮNG** |

**→ RÚT đề xuất đổi mặc định `ar_tail_as_agenda_opener=True`.** Lưới verbatim **hại ở prompt mở**
(dội scaffolding lạc đề) nhưng **CỨU ở QA/frame** — đúng như tên gọi và README §0.4.1 *"giữ đúng
fact câu đã thuộc"*. Đổi toàn cục = đánh đổi 3 ca QA lấy 5 ca văn, mà QA là thứ guard dùng.

**Việc đáng làm, theo thứ tự:**

1. **Dọn scaffolding khỏi CORPUS** (`Lịch sử : Tiêu đề :`, `Hội thoại :`, `Truyện :`) — nó nằm
   trong **dữ liệu train**, hại cả hai chế độ. Sửa gốc rẻ hơn và không đánh đổi.
2. **varlm vẫn NO-OP** kể cả `weight=1.0` + nets-off + sidecar 119k ctx EOS (`0` = `0.5` = `1.0` ở
   4/5 bucket; chỉ `conv` lái được 0.21→0.25). Bí ẩn thật: đo độc lập nó là **kênh mạnh nhất**
   (+7.3pp, README §0.4.2) mà lắp vào không lái. **Đáng đào nhất.**
3. Lưới verbatim: **GIỮ** — nhưng đáng tách theo bucket (QA giữ / prompt mở tắt) thay vì một cờ
   toàn cục.
4. A3 chèn token phá text (`chằm Mạn chằm`) — lỗi sinh thật, chưa có test.

### 9.8 Bài học đo lường (đắt nhất đợt này)

1. **Checkpoint nhỏ đánh lừa NGƯỢC DẤU** — 3 lần: CO có hướng (+1.75 → −0.12), đa neo K=2
   (+13.3 → −0.16), ngưỡng EOS (P=1.000 → 0.057 → **trượt**). Corpus toy đồng nhất (1 câu/đơn
   vị) là nguồn gốc. Xem memory `long-range-cooccurrence-negative`.
2. **Đo nhầm CHẾ ĐỘ** (§9.3) — "+4.0pp" là artefact của trạng thái bất đối xứng sẽ biến mất
   khi train lại.
3. **Đo mô hình KHÔNG tồn tại** — từng dựng `W_seq + 0.25*CO` rồi báo "bug −11pp"; repo không
   bao giờ cộng W_seq với CO (`SEQUENTIAL` tách khỏi STRUCTURAL, đọc qua kênh micro có
   row-normalize). **Đọc kiến trúc TRƯỚC khi dựng harness.**
4. **Thước có thể mù**: `overlap` luôn=1.00 (đo %token-prompt-trong-output, mà output luôn bắt
   đầu bằng prompt); `multi` đo "có dấu câu không". Report TK_017 §16 đọc `overlap=1.00` thành
   "xu hướng verbatim rất mạnh" — kết luận tình cờ đúng, **bằng chứng không hỗ trợ**.
5. **Nhiều đường nạp** — sửa một chỗ ≠ sửa hết. Ba đường EOS, và tôi sót hai lần.

### 9.6 Checkpoint chia hai loại sạch — đợt train không phải lo migrate

Đo 2026-07-15: **tiền-R0** (`grade_v3.pkl` — `tokenize` không tách → **0/2025 token dấu câu**,
`is_punct` chưa từng chạy, cũng không lưu `source_ids`/`slot_chain`) và **ISG-era**
(`network_v10`, `grade_lessons_test`, `grade_v3_big_a2_t3_isg` — 6 token dấu câu **và** có
`slot_chain` → `match_ids` tái tạo được). Không có ca giữa.
