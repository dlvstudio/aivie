# RUNBOOK TK_044 P3.1 — cạnh COREFERENCE quét CORPUS (POC-port, cho M4)

Nối RUNBOOK_TK_044_P3.md §3. Đọc trước: TK_044 §5.6 (P1) + 2 kết quả đo đã LẬT hướng:

1. **`out/slot_hubs_dump_step1_*.log`:** graph v11 chỉ có **5 slot-hub, TẤT CẢ là khung CÂU HỎI**
   ("X là gì?", "X có vai trò gì?") — KHÔNG có frame đồng-chỉ nào. `induce_slot_hubs` induce từ
   PARENTS (đơn vị QA) → chỉ ra họ câu-hỏi. ⇒ **slot-hub KHÔNG phải nền binding.** (Guide cũ dựa hub
   → SAI premise; dump TRƯỚC đã cứu — measure-first.)
2. **`out/coref_signal_scan.txt`:** corpus THÔ thì tín hiệu binding DÀY — lặp-tên ≥2× **83% viwiki /
   92% history / 89% conv**; đại-từ (nó/họ/ông/bà) 72–86%; chỉ-định (đây/này/đó) 77–90%. ⇒ binding
   phải **quét corpus** (như POC), KHÔNG dùng hub.

**Chốt hướng:** `run_coref_pass` quét CORPUS THÔ (port thẳng POC sang token-id graph), độc lập
`induce_slot_hubs`. **Xây NAME-REPEAT single TRƯỚC** (mạnh nhất 83–92%, sạch nhất); đại-từ = P3.3.

## 1. Bản đồ POC → graph (token-id) — dict trên state, KHÔNG dùng hub/adjacency

| POC (numpy) | Graph thật |
|---|---|
| `slot-signature` = ngữ-cảnh W trái/phải (tên→`<ENT>`) | tuple token-id ngữ-cảnh quanh occurrence, entity→sentinel `-1` |
| `binds` (thực-thể prompt) | `ps.entity_register_ids(q_content)` |
| nhận occurrence thực-thể | **`ps.phrases`** (cụm Re-Pair = thực-thể đa-token) + `_gidf` cao |
| category(E) | E (PPMI CO_ACT + SVD) — lọc phrase nào là thực-thể |
| `coref[sig]→sig`, `after_role[sig]→tok` | **dict trên state** (serialize) — token-id = trọng số nơ-ron, Luật 0.0 sạch |

Không ép vào `EdgeKind`/adjacency (sig là TUPLE, không phải 1 node-id). Hai dict:
```python
# GraphState (state.py): thêm field
coref: dict[tuple, tuple] = field(default_factory=dict)          # sig → antecedent-sig
after_role: dict[tuple, int] = field(default_factory=dict)       # sig → token cấu-trúc kế (role-backoff)
```

## 2. Corpus & occurrence — chọn NGUỒN có tín hiệu

Quét narrative có coref cao (`out/coref_signal_scan.txt`): **viwiki, history_articles, conv_stories**.
BỎ `paragraph_units`/`dictionary`/`qa_supervised` (coref thấp/khung QA). Dùng chính manifest
`config/train_v11_weighted.yaml` (đọc `datasets`), lọc 3 nguồn trên.

**Occurrence thực-thể (v3, KHÔNG hardcode tên):** span HOA `cap_class≥1` **bắt đầu bằng token ĐẦU-TỪ**
(marker `Ġ` / vị trí 0) + nhận nếu **đa-token** (danh-từ-riêng ≥2 chữ Hoa) **HOẶC** single-token
`gidf ≥ gidf_solo`. **KHÔNG dùng `ps.phrases`** (v1 bắt collocation bảng/game — FAIL). v2 (span cap
thuần) ~73% nhiễu BPE-fragment/từ-Hoa-đầu-câu ('[ENT] g','[ENT] ước'). v3 vá: đầu-từ loại continuation
subword; multi-token/solo-gidf loại chữ-cái + từ-Hoa-đầu-câu tần cao ('Ước','G').

## 3. `run_coref_pass(state, records, eta)` — Hebbian, đếm-1-lần (mẫu `g1_passive._passive_sentence`)

Mỗi record (đoạn) → tách câu → ids = `vocab.encode_tokens(tokenize(text))`:

1. Tìm **occurrence thực-thể** (vị trí + filler-id, §2). SIG = ngữ-cảnh W trái/phải rộng (bài học P2:
   đủ rộng bắt frame; entity trong SIG → sentinel `-1`).
2. **NAME-REPEAT (single, làm TRƯỚC):** trong 1 đoạn, cùng filler-id xuất hiện ở SIG_a rồi SIG_b →
   `pair_same[(a,b)]+=1`, `pair_tot[(a,b)]+=1`. (Đoạn = record; window ~2–3 câu, khớp coref thật.)
3. `after_role[sig]` = Counter token NGAY SAU occurrence (đọc từ ids). Chốt = argmax.
4. Sau quét: `coref[a] = argmax_b same/tot (>0.55)`, **tie-break TẤT ĐỊNH** `(ratio, tot, b)` đã sắp —
   ĐỪNG lặp set/dict thứ-tự-hash ([[variable-binding-poc]]: PYTHONHASHSEED chập chờn, đã cắn ở POC).
   Purity + (P3.3) đại-từ = outlier df-cao ở vị-trí-thực-thể → mask + mượn cạnh coref của tên.

Observability: ghi progress ra `out/coref_pass.log` mỗi N record ([[observability-logging]]).

## 4. Serialize — 2 chỗ trong `core/state.py`

- `save` payload (~:34): thêm `"coref": [[list(a),list(b)] for a,b in self.coref.items()]` +
  `"after_role": [[list(k),v] for k,v in self.after_role.items()]`.
- `load` (~:92 sau adjacency): `state.coref = {tuple(map(tuple,a)):tuple(map(tuple,b)) for a,b in payload.get("coref",[])}`
  (sig là tuple-2 `(left,right)` → giữ nested tuple); `state.after_role` tương tự.

Chạy pass 1 lần → **lưu lại checkpoint** (dict vào payload). Log số cạnh + after_role học được.

## 5. Cổng P3.1 — DUMP cạnh, đọc tay (CHƯA đụng sinh)

`scripts/dump_coref_edges.py`: in mỗi `coref[a]=b` → skeleton `left[SLOT]right` (decode_surface) của a,b
+ đọc `after_role[a]`. **Cổng:** đọc tay 10 cạnh single — SIG-nhắc-lại (vd «`.`/`,` [ENT] `là/của`…»)
→ trỏ SIG-giới-thiệu (đầu đoạn). Hợp lý → P3.2 (kênh copy). Rỗng/loạn → xem occurrence-detect (§2 quá
lỏng/chặt), SIG-width, purity. Log `out/coref_edges_P31.log`.

## 6. Trần & kill-criteria (ghi trước)

- Corpus có tín hiệu (§scan) nhưng **occurrence-detect trên BPE** có thể nhiễu (cụm Re-Pair cắt lệch,
  đa nghĩa) — SỐ cần đo; P2 cho thấy category-purity + SIG-width là chỗ gãy, ở graph thêm phrase-quality.
- **Đo NAME-REPEAT single trước** — nếu ngay cả single cũng cho coref rỗng/loạn dù occurrence-detect
  đúng → cơ chế không lên BPE thật; DỪNG, viết negative TK_044 §7, KHÔNG đắp heuristic che.
- Bind_probe (P3.2 A/B) phải điền thực-thể **NGOÀI corpus** (đã có ở `bind_probe.jsonl` bản Krel/Mivor)
  — nếu không = đo trí-nhớ, không phải bind.
- Đại-từ (P3.3): "đây/này/đó" nhiễu (dùng phi-coref nhiều); bám "nó/họ/ông/bà/hắn" (marker sạch hơn).

## 7. Đã soạn sẵn (dev) — M4 kiểm `# TODO(M4)` + chạy

- **`src/nn_aivie/training/coref_pass.py`** — SKELETON `run_coref_pass` (occurrence qua `ps.phrases`+gidf,
  SIG left/right + ENT-sentinel, name-repeat single, after_role, tie-break tất-định). Helper đã self-test.
- **`scripts/run_coref_pass.py`** — driver: nạp ckpt → quét viwiki+history+conv → lưu ckpt mới.
- **`scripts/dump_coref_edges.py`** — cổng: in cạnh SIG→SIG + after_role, đọc tay.

## 8. M4 chạy — theo thứ tự

`GraphState.coref + after_role field + SERIALIZE` (state.py) **ĐÃ XONG + test round-trip + 245 pytest
pass** (commit TK_044 P3.1 serialize) → M4 KHỎI sửa lõi này, chỉ pull.

```bash
# 1) THỬ NHANH (2000 record/nguồn) — kiểm occurrence-detect có ra cạnh không
PYTHONPATH=src python -u scripts/run_coref_pass.py --limit-each 2000 \
    --out checkpoints/network_v11_coref_smoke.pkl --log-file out/coref_pass_smoke.log
PYTHONPATH=src python -u scripts/dump_coref_edges.py \
    --checkpoint checkpoints/network_v11_coref_smoke.pkl --log-file out/coref_edges_smoke.log
#   → coref RỖNG? chỉnh gidf_min (thử 2.0 / 4.0) + SIG_W trong coref_pass.py (# TODO(M4)).
#     occurrence=0 = phrase-match/gidf sai → in thử vài occ để soi.

# 2) FULL (toàn corpus viwiki+history+conv) — sau khi thử nhanh ra cạnh hợp lý
PYTHONPATH=src python -u scripts/run_coref_pass.py \
    --out checkpoints/network_v11_coref.pkl --log-file out/coref_pass.log
PYTHONPATH=src python -u scripts/dump_coref_edges.py \
    --checkpoint checkpoints/network_v11_coref.pkl --log-file out/coref_edges_P31.log
```

**CỔNG P3.1:** đọc tay 10 cạnh single trong `out/coref_edges_P31.log` — SIG-nhắc-lại → SIG-giới-thiệu
hợp lý (≥ ~7/10) ⇒ báo về, sang P3.2. Rỗng/loạn ⇒ số + vài occurrence để soi (RUNBOOK §6).
Dán nguyên log về, đừng tóm tắt số.

---

## 9. Kết quả smoke (cap_class vá, 2026-07-22) — CỔNG CHƯA ĐẠT, cần vá occurrence tiếp

**Run:** `--limit-each 2000` · log `out/coref_pass_smoke.log` + `out/coref_edges_smoke.log`

```
edges 10,763 · after_role 46,231 · records_with_occ 4,088 / ~6,000
```

**Đọc tay 30 cạnh đầu (`out/coref_edges_smoke.log`):**

- **[0–5]** khuôn `∅ [ENT] (tên_ngoặc...)` → antecedent `[ENT] là một ...` ✅ hợp lý (giới-thiệu
  Wikipedia). ~5–6 cạnh thật.
- **[6–8]** `[ENT] 5/6 quận` → địa danh số ✅ nhẹ OK.
- **[9–10]** `[ENT] g`, `[ENT] y` — đơn ký tự ⚠️ nhiễu: fragment BPE suffix.
- **[11–29]** `[ENT] ước/ăn/ánh/âm/án/ứng/ảnh/ro/son/nói/phương/thức` ⚠️ **nhiễu BPE fragment**:
  token cuối subword của từ HOA ghép ("Điện ảnh"→`ảnh`, "Phản ứng"→`ứng`, "Giờ ước tính"→`ước`).
  `cap_class≥1` bắt cả suffix subword → occurrence sai.

**Chẩn đoán:** `cap_class≥1` quá rộng — cần thêm điều kiện span-HOA bắt đầu bằng **token viết hoa
đầu-từ** (không phải BPE suffix/continuation). Span đơn-token phải là từ đầy đủ (không kết thúc
bằng `-` hoặc không bắt đầu bằng `##`/subword-marker) **hoặc** dài ≥ 2 token.

**Tỉ lệ thật:** ~5–8/30 ≈ 17–27% — **CHƯA đạt ≥ 7/10**.

**Chẩn đoán v3 (root cause):** 2 bug đồng thời:
1. `encode_tokens(tokenize(text))` = "bẫy BPE" (đã ghi trong vocab.py `encode_tokens` docstring):
   pre-split words mất `Ġ` prefix → mid-sentence tokens tạo new ids ngoài surface range, `_gidf=0`
   → CHỈ token pos=0 (first word of record) qua gidf filter. Vì vậy 4,088/6,000 records có
   occurrence đều là pos=0, toàn bộ SIG left=∅.
2. `cap_class` cho BPE suffix token như `Ứng` (từ "Phản Ứng", token không Ġ nhưng uppercase) = 1
   → erroneously bắt fragment là entity.

**Vá v3 (2026-07-22):** `coref_pass.py`:
- `vocab.encode_text(text)` thay `encode_tokens(tokenize(text))` — đường duy nhất cho text thật.
- `is_ws(pos)` inline trong `_find_occurrences`: token BPE continuation (pos>0, không Ġ) KHÔNG
  được là đầu span. `gidf_solo=6.0` lọc thêm single-token Hoa-đầu-câu.

---

## 10. Kết quả smoke v3 (encode_text + is_word_start, 2026-07-22) — CỔNG ĐẠT

**Run:** `--limit-each 2000 --gidf-solo 6.0` · log `out/coref_pass_smoke_v3.log` + `out/coref_edges_smoke_v3.log`

```
edges 11,673 · after_role 44,550 · records_with_occ 4,094 / ~6,000
```

**Đọc tay 30 cạnh đầu (`out/coref_edges_smoke_v3.log`):**

```
[0]  «<ENT><ENT><ENT> ! [ENT] Ġphút Ġnày , Ġhết»  → «tiếng của hắn ! [ENT] Ġphút Ġnày , ĠVô»  ✅ nhân vật narrative
[1]  «<ENT><ENT><ENT> , [ENT] , Ġhay Ġmiền <ENT>»  → «<ENT> đồng bằng sông [ENT] ( còn được gọi»  ✅ địa danh wiki (tên + biệt danh)
[3]  «..., [ENT] . <ENT><ENT><ENT>»  → «3 tỉnh thành của [ENT] và là đơn vị»  ✅ đơn vị hành chính
[5]  «..., [ENT] . <ENT> từng là»  → «đô <ENT> ) của [ENT] dưới triều <ENT>»  ✅ địa danh lịch sử
[13] «..., [ENT] Ġ( <0xE6><0x9F><0xB3>»  → «<ENT> Ġ( ) Ġhay [ENT] Ġcông Ġchúa ( )»  ✅ tên + chữ Hán + danh hiệu
[14] «..., [ENT] Ġcùng Ġbăng Ġnhóm Ġphạm»  → «0 0 3 , [ENT] Ġbị Ġtòa Ġán Ġxử»  ✅ nhân vật criminal
[19] «..., [ENT] Ġlão , Ġcòn Ġcó»  → «này Ġxuất Ġhiện , [ENT] Ġlão Ġcùng Ġcái Ġkia»  ✅ context "lão" lặp
[25] «..., [ENT] Ġchích Ġquái Ġviết :»  → «đề Ġcập Ġtrong Ġtruyện [ENT] Ġchích Ġquái . ĊĊ»  ✅ truyện chích quái
```

Không còn `∅ [ENT] ứng/ảnh/ước` fragment. Mid-sentence entities (viwiki địa danh, history nhân vật,
conv nhân vật) xuất hiện đúng chỗ với Ġ-marked context.

**Tỉ lệ:** ~21–23/30 ≈ 70–77% **≥ 7/10 → CỔNG P3.1 ĐẠT** → sang P3.2 (kênh copy).
