# TK_018 — BÀN GIAO CHO CURSOR (Claude hết session, 2026-07-05 đêm)

| Field | Value |
|-------|-------|
| **ID** | TK_018 |
| **Status** | 🔵 HANDOFF — đọc file này TRƯỚC khi làm bất kỳ gì |
| **Checkpoint MỚI NHẤT** | `checkpoints/network_v4_full.pkl` (= convpool + 15 000 pattern_templates sidecar; parents 64 573; vocab 77 464). Default paths hiện trỏ `network_v4_convpool.pkl` |
| **Ticket nền** | TK_015 (từ điển/thước) → TK_016 (D1′/D3) → TK_017 §1-15b (gate/R1-R3b) — đọc TK_017 là đủ 90% context |
| **Test suite** | 304+ pass `-m "not slow"` (xác nhận cuối phiên Claude) |

---

## 1. TÓM TẮT THÀNH QUẢ TOÀN CHUỖI (đều có số đo trong TK_015/016/017)

| # | Thành quả | Số chốt | Ticket |
| :-- | :--- | :--- | :--- |
| 1 | **Thước không bão hoà** `next_token_lm` (log₂rank, held-out 200 câu essay) | baseline `network_v4` = **4.028 bits / top1 22.1%**; nhạy Δ1.0 giữa checkpoint | TK_015 §5b |
| 2 | **C0**: 2 tường đo được (đa-nghĩa blur; cạnh xa seq=0) | sim(bóng,vôi)=0.28; nếu→thì seq=0.00 | TK_015 §5c |
| 3 | **D1 substrate từ điển**: 36 764 sense + 14 POS hub (tách nghĩa thật: cân#3→gram/kg) | flat-dump ÂM (4.40) → D1′ fix về 4.028 | TK_015 §5d, TK_016 §7d |
| 4 | **D3 FRAME_BRACKET**: 35 205 cạnh xa có hướng (tuy→nhưng 3.57, nếu→thì 2.57) | kênh phẳng ÂM (+0.45); runtime gate wired Δ−0.052 | TK_016 §7b, TK_017 §8-11 |
| 5 | **R2 q·k directed** (`attention_qk.py`, doc 03 lần đầu build thật) | ruler 4.028→**3.926** (kênh đầu tiên DƯƠNG aggregate); unseen Δ=0 | TK_017 §13 |
| 6 | **2 ĐỊNH LUẬT** (5 thí nghiệm âm hội tụ): (1) cắm-phẳng-luôn-pha-loãng → cần GATE/CHỌN; (2) trần order-1 Markov greedy → continuation = truy hồi mẫu + completion | TK_017 §7b③, §13, §14 |
| 7 | **R3b: CONTINUATION MẠCH LẠC ĐẦU TIÊN** — pool văn nói two-tier (priority=0.3) + `_conv_pool_tail` phát đuôi mẫu | 5-6/7 prompt khung mạch lạc, 0-1s/prompt, guard QA/ruler NGUYÊN | TK_017 §15, §15b |
| 8 | **Observability**: load/build ghi log 5 bước `out/logs/`; vá race server; UI panel "Tách nghĩa" + `/api/embedding/senses` | — | TK_016, memory |

**Data assets:** `data/dict/vi_dict_raw.jsonl` (36 764 nghĩa), `data/conv/{stories,chat}_raw.jsonl` (10k items), `data/eval_v4/heldout_lm.jsonl` (200 câu — thước; KHÔNG sửa file này).

## 2. BẢN ĐỒ CODE CHANGES (staged + unstaged, theo việc)

| Việc | File | Ghi chú |
| :--- | :--- | :--- |
| Thước | `eval/metrics.py::next_token_lm`, `scripts/eval_lm_ruler.py` | bpb cũ = proxy giả, đừng dùng |
| Từ điển D1′ | `training/ingest_dictionary.py`, `scripts/{fetch_vi_dictionary,build_dict_checkpoint}.py`, `EdgeKind.{SENSE,POS_ANCHOR,SYNONYM,ANTONYM,LEXATTR}`, vocab `Sense:/POS:` prefix | KHÔNG flat-dump parent |
| Perf | `parent_store.add_example(defer_index=)`, `embedding.concept_meanpool` 1-pass, `viz/server get_state` lock, `GraphState.load(logger=)` | |
| D3 | `training/frame_bracket.py`, `hca/frame_bracket_gate.py`, `EdgeKind.FRAME_BRACKET`, `scripts/{probe_d3_brackets,eval_d3_ruler,eval_d3_bracket_positions}.py` | mine giờ PERSIST tại build_conv_pool |
| R1 route | `generate_hebbian`: `_filter_plan_on_focus`, opener-continuation check, gate trong `_wseq_fallback`; cờ `ar_fallback_focus_gate`, `ar_opener_continuation` | |
| R2 | `hca/attention_qk.py`, blend trong `_wseq_fallback`; cờ `ar_qk_weight`, `ar_continuation_topic_*` | |
| R3/R3b | `scripts/{fetch_vi_conversational,build_conv_checkpoint,build_conv_pool}.py`, `generate_hebbian::{_conv_pool_indices,_conv_pool_tail,_longest_suffix_cut}`; cờ `ar_conv_pool`, `conv_pool_min_score=0.10`, `generation_min_priority=0.5` (Cursor §15b) | |
| R4b (Cursor, CHƯA note đủ) | `state.pattern_templates` sidecar + serialize; `generate_hebbian::{_tail_from_pattern_templates,_emit_tail,_prompt_is_open_continuation,_finish_or_pattern}`; cờ `ar_tail_as_agenda*`; `scripts/build_essay_opener_pool.py` → `network_v4_full.pkl` | xem REVIEW §3 |
| Viz | `viz/export_embedding.py::embedding_senses`, `viz/server/main.py` route `/api/embedding/senses`, `viz/web/src/{EmbeddingExplorer,api,styles}` + build | |

## 3. ⚠️ REVIEW ĐỢT R4b CỦA CURSOR (Claude, cuối phiên) — 1 VI PHẠM LUẬT 0 PHẢI SỬA

Đợt unstaged mới nhất (R4b: tail→agenda A3 + pattern_templates sidecar + mở route pattern-completion) — **ý tưởng ĐÚNG** (đúng nấc "sinh thật" trong sổ nợ R3b §15), NHƯNG:

1. **🔴 VI PHẠM LUẬT 0 — PHẢI SỬA TRƯỚC KHI COMMIT:** `_prompt_is_open_continuation`
   (generate_hebbian ~dòng 575) hardcode **danh sách WH tiếng Việt ở runtime inference**:
   `{"ai","gì","nào","sao","đâu"}`, `"bao"+"nhiêu"`, `"mấy"`, `{"ai","hỏi"}` — đúng thứ README
   Mục 0 CẤM ("danh sách WH / loại câu hỏi"). Đây là vá che regression Para bằng heuristic
   ngôn ngữ ("ảo tưởng sức mạnh").
   **Fix Luật-0-sạch (máy CÓ SẴN — TK_014):** prompt là "câu hỏi" ⇔ `ps.query_anchor_clusters(
   q_ids)` chạm cụm WH/anchor ĐÃ HỌC ("là gì", "ở đâu"… là ClusterNeuron ISG có sẵn, TK_014
   build đúng cho việc này) HOẶC chứa frame token. Thay hàm bằng check đó, đo lại Para/Syn.
2. **🟡 Perf:** `_tail_from_pattern_templates` quét TUYẾN TÍNH 15 000 templates ×
   `_longest_suffix_cut` MỖI prompt → cần index (token-cuối-fold → template ids) trước khi
   scale templates.
3. **🟡 Chưa đo/chưa note:** `network_v4_full.pkl` + §R4b chưa có mục ticket với số đo A/B
   (tail-as-agenda ON vs OFF: mạch lạc giữ? verbatim_rate giảm?); eval full ghi nhận
   composition 60/0/93 (n=5) + **anaphor 33%** ("sản" đại từ giả lọt) trên convpool —
   **chưa diff với network_v4 baseline** → chưa biết có regression anaphor/composition không.
4. **🟡 Suite từng ĐỎ khi bàn giao:** `paths.py` đổi default → `network_v4_full.pkl` nhưng
   `tests/test_paths.py` chưa cập nhật (2 fail). **Claude đã fix** (thêm full vào allowed set
   + thứ tự ưu tiên full→convpool) → **309 pass**. Bài học: đổi default checkpoint phải chạy
   suite.
5. **🟡 Default = `network_v4_full.pkl` nhưng checkpoint này CHƯA qua guard đo riêng** (eval
   full mới chạy trên convpool; full = convpool + templates, kỳ vọng giống — nhưng route
   pattern-templates MỚI có thể đổi Para/Syn khi thiếu frame). H2 phải đo TRÊN full.
6. 🟢 Điểm tốt: sidecar templates KHÔNG vào ParentStore (đúng bài học dilution);
   `generation_min_priority=0.5` chặn pool khỏi best_match (fix Para 88 đúng cách); frozen
   vocab; `_longest_suffix_cut` fix khớp cụm; cờ ablate đầy đủ; `_emit_tail` (tail→agenda A3)
   đúng nấc "sinh thật" của sổ nợ R3b.

## 4. VIỆC CẦN LÀM (thứ tự ưu tiên)

| # | Việc | Acceptance |
| :-- | :--- | :--- |
| **H1** | **Sửa vi phạm Luật 0** (§3.1): thay WH-list bằng `query_anchor_clusters`/frame check | Para/Syn/QA giữ (100/88/67); prompt mở vẫn route pattern; grep không còn danh sách WH |
| **H2** | **Đo + note R4b** vào TK_017 §15c: A/B `ar_tail_as_agenda` on/off trên 7 prompt khung + ruler + guard full; diff anaphor/composition convpool-vs-v4 (33% là regression hay artifact gold?) | bảng số trong ticket |
| **H3** | Index cho `pattern_templates` (§3.2) | O(1) lookup; smoke 7 prompt <1s |
| **H4** | Quyết checkpoint default: `network_v4_full` nếu H1+H2 sạch → cập nhật `paths.py` + viz + README §0.6 + v4.md nhật ký | eval_v4 full + 304 test |
| **H5** | Sổ nợ còn: `bind_sense_resonance` batch-hoá rồi CHẠY (TK_016 §7d; rank_cache vô dụng — cần index đảo token→sense); "Mèo là gì" ranking bug (TK_017 §12); D2 head-tách-POS; D4 stack neo | từng món một, đo trước |

## 5. CÁCH TEST (copy-paste)

```bash
# 1. Suite (phải ≥304 pass)
python -m pytest tests/ -q -m "not slow"

# 2. Thước — KHÔNG được tụt khỏi 4.028/22.1 (checkpoint nào thì ghi rõ)
python scripts/eval_lm_ruler.py --checkpoint checkpoints/network_v4_full.pkl

# 3. Guard eval (QA 100/100, Para 88, Syn 67)
python scripts/eval_v4.py --checkpoint checkpoints/network_v4_convpool.pkl --fast \
  --skip-section essay --skip-section gen-ar

# 4. Continuation đọc tay (thành quả R3b — PHẢI giữ mạch lạc)
#    prompt KHÔNG frame: "nếu trời mưa" / "nếu trời mưa thì" / "vì em chăm học" /
#    "khi mùa hè đến" / "tuy vất vả" — kỳ vọng câu văn đọc được, KHÔNG salad ". , và các"
#    QA có frame:  "Hỏi: Thủ đô Việt Nam là gì ? Đáp:" → "Thủ đô Việt Nam là Hà Nội."

# 5. Graph UI
AIVIE_CHECKPOINT=checkpoints/network_v4_full.pkl python -m viz.server.main
#    Parent Explorer (A3): prompt khung ở trên; Embedding E → panel "Tách nghĩa": cân / đá / đông

# 6. Log mọi process dài ở out/logs/ (tail -f) — quy tắc bắt buộc của tác giả
```

## 6. QUY TẮC LÀM VIỆC (tác giả — bắt buộc)

1. **Đo TRƯỚC khi tin; số âm tính là kết quả hợp lệ — báo thẳng, ghi ticket.** Mỗi cơ chế mới: A/B + guard (QA/Para/Syn + ruler) + đọc tay.
2. **Luật 0**: cấm hardcode từ vựng/WH-list ở runtime inference (H1 là ví dụ). Luật 0.0: inference không đọc content-cache.
3. **Luật 3.5**: cơ chế đo yếu KHÔNG tắt — BẬT + ghi SỔ NỢ với giá đã đo.
4. **Mọi process dài ghi log ra file** (`util/progress.get_run_logger`) — kể cả load.
5. **Tác giả TỰ commit** — không commit thay.
6. Thước cẩn thận: metric fire theo GOLD = oracle (cận trên), không phải gain khả thi (bài học Mode B TK_017 §11 F1).
