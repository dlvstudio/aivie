# TK_019 — DỌN LEGACY: chỉ giữ đường sinh token-by-token (chỉ đạo tác giả 2026-07-06)

| Field | Value |
|-------|-------|
| **ID** | TK_019 |
| **Status** | ✅ DONE (Claude) — 135 test pass; guard đo lại; 1 giá ghi sổ nợ. **Lần 2 (2026-07-09):** hợp nhất A3 multi-only → **131 test** pass. |
| **Chỉ đạo** | Bỏ hết code legacy + test của chúng: anaphora, ghép fact, nối fact bằng đại từ… Chỉ giữ sinh token-by-token tốt. |
| **Kèm** | **H1 (TK_018 §3.1) FIXED** — gỡ WH-list hardcode (vi phạm Luật 0) |

## 1. ĐÃ XÓA (file + tuyến)

| Nhóm | Xóa |
| :--- | :--- |
| **Chuỗi HCA v2 chết** | `hca/{generate,matcher,qa_context,answer_closure,answer_verify,forward,discourse}.py`, `viz/trace.py` + route `/api/trace` + CLI trace/TraceRequest |
| **Anaphora/đại từ** | kênh `entity`/`discourse` trong AR loop; `best_anaphor_for_entity`/`strip_leading_entity`/`pick_discourse_token` (bridge); `parent_store.{build_pron_stats,entity_anaphor_pref}`; `eval_anaphor`+AnaphorCase; cờ `ar_anaphor_enabled/anaphor_pref_weight/ar_entity_*/ar_discourse_*/ar_repeat_entity_penalty` |
| **Ghép fact/compose** | nhánh `multi=True` (plan/generate — giữ chữ ký deprecated); `pick_facts_coherent`; `compose_episode_weight`; `eval_composition` + section; `generate_composed` = shim → crisp |
| **Teach-pack v2** | `teaching/{compile,parse_nvl,parse_structured,position,segments,load,imprint}.py` (GIỮ `align.py`+`ast.py` — tokenize dùng khắp nơi); CLI teach/teach-pack/convert; `training/g3_dopamine` (runner bỏ nhánh g3); g2 gỡ block QFocus/fingerprint (giữ REINFORCED gold) |
| **Eval/scripts v2** | `eval/{niah,qa_v3}.py`; `qa_focus` viết lại tối giản (QACase+load); `bits_per_byte` proxy giả; `scripts/{eval_suite,try_prompts,essay_demo,export_viz,prompt_one,inference_samples,run_train_eval,train_cumulative,dump_copula_rules,debug_*,build_*curriculum*}.py` |
| **Tests** | ~40 file test legacy (teach-based/v2/anaphor/compose). GIỮ + SỬA test đường sống: eval_v4, generate_hebbian, parent_store, pattern_completion, content_cache, ingest, g1, seg_graph, frame_bracket_gate, paths… |

## 2. MỚI: `hca/readout.py` — hàm sống gom về một chỗ

`decode_surface` · `mask_non_surface` · `apply_repetition_penalty` (thuần window, **không phạt token prompt** — đo: phạt prompt làm QA rớt) · `spreading_activation_read` (copy NGUYÊN semantics forward.py). Mọi import cũ (`hca.generate`/`hca.matcher`/`hca.forward`) → `hca.readout`.

## 3. H1 — Luật 0 FIXED

`_prompt_is_open_continuation` (R4b Cursor) chứa WH-list tiếng Việt hardcode → **thay bằng
`ps.query_anchor_clusters`** (TK_014): câu hỏi ⇔ query chạm cụm WH/anchor ĐÃ HỌC. Đo: Para
"Người chữa bệnh là ai" (không frame WH-list cũ chặn) vẫn đúng "Bác sĩ chữa bệnh." ✓

## 4. GUARD SAU DỌN (network_v4_full, đo 2026-07-06)

| Bộ | retr | gen-hy | vb-hy | Trước dọn |
| :--- | :---: | :---: | :---: | :--- |
| QA val | **100%** | **86%** | 14% | 100/100(?)/14 |
| Paraphrase | 88% | 88% | 62% | 88 |
| Synonym | 67% | 67% | 17% | 67 |
| Ruler | **4.028 / 22.1%** | | | = baseline NGUYÊN |
| Continuation 5 prompt | mạch lạc GIỮ ("nếu trời mưa tầm tã."; "tuy vất vả đến mức nào." — **hết nhiễu kiếm hiệp**) | | | |
| Tests | **135 pass** (suite chỉ còn đường sống, chạy 3-40s thay ~90s) | | | |

## 5. SỔ NỢ TK_019 (giá đã đo + quyết định)

1. **QA gen-hy 100→86 (−1 ca "Đoàn kết giúp em"→"mạnh mẽ" thay "mạnh hơn")** — giá của bỏ kênh
   entity/discourse (chúng phá thế hòa ca này). Đã đo 2 lưới cứu: `essay_ar_min_cov=0.99` và
   key-token-gidf — **CẢ HAI đẩy vb 14→100%** (máy tra cứu, phản Trụ 4) → KHÔNG lưới; giữ sinh
   thật. Hướng fix thật: bigram "mạnh→hơn." thua "mạnh→mẽ" = việc của substrate chuỗi (D2 head
   / order-2), không phải thêm kênh.
2. `retr` QA vẫn 100% — trí nhớ mạng NGUYÊN VẸN; chỉ decode trượt 1 ca.
3. Repetition-penalty mới không phạt token prompt (semantics đúng hơn v2-exempt).
4. `g2_imitation` giữ phần REINFORCED gold; **rebuild checkpoint tương lai** bằng train_v3 sẽ
   thiếu QFocus/rule edges v2 — các checkpoint hiện tại KHÔNG ảnh hưởng (đã build).
5. `data/` gold anaphor/composition jsonl GIỮ (data không hại; eval không đọc nữa).

## 6. Việc tiếp (kế thừa TK_018 H2-H5)

- H2: đo A/B `ar_tail_as_agenda` + note R4b đầy đủ vào TK_017 §15c.
- H3: index cho 15k `pattern_templates` (đang quét tuyến tính).
- H4: quyết default checkpoint (`network_v4_full` — guard trên đã có) + cập nhật README/v4.md.
- H5: bind_sense_resonance batch; "Mèo là gì" ranking; D2 head-POS.

## 7. Lần 2 — Hợp nhất A3 multi-only (2026-07-09)

Chỉ đạo tác giả: chỉ giữ **A3 token-by-token + multi**; bỏ QA/crisp/essay/pool-OFF và eval đa-mode.

### 7.1 Đã xóa

| Nhóm | File / symbol |
| :--- | :--- |
| Mode sinh | `generate_hybrid`, `generate_composed`, `generate_essay` |
| Config | `ar_paragraph_pool` (pool đoạn **luôn bật** trong discourse-route) |
| Scripts | `scripts/eval_v4.py`, `scripts/verify.py`, `scripts/eval_paragraph_ab.py`, `scripts/essay_ab_t3.py` |
| Tests | `tests/test_eval_v4.py`, `tests/test_hf_corpus.py`; essay/episode/SFT/tk010; ablation legacy |

### 7.2 Đường sống duy nhất

- `hca/__init__.py` → `generate`, `HCAConfig`
- `generate(prompt, multi=True)` → `generate_autoregressive` (agenda + paragraph pool + VarLM)
- Viz: `completion_graph` luôn multi; API `mode=autoregressive` only
- Eval discourse: `scripts/eval_discourse_prompts.py`

### 7.3 Checkpoint & đo

| | |
| :--- | :--- |
| Default | `network_v10_histwiki.pkl` / `network_v10_histwiki_paragraph.pkl` |
| VarLM | `network_v10_histwiki_paragraph.varlm.pkl` (cluster-aware) |
| Guard | QA/Para/Syn **86/88/67** giữ trên v10 paragraph |
| Tests | **131 pass** (`out/pytest_collect.log`) |

Chi tiết: README §0.4.1, `TK_035` §7i–7j, `docs/NN-AIVIE v4.md` nhật ký 2026-07-09.
