# QA Context Grounding — Spec

> **Trạng thái:** Một phần đã implement (QFocus, fingerprint, query readout); **Answer closure** chưa — xem `docs/ANSWER_CLOSURE_SPEC.md`.

## Nguyên tắc

**Không hardcode từ vựng / loại câu hỏi trong runtime.** Mọi disambiguation học trên đồ thị tại teach/G2; runtime chỉ đọc cạnh `rule_colocation` + `REINFORCED`.

Giữ lại ở runtime: **cấu trúc frame** (`Hỏi:` / `Đáp:`), **dấu câu**, **JSON brackets** — không phải ngữ nghĩa tiếng Việt.

---

## Kiến trúc

1. **QA frame** — `Hỏi:` / `Đáp:` (zone isolation, train mask)
2. **QASession** — salient Q, post A, **fingerprint** từ chữ ký câu hỏi
3. **Matcher** — đọc đồ thị; suppress Var:A hub khi không có cạnh ngữ cảnh
4. **Query-conditioned readout** — `query_conditioned_read()` boost path từ Q tokens → A (HCA)
5. **Answer closure** *(planned)* — `assess_answer_closure()` — xem spec riêng

---

## QFocus & Q-fingerprint

### Định nghĩa

**QFocus** là **node cấu trúc** trên đồ thị, **không** phải hardcode ngữ nghĩa:

```
Chữ ký (signature) = token nội dung Q nối bằng '_'
Node id            = QFocus:{signature}

Chó màu gì ?     →  QFocus:Chó_màu_gì
Con mèo là gì ?  →  QFocus:Con_mèo_là_gì
Mẹ làm gì ?      →  QFocus:Mẹ_làm_gì
```

- Token nội dung = bỏ `Hỏi:`, `?`, dấu câu thừa (logic trong `QASession`)
- **Teach** (`compile._add_qa_bindings`): imprint `QFocus:{sig} → full A span` + cạnh Q token → fingerprint
- **Runtime** (`resolve_qa_fingerprint`): khớp chữ ký chính xác, hoặc QFocus có colocation Q→fp mạnh nhất trên đồ thị
- **Rule id** (`resolve_qa_rule_id`): đọc từ đồ thị — **không** hardcode `vi_QA_hoi_dap` lúc generate

### QFocus có phải “một nơ-ron cho một câu hỏi”?

| Phạm vi | Phù hợp? | Ghi chú |
|:---|:---:|:---|
| QA ngắn lớp 3–7 (~5–15 token Q) | ✓ | Mỗi cặp Hỏi–Đáp trong lesson ≈ một fingerprint |
| Paraphrase / Q lạ (OOV) | ✗ | Không có node teach → fingerprint yếu |
| Câu essay / multi-part dài | ✗ | Không scale hash full-text; cần **agenda segment** + topic hub |

**Kết luận trao đổi:** QFocus **đúng cho POC QA ngắn**; essay cần kiến trúc mở rộng (Mục “Câu hỏi dài”).

---

## Module

| File | Vai trò |
|:---|:---|
| `hca/qa_context.py` | Parse session, graph readout, `qa_answer_complete` *(sẽ thay closure)* |
| `hca/generate.py` | `query_conditioned_read`, stop on closure |
| `hca/matcher.py` | Zone guard, `apply_qa_context_scores`, frame EOS |
| `teaching/compile.py` | `_add_qa_bindings` — fingerprint + entity bind |
| `training/g2_imitation.py` | Củng cố gold QA path |
| `viz/trace.py` | **Cần sync** với generate (closure + query readout) |
| `eval/qa_focus.py` | Eval đọc `data/grade_XX/qa/val.jsonl`, không `must_contain` runtime |

---

## Train lại?

| Thay đổi | Teach | G1/G2 |
|:---|:---:|:---:|
| Chỉ zone / anchor runtime | — | — |
| QA bindings / fingerprint | **Có** | **G2** |
| Thêm lesson examples | Có | Tuỳ chọn |
| Answer closure (chỉ runtime) | — | — |
| Segment / Var:P / must_cover | **Có** | **G2** |

---

## Smoke prompts

```
Hỏi: Chó màu gì ? Đáp:       → Chó màu …
Hỏi: Mẹ làm gì ? Đáp:        → Mẹ nấu cơm
Hỏi: Em học gì ? Đáp:        → Em học …
Hỏi: Con mèo là gì ? Đáp:    → Con mèo là thú cưng.
```

Graph UI trace fields: `fingerprint`, `fingerprint_label`, `salient_q`, `post_tokens`, *(planned)* `closure_status`.

---

## Case study 1: bleed `Chó → chạy/ăn/sủa`

**Nguyên nhân cũ:** colocation `(Chó, a_token)` gộp từ mọi example QA.

**Fix:** fingerprint `QFocus:Chó_màu_gì` chỉ bind answer path của example màu — runtime đọc cạnh fingerprint, không if/else `màu`/`làm`/verb list.

---

## Case study 2: `Con mèo là gì` — CLI đúng, UI bleed

| | Output |
|---|--------|
| CLI `generate()` | `Con mèo là thú cưng.` |
| Graph UI trace | `Con mèo là thú cưng. ngủ. màu trắng.` |

**Nguyên nhân:**

1. **Trace** không gọi `qa_answer_complete` / closure → chạy tới `max_len`
2. Sau `.`, **fingerprint gate yếu** → colocation `(Con mèo, ·)` kéo token từ QA *làm gì* / *màu gì*
3. Token `ngủ`, `màu trắng` là **mảnh vị ngữ** (thiếu chủ ngữ) — graph có kiến thức nhưng không ghép câu

**Hướng xử lý:** `docs/ANSWER_CLOSURE_SPEC.md` (SATISFIED + sync trace + compound teach).

---

## Case study 3: cụt sớm `Em học gì`

**Hiện tượng:** `Em học` (thiếu object).

**Nguyên nhân:** `qa_answer_complete` trả `True` khi `len(post)==1` và không thấy fp continuation — quá sớm.

**Fix:** closure chỉ SATISFIED khi đủ agenda + coverage + closure token cuối.

---

## Câu hỏi dài (essay) — không dùng QFocus full-text

Ví dụ (grade 7):

```
Hãy mô tả về loài hổ, cách chúng kiếm ăn. Đồng thời mô tả khí hậu …
Liệt kê loài hổ ở Việt Nam. Loài nào trong sách đỏ?
```

**Thiết kế đề xuất:**

- **Agenda:** `Lbl:ans_seg_1` … `Lbl:ans_seg_n` trong gold A
- **Coverage:** salient Q tokens (`hổ`, `Việt Nam`, `sách đỏ`) bắt buộc xuất hiện (teach `must_cover`)
- **Hub:** `QFocus:essay_tiger` hoặc topic node, không hash 200 token Q
- **Distill:** LLM mã nguồn mở → jsonl → G2 (`docs/study/01_OPEN_LLM_TO_HEBBIAN_TRANSFER.md`)

---

## Answer closure (tóm tắt)

Chi tiết đầy đủ: **`docs/ANSWER_CLOSURE_SPEC.md`**.

**SATISFIED** khi **đồng thời**:

1. Agenda segment rỗng (không còn Var/slot bắt buộc)
2. Không còn fingerprint-gated continuation
3. 100% salient Q coverage (khi có teach `must_cover`)
4. Token cuối là `.`/`!`/`?` trên **segment cuối**

**`.` không đồng nghĩa SATISFIED** — có thể là `SECTION_PAUSE` giữa các vế.

---

## Liên kết

- `docs/ANSWER_CLOSURE_SPEC.md` — đặc tả closure, roadmap implement
- `README.md` Mục 0.2 — bài toán & kế hoạch mai
- `docs/study/01_OPEN_LLM_TO_HEBBIAN_TRANSFER.md` — distill essay gold
