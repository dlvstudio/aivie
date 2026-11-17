# Answer Closure — Đặc tả thoả mãn câu trả lời

> **Trạng thái:** Spec (chưa implement) — bắt đầu triển khai 2026-06-29  
> **Liên quan:** `docs/QA_CONTEXT_SPEC.md`, README Mục 0.2, `src/nn_aivie/hca/qa_context.py`

---

## 1. Bài toán (problem statement)

### 1.1 Bối cảnh

Mạng **nn-aivie** sinh câu trả lời QA autoregressive trong vùng `Hỏi: … Đáp: …`. POC hiện tại:

- **Đúng** trên một số prompt ngắn đã dạy (vd. `Con mèo là thú cưng.`)
- **Sai** theo hai chiều:
  - **Cụt sớm:** `Em học gì` → `Em học` (dừng trước khi đủ câu)
  - **Thừa / mảnh vị ngữ:** `… thú cưng. ngủ. màu trắng.` (bleed từ QA khác cùng anchor, thiếu chủ ngữ)

### 1.2 Yêu cầu (requirements)

| ID | Yêu cầu |
|:---|:---|
| **R1** | Mỗi bước token, runtime **đánh giá** câu trả lời đã thoả mãn chưa — **không** hardcode từ/loại câu hỏi |
| **R2** | **INCOMPLETE** ⇒ **tiếp tục sinh**; **suppress `<EOS>`** |
| **R3** | **SATISFIED** ⇒ mới **boost `<EOS>`** và dừng generate |
| **R4** | Dấu `.` `,` xuống dòng / marker section là **token discourse** học trên đồ thị — **`.` ≠ SATISFIED** tuyệt đối |
| **R5** | Câu hỏi **dài multi-part** (essay) dùng **cùng cơ chế** agenda/segment, không hash nguyên văn Q vào một QFocus khổng lồ |
| **R6** | Văn phong **compound + đại từ** (`Con mèo màu trắng, nó thích ngủ, …`) qua teach `Var:P`, không map cứng `mèo→nó` |
| **R7** | Graph UI **trace** và CLI **`generate()`** dùng **cùng** pipeline closure |

### 1.3 Non-goals (POC phase này)

- Classifier LLM “user hài lòng chưa”
- Parser ngữ nghĩa tiếng Việt runtime (WH-word list, NER cứng)
- Import weight Transformer

---

## 2. Tóm tắt trao đổi thiết kế (design log)

### 2.1 QFocus — có phải một nơ-ron / câu hỏi?

- **`QFocus:{signature}`** = node **cấu trúc**; `signature` = token nội dung Q nối `_` (vd. `QFocus:Con_mèo_là_gì`).
- **Teach/G2** imprint cạnh fp → full answer path.
- **Giới hạn:** mỗi **văn bản Q khác nhau** ≈ một fingerprint; câu Q **dài / paraphrase** → node sparse, không generalize như LLM.
- **Hướng mở rộng:** fingerprint **compositional** (topic hub) + **agenda segment**, không hash cả đoạn essay.

### 2.2 Case `Con mèo là gì` — bleed vs CLI

| | CLI `generate()` | Graph UI trace |
|---|------------------|----------------|
| Output | `Con mèo là thú cưng.` ✓ | `… thú cưng. ngủ. màu trắng.` ✗ |
| Nguyên nhân | `qa_answer_complete` dừng sau `cưng.` | trace **không** gọi closure; chạy đủ `max_len` |
| Bleed token | — | `ngủ` (QA *làm gì*), `màu trắng` (QA *màu gì*) — cùng anchor `Con mèo` |

**Insight:** Kiến thức multi-aspect **có trên đồ thị** nhưng sinh thành **mảnh vị ngữ** vì thiếu chủ ngữ / section / closure.

### 2.3 Văn phong: lặp S vs đại từ

Mục tiêu văn phong (grade 4+):

```
Con mèo màu trắng, nó thích ngủ, là thú cưng.
```

Cần **`Var:P`** (pronoun/ref) + teach S↔P + **SECTION_PAUSE** sau `,` (anaphor step), không phạt lặp anchor sau dấu phẩy.

### 2.4 Câu hỏi essay (ví dụ hổ)

```
Hãy mô tả về loài hổ, cách chúng kiếm ăn. Đồng thời mô tả khí hậu …
Liệt kê loài hổ ở Việt Nam. Loài nào trong sách đỏ?
```

**Không** fit POC QFocus full-text. Cần:

- Agenda **n section** (`Lbl:ans_seg_1` … `Lbl:ans_seg_n`)
- **Coverage** salient Q (`hổ`, `Việt Nam`, `sách đỏ`, …)
- `$K_{\max}` / hub / retroactive cho prefix Q dài

---

## 3. Trạng thái closure (AnswerClosure)

```python
# Pseudocode — module đề xuất: hca/answer_closure.py hoặc mở rộng qa_context.py

class ClosureStatus(Enum):
    INCOMPLETE = "incomplete"       # còn slot/seg/coverage → sinh tiếp
    SECTION_PAUSE = "section_pause" # vế/section xong, còn section khác
    SATISFIED = "satisfied"         # đủ điều kiện → cho EOS
```

### 3.1 Công thức SATISFIED (AND — tất cả)

```
SATISFIED ⇔
    open_segments(session) == ∅
 ∧ ¬has_fp_continuation(last, session, qa_rule)
 ∧ coverage_required(salient_q, post, adjacency) == 1.0
 ∧ is_final_closure(last, modality)   # . ! ? ở segment CUỐI
```

### 3.2 Chi tiết từng điều kiện

#### (1) Agenda rỗng

- **Segment** = `Lbl:ans_seg_k` trong vùng `Đáp:` (dạy qua lesson).
- Segment **mở** nếu còn **Var bắt buộc** eligible (`matcher._eligible_var_ids`).
- **Câu ngắn lớp 3:** mặc định **1 segment** (implicit từ rule QA + Var A) — tương thích ngược.

#### (2) Không continuation fingerprint-gated

- Dùng `_chain_has_continuation(chain_id, post_set, qa_rule, fp=session.fingerprint_id)`.
- Ngưỡng như hiện tại (~0.12); có thể tune qua config.

#### (3) Coverage salient Q

- `salient_q_ids` từ `QASession` (token nội dung Q, bỏ dấu câu).
- Teach thêm cạnh **`must_cover(q_id)`** cho essay: colocation `(qa_rule, q, ·)` bắt buộc xuất hiện trong A.
- **POC phase 1:** nếu chưa có cạnh `must_cover`, coi coverage = 1.0 (chỉ dùng agenda + continuation).

#### (4) Final closure token

- Token cuối ∈ `{`.`, `!`, `?`}` hoặc JSON close (modality).
- **SECTION_PAUSE:** có closure **nhưng** `open_segments` vẫn còn phần tử **khác** segment vừa đóng.

### 3.3 Bảng quyết định

| Trạng thái | Agenda | fp cont. | Coverage | Token cuối | EOS |
|:---|:---:|:---:|:---:|:---|:---:|
| INCOMPLETE | còn mở | any | thiếu | any | **suppress** |
| SECTION_PAUSE | còn seg khác | false trong seg | seg ok | `.` / `\n` / `[k]` | **suppress** |
| SATISFIED | ∅ | false | 100% | `.`/`!`/`?` seg cuối | **boost** |

### 3.4 Điều kiện **KHÔNG** dùng

- `len(post) >= N`
- `last == "."` ⇒ dừng (bug hiện tại trong `qa_answer_complete`)
- `must_contain("từ")` eval-style trong runtime
- `is_frame_complete` một mình (chỉ grammar frame S–V, không QA agenda)

---

## 4. Vòng generate (target architecture)

```
for step in max_len:
    scores = dual_context_read + query_conditioned_read + matcher
    closure = assess_answer_closure(session, adjacency, vocab)

    if closure.status != SATISFIED:
        scores[EOS] *= eos_suppress   # vd. 0.01
        scores = apply_incomplete_boost(scores, closure)  # slot/seg thiếu

    if closure.status == SECTION_PAUSE:
        boost punct: , \n [k+1]; boost Var:P / anchor

    tok = argmax(scores)
    append(tok)

    if tok == EOS and closure.status == SATISFIED:
        break
    # KHÔNG break chỉ vì tok == '.'
```

**Thay thế:** `qa_answer_complete()` → wrapper gọi `assess_answer_closure().status == SATISFIED`.

---

## 5. Dấu câu & modality

| Token | Vai trò | Học từ |
|:---|:---|:---|
| `,` | Nối vế trong segment / trước `Var:P` | colocation teach |
| `.` `!` `?` | Đóng segment hoặc toàn bài | colocation + closure rules |
| `\n`, `[1]` `[2]` … | Ranh giới section (essay) | lesson essay template |
| `<EOS>` | Chỉ khi SATISFIED | REINFORCED sau gold EOS trong G2 |

**Modality** (`AnswerModality` hiện có): mở rộng thêm **`COMPOUND`** khi gold có `,` trong A và teach COMPOUND path.

---

## 6. Curriculum & teach (không runtime hardcode)

### 6.1 Câu ngắn (grade 3) — giữ tương thích

- Một segment, fingerprint như hiện tại.
- Closure = đủ chain + `.` + không fp continuation.

### 6.2 Compound + anaphor (grade 4+)

Thêm vào `lessons/vi_QA_hoi_dap.lesson.yaml` (hoặc `vi_QA_compound.lesson.yaml`):

```yaml
- "Hỏi: Con mèo là gì ? Đáp: Con mèo màu trắng, nó thích ngủ, là thú cưng."
- "Hỏi: Bố làm gì ? Đáp: Bố nấu cơm, ông ấy nấu ngon."
```

Teach compile:

- `Var:P` align với cùng entity `Var:S`
- `Lbl:ans_seg_1` nếu multi-section; COMPOUND = một seg nhiều vế

### 6.3 Essay (grade 6–7) — phase 2

```yaml
# Ví dụ gold (distill hoặc soạn tay)
Hỏi: [câu dài về hổ …]
Đáp:
[1] Loài hổ là … .
[2] Chúng kiếm ăn … .
[3] Khí hậu ưa thích … .
[4] Ở Việt Nam có … .
[5] Sách đỏ ghi … .
```

Imprint: `QFocus:essay_species_VN` hoặc hub `hổ` + segment labels.

---

## 7. G4 Verify (optional, sau SATISFIED cơ bản)

- Structural node `[Verify]` (roadmap README / DeepSeek R1 analogy).
- Sau khi graph nghi SATISFIED: đọc lại coverage + agenda.
- Thiếu → chuyển INCOMPLETE, boost mở segment thiếu (self-repair).

---

## 8. Module & file mapping (implementation)

| Thành phần | File | Việc |
|:---|:---|:---|
| `AnswerClosure`, `assess_answer_closure()` | `hca/qa_context.py` hoặc `hca/answer_closure.py` | **Mới** |
| Thay `qa_answer_complete` | `hca/qa_context.py`, `hca/generate.py` | Refactor |
| EOS suppress/boost | `hca/matcher.py` `apply_frame_closure` | Mở rộng |
| Sync trace | `viz/trace.py` | +query_readout + closure + early stop |
| Teach segment / must_cover | `teaching/compile.py` | Phase 2 |
| `Var:P` slot | `teaching/ast.py`, lesson yaml | Phase 2 |
| Tests | `tests/test_answer_closure.py` | **Mới** |
| Graph UI fields | trace dict: `closure_status`, `coverage`, `open_segments` | Phase 1 |

---

## 9. Kế hoạch triển khai (mai — 2026-06-29)

### Phase 1 — Closure cơ bản (1–2 ngày)

- [ ] **P1.1** `assess_answer_closure()` + `ClosureStatus`
- [ ] **P1.2** Bỏ rule “`.` ⇒ return True” tuyệt đối; tách SECTION_PAUSE vs SATISFIED
- [ ] **P1.3** `generate()`: break chỉ khi SATISFIED; suppress EOS khi INCOMPLETE
- [ ] **P1.4** Sync `viz/trace.py` với `generate()` (+ `query_conditioned_read`)
- [ ] **P1.5** Tests: `Con mèo là gì` dừng đúng; không bleed trên trace; `Em học gì` không cụt nếu chain đủ
- [ ] **P1.6** Inference Lab hiển thị `closure_status`, `coverage`

### Phase 2 — Compound & coverage (3–5 ngày)

- [ ] **P2.1** Lesson compound + `Var:P` + teach compile
- [ ] **P2.2** `must_cover` edges + coverage trong closure
- [ ] **P2.3** Anaphor step sau `,` (boost P, không subject_penalty)
- [ ] **P2.4** Train grade 3 + eval

### Phase 3 — Essay agenda (1–2 tuần)

- [ ] **P3.1** `Lbl:ans_seg_k` trong lesson AST
- [ ] **P3.2** 1–2 essay gold grade 7 (hổ / sông Mekong)
- [ ] **P3.3** Distill pipeline (`docs/study/01_OPEN_LLM_TO_HEBBIAN_TRANSFER.md`)
- [ ] **P3.4** G4 `[Verify]` smoke

---

## 10. Test cases (acceptance)

| Prompt | Kỳ vọng sau Phase 1 |
|:---|:---|
| `Hỏi: Con mèo là gì ? Đáp:` | `Con mèo là thú cưng.` — trace = CLI, không `ngủ.` |
| `Hỏi: Em học gì ? Đáp:` | Câu đủ (vd. `Em học toán.`) nếu chain teach đủ |
| `Hỏi: Chó màu gì ? Đáp:` | `Chó màu …` không bleed `chạy/ăn/sủa` |

| Prompt | Kỳ vọng sau Phase 2 |
|:---|:---|
| Compound lesson | `…, nó …, …` với đại từ |

| Prompt | Kỳ vọng sau Phase 3 |
|:---|:---|
| Essay hổ (5 section gold) | 5 vế + EOS; coverage salient |

---

## 11. Tham chiếu code hiện tại (baseline)

```365:410:src/nn_aivie/hca/qa_context.py
def qa_answer_complete(...):
    ...
    if last in _PUNCT or last.endswith((".", "!", "?")):
        return True   # ← cần thay bằng assess_answer_closure
    ...
    if len(post_ids) == 1:
        return True   # ← gây cụt sớm
```

---

## 12. Liên kết tài liệu

- `docs/QA_CONTEXT_SPEC.md` — QFocus, frame, bleed case study
- `README.md` Mục 0, 0.1, 0.2 — luật kiến trúc, điểm yếu, roadmap
- `docs/study/01_OPEN_LLM_TO_HEBBIAN_TRANSFER.md` — distill gold essay
- `README.md` B.9 — matcher slot close, directed colocation (phụ thuộc closure)
