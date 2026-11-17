# HCA Composition — Spec kiến trúc (sinh câu bằng kết hợp, không hardcode)

> **Trạng thái:** Spec + implement dần (bắt đầu 2026-06-29)
> **Nguyên tắc:** Mọi cải tiến ở đây là **cơ chế đồ thị tổng quát** áp dụng cho **mọi node**, học từ training (teach/G1–G4). **Không** rule if/else từ vựng, **không** routing theo loại câu hỏi (Luật README Mục 0).

---

## 1. Vấn đề (đặt lại ở mức kiến trúc)

LLM không "phát minh từ hư vô" — nó **kết hợp / nội suy / khái quát hóa / ngoại suy** từ thống kê đã học. nn-aivie cần **cùng nguyên lý**, nhưng cơ chế là **cạnh đồ thị + chồng chập kích hoạt**, không phải softmax tham số.

Hiện tại mạng **chỉ nhớ lại một path** rồi argmax — không ghép path. Ba khiếm khuyết kiến trúc:

### K1. Biểu diễn câu hỏi *localist*

`QFocus:Con_mèo_là_gì` = **một node cho cả câu**. Hai câu khác chữ → hai node rời, không chia sẻ. **Về bản chất không khái quát hóa được** (giống mỗi câu là một token, không có embedding).

### K2. Nhánh distributed bị tắt

`query_conditioned_read(..., query_ids=())` — readout chỉ dùng fingerprint localist, **bỏ** đường khái quát hóa qua token nội dung. Trong khi G2 **đã train** cạnh `Q_token → A_token` (REINFORCED, context bind) — nhưng inference không đọc.

### K3. Readout 1-hop + winner-take-all

`dual_context_read` và `apply_answer_chain_scores` đi **một chain từ token cuối**; activation không lan nhiều bước; argmax + boost gộp về **một path thắng**. Không có chồng chập để sinh tổ hợp mới.

---

## 2. Bốn cơ chế (tổng quát, học từ training)

| Năng lực (idea) | Khiếm khuyết | Cơ chế kiến trúc | Phạm vi |
|:---|:---|:---|:---|
| **Khái quát hóa** | K1, K2 | **Distributed query**: câu hỏi = chồng chập kích hoạt token nội dung; fingerprint hạ xuống *một nguồn* | mọi token |
| **Kết hợp / nội suy** | K3 | **Spreading activation đa hop** + readout chồng chập chuẩn hóa (không argmax thô) | mọi node |
| **Củng cố (Hebb thật)** | EMERGENT chưa dùng | **Consolidation**: hai ngữ cảnh cùng kích hoạt một đích lặp lại → tự sinh cạnh `EMERGENT` | mọi node |
| **Ngoại suy có kiểm soát** | không chặn | **G4 Verify**: token không có đường dẫn học được → ức chế (không bịa fact ngoài đồ thị) | mọi node |

> Composition là **hành vi mặc định của readout**, nổi lên từ cấu trúc + chồng chập — **không** "nếu compound thì chọn rule compound".

---

## 3. Cơ chế #1 — Distributed query representation

### 3.1 Ý tưởng

Câu hỏi không phải một node hash, mà là **tập kích hoạt** của các token nội dung salient. Readout có điều kiện = **chồng chập** đóng góp từ:

```
score[v] = Σ_q∈salient_Q  readout(q → v) · w_q          (distributed — khái quát hóa)
         + readout(fingerprint → v) · w_fp               (localist — gợi nhớ chính xác, hạ trọng số)
         + readout(anchor → v) · w_anchor
```

- **Paraphrase** (`Mèo là gì`, `Con mèo là con gì`) chia sẻ phần lớn token nội dung → kích hoạt cùng vùng đáp án qua cạnh `q→a` đã train. Generalization **nổi lên**, không cần node mới.
- **Fingerprint** vẫn còn nhưng chỉ là *một nguồn* — câu đã dạy nguyên văn vẫn mạnh, câu lạ vẫn có tín hiệu.

### 3.2 Hạ tầng đã có sẵn (chỉ chưa đọc)

- `query_conditioned_read` đã nhận `query_ids` (chồng chập nhiều nguồn) — hiện truyền rỗng.
- G2 (`g2_imitation`) **đã** imprint `prefix_token → gold_answer` REINFORCED (gồm token Q) → đúng substrate cho distributed query.

### 3.3 Thay đổi

- `generate.py`: `query_ids = session.salient_q_ids` (thay `()`).
- `viz/trace.py`: đồng bộ y hệt.
- `query_conditioned_read`: giữ chữ ký; có thể thêm chuẩn hóa theo `|query_ids|` để câu dài không lấn át.
- **Không** thêm rule nào.

### 3.4 Đo lường (bắt buộc — tránh ảo tưởng)

| Phép đo | Mục tiêu |
|:---|:---|
| `Con mèo là gì` (đã dạy) | Không thoái lui so với trước |
| **Paraphrase chưa dạy** (`Mèo là gì`, `Con mèo là con gì`) | Sinh đáp án hợp lý → bằng chứng khái quát hóa |
| QA val accuracy | Không giảm |
| next-token acc | Theo dõi xu hướng |

---

## 4. Cơ chế #2 — Spreading activation đa hop

Cho activation lan 2–3 hop có decay trong `forward.py`:

```
a^(0) = onehot(nguồn)
a^(t+1)[v] = Σ_u a^(t)[u] · readout(u→v) · γ          (γ < 1 decay)
score = Σ_t β_t · a^(t)
```

- Nội suy nổi lên ở **hub chung** (vd. `Con mèo` nối path *là gì* và *màu gì*).
- Tổng quát: không phụ thuộc QA; là phiên bản Hebbian của "stack nhiều lớp attention".
- Rủi ro: nhiễu lan rộng → cần `γ`, ngưỡng cắt, chuẩn hóa. Đo kỹ.

**Đã triển khai** (`forward.spreading_activation_read`, wired ở `generate.py` + `trace.py`):

- Seeds = salient Q tokens + fingerprint (w=1.5) + anchor cuối; lan `spread_hops=2` hop, decay `spread_gamma=0.45`.
- Chuẩn hóa theo `row_max` mỗi hop để chặn hub bùng nổ; cắt frontier (`spread_max_frontier`) + ngưỡng (`spread_threshold`).
- **Gate cuối theo `row_readout(w0)`**: chỉ cộng cho token là hậu duệ hợp lệ của w0 → giữ tính hợp lý token kế tiếp, lan đa hop chỉ dùng để *định hướng* trong tập ứng viên đó.
- Đo (grade 3): các ca đã dạy không hồi quy; paraphrase `Con mèo là con gì` tổng quát đúng (`thú cưng.`). Bleed `Con chó là gì → màu trắng` tồn tại **cả khi tắt spread** ⇒ là **thiếu gold** (chỉ dạy chó→màu), không phải lỗi cơ chế. 166 test pass.

---

## 5. Cơ chế #3 — Emergent consolidation (Hebb thật)

`"Neurons that fire together, wire together"` ở mức **tạo cạnh mới**:

- Trong/ sau train: nếu hai ngữ cảnh khác nhau **cùng kích hoạt mạnh** một đích lặp lại → thêm cạnh `EdgeKind.EMERGENT` giữa chúng.
- Hệ quả: mạng **tự khái quát** từ các mẫu rời rạc thành cấu trúc chia sẻ — không cần teacher liệt kê mọi tổ hợp.
- Là **cơ chế training**, tổng quát mọi node; có decay/prune để tránh phình.

**Đã triển khai** (`training/consolidate.py`, gọi sau G2 trong `run_train_eval.py`):

- Phát hiện node **đồng vai**: hai nguồn chia sẻ tập đích REINFORCED. Tương đồng = **Jaccard có trọng số IDF** theo in-degree đích → chia sẻ đích hiếm là bằng chứng mạnh, chia sẻ hub thì yếu.
- **Kế thừa**: mỗi node nhận các đích mạnh *không phải hub* của anh em qua cạnh `EMERGENT` (cap số cạnh/cặp, bỏ node cấu trúc/đặc biệt/dấu câu).
- `EMERGENT` lần đầu **vào readout** ở scale nhỏ `READOUT_EM_SCALE=0.12` (trước đó là cạnh chết) → chỉ *nudge*, không lấn đường gold.
- Đo (grade 3): ~46 cặp đồng vai, ~220 cạnh emergent hợp lý (`Lan↔Em`, `Bố→học`, `không→là`). **Không hồi quy** QA đã dạy, free-gen, 169 test pass. Trên corpus nhỏ hiện tại các cạnh REINFORCED đã dạy vẫn áp đảo nên đổi sinh nhìn thấy còn nhỏ — giá trị tăng theo độ lớn corpus/cấu trúc anh em. (Bleed `rất→Em` là **co_activation có sẵn**, không phải emergent — đã xác minh.)

---

## 6. Cơ chế #4 — G4 Verify (ngoại suy có kiểm soát)

- Node cấu trúc `[Verify]` (roadmap, DeepSeek-R1 analogy).
- Trước khi chốt token / EOS: đọc lại đồ thị, token **không** có đường dẫn học được (cạnh dưới ngưỡng) → ức chế.
- Đảm bảo "ngoại suy = lắp fact đã học theo frame đã dạy", **không** bịa fact ngoài corpus.

**Đã triển khai** (`hca/answer_verify.py`, gọi sau closure trong `generate.py` + `trace.py`, scope: phiên QA):

- Anchor ngữ cảnh = {w0} ∪ salient Q ∪ fingerprint ∪ vài token đáp gần nhất.
- `grounding(v)` = đường dẫn học được mạnh nhất (REINFORCED + EMERGENT + CO_ACTIVATION + rule-collocation) từ một anchor bất kỳ tới v. `grounding(v) < verify_min` → `scores[v] = -inf` ⇒ thà dừng còn hơn bịa.
- Đo (grade 3): verify ON/OFF cho kết quả **giống hệt** — vì corpus nhỏ, mọi ứng viên thắng đều đã có grounding (vd `chó→màu`). G4 hiện là **lưới an toàn latent** chống *hallucination* token vô căn cứ (unit test xác nhận chặn token không đường dẫn, giữ token có đường dẫn). 172 test pass.
- **Giới hạn quan trọng**: các bleed còn lại (`Con chó là gì → màu trắng`, `Con mèo là con gì → thú chó`) **không phải hallucination** — token đều *có* grounding học được, chỉ *sai liên quan truy vấn*. Đây là vấn đề **độ phủ gold + ràng buộc truy vấn**, ngoài phạm vi G4 grounding-only.

---

## 7. Thứ tự triển khai

| Bước | Cơ chế | Rủi ro | Trạng thái |
|:---|:---|:---|:---|
| **#1** | Distributed query | Thấp | ✅ xong |
| **#2** | Spreading activation | Trung bình (nhiễu) | ✅ xong (gate theo w0) |
| **#3** | Emergent consolidation | Trung bình (phình graph) | ✅ xong (IDF + chặn hub) |
| **#4** | G4 Verify | Cao (cần CoT-graph) | ✅ xong (grounding guard; latent ở grade-3) |

Mỗi bước: **implement → train → đo (bảng §3.4) → giữ nếu cải thiện thật**. Không heuristic che kết quả.

---

## 7b. Nâng cấp toàn bộ — chống thiên lệch tần suất (2026-06-29)

Đo sau #1–#4 lộ ra nút thắt thật: **token/chủ thể tần suất cao ("Con", "mèo", "trắng") lấn át truy vấn**. Bốn sửa đổi nguyên lý (thống kê đồ thị, không luật ngôn ngữ):

1. **Augment gold paraphrase nhất quán chủ thể** (`scripts/augment_qa_paraphrase.py`): mỗi fact hỏi nhiều cách (`là gì`/`là con gì`/`có màu gì`…) cùng map một đáp; khử xung đột một-câu-hỏi-nhiều-đáp.
2. **IDF query + chuẩn hoá row-max** (`query_conditioned_read`): token truy vấn generic ("Con") đóng góp ít, chuẩn hoá để edge tần suất cao không áp đảo.
3. **Copy/pointer** (`copy_query_read`): boost token nội dung của truy vấn chưa xuất hiện trong đáp (mẫu QA phổ quát: đáp tham chiếu thực thể câu hỏi).
4. **Fingerprint khớp theo IDF-overlap** (`resolve_qa_fingerprint`) — *sửa bug gốc rễ*: trước đây fuzzy chọn QFocus theo khối lượng collocation → mọi "màu gì" rơi về fingerprint tần suất cao nhất (mèo). Nay khớp theo độ phủ token phân biệt (IDF trên corpus QFocus) → câu hỏi "chó" khớp đúng `QFocus:Con_chó_màu_gì`.
5. **Gate post_chain theo nhất quán fingerprint** (`q_context_weight`): chuỗi bigram tần suất của đáp ("Con→mèo") bị damp ×0.12 nếu fingerprint truy vấn không hỗ trợ token đó.

**Kết quả (grade 3, tập lõi 13 ca): 11/13** chứa đáp đúng. Thắng lợi chính:
- `Con chó là gì` → `Con chó là thú cưng` ✅ (trước: `màu trắng`, bleed mèo).
- `Con chó màu gì` → `Con chó màu nâu` ✅ (trước: `trắng`, sai màu).
- `Con chó là con gì` → chủ thể `chó` đúng (trước: `mèo`).
- Toàn bộ ca mèo + QA phi-động-vật giữ đúng. **172 test pass**.

**Giới hạn còn lại (trung thực)**: paraphrase có từ chức năng ("có", "con") + chủ thể train thưa (gà, chim) đôi khi **rớt chủ thể** (`Con chó làm gì → Con màu trắng`). Nguyên nhân: `matcher_boost` (magnitude ~100) áp đảo copy/q/spread (~đơn vị) → điều khiển nằm hoàn toàn trong matcher; và `answer_anchor_ids` lấy token đầu generic ("Con") làm chủ thể neo thay vì token phân biệt ("chó"). **Bước tiếp**: refactor anchor/salient theo IDF (đổi signature diện rộng, cần train lại) — để lần sau.

---

## 8. Liên kết

- `README.md` Mục **0** (luật), **0.2** (closure), **0.3** (composition)
- `docs/QA_CONTEXT_SPEC.md` — QFocus, fingerprint
- `docs/ANSWER_CLOSURE_SPEC.md` — closure, agenda
- `docs/study/01_OPEN_LLM_TO_HEBBIAN_TRANSFER.md` — distill teacher (sinh tổ hợp offline)
- `src/nn_aivie/hca/forward.py` — `dual_context_read`, `query_conditioned_read`
