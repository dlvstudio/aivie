# 02 — Neuro-Symbolic Hybrid Matrix (NN-AIVIE v3)

> Nguồn: `docs/NN-AIVIE v3.docx`. Tuân thủ **Luật 0** (README Mục 0): không hardcode từ ngữ/ý
> niệm vào runtime; embedding **dẫn xuất từ chính đồ thị**, không vay tri thức ngoài.

## 0. TINH THẦN CỐT LÕI — "1 ví dụ = 1 nơ-ron cha, rồi mới phân rã"

Đây là **trục chính** của v3 (docx §1), khác hẳn cách co-occurrence/clustering rời rạc:

> Khi đưa **một câu / một cụm / cả một đoạn văn** vào lúc train, v3 **coi nó như MỘT nơ-ron**
> (nơ-ron cha vĩ mô `n₁`). **Từ đó mới phân rã** xuống nơ-ron trung gian và nơ-ron token.

**Quá trình phân rã (top-down → hierarchical):**

1. **Tạo nơ-ron cha:** mỗi ví dụ (vd `"Bác sĩ chữa bệnh."`, hay cả đoạn mô tả con hổ) → một
   nơ-ron cha `P`. Ban đầu mọi token con nối **thẳng** vào `P`.
2. **Index + Power:** mỗi cạnh con→cha mang **index cứng** (thứ tự xuất hiện 1,2,3…) để giữ
   trật tự từ; mỗi nơ-ron giữ **power** = tần suất (đếm số lần lặp khi train).
3. **Sinh nơ-ron trung gian (ISG — TK_013, 2026-07-03):** bigram liền kề trong chuỗi ingest
   tích lũy tần suất; đủ `min_frequency` → `ClusterNeuron` (power = count) + **rewire** mọi cha
   chứa cặp; index trên **link** (`parent_idx`, `slot_index`), không trên neuron. Thay thế
   đường Re-Pair batch cho ingest mới (`core/seg_graph.py`). Re-Pair batch vẫn có qua
   `--legacy-repair`.

**Suy luận = lan xuôi + pattern completion (docx §1.2):**

- **Lan xuôi (bottom-up):** prompt (`"Hỏi: Ai chữa bệnh Đáp:"`) → token tầng dưới kích hoạt lan
  lên; **nơ-ron cha khớp nhiều thành phần nhất** tích nhiều năng lượng nhất → thắng.
- **Lan ngược (top-down / pattern completion):** từ cha thắng, duyệt con **theo index**; thành
  phần đã có trong prompt thì bỏ qua, **thành phần còn thiếu** (đoạn sau `Đáp:`) được **dội ngược
  năng lượng** xuống token để sinh ra → lấp đầy đáp `"Bác sĩ chữa bệnh."`.
- **Case "con kiến" (tri thức thiếu hụt):** chưa từng học dạng hỏi-đáp trực tiếp nhưng đã có dữ
  liệu mô tả → các nơ-ron đặc điểm vĩ mô vẫn kích hoạt mạnh, lan ngược chọn nhánh con **chưa
  kích hoạt** để sinh tiếp.

→ Đây là **máy pattern-completion từ nơ-ron cha**, KHÔNG phải bigram/co-occurrence rời rạc.

## 1. Bài học Phase 1–3 (đã đo) — vì sao bản đầu sai hướng

Bản đầu tôi xây **co-occurrence (PPMI) + clustering bottom-up (`neurogenesis.py` Bron-Kerbosch)
+ W_seq bigram** rồi tưởng "đã hiện thực v3". **Sai trục:** đó là gom cụm thống kê rời rạc, KHÔNG
phải "1 ví dụ = 1 nơ-ron cha → phân rã → pattern completion". Hệ quả đo được:

| Bộ | baseline cũ `generate()` | hybrid co-occurrence (sai trục) |
|:---|:---:|:---:|
| QA val (7) | 43% | **0%** |
| Paraphrase giữ kín (8) | 25% | **0%** |

Root-cause (đã trace): với `Ai chữa bệnh`, không kênh nào **dội ra** được thực thể `Bác sĩ` —
`base`=readout(`Đáp:`) toàn token tần suất (`bằng`,`gì`,`nào`), macro co-occurrence cũng nhiễu.
**Thiếu chính cơ chế nơ-ron-cha + pattern completion.** Fact CÓ trong data thật (`Bác sĩ`×5,
`chữa bệnh`×11) nhưng co-occurrence rời rạc quá yếu để truy hồi.

## 2. Ba cấu trúc (đặt đúng vào khung nơ-ron cha)

1. **A — phân cấp nơ-ron cha→trung gian→token (`[N×N]` thưa):** cạnh mang **index** (thứ tự con)
   + **power** (tần suất). **Đã code (TK_013):** `SegmentGraph` — link ngang per-parent, cụm
   online + rewire; `phrases[]` vẫn materialize từ cụm ISG để tương thích `rank_matches`.
2. **E — embedding mềm `[N×d]` (docx §2.2):** mỗi nơ-ron (kể cả cụm/cha) có vector **dẫn xuất từ
   đồ thị** (PPMI+SVD; cụm = mean-pool con). Dùng **khớp mờ cosine** để paraphrase/biến thể
   (`chữa`↔`khám`, `chó`↔`cún`) kích hoạt cùng nơ-ron cha. *(Đã code: `core/embedding.py`, 7
   test; cụm semantic nổi đúng — `chó`~`mèo` 0.82, `Hà`~`Nội` 0.99.)*
3. **W_seq — chuyển dịch vị trí `[N×N]` (docx §2.3):** thay **index cứng** bằng Markov mềm để
   văn phong tự nhiên, chống đảo ngữ què cụt. *(Đã code: `EdgeKind.SEQUENTIAL`, populate G1.)*

## 3. Vòng suy luận (docx §4) — bám pattern completion

```
1. Fuzzy a0   : cosine trên E → kích hoạt mềm nơ-ron khớp (token + cụm + cha)
2. Lan xuôi   : a = σ(Aᵀ·a) lặp → nơ-ron CHA tích năng lượng cao nhất = khung tri thức
3. Lan ngược  : a_macro = a_parent · A → dội xuống lấp CON còn thiếu (đáp sau "Đáp:") theo index
4. Micro      : a_micro = a_trace · W_seq → mượt văn phong (Markov ngắn hạn, decay γ)
5. Combine    : a_final = a_macro ⊙ a_micro → softmax(·/T) → multinomial
```

## 4. Việc cần làm (đặt lại theo đúng trục)

- **[CHÍNH] Nơ-ron cha + phân rã:** train ⇒ mỗi ví dụ → nơ-ron cha; **ISG (TK_013)** sinh cụm
  online từ bigram ngang + rewire; `rebuild_seg_graph.py` migrate checkpoint cũ. Re-Pair batch
  (`decompose_repair`) giữ qua `--legacy-repair`.
- **[CHÍNH] Pattern completion:** generate = lan xuôi chọn cha + lan ngược lấp con theo index,
  dùng frame `Đáp:` (được phép) làm mốc "con còn thiếu".
- **[HỖ TRỢ, đã có]** E khớp mờ (paraphrase) + W_seq (mượt) + temperature.
- **KHÔNG dạy ngữ pháp** (no NVL/Rule/Var) — ngữ pháp & cấu trúc đáp **tự nổi** từ thứ tự
  index trong nơ-ron cha + W_seq. Dữ liệu train = **thật, kiến thức thật lớp 3–7** (loại HF
  general/sentences nhiễu).

## 4b. KẾT QUẢ (2026-06-30) — nơ-ron cha ĐẠT, vượt baseline ở paraphrase

Đã hiện thực đúng trục: `core/parent_store.py` (1 ví dụ = 1 nơ-ron cha, con index + power) +
`generate_hebbian` viết lại theo **lan xuôi chọn cha (IDF) → lan ngược pattern completion**.
Train thuần `scripts/train_v3.py` (KHÔNG teaching) trên dữ liệu thật lớp 3–7 → 1012 nơ-ron cha
(250 QA, 762 phát biểu). A/B trên **cùng thước đo gold-overlap** (`eval/qa_v3.py`):

| Bộ | baseline cũ (có teaching) | **v3 nơ-ron cha (KHÔNG teaching)** |
|:---|:---:|:---:|
| QA val (7) | 86% | **86%** |
| Paraphrase giữ kín (8) | 62% | **100%** |

→ **Ngang baseline ở QA đã thuộc, VƯỢT hẳn ở tổng quát hóa paraphrase** — đúng case "con kiến":
câu hỏi diễn đạt khác (`Người chữa bệnh là ai`, `Thủ đô của Việt Nam là gì`) vẫn dội đúng đáp
(`Bác sĩ chữa bệnh`, `Thủ đô Việt Nam là Hà Nội.`) nhờ khớp **token nội dung phân biệt** với
nơ-ron cha. **Không một dòng dạy ngữ pháp.**

**Ablation (Luật 0):** fuzzy E ON/OFF cho **cùng** 86%/100% — paraphrase ở bộ này tổng quát hóa
nhờ **khớp IDF token nội dung chung**, chưa cần E. Giữ E như cơ chế cho paraphrase **đồng nghĩa
khác mặt chữ** (cần bộ eval synonym để chứng minh — chưa có). Phân rã BPE/Re-Pair (trung gian)
cũng để dành: cha-level matching đã đủ cho bộ hiện tại.

Còn 1/7 QA val sai (`Đoàn kết giúp em thế nào` → khớp nhầm cha hình học) — token câu hỏi quá
generic; ISG (TK_013) đã sửa case `ở đâu` không nổi từ Re-Pair (cluster power=19 trên `_t3_isg`).

## 4c. KẾT QUẢ ISG (2026-07-03) — phân rã online qua link ngang

Spec: `TK_013_INCREMENTAL_SEG_GRAPH.md`. Checkpoint `grade_v3_big_a2_t3_isg.pkl`.

- Ingest: mỗi `add_example` → `slot_chain` + đếm bigram; batch rebuild 44k cha ~40s ingest +
  ~2 phút promote (40k cụm, heap).
- Case `ở đâu`: Re-Pair rank #4231 (không bao giờ merge); ISG → cluster **power=19**.
- Eval fast: QA 100/100, Para 88%, Syn 67% (ngang guard `_t3`).

## 5. Tài sản đã chứng minh GIỮ LẠI
- **E** (`core/embedding.py`) — 7/7 test; cụm ngữ nghĩa nổi đúng từ thống kê mạng, tôn trọng
  Luật 0. Dùng cho khớp mờ ở bước 1.
- **W_seq** (`EdgeKind.SEQUENTIAL`) — tín hiệu micro/fluency.
- **Eval thuần không-fingerprint** (`eval/qa_v3.py`) — chấm theo gold-overlap, A/B công bằng.
- **Train thuần không-teaching** (`scripts/train_v3.py`) — pipeline dữ liệu thật lớp 3–7.
- Skeleton `hca/generate_hebbian.py` — sẽ **viết lại lõi macro** theo nơ-ron cha + pattern
  completion (thay direct-readout co-occurrence hiện tại).

## 5c. Giới hạn đã đo — synonym probe (2026-06-30)

Bộ `data/grade_03/qa/synonym_probe.jsonl` (6 câu hỏi dùng **từ đồng nghĩa khác mặt chữ** cho
fact đã thuộc: `trị`≈`chữa`, `cấy`≈`trồng`, `khâu`≈`may`, `hôm`≈`ngày`). Kết quả **17% (1/6)**,
và **E ON/OFF/strong cho cùng số** → embedding **không** cứu được synonym ở corpus nhỏ:

- **Đa nghĩa (polysemy):** một vector/token. `trị` trong đồ thị nghiêng nghĩa *chính trị*
  (`trị`~`chính`/`ASEAN`) → fuzzy kéo SAI về cha lịch sử/chính trị, không phải "chữa bệnh".
- **Thưa thớt:** `cấy` quá hiếm → embedding vô dụng; `khâu` OOV.
- **Pha loãng:** token đồng nghĩa lạ (idf cao) làm loãng coverage của token phân biệt CHUNG
  (`lúa`,`bệnh`) → cha đúng rớt khỏi top-5.

→ Đây là **trần thật** của embedding suy-từ-đồ-thị trên data nhỏ (Luật 0: không pretrain). Vá
synonym cần **nhiều data hơn** (để đồng nghĩa đồng-hiện) hoặc nguồn embedding khác (vi phạm Luật
0). **Không** cố grind synonym bằng heuristic. Paraphrase *chia sẻ mặt chữ phân biệt* vẫn 100%.

> Thử "vá nhanh" loại token df=0 khỏi mẫu số coverage: **không cải thiện** synonym (token đồng
> nghĩa có df>0) và làm vỡ ràng buộc "chỉ chung `Hỏi:` → không khớp" → **đã revert** (Luật 0).

## 6. Track HOÃN — code-gen AST + dual-mode symbol table (docx phần bổ sung)
Chuỗi dài (truyện/code) cần trừu tượng hóa cấu trúc liên-mẫu: Soft (Hebbian vị trí → nơ-ron
`[SETUP]`/`[CONFLICT]` tự nổi) + Hard Symbol Table (tên biến/hàm chính xác) + AST + self-verify.
**Không code lần này** — phase nghiên cứu sau khi lõi QA tiếng Việt (nơ-ron cha) ổn.
