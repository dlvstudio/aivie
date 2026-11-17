# Tận dụng LLM mã nguồn mở → chuyển hoá điểm mạnh sang nn-aivie

Tài liệu này trả lời: *với những gì đã học từ gpt-2, nanochat, DeepSeek (xem các file study khác), làm sao **không copy backprop** mà vẫn chuyển **cơ chế** sang mạng Hebbian?*

**Nguyên tắc nn-aivie:** không nạp weight Transformer vào runtime; **distill hành vi / cấu trúc / curriculum** qua teach + G1–G4 + eval.

---

## 1. LLM mã nguồn mở dùng làm gì?

| Vai trò | Repo gợi ý | Cách dùng với nn-aivie |
| :--- | :--- | :--- |
| **Giáo viên sinh dữ liệu** | Qwen2.5, Llama 3, Mistral, Gemma (HF) | Sinh cặp QA, câu compound, paraphrase → **jsonl / lesson only** |
| **Giáo viên sinh trace** | nanochat SFT, bất kỳ chat model | Chuỗi token gold + mask → **G2 rule colocation imprint** |
| **Kiến trúc tham chiếu** | gpt-2, nanochat, DeepSeek-V3 | Map cơ chế → HCA / matcher (bảng dưới) |
| **Metric & benchmark** | nanochat bpb, DeepSeek NIAH | `eval/metrics.py`, `eval/niah.py` — so sánh xu hướng |
| **Tokenizer / BPE** | gpt-2 `encoder.py`, HF tokenizers | Chuẩn hóa corpus; **không** bắt buộc cùng vocab với SLM |

**Không làm:** fine-tune Transformer rồi gắn vào infer Hebbian; import softmax attention weights vào $A$.

---

## 2. Bảng chuyển hoá: điểm mạnh LLM → hành động Hebbian

### 2.1 Ngữ cảnh & attention

| Điểm mạnh LLM | Nguồn study | Chuyển hoá nn-aivie | Trạng thái |
| :--- | :--- | :--- | :--- |
| Causal KV cache | [gpt2_attention.md](gpt2_attention.md) | `QASession` + fingerprint + active rules | ✅ POC |
| Q·Kᵀ chọn token liên quan | gpt-2 `attn()` | `query_conditioned_read` + salient Q + `QFocus:*` | ✅ v1 |
| Sliding local + global layer | [nanochat_context.md](nanochat_context.md) SSSL | $K_{\max}$ decay + hop tới `Hỏi:` / anchor | ✅ một phần |
| RoPE / long position | nanochat, DeepSeek YaRN | `R_{-k}` + Scope Neurons + $K_{\max}$ theo lớp | ⏳ curriculum |
| Nén cache (MLA) | [deepseek_memory_reasoning.md](deepseek_memory_reasoning.md) | Composite hub / session summary node sau mỗi turn | 📋 Phase 2 |

**Hành động tiếp:** HCA v2 — lớp cuối “full hop” tới mọi salient Q + fingerprint (nanochat layer L); lớp trước chỉ chain local.

### 2.2 Huấn luyện & curriculum

| Điểm mạnh LLM | Chuyển hoá | Trạng thái |
| :--- | :--- | :--- |
| Pretrain trên corpus lớn | G1 passive co-occurrence + HF stream | ✅ |
| SFT mask phần assistant | `train_mask` sau `Đáp:` | ✅ |
| Task mixture (QA + chat + …) | manifest multi-dataset + G2 shuffle | ✅ grade 3 |
| Chinchilla / depth dial | `--grade` → pack size + $K_{\max}$ + corpus rows | ✅ một phần |
| **Distillation** | Teacher generate → jsonl → teach/G2 **path imprint** | 📋 chưa pipeline |
| RL / R1 reasoning | G3 dopamine + G4 `[Verify]` self-talk | ⏳ G3 optional; G4 roadmap |

**Pipeline distillation đề xuất (offline, không backprop):**

```
Open LLM (prompt) → gold answer text
       ↓
tokenize + align QA frame (Hỏi:/Đáp:)
       ↓
data/grade_XX/qa/train.jsonl  (+ optional compound lesson)
       ↓
teach-pack (QFocus imprint) + G2 teacher-forced steps
       ↓
eval_suite / human spot-check
```

Giữ **lọc chất lượng:** chỉ giữ câu phù hợp lớp, không vi phạm luật Mục 0 README (teacher = data, không = runtime if/else).

### 2.3 Routing & chuyên môn

| Điểm mạnh LLM | Chuyển hoá | Trạng thái |
| :--- | :--- | :--- |
| MoE expert routing | DeepSeek `Gate` | Rule arbitration — subgraph theo `Rule:*` active | ✅ matcher |
| Multi-domain corpus | nanochat multi-task | `toan/`, `dia_ly/`, … + `domain` trong parquet | ✅ 2026-06 |
| Structured output JSON | chat templates | QA JSON modality + `infer_modality` | ✅ một phần |

### 2.4 Sinh văn bản & văn phong

| Điểm mạnh LLM | Giới hạn Hebbian hiện tại | Chuyển hoá |
| :--- | :--- | :--- |
| Câu ghép, đại từ tự nhiên | Lặp S hoặc cụt vế | Teacher sinh mẫu `S …, nó …, là …` → lesson Var:P + COMPOUND modality |
| Paraphrase đa dạng | Overfit literal lesson | LLM sinh 5–10 paraphrase / QFocus → G1+G2 |
| Coreference xa | $K_{\max}=16$ | YaRN-style: tăng $K$ theo lớp + retroactive wave |

---

## 3. Ba kênh “ăn kiến thức” từ LLM (theo mức can thiệp)

### Kênh 1 — Chỉ dữ liệu (an toàn nhất, khuyến nghị trước)

- LLM sinh **QA jsonl**, fact môn học, hội thoại → `build_subject_knowledge.py` / `qa/train.jsonl`
- Paraphrase lesson examples → tăng đa dạng colocation **cùng QFocus**
- **Không** đụng runtime

### Kênh 2 — Distillation trace (cốt lõi SLM Hebbian)

- Lấy **chuỗi token** teacher làm gold; G2 imprint từng bước `rule_colocation(w_{t-1}, w_t)` dưới mask
- Tương đương R1 stage 4 “distillation SFT” nhưng cập nhật $A$ thay vì $W$
- Có thể dùng **nhiều teacher** (Việt tốt: Qwen2.5-Instruct, VinLlama, …) → mixture trong manifest

### Kênh 3 — Khởi tạo / gợi ý cấu trúc (tùy chọn, cần ablation)

- **Soft co-occurrence:** logits teacher top-k → trọng số G1 nhẹ trên cặp token (vẫn Hebbian, không lưu $W$)
- **Hub nodes:** cluster embedding teacher → đặt tên Composite node (offline); không bắt buộc
- **Rủi ro:** dễ thành hybrid backprop tinh vi — chỉ thử sau khi Kênh 1–2 plateau

---

## 4. Lộ trình ưu tiên (12 tuần gợi ý)

| Tuần | Việc | LLM leverage |
| :---: | :--- | :--- |
| 1–2 | Sync trace + QA stop; siết fingerprint sau `.` | — |
| 3–4 | Script `distill_qa.py`: HF model → jsonl grade 3–5 | Kênh 1 |
| 5–6 | Compound + anaphor lesson (teacher viết mẫu) | Kênh 1 + teach |
| 7–8 | G2 batch trên distilled paths; eval QA val | Kênh 2 |
| 9–10 | NIAH dài + tăng $K_{\max}$ lớp 6–7 | DeepSeek eval pattern |
| 11–12 | Session hub (MLA-inspired); ablation G3 | Kênh 3 thử nghiệm |

---

## 5. Công cụ mở nguồn cụ thể

| Mục đích | Gợi ý |
| :--- | :--- |
| Sinh QA tiếng Việt | `Qwen/Qwen2.5-7B-Instruct`, `vilm/VinLlama-2.7B-chat` (HF) |
| Pipeline chat jsonl | Tham khảo `nanochat/scripts/chat_sft.py` (format, không copy trainer) |
| Tokenizer | `gpt2` BPE hoặc `Vietnamese` sentencepiece — thống nhất trước khi distill |
| Eval tham chiếu | Chạy cùng prompt trên teacher vs nn-aivie checkpoint — bảng so sánh thủ công |
| Embedding eval (optional) | `sentence-transformers` chỉ để **lọc** duplicate jsonl, không vào graph |

---

## 6. Liên kết tài liệu study

| File | Nội dung |
| :--- | :--- |
| [00_LLM_TO_HEBBIAN_MAP.md](00_LLM_TO_HEBBIAN_MAP.md) | Bản đồ khái niệm ngắn |
| [gpt2_attention.md](gpt2_attention.md) | KV, causal, readout |
| [nanochat_context.md](nanochat_context.md) | SSSL, BOS pack, SFT mask |
| [deepseek_memory_reasoning.md](deepseek_memory_reasoning.md) | MLA, MoE, R1 stages |
| **01_OPEN_LLM_TO_HEBBIAN_TRANSFER.md** (file này) | Chiến lược chuyển hoá & distillation |
| [ANSWER_CLOSURE_SPEC.md](../ANSWER_CLOSURE_SPEC.md) | Thoả mãn câu trả lời, essay agenda, roadmap implement |

---

## 7. Answer closure & essay gold (liên kết POC)

Distill từ teacher LLM không chỉ sinh **cặp QA ngắn** mà còn **essay multi-section** cho grade 6–7:

```
Teacher prompt → gold A có [1]…[n] section → jsonl → teach Lbl:ans_seg_k + must_cover
```

Runtime **không** copy stop sequence của chat theo chuỗi ký tự cố định — dùng **`assess_answer_closure()`** (agenda + coverage + fp continuation). Chi tiết:

- [`docs/ANSWER_CLOSURE_SPEC.md`](../ANSWER_CLOSURE_SPEC.md) — spec closure & roadmap P1–P3
- [`docs/QA_CONTEXT_SPEC.md`](../QA_CONTEXT_SPEC.md) — QFocus, bleed case studies
- README Mục **0.2** — bài toán & kế hoạch implement

---

## 8. Tóm tắt một câu

**LLM mã nguồn mở là giáo viên và kiến trúc tham chiếu — nn-aivie học bằng imprint đồ thị (teach/G2/G3), không sao chép softmax weights; điểm mạnh cần chuyển thành cache session, mixture curriculum, distillation trace, và eval benchmark — không thành hardcode runtime.**
