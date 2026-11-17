# LLM → Hebbian SLM — Bản đồ khái niệm

Tài liệu chuyển ngữ các khái niệm từ LLM mã nguồn mở (gpt-2, nanochat, DeepSeek) sang kiến trúc **nn-aivie** (Hebbian graph + HCA).

## Nguyên tắc

- **Không** copy backprop / Transformer weights
- **Dịch** cơ chế: attention → co-activation readout; KV cache → session state; SFT → teach + G2 imitation

## Ma trận chuyển đổi

| Khái niệm LLM | Ví dụ nguồn | Bản dịch Hebbian | Module nn-aivie |
|:---|:---|:---|:---|
| Scaled dot-product attention | gpt-2 `attn()` | Co-activation: $\sum_k d_k \cdot A[w_0,t_k]\cdot A[t_k,v]$ | `hca/forward.py` |
| Query-conditioned attention | Transformer Q·Kᵀ | Query readout từ salient Q + `QFocus:*` | `forward.query_conditioned_read` |
| KV cache | gpt-2 `past`, nanochat `KVCache` | `QASession`, active rules, fingerprint, filled slots | `hca/qa_context.py` |
| Causal mask | gpt-2 `attention_mask` | Autoregressive history only (đã có) | `generate.py` |
| Positional encoding | gpt-2 `wpe`, nanochat RoPE | `R_{-k}` position nodes + decay | `teaching/position.py` |
| Sliding window | nanochat `SSSL` | $K_{\max}$ local + global hop tới `Hỏi:` | `hca/config.py`, `forward.py` |
| BOS-aligned packing | nanochat `dataloader.py` | Mọi row train bắt đầu frame (`Hỏi:` / lesson BOS) | `data/qa.py`, corpus |
| SFT / chat template | nanochat `chat_sft.py` | Direct Teach + G2 trên mask answer | `teaching/`, `g2_imitation.py` |
| Task mixture | nanochat `TaskMixture` | Shuffle QA focus types trong G2 | `data/dialogue/`, manifest |
| MLA low-rank cache | DeepSeek-V3 `MLA` | Composite / co-fire hub nodes (roadmap) | Phase 2 README |
| YaRN long context | DeepSeek-V3 | Scope Neurons + $K_{\max}$ curriculum theo lớp | README Mục III |
| MoE experts | DeepSeek-V3 `MoE` | Rule-scoped subgraphs (chỉ rule active) | `matcher.py` |
| CoT / reasoning | DeepSeek-R1 | G4 self-talk + `[Verify]` structural nodes | roadmap G4 |
| Distillation | R1 → small models | Gold trace → G2 rule colocation imprint | `g2_imitation.py` — xem [01_OPEN_LLM_TO_HEBBIAN_TRANSFER.md](01_OPEN_LLM_TO_HEBBIAN_TRANSFER.md) |
| **Answer stopping / EOS policy** | Chat template, stop sequences | **Answer closure:** agenda + coverage + fp cont. → SATISFIED → `<EOS>` | [`ANSWER_CLOSURE_SPEC.md`](../ANSWER_CLOSURE_SPEC.md) — 📋 chưa implement |
| Bits-per-byte eval | nanochat | Metric so sánh SLM (eval harness) | `eval/metrics.py` |

## Lộ trình đọc code

| Tuần | Repo | Doc |
|:---|:---|:---|
| 1–2 | gpt-2 | [gpt2_attention.md](gpt2_attention.md) |
| 3–5 | nanochat | [nanochat_context.md](nanochat_context.md) |
| 6–8 | DeepSeek-V3/R1 | [deepseek_memory_reasoning.md](deepseek_memory_reasoning.md) |
| 9+ | LLM mở (HF) + chiến lược | [01_OPEN_LLM_TO_HEBBIAN_TRANSFER.md](01_OPEN_LLM_TO_HEBBIAN_TRANSFER.md) |

## Milestone nn-aivie

1. **POC (hiện tại):** QA frame, rule matcher, neuron-first chain, Q-fingerprint
2. **POC+ (2026-06-29):** Answer closure, trace↔generate sync, compound/anaphor teach — [`ANSWER_CLOSURE_SPEC.md`](../ANSWER_CLOSURE_SPEC.md)
3. **HCA v2:** query readout + dual-scale window
4. **SLM đa năng:** chat curriculum, GPU Hebbian batch, G4 verify, code lessons
