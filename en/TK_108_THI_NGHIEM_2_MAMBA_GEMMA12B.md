# TK_108 — Experiment 2: replacing Gemma 4 12B attention layers with Mamba-2 (on the M4)

*Night of 06→07/11/2026 · Mac M4 24 GB · MLX 0.32 · `mlx-community/gemma-4-12B-it-4bit` · script [scripts/exp2_mamba_layer.py](../scripts/exp2_mamba_layer.py) · results in `out/exp2_mamba_layer/`*
*English translation of the Vietnamese original in the repository root.*

## Questions

If the sliding-window (1024-token) attention layers of Gemma 4 12B are switched to Mamba:

1. How much capability does the model lose?
2. Does the model still work on prompts and data types not seen during training?
3. Does the model still work on sequences longer than those used in training?

## Summary

- **Replacing 3/40 sliding-window layers (P1, ~2.5M tokens, 2.2 hours on the M4):** Mamba recovers about **70% of what is lost** compared with dropping the attention entirely. The model still answers almost exactly like the original, including tool calls.
- **Generalises well:**
  - Trained only on Vietnamese chat (office, code and translation removed), yet on **English code** and **office** it reaches a comparable level.
  - **8K** sequences (4× the training length) are no worse.
- **The type of data matters:** training on conversations is much better than on raw web text (3-layer KL 0.52 vs 0.91). The chat version of Gemma predicts raw text very poorly to begin with.
- **Not yet "lossless":**
  - With just 3 layers replaced, about 15–20% of predicted tokens already differ from the original.
  - Generation shows a few small errors the original doesn't make.
  - Layer 14 is the hardest.
- **P2 (matching the final output) on the M4: not working yet.**
  - In bf16, the gradient through 48 layers points the wrong way.
  - In fp32, the gradient points the right way but training still increases KL, even with a very small lr.
  - Needs to be redone on a GPU with much larger batches (see section 5).

## 1. Method

- **Mamba-2 initialised from that layer's own attention:**
  - Q→C, K→B, V→x, O→out; q_norm and k_norm kept.
  - New parts: conv1d (initialised as identity), dt, A, D, and per-head output normalisation.
  - Per-head memory from 8 to 2048 tokens.
  - 47M parameters per layer.
- **P1:** each Mamba learns to imitate the attention output at its own layer, with inputs taken from the original model. The loss is MSE after `post_attention_layernorm`.
- **Layers replaced:** 1, 14, 30 (early, middle, late; all sliding-window). 1200 steps × 2048 tokens.
- **Evaluation on unseen data** (8 sequences × 2048; the 8K set is 2 sequences × 8192):
  - Vietnamese chat (multiturn, a different source from the training data);
  - Vietnamese office tasks;
  - English code;
  - 8K-long conversations.
- **Metrics:**
  - Mean per-token KL(original ‖ replaced): lower = closer to the original.
  - Share of top-1 predictions matching the original.
- **Compared:** original, **drop** (the layer's attention output = 0), **Mamba**, and **Mamba initialised only, untrained**.

## 2. Main results (P1, trained on Vietnamese chat)

KL / top-1 agreement with the original:

| Eval set | Drop 3 layers | Mamba 3 layers | Mamba layer 1 | layer 14 | layer 30 |
|---|---|---|---|---|---|
| Vietnamese chat | 1.69 / 63% | **0.52 / 81%** | 0.30 / 88% | 0.45 / 84% | 0.08 / 90% |
| Vietnamese office (unseen) | 2.09 / 61% | **0.63 / 80%** | 0.28 / 89% | 0.43 / 85% | 0.10 / 90% |
| English code (unseen) | 1.37 / 74% | **0.48 / 85%** | 0.22 / 92% | 0.34 / 88% | 0.08 / 92% |
| 8K long (4× training length) | 1.31 / 69% | **0.39 / 85%** | 0.24 / 90% | 0.31 / 88% | 0.07 / 92% |

**Mamba initialised only, untrained** (Vietnamese chat): 3 layers give KL 1.42 and 65% agreement, close to dropping (1.69). So copying Q/K/V/O alone is not enough; P1 training is what makes the difference.

**Trained on raw web text** (same configuration, evaluated on the conversation sets): 3 layers give KL 0.91 / 0.92 / 0.74 / 0.69, clearly worse than training on conversations. Reason: the chat version of Gemma predicts raw text very poorly (perplexity in the thousands). Ollama is the same: "The capital of France is" yields "...-". So teaching Mamba on raw text teaches it to imitate an abnormal operating mode of the model.

## 3. Real generation (6 of the 28 cases, 3 layers replaced)

- Reasoning is almost **identical** to the original: same structure, same ideas, same figures (119,101,600 / 19,101,600 VND). The tool call is identical: `get_weather("Đà Nẵng","2026-11-07")`.
- **New small errors:**
  - Spelling: corrects "dành" to "đạt", whereas the original correctly gives "giành".
  - Interest: one line writes "119,101.6" (wrong separator), the final total is still right.
  - COUNTIFS: repeats "Condition Condition".
- **Speed:** ~23.7 seconds vs ~22 seconds for the original. Slower because Mamba runs in Python/MLX without a dedicated kernel.
- Note: the model's thinking used up the whole 700-token limit, so this comparison is mostly about reasoning. File: `out/exp2_mamba_layer/chat/gen_train.json`.

## 4. Interpretation

1. **Generalisation to unseen data types: YES.**
   - Trained on Vietnamese chat, yet English code holds up even better than Vietnamese chat.
   - Mamba learns the general "mixing of information across tokens", not the content.
2. **Length: YES.**
   - Trained at 2K, running at 8K is still fine, KL even lower.
   - Reason: Mamba only replaces short-range viewing; long-range viewing is still handled by the 8 global layers.
3. **But losses accumulate.**
   - Each layer still has a small error (KL 0.08–0.45), and 3 layers add up to 0.4–0.6.
   - Replacing all 40 layers with P1 alone would certainly break the model visibly.
   - Replacing many layers requires **P2** so the layers compensate for each other's errors, plus more data.
   - An 8-layer replacement experiment is running to measure how fast the errors accumulate (section 6).
4. **Layers are not alike.**
   - Layer 30, right after a global layer, is easy to replace (KL 0.08).
   - Layer 14 is hard: its per-layer error is only 0.005 yet the output KL is 0.45 — a sensitive layer.
   - Consequence: replace the easy layers and keep attention in the sensitive ones.

## 5. P2 on the M4 — not achieved, and why

| Run | Configuration | Validation KL (step 0 → later) | Notes |
|---|---|---|---|
| a | bf16, L=1024 | — | 29.5 GB, spills to swap, 22 tokens/second, abandoned |
| b | bf16, L=512, lr 5e-5 | train KL 0.30 → 1.7 | increases |
| c | fp32, L=256×4, lr 2e-5 | 0.49 → 0.76 (step 25) | increases |
| d | fp32, L=256×4, lr 2e-6 | 0.49 → 0.48 → 0.64 (step 50) | decreases then increases |

**What was checked:**
- In **bf16**, following the gradient **increases** KL. The gradient through 48 bf16 layers is so noisy it is no longer a descent direction.
- In **fp32**, the gradient points the right way on a fixed batch: KL 0.279 → 0.239.
- Gradients of each component (sliding attention, global attention, MLP, output) all match finite differences.

**So the problem is in the optimisation, not the formulation.** Suspects:
1. **Batch far too small.** 1024 tokens per update, whereas the GPU plan is about 0.5M tokens.
2. **Adam moves all 47M weights per layer at once** based on a weak signal from the output.
3. **Length mismatch.** Trained at 256 but validated at 512.

**How to do it on a GPU:**
- About 100× larger batches, 2K–8K sequences.
- Keep the P1 loss (per-layer MSE) as an "anchor" so Mamba doesn't drift from what it learned.
- Or let P2 adjust only the small parameters (dt, A, D, normalisation, out_proj).

## 6. Replacing 8 layers — the model collapses, and why

P1 for 800 steps on conversation data for 8 layers: 1, 3, 7, 9, 14, 20, 26, 30. Directory: `out/exp2_mamba_layer/chat8/`.

**All 8 layers replaced at once (KL / top-1 agreement):**

| Eval set | Drop 8 layers | Mamba 8 layers |
|---|---|---|
| Vietnamese chat | — | 4.98 / 32% |
| Vietnamese office | 5.85 / 23% | 5.28 / 31% |
| English code | 3.09 / 53% | 1.41 / 71% |
| 8K long | 4.78 / 27% | 4.32 / 39% |

**Diagnosis by layer group** (Vietnamese chat, 2 sequences × 2048; [scripts/exp2_diag_subsets.py](../scripts/exp2_diag_subsets.py)):

| Replaced | KL | Top-1 agreement |
|---|---|---|
| layer 1 / 3 / 7 / 9 alone | 0.29 / 0.10 / **0.97** / **2.26** | 87 / 93 / 72 / 55% |
| layer 14 / 20 / 26 / 30 alone | 0.39 / 0.45 / 0.21 / 0.08 | 84 / 81 / 86 / 90% |
| layers 1, 3 → add 7 → add 9 | 0.29 → 0.95 → **4.47** | 88 → 72 → 32% |
| layers 14, 20, 26, 30 (second half) | **0.82** (below the sum of singles ≈ 1.13) | 73% |
| layers 1, 14, 30 | 0.62 | 78% |
| all 8 layers | 5.09 | 27% |

**Interpretation:**
- **The culprits are layers 7 and 9**, between global layers 5 and 11.
  - Their per-layer errors are very low (~0.008), but the output breaks badly.
  - Replaced together, the error is **amplified**: 0.97 and 2.26 alone, 4.47 together.
  - Hypothesis: early–middle layers do jobs requiring precision within the window, such as copying or looking up exact tokens — a known weakness of Mamba.
- **The second half of the model is easy to replace,** and replacing several layers there costs less than the sum of the singles.
- **Per-layer error does NOT predict sensitivity.** Output KL must be measured layer by layer.
- **Consequence:** the 40 sliding-window layers cannot all be converted blindly.
  1. Scan the sensitivity of all 40 layers.
  2. Convert only easy layers, keep attention in sensitive ones — a measurement-driven selective hybrid.
  3. Whether P2 can rescue layers 7 and 9 is unknown.

## 6b. Option A for P2 on the M4 — still no decrease, and the reason is now clear

**Option A:**
- Adjust only ~0.3M small parameters (dt, A, D, conv, normalisation); freeze Q/K/V/O.
- 8K tokens per update (32 × 256), fp32, P1 anchor λ = 2.
- Initialised from the 3-layer set 1/14/30.

**Result:** validation KL 0.492 → 0.533 → 0.525 → 0.586 (steps 0 / 5 / 10 / 15). KL on the training batches also rose slightly (moving average 1.15 → 1.24). Stopped at step 16 (`out/exp2_mamba_layer/kdA/`).

**Direct probe** ([scripts/exp2_kd_probe.py](../scripts/exp2_kd_probe.py)): fp32 gradient pooled over 8 batches × 512 tokens, one normalised step of length η:

| η | KL on those same 8 batches | KL on a fixed validation set |
|---|---|---|
| 0 | 1.22 | 0.492 |
| 0.01 | **0.72** | 0.508 |
| 0.03 | 1.05 | 0.754 |
| 0.1 | 1.03 | 0.553 |
| 0.3 | 1.11 | 0.704 |

**Conclusions:**
- **The P2 loss landscape is extremely steep and rugged.** Gradient norm about 290. A tiny step (0.01) cuts KL on that batch by 40%, but **KL on other data rises**, and the effect is not monotonic in η.
- So a gradient from a few thousand tokens mostly **fits that particular batch** and carries no general signal.
- P2 must pool gradients over **very many tokens per update** (hundreds of thousands), as published work does on GPUs.
- The M4 only manages about 50 tokens/second when backpropagating through the whole model, i.e. about 3 hours per 0.5M-token update. Not feasible.
- **This is not evidence that P2 is wrong**, it is evidence that P2 needs GPU scale. Validating P2 remains open: it needs a GPU pilot of about $20–40, batches of ~0.5M tokens, the 3 layers 1/14/30 first, with the criterion that validation KL decreases steadily below 0.49.

## 7. Conclusions for the plan

- **Technically doable:** initialising from attention, P1 training, and generalisation to unfamiliar data and long sequences all work.
- **The current advice stands:** use the original Gemma 4 plus LoRA and RAG when capability is needed. Converting to Mamba is only worth it if P2 on a GPU can push the KL of many layers close to 0.
- Next cheap steps:
  1. Look at the 8-layer replacement results.
  2. If we still want to continue: a ~$20–40 GPU pilot, large-batch P2, about the 12 easiest layers.

## 8. Sensitivity scan of all 40 sliding layers (07/11, M4)

Mamba P1 for every sliding layer (chat8 for 1,3,7,9,14,20,26,30 + sg1–sg4, 500 steps L=1024 per group of 8 layers). KL vs the original measured on 2 chat_vi + 1 code_en × 2048 (`scripts/exp2_sensitivity.py`, results in `out/exp2_mamba_layer/sensitivity/`).

**Per layer (Mamba KL; KL when dropped in parentheses):**
- Very easy ≤0.10: 43 (0.26), 40, 37, 39, 30, 31, 45, 3, 25, 34 — almost all in the second half (30–45).
- Easy 0.10–0.30: 27, 44, 2, 36, 26, 24, 4, 1, 22, 33, 38, 46 (dropped: 4.8!).
- Medium 0.30–0.50: 10, 21, 42, 32, 0 (dropped: 5.0!), 6, 14, 18, 19, 20, 16, 15.
- Hard 0.6–0.8: 12, 13 (dropped: 10–12 — vital layers), 7, 28.
- Broken: 8 (1.51 > dropped 0.66), 9 (1.67).

**Greedy combination (add the easiest layer each time):** 1→0.05 · 3→0.19 · 4→0.27 · 6→0.48 · 8→0.61 · 10→0.70 · 12→0.87 · 13→0.92 · 14→1.11 · 16→1.34.
- Nearly ADDITIVE (8 layers: sum of singles 0.59 vs measured 0.61) — very different from chat8 (KL 4.98, amplified by 7/9).
- Choosing the right layers: 8 Mamba layers at KL 0.61 instead of 4.98 (8× better).

**Conclusion:** with P1 alone (no P2) we can replace ~4–6 layers (8–12% of layers) at KL<0.5, ~10–13 layers at KL<1. The second half of the network (after global layer 29) is where Mamba belongs; keep attention at 7–9, 12–13, 28. P2 on a GPU is required to push to 20–40 layers.

### 8b. 8-layer hybrid generation (`out/exp2_mamba_layer/hyb8/gen_hyb8.json`)
Layers [43,40,37,39,30,31,45,3] combined from chat8/sg3/sg4. 6 real cases, 300 tokens (still in the thinking phase).
- Content/reasoning nearly preserved: email, 3-point summary, idioms (correct meanings), compound interest 119,101,600 correct, SUMPRODUCT/COUNTIFS chosen correctly, SQL task understood correctly.
- Characteristic errors: LOST CLOSING DELIMITERS / exact copying — `"Huỷ)`, `A2:500` instead of `A2:A500`, `'2026-10-01 AND` missing a quote, `AND_at_date` instead of `created_at`, `to__city`, missing closing `**`/backticks. → the second-half layers do pair-matching/exact copying (attention's job); most harmful for code/SQL/formulas.
- Speed 12 s vs 10 s (Mamba has no optimised kernel yet).

## 9. Replacing the GLOBAL layers with Mamba (07–08/11, M4)
Rationale: the KV of the 40 sliding layers is already fixed (1024 window) → replacing them gains almost nothing; the cost at 256K is in the 8 global layers. The mixer is generalised for global layers (H=16, Hkv=1, D=512, K=V, memory 8→32768). P1: gla [5,11,17,23] L2048×500; glb [29,35,41,47] L1024×1000 (L2048 hit swap). `scripts/exp2_global.py`, results in `out/exp2_mamba_layer/gl/global_eval.json`.

**KL per layer (drop → Mamba, 2K / 8K):** 5: 0.75→0.35/0.27 · 11: 1.64→**16.0**/14.8 (broken) · 17: 0.99→0.50/0.50 · 23: 0.68→0.27/0.25 · 29: 0.63→0.14/0.16 · 35: 0.56→0.07/0.06 · 41: 0.50→0.19/0.19 · 47: **12.1**→0.58/0.47. 8K ≈ 2K even though training was ≤2K.
Combinations: 7 layers (excluding 11) KL 1.53; additionally dropping 11 → 2.69; all 8 Mamba 12.9; all 8 dropped → 16.1.

**Needle in a haystack (a 4-digit code in ~7,600 tokens, 5 positions 4–49%):** original 5/5; Mamba in EACH single global layer (5,17,23,29,35,41,47) 5/5; layer 11: 0/5; **7 layers at once (global attention left only at 11): 0/5** — fluent answers but invented ("The code is 123", "not in the document").
→ When a single layer is replaced, the remaining global layers still do the retrieval; Mamba P1 does NOT learn precise long-range retrieval by itself. Consistent with the exact-copy errors in §8b.

**Layer 11:** nmse_post 0.11 (no NaN, not the BOS position) yet KL 16 → the mixer produces a harmful direction even when the post-norm error is small; keep attention at 11.

### 9b. How many global attention layers must be kept? (08/11)
Mamba P1 in the remaining global layers; KL 2K · needle 5 cases ~7,600 tokens:
| attention kept at | KL 2K | needle |
|---|---|---|
| 11,47 | 0.98 | 0/5 |
| 11,29 | 1.38 | 0/5 |
| 11,23,47 | 0.92 | 0/5 |
| 11,29,47 | 0.86 | 0/5 |
| 5,11,29,47 | 0.77 | 0/5 |
| **11,23,35,47** | 0.83 | **5/5** |
| **11,17,29,41** | 1.00 | **5/5** |
- Needs ≥4/8 global attention layers AND evenly spread (~12 layers apart); 5,11,29,47 (gap 11→29) fails despite the lowest KL → KL does not predict retrieval.
- When it fails it invents fluently ("The code is 101", "123456").
- P1-only conclusion: ~50% KV saving at long context (4/8 global → Mamba). Going further needs P2/retrieval training on a GPU.
