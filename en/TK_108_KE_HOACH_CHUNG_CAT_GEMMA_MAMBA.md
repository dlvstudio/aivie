# TK_108 — Option 2: distilling Gemma 4 into a hybrid Mamba architecture · P0 check results

*06/11/2026 · checked on the M4 (24 GB) · follows the [Gemma 4 26B evaluation](TK_108_DANH_GIA_GEMMA4_THUC_TE.md)*
*English translation of the Vietnamese original in the repository root.*

## Summary

| P0 check | Result |
|---|---|
| License | ✅ **Apache 2.0**, not gated. Modifying the architecture, continued training and release are allowed. |
| Is the 12B architecture suitable for Mamba replacement? | ✅ 48 layers: **40 sliding-window attention layers** and 8 global layers (5:1 ratio). Sliding-window layers map cleanly onto Mamba-2. |
| Quality of the original 12B (the "ceiling" for the hybrid) | ✅ **9.3/10** on 22 Vietnamese code cases, 21/22 usable. The 26B-A4B scores 9.8. |
| Running on the M4 (MLX / llama.cpp) | ✅ Feasible: MLX already has both `gemma4` and `mamba2`, which can be combined into one model. |
| **Practical benefit of switching to Mamba** | ❌ **Almost zero.** Gemma 4 is already memory-efficient for context: sliding-window layers only remember 1024 tokens, global layers share K=V with 1 head. Context memory at 32K tokens is only about 0.6 GB. Switching to Mamba saves about 0.25 GB and is barely faster (details in section 3). |

**Conclusion:**
- Technically doable. But **not worth roughly $2–3K** for the 12B: we would spend money and struggle to get back to 9.3 points, while gaining almost no speed or memory.
- **Recommendation:** for a real Vietnamese assistant, go with **option 1**, a Vietnamese LoRA on Gemma 4 12B or 26B-A4B (section 5).
- If the goal is a **Mamba-architecture model of our own**, take the cheaper route: distil Gemma into our own aivie 2.5B (section 6).

---

## 1. Original Gemma 4 12B on the 22 cases (same task set and grading as c6l1, Qwen, Gemma 26B)

`ollama gemma4:12b`, 4-bit, 7.7 GB. Settings: `MAXNEW=6144`, `CTX=16384`. No case was cut off (longest 4913 tokens). Average **~44 seconds per turn, ~43 tokens/second**.

| # | Case | Score | Comments |
|---|---|---|---|
| 01 | Vietnamese phone number | 9 | Ran 14 cases, all correct. Slightly lenient in accepting `+840…`. |
| 02 | pandas revenue | 10 | Ran: Excel with 2 sheets. Only creates the sample file **when `sales.csv` doesn't exist** (safer than the 26B). |
| 03 | mutable default `[]` bug | 10 | Ran: `['trà']`, `['cà phê']`. |
| 04 | KeyError 'email' | 10 | Correct explanation, 3 fixes. |
| 05 | SQL top 5 in Hanoi | 9 | Correct. But wrongly says `<= '2024-12-31'` is "fine" for a datetime column (it actually misses 31/12). |
| 06 | debounce | 8 | Textbook debounce. The real `fetch('/api/search?q=…encodeURIComponent')` line is there but commented out, while fake data is used. |
| 07 | login page | 10 | Mobile-friendly, 16px input text, has "Forgot password". |
| 08 | FastAPI todo | 10 | Create, Update, Response models, 404 on update and delete, delete returns 204. |
| 09 | email regex with de-dup | 10 | Ran: correct and order-preserving. |
| 10 | write pytest | 10 | Parametrize, edge cases, property and error checks. Ran: **11/11 passed**. |
| 11 | O(1) LRUCache | 10 | Ran: correct on a LeetCode-style sequence. |
| 12 | bash backup, keep 7 copies | 9 | `ls -1tr … \| head -n -7 \| xargs -r rm` correct on Linux. Doesn't check tar errors, variables unquoted. |
| 13 | Excel | 10 | VLOOKUP, XLOOKUP, IFERROR, `SUMIF "SP0*"`. |
| 14 | Python to JS | **3** | **Fails on run with `ReferenceError: clean_word`.** The model even left a "small fix here" line in the code, and if the variable name is fixed every word is counted twice. The output it printed is made up, not real output. |
| 15 | SQLite review | 9 | Catches SQL injection (with an example), plain-text passwords and the unclosed connection. The fix uses unsalted sha256 (with a note to use bcrypt). |
| 16 | 3-turn conversation | 10 | Three correctly connected turns. Ran: **3/3 tests passed**. |
| 17 | number-guessing game | 9 | All features. Prints a superfluous "you have 5 seconds to think" though there is no time limit. |
| 18 | git: committed `.env` by mistake | 10 | `rm --cached`, add `.gitignore`, then `commit --amend`. Notes the already-pushed case and suggests `.env.example`. |
| 19 | asyncio + aiohttp | 10 | `Semaphore(3)`, `ClientTimeout(10)`, gather, separates timeout, HTTP and connection errors. |
| 20 | set vs list | 10 | Easy to follow. Ran the timing example. |
| 21 | refactor discounts | 10 | Three approaches (dict of lambdas, config table, Strategy). Ran: matches the original code. |
| 22 | React todo | 9 | All correct. Only uses `onKeyPress` (deprecated). |

**Average 9.3/10 · usable (≥7): 21/22 · broken: 1/22 (case 14).**

Compared with the others: c6l1 scores 1.3 · Qwen2.5-Coder 7B 7.3 · Gemma 4 26B-A4B 9.8.
- The 12B is only slightly behind the 26B, at half the size (7.7 GB vs 16 GB) and the same speed.
- If we distil, this is **the ceiling**: at best the hybrid reaches about 9.3.

---

## 2. Gemma 4 12B architecture (read directly from the config and tensors)

- **Sizes:** hidden 3840, MLP 15360, vocab 262,144 (tied embeddings). `final_logit_softcapping` 30. A single `model.safetensors` of **23.9 GB** in bf16.
- **Global layers** at positions [5, 11, 17, 23, 29, 35, 41, 47]. The other 40 are 1024-token sliding window.
- **Sliding-window layers:**
  - 16 Q heads, 8 KV heads, head 256. `q_proj [4096,3840]`, `k_proj`/`v_proj [2048,3840]`, `o_proj [3840,4096]`.
  - Has `q_norm`/`k_norm`. RoPE 1e4.
- **Global layers:**
  - Head 512, **1 KV head, K = V** (no `v_proj`). `q_proj [8192,3840]`, `k_proj [512,3840]`.
  - Partial rotary 0.25, RoPE 1e6.
- **Other components:** `layer_scalar` and 4 norms per layer. The full model (`Gemma4UnifiedForConditionalGeneration`) includes image and audio parts; the hybrid only needs the text part.
- **Mapping a sliding-window layer to Mamba-2** (following "Mamba in the Llama" / MOHAWK):

  | Mamba-2 component | Taken from Gemma | Details |
  |---|---|---|
  | C | Q | |
  | B | K | ngroups = 8, d_state = 256 |
  | x | V | 8 heads repeated to 16, d_inner = 4096 |
  | out_proj | O | |
  | `dt`, `A`, `D`, `conv1d` | — | freshly initialised, very few parameters |

- **E4B is unsuitable for a pilot**, because it has PLE (`hidden_size_per_layer_input` 256) and 18 KV-sharing layers — considerably more complex.
- **31B** has 60 layers (50 sliding-window, 10 global), the same pattern as the 12B.

## 3. Why switching to Mamba gains almost nothing (P0's main finding)

Mamba pays off when it replaces **global attention**, whose context memory and cost grow with length. But in Gemma 4 12B:

| Component | Context memory (bf16) | If switched to Mamba |
|---|---|---|
| 40 sliding-window layers | Fixed 1024 × (2048 K + 2048 V) × 2 bytes ≈ 8 MB per layer, **~0.34 GB** in total, does not grow with length | Mamba state 4096 × 256 × 2 bytes ≈ 2 MB per layer, ~0.08 GB in total, **saves ~0.25 GB** |
| 8 global layers | 8 × 512 × 2 bytes = **8 KB per token**: ~0.26 GB at 32K, ~1 GB at 128K | Keep as-is — these are the layers that provide long-range memory |
| Weights | 7.7 GB (4-bit) to 24 GB (bf16) | Practically unchanged |

- **Speed:**
  - During generation, time is dominated by reading weights. Attention within a 1024-token window is already very cheap.
  - During prompt reading, the sliding-window part is already linear in length.
  - So Mamba **does not change the complexity class**, it only shaves a few percent.
- **Quality:** distillation always loses something. Published work loses roughly 1–5 benchmark points when replacing 50–75% of the layers. The hybrid would land at about **8–9 points** while the original is already at 9.3.
- **Replacing the 8 global layers too**, to benefit at very long context: these are exactly the layers that retrieve far-away information. From experience in TK_105/TK_107, long-range recall would drop sharply. Not advisable.

→ **Gemma 4 is already an "efficient hybrid" architecture.** Option 2 on the 12B buys the label "Mamba architecture" at the cost of money and some quality.

## 4. If we still want option 2 (for reference, not recommended)

| Phase | Work | Data / resources | Gate |
|---|---|---|---|
| P0 | Check (**done**, this file) | — | — |
| P1 | Per-layer alignment: Mamba-2 learns to mimic the attention output (MSE) for 20 sliding-window layers (50%) | 0.5–1B tokens, 1–2 H100, ~1 day | Per-layer MSE < 5% |
| P2 | Whole-model distillation with KL on logits. Ramp from 50% to 100% of sliding-window layers; length 2K, then 8K, then 16K | 3–8B tokens, 8×H100, ~5 days | Perplexity ≤ original +5% |
| P3 | Vietnamese SFT with rebalanced data, then teacher loop/DPO | C6 SFT data (already on the Mac) | — |
| P4 | Re-evaluate | 22 real cases, HE/MBPP/C7/tool/office, VMLU, needle 8–32K | ≥ 90% of the original's score (≥ 8.4/10) |
| P5 | Run on the M4 | MLX: combine `gemma4_text.py` with a Mamba-2 mixer | ≥ 40 tokens/second |

- **Cost:** a 50% pilot about $500–800; the 100% version about $2–3K (~1000 H100 hours).
- **Data to rebuild:** `data/cpt` (17 GB) and `data/cx3` (8 GB) were lost with the box. The build scripts remain (`scripts/build_cpt_data.py`, `scripts/build_code_expert_data.py`).

## 5. Recommended: option 1, a Vietnamese LoRA on Gemma 4 (architecture unchanged)

- **Base:** Gemma 4 12B (balanced) or 26B-A4B (best). Both already score 9.3–9.8 on code without any training.
- **LoRA goals:**
  - The "aivie" style and identity.
  - Tool use in our own format.
  - Office/Excel and knowledge about Vietnam.
  - More concise answers, since both versions are verbose.
- **Data:** C6 SFT, teacher-loop data, tool, office — all backed up on the Mac (`box_backup/data_1105/box_data_text.tgz`). It must be **rebalanced** to avoid c6l1's "see code → fix it" failure.
- **Cost:**
  - QLoRA on the 12B on 1 A100/H100 for a few hours, about **$20–60**.
  - Or try MLX-LoRA right on the M4 with the 4-bit 12B (slow but free).
- **Acceptance gate:** no drop on the 22 cases (≥ 9.0), correct identity and tool format, VMLU not lower.

## 6. If what we need is "our own Mamba model": distil into aivie 2.5B

- **Teacher:** Gemma 4 12B or 26B generates answers and (top-k) logits for a diverse task set.
- **Student:** our own 2.5B hybrid Mamba (c6 base). At this small size Mamba really pays off: fast and light on the M4.
- **Limit:** "understanding = model size" (the knowledge wall) still applies, so the realistic expectation is **5–7 points**, i.e. on par with Qwen-Coder 3B–7B, not 9.
- **Cost:** about $100–300 of GPU.

## Appendix — re-running

```bash
ollama pull gemma4:12b
MODEL=gemma4:12b OUT=out/real_eval_gemma4_12b MAXNEW=6144 CTX=16384 python3 scripts/real_use_eval_ollama.py data/eval_clean/real_code_22.json
```
