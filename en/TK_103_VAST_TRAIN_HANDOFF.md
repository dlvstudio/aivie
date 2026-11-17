# TK_103 — Handoff: training the 1.3B hybrid on a Vast H100 (full power)

*English translation of the Vietnamese original in the repository root.*

> Written: 2026-09-10 · Follows [TK_100](TK_100_GPU_SCALEUP_PLAN.md) §15, [TK_101](TK_101_M4_DATA_RUNBOOK.md), [TK_102](TK_102_GPU_SMOKE_FINDINGS.md)
> Status: **DONE** — pretraining 143,433/143,433 (2026-09-29 00:07Z) + SFT 5,216/5,216 (2026-09-29 04:38Z). Evaluation results in **§9**. The Vast box is no longer needed (SSH refused as of 2026-09-29 12:28).

## 0. TL;DR — where we are

- The **1.277B hybrid Mamba+Attention model** has finished pretraining on **18.8B tokens** (143,433 steps) on a **full-power Vast.ai H100 80GB** (700W, 26.2K tok/s, MFU ~20%).
- **SFT** for 2 epochs on `data/sft_vi/sft_mix.jsonl` (85K conversations) → val loss **2.431 → 0.54**.
- Checkpoints live on **HF** (see §3) and the **M4 Mac** (`checkpoints/hf_hybrid/`).
- Evaluation conclusion (§9): SFT teaches **format/behaviour** (answers directly, knows when to stop, emails, refusals, short-sentence translation); **knowledge is still fabricated**, **code is nearly useless**.

## 1. Box connection (historical — the box is gone)

```bash
ssh -i ~/.ssh/<key> -p 1352 root@<IP>       # direct; proxy: -p 37784 root@<vast-proxy>
tmux attach -t train                                     # (Ctrl+B D to detach, do NOT kill)
tail -f /workspace/nn-aivie/train.log                    # pretrain · SFT: sft.log
```

- The old box `<IP>:40071` is lost. To filter Vast noise: `... 2>&1 | grep -v "Welcome\|authentication\|Have fun"`.

## 2. Training config (MUST be kept identical when resuming — read from `cfg` in the checkpoint)

```
d=2048 layers=48 attn_every=8 (→6 attn @ [7,15,23,31,39,47]) d_state=16 heads=16 kv_heads=None
ctx=2048 · bs=8 grad_accum=8 (=131,072 tok/step) · lr=3e-4 min_lr=3e-5 warmup=200
tokens=18,800,060,992 · grad_ckpt=False compile=False
```
⚠️ **The trainer defaults are `grad_accum=16 / tokens=26e9` — DIFFERENT.** You must pass `--grad-accum 8 --tokens 18800060992` explicitly, otherwise the cosine LR schedule shifts → broken resume.

Pretraining command (resume). `--hf-repo` is **MANDATORY** — lesson of 2026-09-21: no HF backup → steps 15k–90k were lost when the old box was destroyed.
```bash
cd /workspace/nn-aivie
# the WRITE token already lives in $HF_HOME/token (chmod 600) — DON'T paste tokens into commands/chat
PYTHONPATH=scripts python scripts/train_hybrid_scale.py \
  --data /workspace/nn-aivie/pretrain_vi \
  --ctx 2048 --d 2048 --layers 48 --attn-every 8 --d-state 16 --heads 16 \
  --bs 8 --grad-accum 8 --lr 3e-4 --min-lr 3e-5 --warmup 200 \
  --tokens 18800060992 --eval-every 500 --save-every 1000 \
  --hf-repo donaldluong/nn-aivie-hybrid-1p3b-ckpt --hf-every 1000 \
  --init-from /workspace/nn-aivie/ckpt/<latest checkpoint> \
  --out /workspace/nn-aivie/ckpt/hybrid_1p3b_cont.pt
```

SFT command as run (`scripts/ensure_sft.sh` generates `run_sft.sh`; onstart calls it again if the box restarts, re-running from scratch because SFT can't resume yet; on completion → `ckpt/sft.done`):
```bash
PYTHONPATH=scripts:src python scripts/train_hybrid_sft.py \
  --init-from /workspace/nn-aivie/ckpt/hybrid_1p3b_cont.pt \
  --data data/sft_vi/sft_mix.jsonl --tokenizer checkpoints/bpe_v1.json \
  --max-len 2048 --bs 8 --grad-accum 4 --lr 2e-5 --min-lr 2e-6 --warmup 100 --epochs 2 \
  --eval-every 200 --save-every 500 \
  --hf-repo donaldluong/nn-aivie-hybrid-1p3b-sft --hf-every 500 --hf-name hybrid_1p3b_sft.pt \
  --out /workspace/nn-aivie/ckpt/hybrid_1p3b_sft.pt
```

Manually push one milestone (A/B, safe if the box dies midway):
```bash
HF_HUB_DISABLE_XET=1 PYTHONPATH=scripts python scripts/hf_backup_ckpt.py \
  --path /workspace/nn-aivie/ckpt/hybrid_1p3b_cont.pt \
  --repo donaldluong/nn-aivie-hybrid-1p3b-ckpt --step <N>
```

## 3. Checkpoints — where they are

| Version | HF | M4 Mac | Size |
|:---|:---|:---|:---|
| Final pretrain **step 143,433** (with `opt`) | `donaldluong/nn-aivie-hybrid-1p3b-ckpt` / `hybrid_1p3b.pt` | `checkpoints/hf_hybrid/backup_hybrid_1p3b_pretrain_step143433.pt` | 15.3GB |
| Pretrain step 143,000 (spare) | `donaldluong/nn-aivie-hybrid-1p3b-ckpt-a` | — | 15.3GB |
| **Final SFT step 5,216** (`state` only) | `donaldluong/nn-aivie-hybrid-1p3b-sft-a` (pointer `…-sft/latest.json`) | `checkpoints/hf_hybrid/hybrid_1p3b_sft_step5216.pt` | 5.1GB |

Format: `{"cfg", "state", "opt"?, "step"}`. `cfg` holds all the original parameters:
```python
import torch; o=torch.load("<ckpt>",map_location="cpu",mmap=True,weights_only=False); print(o["step"], o["cfg"])
```

Download to the Mac (reads `latest.json` → `slot_repo`, retries if the slot is deleted midway, deletes older copies with the same prefix):
```bash
PYTHONPATH=scripts:src .venv/bin/python scripts/download_hybrid_ckpt.py                          # pretrain
PYTHONPATH=scripts:src .venv/bin/python scripts/download_hybrid_ckpt.py \
  --repo donaldluong/nn-aivie-hybrid-1p3b-sft --prefix hybrid_1p3b_sft                           # SFT
```
⚠️ Files named `hybrid_1p3b_step*.pt` are deleted on the next download → copies to keep must be renamed (e.g. `backup_…`).

## 4. HF A/B backup (final version, ran reliably through pretraining + SFT)

- The pointer repo `<repo>` only holds `latest.json` (`step`, `bytes`, `slot_repo`). Checkpoints live in slot `<repo>-a` / `<repo>-b`.
- Each milestone: upload to the **NON-active** slot → verify size (`get_paths_info`) → write `latest.json` (slot + pointer) → **only then** delete the old slot. Upload fails → **KEEP** the old slot. There is never a window with nothing on HF.
- Pretraining: sidecar `hf_backup_watch.py --every 1000` (tails `train.log`, catches the `LƯU … @step N` line). SFT: in-process backup (`maybe_hf_backup`, `--hf-every 500`).
- Always `HF_HUB_DISABLE_XET=1` on the box.

## 5. REMAINING WORK

- [x] Pretrain 143,433 steps · [x] HF A/B backup · [x] 2-epoch SFT · [x] download both to the M4 · [x] prompt evaluation (§9).
- [ ] **User: Destroy the Vast box** (if only Stopped) — everything worth keeping is on HF + M4.
- [ ] **User: REVOKE the WRITE HF token** that leaked in chat (and the READ token): https://huggingface.co/settings/tokens.
- [ ] Improvement directions (choose by goal, see §9.4): code → continued pretraining on code; knowledge → RAG; translation → add PhoMT/MTet to SFT; measure quantitatively with `eval_vmlu.py`.

## 6. TRAPS hit & how to avoid them (important for the next session)

1. **Vast host power-capping the GPU** (the biggest trap): the first box was capped at 500/700W → clock 930/1980MHz → **only 15K tok/s** (vs 26K at full power). The container CANNOT lift it (`nvidia-smi -pl 700` = Insufficient Permissions). → **ALWAYS check `nvidia-smi -q -d POWER` (Current vs Default Limit) + `-q -d CLOCK` RIGHT after renting, BEFORE a long run.** DLPerf/TFLOPS in the Vast listing are NOT reliable for detecting a cap.
2. **mamba-ssm 2.2.2 must be built from GIT** (`state-spaces/mamba@v2.2.2`, `--no-build-isolation --no-deps`), NOT from PyPI (the sdist lacks csrc/*.cpp → ninja fails). Then sed out the `import MambaLMHeadModel` line from `__init__.py`. Patched into `setup_container.sh` (commit a402140).
3. **Vast `/workspace` = a directory on the SAME overlay disk as `/root`** (not a separate volume). Stop keeps the disk; **Destroy WIPES it**. `vastai copy` between 2 instances only runs at ~11MB/s; box→box rsync via the proxy ~6MB/s. **HF download is fastest (~30MB/s)** for data.
4. **The `hf` CLI is broken on this image** (`Typer … suggest_commands`) → use the `huggingface_hub` **Python API**.
5. **No HF backup = losing weeks of training.** 2026-09-21: the old box went offline, checkpoints 15k–90k were not on HF. Long runs **must** have `--hf-repo` + a WRITE token in `$HF_HOME/token`.
6. **HF 403 storage limit when overwriting a 15GB file** in a private repo (squashing history does NOT free quota; only deleting the repo does). Delete-then-upload creates a window with **no copy at all** if the box dies midway → switched to **A/B slots** (§4).
7. **XET uploads hang at 100%** → `HF_HUB_DISABLE_XET=1`. (Downloads on the Mac still work over XET: the `.incomplete` file looks stuck but `nettop` shows bytes arriving at ~5.5MB/s — XET buffers before writing, it isn't hung.)
8. **A running trainer keeps the old code in RAM** — editing `hf_backup_ckpt.py` on disk has no effect until restart → use a sidecar watcher (separate process) instead of restarting training.
9. **Vast onstart** (`/workspace/onstart.sh`) must change per phase: after pretraining call `ensure_sft.sh`, otherwise a box restart re-runs pretraining.
10. PowerShell ↔ ssh: use `@'…'@` (single-quoted) blocks so bash `$`/heredocs aren't swallowed by PowerShell; an unquoted heredoc once wrote the **real token** into `.bashrc` → always `export HF_TOKEN=$(cat /workspace/.hf/token)`.

## 7. Security constraints (KEEP)

- **DON'T push binaries** to git: before each commit, `git diff --cached --name-only | grep -E "\.(pt|npz|npy|index|tar|zst)$"` must be empty.
- **Unsigned commits**: `git commit --no-gpg-sign`.
- Commit attribution: `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.
- HF token: only in `/workspace/.hf/token` (chmod 600) — never printed in chat/argv/logs/git.

## 8. Actual time/cost

| | Value |
|:---|:---|
| Pretraining (new box, from step 11k) | 2026-09-21 → 2026-09-29 00:07Z, ~7.6 days, stable 26.2K tok/s |
| SFT | 2026-09-29 01:38Z → 04:38Z, **~3 hours** (~2.05 s/step, 69GB VRAM, ~600W) |
| Box price | $2.281/h (~$24 per billion pretraining tokens) |

## 9. Results

### 9.1 Pretraining

| Step | loss | bits |
|:---|:---|:---|
| 115,500 | 1.983 | 2.967 |
| 142,000 | 1.980 | 2.822 |
| 142,500 | 1.981 | 2.808 |
| 143,000 | 1.977 | 2.896 |

Base-LM prompts (M4, `gen_hybrid_scale.py`, steps 99k → 114k): fluent blog/SEO-style Vietnamese, less off-topic as steps increase; general knowledge slightly better (Nguyễn Du – The Tale of Kiều, 3,254 lines; the Earth orbits the Sun) but still fabricated (Hanoi "the largest in VN", air "dry and humid"); **code = numeric garbage, translation = doesn't follow instructions** (only the few-shot `English: … Vietnamese:` pattern produced "Xin chào, bạn khỏe không?").

### 9.2 SFT

- Data: 83,469 train · 1,703 val (2 samples with empty assistant turns dropped). Big sources: dich-synth 10k, office-synth 10k, code-synth 10k, aya-vi 8k, gemma4-26b-dialog 5.4k, code-long 5k, office-long 5k, sea-instruct 4k, bkai-selfchat 4k, legal-convo 3.5k, … (26 sources). Average ~450 tokens per conversation.
- Loss masked to assistant tokens only; template `Hỏi: …\nĐáp: …<EOS>` ("Question: … Answer: …").

| Step | val loss |
|:---|:---|
| 0 (before SFT) | 2.431 |
| 200 | 0.994 |
| 1,000 | 0.786 |
| 1,200 | 0.620 |
| 4,600 | 0.830 |
| 4,800 | 0.722 |
| 5,000 | 0.521 |
| 5,200 | 0.460 |
| final | **0.542** |

Val fluctuates a lot because each evaluation uses only 10 random batches; the downward trend is clear. The big drop in the first 200 steps = learning the Q/A template + stopping at `<EOS>`.

### 9.3 SFT prompt evaluation (M4, `gen_hybrid_scale.py --chat`)

Run with: `--chat --n 160 --ctx 512 --temp 0.7 --rep-pen 1.1` (general) · `--n 200 --temp 0.5 --rep-pen 1.05` (code/translation). ~5 tok/s on MPS (pure-PyTorch Mamba).

| Group | Result |
|:---|:---|
| Following instructions / stopping | ✅ "What is the capital of Vietnam?" → "The capital of Vietnam is Hanoi." · 3 tips for learning English → exactly 3 points, then stops |
| Office | ✅ Leave-request email in the right office tone, with a `[Date]` placeholder |
| Health | ✅ High fever: markdown heading + bullets, cool compresses, drink water, paracetamol "if the doctor allows" |
| Safety | ✅ Refuses to forge documents, cites the law (article/decree numbers may be invented) |
| Short-sentence translation | ✅ "AI is transforming healthcare" → correct Vietnamese · "Hello, how are you today?" → correct · "Việt Nam là đất nước xinh đẹp" → "Vietnam is beautiful." · ⚠️ fox → "dog"; "I want to learn machine-learning programming" (in Vietnamese) → answers like an assistant instead of translating |
| Knowledge | ⚠️ Nguyễn Du 1765–1820, Tiên Điền–Nghi Xuân–Hà Tĩnh, father Nguyễn Khản ✅ (mother invented) · ❌ "78% oxygen, 21% nitrogen" (swapped) · ❌ the Earth orbits the Sun in "258 days" · ❌ Hanoi "335 km²" · ⚠️ Tấm Cám plot invented, doesn't keep to "3 sentences" |
| Code | ❌ Fibonacci / prime numbers / SQL / async-await: wrong syntax, meaningless repetition. Only learned the **form** (```` ```python ```` blocks, `import`, `pd.read_csv`, `print(df.head())`) |

### 9.4 Conclusions & directions

- Confirms the thesis of TK_100 §S4 / the Mamba-on-M4 work: **SFT teaches FORMAT, not KNOWLEDGE.** Assistant behaviour is there; factual errors are the pretraining limit (1.3B × 18.8B tokens).
- **Code is weak** because the pretraining corpus is almost all Vietnamese with little code; 15k code samples in SFT can't compensate → needs **continued pretraining on code** (The Stack / StarCoderData) if code is a goal.
- **Knowledge** → **RAG** (existing FAISS) is cheaper than more pretraining.
- **Translation** → add bilingual data (PhoMT, MTet) to the SFT mix, **keep translation in the overall mix**.

### 9.5 VMLU (hard numbers, 2026-09-29) — new `ours` backend in `eval_vmlu.py`

Scored by **log-likelihood of the letter token** (" A".." E" are each 1 token → 1 forward per question, no generation; ~0.4s/question on MPS). SFT version `hybrid_1p3b_sft_step5216.pt`, `validation` split (744 questions, 0-shot):

| | ACCURACY | Notes |
|:---|:---|:---|
| **HybridLM 1.3B SFT (ours)** | **24.60%** (183/744) | = random (25% for 4 choices) |
| Random | 25.0% | lower bound |
| SeaLLM-7B-v2.5 | ~53.3% | upper reference |

→ **Hard confirmation: the model has NO reliable knowledge/reasoning** (correct only by chance). Many subjects 0–7%. This is NOT a training bug — it's the ceiling of 1.3B × 18.8B Vietnamese tokens. Every later knowledge improvement is re-measured with this exact command (baseline = 24.6%).

Command: `PYTHONPATH=scripts:src .venv/bin/python scripts/eval_vmlu.py --backend ours --ckpt checkpoints/hf_hybrid/hybrid_1p3b_sft_step5216.pt --split validation`

### 9.6 RAG (`rag_hybrid.py`) — bring in external facts to offset the knowledge ceiling

BM25F over viwiki (285k articles) → idf-weighted `_focus` → SFT template `Hỏi:{sources+question}\nĐáp:` → HybridLM 1.3B SFT generates → verify the grounded ratio. Idea: VMLU shows the model EXPRESSES well but FABRICATES facts → let retrieval handle the facts and the model just quote/paraphrase. Demo results: see §10.

## 10. RAG demo results (2026-09-29) — THE DECISIVE TEST

4 questions, `rag_hybrid.py --demo --max-tokens 140`, k=3, MPS (pure PyTorch, kernel=False):

| Question | Time | grounded | Verdict |
|:---|:---|:---|:---|
| How many moons does Mars have? | 321s | 0.64 | ❌ "has 4 moons: Mars (Saturn), Jupiter…" — **FABRICATED**, failed to extract "2: Phobos/Deimos" from the source |
| What is climate change? | 96s | 1.00 | ✅ copied the definition almost verbatim from the source — CORRECT |
| Which countries does the Mekong flow through? | 60s | 1.00 | ⚠️ "three countries: China, Laos, Thailand" — **INCOMPLETE** (really 6). grounded=1.0 but still WRONG |
| What is artificial intelligence? | 298s | 0.85 | ✅ fluent, correct definition of AI — but wordy, cut off midway |

**The real conclusions (important):**
1. **RAG DOES help DEFINITION/DESCRIPTION questions** — when the answer *is a sentence already in the source*, the 1.3B copies/paraphrases it well (climate change, AI: correct, high grounding). A real step up from asking cold (VMLU 24.6% = all fabricated).
2. **RAG CAN'T rescue questions needing exact EXTRACTION/COUNTING** — "how many moons" → fabricated list; "which countries" → undercounted. The 1.3B is **too weak to reason over context**; it only copies generic passages.
3. **The grounded ratio is MISLEADING**: Mekong grounded=1.0 yet WRONG (all words appear in the source, the model picked the wrong/incomplete ones). → don't trust grounding alone; real right/wrong evaluation is needed.
4. **M4 speed is NOT practical**: 60–320s per question (Mamba has no CUDA kernel on MPS). Fine for a demo, not for an interactive assistant.

**Direction from here (honestly):**
- The RAG pipeline (retrieve→ground→generate→verify) **works and is model-independent**. The bottleneck now is that **the 1.3B generator is too weak to extract precisely**.
- **Upgrade the generator** for hard questions: use a stronger model in RAG — the project already has `assistant_ours.py` (QLoRA Qwen-3B, reported grounding 0.8–1.0) and `rag_seallm.py` (SeaLLM-7B). Our 1.3B suits being a **Vietnamese text generator** for definition questions, not reasoning questions.
- **If keeping the 1.3B**: add **extraction-style SFT** ("answer BASED ON the following context", with counting/listing questions) so the model learns to *stick to the source* instead of fabricating; and improve `_focus` so the sentence containing the answer sits at the start of the context.
- **Measure RAG quantitatively** (not just a 4-question demo): a set of ~50–100 questions with answers, scored truly right/wrong (not just grounded) — before/after RAG — to get a number like VMLU.

### 10.1 RAG measured on VMLU (hard numbers, 2026-09-29)

`eval_vmlu.py --backend ours --rag --rag-k 3` (stuffs 3 BM25F+focus viwiki passages into the prompt, same 744 validation questions, same letter log-likelihood scoring):

| Config | Accuracy | Δ vs baseline |
|:---|:---|:---|
| SFT (no RAG) | 24.60% (183/744) | — |
| **SFT + RAG viwiki k=3** | **24.06% (179/744)** | **−0.54%** (4 questions worse, within noise) |
| Random | 25.0% | — |

**Conclusion: RAG does NOT raise VMLU.** Because (1) viwiki doesn't cover many subjects (math/physics/law/reading comprehension); (2) MCQ answers are not sitting in the retrieved wiki passages; (3) the 1.3B is too weak to *read the context and then choose* — it chooses on surface features. Stuffing long context also dilutes the signal → slightly lower.

**The full RAG picture (demo §10 + numbers §10.1):** RAG only helps when **the answer IS a sentence already in the source** (definitions/descriptions → the model copies it well). It does NOT help **reasoning/extraction/counting/MCQ** questions. → RAG is no cure-all for the 1.3B; its value is narrow.

**Direction decision (based on the numbers):** the retrieval pipeline is good + model-independent; the bottleneck is the **1.3B generator**. For a truly usable assistant → swap in a stronger generator, keep retrieval: `assistant_ours.py` (QLoRA Qwen-3B) or `rag_seallm.py` (SeaLLM-7B). Our 1.3B = a **research artifact** (proving the hybrid Mamba+Attn pipeline trains end to end on Vietnamese), suitable as a text generator for definition questions, not as a knowledge/reasoning core.

## 11. Length-extrapolation probe (2026-09-29) — DIRECTION (b): long context

`scripts/probe_length_extrap.py` — measures bits/token by position on 8,192-token viwiki sequences (4× training context), **pretrain** checkpoint step143433, 3 zero-cost context-extension modes (RoPE monkey-patch).

| RoPE mode | in-dist (<2048) | extrapolated (>2048) | Δ |
|:---|:---|:---|:---|
| none (original) | 3.641 | 4.516 | **+24%** (degrades) |
| **ntk** (base×scale^(hd/(hd-2))) | 3.696 | 3.621 | **−2%** ✅ |
| **pi** (position compression ×2048/L) | 3.840 | 3.668 | **−4%** ✅ |

**Conclusions:**
- The original model does **NOT collapse** at long context (bits ~4.5, not ~random) → the **Mamba layers carry information well past 2048**; a pure Transformer with ctx=2048 would be worse.
- The +24% degradation comes from **RoPE in the 6 attention layers** (rotation frequencies unseen beyond position 2048). **NTK/PI fix it CLEANLY → Δ≈0, FREE, no training.**
- **We immediately get 4× context (2048→8192) at zero cost.** A real structural advantage (Mamba O(1) state + attention in only 6/48 layers) that a same-size Transformer lacks.

**Honest caveat:** flat perplexity is a *necessary condition*, NOT sufficient to claim distant information is *usable*. The next step must test a **Vietnamese needle-in-a-haystack** (hide a fact at ~6000, ask at 8000, check retrieval) + try **16K/32K**. If the needle passes → a sharp long-context story (small model, long context, low cost), matching the nn-aivie3 streaming vision.

## 12. Needle-in-a-haystack (2026-09-29) — how far does long context REALLY go

`scripts/needle_haystack.py` — hides "the secret code is <word>" (5 single-token words) at various depths in a viwiki haystack, asks at the end, measures argmax over 5 choices (chance = 20%). Pretrain checkpoint step143433.

| Context | RoPE | Retrieval | |
|:---|:---|:---|:---|
| **8192** (4× training) | ntk | **20%** (chance) | ❌ can't retrieve distant facts |
| **1024** (within training) | none | **93%** | ✅ retrieves facts well |

**Conclusion (clean separation):**
- The model **HAS** needle-retrieval ability (93% at ctx 1024) — good induction, format OK. **Not a weak model.**
- Failing at 8192 is **exactly the long-context limit**: NTK/PI fix *perplexity* (§11) but do NOT by themselves give long-range *retrieval* — attention heads must **learn** to attend across new positions; compressing positions isn't enough.
- ⟹ **Flat perplexity ≠ usable long context.** A valuable lesson, measured on our own model.

**Direction (b) — verdict:** FEASIBLE but NOT free. The retrieval ability exists (93%); it just needs **length-extension fine-tuning** (continue training at 8K–16K context on long Vietnamese documents, NTK/PI enabled) so attention learns long-range lookup. Estimate: 1×H100 a few hours → 1 day, ~$30–150 (8K context uses ~4× the memory of 2048 so smaller batches, but length adaptation converges quickly in few steps). Then re-run `needle_haystack.py --len 8192` to confirm retrieval recovers.

## 13. Testing Zamba2-2.7B-Instruct (2026-09-29) — a hybrid Mamba2+Attn base ALREADY trained on ~3T tokens

`scripts/test_zamba2.py` (transformers 5.16.1, MPS fp16, `PYTORCH_ENABLE_MPS_FALLBACK=1`).
**Trap:** transformers 5.16 builds Zamba2 incorrectly with `block_id = layer_id % num_mem_blocks` (it should be the order in `hybrid_layer_ids`) → the 2.7B layout `[6,12,…,42,47,51]` gets the parity wrong → `ValueError tie_weights_keys`. The script monkey-patches `Zamba2Model.get_layers` (checkpoint: layer 6 = block 0 adapters 0,2,4,6,8; layer 12 = block 1 adapters 1,3,5,7). transformers 5.x: `apply_chat_template(..., return_dict=True)`.

| Prompt | Zamba2 | Our 1.3B |
|:---|:---|:---|
| EN knowledge (moons of Mars) | ✅ Phobos, Deimos | ❌ invents "4 moons" |
| VI knowledge (same question) | ❌ "Mars is a country… moon Hanoi" | ⚠️ fluent but fabricated |
| VI leave-request email | ❌ nonsense + `NAME_1…` | ✅ right office tone |
| EN leave-request email | ✅ polished | — |
| EN meeting summary → 3 action items | ✅ accurate, with owners | — |
| Programming (Vietnamese prompt) | ✅ Fibonacci + memo correct (answers in EN) | ❌ garbage |
| EN→VI translation | ❌ garbled | ⚠️ nearly right |
| VI→EN translation | ✅ correct | — |

**Pattern:** Zamba2 UNDERSTANDS Vietnamese but CAN'T WRITE Vietnamese; strong in English (knowledge, code, office). The two models complement each other: ours = fluent Vietnamese writing; Zamba2 = knowledge/code/EN.
**M4 speed:** 1–6 tok/s on MPS (0.1 tok/s on CPU) — Mamba2 runs the pure-torch fallback, no kernel. Mamba's performance advantage only kicks in on CUDA; on the Mac, Transformers via MLX are much faster.
**Gap to fine-tune (if Vietnamese is needed):** Vietnamese writing → LoRA SFT on `sft_mix` (85K Vietnamese conversations, already have) + optional continued pretraining on the 18.8B-token Vietnamese corpus (already on HF).

## 14. Pivot through English (approach A, zero training) — `scripts/pivot_zamba2_vi.py`

Flow: **Vietnamese question → [0] `vinai-translate-vi2en-v2` translates the question → [1] Zamba2 answers in English → [2] `vinai-translate-en2vi-v2` translates line by line** (keeping code blocks, `inline code`, list markers, **bold**). "Translate into Vietnamese: …" → straight to the translation model. M4/MPS: 14–43s per question.

| Prompt | Zamba2 asked directly in Vietnamese (§13) | One-way pivot (question not translated) | **Two-way pivot** |
|:---|:---|:---|:---|
| Moons of Mars | ❌ "country founded in 1954" | ❌ "…is NAME_1" | ✅ Phobos, Deimos |
| Mekong River | — | ✅ all 6 countries | ✅ all 6 countries |
| Who wrote The Tale of Kiều | — | ✅ Nguyễn Du | ⚠️ Nguyễn Du, but the question was translated to "Kieu Kieu" → "Kiều Kiều is a novel" |
| Leave-request email | ❌ nonsense | ⚠️ invents the reason "family reunion" | ⚠️ right idea, but machine-translated prose (literal renderings of "I hope this email finds you well", "Boss's Name", "covered by colleagues") |
| Meeting summary, 3 items | — | ⚠️ wrong details + "ship launch" | ✅ correct 3 items, correct Friday |
| Fibonacci code | ✅ (answers in EN) | ✅ | ✅ code untouched, explanation in Vietnamese |
| Direct translation | ❌ | ✅ | ✅ |

**Conclusion:** the two-way pivot turns Zamba2 (which can't write Vietnamese) into a **usable** Vietnamese assistant for knowledge/code/office, with zero training. The remaining errors are in the **translation model**: (a) literal machine-translation prose for long texts (email); (b) Vietnamese proper names get mangled ("Truyện Kiều" → "Kieu Kieu"). Fixes: include the original Vietnamese question alongside the translation in the prompt; or approach B (LoRA Zamba2 on PhoMT/MTet + pivot data) so one model does it all with more natural prose.

## 15. Pivot with Gemma 4 26B as the translator (`--translator gemma`)

`pivot_zamba2_vi.py --translator gemma`: Gemma 4 26B-A4B (Ollama `gemma4:26b-a4b-it-qat`, local) translates both directions, **translating WHOLE PASSAGES** (with context). **Trap:** Gemma 4 "thinks" first by default (~1,300 tokens, 30s per turn) → `"think": false` is mandatory (down to 1–2s, same quality). Runs **in phases** (Gemma translates the question → `keep_alive: 0` frees ~15GB → Zamba2 answers → freed → Gemma translates the answer) because the 26GB machine can't hold both. ```code``` blocks are masked as ⟦CODE_i⟧ before translation. 7 questions: 155s.

| Prompt | VinAI (§14) | **Gemma** |
|:---|:---|:---|
| Mars | ✅ | ✅ Phobos, Deimos |
| Mekong | ✅ "Mekong River" | ✅ proper Vietnamese name, all 6 countries |
| The Tale of Kiều | ⚠️ question became "Kieu Kieu" | question translated CORRECTLY as "The Tale of Kieu" but **Zamba2** answered "NAME_1" ❌ (Zamba2's fault, not the translator's) |
| Leave-request email | ⚠️ machine-translated prose | ✅ **natural, as if written by a Vietnamese speaker**; placeholder left in English `[Boss's Name]`; adds slight content not in the original |
| Meeting summary | ✅ correct 3 items | ✅ correct 3 items + owners (a well-translated question → Zamba2 follows the format better) |
| Code | ✅ | ✅ smooth explanation ("memoization" glossed correctly) |
| Direct translation | ✅ | ✅ natural phrasing |

**Conclusion:** Gemma is the best translator (natural, keeps proper names, understands context). Remaining errors are in **Zamba2's Vietnam-specific knowledge** (The Tale of Kiều → NAME_1, unstable between runs).
**Important observation:** Gemma 26B-A4B on its own is much stronger than Zamba2 and runs faster on the M4 (~69 tok/s thanks to MoE with 4B active, vs Zamba2's 1–6 tok/s without kernels). Here it's only the translator — on the M4, having Gemma answer directly in Vietnamese may beat the whole pipeline. The Zamba2 pipeline is only worth keeping if the goal is a small, fine-tunable "brain" with the hybrid architecture.

## 16. Approach B pilot — LoRA Zamba2 for Vietnamese (running, 2026-09-29)

**Goal:** 1 model (Zamba2-2.7B + LoRA) answering Vietnamese questions in Vietnamese in a single pass, B2 format: `<en>EN draft</en><vi>VI version</vi>`.
**Teacher = Gemma 4 26B** (not Zamba2): measured on the M4, Zamba2 generates 284 samples/hour, **batched generation is even slower (94 samples/hour)** — the Mamba2 fallback can't exploit batching → 2K samples ~9 hours. Gemma ~350–490 samples/hour.

| Script | Job |
|:---|:---|
| `gen_pilot_b.py` | 2,000 single-turn questions from `sft_mix` (evenly across 25 sources) + 60 held out; each Gemma call returns `<q_en>`, `<en>`, `<vi>`; writes sample by sample, re-runs resume → `data/pilot_b/distill.jsonl` |
| (PhoMT) | 1,500 pairs from `ura-hcmut/PhoMT`, shuffled with a 20K buffer → `data/pilot_b/phomt.jsonl` |
| `train_zamba2_lora.py` | B2 + 30% English rehearsal (q_en→en, against forgetting) + translation 70/30 En→Vi/Vi→En; loss on the assistant part only, truncated after `<|im_end|>`; LoRA r64/α128 on `in_proj,out_proj,linear,q/k/v/o_proj,gate_up_proj,down_proj`; `--tiny` CPU smoke |
| `setup_zamba2_box.sh` | box: transformers 5.16.1 + peft + `kernels` (prebuilt Mamba2 kernel on the Hub `kernels-community/mamba-ssm`, NOT self-built) |
| `eval_pilot_b.py` | `gen` (on the box, base vs student, 60 held out + the 7-question set) → `judge` (Mac, Gemma grades 1–5 naturalness/correctness, `format: json`) |

**New traps (transformers 5.16 + Zamba2):**
1. Training needs `use_cache=False` — forward creates a `DynamicCache(config)` without a `'mamba'` layer type → `KeyError: 'mamba'`. (Generation uses Zamba2's own cache so it doesn't hit this.)
2. `patch_zamba2_block_id()` is still needed (§13).
3. Mamba2 kernel: transformers 5.16 prefers hub kernels (`use_kernels=True`) → `mamba-ssm` package → pure torch. Pilot ~3–4M tokens: pure torch on an H100 ~20 minutes < time to build mamba-ssm → don't build.

**GPU estimate:** setup ~10 min + training ~10–30 min + eval generation ~10 min → **< 1 H100-hour (~$2)**.

## 17. Approach B pilot results (2026-09-30)

**Training** (Vast H100 `<IP>:16800`, full 700W): 3,946 samples (B2 1,994 · EN rehearsal 574 · PhoMT 1,500), `--max-len 1536`, batch 8 × accumulation 2, LoRA r64, 2 epochs = 494 steps, **14 minutes**, ~5,000 tok/s. Val loss (122 samples) **1.209 → 0.632**, decreasing steadily, no memorisation yet. 469MB adapter in `checkpoints/zamba2_vi_lora/` on the Mac (git-ignored).
The environment cost ~1 hour due to 4 traps (NGC `PIP_CONSTRAINT`; pure-torch Mamba2 OOM at 40GB; broken hub kernels; had to install mamba-ssm Triton + build causal-conv1d) — packaged into `setup_zamba2_box.sh`. Total box time ~1.5 hours (~$3).

**Evaluation** (`eval_pilot_b.py`: generation on the box ~3 min per model, Gemma grading on the Mac), 60 held out + the 7-question set:

| | Vietnamese naturalness | Correct/useful | Samples with vi=1 | Samples ok≥4 |
|:---|:---|:---|:---|:---|
| Zamba2 base | 1.63 | 1.34 | 53/67 | 4/67 |
| **+ LoRA pilot B** | **2.63** | **1.93** | 28/67 | 11/67 |

Improved: code/SQL, translation, general questions. Still weak: law, health, everyday conversation, Vietnam-specific knowledge; words used with the wrong meaning, invented terms, long answers cut off (B2 writes EN+VI so twice the length). 7-question set: code ✅, Mekong ✅ (wrong name for Cambodia), email ⚠️, The Tale of Kiều ⚠️ (adds inventions), Mars ❌ (the EN draft drifted to "venomous snakes"), meeting summary ❌.

**Conclusion:** B has a real effect (Vietnamese goes from nonsense → readable) but is far from usable and well behind the Zamba2 + Gemma translation pipeline. **Bottleneck: Zamba2 has almost no Vietnamese foundation from pretraining** — 2K SFT samples can only teach format/tone, not the language. Next step if pursuing B: continued pretraining on 1–2B tokens of the 18.8B Vietnamese corpus (HF) before SFT (~$50–100), 10× SFT data (generated with Gemma on GPU for speed), consider B1 (Vietnamese only) to avoid truncation. Note `format_ok` counts direct translation requests (which answer directly, without tags) as "wrong" — so this metric is slightly lower than reality.

## 18. Comparison with Gemma 4 26B answering directly + an independent judge (2026-09-30)

`eval_pilot_b.py gen-ollama --tag gemma` (Gemma answers the 67 questions directly on the M4) + `judge --judge-model <ollama>`. All 3 sets re-graded with **SeaLLM-7B-v2.5** (independent, not the teacher) to check self-grading bias.

| Approach | Gemma judge (naturalness / correctness) | SeaLLM judge (naturalness / correctness) |
|:---|:---|:---|
| Zamba2 base | 1.63 / 1.34 | 2.43 / 1.85 |
| Zamba2 + LoRA pilot B | 2.63 / 1.93 | 3.16 / 2.61 |
| **Gemma 26B answering directly** | 5.00 / 4.57 | **3.94 / 3.51** |

**Conclusions:**
- The **ranking holds under both judges**: Gemma > pilot B > Zamba2 base → a robust conclusion.
- Gemma gives itself a **perfect 5.00** = clear self-grading bias. With an independent judge, the Gemma − pilot B gap is ~0.8–0.9 points (not ~2.5 as Gemma self-grades). SeaLLM "compresses the scale" more (lenient with poor answers, strict with good ones).
- Pilot B sits **in between** Zamba2 base and Gemma. Speed: Gemma on the M4 ~15–40s per question (long answers ~2,100 characters; one question took 28 minutes because the lid was closed and the machine slept); Zamba2 on the M4 is much slower (1–6 tok/s).
- **For practical use on the Mac: Gemma 26B answering directly is the best choice**, $0, no pipeline needed. Route B is only worth pursuing if the goal is a small autonomous model running on GPU/CUDA, and it must get Vietnamese continued pretraining first.

## 19. VMLU of the original Zamba2-2.7B-Instruct — stopped midway (2026-09-30)

`eval_vmlu.py --backend hf-ll --model Zyphra/Zamba2-2.7B-instruct --shuffle` (letter log-likelihood, question wrapped in the chat template; each letter A–E is exactly 1 token). M4 ~16s per question → stopped at **150/744 questions: 29.3%** (at 50: 26.0% · 100: 30.0%) to free the GPU for the chat UI. The order was shuffled (seed 0), so it's an unbiased estimate over all subjects, error ±~7 points (95%).
→ The original Zamba2 asked knowledge questions in Vietnamese is **only slightly better than guessing (25%)** even though its English is at the Gemma-2-2B level: the knowledge hasn't transferred to Vietnamese → Vietnamese continued pretraining has a lot of headroom (estimate ~38–45% after CPT+SFT). For an exact number: re-run on a CUDA box (a few minutes) instead of the M4.

## 20. The 1.3B's assistant starting point + discovering EVALUATION DATA CONTAMINATION (2026-09-30)

`eval_pilot_b.py gen-http --tag ours13b` (calls the UI server, temperature 0, rep-pen 1.1, ≤600 tokens; ~8s per question thanks to step-by-step generation) → graded by Gemma + SeaLLM.
Raw scores over 67 questions: Gemma 3.99/2.97 · SeaLLM 3.44/2.91 (naturalness/correctness) — higher even than Zamba2+LoRA. **BUT inflated:** the 60 held-out questions were taken from `sft_mix` = the 1.3B's own SFT data. 21/60 answers almost copy the reference answer verbatim (similarity ≥0.8; control Gemma: 0 answers ≥0.5).

| Model | 7 clean questions (SeaLLM) | 36 non-copied questions (SeaLLM) |
|:---|:---|:---|
| Zamba2 base | 2.29 / 1.71 | 2.31 / 1.64 |
| Zamba2 + LoRA | 3.00 / 2.57 | 2.86 / 2.33 |
| **Our 1.3B** | **3.29 / 2.57** | **3.31 / 2.89** |
| Gemma 26B | 4.00 / 3.29 | 3.94 / 3.42 |

**Conclusion:** the 1.3B's Vietnamese is much more natural than Zamba2's (as expected: a foundation of 18.8B Vietnamese tokens); correctness on the 7 clean questions matches Zamba2+LoRA, clearly below Gemma. The 36 "non-copied" questions are still in the SFT data → still partly contaminated. **The current evaluation set is NOT usable for the 1.3B.** Mandatory next step: build a new clean evaluation set (questions NOT in `sft_mix`, near-duplicate checked), including questions with verifiable answers to measure the fabrication rate.

## 21. The 1.3B's REAL starting point on the clean evaluation set eval_v1 (2026-09-30)

`eval_clean.py` on `data/eval_clean/eval_v1.jsonl` (158 questions, SFT overlap ≤0.33). The 1.3B via the UI server (temperature 0, ≤500 tokens); Gemma 26B answering directly (≤600 tokens).

| | Open: naturalness / correctness (Gemma · SeaLLM) | Facts correct (40) | Non-existent: admits not knowing (20) | Math correct (18) |
|:---|:---|:---|:---|:---|
| **Our 1.3B** | 4.29 / 2.04 · 3.61 / 2.71 | **2–5%** (~95% fabricated, never says it doesn't know) | **0%** (100% fabricated) | **0/18** |
| Gemma 26B | 5.00 / 4.60 · 4.00 / 3.38 | 30–35% (~58% fabricated — the viwiki facts are fairly obscure) | ~80% | 17–18/18 |

The 1.3B's open-question "correctness" by group (Gemma): office 2.5 · translation 2.4 · knowledge 2.3 · law 2.3 · daily life 2.2 · reasoning 2.0 · summarisation 1.6 · **programming 1.0**.

**Checking the judges against objective measures** (reference answer contained in the response; final numeric answer; "don't know" phrases): matches the **Gemma** judge (1.3B: facts 2/40, don't-know 0/20, math 0/18). **SeaLLM is far too lenient on the classification part** (says the 1.3B admits not knowing 90%, math 22%) → use SeaLLM only as a second opinion for OPEN questions; for questions with answers use Gemma + objective checks.

**Conclusion — a clear improvement priority order:**
1. **Honesty**: the 1.3B never says "I don't know" (0/20 fabricated entities) → teaching it to admit not knowing (SFT/DPO) is job #1, and easy to measure.
2. **Reasoning/math**: 0/18, no calculation at all → an extremely low baseline; any improvement (teaching scratch work, majority voting, GRPO) is immediately measurable.
3. **Programming**: 1.0/5 → the pretraining corpus has almost no code.
4. Natural Vietnamese is a real strength (4.29 per Gemma — higher than every model except Gemma).

**Set errors to fix in v2:** a few "non-existent" questions are actually real or ambiguous (saltwater batteries, filmless projectors, the common name "Nguyễn Ngọc Diệp"); the viwiki facts are somewhat obscure (Gemma only 35%).
