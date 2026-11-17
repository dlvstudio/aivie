# TK_107 — Growing the 1.3B trunk → ~2.6B (~3.4B total) · context 2K → 8K · WSD roadmap 20 → 50 → 100B tokens

*English translation of the Vietnamese original in the repository root.*

User decision (2026-10-06): move to ~3B + long context; **option A — staged** (a 20B-token milestone, evaluate, then
decide whether to extend to 50/100B). V5 image reading is paused (it must be retrained on the new trunk; the V5 data is kept for re-use).

## 1. Architecture

| | Current (v5) | Target |
|---|---|---|
| Trunk | 1.277B · d=2048 · 48 blocks (42 Mamba + 6 attention @7,15,…,47) | **2.488B** · d=2048 · **96 blocks (84 Mamba + 12 attention @7,14,23,30,…,94)** |
| Context | 2,048 · RoPE base 1e4 | **8,192** · RoPE base **1e6** (ABF) |
| Tokenizer | bpe_v1 (32,002) | unchanged |

**Identity block insertion** (`scripts/expand_depth.py`, LLaMA-Pro/SOLAR style): old block i → position 2i; the new block at 2i+1 copies the weights
of the nearest old block of the same type, then **sets its output layer = 0** (Mamba `out_proj`, attention `wo`) → `x + block(x) = x`.
A new block is attention when i ≡ 3 (mod 8) (inserted between 2 old attention blocks). **Checked on the real trunk (chat_v2): logit deviation 0.00 ·
argmax 48/48** → continued training starts with no loss shock. New cfg: `layers=96, attn_layers=[…], rope_base, new_blocks=[1,3,…,95]`.

## 2. Trainer (`scripts/train_hybrid_scale.py`, upgraded)

- **Multi-GPU**: `torchrun --standalone --nproc_per_node N` (DDP, `no_sync` during gradient accumulation, rank 0 logs/saves/evaluates).
- **WSD schedule** (`--sched wsd --decay-frac 0.1`): warm-up → hold lr → linear decay at the end; saves `*_stable.pt` (with optimizer)
  right before the decay → **extending** = `--init-from *_stable.pt --tokens <larger>`.
- `--train-new-only` (only new blocks learn), `--reset-step` (new phase / different parameter set → fresh optimizer), `--rope-base`,
  `--data "dirA:0.5,dirB:0.3,…"` (weighted group mixing), builds the model from the ckpt's cfg (`build_from_cfg`).
- **Bug caught while testing**: cfg saved `rope_base=0` (a default argument overwrote the real value → RoPE gives inf/nan) — fixed.
- Tested on the M4 (tiny model): train from scratch → insert blocks (match 0.00) → 2-process DDP training only new blocks (old blocks change 0 params)
  → stable checkpoint at the right step → extend from the stable checkpoint → unfreeze everything → group mixing at 1K context: **all work correctly**.
- `hybrid_infer.py`, `vlm_1p3b.py` load the new config too (per-block rope_base). ~~Remaining (P3): experts/LoRA/image reading hard-code
  `POS`~~ → fixed (§6).

## 3. Data (P1)

- **Educational-value classifier** (`scripts/edu_classifier.py`): Gemma-31B scores 40K Vietnamese web passages (additive 0–5 scale like
  FineWeb-Edu, adapted to Vietnamese) → a fastText-style model (bag of words + hashed bigrams, pure PyTorch) → keep scores ≥ 2.5.
- **Shard building** (`scripts/build_cpt_data.py`): stream from HF → normalise **keeping line breaks** → heuristic filters (the old version joined all
  lines, so the "repeated lines" filter did nothing) → educational scoring → exact de-duplication → parallel tokenisation (~1M tokens/s per core,
  measured) → texts ≥4K tokens go to the `long` group; one process per source (shards `shard_<source>_<n>.npy`).

| Group | Sources (access verified) | Milestone 1 |
|---|---|---|
| vi_web | FineWeb-2 vie (ODC-BY) · HPLT 2.0 vie (CC0) · [CulturaX: gated, needs user login] | 9B |
| long | texts ≥4K tokens from web/wiki | 2.4B |
| vi_wiki | Wikipedia vi · Wikisource vi | 0.8B |
| en_edu | FineWeb-Edu sample-100BT | 3B |
| math | FineMath-4+ | 1B |
| code | github-code-clean (permissive licenses; Stack-Edu only has blob ids, no contents) | 1.5B |
| replay | old pretrain_vi (18.8B, HF `nn-aivie-pretrain-vi`) | mixed at 9–12% |

## 4. Milestone 1 roadmap (~20B tokens) — `scripts/box_run_p2_cpt.sh` (8×H100)

| Phase | Trains | Context / RoPE | Tokens | Schedule |
|---|---|---|---|---|
| A1 | new blocks only | 2K / 1e4 | 4B | lr held at 3e-4 |
| A2 | new blocks only | 8K / **1e6** | 6B | lr held at 3e-4 |
| B | everything | 8K / 1e6 | 10B | 1.5e-4, decay over the last 10% → `cpt_b_stable.pt` for extension |

Global batch ~1M tokens/step; save every 500 steps, **HF backup every 1,000 steps** (`--hf-repo`). At the start the script checks the block insertion
matches and quickly measures speed/memory at 2K and 8K. At the end: `eval_longctx.py` (loss by position + Vietnamese "needle" 1K–8K) 1.3B vs 2.6B.

After milestone 1 (P3, 1×H100 ~1.5–2 days): chat SFT, retrain the 3 experts, teacher loop, image reading (V4 recipe + V5 data, 12 native tiles).

## 6. P3 — code for the 96-block trunk (2026-10-11, done on the Mac before renting a box)

- **Expert insertion points = the 12 attention blocks** `[7,14,23,30,39,46,55,62,71,78,87,94]` (user chose 12, not 6), each FFN
  **half-size** → total expert parameters unchanged (3 sets = 755M). `expert_ffn_1p3b.set_pos(cfg)` modifies `POS` IN PLACE (every
  `from … import POS` sees the same list); `hid_eff()` converts `--hid` (defined for 6 points) to the real size. The old 1.3B trunk still gets 6 points at
  the old size → v5 / vlm_v4 still load in the UI.
- Every place that loads a trunk goes through `build_from_cfg` (experts, VLM, SFT). **Dangerous bug caught**: `train_hybrid_sft.py` built
  `HybridLM` without passing `rope_base` → SFT of the 8K trunk ran with RoPE 1e4, silently breaking long context; and it read `attn_layers`
  as a string (the new cfg has a list) → crash.
- SFT: `--tok-budget` (micro-batch ≤ token budget after padding, keeps the longest samples), `--optim adamw8bit`, `--device`.
- Data `build_sft_p3.py` (version 2 — the new trunk has NOT been through any SFT, so it combines ALL of the 1.3B's SFT rounds): sft_mix 85,174 +
  the sft_v2-specific part 15,168 (source-grounded, don't-know, math) + Vietnamese DailyDialog 10,323 + multiturn 8,665 = 119,330 short
  (50.9M tokens, p50 256) + **6,000 long samples of 2.6K–8.2K** (4.8%, 33.7M tokens) → ~85M tokens/epoch, 2 epochs of SFT. Long samples are generated from viwiki:
  bundle several articles and ask "What is X?" (→ "According to document k, …"), "Which document talks about X?", and X not in the bundle
  (→ "no information", teaching the model to admit it doesn't know).
- `smoke_p3.py` on the M4 with `hybrid_2p6b_a1_bf16.pt`: correct insertion points · 3 new expert sets match the trunk at 0.00 · training only experts
  leaves the trunk unchanged · step-by-step Runner with experts matches the forward (fp32, 7e-7, 39/39) · tiny SFT (RoPE 1e6, long samples) loss decreases.
  **Mac trap**: pure-PyTorch Mamba + 8K sequences takes a whole minute per step; MPS running out of memory does NOT raise an error but gives a fake loss of 0.000.
- `box_run_p3.sh` (1×H100): smoke → SFT (falls back to 8-bit AdamW on OOM) → 3 new experts (`trainGEN`, 3,000 steps) → evaluation
  (eval_v1, easy20, office, textbooks, math, long context) vs v5. `.done` markers per step, HF backup `nn-aivie-2p6b-p3`.
  `pack_p3_data.sh` packs the data on the Mac → HF dataset (the box downloads it). The teacher loop (P3.3) and image reading (P3.4) run afterwards.
- Old data lost on the Mac: `data/sft_chat/chat_mix.jsonl` (prep now takes chat from sft_v2), `easy20.jsonl` (rebuilt from
  the v5 grading results), `data/vlm` (rebuilt with `gen_vlm_data.py` on the box when P3.4 comes).

## 5. What the user must do before running

1. **P1**: rent a **1×H100 box, disk ≥500GB, ≥32 CPU cores**; `hf auth login` (WRITE token, pasted on the box by the user) → `scripts/box_run_p1_data.sh`.
2. **P2**: rent **8×H100 SXM, disk ≥500GB** (check it isn't power-capped: `nvidia-smi -q -d POWER`); `hf auth login` →
   `scripts/box_run_p2_cpt.sh`. Milestone 1 estimate: ~3.5 days, ~$1,700–2,200 (rental prices vary).
3. (Optional) accept the CulturaX terms on HF to add that Vietnamese source.

## 7. The 20B milestone + P3 re-ordering (2026-10-12)

- **P2 DONE** 2026-10-11 23:39 UTC. Fixed val (bits, 2K): trunk after insertion 3.47 → A1 2.73 → A2 2.67 → **B 2.31** (every group improves; code
  3.69→0.98, math 3.77→1.88, Vietnamese web 3.05→2.21). 8K: A2 2.60 → B 2.22. Vietnamese needle 4K **20/24**, 8K **14/24** (1.3B: 0/24, 0/24).
  44-question multiple choice: 1.3B 36 → 2.5B 38. User: stop at 20B (extend later from `cpt_b_stable.pt` if needed). A1/A2 deleted on the box.
- **Image-reading data lost** with the TK_106 box (the old script didn't back up the data) → rebuilt on the P2 box (`box_vlm_data.sh`: A = CPU
  while training, B = 3 Gemma teachers on GPUs 1–3, image viewing split into 3 parts by crc32) + `box_vlm_code.sh` (code images). Pushed to the public HF repo
  `nn-aivie-vlm-data` (excluding original images from external sets; news pages embedding COCO images → only if the user agrees).
- **Code images (user request)**: `gen_screens.py` adds code with 1 injected logic bug (`mutate`), an editor + TERMINAL with a traceback pointing to the RIGHT line
  (error matching the line's content: division by a variable → ZeroDivisionError, string key → KeyError…), photographed screens (`--photo-frac`, with a margin
  so text isn't cropped), and saves the exact code (`ref`) for the grader. Tested on the M4: the V4 1.3B reads "line 3" as line 2 and invents a ValueError.
- **Re-ordering (user)**: P3.1 SFT → P3.2 experts → P3.5 evaluation → **P3.4 image reading** → **P3.3' JOINT text + image teacher loop** (`vlm_loop.py`:
  ~30% of conversations with images, 35% of those code; the teacher looks at the image + notes the exact content, grades text/numbers/code strictly, writes a correction)
  → buildV5/trainGEN/DPO for the text experts + `fix_<tag>.jsonl` to continue training image reading. Reason: the teacher loop fixes the experts AFTER image reading
  exists → no drift, no retraining needed; and image reading is also graded and corrected by the teacher (previously only teacher-written data + evaluation grading).
