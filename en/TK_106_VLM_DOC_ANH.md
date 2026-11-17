# TK_106 — Our model READS IMAGES (VLM, Gemma 26B/31B style)

*English translation of the Vietnamese original in the repository root.*

Goal: **a single checkpoint** — chat can attach images (photos, screenshots of Excel/Word/email/code,
charts), with images and text interleaved in the same sequence, over multiple turns. No external model is bolted on as the "brain": understanding + speaking Vietnamese
remain our 1.3B trunk + experts; only the "eyes" (SigLIP2) are an open image encoder (training our own eyes would need billions of image–text pairs).

## 1. Architecture (`scripts/vlm_1p3b.py`)

```
Image ─► split into 384px tiles (grid by aspect ratio, ≤12 tiles; multi-tile images + 1 downscaled overview)
      ─► SigLIP2 so400m-patch14-384 (428M, frozen; 27×27 patches/tile)
      ─► pixel-shuffle 3×3 (729 → 81 tokens/tile, 10,368 dims) ─► MLP (LN → 4096 → 2048, 51M)
      ─► RMSNorm × emb_rms × gain  ⇒ image tokens THE SAME SIZE as text tokens (TK_105 lesson: soft tokens blowing up ×3.4 → broken)
Sequence: "Hỏi: " <image> [overview 81] (<row> [tile]…)… </image> "\n" question "\n" "Đáp: " …   (id −1 = image-token slot)
1.3B trunk + code/office/academic experts (frozen) + VISION EXPERT (GatedFFN hid 4096, gate init −2, output layer = 0)
```

| Image | Grid (≤12 tiles) | Image tokens |
|---|---|---|
| 384×384 / small image | 1×1 | 83 |
| 390×844 (phone) | 1×2 | 247 |
| 1440×900 | 3×2 | 571 |
| 1920×1080 | 4×2 (≤16 tiles: 5×3) | 733 (1,301) |

Checkpoint: adds a `"vision"` entry (bf16 eyes + config + projector + max_tiles) and `experts["vision"]`.
`hybrid_infer.Runner.prefill/generate(vis=…)` replaces the negative ids with image vectors; UI (`serve_hybrid_ui.py` +
`hybrid_ui.html`): 📎 button / paste / drag-and-drop images (≤4 images per turn, long side ≤2048px, PNG keeps text sharp), image tokens cached by
sha1, later turns re-use the conversation state (the eyes are not re-run). Original images are kept only in page memory (localStorage
stores only thumbnails for display).

The eyes download only the vision tower via HTTP Range (`scripts/fetch_vision_tower.py`): 1.71GB instead of the whole 4.54GB file → 0.86GB in bf16.

## 2. V0 on the M4 — RESULTS (2026-10-05)

- **smoke**: image-token rms 0.0242 = text-token 0.0242 ✓; eyes 0.2–1.5s per image (1–9 tiles); prefill 744 tokens + generate 40 tokens in 4.4s.
  Untrained → the model "invents" descriptions (a bicycle…) — as expected.
- **overfit** (projector only, 8 self-drawn images of text + background colour, 96 steps, ~10s/step on MPS): loss 2.92 → 0.004;
  **8/8 descriptions verbatim correct** ("The image has the word "Toán" in white on a green background.") → gradients flow correctly through the whole pipeline.
- **UI**: packed into 1 checkpoint (7.65GB) → send an image through the interface, correct answer, ~19–21 tok/s, turn 2 re-uses the state.
  (The toy V0 checkpoint was deleted; the `hybrid-ui-vlm` config already points at `hybrid_1p3b_vlm_v1.pt`.)

## 3. V1 — data (box; the Gemma-4-31B FP8 teacher CAN READ IMAGES: `Gemma4ForConditionalGeneration`, has `vision_config`)

| Type | Source | Verification |
|---|---|---|
| **Self-generated screenshots** (`scripts/gen_screens.py`) | Excel (sales/payroll/inventory, selected cell + formula), mailbox (emails from data/office), Word (textbook lessons), VS Code (code_tb), SVG bar charts, chat app (multiturn); many fonts/sizes/light-dark/widths | **Exact answers** (read a cell, formula, column total, sender, line N, column value…); questions needing understanding (summarise, draft a reply, explain code) → `teach`: the teacher writes from the ORIGINAL TEXT |
| Natural image descriptions | COCO / LLaVA-Pretrain images → Gemma looks at the image and writes a Vietnamese description | Gemma self-grades |
| Vietnamese image QA | ViVQA, OpenViVQA, UIT-ViIC, KTVIC | answers provided |
| Charts/documents | ChartQA, DocVQA (translated) + self-drawn charts | original figures |
| Multi-turn chat with images | Gemma plays the user asking about the image (like teacher loop v5) | graded 0–10 + corrected |

The box needs: `pip install playwright && playwright install --with-deps chromium` + Vietnamese fonts (`fonts-noto`).

## 4. V2 — training (1 H100)

| Stage | Learns | Data | Estimate |
|---|---|---|---|
| 1 alignment | projector | ~500K images + descriptions | ~4h |
| 2 Q&A | projector + vision expert (+ `--tower-last 6` last layers of the eyes) | ~300K conversations with images (screens, descriptions, VQA, charts) | ~15–20h |
| 3 teacher loop | SFT + DPO on Gemma's corrections (looking at the image) | ~5K turns | ~6–10h |

Evaluation: ViVQA, exact reading of text/numbers on held-out screenshots (string match), chart figures, Gemma-graded conversations; plus the old
text benchmarks (textbooks, math, office, 5-turn chat) to make sure nothing is forgotten.

When the trunk changes (3B, direction B) only the projector + vision expert need retraining (~1 day); the V1 data is re-used as-is.

## 5. V1 + V2 — RESULTS (2026-10-05/06, 1 H100)

**V1 data** (`box_run_vlm_v1b.sh`, ~4h, mostly the Gemma-31B teacher generating ~1.5K tokens/s):
44,193 self-generated screenshots (+16,804 understanding questions filled in by the teacher) · 32,170 natural images described by the teacher (COCO 20K, KTVIC 2.8K,
OpenViVQA 10.2K) · 24,093 translated the_cauldron samples (chartqa, textvqa, st_vqa, docvqa, scienceqa, ai2d, tabmwp, infographic,
screen2words) · 3,768 Vietnamese charts → **align 69,203 · sft 128,496 (12,500 text-only)** · test: nat 1,620, screen 1,000,
chart 424, ovqa 1,070.

**V2 training** (trunk + 3 experts frozen): stage 1 (projector, 1,081 steps × 64, ~40 min) val 2.71 → 1.68;
stage 2 (projector + vision expert + last 6 eye layers, 4,015 steps × 32, 6K-token budget per batch, ~3.6h) val 1.27 → **0.72**.

**Evaluation (200 questions per set, teacher-graded against reference answers):**

| Set | Correct | Partial | Value match | Notes |
|---|---|---|---|---|
| Vietnamese charts (vichart) | **38%** | 8% | 13% | trends/comparisons good, specific numbers poor |
| OpenViVQA (Vietnamese images with text) | 16% | 3% | 7% | misreads signs/numbers |
| Self-generated screenshots | 7.5% | 8.5% | 12.9% | copy similarity 0.205; Excel row counting correct 8/22 |
| Natural images (detailed description) | 0.5% | 19.5% | — | gets the main scene right, wrong details (colours, counts) → harsh teacher grading |
| Text-only textbooks: with vision expert | 40.7% | 18.7% | | **no forgetting** |
| Text-only textbooks: original v5 | 39.3% | 19.3% | | |

Diagnosis: the model **sees the layout + large text** (chart titles, number of bars, STOP signs, main scene) but **cannot yet read small text/numbers**
(bar values, Excel cells, code lines → invents from memory). Hypotheses: (a) not enough text-reading data (~60K samples × 1 pass; real VLMs
use millions), (b) 3×3 patch-per-token compression (42×42px) loses fine text. → V3 diagnostic: +40K synthetic text images
(`gen_ocr.py`, PIL) mixed 50/50, continue 1,500 steps, re-evaluate.

**Traps hit (patched into the scripts):**
- The container has only 1,792 threads (`pids.max`): 25 Chromes plus vLLM → vLLM crashes with "Resource temporarily unavailable", SSH flaky.
- HF single-stream downloads ~1.5MB/s; downloading 4 sources in parallel is much faster (COCO 60 images/s).
- 2 overlapping translation processes → 576 duplicate samples → `build` de-duplicates by id.
- Disk full when saving step 500 (120GB disk, 6.6GB assistant checkpoint + test files) → save only the trained eye layers, write `.tmp` then rename.
- vLLM at 0.45 memory leaves too little KV room for Gemma-31B (needs ≥6.9GB) → evaluation split into 2 phases: generate (6 processes, vLLM off) then grade.
- `--device` defaults to mps/cpu → on the box cuda must be selected explicitly.

**M4:** `checkpoints/hf_hybrid/hybrid_1p3b_vlm_v1.pt` (7.85GB, 4 experts) · UI `hybrid-ui-vlm` on port 8796 (~19 tok/s;
4–6s wait for multi-tile images).

## 6. V3 diagnostic + V4 architecture fix — RESULTS (2026-10-06)

**V3** (+40K synthetic text images `gen_ocr.py`, continue 1,500 steps, V1 architecture): screenshots 7.5→9%, value match 12.9→12.9%,
copy similarity 0.205→0.178 — **more data DID NOT help** → the bottleneck is the architecture (the model recognises the topic and then writes from memory).

**V4** = **2×2** merging (pad 27→28 → 196 tokens/tile instead of 81) · drop the overview image when there are multiple tiles · ≤8 tiles (1080p → 4×2 = 1,572 tokens) ·
**LoRA r64 on wq/wk/wv/wo of the 6 attention blocks** (the trunk learns to COPY from image tokens) · V1 data + 40K text images (168,496 samples).
Stage 1: val 2.71→1.58 (V1 1.68) · stage 2 (3,500 steps × 32, ~4.2h): val 1.40→0.61.

| Set (200 questions) | V1 | **V4** |
|---|---|---|
| Screenshots · correct | 7.5% | **38.5%** |
| · value match | 12.9% | **41.9%** |
| · copy similarity | 0.205 | **0.557** |
| OpenViVQA · correct | 16% | **28.5%** |
| Natural images · correct / partial | 0.5 / 19.5% | 1 / **65%** |
| Vietnamese charts · correct | 38% | 32% |
| Text textbooks (150): LoRA + expert / expert only / original v5 | 40.7 (V1) | 36.0 / 38.0 / 39.3 |

By screenshot type (correct/total, V1 → V4): email 2→19/29 · chat 0→10/18 · document 1→6/15 · Excel 12→23/52 · chart 0→13/55 ·
code 0→6/31. Still weak: small numbers on chart bars, specific Excel cell values, line N of code.

**One checkpoint, no text forgetting:** `hybrid_1p3b_vlm_v4.pt` keeps the LoRA SEPARATE (`vision.lora`, not merged into the trunk) + the vision expert wrapped in
`Toggle` → the UI only enables both when the conversation HAS IMAGES; text chat runs exactly like v5 (Ohm's law, Điện Biên Phủ answered exactly as v5).
Trap: `use_gate` MUST be set before wrapping in Toggle (setting it on the wrapper → the expert runs ungated → garbage output).
