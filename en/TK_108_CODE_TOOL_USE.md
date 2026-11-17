# TK_108 — aivie writes code and uses tools (code + tool use)

*English translation of the Vietnamese original in the repository root.*

*Written 2026-10-13. Follows TK_107 (2.5B trunk / 8K, 20B-token milestone, P3 running).*

> **Latest milestone (10/22): cx3b** — a large 1.21B code expert: tool use 55%, HumanEval 13.4%, MBPP 23.3%. Before TK_108 these numbers were 0 · 6.1 · 5.1.
> Architecture: see [TK_108_KIEN_TRUC_CX3B.md](TK_108_KIEN_TRUC_CX3B.md); the process: see §9–§13.

## 0. Goal and current baseline

**Goal:** aivie writes correct code, **runs it itself – reads the error – fixes it**, and **works in a project directory with tools** (reading/writing files, running
tests, searching), explaining code in Vietnamese with correct terminology.

**Baseline measured on the Mac (10/13, harness `scripts/code_eval.py`, 40 HumanEval problems + 20 tool-use tasks):**

| Model | HumanEval single pass | After ≤2 self-fix rounds | Tool use (20 tasks) |
|---|---|---|---|
| 1.3B v5 (with code expert) | 17.5% | 17.5% | 0% — 86% of turns call no tool |
| 2.5B SFT step 2,500 (no expert yet) | 0% — copies the docstring then explains, never writes the function body | 5% | measuring |

**Measurable targets (estimates, to be adjusted once real numbers exist):** HumanEval ≥ 30% single pass, ≥ 40% after self-fixing; tool use ≥ 50% of tasks succeed;
MBPP ≥ 35%. A 2.5B model has a ceiling: the best small code models are trained on trillions of tokens, we have 20B → the lever is **high-quality
verified data + RL from tests + the tool loop**, not token volume.

## 1. Already done (C0)

- `scripts/code_agent.py` — a tool-enabled runner: **sandbox** (macOS `sandbox-exec` / Linux `bwrap`: no network, no writes outside the working
  directory, reads of `~/.ssh`/the HF token blocked; CPU/memory/time/output limits), **debug** mode (write → run public tests →
  feed errors back → fix) and **tool** mode (protocol `<tool>{"name","args"}</tool>` → `<obs>…</obs>`, finishing with `<done>…</done>`),
  **per-turn trajectory logging**. **A Claude Code / Cursor-style toolset** (upgraded 10/13):

  | Tool | Parameters | Behaviour |
  |---|---|---|
  | `list_dir` | `path` | list a directory |
  | `glob` | `pattern` | find files by name pattern (`**/*.py`) |
  | `grep` | `pattern` (regex), `path`, `glob`, `context`, `ignore_case` | search in files, with context lines, at most 80 matching lines |
  | `search_code` | `query` (natural language) | find relevant functions/classes (BM25 over def/class blocks) — like Cursor's "codebase search" |
  | `read_file` | `path`, `offset`, `limit` | read in CHUNKS, with line numbers, pointing to where to continue |
  | `edit_file` | `path`, `old`, `new`, `all` | replace an `old` snippet that must be UNIQUE (like Claude Code's `Edit`); **must `read_file` first**; if not found → suggests the closest matching line; returns the edited region |
  | `write_file` | `path`, `content` | create a new file (existing files must be read first) |
  | `delete_file` | `path` | delete a file |
  | `bash` | `command`, `timeout` | shell inside the sandbox, NO network |
  | `run_tests` | — | the project's test command |
  | `lint` | `path` | syntax errors + pyflakes (undefined names, unused imports) |

  Checked: every tool + error paths (editing without reading, non-unique `old`, whitespace mismatch, writing outside the project, bash without network);
  a scripted fake model following "glob → run tests → read → edit_file on the right line → lint → test" → **40/40**.
- `scripts/code_eval.py` — HumanEval 164 (public tests = the `>>>` examples the canonical solution passes: 69 problems), MBPP sanitized test 257,
  **40 tool-use tasks** built from MBPP train/validation (20 fixing planted bugs + 20 writing function bodies; checked: the original solution passes, the handed-out task fails;
  a fake model writing the correct solution → 40/40).
- Traps caught: Python 3.12 `-I` prevents tests from importing `src.py`; the `-E` flag ignores `PYTHONDONTWRITEBYTECODE` → stale `.pyc` re-used
  after a file edit (same size, same second) → wrong grading → use `-B`; doctests glue `"""` onto the expected output.
- Code-image data (TK_107): 20K code/terminal images for training + 3K for the teacher loop + 300 test, already on HF.

## 2. Data — sources, licenses, how to get them (checked on HF on 10/13)

### 2.1 Open-source code (mid-training)

| Source | Content | License / access | Content available? | Use |
|---|---|---|---|---|
| **Stack-Edu** (`HuggingFaceTB/stack-edu`) | GitHub code scored for "educational value", many languages, with `detected_licenses` | not gated; each file under its original license | **NO** — only `blob_id`; content downloads from Software Heritage S3 (needs an AWS account) | main source, filter to permissive licenses |
| **The Stack v2 / dedup** | the largest open code corpus, with an opt-out mechanism | auto-gated (accept terms) | NO — as above | backup |
| **The Stack v1 dedup** | older version | auto-gated | **YES** | backup if S3 is a problem |
| `codeparrot/github-code-clean` | GitHub code in 30+ languages, basic filtering | Apache-2.0 (dataset); each file carries its license | **YES** | fast source, no AWS needed |
| `OpenCoder-LLM/opc-annealing-corpus` | `algorithmic_corpus` · `synthetic_code_snippet` · `synthetic_qa` (used for OpenCoder's LR-decay phase) | ODC-By | YES | **the "annealing" phase** |
| `OpenCoder-LLM/opc-fineweb-code-corpus` | code-related web pages filtered from FineWeb | MIT | YES | technical prose |
| **CommitPackFT** (`bigcode/commitpackft`) | real commits: code before/after + change description | MIT | YES (load files directly — the old loader script doesn't work with new `datasets`) | **bug fixing / code editing** |
| Notebooks (`codeparrot/github-jupyter-*`) | code interleaved with explanations + outputs | Apache-2.0 | YES | "write → run → read output" |
| `codeparrot-clean` | Python | — | YES | **already used in P1** (~8% of the 20B) |

**Licenses:** keep only MIT / Apache / BSD / ISC / Unlicense / CC0 files (per `detected_licenses` or the file's license field). GPL/
AGPL/LGPL and "unknown" → **excluded** (aivie is a commercial product).

### 2.2 Open books and documentation

| Source | License | Commercial |
|---|---|---|
| Python docs (docs.python.org), Rust Book, Go docs, PostgreSQL docs, Node.js docs | PSF / MIT·Apache / BSD / PostgreSQL / MIT | ✅ |
| MDN Web Docs | CC BY-SA 2.5 | ✅ (with attribution) |
| SICP, Dive Into Python 3, Open Data Structures | CC BY-SA | ✅ |
| Computer-science Wikipedia (vi + en) | CC BY-SA | ✅ |
| Think Python, Eloquent JavaScript, Pro Git | CC BY-**NC** | ❌ research only — NOT in the commercial version |
| Vietnamese university textbooks, VNOI wiki, Vietnamese tech blogs | check each source | varies |

Estimated 100–200M tokens. Vietnamese material is scarce → **the teacher rewrites/translates** open books into Vietnamese (code kept unchanged) + **"synthetic
books" following a syllabus** (extending `gen_code_textbook.py`); every code example must **run**.

### 2.3 Real-work environments (tool use / RL)

| Source | Content | License |
|---|---|---|
| **SWE-Gym** (`SWE-Gym/SWE-Gym`) | real issues + `base_commit` + `patch` + `test_patch` + `FAIL_TO_PASS` / `PASS_TO_PASS` | MIT |
| SWE-bench Lite | 300 tasks — **for MEASUREMENT only**, not training | — |
| Self-built tasks (like the existing 40) | planted bugs / removed function bodies / tracebacks from code with tests | self-made |

## 3. Data filtering (pipeline `build_code_cpt.py`, run on a CPU box like P1)

1. **Licenses** (§2.1) · drop auto-generated / minified one-line / data files (lines > 1,000 characters, abnormal alphanumeric ratio, `.min.js`, lock files).
2. **Deduplication**: exact (hash) + near (MinHash, as in P1).
3. **Quality scoring**: Stack-Edu already has `int_score` (keep ≥ 3); other sources → an `edu_classifier.py`-style educational scorer for code.
4. **Secrets / personal data**: scan for API keys, tokens, passwords, emails, private keys → mask or drop the file.
5. **Benchmark leakage**: drop files containing HumanEval, MBPP, SWE-bench Lite problems/solutions (13-token n-gram matching).
6. **Packing by repository**: files from the same repo placed together (README → source → tests) to learn how files call each other; a portion in **FIM**
   format (fill-in-the-middle: prefix / suffix / middle) for mid-line completion.

## 4. Phased plan

| Phase | Work | Data | Machine / time (est.) |
|---|---|---|---|
| **C0** ✅ | Harness + tool-enabled runner | — | Mac |
| **C1** | Full baseline (HumanEval 164, MBPP 257, tool use 40) for v5, the final SFT version, the SFT + expert version | — | 1×H100, a few hours (the Mac is too slow: 1–3 min per problem) |
| **C2** | **Code data that has really been run**: the teacher generates problems + tests → several solutions → keep solutions that PASS; task types: write a function, fix from a traceback, write tests, refactor, SQL, bash, code comprehension (vi + en) | 100–200K samples | teacher on 1–2×H100, ~1 day |
| **C3** | **Tool-use trajectories**: the teacher drives `code_agent.py` (tool mode) on self-built tasks + SWE-Gym, keeping SUCCESSFUL trajectories (tests pass) | 20–50K trajectories | 1–2×H100, ~1 day |
| **C4** | Retrain the **code expert** (or extra SFT) on C2 + C3 + code images; loss only on model turns, NOT on `<obs>` | — | 1×H100, ~0.5 day |
| **C5** | **RL from tests** (GRPO): 8 solutions per problem, reward = tests pass (+ a small reward for correct tool format); both single-pass and multi-turn tool tasks | problems with tests (C2 + tool use) | 1×H100, 1–2 days |
| **C6** (if still at the ceiling) | **Code mid-training** from `cpt_b_stable.pt` (public on HF): 3–10B tokens; extend context to 16–32K with repo-packed data; the LR-decay phase uses books + opc-annealing + high-scoring Stack-Edu | §2.1–2.2 | 4×H100, 1–3 days (~40K tok/s → 10B ≈ 70 hours) |

**Suggested mix for C6:** Stack-Edu / github-code-clean (Python, JS/TS, SQL, bash, Java, C++, Go) ~40% · CommitPackFT + notebooks ~10% ·
open books + docs ~5% (upweighted in the LR-decay phase) · maths ~10% · **Vietnamese + old data revision ~35%** (against forgetting — the
"only diverse data helps" lesson). After C6, SFT + experts (P3) must be re-run on the new trunk.

**Proposed order:** C1 → C2 → C3 → C4 → C5, measuring after each step with C1; only do C6 if C2–C5 are not enough.

## 5. The runtime in real use (UI / API)

- Chat UI: a **"Run"** button for code blocks → sandbox → show the output; a **code assistant** mode that loops "write → run tests → fix" up to N
  rounds and answers WITH real execution results.
- Tool use in the UI uses exactly the C3 protocol (which the model learned) — a virtual project directory per conversation.
- RAG over library documentation (the open docs of §2.2) for API questions.

## 6. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Copyright / licensing | permissive licenses only; exclude NC/GPL/unknown; keep a source + license list in the data README |
| Code containing secrets / personal data | scan and mask during filtering |
| Benchmark leakage → inflated scores | n-gram filtering against HumanEval/MBPP/SWE-bench; the tool-use set is built from the MBPP splits NOT used for measurement |
| Model leaning toward "explaining instead of writing code" (seen at SFT step 2,500) | high share of "write complete code" samples; RL rewards from tests |
| Forgetting Vietnamese / chat as code increases | ~35% revision; gated experts (frozen trunk) for C4 |
| Sandbox escape when running model code | OS sandbox + resource limits; run RL/teacher on disposable boxes with no important keys |
| Stack-Edu needs AWS to download content | start with sources that include content (github-code-clean, opc, CommitPackFT); add Stack-Edu once an AWS account exists |

## 7. Decisions

**10/13: the user chose option C** — keep the order: finish P3 (TK_107) first, then C1 → C5; no parallel box rental, no mid-training first.

### Open questions

1. Order C1 → C5 (no mid-training first) — agreed?
2. Teacher for C2/C3: Gemma-4-31B (current) or also a strong open-licensed code model (check the license before use)?
3. An AWS account to download Stack-Edu / The Stack v2 content (only needed for C6).
4. Box: run C1–C5 on the current 1×H100 after P3 finishes, or rent another box so the teacher can run in parallel?

## 8. Parallel work while waiting for P3.3' (10/14)

- **Verifying the P3 gate** (`out/bao_cao_kiem_chung_cong_expert.md`): a closed gate is not the cause. Results per mode:
  - normal: HumanEval 4/40, Excel 17/27
  - off: 0/40, 3/27
  - open: garbage output
- **The teacher-free part of C2:** `scripts/c2_build_verified.py` → `data/c2/oci_verified.jsonl`.
  - **99,658 Python problems** with tests re-run in the sandbox, 97.1% pass; `tests` kept for use in C5.
  - Source: OpenCodeInstruct parts 0–3/50 (CC-BY-4.0). Problems similar to HumanEval/MBPP removed.
  - Still missing: a Vietnamese version (teacher translates the prompts), bug-fix / SQL / bash types. To extend: add parts 4–49 (~25K passing problems each).
- **Two-stage expert:** `expert_office_1p3b.py trainGEN2`.
  - Stage A: gate fully open, each step enables one expert on its own domain data (`--domain`).
  - Stage B: freeze the FFNs, gate initialised nearly open (`--gate-init 3`), learn the gate on a shared `--mix`.
  - Trial-run on the Mac (1.3B trunk, MPS). Used for C4.

**10/14: the user's decisions**
- TK_108 runs on the **current H100 box** (Vast P3). Once P3 finishes, go straight to C1.
- How C1 runs: `scripts/box_run_c1.sh` waits for "P3.3' DONE", then runs 3 UI servers in parallel on the GPU to measure HumanEval, MBPP and tool use.
- Where each step learns:
  - **C2–C5:** create a **NEW code expert** with `trainGEN2`, then do RL on that expert. The trunk, the office/academic experts and the vision part are all frozen.
  - **C6:** only when hitting the ceiling. Let the trunk learn more code, then retrain every expert.

**10/15: choosing the teacher for C2/C3**
- **C2:** keep Gemma-4-31B. Reason: the test-running filter already removes wrong solutions, and Gemma translates Vietnamese prompts well.
- **Start of C3:** a small trial to choose the teacher, NOT decided from leaderboards.
  - Run Gemma-4-31B and Qwen3-Coder-30B-A3B on the same ~50 tool-use tasks through the `code_agent` runner (tool mode).
  - Measure success rate (tests pass) and speed. The winner generates the C3 trajectories.
- Before using Qwen: check the license on the official model card (aggregator sites say Apache 2.0).
- Reference numbers, mostly from aggregator sites:
  - Gemma-4-31B: LiveCodeBench ~80%.
  - Qwen3-Coder-30B-A3B: SWE-bench Verified ~52%. An MoE with only 3B active parameters, so it generates fast.

**10/15: measuring the `bpe_v1` tokenizer (~32K tokens) on code** (2,000 Python solutions from C2; unit: characters/token)

| Tokenizer | Python code | English prompts | Vietnamese |
|---|---|---|---|
| aivie | 2.57 | 2.44 | 2.95 |
| Qwen2.5-Coder (151K) | 3.96 | 3.58 | 3.25 |

- For the same code, aivie uses about **1.5×** as many tokens as Qwen-Coder.
- Exact round trip 2000/2000; no UNK; 4- or 8-space indentation each collapses into 1 token.
- **Weakness:** keywords at the start of a line (after the indentation token, so no `Ġ` prefix) get fragmented: `return` → `ret|urn`, `while`, `elif`, `except`, `raise`, `with`, `yield`. Also with a leading space: `except`, `lambda`, `False`, `append`, `dict`, `enumerate`.
- 14.7% of code tokens are just whitespace or newlines.
- **Conclusion:** doesn't block C2–C5. For C6, consider adding ~4–8K merges learned from code plus new embedding rows. With the "large expert" approach, only the new embedding rows could be trained while the old rows stay frozen.

## 9. C1 → C4: baseline, code data that has really run, tool-use trajectories (10/15–16)

**C1 — p3r1 baseline (10/15):** HumanEval 6.1 · MBPP 5.1 · tool use 0% (82% of turns call no tool). Excel specifically: the P3.3' teacher loop
dropped Excel from 17/27 to 9/27 because the teacher wrote formulas inline in sentences. Fixed with `build_office_fix.py` (putting formulas in
```` ```excel ```` blocks) and further training of the office expert → **p3r1xl**: Excel 13/27, formulas in blocks 27/27.

**C2 — code data that has really run (`c2_teacher.py`, teacher Gemma-4-31B):**

| Type | Generated | Kept |
|---|---|---|
| vi — translated prompts, code blocks kept identical | 30,000 | 29,999 |
| debug — planted bug → real error → fix that passes tests | 20,000 | 12,908 |
| predict — output taken from actually running the code | 10,000 | 9,111 |
| sql — 2 independent solutions must produce identical results | 8,000 | 7,654 |
| oci_verified (OpenCodeInstruct, no teacher needed) | — | 99,658 |

**C2b — multilingual code:**
- A: 10 languages, syntax-checked with tree-sitter/sqlglot, yielding 361,824 samples.
- C2b+: Next.js, Tailwind/WebSight, Svelte 5, Kotlin Compose, Flutter, shadcn.
- B: the teacher writes tests then really runs them with node, tsc, javac, dotnet, php, go, bash, Chromium (HTML screenshotted at 2 widths and graded by the teacher looking at the image) and SQL Server 2022 (T-SQL). The teacher fixes at most 2 rounds and the full teacher log is kept.
- Rescued ~35% of samples whose OpenCoder prompts contained Chinese by having the teacher translate them.
- From the teacher logs, extra "wrong → error → fix" data was built (`c2b_fixmix.py`): wrong turns marked `nolearn`, only the fix that runs is learned.

**C3 — tool-use trajectories (`c3_teacher.py`):**
- Small project tasks of the fix and implement types, built from C2.
- Small trial to choose the teacher on 50 tasks: **Gemma 41/50 · Qwen3-Coder 28/50** → Gemma chosen.
- Only successful, protocol-correct trajectories are kept.
- **C3-R:** tasks with planted WRONG turns (TAB, bad edits, repeats; `nolearn`); the teacher must recover within ≤ 15 steps.

**C4 — small code expert (0.3B parameters, 12 insertion points):**

| Version | HumanEval | MBPP | Tool use | Notes |
|---|---|---|---|---|
| c4e (early C4, two-stage `trainGEN2`) | 7.3 | 9.3 | 0 | 99.8% of turns call no tool |
| c4b (C4 round 2, + C3 45%) | 11.0 | 11.7 | 32.5 | |
| c4c (C4 round 3, + C2b + C3-R) | 11.0 | 9.7 | 22.5 with runner v2 → **45 with runner v3** | MultiPL-E 2.4 |

**Tool runner v2 → v3 (`code_agent.py`)** — fixed 27/40 tasks that failed because of loops on errors:
- An identical call to the previous turn → the model is told to change approach.
- `edit_file` with a TAB or space mismatch → points to the exact region in the file (v3 auto-matches indentation).
- `write_file` with identical content → reported back to the model.
- Set `AIVIE_TOOL_GUARD=0` to use the old runner for comparisons.

**Multilingual harness `code_eval_ml.py`:** MultiPL-E (MIT), HumanEval × JS/TS/Java/C#/PHP/Go/Bash. Each run samples 30 problems per language, really executed in the sandbox.

## 10. C5: DPO and GRPO — no better, stopped (10/17)

- **C5a DPO** (2,009 correct ↔ wrong pairs from the teacher logs, base c4c): HumanEval 7.9 vs 11.0, MBPP ~7 vs 9.7, MultiPL-E lower. **Dropped.**
- **C5b GRPO** (`c5b_grpo.py`, reward = tests pass): stopped at ~172 steps.
  - A fixed 24-problem check set stayed at 12–13/24. Python 39% → 41%.
  - Multilingual ~0%, so 40% of problems gave no reward signal.
  - Conclusion: **the ceiling is the small expert's capacity, not the learning method** → switch to a large expert.

## 11. LARGE code expert pilot (10/17–18) — success

- 24 points × hidden 8192 = 1.21B parameters.
- The expert is pretrained on its own on the frozen trunk (`trainCPTX`, gate open): 321M tokens (`build_code_expert_data.py`).
- Then SFT with exactly the C4 round-3 data.

| Version | Tool v3 | HumanEval | MBPP | MultiPL-E | Excel | easy20 |
|---|---|---|---|---|---|---|
| c4c (small expert, baseline) | 45 | 11.0 | 9.7 | 2.4 | 16/27 | 15/20 |
| cx (large expert + SFT) | 45 | 13.4 | 17.1 | 5.7 | 14/27 | 16/20 |
| cx2 (cx + 600-step tool-use top-up SFT) | 47.5 | 13.4 | 15.6 | — | 14/27 | 17/20 |

**Measurement traps caught in the pilot:**
- 4 UI servers running the large expert need 81/80 GB. Long prompts OOM'd, the server swallowed the error and returned empty, so the first measurement gave tool use 5–10% (a broken number). Now the server prints `[lỗi /api/chat]` ("[error /api/chat]") and at most 2 servers run per 80 GB GPU.
- The old UI had `MAXCTX` 2000, while C3 trajectories were trained at 4096. Added `--max-ctx 4096`. Raising max_new to 1024 while leaving ctx at 2000 dropped tool use from 47.5 to 20.
- A fork bomb from model-generated code (MultiPL-E sh) left 5,653 processes and exhausted the box. Fixed with `run_group` + `kill_group` and RLIMIT_NPROC for the sbx user.

## 12. The real run: cx3 → cx3b (10/19–22, 2×A100 SXM4 80 GB box, $1.67/hour)

**Data `data/cx3`:**
- 3.4B tokens, pushed to HF `nn-aivie-cpt-data/cx3` (public).
- The `gh` group comes from `codeparrot/github-code-clean`, keeping only mit, apache-2.0, bsd, isc, unlicense, cc0 licenses, with per-language caps.
- Held-out set identical to the pilot's. The first round used 1.5B tokens.

**Pretraining cx3pt:**
- Ran `box_run_cx3_pt.sh`: 2-GPU `torchrun`, manual all_reduce, `--resume`, 23,000 steps, 9.1K tokens/second, 47 hours.
- `box_cx3_hf_sync.sh` pushes the expert + resume state to HF every 1,000 steps (squashing history) in case the box is reclaimed.
- Loss 25–27% lower than the equivalent pilot point on algo, snip, files (table in `TK_108_KIEN_TRUC_CX3B.md` §5).

**SFT:**
- **cx3** (`box_run_cx3_post.sh`): mix of C4 round 3 + C3-R2, 1,500 steps, lr 1e-4.
- **cx3b** (`box_run_cx3b.sh`): 600-step tool-use top-up SFT, lr 5e-5.

| Version | Tool v3 | HumanEval | MBPP | MultiPL-E | Excel | easy20 |
|---|---|---|---|---|---|---|
| cx2 | 47.5 | 13.4 | 15.6 | 5.7 | 14/27 | 17/20 |
| cx3 | 42.5 | 12.2 | 22.2 | 6.7 (ts 16.7 · java 13.3 · js 6.7 · go 6.7 · php 3.3 · cs 0 · sh 0) | 16/27 | 16/20 |
| **cx3b** | **55.0** (fix 75 · implement 35) | 13.4 | **23.3** (24.9 after fixing) | — | 15/27 | 16/20 |

**Lessons:**
- In the pilot the tool-use top-up SFT looked "unnecessary" (cx 45 ≈ cx2 47.5). But after long pretraining it is **essential** (cx3 42.5 → cx3b 55). It is now a fixed step in the recipe.
- Long pretraining clearly helps MBPP and multilingual code. HumanEval alone barely moves (12–13%) across cx2, cx3, cx3b.

**Traps on the new box (NGC image):**
- `hf` typer error → `PIP_CONSTRAINT= pip install -U "typer>=0.16"`.
- The system Python lacks `transformers`, so the UI dies when loading SigLIP.
- `PIP_CONSTRAINT` pins `datasets` 2.2.1 in the evaluation venv, mismatching the new `fsspec`, causing `'**' can only be an entire path component`. Fixed with `datasets>=3` + `numpy>=2`.
- `hf download` with several `--include` flags sometimes misses files (`data/c2/oci_verified.jsonl`). The preparation script now checks every file.

**Hardware (speed measured on this exact job):**

| Box | Speed | Price |
|---|---|---|
| 1×H100 SXM | ~9.0K tokens/second | |
| 2×A100 SXM4 | 9.1K tokens/second | $1.67/hour — cheapest per token |
| 4×A100 | 17.9K tokens/second | $6.4/hour |

## 13. Real-world testing on the M4 (10/21) and next steps

- **8 real-world, multi-constraint tasks on cx2:**
  - All 3/3 tasks with real tests failed: computing a VAT invoice, counting 5xx errors in an nginx log, debugging a mutable default `gio=[]`.
  - T-SQL didn't JOIN the tables. Excel used SUMIF where SUMIFS was needed. JS debounce wrong. .NET optimisation advice went off-topic.
  - Only the email was good.
  - Conclusion: the model writes code from templates and doesn't follow the task's constraints. This is the capacity of the frozen trunk.
- `serve_hybrid_ui.py --half`: loads in bf16. In fp32 this version needs ~17 GB, so the 24 GB M4 swapped and couldn't produce a single answer in 2.5 hours.
- **Next steps to decide:**
  1. Re-run the 8 real-world tasks on cx3b to compare directly with cx2.
  2. Continue pretraining the expert to the full 3.4B tokens (`--resume` from `nn-aivie-cx3-ckpt`), then SFT and the tool-use top-up.
  3. Do C6 (learning into the trunk) if needed to break through the comprehension ceiling.

**10/22: the same 8 real-world tasks on cx3b** (`scripts/m4_probe_real.py`, answers saved in `out/m4_probe_cx3b.jsonl`):
- Results almost identical to cx2: the 3 tasks with real tests still all fail, T-SQL still doesn't JOIN, Excel still SUMIF (this time the right column B but missing the month condition), debounce wrong, .NET advice drifts into writing a C# class, only the email is good.
- 3 extra short tasks to make sure M4 bf16 doesn't break the model:
  - English prompt in exact MBPP format (`Your code should pass: assert …`) → **PASS**.
  - Short Vietnamese prompt (reverse word order) → fail: splits on commas.
  - Prompt without a stated language → the model writes JavaScript.
- **Conclusion:** the model improves on **exactly the benchmark prompt format** (MBPP 15.6 → 23.3; in-runner tool use 47.5 → 55), but this doesn't transfer to free-form prompts or multi-constraint Vietnamese prompts.
  - Cause 1: the frozen trunk understands prompts poorly.
  - Cause 2: the SFT data is skewed toward the "short English prompt → one function" format.
- **A cheap fix before C6:** more SFT on real-world, multi-constraint Vietnamese tasks with tests (teacher-generated + really run, as in C2). E.g. invoicing logic, logs, SQL reports with JOINs, multi-condition Excel. And measure with a set of such tasks instead of only looking at HumanEval/MBPP.

## 14. C7: real-world Vietnamese tasks (10/22)

- **Data (`scripts/c7_real.py`):** teacher Gemma-4-31B FP8 running 2 vLLM instances, one per A100 GPU. Tasks of 8 types: business Python, data Python, string Python, debug, class, SQL, Excel, JS.
- **Two layers of verification by real execution:**
  1. The reference solution must pass 8–12 tests.
  2. The teacher re-solves independently from the prompt alone, and that solution must pass the same tests.
  Keep rate ~28%, ~13.5 tasks/minute.
- **The real-world benchmark** `data/c7/eval.jsonl` (355 tasks, never trained on) is **the main yardstick from now on**. **Training set** 3,000 tasks (the user lowered it from 10K to 3K to measure within the day).
- **cx3c = cx3b + 700 SFT steps** (C7 35% + tool-use / code / office revision, lr 5e-5). C7 loss 0.80 → 0.41.

| Version | C7 benchmark (355) | C7 Excel | Business Python | HE | MBPP | Tool |
|---|---|---|---|---|---|---|
| cx3b | 0.6% | 1/39 | 0/116 | 13.4 | 23.3 | 55 |
| cx3c | 2.5% | 6/39 | 0/116 | **15.9** | **26.5** | 50 |

**Traps hit while running C7:**
- The `formulas` library hangs forever while holding a shared lock. Fix: compute Excel in a child process with a timeout.
- 2-GPU TP fails on custom all-reduce inside the container. Fix: run 2 vLLM instances, one per GPU.
- The FP8 kernel on A100 needs `LD_LIBRARY_PATH` pointing to `nvidia/cu13/lib`.
- The teacher copied script text into prompts. Fix: filter with `strip_tag`.
- `pkill -f` in the same ssh command kills the ssh session itself. Fix: copy the script file to the box and run it with `setsid`.

## 15. C8: LoRA on the trunk's mixing blocks (10/22–23)

**Hypothesis:** the ceiling lies in the frozen trunk. FFN experts don't touch the part that reads and connects the ideas in a prompt, which lives in Mamba and attention.

**Method:**
- `CodeLoRA` (`expert_ffn_1p3b.py`) attached to `wq/wk/wv/wo` and `in_proj/out_proj` from block 3 upward, 210 projection layers in total.
- Trained together with the code expert: `trainGEN --lora-r`, lr 2e-4.
- Clean comparison with cx3c: same base cx3b, same data mix, same 700 steps.

| Version | LoRA | C7 benchmark | Excel | Business Python | SQL | Tool | Office |
|---|---|---|---|---|---|---|---|
| cx3c | — | 2.5% | 6/39 | 0/116 | 1/43 | 50 | — |
| cx3d | r32 (+49M) | 3.4% | 8/39 | 1/116 | 1/43 | 55 (fix 80) | pending |
| cx3e | r128 (+195M) | 3.7% | 7/39 | 3/116 | 2/43 | 55 | Excel 15/27 · easy20 15/20 |

**Conclusions:**
- LoRA adds only 3–4 tasks, within the noise threshold. r128 is no better than r32, so LoRA capacity is not the bottleneck.
- Lightly tuning the trunk with 3K tasks doesn't break through the ceiling either.
- For a 2.5B trunk trained on 20B tokens (8% code), the remaining direction is **C6**: deep training into the trunk with 10–20B code tokens + Vietnamese revision, then re-running the P3 SFT. Proposed parameters and costs were presented to the user on 10/23. To decide: token count, machine type, and whether to add code tokens to the tokenizer.
