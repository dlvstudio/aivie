# TK_108 — Real-world quality evaluation of c6l1 (code focus)

*06/11/2026 · checkpoint `box_backup/c6sft/c6l1_vlm_bf16.pt` (the main version after the c6l1 teacher loop) · run on the M4, `--half`, context 8192, UI on port 8805*
*English translation of the Vietnamese original in the repository root.*

## Summary

**c6l1 is not yet usable as a real coding assistant.** The test set is 22 real-use cases written in Vietnamese the way users typically ask.

- **Average score 1.3/10.**
- **0/22 cases usable as-is.** Only 5/22 partly correct (4–6 points).
- **13/22 cases scored 0–1:** wrong task, code doesn't run, or wrong explanation.

c6l1's benchmark scores (HumanEval 26.8 · MBPP 39.3) **do not reflect real experience**. The reason is in section 4.

Two bright spots:
- **Speed** 25–30 tokens/second on the M4. No server errors, no empty responses.
- **Fluent Vietnamese,** tidy markdown and code blocks.

The biggest failure is caused by **the data, not by the decoding settings**. I re-ran 6 cases at temperature 0.2 (instead of 0.7) and the results did not improve.

---

## 1. Method

- **22 real cases**: Python functions, pandas, bug fixing, reading a traceback, SQL, JS (debounce), HTML/CSS, FastAPI, regex, writing pytest, LRU cache, bash, Excel formulas, Python→JS, security review, a 3-turn conversation, a CLI game, git, asyncio, explaining a concept, refactoring, React.
- **Generation settings** as the UI defaults: temperature 0.7, top_p 0.9, repetition penalty 1.1, at most 2048 tokens (3072 for the game case).
- **Grading:**
  - I read every answer and scored it 0–10 on: meets the request, code runs, explanation is not wrong.
  - Pure computational snippets were actually run on the Mac (`python3 -I`/`node`, in a temporary directory, 10-second limit).
  - Code touching the network, system files or the shell was only read, not run.
- **Raw data:**
  - Full answers in `out/real_eval_c6l1/*.json`, the temperature-0.2 run in `out/real_eval_c6l1/t02/`.
  - Task set and run script in the appendix.

## 2. Scores

| # | Case | Type | Score | Main comments |
|---|---|---|---|---|
| 01 | Vietnamese phone number | Python function | 3 | Works for numbers starting with 0, but **doesn't handle `+84`**. The results in its own example contradict its code. Ran: `0912…` → True, `+84912…` → False. |
| 02 | pandas revenue | data | 1 | Uses a non-existent `thang` column. Sums `so_luong` (quantity) instead of revenue. `.loc[...max()]` wrong. Excel opened with `mode='a'` fails when the file doesn't exist. |
| 03 | mutable default `[]` bug | debug | **0** | Returns **the buggy code unchanged** and invents "the `or` operator on line 12". |
| 04 | KeyError 'email' | debug | 1 | Nonsense explanation ("the traceback function only runs lines 2–100"). Returned code unchanged. |
| 05 | SQL top 5 Hanoi customers 2024 | SQL | **6** | JOIN, filter, GROUP BY, ORDER BY, LIMIT correct. **Missing the order count** (COUNT). Adds an unrequested `HAVING > 5 million`. GROUP BY name instead of id. |
| 06 | search-box debounce | JS | **0** | **No debounce function.** No input box. Two duplicated code blocks. Hits the 2048-token ceiling. |
| 07 | login page | HTML/CSS | 4 | The form works, has a viewport. But the title is "Login **successful!**". **No "Forgot password" link.** English "Password" label. Broken placeholder characters "� dần!". |
| 08 | FastAPI todo | web | **0** | **Doesn't use FastAPI.** Writes a `requests` client calling a non-existent API. Puts `@dataclass` on a method. |
| 09 | email regex with de-dup | string processing | 4 | The regex extracts the emails correctly but **doesn't de-duplicate** (ran: 3 emails, 1 duplicate). The explanation's "sorting", "re.sub", "O(n²)" are all wrong. |
| 10 | write pytest | test | **0** | **Writes no tests.** "Fixes" a correct function into a wrong one. |
| 11 | O(1) LRUCache | algorithm | 2 | The OrderedDict idea is right, but `put` deletes and **doesn't re-insert** when the key exists (ran: `get(1)` → -1). `keys().next()` errors. Ends by repeating `*****` until the 2048-token limit. |
| 12 | bash backup, keep 7 copies | bash | **0** | Runs an unrelated `git status`. `tar` with `"$webroot/*"` in quotes. No 7-copy rotation. |
| 13 | Excel unit price / amount / SUMIF | Excel | 1 | Gives only one wrong INDEX/MATCH formula searching for the string "SP0". Misses all 3 parts. Invents a result "50000000". |
| 14 | Python to JS | translation | 4 | Runs, but **drops the punctuation stripping and lowercasing**. Returns words instead of (word, count) pairs. Ran: `['Mèo','ăn','cá.']`, differs from the Python version. |
| 15 | SQLite login review | security review | **0** | **Misses SQL injection** and plain-text passwords. Invents a "bare quote" bug. Keeps the dangerous f-string. |
| 16 | 3-turn conversation (JSON → type hints → tests) | multi-turn | 1 | Turn 1 uses `sum(data.values())`, wrong for a list. Turn 2 has a **syntax error** (`return` outside a function). Turn 3 has no tests and uses an undefined `tmp_path`. |
| 17 | CLI number-guessing game | small app | **0** | No `input()`, the computer guesses by itself. No high-score saving. Says to `pip install random`. |
| 18 | git: committed `.env` by mistake | tooling | 2 | Step 1 `reset --soft HEAD~1` is right. Then it says `reset --hard`, **losing data**. No mention of `.gitignore` or `git rm --cached`. |
| 19 | asyncio + aiohttp, limit 3 | async | **0** | No semaphore, no gather, invented aiohttp API. 11 URLs though the task says 10. Says to `pip install asyncio`. |
| 20 | set vs list for `in` lookups | explanation | **0** | Explanation **backwards and wrong** ("sets use recursion"). No timing. |
| 21 | refactor discounts | refactor | **0** | Returns **the code unchanged** and invents a bug it "fixed". |
| 22 | React todo | React | **0** | Writes **plain HTML, no React**. `getElementById('todo-list')` points at a non-existent element. |

**By group:**

| Group | Cases | Avg score |
|---|---|---|
| Write new code from a clear spec (01, 02, 05, 09, 11, 13) | 6 | 2.8 |
| Front-end, web (06, 07, 08, 22) | 4 | 1.0 |
| Existing code, task other than fixing (10, 14, 15, 21) | 4 | 1.0 |
| Bug fixing, debugging (03, 04) | 2 | 0.5 |
| Small apps, scripts (12, 17, 19) | 3 | 0 |
| Knowledge, tooling (18, 20) | 2 | 1.0 |
| Multi-turn (16) | 1 | 1.0 |

**Temperature 0.2 trial (6 cases: 01, 03, 08, 10, 11, 21):** none improved.
- Case 01 drops the code entirely and only lists meaningless `assert`s.
- Cases 03, 10, 21 still return the code unchanged and claim to have "fixed the bug".
- Case 11 is closer, but stores `[value, list]`, corrupting the values.

So the failure is not in the decoding.

---

## 3. Five main failure modes

### 3.1 "See code → fix a bug". The most serious one.
When the prompt contains a code block, the model **almost always answers in a bug-fix template**. Answers open with "The bug is on line…", "The main bug…" or "The previous bug…". The model then invents a bug and returns the code unchanged or worse, **even when the task asks for tests, a refactor, a review or a translation.**
- Seen in cases 03, 04, 10, 15, 21. At temperature 0.2 it is even more pronounced.
- The model even "fixes" correct code (case 10), which is worse than simply answering wrong.

### 3.2 Ignoring the requested technology or constraints
- FastAPI becomes a `requests` client (08). React becomes plain HTML (22). Debounce becomes no debounce (06).
- "De-duplicate" (09), "+84" (01), "order count" (05), "keep 7 copies" (12), "at most 3 requests" (19) are all dropped.
- The model gets the general topic but **does not follow the spec**.

### 3.3 Confident fabricated explanations
Examples: `pip install random` / `pip install asyncio`, "collections requires an external library", "sets use recursion", "bare quote", "O(n²)". Beginners may believe these, which makes it **dangerous**.

### 3.4 Teacher-loop opening templates, out of context
Many answers open with "I've added CSS…", "I've updated…", "The implemented solution…", "The previous bug was…", even on the first turn when nothing exists to "add" to.
- This is a trace of **later-turn correction** samples (the teacher fixing an earlier answer) being learned as first-turn answers.
- In `loop1.jsonl` there are 377 samples starting "<done>Đã triển khai" (implemented), 286 "<done>Đã sửa lỗi" (bug fixed), 193 "Lỗi nằm ở" (the bug is in).

### 3.5 Degeneration on long answers
- Infinite character repetition (`*****…` in case 11), duplicated code blocks (case 06), hitting the token ceiling.
- Beyond about 600 tokens, quality drops clearly.

---

## 4. Why the benchmarks are high but real use is poor

| | HumanEval / MBPP | Real use |
|---|---|---|
| Prompt language | English, short | Vietnamese, natural description |
| Form | function signature and docstring given, only the body is needed | design from scratch: choose libraries, structure, multiple files or parts |
| Scope | 1 pure function, 5–20 lines | APIs, front-end, scripts, many constraints at once |
| Task type | write a function | write, fix, test, review, refactor, explain, multi-turn |

The skill of filling in a function body from a docstring (HE 26.8) does not transfer to following a multi-constraint Vietnamese spec. The C7 set is closer to reality, and there it scores only **4.8%**, consistent with the table above.

## 5. Root cause, with numbers from the training data

I went through `sft_b.jsonl` (215K samples, the basis of c6l1). It contains **44,275 samples with a code block in the prompt**:

| Task on existing code | Samples | Share |
|---|---|---|
| Bug fix | 12,974 | **29.3%** |
| Language translation | 1,521 | 3.4% |
| Refactor | 291 | 0.7% |
| Explanation | 102 | 0.2% |
| Review / security | 87 | 0.2% |
| **Write tests** | **66** | **0.1%** |
| Other (mostly tool/agent trajectories) | 29,234 | 66.0% |

The most common openings in SFT-B are "<done>Đã triển khai" (6,228), "<done>Đã sửa lỗi" (6,197), "Lỗi nằm ở" (4,296) and "Lời giải triển khai" (2,194).

Conclusion: bug-fix samples outnumber test-writing or review samples by **about 200×**. The model learned the rule "if there's code, say 'The bug is in…'". This is a data-distribution skew, not a model-size limit — far cheaper to fix than adding parameters.

There is also a size-related cause:
- A self-pre-trained 2.5B model: about 20B pre-training tokens plus 10B tokens of code-heavy continued training.
- Usable 1.5–3B code models typically need several trillion tokens.
- So failure modes 3.2, 3.3 and 3.5 will be hard to eliminate with SFT alone.

---

## 6. Recommendations, in order of value

1. **Rebalance the code SFT data** (cheap, biggest impact):
   - Add 20–30K Vietnamese samples for tasks other than bug fixing: writing tests, security review, refactoring, explanation, translation, writing from multi-constraint specs (FastAPI, React, bash, asyncio, pandas).
   - **Execute-check every sample** in the sandbox as in C2/C3. This is what the teacher does well.
   - Add **contrastive samples**: correct code plus a request for tests, where the answer is tests, not "The bug is in".
   - Cut bug-fix samples to about 10% of samples with code.
2. **Filter out-of-context opening templates** in `build_c6_loop.py` and `build_sft_c6.py`:
   - Drop or rewrite teacher corrections opening with "I've added/updated…", "The previous bug…" when they are a first turn.
   - Remove the `<done>` prefix from ordinary chat answers (keep it only in tool trajectories).
3. **Add these 22 cases to the regression suite** (`out/real_eval_c6l1` as the baseline). Grow to about 100 cases, some auto-executed, so the next round measures "real use" rather than only HE/MBPP.
4. **Spec-based teacher grading**: in the next teacher loop, have the teacher grade strictly against the task's constraint checklist ("does it use FastAPI", "does it de-duplicate"…), not just by impression.
5. **Fight long-answer degeneration**: add complete 600–2000-token code samples. When generating code, consider an n-gram repetition penalty or stopping on character repetition.
6. **Longer term**: a genuinely usable coding assistant needs (a) many more code pre-training tokens, or (b) the model acting as an orchestrator that uses tools — document RAG, running code to self-check, then fixing based on the run results. Current tool results (52.5%, fixes 70% of errors) show the model does better when it can run code and see the output.

## 7. Strengths worth keeping

- 25–30 tokens/second on the M4 (`--half`). Stable 8K context, no server errors.
- Natural Vietnamese, clear answer structure: code blocks, steps, notes.
- Decent basic SQL (case 05). Short snippets usually run (01, 09, 14), they just miss constraints.
- A habit of asking back or suggesting a next step at the end of an answer.

---

## Appendix — re-running

- Task set: `data/eval_clean/real_code_22.json`
- Script: `scripts/real_use_eval.py`. The script uses temperature 0.7; to measure another version change `URL` and `OUT`.

```bash
# start the c6l1 UI (port 8805), then run all 22 cases → out/real_eval_c6l1/<id>.json
python3 scripts/real_use_eval.py data/eval_clean/real_code_22.json
# run selected cases against another server
URL=http://localhost:8804 OUT=out/real_eval_b python3 scripts/real_use_eval.py data/eval_clean/real_code_22.json 03_bugfix 10_pytest
```

Each result file stores every conversation turn: question, answer, token count and time.
