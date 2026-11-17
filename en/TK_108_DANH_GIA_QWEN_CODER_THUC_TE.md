# TK_108 — Real-use evaluation of Qwen2.5-Coder on the M4 (code focus)

*06/11/2026 · Ollama 0.40 · `qwen2.5-coder:7b` and `qwen2.5-coder:3b` (Ollama's default 4-bit builds) · same 22 cases and same grading as the [c6l1 evaluation](TK_108_DANH_GIA_C6L1_THUC_TE.md)*
*English translation of the Vietnamese original in the repository root.*

## Summary

| | Qwen2.5-Coder 7B | Qwen2.5-Coder 3B |
|---|---|---|
| **Average score** | **7.3 / 10** | **5.7 / 10** |
| Usable as-is (≥ 7 points) | **14 / 22** | 12 / 22 |
| Partly right (4–6) | 8 / 22 | 3 / 22 |
| Broken (0–3) | **0 / 22** | 7 / 22 |
| Speed on the M4 | ~46 tokens/second | ~87 tokens/second |
| Size | 4.7 GB | 1.9 GB |

- **The 7B is usable as an everyday coding assistant** for common tasks such as pandas, SQL, FastAPI, React, debounce, bug fixing, security review, refactoring and Excel formulas.
- **No case is completely broken.** The remaining errors are mostly in **detailed constraints**: regex precedence wrong, the 3-request limit dropped, 7-copy backup rotation wrong, a test with one wrong assert.
- **The 3B** does the basic cases well but makes more logic errors, so it is less trustworthy.
- Both answer in clear Vietnamese and do the kind of task asked. They do not have c6l1's "see code → fix a bug" failure.

---

## 1. Method

- **Task set:** `data/eval_clean/real_code_22.json`, 22 Vietnamese cases, including one 3-turn conversation.
- **Generation settings** identical to the c6l1 run: temperature 0.7, top_p 0.9, repetition penalty 1.1, at most 2048 tokens, context 8192.
- **Script:** `scripts/real_use_eval_ollama.py`. Raw answers are in `out/real_eval_qwen7b/` and `out/real_eval_qwen3b/`.
- **Grading:**
  - I read every answer and scored it 0–10 on: meets the request, code runs, explanation is not wrong.
  - Pure computational code was actually run on the Mac, in a temporary directory, with a time limit (Python `-I`, pytest, pandas via `.venv`, node).
  - Code touching the network, the system or the shell was only read, not run.

## 2. Qwen2.5-Coder 7B scores

| # | Case | Score | Comments |
|---|---|---|---|
| 01 | Vietnamese phone number | 4 | Good structure, but the regex `^03\|05\|…\d{8}$` has **wrong precedence**. Ran: `03123456789` (11 digits) comes out **valid**, `+84912345678` comes out **invalid**. |
| 02 | pandas revenue | **10** | Ran: monthly revenue and top 3 correct, Excel file has exactly 2 sheets. |
| 03 | mutable default `[]` bug | 9 | Correct diagnosis and proper fix (`None`, then a new list). The last explanatory sentence is a bit muddled. |
| 04 | KeyError 'email' | 9 | Correct explanation, fixed with `if 'email' in user`. |
| 05 | SQL top 5 in Hanoi | 9 | Total, order count, GROUP BY id. `YEAR()` only exists in MySQL. |
| 06 | search debounce | 9 | Textbook debounce with input, fetch and `<ul>` rendering. Missing `encodeURIComponent`. |
| 07 | login page | 9 | Compact, works on mobile, Vietnamese text, has a "Forgot password" link. |
| 08 | FastAPI todo | 8 | Correct FastAPI and Pydantic, 404 on update. Delete does not return 404. The example sends `completed` in the body but the code reads it from the query. |
| 09 | email regex with de-dup | 9 | Ran: `['an.nguyen@gmail.com', 'sales@cty.vn']`, correct. |
| 10 | write pytest | 5 | Well-intended tests, but one assert is **wrong**: expects ValueError for `tong < 0`. Ran pytest: **FAILED**. |
| 11 | O(1) LRUCache | 9 | Doubly linked list plus dict, textbook. Ran: correct results. |
| 12 | bash backup, keep 7 copies | 4 | Backup is correct, but **rotation is wrong**: it counts only the file just created, so nothing is ever deleted. Variables not quoted. |
| 13 | Excel | 9 | All 3 parts: VLOOKUP, `=B2*C2`, `SUMIFS(D:D,A:A,"SP0*")`. Doesn't mention `;` separators on Vietnamese locale. |
| 14 | Python to JS | 6 | Correct logic, but **requires lodash** though nobody asked for it. Ran: `Cannot find module 'lodash'`. |
| 15 | SQLite review | 8 | Catches SQL injection and fixes it with `?` parameters. Doesn't mention plain-text passwords. |
| 16 | 3-turn conversation | 6 | Turns 1 and 2 good. Turn 3 has a test missing `import json`, and expects an exception the function already catches. Ran: **1 passed, 1 failed**. |
| 17 | number-guessing game | 5 | All features: input, error handling, high score, replay. But it **only counts a turn on a correct guess**, so it always says "1 try". |
| 18 | git: committed `.env` by mistake | 4 | `git rm --cached` then a new commit leaves **`.env` in the old commit**, exposed on push. Should be `reset --soft` or `commit --amend`. The `.gitignore` part is right. |
| 19 | asyncio + aiohttp | 5 | gather, timeout, prints failing URLs, but **no 3-request limit**, and admits it didn't implement one. |
| 20 | set vs list | 7 | Has a timing example on 1 million elements. Explanation a bit shallow, no mention of hash tables. |
| 21 | refactor discounts | 8 | Dict with `.get`, compact and correct. Ran: matches the original. The 1-million threshold still appears twice. |
| 22 | React todo | 8 | Full hooks: add, strike through, delete, count. Label says "Completed" but shows the number of **unfinished** items. Mutates objects in state directly. |

## 3. Qwen2.5-Coder 3B scores (summary)

| # | Score | Comments |
|---|---|---|
| 01 | 2 | `lstrip('+84')` strips character by character, so the logic is wrong. The results in its own example are wrong too. |
| 02 | 7 | Correct, but the top-3 sheet is written with `index=False`, so **product names are lost**. |
| 03 | 1 | Does not fix the bug; keeps the `[]` default. |
| 04 | 8 | Uses `.get()`, correct. |
| 05 | 8 | Complete, but GROUP BY name instead of id. |
| 06 | 9 | Correct, with `encodeURIComponent`. |
| 07 | 8 | Correct and complete; inputs lack `box-sizing`, so they may overflow the box. |
| 08 | 7 | Correct FastAPI; delete doesn't return 404. |
| 09 | 9 | Correct (`dict.fromkeys`). |
| 10 | 3 | Two asserts miscomputed, e.g. `chia_deu(15,3)` written as `[4,4,3]`. |
| 11 | 8 | Correct code; comments in the example wrong. |
| 12 | 6 | Deletes files older than 7 **days** instead of keeping 7 **copies**. |
| 13 | 4 | Wrong sheet-name syntax; SUMIF sums quantities instead of amounts. |
| 14 | 2 | Requires lodash; `\w` in JS **shreds accented words** (ran: "mèo" becomes `'m','o'`). |
| 15 | 8 | Catches both SQL injection and plain-text passwords. |
| 16 | 7 | Three coherent turns, tests using `tmp_path`. |
| 17 | 2 | Opens the file with `'a'` then calls `read()`, so it **fails immediately**. Mixes in the Chinese character "猜". |
| 18 | 7 | `git reset HEAD^` keeps the changes, with `rm --cached` and `.gitignore`. |
| 19 | 2 | `asyncio.gather(..., limit=3)` is a non-existent parameter, so it fails when run. |
| 20 | 6 | Correct hash-table explanation, but includes garbage text ("Khiển thị…"). |
| 21 | 8 | Same as the 7B. |
| 22 | 3 | Clicking an item **deletes** it; strike-through and counting logic wrong. |

---

## 4. Observations

**7B strengths:**
- **Does the kind of task asked.** Asked for tests, it writes tests; asked to review, it reviews; asked to refactor, it refactors.
- **Uses the requested technology:** FastAPI, React hooks, debounce, pandas, SUMIFS.
- **Code runs out of the box in most cases.** Of the 9 cases I ran, 6 were fully correct.
- **Explanations are correct and not made up.** No "pip install random" like c6l1.
- Natural Vietnamese, step-by-step presentation with run instructions.

**7B weaknesses:**
- **Often misses one detailed constraint:** "+84", "at most 3 requests", "keep 7 copies", 404 on delete. Users need to review the code rather than copy blindly.
- **Small, hard-to-spot logic errors:** a counter in the wrong place (case 17), wrong regex precedence (case 01), a wrongly expected assert (case 10).
- **Unsafe git advice for `.env`:** the secret stays in history. This is the most dangerous kind of error in the whole set.
- Pulls in external libraries (lodash) unnecessarily.

**The 3B:** twice as fast and two and a half times lighter. Common tasks are as good as the 7B (cases 06, 09, 15, 18, 21). With more complex logic it is clearly wrong more often, and some cases fail immediately when run (17, 19). Good for quick suggestions, not for delegating work.

## 5. Usage suggestions

- **For daily use on the M4:** choose `qwen2.5-coder:7b`. Run with `ollama run qwen2.5-coder:7b`, or connect an editor through the `localhost:11434` API.
- **Always run and re-read** the detailed constraints. This model gets about 80–90% of the job done; the last 10–20% is usually where the mistakes are.
- **To raise quality:**
  - (a) Add a self-run-code-and-fix-from-errors loop. The errors in cases 01, 10, 14, 16, 17 would all surface immediately when run.
  - (b) Fine-tune on rebalanced Vietnamese data drawn from the c6l1 evaluation, to learn the common constraints.
  - (c) Try the 14B/32B versions when a GPU is available.
- **This benchmark** (22 cases, same script) can be re-used to compare when switching models or after fine-tuning.

## Appendix — re-running

```bash
ollama pull qwen2.5-coder:7b
MODEL=qwen2.5-coder:7b OUT=out/real_eval_qwen7b python3 scripts/real_use_eval_ollama.py data/eval_clean/real_code_22.json
```
