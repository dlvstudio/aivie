# TK_108 — Real-use evaluation of Gemma 4 26B-A4B on the M4 (code focus)

*06/11/2026 · Ollama 0.40 · `gemma4:26b-a4b` (MoE: 26B parameters, ~4B active per token; 4-bit, 16 GB) · same 22 cases and same grading as [c6l1](TK_108_DANH_GIA_C6L1_THUC_TE.md) and [Qwen2.5-Coder](TK_108_DANH_GIA_QWEN_CODER_THUC_TE.md)*
*English translation of the Vietnamese original in the repository root.*

## Summary

| | Gemma 4 26B-A4B |
|---|---|
| **Average score** | **9.8 / 10** |
| Usable as-is (≥ 7 points) | **22 / 22** |
| Partly right (4–6) | 0 |
| Broken (0–3) | 0 |
| Time per turn | **~47 seconds** (about 25–35 seconds of thinking first, then the answer) |
| Generation speed | ~30–65 tokens/second |
| RAM needed | ~17–18 GB (runs on the 24 GB M4, but heavy apps should be closed) |

- **This is the best model of all those tried on the M4.** 22/22 cases met the requirements, and every detailed constraint was handled.
- Examples of constraints it got right: "+84", "keep 7 copies", "at most 3 requests" (with a semaphore), "404 on delete", "de-duplicate while keeping order", and the review caught both SQL injection and plain-text passwords.
- All the code I ran was **100% correct**: 7 snippets executed, 2 pytest suites (12/12 and 3/3 tests passed).
- **The price:** the model is slower because it always "thinks" before answering, and answers are long (~2000 tokens per turn). The token limit must be set high (≥ 6144), otherwise answers get cut off mid-way (see section 3).

---

## 1. Method

- **Task set:** `data/eval_clean/real_code_22.json`, 22 Vietnamese cases, including one 3-turn conversation.
- **Generation settings:** same as the other models — temperature 0.7, top_p 0.9, repetition penalty 1.1. The difference: context 16,384 and a **6144-token** limit for the 11 cases that had to be re-run; reason in section 3.
- **Script:** `scripts/real_use_eval_ollama.py`. New environment variables `MAXNEW`/`CTX` raise the limits; the model's thinking is saved in the `thinking` field. Raw answers are in `out/real_eval_gemma4_26b/`.
- **Grading:**
  - I read every answer and scored it 0–10 on: meets the request, code runs, explanation is not wrong.
  - Pure computational code was actually run on the Mac, in a temporary directory, with a time limit.
  - Code touching the network or shell was only read, not run.

## 2. Scores

| # | Case | Score | Comments |
|---|---|---|---|
| 01 | Vietnamese phone number | **10** | Regex `^(\+84\|0)(3\|5\|7\|8\|9)[0-9]{8}$` correct. Ran 15 cases: `+84…`, too few digits, too many, `+840…` and whitespace all handled correctly. |
| 02 | pandas revenue | 9 | All correct, and it converts `Period` to string so the Excel export does not fail. One catch: it **overwrites `sales.csv` with sample data**; there is a warning in a comment, but running it as-is on real data would destroy the file. |
| 03 | mutable default `[]` bug | **10** | Explains the root cause correctly, fixes with `None`, includes an example. Ran: `['trà']`, `['cà phê']`. |
| 04 | KeyError 'email' | **10** | Gives 3 options (`.get`, `if`, fix the data) and sound advice. |
| 05 | SQL top 5 in Hanoi | **10** | Has both total amount and order count, GROUP BY id with an explanation of why. Filters on a date range so an index can be used. |
| 06 | search debounce | 8 | Textbook debounce, the HTML page runs immediately. But to make it runnable it **switched to a fake API (jsonplaceholder)** instead of `/api/search` as asked, and has no `encodeURIComponent`. |
| 07 | login page | **10** | Nice, mobile-friendly (16px input text so iPhone doesn't auto-zoom), Vietnamese text, has "Forgot password". |
| 08 | FastAPI todo | **10** | Separate Create/Update/Out models, auto-increment id, PATCH status, **404 for both update and delete**, delete returns 204. |
| 09 | email regex with de-dup | **10** | Uses `dict.fromkeys`. Ran: correct. |
| 10 | write pytest | **10** | Parametrize, edge cases, property checks (sum, length, difference ≤ 1), negatives and error messages. Ran: **12/12 passed**. |
| 11 | O(1) LRUCache | **10** | Hand-written doubly linked list with sentinel nodes at both ends. Ran: correct on a LeetCode-style sequence. |
| 12 | bash backup, keep 7 copies | **10** | `ls -t … \| tail -n +8 \| xargs rm -f` correct, checks tar's exit code, has `mkdir -p`, suggests cron. |
| 13 | Excel | **10** | Both VLOOKUP and XLOOKUP, IFERROR, `=B2*C2`, `SUMIF(A:A,"SP0*",D:D)`, with a summary table. |
| 14 | Python to JS | **10** | No external libraries, strips punctuation with an equivalent regex. Ran: `[['mèo',3],['ăn',2],['cá',1]]`, matches the Python version. |
| 15 | SQLite review | 9 | Caught SQL injection (with an attack example), plain-text passwords (move to bcrypt) and the unclosed connection. One small error: `with sqlite3.connect()` does **not** actually close the connection, it only commits. |
| 16 | 3-turn conversation | **10** | Three connected turns: function, then type hints and error handling, then tests with `tmp_path`. Ran: **3/3 passed**. |
| 17 | number-guessing game | **10** | Input, error handling, 1–100 bounds, correct turn counting, high score and replay. |
| 18 | git: committed `.env` by mistake | **10** | `reset --soft HEAD~1`, then `.gitignore`, `rm --cached`, commit again. Correct and safe, plus a `.env.example` suggestion. |
| 19 | asyncio + aiohttp | **10** | `Semaphore(3)`, `ClientTimeout(10)`, gather, prints failing URLs and separates error types (timeout, status, DNS). |
| 20 | set vs list | **10** | Everyday-life explanation, O(1) hash table vs O(n), timing on 10 million elements. |
| 21 | refactor discounts | **10** | Rule table (threshold, factor), Strategy pattern for complex cases, with a test. |
| 22 | React todo | **10** | Hooks and form submit, timestamp ids, toggle without mutating state, delete and a count of open items. |

---

## 3. Important notes for use

1. **The token limit must be high.**
   - Gemma 4 thinks before answering; this costs about 500–2500 tokens, and Ollama counts it towards `num_predict`.
   - In the first run with a 2048 limit, **11/22 cases were cut off mid-code**. In a UI or the API, set `num_predict` to 6144 or more and context to 16K or more.
2. **Slower than Qwen-Coder.** About 47 seconds per turn on average vs 5–20 seconds. If you only need quick answers you can disable thinking (`"think": false` in the API), but quality with thinking disabled was not measured.
3. **Long answers.** Answers usually include explanations, tables, how-to-run instructions and tips. Great for learning and understanding, verbose if you only want code.
4. **RAM:** the model takes about 16 GB. On the 24 GB M4 it should run alone, not alongside the c6l1 UI or other models.
5. **Still review the output.** Three small deductions: overwriting the sample `sales.csv` (case 02), using a fake API instead of the real endpoint (case 06), and wrongly saying `with` closes the sqlite connection (case 15). No logic errors.

## 4. Conclusion

- **Gemma 4 26B-A4B is usable as a high-quality Vietnamese coding assistant right on the M4.** For everyday tasks this is close to a commercial assistant.
- It does the kind of work asked, keeps every detailed constraint, and explains correctly.
- **Suggested usage:**
  - Use **Gemma 4 26B-A4B** for work that must be right: writing modules, review, tests, explanations.
  - Use **Qwen2.5-Coder 7B** for quick suggestions.
- **Direction for the project:**
  - Gemma 4 was already the teacher in earlier rounds (the 31B version on the box). This result shows the Gemma 4 line itself is a good base for our Vietnamese assistant.
  - We could add document RAG and a self-run-code loop. Or LoRA fine-tune the 26B-A4B on the Vietnamese data we already have (teacher loop, office, tool), instead of continuing to raise the 2.5B model from scratch.

## Appendix — re-running

```bash
ollama pull gemma4:26b-a4b
MODEL=gemma4:26b-a4b OUT=out/real_eval_gemma4_26b MAXNEW=6144 CTX=16384 python3 scripts/real_use_eval_ollama.py data/eval_clean/real_code_22.json
```
