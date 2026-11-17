# TK_108 — Gemma 4 on the M4: Vietnamese · office · tools (self-graded)

*06/11/2026 · Ollama · `gemma4:12b` (7.7 GB) and `gemma4:26b-a4b` (16 GB), thinking mode on (default), `MAXNEW=6144`, `CTX=16384`.*
*English translation of the Vietnamese original in the repository root.*

*The task set is 28 realistic cases I wrote myself (`data/eval_clean/real_vto_28.json`). I read every answer and scored it 0–10, the same grading as the [22 code cases](TK_108_DANH_GIA_GEMMA4_THUC_TE.md). For tools, the tool results are pre-written and supplied in the second turn.*

## Summary

| | Gemma 4 12B | Gemma 4 26B-A4B |
|---|---|---|
| Vietnamese (12 cases) | 7.9 | **8.8** |
| Office (9 cases) | 9.3 | **9.4** |
| Tools (7 cases) | 9.7 | **10** |
| **Overall (28 cases)** | **8.8** | **9.3** |
| Code (22 cases, previous run) | 9.3 | 9.8 |
| Time per turn / speed | ~43 s · ~27 tokens/s | **~22 s · ~55 tokens/s** |

- **Tools:** both versions are nearly perfect. Right names, right arguments, they resolve dates themselves ("tomorrow", "this Sunday"). They don't call tools when not needed, ask back when an email address is missing, make 2 calls in parallel, write correct SQL, and answer correctly from the tool results.
- **Office:** correct Excel formulas (COUNTIFS, TEXTAFTER, the name-splitting trick for old Excel, NETWORKDAYS.INTL code 11). Proper VBA, correct arithmetic on a sales table. Meeting minutes and official letters have the full national header, motto and document number; the 26B version even cites Decree 30/2020 correctly.
- **Vietnamese:** very natural writing. Consistent "bác–cháu" (elder–younger) forms of address over 2 turns, a Southern accent with "nè, nhen, á". Correctly fixes dành/giành, điều/đều, xung/sung, kỷ/kỹ confusions; diacritic restoration is perfect.
- **Two real weaknesses:**
  1. **Stale knowledge.** Both say Vietnam has 63 provinces; the correct number has been 34 since 1/7/2025. The models' knowledge predates the merger.
  2. **The 12B hallucinates.** It fluently summarised "Dòng sông không tên" by Nguyễn Huy Thiệp, a work that does not exist. The 26B clearly says there is no such work.
- **Surprise:** the 26B-A4B is **twice as fast** as the 12B on the M4. The reason is that 26B-A4B only runs about 4B parameters per token, while 12B runs all 12B. If RAM allows (about 16–18 GB), **26B-A4B is better in every respect**.

---

## Scores

### Vietnamese

| Case | 12B | 26B | Comments |
|---|---|---|---|
| v01 leave-request email to the boss | 9 | 9 | Polite, correct "em – anh Tuấn" address, mentions handing over work. Offers 3 versions, a bit long for "don't make it too long". |
| v02 summarise in 3 points | 8 | 8 | Exactly 3 points, right main ideas, but drops the numbers (34 provinces, 3,321 communes). The 12B writes "will operate", wrong tense. |
| v03 translate EN → VI (business news) | 9 | 9 | Good translation ("biên lợi nhuận bị thu hẹp", "lạc quan thận trọng"). Only "đơn hàng tồn đọng" has a negative connotation, not quite "backlog". |
| v04 translate VI idioms → EN | 8 | 10 | Both use "last minute / eleventh hour / headless chickens". The 12B has "left it until the wire", which is not standard English (should be "down to the wire"). |
| v05 explain 3 idioms | 10 | 10 | Correct meanings, sensible everyday examples. |
| v06 spelling fixes | 10 | 10 | Fixes all 4 errors and explains dành/giành correctly. |
| v07 restore diacritics in a message | 10 | 10 | Perfect. |
| v08 current number of provinces | **2** | **2** | Answers "63 (58 provinces + 5 cities)". Wrong: since 1/7/2025 there are only 34. |
| v09 compound vs simple interest | 10 | 10 | 119,101,600 VND and 18,000,000 VND interest, both correct, with a comparison table. |
| v10 Facebook cake-sale post, Southern voice | 9 | 9 | Natural voice, correct prices and combos, under 120 words. A few slightly forced phrases ("tiết kiệm tốn luôn"). |
| v11 made-up literary work | **1** | **10** | The 12B invents a long summary. The 26B says clearly that the work does not exist and suggests possible mix-ups. |
| v12 "cháu – bác" conversation, 2 turns | 9 | 9 | Consistent "bác" role, correct content (green/brown ratio, aerobic composting, odour control). A bit long. |

### Office

| Case | 12B | 26B | Comments |
|---|---|---|---|
| o01 COUNTIFS for March 2026 | 10 | 8 | 12B uses `">="&DATE(2026,3,1)`, correct on any machine. 26B writes the date as text `">=01/03/2026"`, which may be misread on machines with US date format (it also gives a correct SUMPRODUCT version). |
| o02 split surname / given name | 10 | 10 | `TEXTAFTER(…," ",-1)` and the old `TRIM(RIGHT(SUBSTITUTE(…REPT(" ",100)),100))` both correct. |
| o03 working days, Sundays off only | 10 | 10 | `NETWORKDAYS.INTL(A2,B2,11,E2:E10)` correct, code 11 = Sunday only. |
| o04 meeting minutes, official format | 7 | 9 | All sections present. The 12B breaks the header table (date repeated in both columns). The 26B formats it better and cites Decree 30/2020. |
| o05 automatic table of contents in Word | 10 | 10 | Headings, then References → Table of Contents, then Update Field. Correct and complete. |
| o06 8-slide outline | 9 | 9 | All 8 slides, 3 points each with chart suggestions, sticks to the figures (+15%, −3%). |
| o07 calculations on a sales table | 10 | 10 | Totals 380/345/295/233, correct ranking, Dũng +63.3% strongest, Chi −40%. Sensible commentary. |
| o08 VBA: highlight rows with stock < 10 | 10 | 10 | Finds the last row, checks the sheet exists, colours or clears whole rows, disables screen updating while running. |
| o09 debt-collection letter | 8 | 9 | Polite and firm, amount in words, cites the right contract number. The 12B uses document number "…/2024", inconsistent with 2026. |

### Tools

| Case | 12B | 26B | Comments |
|---|---|---|---|
| t01 "tomorrow" in Đà Nẵng | 10 | 10 | `get_weather(Đà Nẵng, 2026-11-07)`. |
| t02 find the cheapest ticket then email (2 steps) | 10 | 10 | Searches, then emails choosing the right Vietjet 13:30 flight at 1,290,000. |
| t03 question needing no tool | 10 | 10 | Answers "Tokyo" directly. |
| t04 missing email address | 10 | 10 | Asks for the email address instead of inventing one. |
| t05 two cities | 10 | 10 | Two parallel calls. |
| t06 tool returns an error | 8 | 10 | Both correctly compute "Sunday" as 08/11. 26B calls "Sài Gòn", gets the error and retries with "Ho Chi Minh City" — very sensible. 12B already called "Ho Chi Minh City" in the first turn, so my fake error message didn't match. It retried with "Saigon", still fine but without explanation. |
| t07 SQL then answer | 10 | 10 | Writes 2 correct SQL queries (SUM; GROUP BY + ORDER BY + LIMIT 3), then answers with the right numbers. |

---

## Conclusions and next steps

1. **Usable right away** for tools and office work in both versions. If RAM allows, use **26B-A4B**: higher scores, twice as fast, fewer hallucinations.
2. **Two gaps to close for a real Vietnamese assistant:**
   - **Up-to-date knowledge about Vietnam:** administrative boundaries after 1/7/2025, new laws, prices. Fine-tuning cannot fix this at the root. Better to **add RAG** over our own document store (BM25/viwiki already exists), plus a few sources of new legal texts.
   - **Hallucination in the 12B:** if 12B is chosen as the base, the LoRA needs "admit you don't know" data (the old unanswerable set exists). The 26B already declines correctly on its own.
3. **Both are verbose.** They often offer 2–3 options when the user asked for one. A style LoRA can fix this easily.
4. **Direction:** continue with option 1 (Vietnamese LoRA on Gemma 4), preferring the 26B-A4B base, with RAG for new knowledge. No need to convert to Mamba (see [plan P0](TK_108_KE_HOACH_CHUNG_CAT_GEMMA_MAMBA.md)).

## Re-running

```bash
MODEL=gemma4:26b-a4b OUT=out/real_vto_gemma4_26b_a4b MAXNEW=6144 CTX=16384 python3 scripts/real_use_eval_ollama.py data/eval_clean/real_vto_28.json
MODEL=gemma4:12b OUT=out/real_vto_gemma4_12b MAXNEW=6144 CTX=16384 python3 scripts/real_use_eval_ollama.py data/eval_clean/real_vto_28.json
```
