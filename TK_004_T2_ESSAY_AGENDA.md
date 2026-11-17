# TK_004 — Trụ 7 T2: Essay agenda từ gold dài vi-alpaca (t1 SFT → sinh bài nhiều đoạn)

| Field | Value |
|-------|-------|
| **ID** | TK_004 |
| **Title** | Nạp gold dài vi-alpaca thành t1 SFT; sinh bài dài bám agenda t1; closure khi phủ đủ ý |
| **Status** | **Implemented 2026-07-02** — mục 5 Kết quả. **Superseded 2026-07-09:** `generate_essay` + eval essay đã gỡ (TK_019 §7); essay agenda không còn đường sinh riêng — dùng `generate(multi=True)` + discourse eval. |
| **Repo** | `nn-aivie`, nhánh `cursor2` |
| **Nguồn** | Trụ 7 (ý tưởng gốc tác giả) — `docs/NN-AIVIE v4.md`; substrate T1 đã xong (TK_002 mục 7) |
| **Liên quan** | `AbstractNeuron`, `bind_episode_by_resonance`, `bkai-foundation-models/vi-alpaca` |

---

## 1. Đề bài

Đóng nốt **nửa sau ý tưởng t1**: gold SFT dài → t1; prompt bài-dài → **đi lên t1** (khớp) →
**phân rã xuống** (agenda = member theo thứ tự) → sinh bài nhiều đoạn, **dừng khi phủ đủ ý**
(= A4 agenda/closure tái sinh trên nền v3, bỏ QFocus/Rule v2).

**Định vị trung thực (chống thổi phồng):** T2 phase-1 là *truy hồi kế hoạch diễn ngôn + ghép
segment* — verbatim_rate mức essay sẽ ~cao; giá trị mới nằm ở **chọn ý + thứ tự + closure từ
cấu trúc t1**, không phải sinh token-level (đó là Trụ 4). Phải ghi rõ trong docs khi báo số.

## 2. Thiết kế

1. **Ingest** (`ingest_hf_vi.py --source vi_alpaca`): instruction + output từ
   `bkai-foundation-models/vi-alpaca` (datasets-server /rows, không gated). Lọc: output đủ dài
   (≥3 câu qua `is_quality_vi_sentence`), instruction là lệnh viết dài (tự nhiên qua độ dài
   output — KHÔNG regex từ khóa "viết/blog", Luật 0 áp cho cả data filter heuristic từ vựng).
   → `data/real_vi/essays.jsonl`: `{instruction, sentences[], title}`.
2. **Thí nghiệm rẻ trước** (nếp T0): 200 bài → đo % câu qua gate + % câu cộng hưởng parents
   big_a2 (`rank_matches` bag-level) → mới quyết build full.
3. **SFT t1** (`AbstractNeuron` + field mới `match_ids`): câu bài luận ingest làm statement
   parent **priority thấp (0.5, ngoài miền — không lọt pool curated two-tier)**; t1 = member
   xuất xứ THEO THỨ TỰ + member cộng hưởng từ tri thức cũ (np1/np2 — cơ chế tác giả);
   `match_ids` = token instruction (chìa khớp). Build trên `_t1` → `grade_v3_big_a2_t2.pkl`.
4. **Sinh** (`generate_essay`, opt-in — KHÔNG đụng generate_hybrid/composed mặc định):
   query → khớp t1 SFT (IDF coverage trên match_ids, tái dùng nguyên tắc rank_matches) →
   agenda = members thứ tự → ghép segment (completion từng member) → closure khi phủ đủ /
   trần segment; không khớp → fallback `generate_composed`.
5. **Eval** (`eval_essay`): gold = **instruction PARAPHRASE** của bài đã nạp (đo khái quát
   hóa khớp — không chấm verbatim trên chính instruction train). Metric: coverage nhóm ý +
   ordering (thứ tự ý theo gold) + on_topic. ~8 ca đầu.

## 3. Acceptance

- Thí nghiệm rẻ có số (gate-rate, resonance-rate) ghi vào ticket TRƯỚC khi build full.
- Essay eval: coverage ≥ 60% + ordering ≥ 60% trên bộ paraphrase đầu (baseline mới — chưa có
  số cũ để so; số thật quan trọng hơn ngưỡng).
- **Guard bất di bất dịch:** trên `_t2`: QA val 100, Para ≥88, Syn ≥50, composition ≥64/·/83,
  anaphor ≥50 (baseline TK_002); `grade_v3.pkl`/`grade_v3_big_a2.pkl` không đụng tới.
- pytest xanh (≥270); test toy cho match_episode/generate_essay/serialization match_ids.

## 4. Không làm

- KHÔNG token-level AR cho essay (Trụ 4). KHÔNG re-ingest wiki. KHÔNG đổi default flags
  ngoài cờ essay mới. KHÔNG train lại E.

---

## 5. KẾT QUẢ (2026-07-02, Claude implement)

**Thí nghiệm rẻ (đèn xanh trước build):** 200 bài đầu — 55/60 bài cộng hưởng tri thức cũ
(TB 18.5 member), vocab overlap 99%, văn sạch (thêm backtick vào markup filter `quality.py`).

**Build:** 500 essay → `grade_v3_big_a2_t2.pkl`: +3 680 cha essay (priority 0.5, ngoài miền),
500 episode SFT, **2 080 member cộng hưởng (TB 4.2/bài — cơ chế np1/np2 chạy thật)**.
Build ~60 phút (nợ hiệu năng: resonance scan chưa index).

**Essay eval lần đầu (gold 8 ca instruction-PARAPHRASE):**

| Metric | Số | Ghi chú |
|---|---|---|
| coverage | **67%** | 5/8 ca hoàn hảo (cov 100 + ord 100) |
| **ordering** | **75%** | thứ tự ý bám đúng agenda t1 |
| on_topic | 41% | metric mượn từ composition — không hợp essay (segment hợp lệ không lặp từ chủ đề); cần metric riêng |

3 ca fail đã chẩn đoán: (1) "đọc sách" khớp nhầm bài khác qua token instruction phổ biến
("lợi/ích") → match_episode cần đơn vị CỤM; (2) "căng thẳng" khớp bài chiến tranh (đa nghĩa —
cùng trần "trị/giá trị" T0); (3) "tập thể dục" đáp ĐÚNG chủ đề nhưng từ bài thể dục KHÁC trong
pool → artifact gold khóa-bài (đáp thật tốt).

**Guard (kèm ĐỔI LUẬT sổ nợ — fuzzy/order2/episode/pref BẬT):**

| Checkpoint | QA | Para | Syn | Comp | Anaphor |
|---|---|---|---|---|---|
| `_t2` | 100 | 88 | **67** (+17 nhờ fuzzy) | — | 50 |
| big_a2 | 100 | 88 | **67** | 64/2/83 ✓ | 50 |
| grade_v3 | 100 | **62** (−13, giá fuzzy corpus nhỏ — sổ nợ) | 50 | 93/0/100 ✓ | 100 |

**274 test pass.** Còn lại → việc tiếp §7.5 v4 doc: match_episode theo cụm; metric on_topic
riêng cho essay; gold artifact 2 ca; đa nghĩa instruction.

**Bổ sung UI (2026-07-02):** mode **Essay (bài dài t1)** trong Parent Explorer —
`export_graph_v3.completion_graph(mode="essay")`: node `episode` (tím hồng) + chuỗi member
theo agenda + cạnh query→t1 ("đi lên") ; panel hiện t1 khớp (score/title/member/cộng hưởng)
+ bài trả lời; không khớp → báo fallback Compose. Files: `viz/server/main.py`,
`src/nn_aivie/viz/export_graph_v3.py`, `viz/web/src/{api.ts,GraphExplorer.tsx}` (đã build dist).
