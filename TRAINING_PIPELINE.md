# TRAINING — một script, một schema text

> Cập nhật **2026-07-12**. **Train:** [`scripts/train.py`](scripts/train.py) duy nhất.  
> Pipeline: **batch ingest** (TK_027) + **G1 batch** (B1). **Finalize KHÔNG còn là bottleneck**
> sau **6 fix perf** 2026-07-12/13 ([`TK_039`](TK_039_TRAIN_PERF.md), §0.1) — bài dài
> (`history_articles` 158 bài) từ **~22 phút → ~3 phút**. Fix #1–#4, #6 **byte-identical**;
> **fix #5** (hoãn skip_ledger, mặc định BẬT) đổi skip-span assignment tối đa ~26% trên corpus
> dict-nặng (0% domain lõi); **fix #6** gọn RAM bake (chống swap). Vẫn 1 process = 1 core.

---

## 0. Mang sang máy khác (train nhanh hơn)

> **Toàn bộ 3 fix perf (§0.1) nằm trong `src/`** — copy `src/` là mang theo. Không cần checkpoint cũ.

### Cần copy (~100 MB, không cần checkpoint cũ)

```text
nn-aivie/
  pyproject.toml
  src/                        # ← chứa fix perf 2026-07-12
  scripts/
  config/train_v10.yaml
  data/text_corpus/*.jsonl    # ~95 MB, 65.160 dòng
```

**Không bắt buộc:** `checkpoints/`, `out/`, `data/` legacy, file `.pkl` lớn.

### A. Trên máy NGUỒN — đóng gói (một lệnh)

```bash
cd nn-aivie
tar czf nn-aivie-train-pack.tgz \
  pyproject.toml src scripts config/train_v10.yaml data/text_corpus
# copy sang máy đích, ví dụ:
#   scp nn-aivie-train-pack.tgz user@host:~/
```

### B. Trên máy ĐÍCH — giải nén + cài + train (copy-paste cả khối)

```bash
mkdir -p nn-aivie && tar xzf nn-aivie-train-pack.tgz -C nn-aivie
cd nn-aivie
python -m venv .venv && source .venv/bin/activate    # Python ≥ 3.10
pip install -e .                                     # numpy, scipy, pyyaml, pyarrow
mkdir -p out/logs checkpoints
export PYTHONPATH=src

# Train full v10, chạy nền, log ra file (an toàn khi SSH đứt)
nohup python scripts/train.py \
  --manifest config/train_v10.yaml \
  --out checkpoints/network_v10.pkl \
  --bake > out/train_v10_full.log 2>&1 &

tail -f out/train_v10_full.log        # theo dõi; log chi tiết: out/logs/train_*.log
```

Muốn chạy cả test: `pip install -e ".[dev]"` (thêm pytest).

**Lưu ý:** Checkpoint chỉ ghi **một lần cuối** sau 8/8 dataset + bake. Giữa chừng không có `.pkl` tạm — nếu cần an toàn khi train lâu, dùng **train từng giai đoạn** (§7).

### Sau khi train xong — export audit (tùy chọn)

```bash
PYTHONPATH=src python scripts/export_grade_audit.py \
  --checkpoint checkpoints/network_v10.pkl \
  --prefix network_v10 \
  --no-bake
```

Xuất `out/network_v10_*.tsv` + `*_audit.json` + `*_verify.json`.

---

## 0.1. Tăng tốc finalize — 6 fix perf (2026-07-12/13)

> **Chi tiết đầy đủ: [`TK_039_TRAIN_PERF.md`](TK_039_TRAIN_PERF.md)** (chẩn đoán + đo + verify).
>
> **Chẩn (profile `history_articles` 45 bài = 1838s):** finalize bị chi phối bởi các hàm O(n²)
> trên bài dài, **không** phải index-rebuild. Tất cả **build-time thuần** — **không đụng `hca/`
> (hàm sinh)**, **không đổi format checkpoint**. **Fix #1–#4 byte-identical** (verify FULLSIG:
> clusters, cluster_of_pair, composite_count, bigram_count, links, chains, skip-spans khớp tuyệt
> đối). **Fix #5 KHÔNG byte-identical** — xem cảnh báo cuối mục.

| Fix | Hotspot | Cơ chế | Byte-identical? |
|-----|---------|--------|:---:|
| **#1** | `_rebuild_parent_links` (504s) | Hoãn rebuild link (seg_h) → **1 lần/cha** cuối `finalize_promotions` | ✅ |
| **#2** | skip_ledger `_right_at_slot` (~470s) | Hoist `flat` + prefix-sum khỏi vòng lặp đôi `si×sj` → O(len²·gap)→O(len·gap) | ✅ |
| **#3** | `_refresh_pair_occurrences_parent` (93%) | Bỏ lưu vị trí; membership `pair→{cha}` + quét on-demand; `composite_count` cập nhật ở biên merge | ✅ |
| **#4** | `materialize` re-scan (phase `gán skip_span_ids`) | Tái dùng `_parent_hits` cho cha vừa record cùng flush → khỏi scan lại (cắt 64–94% quét) | ✅ |
| **#5** | skip_ledger lặp mỗi dataset | **Hoãn record+materialize → 1 lần cuối train** (`defer_skip`, mặc định BẬT). O(dataset×cha)→O(cha) | ⚠️ **KHÔNG** |
| **#6** | `mine_frame_brackets` spike RAM (bake) | **2-pass**: pass1 đếm, lọc survivor, pass2 gom gaps/prov chỉ cho survivor → peak RAM vài GB→~1GB. Giá: 1.5× CPU (fix RAM, không phải tốc độ) | ✅ |

**Đo (history_articles, cùng máy, fix #1–#4 byte-identical):**

| Quy mô | Gốc | fix #1+#2 | **+ #3** |
|--------|-----|-----------|--------------|
| N=20 (back-to-back) | 865s | 559s | **43.5s** |
| N=45 | 1838s | ~500s | **~42s** |
| **Full 158 bài (grade+history)** | ~2.5–3h | (kẹt) | **~3.5 phút** |

**Verify byte-identical (fix #1–#4):** so FULLSIG checkpoint nhỏ trước/sau.
Nhanh: `pytest tests/test_seg_graph.py tests/test_pipeline_batch.py tests/test_skip_ledger.py -q` (21 pass).

### Fix #5 — hoãn skip_ledger (mặc định BẬT, KHÔNG byte-identical)

`skip_span_ids` **không được đọc giữa các dataset** → thay vì materialize skip_ledger **mỗi
dataset** (lặp lại trên số cha tăng dần = bậc-hai), hoãn **một lần ở cuối** (`run_manifest(defer_skip=True)`
→ `finalize_skip_ledger_deferred`). Tắt bằng `defer_skip=False`.

**Cái giá đã ĐO** (per-dataset vs deferred, cùng bộ dataset):

| Corpus | Kho span | Tổng gán | **Cha đổi skip_span_ids** |
|--------|:---:|:---:|:---:|
| grade+subject+history (domain lõi) | 626=626 | 3268=3268 | **0.0%** |
| + dictionary (phrase tăng mạnh) | 2248=2248 | 15959=15959 | **26.1%** |

→ **Kho span + tổng gán BẤT BIẾN** (không mất tín hiệu, chỉ phân bố lại); **0%** trên domain lõi;
26% chỉ ở **cha-từ-điển** khi phrase tăng nhiều. Non-skip state (chains/clusters/links) **MATCH tuyệt đối**.

**Ảnh hưởng chất lượng sinh:** per-parent `skip_span_ids` là tín hiệu **coverage phụ** trong
`rank_matches` (cha chứa skip-span của câu hỏi → +`_skip_idf`). Vì **guard QA/Para/Syn đo trên domain
lõi (skip 0%)** → guard **không đổi**; đáp lấy từ **cha-từ-điển** có thể xê dịch thứ hạng nhẹ ở ca sát
nút (theo hướng nhất quán hơn, inventory không mất). **Nợ:** chưa spot-check guard trực tiếp với fix#5 —
làm sau lần train full tới.

> **Trạng thái:** diff ở `src/nn_aivie/core/{seg_graph,skip_ledger,parent_store}.py` +
> `src/nn_aivie/training/{pipeline,frame_bracket}.py`. Copy `src/` sang máy khác là mang theo toàn bộ.

### Fix #6 — 2-pass FRAME_BRACKET (byte-identical, đánh đổi RAM lấy CPU)

Bake `mine_frame_brackets` giữ **~6.5M cặp thô × {count, gaps, prov, kinds}** cùng lúc → spike
**vài GB** ở đỉnh đồ thị → **swap** trên 24GB (làm treo bake, mất cả train). fix#6 tách 2-pass:
pass1 chỉ đếm → lọc survivor → pass2 gom gaps/prov chỉ cho survivor. **Peak vài GB→~1GB**,
FRAME_BRACKET edges + `hlink_meta` **byte-identical** (FBSIG khớp). **Giá: ~1.5× CPU** (2× scan)
— là fix RAM, không phải tốc độ; chỉ đáng khi máy đang swap (ca 24GB). Bake chạy 1 lần/train.

**⚠️ Vận hành:** `save_checkpoint(bake_derived=True)` chạy **bake TRƯỚC khi ghi `.pkl`** → bake
OOM = mất cả train. An toàn hơn: train **KHÔNG** `--bake` (ghi pkl ngay) + `caffeinate -i` (chặn
sleep), rồi **bake riêng** `bake_checkpoint_indices.py` (retry được). Xem TK_039 §5.

---

## 1. Schema corpus (duy nhất)

File `*.jsonl` trong `data/text_corpus/`:

- **Mỗi dòng** = một JSON **string** (một đơn vị train)
- Câu, đoạn, bài, khối Q&A đều là một string — không object, không mảng trên một dòng
- Xuống dòng trong đoạn = `\n` **trong** chuỗi JSON (một dòng vật lý file)

```text
"phở Việt Nam rất ngon ."
"Hỏi: Phở là gì ? Đáp: Phở là món truyền thống ."
"Đoạn một .\n\nĐoạn hai ."
```

Lúc train: `normalize_ws(text)` — `\n` → khoảng trắng → tokenize → một nơ-ron cha.

**Chuẩn hóa nguồn** (chỉ khi đổi data gốc, không cần trên máy đích nếu đã có `text_corpus/`):

```bash
PYTHONPATH=src python scripts/normalize_corpus.py
```

---

## 2. Lệnh train

### From scratch (full v10)

```bash
PYTHONPATH=src python scripts/train.py \
  --manifest config/train_v10.yaml \
  --out checkpoints/network_v10.pkl \
  --bake
```

### Train tiếp thêm một file

```bash
PYTHONPATH=src python scripts/train.py \
  --checkpoint checkpoints/network_v10.pkl \
  --out checkpoints/network_v10.pkl \
  --dataset data/text_corpus/histwiki_articles.jsonl \
  --priority 0.85 \
  --bake
```

### Tham số CLI (`scripts/train.py`)

| Tham số | Mặc định | Ý nghĩa |
|---------|----------|---------|
| `--manifest` | — | YAML danh sách dataset |
| `--checkpoint` | — | Nạp graph có sẵn (bỏ qua = graph trống) |
| `--out` | **bắt buộc** | Checkpoint đích |
| `--bake` | tắt | Mine skip spans + seq index + FRAME_BRACKET lúc **save** |
| `--eta` | 0.05 | Cường độ cạnh G1/W_seq |
| `--dim` | 96 | Chiều embedding E |
| `--max-texts` | 0 | Giới hạn dòng/file (0 = hết) |
| `-v` / `--verbose` | tắt | Log từng bài → `out/train_progress.jsonl` |
| `--log-every` | 0 | Verbose: mỗi N bài (file lớn: 20–50) |

**`--bake`:** lúc save gọi `bake_derived_indices` — skip spans, `seq_bi`/`seq_tri`, FRAME_BRACKET. Train lâu nên **bật** để load/generate sau không mine lazy. Có thể tắt khi debug ingest, bake sau:

```bash
PYTHONPATH=src python scripts/bake_checkpoint_indices.py checkpoints/network_v10.pkl --in-place
```

### Ngăn máy sleep lúc train (macOS)

Train lâu (4–8h) nên dùng `caffeinate -i` để máy không vào idle sleep:

**Bash / zsh (nohup + background):**
```bash
export PYTHONPATH=src
nohup caffeinate -i python scripts/train.py \
  --manifest config/train_v10.yaml \
  --out checkpoints/network_v10.pkl \
  --bake > out/train_v10_full.log 2>&1 &

tail -f out/train_v10_full.log
```

**PowerShell (background job):**
```powershell
$env:PYTHONPATH = "src"
Start-Job -ScriptBlock {
  Set-Location /path/to/nn-aivie
  $env:PYTHONPATH = "src"
  caffeinate -i .venv/bin/python scripts/train.py `
    --manifest config/train_v10.yaml `
    --out checkpoints/network_v10.pkl `
    --bake *> out/train_v10_full.log
} | Out-Null

# Theo dõi:
Get-Content out/train_v10_full.log -Wait -Tail 20
```

**Tùy chọn caffeinate:**
- `-i` (idle) — ngăn máy vào idle sleep (khuyến nghị)
- `-s` (system) — ngăn cả system sleep (tình huống đặc biệt)

---

## 3. Manifest v10

[`config/train_v10.yaml`](config/train_v10.yaml) — thứ tự nạp:

| # | File | Dòng | priority | Ghi chú |
|---|------|------|----------|---------|
| 1 | `grade_lessons.jsonl` | 476 | 0.95 | SGK lớp 3–7, nền |
| 2 | `subject_articles.jsonl` | 25 | 0.92 | Bài tổng hợp môn |
| 3 | `history_articles.jsonl` | 158 | 0.88 | Bài sử dài — **finalize chậm** |
| 4 | `dictionary.jsonl` | 26.244 | 0.6 | Nạp text nhanh; finalize lâu |
| 5 | `doc_add_articles.jsonl` | 2.680 | 0.6 | |
| 6 | `conv_stories.jsonl` | 9.917 | 0.6 | |
| 7 | `paragraph_units.jsonl` | 25.000 | 0.36 | **finalize lâu** |
| 8 | `qa_supervised.jsonl` | 660 | 0.95 | **Nạp cuối** — Hỏi/Đáp ghi đè |

**Tổng:** ~65.160 đơn vị.

Tham số trong YAML (ngoài `path`):

| Tham số | Mặc định | Ý nghĩa |
|---------|----------|---------|
| `priority` | 0.6 | Trọng số parent |
| `eta` | 0.05 | G1/W_seq |
| `max_texts` | 0 | Giới hạn dòng/file |
| `split_punct` | true | Tách `,` `.` thành token riêng (R0) |

---

## 4. Theo dõi tiến trình

### Log mặc định (không `-v`)

- **Stdout / nohup:** `out/train_v10_full.log` (nếu redirect)
- **Logger:** `out/logs/train_<YYYYMMDD_HHMMSS>.log` — luôn ghi từng phase

Dòng điển hình:

```text
▶ [4/8] dictionary.jsonl priority=0.6
ingest_text 26000/26000 (100.0%) — n=26000
    ISG: promote 500 clusters=...
    skip_ledger record_parent ...
✓ ingest dictionary.jsonl  (....s)
```

**Im lặng sau `ingest_text 100%` là bình thường** — đang `finalize_batch_ingest` (ISG heap + skip ledger). CPU cao, log chưa nhảy. Sau fix perf (§0.1) pha này **nhanh hơn nhiều** nhưng vẫn là pha nặng nhất.

### Verbose (debug / file nhỏ)

```bash
PYTHONPATH=src python scripts/train.py \
  --manifest config/train_v10.yaml \
  --out checkpoints/network_v10.pkl \
  -v --log-every 25
```

→ `out/train_progress.jsonl` (JSON từng bài: cha#, slots, cụm).

### Thời gian tham khảo (MacBook, 1 process CPU ~100%)

**Sau fix perf 2026-07-12 (§0.1)** — đo trực tiếp, mạng byte-identical:

| Giai đoạn | Trước (2026-07-11) | **Sau fix (2026-07-12)** |
|-----------|--------------------|--------------------------|
| `grade_lessons` (476) | ~24s | **~6s** |
| `subject_articles` (25) | ~9s | ~2s |
| `history_articles` (158) | **~1h45** | **~3 phút** |
| `dictionary` (26k) | finalize >1h | *chưa đo lại — cùng cơ chế, kỳ vọng giảm mạnh* |
| `paragraph_units` (25k) | tương tự dictionary | *chưa đo lại* |
| **Full v10** | ~4–8h | *chưa đo lại — re-measure trên máy đích* |

> `dictionary`/`paragraph_units`/full v10 **chưa đo lại** với fix mới (tôi mới verify trên
> grade+history). Cùng đường finalize nên kỳ vọng giảm tương tự, nhưng **hãy đo thực tế** trên
> máy đích thay vì tin ước lượng cũ.

Máy nhiều core / server: một process Python vẫn chủ yếu 1 core — cần máy **GHz cao + RAM đủ** (graph lớn ~500MB–2GB+ RAM lúc finalize).

---

## 5. File text_corpus

| File | Đơn vị / dòng | Nguồn |
|------|----------------|-------|
| `grade_lessons.jsonl` | Một bài (Môn / Lớp / Tiết / Bài) | `GRADE_KNOWLEDGE` + parquet |
| `subject_articles.jsonl` | Một bài tổng hợp / môn / lớp | parquet môn học |
| `history_articles.jsonl` | Một bài sử (curated + wiki) | `history_vi` + wiki para |
| `dictionary.jsonl` | Một lemma (nghĩa + ví dụ) | `dict/vi_dict_raw.jsonl` |
| `doc_add_articles.jsonl` | Một bài kiến thức | `statements_doc_add.jsonl` |
| `conv_stories.jsonl` | Một truyện / hội thoại | `data/conv/` |
| `paragraph_units.jsonl` | Đoạn 2–4 câu (phụ) | doc + conv |
| `qa_supervised.jsonl` | Hỏi/Đáp tự nhiên | QA thủ công |

File cũ (`dictionary_examples.jsonl`, `grade_3_7.jsonl`, …) — **không** dùng trong manifest v10.

Logic chuẩn hóa: [`src/nn_aivie/training/normalize_corpus.py`](src/nn_aivie/training/normalize_corpus.py).

---

## 6. Pipeline train (code)

[`src/nn_aivie/training/pipeline.py`](src/nn_aivie/training/pipeline.py):

```text
for each dataset in manifest:
    begin_g1_batch()
    begin_ingest_batch(defer_index=True)
    for each line in jsonl:
        normalize_ws → tokenize (split_punct)
        g1_and_wseq (buffered) + add_example(defer_index=True)
    flush_g1_batch()
  finalize_batch_ingest(defer_skip=True):   # ← đã tối ưu §0.1
    ISG promote (heap batch)         # fix #1: hoãn rebuild link; fix #3: membership + on-demand pos
    # skip_ledger HOÃN (fix #5) — không chạy per-dataset nữa
finalize_skip_ledger_deferred:       # ← MỘT lần cuối (fix #5)
    record_parent × N cha (fix #2 prefix-sum)
    materialize + gán skip_span_ids (fix #4: reuse _parent_hits, khỏi scan lại)
finalize_graph: slot_hubs + embedding E
save checkpoint (+ bake nếu --bake)
```

**Khác biệt so ingest online cũ:** ít ISG cluster hơn (~1148 vs ~1773 trên `grade_lessons` đơn lẻ) do promote batch theo heap; **G1 edges giữ nguyên**; skip ledger materialize đầy đủ hơn.

Module liên quan:

| File | Vai trò |
|------|---------|
| `src/nn_aivie/training/g1_batch.py` | Buffer CO_ACTIVATION + SEQUENTIAL |
| `src/nn_aivie/training/pipeline.py` | `run_manifest` — **fix #5 (`defer_skip`) + gọi `finalize_skip_ledger_deferred`** §0.1 |
| `src/nn_aivie/core/parent_store.py` | `finalize_batch_ingest(defer_skip)`, `finalize_skip_ledger_deferred` — **fix #4 (reuse) + #5 (hoãn)** §0.1 |
| `src/nn_aivie/core/seg_graph.py` | ISG promote/rewire — **fix #1 (hoãn link) + #3 (membership on-demand)** §0.1 |
| `src/nn_aivie/core/skip_ledger.py` | Skip span — **fix #2 (prefix-sum) + #4 (materialize reuse)** §0.1 |
| `scripts/export_grade_audit.py` | Export TSV/JSON + verify sau train |

---

## 7. Train từng giai đoạn (checkpoint giữa chừng)

Train full một lần **không** lưu giữa các file. Để tránh mất giờ khi crash, nạp tuần tự và **save sau mỗi nhóm**:

```bash
export PYTHONPATH=src

# Giai đoạn 1 — nền SGK + sử
python scripts/train.py \
  --dataset data/text_corpus/grade_lessons.jsonl --priority 0.95 \
  --dataset data/text_corpus/subject_articles.jsonl --priority 0.92 \
  --dataset data/text_corpus/history_articles.jsonl --priority 0.88 \
  --out checkpoints/v10_s1.pkl --bake

# Giai đoạn 2 — từ điển + bổ trợ
python scripts/train.py --checkpoint checkpoints/v10_s1.pkl \
  --dataset data/text_corpus/dictionary.jsonl --priority 0.6 \
  --dataset data/text_corpus/doc_add_articles.jsonl --priority 0.6 \
  --dataset data/text_corpus/conv_stories.jsonl --priority 0.6 \
  --out checkpoints/v10_s2.pkl --bake

# Giai đoạn 3 — đoạn + QA
python scripts/train.py --checkpoint checkpoints/v10_s2.pkl \
  --dataset data/text_corpus/paragraph_units.jsonl --priority 0.36 \
  --dataset data/text_corpus/qa_supervised.jsonl --priority 0.95 \
  --out checkpoints/network_v10.pkl --bake
```

Hoặc từng file một với cùng `--checkpoint` / `--out`.

---

## 8. Script cũ & eval

`build_*.py`, `train_v3.py`, `train_from_scratch_tk036.py` — **legacy**. Dùng `normalize_corpus.py` + `train.py`.

```bash
PYTHONPATH=src python scripts/bake_checkpoint_indices.py checkpoints/network_v10.pkl --in-place
PYTHONPATH=src python -u scripts/eval_discourse_prompts.py --checkpoint checkpoints/network_v10.pkl
```

Test nhanh pipeline batch:

```bash
PYTHONPATH=src pytest tests/test_pipeline_batch.py tests/test_g1_batch.py -q
```

---

## 9. Module

| File | Vai trò |
|------|---------|
| `scripts/train.py` | Entry train duy nhất |
| `scripts/normalize_corpus.py` | Legacy → text_corpus |
| `scripts/export_grade_audit.py` | Audit TSV/JSON sau train |
| `src/nn_aivie/training/text_corpus.py` | Đọc/ghi jsonl string |
| `src/nn_aivie/training/dataset_spec.py` | Manifest YAML |
| `src/nn_aivie/training/pipeline.py` | Ingest + finalize |
| `src/nn_aivie/training/checkpoint_bake.py` | Bake skip / seq / FRAME_BRACKET |
