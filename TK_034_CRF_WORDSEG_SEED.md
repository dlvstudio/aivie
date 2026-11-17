# TK_034 — CRF WordSeg → seed_pairs (graph-native clusters)

| Field | Value |
|-------|-------|
| **ID** | TK_034 |
| **Status** | 🟡 IMPLEMENT — train pipeline self-contained đã chạy, đang train full model |
| **Liên quan** | TK_013 (ISG), TK_033 (seed_pairs), TK_032 (underthesea study) |
| **Luật 0** | CRF chỉ chạy **offline/build**. Runtime `generate()` chỉ đọc cluster đã build. |

## 1) Mục tiêu

TK_033 S1 đã nâng coverage gold (underthesea) trên `history_vi.jsonl` từ **22.8% → 86.5%** bằng
dictionary+titles seed. Còn ~13.5% miss: cụm hiếm / ngoài từ điển / biến thể.

Mục tiêu TK_034: thêm nguồn seed_pairs từ **một model word segmentation kiểu underthesea (CRF B-W/I-W)**
— nhưng ta có thể:
- dùng model sẵn (baseline), và/hoặc
- tự train lại theo recipe underthesea (VLSP/UTS) rồi swap model artifacts.

Đầu ra vẫn là **`seed_pairs` (cặp âm-tiết liền kề)** để ISG promote cluster như TK_033 (không đổi
kiến trúc slot-chain).

## 2) Tại sao không “chưng cất” trực tiếp underthesea?

Chưng cất output underthesea là đủ cho S2, nhưng tác giả muốn “học cách train” để:
- có thể retrain trên dataset mới / domain-adapt,
- kiểm soát feature/template (đặc biệt `is_in_dict`), giảm over-merge,
- vẫn giữ được vật chất hóa merge → cluster neuron.

## 3) Cách làm (Hướng A: train giống underthesea)

### 3.1 Recipe train (tham chiếu underthesea)

Underthesea train CRF với feature template (ngữ cảnh ±2, bigram, `.lower`, `.isdigit`, `.istitle`,
`is_in_dict`) và dataset VLSP2013_WTK / UTS_WTK. Artifacts:

- `models.bin`: CRF weights
- `features.bin`: feature space/template
- `dictionary.bin`: lexicon cho `is_in_dict`

### 3.2 Tích hợp vào nn-aivie (build-time)

Thêm script:

```
python scripts/build_segment_seed_ws_crf.py \
  --checkpoint checkpoints/network_v7_history.pkl \
  --out checkpoints/network_v7_history_wsseed.pkl \
  --model-dir /path/to/ws_crf_model_dir \
  --corpus data/real_vi/history_vi.jsonl \
  --corpus data/real_vi/statements.jsonl \
  --corpus data/real_vi/statements_doc.jsonl \
  --corpus data/real_vi/statements_doc_add.jsonl \
  --max-promotions 80000
```

Luồng:
- load checkpoint → lấy vocab + ParentStore.seg
- chạy WS-CRF model lên corpus thật → dự đoán merge (I-W)
- gom các “từ ghép” predicted (chuỗi token nối bằng dấu cách) → sinh `seed_pairs`
- set `ps.seg.seed_pairs |= seed_pairs` → `finalize_promotions()` → save checkpoint mới

## 4) Tiêu chí đo

- **Coverage**: dùng `scripts/underthesea_probe/distill_segment_gold.py --checkpoint <ckpt>`:
  - baseline S1: 86.5% (segseed)
  - mục tiêu WS-seed bổ sung: tiến gần **95–100%** (tùy filter)
- **Guard**: QA/Para/Syn không tụt (giống TK_033)
- **Over-merge**: log top seed pairs, kiểm tra một số case “Lý do / lý thuyết” v.v.

## 5) Ghi chú Luật 0

CRF model không chạy trong inference. Nó chỉ là **curriculum/build tool** để nạp “quyết định merge”
vào graph như cluster neurons.

## 6) Tiến độ implement (2026-07-08)

### 6.1 Pipeline self-contained trong nn-aivie (không phụ thuộc underthesea)

| Thành phần | File |
|---|---|
| Feature templates (±2 window, bigram, lower, isdigit, istitle, is_in_dict) | `src/nn_aivie/ws_crf/ws_features.py` |
| Model load/tag/merge B-W→I-W | `src/nn_aivie/ws_crf/ws_model.py` |
| Corpus reader (CoNLL tab UTS_WTK + underscore VLSP-style, auto-detect) | `src/nn_aivie/ws_crf/ws_corpus.py` |
| Trainer (pycrfsuite, L-BFGS c1/c2) + eval token-acc & word-F1 | `scripts/train_ws_crf.py` |
| Build seed_pairs từ WS-CRF predictions → checkpoint | `scripts/build_segment_seed_ws_crf.py` |

Dependency: `pip install -e ".[ws]"` (python-crfsuite, datasets) — chỉ dev/build, không cần runtime.

### 6.2 Dataset

- `undertheseanlp/UTS_WTK` (HF, public) subset **large**: train 95.000 câu / 2,2M token,
  validation 2.500 câu, test 2.500 câu. Tải về `data/ws_uts_wtk/large/`.
- `undertheseanlp/VLSP2013_WTK` là repo private (401) → không dùng được, UTS_WTK large là đủ.

### 6.3 Kết quả train

| Model | Train data | Iter | Token acc | Word F1 |
|---|---|---|---|---|
| `out/models/ws_crf_uts_10k` | 10k câu (sample) | 100 | 0.9825 | 0.9685 |
| `out/models/ws_crf_uts_full` | 95k câu (full) | 300 | **0.9914** | **0.9844** (P 0.9832 / R 0.9856) |

Tham chiếu: underthesea công bố F1 ~0.977 trên UTS_WTK — full train (95k, 300 iter) đạt **0.9844**,
vượt cả con số công bố. CRF self-trained trong nn-aivie đạt chất lượng SOTA-tier trên UTS_WTK.
(Kết thúc 2026-07-08 20:59, ~4h17m wall; model.crfsuite 3.77 MB.)

### 6.4 Việc còn lại

- [x] Chờ full train xong, ghi eval vào bảng trên (F1 0.9844 — 2026-07-08)
- [ ] Chạy `build_segment_seed_ws_crf.py` trên checkpoint segseed → `network_v7_history_wsseed.pkl`
- [ ] Đo coverage bằng `distill_segment_gold.py --checkpoint wsseed`
- [ ] Chạy guard QA/Para/Syn

