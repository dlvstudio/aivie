# TK_033 — DICTIONARY-SEED CLUSTER: nơ-ron cụm từ từ điển (học nguyên lý underthesea, không runtime NLP)

| Field | Value |
|-------|-------|
| **ID** | TK_033 |
| **Status** | 🟢 DONE (Phase S1) — seed_pairs + build ckpt + đo coverage + guard giữ |
| **Phụ thuộc** | TK_013 (ISG), TK_015 (dictionary substrate), TK_032 (study underthesea + probe) |
| **Nguồn** | Trao đổi tác giả: đọc mã underthesea → áp dụng **nguyên lý segment + từ điển** cho nn-aivie graph-native; không gọi underthesea lúc `generate` (Luật 0) |
| **Study** | [`docs/study/10_SEGMENT_NEURON_FROM_UTS.md`](docs/study/10_SEGMENT_NEURON_FROM_UTS.md), [`docs/study/09_UNDERTHESEA_VI_NLP.md`](docs/study/09_UNDERTHESEA_VI_NLP.md) |

---

## 1. Bài toán

Mạng nn-aivie ingest tiếng Việt bằng **âm tiết rời** (`tokenize()` = `split()`). Đơn vị học
W_seq / VarLM / ISG retrieval là bigram **âm tiết**, trong khi từ tiếng Việt thường là **cụm
đa âm tiết** (`Thăng Long`, `dời đô`, `Lý Thái Tổ`).

**Hệ quả quan sát:**

| Triệu chứng | Nguyên nhân gốc |
| :--- | :--- |
| Văn rời / collocations yếu | W_seq order-1 trên âm tiết; VarLM (TK_031) cải thiện nhưng ctx vẫn trên token sai ranh giới |
| Fact match cụm yếu | `rank_matches` / `facts_about` khó khớp `Nhà Lý` như một unit khi graph chỉ có `Nhà` + `Lý` |
| ISG không bắt cụm hiếm | Promote cluster **chỉ khi** `bigram_count >= min_frequency` (mặc định 3) — corpus mỏng bỏ lỡ hầu hết từ đúng |

**Câu hỏi ticket:** Làm sao cho đồ thị có **nơ-ron cụm từ đúng ranh giới** trên corpus mỏng
(lịch sử VN, QA curated) **mà không** hardcode rule tiếng Việt lúc inference và **không** phụ thuộc
underthesea runtime?

---

## 2. Ý tưởng (từ underthesea, tái đúc graph-native)

Đọc mã underthesea (`word_tokenize`, CRF `B-W`/`I-W`, feature `is_in_dict` trên **72k cụm**)
cho thấy ba nguyên lý — **không phải** “cài thư viện”:

1. **Nguyên tử = âm tiết** — trùng `align.tokenize()` nn-aivie.
2. **Ghép từ = quyết định merge có nhãn** — underthesea dùng CRF + từ điển; nn-aivie tương đương
   **ISG cluster** + **seed từ lexicon**.
3. **Nhãn ngữ pháp gắn trên cụm đã ghép** — phase sau (POS/Entity trên cluster, TK_033 phase 2+).

**Dictionary-seed cluster (phase 1 — ticket này):**

> Nạp danh sách cụm đa âm tiết từ **từ điển mở** (HF `tsdocode/vietnamese-dictionary` đã có, hoặc
> UVD ~72k) → đánh dấu cặp token-id `(a,b)` là **seed** → `promote_pair(a,b)` **bỏ qua**
> `min_frequency` → tạo `ClusterNeuron` + rewire parent như ISG hiện tại.

Tương đương phần **`is_in_dict`** của underthesea: “cặp này là một từ” — nhưng **vật chất hóa
thành nơ-ron cụm** trong graph, không feature CRF lúc chạy.

**Luật 0:** Từ điển = **dữ liệu huấn luyện / curriculum** (README Mục 0 cho phép). Runtime chỉ
đọc cluster + cạnh đã build — không `if is_in_dict(...)` trong `generate_hebbian.py`.

---

## 3. Hiện trạng

### 3.1 ISG (TK_013) — đã có hạ tầng cluster

- [`core/seg_graph.py`](src/nn_aivie/core/seg_graph.py): `ClusterNeuron`, `promote_pair`, rewire
  `slot_chain` parent, `CLUSTER_OFFSET`.
- Promote điều kiện: `bigram_count[(a,b)] >= min_frequency` (default **3**).
- Rebuild: [`scripts/rebuild_seg_graph.py`](scripts/rebuild_seg_graph.py).

### 3.2 Từ điển — đã nạp substrate nhãn, chưa seed segment

- HF `tsdocode/vietnamese-dictionary` (~36k mục) → [`ingest_dictionary.py`](src/nn_aivie/training/ingest_dictionary.py)
  (sense/POS hub, TK_015) — **không** tạo cluster từ mục đa âm tiết.
- Underthesea UVD ~72k — tùy chọn bổ sung; không bắt buộc phase 1.

### 3.3 Probe đã chạy (TK_032)

| Log | Kết quả chính |
| :--- | :--- |
| `out/tk032_segment.log` | **347** cụm gold; ISG-discoverable **22.8%**; ISG-MISSED **77.2%** trên `history_vi.jsonl` |
| `out/tk032_tokenize.log` | 100% câu khác tokenization ws vs underthesea; −21.7% token |
| `out/tk032_varlm.log` | Corpus nhỏ: whitespace thắng VarLM — **chưa** kết luận đổi tokenize toàn pipeline |
| `out/tk032_ner.log` | NER↔Entity overlap 70.9% — phase sau |

**Ví dụ ISG bỏ lỡ (cần seed):** `Khổng Tử`, `Đại Việt`, `Như Nguyệt`, `Trần Cảnh`, `đại học`,
`xâm lược`, `triều Lý` (cặp yếu).

**Ví dụ ISG đã bắt (tần cao):** `Thăng Long`, `Việt Nam`, `dời đô`, `Lý Thái Tổ`, `kháng chiến`.

### 3.4 Guard hiện tại

Checkpoint tham chiếu: `network_v7_history.pkl` — guard **QA 86 / Para 88 / Syn 67** (TK_028/031).
Mọi thay đổi phải **giữ hoặc ghi âm tính** trên guard.

---

## 4. Đề xuất giải pháp (phased)

### Phase S1 — Dictionary-seed cluster (scope ticket này)

**Mục tiêu:** Cụm có trong từ điển (và optional `title` wiki/history) → cluster tồn tại trong
graph dù `bigram_count < 3`.

**Không làm trong S1:** đổi `tokenize()` ingest toàn corpus; gọi underthesea mỗi câu; rebuild
VarLM sidecar (để phase S2 sau khi đo).

| Bước | Việc |
| :--- | :--- |
| S1a | Thu thập `seed_pairs: set[tuple[int,int]]` từ HF dictionary words ≥2 âm tiết |
| S1b | Optional: thêm seed từ `title` trong `history_vi.jsonl` / `statements_doc` (`fixed_words` tương đương) |
| S1c | `SegmentGraph.seed_pairs` + promote bỏ ngưỡng trong `_check_promotions` / `finalize_promotions` |
| S1d | Script `build_segment_seed.py`: load ckpt → gán seed → `finalize_promotions` hoặc promote trực tiếp → save ckpt mới |
| S1e | Đo: `distill_segment_gold.py` (coverage↑); guard QA/Para/Syn; probe cụm flagship |

**Chuỗi cụm ≥3 âm tiết (S1b mở rộng):** promote lần lượt cặp kề trong cụm (left-to-right), giống
underthesea merge I-W — dùng `promote_pair` lặp; cluster trái có thể là `ClusterNeuron` (cần mở
rộng `promote_pair` nhận `left`/`right` là token hoặc cluster — **hoặc** S1 chỉ seed **bigram**
đầu tiên của mục từ điển, S1.5 mở rộng trigram).

**Đề xuất S1 thuần:** seed **mọi cặp âm tiết liền kề** trong mỗi mục từ điển đa âm tiết
(`"Lý Thái Tổ"` → `(Lý,Thái)`, `(Thái,Tổ)`). Đơn giản, không đổi `promote_pair` signature.

### Phase S2 — Segment gold distill (ticket sau, TK_033B hoặc mở rộng)

- Chạy underthesea `word_tokenize` **1 lần** lúc build trên corpus → cặp `I-W` → `EdgeKind.SEG_MERGE`
  hoặc seed bổ sung (cụm không có trong từ điển).
- Lọc over-merge (`Lý mở`, `Lý do`) — ưu tiên cặp cũng có trong từ điển.

### Phase S3 — POS/Entity trên cluster (ticket sau)

- Wire `POS:` / `Entity:` hub tới cluster thay token đầu — doc 10 §Lớp 3.

---

## 5. Thiết kế kỹ thuật (Phase S1)

### 5.1 `SegmentGraph` — field mới

```python
# seg_graph.py
seed_pairs: set[tuple[int, int]] = field(default_factory=set)

def should_promote(self, pair: tuple[int, int]) -> bool:
    if pair in self.seed_pairs:
        return pair not in self.cluster_of_pair
    count = self.bigram_count.get(pair, 0)
    return count >= self.min_frequency and pair not in self.cluster_of_pair
```

Dùng `should_promote` trong `_check_promotions` và `finalize_promotions` thay check inline.

**Serialize:** thêm `seed_pairs` vào `to_payload` / `from_payload` (list of pairs) — checkpoint
tương thích: thiếu field → `seed_pairs=∅` (hành vi cũ).

### 5.2 `scripts/build_segment_seed.py` (mới)

```
python scripts/build_segment_seed.py \
  --checkpoint checkpoints/network_v7_history.pkl \
  --out checkpoints/network_v7_history_segseed.pkl \
  --dict-source hf \
  --extra-titles data/real_vi/history_vi.jsonl \
  --log-file out/tk033_build_seed.log
```

Luồng:

1. `load_checkpoint`
2. Parse dictionary entries → list words (≥2 syllables = ≥1 space trong surface)
3. `encode_tokens` từng word → consecutive pairs → `seed_pairs`
4. Optional: `title` jsonl → thêm pairs + whole-title bigrams
5. Gán `state.parents.seg.seed_pairs`
6. `rebuild_seg_graph` hoặc loop `promote_pair` cho mọi seed pair chưa có cluster
7. `save_checkpoint`

**Nguồn từ điển phase 1:** tái dùng [`scripts/fetch_vi_dictionary.py`](scripts/fetch_vi_dictionary.py)
/ cache HF — không cần `pip install underthesea` cho S1.

### 5.3 Không đổi (S1)

- `generate_hebbian.py` — không import NLP
- `align.tokenize()` — vẫn âm tiết; cluster chỉ thay **slot_chain** parent sau rewire
- `ingest_dictionary.py` — giữ nguyên; seed segment là đường **song song**

### 5.4 Rủi ro & mitigations

| Rủi ro | Mitigation |
| :--- | :--- |
| Seed quá nhiều (~50k+ cặp từ 36k mục) | Chỉ cặp liền kề; cap `max_seed_promotions`; log số cluster |
| Cụm từ điển sai / hiếm dùng | Chỉ HF curated dict; không seed mục 1 âm tiết |
| Phá guard QA/Para/Syn | Đo guard trước/sau; revert nếu hại |
| `promote_pair` không có occurrence trong parent | Vẫn tạo cluster + `cluster_of_pair`; rewire khi parent sau này chứa cặp (giống ISG) |
| VarLM/W_seq vẫn trên âm tiết | Chấp nhận S1; S2 rebuild VarLM trên slot/cluster nếu đo dương |

---

## 6. Kế hoạch đo (measure-first)

### Tiêu chí S1 — chấp nhận

| Metric | Baseline | Mục tiêu S1 |
| :--- | :--- | :--- |
| ISG coverage (`tk032_segment.log`) | 22.8% discoverable | **≥90%** cho cụm có trong từ điển HF |
| `#clusters` seed | — | log + không explode (>100k cần review) |
| Guard QA / Para / Syn | 86 / 88 / 67 | **giữ** (±0 cho phép ghi âm tính) |
| Flagship đọc tay | `Thăng Long`, `Lý Thái Tổ` trong slot_chain parent lịch sử | cluster xuất hiện |

### Lệnh đo

```bash
pip install underthesea   # chỉ cho probe gold, không runtime
export PYTHONPATH=src

# Trước / sau build seed
python scripts/underthesea_probe/distill_segment_gold.py

# Guard (checkpoint mới)
python scripts/eval_v4.py --checkpoint checkpoints/network_v7_history_segseed.pkl --fast

# pytest liên quan seg
pytest tests/ -k seg -q
```

### Tiêu chí S1 — **không** yêu cầu (phase sau)

- VarLM bits giảm (cần rebuild sidecar)
- `Triều Lý` retrieval đúng (nợ TK_028/029)
- Essay / composition mượt hơn

---

## 7. Changes (file checklist)

| File | Thay đổi | Phase |
| :--- | :--- | :---: |
| [`src/nn_aivie/core/seg_graph.py`](src/nn_aivie/core/seg_graph.py) | `seed_pairs`, `should_promote`, serialize | S1 |
| [`scripts/build_segment_seed.py`](scripts/build_segment_seed.py) | **mới** — build seed từ dict + titles | S1 |
| [`scripts/rebuild_seg_graph.py`](scripts/rebuild_seg_graph.py) | optional `--seed-from-dict` shortcut | S1 |
| [`scripts/underthesea_probe/distill_segment_gold.py`](scripts/underthesea_probe/distill_segment_gold.py) | thêm mode đo sau seed (đọc ckpt clusters) | S1 |
| [`tests/test_seg_graph.py`](tests/test_seg_graph.py) | test promote seed khi count=0 | S1 |
| `TK_033_DICTIONARY_SEED_CLUSTER.md` | ticket này | — |
| [`docs/study/10_SEGMENT_NEURON_FROM_UTS.md`](docs/study/10_SEGMENT_NEURON_FROM_UTS.md) | link TK_033 khi implement xong | S1 |
| [`TK_032_UNDERTHESEA_VI_NLP.md`](TK_032_UNDERTHESEA_VI_NLP.md) | trỏ implementation → TK_033 | S1 |

**Không sửa trong S1:** `generate_hebbian.py`, `align.py`, `build_varlm.py`, `pyproject.toml`.

---

## 11. Kết quả implement Phase S1 (2026-07-08)

### 11.1 Build seed + checkpoint mới

- **Script mới**: [`scripts/build_segment_seed.py`](scripts/build_segment_seed.py)
- **Checkpoint output**: `checkpoints/network_v7_history_segseed.pkl`
- **Log**: `out/tk033_build_seed_*.log` (FileProgressLog)

Chạy:

```bash
python scripts/build_segment_seed.py \
  --checkpoint checkpoints/network_v7_history.pkl \
  --out checkpoints/network_v7_history_segseed.pkl \
  --dict-raw data/dict/vi_dict_raw.jsonl \
  --extra-titles data/real_vi/history_vi.jsonl \
  --max-promotions 80000
```

Kết quả chính (từ log):

- `dict_raw rows=36764`, **multi-word=24536** → `seed_pairs=22209`
- `history_vi titles=173` → `seed_pairs+=59` → **seed_pairs total = 22268**
- `promoted=13342` (đa số seed count=0/1/2) → `clusters=144295`

### 11.2 Coverage: gold (underthesea) vs ISG vs Seed

Probe mới (đã mở rộng để đọc ckpt clusters):

```bash
python scripts/underthesea_probe/distill_segment_gold.py \
  --checkpoint checkpoints/network_v7_history_segseed.pkl \
  --out out/tk033_segment_seed.log
```

Kết quả `out/tk033_segment_seed.log` trên `history_vi.jsonl` (173 câu):

- **ISG-discoverable** (tần suất ≥3): **79/347 = 22.8%** (baseline TK_032)
- **Seed-discoverable** (tần suất ≥3 **HOẶC** cluster có trong ckpt): **300/347 = 86.5%**

### 11.3 Guard (QA/Para/Syn)

```bash
python scripts/eval_v4.py --checkpoint checkpoints/network_v7_history_segseed.pkl --fast
```

Kết quả: **QA 86 / Para 88 / Syn 67 (GIỮ)**.


---

## 8. Lộ trình thực hiện (ước lượng)

| Tuần | Việc | Deliverable |
| :--- | :--- | :--- |
| 1 | S1a–c: `seed_pairs` + unit test | PR nhỏ `seg_graph.py` |
| 1 | S1d: `build_segment_seed.py` + chạy trên `network_v7_history.pkl` | `network_v7_history_segseed.pkl` |
| 1 | S1e: probe + guard eval | `out/tk033_*.log`, cập nhật status ticket |
| 2+ | S2 distill (nếu S1 dương) | ticket mở rộng / TK_033B |
| 2+ | S3 POS trên cluster | doc 10 phase 3 |

---

## 9. Quyết định cần duyệt trước code

1. **Nguồn seed S1:** chỉ HF dictionary (~36k) hay thêm UVD 72k? *(Đề xuất: HF trước — đã có pipeline.)*
2. **Có seed `title` wiki/history không?** *(Đề xuất: có — `Nhà Lý`, `Văn Miếu` không chắc có trong dict.)*
3. **Checkpoint output:** `network_v7_history_segseed.pkl` riêng hay ghi đè? *(Đề xuất: file mới + `viz_default_checkpoint` chỉ đổi sau guard pass.)*
4. **Trigram trong dict** (`"Quốc Tử Giám"`): S1 chỉ bigram liền kề hay promote chuỗi? *(Đề xuất S1: bigram liền kề only — đủ cho 90% case, đơn giản.)*

---

## 10. Kết luận ticket

**Dictionary-seed cluster** = đưa lexicon tiếng Việt vào ISG như **danh sách cặp được phép
gộp thành cluster ngay**, khắc phục **77.2% cụm gold bị ISG bỏ lỡ** trên corpus mỏng — học
**ý tưởng từ điển** của underthesea (`is_in_dict`) mà **không** phụ thuộc CRF runtime. Đây là
bước substrate rẻ nhất trước segment distill (S2) và nhãn trên cluster (S3).

**Trạng thái:** PLAN — sẵn sàng implement Phase S1 sau khi tác giả duyệt §9.
