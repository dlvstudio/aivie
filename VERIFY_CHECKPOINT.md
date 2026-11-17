# VERIFY CHECKPOINT — kiểm checkpoint sau train (2026-07-16)

| Field | Value |
|-------|-------|
| **Script** | `scripts/verify_checkpoint.py` |
| **Nguồn** | Đợt thay đổi Luật 0 (2026-07-15/16): gỡ `is_punct`/`PUNCT_TOKENS` + EOS cuối câu train |
| **Vì sao** | Cả ba thay đổi đợt này **hỏng trong im lặng** nếu sót — không lỗi, không cảnh báo |

## 1. Lệnh

```bash
PYTHONPATH=src python -u scripts/verify_checkpoint.py checkpoints/network_v10.pkl
```

Exit `0` = tất cả PASS · Exit `1` = có FAIL (in rõ mục nào + vì sao).

Checkpoint 1.9 GB mất **5–15 phút** để load — phần lớn là `resync_match_ids` quét mọi cha.
Nên chạy nền + ghi log (Luật observability):

```bash
PYTHONPATH=src nohup python -u scripts/verify_checkpoint.py \
  checkpoints/network_v10.pkl > out/verify_v10.log 2>&1 &
tail -f out/verify_v10.log
```

## 2. Kiểm gì, và vì sao mỗi mục tồn tại

| # | Mục | FAIL nghĩa là | Sửa |
| :-- | :--- | :--- | :--- |
| 1 | **EOS cuối `source_ids`** | `_with_eos` không chạy trên đường train đã dùng | xem §3 |
| 2 | **EOS ngoài `match_ids`** | EOS lọt vào retrieval → mọi cha đều có nó → nhiễu | `is_surface_token(SPECIAL_EOS)` phải = False |
| 3 | **Dấu câu vào `_gdf`** (`'.'` gidf < 2.0) | dấu câu vắng khỏi thống kê → `gidf('.')`=7.17 = **cao nhất corpus**, gấp 6× `Việt` → dấu chấm thành **neo thực thể** | `_is_match_token` còn lọc `is_punct`? `resync_match_ids` có chạy lúc load? |
| 4 | **Mốc kết học từ EOS** (`_ends_unit('.')`=True, từ nội dung=False) | `at_boundary` (TK_009) mù → sau dấu chấm không tin trật tự nguồn → bug `"Việc đọc sách"→"đọc Việc sách"` | cần #1 **và** #9 (varlm); kiểm `ar_eos_boundary_min_lift` |
| 5 | **Bake FRAME_BRACKET** (edges/meta/**endpoints** > 0) | `endpoints=0` = checkpoint **KẸT**: bake báo `skipped`, runtime bị guard checkpoint-lớn chặn → phrase opener (TK_037) chết âm thầm | train lại với `--bake`, hoặc `scripts/bake_checkpoint_indices.py <ckpt> --in-place` |
| 6 | **Bake skip_spans** | thiếu chỉ số suy dẫn | như trên |
| 7 | **VarLM sidecar nạp được** | thiếu = **no-op im lặng** (`else: state.varlm = None`, không log) → `ar_varlm_weight` vô nghĩa | `scripts/build_varlm.py` |
| 8 | **VarLM khớp vocab** (0 token-id ngoài vocab) | sidecar build từ checkpoint KHÁC → **load OK nhưng điểm sai** | build lại **SAU** khi train xong |
| 9 | **VarLM học mốc kết** (ctx có EOS làm token kế) | corpus build varlm không có EOS | build lại sau khi train |
| 10 | **VarLM cluster-aware** | `--use-clusters` không được truyền | build lại với cờ đúng |

## 3. Đường train nào đã dùng? (bẫy đã cắn thật)

Repo có **BA** đường nạp, và EOS phải được nối ở CẢ BA:

| lệnh | đường | file |
| :--- | :--- | :--- |
| `scripts/train.py --manifest …` | `pipeline.run_manifest` → `pipeline.ingest_text_file` | `training/pipeline.py` |
| `scripts/train_v3.py --extra-dir …` | `ingest_statements` / `ingest_qa` | `training/ingest.py` |
| `scripts/build_varlm.py` | tự tokenize → `lm.add_sequence` | `scripts/build_varlm.py` |

Bẫy thật, VẤP HAI LẦN:
1. `_with_eos` ban đầu chỉ thêm vào `ingest.py`. `train.py` — lệnh train **chính** — đi lối
   `pipeline.py` → suýt bake 7.7 giờ ra checkpoint không có mốc kết. Bắt lúc soạn lệnh.
2. Sót tiếp `build_varlm.py` (nó tự tokenize) → verify 2026-07-16 FAIL mục 9
   (`varlm học mốc kết = 0 ctx có EOS`). Sót ngay TRONG LÚC viết chính mục §3 này.

Test `tests/test_eos_boundary.py::test_train_pipeline_appends_eos` khoá đường `pipeline.py`.

## 4. Vì sao không tin log train

Log train **không nhắc EOS lần nào** (`_with_eos` không log), và nó **không** in `gidf`. Nó
báo `FRAME_BRACKET edges=… endpoints=…` — chỉ đủ cho mục 5. Bảy mục còn lại chỉ biết bằng
cách **load checkpoint và đo**.

Tiền lệ: bản eval `tk035_discourse_prompts` chạy **50 phút** quét `varlm=0/0.35/0.5` ra kết
quả **giống từng byte** ở 129/129 dòng — vì sidecar thiếu. Report `TK_017 §16` kết luận nhầm
*"biến varlm chưa tạo khác biệt"*. Không log nào báo; chỉ có đo mới thấy.

## 5. Kỳ vọng trên `network_v10.pkl` (train 2026-07-16)

```
cha=65,098  vocab=126,518
[PASS] EOS cuối source_ids       ~2000/2000
[PASS] EOS ngoài match_ids       0/2000
[PASS] dấu câu vào thống kê      '.' gidf ≈ 1.1   (KHÔNG phải 7.168)
[PASS] mốc kết học từ EOS        '.' lift = 16.0 (ngưỡng 2.0), ends_unit=True; 'là'/'của'/'và' = False
[PASS] bake FRAME_BRACKET        edges=1,812,272  endpoints=357,138
[PASS] bake skip_spans           30,000
[PASS] varlm khớp vocab          0 id ngoài vocab
```

Checkpoint **cũ** (`grade_lessons_test.pkl`, train trước 15-07) sẽ FAIL mục 1/4/7 — **đúng
như kỳ vọng**, không phải lỗi: chúng train trước khi có EOS. Mục 3 vẫn PASS nhờ
`resync_match_ids` vá lúc load.

## 6. Sổ nợ liên quan

- `FOLD_STRIP` + `align._PUNCT_CHARS` — 2 hằng dấu câu còn lại; gỡ = đổi tokenize = đổi vocab
  = mọi checkpoint vô giá trị. Xem **TK_012 §9.5**.
- `w0_tok.endswith(":")` — chỗ so-string **cuối cùng** trong inference.
- `resync_match_ids` chạy **vô điều kiện** mỗi lần load → checkpoint mới (đã train bằng luật
  mới) không cần nó. Có thể gate bằng cách kiểm một cha xem `match_ids` đã có dấu câu chưa.
  Chưa đo cái giá; nghi là nguyên nhân load `network_v10` chậm hơn hẳn.

## 7. Kết quả verify LẦN ĐẦU — 2026-07-16 (8/10, ĐÃ LỖI THỜI → xem §9)

> ⚠️ Mục này là bản chạy **lần đầu**, còn 2 FAIL. Cả hai là **lỗi CODE, đã sửa** (§8); sau khi
> sửa + build lại sidecar → **10/10 PASS** (§9). Giữ đây làm lịch sử.

Lệnh đã chạy:

```bash
PYTHONPATH=src ~/Documents/learn/nn-aivie/.venv/bin/python -u \
  scripts/verify_checkpoint.py checkpoints/network_v10.pkl
```

Thời gian load checkpoint: ~226s (1,850 MB).

Tóm tắt:

- PASS: EOS cuối `source_ids` = 2000/2000
- PASS: EOS ngoài `match_ids` = 0/2000
- PASS: dấu câu vào thống kê `'.'` gdf=38,976 gidf=1.513
- PASS: bake FRAME_BRACKET = edges 1,812,272; endpoints 357,138
- PASS: bake skip_spans = 30,000
- PASS: varlm khớp vocab = 0 token-id ngoài vocab
- PASS: varlm cluster-aware = 3,036,655 ctx-element là cluster id
- FAIL: mốc kết học từ EOS: `'.' P(→EOS)=0.057`, `ends_unit=False`
- FAIL: varlm học mốc kết: `0` ctx có EOS làm token kế

Kết luận nhanh cho cowork review:

- Checkpoint đã bake đầy đủ và sidecar varlm nạp được, vocab khớp.
- Tuy nhiên còn fail 2 mục liên quan trực tiếp đến tín hiệu EOS-boundary (mục 4 và 9).
- Trạng thái hiện tại: **không đạt verify full** (exit code 1).

## 8. Đính chính §7 — hai FAIL là lỗi CODE, không phải checkpoint

Verify 2026-07-16 trên `network_v10.pkl` (8/10 PASS) trượt mục 4 và 9. **Cả hai do code, đã
sửa** — checkpoint train đúng, KHÔNG cần train lại:

**FAIL mục 4 — ngưỡng hiệu chỉnh trên corpus đồ chơi.** `'.' P(→EOS)=0.057` < ngưỡng `0.25`.
EOS *có* học được (0.057 > 0; `2000/2000` cha kết thúc bằng EOS). Lỗi: `ar_eos_boundary_min_p
= 0.25` chốt từ corpus 3 câu — nơi **mỗi đơn vị đúng một câu** → `P = 1.000`. Corpus thật là
đoạn 2-4 câu + bài dài (`history_articles` 158 bài nguyên vẹn) → `.` xuất hiện ~17 lần/đơn vị,
chỉ **1 lần** trước EOS.

→ Thay bằng **LIFT** = `P(EOS|w0) / P(EOS) nền`, ngưỡng `ar_eos_boundary_min_lift = 2.0`.
Tự hiệu chỉnh theo corpus. Đo:

| corpus | `P(.→EOS)` | base | **lift** |
| :--- | --: | --: | --: |
| đơn vị 1 câu | **1.000** | 0.167 | **6.0** |
| đơn vị 6 câu | **0.167** | 0.024 | **7.0** |
| `network_v10` (dự đoán) | 0.057 | ~0.007 | **~8** |

`P` đổi 6×, **lift gần như không đổi**. Test `test_boundary_survives_long_units` khoá.

**FAIL mục 9 — `build_varlm` là đường nạp THỨ BA.** Nó tự tokenize (`build_varlm.py:63`),
không qua `ingest.py` lẫn `pipeline.py` → sidecar không thấy EOS → VarLM không bao giờ đề
xuất mốc kết dù đồ thị đã học. Đã nối EOS vào `_feed_text`.

→ **Chỉ cần build lại sidecar**, không train lại:

```bash
PYTHONPATH=src python -u scripts/build_varlm.py \
  --checkpoint checkpoints/network_v10.pkl --use-clusters --paragraph-units
PYTHONPATH=src python -u scripts/verify_checkpoint.py checkpoints/network_v10.pkl
```

---

## 9. Kết quả verify SAU KHI SỬA — 2026-07-16: **10/10 PASS** ✅

Sau khi (a) đổi ngưỡng tuyệt đối → **lift** và (b) nối EOS trong `build_varlm` (đường nạp thứ
BA), rồi **build lại sidecar** — checkpoint KHÔNG train lại:

```
loaded: cha=65,098 vocab=126,518                          (load 206s / 1,850 MB)
[PASS] EOS cuối source_ids: 2000/2000
[PASS] EOS ngoài match_ids: 0/2000
[PASS] dấu câu vào thống kê: '.' gdf=38,976 gidf=1.513    ('Việt' 3.781, 'là' 1.894)
       P(EOS) nền = 0.00358
[PASS] mốc kết học từ EOS: '.' P(→EOS)=0.057 lift=16.0 (ngưỡng 2.0) ends_unit=True;
       từ nội dung nhận nhầm làm mốc: không
[PASS] bake FRAME_BRACKET: edges=1,812,272 meta=1,812,272 endpoints=357,138
[PASS] bake skip_spans: 30,000
       varlm: max_ctx=4 lam=0.45 use_clusters=True n_tokens=9,654,468
[PASS] varlm khớp vocab: 0 token-id NGOÀI vocab
[PASS] varlm học mốc kết: 119,093 ctx có EOS làm token kế    (trước: 0)
[PASS] varlm cluster-aware: 3,114,704 ctx-element là cluster id
✅ TẤT CẢ PASS
```

**LIFT chứng minh được thiết kế:** chính `P=0.057` này làm ngưỡng tuyệt đối `0.25` **trượt** ở
§7 — mà lift cho **16.0**, vượt xa ngưỡng 2.0. `P` phụ thuộc độ dài đơn vị; lift thì không.

### 9.1 Verify XANH ≠ cơ chế được DÙNG

Rebuild sidecar (119k ctx EOS) + `at_boundary` bật (lift=16.0) → **eval summary GIỐNG HỆT bản
trước**. Vì `at_boundary` và `ar_varlm_weight` sống **bên trong `generate_autoregressive`**, mà
lưới verbatim (`ar_tail_as_agenda_opener=False`) return **trước** khi tới đó:

```python
if verbatim_only or not config.ar_tail_as_agenda:
    return generated + tail        # generate_hebbian.py:1023 — KHÔNG vào A3
```

→ Verify kiểm **checkpoint có mang tín hiệu không**; nó KHÔNG kiểm **đường sinh có dùng tín hiệu
đó không**. Hai câu hỏi khác nhau. Chi tiết: **TK_012 §9.7.1**, TK_009 §7.2.
