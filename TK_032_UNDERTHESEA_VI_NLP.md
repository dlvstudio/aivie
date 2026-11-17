# TK_032 — UNDERTHESEA VI NLP (ingest-only substrate)

| Field | Value |
|-------|-------|
| **ID** | TK_032 |
| **Status** | 🟡 STUDY + PROBES — `read-uts-code` ✅ done (underthesea 9.5.0 @ `../underthesea`) |
| **Nguồn** | Phân tích underthesea làm công cụ **ingest/build** (Luật 0), không runtime NLP lúc generate |

## Checklist nghiên cứu

| Task | Status |
| :--- | :---: |
| `read-uts-code` — đọc API + model underthesea | ✅ |
| `docs/study/09_UNDERTHESEA_VI_NLP.md` | ✅ |
| Probes `scripts/underthesea_probe/` | ✅ |
| Chạy probe + cập nhật số liệu §5 study doc | ✅ (xem log `out/tk032_*.log`) |

## Key facts (tóm tắt kỹ thuật)

- **underthesea 9.5.0**, Apache-2.0, model CRF trong `underthesea_core`
- `word_tokenize`: `regex_tokenize` + `FastCRFSequenceTagger` `ws_crf_vlsp2013`, API `fixed_words`
- `pos_tag`: `word_tokenize` → CRF `pos_crf_2017_10_11.bin`
- `ner`: `chunk()` → `ner_crf_2017_10_12.bin`; `deep=True` cần transformers
- `sent_tokenize`: PunktSentenceTokenizer
- `text_normalize`: quy tắc `token_normalize`
- Từ điển UVD ~72k mục qua `underthesea download-data` (nn-aivie đã dùng HF `tsdocode/vietnamese-dictionary`)

## Probe logs

- `out/tk032_tokenize.log` — whitespace vs underthesea
- `out/tk032_ner.log` — NER vs Entity:title
- `out/tk032_varlm.log` — VarOrderLM A/B tokenization
- `out/tk032_segment.log` — segment gap: ISG tần-suất vs gold underthesea (77.2% miss trên history_vi)

## Cơ chế underthesea (đọc mã nguồn) + spec áp dụng

- **Nguyên lý:** tách từ = chuỗi quyết định merge B-W/I-W trên âm tiết (CRF + từ điển 72k feature); nhãn POS/chunk/NER xếp tầng trên cụm đã ghép.
- **Áp dụng nn-aivie:** dictionary-seeded cluster + segment gold distill (SEG_MERGE) + nhãn trên cluster — build-time, Luật 0 sạch.
- Spec đầy đủ: [`docs/study/10_SEGMENT_NEURON_FROM_UTS.md`](docs/study/10_SEGMENT_NEURON_FROM_UTS.md)
- **Implementation plan:** [`TK_033_DICTIONARY_SEED_CLUSTER.md`](TK_033_DICTIONARY_SEED_CLUSTER.md) (dictionary-seed cluster Phase S1)

## Study doc

- [`docs/study/09_UNDERTHESEA_VI_NLP.md`](docs/study/09_UNDERTHESEA_VI_NLP.md) — khảo sát + probe tokenize/NER/VarLM
- [`docs/study/10_SEGMENT_NEURON_FROM_UTS.md`](docs/study/10_SEGMENT_NEURON_FROM_UTS.md) — cơ chế segment + spec nơ-ron cụm
