# TK_032 — Underthesea probes

So sánh tokenization / NER / VarOrderLM giữa stack nn-aivie hiện tại và [underthesea](https://github.com/undertheseanlp/underthesea) trên corpus lịch sử Việt.

## Cài đặt

```bash
pip install underthesea
```

Lần chạy đầu underthesea có thể tải model CRF (~vài MB).

## Chạy (từ repo root)

```bash
cd /path/to/nn-aivie
export PYTHONPATH=src

python scripts/underthesea_probe/compare_tokenization.py
python scripts/underthesea_probe/ner_entity_overlap.py
python scripts/underthesea_probe/varlm_bits_ab.py
```

Hoặc một lệnh:

```bash
PYTHONPATH=src python scripts/underthesea_probe/compare_tokenization.py && \
PYTHONPATH=src python scripts/underthesea_probe/ner_entity_overlap.py && \
PYTHONPATH=src python scripts/underthesea_probe/varlm_bits_ab.py
```

## Đầu ra

| Script | Log |
| :--- | :--- |
| `compare_tokenization.py` | `out/tk032_tokenize.log` |
| `ner_entity_overlap.py` | `out/tk032_ner.log` |
| `varlm_bits_ab.py` | `out/tk032_varlm.log` |
| `distill_segment_gold.py` | `out/tk032_segment.log` |

`distill_segment_gold.py` đo khoảng cách segment: bao nhiêu TỪ gold (underthesea) mà ISG thuần-tần-suất bỏ lỡ trên corpus — luận cứ cho [`docs/study/10_SEGMENT_NEURON_FROM_UTS.md`](../../docs/study/10_SEGMENT_NEURON_FROM_UTS.md).

`ner_entity_overlap.py` bỏ qua an toàn nếu không có checkpoint tại `viz_default_checkpoint()` (`checkpoints/network_v7_history.pkl` hoặc fallback trong `src/nn_aivie/core/paths.py`).

## Tài liệu

Kết quả probe được tổng hợp trong [`docs/study/09_UNDERTHESEA_VI_NLP.md`](../../docs/study/09_UNDERTHESEA_VI_NLP.md).
