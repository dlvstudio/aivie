# Tài liệu nn-aivie

## Spec kiến trúc (runtime & QA)

| File | Mô tả |
|:---|:---|
| [QA_CONTEXT_SPEC.md](QA_CONTEXT_SPEC.md) | QFocus, fingerprint, frame QA, case studies bleed/cụt |
| [ANSWER_CLOSURE_SPEC.md](ANSWER_CLOSURE_SPEC.md) | **Answer closure** — thoả mãn câu trả lời, agenda, essay, roadmap P1–P3 |

README gốc: [../README.md](../README.md) — Mục **0** (luật), **0.1** (điểm yếu), **0.2** (bài toán & roadmap mai).

## Study — LLM → Hebbian

| File | Mô tả |
|:---|:---|
| [study/00_LLM_TO_HEBBIAN_MAP.md](study/00_LLM_TO_HEBBIAN_MAP.md) | Bản đồ khái niệm |
| [study/01_OPEN_LLM_TO_HEBBIAN_TRANSFER.md](study/01_OPEN_LLM_TO_HEBBIAN_TRANSFER.md) | Distillation & chuyển hoá cơ chế |
| [study/gpt2_attention.md](study/gpt2_attention.md) | gpt-2 attention → HCA |
| [study/nanochat_context.md](study/nanochat_context.md) | nanochat context / SFT |
| [study/deepseek_memory_reasoning.md](study/deepseek_memory_reasoning.md) | DeepSeek MLA, R1 |

## Bắt đầu implement (2026-06-29)

1. Đọc README **Mục 0.2** + [ANSWER_CLOSURE_SPEC.md](ANSWER_CLOSURE_SPEC.md) §9 (checklist P1)
2. Code: `assess_answer_closure()` → `generate.py` + `viz/trace.py`
3. Test: `tests/test_answer_closure.py`
