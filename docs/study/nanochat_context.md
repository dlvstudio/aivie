# nanochat — Context, Window, Training (study notes)

Repo: `/Users/dlv/Documents/code/nanochat` · Full train → SFT → chat pipeline.

## Architecture highlights

| Component | File | Idea |
|:---|:---|:---|
| GPT + RoPE | `nanochat/gpt.py` | QK-norm, GQA, value embeddings, smear |
| Sliding window | `gpt.py::_compute_window_sizes` | Pattern `SSSL`: short (~¼ seq) then **full on last layer** |
| Flash attention | `nanochat/flash_attention.py` | FA3/SDPA + window tuple `(left, right)` |
| KV cache | `nanochat/engine.py::KVCache` | Preallocated `(layers, B, T, H, D)` |
| Data packing | `nanochat/dataloader.py` | BOS-aligned best-fit — every row starts BOS |
| Pretrain | `scripts/base_train.py` | `--depth` dial, Chinchilla-style token horizon |
| SFT | `scripts/chat_sft.py` | `TaskMixture` shuffled multi-task |

## Sliding window (`SSSL`)

```python
# gpt.py — pattern tiled per layer; final layer always L (full context)
long_window = sequence_len      # L
short_window = ceil(seq/4, 128) # S
```

**Insight:** Early layers local patterns; **final layer global** — maps to nn-aivie: local $K_{\max}$ decay + global hop to `Hỏi:` anchor.

## BOS-aligned packing

Every training row starts with BOS → any token can attend to document start.

**nn-aivie analogue:** QA rows start `Hỏi:`; chat rows start `User:` — frame token as global anchor.

## Training stages

1. **Pretrain:** random document packing, `--target-param-data-ratio` (default 12)
2. **SFT:** assistant-mask only (like nn-aivie `train_mask` after `Đáp:`)
3. **Optional RL:** `chat_rl.py`

## Mapping → nn-aivie

| nanochat | Hebbian | Status |
|:---|:---|:---|
| SSSL window | Local K + global anchor readout | `query_conditioned_read` + config |
| BOS packing | Frame-first corpus rows | `data/qa.py`, `data/chat.py` |
| TaskMixture SFT | G2 shuffle QA + dialogue | `dialogue` dataset in manifest |
| depth dial | grade pack + `K_max` by grade | `hca_config_for_grade` |
| bits-per-byte | eval metric | `eval/metrics.py` |
| MuonAdamW | N/A (backprop) | skip |

## Prototype spikes

1. `hca_config`: document global anchor ids in query readout
2. `data/grade_03/dialogue/` — multi-turn jsonl for G2 mixture
3. `scripts/eval_suite.py` — smoke metrics table

## Takeaways

- **Hybrid local+global context** is production-proven — adopt for HCA v2 without softmax
- **Answer-only mask** in SFT = nn-aivie QA mask — already aligned
- Scaling laws (depth → batch/LR/horizon) suggest single **grade dial** for teach steps + corpus size
