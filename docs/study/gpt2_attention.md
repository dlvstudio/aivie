# GPT-2 — Attention & Generation (study notes)

Repo: `/Users/dlv/Documents/code/gpt-2` · Inference-only TensorFlow 1.x.

## Data flow

```
tokens → wte+wpe → N×(Attn+MLP) → logits @ wte.T → sample next token
                ↑ past (KV cache per layer)
```

## Key files

| File | Role |
|:---|:---|
| `src/model.py` | `attn()`, `block()`, `model()`, `positions_for()` |
| `src/sample.py` | `sample_sequence()` — autoregressive loop + KV concat |
| `src/encoder.py` | BPE tokenizer |

## Attention mechanism (`model.py`)

- Multi-head causal self-attention: `Q,K,V = conv1d(x)` → `softmax(QKᵀ/√d) V`
- **Causal mask:** lower-triangular — token chỉ nhìn quá khứ
- **KV cache:** `present` stacked as `past`; decode feeds **last token only**
- **Positions:** `past_length + arange(seq)` — absolute position with cache offset

## Generation loop (`sample.py`)

1. Prefill: full `context` → build `past`
2. Loop: `prev` = last sampled token, `past` = accumulated K/V
3. Logits[:, -1] → temperature / top-k / top-p → multinomial
4. Stop at `length` or `n_ctx`

## Mapping → nn-aivie

| GPT-2 | Hebbian equivalent | Action |
|:---|:---|:---|
| Q·Kᵀ softmax | Co-activation on $A$ with decay weights | `dual_context_read` |
| KV cache | Session state (`QASession`, rules, fingerprint) | `qa_context.py` |
| `positions_for` | `R_{-k}` position nodes | `teaching/position.py` |
| Causal mask | Autoregressive history slice | `forward.py` `history[-k_max:]` |
| Weight-tied LM head | Row readout `A[w0, ·]` | `adjacency.row_readout` |
| top-k / top-p | Optional decode (future) | not yet in generate |

## Prototype spike (implemented)

- **Query-conditioned readout:** salient Q tokens act as static "queries" over graph rows — see `hca/forward.py::query_conditioned_read`
- **Session position:** trace shows `d_frame`/`d_bind` per history token (Graph UI)

## Takeaways

1. Context = **weighted sum of past representations** — HCA decay $d_{\text{frame}}$, $d_{\text bind}$ is the Hebbian analogue
2. Incremental decode needs **persistent state** — not full re-read; maps to session/fingerprint not re-encoding full prompt each step
3. Simplest reference implementation (~175 LOC model) — good baseline before nanochat complexity
