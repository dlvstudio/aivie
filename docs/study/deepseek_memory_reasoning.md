# DeepSeek-V3 / R1 — Memory & Reasoning (study notes)

Repos:
- `/Users/dlv/Documents/code/DeepSeek-V3` — inference (MLA, MoE, YaRN)
- `/Users/dlv/Documents/code/DeepSeek-R1` — README + paper only

## DeepSeek-V3 architecture (inference code)

| Innovation | Location | Summary |
|:---|:---|:---|
| **MLA** | `inference/model.py::MLA` | Low-rank KV latent (`kv_lora_rank=512`); cache compact `kv_cache` + `pe_cache` |
| **YaRN** | `precompute_freqs_cis()` | Extended context via rope scaling + `mscale` |
| **MoE** | `MoE`, `Gate` | 256 experts, 8 active; grouped routing |
| **128K context** | README NIAH benchmarks | Needle-in-haystack eval |

## DeepSeek-R1 (documentation)

4-stage pipeline (from README / arXiv 2501.12948):

1. Cold-start SFT (small CoT seed)
2. RL reasoning
3. RL + language consistency
4. Distillation SFT to V3 / dense models

Emergent: long CoT, self-verification, reflection.

## Mapping → nn-aivie

| DeepSeek | Hebbian analogue | Phase |
|:---|:---|:---|
| MLA compressed cache | Composite hub nodes summarizing history | Phase 2 |
| YaRN / mscale | Scope Neurons widen $K_{\max}$ by grade | curriculum |
| NIAH eval | Retrieval test in eval harness | `eval/niah.py` |
| MoE routing | Rule arbitration — activate subgraph | `matcher.py` |
| R1 CoT | `Suy nghĩ:` / verify lesson + G4 self-talk | Phase C |
| Distillation traces | G2 imprint on teacher paths | `g2_imitation.py` |

## What NOT to port

- FP8 training stack, 671B MoE scale, MTP speculative decode
- Full RL loop — replace with dopamine G3 + structural verify

## Prototype spikes

1. **NIAH-style test:** embed needle fact in long prefix, ask QA at end — `eval/niah.py`
2. **Verify node roadmap:** structural `[Verify]` in lesson AST (G4)
3. **Session summary node:** after each turn, co-fire hub (MLA-inspired)

## Takeaways

- **Compress past** without losing retrieval = key to long context on Hebbian budget
- **Staged train** (teach → imitate → reinforce → verify) mirrors R1 pipeline shape
- **Distillation** over raw RL fits Direct Teaching philosophy — imprint traces, don't backprop
