"""aivie generate command."""

from __future__ import annotations

import time

from nn_aivie.core.state import GraphState
from nn_aivie.hca.readout import decode_surface
from nn_aivie.hca.generate_hebbian import generate


def run_generate(
    prompt: str,
    checkpoint: str,
    max_len: int = 48,
    *,
    trace_log: bool = False,
    mode: str = "w_skip",
) -> None:
    """Sinh HỢP NHẤT (P1): truy hồi nơ-ron cha → agenda → A3 sinh token-by-token (như LLM).
    Checkpoint cũ không có parents sẽ tự fallback W_seq trong `generate`.

    `mode=frame_da_khe` (TK_050): khung–đa-khe P112. `mode=compose_2cha` (TK_051):
    ghép ≥2 cha P203. `mode=analogy_la` (TK_046 P98G): điền ``X là ?`` qua cầu
    thuộc_tính. Các mode trên opt-in; mặc định vẫn W_skip primary.

    `trace_log=True` → ghi file log tiến trình + trace TỪNG TOKEN (w0, token chọn, agenda
    còn thiếu, top scores) ra `out/logs/` — soi tính đúng đắn ngay lúc sinh (TK_011 A).
    """
    from nn_aivie.util.progress import get_run_logger
    from nn_aivie.hca.config import HCAConfig

    log = get_run_logger("generate") if trace_log else None
    if log is not None:
        log.log(f"checkpoint={checkpoint} max_len={max_len} mode={mode}")
        log.log(f"prompt: {prompt}")

    t0 = time.time()
    state = GraphState.load(checkpoint)
    if log is not None:
        log.log(f"loaded ({time.time() - t0:.1f}s) "
                f"vocab={len(state.vocab)} parents={len(state.parents) if state.parents else 0}")

    trace: list[dict[str, object]] | None = [] if trace_log else None
    t1 = time.time()
    cfg = HCAConfig(generation_mode=mode) if mode and mode != "w_skip" else None
    ids = generate(state, prompt, max_len=max_len, config=cfg, trace_out=trace)
    text = decode_surface(state, ids)
    print(text)

    if log is not None:
        for step in trace or []:
            top = ", ".join(
                f"{d['token']}:{d['score']:.2f}" for d in (step.get("top") or [])[:5]
            )
            log.log(
                f"  step {step['step']:>2} w0={step['w0']!r} → {step['chosen']!r}"
                f"  pending={step.get('agenda_pending')}  top=[{top}]"
            )
        log.log(f"generated {len(ids)} tokens ({time.time() - t1:.1f}s): {text}")
