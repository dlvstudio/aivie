"""
TK_015 D1 / TK_016 D1′ — nạp TỪ ĐIỂN làm substrate CÓ NHÃN (sense/POS/attr).

Luật 0 SẠCH: từ điển = nhãn + co-occurrence học được — **KHÔNG** flat-dump định nghĩa/ví dụ
thành nơ-ron cha (D1′: tránh pha loãng pool sinh). Định nghĩa chỉ gom token → cạnh sense↔nội-dung;
occurrence corpus gắn sau qua `bind_sense_resonance` (chưa chạy trong build hiện tại).
"""

from __future__ import annotations

from nn_aivie.core.edge_kinds import EdgeKind
from nn_aivie.core.state import GraphState
from nn_aivie.teaching.align import tokenize_punct
from nn_aivie.training.ingest import encode_content


def _content_ids(state: GraphState, text: str) -> tuple[list[int], list[int]]:
    # split_punct (R0): tách dấu câu thành token riêng (như ingest HF) → không sinh token
    # dính dấu "hòn:"/"giờ!" nhiễu hồ sơ sense; lọc luôn token dấu-câu khỏi content sense.
    ids = encode_content(state.vocab, text, legacy_tok=tokenize_punct)
    content = [i for i in ids if state.vocab.is_surface(i)]
    return ids, content


def ingest_dictionary(
    state: GraphState,
    entries: list[dict],
    *,
    logger=None,
    log_every: int = 2000,
) -> dict[str, int]:
    """`entries`: [{word, senses:[{pos, meaning, example}]}]. Trả thống kê build (parents=0)."""
    vocab = state.vocab
    stats = {"words": 0, "senses": 0, "parents": 0, "pos_hubs": 0}
    pos_seen: set[str] = set()
    total = len(entries)
    for e_i, entry in enumerate(entries):
        word = str(entry.get("word", "")).strip()
        senses = entry.get("senses") or []
        if not word or not senses:
            continue
        # neo vào token ĐẦU của từ khoá (từ đơn = chính token đa nghĩa; đa từ = token dẫn)
        head_tok = word.split()[0]
        lemma_id = vocab.id_of(head_tok)
        stats["words"] += 1
        for k, sense in enumerate(senses, start=1):
            pos = str(sense.get("pos", "")).strip()
            sid = vocab.sense_id(word, k)
            if pos:
                pid = vocab.pos_id(pos)
                if pos not in pos_seen:
                    pos_seen.add(pos)
                    stats["pos_hubs"] += 1
                state.adjacency.add(sid, pid, 1.0, EdgeKind.POS_ANCHOR)
                state.adjacency.add(lemma_id, pid, 1.0, EdgeKind.LEXATTR)
            state.adjacency.add(lemma_id, sid, 1.0, EdgeKind.SENSE)
            state.adjacency.add(sid, lemma_id, 1.0, EdgeKind.SENSE)
            # định nghĩa + ví dụ: ingest thường (frame) + gom nội dung để nối sense
            sense_content: list[int] = []
            for text in (sense.get("meaning"), sense.get("example")):
                if not text or not str(text).strip():
                    continue
                _ids, content = _content_ids(state, str(text))
                sense_content.extend(content)
            # STAR co-activation sense↔nội-dung (chỉ nghĩa này) → embedding sense = hồ sơ riêng.
            # PPMI nhấn token đặc thù của nghĩa (đá#kick→bóng/sút; đá#stone→vôi) → 2 nghĩa TÁCH.
            for c in set(sense_content):
                if c == lemma_id:
                    continue
                state.adjacency.add(sid, c, 1.0, EdgeKind.CO_ACTIVATION)
                state.adjacency.add(c, sid, 1.0, EdgeKind.CO_ACTIVATION)
            stats["senses"] += 1
        if logger is not None and (e_i + 1) % log_every == 0:
            logger.progress("ingest_dict", e_i + 1, total, extra=f"senses={stats['senses']}")
    return stats
