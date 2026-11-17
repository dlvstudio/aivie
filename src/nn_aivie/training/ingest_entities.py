"""
TK_027 — nạp THỰC-THỂ tên-riêng (title bài wiki) làm substrate CÓ NHÃN, analog `ingest_dictionary`.

Vì sao: đồng-tự tên-riêng ("Lý"-triều vs "lý"-thuyết) KHÔNG tách được ở tầng token/fold/sense
(TK_026: "Lý" fold về "lý" → gọi mọi nghĩa lý*). Nhưng `statements_doc.jsonl` có field `title` =
tên bài = THỰC THỂ. Gom câu theo title → nơ-ron `Entity:title` với hồ-sơ co-act SẠCH (từ vựng
bài đó), **key theo cụm title (không fold)** → probe đo tách sạch ("Lý Thái Tổ" ~ "Lý thuyết dây"
cosine 0.085; fact "Lý thuyết điện từ" khoá vào entity vật-lý 2.28 vs dynasty ≤0.14).

Luật 0 sạch: nhãn thực-thể + co-occurrence học được; chỉ nối token ĐÃ CÓ trong vocab (không đổ
text trần → không pha loãng, không token rác).
"""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict

from nn_aivie.core.edge_kinds import EdgeKind
from nn_aivie.core.state import GraphState
from nn_aivie.teaching.align import tokenize_punct
from nn_aivie.training.ingest import encode_content


def ingest_entities(
    state: GraphState,
    statements_path: str | list[str],
    *,
    top_tokens: int = 40,
    min_sentences: int = 5,
    max_df_frac: float = 0.3,
    logger=None,
) -> dict[str, int]:
    """Mỗi `title` (≥ min_sentences câu) → `Entity:title` + cạnh CO_ACTIVATION tới top-K token
    nội-dung ĐẶC THÙ (count×idf, bỏ hư-từ df> max_df_frac×#title). Chỉ token đã có trong vocab.
    `statements_path` = 1 đường hoặc list (gộp nhiều nguồn cùng title)."""
    vocab = state.vocab
    paths = [statements_path] if isinstance(statements_path, str) else list(statements_path)
    docs: dict[str, list[str]] = defaultdict(list)
    for p in paths:
        with open(p, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                t = str(r.get("title", "")).strip()
                if t:
                    docs[t].append(r.get("text", ""))

    # bag per-title (token vocab hiện có, bỏ dấu/phi-bề-mặt) + df theo #title
    title_bag: dict[str, Counter] = {}
    df: Counter = Counter()
    for title, sents in docs.items():
        if len(sents) < min_sentences:
            continue
        bag: Counter = Counter()
        for s in sents:
            for tid in encode_content(vocab, s, legacy_tok=tokenize_punct, lookup_only=True):
                if not vocab.is_surface(tid):
                    continue
                bag[tid] += 1
        if bag:
            title_bag[title] = bag
            # df theo FOLD-GROUP: "Lý"(hoa) ≡ "lý"(thường) → token tên xuất hiện ở cả bài triều
            # lẫn bài lý-thuyết → df cao → idf thấp → không áp đảo (thay case-sensitive df sai).
            for fg in {vocab.fold_group_id(t) for t in bag}:
                df[fg] += 1

    n_titles = len(title_bag)
    maxdf = max_df_frac * n_titles
    stats = {"titles_seen": len(docs), "entities": 0, "links": 0}
    for i, (title, bag) in enumerate(title_bag.items()):
        eid = vocab.entity_id(title)
        scored = []
        for t, c in bag.items():
            fg = vocab.fold_group_id(t)
            if df[fg] > maxdf:  # hư-từ / token phổ biến khắp bài → không đặc thù
                continue
            idf = math.log((n_titles + 1) / (1 + df[fg])) + 1.0
            tf = 1.0 + math.log(c)  # sublinear TF: token TÊN tần cao (Lý 80×) không nhấn chìm
            scored.append((tf * idf, t))  # token đặc thù (Uẩn/dây, tần thấp nhưng idf cao)
        scored.sort(reverse=True)
        top = scored[:top_tokens]
        nrm = math.sqrt(sum(w * w for w, _ in top)) or 1.0  # L2-normalize → hồ-sơ so sánh được
        added = 0
        for w, t in top:
            wn = w / nrm
            state.adjacency.add(eid, t, wn, EdgeKind.CO_ACTIVATION)
            state.adjacency.add(t, eid, wn, EdgeKind.CO_ACTIVATION)
            added += 1
            stats["links"] += 1
        if added:
            stats["entities"] += 1
        if logger is not None and (i + 1) % 200 == 0:
            logger.progress("ingest_entities", i + 1, n_titles, extra=f"ents={stats['entities']}")
    return stats
