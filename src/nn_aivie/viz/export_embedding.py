"""
Export embedding E cho Graph UI — tra cứu cosine, hàng xóm, so sánh cặp token.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from nn_aivie.core.edge_kinds import EdgeKind
from nn_aivie.core.state import GraphState
from nn_aivie.core.vocab import Vocabulary, is_surface_token


def _resolve_id(vocab: Vocabulary, token: str) -> int | None:
    """Tra id có sẵn — không gọi id_of (tránh tạo node mới khi gõ nhầm)."""
    t = token.strip()
    if not t:
        return None
    return vocab._token_to_id.get(t)


def _kind(token: str) -> str:
    """TK_015/016 — loại node để UI tô màu: surface | sense | pos | struct."""
    if token.startswith("Sense:"):
        return "sense"
    if token.startswith("POS:"):
        return "pos"
    return "surface" if is_surface_token(token) else "struct"


def _surface_neighbors(state: GraphState, nid: int, topk: int) -> list[dict[str, Any]]:
    """Hàng xóm cosine LỌC còn surface (bỏ dấu câu + node cấu trúc) — hồ sơ nghĩa sạch."""
    emb, vocab = state.embedding, state.vocab
    out: list[dict[str, Any]] = []
    for i, s in emb.nearest(nid, topk=topk * 4):
        tok = vocab.token_of(i)
        if is_surface_token(tok):
            out.append({"id": int(i), "token": tok, "similarity": round(float(s), 4)})
        if len(out) >= topk:
            break
    return out


def embedding_senses(state: GraphState, word: str, *, topk: int = 8) -> dict[str, Any]:
    """TK_015/016 — TÁCH NGHĨA: token đơn (blur) vs từng sense-neuron (hồ sơ hàng xóm riêng).

    Trực quan hoá kết quả D1: vector token đa nghĩa bị TRỘN; mỗi Sense:word#k có cụm hàng xóm
    riêng (đá#1→{hòn,tảng}, cân#3→{gram,kg,lạng}).
    """
    vocab, emb = state.vocab, state.embedding
    if emb is None:
        raise ValueError("no embedding in checkpoint")
    w = word.strip()
    wid = _resolve_id(vocab, w)
    result: dict[str, Any] = {"word": w, "token_found": wid is not None}
    if wid is not None:
        result["token_blur"] = _surface_neighbors(state, wid, topk)
    senses: list[dict[str, Any]] = []
    k = 1
    while (sid := vocab._token_to_id.get(f"Sense:{w}#{k}")) is not None:
        pos_hubs = [
            vocab.token_of(j).removeprefix("POS:").replace("_", " ")
            for j in state.adjacency.layer(EdgeKind.POS_ANCHOR).row(sid)
        ]
        senses.append({
            "sense": f"{w}#{k}",
            "id": int(sid),
            "pos": pos_hubs,
            "neighbors": _surface_neighbors(state, sid, topk),
        })
        k += 1
    result["senses"] = senses
    result["n_senses"] = len(senses)
    return result


def _token_rows(vocab, limit: int = 500) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for i in range(len(vocab)):
        tok = vocab.token_of(i)
        if not tok or not is_surface_token(tok):
            continue
        rows.append({"id": i, "token": tok})
        if len(rows) >= limit:
            break
    return rows


def embedding_info(state: GraphState) -> dict[str, Any]:
    emb = state.embedding
    if emb is None or emb.matrix.size == 0:
        return {
            "available": False,
            "vocab_size": len(state.vocab),
            "dim": 0,
            "message": "Checkpoint không có embedding E (chạy train_v3 để build PPMI+SVD).",
        }
    mat = emb.matrix
    norms = np.linalg.norm(mat, axis=1)
    nonzero = int(np.sum(norms > 1e-9))
    return {
        "available": True,
        "vocab_size": int(mat.shape[0]),
        "dim": int(mat.shape[1]),
        "nonzero_rows": nonzero,
        "norm_mean": float(norms[norms > 0].mean()) if nonzero else 0.0,
        "norm_max": float(norms.max()) if norms.size else 0.0,
        "sample_tokens": _token_rows(state.vocab, limit=24),
    }


def embedding_lookup(
    state: GraphState,
    token: str,
    *,
    topk: int = 16,
    min_sim: float = 0.0,
    vector_preview: int = 0,
) -> dict[str, Any]:
    emb = state.embedding
    vocab = state.vocab
    if emb is None:
        raise ValueError("no embedding in checkpoint")

    tid = _resolve_id(vocab, token)
    if tid is None:
        matches = [
            t
            for i in range(len(vocab))
            if (t := vocab.token_of(i))
            and is_surface_token(t)
            and token.strip().lower() in t.lower()
        ][:20]
        return {
            "found": False,
            "query": token,
            "suggestions": matches,
        }

    vec = emb.vector(tid)
    norm = float(np.linalg.norm(vec))
    n_show = vector_preview if vector_preview > 0 else emb.dim
    n_show = min(n_show, emb.dim)
    preview = [round(float(x), 4) for x in vec[:n_show]]

    near = emb.nearest(tid, topk=topk, min_sim=min_sim)
    neighbors = [
        {
            "id": nid,
            "token": vocab.token_of(nid),
            "similarity": round(sim, 4),
        }
        for nid, sim in near
    ]

    return {
        "found": True,
        "id": tid,
        "token": vocab.token_of(tid),
        "dim": emb.dim,
        "norm": round(norm, 4),
        "vector_preview": preview,
        "vector_preview_dims": list(range(n_show)),
        "neighbors": neighbors,
        "parent_df": state.parents._df.get(tid, 0) if state.parents else 0,
    }


def embedding_compare(state: GraphState, token_a: str, token_b: str) -> dict[str, Any]:
    emb = state.embedding
    vocab = state.vocab
    if emb is None:
        raise ValueError("no embedding in checkpoint")

    def resolve(t: str) -> tuple[int, str] | None:
        tid = _resolve_id(vocab, t)
        if tid is None:
            return None
        return tid, vocab.token_of(tid)

    a = resolve(token_a)
    b = resolve(token_b)
    if a is None or b is None:
        return {
            "found": False,
            "token_a": token_a,
            "token_b": token_b,
        }

    ta, la = a
    tb, lb = b
    sim = emb.similarity(ta, tb)
    return {
        "found": True,
        "token_a": la,
        "id_a": ta,
        "token_b": lb,
        "id_b": tb,
        "similarity": round(sim, 4),
        "norm_a": round(float(np.linalg.norm(emb.vector(ta))), 4),
        "norm_b": round(float(np.linalg.norm(emb.vector(tb))), 4),
    }


def embedding_search(state: GraphState, query: str, *, limit: int = 40) -> dict[str, Any]:
    """Tìm token vocabulary theo substring (duyệt surface)."""
    q = query.strip().lower()
    vocab = state.vocab
    hits: list[dict[str, Any]] = []
    if not q:
        return {"query": query, "tokens": hits}

    emb = state.embedding
    for i in range(len(vocab)):
        tok = vocab.token_of(i)
        # TK_016: cho phép tìm node Sense:/POS: (trước lọc surface-only → không khám phá được)
        if not tok or (not is_surface_token(tok) and _kind(tok) == "struct"):
            continue
        if q not in tok.lower():
            continue
        row: dict[str, Any] = {"id": i, "token": tok, "kind": _kind(tok)}
        if emb is not None and i < emb.matrix.shape[0]:
            row["norm"] = round(float(np.linalg.norm(emb.vector(i))), 4)
        if state.parents is not None:
            row["parent_df"] = state.parents._df.get(i, 0)
        hits.append(row)
        if len(hits) >= limit:
            break
    return {"query": query, "tokens": hits, "shown": len(hits)}
