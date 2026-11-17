"""
Export đồ thị v3 cho Graph UI — nơ-ron cha + pattern completion (thay đồ thị rule cũ).

Hai khung nhìn:
  - `list_parents`   : duyệt/tìm nơ-ron cha (câu hỏi → đáp, power).
  - `completion_graph`: với một prompt, dựng đồ thị **lan xuôi** (token câu hỏi → các nơ-ron cha
    ứng viên, có điểm) + **lan ngược** (cha thắng → token đáp theo index). Đúng cơ chế docx §1.
"""

from __future__ import annotations

from typing import Any, Literal

from nn_aivie.core.parent_store import _content_ids
from nn_aivie.core.seg_graph import CLUSTER_OFFSET
from nn_aivie.core.state import GraphState
from nn_aivie.hca.config import HCAConfig
from nn_aivie.hca.readout import decode_surface
from nn_aivie.hca.generate_hebbian import (
    _build_agenda,
    _hub_guarded_completion,
    _prefer_atomic_facts,
    _split_frame,
    generate,
)

GenerateMode = Literal["autoregressive"]


def _parent_texts(state: GraphState, idx: int) -> dict[str, str]:
    rec = state.parents.parents[idx]
    tok = state.vocab.token_of
    emit = state.vocab.detokenize(rec.emit_ids)
    answer = state.vocab.detokenize(rec.answer_ids)
    match = state.vocab.detokenize(rec.match_ids)
    return {
        "question": match,
        "answer": answer if rec.is_qa else emit,
        "emit": emit,
    }


def list_parents(
    state: GraphState, query: str = "", limit: int = 60
) -> dict[str, Any]:
    """Duyệt nơ-ron cha (lọc substring nếu có query). Sắp theo power giảm dần."""
    if state.parents is None:
        return {"parents": [], "total": 0, "qa": 0, "statement": 0}
    q = query.strip().lower()
    rows: list[dict[str, Any]] = []
    for idx, rec in enumerate(state.parents.parents):
        texts = _parent_texts(state, idx)
        if q and q not in texts["emit"].lower():
            continue
        rows.append(
            {
                "idx": idx,
                "is_qa": rec.is_qa,
                "power": rec.power,
                "question": texts["question"],
                "answer": texts["answer"],
                "emit": texts["emit"],
            }
        )
    rows.sort(key=lambda r: -r["power"])
    total = len(state.parents.parents)
    qa = sum(p.is_qa for p in state.parents.parents)
    return {
        "parents": rows[:limit],
        "total": total,
        "qa": qa,
        "statement": total - qa,
        "shown": min(limit, len(rows)),
    }



def _append_token_chain(
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    *,
    source_id: str,
    token_ids: list[int],
    tok,
    vocab,
    prefix: str,
    kind: str = "completion",
    edge_kind: str = "complete",
    max_tokens: int = 32,
) -> str:
    """Nối chuỗi token (lan ngược / đáp ghép) — trả id node cuối."""
    prev = source_id
    shown = token_ids[:max_tokens]
    for j, cid in enumerate(shown):
        if tok(cid).startswith("<"):
            continue
        nid = f"{prefix}:{j}"
        nodes.append({"id": nid, "label": "↳ " + vocab.detokenize([cid]), "kind": kind, "index": j})
        edges.append({"source": prev, "target": nid, "weight": 1.0, "kind": edge_kind})
        prev = nid
    if len(token_ids) > max_tokens:
        nid = f"{prefix}:more"
        nodes.append({
            "id": nid,
            "label": f"… +{len(token_ids) - max_tokens} token",
            "kind": kind, "index": max_tokens,
        })
        edges.append({"source": prev, "target": nid, "weight": 1.0, "kind": edge_kind})
        prev = nid
    return prev


def _pick_compose_facts(
    state: GraphState,
    q_content: list[int],
    config: HCAConfig,
    *,
    max_facts: int,
) -> tuple[tuple[int, ...], list[int]]:
    """Chọn fact cho viz agenda multi (facts_about)."""
    ps = state.parents
    focus = ps.top_focus_tokens(tuple(q_content), k=2) if q_content else ()
    picked: list[int] = []
    if focus:
        # TK_019: pick_facts_coherent (episode compose) đã bỏ — chỉ facts_about
        picked = ps.facts_about(
            focus, top_k=max_facts, vocab=state.vocab,
            theme_weight=config.compose_theme_weight,
        )
        picked = _prefer_atomic_facts(picked, ps, state.vocab)
    return focus, picked


def _viz_compose_graph(
    state: GraphState,
    vocab,
    q_content: list[int],
    config: HCAConfig,
    prompt: str,
    prompt_ids: list[int],
    *,
    max_facts: int,
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    comp_ids: list[int] | None = None,
) -> dict[str, Any] | None:
    """Đồ thị Compose: thực thể trọng tâm → N fact → chuỗi đáp ghép."""
    tok = vocab.token_of
    focus, picked = _pick_compose_facts(state, q_content, config, max_facts=max_facts)
    if len(picked) < 2:
        return None
    facts_out: list[dict[str, Any]] = []
    prev_fact_id: str | None = None
    q_set = set(q_content)
    for fi, pi in enumerate(picked):
        rec_f = state.parents.parents[pi]
        text = vocab.detokenize(rec_f.emit_ids)
        fnid = f"fact:{pi}"
        if not any(n["id"] == fnid for n in nodes):
            nodes.append({"id": fnid, "label": text[:56], "kind": "fact", "index": fi})
        for ft in focus:
            if ft in rec_f.emit_ids and ft in q_set:
                edge = {
                    "source": f"q:{ft}", "target": fnid,
                    "weight": 1.0, "kind": "compose_fact",
                }
                if edge not in edges:
                    edges.append(edge)
        if prev_fact_id is not None:
            edges.append({
                "source": prev_fact_id, "target": fnid,
                "weight": 1.0, "kind": "compose_link",
            })
        prev_fact_id = fnid
        facts_out.append({"idx": pi, "text": text})
    if comp_ids is None:
        comp_ids = generate(state, prompt, config=config, max_facts=max_facts, multi=True)
    comp_text = decode_surface(state, comp_ids)
    answer_ids = [
        i for i in comp_ids[len(prompt_ids):]
        if not tok(i).startswith("<")
    ]
    if answer_ids and prev_fact_id:
        _append_token_chain(
            nodes, edges, source_id=prev_fact_id, token_ids=answer_ids, tok=tok, vocab=vocab,
            prefix="comp_out", kind="completion", edge_kind="compose_out", max_tokens=24,
        )
    return {
        "focus": [tok(t) for t in focus],
        "facts": facts_out,
        "answer": comp_text.split("Đáp:")[-1].strip() if "Đáp:" in comp_text else comp_text,
        "full": comp_text,
    }


def _append_viz_edge(edges: list[dict[str, Any]], edge: dict[str, Any]) -> None:
    if edge not in edges:
        edges.append(edge)


def _parent_match_activation_edges(
    ps,
    rec,
    pidx: int,
    q_content: list[int],
    q_set: set[int],
    *,
    is_matched: bool,
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    tok,
    vocab,
) -> None:
    """Cạnh lan xuôi query→cha — delegate TK_023 `ParentStore.stimulus_activation_edges`."""
    del rec, q_set
    ps.stimulus_activation_edges(
        pidx, q_content, vocab,
        is_matched=is_matched, nodes=nodes, edges=edges, tok=tok,
    )


def completion_graph(
    state: GraphState,
    prompt: str,
    *,
    top_candidates: int = 6,
    max_len: int = 12,
    config: HCAConfig | None = None,
    mode: GenerateMode = "autoregressive",
    multi: bool = True,
    max_facts: int = 3,
    cache=None,
) -> dict[str, Any]:
    """Đồ thị lan xuôi (token → cha ứng viên) + lan ngược (cha thắng → đáp theo index).

    TK_012: nhãn episode/tag đọc từ content-cache (UI/debug); label fallback = decode token."""
    config = config or HCAConfig()
    vocab = state.vocab
    tok = vocab.token_of
    prompt_ids = list(vocab.encode_prompt(prompt))
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []

    if state.parents is None or len(state.parents) == 0 or not prompt_ids:
        gen = generate(state, prompt, max_len=max_len, config=config, multi=True)
        ar_text = decode_surface(state, gen)
        return {
            "nodes": nodes, "edges": edges, "candidates": [], "matched": None,
            "generated": ar_text, "completion": "", "hub": None,
            "compose": None, "autoregressive": {"multi": True, "agenda": [], "answer": ar_text, "steps": []},
            "essay": None,
            "fact_stimulus": None,
            "generate_mode": "autoregressive", "hybrid_generated": ar_text,
            "meta": {"n_parents": 0 if state.parents is None else len(state.parents)},
        }

    q_ids, _ = _split_frame(prompt_ids, vocab)
    q_seq = _content_ids(q_ids or prompt_ids, vocab, skip=set())
    q_content = list(dict.fromkeys(q_seq))
    # Hướng 1 (TK_014): kích thích cụm phổ biến (WH/frame) như "ở đâu" — hiện trong đồ thị.
    state.parents.match_common_clusters = config.match_common_clusters
    units = state.parents.query_match_units(q_seq) if q_seq else None

    ranked = state.parents.rank_matches(
        q_ids or prompt_ids, vocab, state.embedding,
        fuzzy_topk=config.hybrid_fuzzy_topk,
        fuzzy_min_sim=config.hybrid_fuzzy_min_sim,
        fuzzy_scale=config.hybrid_fuzzy_weight,
        top_n=top_candidates,
    )
    del cache, mode, multi  # A3 multi là đường duy nhất; tham số giữ tương thích API
    matched_idx = (
        ranked[0][0]
        if ranked and ranked[0][1] >= config.hybrid_parent_min_score
        else None
    )

    # --- query token nodes (tầng dưới) ---
    for i, qid in enumerate(q_content):
        nodes.append({"id": f"q:{qid}", "label": vocab.detokenize([qid]), "kind": "query"})

    # --- Hướng 2 (TK_014): anchor cụm được kích thích bởi câu hỏi (VD "ở đâu") + nhóm gom ---
    anchor_groups: list[dict[str, Any]] = []
    ps_a = state.parents
    if q_seq and ps_a.seg.clusters:
        for ci in ps_a.query_anchor_clusters(tuple(q_seq)):
            c = ps_a.seg.clusters[ci]
            acid = f"cl:{ci}"
            label = vocab.detokenize([c.left, c.right])
            if not any(n["id"] == acid for n in nodes):
                nodes.append(
                    {"id": acid, "label": label, "kind": "cluster",
                     "power": c.power, "activated": True}
                )
            # cạnh q→cl do _parent_match_activation_edges vẽ (có index con + trọng số)
            # nhóm gom: mọi cha X→anchor, filler X (thực thể được hỏi)
            fillers = ps_a.anchor_left_fillers(ci)
            def _flabel(ref) -> str:
                if ref.kind == "cluster":
                    cc = ps_a.seg.clusters[ref.id - CLUSTER_OFFSET]
                    return f"{tok(cc.left)} {tok(cc.right)}"
                return tok(ref.id)
            n_members = len(ps_a.anchor_members(ci))
            if n_members == 0:
                continue  # cụm nhiễu adjacency (đã rewire đi) — không mang thông tin gom
            filler_txts = list(dict.fromkeys(_flabel(r) for r, _pi in fillers))
            anchor_groups.append(
                {"cluster": label, "cluster_idx": ci, "power": c.power,
                 "n_members": n_members,
                 "fillers": filler_txts[:12]}
            )

    # --- nơ-ron mẫu chung (slot hub): câu hỏi thuộc một họ mẫu? (docx Cross-sample) ---
    hub_info: dict[str, Any] | None = None
    hub_id: str | None = None
    hubs = getattr(state.parents, "hubs", None)
    if hubs and q_seq:
        hit = state.parents.hub_for_query(tuple(q_seq))
        if hit is not None:
            hub, slot_tok = hit
            hidx = next(i for i, h in enumerate(hubs) if h is hub)
            hub_id = f"hub:{hidx}"
            left_txt = vocab.detokenize(hub.left)
            right_txt = vocab.detokenize(hub.right)
            skeleton = f"{left_txt} ___ {right_txt}".strip()
            slot_known = slot_tok in hub.fillers
            filler_tokens = [vocab.detokenize([f]) for f in hub.fillers]
            nodes.append(
                {
                    "id": hub_id,
                    "label": skeleton,
                    "kind": "hub",
                    "slot": vocab.detokenize([slot_tok]),
                    "slot_known": slot_known,
                    "fillers": filler_tokens,
                    "n_fillers": len(hub.fillers),
                }
            )
            skel_set = set(hub.left) | set(hub.right)
            for qid in q_content:
                if qid in skel_set:
                    edges.append(
                        {"source": f"q:{qid}", "target": hub_id, "weight": 1.0, "kind": "hub_frame"}
                    )
            if slot_tok in q_content:
                edges.append(
                    {"source": f"q:{slot_tok}", "target": hub_id, "weight": 1.0, "kind": "hub_slot"}
                )
            hub_info = {
                "skeleton": skeleton,
                "slot": tok(slot_tok),
                "slot_known": slot_known,
                "fillers": filler_tokens,
                "n_fillers": len(hub.fillers),
            }

    # --- guard hub: filler LẠ → đáp phải slot-consistent (route/suppress); đồng bộ viz==sinh ---
    guard_answer: tuple[int, ...] | None = None
    guard_suppressed = False
    if hub_info is not None and config.hybrid_hub_guard:
        res = _hub_guarded_completion(state, q_ids, prompt_ids, config)
        if res is not False:  # filler lạ: hub can thiệp → không còn một cha "thắng" sạch
            matched_idx = None
            if res:
                guard_answer = res
            else:
                guard_suppressed = True
            if hub_info is not None:
                hub_info["guard"] = "reroute" if guard_answer else "suppress"

    # --- candidate parent nodes + cạnh lan xuôi (phân rã thật từng cha) ---
    ps = state.parents
    q_set = set(q_content)
    candidates_out: list[dict[str, Any]] = []
    for rank, (pidx, score) in enumerate(ranked):
        rec = state.parents.parents[pidx]
        texts = _parent_texts(state, pidx)
        is_matched = pidx == matched_idx
        nodes.append(
            {
                "id": f"p:{pidx}",
                "label": texts["question"] or texts["emit"],
                "kind": "parent" if is_matched else "parent_alt",
                "score": round(score, 3),
                "is_qa": rec.is_qa,
                "answer": texts["answer"],
            }
        )
        candidates_out.append(
            {
                "idx": pidx, "score": round(score, 3), "rank": rank,
                "is_qa": rec.is_qa, "question": texts["question"],
                "answer": texts["answer"], "matched": is_matched,
            }
        )
        _parent_match_activation_edges(
            ps, rec, pidx, q_content, q_set,
            is_matched=is_matched, nodes=nodes, edges=edges, tok=tok, vocab=vocab,
        )

    # --- lan ngược: cha thắng (hoặc hub route) → token đáp theo index (tầng trên) ---
    completion_text = ""
    comp_ids: tuple[int, ...] = ()
    comp_source: str | None = None
    if guard_answer is not None:  # hub route novel filler → đáp từ hub
        comp_ids = guard_answer
        comp_source = hub_id
    elif matched_idx is not None:
        comp_ids = state.parents.completion(matched_idx)
        comp_source = f"p:{matched_idx}"
    if comp_ids and comp_source is not None and not guard_suppressed:
        completion_text = vocab.detokenize(comp_ids)
        prev = comp_source
        for j, cid in enumerate(comp_ids):
            nid = f"c:{j}"
            nodes.append(
                {"id": nid, "label": "↳ " + vocab.detokenize([cid]), "kind": "completion", "index": j}
            )
            edges.append(
                {"source": prev, "target": nid, "weight": 1.0, "kind": "complete"}
            )
            prev = nid

    compose_info = None
    comp_text = ""

    # --- A3: generate() token-by-token (multi) + trace cho viz ---
    q_for_agenda = q_ids or prompt_ids
    agenda_ids = _build_agenda(
        state, q_for_agenda, config, max_facts=max_facts, multi=True
    )
    ar_steps: list[dict[str, object]] = []
    fact_plan_holder: dict[str, Any] = {}
    ar_ids = generate(
        state,
        prompt,
        max_len=max(max_len, 48),
        config=config,
        multi=True,
        max_facts=max_facts,
        trace_out=ar_steps,
        fact_plan_out=fact_plan_holder,
    )
    fact_stimulus = fact_plan_holder or None
    ar_text = decode_surface(state, ar_ids)
    ar_answer = ar_text.split("Đáp:")[-1].strip() if "Đáp:" in ar_text else ar_text
    ar_info = {
        "multi": True,
        "agenda": [tok(t) for t in agenda_ids],
        "answer": ar_answer,
        "steps": ar_steps,
    }

    compose_info = _viz_compose_graph(
        state, vocab, q_content, config, prompt, prompt_ids,
        max_facts=max_facts, nodes=nodes, edges=edges,
        comp_ids=ar_ids,
    )
    comp_text = compose_info["full"] if compose_info else ""
    new_ids = ar_ids[len(prompt_ids):]
    if new_ids:
        prev_ar = comp_source or (
            f"q:{q_content[-1]}" if q_content else f"q:{prompt_ids[-1]}"
        )
        for j, cid in enumerate(new_ids):
            if tok(cid).startswith("<"):
                continue
            nid = f"ar:{j}"
            nodes.append(
                {"id": nid, "label": vocab.detokenize([cid]), "kind": "ar_token", "index": j}
            )
            edges.append(
                {"source": prev_ar, "target": nid, "weight": 1.0, "kind": "ar_step"}
            )
            prev_ar = nid

    generated = ar_text

    return {
        "nodes": nodes,
        "edges": edges,
        "candidates": candidates_out,
        "matched": matched_idx,
        "completion": completion_text,
        "generated": generated,
        "hybrid_generated": generated,
        "hub": hub_info,
        "anchor_groups": anchor_groups,
        "compose": compose_info,
        "essay": None,
        "autoregressive": ar_info,
        "fact_stimulus": fact_stimulus,
        "generate_mode": "autoregressive",
        "meta": {
            "n_parents": len(state.parents),
            "n_hubs": len(hubs) if hubs else 0,
            "q_content": [tok(q) for q in q_content],
            "min_score": config.hybrid_parent_min_score,
        },
    }
