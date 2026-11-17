#!/usr/bin/env python
"""P85 (TK_045 §13.59) — BA NHÁNH: bản lề vs **truy hồi cha** vs **chép nguyên câu**.

**Vì sao.** Mười ba cơ chế đã thử **đều** xếp lại hoặc lọc bớt cái bể ứng viên do **bản lề**
sinh ra (`last_content(A) → first_content(B)` qua `W_skip`). Không cơ chế nào đổi **nguồn sinh
ra bể**. Chúng cùng chết vì cùng một lý do: không vá được một đại lượng thiếu thông tin bằng
cách xếp lại nó.

Bất đối xứng đã đo được, và chưa ai khai thác:

| việc | hệ làm tốt cỡ nào |
|---|---|
| **truy hồi CHA** theo cả đề (`rank_matches`) | +31pp (§12.122) · +13.5pp (§12.126) |
| **xếp mối nối** theo bản lề | hạng trung vị **51/125** — ngang ngẫu nhiên (§13.31) |

Ở `max_clauses=1` (mặc định từ §13.52), **chất lượng đầu ra = chất lượng ĐÚNG MỘT phép chọn**.
Đổi bộ chọn từ mức ngẫu nhiên sang mức +31pp là đòn bẩy lớn nhất còn lại.

## Ba nhánh

| nhánh | chọn thế nào |
|---|---|
| `ban_le` | hiện tại — bản lề sinh bể, cổng fold lọc, `max_clauses=1` |
| `truy_hoi_cha` | `rank_matches(ĐỀ)` → cha tốt nhất **khác cha của đề** → mệnh đề MỞ của cha đó |
| `chep_cau` | `rank_matches(ĐỀ)` → cha tốt nhất → **phát nguyên emit của cha** |

`chep_cau` là **TRẦN TRÊN**, cố ý:
- nếu nó cũng thấp ⇒ vấn đề nằm ở **truy hồi**, không ở ghép, và mọi công vào khâu sinh là
  nhắm sai chỗ;
- nếu nó cao ⇒ biết chính xác kiến trúc đang mất bao nhiêu vì cố **ghép**.

§12.72 từng loại hướng chép vì "chép 0,93–1,00" — nhưng lúc đó **chưa ai biết cái thay thế cho
ra 87% hỏng** (§13.43). Ba tiêu chí của dự án là *đúng ngữ cảnh · không salad · trôi chảy*;
"sáng tạo" không nằm trong đó. Nên trần này **phải** được đo, không được giả định.

Ba nhánh chạy trên **BỘ ĐỀ RỜI NHAU** (§13.51): `chep_cau` phát nguyên câu nên nếu dùng chung
đề, giám khảo ghép cặp được ngay và biết nhánh nào là nhánh nào.

Chạy:
  PYTHONPATH=src python scripts/gen_three_arm.py --n 150
"""
from __future__ import annotations

import argparse
import gc
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from nn_aivie.core.checkpoint_io import load_checkpoint  # noqa: E402
from nn_aivie.hca.clause_layer import (  # noqa: E402
    CoverageGate,
    build_clause_layer,
    compose_chain,
    content_folds,
)
import numpy as np  # noqa: E402

from nn_aivie.hca.context_positional import is_structural_token  # noqa: E402
from nn_aivie.hca.readout import decode_surface  # noqa: E402
from nn_aivie.hca.w_skip import ensure_w_skip  # noqa: E402
from nn_aivie.util.progress import get_run_logger  # noqa: E402

EMPTY = "(TRỐNG — hệ không sinh được gì)"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/mini_sent_ep_qa.pkl")
    ap.add_argument("--n", type=int, default=150, help="số đề MỖI nhánh (bộ đề rời nhau)")
    ap.add_argument("--top-n", type=int, default=8, help="top-N của rank_matches")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--arms", default="ban_le,chep_cau,ar",
                    help="danh sách nhánh, cách nhau bởi dấu phẩy")
    ap.add_argument("--ws-model", default="out/models/ws_crf_uts_full")
    ap.add_argument("--label-map", default="out/token_labels_mini.json",
                    help="bản đồ nhãn để CẤM thay khe `số` (TK_047 §17/P102)")
    ap.add_argument("--thay-lap-gap", action="store_true",
                    help="thay để LẤP phần đề khuôn bỏ sót (§10) thay vì tăng chồng lấp")
    ap.add_argument("--min-share", type=int, default=3,
                    help="bối cảnh chung tối thiểu khi dựng lớp thay thế (TK_048)")
    ap.add_argument("--topk-lop", type=int, default=8,
                    help="số bạn cùng lớp giữ lại mỗi đơn vị")
    ap.add_argument("--max-thay", type=int, default=0,
                    help="P108: trần số khe thay cho ep_da_khe_* (0 = không trần)")
    ap.add_argument("--slot-rank", choices=("ncand", "bam"), default="ncand",
                    help="P109: xếp khe theo |ứng viên| (ncand) hoặc độ tăng bám đề (bam)")
    ap.add_argument("--min-gain", type=int, default=0,
                    help="P110: chỉ đề xuất khe khi bam(best)-bam(x) ≥ ngưỡng (với slot-rank=bam)")
    ap.add_argument("--no-co-graph", action="store_true",
                    help="P109 ablation: không bổ sung CO_ACTIVATION graph vào _co")
    ap.add_argument("--decode-unit", choices=("token", "thuc_the"), default="token",
                    help="P114: điền khe theo token (P112) hoặc đơn vị thuc_the/entity")
    ap.add_argument("--era-filter", action="store_true",
                    help="P115: cấm ứng viên nhảy triều/niên đại so với đề")
    ap.add_argument("--soft-giao-single-only", action="store_true",
                    help="P115: soft-giao (lop) chỉ cho đơn vị 1-token")
    ap.add_argument("--hybrid-if-zero", action="store_true",
                    help="P115: token hybrid chỉ khi 0 thay ở tầng đơn vị")
    ap.add_argument("--second-min-gain", type=int, default=0,
                    help="P115: khe thứ 2+ cần bam-gain ≥ ngưỡng")
    ap.add_argument("--join-d-max", type=int, default=8,
                    help="P203 ship: khoảng cách W_skip tối đa khi kiểm bản lề A→B")
    ap.add_argument("--join-min-count", type=float, default=1.0,
                    help="P203 ship: tổng count W_skip tối thiểu để cho phép nối")
    ap.add_argument("--out", default="out/three_samples.txt")
    ap.add_argument("--out-key", default="out/three_key.json")
    ap.add_argument("--out-meta", default="out/three_meta.json")
    args = ap.parse_args()

    logger = get_run_logger("gen_three_arm")
    t0 = time.time()
    logger.log(f"P85 ba nhánh · {args.checkpoint} · {args.n} đề/nhánh · BỘ ĐỀ RỜI NHAU")

    gc.disable()
    state, _ = load_checkpoint(str(_ROOT / args.checkpoint))
    gc.enable()
    layer = build_clause_layer(state, pmi_theta=0.0, logger=logger)
    ws = ensure_w_skip(state, d_max=8, min_count=1, content_only=False)
    ps = state.parents
    logger.log(f"tầng mệnh đề {len(layer):,} ({time.time()-t0:.0f}s)")

    rng = random.Random(args.seed)
    openers = [c for c in layer.clauses if c.opens_parent and len(c.tokens) >= 4]
    rng.shuffle(openers)
    arm_names = [x.strip() for x in args.arms.split(",") if x.strip()]
    need = len(arm_names) * args.n
    if len(openers) < need:
        raise SystemExit(f"⛔ không đủ đề rời nhau: cần {need}, có {len(openers)}")
    slices = [openers[i * args.n:(i + 1) * args.n] for i in range(len(arm_names))]

    def ranked_parent(p) -> int | None:
        """Cha tốt nhất theo `rank_matches`, LOẠI cha của chính đề.

        Không loại thì nhánh này thành 'phát lại chính cha của đề' — tức đo trí nhớ, không
        đo truy hồi. Đúng cái bẫy `KHÁC-CHA 99.8%` của §13.6 đã dựng để tránh.
        """
        got = ps.rank_matches(
            list(p.tokens), state.vocab, getattr(state, "embedding", None),
            top_n=args.top_n)
        for pi, _sc in got:
            if int(pi) != int(p.parent_idx):
                return int(pi)
        return None

    def arm_ban_le(p) -> tuple[str, dict]:
        gate = CoverageGate(
            theme_folds=frozenset(content_folds(state, p.tokens)), anchor_min_folds=1)
        ch, st = compose_chain(state, layer, ws, p.tokens, gate=gate,
                               rng=random.Random(args.seed))
        return (decode_surface(state, [int(t) for c in ch for t in c.tokens]).strip(),
                {"n_clauses": len(ch), "stop": st["stop"]})

    def arm_truy_hoi(p) -> tuple[str, dict]:
        pi = ranked_parent(p)
        if pi is None:
            return "", {"n_clauses": 0, "stop": "khong_truy_hoi"}
        ids = layer.by_parent.get(pi) or []
        opens = [i for i in ids if layer.clauses[i].opens_parent] or ids
        if not opens:
            return "", {"n_clauses": 0, "stop": "cha_khong_menh_de"}
        cl = layer.clauses[opens[0]]
        return (decode_surface(state, [int(t) for t in cl.tokens]).strip(),
                {"n_clauses": 1, "stop": "truy_hoi", "parent": pi})

    # --- §13.64 — điều kiện hoá theo CHỦ THỂ bằng E ---------------------------------------
    # §13.63-A: biết chủ thể hơn 2,19× so với chỉ biết khung; §13.63-B: E khôi phục liên kết
    # CHƯA TỪNG THẤY ở 2,44× sàn. `rank_matches` đã dùng E cho khớp mờ, nhưng KHÔNG xếp lại
    # theo độ gần CHỦ THỂ ↔ ỨNG VIÊN — đó là chỗ trống này lấp.
    _E = np.asarray(state.embedding.matrix, dtype=np.float64)
    if _E.shape[0] < len(state.vocab):
        _E = np.pad(_E, ((0, len(state.vocab) - _E.shape[0]), (0, 0)))
    _En = _E / (np.linalg.norm(_E, axis=1, keepdims=True) + 1e-12)

    def _vec(ids) -> np.ndarray | None:
        rows = [int(t) for t in ids
                if 0 <= int(t) < len(state.vocab)
                and state.vocab.is_surface(int(t))
                and not is_structural_token(state.vocab, int(t))]
        if not rows:
            return None
        w = _En[rows].mean(axis=0)
        nrm = float(np.linalg.norm(w))
        return w / nrm if nrm > 0 else None

    def arm_chep_e(p) -> tuple[str, dict]:
        """Truy hồi rồi XẾP LẠI top-N theo độ gần chủ-thể↔cha trên E."""
        got = ps.rank_matches(
            list(p.tokens), state.vocab, getattr(state, "embedding", None),
            top_n=args.top_n)
        cands = [int(pi) for pi, _ in got if int(pi) != int(p.parent_idx)]
        if not cands:
            return "", {"n_clauses": 0, "stop": "khong_truy_hoi"}
        qv = _vec(p.tokens)
        if qv is None:
            best = cands[0]
        else:
            def sim(pi: int) -> float:
                pv = _vec(ps.parents[pi].emit_ids)
                return float(qv @ pv) if pv is not None else -1.0
            best = max(cands, key=sim)
        emit = list(ps.parents[best].emit_ids)
        return (decode_surface(state, [int(t) for t in emit]).strip(),
                {"n_clauses": -1, "stop": "chep_E", "parent": best})

    # --- TK_046 §11 — ĐƯỜNG AR: sinh TỪNG TOKEN, chưa từng được đo tuyệt đối --------------
    # Mọi con số 0,5% / 13,0% / 53,3% đều thuộc đường tầng-mệnh-đề và đường CHÉP. Đường AR
    # (`generate()`) là nơi ý tưởng suy-luận-cạnh có hiệu lực thật — cạnh `khỉ → động vật`
    # đẩy token kế, tức hệ TỰ PHÁT ra câu chưa từng tồn tại, thay vì chép câu của người khác.
    # Không có mốc này thì bật suy luận lên cũng không biết nó cải thiện hay làm hỏng gì.
    from nn_aivie.hca.config import HCAConfig  # noqa: PLC0415
    from nn_aivie.hca.generate_hebbian import generate  # noqa: PLC0415

    _cfg_ar = HCAConfig()

    def arm_ar(p) -> tuple[str, dict]:
        pr_ids = [int(t) for t in p.tokens]
        pr_txt = decode_surface(state, pr_ids).strip()
        try:
            out = generate(state, pr_txt, config=_cfg_ar, seed=args.seed)
        except Exception as e:  # noqa: BLE001
            return "", {"n_clauses": 0, "stop": f"loi_{type(e).__name__}"}
        ids = [int(t) for t in out]
        # bỏ tiền tố ĐỀ để chỉ còn phần SINH — so được với các nhánh khác
        k = 0
        while k < min(len(ids), len(pr_ids)) and ids[k] == pr_ids[k]:
            k += 1
        cont = decode_surface(state, ids[k:]).strip()
        return cont, {"n_clauses": -2, "stop": "ar", "n_tok": len(ids) - k}

    def arm_chep(p) -> tuple[str, dict]:
        pi = ranked_parent(p)
        if pi is None:
            return "", {"n_clauses": 0, "stop": "khong_truy_hoi", "ent_keep": 0.0}
        emit = [int(t) for t in ps.parents[pi].emit_ids]
        return (decode_surface(state, emit).strip(),
                {"n_clauses": -1, "stop": "chep", "parent": pi,
                 "n_src": 1, "parents": [pi],
                 "ent_keep": _ent_keep(p, emit)})


    # --- TK_047 §9 — KHUÔN CỤ THỂ + THAY KHE ----------------------------------------------
    # P99 qua cổng: (a,c,b) có thật ở cha khác ⇒ c thay được x. Ở đây thêm hai thứ P99 chưa
    # có, vì P99 chỉ đo SUBSTRATE chứ không sinh: (1) CHỦ ĐỀ của ĐỀ lái phép thay — thay chỉ
    # khi ứng viên bám đề CHẶT HƠN token gốc; (2) lọc VỠ TỪ bằng WS-CRF (`bắt ông → bắt đầu`).
    from nn_aivie.ws_crf.ws_model import WSCRFModel  # noqa: PLC0415
    _wsm = WSCRFModel.load(_ROOT / args.ws_model)
    # TK_047 §17 (P102) — CẤM THAY KHE MANG NHÃN `số`. Kiểm sự thật trên 66 mẫu cho thấy
    # 100% lỗi sự thật nằm ở NIÊN ĐẠI (tên người/địa danh khớp hết), và hai lỗi là CẶP: cùng
    # sự kiện, một bản đúng một bản lệch ⇒ phép thay phá một niên đại VỐN ĐÚNG ở chỗ khác.
    _so: set[int] = set()
    _mf = _ROOT / args.label_map
    if _mf.exists():
        _m = json.loads(_mf.read_text(encoding="utf-8")).get("tu_loai") or {}
        for _ts, _d in _m.items():
            if _d and max(_d, key=_d.get) == "số":
                _so.add(int(_ts))
        logger.log(f"cấm thay khe `số`: {len(_so):,} token")
    else:
        logger.log(f"⚠️ không có {args.label_map} ⇒ KHÔNG cấm được khe `số` (P102 §17)")
    _nv = len(state.vocab)
    _keep = np.zeros(_nv, dtype=bool)
    for _t in range(_nv):
        if is_structural_token(state.vocab, _t) or not state.vocab.is_surface(_t):
            continue
        try:
            if float(ps._gidf(_t)) >= 4.0:      # noqa: SLF001
                _keep[_t] = True
        except Exception:
            continue

    def _ent_keep(p, out_ids: list[int]) -> float:
        """Tỷ lệ token keep của đề còn xuất hiện trong emit (TK_051 cổng 2)."""
        th = {int(t) for t in p.tokens if 0 <= int(t) < _nv and _keep[int(t)]}
        if not th:
            return 1.0
        ou = {int(t) for t in out_ids if 0 <= int(t) < _nv}
        return round(len(th & ou) / len(th), 4)
    _tri: dict[tuple[int, int], set[int]] = {}
    _co: dict[int, set[int]] = {}
    from nn_aivie.core.edge_kinds import EdgeKind  # noqa: PLC0415
    for _rec in ps.parents:
        _e = [int(t) for t in _rec.emit_ids]
        for _i in range(1, len(_e) - 1):
            if 0 <= _e[_i] < _nv and _keep[_e[_i]]:
                _tri.setdefault((_e[_i - 1], _e[_i + 1]), set()).add(_e[_i])
        _u = {t for t in _e if 0 <= t < _nv and _keep[t]}
        for _t in _u:
            _co.setdefault(_t, set()).update(_u)
    # P108 — bổ sung hàng xóm CO_ACTIVATION từ graph (Hebbian kế thừa tính chất).
    # Giới hạn top-16/token để không bão hoà kiểu §13.20.
    _n_co_g = 0
    if not args.no_co_graph:
        _co_layer = state.adjacency.layer(EdgeKind.CO_ACTIVATION)
        for _t in list(_co.keys()):
            _row = _co_layer.row(_t)
            if not _row:
                continue
            _top = sorted(
                ((float(_w), int(_j)) for _j, _w in _row.items()
                 if float(_w) >= 0.05 and 0 <= int(_j) < _nv and _keep[int(_j)]
                 and int(_j) != _t),
                reverse=True,
            )[:16]
            for _w, _j in _top:
                if _j not in _co[_t]:
                    _co[_t].add(_j)
                    _n_co_g += 1
    logger.log(f"khuôn: bối cảnh {len(_tri):,} · token có đồng hiện {len(_co):,}"
               f" · +CO graph {_n_co_g:,}"
               + (" (tắt)" if args.no_co_graph else ""))

    # TK_048 / TK_049 — lớp thay thế (paradigmatic). Ưu tiên đọc cạnh SYNONYM đã bake
    # (P106); nếu checkpoint chưa bake thì tính lại tại chỗ như P103/P105.
    _lop_nb: dict[int, list[int]] = {}
    _need_lop = any(
        a in ("ep_giao", "ep_lop", "ep_mot_khe_giao", "ep_da_khe_giao", "compose_2cha")
        or a.endswith("_giao") or a.endswith("_lop")
        for a in arm_names)
    _syn_layer = state.adjacency.layer(EdgeKind.SYNONYM)
    # đếm nhanh: frozen CSR dùng len(indices); thawed dùng len(_data)
    if getattr(_syn_layer, "_frozen", False):
        _n_syn_edges = int(len(getattr(_syn_layer, "_indices", ())))
    else:
        _n_syn_edges = len(getattr(_syn_layer, "_data", {}) or {})
    if _need_lop and _n_syn_edges > 0:
        for _t in range(_nv):
            if not _keep[_t]:
                continue
            row = _syn_layer.row(_t)
            if not row:
                continue
            ranked = sorted(
                ((float(w), int(j)) for j, w in row.items()
                 if 0 <= int(j) < _nv and _keep[int(j)] and int(j) != _t),
                reverse=True)
            if ranked:
                _lop_nb[_t] = [j for _w, j in ranked[: args.topk_lop]]
        logger.log(f"lớp thay thế từ SYNONYM bake: {len(_lop_nb):,} token "
                   f"({_n_syn_edges:,} cạnh hướng)")
    elif _need_lop:
        ctx_all: dict[int, set] = defaultdict(set)
        for _rec in ps.parents:
            _e = [int(t) for t in _rec.emit_ids]
            for _i in range(1, len(_e) - 1):
                if 0 <= _e[_i] < _nv and _keep[_e[_i]]:
                    ctx_all[_e[_i]].add((_e[_i - 1], _e[_i + 1]))
        ctx_all = {t: c for t, c in ctx_all.items() if len(c) >= 3}
        inv: dict[tuple, list[int]] = defaultdict(list)
        for t, cs in ctx_all.items():
            for c in cs:
                inv[c].append(t)
        for x, cs in ctx_all.items():
            sh: dict[int, int] = defaultdict(int)
            for c in cs:
                for y in inv[c]:
                    if y != x:
                        sh[y] += 1
            nx = len(cs) ** 0.5
            sc = sorted(((m / (nx * len(ctx_all[y]) ** 0.5), y)
                         for y, m in sh.items() if m >= args.min_share), reverse=True)
            if sc:
                _lop_nb[x] = [y for _s, y in sc[: args.topk_lop]]
        logger.log(f"lớp thay thế (tính lại): {len(_lop_nb):,} đơn vị có bạn cùng lớp "
                   f"(min_share={args.min_share}, topk={args.topk_lop})")

    from nn_aivie.hca.frame_slot_gen import (  # noqa: PLC0415
        FrameSlotConfig,
        FrameSlotIndex,
        build_unit_slot_maps,
        fill_da_khe,
        load_proper_noun_tokens,
    )
    _pn: set[int] = set()
    _unit_tri = _unit_lop = _rep_tokens = None
    if any(a == "ep_da_khe_giao" for a in arm_names) and args.decode_unit == "thuc_the":
        _pn = load_proper_noun_tokens(_ROOT / args.label_map)
        if not _pn:
            raise SystemExit("⛔ --decode-unit thuc_the cần nhãn `tên riêng` trong label-map")
        logger.log(f"P114 decode thuc_the · token tên riêng {len(_pn):,}")
        _ucfg = FrameSlotConfig(decode_unit="thuc_the", topk_lop=args.topk_lop)
        _unit_tri, _unit_lop, _rep_tokens = build_unit_slot_maps(
            state, _keep, _ucfg, _wsm, _pn, logger=logger)
        logger.log(
            f"chỉ mục đơn vị: bối cảnh {len(_unit_tri):,} · lớp {len(_unit_lop):,}"
        )
    _frame_slot_idx = FrameSlotIndex(
        keep=_keep, so=_so, tri=_tri, co=_co, lop_nb=_lop_nb, ws_model=_wsm,
        n_syn_edges=_n_syn_edges, n_co_graph=_n_co_g, pn=_pn,
        unit_tri=_unit_tri, unit_lop=_unit_lop, rep_tokens=_rep_tokens,
    )

    def _vo_tu(ids: list[int], k: int) -> bool:
        """Phép thay ở vị trí k có làm VỠ TỪ không (WS-CRF nói token k dính vào láng giềng)."""
        txt = decode_surface(state, ids)
        syl = txt.split()
        if len(syl) < 2:
            return False
        tags = _wsm.tag(syl)
        if len(tags) != len(syl):
            return False
        pre = decode_surface(state, ids[: k + 1])
        if not txt.startswith(pre) or txt[len(pre):][:1].isalnum():
            return True                     # không căn được ⇒ từ chối, phía an toàn
        si = len(pre.split()) - 1
        if not (0 <= si < len(tags)):
            return True
        return tags[si] == "I-W" or (si + 1 < len(tags) and tags[si + 1] == "I-W")

    def arm_khuon(p) -> tuple[str, dict]:
        pi = ranked_parent(p)
        if pi is None:
            return "", {"n_clauses": 0, "stop": "khong_truy_hoi"}
        ids = [int(t) for t in ps.parents[pi].emit_ids]
        theme = {int(t) for t in p.tokens if 0 <= int(t) < _nv and _keep[int(t)]}
        if not theme:
            return (decode_surface(state, ids).strip(),
                    {"n_clauses": -3, "stop": "khuon_khong_chu_de", "parent": pi,
                     "n_thay": 0})

        # ⚠️ LỖI THIẾT KẾ đã đo (§10): bản đầu thay khi `|đồng hiện(c) ∩ chủ đề|` LỚN HƠN của
        # token gốc. Nhưng `rank_matches` ĐÃ chọn cha bám đề nhất ⇒ token gốc gần như luôn
        # thắng ⇒ 94,5% ca trùng nguyên văn là "có ứng viên mà không cái nào bám hơn". Phép
        # thay đang tối ưu ĐÚNG hàm mục tiêu truy hồi vừa tối ưu xong.
        #
        # Sửa: nhắm phần đề khuôn **BỎ SÓT**. `gap` = token chủ đề của đề mà khuôn không phủ.
        # Đây là hàm mục tiêu truy hồi KHÔNG tối ưu, nên không tự triệt tiêu.
        # ⚠️ Bản đầu của `gap` dùng ĐỒNG HIỆN để xét "khuôn đã phủ chưa" ⇒ gap luôn RỖNG và
        # 0/120 phép thay, vì token phổ biến đồng hiện với mọi thứ (§13.20 lần thứ n). Định
        # nghĩa lại KHÔNG dùng đồng hiện: gap = token chủ đề của đề mà khuôn KHÔNG CÓ MẶT.
        phu = {t for t in ids if 0 <= t < _nv and _keep[t]}
        gap = theme - phu

        def bam(c: int) -> int:
            return len(_co.get(c, ()) & theme)

        def lap_gap(c: int) -> int:
            """Ứng viên có ĐƯA VÀO một token của đề mà khuôn đang thiếu không.

            Chỉ xét c CÓ MẶT trong gap — không dùng `co[c] & gap`, vì vế đó bão hoà (§13.20)
            và biến mọi ứng viên thành "có lấp gap". Luật này bảo thủ nhưng CHÍNH XÁC: đầu ra
            phải nói tới thứ đề nói tới."""
            return 1 if c in gap else 0

        # Điền THEO ĐỘ CHẮC CHẮN: khe có mức tăng bám-đề LỚN NHẤT đi trước (TK_047 §3b).
        prop: list[tuple[int, int, int]] = []      # (mức tăng, vị trí, ứng viên)
        for i in range(1, len(ids) - 1):
            x = ids[i]
            if not (0 <= x < _nv) or not _keep[x] or x in _so:
                continue
            cands = _tri.get((ids[i - 1], ids[i + 1]), set()) - {x} - _so
            if not cands:
                continue
            if args.thay_lap_gap:
                base = lap_gap(x)
                best = max(cands, key=lambda c: (lap_gap(c), bam(c), -c))
                if lap_gap(best) > base:            # ứng viên chạm phần đề khuôn BỎ SÓT
                    prop.append((lap_gap(best) - base, i, best))
            else:
                base = bam(x)
                best = max(cands, key=lambda c: (bam(c), -c))
                if bam(best) > base:                # chỉ thay khi CHẶT HƠN, không thay bừa
                    prop.append((bam(best) - base, i, best))
        prop.sort(reverse=True)

        n_thay = 0
        for _gain, i, c in prop:
            old = ids[i]
            ids[i] = c
            if _vo_tu(ids, i):
                ids[i] = old                        # hoàn tác: phép thay làm vỡ từ
                continue
            n_thay += 1
        goc = [int(t) for t in ps.parents[pi].emit_ids]
        return (decode_surface(state, ids).strip(),
                {"n_clauses": -3, "stop": "khuon", "parent": pi, "n_thay": n_thay,
                 "n_khe_de_xuat": len(prop), "trung_nguyen_van": ids == goc})

    def _arm_ep(mode: str, licence: str = "bo_ba"):
        """TK_047 §12 — ÉP THAY đúng 1 khe, để tách HƯ HẠI khỏi CHỌN MẪU.

        §11 đo `có thay` thấp hơn `trùng nguyên văn` 37,2 điểm, nhưng phép tách đó **không
        ngẫu nhiên hoá**: luật thay chỉ kích hoạt khi truy hồi tìm được khuôn KÉM khớp, nên
        nhóm "có thay" có thể vốn đã khó hơn. Ở đây khuôn được bốc ngẫu nhiên vào nhánh ⇒
        hai nhóm cùng phân bố độ khó.

        `mode="ngau_nhien"` bốc ứng viên ĐỀU trong các ứng viên hợp lệ ⇒ đo phép cấp phép
        MỘT MÌNH có đủ không. `mode="chu_de"` bốc theo điểm bám đề ⇒ hiệu số hai nhánh là
        GIÁ TRỊ RIÊNG của tiêu chí chủ đề.

        `licence` (TK_048): `bo_ba` = (a,c,b) có thật · `lop` = bạn cùng lớp ·
        `giao` = giao cả hai.
        """
        def _fn(p) -> tuple[str, dict]:
            pi = ranked_parent(p)
            if pi is None:
                return "", {"n_clauses": 0, "stop": "khong_truy_hoi"}
            ids = [int(t) for t in ps.parents[pi].emit_ids]
            theme = {int(t) for t in p.tokens if 0 <= int(t) < _nv and _keep[int(t)]}
            r = random.Random(args.seed * 7919 + pi + hash(licence) % 9973)
            slots = []
            for i in range(1, len(ids) - 1):
                x = ids[i]
                if not (0 <= x < _nv) or not _keep[x] or x in _so:
                    continue
                _tri_c = _tri.get((ids[i - 1], ids[i + 1]), set()) - {x} - _so
                _lop_c = set(_lop_nb.get(x, ())) - {x} - _so
                if licence == "lop":
                    cs = _lop_c
                elif licence == "giao":
                    cs = _tri_c & _lop_c
                else:
                    cs = _tri_c
                if cs:
                    slots.append((i, sorted(cs)))
            r.shuffle(slots)
            for i, cs in slots:
                if mode == "chu_de" and theme:
                    c = max(cs, key=lambda z: (len(_co.get(z, ()) & theme), -z))
                else:
                    c = r.choice(cs)
                old = ids[i]
                ids[i] = c
                if _vo_tu(ids, i):
                    ids[i] = old
                    continue
                return (decode_surface(state, ids).strip(),
                        {"n_clauses": -4, "stop": f"ep_{licence}_{mode}", "parent": pi,
                         "n_thay": 1, "licence": licence})
            return (decode_surface(state, ids).strip(),
                    {"n_clauses": -4, "stop": f"ep_{licence}_{mode}_khong_thay_duoc",
                     "parent": pi, "n_thay": 0, "trung_nguyen_van": True,
                     "licence": licence})
        return _fn

    def _arm_da_khe(licence: str = "giao"):
        """TK_049 — điền nhiều khe giao (P113: luật trong frame_slot_gen)."""

        def _fn(p) -> tuple[str, dict]:
            pi = ranked_parent(p)
            if pi is None:
                return "", {"n_clauses": 0, "stop": "khong_truy_hoi"}
            goc = [int(t) for t in ps.parents[pi].emit_ids]
            theme = {int(t) for t in p.tokens if 0 <= int(t) < _nv and _keep[int(t)]}
            fcfg = FrameSlotConfig(
                licence=licence,
                slot_rank=args.slot_rank,
                min_gain=args.min_gain,
                max_thay=args.max_thay,
                no_co_graph=args.no_co_graph,
                topk_lop=args.topk_lop,
                min_share=args.min_share,
                seed=args.seed,
                decode_unit=args.decode_unit,
                era_filter=args.era_filter,
                soft_giao_single_only=args.soft_giao_single_only,
                hybrid_if_zero=args.hybrid_if_zero,
                second_min_gain=args.second_min_gain,
            )
            ids, m = fill_da_khe(state, goc, theme, _frame_slot_idx, fcfg, parent_idx=pi)
            m["ent_keep"] = _ent_keep(p, ids)
            m["n_src"] = 1
            m["parents"] = [pi]
            return (decode_surface(state, ids).strip(),
                    {"n_clauses": -5, **m})
        return _fn

    def arm_compose_2cha(p) -> tuple[str, dict]:
        """TK_051 ship — một luật với `nn_aivie.hca.compose_2cha`."""
        from nn_aivie.hca.compose_2cha import ComposeConfig, compose_two_parents
        ccfg = ComposeConfig(
            join_d_max=args.join_d_max,
            join_min_count=args.join_min_count,
            top_n=max(int(args.top_n), 16),
            max_thay=args.max_thay,
            min_gain=args.min_gain,
            slot_rank=args.slot_rank,
            no_co_graph=args.no_co_graph,
            seed=args.seed,
        )
        ids, m = compose_two_parents(
            state, list(p.tokens), index=_frame_slot_idx, cfg=ccfg, ws=ws,
            skip_parent=int(p.parent_idx),
        )
        if not ids:
            return "", {"n_clauses": 0, **m}
        # Nối surface có khoảng trắng giữa hai emit khi joined
        if m.get("joined") and len(m.get("parents") or []) >= 2:
            pa, pb = m["parents"][0], m["parents"][1]
            ta = decode_surface(state, list(ps.parents[pa].emit_ids)).strip()
            tb = decode_surface(state, list(ps.parents[pb].emit_ids)).strip()
            out = f"{ta} {tb}".strip()
        else:
            out = decode_surface(state, ids).strip()
        return out, {"n_clauses": -6, **m}

    _ALL = {"ep_ngau_nhien": _arm_ep("ngau_nhien", "bo_ba"),
            "ep_chu_de": _arm_ep("chu_de", "bo_ba"),
            "ep_bo_ba": _arm_ep("ngau_nhien", "bo_ba"),
            "ep_giao": _arm_ep("ngau_nhien", "giao"),
            "ep_mot_khe_giao": _arm_ep("ngau_nhien", "giao"),
            "ep_da_khe_giao": _arm_da_khe("giao"),
            "ep_lop": _arm_ep("ngau_nhien", "lop"),
            "compose_2cha": arm_compose_2cha,
            "ban_le": arm_ban_le, "chep_cau": arm_chep, "ar": arm_ar,
            "chep_cau_E": arm_chep_e, "truy_hoi": arm_truy_hoi, "khuon_thay": arm_khuon}
    missing = [nm for nm in arm_names if nm not in _ALL]
    if missing:
        raise SystemExit(f"⛔ nhánh không biết: {missing} · có: {sorted(_ALL)}")
    if any(nm in ("ep_giao", "ep_lop", "ep_mot_khe_giao", "ep_da_khe_giao")
           for nm in arm_names) and not _lop_nb:
        raise SystemExit("⛔ cần lớp thay thế nhưng lop_nb rỗng — kiểm corpus/keep/bake")
    for nm in arm_names:
        if nm in ("ep_giao", "ep_mot_khe_giao", "ep_da_khe_giao", "ep_lop",
                  "ep_bo_ba", "ep_ngau_nhien"):
            lic = {"ep_giao": "giao", "ep_mot_khe_giao": "giao",
                   "ep_da_khe_giao": "giao", "ep_lop": "lop"}.get(nm, "bo_ba")
            logger.log(f"cấp phép nhánh {nm}: {lic}")
        if nm == "compose_2cha":
            logger.log(
                f"P202 compose_2cha: dual retrieve · W_skip join "
                f"d≤{args.join_d_max} min_count≥{args.join_min_count} · fallback P112"
            )
    arms = tuple((nm, _ALL[nm]) for nm in arm_names)
    rows, meta = [], []
    for (name, fn), prompts in zip(arms, slices):
        n_empty = 0
        for i, p in enumerate(prompts):
            if i and i % 50 == 0:
                logger.log(f"  {name} {i}/{len(prompts)} ({time.time()-t0:.0f}s)")
            txt, m = fn(p)
            if not txt:
                n_empty += 1
            rows.append((name, decode_surface(state, list(p.tokens)).strip(), txt or EMPTY))
            meta.append({"cfg": name, "empty": not txt, **m})
        logger.log(f"  {name}: trống {n_empty}/{len(prompts)} = {n_empty/len(prompts):.1%}")

    order = list(range(len(rows)))
    random.Random(20260804).shuffle(order)
    rows = [rows[i] for i in order]
    meta = [meta[i] for i in order]
    key = {}
    for j, (m, r) in enumerate(zip(meta, rows)):
        sid = f"A{j:03d}"
        m["id"] = sid
        key[sid] = m["cfg"]
        rows[j] = (sid, r[1], r[2])

    op = _ROOT / args.out
    op.parent.mkdir(parents=True, exist_ok=True)
    with open(op, "w", encoding="utf-8") as f:
        f.write("# CHẤM MÙ TUYỆT ĐỐI — mỗi mẫu chấm ĐỘC LẬP, không so với mẫu nào khác.\n")
        f.write(f"# n={len(rows)} mẫu · KHÔNG LỌC · thang: dung_duoc / tam / hong.\n\n")
        for sid, pr, txt in rows:
            f.write(f"[{sid}] ĐỀ: {pr}\n   → {txt}\n\n")
    (_ROOT / args.out_key).write_text(
        json.dumps(key, ensure_ascii=False, indent=1), encoding="utf-8")
    (_ROOT / args.out_meta).write_text(
        json.dumps({"checkpoint": args.checkpoint, "n_moi_nhanh": args.n,
                    "rows": meta}, ensure_ascii=False, indent=1), encoding="utf-8")
    logger.log(f"\n→ {args.out} ({len(rows)} mẫu) · khoá {args.out_key}")


if __name__ == "__main__":
    main()
