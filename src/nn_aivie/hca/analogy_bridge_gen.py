"""TK_046 P98G — điền khung ``X là ?`` qua cầu thuộc_tính (L=1..2).

Opt-in: ``generate(mode=analogy_la)`` / ``aivie generate --mode analogy_la``.
Không đổi mặc định W_skip. Donor khớp khi chung thuộc_tính Z (không dùng Lbl).
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from nn_aivie.core.state import GraphState
from nn_aivie.core.tokenizer import MARK_SPACE
from nn_aivie.core.frame_bracket_ref import cluster_adj_id, phrase_adj_id

PropKey = tuple[int, ...]

# Trái hợp lệ trước head trong triple «X là C» (không phải lop ngữ nghĩa).
_LA_LEFT_OK = frozenset({"con", "cái", "một", "các", "những"})

STOP = frozenset({
    "con", "cái", "là", "màu", "các", "những", "một", "của", "và",
    "cho", "với", "trong", "trên", "dưới", "vào", "từ", "đến", "về", "như",
    "khi", "mà", "để", "bị", "được", "đã", "đang", "sẽ", "cũng", "rất",
    "này", "đó", "kia", "nào", "gì", "ai", "đây", "Đây", "Đó", "đó",
    "người", "nơi", "việc", "theo", "sau", "trước", "giữa", "hay", "hoặc",
    "nhưng", "vì", "nếu", "do", "bởi", "qua", "lại", "ra", "lên", "xuống",
    "đi", "làm", "có", "không", "chỉ", "còn", "vẫn", "thì", "gọi", "tức",
    "tên", "di", "hành", "cách", "coi", "xem", "EOS", "<EOS>",
})
BAD_HEAD = frozenset({
    "gọi", "coi", "xem", "tức", "tên", "hay", "dù", "phải", "còn", "vẫn",
    "cũng", "dưới", "trên", "nhất", "nay", "từng", "nghĩa", "truyền", "chính",
    "sự", "Đây", "đây", "Đó", "đó", "Nó", "nó", "họ", "mình", "người",
    "việc", "chỉ", "vua", "thứ", "hiệu", "xưng", "nguyên", "trái", "Vì",
    "vì", "Do", "Nếu", "Khi", "Sau", "Tuy", "Dù", "Chính", "Bit", "ADN",
    "vừa", "thường", "dẫu", "thực", "hoặc", "như", "lần", "Gen", "Thuế",
    "Nghĩa", "Kết", "kết",
})
BAD_PRED = frozenset({
    "người", "nơi", "vị", "số", "hai", "lần", "cách", "di", "hành", "cũng",
    "rất", "không", "có", "được", "đã", "này", "đó", "một", "các", "những",
    "biểu", "loại", "vùng", "cố", "kinh",
    # mảnh «là cùng nhau / là cùng X» và pred chức từ
    "cùng", "nhau", "như", "với", "và", "của", "để", "khi", "mà", "bị",
    "rồi", "nữa", "thôi", "vậy", "thế", "nên", "liền", "đều", "càng",
    "hẳn", "phàm", "tuy", "sợ", "đúng",
    # mảnh chức danh «là Tổng thống / Tổng giám đốc» → pred Trần «Tổng»
    "Tổng", "Phó", "Chủ", "Ủy", "Giám", "Đốc", "Trưởng", "Vụ", "Sở",
    "Bí", "Thư", "Bộ", "Ban", "Cục",
})
# Head chức từ / discourse — không làm donor (kéo pred nhiễu).
BAD_DONOR = frozenset({
    "ngươi", "Ngươi", "đều", "Đều", "liền", "nên", "càng", "hẳn", "phàm",
    "tuy", "Hắn", "hắn", "ta", "Ta", "thân", "sợ", "đúng", "tính",
    # chủ ngữ giả discourse (học sinh ← đặc biệt / Mặc dù là Vương…)
    "đặc biệt", "Đặc biệt", "mặc dù", "Mặc dù", "thậm chí", "Thậm chí",
    "tuy nhiên", "Tuy nhiên", "do đó", "Do đó", "vì vậy", "Vì vậy",
    "bởi vậy", "Bởi vậy", "thế nên", "Thế nên", "tuy vậy", "Tuy vậy",
})
# Họ VN / đệm tên — không dùng làm cầu thuộc_tính (case Việt Nam—Nguyễn Văn—ông).
VN_SURNAME = frozenset({
    "Nguyễn", "Trần", "Lê", "Phạm", "Hoàng", "Huỳnh", "Phan", "Vũ", "Võ",
    "Đặng", "Bùi", "Đỗ", "Hồ", "Ngô", "Dương", "Lý", "Đào", "Đinh", "Lương",
    "Trịnh", "Đoàn", "Mai", "Trương", "Tô", "Lâm", "Hà", "Tăng", "Chu",
    "Cao", "Thái", "Tạ", "Quách", "Hứa", "Liêu", "La", "Bạch",
})
VN_SURNAME_LOWER = frozenset(s.lower() for s in VN_SURNAME)
NAME_MIDDLE = frozenset({
    "Văn", "Thị", "Đức", "Minh", "Công", "Hữu", "Xuân", "Quốc", "Anh",
    "văn", "thị",
})
# Token sau pred viết hoa → chức danh (Tổng thống); khác với tên (Vương gia / Hồ Liêm).
TITLE_CONTINUATION = frozenset({
    "thống", "giám", "đốc", "tịch", "trưởng", "binh", "hành", "ủy", "viên",
    "sư", "tướng", "lệnh", "quản", "lý", "đốc", "phó", "cục", "vụ", "sở",
    "ban", "ngành", "cục", "bộ",
})


@dataclass
class AnalogyBridgeConfig:
    """Defaults = P98G siết hub (v12) · giữ mini ĐẠT hình thức."""

    min_support: int = 2
    prop_gidf_min: float = 4.5
    compound_bonus: float = 2.5
    pred_top_ban: int = 40
    min_strength: float = 1.0
    # Cầu quá phổ biến (vd cẩn+thận ~238 trên v12) → bỏ.
    max_prop_df: int = 40
    # Triple (x,là,c) chỉ nhận khi c kết thúc cụm (token sau = punct/EOS).
    require_la_complete: bool = True
    # Ưu tiên cụm phrase/ISG/Entity hơn token đơn khi neo thuộc_tính.
    prefer_compound: bool = True
    # Độ dài tối đa span cụm khi quét ngược từ trước prop.
    max_head_span: int = 6
    # True: chỉ điền c không có (a,là,c) trong train (creative).
    # False: dùng mọi triple donor (try-prompt / sinh mở).
    creative_only: bool = False
    # Nhóm HƯ TỪ suy từ đồ thị thay bảng tay — xem `hu_tu_tu_do_thi`.
    # ⚠️ MẶC ĐỊNH TẮT: đo được là nó phá smoke ship trên mini (lý do trong
    # docstring `hu_tu_tu_do_thi`). Bật để A/B, đừng bật để ship khi chưa đo lại.
    hu_tu_tu_gidf: bool = False
    stop_gidf_max: float = 4.5


@dataclass
class AnalogyBridgeIndex:
    props: dict[int, set[PropKey]]
    edge_left_cls: dict[tuple[PropKey, int], set[str]]
    edge_sents: dict[tuple[PropKey, int], list[str]]
    la_ac: dict[int, set[int]]
    la_ids: set[int]
    # Biến thể `là` dùng khi phát ra (max df), KHÔNG phải min(la_ids).
    la_emit_id: int
    p_heads: dict[PropKey, set[int]]
    pred_hub: set[int]
    pred_df: np.ndarray
    prop_df: dict[PropKey, int]
    b_to_c_all: dict[int, set[int]]
    triples: list[tuple[int, int, int]]
    parents_a_la: dict[int, set[int]]
    stop: set[str]
    # id >= len(vocab): cụm ảo "học sinh" khi không map được Entity:
    compound_surf: dict[int, str] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)


def surface(vocab: object, tid: int, compound_surf: dict[int, str] | None = None) -> str:
    if compound_surf and int(tid) in compound_surf:
        return compound_surf[int(tid)]
    return vocab.token_of(int(tid)).replace(MARK_SPACE, "").strip()


def prop_label(
    vocab: object, key: PropKey, compound_surf: dict[int, str] | None = None
) -> str:
    return "+".join(surface(vocab, t, compound_surf) for t in key)


def _build_entity_surface_map(vocab: object, n: int) -> dict[str, int]:
    """surface không prefix → Entity id (vd 'học sinh' → Entity:học sinh)."""
    out: dict[str, int] = {}
    for i in range(n):
        raw = vocab.token_of(i)
        if not str(raw).startswith("Entity:"):
            continue
        lab = str(raw)[7:].replace(MARK_SPACE, "").strip()
        # bỏ rác EOS
        if not lab or lab.endswith("<EOS>") or "Ċ" in lab:
            continue
        out.setdefault(lab, i)
        out.setdefault(lab.lower(), i)
    return out


def _is_punct(vocab: object, tid: int) -> bool:
    s = surface(vocab, tid)
    return (not s) or s in {"<EOS>"} or all(not ch.isalnum() for ch in s)


def _content_ok(vocab: object, tid: int, *, stop: set[str]) -> bool:
    s = surface(vocab, tid)
    if not s or s in stop or s.lower() in stop:
        return False
    if s.isdigit() or len(s) <= 1:
        return False
    if all(not ch.isalnum() for ch in s):
        return False
    return True



def _is_surname_token(surf: str) -> bool:
    return surf in VN_SURNAME or surf.lower() in VN_SURNAME_LOWER


def _prop_is_name_like(vocab: object, key: PropKey) -> bool:
    """Cầu họ/tên (Nguyễn, Bùi, Nguyễn+Văn) — không phải thuộc_tính."""
    surfs = [surface(vocab, t) for t in key]
    if any(_is_surname_token(s) for s in surfs):
        return True
    if len(surfs) == 2 and surfs[1] in NAME_MIDDLE and surfs[0][:1].isupper():
        return True
    return False


def _prop_is_ho_bridge(vocab: object, key: PropKey) -> bool:
    """Cầu định danh họ (mini anh←họ→Hồ); khác hành vi kèm tên."""
    return any(surface(vocab, t) in ("họ", "Họ") for t in key)


def hu_tu_tu_do_thi(state: GraphState, *, gidf_max: float) -> frozenset[str]:
    """Hư từ suy TỪ ĐỒ THỊ: token df cao ⇒ gidf thấp (§0.4.2).

    Thay `STOP`/`BAD_HEAD`/`BAD_PRED` viết tay. Đo A/B/C trên
    `probe_analogy_exec_v5.py --scaffold {full,off,gidf}`, mini
    (`out/scaffold_ab_arm*.json`):

    | nhánh | ca | cầu hit@1 | cầu/embedding | FP |
    |---|---:|---:|---:|---:|
    | full (~300 từ tay) | 16 | 100,0% | 3,2× | 0% |
    | off (rỗng) | 31 | 87,1% | 4,5× | 0% |
    | **gidf (69 từ)** | **40** | **90,0%** | **5,1×** | **0%** |

    Bảng tay chủ yếu THU HẸP tập ca, không gánh kết quả. gidf cho 2,5× số ca
    mà cổng vẫn đạt và tỉ số so embedding tốt nhất.

    🔴 NHƯNG KHÔNG CHUYỂN ĐƯỢC SANG MODULE NÀY — vì thế mặc định
    `hu_tu_tu_gidf=False`. Đo trên `mini_sent_ep_qa.pkl`: bật lên thì smoke
    ship `anh/mẹ/em là → Hồ` **mất hẳn** (`khong_dien`).

    Nguyên nhân (đã truy tới gốc, không phải ngưỡng): **đáp án gold của nhiệm
    vụ này là danh từ riêng TẦN SUẤT CAO.** `Nguyễn` có gidf **4,09 < 4,5** ⇒
    tập gidf cấm nó làm predicate ⇒ `Nguyễn` biến khỏi `la_ac`, `cha` rụng khỏi
    head, cầu `họ` mất sạch head ⇒ không còn ứng viên. Nới `max_prop_df`
    (40→10⁹) **không** cứu được vì cầu chết từ khâu dựng.

    Bảng tay làm được một việc gidf không làm được: **giữ họ người phổ biến
    trong khi vẫn cấm hư từ phổ biến**. Tần suất một mình không tách nổi hai
    loại đó. Probe không lộ ra vì chọn ca của nó đòi head VIẾT HOA, nên tập ca
    arm C vốn đã loại các ca gãy này.

    ⚠️ Chỉ thay được nhóm HƯ TỪ. `VN_SURNAME` / `NAME_MIDDLE` /
    `TITLE_CONTINUATION` / `BAD_DONOR` là TRI THỨC THẾ GIỚI — gidf không suy
    ra được; còn là nợ Luật 0, chưa đo.
    """
    ps = state.parents
    assert ps is not None
    out = {
        surface(state.vocab, t)
        for t in range(len(state.vocab))
        if ps._gidf(t) < gidf_max  # noqa: SLF001
    }
    out.discard("")
    return frozenset(out)


def _la_pred_allowed(
    vocab: object,
    e: list[int],
    i: int,
    c: int,
    cs: str,
    *,
    require_complete: bool,
    bad_pred: frozenset[str] | set[str],
) -> bool:
    """Siết pred sau 'là': chặn «Tổng thống»; giữ Hồ Liêm / Vương gia."""
    if cs in bad_pred:
        return False
    if not require_complete:
        return True
    if i + 3 >= len(e) or _is_punct(vocab, e[i + 3]):
        return True
    nxt = surface(vocab, e[i + 3])
    # chức danh cắt cụt
    if nxt.lower() in TITLE_CONTINUATION:
        return False
    # tên riêng (Hồ Liêm cả hoa; Vương gia — giữ token đầu viết hoa)
    if cs[:1].isupper():
        return True
    return False


def build_analogy_bridge_index(
    state: GraphState,
    cfg: AnalogyBridgeConfig | None = None,
) -> AnalogyBridgeIndex:
    cfg = cfg or AnalogyBridgeConfig()
    v, ps = state.vocab, state.parents
    if ps is None or len(ps) == 0:
        raise RuntimeError("analogy_bridge_gen cần ParentStore")
    n = len(v)
    if not hasattr(ps, "_gdf"):
        # `strength()` chấm điểm bằng gidf; thiếu `_gdf` thì mọi cầu có cùng
        # trọng số 1.0 và cơ chế im lặng biến thành đếm donor thuần.
        raise RuntimeError(
            "analogy_bridge_gen: ParentStore thiếu `_gdf` — gidf không tính được"
        )
    if cfg.hu_tu_tu_gidf:
        _hu_tu = hu_tu_tu_do_thi(state, gidf_max=cfg.stop_gidf_max)
        stop = set(_hu_tu) | {s.lower() for s in _hu_tu}
        bad_head: frozenset[str] | set[str] = _hu_tu
        bad_pred: frozenset[str] | set[str] = _hu_tu
    else:
        stop = set(STOP) | {s.lower() for s in STOP}
        bad_head, bad_pred = BAD_HEAD, BAD_PRED
    la_ids = {i for i in range(n) if surface(v, i) == "là"}
    if not la_ids:
        raise RuntimeError("analogy_bridge_gen: không tìm thấy token 'là'")
    # Biến thể `là` hay xuất hiện nhất (có/không MARK_SPACE) — id nhỏ nhất là
    # thứ tự vocab, không phải thứ mạng dùng. Tie-break theo id cho tất định.
    la_emit_id = max(la_ids, key=lambda t: (int(ps._gdf.get(t, 0)), -int(t)))  # noqa: SLF001

    ent_map = _build_entity_surface_map(v, n)
    compound_surf: dict[int, str] = {}
    compound_key_to_id: dict[tuple[int, ...], int] = {}
    n_compound_edges = 0
    n_head_cluster = 0
    n_head_phrase = 0
    n_head_entity = 0

    # Tra cứu cụm toàn cục: ISG span + phrase token-tuple
    cluster_by_span: dict[tuple[int, ...], int] = {}
    seg = getattr(ps, "seg", None)
    if seg is not None and getattr(seg, "clusters", None):
        for ci, c in enumerate(seg.clusters):
            try:
                sp = tuple(int(t) for t in c.token_span())
            except Exception:
                continue
            if len(sp) >= 2:
                cluster_by_span[sp] = ci
        for (a, b), ci in getattr(seg, "cluster_of_pair", {}).items():
            cluster_by_span.setdefault((int(a), int(b)), int(ci))
    phrase_by_span: dict[tuple[int, ...], int] = {}
    if ps.phrases:
        for j, ph in enumerate(ps.phrases):
            if len(ph) >= 2:
                phrase_by_span[tuple(int(t) for t in ph)] = j

    def _span_label(toks: tuple[int, ...]) -> str:
        return " ".join(surface(v, t) for t in toks)

    def head_id_for_span(toks: tuple[int, ...], *, kind: str) -> int:
        """kind: cluster|phrase|entity — không tạo virtual id (tránh phình DF)."""
        nonlocal n_head_cluster, n_head_phrase, n_head_entity
        if len(toks) == 1:
            return int(toks[0])
        if toks in compound_key_to_id:
            return compound_key_to_id[toks]
        lab = _span_label(toks)
        if kind == "cluster" and toks in cluster_by_span:
            hid = int(cluster_adj_id(cluster_by_span[toks]))
            compound_key_to_id[toks] = hid
            compound_surf[hid] = lab
            n_head_cluster += 1
            return hid
        if kind == "phrase" and toks in phrase_by_span:
            hid = int(phrase_adj_id(phrase_by_span[toks]))
            compound_key_to_id[toks] = hid
            compound_surf[hid] = lab
            n_head_phrase += 1
            return hid
        eid = ent_map.get(lab) or ent_map.get(lab.lower())
        if eid is not None:
            compound_key_to_id[toks] = eid
            compound_surf[eid] = lab  # surface hiển thị không prefix Entity:
            n_head_entity += 1
            return eid
        raise KeyError(f"head_id_for_span: no compound for {lab!r} kind={kind}")

    def resolve_head_ending_at(e: list[int], end: int) -> tuple[int, tuple[int, ...], str]:
        """Ưu tiên span dài nhất: ISG cluster > phrase > Entity > token."""
        if not cfg.prefer_compound:
            return int(e[end]), (e[end],), "token"
        max_L = min(int(cfg.max_head_span), end + 1)
        # dài → ngắn; cùng dài: cluster > phrase > entity
        for L in range(max_L, 1, -1):
            s = end - L + 1
            toks = tuple(e[s : end + 1])
            if any(
                t in la_ids or _is_punct(v, t) or not _content_ok(v, t, stop=stop)
                for t in toks
            ):
                continue
            if toks in cluster_by_span:
                return head_id_for_span(toks, kind="cluster"), toks, "cluster"
            if toks in phrase_by_span:
                return head_id_for_span(toks, kind="phrase"), toks, "phrase"
            lab = _span_label(toks)
            if lab in ent_map or lab.lower() in ent_map:
                return head_id_for_span(toks, kind="entity"), toks, "entity"
        # L=2 Entity dù không có phrase/ISG (vd học sinh bị overlap chặn)
        if end >= 1:
            toks = (e[end - 1], e[end])
            if (
                all(_content_ok(v, t, stop=stop) for t in toks)
                and toks[0] not in la_ids
                and toks[1] not in la_ids
            ):
                lab = _span_label(toks)
                if lab in ent_map or lab.lower() in ent_map:
                    return head_id_for_span(toks, kind="entity"), toks, "entity"
        return int(e[end]), (e[end],), "token"

    def span_covers_as_cum(e: list[int], start: int, end: int) -> bool:
        """True nếu [start..end] là cụm phrase/ISG/Entity."""
        if end - start < 1:
            return False
        toks = tuple(e[start : end + 1])
        if toks in cluster_by_span or toks in phrase_by_span:
            return True
        lab = _span_label(toks)
        return lab in ent_map or lab.lower() in ent_map

    props: dict[int, set[PropKey]] = defaultdict(set)
    # Chỉ head chính (cụm/Entity/token primary) — dual token không phình prop_df.
    props_for_df: dict[int, set[PropKey]] = defaultdict(set)
    edge_left_cls: dict[tuple[PropKey, int], set[str]] = defaultdict(set)
    edge_sents: dict[tuple[PropKey, int], list[str]] = defaultdict(list)
    la_ac: dict[int, set[int]] = defaultdict(set)
    parents_a_la: dict[int, set[int]] = defaultdict(set)
    triples: list[tuple[int, int, int]] = []

    def add_edge(
        x: int, key: PropKey, pi: int, text: str, *, primary: bool,
    ) -> None:
        props[x].add(key)
        if primary and x < n:
            props_for_df[x].add(key)
        if len(edge_sents[(key, x)]) < 2:
            edge_sents[(key, x)].append(
                f"[{pi}] L={len(key)} "
                f"cầu={prop_label(v, key, compound_surf)!r} "
                f"head={surface(v, x, compound_surf)!r} | {text}"
            )

    for pi, rec in enumerate(ps.parents):
        e = [int(t) for t in rec.emit_ids]
        text = " ".join(surface(v, t) for t in e)

        for j in range(1, len(e)):
            p1 = e[j]
            if _is_punct(v, p1) or p1 in la_ids or not _content_ok(v, p1, stop=stop):
                continue
            end = j - 1
            if _is_punct(v, e[end]) or e[end] in la_ids:
                continue
            if not _content_ok(v, e[end], stop=stop):
                continue
            # p1 đang nằm trong cụm dài hơn (… end p1 …) → không phải thuộc_tính
            if cfg.prefer_compound and span_covers_as_cum(e, end, j):
                continue

            hid, head_toks, kind = resolve_head_ending_at(e, end)
            if kind != "token":
                n_compound_edges += 1
            # Dual-write: cụm + token cuối — giữ mini «trai/ông là Hồ» khi resolve token.
            heads_prop = [hid]
            if kind != "token" and e[end] != hid:
                heads_prop.append(int(e[end]))

            # Không neo cầu họ/tên (ông Nguyễn… / Việt Nam Nguyễn Văn…).
            p1_surf = surface(v, p1)
            skip_l1 = _is_surname_token(p1_surf)

            for h in heads_prop:
                primary = h == hid
                if not skip_l1 and not _prop_is_name_like(v, (p1,)):
                    add_edge(h, (p1,), pi, text, primary=primary)
                if j + 1 < len(e):
                    p2 = e[j + 1]
                    if (
                        not _is_punct(v, p2)
                        and p2 not in la_ids
                        and _content_ok(v, p2, stop=stop)
                    ):
                        pair_lab = f"{surface(v, p1)} {surface(v, p2)}"
                        if pair_lab in ent_map or pair_lab.lower() in ent_map:
                            continue
                        key2 = (p1, p2)
                        if _prop_is_name_like(v, key2):
                            continue
                        add_edge(h, key2, pi, text, primary=primary)

        for i in range(len(e) - 2):
            if e[i + 1] not in la_ids:
                continue
            c = e[i + 2]
            # 'là': ưu tiên cụm trước là, đồng thời giữ token cuối (mini trai là Hồ).
            hid, head_toks, kind = resolve_head_ending_at(e, i)
            heads_la = [hid]
            if hid != e[i]:
                heads_la.append(e[i])
            for b in heads_la:
                parents_a_la[b].add(pi)
                bs = surface(v, b, compound_surf)
                cs = surface(v, c)
                if bs in bad_head or cs in bad_pred or bs in bad_pred:
                    continue
                if bs in BAD_DONOR:
                    continue
                if b < n and b not in compound_surf and not _content_ok(v, b, stop=stop):
                    continue
                if b >= n and b not in compound_surf:
                    continue
                if not _content_ok(v, c, stop=stop):
                    continue
                # token dual: dùng i; cụm: token đầu span
                check_i = i if b == e[i] else (i - len(head_toks) + 1)
                check_i = max(check_i, 0)
                ok_left = (
                    check_i == 0
                    or _is_punct(v, e[check_i - 1])
                    or surface(v, e[check_i - 1]) in _LA_LEFT_OK
                )
                if not ok_left:
                    continue
                if not _la_pred_allowed(
                    v, e, i, c, cs, require_complete=cfg.require_la_complete,
                    bad_pred=bad_pred,
                ):
                    continue
                la_ac[b].add(c)
                triples.append((pi, b, c))

    tok_ok = np.zeros(n, dtype=bool)
    for t in range(n):
        if not _content_ok(v, t, stop=stop) or t in la_ids:
            continue
        try:
            if float(ps._gidf(t)) >= cfg.prop_gidf_min:  # noqa: SLF001
                tok_ok[t] = True
        except Exception:
            continue

    def key_ok(key: PropKey) -> bool:
        return all(0 <= t < n and tok_ok[t] for t in key)

    # DF chỉ đếm head primary vocab/Entity (< n)
    prop_df_count: dict[PropKey, int] = defaultdict(int)
    for a, keys in props_for_df.items():
        for k in keys:
            if key_ok(k):
                prop_df_count[k] += 1

    for a in list(props):
        props[a] = {
            k for k in props[a]
            if key_ok(k)
            and prop_df_count[k] <= cfg.max_prop_df
            and not _prop_is_name_like(v, k)
        }
        if not props[a]:
            del props[a]

    pred_df = np.zeros(n, dtype=np.int32)
    for _h, cs in la_ac.items():
        for c in cs:
            if c < n:
                pred_df[c] += 1
    top_ban: set[int] = set()
    for i in np.argsort(-pred_df)[: max(cfg.pred_top_ban * 3, 0)]:
        if pred_df[i] <= 0:
            continue
        if surface(v, int(i))[:1].isupper():
            continue
        top_ban.add(int(i))
        if len(top_ban) >= cfg.pred_top_ban:
            break
    pred_hub = set(top_ban)
    for c in range(n):
        if surface(v, c) in bad_pred or not _content_ok(v, c, stop=stop):
            pred_hub.add(c)
    for h in list(la_ac):
        hs = surface(v, h, compound_surf)
        if hs in BAD_DONOR:
            del la_ac[h]
            continue
        la_ac[h] = {c for c in la_ac[h] if c not in pred_hub}
        if not la_ac[h]:
            del la_ac[h]
    triples = [(pi, b, c) for pi, b, c in triples if c not in pred_hub and b in la_ac]

    p_heads: dict[PropKey, set[int]] = defaultdict(set)
    for head in la_ac:
        for key in props.get(head, ()):
            p_heads[key].add(head)

    b_to_c_all: dict[int, set[int]] = defaultdict(set)
    for _pi, b, c in triples:
        b_to_c_all[b].add(c)

    all_heads: set[int] = set(props) | set(la_ac)
    for hs in p_heads.values():
        all_heads |= hs

    n_l1 = sum(1 for s in props.values() for k in s if len(k) == 1)
    n_l2 = sum(1 for s in props.values() for k in s if len(k) == 2)
    n_prop_drop = sum(1 for _k, d in prop_df_count.items() if d > cfg.max_prop_df)
    return AnalogyBridgeIndex(
        props=dict(props),
        edge_left_cls=dict(edge_left_cls),
        edge_sents=dict(edge_sents),
        la_ac=dict(la_ac),
        la_ids=la_ids,
        la_emit_id=int(la_emit_id),
        p_heads=dict(p_heads),
        pred_hub=pred_hub,
        pred_df=pred_df,
        prop_df=dict(prop_df_count),
        b_to_c_all=dict(b_to_c_all),
        triples=triples,
        parents_a_la=dict(parents_a_la),
        stop=stop,
        compound_surf=compound_surf,
        meta={
            "n_l1": n_l1,
            "n_l2": n_l2,
            "n_triples": len(triples),
            "n_prop_keys_dropped_df": n_prop_drop,
            "max_prop_df": cfg.max_prop_df,
            "n_compound_edge_attach": n_compound_edges,
            "n_head_cluster": n_head_cluster,
            "n_head_phrase": n_head_phrase,
            "n_head_entity": n_head_entity,
            "n_compound_surf": len(compound_surf),
            "n_entity_map": len(ent_map),
            "n_cluster_spans": len(cluster_by_span),
            "n_phrase_spans": len(phrase_by_span),
            "hu_tu_tu_gidf": cfg.hu_tu_tu_gidf,
            "n_hu_tu": len(stop),
            "prefer_compound": cfg.prefer_compound,
            "max_head_span": cfg.max_head_span,
            "n_heads": len(all_heads),
        },
    )


def resolve_head(
    vocab: object,
    word: str,
    props: dict[int, set[PropKey]],
    n: int,
) -> int | None:
    """Cùng surface → chọn id có nhiều props nhất (vd anh 2149 ≠ 1550)."""
    cands = [i for i in range(n) if surface(vocab, i) == word and props.get(i)]
    if cands:
        return int(max(cands, key=lambda i: len(props[i])))
    cands = [i for i in range(n) if surface(vocab, i) == word]
    return int(cands[0]) if cands else None


def resolve_head_phrase(
    vocab: object,
    phrase: str,
    props: dict[int, set[PropKey]],
    n: int,
    *,
    compound_surf: dict[int, str] | None = None,
) -> tuple[int | None, str]:
    """Ưu tiên cụm/Entity đúng phrase; rồi token cuối (vd anh / mẹ)."""
    phrase = (phrase or "").strip()
    if not phrase:
        return None, phrase

    # 1) Exact match trên compound_surf (ISG / phrase / Entity đã neo)
    if compound_surf:
        hits = [
            hid for hid, lab in compound_surf.items()
            if lab == phrase and props.get(hid)
        ]
        if not hits:
            pl = phrase.lower()
            hits = [
                hid for hid, lab in compound_surf.items()
                if lab.lower() == pl and props.get(hid)
            ]
        if hits:
            return int(max(hits, key=lambda i: len(props.get(i, ())))), phrase

    # 2) Entity: đúng surface (kể cả chưa gắn compound_surf)
    ent = f"Entity:{phrase}"
    for i in range(n):
        if vocab.token_of(i) != ent:
            continue
        if props.get(i):
            return i, phrase

    parts = phrase.split()
    head_w = phrase
    a: int | None = None
    for cand in ([phrase] if phrase else []) + list(reversed(parts)):
        if not cand:
            continue
        a = resolve_head(vocab, cand, props, n)
        if a is not None and props.get(a):
            return a, cand
        if a is not None and head_w == phrase:
            head_w = cand
    if a is None and parts:
        head_w = parts[-1]
        a = resolve_head(vocab, head_w, props, n)
    return a, head_w


def _head_part_surfaces(
    vocab: object,
    head_id: int,
    compound_surf: dict[int, str] | None,
) -> set[str]:
    """Token thành phần của head cụm (vd học sinh → {học, sinh})."""
    lab = surface(vocab, head_id, compound_surf)
    parts = [p for p in lab.split() if p]
    if len(parts) >= 2:
        return set(parts)
    return set()


def _dedupe_compound_donors(
    vocab: object,
    dons: set[int],
    compound_surf: dict[int, str] | None,
) -> set[int]:
    """Bỏ donor là token thành phần của donor cụm khác (nữ ⊂ Phụ nữ)."""
    if len(dons) < 2:
        return dons
    surfs = {d: surface(vocab, d, compound_surf) for d in dons}
    drop: set[int] = set()
    for d, s in surfs.items():
        if " " in s or not s:
            continue
        for o, osurf in surfs.items():
            if o == d:
                continue
            parts = osurf.split()
            if len(parts) >= 2 and s in parts:
                drop.add(d)
                break
    return dons - drop if drop else dons


def aggregate_scores(
    state: GraphState,
    index: AnalogyBridgeIndex,
    a: int,
    cfg: AnalogyBridgeConfig,
    *,
    b2c: dict[int, set[int]] | None = None,
) -> tuple[
    dict[int, float],
    dict[int, set[int]],
    dict[int, list[tuple[PropKey, list[int], float]]],
]:
    """score / donors / bridges cho mọi predicate c từ seed a."""
    ps = state.parents
    assert ps is not None
    acc_score: dict[int, float] = defaultdict(float)
    acc_dons: dict[int, set[int]] = defaultdict(set)
    acc_bridges: dict[int, list[tuple[PropKey, list[int], float]]] = defaultdict(list)
    # Donor = đuôi/token của cụm head (sinh ⊂ học sinh) → trùng dual-write, bỏ.
    head_parts = _head_part_surfaces(state.vocab, a, index.compound_surf)

    def strength(n_donors: int, key: PropKey, c: int) -> float:
        g = 1.0
        for t in key:
            g *= float(ps._gidf(t))  # noqa: SLF001
        g = float(g ** (1.0 / max(len(key), 1)))
        bonus = cfg.compound_bonus if len(key) == 2 else 1.0
        pdf = float(index.prop_df.get(key, 1))
        return float(n_donors) * g * bonus / (
            (1.0 + np.log1p(index.pred_df[c])) * (1.0 + np.log1p(pdf))
        )

    src_map = b2c if b2c is not None else (
        {h: set(cs) for h, cs in index.la_ac.items()}
        if cfg.creative_only
        else index.b_to_c_all
    )

    for key in index.props.get(a, ()):
        # cầu hub còn sót (df đo trước cắt) — bỏ
        if index.prop_df.get(key, 0) > cfg.max_prop_df:
            continue
        cd: dict[int, set[int]] = defaultdict(set)
        for b in index.p_heads.get(key, ()):
            if b == a:
                continue
            bs = surface(state.vocab, b, index.compound_surf)
            if bs in BAD_DONOR:
                continue
            if head_parts and bs in head_parts:
                continue
            for c in src_map.get(b, ()):
                if c == a or c in index.pred_hub:
                    continue
                cd[c].add(b)
        for c, dons in cd.items():
            # Donors trùng đuôi cụm nhau (nữ ⊂ Phụ nữ) → support ảo, giữ donor dài hơn.
            dons = _dedupe_compound_donors(state.vocab, dons, index.compound_surf)
            if len(dons) < cfg.min_support:
                continue
            # Pred là họ VN chỉ qua cầu «họ» (tránh mẹ+cưng → Nguyễn; giữ mini Hồ).
            if _is_surname_token(surface(state.vocab, c)) and not _prop_is_ho_bridge(
                state.vocab, key
            ):
                continue
            sc = strength(len(dons), key, c)
            acc_score[c] += sc
            acc_dons[c] |= dons
            acc_bridges[c].append((key, sorted(dons), sc))
    return acc_score, acc_dons, acc_bridges


def fill_predicate(
    state: GraphState,
    index: AnalogyBridgeIndex,
    a: int,
    cfg: AnalogyBridgeConfig | None = None,
    *,
    b2c: dict[int, set[int]] | None = None,
) -> tuple[int | None, dict[str, Any]]:
    cfg = cfg or AnalogyBridgeConfig()
    acc_score, acc_dons, acc_bridges = aggregate_scores(
        state, index, a, cfg, b2c=b2c)
    if not acc_score:
        return None, {"stop": "khong_dien", "head": a}
    ranked = [
        (c, sc) for c, sc in acc_score.items() if sc >= cfg.min_strength
    ]
    if not ranked:
        return None, {"stop": "duoi_min_strength", "head": a}
    best_c, best_sc = max(ranked, key=lambda kv: kv[1])
    bridges = sorted(acc_bridges[best_c], key=lambda t: -t[2])
    v = state.vocab
    cs = index.compound_surf
    return best_c, {
        "stop": "analogy_la",
        "head": a,
        "head_surf": surface(v, a, cs),
        "pred": best_c,
        "pred_surf": surface(v, best_c, cs),
        "score": float(best_sc),
        "n_bridges": len(bridges),
        "bridges": [prop_label(v, k, cs) for k, _, _ in bridges],
        "donors": [surface(v, d, cs) for d in sorted(acc_dons[best_c])[:8]],
    }


def parse_la_prompt(prompt: str) -> tuple[str, bool]:
    """Trả (head_phrase, ends_with_la)."""
    raw = prompt.strip()
    # bỏ dấu câu cuối
    while raw and raw[-1] in ".?!…":
        raw = raw[:-1].rstrip()
    ends = False
    head = raw
    if raw.endswith(" là"):
        head = raw[: -len(" là")].strip()
        ends = True
    elif raw.endswith("là"):
        head = raw[: -len("là")].strip()
        ends = True
    return head, ends


def generate_analogy_bridge(
    state: GraphState,
    prompt: str,
    *,
    index: AnalogyBridgeIndex | None = None,
    cfg: AnalogyBridgeConfig | None = None,
) -> tuple[list[int], dict[str, Any]]:
    """Continuation sau prompt: predicate (và ``là`` nếu prompt chưa có)."""
    cfg = cfg or AnalogyBridgeConfig()
    if index is None:
        index = build_analogy_bridge_index(state, cfg)
    v = state.vocab
    n = len(v)
    head_phrase, ends_la = parse_la_prompt(prompt)
    if not head_phrase:
        return [], {"stop": "prompt_rong"}
    a, head_w = resolve_head_phrase(
        v, head_phrase, index.props, n, compound_surf=index.compound_surf,
    )
    if a is None:
        return [], {"stop": "khong_resolve_head", "head_phrase": head_phrase}
    pred, meta = fill_predicate(state, index, a, cfg)
    meta["head_phrase"] = head_phrase
    meta["head_resolved"] = head_w
    meta["ends_with_la"] = ends_la
    if pred is None:
        return [], meta
    out: list[int] = []
    if not ends_la:
        out.append(int(index.la_emit_id))
    out.append(int(pred))
    return out, meta
