"""Tín hiệu nào tách LỚP CHUNG CHUNG khỏi LỚP THẬT? (§68)

§67.B: nhóm đúng tồn tại ở 5,1% ca nhưng luôn thua nhóm TO HƠN của `loại` /
`phần` / `người`. §64 đã quét ĐỊNH TỪ ra khỏi nhãn; danh từ chung chung thì vẫn
còn, và chúng thu gom mọi đồng thuận.

Đo sức phân biệt của bốn tín hiệu trên hai nhóm đối chứng, TRƯỚC khi vá:

  CHUNG : loại · phần · người · sự · nơi · điều · hình thức · tập hợp
  THẬT  : chim · thú · cây · cá · loài · kim loại · hành tinh · tỉnh

Tín hiệu (không tín hiệu nào là danh sách từ):
  1. `indeg`      — số chủ ngữ khác nhau (nền, §24 đã thử ở dạng LỌC và hỏng)
  2. `thuần`      — thành viên có chia nhau NHÃN KHÁC không (độ chụm của lớp)
  3. `có cha`     — bản thân nhãn có được xếp lớp không («chim là động vật»)
  4. `tv đồng hiện` — thành viên có hay đứng cạnh nhau trong câu không

    PYTHONHASHSEED=0 PYTHONPATH=src python scripts/do_lop_chung_vs_that.py
"""
from __future__ import annotations

import gc
import sys
import time
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
gc.disable()

from nn_aivie.core.checkpoint_io import load_checkpoint  # noqa: E402
from nn_aivie.hca import hub_spread as H  # noqa: E402
from nn_aivie.hca.config import HCAConfig  # noqa: E402

T0 = time.time()
CHUNG = ["loại", "phần", "người", "sự", "nơi", "điều", "hình thức", "tập hợp"]
THAT = ["chim", "thú", "cây", "cá", "loài", "kim loại", "hành tinh", "tỉnh"]


def main() -> int:
    ck = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/classrich2.pkl"
    cfg = HCAConfig()
    state, _ = load_checkpoint(ck)
    v = state.vocab
    wsp = H.load_ws_spans(ck.replace(".pkl", ".ws_spans.json"))
    pm = H.load_pos_map(cfg.hub_chain_pos_dict)
    idx = H.build_hub_spread_index(
        state, d_max=cfg.hub_chain_d_max, min_count=cfg.hub_chain_min_count,
        ws_spans=wsp, head_rule=cfg.hub_chain_head_rule,
        head_rule_trai=cfg.hub_chain_head_rule_trai,
        label_gidf_q=cfg.hub_chain_gidf_q, head_gidf_q=cfg.hub_chain_head_gidf_q,
        label_indeg_q=cfg.hub_chain_indeg_q,
        chu_ngu_ket_hop_q=cfg.hub_chain_chu_gia_q, pos_map=pm)
    print(f"\n### {ck} · {len(state.parents.parents)} cha\n")

    be = idx.via_pivot
    rev: dict[int, list[int]] = {}
    for s_, dd in be.items():
        for d in dd:
            rev.setdefault(int(d), []).append(int(s_))

    def ids_cua(w: str) -> list[int]:
        r = [int(x) for x in (H.resolve_surface_ids(v, w) or [])]
        for t, s_ in idx.label_surf.items():
            if s_ == w:
                r.append(int(t))
        return r

    def nhan_cua(t: int) -> set[int]:
        return set(be.get(int(t), {}))

    def do(w: str) -> tuple[int, float, int, float]:
        tids = ids_cua(w)
        tv = [m for t in tids for m in rev.get(t, ())]
        tv = list(dict.fromkeys(tv))[:40]
        indeg = max((int(idx.label_indeg.get(t, 0)) for t in tids), default=0)
        # THUẦN: cặp thành viên có chia nhau nhãn KHÁC không
        cap = list(combinations(tv[:20], 2))[:120]
        chung = sum(1 for a, b in cap
                    if (nhan_cua(a) & nhan_cua(b)) - set(tids))
        thuan = chung / max(1, len(cap))
        # CÓ CHA: bản thân nhãn có được xếp lớp không
        cha = sum(1 for t in tids if be.get(t))
        # ĐỒNG HIỆN giữa thành viên
        dh = sum(1 for a, b in cap
                 if b in idx.fwd.get(a, {}) or a in idx.fwd.get(b, {}))
        return indeg, thuan, cha, dh / max(1, len(cap))

    print(f"  {'nhãn':<12} {'loại':<6} {'indeg':>6} {'thuần':>7} {'cha':>4} {'tv đồng hiện':>13}")
    tong = {"CHUNG": [], "THẬT": []}
    for ten, ds in (("CHUNG", CHUNG), ("THẬT", THAT)):
        for w in ds:
            a, b, c, d = do(w)
            tong[ten].append((a, b, c, d))
            print(f"  {w:<12} {ten:<6} {a:>6} {b:>7.2f} {c:>4} {d:>13.2f}")
    print()
    for ten in ("CHUNG", "THẬT"):
        xs = tong[ten]
        n = max(1, len(xs))
        print(f"  TRUNG BÌNH {ten:<6}: indeg {sum(x[0] for x in xs)/n:>7.1f} · "
              f"thuần {sum(x[1] for x in xs)/n:.3f} · "
              f"cha {sum(x[2] for x in xs)/n:.2f} · "
              f"đồng hiện {sum(x[3] for x in xs)/n:.3f}")
    print(f"\n[{time.time() - T0:.0f}s] xong")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
