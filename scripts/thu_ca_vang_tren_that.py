"""Ca «con khỉ» TRÊN DỮ LIỆU THẬT, dùng LỚP THẬT của chân lý v2.

Mọi phép đo trước hoặc chạy trên toy (5 câu vàng + nhiễu) hoặc trên bộ ca đào
máy móc (§60.8: 46% là mảnh BPE, lớp là `khoảng`/`sự`/`hai`). Chưa ai hỏi câu
đơn giản nhất: **lấy một lớp THẬT của `classrich` (`loài` → Chim · Cá sấu ·
Gà · Rùa · cá chép · gấu trúc), một thuộc tính THẬT chạm ≥2 thành viên — chuỗi
có ra thành viên CHƯA chạm thuộc tính đó không?**

Đó đúng là cấu trúc ca vàng, chỉ khác: dữ liệu do đời thật viết.

    PYTHONHASHSEED=0 PYTHONPATH=src python scripts/thu_ca_vang_tren_that.py [lớp]
"""
from __future__ import annotations

import gc
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
gc.disable()

from nn_aivie.core.checkpoint_io import load_checkpoint  # noqa: E402
from nn_aivie.core.tokenizer import MARK_SPACE  # noqa: E402
from nn_aivie.hca import hub_spread as H  # noqa: E402
from nn_aivie.hca.config import HCAConfig  # noqa: E402

T0 = time.time()


def main() -> int:
    ck = "checkpoints/classrich2.pkl" if "-2" in sys.argv else "checkpoints/classrich.pkl"
    chi_lop = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] != "-" else ""
    bien = (len(sys.argv) <= 2) or sys.argv[2] != "0"      # A/B §62
    hq = float(sys.argv[3]) if len(sys.argv) > 3 else None  # §64 head_gidf_q
    cfg = HCAConfig()
    state, _ = load_checkpoint(ck)
    v, ps = state.vocab, state.parents
    S = [str(v.token_of(i)).replace(MARK_SPACE, "").strip() for i in range(len(v))]
    lop = json.load(open("out/chan_ly_v2_classrich2.json" if "-2" in sys.argv else "out/chan_ly_v2_classrich.json"))["lop"]
    pm = H.load_pos_map(cfg.hub_chain_pos_dict)
    tu = {k.replace(MARK_SPACE, "").strip() for k in pm}
    # §67 (A): BỘ CA phải đúng HÌNH DẠNG ca vàng — thuộc tính là TÍNH TỪ
    # (`nghịch ngợm`), không phải `rồi`/`quả`/`thân` do đồng-hiện nhặt ra.
    tu_loai = {k.replace(MARK_SPACE, "").strip(): x for k, x in pm.items()}
    chi_tinh_tu = "tinhtu" in sys.argv
    wsp = H.load_ws_spans(ck.replace(".pkl", ".ws_spans.json"))
    idx = H.build_hub_spread_index(
        state, d_max=cfg.hub_chain_d_max, min_count=cfg.hub_chain_min_count,
        head_rule=cfg.hub_chain_head_rule,
        head_rule_trai=cfg.hub_chain_head_rule_trai,
        label_gidf_q=cfg.hub_chain_gidf_q, label_indeg_q=cfg.hub_chain_indeg_q,
        chu_ngu_ket_hop_q=cfg.hub_chain_chu_gia_q, pos_map=pm, ws_spans=wsp,
        bien_tu_pivot=bien, head_gidf_q=hq,
        kenh_vi_ngu=("vingu" in sys.argv),
        lop_trang=(set(json.load(open("out/lop_tu_dien.json")))
                   if "lopdt" in sys.argv else None))
    print(f"  cờ: head_gidf_q={hq}")
    print(f"\n### {ck} · {len(ps.parents)} cha\n")

    # ⚠️ ĐƠN VỊ: thành viên của chân lý v2 là TỪ (span WS), nên thuộc tính cũng
    # phải là TỪ. Bản đầu lấy token ⇒ ca ra `giả`/`danh`/`khai` = mảnh của «học
    # giả»/«danh tiếng»/«công khai», và chuỗi bị chấm trên thứ không tồn tại.
    span = {tuple(int(x) for x in sp) for sp in (wsp or ())}
    dai = max((len(x) for x in span), default=1)
    tok_par: dict[str, set[int]] = defaultdict(set)
    par_tok: list[set[str]] = []
    for pi, rec in enumerate(ps.parents):
        seq = [int(t) for t in rec.emit_ids]
        ts: set[str] = set()
        i = 0
        while i < len(seq):
            hit = None
            for L in range(min(dai, len(seq) - i), 1, -1):
                if tuple(seq[i:i + L]) in span:
                    hit = L
                    break
            if hit:
                ts.add(" ".join(S[t] for t in seq[i:i + hit]))
                i += hit
            else:
                if S[seq[i]] and not S[seq[i]].startswith("<"):
                    ts.add(S[seq[i]])
                i += 1
        par_tok.append(ts)
        for s_ in ts:
            tok_par[s_].add(pi)
    print(f"  đơn vị TỪ: {len(span)} span · {len(tok_par)} bề mặt\n")

    tap = {chi_lop: lop[chi_lop]} if chi_lop else lop
    ket = {"đạt": 0, "②_sai": 0, "③_thiếu": 0, "không_có_ca": 0}
    # ĐO BƯỚC ① — chưa mục nào của TK_046 đo nó. Cơ chế phiếu ở ② cần ≥2 hạt
    # CÙNG LỚP mới phân biệt được; nếu ① hiếm khi cho 2 đồng-thành-viên thì ②
    # không thể đúng dù vá bao nhiêu, và mọi vòng vá ② là vá sai chỗ.
    dem_hat: dict[int, int] = {0: 0, 1: 0, 2: 0, 3: 0}
    hang60: list[list[int]] = []
    chia_nhan: list[int] = []
    thieu_canh: list = []
    for c, mem in sorted(tap.items(), key=lambda kv: -len(kv[1])):
        mem = [m for m in mem if H.resolve_surface_ids(v, m)]
        if len(mem) < 3:
            continue
        # thuộc tính chạm ≥2 thành viên, chưa chạm ≥1 thành viên khác
        cham: dict[str, set[str]] = defaultdict(set)
        for m in mem:
            for pi in list(tok_par.get(m, ()))[:200]:
                for p in par_tok[pi]:
                    cham[p].add(m)
        # THUỘC TÍNH phải là TỪ THẬT và KHÔNG phổ biến — `nhất`/`thường` chạm
        # mọi thành viên nhưng không phân biệt gì (đúng bệnh `phổ biến` §54).
        ung = [(p, ms) for p, ms in cham.items()
               if len(ms) >= 2 and len(ms) < len(mem)
               and p not in mem and p != c and len(p) > 1
               and p.replace(" ", "").isalpha() and p[:1].islower()
               and (" " in p or p in tu) and len(tok_par.get(p, ())) <= 300
               and (not chi_tinh_tu or "tính từ" in tu_loai.get(p, ""))
               and len(H.resolve_surface_ids(v, p) or []) > 0]
        if not ung:
            ket["không_có_ca"] += 1
            continue
        ung.sort(key=lambda kv: -len(kv[1]))
        for p, ms in ung[:3]:                # 3 thuộc tính MẠNH NHẤT mỗi lớp
         if True:
          khi = sorted(set(mem) - ms)          # «con khỉ» = chưa chạm thuộc tính
          pid = H.resolve_surface_ids(v, p)
          ctx = list(v.encode_prompt(p))
          s1 = H.score_hubs_from(state, idx, pid[:4], topk=60, gidf_min=0.0,
                                 channel="typed_first", kenh_thuoc_tinh=True)
          # ① CÓ CHỨA đồng-thành-viên ở đâu đó trong 60 không, hay KHÔNG HỀ có?
          # Hai chẩn đoán khác nhau: «không TÌM được» vs «không BIỂU DIỄN được»
          # (bài học nặng số 2 của ticket).
          _ms = set(mem)
          _dtv = [h for h in s1 if h.surface in _ms]
          hang60.append([i for i, h in enumerate(s1, 1) if h.surface in _ms][:2])
          # Khi bể CÓ ≥2 đồng-thành-viên: chúng có CHIA NHAU một nhãn không?
          # Nếu không thì luật chọn hạt nào cũng vô ích — nghẽn là CẠNH, không
          # phải xếp hạng (B chết, C là lối duy nhất).
          if len(_dtv) >= 2:
              _nh2 = []
              for _h in _dtv[:4]:
                  _t = set()
                  for _i in (H.resolve_surface_ids(v, _h.surface) or [_h.tid])[:3]:
                      _t |= set(idx.via_pivot.get(int(_i), {}))
                  _nh2.append(_t)
              _chung = set.intersection(*_nh2[:2]) if len(_nh2) >= 2 else set()
              chia_nhan.append(1 if _chung else 0)
              if not _chung:
                  thieu_canh.append((p, [h.surface for h in _dtv[:2]],
                                     [len(x) for x in _nh2[:2]]))
          cand = [h for h in s1 if h.surface != p]
          ty = [h for h in cand if h.typed]
          # A/B §66: "cooc" = BỎ luật hạt-chỉ-typed của §59. §59 được chốt trên
          # TOY, nơi câu «nghịch ngợm như loài chó» tạo cạnh typed sẵn. Trên
          # dữ liệu thật kênh `như` thưa, nên luật đó có thể đang bóp chết ①.
          hat = (cand if "cooc" in sys.argv else (ty if ty else cand))[:4]
          if "dongthuan" in sys.argv and len(ty) < 2:
              # §66: CHỌN HẠT THEO ĐỒNG THUẬN LỚP thay vì top-k theo điểm.
              # ① top-60 CÓ ≥2 đồng-thành-viên ở 11,8% ca, nhưng top-4 theo
              # điểm lấy đúng 0%. Thay vì xếp hạng từng ứng viên rời rạc, tìm
              # NHÓM ứng viên đã cùng trỏ về một nhãn — đó chính là định nghĩa
              # «cùng lớp». Cùng trực giác «đồng thuận» của §21, đặt ở ① thay
              # vì ②.
              _be = idx.via_pivot
              _nhan: dict[int, list] = defaultdict(list)
              for h in cand[:60]:
                  for i2 in (H.resolve_surface_ids(v, h.surface) or [h.tid])[:3]:
                      for d2 in _be.get(int(i2), {}):
                          if h not in _nhan[int(d2)]:
                              _nhan[int(d2)].append(h)
              _dt2 = H.chon_hat_dong_thuan(
                  idx, v, cand, topk=4,
                  pos_map=(pm if "danhtu" in sys.argv else None))
              if _dt2:
                  hat = _dt2
          if "ctxnhan" in sys.argv and len(ty) < 2:
              # B3 (§67): nhóm thắng phải là nhóm mà NHÃN của nó từng được mô
              # tả bằng CHÍNH thuộc tính P. `label_ctx` (§43) giữ túi token ngữ
              # cảnh của các câu sinh ra cạnh vào nhãn — nếu P nằm trong túi đó
              # thì lớp này đúng là lớp hay mang thuộc tính P.
              _nh3: dict[int, list] = defaultdict(list)
              for _h in cand[:60]:
                  for _i in (H.resolve_surface_ids(v, _h.surface) or [_h.tid])[:3]:
                      for _d in idx.via_pivot.get(int(_i), {}):
                          if _h not in _nh3[int(_d)]:
                              _nh3[int(_d)].append(_h)
              _pid = set(int(x) for x in (H.resolve_surface_ids(v, p) or []))
              _kh = [(len(_g), _g) for _lid, _g in _nh3.items() if len(_g) >= 2
                     and (_pid & set(idx.label_ctx.get(int(_lid), {})))]
              if _kh:
                  hat = max(_kh, key=lambda kv: kv[0])[1][:4]
          if "chum" in sys.argv and len(ty) < 2:
              # B2 (§67): CHÙM — chấm NHIỀU nhóm hạt bằng chính bước ②, lấy
              # nhóm cho nhãn mạnh nhất. Chọn nhóm trước rồi mới chạy ② là ép
              # ① quyết định mà không có thông tin của ②.
              _nh: dict[int, list] = defaultdict(list)
              for _h in cand[:60]:
                  for _i in (H.resolve_surface_ids(v, _h.surface) or [_h.tid])[:3]:
                      for _d in idx.via_pivot.get(int(_i), {}):
                          if _h not in _nh[int(_d)]:
                              _nh[int(_d)].append(_h)
              _nhom = sorted((g for g in _nh.values() if len(g) >= 2),
                             key=len, reverse=True)[:5]
              _best = None
              for _g in _nhom:
                  _s = H.score_hubs_from(
                      state, idx, [x.tid for x in _g[:4]], topk=5, gidf_min=0.0,
                      channel="typed_first", rank=cfg.hub_chain_rank,
                      loc_pos=True, ctx_ids=ctx)
                  _s = [x for x in _s if x.surface not in {y.surface for y in _g}
                        and x.surface != p]
                  if _s and (_best is None or _s[0].score > _best[0]):
                      _best = (_s[0].score, _g)
              if _best:
                  hat = _best[1][:4]
          if not hat:
              ket["②_sai"] += 1
              continue
          _dong = sum(1 for h in hat if h.surface in set(mem))
          dem_hat[min(3, _dong)] = dem_hat.get(min(3, _dong), 0) + 1
          s2 = H.score_hubs_from(state, idx, [h.tid for h in hat], topk=20,
                                 gidf_min=0.0, channel="typed_first",
                                 rank=cfg.hub_chain_rank, loc_pos=True, ctx_ids=ctx)
          cam = {h.surface for h in hat} | {p}
          ok = [h for h in s2 if h.surface not in cam]
          nhan = ok[0].surface if ok else None
          dung_nhan = nhan is not None and nhan in (c, c.split()[0])
          ra: list[str] = []
          if ok:
              nb = H.resolve_surface_ids(v, nhan) or [ok[0].tid]
              ra = [m.surface for m in H.reverse_peers(
                  state, idx, [ok[0].tid] + nb[:3], topk=30, gidf_min=0.0,
                  channel="typed_first", ctx_ids=ctx,
                  ke_ctx=cfg.hub_chain_ke_ctx)]
          trung = [x for x in ra if x in khi]
          if not dung_nhan:
              ket["②_sai"] += 1
          elif trung:
              ket["đạt"] += 1
          else:
              ket["③_thiếu"] += 1
          dau = "✓" if (dung_nhan and trung) else ("~" if dung_nhan else "✗")
          print(f"  {dau} lớp={c!r:<14} P={p!r:<12} hạt={[h.surface for h in hat][:3]}")
          print(f"      ② → {nhan!r:<14} (đúng={dung_nhan})   khỉ cần = {khi[:4]}")
          print(f"      ③ → {ra[:6]}   TRÚNG = {trung[:3]}")
    print(f"\n  {ket}")
    tong = max(1, sum(dem_hat.values()))
    co = [x for x in hang60 if x]
    hai = [x for x in hang60 if len(x) >= 2]
    print(f"  ① trong TOP-60: có ≥1 đồng-thành-viên {len(co)}/{len(hang60)} "
          f"({100*len(co)/max(1,len(hang60)):.1f}%) · có ≥2: {len(hai)} "
          f"({100*len(hai)/max(1,len(hang60)):.1f}%)")
    if hai:
        print(f"     hạng của 2 đồng-thành-viên đầu (10 ca): {hai[:10]}")
    if chia_nhan:
        print(f"  ① ca có ≥2 đồng-thành-viên trong bể: {len(chia_nhan)} · "
              f"CHIA NHAU một nhãn: {sum(chia_nhan)} "
              f"({100*sum(chia_nhan)/len(chia_nhan):.1f}%)")
        print(f"     5 ca KHÔNG chia (P, hai thành viên, số nhãn mỗi bên): "
              f"{thieu_canh[:5]}")
    print(f"  ① số hạt LÀ ĐỒNG-THÀNH-VIÊN của lớp đích: "
          f"0 → {dem_hat.get(0,0)} ({100*dem_hat.get(0,0)/tong:.1f}%) · "
          f"1 → {dem_hat.get(1,0)} · ≥2 → {dem_hat.get(2,0)+dem_hat.get(3,0)} "
          f"({100*(dem_hat.get(2,0)+dem_hat.get(3,0))/tong:.1f}%)")
    print(f"\n[{time.time() - T0:.0f}s] xong")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
