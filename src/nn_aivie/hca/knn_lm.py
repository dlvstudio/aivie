"""§TK_056 — kNN-LM cho đường SINH.

Ý tưởng (Khandelwal 2020, hội tụ với TK_042): mỗi bước sinh, lấy vector LEFT-
context của encoder (`ctx_encoder.causal_last`), truy k hàng xóm gần nhất trong
DATASTORE `(h, next_token)` (dựng bởi `scripts/dung_datastore.py`), gộp next-token
của hàng xóm thành phân bố `P_kNN`, rồi TRỘN vào điểm vote W_skip:

    P = (1−λ)·P_Wskip + λ·P_kNN     (chỉ trên tập ỨNG VIÊN SỐNG, tôn trọng mask)

POC (§TK_056): encoder h hơn đếm 1.42 bits, hơn túi-tĩnh 0.95 bits — phá tường
đếm. Torch là phụ thuộc TÙY CHỌN; thiếu torch/datastore/encoder → `load_knn_lm`
trả None và đường sinh chạy y như cũ (không kNN).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

_DS_PATH = "checkpoints/net_ls_concept_conf.datastore.npz"


class KnnLM:
    """Datastore kNN + encoder ngữ cảnh → phân bố next-token trộn vào vote."""

    def __init__(self, encoder, keys_np, vals, torch_mod, dev) -> None:
        self._enc = encoder            # ContextEncoder
        # §TK_063 — keys giữ ở NUMPY fp32 (CPU), matmul similarity bằng numpy BLAS
        # gemv. MPS matmul (N≈1M, d) dính "known issue" → rơi metal path chậm
        # (nặng ở d≥384: vài giây/bước). numpy gemv ~vài ms, KHÔNG phụ thuộc MPS.
        self._Knp = np.ascontiguousarray(keys_np, dtype=np.float32)  # (N, d) L2-chuẩn hoá
        self._vals = vals              # np int32 (N,)
        self._t = torch_mod
        self._dev = dev

    def next_dist(self, out_ids, n_vocab, k=64, T=0.08):
        """Phân bố next-token (np, tổng≈1 trên các token hàng xóm) từ kNN; None
        nếu không lấy được vector ngữ cảnh.

        Tối ưu latency: matmul trên MPS (nhanh) NHƯNG top-k bằng numpy
        argpartition trên CPU — `torch.topk` trên MPS quá chậm với ~1M hàng
        (23s/câu). argpartition ~ vài ms."""
        h = self._enc.causal_last(out_ids)
        if h is None:
            return None
        sim = self._Knp @ h.astype(np.float32)      # (N,) numpy BLAS gemv — né MPS
        kk = min(k, sim.shape[0])
        idx = np.argpartition(sim, -kk)[-kk:]       # top-k không sắp xếp
        sv = sim[idx].astype(np.float64)
        sv -= sv.max()
        w = np.exp(sv / T); w /= w.sum()
        nb = self._vals[idx]
        P = np.zeros(n_vocab, np.float64)
        np.add.at(P, nb, w)
        s = P.sum()
        return P / s if s > 0 else None

    def rerank(self, out_ids, scores, n_vocab, a=0.6, k=64, T=0.08, eps=1e-4):
        """§TK_082 (Pha 1b) — RERANK product-of-experts, thay TRỘN-λ (đo được: trơ).

        Trên tập ứng viên SỐNG của W_skip, kết hợp NHÂN thay vì cộng:
            comb[t] = base[t]^(1−a) · (sup[t]+eps)^a
        base = P_Wskip (chuẩn hoá trên sống), sup = P_kNN (datastore SẠCH) hạn
        trên sống. kNN XẾP-LẠI thứ tự ứng viên LM (không hồi sinh token −inf/chết),
        DẬP ứng viên W_skip lạc-chủ-đề mà datastore factual không hậu thuẫn, KÉO
        về token đúng chủ đề. `a`∈[0,1]: 0=thuần LM, 1=thuần kNN. Rơi về nguyên
        trạng nếu kNN không nói gì về ứng viên sống (an toàn, không kẹt)."""
        live = np.isfinite(scores) & (scores > 0)
        if not live.any():
            return scores
        P = self.next_dist(out_ids, n_vocab, k, T)
        if P is None:
            return scores
        sp = np.where(live, scores, 0.0); tot = float(sp.sum())
        if tot <= 0:
            return scores
        base = sp / tot
        sup = np.where(live, P, 0.0); ssup = float(sup.sum())
        if ssup <= 0:
            return scores                      # kNN mù trên ứng viên sống ⇒ giữ
        sup = sup / ssup
        comb = np.zeros_like(base)
        comb[live] = (base[live] ** (1.0 - a)) * ((sup[live] + eps) ** a)
        cs = float(comb.sum())
        if cs <= 0:
            return scores
        out = scores.copy()
        out[live] = (comb[live] / cs) * tot    # giữ tổng thang điểm sống
        return out

    def blend(self, out_ids, scores, lam, n_vocab, vocab=None, k=64, T=0.08,
              widen_k=0, widen_recent=4, widen_gain=1.0):
        """TRỘN P_kNN vào `scores`. Nội suy `P=(1−λ)·P_Wskip+λ·P_kNN` trên tập
        ứng viên = SỐNG (finite & >0) ∪ NỚI (top-k kNN là surface mà W_skip chưa
        bầu). Token nới chỉ mang khối kNN (W_skip=0) nên cạnh tranh CÔNG BẰNG,
        không auto-thắng. Tôn trọng tường cứng: KHÔNG hồi sinh token `-inf`
        (surface/structural đã bị mask), bỏ token vừa dùng gần (chống lặp/consume).

        `widen_k=0` (MẶC ĐỊNH) ⇒ TẮT nới, chỉ xáo lại tập sống. Nới đã dựng +
        an toàn nhưng ĐO ĐƯỢC vô ích ở encoder hiện tại (masked-train, ngữ cảnh
        2-token): top-kNN bị tần suất chi phối ("một"/"là" — W_skip đã có) hoặc
        nhiễu chéo miền ("lọ"/"xe"); gain=1 thắng 0/22, gain=6 chỉ nổi nhiễu.
        BẬT (widen_k>0) khi có causal encoder TK_059 làm `h` sắc hơn."""
        live = np.isfinite(scores) & (scores > 0)
        if not live.any():   # W_skip cạn ⇒ để generate dừng tự nhiên, không ép
            return scores
        P = self.next_dist(out_ids, n_vocab, k, T)
        if P is None:
            return scores
        tot = float(scores[live].sum())
        if tot <= 0:
            return scores
        cand = live.copy()
        # ── NỚI: thêm token kNN-top mà W_skip bỏ sót (surface, không cứng) ──
        if vocab is not None and widen_k > 0:
            recent = {int(t) for t in out_ids[-widen_recent:]} if widen_recent else set()
            try:
                from nn_aivie.hca.context_positional import is_structural_token as _struct
            except Exception:
                _struct = None
            order = np.argsort(P)[::-1][:widen_k]
            for tok in order:
                tok = int(tok)
                if P[tok] <= 0:
                    break
                if cand[tok] or not np.isfinite(scores[tok]) or scores[tok] > 0:
                    continue                      # đã sống / bị -inf cứng
                if tok in recent:
                    continue                      # vừa dùng ⇒ chống lặp/consume
                if not vocab.is_surface(tok):
                    continue
                if _struct is not None and _struct(vocab, tok):
                    continue
                cand[tok] = True
        sp = np.where(live, scores, 0.0)
        sp = sp / tot                              # P_Wskip (0 ở token nới)
        b = (1.0 - lam) * sp + lam * P
        if widen_gain != 1.0:
            # KHUẾCH ĐẠI khối token nới (chỉ mang kNN) để cạnh tranh được với
            # vote W_skip tự tin — nếu không, nới FIRING nhưng THẮNG=0 (đo được).
            widen_only = cand & ~live
            b = np.where(widen_only, lam * P * widen_gain, b)
        b = np.where(cand, b, 0.0)
        bs = float(b.sum())
        if bs <= 0:
            return scores
        out = scores.copy()
        out[cand] = (b[cand] / bs) * tot           # giữ tổng thang điểm sống
        return out


    def hybrid_blend(self, out_ids, scores, n_vocab, vocab, conf_lo=0.10,
                     conf_hi=0.50, w_max=0.7, rep_pen=1.6, temp=1.0,
                     lm_topk=12, recent=4, eos_id=None, prompt_len=0, min_gen=6):
        """§TK_061 — SINH LAI cổng-độ-tin. W_skip TỰ TIN (đỉnh vote áp đảo) → giữ
        nguyên (bảo vệ SỰ THẬT grounded). W_skip PHÂN VÂN/CẠN → để causal-LM DẪN
        cho MƯỢT (thay vì cụt/rác), có repetition-penalty. Tôn trọng -inf cứng
        (chỉ token surface finite), chống lặp gần, bỏ structural.

        Độ tin = khe hở đỉnh: (top1−top2)/top1 trên vote dương. w_lm nội suy từ
        [conf_lo,conf_hi]. conf≥conf_hi ⇒ w_lm=0 (W_skip quyết). Vote cạn ⇒ conf=0
        ⇒ causal-LM cứu (sinh tiếp thay vì dừng)."""
        lo = self._enc.causal_logits_last(out_ids)
        if lo is None:
            return scores
        lo = lo.astype(np.float64).copy()
        for tk in {int(t) for t in out_ids[-12:]}:
            if 0 <= tk < n_vocab and lo[tk] > 0:
                lo[tk] /= rep_pen
            elif 0 <= tk < n_vocab:
                lo[tk] *= rep_pen
        live = np.isfinite(scores) & (scores > 0)
        pos = scores[live]
        # §TK_061 — CHỐNG-CỤT: W_skip đòi EOS khi output còn NGẮN ⇒ để causal-LM
        # nối tiếp cho mượt (thứ cần ĐỘ DÀI). Không đụng câu đã dài (grounded def
        # tự dừng đúng chỗ). Chỉ ép khi EOS đang là đỉnh và gen_len < min_gen.
        gen_len = max(0, len(out_ids) - int(prompt_len))
        force_continue = False
        if eos_id is not None and 0 <= int(eos_id) < n_vocab and gen_len < min_gen and pos.size:
            fin = np.where(np.isfinite(scores), scores, -np.inf)
            if int(np.argmax(fin)) == int(eos_id) and fin[int(eos_id)] > 0:
                force_continue = True
        if pos.size == 0:
            conf = 0.0; tot = 0.0
        elif pos.size == 1 and not force_continue:
            return scores                       # 1 ứng viên: W_skip bị ép, giữ
        else:
            sp = np.sort(pos)[::-1]
            conf = float((sp[0] - sp[1]) / (sp[0] + 1e-9)) if sp.size >= 2 else 1.0
            tot = float(pos.sum())
        if force_continue:
            conf = 0.0                           # ép causal-LM dẫn để nối tiếp
        w_lm = float(np.clip((conf_hi - conf) / (conf_hi - conf_lo + 1e-9), 0.0, 1.0)) * w_max
        if w_lm <= 0.0:
            return scores                        # W_skip tự tin ⇒ giữ SỰ THẬT
        try:
            from nn_aivie.hca.context_positional import is_structural_token as _struct
        except Exception:
            _struct = None
        surf = np.isfinite(scores)               # không -inf cứng ⇒ surface
        if force_continue and eos_id is not None:
            surf[int(eos_id)] = False            # cấm EOS ⇒ buộc nối tiếp
        lom = np.where(surf, lo, -np.inf)
        rec = {int(t) for t in out_ids[-recent:]} if recent else set()
        cand = live.copy()
        if force_continue and eos_id is not None:
            cand[int(eos_id)] = False
        order = np.argsort(lom)[::-1][:lm_topk]
        for tok in order:
            tok = int(tok)
            if not np.isfinite(lom[tok]):
                break
            if tok in rec:
                continue
            if _struct is not None and _struct(vocab, tok):
                continue
            cand[tok] = True
        if not cand.any():
            return scores
        # P_causalLM (softmax trên cand), P_Wskip (chuẩn hoá vote dương trên cand)
        lc = np.where(cand, lo, -np.inf); lc = lc - np.nanmax(lc[np.isfinite(lc)])
        plm = np.where(cand, np.exp(lc / temp), 0.0); s = plm.sum()
        if s <= 0:
            return scores
        plm /= s
        pw = np.where(live, scores, 0.0); sw = pw.sum()
        pw = pw / sw if sw > 0 else pw
        b = (1.0 - w_lm) * pw + w_lm * plm
        b = np.where(cand, b, 0.0); bs = b.sum()
        if bs <= 0:
            return scores
        out = scores.copy()
        scale = tot if tot > 0 else 1.0          # cạn ⇒ scale=1 để sinh tiếp
        out[cand] = (b[cand] / bs) * scale
        if force_continue and eos_id is not None:
            out[int(eos_id)] = 0.0               # dìm EOS ⇒ nối token thật thắng
        return out


    def lm_extend(self, ids, eos_id=None, max_new=24, rep_pen=1.6,
                  knn_ground=0.4, anchor=0.0, T=0.08, k=64, vocab=None, min_ok=1,
                  anchor_ramp=0.0, conf_stop=0.0, conf_patience=3, conf_after=4,
                  conf_maxextra=20):
        """§TK_061-v2 BÀN GIAO SPAN — nối tiếp `ids` bằng causal-LM cho MƯỢT khi
        W_skip cụt. W_skip neo mở đầu GROUNDED, causal-LM lo phần đuôi.

        §TK_062 CHỐNG-DRIFT: mỗi bước trộn logits causal-LM (trôi chảy) với phân
        bố kNN từ datastore (thực tế corpus) → `P=(1−g)·P_LM+g·P_kNN`. kNN kéo về
        chuỗi THẬT SỰ xuất hiện ⇒ bớt ẢO GIÁC (kNN-LM Khandelwal). `anchor>0`
        cộng thêm thiên lệch về CENTROID embedding của token nội-dung trong prompt
        (giữ CHỦ ĐỀ). Một forward/bước (h dùng cho cả logits lẫn truy kNN).

        Trả list ids đã nối. Dừng ở EOS hoặc `max_new` token mới."""
        t = self._t
        m = self._enc._m
        # §TK_064 — maxlen THẬT của model (bảng pos), để model ngữ-cảnh-dài (128)
        # dùng đủ cửa sổ khi nối; rơi về 48 nếu không đọc được.
        try:
            _ML = int(m.pos.num_embeddings)
        except Exception:
            from nn_aivie.hca.ctx_encoder import MAXLEN as _ML
        V = 32002
        W = m.emb.weight[:V]
        out = [int(x) for x in ids]
        bg = set()
        # neo chủ đề: centroid embedding các token NỘI-DUNG của prompt gốc (bỏ
        # hư từ/cấu trúc để centroid không bị "và/là" pha loãng)
        anchor_sim = None
        if anchor > 0:
            emb = W.detach().float().cpu().numpy()
            _struct = None
            if vocab is not None:
                try:
                    from nn_aivie.hca.context_positional import is_structural_token as _struct
                except Exception:
                    _struct = None
            seed = [tok for tok in out if 0 <= tok < V]
            if _struct is not None:
                seed = [tok for tok in seed if not _struct(vocab, tok)] or seed
            if seed:
                c = emb[np.asarray(seed)].mean(0)
                cn = np.linalg.norm(c)
                if cn > 0:
                    En = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)
                    anchor_sim = (En @ (c / cn)).astype(np.float64)  # cos[V]
        lowconf = 0                       # §TK_065 — đếm bước độ-tin thấp liên tiếp
        want_stop = False                 # đã tụt-tin: dừng ở DẤU CÂU kế (≥1 câu trọn)
        extra = 0                         # số token sinh thêm sau khi muốn dừng
        _BND = (".", "!", "?", "…", "*", ";", ":")
        MN = max(1, int(max_new) - 1)

        def _is_bnd(tok):
            if vocab is None:
                return False
            try:
                s = vocab.detokenize([int(tok)]).strip()
            except Exception:
                return False
            return bool(s) and s[-1] in _BND

        with t.no_grad():
            for i in range(int(max_new)):
                e = out[-_ML:]
                x = t.zeros(1, _ML, dtype=t.long)
                x[0, : len(e)] = t.tensor(e)
                pm = t.arange(_ML)[None] >= len(e)
                h = m.hidden(x, pm, causal=True)[0, len(e) - 1]
                lo = (h @ W.T).float().cpu().numpy()
                lo = lo - lo.max()
                plm = np.exp(lo); plm /= plm.sum()
                # §TK_065 DỪNG THEO ĐỘ-TIN: LM mất mạch (đỉnh phân bố tụt) → LATCH
                # want_stop; KHÔNG cắt ngay mà dừng ở DẤU CÂU kế (đảm bảo ≥1 câu
                # trọn, không treo giữa cụm). Nếu +conf_maxextra token vẫn chưa ra
                # dấu câu → dừng cứng. Chỉ xét sau conf_after token.
                if conf_stop > 0.0 and not want_stop and i >= int(conf_after):
                    if float(plm.max()) < conf_stop:
                        lowconf += 1
                        if lowconf >= int(conf_patience):
                            want_stop = True
                    else:
                        lowconf = 0
                mix = plm
                if knn_ground > 0:
                    hn = h.float().cpu().numpy()
                    nn_ = np.linalg.norm(hn)
                    if nn_ > 0:
                        sim = self._Knp @ (hn / nn_).astype(np.float32)  # numpy gemv — né MPS
                        kk = min(k, sim.shape[0])
                        idx = np.argpartition(sim, -kk)[-kk:]
                        sv = sim[idx].astype(np.float64); sv -= sv.max()
                        w = np.exp(sv / T); w /= w.sum()
                        Pk = np.zeros(V, np.float64); np.add.at(Pk, self._vals[idx], w)
                        s = Pk.sum()
                        if s > 0:
                            mix = (1.0 - knn_ground) * plm + knn_ground * (Pk / s)
                if anchor_sim is not None:
                    # §TK_065 anchor LŨY TIẾN: yếu đầu đoạn (chảy tự nhiên) →
                    # mạnh dần về cuối (kéo về chủ đề khi drift tích tụ).
                    eff = anchor * (1.0 + anchor_ramp * (i / MN))
                    mix = mix * np.exp(eff * anchor_sim)  # thiên lệch chủ đề
                # repetition penalty + no-repeat
                for tk in set(out[-12:]):
                    mix[tk] /= (rep_pen * rep_pen)
                if out:
                    for a, b in bg:
                        if a == out[-1]:
                            mix[b] = 0.0
                    mix[out[-1]] = 0.0
                nx = int(np.argmax(mix))
                if eos_id is not None and nx == int(eos_id):
                    break
                if out:
                    bg.add((out[-1], nx))
                out.append(nx)
                # đã muốn dừng → kết ở dấu câu kế (hoặc dừng cứng sau conf_maxextra)
                if want_stop:
                    if _is_bnd(nx):
                        break
                    extra += 1
                    if extra >= int(conf_maxextra):
                        break
        return out


def load_knn_lm(encoder, ds_path: str | Path = _DS_PATH) -> "KnnLM | None":
    """Nạp datastore + gắn encoder. Trả None (âm thầm) nếu thiếu torch / file /
    encoder — đường sinh rơi về vote thuần, không vỡ."""
    if encoder is None:
        return None
    p = Path(ds_path)
    if not p.exists():
        return None
    try:
        import torch
    except Exception:
        return None
    try:
        d = np.load(str(p))
        keys = d["keys"].astype(np.float32)   # §TK_063 — CPU numpy, matmul né MPS
        vals = d["vals"].astype(np.int32)
        dev = "mps" if torch.backends.mps.is_available() else "cpu"
    except Exception:
        return None
    return KnnLM(encoder, keys, vals, torch, dev)
