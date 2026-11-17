"""Giao-diện chat (kiểu ChatGPT) cho HybridLM 1.3B của ta — chạy cục-bộ trên Mac.

    .venv/bin/python scripts/serve_hybrid_ui.py --port 8790        # rồi mở http://localhost:8790

• Sinh TỪNG BƯỚC (scripts/hybrid_infer.py): Mamba giữ trạng-thái + KV-cache attention → ~15 tok/s trên M4
  (vòng cũ tính lại cả ngữ-cảnh mỗi token: ~0,5 tok/s ở ngữ-cảnh 300).
• Giữ trạng-thái hội-thoại: lượt sau chỉ NẠP TIẾP câu hỏi mới (không nạp lại lịch-sử) khi lịch-sử khớp.
• Khuôn SFT đúng như lúc train (train_hybrid_sft.build_example): "Hỏi: …\\n" + "Đáp: " + trả-lời + <EOS>.
• Tuỳ-chọn tra viwiki (BM25F, như rag_hybrid.py) — CHỈ nạp khi người dùng bật lần đầu (~20-40s): dựng chỉ-mục
  BM25 bằng Python thuần giữ GIL → nếu nạp nền lúc khởi-động, luồng sinh chữ bị bỏ đói (đo: chờ 50s, 2,9 tok/s).
• Công-tắc "Trò chuyện": gắn câu dặn hệ-thống CHAT_SYS (bản giao-tiếp v2 train kèm câu dặn này → tán gẫu; không câu
  dặn → trợ-lý làm việc). Công-tắc "Máy tính": hễ model viết xong "<biểu-thức> =" thì tính hộ và chèn kết-quả đúng
  (scripts/math_calc.py — toán eval_v1 ×3).
Chỉ dùng thư-viện chuẩn cho HTTP; trả lời stream dạng Server-Sent Events.
"""
from __future__ import annotations
import argparse, hashlib, json, os, re, sys, threading, time
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, os.path.join(ROOT, "src")]
os.chdir(ROOT)

import torch
from hybrid_infer import load, Runner
from nn_aivie.core.tokenizer import BPETokenizer
from nn_aivie.core.vocab import Vocabulary

CHAT_SYS = "Hãy trò chuyện tự nhiên như người bạn."
MAXCTX = 2000                     # model train ở ctx 2048 (RoPE chưa giãn) → giữ dưới
S = {}                            # trạng-thái server
LOCK = threading.Lock()
STOP = threading.Event()
CACHE: "OrderedDict[str, dict]" = OrderedDict()   # conv_id → {key, ids, state}  (LRU 4)
VCACHE: "OrderedDict[str, torch.Tensor]" = OrderedDict()   # sha1(ảnh) → token ảnh (LRU 16)
IMG = -1                          # id giữ chỗ token ảnh (vlm_1p3b.IMG)


# ── RAG (viwiki) ─────────────────────────────────────────────────────────────────────
class Rag:
    def __init__(self):
        from retrieval_v2 import load_corpus_v2, tok, STOP as SW, extract_title
        self.tok, self.SW, self.title = tok, SW, extract_title
        self.bm, self.snips = load_corpus_v2(285000)

    def focus(self, text, query, win=700, step=140):
        qset = {t for t in self.tok(query) if t not in self.SW and len(t) >= 2}
        if not qset or len(text) <= win:
            return text[:win]
        low = text.lower()
        wt = {t: self.bm.idf.get(t, 0.0) / (1 + low.count(t)) for t in qset}
        best_s, best = 0, -1.0
        for s in range(0, len(low) - win + 1, step):
            sc = sum(wt[t] for t in qset if t in low[s:s + win])
            if sc > best:
                best, best_s = sc, s
        return text[best_s:best_s + win] if best_s == 0 else "…" + text[best_s:best_s + win]

    def augment(self, q, k=3):
        ids = self.bm.top(q, k=k)
        src = [{"title": self.title(self.snips[i]), "text": self.focus(self.snips[i], q)} for i in ids]
        ctx = "\n".join(f"[Nguồn {j+1} — {s['title']}] {s['text']}" for j, s in enumerate(src))
        return (f"Dựa vào các nguồn dưới đây, trả lời ngắn gọn bằng tiếng Việt. "
                f"Nếu nguồn không có, nói không tìm thấy.\n{ctx}\nCâu hỏi: {q}"), src


def rag_state():
    if "rag" in S:
        return "ready"
    if "rag_err" in S:
        return "error"
    return "loading" if S.get("rag_loading") else "idle"


def _load_rag():
    S["rag_loading"] = True
    try:
        t = time.time()
        S["rag"] = Rag()
        print(f"[rag] kho viwiki sẵn sàng ({time.time()-t:.0f}s)", flush=True)
    except Exception as e:                                     # không có kho → tắt RAG, UI vẫn chạy
        S["rag_err"] = f"{type(e).__name__}: {e}"
        print(f"[rag] LỖI: {S['rag_err']}", flush=True)


class Toggle(torch.nn.Module):
    """bọc một expert: off=True → bỏ qua (x đi thẳng)."""
    def __init__(self, m):
        super().__init__()
        self.m, self.off = m, False

    def forward(self, x):
        return x if self.off else self.m(x)


# ── Ảnh (checkpoint có mục "vision" — scripts/vlm_1p3b.py) ──────────────────────────
def vis_of(src):
    """data-URL ảnh → token ảnh (n, d) trên thiết bị model; nhớ đệm theo sha1 (lượt sau / nạp lại không chạy mắt lại)."""
    k = hashlib.sha1(src.encode()).hexdigest()
    if k in VCACHE:
        VCACHE.move_to_end(k)
        return VCACHE[k]
    import vlm_1p3b as V
    px, grids = V.prep_images([src], S["max_tiles"])
    with torch.no_grad():
        v = S["vision"](px, grids)[0]
    VCACHE[k] = v
    while len(VCACHE) > 16:
        VCACHE.popitem(last=False)
    return v


def enc_umsg(c, images):
    """→ (ids, [token ảnh]) — khuôn như vlm_1p3b.build_example: "Hỏi: " <ảnh…> "\n" câu "\n"."""
    tok = S["tok"]
    if images and S.get("vision") is not None:
        vs = [vis_of(x) for x in images]
        ids = tok.encode("Hỏi: ")
        for v in vs:
            ids += [IMG] * v.shape[0]
        return ids + tok.encode("\n" + c.strip() + "\n"), vs
    return tok.encode("Hỏi: " + c.strip() + "\n"), []


# ── Khuôn hội-thoại ──────────────────────────────────────────────────────────────────
def enc_user(c, images=None):
    ids, vs = enc_umsg(c, images)
    return ids + S["tok"].encode("Đáp: "), vs


def enc_history(msgs, system):
    tok, eos = S["tok"], S["eos"]
    ids = tok.encode(system.strip() + "\n") if system.strip() else []
    vis = []
    for m in msgs:
        c = m["content"].strip()
        if m["role"] == "user":
            e, vs = enc_umsg(c, m.get("images"))
            ids += e; vis += vs
        elif m["role"] == "assistant":
            ids += tok.encode("Đáp: ") + tok.encode(c) + [eos]
    return ids, vis


def calc_generate(r, ids, n, temp, top_p, rep_pen, eos, stop, state, hist, out, recent=64, vis=None):
    """Như Runner.generate + MÁY TÍNH: gặp '<biểu-thức> =' → ép token kết-quả đúng (rồi cấm chữ-số ở bước kế)."""
    import numpy as np
    from math_calc import EXPR, calc, digit_ids
    rng = np.random.default_rng(0)
    lo, state = r.prefill(ids, state=state, vis=vis)
    lo = lo[-1]
    hist = list(hist or []) + list(ids)
    out["state"] = state
    gen, just = [], False
    while len(gen) < n:
        if stop():
            return
        l = lo.float().cpu().numpy().astype(np.float64)
        for t in set(hist[-recent:]):
            if 0 <= t < len(l):                                      # id âm = token ảnh
                l[t] -= rep_pen
        if just:
            l[digit_ids(S["vocab"], len(l))] = -1e9
            just = False
        if temp <= 0:
            nxt = int(l.argmax())
        else:
            l = (l - l.max()) / temp
            p = np.exp(l); p /= p.sum()
            order = np.argsort(p)[::-1]
            keep = order[: max(1, int(np.searchsorted(np.cumsum(p[order]), top_p)) + 1)]
            nxt = int(rng.choice(keep, p=p[keep] / p[keep].sum()))
        if eos is not None and nxt == eos:
            return
        forced = [nxt]
        gen.append(nxt)
        text = S["vocab"].detokenize(gen)
        if text.endswith("="):
            tail = text[-80:]
            m = EXPR.search(tail)
            # chốt chặn: biểu-thức chỉ là ĐOẠN CUỐI của một biểu-thức lớn có đơn-vị ("120 kg - 1/3 =") → không tính
            cut = bool(m) and re.search(r"[\+\-\*x×/:]\s*$", tail[:m.start()]) is not None
            v = calc(m.group(1)) if m and not cut else None
            if v is not None:
                extra = S["tok"].encode(" " + v)
                forced += extra; gen += extra; just = True
        for t in forced:
            yield t
            hist.append(t)
            lo, state = r.step(t, state)
            out["state"] = state


def key_of(system, msgs):
    raw = json.dumps([system.strip(), [(m["role"], m["content"].strip(),
                                        [hashlib.sha1(x.encode()).hexdigest() for x in m.get("images") or []]) for m in msgs]],
                     ensure_ascii=False)
    return hashlib.sha1(raw.encode()).hexdigest()


# ── HTTP ─────────────────────────────────────────────────────────────────────────────
class H(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def _json(self, code, obj):
        b = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            b = open(os.path.join(HERE, "hybrid_ui.html"), encoding="utf-8").read()
            for k_, v_ in S.get("ui", {}).items():               # nhãn theo checkpoint đang chạy (trước đây viết cứng "1.3B")
                b = b.replace(k_, v_)
            b = b.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Cache-Control", "no-store")        # sửa trang (vd mặc định max_new) phải tới tab đang mở khi tải lại
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)
        elif self.path == "/api/info":
            self._json(200, {"model": S["name"], "device": S["dev"], "busy": LOCK.locked(), "rag": rag_state(),
                             "rag_err": S.get("rag_err"), "vision": S.get("vision") is not None})
        else:
            self._json(404, {"error": "không có"})

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(n) or b"{}")
        if self.path == "/api/stop":
            STOP.set()
            return self._json(200, {"ok": True})
        if self.path == "/api/rag":                                    # bật tra-cứu lần đầu → nạp kho
            if rag_state() == "idle":
                threading.Thread(target=_load_rag, daemon=True).start()
            return self._json(200, {"rag": rag_state()})
        if self.path != "/api/chat":
            return self._json(404, {"error": "không có"})
        if not LOCK.acquire(blocking=False):
            return self._json(429, {"error": "Model đang trả lời câu khác — chờ xong hoặc bấm Dừng."})
        try:
            self._chat(body)
        finally:
            LOCK.release()

    def _send(self, obj):
        self.wfile.write(f"data: {json.dumps(obj, ensure_ascii=False)}\n\n".encode())
        self.wfile.flush()

    def _chat(self, b):
        STOP.clear()
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        r, tok, vocab, eos = S["runner"], S["tok"], S["vocab"], S["eos"]
        msgs = [m for m in b.get("messages", []) if m.get("content", "").strip() or m.get("images")]
        system = CHAT_SYS if b.get("chat_mode") else (b.get("system", "") or "")
        mode = b.get("mode", "chat")
        n = max(1, min(int(b.get("max_new", 300)), MAXCTX - 64))
        nb = min(n, 1024)          # chỗ DÀNH cho câu trả lời khi cắt lịch sử — câu trả lời thật co theo chỗ còn trống (dưới)
        opts = dict(temp=float(b.get("temperature", 0.7)), top_p=float(b.get("top_p", 0.9)),
                    rep_pen=float(b.get("rep_pen", 1.1)))
        conv = b.get("conv_id", "")
        sources, cached, state, hist, vis = [], False, None, [], []
        try:
            if mode == "raw":                                          # viết tiếp văn-bản thô
                feed, eos_stop = tok.encode(msgs[-1]["content"] if msgs else ""), None
            else:
                if not msgs or msgs[-1]["role"] != "user":
                    return self._send({"type": "error", "error": "Tin nhắn cuối phải là của người dùng."})
                q = msgs[-1]["content"]
                if b.get("rag"):
                    if "rag" not in S:
                        if rag_state() == "idle":
                            threading.Thread(target=_load_rag, daemon=True).start()
                        self._send({"type": "error", "error": "Kho viwiki đang nạp (~20-40 giây) — chờ chấm "
                                    "xanh ở góc trái rồi gửi lại, hoặc tắt Tra cứu viwiki."
                                    + (f" ({S['rag_err']})" if "rag_err" in S else "")})
                        return
                    q, sources = S["rag"].augment(q)
                tail, tvis = enc_user(q, msgs[-1].get("images"))
                prev_key = key_of(system, msgs[:-1])
                c = CACHE.get(conv)
                if c and c["key"] == prev_key and c["state"]["pos"] + len(tail) + nb <= MAXCTX:
                    state, hist, feed, vis, cached = c["state"], c["ids"], tail, tvis, True
                    CACHE.move_to_end(conv)
                else:                                                  # nạp lại: cắt lượt cũ nếu quá dài
                    prev = msgs[:-1]
                    base, bvis = enc_history(prev, system)
                    while prev and len(base) + len(tail) + nb > MAXCTX:
                        prev = prev[2:] if len(prev) >= 2 else []
                        base, bvis = enc_history(prev, system)
                    feed, vis = base + tail, bvis + tvis
                    if len(feed) + nb > MAXCTX:                         # câu hỏi quá dài → cắt đầu
                        if vis:
                            return self._send({"type": "error", "error": f"Ảnh + câu hỏi dài {len(feed)} token, vượt ngữ "
                                               f"cảnh {MAXCTX} — dùng ảnh nhỏ hơn hoặc câu ngắn hơn."})
                        feed = feed[-(MAXCTX - nb):]
                eos_stop = eos
            n = max(1, min(n, MAXCTX - (state["pos"] if state else 0) - len(feed)))   # không vượt ngữ cảnh đã học
            self._send({"type": "meta", "prefill": len(feed), "cached": cached, "sources": sources,
                        "ctx": (state["pos"] if state else 0) + len(feed)})
            t0 = time.time()
            out_ids, box, ttft = [], {}, None
            vv = torch.cat(vis) if vis else None
            if S.get("vision") is not None:                            # phần đọc ảnh (LoRA + expert thị giác) chỉ bật
                import vlm_1p3b as V                                   # khi hội thoại có ảnh → chat chữ y hệt v5
                on = any(m.get("images") for m in msgs)
                V.LoRA.on = on
                for b_ in S.get("vexp", []):
                    b_.off = not on
            gen = (calc_generate(r, feed, n, opts["temp"], opts["top_p"], opts["rep_pen"], eos_stop, STOP.is_set,
                                 state, hist, box, vis=vv) if b.get("calc") and mode != "raw" else
                   r.generate(feed, n=n, eos=eos_stop, stop=STOP.is_set, state=state, hist=hist, out=box, vis=vv, **opts))
            for t in gen:
                if ttft is None:
                    ttft = time.time() - t0
                out_ids.append(t)
                self._send({"type": "delta", "text": vocab.detokenize(out_ids)})
            dt = time.time() - t0
            text = vocab.detokenize(out_ids).strip()
            gen_t = dt - (ttft or dt)
            self._send({"type": "done", "text": text, "tokens": len(out_ids),
                        "tok_s": round((len(out_ids) - 1) / gen_t, 1) if len(out_ids) > 1 and gen_t > 0 else None,
                        "ttft": round(ttft if ttft is not None else dt, 1), "stopped": STOP.is_set(),
                        "cached": cached})
            if mode != "raw" and "state" in box:                       # lưu trạng-thái cho lượt sau
                _, st = r.step(eos, box["state"])                      # khép lượt bằng <EOS> như lúc train
                CACHE[conv] = {"key": key_of(system, msgs + [{"role": "assistant", "content": text}]),
                               "ids": (hist or []) + feed + out_ids + [eos], "state": st}
                CACHE.move_to_end(conv)
                while len(CACHE) > 4:
                    CACHE.popitem(last=False)
        except (BrokenPipeError, ConnectionResetError):
            STOP.set()
        except Exception as e:
            print(f"[lỗi /api/chat] {type(e).__name__}: {str(e)[:300]}", flush=True)   # trước đây nuốt im → đo sai (OOM 18/10)
            if isinstance(e, torch.cuda.OutOfMemoryError):
                torch.cuda.empty_cache()
            try:
                self._send({"type": "error", "error": f"{type(e).__name__}: {e}"})
            except Exception:
                pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/hf_hybrid/hybrid_1p3b_assistant_v5.pt")
    ap.add_argument("--port", type=int, default=8790)
    ap.add_argument("--no-rag", action="store_true")
    ap.add_argument("--max-ctx", type=int, default=2000, help="ngữ cảnh tối đa (token). 2000 = mức bản 1.3B; thân 2,6B "
                    "học 8K, quỹ đạo tool-use C3 train tới 4096 → nên 4096 khi dùng công cụ")
    ap.add_argument("--adapters", default="", help="expert GatedFFN (TK_104 §14), vd checkpoints/hf_hybrid/expert_code/code_VI_bf16.pt")
    ap.add_argument("--hid", type=int, default=8192, help="cỡ ẩn FFN của expert")
    ap.add_argument("--half", action="store_true", help="nạp thân + expert ở bf16 (M4 24 GB: fp32 ~17 GB → tràn swap, 21/10)")
    ap.add_argument("--expert-mode", default="normal", choices=["normal", "open", "off"],
                    help="kiểm chứng cổng: normal = cổng như đã học · open = bỏ cổng (expert chữ mở hẳn) · off = tắt expert chữ")
    a = ap.parse_args()
    globals()["MAXCTX"] = a.max_ctx
    dev = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"     # box: đo C1 trên GPU
    t = time.time()
    DT = torch.bfloat16 if a.half else torch.float32
    if a.half:                                   # ép bf16 trên CPU trước rồi mới lên thiết bị → không bao giờ giữ bản fp32 trên MPS
        m, c, step = load(a.ckpt, "cpu")
        m = m.to(DT).to(dev)
    else:
        m, c, step = load(a.ckpt, dev)
    npar = sum(p.numel() for p in m.parameters()) / 1e9
    kind = "SFT" if "sft" in os.path.basename(a.ckpt) or "chat" in os.path.basename(a.ckpt) else "nền"
    name = f"HybridLM {npar:.1f}B · {kind} bước {step}"
    # expert nằm SẴN trong checkpoint (mục "experts") — một checkpoint duy nhất; --adapters chỉ để thử file rời
    import torch.nn as nn
    from expert_ffn_1p3b import GatedFFN, POS, set_pos
    set_pos(c)                                   # điểm cắm expert theo thân (1.3B: 6 · 2,5B TK_107: 12)
    sets = []
    for nm_, e in torch.load(a.ckpt, map_location="cpu", mmap=True, weights_only=False).get("experts", {}).items():
        sets.append((nm_, e["state"], e.get("hid", a.hid), e.get("gated", True)))
    if a.adapters:
        sets.append((os.path.basename(a.adapters).split(".")[0], torch.load(a.adapters, map_location="cpu"), a.hid, True))
    ad = None
    if sets:
        per = {}
        if a.expert_mode == "off":
            sets = [x for x in sets if x[0] == "vision"]
        elif a.expert_mode == "open":
            sets = [(n_, sd, h_, False if n_ != "vision" else g_) for n_, sd, h_, g_ in sets]
        for nm_, sd, hid, gated in sets:
            sdf = {k: v.float() for k, v in sd.items()}
            pos_ = sorted({int(k.split(".")[0]) for k in sdf})                      # vị trí cắm + cỡ ẩn đọc từ trọng số
            md = nn.ModuleDict({str(i): GatedFFN(c["d"], sdf[f"{i}.w3.weight"].shape[1], exact=True) for i in pos_})
            md.load_state_dict(sdf)
            for b_ in md.values():
                b_.use_gate = gated              # cổng theo từng token: code → mở, trò chuyện → khép
            if nm_ == "vision":                      # công tắc: tắt expert thị giác khi hội thoại không có ảnh
                md = nn.ModuleDict({k: Toggle(v) for k, v in md.items()})       # (bọc SAU khi đặt use_gate)
                S["vexp"] = list(md.values())
            for k_ in md:
                per.setdefault(k_, []).append(md[k_])
            print(f"[expert] {nm_} ({len(md)} điểm cắm · hid {sdf[f'{pos_[0]}.w3.weight'].shape[1]}, "
                  f"{'có cổng' if gated else 'mở hẳn'})", flush=True)
        ad = nn.ModuleDict({k: nn.Sequential(*v) for k, v in per.items()}).to(dev, DT).eval()
        name += " + expert " + ", ".join(x[0] for x in sets)
    S.update(runner=Runner(m, dev, ad), dev=dev, name=name)
    size = f"{npar:.1f}B".replace(".", ",")
    tag = re.search(r"p3_([a-z0-9]+)_bf16", os.path.basename(a.ckpt))
    big = npar > 2
    S["ui"] = {"__SIZE__": size, "__TITLE__": f"aivie {size}" + (f" · {tag.group(1)}" if tag else ""),
               "__NOTE__": (f"Thân {size} tự train trên 20 tỷ token (tiếng Việt + code)"
                            + (" + expert " + ", ".join(x[0] for x in sets) if sets else "")
                            + " — viết code ngắn khá, đề thực tế nhiều ràng buộc còn yếu; có thể bịa kiến thức, hãy kiểm tra lại."
                            if big else f"Model {size} tự train trên 18,8 tỷ token tiếng Việt — viết trôi chảy nhưng hay bịa kiến thức. "
                            "Hãy kiểm tra lại thông tin quan trọng.")}
    obj_v = torch.load(a.ckpt, map_location="cpu", mmap=True, weights_only=False)
    if obj_v.get("code_lora"):                                 # TK_108 C8: LoRA khối trộn học cùng expert code (luôn bật)
        from expert_ffn_1p3b import load_code_lora
        load_code_lora(m, obj_v["code_lora"]["state"])
        S["name"] = name = name + f" + LoRA khối trộn r{obj_v['code_lora'].get('r', '?')}"
        print(f"[lora] CodeLoRA khối trộn: {len(obj_v['code_lora']['state']) // 2} lớp chiếu", flush=True)
    if obj_v.get("vision"):                                    # ĐỌC ẢNH: mắt SigLIP2 + bộ chiếu nằm sẵn trong checkpoint
        import vlm_1p3b as V
        S["vision"], S["max_tiles"] = V.vision_from_ckpt(obj_v, m, dev)
        S["name"] = name = name + " · đọc ảnh"
        print(f"[vision] mắt SigLIP2 + bộ chiếu (tối đa {S['max_tiles']} ô/ảnh)", flush=True)
    del obj_v
    tok = BPETokenizer.load("checkpoints/bpe_v1.json")
    S.update(tok=tok, vocab=Vocabulary(tok), eos=tok.token_to_id["<EOS>"])
    print(f"[model] {S['name']} trên {dev} ({time.time()-t:.0f}s)", flush=True)
    if a.no_rag:
        S["rag_err"] = "tắt bằng --no-rag"
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), H)
    print(f"[ui] http://localhost:{a.port}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
