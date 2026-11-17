"""UI chat cho model GIAO-TIẾP Việt–Anh (TK_104, train_gt.py) — cùng giao-thức với serve_hybrid_ui.py (model 1.3B
vẫn chạy riêng ở cổng 8790; server này mặc-định cổng 8791).

Định-dạng prompt đúng như lúc train (build_corpus.py):
  hội-thoại   "A: <bạn>\\nB: <model>\\nA: …\\nB:"   → dừng ở xuống dòng
  dịch        "Dịch sang tiếng Việt: <câu>\\n=>"  /  "Translate to English: <câu>\\n=>"
  viết tiếp   văn-bản thô (truyện)                  → dừng ở <eos>

    .venv/bin/python scripts/serve_gt_ui.py [--ckpt out/curriculum/gt/gt2_big.pt --port 8791]
"""
from __future__ import annotations
import argparse, json, os, sys, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "curriculum", "giaotiep"))
sys.path.insert(0, os.path.join(ROOT, "curriculum"))
from train_gt import GPT  # noqa: E402

CTX = 512                                                    # model chỉ được train với cửa-sổ 512 token
S, LOCK, STOP = {}, threading.Lock(), threading.Event()
PROMPT = {"dich_ev": "Dịch sang tiếng Việt: {q}\n=>", "dich_ve": "Translate to English: {q}\n=>"}


def load(ckpt, tok_path, dev):
    from tokenizers import Tokenizer
    ck = torch.load(ckpt, map_location="cpu")
    st, a = ck["state"], ck["args"]
    nl = 1 + max(int(k.split(".")[1]) for k in st if k.startswith("blocks."))
    m = GPT(st["emb.weight"].shape[0], a["d"], nl, a["heads"])
    m.load_state_dict(st)
    m.to(dev).eval()
    n = sum(p.numel() for p in m.parameters())
    return m, Tokenizer.from_file(tok_path), n, a


def build(mode, msgs, max_new):
    """→ (ids prompt, chuỗi dừng). Hội-thoại: bỏ bớt lượt cũ cho vừa cửa-sổ."""
    tok = S["tok"]
    if mode == "raw":
        ids = tok.encode(msgs[-1]["content"]).ids
        return ids[-(CTX - max_new):], None
    if mode in PROMPT:
        return tok.encode(PROMPT[mode].format(q=msgs[-1]["content"].strip())).ids[-(CTX - max_new):], "\n"
    turns = [f"{'A' if m['role'] == 'user' else 'B'}: {m['content'].strip()}" for m in msgs]
    while True:
        ids = tok.encode("\n".join(turns) + "\nB:").ids
        if len(ids) + max_new <= CTX or len(turns) <= 1:
            return ids[-(CTX - max_new):], "\n"
        turns = turns[1:]


@torch.no_grad()
def generate(ids, n, stop_str, temp, top_p, rep_pen):
    m, tok, dev, eos = S["m"], S["tok"], S["dev"], S["eos"]
    ids, out = list(ids), []
    g = torch.Generator(device="cpu").manual_seed(int(time.time()))
    for _ in range(n):
        if STOP.is_set():
            break
        lo = m(torch.tensor([ids[-CTX:]], device=dev))[0, -1].float().cpu()
        if rep_pen and out:                                  # trừ điểm token vừa dùng (64 token gần nhất)
            lo[torch.tensor(sorted(set(out[-64:])))] -= rep_pen
        if temp <= 0:
            nxt = int(lo.argmax())
        else:
            p = torch.softmax(lo / temp, -1)
            sp, si = torch.sort(p, descending=True)
            keep = torch.cumsum(sp, 0) - sp < top_p
            sp = sp * keep
            nxt = int(si[torch.multinomial(sp / sp.sum(), 1, generator=g)])
        if nxt == eos:
            break
        ids.append(nxt); out.append(nxt)
        text = tok.decode(out)
        if stop_str and stop_str in text:
            yield text.split(stop_str)[0], True
            return
        yield text, False


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
            b = open(os.path.join(HERE, "gt_ui.html"), "rb").read()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)
        elif self.path == "/api/info":
            self._json(200, {"model": S["name"], "device": S["dev"], "busy": LOCK.locked(), "rag": "off",
                             "detail": S["detail"]})
        else:
            self._json(404, {"error": "không có"})

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(n) or b"{}")
        if self.path == "/api/stop":
            STOP.set()
            return self._json(200, {"ok": True})
        if self.path == "/api/rag":
            return self._json(200, {"rag": "off"})
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
        msgs = [m for m in b.get("messages", []) if m.get("content", "").strip()]
        mode = b.get("mode", "chat")
        n = max(8, min(int(b.get("max_new", 80)), 400))
        try:
            if not msgs or msgs[-1]["role"] != "user":
                return self._send({"type": "error", "error": "Tin nhắn cuối phải là của người dùng."})
            ids, stop_str = build(mode, msgs, n)
            self._send({"type": "meta", "prefill": len(ids), "cached": False, "sources": [], "ctx": len(ids)})
            t0, ttft, text, k = time.time(), None, "", 0
            for text, _ in generate(ids, n, stop_str, float(b.get("temperature", 0.6)),
                                    float(b.get("top_p", 0.9)), float(b.get("rep_pen", 1.1))):
                ttft = ttft or (time.time() - t0)
                k += 1
                self._send({"type": "delta", "text": text.strip()})
            dt = time.time() - t0
            gen_t = dt - (ttft or dt)
            self._send({"type": "done", "text": text.strip(), "tokens": k,
                        "tok_s": round((k - 1) / gen_t, 1) if k > 1 and gen_t > 0 else None,
                        "ttft": round(ttft if ttft is not None else dt, 2), "stopped": STOP.is_set(), "cached": False})
        except (BrokenPipeError, ConnectionResetError):
            STOP.set()
        except Exception as e:
            try:
                self._send({"type": "error", "error": f"{type(e).__name__}: {e}"})
            except Exception:
                pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="out/curriculum/gt/gt2_big.pt")
    ap.add_argument("--tok", default="data/giaotiep/tok.json", help="v1 (gt_*.pt): data/giaotiep/v1/tok.json")
    ap.add_argument("--port", type=int, default=8791)
    a = ap.parse_args()
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    t = time.time()
    m, tok, npar, args = load(a.ckpt, a.tok, dev)
    S.update(m=m, tok=tok, dev=dev, eos=tok.token_to_id("<eos>"),
             name=f"Giao tiếp {npar/1e6:.0f}M ({os.path.basename(a.ckpt)[:-3]})",
             detail=f"d={args['d']} · {args.get('layers')} tầng · ngữ cảnh {CTX} token")
    print(f"[model] {S['name']} trên {dev} ({time.time()-t:.0f}s)", flush=True)
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), H)
    print(f"[ui] http://localhost:{a.port}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
