"""TK_105 · UI chat cho MẠNG GHÉP 1.3B + Zamba2-7B (chạy trên GPU; mở từ Mac qua đường hầm SSH).

Mỗi câu hỏi (một lượt, tiếng Việt):
  1) Zamba2-7B đọc câu Việt thô → tự viết NHÁP tiếng Anh (tham lam, có cache)
  2) ẩn Zamba2 trên [câu hỏi + nháp] → 1.3B đáp TIẾNG VIỆT qua cầu dịch-ra (cross-attention gđ3b), phát từng chữ
Dùng lại giao diện scripts/hybrid_ui.html (cùng giao thức /api/info, /api/chat SSE); nháp hiện trong ô "Nháp của Zamba2".

    /workspace/zenv/bin/python scripts/serve_fused_ui.py --xattn ckpt/tk105/stage3b.pt --port 8800
    (Mac) ssh -N -L 8800:localhost:8800 -p 16500 root@<box>   → http://localhost:8800
"""
from __future__ import annotations
import argparse, json, os, sys, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(os.path.dirname(HERE), "src")]
import torch
import tk105_stage3 as S3

SUF = ("\nAnswer the request above in English. If it needs calculation, think step by step and end with the final answer. "
       "If you do not know, or the thing does not exist, say you do not know.<|im_end|>\n<|im_start|>assistant\n")
LOCK, STOP = threading.Lock(), threading.Event()
ST = {}


def html():
    h = open(os.path.join(HERE, "hybrid_ui.html"), encoding="utf-8").read()
    h = h.replace("Nguồn viwiki (${m.sources.length})",
                  "${m.sources[0].kind === 'draft' ? 'Nháp của Zamba2-7B (tiếng Anh, bên trong mạng)' : 'Nguồn viwiki (' + m.sources.length + ')'}")
    h = h.replace("s.text.slice(0, 300)", "s.text.slice(0, s.kind === 'draft' ? 4000 : 300)")
    h = h.replace("Model 1.3B tự train trên 18,8 tỷ token tiếng Việt — viết trôi chảy nhưng hay bịa kiến thức.",
                  "MẠNG GHÉP: Zamba2-7B nghĩ nháp (tiếng Anh) → 1.3B của ta nói tiếng Việt qua cầu nối. Mỗi lượt hỏi độc lập.")
    return h.encode()


@torch.no_grad()
def draft(P, q, n):
    ids = P.pre + P.ztok(q, add_special_tokens=False).input_ids[:256] + P.ztok(SUF, add_special_tokens=False).input_ids
    x = torch.tensor([ids], device=P.dev)
    y = P.z.generate(x, max_new_tokens=n, do_sample=False, repetition_penalty=1.05)
    return P.ztok.decode(y[0, x.shape[1]:], skip_special_tokens=True).strip()


@torch.no_grad()
def speak(P, q, H, M, n, temp, rep_pen):
    """1.3B sinh từng token (không cache, ~0,03 giây/bước trên H100) — yield chuỗi đã giải mã."""
    s = P.prompt_ids(q)
    p0 = len(s)
    for _ in range(n):
        if STOP.is_set():
            break
        X = torch.tensor([s], device=P.dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            l_ = P.fwd(X, H, M)[0, -1].float()
        rec = torch.tensor(sorted(set(s[max(p0, len(s) - 64):])), dtype=torch.long, device=l_.device)
        if len(rec) and rep_pen > 1:
            v = l_[rec]; l_[rec] = torch.where(v > 0, v / rep_pen, v * rep_pen)
        t = int(torch.multinomial(torch.softmax(l_ / temp, -1), 1)) if temp > 0 else int(l_.argmax())
        if t == P.eos:
            break
        s.append(t)
        yield P.voc.detokenize(s[p0:])


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _json(self, code, obj):
        b = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code); self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)

    def _send(self, obj):
        self.wfile.write(f"data: {json.dumps(obj, ensure_ascii=False)}\n\n".encode()); self.wfile.flush()

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            b = html()
            self.send_response(200); self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)
        elif self.path == "/api/info":
            self._json(200, {"model": "MẠNG GHÉP · Zamba2-7B nghĩ → 1.3B nói (TK_105 gđ3b)", "device": "H100",
                             "busy": LOCK.locked(), "rag": "error", "rag_err": "mạng ghép không dùng viwiki"})
        else:
            self._json(404, {"error": "không có"})

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        b = json.loads(self.rfile.read(n) or b"{}")
        if self.path == "/api/stop":
            STOP.set(); return self._json(200, {"ok": True})
        if self.path == "/api/rag":
            return self._json(200, {"rag": "error"})
        if self.path != "/api/chat":
            return self._json(404, {"error": "không có"})
        if not LOCK.acquire(blocking=False):
            return self._json(429, {"error": "Mạng ghép đang trả lời câu khác — chờ xong hoặc bấm Dừng."})
        try:
            self._chat(b)
        finally:
            LOCK.release()

    def _chat(self, b):
        STOP.clear()
        self.send_response(200); self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache"); self.end_headers()
        P = ST["P"]
        msgs = [m for m in b.get("messages", []) if m.get("content", "").strip()]
        if not msgs or msgs[-1]["role"] != "user":
            return self._send({"type": "error", "error": "Tin nhắn cuối phải là của người dùng."})
        q = msgs[-1]["content"].strip()
        try:
            t0 = time.time()
            d = draft(P, q, ST["n_draft"])
            emb, mask, _ = P.zin([q], P.suf_qa_ui, targets=[d])
            Hs, M = P.zstate(emb, mask)
            self._send({"type": "meta", "prefill": int(mask.sum()), "cached": False, "ctx": int(mask.sum()),
                        "sources": [{"kind": "draft", "title": f"Zamba2 nghĩ trong {time.time()-t0:.1f}s", "text": d}]})
            t1, ttft, text, k = time.time(), None, "", 0
            n = max(1, min(int(b.get("max_new", 400)), 800))
            for text in speak(P, q, Hs, M, n, float(b.get("temperature", 0.0)) if b.get("temperature", 0) < 0.35 else 0.0,
                              ST["rep_pen"]):
                k += 1
                ttft = ttft or (time.time() - t0)
                self._send({"type": "delta", "text": text})
            dt = time.time() - t1
            self._send({"type": "done", "text": text.strip(), "tokens": k, "tok_s": round(k / dt, 1) if dt > 0 else None,
                        "ttft": round(ttft or 0, 1), "stopped": STOP.is_set(), "cached": False})
        except (BrokenPipeError, ConnectionResetError):
            STOP.set()
        except Exception as e:
            self._send({"type": "error", "error": f"{type(e).__name__}: {e}"})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="ckpt/hybrid_1p3b_chat_v2.pt")
    ap.add_argument("--layer", type=int, default=23)
    ap.add_argument("--xattn", default="ckpt/tk105/stage3b.pt")
    ap.add_argument("--bridge-in", default="")
    ap.add_argument("--port", type=int, default=8800)
    ap.add_argument("--n-draft", type=int, default=384)
    ap.add_argument("--rep-pen", type=float, default=1.15)
    a = ap.parse_args()
    t = time.time()
    P = S3.P3(a)
    P.rep_pen = a.rep_pen
    P.suf_qa_ui = P.ztok(SUF, add_special_tokens=False).input_ids
    ST.update(P=P, n_draft=a.n_draft, rep_pen=a.rep_pen)
    print(f"[mạng ghép] nạp xong {time.time()-t:.0f}s · http://localhost:{a.port}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", a.port), H).serve_forever()


if __name__ == "__main__":
    main()
