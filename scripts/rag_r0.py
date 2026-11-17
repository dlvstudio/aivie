"""TK_108 R0 — ĐO TRƯỚC khi làm RAG: B bịa sự kiện vì TÌM SAI hay vì ĐỌC SAI?

Bộ câu hỏi sự kiện SẠCH dựng từ viwiki (thầy đặt câu hỏi + đáp án ngắn từ một đoạn chọn ngẫu nhiên, lưu NGUYÊN VĂN đoạn gốc
→ không phụ thuộc thứ tự bài trong kho, chạy được cả Mac lẫn box):
  single   1 ý nằm trong 1 câu của đoạn
  extract  trích nhiều ý / đếm / liệt kê trong đoạn (lỗi cũ của 1.3B: "Mê Kông chảy qua 3 nước")
  multi    cần ghép 2 đoạn xa nhau của CÙNG bài
  fake     thực thể KHÔNG có thật (thầy bịa, đã kiểm không có tiêu đề nào trong kho khớp) → đúng = nói không biết
Đo 3 cách cho mỗi câu: none (hỏi chay) · rag (BM25F top-k, đúng khuôn Rag.augment của UI) · oracle (đoạn gốc + 2 đoạn nhiễu)
→ rag ≈ oracle: tìm đủ tốt, lỗi ở ĐỌC · oracle ≫ rag: lỗi ở TÌM · oracle thấp: model chưa biết dùng nguồn.

  python scripts/rag_r0.py corpus                      # (box) dựng data/text_corpus/viwiki.jsonl từ HF wikimedia/wikipedia 20231101.vi
  python scripts/rag_r0.py build --llm http://127.0.0.1:8000/v1 --model gemma4-31b
  python scripts/rag_r0.py eval --url http://localhost:8891,http://localhost:8892 --tag c6b --modes none,rag,oracle
  python scripts/rag_r0.py judge --tag c6b --llm http://127.0.0.1:8000/v1
  python scripts/rag_r0.py report --tags c6b,c6l1
Thử nhanh trên Mac: build --n 8 --llm http://127.0.0.1:11434/v1 --model <ollama> · eval --url http://localhost:8804 --limit 4
"""
from __future__ import annotations
import argparse, json, os, random, re, sys, time, urllib.error, urllib.request
from concurrent.futures import ThreadPoolExecutor
sys.path[:0] = [os.path.dirname(os.path.abspath(__file__))]

OUT = "data/rag"
EVAL = f"{OUT}/r0_eval.jsonl"
RES = "out/rag_r0"
KINDS = {"single": 0.4, "extract": 0.25, "multi": 0.2, "fake": 0.15}


# ── tiện ích ──────────────────────────────────────────────────────────────────────────────
def teacher(llm, model, prompt, temp=0.7, mt=800):
    body = {"model": model, "temperature": temp, "max_tokens": mt, "messages": [{"role": "user", "content": prompt}],
            "response_format": {"type": "json_object"}, "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(f"{llm.rstrip('/')}/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    txt = json.loads(urllib.request.urlopen(req, timeout=600).read())["choices"][0]["message"]["content"]
    m = re.search(r"\{.*\}", txt, re.S)
    return json.loads(m.group(0)) if m else {}


def rows(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()] if os.path.exists(p) else []


class Kho:
    """BM25F viwiki (retrieval_v2) + cửa sổ tập trung giống Rag.focus của serve_hybrid_ui."""
    def __init__(self, n=285000):
        from retrieval_v2 import load_corpus_v2, tok, STOP, extract_title
        self.tok, self.SW, self.title = tok, STOP, extract_title
        self.bm, self.snips = load_corpus_v2(n)

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

    def search(self, q, k=3):
        return [{"title": self.title(self.snips[i]), "text": self.focus(self.snips[i], q)} for i in self.bm.top(q, k=k)]


def rag_prompt(q, src):                     # GIỮ Y KHUÔN Rag.augment (serve_hybrid_ui) — đo đúng cái người dùng nhận
    ctx = "\n".join(f"[Nguồn {j+1} — {s['title']}] {s['text']}" for j, s in enumerate(src))
    return (f"Dựa vào các nguồn dưới đây, trả lời ngắn gọn bằng tiếng Việt. "
            f"Nếu nguồn không có, nói không tìm thấy.\n{ctx}\nCâu hỏi: {q}")


# ── corpus (box) ──────────────────────────────────────────────────────────────────────────
def cmd_corpus(a):
    from datasets import load_dataset
    os.makedirs("data/text_corpus", exist_ok=True)
    ds = load_dataset("wikimedia/wikipedia", "20231101.vi", split="train", streaming=True)
    n = 0
    with open("data/text_corpus/viwiki.jsonl", "w", encoding="utf-8") as f:
        for r in ds:
            t = r["text"] if r["text"].startswith(r["title"]) else r["title"] + ": " + r["text"]   # extract_title tách ở ':'
            if len(t) < 200:
                continue
            f.write(json.dumps(t[:6000], ensure_ascii=False) + "\n"); n += 1
            if n >= a.n_docs:
                break
    print(f"### viwiki {n} bài → data/text_corpus/viwiki.jsonl", flush=True)


# ── build ─────────────────────────────────────────────────────────────────────────────────
P_Q = """Bạn soạn câu hỏi KIỂM TRA KIẾN THỨC SỰ KIỆN cho một trợ lý AI tiếng Việt. Dưới đây là {what} trích từ Wikipedia
tiếng Việt (bài "{title}").
{body}
Yêu cầu: {task}
- Câu hỏi tự nhiên như người dùng thật hỏi, ĐỦ NGỮ CẢNH để hiểu khi không có đoạn văn (nêu rõ tên thực thể; KHÔNG viết
  "theo đoạn văn", "bài viết này"). Một câu hỏi, có đáp án DUY NHẤT, kiểm chứng được từ văn bản trên.
- "dap_an": ngắn gọn (tên, số, năm, danh sách…), đúng tuyệt đối theo văn bản. "y_chinh": các ý bắt buộc phải có để tính đúng.
- Nếu văn bản không đủ để đặt câu hỏi như vậy (danh sách rời rạc, trang định hướng…), trả "ok": false.
Trả về DUY NHẤT JSON: {{"ok": true, "cau_hoi": "...", "dap_an": "...", "y_chinh": ["..."]}}"""
TASK = {"single": "hỏi MỘT sự kiện cụ thể (ai, khi nào, ở đâu, bao nhiêu…) nằm trong một câu của đoạn.",
        "extract": "hỏi cần TRÍCH NHIỀU Ý hoặc ĐẾM/LIỆT KÊ (vd những nước nào, bao nhiêu thành phần, kể tên các…) — đáp án "
                   "gồm ≥ 2 mục lấy từ đoạn.",
        "multi": "hỏi cần GHÉP thông tin từ CẢ HAI đoạn (mỗi đoạn góp một phần đáp án)."}
P_FAKE = """Hãy BỊA một thực thể KHÔNG có thật nhưng nghe rất hợp lý trong bối cảnh Việt Nam hoặc thế giới — loại: {loai}.
Đặt một câu hỏi sự kiện về nó như người dùng thật hỏi (vd năm thành lập, ai sáng lập, ở đâu, nổi tiếng vì gì).
Tên phải ĐỦ CỤ THỂ để không trùng thực thể thật nổi tiếng nào. Trả về DUY NHẤT JSON:
{{"ten": "tên thực thể bịa", "cau_hoi": "..."}}"""
FAKE_TYPES = ["nhà khoa học Việt Nam", "vị vua/chúa thời phong kiến", "xã/huyện/thị trấn", "con sông hoặc ngọn núi",
              "công ty công nghệ", "trận đánh lịch sử", "nhà văn/nhà thơ và tác phẩm", "loài động vật/thực vật",
              "giải thưởng/cuộc thi", "món ăn đặc sản địa phương", "hiệp ước/hội nghị quốc tế", "cầu thủ bóng đá"]


BAD_Q = re.compile(r"đoạn (văn|trích|trên|này)|bài viết|văn bản|theo (đoạn|bài)|được nhắc|nêu trên", re.I)


def ok_q(q):
    """một câu hỏi thật: có dấu hỏi, không dài, không trỏ vào "đoạn văn" (người dùng thật không có đoạn văn)."""
    q = q.strip()
    return 10 < len(q) < 250 and q.count("?") == 1 and q.endswith("?") and not BAD_Q.search(q)


def window(text, rng, n=900):
    if len(text) <= n:
        return text
    s = rng.randrange(0, len(text) - n)
    s = text.rfind(". ", 0, s) + 2 if text.rfind(". ", 0, s) > 0 else 0
    return text[s:s + n]


def cmd_build(a):
    kho = Kho(a.max_docs)
    rng = random.Random(a.seed)
    os.makedirs(OUT, exist_ok=True)
    want = {k: max(1, round(a.n * w)) for k, w in KINDS.items()}
    good = [i for i, t in enumerate(kho.snips) if len(t) >= 1800 and "có thể là" not in t[:80].lower()]
    print(f"[r0] kho {len(kho.snips)} bài · {len(good)} bài đủ dài · cần {want}", flush=True)

    def job(kind, k):
        r = random.Random(a.seed * 100000 + list(KINDS).index(kind) * 10000 + k)   # hash() đổi theo tiến trình → không tái lập
        try:
            if kind == "fake":
                j = teacher(a.llm, a.model, P_FAKE.format(loai=r.choice(FAKE_TYPES)), temp=1.0)
                ten, q = (j.get("ten") or "").strip(), (j.get("cau_hoi") or "").strip()
                if not ten or not ok_q(q):
                    return None
                hits = [kho.title(kho.snips[i]).lower() for i in kho.bm.top(ten, k=5)]
                if any(ten.lower() in h for h in hits):
                    return None                                        # tên bịa trùng tiêu đề có thật → bỏ
                return {"kind": kind, "q": q, "gold": "", "y_chinh": [], "entity": ten, "passages": []}
            i = r.choice(good)
            text, title = kho.snips[i], kho.title(kho.snips[i])
            if kind == "multi":
                h = len(text) // 2
                p1, p2 = window(text[:h], r, 700), window(text[h:], r, 700)
                body, what, ps = f"[Đoạn 1] {p1}\n[Đoạn 2] {p2}", "HAI đoạn", [p1, p2]
            else:
                p = window(text, r)
                body, what, ps = p, "một đoạn", [p]
            j = teacher(a.llm, a.model, P_Q.format(what=what, title=title, body=body, task=TASK[kind]))
            if not j.get("ok") or not ok_q(j.get("cau_hoi") or "") or not j.get("dap_an"):
                return None
            return {"kind": kind, "q": j["cau_hoi"].strip(), "gold": str(j["dap_an"]).strip(),
                    "y_chinh": [str(x) for x in j.get("y_chinh", [])], "title": title, "passages": ps}
        except Exception as e:
            print(f"[r0] lỗi {kind}: {type(e).__name__} {str(e)[:100]}", flush=True)
            return None

    out = []
    with ThreadPoolExecutor(a.workers) as ex:
        for kind, n in want.items():
            got, k = [], 0
            while len(got) < n and k < n * 3:                          # thầy trả ok=false / trùng → đặt thêm
                batch = list(ex.map(lambda kk: job(kind, kk), range(k, k + (n - len(got)) + 2)))
                k += (n - len(got)) + 2
                got += [x for x in batch if x][: n - len(got)]
            out += got
            print(f"[r0] {kind}: {len(got)}/{n}", flush=True)
    for j, x in enumerate(out):
        x["id"] = f"r0_{x['kind']}_{j}"
    with open(a.out or EVAL, "w", encoding="utf-8") as f:
        for x in out:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    print(f"### r0 build {len(out)} câu → {a.out or EVAL}", flush=True)


# ── eval ──────────────────────────────────────────────────────────────────────────────────
def ask(url, prompt, max_new):
    """có máy chủ bận (429 — máy chủ của ta sinh từng yêu cầu một, vừa gửi 'done' vẫn còn giữ khoá) → chờ rồi gửi lại."""
    for _ in range(200):
        try:
            return ask1(url, prompt, max_new)
        except urllib.error.HTTPError as e:
            if e.code != 429:
                raise
            time.sleep(3)
    raise RuntimeError("máy chủ bận quá lâu (429)")


def ask1(url, prompt, max_new):
    """máy chủ của ta (/api/chat, đọc luồng SSE tới 'done') hoặc kiểu OpenAI (/v1)."""
    if url.rstrip("/").endswith("/v1"):
        body = {"model": "m", "temperature": 0, "max_tokens": max_new, "messages": [{"role": "user", "content": prompt}]}
        req = urllib.request.Request(url.rstrip("/") + "/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        return json.loads(urllib.request.urlopen(req, timeout=900).read())["choices"][0]["message"]["content"]
    body = {"conv_id": f"r0-{time.time_ns()}", "messages": [{"role": "user", "content": prompt}], "temperature": 0.0,
            "top_p": 1.0, "rep_pen": 1.1, "max_new": max_new, "rag": False}
    req = urllib.request.Request(url.rstrip("/") + "/api/chat", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    out = ""
    with urllib.request.urlopen(req, timeout=1800) as r:
        for line in r:
            line = line.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            ev = json.loads(line[5:])
            if ev.get("type") == "error":
                raise RuntimeError(ev.get("error"))
            if ev.get("type") == "done":
                return ev.get("text", "")
            if ev.get("type") == "delta":
                out = ev.get("text", out)
    return out


def cmd_eval(a):
    qs = rows(a.eval)[: a.limit or None]
    modes = a.modes.split(",")
    kho = Kho(a.max_docs) if {"rag", "oracle"} & set(modes) else None
    os.makedirs(RES, exist_ok=True)
    urls = a.url.split(",")
    for mode in modes:
        path = f"{RES}/{a.tag}_{mode}.jsonl"
        done = {x["id"] for x in rows(path)}
        todo = [x for x in qs if x["id"] not in done]
        print(f"[r0] {a.tag} · {mode}: {len(todo)} câu · {len(urls)} máy chủ", flush=True)

        def one(k, x):
            src, hit = [], None
            if mode == "none":
                prompt = x["q"]
            else:
                ret = kho.search(x["q"], k=a.k)
                hit = bool(x.get("title")) and any(s["title"] == x["title"] for s in ret)
                if mode == "rag":
                    src = ret
                else:                                                  # oracle: đoạn gốc + 2 đoạn nhiễu (khác bài gốc)
                    noise = [s for s in ret if s["title"] != x.get("title")][:2]
                    gold = [{"title": x.get("title", ""), "text": p} for p in x["passages"]]
                    src = gold + noise
                    random.Random(k).shuffle(src)
                prompt = rag_prompt(x["q"], src)
            t = time.time()
            try:
                ans = ask(urls[k % len(urls)], prompt, a.max_new)
            except Exception as e:
                ans = f"[LỖI {type(e).__name__}: {str(e)[:120]}]"
            return {"id": x["id"], "kind": x["kind"], "mode": mode, "q": x["q"], "answer": ans.strip(),
                    "retrieved": [s["title"] for s in src] if mode == "rag" else None, "hit": hit, "sec": round(time.time() - t, 1)}

        with open(path, "a", encoding="utf-8") as f, ThreadPoolExecutor(len(urls)) as ex:
            for j, r in enumerate(ex.map(lambda kx: one(*kx), enumerate(todo))):
                f.write(json.dumps(r, ensure_ascii=False) + "\n"); f.flush()
                if (j + 1) % 25 == 0:
                    print(f"[r0] {mode} {j+1}/{len(todo)}", flush=True)


# ── judge ─────────────────────────────────────────────────────────────────────────────────
P_J = """Chấm câu trả lời của một trợ lý AI cho câu hỏi SỰ KIỆN, so với ĐÁP ÁN CHUẨN.
Câu hỏi: {q}
Đáp án chuẩn: {gold}
Ý bắt buộc: {y}
Câu trả lời của trợ lý: {ans}
"diem": 2 = đúng và đủ mọi ý bắt buộc (diễn đạt khác vẫn tính) · 1 = đúng một phần (thiếu ý / thừa ý sai nhỏ) ·
0 = sai, lạc đề, hoặc nói không biết. "bia": true CHỈ KHI câu trả lời khẳng định một thông tin SAI cụ thể (tên, số, năm…);
nói không biết / không tìm thấy / từ chối đoán → "bia": false.
Trả về DUY NHẤT JSON: {{"diem": 0|1|2, "bia": true|false, "ly_do": "một câu ngắn"}}"""
P_JF = """"{ten}" là thực thể KHÔNG có thật. Người dùng hỏi: {q}
Câu trả lời của trợ lý: {ans}
"nhan": true nếu trợ lý nói rõ là không biết / không tìm thấy thông tin / thực thể có thể không tồn tại (và KHÔNG bịa chi
tiết); false nếu trợ lý bịa thông tin như thể thực thể có thật (kể cả khi kèm câu rào đón).
Trả về DUY NHẤT JSON: {{"nhan": true|false, "ly_do": "một câu ngắn"}}"""


def cmd_judge(a):
    qs = {x["id"]: x for x in rows(a.eval)}
    for mode in a.modes.split(","):
        path = f"{RES}/{a.tag}_{mode}.jsonl"
        res = rows(path)
        todo = [r for r in res if "diem" not in r and "nhan" not in r]
        if not todo:
            continue

        def one(r):
            x = qs[r["id"]]
            try:
                if x["kind"] == "fake":
                    j = teacher(a.llm, a.model, P_JF.format(ten=x["entity"], q=x["q"], ans=r["answer"][:2000]), temp=0)
                    r["nhan"] = bool(j.get("nhan"))
                else:
                    j = teacher(a.llm, a.model, P_J.format(q=x["q"], gold=x["gold"], y="; ".join(x["y_chinh"]) or x["gold"],
                                                       ans=r["answer"][:2000]), temp=0)
                    r["diem"], r["bia"] = int(j.get("diem", 0)), bool(j.get("bia"))
                r["ly_do"] = j.get("ly_do", "")
            except Exception as e:
                print(f"[r0] lỗi chấm {r['id']}: {type(e).__name__}", flush=True)
            return r

        with ThreadPoolExecutor(a.workers) as ex:
            res = list(ex.map(lambda r: one(r) if ("diem" not in r and "nhan" not in r) else r, res))
        with open(path, "w", encoding="utf-8") as f:
            for r in res:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"[r0] chấm {a.tag} · {mode}: {len(todo)} câu", flush=True)


def cmd_report(a):
    print(f"{'bản · cách':22} {'single':>9} {'extract':>9} {'multi':>9} {'fake:nhận':>10} {'bịa':>6} {'tìm trúng':>10}")
    for tag in a.tags.split(","):
        for mode in a.modes.split(","):
            res = rows(f"{RES}/{tag}_{mode}.jsonl")
            if not res:
                continue
            cell = []
            for k in ("single", "extract", "multi"):
                rr = [r for r in res if r["kind"] == k and "diem" in r]
                cell.append(f"{100 * sum(r['diem'] for r in rr) / (2 * len(rr)):.0f}%" if rr else "–")
            ff = [r for r in res if r["kind"] == "fake" and "nhan" in r]
            nf = [r for r in res if r["kind"] != "fake" and "diem" in r]
            hits = [r["hit"] for r in res if r.get("hit") is not None]
            pct = lambda xs: f"{100 * sum(xs) / len(xs):.0f}%" if xs else "–"
            print(f"{tag + ' · ' + mode:22} {cell[0]:>9} {cell[1]:>9} {cell[2]:>9} {pct([r['nhan'] for r in ff]):>10} "
                  f"{pct([r['bia'] for r in nf]):>6} {(pct(hits) if mode == 'rag' else '–'):>10}")
    print("điểm = (2·đúng + 1·một phần) / (2·số câu) · fake:nhận = tỉ lệ nói không biết với thực thể bịa · bịa = tỉ lệ khẳng định "
          "thông tin SAI · tìm trúng = bài gốc có trong top-k BM25")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["corpus", "build", "eval", "judge", "report"])
    ap.add_argument("--llm", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--model", default="gemma4-31b")
    ap.add_argument("--n", type=int, default=350, help="build: tổng số câu (chia theo KINDS)")
    ap.add_argument("--n-docs", type=int, default=290000, help="corpus: số bài viwiki")
    ap.add_argument("--max-docs", type=int, default=285000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out", default="")
    ap.add_argument("--eval", default=EVAL)
    ap.add_argument("--url", default="http://localhost:8804", help="eval: một hay nhiều máy chủ, phân tách bằng dấu phẩy")
    ap.add_argument("--tag", default="c6b"); ap.add_argument("--tags", default="c6b")
    ap.add_argument("--modes", default="none,rag,oracle")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--max-new", type=int, default=300)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    {"corpus": cmd_corpus, "build": cmd_build, "eval": cmd_eval, "judge": cmd_judge, "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    main()
