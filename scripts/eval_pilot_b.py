"""§TK_103 · ĐÁNH-GIÁ PILOT PHƯƠNG ÁN B.

  gen   : sinh câu trả lời cho held-out (data/pilot_b/heldout.jsonl, KHÔNG có trong train) + bộ 7 câu
          chuẩn. Chạy TRÊN BOX GPU ngay sau train (student B2 viết EN+VI ~450 token: M4 ~90s/câu).
            --adapter ckpt/zamba2_vi_lora --tag student   (tách phần <vi>…</vi>)
            --tag base                                    (Zamba2 gốc hỏi thẳng tiếng Việt — mốc dưới)
  judge : trên Mac, Gemma 4 26B (Ollama) chấm từng câu 1-5: độ tự-nhiên tiếng Việt + đúng/hữu-ích.
          Lưu ý: thầy (Gemma) cũng là giám-khảo → có thể thiên-vị văn giống mình; chỉ dùng để SO SÁNH.

    PYTHONPATH=scripts python scripts/eval_pilot_b.py gen --adapter ckpt/zamba2_vi_lora --tag student
    PYTHONPATH=scripts python scripts/eval_pilot_b.py gen --tag base
    .venv/bin/python scripts/eval_pilot_b.py judge out/pilot_b_gen_base.jsonl out/pilot_b_gen_student.jsonl
"""
from __future__ import annotations
import argparse, json, os, re, statistics, sys, time, urllib.request

sys.path.insert(0, "scripts")
SUITE = [
    "Sao Hỏa có mấy vệ tinh? Tên là gì?",
    "Sông Mê Kông chảy qua những nước nào?",
    "Truyện Kiều do ai viết?",
    "Viết email xin nghỉ phép 2 ngày gửi sếp, lý do việc gia đình, giọng lịch sự.",
    "Tóm tắt biên bản họp thành 3 việc cần làm, có người phụ trách: 'Linh sẽ chốt ngân sách Q3 "
    "trước thứ Sáu. Minh cần sửa lỗi đăng nhập trước khi phát hành. Mọi người nên đọc lại tài liệu "
    "onboarding mới. Dự kiến phát hành ngày 15.'",
    "Viết hàm Python tính số Fibonacci thứ n bằng đệ quy có memoization.",
    "Dịch sang tiếng Việt: 'Artificial intelligence is transforming how businesses operate.'",
]
_VI = re.compile(r"<vi>\s*(.*?)\s*(?:</vi>|$)", re.S)
_EN = re.compile(r"<en>\s*(.*?)\s*(?:</en>|$)", re.S)


def questions():
    qs = [{"id": f"suite_{i}", "source": "suite", "q": q} for i, q in enumerate(SUITE)]
    for line in open("data/pilot_b/heldout.jsonl", encoding="utf-8"):
        r = json.loads(line)
        qs.append({"id": r["id"], "source": r["source"], "q": r["q"]})
    return qs


def cmd_gen(a):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from test_zamba2 import patch_zamba2_block_id, MODEL
    patch_zamba2_block_id()
    dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    dt = torch.bfloat16 if dev == "cuda" else (torch.float16 if dev == "mps" else torch.float32)
    tok = AutoTokenizer.from_pretrained(MODEL)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    kw = {"use_kernels": True} if a.use_kernels else {}
    m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=dt, **kw).to(dev)
    if a.adapter:
        from peft import PeftModel
        m = PeftModel.from_pretrained(m, a.adapter).merge_and_unload()
    m.eval()
    bs = a.bs or (8 if dev == "cuda" else 1)
    qs = questions()
    os.makedirs("out", exist_ok=True)
    path = f"out/pilot_b_gen_{a.tag}.jsonl"
    T0 = time.time()
    with open(path, "w", encoding="utf-8") as f:
        for i in range(0, len(qs), bs):
            chunk = qs[i:i + bs]
            texts = [tok.apply_chat_template([{"role": "user", "content": q["q"]}],
                                             add_generation_prompt=True, tokenize=False) for q in chunk]
            enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(dev)
            t = time.time()
            with torch.no_grad():
                out = m.generate(**enc, max_new_tokens=a.max_new, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
            dt_ = (time.time() - t) / len(chunk)
            for q, o in zip(chunk, out):
                raw = tok.decode(o[enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
                mv, me = _VI.search(raw), _EN.search(raw)
                vi = mv.group(1).strip() if (a.adapter and mv) else raw
                f.write(json.dumps({**q, "tag": a.tag, "raw": raw, "vi": vi,
                                    "en": me.group(1).strip() if me else "",
                                    "format_ok": bool(mv) if a.adapter else None,
                                    "sec": round(dt_, 1)}, ensure_ascii=False) + "\n")
            f.flush()
            print(f"[{time.time()-T0:.0f}s] {min(i+bs, len(qs))}/{len(qs)}", flush=True)
    print(f"### LƯU {path}", flush=True)


def cmd_gen_http(a):
    """Model 1.3B của ta qua server UI đang chạy (serve_hybrid_ui.py) — không nạp model lần hai.
    Mỗi câu một conv_id mới (không mang lịch-sử); nhiệt-độ 0 như các model khác trong bảng."""
    qs = questions()
    os.makedirs("out", exist_ok=True)
    path = f"out/pilot_b_gen_{a.tag}.jsonl"
    T0 = time.time()
    with open(path, "w", encoding="utf-8") as f:
        for i, q in enumerate(qs, 1):
            body = {"conv_id": f"eval-{a.tag}-{q['id']}", "messages": [{"role": "user", "content": q["q"]}],
                    "temperature": 0, "top_p": 1.0, "rep_pen": a.rep_pen, "max_new": a.max_new, "rag": a.rag}
            for attempt in range(60):                           # 429 = UI đang bận câu khác → chờ
                try:
                    req = urllib.request.Request(f"{a.url}/api/chat", data=json.dumps(body).encode(),
                                                 headers={"Content-Type": "application/json"})
                    done, t = None, time.time()
                    with urllib.request.urlopen(req, timeout=900) as res:
                        for line in res:
                            line = line.decode().strip()
                            if line.startswith("data: "):
                                ev = json.loads(line[6:])
                                if ev["type"] in ("done", "error"):
                                    done = ev
                    break
                except urllib.error.HTTPError as e:
                    if e.code != 429:
                        raise
                    time.sleep(10)
            ans = (done or {}).get("text", "") if (done or {}).get("type") == "done" else ""
            f.write(json.dumps({**q, "tag": a.tag, "raw": ans, "vi": ans, "en": "", "format_ok": None,
                                "sec": round(time.time() - t, 1), "tok_s": (done or {}).get("tok_s"),
                                "err": (done or {}).get("error")}, ensure_ascii=False) + "\n")
            f.flush()
            if i % 10 == 0 or i == len(qs):
                print(f"[{time.time()-T0:.0f}s] {i}/{len(qs)}", flush=True)
    print(f"### LƯU {path}", flush=True)


JUDGE = """You are grading an AI assistant's answer to a user question written in Vietnamese.

Question:
{q}

Answer:
{ans}

Score two things from 1 to 5:
- "vi": how natural and fluent the Vietnamese is (5 = reads like a native speaker wrote it; 1 = not Vietnamese or unreadable).
- "ok": how correct and helpful the answer is for the question (5 = fully correct and useful; 1 = wrong, irrelevant or empty).
Return ONLY a JSON object like {{"vi": 4, "ok": 3, "why": "one short sentence"}}."""


def ollama_json(prompt, model="gemma4:26b-a4b-it-qat"):
    body = {"model": model, "stream": False, "format": "json",
            "options": {"temperature": 0}, "messages": [{"role": "user", "content": prompt}]}
    if model.startswith("gemma4"):
        body["think"] = False                    # Gemma 4 mặc-định 'suy nghĩ' ~30s/lượt
    return json.loads(ollama_chat(body)["message"]["content"])


def ollama_chat(body):
    req = urllib.request.Request("http://127.0.0.1:11434/api/chat", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=900).read())


def cmd_gen_ollama(a):
    """Model trong Ollama trả lời THẲNG (mốc trên, vd Gemma 26B) — cùng câu hỏi, cùng định-dạng file."""
    qs = questions()
    os.makedirs("out", exist_ok=True)
    path = f"out/pilot_b_gen_{a.tag}.jsonl"
    T0 = time.time()
    with open(path, "w", encoding="utf-8") as f:
        for i, q in enumerate(qs, 1):
            body = {"model": a.model, "stream": False, "options": {"temperature": 0, "num_predict": a.max_new},
                    "messages": [{"role": "user", "content": q["q"]}]}
            if a.model.startswith("gemma4"):
                body["think"] = False
            t = time.time()
            ans = ollama_chat(body)["message"]["content"].strip()
            f.write(json.dumps({**q, "tag": a.tag, "raw": ans, "vi": ans, "en": "", "format_ok": None,
                                "sec": round(time.time() - t, 1)}, ensure_ascii=False) + "\n")
            f.flush()
            if i % 10 == 0 or i == len(qs):
                print(f"[{time.time()-T0:.0f}s] {i}/{len(qs)}", flush=True)
    print(f"### LƯU {path}", flush=True)


def cmd_judge(a):
    for path in a.files:
        rows = [json.loads(l) for l in open(path, encoding="utf-8")]
        suffix = "" if a.judge_model.startswith("gemma4") else "_" + a.judge_model.split(":")[0].split("/")[-1]
        out_path = path.replace(".jsonl", f".judged{suffix}.jsonl")
        with open(out_path, "w", encoding="utf-8") as f:
            for r in rows:
                try:
                    s = ollama_json(JUDGE.format(q=r["q"], ans=r["vi"][:4000] or "(trống)"), a.judge_model)
                    r["j_vi"], r["j_ok"], r["j_why"] = int(s["vi"]), int(s["ok"]), s.get("why", "")
                except Exception as e:
                    r["j_err"] = f"{type(e).__name__}: {e}"
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        good = [r for r in rows if "j_vi" in r]
        tag = rows[0].get("tag", path)
        fmt = [r["format_ok"] for r in rows if r.get("format_ok") is not None]
        print(f"\n== {tag} · giám-khảo {a.judge_model} · {len(good)}/{len(rows)} câu chấm được ==")
        print(f"   tự-nhiên tiếng Việt TB {statistics.mean(r['j_vi'] for r in good):.2f}/5 · "
              f"đúng/hữu-ích TB {statistics.mean(r['j_ok'] for r in good):.2f}/5"
              + (f" · đúng định-dạng <vi> {sum(fmt)}/{len(fmt)}" if fmt else "")
              + f" · {statistics.mean(r['sec'] for r in rows):.1f}s/câu (lúc sinh)")
        for grp in ("suite", None):
            sub = [r for r in good if (r["source"] == "suite") == (grp == "suite")]
            if sub:
                print(f"   {'bộ 7 câu' if grp else 'held-out'}: vi {statistics.mean(r['j_vi'] for r in sub):.2f}"
                      f" · ok {statistics.mean(r['j_ok'] for r in sub):.2f} (n={len(sub)})")
        print(f"   → chi-tiết: {out_path}")


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    g = sp.add_parser("gen")
    g.add_argument("--adapter", default=None)
    g.add_argument("--tag", required=True)
    g.add_argument("--max-new", type=int, default=600)
    g.add_argument("--bs", type=int, default=0)
    g.add_argument("--use-kernels", action="store_true")
    o = sp.add_parser("gen-ollama")
    o.add_argument("--model", default="gemma4:26b-a4b-it-qat")
    o.add_argument("--tag", required=True)
    o.add_argument("--max-new", type=int, default=800)
    h = sp.add_parser("gen-http")
    h.add_argument("--tag", required=True)
    h.add_argument("--url", default="http://localhost:8790")
    h.add_argument("--max-new", type=int, default=600)
    h.add_argument("--rep-pen", type=float, default=1.1)
    h.add_argument("--rag", action="store_true")
    j = sp.add_parser("judge")
    j.add_argument("files", nargs="+")
    j.add_argument("--judge-model", default="gemma4:26b-a4b-it-qat")
    a = ap.parse_args()
    {"gen": cmd_gen, "gen-ollama": cmd_gen_ollama, "gen-http": cmd_gen_http, "judge": cmd_judge}[a.cmd](a)


if __name__ == "__main__":
    main()
