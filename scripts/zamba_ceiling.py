"""§TK_105 · Bước 0 — ĐO TRẦN: Zamba2-7B-Instruct biết bao nhiêu nếu hiểu đúng câu hỏi?

Trước khi xây cầu-nối ẩn (1.3B ↔ Zamba2), đo xem "bộ não tiếng Anh" có kiến-thức mà 1.3B thiếu không.
Cùng một bộ câu hỏi, Zamba2 đọc 2 kiểu:
  vi : câu tiếng Việt gốc                      → Zamba2 tự hiểu tiếng Việt (yếu, pilot B)
  en : bản dịch tiếng Anh chuẩn (Gemma dịch)   → TRẦN: cầu-nối hoàn hảo cũng chỉ chuyển được chừng này

  translate (Mac, Ollama Gemma) : eval_v1 (158 câu) + VMLU validation (744 câu) → data/zamba_ceiling/*_en.jsonl
  gen       (box H100, venv /workspace/zenv) : Zamba2 trả lời — eval_v1 sinh tự-do; VMLU chấm log-xác-suất chữ cái
  judge     (Mac hoặc box, Ollama Gemma) : chấm eval_v1 bằng đúng prompt của eval_clean.py
  report    : bảng so với 1.3B (sftv2) — sự-kiện đúng, nhận không-biết, toán, VMLU

    .venv/bin/python scripts/zamba_ceiling.py translate
    /workspace/zenv/bin/python scripts/zamba_ceiling.py gen            # trên box
    .venv/bin/python scripts/zamba_ceiling.py judge
    .venv/bin/python scripts/zamba_ceiling.py report
"""
from __future__ import annotations
import argparse, json, os, sys, time, re, concurrent.futures as cf, urllib.request
from collections import defaultdict

DIR = "data/zamba_ceiling"
EV_EN = f"{DIR}/evalv1_en.jsonl"
VM_EN = f"{DIR}/vmlu_val_en.jsonl"
GEMMA = "gemma4:26b-a4b-it-qat"
ZAMBA = "Zyphra/Zamba2-7B-Instruct"
T0 = time.time()


def log(*a):
    print(f"[{time.time()-T0:.0f}s]", *a, flush=True)


def ollama(body, url="http://127.0.0.1:11434", timeout=900):
    body["think"] = False
    req = urllib.request.Request(f"{url}/api/chat", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=timeout).read())["message"]["content"]


def jl(path):
    return [json.loads(l) for l in open(path, encoding="utf-8")] if os.path.exists(path) else []


# ───────────────────────── translate ─────────────────────────
P_TR_Q = """Translate this Vietnamese question into natural, accurate English.
Keep names, numbers, units and formulas exactly. Do not answer it.
Return ONLY JSON: {{"en": "..."}}

Vietnamese: {q}"""

P_TR_MCQ = """Translate this Vietnamese multiple-choice question and its options into accurate English.
Keep names, numbers, formulas and the option letters (A., B., ...) exactly. Do not answer it.
Return ONLY JSON: {{"question": "...", "choices": ["A. ...", "B. ...", ...]}} with the same number of choices.

{payload}"""


def tr_ev(r, url):
    out = json.loads(ollama({"model": GEMMA, "stream": False, "format": "json", "options": {"temperature": 0},
                             "messages": [{"role": "user", "content": P_TR_Q.format(q=r["q"])}]}, url))
    return {**r, "q_en": out["en"].strip()}


def tr_mcq(r, url):
    payload = json.dumps({"question": r["question"], "choices": r["choices"]}, ensure_ascii=False)
    out = json.loads(ollama({"model": GEMMA, "stream": False, "format": "json", "options": {"temperature": 0},
                             "messages": [{"role": "user", "content": P_TR_MCQ.format(payload=payload)}]}, url))
    ch = [str(c).strip() for c in out["choices"]]
    if len(ch) != len(r["choices"]):
        raise ValueError(f"số lựa chọn lệch {len(ch)} vs {len(r['choices'])}")
    ch = [c if re.match(r"^[A-E]\.", c) else f"{'ABCDE'[i]}. {c}" for i, c in enumerate(ch)]
    return {**r, "question_en": out["question"].strip(), "choices_en": ch}


def run_tr(rows, fn, path, key, workers, url):
    done = {r[key] for r in jl(path)}
    todo = [r for r in rows if r[key] not in done]
    log(f"{path}: đã có {len(done)}, còn {len(todo)}")
    with open(path, "a", encoding="utf-8") as f, cf.ThreadPoolExecutor(workers) as ex:
        futs = {ex.submit(fn, r, url): r for r in todo}
        for i, fu in enumerate(cf.as_completed(futs), 1):
            try:
                f.write(json.dumps(fu.result(), ensure_ascii=False) + "\n"); f.flush()
            except Exception as e:
                log("lỗi", futs[fu][key], type(e).__name__, str(e)[:100])
            if i % 25 == 0:
                log(f"{i}/{len(todo)}")


def cmd_translate(a):
    os.makedirs(DIR, exist_ok=True)
    ev = jl("data/eval_clean/eval_v1.jsonl")
    run_tr(ev, tr_ev, EV_EN, "id", a.workers, a.ollama)
    from datasets import load_dataset
    vm = [dict(x) for x in load_dataset("Hamana0509/VMLU", split="validation")]
    run_tr(vm, tr_mcq, VM_EN, "id", a.workers, a.ollama)
    log(f"### xong · eval_v1 {len(jl(EV_EN))}/{len(ev)} · VMLU {len(jl(VM_EN))}/{len(vm)}")


# ───────────────────────── gen (box) ─────────────────────────
SYS_EN = "You are a helpful, honest assistant. If you do not know something, say so."
MCQ_EN = "The following is a multiple-choice question. Choose ONE correct option and answer with only its letter."
MCQ_VI = ("Đây là câu hỏi trắc nghiệm. Hãy chọn MỘT đáp án đúng và chỉ trả lời bằng "
          "một chữ cái (A, B, C, D hoặc E).")
# thứ tự ưu tiên: câu có đáp-án chuẩn trước (đo trần kiến-thức), câu mở sau cùng (nếu còn giờ GPU)
ORDER = {"fact": 0, "unanswerable": 1, "math": 2, "open": 3}


def cmd_gen(a):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(ZAMBA)
    md = AutoModelForCausalLM.from_pretrained(ZAMBA, torch_dtype=torch.bfloat16, device_map="cuda").eval()
    log("nạp xong", ZAMBA)

    def prompt(user, prefix=""):
        msgs = [{"role": "user", "content": user}]
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True) + prefix

    # VMLU trước: chỉ 1 lượt forward/câu, vài phút — luôn có số dù hết giờ GPU giữa chừng
    letter_ids = {}
    for L in "ABCDE":
        ids = {tok.encode(s, add_special_tokens=False)[-1] for s in (L, " " + L)}
        letter_ids[L] = sorted(ids)
    vm_out = f"out/zceil_vmlu.jsonl"
    if not os.path.exists(vm_out):
        rows, res = jl(VM_EN), []
        for i, r in enumerate(rows, 1):
            rec = {"id": r["id"], "gold": r["answer"]}
            n = len(r["choices"])
            for lang in ("vi", "en"):
                if lang == "vi":
                    u = f"{MCQ_VI}\n\nCâu hỏi: {r['question']}\n" + "\n".join(r["choices"])
                else:
                    u = f"{MCQ_EN}\n\nQuestion: {r['question_en']}\n" + "\n".join(r["choices_en"])
                x = tok(prompt(u, "Answer:" if lang == "en" else "Đáp án:"), return_tensors="pt",
                        add_special_tokens=False).input_ids.cuda()
                with torch.no_grad():
                    lo = md(input_ids=x).logits[0, -1].float()
                sc = {L: max(lo[j].item() for j in letter_ids[L]) for L in "ABCDE"[:n]}
                rec[lang] = max(sc, key=sc.get)
            res.append(rec)
            if i % 100 == 0:
                acc = {l: sum(z[l] == z["gold"] for z in res) / len(res) for l in ("vi", "en")}
                log(f"VMLU {i}/{len(rows)} · {acc}")
        with open(vm_out, "w") as f:
            for z in res:
                f.write(json.dumps(z) + "\n")
        log(f"### VMLU: " + str({l: round(100 * sum(z[l] == z["gold"] for z in res) / len(res), 1)
                                  for l in ("vi", "en")}))

    rows = sorted(jl(EV_EN), key=lambda r: ORDER[r["type"]])
    for lang in ("en", "vi"):
        path = f"out/evalclean_zamba7b_{lang}.jsonl"
        done = {r["id"] for r in jl(path)}
        with open(path, "a", encoding="utf-8") as f:
            for i, r in enumerate(rows, 1):
                if r["id"] in done or (a.skip_open and r["type"] == "open"):
                    continue
                q = r["q_en"] if lang == "en" else r["q"]
                msgs = [{"role": "system", "content": SYS_EN}, {"role": "user", "content": q}]
                x = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").to("cuda")
                if not torch.is_tensor(x):
                    x = x["input_ids"].to("cuda")
                t = time.time()
                with torch.no_grad():
                    y = md.generate(x, max_new_tokens=a.max_new, do_sample=False, repetition_penalty=1.05)
                ans = tok.decode(y[0, x.shape[1]:], skip_special_tokens=True).strip()
                f.write(json.dumps({**r, "tag": f"zamba7b_{lang}", "answer": ans,
                                    "sec": round(time.time() - t, 1)}, ensure_ascii=False) + "\n")
                f.flush()
                if i % 10 == 0:
                    log(f"{lang} {i}/{len(rows)} · {time.time()-t:.1f}s/câu")
        log(f"### LƯU {path}")
    log("### ZCEIL XONG")


# ───────────────────────── judge / report ─────────────────────────
def cmd_judge(a):
    sys.path.insert(0, "scripts")
    from eval_clean import judge_one, judged_path
    for lang in ("en", "vi"):
        path = f"out/evalclean_zamba7b_{lang}.jsonl"
        rows = jl(path)
        with open(judged_path(path, GEMMA), "w", encoding="utf-8") as f:
            for r in rows:
                try:
                    r["j"] = judge_one(r, GEMMA)   # chấm theo câu tiếng Việt gốc + đáp-án chuẩn
                except Exception as e:
                    r["j_err"] = f"{type(e).__name__}: {e}"
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        log(f"### chấm xong {path}")


def vmlu_ours():
    p = "out/vmlu_sft_preds.json"
    if not os.path.exists(p):
        return None
    gold = {r["id"]: r["answer"] for r in jl(VM_EN)}
    pr = [x for x in json.load(open(p)) if x["id"] in gold]
    return round(100 * sum(x["answer"] == gold[x["id"]] for x in pr) / max(1, len(pr)), 1), len(pr)


def cmd_report(a):
    sys.path.insert(0, "scripts")
    from eval_clean import summarize
    print(f"{'model':16s} | {'sự-kiện đúng/không-biết/sai':>27s} | {'bịa-thực-thể: nhận/bịa':>23s} | "
          f"{'toán':>5s} | {'mở: đúng':>8s}")
    for tag in ("sftv2", "chat13v2", "zamba7b_vi", "zamba7b_en"):
        s = summarize(tag, GEMMA)
        if not s:
            continue
        f, u, m = s.get("fact", {}), s.get("unanswerable", {}), s.get("math", {})
        print(f"{tag:16s} | {f.get('correct',0):9d}% / {f.get('abstain',0):3d}% / {f.get('wrong',0):3d}% | "
              f"{u.get('abstain',0):12d}% / {u.get('hallucinated',0):3d}% | {m.get('correct',0):4d}% | "
              f"{s.get('open_ok', 0):8.2f}")
    vm = jl("out/zceil_vmlu.jsonl")
    if vm:
        acc = {l: round(100 * sum(z[l] == z["gold"] for z in vm) / len(vm), 1) for l in ("vi", "en")}
        print(f"\nVMLU validation ({len(vm)} câu) · Zamba2-7B đọc tiếng Việt {acc['vi']}% · đọc bản Anh {acc['en']}%"
              f" · ngẫu-nhiên ≈25%")
        o = vmlu_ours()
        if o:
            print(f"VMLU 1.3B (sft, {o[1]} câu): {o[0]}%")
        # theo môn: môn nào bản Anh hơn hẳn bản Việt (= kiến-thức có, chỉ thiếu cầu ngôn-ngữ)
        by = defaultdict(lambda: [0, 0, 0])
        for z in vm:
            k = z["id"].split("-")[0]
            by[k][0] += 1; by[k][1] += z["vi"] == z["gold"]; by[k][2] += z["en"] == z["gold"]
        gap = sorted(((v[2] - v[1]) / v[0], k, v) for k, v in by.items() if v[0] >= 8)
        print("môn lệch nhất (en−vi):", [(k, f"{v[1]}/{v[0]}→{v[2]}/{v[0]}") for g, k, v in gap[-5:]])
        # 1.3B vs Zamba2 theo môn: trần của mạng ghép là HỢP hai model — môn nào 1.3B hơn = kiến-thức Việt riêng
        if os.path.exists("out/vmlu_sft_preds.json"):
            gold = {z["id"]: z["gold"] for z in vm}
            ours = {x["id"]: x["answer"] for x in json.load(open("out/vmlu_sft_preds.json")) if x["id"] in gold}
            cmp_ = defaultdict(lambda: [0, 0, 0, 0])   # n, 1.3B, zamba vi, zamba en
            for z in vm:
                c = cmp_[z["id"].split("-")[0]]
                c[0] += 1; c[1] += ours.get(z["id"]) == z["gold"]; c[2] += z["vi"] == z["gold"]; c[3] += z["en"] == z["gold"]
            tot = [sum(c[i] for c in cmp_.values()) for i in range(4)]
            print(f"\ntổng {tot[0]} câu · 1.3B {tot[1]} · Zamba2-vi {tot[2]} · Zamba2-en {tot[3]} · "
                  f"HỢP (1.3B đúng hoặc Zamba2-en đúng) "
                  f"{sum((ours.get(z['id']) == z['gold']) or (z['en'] == z['gold']) for z in vm)}")
            print("môn (n) | 1.3B | Zamba2-vi | Zamba2-en")
            for k, c in sorted(cmp_.items(), key=lambda kv: (kv[1][1] - kv[1][3]) / kv[1][0], reverse=True):
                print(f"{k:>4s} ({c[0]:2d}) | {c[1]:3d} | {c[2]:3d} | {c[3]:3d}   {next(z for z in jl(VM_EN) if z['id'].startswith(k + '-'))['question'][:60]}")


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    t = sp.add_parser("translate"); t.add_argument("--workers", type=int, default=2)
    t.add_argument("--ollama", default="http://127.0.0.1:11434")
    g = sp.add_parser("gen"); g.add_argument("--max-new", type=int, default=320)
    g.add_argument("--skip-open", action="store_true")
    sp.add_parser("judge"); sp.add_parser("report")
    a = ap.parse_args()
    {"translate": cmd_translate, "gen": cmd_gen, "judge": cmd_judge, "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    main()
