"""TK_107 P3.1 — dữ liệu SFT cho thân 2,5B / 8K. Thân mới CHƯA qua SFT nào → gộp đủ mọi đợt SFT của 1.3B một lần:
  sft_mix      85K  SFT 1 của 1.3B (dịch, văn phòng, code, aya-vi, hội thoại Gemma, pháp luật, openorca…)
  sft_v2       ~15K phần RIÊNG của chat_v2: bám nguồn (ViQuAD) · tự biết không biết · toán từng bước (bỏ "replay" = sft_mix)
  dailydialog  ~10K hội thoại khẩu ngữ Việt (bước "chat" của 1.3B; chỉ split train — bộ chấm dùng test đã lọc trùng)
  multiturn    ~8,7K hội thoại nhiều lượt (bỏ 200 dòng đầu ⊇ 150 mẫu val của expert)
  dài          ~5%  mẫu 2,5K–8K token (dưới đây)
Mẫu DÀI: SFT chỉ trên hội thoại ngắn làm model "quên" cách đọc ngữ cảnh dài vừa học ở pha A2/B. Tự sinh từ SFT chỉ trên hội thoại ngắn làm model "quên" cách đọc ngữ cảnh dài vừa học ở pha A2/B. Mẫu dài tự sinh từ
viwiki, không cần thầy: ghép nhiều bài Wikipedia thành bó tài liệu, hỏi về MỘT bài nằm ở vị trí ngẫu nhiên:
  qa      "X là gì?"            → "Theo tài liệu k, <câu mở đầu của bài X>"   (tìm + chép đúng chỗ)
  which   "Tài liệu nào nói về X?" → "Tài liệu k."
  none    hỏi X KHÔNG có trong bó → "Các tài liệu được cung cấp không có thông tin về X."  (dạy nhận không biết)

    PYTHONPATH=scripts:src python scripts/build_sft_p3.py --out data/sft_p3/sft_p3.jsonl
"""
import argparse, json, os, random, re, sys
sys.path[:0] = ["scripts", "src"]

LEAD = re.compile(r"^([^\n.()]{2,60}?)(?: \([^)]{0,80}\))? (?:là|hay) ")
SENT = re.compile(r"^(.{20,400}?[.!?])(?:\s|$)", re.S)


def wiki_docs(path, n_max, rng):
    """bài viwiki có câu mở đầu kiểu định nghĩa "X là …" → (thực thể, câu mở đầu, văn bản)."""
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if rng.random() > 0.35:                       # lấy mẫu rải khắp file, không chỉ đầu file
                continue
            t = json.loads(line)
            t = t if isinstance(t, str) else t.get("text", "")
            if not 1500 <= len(t) <= 20000:
                continue
            m, s = LEAD.match(t), SENT.match(t)
            if not (m and s):
                continue
            ent = m.group(1).strip()
            if len(ent.split()) > 8 or not ent[0].isupper():
                continue
            out.append((ent, s.group(1).strip(), t))
            if len(out) >= n_max:
                break
    return out


def long_example(docs, rng, tok, lo, hi, cap):
    """một bó tài liệu dài ~[lo, hi] token + câu hỏi về một bài trong bó (hoặc bài ngoài bó)."""
    target = rng.randint(lo, hi)
    bundle, total = [], 0
    while total < target:
        ent, lead, text = docs[rng.randrange(len(docs))]
        if any(ent == b[0] for b in bundle):
            continue
        ids = tok.encode(text)
        if len(ids) > cap:                                 # cắt bài quá dài ở ranh giới đoạn
            text = text[: int(len(text) * cap / len(ids))].rsplit("\n", 1)[0]
            ids = tok.encode(text)
        bundle.append((ent, lead, text)); total += len(ids)
    kind = rng.choices(["qa", "which", "none"], [0.55, 0.25, 0.20])[0]
    body = "\n\n".join(f"[Tài liệu {i + 1}]\n{t}" for i, (_, _, t) in enumerate(bundle))
    if kind == "none":
        for _ in range(50):
            ent = docs[rng.randrange(len(docs))][0]
            if ent not in body:                            # chắc chắn bó KHÔNG nhắc tới thực thể này
                break
        else:
            return None
        q, ans = f"{ent} là gì?", f"Các tài liệu được cung cấp không có thông tin về {ent}."
    else:
        k = rng.randrange(len(bundle))
        ent, lead, _ = bundle[k]
        if sum(t.count(ent) > 0 for _, _, t in bundle) > 1:   # thực thể xuất hiện ở bài khác → câu trả lời mơ hồ
            return None
        if kind == "qa":
            q, ans = f"{ent} là gì?", f"Theo tài liệu {k + 1}, {lead}"
        else:
            q, ans = f"Tài liệu nào nói về {ent}?", f"Tài liệu {k + 1}."
    user = f"Dựa vào các tài liệu sau, trả lời câu hỏi.\n\n{body}\n\nCâu hỏi: {q}"
    return {"source": f"long_{kind}", "n_docs": len(bundle),
            "messages": [{"role": "user", "content": user}, {"role": "assistant", "content": ans}]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mix", default="data/sft_vi/sft_mix.jsonl")
    ap.add_argument("--chat", default="data/sft_v2/sft_v2.jsonl")
    ap.add_argument("--dialog", default="data/giaotiep/dd_envi.jsonl")
    ap.add_argument("--multiturn", default="data/multiturn/train.jsonl")
    ap.add_argument("--wiki", default="data/text_corpus/viwiki.jsonl")
    ap.add_argument("--n-long", type=int, default=6000)
    ap.add_argument("--min-tok", type=int, default=2500)
    ap.add_argument("--max-tok", type=int, default=7500)
    ap.add_argument("--doc-cap", type=int, default=1800, help="tối đa token mỗi bài trong bó")
    ap.add_argument("--max-len", type=int, default=8192)
    ap.add_argument("--out", default="data/sft_p3/sft_p3.jsonl")
    ap.add_argument("--seed", type=int, default=107)
    a = ap.parse_args()
    from nn_aivie.core.tokenizer import BPETokenizer
    from train_hybrid_sft import build_example
    tok = BPETokenizer.load("checkpoints/bpe_v1.json")
    rng = random.Random(a.seed)

    src = {}
    rows = [{"source": "mix:" + r.get("source", "?"), "messages": r["messages"]}
            for r in map(json.loads, open(a.mix, encoding="utf-8"))]
    src["sft_mix"] = len(rows)
    v2 = [r for r in map(json.loads, open(a.chat, encoding="utf-8")) if not r.get("source", "").startswith("replay")]
    rows += v2; src["sft_v2_rieng"] = len(v2)
    dd = []
    for r in map(json.loads, open(a.dialog, encoding="utf-8")):
        turns = [t.strip() for t in r["vi"] if t.strip()]
        if r["split"] != "train" or len(turns) < 2:
            continue
        dd.append({"source": "dailydialog_vi", "messages": [{"role": "user" if i % 2 == 0 else "assistant", "content": t}
                                                            for i, t in enumerate(turns)]})
    rows += dd; src["dailydialog"] = len(dd)
    mt = [{"source": "multiturn", "messages": r["messages"]}
          for r in map(json.loads, list(open(a.multiturn, encoding="utf-8"))[200:])]
    rows += mt; src["multiturn"] = len(mt)
    n_short = len(rows)
    print(f"ngắn: {src} = {n_short}", flush=True)
    docs = wiki_docs(a.wiki, 40000, rng)
    print(f"viwiki: {len(docs)} bài có câu định nghĩa", flush=True)
    longs, kinds, lens, tries = [], {}, [], 0
    while len(longs) < a.n_long and tries < a.n_long * 5:
        tries += 1
        r = long_example(docs, rng, tok, a.min_tok, a.max_tok, a.doc_cap)
        if r is None:
            continue
        n = len(build_example(r["messages"], tok, 10 ** 9)[0])
        if n > a.max_len:                                  # không để bị cắt mất câu trả lời ở đuôi
            continue
        longs.append(r); lens.append(n); kinds[r["source"]] = kinds.get(r["source"], 0) + 1
        if len(longs) % 500 == 0:
            print(f"  {len(longs)}/{a.n_long} mẫu dài", flush=True)
    rows += longs
    rng.shuffle(rows)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    lens.sort()
    stats = {"sources": src, "short": n_short, "long": len(longs), "long_frac": round(len(longs) / len(rows), 3), "kinds": kinds,
             "long_tok": {"min": lens[0], "p50": lens[len(lens) // 2], "max": lens[-1], "sum": sum(lens)}}
    json.dump(stats, open(a.out.replace(".jsonl", "_stats.json"), "w"), ensure_ascii=False, indent=1)
    print(json.dumps(stats, ensure_ascii=False))


if __name__ == "__main__":
    main()
