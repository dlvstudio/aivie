"""§TK_103 · SFT lần 2 — dữ-liệu "TỰ BIẾT MÌNH KHÔNG BIẾT" cho model 1.3B (kiểu R-Tuning).

Mốc §21: 1.3B nhận "không biết" 0/20 với thực-thể bịa, sự-kiện đúng 2–5% (bịa ~95%).
Không dạy từ-chối MỌI câu sự-kiện (sẽ thành từ-chối lung-tung) — dựa vào chính kiến-thức của model:

  closed   : câu hỏi ViQuAD 2.0 (train, CÓ đáp-án) hỏi KHÔNG kèm đoạn văn, chỉ lấy câu nêu rõ tên đối-tượng
             (tự đủ nghĩa). Model 1.3B trả lời (nhiệt-độ 0) → đúng (đáp-án chuẩn có trong câu trả-lời) thì
             GIỮ câu trả-lời của nó; sai thì mục-tiêu = câu từ-chối.
  fake     : Gemma soạn câu hỏi về thực-thể BỊA với cách hỏi KHÁC bộ eval_v1 (sản-phẩm, công-ty, bài hát,
             giải thưởng, món ăn, công-trình…) → mục-tiêu = "không tìm thấy thông tin về X"; kiểm trùng với eval_v1.

    PYTHONPATH=scripts:src .venv/bin/python scripts/gen_selfknow.py --closed 800 --fake 600
    → data/sft_v2/selfknow_closed.jsonl · data/sft_v2/selfknow_fake.jsonl
"""
from __future__ import annotations
import argparse, json, os, random, re, sys, time, unicodedata, urllib.request

sys.path[:0] = ["scripts", "src"]
OUT = "data/sft_v2"
RNG = random.Random(20261001)

ABSTAIN = [
    "Tôi không chắc chắn về thông tin này nên không muốn đưa ra câu trả lời có thể sai. Bạn nên kiểm tra lại ở nguồn đáng tin cậy như sách, báo chính thống hoặc Wikipedia.",
    "Xin lỗi, tôi không có đủ thông tin chính xác để trả lời câu hỏi này. Bạn có thể bật chế độ tra cứu hoặc tìm thêm ở nguồn đáng tin cậy.",
    "Tôi không biết chắc câu trả lời cho câu hỏi này. Để tránh cung cấp thông tin sai, bạn nên tra cứu thêm từ nguồn chính thống.",
    "Câu hỏi này cần một chi tiết cụ thể mà tôi không nắm chắc. Tôi không muốn đoán, bạn hãy kiểm tra lại ở tài liệu đáng tin cậy nhé.",
]
FAKE_ABSTAIN = [
    "Tôi không tìm thấy thông tin nào về {x}. Có thể tên này chưa chính xác hoặc không tồn tại. Bạn kiểm tra lại tên, hoặc cho tôi thêm chi tiết nhé.",
    "Tôi không có thông tin về {x}. Tôi không muốn đoán vì dễ đưa ra thông tin sai. Bạn có thể cho biết bạn đọc được tên này ở đâu không?",
    "Theo những gì tôi biết, không có thông tin đáng tin cậy nào về {x}. Có thể tên bị nhầm hoặc đây không phải là thứ có thật.",
]


def n(s):
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", s.lower())).strip()


def build_closed(k):
    """Câu hỏi ViQuAD tự-đủ-nghĩa (có tên đối-tượng) → 1.3B trả lời → nhãn đúng/sai."""
    from datasets import load_dataset
    import torch
    from hybrid_infer import load, Runner
    from nn_aivie.core.tokenizer import BPETokenizer
    from nn_aivie.core.vocab import Vocabulary
    ds = load_dataset("taidng/UIT-ViQuAD2.0", split="train")
    pool = [r for r in ds if not r["is_impossible"] and r["answers"]["text"]
            and len(r["answers"]["text"][0]) <= 40 and n(r["title"]) in n(r["question"])]
    RNG.shuffle(pool)
    pool = pool[:k]
    print(f"[closed] {len(pool)} câu ViQuAD có tên đối-tượng trong câu hỏi", flush=True)
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    m, _, _ = load("checkpoints/hf_hybrid/hybrid_1p3b_assistant_v5.pt", dev)
    r = Runner(m, dev)
    tok = BPETokenizer.load("checkpoints/bpe_v1.json")
    vocab, eos = Vocabulary(tok), tok.token_to_id["<EOS>"]
    path = f"{OUT}/selfknow_closed.jsonl"
    done = {json.loads(l)["id"] for l in open(path, encoding="utf-8")} if os.path.exists(path) else set()
    T0, right = time.time(), 0
    with open(path, "a", encoding="utf-8") as f:
        for i, x in enumerate(pool, 1):
            if x["id"] in done:
                continue
            ids = tok.encode("Hỏi: " + x["question"].strip() + "\n") + tok.encode("Đáp: ")
            out = list(r.generate(ids, n=64, temp=0, rep_pen=1.1, eos=eos))
            ans = vocab.detokenize(out).strip()
            ok = any(n(g) in n(ans) for g in x["answers"]["text"])
            right += ok
            f.write(json.dumps({"id": x["id"], "q": x["question"], "gold": x["answers"]["text"][0],
                                "model_answer": ans, "correct": ok}, ensure_ascii=False) + "\n")
            f.flush()
            if i % 50 == 0:
                el = time.time() - T0
                print(f"[closed] {i}/{len(pool)} · đúng {right} · {el/i:.1f}s/câu", flush=True)
    print(f"### closed xong · đúng {right}", flush=True)


def gemma(prompt, temp=0.9):
    body = {"model": "gemma4:26b-a4b-it-qat", "stream": False, "think": False, "format": "json",
            "options": {"temperature": temp}, "messages": [{"role": "user", "content": prompt}]}
    req = urllib.request.Request("http://127.0.0.1:11434/api/chat", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        return json.loads(json.loads(urllib.request.urlopen(req, timeout=900).read())["message"]["content"])
    except json.JSONDecodeError:
        return {}


def build_fake(k):
    from build_eval_clean import shingles
    ev = [json.loads(l)["q"] for l in open("data/eval_clean/eval_v1.jsonl", encoding="utf-8")]
    ev_sh = [shingles(q) for q in ev]
    kinds = ["sản phẩm công nghệ hoặc ứng dụng điện thoại", "công ty hoặc thương hiệu Việt Nam", "bài hát hoặc album nhạc",
             "giải thưởng hoặc cuộc thi", "món ăn đặc sản vùng miền", "công trình kiến trúc hoặc cây cầu",
             "bộ phim hoặc chương trình truyền hình", "loài động vật hoặc thực vật", "phương pháp khoa học hoặc định luật",
             "trường học hoặc viện nghiên cứu", "lễ hội truyền thống", "vận động viên thể thao"]
    items, seen = [], set()
    for rnd in range(6):
        for kind in kinds:
            if len(items) >= k:
                break
            r = gemma(f"""Viết 12 câu hỏi tiếng Việt về {kind} KHÔNG TỒN TẠI (tên do bạn bịa, nghe như thật).
Hỏi tự nhiên, đa dạng kiểu hỏi (ai tạo ra, ra mắt năm nào, nằm ở đâu, có đặc điểm gì, giá bao nhiêu...),
như thể người hỏi tin rằng nó có thật.
Trả về JSON: {{"items": [{{"question": "...", "fake_entity": "tên bịa"}}]}}""")
            for it in r.get("items", []):
                q, x = str(it.get("question", "")).strip(), str(it.get("fake_entity", "")).strip()
                if not q or not x or n(x) in seen:
                    continue
                sq = shingles(q)
                if max(len(sq & e) / max(1, len(sq | e)) for e in ev_sh) >= 0.3:   # tránh làm đẹp điểm eval
                    continue
                seen.add(n(x))
                items.append({"q": q, "fake_entity": x, "kind": kind,
                              "answer": RNG.choice(FAKE_ABSTAIN).format(x=x)})
        print(f"[fake] vòng {rnd+1}: {len(items)} câu", flush=True)
        if len(items) >= k:
            break
    with open(f"{OUT}/selfknow_fake.jsonl", "w", encoding="utf-8") as f:
        for it in items[:k]:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    print(f"### fake xong · {min(k, len(items))} câu", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--closed", type=int, default=800)
    ap.add_argument("--fake", type=int, default=600)
    ap.add_argument("--only", choices=["closed", "fake"], default=None)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if a.only != "closed":
        build_fake(a.fake)
        urllib.request.urlopen(urllib.request.Request(          # nhả Gemma (~15GB) trước khi nạp 1.3B
            "http://127.0.0.1:11434/api/generate", headers={"Content-Type": "application/json"},
            data=json.dumps({"model": "gemma4:26b-a4b-it-qat", "keep_alive": 0}).encode()), timeout=120).read()
    if a.only != "fake":
        build_closed(a.closed)


if __name__ == "__main__":
    main()
