"""§TK_103 · Ghép bộ SFT LẦN 2 cho model 1.3B: trung-thực + bám nguồn + toán từng bước + ôn lại.

Mốc §21 (eval_v1): nhận "không biết" 0/20 · sự-kiện đúng 2–5% · toán 0/18 · câu mở đúng 2,04/5.

  replay      15.000  sft_mix (khuôn trợ-lý đang có — chống quên; [[qlora-data-da-dang]])
  rag_ctx      5.000  ViQuAD 2.0: đoạn văn + câu hỏi → đáp-án; câu KHÔNG trả lời được → "đoạn văn không có"
                      (nhiều khuôn lệnh, gồm cả khuôn RAG của UI) — dạy BÁM NGUỒN
  closed        ~800  gen_selfknow.py: 1.3B tự trả lời câu ViQuAD không kèm nguồn; đúng → giữ câu của nó,
                      sai → câu từ-chối (R-Tuning) — dạy TỰ BIẾT MÌNH KHÔNG BIẾT
  fake          ~600  gen_selfknow.py: thực-thể bịa (cách hỏi khác eval_v1) → từ-chối
  math        ~8.800  longhoang06/Vi-GSM8K: lời giải có phép tính "48/2 = 24 … Đáp án: 72" — dạy TÍNH TỪNG BƯỚC

    PYTHONPATH=scripts .venv/bin/python scripts/build_sft_v2.py   → data/sft_v2/sft_v2.jsonl
"""
from __future__ import annotations
import collections, json, random, sys

sys.path.insert(0, "scripts")
from build_eval_clean import shingles
from gen_selfknow import ABSTAIN

RNG = random.Random(20261001)
OUT = "data/sft_v2/sft_v2.jsonl"

CTX_TEMPLATES = [
    "Dựa vào đoạn văn sau, trả lời câu hỏi.\n\n{c}\n\nCâu hỏi: {q}",
    "Đọc đoạn văn rồi trả lời ngắn gọn. Nếu đoạn văn không có thông tin, hãy nói rõ.\n\n{c}\n\nCâu hỏi: {q}",
    "Dựa vào các nguồn dưới đây, trả lời ngắn gọn bằng tiếng Việt. Nếu nguồn không có, nói không tìm thấy.\n[Nguồn 1 — {t}] {c}\nCâu hỏi: {q}",
    "{c}\n\nTheo nội dung trên, {q_lower}",
]
NO_CTX = [
    "Đoạn văn không có thông tin để trả lời câu hỏi này.",
    "Không tìm thấy thông tin này trong nguồn được cung cấp.",
    "Nội dung trên không đề cập đến điều bạn hỏi, nên tôi không thể trả lời dựa vào nó.",
]


def msg(u, a, src):
    return {"source": src, "messages": [{"role": "user", "content": u}, {"role": "assistant", "content": a}]}


def main():
    from datasets import load_dataset
    ev = [json.loads(l)["q"] for l in open("data/eval_clean/eval_v1.jsonl", encoding="utf-8")]
    ev_sh = [shingles(q) for q in ev]

    def contam(q):
        s = shingles(q)
        return max(len(s & e) / max(1, len(s | e)) for e in ev_sh) >= 0.35

    out = []
    # 1) ôn lại
    mix = [json.loads(l) for l in open("data/sft_vi/sft_mix.jsonl", encoding="utf-8")]
    RNG.shuffle(mix)
    out += [{"source": "replay:" + r.get("source", "?"), "messages": r["messages"]} for r in mix[:15000]]

    # 2) bám nguồn (ViQuAD có đoạn văn) — tách khỏi các câu đã dùng cho closed-book
    closed = [json.loads(l) for l in open("data/sft_v2/selfknow_closed.jsonl", encoding="utf-8")]
    used = {r["id"] for r in closed}
    vq = [r for r in load_dataset("taidng/UIT-ViQuAD2.0", split="train") if r["id"] not in used]
    RNG.shuffle(vq)
    n_ctx = 0
    for r in vq:
        if n_ctx >= 5000:
            break
        if len(r["context"]) > 2500 or contam(r["question"]):
            continue
        tpl = RNG.choice(CTX_TEMPLATES)
        q = r["question"].strip()
        u = tpl.format(c=r["context"].strip(), q=q, t=r["title"], q_lower=q[:1].lower() + q[1:])
        a = RNG.choice(NO_CTX) if r["is_impossible"] else r["answers"]["text"][0].strip()
        out.append(msg(u, a, "rag_ctx:" + ("khong_co" if r["is_impossible"] else "co")))
        n_ctx += 1

    # 3) tự biết mình không biết (sách đóng)
    for r in closed:
        # chỉ giữ ĐOẠN ĐẦU (chứa ý đúng): model hay viết tiếp phần dễ bịa, và bị cắt ở 64 token
        a = r["model_answer"].split("\n")[0].strip() if r["correct"] else RNG.choice(ABSTAIN)
        out.append(msg(r["q"], a, "closed:" + ("dung" if r["correct"] else "tu_choi")))

    # 4) thực-thể bịa
    for r in map(json.loads, open("data/sft_v2/selfknow_fake.jsonl", encoding="utf-8")):
        out.append(msg(r["q"], r["answer"], "fake"))

    # 5) toán từng bước
    for r in load_dataset("longhoang06/Vi-GSM8K", split="train"):
        q, a = r["question"].strip(), r["answer"].strip()
        if q and "Đáp án" in a and not contam(q):
            out.append(msg(q, a, "math:gsm8k"))

    RNG.shuffle(out)
    with open(OUT, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    c = collections.Counter(r["source"].split(":")[0] if not r["source"].startswith(("closed", "rag_ctx")) else r["source"]
                            for r in out)
    print(f"### LƯU {OUT} · {len(out)} mẫu · {dict(sorted(c.items()))}")


if __name__ == "__main__":
    main()
