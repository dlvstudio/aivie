"""TK_104 §15 · HIỆU CHỈNH TRUNG THỰC: dạy 1.3B TRẢ LỜI câu kiến thức phổ thông (đang từ chối ~50%), vẫn TỪ CHỐI thực thể bịa.

Đo trên UI v2 (20 câu dễ, data/eval_clean/easy20.jsonl): nhiệt độ 0 đúng ~7/20, từ chối 10/20 — SFT v2 dạy "không biết"
với câu khó/bịa nhưng thiếu mẫu "câu hiển nhiên thì phải đáp" → model dè dặt với MỌI câu kiến thức.

  gen   : Gemma viết lô 10 cặp hỏi–đáp kiến thức "ai học hết THCS cũng biết" theo chủ đề (đáp ngắn 1–2 câu, tự nhiên)
  check : Gemma (nhiệt độ 0) kiểm từng cặp: đúng chắc chắn + phổ biến → giữ
  build : trộn ĐÁP (câu phổ thông đã kiểm) với TỪ CHỐI (thực thể bịa, data/sft_v2/selfknow_fake.jsonl) ~3:1;
          bỏ câu gần trùng 20 câu kiểm tra (giữ bài kiểm tra sạch) → data/calib/train.jsonl

    python scripts/gen_calib.py gen --n 400 --workers 8      # 400 lô ≈ 4.000 cặp
    python scripts/gen_calib.py build
"""
from __future__ import annotations
import argparse, json, os, random, re, sys, time
import concurrent.futures as cf
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gen_office import ollama, loads_loose, CJK

RAW = "data/calib/raw.jsonl"
EASY = "data/eval_clean/easy20.jsonl"
T0 = time.time()

EASY20 = [("Thủ đô của Việt Nam là gì?", "Hà Nội"), ("Sông nào dài nhất Việt Nam chảy qua nhiều nước Đông Nam Á?", "Mê Kông"),
          ("Ai là tác giả Truyện Kiều?", "Nguyễn Du"), ("Nước sôi ở bao nhiêu độ C ở áp suất thường?", "100"),
          ("Một năm có bao nhiêu tháng?", "12"), ("Thành phố Hồ Chí Minh trước đây tên là gì?", "Sài Gòn"),
          ("Đỉnh núi cao nhất Việt Nam là gì?", "Fansipan"), ("Trái Đất quay quanh cái gì?", "Mặt Trời"),
          ("Quốc khánh Việt Nam là ngày nào?", "2/9"), ("Đơn vị tiền tệ của Việt Nam là gì?", "đồng"),
          ("Tết Nguyên Đán tính theo lịch nào?", "âm"), ("Chữ viết tiếng Việt hiện nay gọi là chữ gì?", "Quốc ngữ"),
          ("Có bao nhiêu châu lục trên thế giới?", "7"), ("Hà Nội nằm ở miền nào của Việt Nam?", "Bắc"),
          ("Vịnh Hạ Long thuộc tỉnh nào?", "Quảng Ninh"), ("Nước có công thức hoá học là gì?", "H2O"),
          ("Một tuần có mấy ngày?", "7"), ("Ai là người đọc Tuyên ngôn độc lập năm 1945?", "Hồ Chí Minh"),
          ("Phở là món ăn nổi tiếng của nước nào?", "Việt Nam"), ("Thủ đô của Nhật Bản là gì?", "Tokyo")]

TOPICS = ["địa lý Việt Nam (tỉnh, sông, núi, biển, vùng miền)", "địa lý thế giới (thủ đô, châu lục, đại dương, quốc gia)",
          "lịch sử Việt Nam phổ thông (sự kiện, nhân vật, ngày lễ)", "lịch sử thế giới phổ thông",
          "văn học Việt Nam (tác giả – tác phẩm học trong trường)", "vật lý phổ thông", "hoá học phổ thông",
          "sinh học và cơ thể người", "động vật và thực vật", "thiên văn (Mặt Trời, Mặt Trăng, hành tinh)",
          "toán học phổ thông (khái niệm, đơn vị, hình học)", "đơn vị đo lường và thời gian", "văn hoá, ẩm thực, phong tục Việt Nam",
          "thể thao phổ biến", "công nghệ và máy tính cơ bản", "sức khoẻ và an toàn thường thức", "kinh tế đời sống (tiền, ngân hàng cơ bản)",
          "âm nhạc, hội hoạ phổ thông", "giao thông và luật đi đường cơ bản", "ngôn ngữ và tiếng Việt (từ loại, chính tả cơ bản)"]

P_GEN = """Viết 10 cặp HỎI–ĐÁP tiếng Việt về chủ đề: {topic}.
Yêu cầu:
- Câu hỏi về kiến thức PHỔ THÔNG, CHẮC CHẮN ĐÚNG, ai học hết lớp 9 ở Việt Nam cũng biết (không hỏi chi tiết hiếm, số liệu dễ thay đổi).
- Câu hỏi đa dạng cách hỏi (tự nhiên như người dùng gõ: "...là gì?", "ai...", "bao nhiêu...", "cho mình hỏi...").
- Câu đáp: trả lời thẳng vào câu hỏi trong 1–2 câu, tự nhiên, có thể thêm một chi tiết ngắn đúng; KHÔNG rào đón.
Trả về DUY NHẤT JSON: {{"pairs": [{{"q": "...", "a": "..."}}, ...]}}"""

P_CHECK = """Kiểm tra từng cặp hỏi–đáp tiếng Việt dưới đây. Với mỗi cặp, "ok" = true CHỈ KHI câu đáp ĐÚNG CHẮC CHẮN
(theo kiến thức phổ thông hiện nay) VÀ câu hỏi là kiến thức phổ biến; sai, mơ hồ, đã lỗi thời hoặc gây tranh cãi → false.
{items}
Trả về DUY NHẤT JSON: {{"ok": [true, false, ...]}} đúng {n} phần tử theo thứ tự."""


def work(args):
    i, topic, url = args
    try:
        o = loads_loose(ollama(P_GEN.format(topic=topic), url, temp=0.9, fmt="json", num_predict=1500))
        ps = [p for p in o["pairs"] if p.get("q") and p.get("a") and not CJK.search(p["q"] + p["a"])][:10]
        if not ps:
            return {"i": i, "topic": topic, "drop": "rỗng"}
        items = "\n".join(f"{k+1}. HỎI: {p['q']}\n   ĐÁP: {p['a']}" for k, p in enumerate(ps))
        c = loads_loose(ollama(P_CHECK.format(items=items, n=len(ps)), url, temp=0, fmt="json", num_predict=300))
        keep = [p for p, ok in zip(ps, c["ok"]) if ok is True]
        return {"i": i, "topic": topic, "pairs": keep, "n_gen": len(ps)}
    except Exception as e:
        return {"i": i, "topic": topic, "drop": f"{type(e).__name__}: {str(e)[:80]}"}


def cmd_gen(a):
    os.makedirs(os.path.dirname(RAW), exist_ok=True)
    rng = random.Random(a.seed)
    specs = [rng.choice(TOPICS) for _ in range(a.n)]
    done = {json.loads(l)["i"] for l in open(RAW)} if os.path.exists(RAW) else set()
    todo = [(i, t, a.ollama) for i, t in enumerate(specs) if i not in done]
    print(f"[calib] còn {len(todo)} lô", flush=True)
    kept = gen = 0
    with open(RAW, "a", encoding="utf-8") as f, cf.ThreadPoolExecutor(a.workers) as ex:
        for k, r in enumerate(ex.map(work, todo), 1):
            f.write(json.dumps(r, ensure_ascii=False) + "\n"); f.flush()
            kept += len(r.get("pairs", [])); gen += r.get("n_gen", 0)
            if k % 20 == 0 or k == len(todo):
                print(f"[{time.time()-T0:.0f}s] lô {k}/{len(todo)} · giữ {kept}/{gen} cặp", flush=True)
    print("### calib xong", flush=True)


def toks(s):
    return set(re.findall(r"\w+", s.lower()))


def cmd_build(a):
    os.makedirs("data/eval_clean", exist_ok=True)
    with open(EASY, "w", encoding="utf-8") as f:
        for q, g in EASY20:
            f.write(json.dumps({"q": q, "gold": g}, ensure_ascii=False) + "\n")
    easy = [toks(q) for q, _ in EASY20]
    pos, seen = [], set()
    for l in open(RAW, encoding="utf-8"):
        try:
            r = json.loads(l)
        except json.JSONDecodeError:
            continue
        for p in r.get("pairs", []):
            k = re.sub(r"\W+", " ", p["q"].lower()).strip()
            t = toks(p["q"])
            if k in seen or any(len(t & e) / max(1, len(t | e)) > 0.5 for e in easy):
                continue
            seen.add(k)
            pos.append({"messages": [{"role": "user", "content": p["q"]}, {"role": "assistant", "content": p["a"]}],
                        "src": "calib_dap"})
    neg = []
    if os.path.exists("data/sft_v2/selfknow_fake.jsonl"):
        for l in open("data/sft_v2/selfknow_fake.jsonl", encoding="utf-8"):
            r = json.loads(l)
            if "q" in r and "answer" in r:
                neg.append({"messages": [{"role": "user", "content": r["q"]}, {"role": "assistant", "content": r["answer"]}],
                            "src": "calib_tu_choi"})
    rng = random.Random(3)
    rng.shuffle(neg)
    neg = neg[: max(1, len(pos) // 3)]
    rows = pos + neg
    rng.shuffle(rows)
    os.makedirs("data/calib", exist_ok=True)
    with open("data/calib/train.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"đáp {len(pos)} · từ chối (thực thể bịa) {len(neg)} → data/calib/train.jsonl ({len(rows)}) · bài kiểm tra {EASY}")


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    g = sp.add_parser("gen"); g.add_argument("--n", type=int, default=400); g.add_argument("--seed", type=int, default=0)
    g.add_argument("--workers", type=int, default=8); g.add_argument("--ollama", default="http://127.0.0.1:11434")
    sp.add_parser("build")
    a = ap.parse_args()
    {"gen": cmd_gen, "build": cmd_build}[a.cmd](a)


if __name__ == "__main__":
    main()
