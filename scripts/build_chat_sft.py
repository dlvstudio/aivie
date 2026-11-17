"""Dữ-liệu fine-tune GIAO-TIẾP cho model 1.3B (khuôn Hỏi/Đáp của train_hybrid_sft.py).

Nguồn (TK_104 · môn giao-tiếp):
  • DailyDialog dịch khẩu-ngữ (data/giaotiep/dd_envi.jsonl, CHỈ phần train — test giữ sạch)
  • hội-thoại "ăn nhập" Gemma sinh (data/giaotiep/grounded*.jsonl)
Mỗi hội-thoại → 1 mẫu nhiều lượt: lượt chẵn = người dùng ("Hỏi:"), lượt lẻ = model ("Đáp:", có loss),
cả bản tiếng Việt và tiếng Anh. Trộn lại một phần sft_v2 (trung-thực, bám nguồn, toán) để KHÔNG QUÊN hành-vi cũ
(bài-học TK_104: không ôn thì quên sạch).

    python scripts/build_chat_sft.py --grounded data/giaotiep/grounded_snap2.jsonl --replay 10000
    → data/sft_chat/chat_mix.jsonl
"""
import argparse, json, os, random


def to_msgs(turns):
    if len(turns) % 2:                                       # lượt cuối là người dùng không ai đáp → bỏ
        turns = turns[:-1]
    return [{"role": "user" if i % 2 == 0 else "assistant", "content": t} for i, t in enumerate(turns)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dd", default="data/giaotiep/dd_envi.jsonl")
    ap.add_argument("--grounded", default="data/giaotiep/grounded.jsonl")
    ap.add_argument("--replay", type=int, default=10000, help="số mẫu sft_v2 trộn lại để ôn")
    ap.add_argument("--out", default="data/sft_chat/chat_mix.jsonl")
    ap.add_argument("--system", default="", help="câu dặn hệ-thống gắn vào MỌI mẫu hội-thoại → tách chế-độ TRÒ CHUYỆN "
                    "khỏi chế-độ TRỢ LÝ (không câu dặn = làm việc được nhờ). Bản 1 không có → model đóng vai người chat cả khi được nhờ soạn email.")
    a = ap.parse_args()
    rng = random.Random(104)
    rows = []
    for path, keep in ((a.dd, lambda r: r["split"] == "train"), (a.grounded, lambda r: True)):
        for l in open(path, encoding="utf-8"):
            try:
                r = json.loads(l)
            except Exception:
                continue
            if not keep(r):
                continue
            for lang in ("vi", "en"):
                m = to_msgs(r[lang])
                if a.system:
                    m = [{"role": "system", "content": a.system if lang == "vi" else "Chat naturally like a friend."}] + m
                if len(m) >= 2:
                    rows.append({"messages": m, "src": f"{os.path.basename(path)}:{lang}"})
    n_dlg = len(rows)
    sft = [json.loads(l) for l in open("data/sft_v2/sft_v2.jsonl", encoding="utf-8")]
    rng.shuffle(sft)
    rows += [{"messages": r["messages"], "src": "sft_v2"} for r in sft[: a.replay]]
    rng.shuffle(rows)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{n_dlg} hội-thoại (vi+en) + {min(a.replay, len(sft))} ôn sft_v2 = {len(rows)} mẫu → {a.out}")


if __name__ == "__main__":
    main()
