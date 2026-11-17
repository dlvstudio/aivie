"""TK_104 §17 · v5 — VÒNG THẦY CHAT–CHẤM–SỬA (ý người dùng): Gemma-4-31B chủ động chat NHIỀU LƯỢT với 1.3B, chấm 0–10 từng
lượt, viết BẢN SỬA cho lượt kém → dữ liệu hiệu chỉnh (SFT trên bản sửa + DPO cặp sửa/kém).

Vì sao chữa được "câu đầu ổn, vài câu sau lan man": hội thoại đi tiếp bằng CHÍNH câu trả lời của 1.3B (on-policy) → bản sửa
nằm đúng trong ngữ cảnh 1.3B tự tạo ra (kể cả khi lượt trước đã lệch) — kiểu DAgger, sửa lỗi dồn qua các lượt.

Mỗi tiến trình con giữ 1 bản 1.3B (thân + 3 expert, Runner có cache); thầy gọi qua vLLM (song song giữa các tiến trình).
    /workspace/zenv/bin/python scripts/v5_loop.py --n-conv 1000 --workers 4 --tag r1 \
        --code ckpt/expert_code/v4c_code.pt --office ckpt/expert_code/v4c_vp.pt --hoc ckpt/expert_code/v4c_hoc.pt
    → data/v5/r1.jsonl (mỗi dòng = 1 lượt: lịch sử, câu hỏi, câu trả lời, điểm, lý do, bản sửa)
"""
from __future__ import annotations
import argparse, json, os, random, sys, time
sys.path[:0] = [os.path.dirname(os.path.abspath(__file__)), os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")]

P_USER = """Bạn đóng vai NGƯỜI DÙNG Việt Nam đang chat với một trợ lý AI. Lĩnh vực: {dom}. Tình huống: {task}.
Cách bạn dẫn dắt hội thoại: {pat}.
{hist}
Hãy viết TIN NHẮN TIẾP THEO của người dùng (lượt {t}/{n}) — tự nhiên như người thật gõ (ngắn hoặc dài tuỳ lúc, có thể xưng
anh/chị/em/mình), bám theo những gì trợ lý vừa nói (kể cả khi trợ lý trả lời sai hay lạc đề thì phản ứng như người thật).
Chỉ viết đúng tin nhắn, không thêm gì khác."""

P_JUDGE = """Bạn là giám khảo nghiêm khắc, chấm câu trả lời của một TRỢ LÝ AI tiếng Việt trong hội thoại nhiều lượt.
{hist}
NGƯỜI DÙNG (mới nhất): {q}
TRỢ LÝ (cần chấm): {ans}

Chấm 0–10: đúng kiến thức/kết quả, làm đúng điều được hỏi, bám câu MỚI NHẤT và nhất quán với các lượt trước (nhớ thông tin đã nêu),
không bịa, không lặp, không lan man, tiếng Việt tự nhiên. Với câu hỏi kiến thức / trò chuyện / tư vấn: câu trả lời CỘC LỐC một câu
(đúng nhưng không giải thích, không gợi mở) tối đa 6 điểm. (10 = hoàn hảo; 5 = dùng được nhưng có lỗi rõ; 0 = sai/lạc đề/vô nghĩa)
Nếu điểm < 8: viết "sua" = câu trả lời TỐT NHẤT mà trợ lý lẽ ra phải viết ở lượt này (tiếng Việt, gọn, đúng, nhất quán với hội thoại,
công thức viết ký hiệu thường, không LaTeX; với câu kiến thức / trò chuyện: nêu thẳng ý chính trước rồi giải thích thêm hoặc ví dụ,
3–6 câu, và khi phù hợp kết bằng một câu hỏi mở để tiếp tục; công thức Excel/Sheets và code LUÔN đặt riêng trong khối ``` — vd
```excel
=SUMIF(B2:B10,"Hà Nội",D2:D10)
``` — không viết lẫn trong câu). Nếu điểm ≥ 8: "sua" để rỗng.
Trả về DUY NHẤT JSON: {{"diem": <0-10>, "ly_do": "một câu ngắn", "sua": "..."}}"""


def fmt_hist(msgs, cap=700):
    if not msgs:
        return "(Chưa có tin nhắn nào.)"
    return "Hội thoại đến giờ:\n" + "\n".join(f"{'NGƯỜI DÙNG' if m['role'] == 'user' else 'TRỢ LÝ'}: {m['content'][:cap]}" for m in msgs)


def worker(rank, a, jobs, out_path):
    import torch
    import expert_office_1p3b as X
    import expert_ffn_1p3b as E
    from gen_office import ollama, loads_loose
    E.BASE = a.base
    if a.device == "cuda":
        torch.cuda.set_device(0)
    r = X.multi_runner(a, a.device)
    tok = E.tok_()
    f = open(out_path + f".part{rank}", "a", encoding="utf-8")
    T0 = time.time()
    for k, (cid, sp) in enumerate(jobs):
        msgs = []
        for t in range(1, sp["n"] + 1):
            try:
                u = ollama(P_USER.format(**sp, hist=fmt_hist(msgs), t=t), a.llm, temp=0.9, num_predict=300).strip()
                if not u:
                    break
                msgs.append({"role": "user", "content": u})
                ans = X.gen_ids(r, tok, X.hist_ids(tok, msgs)[-a.max_ctx:], a.max_new)
                j = loads_loose(ollama(P_JUDGE.format(hist=fmt_hist(msgs[:-1]), q=u, ans=ans[:2500]), a.llm, temp=0, fmt="json",
                                       num_predict=1500))
                rec = {"cid": cid, "turn": t, "spec": sp, "history": msgs[:-1], "user": u, "answer": ans,
                       "diem": float(j.get("diem", -1)), "ly_do": j.get("ly_do", ""), "sua": (j.get("sua") or "").strip()}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n"); f.flush()
                msgs.append({"role": "assistant", "content": ans})     # đi tiếp bằng câu của CHÍNH 1.3B (on-policy)
            except Exception as e:
                print(f"[w{rank}] lỗi {cid}/{t}: {type(e).__name__} {str(e)[:80]}", flush=True)
                break
        if k % 10 == 0:
            print(f"[w{rank}] {k+1}/{len(jobs)} hội thoại · {time.time()-T0:.0f}s", flush=True)
    f.close()


def main():
    import torch.multiprocessing as mp
    from gen_multiturn import DOMAINS, PATTERNS
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-conv", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--tag", default="r1")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--base", default="ckpt/hybrid_1p3b_chat_v2.pt")
    ap.add_argument("--code", default=""); ap.add_argument("--office", default=""); ap.add_argument("--hoc", default="")
    ap.add_argument("--hid-code", type=int, default=8192); ap.add_argument("--hid-off", type=int, default=4096)
    ap.add_argument("--hid-hoc", type=int, default=8192); ap.add_argument("--code-open", action="store_true")
    ap.add_argument("--max-new", type=int, default=350)
    ap.add_argument("--max-ctx", type=int, default=1400, help="giữ tối đa bấy nhiêu token lịch sử (nạp ngữ cảnh dài → hết bộ nhớ GPU)")
    ap.add_argument("--llm", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--device", default="cuda", help="cuda | mps (Mac M4 làm 'người được hỏi' phụ qua đường hầm SSH)")
    ap.add_argument("--mod", type=int, default=0, help="chia hội thoại: chỉ số i với i %% mod == rem")
    ap.add_argument("--rem", type=int, default=0)
    ap.add_argument("--mode", default="all", choices=["all", "only", "skip"], help="only = chỉ phần i%%mod==rem (Mac) · skip = bỏ phần đó (GPU)")
    ap.add_argument("--out", default="", help="mặc định data/v5/<tag>.jsonl")
    a = ap.parse_args()
    rng = random.Random(a.seed)
    specs = []
    for i in range(a.n_conv):
        dom = rng.choice(list(DOMAINS))
        specs.append((f"{a.tag}_{i}", {"dom": dom, "task": rng.choice(DOMAINS[dom]), "pat": rng.choice(PATTERNS),
                                        "n": rng.choice([4, 5, 5, 6])}))
    if a.mod:
        keep = (lambda i: i % a.mod == a.rem) if a.mode == "only" else (lambda i: i % a.mod != a.rem)
        specs = [s_ for i, s_ in enumerate(specs) if a.mode == "all" or keep(i)]
    os.makedirs("data/v5", exist_ok=True)
    out = a.out or f"data/v5/{a.tag}.jsonl"
    done = set()
    for p in [out] + [out + f".part{k}" for k in range(a.workers)]:
        if os.path.exists(p):
            done |= {json.loads(l)["cid"] for l in open(p, encoding="utf-8") if l.strip()}
    specs = [s for s in specs if s[0] not in done]
    print(f"[v5] {len(specs)} hội thoại cần chạy · {a.workers} tiến trình", flush=True)
    chunks = [specs[k::a.workers] for k in range(a.workers)]
    mp.set_start_method("spawn", force=True)
    ps = [mp.Process(target=worker, args=(k, a, chunks[k], out)) for k in range(a.workers)]
    for p in ps:
        p.start()
    for p in ps:
        p.join()
    with open(out, "a", encoding="utf-8") as f:                       # gộp các phần
        for k in range(a.workers):
            p = out + f".part{k}"
            if os.path.exists(p):
                f.write(open(p, encoding="utf-8").read()); os.remove(p)
    rows = [json.loads(l) for l in open(out, encoding="utf-8") if l.strip()]
    import collections
    by_t = collections.defaultdict(list)
    for x in rows:
        if x["diem"] >= 0:
            by_t[x["turn"]].append(x["diem"])
    print("[v5] điểm TB theo lượt:", {t: round(sum(v) / len(v), 2) for t, v in sorted(by_t.items())},
          "· số lượt", len(rows), "· cần sửa (<8)", sum(1 for x in rows if 0 <= x["diem"] < 8), flush=True)


if __name__ == "__main__":
    main()
