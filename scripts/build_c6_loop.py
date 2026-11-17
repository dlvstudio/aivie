"""TK_108 vòng thầy trên SFT-B (thân C6): gộp dữ liệu học tiếp TOÀN THÂN từ kết quả vòng thầy.

  chữ   data/v6/<tag>.jsonl (vlm_loop run): lượt KHÔNG ảnh — điểm < 8 có bản sửa → học bản sửa; điểm ≥ 9 → giữ câu của B.
        Các lượt trợ lý trước đó trong hội thoại là câu CỦA B (on-policy) → nolearn (cho thấy, không cho học).
  tool  data/c3/traj_<tag>.jsonl (c3_teacher run): CHỈ quỹ đạo thành công (test chạy lại QUA + gọi công cụ hợp lệ) —
        của chính B (tự giải được) và của thầy (bài B hỏng)
  ôn    mẫu ngẫu nhiên từ data/sft_c6/sft_b.jsonl (chống quên), ≥ --replay-min, gấp --replay-x lần phần mới

    python scripts/build_c6_loop.py --v6 c6l1a,c6l1b --traj c6l_b0,c6l_b1,c6l_b2,c6l_g --out data/sft_c6/loop1.jsonl
"""
import argparse, collections, json, os, random


def rows(p):
    if not os.path.exists(p):
        print(f"[loop] thiếu {p}", flush=True)
        return []
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v6", default="")
    ap.add_argument("--traj", default="")
    ap.add_argument("--replay", default="data/sft_c6/sft_b.jsonl")
    ap.add_argument("--replay-x", type=float, default=2.0)
    ap.add_argument("--replay-min", type=int, default=10000)
    ap.add_argument("--out", default="data/sft_c6/loop1.jsonl")
    a = ap.parse_args()
    rng = random.Random(61)
    st = collections.Counter()

    txt = []
    for tag in filter(None, a.v6.split(",")):
        for x in rows(f"data/v6/{tag}.jsonl"):
            if x["images"]:
                continue
            ctx = [dict(m, nolearn=True) if m["role"] == "assistant" else m for m in x["history"]]
            ctx.append({"role": "user", "content": x["user"]})
            if 0 <= x["diem"] < 8 and len(x["sua"]) > 10:
                txt.append({"source": "c6loop_sua_" + x["kind"], "messages": ctx + [{"role": "assistant", "content": x["sua"]}]})
                st["chữ: bản sửa"] += 1
            elif x["diem"] >= 9:
                txt.append({"source": "c6loop_tot_" + x["kind"], "messages": ctx + [{"role": "assistant", "content": x["answer"]}]})
                st["chữ: giữ câu tốt"] += 1
            else:
                st["chữ: bỏ (8 điểm / sửa rỗng)"] += 1

    tool, seen = [], set()
    for tag in filter(None, a.traj.split(",")):
        for r in rows(f"data/c3/traj_{tag}.jsonl"):
            if r["ok"] and r["valid"] and r["id"] not in seen:
                seen.add(r["id"])
                tool.append({"source": f"c6loop_tool_{tag}_{r['kind']}", "messages": r["messages"]})
                st[f"tool {tag}: thành công"] += 1
            else:
                st[f"tool {tag}: hỏng"] += 1

    new = txt + tool
    old = rows(a.replay)
    rng.shuffle(old)
    k = min(len(old), max(a.replay_min, int(len(new) * a.replay_x)))
    mix = new + old[:k]
    rng.shuffle(mix)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        for r in mix:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("[loop] " + " · ".join(f"{k} {v}" for k, v in sorted(st.items())), flush=True)
    print(f"### loop {len(mix)} mẫu → {a.out} · chữ {len(txt)} · tool {len(tool)} · ôn {k}", flush=True)


if __name__ == "__main__":
    main()
