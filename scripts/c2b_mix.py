"""TK_108 C4 lần 3 — gộp dữ liệu C2b thành 3 file trộn cho trainGEN (mỗi file = một khoá DATA):
  data/c2b/mix_A.jsonl         phần A (10 ngôn ngữ phổ biến, kiểm cú pháp) — tối đa --per-a mẫu mỗi ngôn ngữ
  data/c2b/mix_plus.jsonl      C2b+ (Next.js, Tailwind/WebSight, Svelte, Kotlin, Flutter, shadcn, glaive…) — tối đa --per-p mỗi nhóm
  data/c2b/mix_verified.jsonl  phần B (thầy viết test + CHẠY THẬT, có bản tiếng Việt) — lấy hết
Chỉ giữ trường messages (+ id, lang). Trộn ngẫu nhiên cố định (seed 308).
    python scripts/c2b_mix.py [--per-a 8000] [--per-p 6000]
"""
import argparse, glob, json, os, random, re

CJK = re.compile(r"[\u3040-\u30ff\u3400-\u9fff\uac00-\ud7af]")   # đề Hán / Nhật / Hàn lẫn từ OpenCoder → bỏ

D = "data/c2b"


def rows(p, cap, rng):
    r = [json.loads(l) for l in open(p, encoding="utf-8")]
    r = [x for x in r if len(CJK.findall(x["messages"][0]["content"])) < 3]
    rng.shuffle(r)
    return [{"id": x["id"], "lang": x.get("lang"), "messages": x["messages"]} for x in r[:cap]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-a", type=int, default=8000)
    ap.add_argument("--per-p", type=int, default=6000)
    a = ap.parse_args()
    rng = random.Random(308)
    A = ["javascript", "typescript", "html_css", "python", "sql", "java", "csharp", "php", "go", "bash"]
    groups = {
        "mix_A": [(f"{D}/{l}.jsonl", a.per_a) for l in A],
        "mix_plus": [(p, a.per_p) for p in sorted(glob.glob(f"{D}/plus_*.jsonl"))],
        "mix_verified": [(p, 10 ** 9) for p in sorted(glob.glob(f"{D}/verified_*.jsonl"))],
    }
    for name, srcs in groups.items():
        out = []
        for p, cap in srcs:
            if os.path.exists(p):
                got = rows(p, cap, rng)
                out += got
                print(f"[mix] {name} ← {os.path.basename(p)}: {len(got)}")
        rng.shuffle(out)
        with open(f"{D}/{name}.jsonl", "w", encoding="utf-8") as f:
            for x in out:
                f.write(json.dumps(x, ensure_ascii=False) + "\n")
        print(f"### {name}: {len(out)} mẫu")


if __name__ == "__main__":
    main()
