"""TK_104 · Tổng-hợp kết-quả thí-nghiệm học theo lớp (out/curriculum/*.json) → 4 bảng cho 4 câu hỏi.

    .venv/bin/python curriculum/analyze.py
"""
import glob, json, os, statistics as st
from collections import defaultdict

R = {os.path.basename(p)[:-5]: json.load(open(p)) for p in glob.glob("out/curriculum/*.json")}


def grp(prefix):
    return [v for k, v in R.items() if k.startswith(prefix + "_s")]


def mean_acc(runs, r, lv):
    xs = [x["final"][f"r{r}"][f"lop{lv}"] for x in runs if f"r{r}" in x.get("final", {})]
    return (st.mean(xs), max(xs) - min(xs)) if xs else (None, None)


def fmt(m):
    return "  —  " if m[0] is None else f"{100*m[0]:5.1f}" + (f"±{100*m[1]/2:3.1f}" if m[1] else "     ")


def main():
    print("== CÂU 1 · Học theo lớp vs trộn ngẫu-nhiên (cùng 6.000 bước) · độ-chính-xác % cuối, r=4 (TB 2 hạt-giống ±nửa-khoảng) ==")
    print(f"{'lịch':14s}" + "".join(f"{'lớp'+str(l):>11s}" for l in range(1, 7)))
    for name, pre in [("ngẫu-nhiên", "rand"), ("theo lớp", "curr"), ("theo lớp+ôn", "cr")]:
        print(f"{name:14s}" + "".join(f"{fmt(mean_acc(grp(pre), 4, l)):>11s}" for l in range(1, 7)))

    print("\n== CÂU 2 · Quên: độ-chính-xác % lớp k NGAY sau khi học xong lớp k → lúc kết-thúc (r=4) ==")
    for name, pre in [("theo lớp (không ôn)", "curr"), ("theo lớp + ôn 30%", "cr")]:
        for run in grp(pre):
            row = []
            for lv in range(1, 5):
                after = next((s["acc"]["r4"][f"lop{lv}"] for s in run["stages"] if s["after_level"] == lv), None)
                end = run["final"]["r4"][f"lop{lv}"]
                row.append(f"lớp{lv} {100*after:4.0f}→{100*end:4.0f}" if after is not None else "")
            print(f"  {name:22s} hạt {run['args']['seed']}: " + " · ".join(row))

    print("\n== CÂU 3 · Nghĩ nhiều vòng: độ-chính-xác % theo số vòng r lúc dùng ==")
    print(f"{'model':26s}{'lớp':>5s}" + "".join(f"{'r='+str(r):>8s}" for r in (1, 2, 4, 8)))
    for name, runs in [("train r=1..4 (cr)", grp("cr")), ("train r=1 cố-định (cr_r1)", grp("cr_r1"))]:
        for lv in (3, 4, 5, 6):
            print(f"{name:26s}{lv:>5d}" + "".join(f"{(lambda m: '  —' if m[0] is None else f'{100*m[0]:6.1f}')(mean_acc(runs, r, lv)):>8s}" for r in (1, 2, 4, 8)))

    print("\n== CÂU 4 · Lớn dần vs cố-định · % cuối r=4 + compute (tầng×bước, tương-đối so lõi 2 tầng) ==")
    base = grp("cr")
    base_c = st.mean(x["layer_steps"] for x in base) if base and "layer_steps" in base[0] else None
    for name, pre in [("cố-định lõi 2 tầng", "cr"), ("lớn dần 1→2 tầng", "crgrow"), ("cố-định lõi 1 tầng", "cr_C1")]:
        runs = grp(pre)
        if not runs:
            continue
        c = st.mean(x.get("layer_steps", 0) for x in runs)
        accs = " · ".join(f"lớp{l} {fmt(mean_acc(runs, 4, l)).strip()}" for l in (3, 4, 5, 6))
        rel = f"{100*c/base_c:5.0f}%" if base_c else "  ?"
        print(f"  {name:22s} compute {rel} · {accs}")


def round2():
    if not grp("mastery") and not grp("ans_r1"):
        return
    print("\n\n######## VÒNG 2 ########")
    print("\n== 1b · LÊN LỚP KHI THI ĐỖ vs trộn ngẫu-nhiên · % cuối, r=4 ==")
    print(f"{'lịch':16s}" + "".join(f"{'lớp'+str(l):>11s}" for l in range(1, 7)))
    for name, pre in [("ngẫu-nhiên", "rand"), ("lên lớp khi đỗ", "mastery")]:
        print(f"{name:16s}" + "".join(f"{fmt(mean_acc(grp(pre), 4, l)):>11s}" for l in range(1, 7)))
    for run in grp("mastery"):
        ps = " · ".join(f"lớp{p['from']}→{p['from']+1} @{p['step']} ({p['reason']} {100*p['dev_acc']:.0f}%)"
                        for p in run.get("promotions", []))
        print(f"  hạt {run['args']['seed']}: {ps or 'không lên lớp lần nào'}")

    print("\n== 3b · NGHĨ THẦM (chỉ ghi đáp-số) · % theo số vòng r lúc dùng ==")
    rs = (1, 2, 4, 8, 16)
    print(f"{'train':12s}{'lớp':>5s}" + "".join(f"{'r='+str(r):>8s}" for r in rs))
    for name, pre in [("1 vòng", "ans_r1"), ("1-4 vòng", "ans_r4"), ("1-8 vòng", "ans_r8")]:
        runs = grp(pre)
        for lv in range(1, 7):
            cells = []
            for r in rs:
                m = mean_acc(runs, r, lv)
                cells.append("  —" if m[0] is None else f"{100*m[0]:6.1f}")
            print(f"{name:12s}{lv:>5d}" + "".join(f"{c:>8s}" for c in cells))

    print("\n== 4b · LỚN DẦN trên lịch ngẫu-nhiên · % cuối r=4 + compute so với cố-định 2 tầng ==")
    base = grp("rand")
    bc = st.mean(x["layer_steps"] for x in base) if base and "layer_steps" in base[0] else None
    for name, pre in [("cố-định 2 tầng", "rand"), ("lớn dần 1→2", "randgrow")]:
        runs = grp(pre)
        if runs:
            c = st.mean(x.get("layer_steps", 0) for x in runs)
            print(f"  {name:16s} compute {100*c/bc:4.0f}% · " if bc else f"  {name:16s} ",
                  " · ".join(f"lớp{l} {fmt(mean_acc(runs, 4, l)).strip()}" for l in (3, 4, 5, 6)))


def plasticity():
    if not grp("plast_decay") and not grp("plast_freeze"):
        return
    print("\n\n######## VÒNG 3 · ĐỘ MỀM-DẺO THEO TẦNG (từ lớp 2: tầng dưới chậm/khoá, tầng trên tự-do) ########")
    rows = [("theo lớp, không ôn", "curr"), ("  + giảm-dần theo tầng", "plast_decay"), ("  + khoá dần sau mỗi lớp", "plast_freeze"),
            ("theo lớp + ôn 30%", "cr"), ("  + ôn + giảm-dần", "plast_decay_cr"), ("  + ôn + khoá dần", "plast_freeze_cr"),
            ("trộn ngẫu-nhiên", "rand")]
    print("\n== % cuối, r=4 (TB các hạt-giống) ==")
    print(f"{'cách học':28s}" + "".join(f"{'lớp'+str(l):>11s}" for l in range(1, 7)))
    for name, pre in rows:
        if grp(pre):
            print(f"{name:28s}" + "".join(f"{fmt(mean_acc(grp(pre), 4, l)):>11s}" for l in range(1, 7)))
    print("\n== GIỮ bài cũ & HỌC bài mới: lớp k ngay sau khi học xong → cuối (r=4, TB hạt) ==")
    for name, pre in rows[:6]:
        runs = grp(pre)
        if not runs:
            continue
        cells = []
        for lv in range(1, 6):
            aft = [next((x["acc"]["r4"][f"lop{lv}"] for x in r["stages"] if x["after_level"] == lv), None) for r in runs]
            aft = [x for x in aft if x is not None]
            end = st.mean(r["final"]["r4"][f"lop{lv}"] for r in runs)
            cells.append(f"lớp{lv} {100*st.mean(aft):3.0f}→{100*end:3.0f}" if aft else f"lớp{lv}  —→{100*end:3.0f}")
        print(f"  {name:26s} " + " · ".join(cells))


if __name__ == "__main__":
    main()
    round2()
    plasticity()
