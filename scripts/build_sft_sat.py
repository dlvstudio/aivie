"""Gom SFT sát thực tế: dialog đời thực (upsample) + gold + HF.

    python3 scripts/build_sft_sat.py
"""
import argparse, json, os, random


def load_jsonl(path):
    rows = []
    if not path or not os.path.exists(path):
        return rows
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        if r.get("messages"):
            rows.append(r)
    return rows


def emit(r, source_tag=None):
    ms = r["messages"]
    nt = sum(m.get("role") == "user" for m in ms)
    out = {
        "source": source_tag or r.get("source") or "sat",
        "n_turns": nt,
        "messages": ms,
    }
    if r.get("id"):
        out["id"] = r["id"]
    if r.get("domain"):
        out["domain"] = r["domain"]
    return out


def add(mix, rows, repeat, cap=0):
    if cap:
        rows = rows[:cap]
    n = 0
    for r in rows:
        rec = emit(r)
        mix.extend([rec] * repeat)
        n += 1
    return n


OFFICE_DOM = frozenset({"van_phong", "lap_trinh"})


def add_by_domain(mix, rows, default_repeat, office_repeat):
    n = n_off = 0
    for r in rows:
        rep = office_repeat if r.get("domain") in OFFICE_DOM else default_repeat
        rec = emit(r)
        mix.extend([rec] * rep)
        n += 1
        if r.get("domain") in OFFICE_DOM:
            n_off += 1
    return n, n_off


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dialogs", default="data/sft/gemma_dialog.jsonl")
    ap.add_argument("--gold", default="data/sft/gold_khuon.jsonl")
    ap.add_argument("--hf-dir", default="data/sft/hf")
    ap.add_argument("--out", default="data/sft_vi/sft_mix.jsonl")
    ap.add_argument("--dialog-repeat", type=int, default=12)
    ap.add_argument("--office-repeat", type=int, default=24,
                    help="upsample dialog/gold domain van_phong+lap_trinh")
    ap.add_argument("--gold-repeat", type=int, default=12)
    ap.add_argument("--code-repeat", type=int, default=4)
    ap.add_argument("--office-hf-repeat", type=int, default=12)
    ap.add_argument("--aya-cap", type=int, default=8000)
    ap.add_argument("--adapt-repeat", type=int, default=1)
    ap.add_argument("--adapt-cap", type=int, default=2500)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    mix, stats = [], {}
    n_d, n_doff = add_by_domain(mix, load_jsonl(a.dialogs), a.dialog_repeat, a.office_repeat)
    stats["dialog"] = n_d
    stats["dialog_office_code"] = n_doff
    n_g, n_goff = add_by_domain(mix, load_jsonl(a.gold), a.gold_repeat, a.office_repeat)
    stats["gold"] = n_g
    stats["gold_office_code"] = n_goff
    stats["aya"] = add(mix, load_jsonl(os.path.join(a.hf_dir, "aya_vi.jsonl")), 1, a.aya_cap)
    stats["oasst_vi"] = add(mix, load_jsonl(os.path.join(a.hf_dir, "oasst2_vi.jsonl")), 2)
    stats["alpaca_soan"] = add(mix, load_jsonl(os.path.join(a.hf_dir, "alpaca_soan_dich.jsonl")), 1, 400)
    stats["sea"] = add(mix, load_jsonl(os.path.join(a.hf_dir, "seadialogues_vi.jsonl")), 1, 900)
    legal = load_jsonl(os.path.join(a.hf_dir, "legal_vi.jsonl"))
    legal.sort(key=lambda r: 0 if r.get("qa_type") == "qa_practical" else 1)
    stats["legal"] = add(mix, legal, 1, 2000)
    stats["vimed"] = add(mix, load_jsonl(os.path.join(a.hf_dir, "vimed_thin.jsonl")), 1, 2500)
    sea_i = load_jsonl(os.path.join(a.hf_dir, "sea_instruct_vi.jsonl"))
    stats["sea_instruct"] = add(mix, sea_i, 1, 4000)
    adapt = load_jsonl(os.path.join(a.hf_dir, "adapted_vi.jsonl"))
    stats["adapt"] = add(mix, adapt, a.adapt_repeat, a.adapt_cap)
    stats["selfchat"] = add(mix, load_jsonl(os.path.join(a.hf_dir, "selfchat_vi.jsonl")), 1, 4000)
    stats["daily"] = add(mix, load_jsonl(os.path.join(a.hf_dir, "daily_vi.jsonl")), 1, 2009)
    stats["sharegpt4"] = add(mix, load_jsonl(os.path.join(a.hf_dir, "sharegpt4_vi.jsonl")), 1, 668)
    stats["safety"] = add(mix, load_jsonl(os.path.join(a.hf_dir, "safety_vi.jsonl")), 1, 1500)
    stats["openorca"] = add(mix, load_jsonl(os.path.join(a.hf_dir, "openorca_vi.jsonl")), 1, 3000)
    stats["pbgdpl"] = add(mix, load_jsonl(os.path.join(a.hf_dir, "pbgdpl_vi.jsonl")), 1, 3000)
    stats["legal_convo"] = add(mix, load_jsonl(os.path.join(a.hf_dir, "legal_convo_vi.jsonl")), 1, 3500)
    stats["ecom"] = add(mix, load_jsonl(os.path.join(a.hf_dir, "ecom_chat_vi.jsonl")), 1, 1482)
    stats["office"] = add(mix, load_jsonl(os.path.join(a.hf_dir, "office_vi.jsonl")), a.office_hf_repeat, 600)
    code_rows = []
    for r in load_jsonl(os.path.join(a.hf_dir, "adapted_code_vi.jsonl")):
        a_txt = (r["messages"][-1].get("content") or "") if r.get("messages") else ""
        blob = a_txt.lower()
        if len(a_txt) < 120:
            continue
        if not any(x in blob for x in ("import ", "def ", "pd.", "select ", "groupby", "read_csv", "read_excel")):
            continue
        code_rows.append(r)
    stats["code_adapt"] = add(mix, code_rows, a.code_repeat, 2000)
    stats["office_synth"] = add(mix, load_jsonl(os.path.join(a.hf_dir, "office_synth.jsonl")), 1, 10000)
    stats["code_synth"] = add(mix, load_jsonl(os.path.join(a.hf_dir, "code_synth.jsonl")), 1, 10000)
    stats["dich_synth"] = add(mix, load_jsonl(os.path.join(a.hf_dir, "dich_synth.jsonl")), 1, 10000)
    stats["office_long"] = add(mix, load_jsonl(os.path.join(a.hf_dir, "office_long.jsonl")), 1, 5000)
    stats["code_long"] = add(mix, load_jsonl(os.path.join(a.hf_dir, "code_long.jsonl")), 1, 5000)

    rng = random.Random(a.seed)
    rng.shuffle(mix)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fp:
        for r in mix:
            fp.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("gốc:", stats)
    print(f"→ {len(mix)} dòng  {a.out}")


if __name__ == "__main__":
    main()
