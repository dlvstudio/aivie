"""§TK_100 · M0 — GOM SFT-MIX TIẾNG VIỆT (chuẩn-hoá → lọc → dedup → cân tỉ-lệ).

Gộp nhiều nguồn chỉ-dẫn (Aya native, vi-alpaca, multi-turn, Bactrian-X, +tuỳ chọn
reasoning/tool-use) về MỘT định-dạng chat thống-nhất: mỗi dòng JSONL =
  {"source","n_turns","messages":[{"role":"system|user|assistant","content":...}]}
áp chat-template lúc SFT sau. Auto-detect schema (alpaca / sharegpt / aya / messages /
xlam) → bền với mọi repo. Không tốn GPU.

SMOKE:
    PYTHONPATH=src .venv/bin/python scripts/build_sft_mix.py \
        --sources aya vi-alpaca --limit 3000 --out data/sft_vi_smoke

THẬT (cân tỉ-lệ, thêm kỹ-năng):
    PYTHONPATH=src .venv/bin/python scripts/build_sft_mix.py \
        --sources aya vi-alpaca multiturn bactrian reasoning xlam-tool \
        --per-source-cap 60000 --out data/sft_vi
"""
import sys, os, re, json, time, argparse, hashlib
T0 = time.time()

# ── Nguồn: name → (repo, config, split, license, ⚠️GPT-lineage, lọc-ngôn-ngữ) ──────
SFT_SOURCES = {
    "aya":       ("CohereForAI/aya_dataset",                    None,  "train", "Apache-2.0", False, "Vietnamese"),
    "vi-alpaca": ("bkai-foundation-models/vi-alpaca",           None,  "train", "cc(⚠️GPT)",  True,  None),
    "vi-alpaca-io": ("bkai-foundation-models/vi-alpaca-input-output-format", None, "train", "cc(⚠️GPT)", True, None),
    "multiturn": ("5CD-AI/Vietnamese-Multi-turn-Chat-Alpaca",   None,  "train", "⚠️GPT",      True,  None),
    "bactrian":  ("MBZUAI/Bactrian-X",                          "vi",  "train", "dịch-máy",   False, None),
    "vi_instruct": ("tmnam20/vi_instruct",                      None,  "train", "mở",         False, None),
    # ── kỹ-năng (tiếng Anh — dạy ĐỊNH-DẠNG, bật bằng cờ; xem §13 Tầng 2) ──
    "reasoning": ("open-r1/Mixture-of-Thoughts",                None,  "train", "mở",         False, None),
    "xlam-tool": ("Salesforce/xlam-function-calling-60k",       None,  "train", "CC-BY(+dk)", False, None),
}


def clean(s):
    return re.sub(r"\s+", " ", str(s or "")).strip()


# ── AUTO-DETECT schema → list messages [{role,content}] (hoặc None nếu hỏng) ───────
def to_messages(row, lang=None):
    # aya: inputs/targets (+language)
    if "inputs" in row and "targets" in row:
        if lang and clean(row.get("language")) not in (lang, "") and clean(row.get("language_code")) not in ("vie", "vi"):
            return None
        u, o = clean(row["inputs"]), clean(row["targets"])
        return [{"role": "user", "content": u}, {"role": "assistant", "content": o}] if u and o else None
    # sharegpt: conversations [{from,value}]
    if "conversations" in row and isinstance(row["conversations"], list):
        role = {"human": "user", "user": "user", "gpt": "assistant", "assistant": "assistant",
                "system": "system", "tool": "assistant"}
        ms = [{"role": role.get(clean(t.get("from")).lower(), "user"), "content": clean(t.get("value"))}
              for t in row["conversations"] if clean(t.get("value"))]
        return ms if len(ms) >= 2 else None
    # messages đã chuẩn: [{role,content}]
    if "messages" in row and isinstance(row["messages"], list):
        ms = [{"role": clean(t.get("role")), "content": clean(t.get("content"))}
              for t in row["messages"] if clean(t.get("content"))]
        return ms if len(ms) >= 2 else None
    # xlam tool-use: query/tools/answers
    if "query" in row and "answers" in row:
        sys_m = f"Bạn có các công cụ (JSON): {clean(row.get('tools'))}" if row.get("tools") else None
        ms = ([{"role": "system", "content": sys_m}] if sys_m else []) + \
             [{"role": "user", "content": clean(row["query"])},
              {"role": "assistant", "content": clean(row["answers"])}]
        return ms if clean(row["query"]) and clean(row["answers"]) else None
    # alpaca: instruction/input/output
    if "instruction" in row and "output" in row:
        u = clean(row["instruction"])
        if clean(row.get("input")):
            u += "\n\n" + clean(row["input"])
        o = clean(row["output"])
        return [{"role": "user", "content": u}, {"role": "assistant", "content": o}] if u and o else None
    return None


def ok(ms, min_a, max_c):
    """Lọc chất-lượng cơ-bản: có cặp user→assistant, đáp đủ dài, không quá dài."""
    if not ms or ms[-1]["role"] != "assistant":
        return False
    if len(ms[-1]["content"]) < min_a:
        return False
    if sum(len(m["content"]) for m in ms) > max_c:
        return False
    if not any(m["role"] == "user" for m in ms):
        return False
    return True


def key_of(ms):
    u = next((m["content"] for m in ms if m["role"] == "user"), "")
    a = ms[-1]["content"]
    return hashlib.blake2b((u[:200] + "||" + a[:200]).encode(), digest_size=16).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", nargs="+", default=["aya", "vi-alpaca", "multiturn", "bactrian"],
                    help=f"chọn trong {list(SFT_SOURCES)}")
    ap.add_argument("--out", default="data/sft_vi")
    ap.add_argument("--limit", type=int, default=3000, help="đọc TỐI-ĐA mỗi nguồn (streaming)")
    ap.add_argument("--per-source-cap", type=int, default=100000, help="GIỮ tối-đa mỗi nguồn (cân tỉ-lệ)")
    ap.add_argument("--min-answer", type=int, default=20)
    ap.add_argument("--max-chars", type=int, default=16000)
    a = ap.parse_args()

    from datasets import load_dataset
    os.makedirs(a.out, exist_ok=True)
    fp = open(os.path.join(a.out, "sft_mix.jsonl"), "w", encoding="utf-8")
    seen = set()
    stats = {"per_source": {}, "dropped": {}, "total": 0, "turns_hist": {}}

    def drop(r):
        stats["dropped"][r] = stats["dropped"].get(r, 0) + 1

    for name in a.sources:
        if name not in SFT_SOURCES:
            print(f"⚠️ bỏ nguồn lạ: {name}", flush=True); continue
        repo, cfg, split, lic, gpt, lang = SFT_SOURCES[name]
        print(f"[{time.time()-T0:.0f}s] === {name}: {repo} (lic={lic}) ===", flush=True)
        try:
            ds = load_dataset(repo, cfg, split=split, streaming=True)
        except Exception as e:
            print(f"⚠️ KHÔNG mở được {name}: {str(e)[:140]}", flush=True); continue
        read = kept = 0
        for row in ds:
            if read >= a.limit or kept >= a.per_source_cap:
                break
            read += 1
            ms = to_messages(row, lang)
            if ms is None:
                drop("schema/lang"); continue
            if not ok(ms, a.min_answer, a.max_chars):
                drop("chất-lượng"); continue
            k = key_of(ms)
            if k in seen:
                drop("trùng"); continue
            seen.add(k)
            nt = sum(m["role"] == "user" for m in ms)
            fp.write(json.dumps({"source": name, "n_turns": nt, "messages": ms}, ensure_ascii=False) + "\n")
            stats["turns_hist"][nt] = stats["turns_hist"].get(nt, 0) + 1
            kept += 1
            if read % 2000 == 0:
                print(f"[{time.time()-T0:.0f}s]   {name}: đọc {read} · giữ {kept}", flush=True)
        stats["per_source"][name] = {"repo": repo, "license": lic, "gpt_lineage": gpt,
                                     "read": read, "kept": kept}
        stats["total"] += kept
        print(f"[{time.time()-T0:.0f}s] {name}: đọc {read} → giữ {kept}", flush=True)

    fp.close()
    stats["created"] = time.strftime("%Y-%m-%d %H:%M")
    with open(os.path.join(a.out, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(f"\n[{time.time()-T0:.0f}s] ===== XONG =====", flush=True)
    print(f"  tổng giữ {stats['total']} mẫu → {os.path.join(a.out,'sft_mix.jsonl')}", flush=True)
    print(f"  phân-bố lượt (n_turns→#): {dict(sorted(stats['turns_hist'].items()))}", flush=True)
    print(f"  loại: {stats['dropped']}", flush=True)
    gpt_src = [n for n, s in stats["per_source"].items() if s["gpt_lineage"]]
    if gpt_src:
        print(f"  ⚠️ nguồn GPT-lineage (rà ToS trước thương-mại): {gpt_src}", flush=True)


if __name__ == "__main__":
    main()
