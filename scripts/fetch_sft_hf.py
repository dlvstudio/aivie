"""Kéo nguồn HF chất lượng → JSONL messages (không ViQuAD / không alpaca bài tập).

    PYTHONPATH=src .venv/bin/python scripts/fetch_sft_hf.py
"""
import json, os, re, time, argparse

OUT = "data/sft/hf"


def ok_pair(u, a, min_u=8, min_a=12, max_c=6000):
    u, a = (u or "").strip(), (a or "").strip()
    if len(u) < min_u or len(a) < min_a:
        return False
    if len(u) + len(a) > max_c:
        return False
    return True


def write_jsonl(path, rows):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fp:
        for r in rows:
            fp.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  LƯU {path}: {len(rows)}", flush=True)


def fetch_aya(limit):
    from datasets import load_dataset
    print("=== Aya Dataset (lọc Vietnamese) ===", flush=True)
    ds = load_dataset("CohereLabs/aya_dataset", split="train", streaming=True)
    rows, seen = [], 0
    for row in ds:
        seen += 1
        lang = (row.get("language") or "") + " " + (row.get("language_code") or "")
        if "vietnamese" not in lang.lower() and "vie" not in lang.lower().split():
            continue
        u, a = row.get("inputs") or "", row.get("targets") or ""
        if not ok_pair(u, a):
            continue
        rows.append({
            "source": "aya-vi",
            "id": f"aya_{len(rows)}",
            "n_turns": 1,
            "messages": [
                {"role": "user", "content": u.strip()},
                {"role": "assistant", "content": a.strip()},
            ],
        })
        if limit and len(rows) >= limit:
            break
        if seen % 20000 == 0:
            print(f"  quét {seen} · giữ {len(rows)}", flush=True)
    write_jsonl(os.path.join(OUT, "aya_vi.jsonl"), rows)
    return len(rows)


def fetch_everyday(limit):
    from datasets import load_dataset
    print("=== everyday-conversations (seed Anh, chờ Gemma) ===", flush=True)
    ds = load_dataset("HuggingFaceTB/everyday-conversations-llama3.1-2k", split="train_sft")
    rows = []
    for i, row in enumerate(ds):
        if limit and i >= limit:
            break
        ms = row.get("messages") or []
        clean = []
        for m in ms:
            role, c = m.get("role"), (m.get("content") or "").strip()
            if role in ("user", "assistant") and c:
                clean.append({"role": role, "content": c})
        if len(clean) < 4:
            continue
        rows.append({
            "source": "everyday-en",
            "id": f"everyday_{i}",
            "topic": row.get("full_topic") or row.get("topic") or "",
            "n_turns": sum(m["role"] == "user" for m in clean),
            "messages": clean,
        })
    write_jsonl(os.path.join(OUT, "everyday_en.jsonl"), rows)
    return len(rows)


def fetch_oasst(limit_vi, limit_en_seed):
    from datasets import load_dataset
    print("=== OASST2 ===", flush=True)
    ds = load_dataset("OpenAssistant/oasst2", split="train")
    rows = list(ds)
    kids = {}
    for r in rows:
        pid = r.get("parent_id")
        if pid:
            kids.setdefault(pid, []).append(r)
    vi, en_seeds = [], []
    for r in rows:
        txt = (r.get("text") or "").strip()
        lang = (r.get("lang") or "").lower()
        role = r.get("role")
        if not txt or len(txt) < 8:
            continue
        if lang in ("vi", "vie", "vietnamese") and role == "prompter":
            ans = [k for k in kids.get(r["message_id"], [])
                   if k.get("role") == "assistant" and (k.get("text") or "").strip()]
            if not ans:
                continue
            a = max(ans, key=lambda k: len(k.get("text") or "")).get("text", "").strip()
            if ok_pair(txt, a, min_a=20):
                vi.append({
                    "source": "oasst2-vi",
                    "id": r["message_id"],
                    "n_turns": 1,
                    "messages": [
                        {"role": "user", "content": txt},
                        {"role": "assistant", "content": a},
                    ],
                })
        if (lang.startswith("en") and role == "prompter" and not r.get("parent_id")
                and 20 <= len(txt) <= 400 and len(en_seeds) < limit_en_seed):
            en_seeds.append({
                "source": "oasst2-en-seed",
                "id": r["message_id"],
                "prompt": txt,
            })
        if limit_vi and len(vi) >= limit_vi and len(en_seeds) >= limit_en_seed:
            break
    write_jsonl(os.path.join(OUT, "oasst2_vi.jsonl"), vi)
    write_jsonl(os.path.join(OUT, "oasst2_en_seeds.jsonl"), en_seeds)
    return len(vi), len(en_seeds)


def parse_sea_convo(text):
    """Tách 'Name: line' → messages xen kẽ user/assistant (chitchat, không hoàn hảo)."""
    if not text:
        return []
    parts = re.split(r"\*\*[^*]{1,80}:\*\*\s*", text.strip())
    parts = [p.strip() for p in parts if p.strip()]
    ms = []
    for i, p in enumerate(parts[:10]):
        p = re.sub(r"\s+", " ", p).strip()
        if len(p) < 8:
            continue
        ms.append({"role": "user" if i % 2 == 0 else "assistant", "content": p[:1200]})
    return ms if len(ms) >= 4 else []


def fetch_sea(limit):
    from datasets import load_dataset
    print("=== SEADialogues (lang=vie, Aya convo) ===", flush=True)
    ds = load_dataset("SEACrowd/SEADialogues", split="train")
    rows = []
    for i, row in enumerate(ds):
        lang = (row.get("lang") or "").lower()
        if lang not in ("vie", "vi", "vietnamese"):
            continue
        text = row.get("generated_convo_aya") or row.get("generated_convo_gpt4omini") or ""
        ms = parse_sea_convo(text)
        if not ms:
            continue
        rows.append({
            "source": "seadialogues-vi",
            "id": f"sea_{i}",
            "n_turns": sum(m["role"] == "user" for m in ms),
            "messages": ms,
        })
        if limit and len(rows) >= limit:
            break
    write_jsonl(os.path.join(OUT, "seadialogues_vi.jsonl"), rows)
    return len(rows)


def _as_messages(raw):
    import ast
    if isinstance(raw, str):
        raw = raw.strip()
        try:
            raw = json.loads(raw)
        except Exception:
            try:
                raw = ast.literal_eval(raw)
            except Exception:
                return []
    if not isinstance(raw, list):
        return []
    out = []
    for x in raw:
        if not isinstance(x, dict):
            continue
        role = (x.get("role") or x.get("from") or "").lower()
        if role in ("system",):
            continue
        if role in ("human", "prompter", "user"):
            role = "user"
        elif role in ("gpt", "bot", "assistant"):
            role = "assistant"
        c = (x.get("content") or x.get("value") or "").strip()
        if role in ("user", "assistant") and len(c) >= 4:
            out.append({"role": role, "content": c[:4000]})
    return out if len(out) >= 2 and out[-1]["role"] == "assistant" else []


VI_MARK = re.compile(
    r"[àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ]",
    re.I,
)


def fetch_legal(limit):
    from datasets import load_dataset
    print("=== duyet/vietnamese-legal-instruct (qa_practical/explain, cap) ===", flush=True)
    limit = limit or 2000
    keep = {
        "qa_practical": max(1, int(limit * 0.40)),
        "explain_simple": max(1, int(limit * 0.30)),
        "key_provisions": max(1, int(limit * 0.20)),
        "scope": max(1, int(limit * 0.10)),
    }
    ds = load_dataset("duyet/vietnamese-legal-instruct", split="train", streaming=True)
    rows, counts = [], {k: 0 for k in keep}
    seen = 0
    for row in ds:
        seen += 1
        qt = (row.get("qa_type") or "").strip()
        if qt not in keep or counts[qt] >= keep[qt]:
            if seen > 80000 and all(counts[k] >= keep[k] for k in keep):
                break
            continue
        ms = _as_messages(row.get("conversations"))
        if not ms:
            continue
        blob = " ".join(m["content"] for m in ms)
        if len(VI_MARK.findall(blob)) < 8:
            continue
        if len(ms[-1]["content"]) > 1800:
            continue
        rows.append({
            "source": "legal-vi",
            "id": f"legal_{row.get('source_id')}_{qt}_{len(rows)}",
            "qa_type": qt,
            "n_turns": sum(m["role"] == "user" for m in ms),
            "messages": ms,
        })
        counts[qt] += 1
        if limit and len(rows) >= limit:
            break
        if all(counts[k] >= keep[k] for k in keep):
            break
        if seen % 20000 == 0:
            print(f"  quét {seen} · giữ {len(rows)}", flush=True)
    write_jsonl(os.path.join(OUT, "legal_vi.jsonl"), rows)
    return len(rows)


def fetch_vimed(limit):
    from datasets import load_dataset
    print("=== ViMed-SFT (lát mỏng, giữ 'đi khám') ===", flush=True)
    try:
        ds = load_dataset("quannguyen204/medical_sft_crawl_vi_10k_v1", split="train")
    except Exception:
        ds = load_dataset("vlinhd11/medical_sft_crawl_vi_10k_v1", split="train")
    rows = []
    for i, row in enumerate(ds):
        ms = [m for m in _as_messages(row.get("messages")) if m["role"] in ("user", "assistant")]
        if len(ms) < 2:
            continue
        a = " ".join(m["content"] for m in ms if m["role"] == "assistant")
        if not re.search(r"khám|bác sĩ|cơ sở y tế|bệnh viện", a, re.I):
            continue
        if re.search(r"\b\d+\s*(mg|ml|viên)\b", a, re.I) and not re.search(r"không kê|không thay|bác sĩ", a, re.I):
            continue
        if len(a) > 2200:
            continue
        rows.append({
            "source": "vimed-thin",
            "id": f"vimed_{i}",
            "n_turns": sum(m["role"] == "user" for m in ms),
            "messages": ms,
        })
        if limit and len(rows) >= limit:
            break
    write_jsonl(os.path.join(OUT, "vimed_thin.jsonl"), rows)
    return len(rows)


def fetch_sea_instruct(limit):
    from datasets import load_dataset
    print("=== SEA-Instruct VN (bản ungated, lọc hỏi ngắn) ===", flush=True)
    ds = load_dataset("trannguyenquynhnhu/SEA-Instruct-2602-fine-tuned", split="train", streaming=True)
    rows, seen = [], 0
    skip_re = re.compile(r"```|viết hàm|leetcode|python code|chứng minh rằng", re.I)
    ask_re = re.compile(r"\?|làm sao|thế nào|có được không|tôi muốn|cho mình|giúp")
    for row in ds:
        seen += 1
        ms = _as_messages(row.get("messages"))
        if not ms:
            continue
        u = ms[0]["content"]
        blob = " ".join(m["content"] for m in ms)
        if len(VI_MARK.findall(blob)) < 10:
            continue
        if len(u) > 700 or skip_re.search(blob):
            continue
        if not ask_re.search(u):
            continue
        rows.append({
            "source": "sea-instruct-vi",
            "id": row.get("conversations_id") or f"seai_{len(rows)}",
            "n_turns": sum(m["role"] == "user" for m in ms),
            "messages": ms,
        })
        if limit and len(rows) >= limit:
            break
        if seen % 5000 == 0:
            print(f"  quét {seen} · giữ {len(rows)}", flush=True)
    write_jsonl(os.path.join(OUT, "sea_instruct_vi.jsonl"), rows)
    return len(rows)


def fetch_pbgdpl(limit):
    from datasets import load_dataset
    print("=== tmquan/pbgdpl-vn-legal-qna (Bộ Tư pháp) ===", flush=True)
    ds = load_dataset("tmquan/pbgdpl-vn-legal-qna", split="train")
    note = "\n\nNội dung mang tính tham khảo. Kiểm tra văn bản hiện hành hoặc cổng DVC trước khi làm hồ sơ."
    rows = []
    for i, row in enumerate(ds):
        u = (row.get("question_text") or row.get("title") or "").strip()
        a = (row.get("answer_text") or "").strip()
        if not ok_pair(u, a, min_u=20, min_a=40, max_c=5000):
            continue
        if len(a) > 2200:
            a = a[:2200].rsplit(" ", 1)[0] + "…"
        rows.append({
            "source": "pbgdpl-vi",
            "id": str(row.get("item_id") or f"pbg_{i}"),
            "n_turns": 1,
            "messages": [
                {"role": "user", "content": u},
                {"role": "assistant", "content": a + note},
            ],
        })
        if limit and len(rows) >= limit:
            break
    write_jsonl(os.path.join(OUT, "pbgdpl_vi.jsonl"), rows)
    return len(rows)


def fetch_legal_convo(limit):
    from datasets import load_dataset
    print("=== chillies/vn-legal-conversation ===", flush=True)
    ds = load_dataset("chillies/vn-legal-conversation", split="train")
    sit = re.compile(r"tôi|mình|mẹ|bố|chồng|vợ|em |anh ", re.I)
    rows = []
    for i, row in enumerate(ds):
        u = (row.get("Question") or "").strip()
        a = (row.get("Answer") or "").strip()
        if not ok_pair(u, a, min_u=24, min_a=50, max_c=4500):
            continue
        if not sit.search(u):
            continue
        if len(a) > 2000:
            a = a[:2000].rsplit(" ", 1)[0] + "…"
        rows.append({
            "source": "legal-convo-vi",
            "id": f"lc_{i}",
            "n_turns": 1,
            "messages": [
                {"role": "user", "content": u},
                {"role": "assistant", "content": a},
            ],
        })
        if limit and len(rows) >= limit:
            break
    write_jsonl(os.path.join(OUT, "legal_convo_vi.jsonl"), rows)
    return len(rows)


def fetch_ecom_chat():
    from datasets import load_dataset
    print("=== 5CD-AI/Vietnamese-Ecommerce-Multi-turn-Chat ===", flush=True)
    ds = load_dataset("5CD-AI/Vietnamese-Ecommerce-Multi-turn-Chat", split="train")
    rows = []
    for i, row in enumerate(ds):
        ms = _as_messages(row.get("conversations"))
        if len(ms) < 2:
            continue
        rows.append({
            "source": "ecom-chat-vi",
            "id": str(row.get("id") if row.get("id") is not None else f"ecom_{i}"),
            "n_turns": sum(m["role"] == "user" for m in ms),
            "messages": ms,
        })
    write_jsonl(os.path.join(OUT, "ecom_chat_vi.jsonl"), rows)
    return len(rows)


TOOLISH = re.compile(
    r"tool_calls|function_call|create_table|\"name\"\s*:\s*\"create_|xuất ra json|trả về json|output định dạng json",
    re.I,
)
OFFICE_OK = re.compile(
    r"\bexcel\b|google sheets|powerpoint|vlookup|xlookup|sumif|countif|\bpivot\b|"
    r"macro|openpyxl|xlsx|cột [a-z]\b",
    re.I,
)
CODE_KEEP = re.compile(
    r"pandas|openpyxl|xlrd|xlsxwriter|\.xlsx\b|\.csv\b|spreadsheet|\bexcel\b|"
    r"read_csv|to_csv|to_excel|SUMIF|VLOOKUP|google.?sheets|"
    r"GROUP BY|SELECT\s+.+\s+FROM|git (merge|rebase|conflict|stash|checkout)|"
    r"smtplib|sqlite3|sqlalchemy|read_sql|to_sql|matplotlib",
    re.I,
)
CODE_SKIP = re.compile(
    r"leetcode|binary tree|linked list|two sum|palindrome|dynamic programming|"
    r"depth-first|breadth-first|\bO\(n|\bimplement a (class|heap|trie)\b|"
    r"n-queens|sudoku solver",
    re.I,
)


def _flatten_office_messages(ms):
    out = []
    for m in ms:
        c = (m.get("content") or "").strip()
        if m.get("role") == "assistant" and c.startswith("{"):
            try:
                o = json.loads(c)
            except Exception:
                return []
            if o.get("loaiVanBan") or o.get("chuDeIT"):
                return []
            formula = (o.get("congThuc_Hoac_Code") or "").strip()
            bits = o.get("giaiThichChiTiet") or o.get("noiDungChiTiet") or []
            if isinstance(bits, list):
                bits = " ".join(str(x) for x in bits)
            bits = (bits or "").strip()
            note = o.get("luuYSuDung") or []
            if isinstance(note, list):
                note = " ".join(str(x) for x in note)
            parts = [p for p in (bits, f"Công thức: {formula}" if formula else "", note) if p]
            if not parts:
                return []
            c = "\n".join(parts)
        out.append({"role": m["role"], "content": c[:4000]})
    return out if len(out) >= 2 and out[-1]["role"] == "assistant" else []


def _row_messages(row):
    for key in ("messages", "conversations", "conversation"):
        if row.get(key):
            ms = _as_messages(row.get(key))
            if ms:
                return ms
    u = (row.get("instruction") or row.get("input") or row.get("prompt") or row.get("question") or "").strip()
    a = (row.get("output") or row.get("response") or row.get("answer") or row.get("targets") or "").strip()
    if len(u) >= 8 and len(a) >= 12:
        return [{"role": "user", "content": u[:4000]}, {"role": "assistant", "content": a[:4000]}]
    return []


def fetch_office_sft(limit):
    from datasets import load_dataset
    print("=== linhhuonglinux-office-dataset-v3 (bỏ tool JSON) ===", flush=True)
    ds = None
    err = None
    for args in (
        ("linhhuonglinux/linhhuonglinux-office-dataset-v3", "sft"),
        ("linhhuonglinux/linhhuonglinux-office-dataset-v3",),
    ):
        try:
            ds = load_dataset(*args, split="train")
            break
        except Exception as e:
            err = e
            print(f"  thử {args}: {e}", flush=True)
    if ds is None:
        raise RuntimeError(f"không load được office-dataset-v3: {err}")
    rows = []
    for i, row in enumerate(ds):
        if any(row.get(k) for k in ("tool_calls", "tools", "function_call")):
            continue
        ms = _flatten_office_messages(_row_messages(row))
        if len(ms) < 2:
            continue
        blob = " ".join(m["content"] for m in ms)
        if ms[-1]["content"].lstrip().startswith("{") or TOOLISH.search(blob):
            continue
        if len(VI_MARK.findall(blob)) < 6:
            continue
        if not OFFICE_OK.search(blob):
            continue
        if re.search(r"quick sort|cấu trúc dữ liệu|leetcode", blob, re.I):
            continue
        rows.append({
            "source": "office-vi",
            "id": str(row.get("id") or f"off_{i}"),
            "n_turns": sum(m["role"] == "user" for m in ms),
            "messages": ms,
        })
        if limit and len(rows) >= limit:
            break
    write_jsonl(os.path.join(OUT, "office_vi.jsonl"), rows)
    return len(rows)


def fetch_magicoder_seeds(limit, out_name="magicoder_en_seeds.jsonl", skip_ids=None):
    from datasets import load_dataset
    print("=== Magicoder OSS-Instruct (seed Python/SQL/Excel, bỏ leetcode) ===", flush=True)
    skip_ids = skip_ids or set()
    ds = load_dataset("ise-uiuc/Magicoder_oss_instruct_75k", split="train")
    rows = []
    for i, row in enumerate(ds):
        rid = f"mgc_{i}"
        if rid in skip_ids:
            continue
        lang = (row.get("lang") or row.get("language") or "").lower()
        q = (row.get("problem") or row.get("instruction") or row.get("query") or "").strip()
        a = (row.get("solution") or row.get("response") or row.get("output") or "").strip()
        blob = f"{lang} {q} {a}"
        if CODE_SKIP.search(blob):
            continue
        if not CODE_KEEP.search(blob):
            continue
        if not ok_pair(q, a, min_u=20, min_a=40, max_c=5000):
            continue
        rows.append({
            "source": "magicoder-en",
            "id": rid,
            "lang": lang or "code",
            "prompt": q,
            "solution": a[:3500],
        })
        if limit and len(rows) >= limit:
            break
    write_jsonl(os.path.join(OUT, out_name), rows)
    return len(rows)


def fetch_selfchat(limit):
    from datasets import load_dataset
    print("=== BKAI vi-self-chat (đa lượt) ===", flush=True)
    ds = load_dataset("bkai-foundation-models/vi-self-chat-sharegpt-format", split="train")
    rows = []
    for i, row in enumerate(ds):
        ms = _as_messages(row.get("conversations"))
        if len(ms) < 4:
            continue
        blob = " ".join(m["content"] for m in ms)
        if len(VI_MARK.findall(blob)) < 8:
            continue
        if len(blob) > 5000:
            continue
        rows.append({
            "source": "bkai-selfchat",
            "id": str(row.get("id") or f"sc_{i}"),
            "n_turns": sum(m["role"] == "user" for m in ms),
            "messages": ms,
        })
        if limit and len(rows) >= limit:
            break
    write_jsonl(os.path.join(OUT, "selfchat_vi.jsonl"), rows)
    return len(rows)


def fetch_daily_vi():
    from datasets import load_dataset
    print("=== hoanghai2110/vietnamese-dataset ===", flush=True)
    ds = load_dataset("hoanghai2110/vietnamese-dataset", split="train")
    rows = []
    for i, row in enumerate(ds):
        ms = _as_messages(row.get("messages"))
        if len(ms) < 2:
            continue
        rows.append({
            "source": "daily-vi",
            "id": f"daily_{i}",
            "n_turns": sum(m["role"] == "user" for m in ms),
            "messages": ms,
        })
    write_jsonl(os.path.join(OUT, "daily_vi.jsonl"), rows)
    return len(rows)


def fetch_sharegpt4():
    import ast
    from datasets import load_dataset
    print("=== kimnt93/viet-sharegpt4 ===", flush=True)
    ds = load_dataset("kimnt93/viet-sharegpt4", split="train")
    rows = []
    for i, row in enumerate(ds):
        hist = row.get("history") or []
        if isinstance(hist, str):
            try:
                hist = ast.literal_eval(hist)
            except Exception:
                hist = []
        ms = []
        for pair in hist:
            if isinstance(pair, (list, tuple)) and len(pair) >= 2:
                u, a = str(pair[0]).strip(), str(pair[1]).strip()
                if u and a:
                    ms.append({"role": "user", "content": u[:2000]})
                    ms.append({"role": "assistant", "content": a[:2500]})
        instr = (row.get("instruction") or "").strip()
        extra = (row.get("input") or "").strip()
        out = (row.get("output") or "").strip()
        u = (instr + ("\n" + extra if extra else "")).strip()
        if u and out:
            ms.append({"role": "user", "content": u[:2000]})
            ms.append({"role": "assistant", "content": out[:2500]})
        if len(ms) < 2 or ms[-1]["role"] != "assistant":
            continue
        blob = " ".join(m["content"] for m in ms)
        if len(VI_MARK.findall(blob)) < 6:
            continue
        rows.append({
            "source": "sharegpt4-vi",
            "id": f"sg4_{i}",
            "n_turns": sum(m["role"] == "user" for m in ms),
            "messages": ms,
        })
    write_jsonl(os.path.join(OUT, "sharegpt4_vi.jsonl"), rows)
    return len(rows)


def fetch_safety(limit):
    from datasets import load_dataset
    print("=== vlinhd11/vietnamese-sft-10k (an toàn/từ chối) ===", flush=True)
    ds = load_dataset("vlinhd11/vietnamese-sft-10k", split="train")
    rows = []
    for i, row in enumerate(ds):
        u, a = (row.get("prompt") or "").strip(), (row.get("response") or "").strip()
        if not ok_pair(u, a, min_a=20):
            continue
        rows.append({
            "source": "safety-vi",
            "id": f"safe_{i}",
            "n_turns": 1,
            "messages": [
                {"role": "user", "content": u},
                {"role": "assistant", "content": a},
            ],
        })
        if limit and len(rows) >= limit:
            break
    write_jsonl(os.path.join(OUT, "safety_vi.jsonl"), rows)
    return len(rows)


def fetch_openorca(limit):
    from datasets import load_dataset
    print("=== vilm/OpenOrca-Viet (cap, bỏ toán/FLAN dài) ===", flush=True)
    ds = load_dataset("vilm/OpenOrca-Viet", split="train", streaming=True)
    skip = re.compile(r"giải .*phương trình|cho a\.|leetcode|```", re.I)
    rows, seen = [], 0
    for row in ds:
        seen += 1
        instr = (row.get("instruction") or "").strip()
        inp = (row.get("input") or "").strip()
        out = (row.get("output") or "").strip()
        out = out.replace("<|end_of_text|>", "").strip()
        u = inp if inp else instr
        if instr and inp and instr.lower().startswith("bạn là"):
            u = inp
        if not ok_pair(u, out, min_a=20, max_c=3500):
            continue
        if skip.search(u) or skip.search(out):
            continue
        if len(VI_MARK.findall(u + out)) < 8:
            continue
        rows.append({
            "source": "openorca-vi",
            "id": f"orca_{len(rows)}",
            "n_turns": 1,
            "messages": [
                {"role": "user", "content": u},
                {"role": "assistant", "content": out[:2500]},
            ],
        })
        if limit and len(rows) >= limit:
            break
        if seen % 20000 == 0:
            print(f"  quét {seen} · giữ {len(rows)}", flush=True)
    write_jsonl(os.path.join(OUT, "openorca_vi.jsonl"), rows)
    return len(rows)


def filter_alpaca():
    src = "data/sft/sft_train.jsonl"
    if not os.path.exists(src):
        return 0
    print("=== lọc vi-alpaca soạn/dịch (bỏ wiki RC) ===", flush=True)
    keep_re = re.compile(r"soạn|email|dịch sang|dịch ra|viết thư|tin nhắn|teams", re.I)
    skip_re = re.compile(r"đoạn văn|dựa vào|ngữ cảnh", re.I)
    rows = []
    for line in open(src, encoding="utf-8"):
        r = json.loads(line)
        if r.get("source") != "vi_alpaca":
            continue
        q = re.sub(r"^Hỏi:\s*", "", r.get("prompt") or "").strip()
        a = (r.get("answer") or "").strip()
        if skip_re.search(q) or not keep_re.search(q):
            continue
        if not ok_pair(q, a, min_a=30):
            continue
        rows.append({
            "source": "alpaca-soan-dich",
            "id": f"alpaca_{len(rows)}",
            "n_turns": 1,
            "messages": [
                {"role": "user", "content": q},
                {"role": "assistant", "content": a},
            ],
        })
    write_jsonl(os.path.join(OUT, "alpaca_soan_dich.jsonl"), rows)
    return len(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aya-limit", type=int, default=0)
    ap.add_argument("--everyday-limit", type=int, default=0)
    ap.add_argument("--oasst-vi", type=int, default=0)
    ap.add_argument("--oasst-en-seeds", type=int, default=800)
    ap.add_argument("--sea-limit", type=int, default=0)
    ap.add_argument("--legal-limit", type=int, default=400)
    ap.add_argument("--vimed-limit", type=int, default=200)
    ap.add_argument("--sea-instruct-limit", type=int, default=600)
    ap.add_argument("--extra-only", action="store_true",
                    help="chỉ kéo legal / vimed / SEA-Instruct (bỏ Aya/OASST đã có)")
    ap.add_argument("--round2", action="store_true",
                    help="self-chat / daily / sharegpt4 / safety / OpenOrca")
    ap.add_argument("--round3", action="store_true",
                    help="pbgdpl / legal-convo / ecommerce chat")
    ap.add_argument("--office-code", action="store_true",
                    help="office-v3 (bỏ tool) + Magicoder seeds")
    ap.add_argument("--magicoder-more", action="store_true",
                    help="thêm seed Magicoder, bỏ id đã có")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    if a.round2:
        n_sc = n_daily = n_sg = n_safe = n_or = 0
        try:
            n_sc = fetch_selfchat(5000)
        except Exception as e:
            print("⚠ selfchat:", e, flush=True)
        try:
            n_daily = fetch_daily_vi()
        except Exception as e:
            print("⚠ daily:", e, flush=True)
        try:
            n_sg = fetch_sharegpt4()
        except Exception as e:
            print("⚠ sharegpt4:", e, flush=True)
        try:
            n_safe = fetch_safety(2000)
        except Exception as e:
            print("⚠ safety:", e, flush=True)
        try:
            n_or = fetch_openorca(4000)
        except Exception as e:
            print("⚠ openorca:", e, flush=True)
        print(f"\n[{time.time()-t0:.0f}s] selfchat={n_sc} daily={n_daily} sg4={n_sg} safety={n_safe} orca={n_or}", flush=True)
        return
    if a.round3:
        n_pb = n_lc = n_ec = 0
        try:
            n_pb = fetch_pbgdpl(3500)
        except Exception as e:
            print("⚠ pbgdpl:", e, flush=True)
        try:
            n_lc = fetch_legal_convo(4000)
        except Exception as e:
            print("⚠ legal-convo:", e, flush=True)
        try:
            n_ec = fetch_ecom_chat()
        except Exception as e:
            print("⚠ ecom:", e, flush=True)
        print(f"\n[{time.time()-t0:.0f}s] pbgdpl={n_pb} legal_convo={n_lc} ecom={n_ec}", flush=True)
        return
    if a.office_code:
        n_off = n_mgc = 0
        try:
            n_off = fetch_office_sft(800)
        except Exception as e:
            print("⚠ office:", e, flush=True)
        try:
            n_mgc = fetch_magicoder_seeds(800)
        except Exception as e:
            print("⚠ magicoder:", e, flush=True)
        print(f"\n[{time.time()-t0:.0f}s] office={n_off} magicoder_seeds={n_mgc}", flush=True)
        return
    if a.magicoder_more:
        skip = set()
        for p in (
            os.path.join(OUT, "magicoder_en_seeds.jsonl"),
            os.path.join(OUT, "adapted_code_vi.jsonl"),
        ):
            if not os.path.exists(p):
                continue
            for line in open(p, encoding="utf-8"):
                if line.strip():
                    skip.add(json.loads(line).get("id"))
        n_mgc = fetch_magicoder_seeds(800, "magicoder_en_seeds_more.jsonl", skip)
        print(f"\n[{time.time()-t0:.0f}s] magicoder_more={n_mgc} skip={len(skip)}", flush=True)
        return
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    if a.extra_only:
        n_legal = n_vimed = n_seai = 0
        try:
            n_legal = fetch_legal(a.legal_limit or None)
        except Exception as e:
            print("⚠ legal:", e, flush=True)
        try:
            n_vimed = fetch_vimed(a.vimed_limit or None)
        except Exception as e:
            print("⚠ vimed:", e, flush=True)
        try:
            n_seai = fetch_sea_instruct(a.sea_instruct_limit or None)
        except Exception as e:
            print("⚠ sea-instruct:", e, flush=True)
        print(f"\n[{time.time()-t0:.0f}s] legal={n_legal} vimed={n_vimed} sea_instruct={n_seai}", flush=True)
        return
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    aya_path = os.path.join(OUT, "aya_vi.jsonl")
    if os.path.exists(aya_path):
        n_aya = sum(1 for _ in open(aya_path, encoding="utf-8"))
        print(f"=== Aya đã có {n_aya} ===", flush=True)
    else:
        n_aya = fetch_aya(a.aya_limit or None)
    n_ev = fetch_everyday(a.everyday_limit or None)
    try:
        n_vi, n_en = fetch_oasst(a.oasst_vi or 10**9, a.oasst_en_seeds)
    except Exception as e:
        print("⚠ OASST2:", e, flush=True)
        n_vi, n_en = 0, 0
    try:
        n_sea = fetch_sea(a.sea_limit or None)
    except Exception as e:
        print("⚠ SEADialogues:", e, flush=True)
        n_sea = 0
    n_al = filter_alpaca()
    n_legal = n_vimed = n_seai = 0
    try:
        n_legal = fetch_legal(a.legal_limit or None)
    except Exception as e:
        print("⚠ legal:", e, flush=True)
    try:
        n_vimed = fetch_vimed(a.vimed_limit or None)
    except Exception as e:
        print("⚠ vimed:", e, flush=True)
    try:
        n_seai = fetch_sea_instruct(a.sea_instruct_limit or None)
    except Exception as e:
        print("⚠ sea-instruct:", e, flush=True)
    print(
        f"\n[{time.time()-t0:.0f}s] aya={n_aya} everyday={n_ev} oasst_vi={n_vi} "
        f"oasst_en_seed={n_en} sea={n_sea} alpaca_soan={n_al} "
        f"legal={n_legal} vimed={n_vimed} sea_instruct={n_seai}",
        flush=True,
    )


if __name__ == "__main__":
    main()
