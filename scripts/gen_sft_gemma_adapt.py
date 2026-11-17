"""Gemma: hội thoại/hỏi Anh → tiếng Việt (user tự nhiên, assistant trợ lý VN).

    python3 scripts/gen_sft_gemma_adapt.py \\
        --everyday data/sft/hf/everyday_en.jsonl \\
        --oasst-seeds data/sft/hf/oasst2_en_seeds.jsonl \\
        --out data/sft/hf/everyday_vi.jsonl
"""
import json, time, argparse, urllib.request, os, re

OLLAMA = "http://127.0.0.1:11434/api/chat"
TEACHER = "gemma4:26b-a4b-it-qat"
SYS = (
    "Bạn chuyển nội dung sang hội thoại trợ lý tiếng Việt. "
    "User: người Việt, giọng tự nhiên — không dịch máy, không mở bằng Hi/Hello. "
    "Assistant: ngắn, hữu ích. Bối cảnh Việt khi hợp lý (đừng mặc định New York, IRS, 911). "
    "CẤM bịa lệ phí/hotline/địa chỉ. Không chắc thì nói kiểm tra nguồn chính thức. "
    "Chỉ in JSON hợp lệ, không markdown."
)


def ollama(user, num_predict=700):
    body = {
        "model": TEACHER,
        "messages": [{"role": "system", "content": SYS}, {"role": "user", "content": user}],
        "stream": False,
        "think": False,
        "options": {"temperature": 0.4, "num_predict": num_predict},
    }
    r = urllib.request.urlopen(
        urllib.request.Request(OLLAMA, data=json.dumps(body).encode(),
                               headers={"Content-Type": "application/json"}),
        timeout=180,
    )
    txt = ((json.load(r).get("message") or {}).get("content") or "").strip()
    m = re.search(r"```(?:json)?\s*([\s\S]*?)```", txt)
    if m:
        txt = m.group(1).strip()
    else:
        txt = re.sub(r"^```(?:json)?\s*|\s*```$", "", txt).strip()
    return txt


# Tiếng Việt dùng Latin + dấu. Đếm ASCII letter sẽ ~0.8 dù câu đúng tiếng Việt.
VI_MARK = re.compile(
    r"[àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ]",
    re.I,
)
EN_HINT = re.compile(
    r"\b(the|you|your|what|how|please|thanks|hello|could|would)\b", re.I
)


def looks_vietnamese(text):
    marks = len(VI_MARK.findall(text))
    if marks >= 3:
        return True
    # câu ngắn toàn dấu ít (vd. "Ok.") — vẫn nhận nếu gần như không có từ Anh
    return marks >= 1 and len(EN_HINT.findall(text)) <= 1


def _loads(txt):
    txt = (txt or "").strip()
    txt = re.sub(r",\s*([}\]])", r"\1", txt)
    for blob in (txt,):
        try:
            return json.loads(blob)
        except Exception:
            pass
    m = re.search(r"\{[\s\S]*\}", txt)
    if m:
        try:
            return json.loads(re.sub(r",\s*([}\]])", r"\1", m.group(0)))
        except Exception:
            pass
    m = re.search(r"\[[\s\S]*\]", txt)
    if m:
        try:
            return json.loads(re.sub(r",\s*([}\]])", r"\1", m.group(0)))
        except Exception:
            pass
    return None


def _normalize_ms(o):
    if isinstance(o, dict):
        o = o.get("messages") or o.get("conversation") or o.get("hội thoại")
    if not isinstance(o, list) or len(o) < 2:
        return None
    out = []
    for x in o:
        if not isinstance(x, dict):
            continue
        role = (x.get("role") or x.get("from") or "").lower()
        if role in ("human", "prompter", "user", "người dùng"):
            role = "user"
        elif role in ("gpt", "bot", "assistant", "trợ lý"):
            role = "assistant"
        c = (x.get("content") or x.get("value") or x.get("text") or "").strip()
        if role in ("user", "assistant") and len(c) >= 4:
            out.append({"role": role, "content": c})
    if len(out) < 2 or out[-1]["role"] != "assistant":
        return None
    blob = " ".join(m["content"] for m in out)
    if not looks_vietnamese(blob):
        return None
    return out


def extract_messages_loose(txt):
    """Gemma hay nhét dấu \" vào giữa câu → JSON gãy. Lấy role/content thủ công."""
    out = []
    for m in re.finditer(r'"role"\s*:\s*"(user|assistant)"', txt):
        rest = txt[m.end():]
        cm = re.search(r'"content"\s*:\s*"', rest)
        if not cm:
            continue
        i = cm.end()
        buf = []
        while i < len(rest):
            c = rest[i]
            if c == "\\" and i + 1 < len(rest):
                buf.append(rest[i:i + 2])
                i += 2
                continue
            if c == '"':
                j = i + 1
                while j < len(rest) and rest[j] in " \t\r\n":
                    j += 1
                if j >= len(rest) or rest[j] in ",}":
                    break
                buf.append('"')
                i += 1
                continue
            buf.append(c)
            i += 1
        raw = "".join(buf)
        try:
            content = json.loads('"' + raw.replace('"', '\\"') + '"')
        except Exception:
            content = raw.replace("\\n", "\n").replace('\\"', '"').strip()
        content = content.strip()
        if len(content) >= 4:
            out.append({"role": m.group(1), "content": content})
    if len(out) >= 2 and out[-1]["role"] == "assistant":
        blob = " ".join(m["content"] for m in out)
        if looks_vietnamese(blob):
            return out
    return None


def parse_messages(txt):
    o = _loads(txt)
    ms = _normalize_ms(o)
    if ms:
        return ms
    # Gemma đôi khi in User:/Assistant: thay vì JSON
    lines = re.split(r"\n(?=(?:User|Assistant|Người dùng|Trợ lý)\s*:)", txt)
    out = []
    for line in lines:
        m = re.match(r"(User|Assistant|Người dùng|Trợ lý)\s*:\s*(.*)", line, re.S)
        if not m:
            continue
        role = "user" if m.group(1) in ("User", "Người dùng") else "assistant"
        c = m.group(2).strip().strip('"')
        if len(c) >= 4:
            out.append({"role": role, "content": c})
    if len(out) >= 2 and out[-1]["role"] == "assistant":
        blob = " ".join(m["content"] for m in out)
        if looks_vietnamese(blob):
            return out
    return extract_messages_loose(txt)


def load_done(path):
    done = set()
    if not os.path.exists(path):
        return done
    for line in open(path, encoding="utf-8"):
        try:
            done.add(json.loads(line).get("id"))
        except Exception:
            pass
    return done


def dump_en_dialog(ms):
    lines = []
    for m in ms:
        who = "User" if m["role"] == "user" else "Assistant"
        if m["content"].lower() in ("hi", "hello", "hey", "hey!", "hi there",
                                    "hello! how can i help you today?"):
            continue
        lines.append(f"{who}: {m['content']}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--everyday", default="data/sft/hf/everyday_en.jsonl")
    ap.add_argument("--oasst-seeds", default="data/sft/hf/oasst2_en_seeds.jsonl")
    ap.add_argument("--code-seeds", default="")
    ap.add_argument("--out", default="data/sft/hf/adapted_vi.jsonl")
    ap.add_argument("--n-everyday", type=int, default=0)
    ap.add_argument("--n-oasst", type=int, default=400)
    ap.add_argument("--n-code", type=int, default=0)
    a = ap.parse_args()

    jobs = []
    if os.path.exists(a.everyday):
        ev = [json.loads(l) for l in open(a.everyday, encoding="utf-8") if l.strip()]
        if a.n_everyday:
            ev = ev[: a.n_everyday]
        for r in ev:
            blob = dump_en_dialog(r.get("messages") or [])
            if len(blob) < 40:
                continue
            prompt = (
                f"Chủ đề: {r.get('topic') or 'đời thường'}\n"
                "Viết lại hội thoại sau thành tiếng Việt, cùng số lượt (bỏ câu chào Hi).\n"
                'JSON: {"messages":[{"role":"user","content":"..."},{"role":"assistant","content":"..."}]}\n\n'
                + blob
            )
            jobs.append((r["id"], "everyday-vi", prompt, 1400))
    if os.path.exists(a.oasst_seeds):
        seeds = [json.loads(l) for l in open(a.oasst_seeds, encoding="utf-8") if l.strip()]
        if a.n_oasst:
            seeds = seeds[: a.n_oasst]
        for r in seeds:
            q = (r.get("prompt") or "").strip()
            prompt = (
                "Người dùng hỏi (gốc Anh). Viết 1 lượt user tiếng Việt tự nhiên "
                "(đặt vào Việt Nam nếu hợp lý) rồi 1 đáp assistant.\n"
                'JSON: {"messages":[{"role":"user","content":"..."},{"role":"assistant","content":"..."}]}\n\n'
                f"Original: {q}"
            )
            jobs.append((r["id"], "oasst-vi-adapt", prompt, 400))
    if a.code_seeds and os.path.exists(a.code_seeds):
        seeds = [json.loads(l) for l in open(a.code_seeds, encoding="utf-8") if l.strip()]
        if a.n_code:
            seeds = seeds[: a.n_code]
        for r in seeds:
            q = (r.get("prompt") or "").strip()
            sol = (r.get("solution") or "")[:1800]
            prompt = (
                "Người Việt hỏi tình huống Excel/SQL/Python ở công ty. "
                "Viết 1 lượt user tự nhiên (đặt VN nếu hợp lý) rồi 1 đáp assistant: "
                "công thức hoặc code ngắn, giải thích 3–6 câu, hỏi thiếu file/phiên bản. "
                "Giữ đúng ý bài gốc, đừng bịa API. Không markdown tiêu đề dài.\n"
                'JSON: {"messages":[{"role":"user","content":"..."},{"role":"assistant","content":"..."}]}\n\n'
                f"Original:\n{q}\n\nHint solution:\n{sol}"
            )
            jobs.append((r["id"], "code-vi-adapt", prompt, 900))

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    done = load_done(a.out)
    jobs = [j for j in jobs if j[0] not in done]
    T0 = time.time()
    print(f"[0s] {len(jobs)} việc · bỏ {len(done)} đã có · {TEACHER}", flush=True)
    fp = open(a.out, "a", encoding="utf-8")
    ok = fail = 0
    for i, (rid, src, prompt, npred) in enumerate(jobs):
        t1 = time.time()
        try:
            ms = parse_messages(ollama(prompt, npred))
            if not ms:
                raise RuntimeError("JSON/vi yếu")
            rec = {
                "source": src,
                "id": rid,
                "n_turns": sum(m["role"] == "user" for m in ms),
                "sec": round(time.time() - t1, 2),
                "messages": ms,
            }
            fp.write(json.dumps(rec, ensure_ascii=False) + "\n"); fp.flush()
            ok += 1
            flag = "ok"
        except Exception as e:
            fail += 1
            flag = f"FAIL {e}"
        print(f"[{time.time()-T0:.0f}s] {i+1}/{len(jobs)} {flag} {time.time()-t1:.1f}s  {rid}", flush=True)
    fp.close()
    print(f"[{time.time()-T0:.0f}s] LƯU {a.out}: ok {ok} · fail {fail}", flush=True)


if __name__ == "__main__":
    main()
