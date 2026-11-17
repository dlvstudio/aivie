"""TK_108 C7 — ĐỀ THỰC TẾ nhiều ràng buộc, tiếng Việt, mọi mẫu KIỂM BẰNG CHẠY THẬT (thầy = vLLM kiểu OpenAI trên box).

Vì sao: cx3b đạt MBPP 23,3 / tool 55 nhưng trượt hết 8 đề thực tế (TK_108 §13): model chỉ giỏi khuôn "đề ngắn tiếng Anh → 1 hàm".
Mỗi mẫu đi 2 lớp kiểm:
  1) thầy (nhiệt độ cao, kịch bản ngẫu nhiên: lĩnh vực × kiểu bài × độ khó) viết ĐỀ như người dùng thật + lời giải mẫu + test;
     lời giải mẫu phải QUA test (SQL: chạy ra kết quả khác rỗng; Excel: công thức tính được trên bảng);
  2) thầy giải LẠI ĐỘC LẬP chỉ từ đề, giọng trợ lý (giải thích ngắn + code) → code phải QUA đúng bộ test đó
     (SQL: trùng kết quả; Excel: trùng giá trị). Câu trả lời lớp 2 = mục tiêu SFT. Test sai/đề mơ hồ bị loại ở lớp 2.
Kiểu bài: py_func (nghiệp vụ) · py_data (list dict / CSV) · py_text (chuỗi tiếng Việt, regex, định dạng) · py_debug (lỗi kinh điển)
          · py_class (lớp có trạng thái) · sql (SQLite: JOIN/GROUP BY/HAVING/cửa sổ) · excel (công thức ô H2) · js (Node, assert)
Lệnh:
  gen   python scripts/c7_real.py gen --keep 10000 [--split train|eval] → data/c7/<split>.jsonl (chạy lại được, nối tiếp)
  eval  python scripts/c7_real.py eval --url http://localhost:8881 --tag cx3b → out/code_eval/c7_<tag>.jsonl + bảng theo kiểu
Nhật ký thầy đầy đủ (cả lượt hỏng) → data/c7/teacher_log/<split>.jsonl
"""
from __future__ import annotations
import argparse, hashlib, json, os, random, re, sqlite3, sys, threading, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(__file__))
import code_agent as CA

OUT = "data/c7"
A = None
_LOG, _LK = None, threading.Lock()

DOMAINS = [
    "bán lẻ / hoá đơn bán hàng", "thương mại điện tử (giỏ hàng, mã giảm giá, phí ship)", "kế toán / thuế GTGT / thuế TNCN",
    "nhân sự / chấm công / tính lương, phụ cấp, làm thêm giờ", "kho hàng / tồn kho / nhập xuất", "ngân hàng / lãi suất / trả góp",
    "giáo dục / điểm số / xếp loại học sinh", "logistics / vận chuyển / tuyến giao hàng", "y tế / lịch hẹn khám / phòng khám",
    "log máy chủ web (nginx, ứng dụng)", "dữ liệu CSV / báo cáo từ Excel xuất ra", "ngày giờ / lịch làm việc Việt Nam (cuối tuần, lễ Tết)",
    "chuỗi tiếng Việt (bỏ dấu, chuẩn hoá họ tên, địa chỉ, số điện thoại)", "xử lý JSON trả về từ API", "nhà hàng / gọi món / chia hoá đơn",
    "đặt vé / đặt phòng khách sạn / giá theo mùa", "bất động sản / tiền thuê / hợp đồng", "quản lý dự án / task / deadline",
    "điện nước / bậc thang giá điện", "bảo hiểm / phí bảo hiểm / quyền lợi", "chứng khoán (thống kê dữ liệu giá, không tư vấn đầu tư)",
    "quán cà phê / tích điểm khách hàng thân thiết", "trường học / thời khoá biểu / phòng học", "sản xuất / định mức nguyên vật liệu",
    "marketing / thống kê chiến dịch, tỉ lệ chuyển đổi", "chăm sóc khách hàng / ticket / SLA", "vận hành IT / giám sát, cảnh báo",
    "tài chính cá nhân / chi tiêu theo danh mục", "thể thao / bảng xếp hạng giải đấu", "thư viện / mượn trả sách / phạt quá hạn",
]
KINDS = [("py_func", .30), ("py_data", .15), ("py_text", .10), ("py_debug", .12), ("py_class", .05),
         ("sql", .12), ("excel", .08), ("js", .08)]
LEVELS = ["vừa (3–4 ràng buộc)", "khó (5–6 ràng buộc, có trường hợp biên)", "khó (nhiều trường hợp biên + kiểm tra đầu vào lỗi)"]
STYLES = ["người dùng nhắn ngắn gọn, đi thẳng vào việc", "người dùng kể bối cảnh công việc trước rồi mới nêu yêu cầu",
          "người dùng liệt kê yêu cầu dạng gạch đầu dòng", "người dùng đưa ví dụ đầu vào/đầu ra cụ thể",
          "người dùng là người mới học, diễn đạt chưa chuẩn thuật ngữ nhưng yêu cầu rõ"]
BUGS = ["tham số mặc định có thể thay đổi (list/dict)", "lệch một (off-by-one) trong vòng lặp hoặc range", "so sánh số thực tiền tệ bằng ==",
        "sửa list trong lúc đang duyệt", "chia nguyên // thay vì chia thường", "copy nông (shallow copy) danh sách lồng nhau",
        "so sánh chuỗi không chuẩn hoá hoa/thường hoặc khoảng trắng", "xử lý sai ngày cuối tháng / năm nhuận",
        "biến dùng chung bị ghi đè trong vòng lặp lồng", "sắp xếp sai khoá hoặc sai thứ tự khi bằng nhau", "quên trường hợp danh sách rỗng",
        "làm tròn sai (round kiểu ngân hàng của Python)"]

COMMON = """Bạn là người ra đề cho bộ dữ liệu huấn luyện trợ lý lập trình tiếng Việt. Hãy tạo MỘT bài THỰC TẾ, cụ thể, KHÁC các bài
quen thuộc trên mạng.
Lĩnh vực: {dom}. Độ khó: {lvl}. Cách người dùng viết: {sty}. Mã kịch bản: {k} (CHỈ để bạn tự đổi tên người, tên hàm, số liệu cho đa dạng — TUYỆT ĐỐI KHÔNG ghi mã này vào đề).
Đề ("de") viết bằng TIẾNG VIỆT tự nhiên như người dùng Việt Nam thật nhắn cho trợ lý, nêu rõ mọi ràng buộc cần thiết để làm đúng
(định dạng đầu vào/ra, làm tròn, trường hợp biên, lỗi cần báo). Không lộ lời giải, không nhắc tới test."""

P_GEN = {
 "py": COMMON + """
Kiểu bài: {what}
Đề phải yêu cầu viết bằng Python và nêu rõ tên + tham số của {unit} (ví dụ `def ten_ham(a, b=...)`). Chỉ dùng thư viện chuẩn.
Trả về DUY NHẤT một đối tượng JSON (không có chữ nào khác):
{{"de": "...", "loi_giai": "code Python hoàn chỉnh", "tests": ["assert ...", "..."]}}
tests: 8–12 dòng assert ĐỘC LẬP (mỗi dòng tự chạy được sau lời giải), phủ trường hợp thường + biên + lỗi.
Kiểm lỗi bằng hàm trợ giúp có sẵn: `assert _raises(lambda: ten_ham(...), ValueError)`. So sánh số thực bằng abs(a-b) < 1e-6.""",
 "sql": COMMON + """
Kiểu bài: câu hỏi báo cáo trên cơ sở dữ liệu nghiệp vụ, cần JOIN ≥ 2 bảng và GROUP BY / HAVING / hàm cửa sổ / truy vấn con tuỳ độ khó.
Đề mô tả các bảng (tên cột, kiểu) như người dùng dán cấu trúc bảng, rồi hỏi bằng tiếng Việt; nêu rõ cột cần trả ra và thứ tự sắp xếp.
Trả về DUY NHẤT một JSON: {{"de": "...", "schema": "CREATE TABLE ...; INSERT INTO ... (SQLite, 15–40 dòng dữ liệu, có dòng gây bẫy như
NULL, trạng thái khác, ngày ngoài khoảng)", "query": "truy vấn SQLite đúng"}}""",
 "excel": COMMON + """
Kiểu bài: công thức Excel trên một bảng dữ liệu. Bảng nằm ở Sheet1 từ ô A1 (hàng 1 là tiêu đề, tối đa 6 cột A–F, 8–25 hàng dữ liệu).
Đề mô tả bảng (cột nào chứa gì, từ hàng nào tới hàng nào) và yêu cầu MỘT công thức đặt ở ô H2 trả về MỘT giá trị
(ví dụ tổng có nhiều điều kiện, đếm có điều kiện, tra cứu, trung bình có điều kiện, theo tháng/năm...).
Dùng hàm phổ biến: SUMIFS, COUNTIFS, AVERAGEIFS, INDEX/MATCH, IF, AND/OR, ROUND, SUMPRODUCT, MAX/MIN có điều kiện, DATE, YEAR, MONTH.
Ngày trong bảng ghi dạng số ngày Excel (số nguyên, ví dụ 46296 ≈ 2026-10-01) và đề nói rõ đó là cột ngày.
Trả về DUY NHẤT một JSON: {{"de": "...", "table": [["Tiêu đề A", ...], [giá trị...], ...], "formula": "=..."}}""",
 "js": COMMON + """
Kiểu bài: hàm JavaScript (Node.js, không thư viện ngoài) xử lý nghiệp vụ.
Đề nêu rõ tên + tham số hàm. Trả về DUY NHẤT một JSON:
{{"de": "...", "loi_giai": "code JavaScript định nghĩa hàm (function ten(...) {{...}})", "tests": ["assert.deepStrictEqual(...)", "..."]}}
tests: 8–12 dòng dùng module `assert` (đã require sẵn biến assert), phủ thường + biên + lỗi (assert.throws).""",
}
WHAT = {
 "py_func": ("viết một hàm nghiệp vụ tính toán có nhiều ràng buộc", "hàm"),
 "py_data": ("xử lý dữ liệu dạng list các dict hoặc chuỗi CSV (lọc, gộp nhóm, thống kê, sắp xếp)", "hàm"),
 "py_text": ("xử lý văn bản / chuỗi tiếng Việt (chuẩn hoá, tách trường, regex, định dạng số tiền / ngày)", "hàm"),
 "py_class": ("viết một lớp (class) nhỏ có trạng thái và vài phương thức; tests tạo đối tượng rồi gọi phương thức", "lớp và các phương thức"),
}
P_DEBUG = COMMON + """
Kiểu bài: SỬA LỖI. Người dùng dán một hàm Python ({bug}) đang cho kết quả sai, mô tả triệu chứng (có thể kèm thông báo lỗi hoặc
ví dụ kết quả sai) và nhờ tìm nguyên nhân + sửa. Code lỗi phải nằm TRONG đề, trong khối ```python.
Trả về DUY NHẤT một JSON: {{"de": "...", "loi_giai": "hàm ĐÃ SỬA hoàn chỉnh (cùng tên, cùng tham số)", "tests": ["assert ...", ...]}}
tests: 6–10 assert, trong đó ít nhất 2 cái làm code LỖI ban đầu chạy sai. Dùng `_raises(lambda: ..., ValueError)` nếu cần kiểm lỗi."""

P_ANS = {
 "py": """{de}

---
Trả lời người dùng bằng TIẾNG VIỆT như một trợ lý lập trình giỏi: 1–3 câu nêu cách làm (và nguyên nhân lỗi nếu là bài sửa lỗi),
rồi đưa TOÀN BỘ code Python hoàn chỉnh trong MỘT khối ```python (đúng tên hàm/lớp và tham số đề yêu cầu, có import nếu cần,
không kèm test, không có lệnh input()), cuối cùng tối đa 3 gạch đầu dòng lưu ý (trường hợp biên đã xử lý). Không dùng LaTeX.""",
 "js": """{de}

---
Trả lời bằng TIẾNG VIỆT: 1–3 câu nêu cách làm, rồi TOÀN BỘ code trong MỘT khối ```javascript (đúng tên hàm, không require thư viện
ngoài, không in ra màn hình, không module.exports), cuối cùng tối đa 3 gạch đầu dòng lưu ý. Không dùng LaTeX.""",
 "sql": """{de}

---
Trả lời bằng TIẾNG VIỆT: viết truy vấn trong MỘT khối ```sql (cú pháp chạy được trên SQLite), rồi 2–4 câu giải thích từng phần
(JOIN, điều kiện lọc, gộp nhóm, sắp xếp). Không dùng LaTeX.""",
 "excel": """{de}

---
Trả lời bằng TIẾNG VIỆT: đưa công thức trong MỘT khối ```excel (một dòng bắt đầu bằng =, dùng dấu phẩy ngăn tham số), rồi giải thích
ngắn từng phần của công thức, cuối cùng 1 câu lưu ý (ví dụ máy dùng định dạng vùng Việt Nam thì thay dấu phẩy bằng chấm phẩy).""",
}
HELPER = '''
def _raises(f, exc=Exception):
    try:
        f()
    except exc:
        return True
    except Exception:
        return False
    return False
'''
CJK = re.compile(r"[぀-ヿ㐀-䶿一-鿿가-힯]")
VI_CH = re.compile(r"[ăâđêôơưạảấầẩẫậắằẳẵặẹẻẽếềểễệỉịọỏốồổỗộớờởỡợụủứừửữựỳỵỷỹ]", re.I)


def logw(**d):
    if _LOG is None:
        return
    with _LK:
        _LOG.write(json.dumps({"t": round(time.time(), 1), **d}, ensure_ascii=False) + "\n")


_RR = [0]


def chat(prompt, max_tokens=2500, temp=0.3):
    urls = A.llm.split(",")                                    # nhiều thầy (vd 1 vLLM / GPU) → chia vòng tròn
    with _LK:
        _RR[0] += 1
        base = urls[_RR[0] % len(urls)]
    body = {"model": A.model, "messages": [{"role": "user", "content": prompt}], "temperature": temp, "max_tokens": max_tokens}
    for k in range(3):
        try:
            h = {"Content-Type": "application/json"}
            if os.environ.get("C7_API_KEY"):                  # thầy qua API ngoài (user tự đặt biến môi trường trên box)
                h["Authorization"] = "Bearer " + os.environ["C7_API_KEY"]
            req = urllib.request.Request(base.rstrip("/") + "/chat/completions", json.dumps(body).encode(), h)
            with urllib.request.urlopen(req, timeout=900) as r:
                return json.loads(r.read())["choices"][0]["message"]["content"]
        except Exception:
            time.sleep(5 * (k + 1))
    return ""


def loads_json(t):
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t.strip())
    i, j = t.find("{"), t.rfind("}")
    return json.loads(t[i:j + 1]) if i >= 0 and j > i else None


def extract_formula(ans):                       # như office_probe_light.extract_formula (ưu tiên khối ```)
    f_ = None
    for blk in re.findall(r"```[a-zA-Z]*\n(.*?)```", ans, re.S):
        f_ = next((ln.strip() for ln in blk.splitlines() if ln.strip().startswith("=")), None)
        if f_:
            break
    if not f_:
        m_ = re.search(r"(=[A-Z]+\(.*\))", ans)
        f_ = m_.group(1).strip() if m_ else None
    if f_ and ";" in f_ and "," not in f_:
        f_ = f_.replace(";", ",")
    return f_


def block(t, langs):
    bs = re.findall(r"```(?:%s)\s*\n(.*?)```" % langs, t, re.S | re.I)
    return max(bs, key=len).strip("\n") if bs else None


# ── chấm (dùng chung cho sinh dữ liệu và đo) ────────────────────────────────────────────────
def grade_py(code, tests):
    r = CA.run_python(HELPER + "\n" + code + "\n\n" + "\n".join(t.strip() for t in tests) + "\n", timeout=15)
    return r["rc"] == 0 and not r["timeout"], r["out"][-1500:]


def grade_js(code, tests):
    import c2b_teacher as T
    prog = "const assert = require('assert');\n" + code + "\n\n" + "\n".join(t.strip() for t in tests) + "\n"
    r = T.run_project({"prog.js": prog}, "node prog.js", 30)
    return r["rc"] == 0 and not r["timeout"], r["out"][-1500:]


XL_CODE = """import json, sys
sys.path.insert(0, sys.argv[1])
from gen_office import check_excel
d = json.load(sys.stdin)
ok, got = check_excel(d["table"], d["formula"], d["expected"])
got = got.item() if hasattr(got, "item") else got
print(json.dumps({"ok": bool(ok), "got": got if isinstance(got, (int, float, str, bool)) or got is None else str(got)}))
"""


def excel_check(table, formula, expected, timeout=30):
    """check_excel trong TIẾN TRÌNH CON có hạn giờ: thư viện formulas có công thức làm treo vô hạn khi đang giữ khoá chung
    → cả bộ sinh đứng (22/10). → (ok, giá trị) · (False, None) nếu lỗi/hết giờ."""
    import subprocess
    try:
        r = subprocess.run([sys.executable, "-c", XL_CODE, os.path.dirname(os.path.abspath(__file__))], capture_output=True,
                           text=True, timeout=timeout, input=json.dumps({"table": table, "formula": formula, "expected": expected}))
        d = json.loads(r.stdout.strip().splitlines()[-1])
        return d["ok"], d["got"]
    except Exception:
        return False, None


def sql_rows(schema, q):
    con = sqlite3.connect(":memory:")
    try:
        con.executescript(schema)
        rows = [tuple(round(v, 4) if isinstance(v, float) else v for v in r) for r in con.execute(q).fetchall()]
        return rows
    except Exception:
        return None
    finally:
        con.close()


def same_rows(a, b, q):
    if a is None or b is None:
        return False
    return a == b if re.search(r"order\s+by", q, re.I) else sorted(map(repr, a)) == sorted(map(repr, b))


def grade(item, ans):
    k = item["kind"]
    if k == "sql":
        q = block(ans, "sql")
        return bool(q) and same_rows(sql_rows(item["schema"], q), [tuple(r) for r in item["expected_rows"]], item["query"]), q
    if k == "excel":
        fm = extract_formula(ans)
        return (excel_check(item["table"], fm, item["expected"])[0] if fm else False), fm
    if k == "js":
        c = block(ans, "javascript|js")
        return (grade_js(c, item["tests"])[0] if c else False), c
    c = block(ans, "python|py") or CA.extract_code(ans)
    return (grade_py(c, item["tests"])[0] if c else False), c


# ── sinh 1 mẫu ─────────────────────────────────────────────────────────────────────────────
def make(k):
    rng = random.Random(f"{A.split}-{A.seed}-{k}")
    kind = rng.choices([x for x, _ in KINDS], [w for _, w in KINDS])[0]
    dom, lvl, sty = rng.choice(DOMAINS), rng.choice(LEVELS), rng.choice(STYLES)
    tag = f"{A.split[0]}{k}"
    if kind == "py_debug":
        prompt = P_DEBUG.format(dom=dom, lvl=lvl, sty=sty, k=tag, bug=rng.choice(BUGS))
    elif kind in WHAT:
        what, unit = WHAT[kind]
        prompt = P_GEN["py"].format(dom=dom, lvl=lvl, sty=sty, k=tag, what=what, unit=unit)
    else:
        prompt = P_GEN[kind].format(dom=dom, lvl=lvl, sty=sty, k=tag)
    raw = chat(prompt, max_tokens=3500, temp=1.0)
    g = None
    try:
        g = loads_json(raw)
    except Exception:
        pass
    logw(ev="gen", id=tag, kind=kind, dom=dom, ok=bool(g), raw=raw[:6000])
    if not g or not isinstance(g.get("de"), str) or len(VI_CH.findall(g["de"])) < 15 or CJK.search(raw):
        return None
    item = {"id": f"c7_{tag}", "kind": kind, "dom": dom, "lvl": lvl, "de": strip_tag(g["de"]).strip()}
    # lớp 1: lời giải mẫu phải chạy đúng
    if kind == "sql":
        rows = sql_rows(g.get("schema", ""), g.get("query", ""))
        if not rows:
            return None
        item.update(schema=g["schema"], query=g["query"], expected_rows=rows)
    elif kind == "excel":
        ok, val = excel_check(g.get("table"), g.get("formula"), 0)
        if val is None or (isinstance(val, str) and val.startswith("#")):
            return None
        item.update(table=g["table"], formula=g["formula"], expected=val)
    else:
        tests = [t for t in g.get("tests", []) if isinstance(t, str) and t.strip()]
        if len(tests) < 5 or not isinstance(g.get("loi_giai"), str):
            return None
        ok, out = (grade_js if kind == "js" else grade_py)(g["loi_giai"], tests)
        logw(ev="ref", id=tag, ok=ok, out=out)
        if not ok:
            return None
        item.update(ref=g["loi_giai"], tests=tests)
    # lớp 2: thầy giải lại độc lập chỉ từ đề → phải qua đúng bộ kiểm
    pk = "py" if kind.startswith("py") else kind
    for temp in (0.3, 0.8):
        ans = chat(P_ANS[pk].format(de=item["de"]), max_tokens=2500, temp=temp).strip()
        if not ans or CJK.search(ans):
            continue
        ok, _ = grade(item, ans)
        logw(ev="ans", id=tag, temp=temp, ok=ok, ans=ans)
        if ok:
            item["messages"] = [{"role": "user", "content": item["de"]}, {"role": "assistant", "content": ans}]
            return item
    return None


def safe(k):
    try:
        return make(k)
    except Exception as e:
        print(f"[c7] lỗi mẫu {k}: {type(e).__name__}: {str(e)[:120]}", flush=True)
        return None


TAGRX = re.compile(r"\s*\((?:mã|ma)(?: dự án| kịch bản| bài| đề)?\s*[te]\d+\)|\s*(?:mã|ma)(?: dự án| kịch bản)?\s*[te]\d+\b", re.I)


def strip_tag(text):
    """bỏ "mã kịch bản" (vd "(mã dự án t95)") thầy lỡ chép vào đề — mã chỉ để đa dạng hoá (22/10)"""
    return TAGRX.sub("", text)


def key_of(item):
    return hashlib.md5(re.sub(r"\W+", " ", item["de"].lower())[:240].encode()).hexdigest()


def gen():
    global _LOG
    os.makedirs(f"{OUT}/teacher_log", exist_ok=True)
    _LOG = open(f"{OUT}/teacher_log/{A.split}.jsonl", "a", encoding="utf-8")
    path, seen_p = f"{OUT}/{A.split}.jsonl", f"{OUT}/{A.split}.seen"
    done = set(open(seen_p).read().split()) if os.path.exists(seen_p) else set()
    have = [json.loads(l) for l in open(path)] if os.path.exists(path) else []
    keys = {key_of(x) for x in have}
    ban = set()                                                # chống rò: đề đo (eval) không được lọt vào train
    if A.split == "train" and os.path.exists(f"{OUT}/eval.jsonl"):
        ban = {key_of(json.loads(l)) for l in open(f"{OUT}/eval.jsonl")}
    kept, n, T0 = len(have), 0, time.time()
    print(f"[c7] {A.split}: đã có {kept} · cần {A.keep}", flush=True)
    todo = (k for k in range(10 ** 7) if str(k) not in done)
    from concurrent.futures import FIRST_COMPLETED, wait
    ex = ThreadPoolExecutor(A.workers)
    with open(path, "a") as fo, open(seen_p, "a") as fs:
        live = {}
        for _ in range(A.workers if kept < A.keep else 0):
            k = next(todo); live[ex.submit(safe, k)] = k      # luồng liên tục: xong 1 bài → nạp 1 bài
        last = time.time()
        while live and kept < A.keep:
            fin, _ = wait(live, return_when=FIRST_COMPLETED)
            for f in fin:
                fs.write(f"{live.pop(f)}\n")                   # đã thử (giữ hay không) → chạy lại không làm lại
                n += 1
                res = f.result()
                if res:
                    if key_of(res) not in keys and key_of(res) not in ban:
                        keys.add(key_of(res)); kept += 1
                        fo.write(json.dumps(res, ensure_ascii=False) + "\n")
                if kept < A.keep:
                    k = next(todo)
                    live[ex.submit(safe, k)] = k
            if time.time() - last > 120:
                fo.flush(); fs.flush(); _LOG.flush(); last = time.time()
                print(f"[c7] {A.split} thử {n} · giữ {kept}/{A.keep} ({(kept - len(have)) / max(n, 1):.0%}) · {time.time()-T0:.0f}s", flush=True)
    ex.shutdown(wait=False, cancel_futures=True)                 # bài dở (kể cả bị treo) bỏ luôn — đã đủ số
    rows = [json.loads(l) for l in open(path)]                   # làm sạch cả mẫu sinh trước khi có strip_tag
    for r in rows:
        r["de"] = strip_tag(r["de"])
        for m in r.get("messages", []):
            m["content"] = strip_tag(m["content"])
    with open(path + ".tmp", "w") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    os.replace(path + ".tmp", path)
    cnt = {}
    for l in open(path):
        k_ = json.loads(l)["kind"]; cnt[k_] = cnt.get(k_, 0) + 1
    print(f"### c7 {A.split}: giữ {kept} · {cnt} · {time.time()-T0:.0f}s → {path}", flush=True)
    _LOG.flush()
    os._exit(0)                                                  # không chờ luồng còn treo


def evaluate():
    items = [json.loads(l) for l in open(f"{OUT}/eval.jsonl")]
    if A.n:
        items = items[:A.n]
    si, sn = map(int, A.shard.split("/"))                         # --shard 1/2 → nửa đầu (chạy song song nhiều máy chủ)
    items = items[si - 1::sn]
    tag = A.tag + (f"_s{si}" if sn > 1 else "")
    m = CA.Model(A.url, max_new=A.max_new)
    os.makedirs("out/code_eval", exist_ok=True)
    res = {}

    def one(it):
        try:
            ans = m.chat([{"role": "user", "content": it["de"]}])
        except Exception as e:
            ans = f"[lỗi] {e}"
        ok, got = grade(it, ans)
        return it, ans, ok, got
    with open(f"out/code_eval/c7_{tag}.jsonl", "w") as fo, ThreadPoolExecutor(A.workers) as ex:
        for i, (it, ans, ok, got) in enumerate(ex.map(one, items), 1):
            r = res.setdefault(it["kind"], [0, 0]); r[0] += ok; r[1] += 1
            fo.write(json.dumps({"id": it["id"], "kind": it["kind"], "ok": ok, "answer": ans}, ensure_ascii=False) + "\n")
            if i % 20 == 0:
                tot = sum(v[0] for v in res.values())
                print(f"[c7 {tag}] {i}/{len(items)} · {100 * tot / i:.1f}%", flush=True)
    tot, n = sum(v[0] for v in res.values()), sum(v[1] for v in res.values())
    by = {k: f"{v[0]}/{v[1]}" for k, v in sorted(res.items())}
    print(f"### c7 {tag} {{'n': {n}, 'đúng_%': {100 * tot / max(n, 1):.1f}, 'theo_kiểu': {by}}}", flush=True)


def report():
    import glob
    res = {}
    fs = sorted(glob.glob(f"out/code_eval/c7_{A.tag}.jsonl") + glob.glob(f"out/code_eval/c7_{A.tag}_s*.jsonl"))
    for f in fs:
        for l in open(f):
            r = json.loads(l); v = res.setdefault(r["kind"], [0, 0]); v[0] += r["ok"]; v[1] += 1
    tot, n = sum(v[0] for v in res.values()), sum(v[1] for v in res.values())
    by = {k: f"{v[0]}/{v[1]}" for k, v in sorted(res.items())}
    print(f"### c7 {A.tag} {{'n': {n}, 'đúng_%': {100 * tot / max(n, 1):.1f}, 'theo_kiểu': {by}}}", flush=True)


def main():
    global A
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["gen", "eval", "report"])
    ap.add_argument("--split", default="train", choices=["train", "eval"])
    ap.add_argument("--keep", type=int, default=10000)
    ap.add_argument("--workers", type=int, default=128)
    ap.add_argument("--llm", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--model", default="gemma4-31b")
    ap.add_argument("--seed", type=int, default=108)
    ap.add_argument("--url", default="http://localhost:8881")
    ap.add_argument("--tag", default="model")
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--shard", default="1/1")
    ap.add_argument("--max-new", type=int, default=1024)
    ap.add_argument("--out", default="data/c7", help="thư mục dữ liệu (thử nhỏ ghi riêng, không lẫn dữ liệu thật)")
    A = ap.parse_args()
    global OUT
    OUT = A.out
    {"gen": gen, "eval": evaluate, "report": report}[A.cmd]()


if __name__ == "__main__":
    main()
