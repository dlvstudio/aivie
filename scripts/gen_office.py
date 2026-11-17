"""§TK_104 §15 · Dữ liệu EXPERT VĂN PHÒNG cho 1.3B — ít mà chất (Gemma làm thầy, máy kiểm/chấm lọc).

Ba dòng dữ liệu:
  text  : soạn thảo + biên tập (email, thông báo, báo cáo, biên bản, công văn, đơn, CV, viết lại giọng, tóm tắt,
          trả lời email, dịch email) + hướng dẫn thao tác Word / PowerPoint / Outlook / Google Docs
  excel : công thức Excel/Sheets KIỂM ĐƯỢC — Gemma ra bảng nhỏ + yêu cầu + công thức + kết quả mong đợi;
          ta dựng file xlsx thật, tính bằng thư viện `formulas`, chỉ giữ bài mà công thức cho ĐÚNG kết quả
  auto  : tự động hoá văn phòng bằng code (VBA, Google Apps Script, Python pandas/openpyxl) — cầu sang expert code

Mỗi mẫu đi qua giám khảo Gemma (đúng/hữu ích + tiếng Việt tự nhiên, thang 1–5) → giữ mẫu ≥4 cả hai.
Lưới đề (loại × ngữ cảnh ngành × người hỏi × giọng × kiểu hỏi) → đa dạng, tránh bài học QLoRA "data hẹp làm hỏng".

    .venv/bin/python scripts/gen_office.py gen --n 200 --workers 2             # Mac: thử chất lượng
    python scripts/gen_office.py gen --n 30000 --workers 32 --out data/office/raw.jsonl   # box H100
    python scripts/gen_office.py judge --workers 32
    python scripts/gen_office.py build        # → data/office/train.jsonl + val.jsonl + eval_office.jsonl (đề giữ riêng)
"""
from __future__ import annotations
import argparse, json, os, random, re, sys, tempfile, time, threading, urllib.error, urllib.request
import concurrent.futures as cf

GEMMA = "gemma4:26b-a4b-it-qat"
RAW = "data/office/raw.jsonl"
JUDGED = "data/office/judged.jsonl"
T0 = time.time()

# ───────── lưới đề ─────────
TEXT_TASKS = {
    "email": ["email mời họp/mời dự sự kiện", "email báo giá cho khách", "email nhắc khách thanh toán công nợ",
              "email xin lỗi khách vì sự cố", "email từ chối đề nghị một cách khéo léo", "email follow-up sau cuộc họp",
              "email cảm ơn đối tác", "email xin nghỉ phép gửi sếp", "email bàn giao công việc",
              "email chào hàng lần đầu", "email phản hồi khiếu nại", "email gửi ứng viên mời phỏng vấn",
              "email thông báo kết quả tuyển dụng (trượt)", "email đề nghị tăng lương", "email nhắc deadline cho team"],
    "van_ban": ["thông báo nội bộ", "báo cáo công việc tuần", "báo cáo kết quả kinh doanh tháng ngắn gọn",
                "biên bản họp", "kế hoạch tổ chức sự kiện", "tờ trình/đề xuất mua sắm thiết bị",
                "công văn gửi đối tác theo thể thức hành chính Việt Nam", "quyết định khen thưởng nhân viên",
                "đơn xin nghỉ việc", "đơn xin nghỉ phép", "thư xin việc", "CV ngắn cho một vị trí",
                "mô tả công việc (JD) tuyển dụng", "nội quy/quy định ngắn", "thư mời tham dự hội nghị",
                "giấy ủy quyền đơn giản", "bài đăng thông báo trên fanpage công ty"],
    "bien_tap": ["viết lại đoạn văn cho trang trọng hơn", "viết lại cho thân thiện, ngắn gọn hơn",
                 "rút gọn một email dài còn 3-4 câu", "tóm tắt một email/biên bản dài thành gạch đầu dòng",
                 "sửa lỗi chính tả, ngữ pháp cho đoạn văn", "soạn câu trả lời cho một email nhận được",
                 "dịch một email tiếng Anh sang tiếng Việt", "dịch một email tiếng Việt sang tiếng Anh",
                 "đặt tiêu đề (subject) hay cho email", "chuyển ghi chú lộn xộn thành biên bản gọn gàng"],
    "thao_tac": ["Word: tạo mục lục tự động", "Word: đánh số trang bắt đầu từ trang 3", "Word: trộn thư (mail merge) từ Excel",
                 "Word: theo dõi thay đổi (track changes) và chấp nhận sửa", "Word: định dạng heading/style thống nhất",
                 "Word: ngắt trang, ngắt section, đổi hướng trang ngang một trang", "Excel: tạo pivot table tổng hợp doanh thu",
                 "Excel: định dạng có điều kiện tô màu ô quá hạn", "Excel: data validation tạo danh sách thả xuống",
                 "Excel: cố định dòng tiêu đề (freeze panes) và in lặp tiêu đề", "Excel: lọc và xoá dòng trùng lặp",
                 "Excel: vẽ biểu đồ cột kết hợp đường", "Excel: tách họ và tên ra hai cột",
                 "PowerPoint: dàn ý slide thuyết trình", "PowerPoint: chèn số slide và logo mọi trang",
                 "Outlook/Gmail: tạo quy tắc lọc thư tự động", "Outlook/Gmail: hẹn giờ gửi email",
                 "Google Sheets: chia sẻ và phân quyền chỉnh sửa", "Google Docs: gõ tiếng Việt và xuất PDF"],
}
AUTO_TASKS = ["macro VBA gộp nhiều sheet thành một", "macro VBA gửi email hàng loạt từ danh sách Excel",
              "Google Apps Script tự gửi email nhắc lịch từ Google Sheets",
              "Python pandas đọc nhiều file Excel và gộp lại", "Python openpyxl tô màu dòng theo điều kiện",
              "Python tách một file Excel lớn thành nhiều file theo phòng ban",
              "Python tạo báo cáo tổng hợp doanh thu theo tháng từ file Excel",
              "VBA đổi tên hàng loạt sheet theo danh sách", "Google Apps Script lưu câu trả lời Google Form sang Docs",
              "Python đọc file CSV, lọc dữ liệu rồi xuất Excel có định dạng"]
EXCEL_FUNCS = ["SUM/SUMIF/SUMIFS", "COUNTIF/COUNTIFS", "AVERAGEIF", "IF lồng nhau / IFS", "VLOOKUP",
               "INDEX + MATCH", "XLOOKUP", "IFERROR", "ROUND/ROUNDUP", "LEFT/RIGHT/MID/LEN/FIND",
               "TEXT định dạng số/ngày", "DATE/YEAR/MONTH/DATEDIF", "MAX/MIN/LARGE/SMALL", "RANK",
               "SUMPRODUCT", "AND/OR trong IF", "CONCATENATE/& ghép chuỗi", "UPPER/PROPER/TRIM"]
CONTEXTS = ["công ty xuất nhập khẩu", "trường THPT", "phòng khám nha khoa", "shop thời trang online",
            "chi nhánh ngân hàng", "UBND phường", "startup phần mềm", "nhà hàng", "công ty xây dựng",
            "công ty logistics", "trung tâm tiếng Anh", "khách sạn", "nhà máy may mặc", "công ty bất động sản",
            "văn phòng luật", "phòng nhân sự", "phòng kế toán", "cửa hàng điện máy", "công ty du lịch",
            "hợp tác xã nông nghiệp", "bệnh viện tuyến huyện", "công ty bảo hiểm", "agency quảng cáo"]
ASKERS = ["nhân viên mới đi làm", "trưởng phòng", "kế toán", "thư ký giám đốc", "giáo viên", "chủ shop nhỏ",
          "sinh viên thực tập", "nhân viên hành chính", "cán bộ phường", "quản lý dự án"]
STYLES = ["ngắn gọn, hơi cộc", "lịch sự, đầy đủ thông tin", "kiểu chat thân mật, có từ lóng nhẹ",
          "dài dòng, kể hoàn cảnh trước rồi mới nhờ", "gõ vội, thiếu dấu câu", "xưng anh/chị, nhờ vả"]


def spec(rng):
    r = rng.random()
    if r < 0.62:
        cat = rng.choice(list(TEXT_TASKS)); return {"kind": "text", "cat": cat, "task": rng.choice(TEXT_TASKS[cat]),
                                                    "ctx": rng.choice(CONTEXTS), "asker": rng.choice(ASKERS),
                                                    "style": rng.choice(STYLES)}
    if r < 0.87:
        return {"kind": "excel", "cat": "excel", "task": rng.choice(EXCEL_FUNCS), "ctx": rng.choice(CONTEXTS),
                "asker": rng.choice(ASKERS), "style": rng.choice(STYLES)}
    return {"kind": "auto", "cat": "tu_dong_hoa", "task": rng.choice(AUTO_TASKS), "ctx": rng.choice(CONTEXTS),
            "asker": rng.choice(ASKERS), "style": rng.choice(STYLES)}


GUIDE = """Yêu cầu với CÂU TRẢ LỜI (như một trợ lý văn phòng giỏi người Việt):
- Tiếng Việt tự nhiên, đúng chính tả; trình bày rõ ràng (gạch đầu dòng/đánh số khi cần), không lan man.
- Làm đúng điều được nhờ. Văn bản soạn sẵn phải dùng được ngay; thông tin người hỏi không cung cấp thì để chỗ trống
  dạng [Tên công ty], [ngày], KHÔNG bịa số liệu, tên người, số điện thoại cụ thể.
- Hướng dẫn thao tác: các bước cụ thể theo menu thật của phần mềm (ghi rõ tên menu tiếng Anh nếu giao diện tiếng Anh).
- CHỈ dùng thông tin có trong yêu cầu: không thêm ngành nghề, dự án, con số, tên riêng mà người hỏi không nói tới.
- Mở đầu ngắn kiểu "Dưới đây là …" rồi vào việc; có thể kết bằng 1 câu gợi ý chỉnh sửa. Không tự xưng tên."""

P_TEXT = """Bạn tạo dữ liệu huấn luyện cho trợ lý văn phòng tiếng Việt.
Bối cảnh: người hỏi là {asker} ở {ctx}. Việc cần nhờ: {task}. Cách người đó hỏi: {style}.
1) Viết YÊU CẦU người đó gõ vào khung chat — tự nhiên như người thật, đủ chi tiết cụ thể của tình huống
   (nếu cần đoạn văn/email gốc để sửa, tóm tắt hoặc dịch thì đưa luôn đoạn đó vào yêu cầu).
   Yêu cầu là lời người thật gõ nên KHÔNG chứa chỗ trống kiểu [Tên], không gọi trợ lý bằng tên.
2) Viết CÂU TRẢ LỜI.
{guide}
Trả về ĐÚNG định dạng sau (không thêm gì khác):
<<<YEU_CAU>>>
(yêu cầu)
<<<TRA_LOI>>>
(câu trả lời)"""

P_AUTO = """Bạn tạo dữ liệu huấn luyện cho trợ lý văn phòng tiếng Việt biết lập trình.
Bối cảnh: người hỏi là {asker} ở {ctx}, nhờ viết {task}. Cách hỏi: {style}.
1) Viết YÊU CẦU người đó gõ (mô tả cụ thể dữ liệu của họ: tên sheet/cột/file), không chứa chỗ trống kiểu [Tên]).
2) Viết CÂU TRẢ LỜI: giải thích ngắn bằng tiếng Việt + MỘT khối code hoàn chỉnh chạy được (```vba / ```javascript /
   ```python), chú thích trong code bằng tiếng Việt, rồi cách chạy từng bước.
{guide}
Trả về ĐÚNG định dạng sau (không thêm gì khác):
<<<YEU_CAU>>>
(yêu cầu)
<<<TRA_LOI>>>
(câu trả lời)"""

P_EXCEL = """Bạn tạo dữ liệu huấn luyện cho trợ lý Excel tiếng Việt. Bài phải KIỂM ĐƯỢC bằng máy.
Bối cảnh: {asker} ở {ctx}; nhóm hàm: {task}. Cách hỏi: {style}.
- "table": bảng nhỏ trên Sheet1 bắt đầu từ A1: dòng 1 là tiêu đề tiếng Việt, 4–8 dòng dữ liệu, tối đa 6 cột,
  chỉ chữ hoặc số (ngày ghi dạng số serial Excel hoặc chuỗi "dd/mm/yyyy" thì đừng tính toán ngày trên chuỗi).
- "request": người hỏi mô tả bảng (cột nào chứa gì, dữ liệu từ dòng 2 đến dòng mấy) và cần tính gì.
- "formula": MỘT công thức Excel bắt đầu bằng "=" trả về MỘT giá trị, tham chiếu ĐÚNG cột/vùng mà "request" nêu
  (cột điều kiện, cột cần tính lấy đúng chữ cái người hỏi nói), đặt ở ô H2.
- "expected": kết quả đúng của công thức (số hoặc chuỗi).
- "answer": câu trả lời tiếng Việt: đưa công thức trong khối ``` , giải thích từng phần, nói kết quả, mẹo/lưu ý
  (ví dụ Excel tiếng Việt dùng dấu ; thay dấu , nếu máy cài định dạng vùng Việt Nam).
{guide}
Trả về DUY NHẤT JSON: {{"table": [[...],[...]], "request": "...", "formula": "=...", "expected": ..., "answer": "..."}}"""


def loads_loose(t):
    """JSON từ văn bản tự do (bỏ ```json … ```, lấy từ { đầu tới } cuối) — KHÔNG ép ngữ pháp JSON trong Ollama:
    lọc ngữ pháp trên bộ token 262K của Gemma ngốn CPU, GPU chỉ chạy ~13%."""
    t = t.strip()
    i, j = t.find("{"), t.rfind("}")
    if i < 0 or j <= i:
        raise ValueError("không thấy JSON")
    return json.loads(t[i:j + 1], strict=False)


def ollama(prompt, url, temp=0.8, fmt=None, num_predict=1800, tries=2):
    for t in range(tries):
        try:
            return _ollama(prompt, url, temp, fmt, num_predict)
        except urllib.error.HTTPError:
            if t == tries - 1:
                raise
            time.sleep(5)


LLM_MODEL = os.environ.get("LLM_MODEL", "gemma4-31b")   # tên model khi phục vụ bằng vLLM (--served-model-name)


def _openai(prompt, url, temp, fmt, num_predict):
    """vLLM / máy chủ tương thích OpenAI (url kết thúc bằng /v1) — nhanh gấp nhiều lần Ollama trên H100."""
    body = {"model": LLM_MODEL, "temperature": temp, "max_tokens": num_predict,
            "messages": [{"role": "user", "content": prompt}],
            "chat_template_kwargs": {"enable_thinking": False}}
    if fmt == "json":
        body["response_format"] = {"type": "json_object"}
    req = urllib.request.Request(f"{url}/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=900).read())["choices"][0]["message"]["content"]


def _ollama(prompt, url, temp, fmt, num_predict):
    if url.rstrip("/").endswith("/v1"):
        return _openai(prompt, url.rstrip("/"), temp, fmt, num_predict)
    body = {"model": GEMMA, "stream": False, "think": False,
            "options": {"temperature": temp, "num_predict": num_predict},
            "messages": [{"role": "user", "content": prompt}]}
    if fmt:
        body["format"] = fmt
    req = urllib.request.Request(f"{url}/api/chat", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=900).read())["message"]["content"]


# ───────── kiểm công thức Excel thật ─────────
_xl_lock = threading.Lock()


def check_excel(table, formula, expected):
    import openpyxl, formulas
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "t.xlsx")
        wb = openpyxl.Workbook(); ws = wb.active; ws.title = "Sheet1"
        for row in table:
            ws.append(row)
        ws["H2"] = formula
        wb.save(p)
        with _xl_lock:                                   # formulas không an-toàn đa luồng
            sol = formulas.ExcelModel().loads(p).finish().calculate()
        got = None
        for k, v in sol.items():
            if k.upper().endswith("SHEET1'!H2"):
                got = v.value[0][0] if hasattr(v, "value") else v
        if got is None:
            return False, None
        got = got.item() if hasattr(got, "item") else got
        if isinstance(expected, (int, float)) or re.fullmatch(r"-?[\d.]+", str(expected).strip()):
            try:
                return abs(float(got) - float(str(expected).replace(",", ""))) < 1e-6 * max(1, abs(float(got))) + 1e-9, got
            except (TypeError, ValueError):
                return False, got
        return str(got).strip() == str(expected).strip(), got


CJK = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uac00-\ud7af]")   # Gemma bản nén đôi khi lẫn chữ Trung/Nhật/Hàn → loại


def work(args):
    i, sp, url = args
    try:
        if sp["kind"] == "excel":
            o = loads_loose(ollama(P_EXCEL.format(**sp, guide=GUIDE), url, fmt="json"))
            ok, got = check_excel(o["table"], o["formula"], o["expected"])
            if not ok:
                return {"i": i, "spec": sp, "drop": f"excel sai: được {got!r} ≠ {o.get('expected')!r}"}
            norm = lambda t: re.sub(r"\s+", "", t).replace(";", ",").upper()
            if norm(o["formula"]) not in norm(o["answer"]):
                return {"i": i, "spec": sp, "drop": "excel: lời giải không chứa đúng công thức đã kiểm"}
            if CJK.search(o["request"] + o["answer"]):
                return {"i": i, "spec": sp, "drop": "lẫn chữ CJK"}
            return {"i": i, "spec": sp, "request": o["request"], "answer": o["answer"], "formula": o["formula"],
                    "table": o["table"], "expected": o["expected"]}
        P = P_AUTO if sp["kind"] == "auto" else P_TEXT
        t = ollama(P.format(**sp, guide=GUIDE), url)
        if "<<<YEU_CAU>>>" not in t or "<<<TRA_LOI>>>" not in t:
            return {"i": i, "spec": sp, "drop": "sai định dạng"}
        q_, a_ = t.split("<<<YEU_CAU>>>", 1)[1].split("<<<TRA_LOI>>>", 1)
        o = {"request": q_.strip(), "answer": a_.strip()}
        if CJK.search(o["request"] + o["answer"]):
            return {"i": i, "spec": sp, "drop": "lẫn chữ CJK"}
        if len(o["request"]) < 15 or len(o["answer"]) < 80:
            return {"i": i, "spec": sp, "drop": "ngắn"}
        return {"i": i, "spec": sp, "request": o["request"], "answer": o["answer"]}
    except Exception as e:
        return {"i": i, "spec": sp, "drop": f"{type(e).__name__}: {str(e)[:80]}"}


def cmd_gen(a):
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    rng = random.Random(a.seed)
    specs = [spec(rng) for _ in range(a.n)]
    done = {json.loads(l)["i"] for l in open(a.out)} if os.path.exists(a.out) else set()
    todo = [(i, s, a.ollama) for i, s in enumerate(specs) if i not in done]
    print(f"[gen] {len(done)} đã có · còn {len(todo)}", flush=True)
    kept = drop = 0
    with open(a.out, "a", encoding="utf-8") as f, cf.ThreadPoolExecutor(a.workers) as ex:
        for k, r in enumerate(ex.map(work, todo), 1):
            f.write(json.dumps(r, ensure_ascii=False) + "\n"); f.flush()
            kept += "drop" not in r; drop += "drop" in r
            if k % 50 == 0 or k == len(todo):
                el = time.time() - T0
                print(f"[{el:.0f}s] {k}/{len(todo)} · giữ {kept} · loại {drop} · {60*k/el:.1f}/phút", flush=True)
    print("### gen xong", flush=True)


# ───────── giám khảo ─────────
P_JUDGE = """Chấm câu trả lời của một trợ lý văn phòng cho yêu cầu tiếng Việt.
Yêu cầu:
{q}

Câu trả lời:
{ans}

Cho điểm 1–5:
- "ok": đúng, làm đúng việc được nhờ, dùng được ngay, không bịa thông tin cụ thể (5 = xuất sắc).
- "vi": tiếng Việt tự nhiên, đúng chính tả, văn phong phù hợp (5 = như người bản xứ viết giỏi).
Trả về DUY NHẤT JSON {{"ok": 4, "vi": 5, "why": "một câu ngắn"}}"""


def judge_work(args):
    r, url = args
    try:
        j = loads_loose(ollama(P_JUDGE.format(q=r["request"], ans=r["answer"][:5000]), url, temp=0, num_predict=200))
        r["j"] = {"ok": int(j["ok"]), "vi": int(j["vi"]), "why": j.get("why", "")}
    except Exception as e:
        r["j_err"] = str(e)[:80]
    return r


def cmd_judge(a):
    rows = []
    for l in open(a.inp, encoding="utf-8"):
        try:
            rows.append(json.loads(l))
        except json.JSONDecodeError:                       # dòng cuối đang ghi dở
            pass
    rows = [r for r in rows if "drop" not in r]
    done = {json.loads(l)["i"] for l in open(a.out)} if os.path.exists(a.out) else set()
    todo = [(r, a.ollama) for r in rows if r["i"] not in done]
    print(f"[judge] còn {len(todo)}", flush=True)
    with open(a.out, "a", encoding="utf-8") as f, cf.ThreadPoolExecutor(a.workers) as ex:
        for k, r in enumerate(ex.map(judge_work, todo), 1):
            f.write(json.dumps(r, ensure_ascii=False) + "\n"); f.flush()
            if k % 200 == 0:
                print(f"[{time.time()-T0:.0f}s] chấm {k}/{len(todo)}", flush=True)
    print("### judge xong", flush=True)


def cmd_build(a):
    """giữ ok≥4 & vi≥4 · khử trùng yêu cầu gần giống · tách đề đánh giá GIỮ RIÊNG (không train)."""
    rows = []
    for l in open(a.inp or (RAW if a.no_judge else JUDGED), encoding="utf-8"):
        try:
            rows.append(json.loads(l))
        except json.JSONDecodeError:                       # dòng cuối đang ghi dở (file vẫn đang sinh)
            pass
    if a.no_judge:   # Excel đã kiểm bằng máy; thử nghiệm: giám khảo chỉ loại ~3% mẫu (1/32)
        good = [r for r in rows if "drop" not in r]
    else:
        good = [r for r in rows if r.get("j", {}).get("ok", 0) >= 4 and r["j"].get("vi", 0) >= 4]
    seen, uniq = set(), []
    for r in good:
        key = re.sub(r"\W+", " ", r["request"].lower())[:80]
        if key not in seen:
            seen.add(key); uniq.append(r)
    rng = random.Random(1); rng.shuffle(uniq)
    if a.eval_from:                                       # giữ NGUYÊN bộ đề cũ để so sánh công bằng giữa các bản
        ev_old = [json.loads(l) for l in open(a.eval_from, encoding="utf-8")]
        old_keys = {re.sub(r"\W+", " ", e["q"].lower())[:80] for e in ev_old}
        rest = [r for r in uniq if re.sub(r"\W+", " ", r["request"].lower())[:80] not in old_keys]
        ev = None
    else:
        ev = uniq[:a.n_eval]
        rest = uniq[a.n_eval:]
    va, tr = rest[:a.n_val], rest[a.n_val:]
    msg = lambda r: {"messages": [{"role": "user", "content": r["request"]}, {"role": "assistant", "content": r["answer"]}],
                     "cat": r["spec"]["cat"]}
    for name, part in (("train", tr), ("val", va)):
        os.makedirs(a.out_dir, exist_ok=True)
        with open(os.path.join(a.out_dir, f"{name}.jsonl"), "w", encoding="utf-8") as f:
            for r in part:
                f.write(json.dumps(msg(r), ensure_ascii=False) + "\n")
    if ev is None:
        print(f"giữ bộ đề cũ {a.eval_from} · loại khỏi train {len(uniq) - len(rest)} mẫu trùng đề")
        ev = []
    with open(os.path.join(a.out_dir, "eval_office.jsonl") if not a.eval_from else os.devnull, "w", encoding="utf-8") as f:
        for r in ev:
            f.write(json.dumps({"id": f"off_{r['i']}", "cat": r["spec"]["cat"], "q": r["request"],
                                "ref": r["answer"], **({k: r[k] for k in ("table", "formula", "expected") if k in r})},
                               ensure_ascii=False) + "\n")
    from collections import Counter
    print(f"tổng {len(rows)} · đạt ≥4 {len(good)} · khử trùng {len(uniq)} → train {len(tr)} · val {len(va)} · eval {len(ev)}")
    print("theo loại:", Counter(r["spec"]["cat"] for r in uniq))


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    g = sp.add_parser("gen"); g.add_argument("--n", type=int, default=200); g.add_argument("--seed", type=int, default=0)
    g.add_argument("--workers", type=int, default=2); g.add_argument("--out", default=RAW)
    j = sp.add_parser("judge"); j.add_argument("--inp", default=RAW); j.add_argument("--out", default=JUDGED)
    j.add_argument("--workers", type=int, default=2)
    b = sp.add_parser("build"); b.add_argument("--n-eval", type=int, default=200)
    b.add_argument("--n-val", type=int, default=300); b.add_argument("--inp", default="")
    b.add_argument("--no-judge", action="store_true", help="bỏ bước giám khảo, giữ mọi mẫu không bị loại")
    b.add_argument("--eval-from", default="", help="giữ nguyên bộ đề này (vd data/office/eval_office.jsonl), loại đề trùng khỏi train")
    b.add_argument("--out-dir", default="data/office")
    for p in (g, j):
        p.add_argument("--ollama", default="http://127.0.0.1:11434")
    a = ap.parse_args()
    {"gen": cmd_gen, "judge": cmd_judge, "build": cmd_build}[a.cmd](a)


if __name__ == "__main__":
    main()
