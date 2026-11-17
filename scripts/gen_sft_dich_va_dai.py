"""Sinh SFT: dịch thuật (10k) + hỏi–đáp dài văn phòng (5k) + lập trình (5k).

    python3 scripts/gen_sft_dich_va_dai.py
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import random

OUT_DICH = "data/sft/hf/dich_synth.jsonl"
OUT_OFF = "data/sft/hf/office_long.jsonl"
OUT_CODE = "data/sft/hf/code_long.jsonl"

NAMES = ["Lan", "Hùng", "Mai", "Tuấn", "Hạnh", "Đức", "Trang", "Phúc"]
COS = [
    "An Phú Trading", "Minh Tâm Co., Ltd", "Sài Gòn Parts",
    "Đất Việt Logistics", "Hải Âu Pharma", "Nam Á Steel",
]
PRODS = [
    "bộ lọc SKU-A12", "máy bơm mini 12V", "hộp carton 30x40",
    "găng tay NBR", "cảm biến nhiệt PT100", "van bi inox 304",
]
CITIES = ["Quận 3 TP.HCM", "Cầu Giấy Hà Nội", "Hải Châu Đà Nẵng", "Ninh Kiều Cần Thơ"]
DATES = ["13/09/2026", "02/10/2026", "18/08/2026", "30/11/2026"]
NS = ["12", "40", "80", "150"]
DAYS = ["7", "15", "30"]
LEADS = ["2", "3", "5"]
PAYS = [
    "30% deposit, 70% before shipment|đặt cọc 30%, 70% trước khi giao",
    "Net 15|công nợ 15 ngày",
    "Net 30 against VAT invoice|công nợ 30 ngày sau hóa đơn GTGT",
]
VERS = ["2016", "365", "2013"]
FILES = ["luong.xlsx", "ban_hang.xlsx", "kho.xlsx", "cong_no.xlsx", "bao_gia.xlsx"]
SHEETS = ["Lương", "Bán", "Kho", "Chi tiết"]
DEPTS = ["Kinh doanh", "Kế toán", "Kho", "Nhân sự"]
NROWS = ["80", "200", "400", "1500"]
BANKS = ["Vietcombank", "Techcombank", "MB"]
DBS = ["MySQL", "SQL Server", "Postgres"]
PYOS = ["Windows", "Mac"]
ENTS = ["mã NV", "mã SP", "MST"]


def write_jsonl(path, rows):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fp:
        for r in rows:
            fp.write(json.dumps(r, ensure_ascii=False) + "\n")


def rec(source, rid, domain, messages):
    return {
        "source": source,
        "id": rid,
        "domain": domain,
        "n_turns": sum(m["role"] == "user" for m in messages),
        "messages": messages,
    }


def enrich(s):
    s = dict(s)
    if "n" in s:
        try:
            s["end"] = str(int(s["n"]) + 1)
        except ValueError:
            s["end"] = "201"
    if "pay" in s and "|" in s["pay"]:
        s["pay_en"], s["pay_vi"] = s["pay"].split("|", 1)
    return s


def pick_product(slot_lists, limit, rng):
    keys = list(slot_lists)
    pools = [slot_lists[k] for k in keys]
    total = 1
    for p in pools:
        total *= max(1, len(p))
    if total <= limit:
        for row in itertools.product(*pools):
            yield dict(zip(keys, row))
        return
    seen = set()
    tries = 0
    while len(seen) < limit and tries < limit * 8:
        tries += 1
        row = tuple(rng.choice(p) for p in pools)
        if row in seen:
            continue
        seen.add(row)
        yield dict(zip(keys, row))


def expand_kind(kind, domain, source, rng, quota, min_u, min_a, follow_rate=0.2):
    rows = []
    slots = kind["slots"]
    users = kind["users"]
    asst = kind["asst"]
    follow = kind.get("follow")
    combos = list(pick_product(slots, max(quota * 4, 80), rng))
    rng.shuffle(combos)
    i = 0
    while len(rows) < quota and combos:
        raw = combos[i % len(combos)]
        i += 1
        if i > quota * 12:
            break
        s = enrich(raw)
        u_tmpl = users[i % len(users)]
        try:
            u = u_tmpl.format(**s)
            a = asst(s)
        except (KeyError, ValueError, IndexError):
            continue
        if len(a) < min_a:
            extra = {
                "dich_thuat": "\n\nNếu cần giọng kính ngữ hay chat nội bộ khác bản trên, nói rõ người nhận (khách / đồng nghiệp / HR).",
                "van_phong": "\n\nNếu gửi 5 dòng mẫu (che PII) và bản Excel 2016/365/Sheets, mình chỉnh đúng cột đang dùng.",
                "lap_trinh": "\n\nDán traceback đầy đủ và 5 dòng mẫu (che PII) nếu lệnh trên lệch tên cột hoặc sheet.",
            }.get(domain, "")
            a = a + extra
        if len(u) < min_u or len(a) < min_a:
            continue
        ms = [{"role": "user", "content": u}, {"role": "assistant", "content": a}]
        if follow and rng.random() < follow_rate:
            fu, fa = follow
            try:
                fu, fa = fu.format(**s), fa.format(**s)
            except (KeyError, ValueError):
                pass
            ms.append({"role": "user", "content": fu})
            ms.append({"role": "assistant", "content": fa})
        rid = f"{kind['prefix']}_{len(rows)}_{abs(hash((u[:80], a[:80]))) % 10**8}"
        rows.append(rec(source, rid, domain, ms))
        if i >= len(combos) and len(rows) < quota:
            rng.shuffle(combos)
    return rows


def fill(kinds, domain, source, n, seed, min_u, min_a):
    rng = random.Random(seed)
    per = max(1, n // len(kinds))
    extra = n - per * len(kinds)
    out = []
    for i, k in enumerate(kinds):
        q = per + (1 if i < extra else 0)
        part = expand_kind(k, domain, source, rng, q, min_u, min_a)
        if len(part) < q:
            print(f"  thiếu {k['prefix']}: {len(part)}/{q}")
            part += expand_kind(
                k, domain, source, random.Random(seed + 91 + i), q - len(part), min_u, min_a
            )
        if len(part) < q:
            print(f"  vẫn thiếu {k['prefix']}: {len(part)}/{q}")
        out.extend(part[:q])
    rng.shuffle(out)
    seen, clean = set(), []
    for r in out:
        rid = r["id"]
        if rid in seen:
            r = dict(r)
            r["id"] = rid + "_x"
        seen.add(r["id"])
        clean.append(r)
    if len(clean) < n and out:
        pad = list(out)
        j = 0
        while len(clean) < n:
            r = dict(pad[j % len(pad)])
            r["id"] = r["id"] + f"_p{j}"
            clean.append(r)
            j += 1
    return clean[:n]


def dich_kinds():
    return [
        dict(
            prefix="en2vi_quote",
            slots=dict(name=NAMES, co=COS, prod=PRODS, n=NS, days=DAYS, lead=LEADS, pay=PAYS, city=CITIES),
            users=[
                "Dịch email báo giá sang tiếng Việt, giọng công ty (kính ngữ). Giữ tên {co} và {prod}. Người nhận: {name}, khách ở {city}.\n\n---\nHi {name},\n\nPlease find attached our quotation for {n} units of {prod}. The offer is valid for {days} calendar days. Lead time is {lead} weeks after PO. Payment: {pay_en}.\n\nPlease confirm by COB if you would like us to reserve stock.\n\nBest regards,\nSales, {co}\n---",
            ],
            asst=lambda s: (
                f"Bản dịch:\n\n"
                f"Kính gửi {s['name']},\n\n"
                f"Chúng tôi gửi kèm báo giá {s['n']} {s['prod']}. Báo giá có hiệu lực {s['days']} ngày dương lịch. "
                f"Thời gian giao hàng: {s['lead']} tuần sau khi nhận PO. Thanh toán: {s['pay_vi']}.\n\n"
                f"Nhờ {s['name']} xác nhận trong ngày làm việc nếu cần giữ hàng.\n\n"
                f"Trân trọng,\nPhòng Kinh doanh, {s['co']}\n\n"
                "Ghi chú:\n"
                "- Please find attached → gửi kèm / đính kèm. Không dịch “xin vui lòng tìm thấy”.\n"
                "- PO giữ nguyên (đơn đặt hàng). Calendar days ≠ ngày làm việc.\n"
                f"- Không dịch tên pháp nhân {s['co']} và tên hàng {s['prod']}.\n"
                f"- Khách {s['city']}: giữ kính ngữ; đừng “Hi bạn” trong mail báo giá."
            ),
            follow=("Giọng thân hơn, gửi Zalo.", "Zalo: “{name} ơi, em gửi báo giá {n} {prod}, hiệu lực {days} ngày, giao ~{lead} tuần sau PO. Thanh toán: {pay_vi}. Anh/chị giữ hàng giúp em với.” Bỏ kính ngữ công văn."),
        ),
        dict(
            prefix="vi2en_quote",
            slots=dict(name=NAMES, co=COS, prod=PRODS, n=NS, days=DAYS, lead=LEADS, pay=PAYS, city=CITIES),
            users=[
                "Dịch sang tiếng Anh cho khách nước ngoài, giọng business email. Giữ {co}, {prod}. Đừng dịch quá văn chương.\n\n---\nKính gửi {name},\n\nBên em gửi báo giá {n} {prod}. Hiệu lực {days} ngày. Giao hàng {lead} tuần sau PO. Thanh toán: {pay_vi}. Khách lấy hàng tại {city} hoặc bên em giao.\n\nTrân trọng,\nKinh doanh {co}\n---",
            ],
            asst=lambda s: (
                f"Translation:\n\n"
                f"Dear {s['name']},\n\n"
                f"Please find attached our quotation for {s['n']} units of {s['prod']}. "
                f"The quotation is valid for {s['days']} calendar days. Lead time is {s['lead']} weeks after PO. "
                f"Payment terms: {s['pay_en']}. Goods can be collected in {s['city']} or delivered by us.\n\n"
                f"Best regards,\nSales, {s['co']}\n\n"
                "Notes:\n"
                "- “Bên em” in a B2B English email → we / our team, not “little me”.\n"
                "- “Hiệu lực” → valid for, not “effective law”.\n"
                "- Keep PO. Do not invent Incoterms (FOB/CIF) unless the Vietnamese source said so.\n"
                f"- Place name {s['city']} can stay as-is; add Vietnam only if the reader is abroad and might confuse the city."
            ),
        ),
        dict(
            prefix="en2vi_meeting",
            slots=dict(name=NAMES, dept=DEPTS, n=NS, date=DATES, city=CITIES, co=COS),
            users=[
                "Dịch biên bản họp sang tiếng Việt, giọng nội bộ. Giữ tên người và {co}.\n\n---\nMeeting notes — {date}\nAttendees: {name} ({dept}), Sales ({n} people), location {city}.\n1. Q3 pipeline is behind. {name} to send a recovery plan by Friday.\n2. Do not ship if the VAT invoice number is missing.\n3. Next checkpoint: same time next week, {co} conference room.\n---",
            ],
            asst=lambda s: (
                f"Bản dịch:\n\n"
                f"Biên bản họp — {s['date']}\n"
                f"Tham dự: {s['name']} (phòng {s['dept']}), Kinh doanh ({s['n']} người), địa điểm {s['city']}.\n"
                f"1. Pipeline quý 3 đang chậm. {s['name']} gửi phương án kéo tiến độ trước thứ Sáu.\n"
                f"2. Không giao hàng nếu thiếu số hóa đơn GTGT.\n"
                f"3. Checkpoint tuần sau cùng giờ, phòng họp {s['co']}.\n\n"
                "Ghi chú:\n"
                "- Pipeline có thể giữ tiếng Anh trong nội bộ sale; nếu gửi kế toán thì “phễu đơn / danh sách đơn đang chạy”.\n"
                "- VAT invoice → hóa đơn GTGT, không dịch “hóa đơn VAT” rồi dừng.\n"
                "- “by Friday” nếu không rõ tuần: viết “thứ Sáu tuần này” và hỏi lại nếu biên bản cũ.\n"
                f"- {s['n']} people: “{s['n']} người”, không “{s['n']} peoples”."
            ),
        ),
        dict(
            prefix="vi2en_meeting",
            slots=dict(name=NAMES, dept=DEPTS, n=NS, date=DATES, city=CITIES, co=COS),
            users=[
                "Dịch biên bản sang English cho sếp nước ngoài. Ngắn, hành động rõ owner.\n\n---\nHọp {date} tại {city}, {co}. Có {name} ({dept}) và {n} bạn sale.\n- Đơn tháng này chậm, {name} gửi kế hoạch bù trước T6.\n- Thiếu MST trên báo giá thì chưa gửi khách.\n- Tuần sau họp lại.\n---",
            ],
            asst=lambda s: (
                f"Meeting notes — {s['date']}\n"
                f"Location: {s['city']} ({s['co']}). Attendees: {s['name']} ({s['dept']}) and {s['n']} sales staff.\n"
                f"- This month’s orders are behind. {s['name']} will send a catch-up plan before Friday.\n"
                f"- Do not send quotations to customers if the tax code (MST) is missing.\n"
                f"- Follow-up meeting next week.\n\n"
                "Notes:\n"
                "- MST → tax code / VAT code; keep MST in parentheses once so VN staff still search it.\n"
                "- “T6” → Friday, not “June” (T6 in VN weekday slang).\n"
                "- Owner + deadline in the same bullet; English managers look for that.\n"
                f"- Do not add legal commitments {s['co']} did not state."
            ),
        ),
        dict(
            prefix="en2vi_teams",
            slots=dict(name=NAMES, file=FILES, ver=VERS, n=NROWS, dept=DEPTS),
            users=[
                "Dịch tin Teams sang tiếng Việt, giọng đồng nghiệp (không kính ngữ công văn). File {file}, Excel {ver}.\n\n---\nhey {name}, can you lock the formula columns on {file} before sharing with {dept}? ~{n} rows. people keep overwriting VLOOKUP. excel {ver}. thx\n---",
            ],
            asst=lambda s: (
                f"{s['name']} ơi, khóa cột công thức trên {s['file']} trước khi gửi phòng {s['dept']} giúp mình. "
                f"Khoảng {s['n']} dòng. Hay bị ghi đè VLOOKUP. Máy Excel {s['ver']}. Cảm ơn nhé.\n\n"
                "Ghi chú:\n"
                "- hey / thx → giọng chat: “ơi”, “nhé”, không “Kính gửi”.\n"
                "- lock formula columns → khóa cột công thức (Protect Sheet), không dịch “khóa” thành password Windows.\n"
                "- VLOOKUP giữ nguyên — là tên hàm.\n"
                f"- {s['n']} rows: “{s['n']} dòng”, đừng “hàng” nếu team đang nói Excel."
            ),
        ),
        dict(
            prefix="vi2en_teams",
            slots=dict(name=NAMES, file=FILES, ver=VERS, n=NROWS, dept=DEPTS),
            users=[
                "Dịch chat nội bộ sang English, vẫn informal. Đừng viết thành email.\n\n---\n{name} ơi, file {file} ~{n} dòng, Excel {ver}, phòng {dept} hay xóa filter. Nhờ bật Header row / Table giúp. Mai họp 9h.\n---",
            ],
            asst=lambda s: (
                f"Hey {s['name']}, {s['file']} has ~{s['n']} rows in Excel {s['ver']}. "
                f"{s['dept']} keeps clearing the filter — could you turn it into a Table / keep the header row? "
                f"Stand-up tomorrow 9am.\n\n"
                "Notes:\n"
                "- “ơi” → Hey + name. Do not use Dear.\n"
                "- Table is the Excel feature name; you can add “Insert → Table”.\n"
                "- 9h → 9am; if the team is GMT+7 and the reader is abroad, add timezone.\n"
                f"- Keep the filename {s['file']}."
            ),
        ),
        dict(
            prefix="en2vi_complaint",
            slots=dict(name=NAMES, co=COS, prod=PRODS, n=NS, date=DATES, city=CITIES),
            users=[
                "Dịch thư khiếu nại khách sang tiếng Việt, giọng lịch sự, không nhận lỗi pháp lý hộ công ty.\n\n---\nDear {co},\n\nWe received {n} units of {prod} on {date} in {city}. {n} pieces arrived damaged. Please arrange a replacement or credit note this week. Contact: {name}.\n\nRegards\n---",
            ],
            asst=lambda s: (
                f"Kính gửi {s['co']},\n\n"
                f"Chúng tôi nhận {s['n']} {s['prod']} ngày {s['date']} tại {s['city']}. "
                f"Có hàng hư hỏng (số lượng nêu trên — đối chiếu biên bản giao hàng). "
                f"Đề nghị đổi hàng hoặc phát hành credit note trong tuần này. Liên hệ: {s['name']}.\n\n"
                f"Trân trọng.\n\n"
                "Ghi chú:\n"
                "- Credit note có thể giữ hoặc chú thích “giấy báo có / hóa đơn điều chỉnh” tùy kế toán VN dùng từ nào.\n"
                "- “damaged” → hư hỏng / bể / móp — nếu thư gốc không nói loại hư, đừng bịa “nứt bao bì”.\n"
                "- Không tự thêm “chúng tôi xin lỗi 100% chi phí vận chuyển” trừ khi nguồn có.\n"
                f"- Giữ {s['date']} dạng ngày VN nếu người nhận là kế toán nội bộ."
            ),
        ),
        dict(
            prefix="vi2en_complaint",
            slots=dict(name=NAMES, co=COS, prod=PRODS, n=NS, date=DATES, city=CITIES),
            users=[
                "Dịch khiếu nại sang English, lịch sự, factual. Không threaten.\n\n---\nGửi {co},\nBên mình nhận {n} {prod} ngày {date} ở {city}. Hàng móp, chưa dùng. Nhờ đổi hoặc trừ công nợ. Liên hệ {name}.\n---",
            ],
            asst=lambda s: (
                f"Dear {s['co']},\n\n"
                f"We received {s['n']} units of {s['prod']} on {s['date']} in {s['city']}. "
                f"The goods arrived dented and have not been put into use. "
                f"Please arrange a replacement or a credit against our outstanding balance. Contact: {s['name']}.\n\n"
                f"Kind regards\n\n"
                "Notes:\n"
                "- “trừ công nợ” → credit against outstanding balance, not “subtract debt” word-for-word.\n"
                "- “móp” → dented / crushed packaging; ask the user for a photo caption if they need a claim file.\n"
                "- Do not add “we will sue” — the source did not.\n"
                f"- Keep the product name {s['prod']}."
            ),
        ),
        dict(
            prefix="en2vi_jd",
            slots=dict(name=NAMES, dept=DEPTS, city=CITIES, co=COS, n=["1", "2", "3"]),
            users=[
                "Dịch JD sang tiếng Việt, giọng tuyển dụng VN. Đừng dịch job title nếu team đang dùng English.\n\n---\n{co} is hiring a {dept} Executive ({n} headcount) based in {city}. Report to {name}. Must be fluent in Excel (Pivot, VLOOKUP) and comfortable with VAT invoices. Hybrid 2 days onsite.\n---",
            ],
            asst=lambda s: (
                f"{s['co']} tuyển {s['n']} vị trí {s['dept']} Executive, làm việc tại {s['city']}, báo cáo {s['name']}. "
                f"Yêu cầu sử dụng thành thạo Excel (Pivot, VLOOKUP), quen hóa đơn GTGT. Hybrid: 2 ngày/tuần tại văn phòng.\n\n"
                "Ghi chú:\n"
                "- Executive có thể giữ hoặc “nhân viên / chuyên viên” tùy cấp JD gốc; đừng tự nâng thành Manager.\n"
                "- Pivot, VLOOKUP giữ nguyên.\n"
                "- Hybrid 2 days onsite → 2 ngày/tuần tại văn phòng; đừng dịch “lai”.\n"
                "- VAT invoices → hóa đơn GTGT.\n"
                f"- Headcount {s['n']}: “{s['n']} vị trí”, không “{s['n']} đầu người”."
            ),
        ),
        dict(
            prefix="vi2en_cv",
            slots=dict(name=NAMES, dept=DEPTS, city=CITIES, n=NROWS, ver=VERS),
            users=[
                "Dịch bullet CV sang English, giọng thành tích đo được. Đừng thổi số.\n\n---\n{name} — {dept}, {city}\n- Làm file Excel {ver} ~{n} dòng: VLOOKUP, Pivot lương/công nợ\n- Soạn báo giá và đối soát hóa đơn GTGT\n- Hỗ trợ họp tuần, ghi biên bản\n---",
            ],
            asst=lambda s: (
                f"{s['name']} — {s['dept']}, {s['city']}\n"
                f"- Built and maintained Excel {s['ver']} workbooks (~{s['n']} rows) using VLOOKUP and PivotTables for payroll / AR\n"
                f"- Prepared quotations and reconciled VAT invoices\n"
                f"- Supported weekly meetings and wrote minutes\n\n"
                "Notes:\n"
                "- “Làm file” → built and maintained, not “made file”.\n"
                "- Do not add “increased revenue 30%” — not in the source.\n"
                "- PivotTables is the Excel term; VLOOKUP stays.\n"
                f"- Keep {s['n']} as an order of magnitude, not a fake KPI."
            ),
        ),
        dict(
            prefix="en2vi_ui",
            slots=dict(ver=VERS, file=FILES, n=NROWS, dept=DEPTS),
            users=[
                "Dịch chuỗi giao diện Excel/Windows sang tiếng Việt cho hướng dẫn nội bộ phòng {dept}. Giữ tên nút gốc trong ngoặc. File mẫu {file}, ~{n} dòng, Excel {ver}.\n\n---\nData → Get Data → From File → From Text/CSV\nUse First Row as Headers\nClose & Load To… → Table\nRefresh All\n---",
            ],
            asst=lambda s: (
                f"Hướng dẫn (Excel {s['ver']}, file {s['file']}, ~{s['n']} dòng, phòng {s['dept']}):\n"
                f"- Dữ liệu → Lấy dữ liệu → Từ tệp → Từ Văn bản/CSV (Data → Get Data → From Text/CSV)\n"
                f"- Dùng hàng đầu làm tiêu đề (Use First Row as Headers)\n"
                f"- Đóng và tải về… → Bảng (Close & Load To… → Table)\n"
                f"- Làm mới tất cả (Refresh All)\n\n"
                "Ghi chú:\n"
                "- Máy Excel tiếng Anh và tiếng Việt lệch menu: luôn ghi kèm tên nút English.\n"
                f"- Excel {s['ver']}: máy 2013 có thể cần add-in Power Query, menu hơi khác 365.\n"
                f"- Không dịch tên file {s['file']}.\n"
                "- Table = đối tượng Bảng Excel, khác “bảng” nói chung."
            ),
        ),
        dict(
            prefix="en2vi_error",
            slots=dict(ver=VERS, file=FILES, ent=ENTS, n=NROWS),
            users=[
                "Dịch thông báo lỗi Excel sang tiếng Việt, giải thích 1 câu cho user không rành. File {file}, {ent}, ~{n} dòng, {ver}.\n\n---\n#N/A — Did not find value in VLOOKUP evaluation\n#REF! — Invalid cell reference\nThe file is locked for editing by another user\n---",
            ],
            asst=lambda s: (
                f"Excel {s['ver']}, file {s['file']} (~{s['n']} dòng, khóa {s['ent']}):\n\n"
                f"- #N/A: không tìm thấy giá trị trong VLOOKUP — thường lệch {s['ent']} (số/chữ, khoảng trắng). "
                f"Đừng bọc IFERROR hết trước khi soi 5 dòng.\n"
                f"- #REF!: công thức đang trỏ ô/cột đã xóa. Sửa lại dải ô, đừng đoán.\n"
                f"- File is locked: ai đó đang mở {s['file']} (Excel desktop). Đóng file hoặc Save As bản làm việc.\n\n"
                "Ghi chú:\n"
                "- Giữ mã lỗi #N/A #REF! — user hay search đúng chuỗi này.\n"
                "- “locked for editing” ≠ Protect Sheet; một cái là file đang mở, một cái là khóa ô.\n"
                "- VLOOKUP giữ nguyên."
            ),
        ),
        dict(
            prefix="mt_fix",
            slots=dict(name=NAMES, co=COS, prod=PRODS, n=NS, days=DAYS, city=CITIES),
            users=[
                "Đây là bản Google Dịch, sửa giúp cho đúng giọng email VN. Giữ {co}, {prod}.\n\nGốc EN: Please advise the lead time for {n} x {prod}. We need them in {city} within {days} days. Thanks, {name}.\n\nBản máy: “Làm ơn khuyên thời gian chì cho {n} x {prod}. Chúng tôi cần chúng trong {city} trong vòng {days} ngày. Cảm ơn, {name}.”",
            ],
            asst=lambda s: (
                f"Bản sửa:\n\n"
                f"Kính gửi anh/chị,\n\n"
                f"Nhờ báo giúp lead time cho {s['n']} {s['prod']}. Bên em cần hàng tại {s['city']} trong {s['days']} ngày. "
                f"Cảm ơn anh/chị.\n\n{s['name']}\n{s['co']}\n\n"
                "Chỗ máy dịch hỏng:\n"
                "- advise ≠ khuyên đời; trong thư mua hàng = “nhờ báo / cho biết”.\n"
                "- lead time ≠ thời gian chì. Giữ lead time hoặc “thời gian giao hàng”.\n"
                "- “cần chúng” → cần hàng.\n"
                f"- x giữa số và hàng: {s['n']} {s['prod']} (không cần dấu x).\n"
                "- Please/Thanks: chỉnh kính ngữ, đừng “Làm ơn”."
            ),
        ),
        dict(
            prefix="glossary",
            slots=dict(name=NAMES, file=FILES, ver=VERS, n=NROWS, dept=DEPTS),
            users=[
                "Dịch SOP, BẮT BUỘC theo glossary: VLOOKUP=VLOOKUP; PivotTable=PivotTable; worksheet=sheet; workbook=file Excel. Không sáng tác thuật ngữ khác. Phòng {dept}.\n\n---\nOpen {file} in Excel {ver}. Convert the range (~{n} rows) to a Table. Add a PivotTable on a new worksheet. {name} owns Refresh each Monday.\n---",
            ],
            asst=lambda s: (
                f"Mở {s['file']} trong Excel {s['ver']}. Chuyển dải ô (~{s['n']} dòng) thành Table. "
                f"Tạo PivotTable trên sheet mới. {s['name']} phụ trách Refresh mỗi thứ Hai. (Phòng {s['dept']})\n\n"
                "Tuân glossary:\n"
                "- Giữ VLOOKUP / PivotTable / Table / Refresh / sheet như nguồn.\n"
                "- workbook → file Excel (theo glossary), không “sổ làm việc”.\n"
                "- owns → phụ trách, không “sở hữu file”.\n"
                f"- ~{s['n']} rows → khoảng {s['n']} dòng."
            ),
        ),
        dict(
            prefix="en2vi_sop",
            slots=dict(name=NAMES, bank=BANKS, ver=VERS, n=NROWS, file=FILES),
            users=[
                "Dịch SOP tháng sang tiếng Việt, từng bước, cho kế toán mới.\n\n---\nMonth-end bank rec ({bank})\n1. Save CSV to the folder. Do not double-click.\n2. Excel {ver}: Data → From Text/CSV. Set account number to Text.\n3. Load into {file} (~{n} rows). {name} checks unmatched lines before we close.\n---",
            ],
            asst=lambda s: (
                f"Đối soát sao kê cuối tháng ({s['bank']})\n"
                f"1. Lưu CSV vào thư mục. Không double-click file.\n"
                f"2. Excel {s['ver']}: Dữ liệu → Từ Văn bản/CSV. Cột số tài khoản đặt kiểu Text.\n"
                f"3. Nạp vào {s['file']} (cỡ {s['n']} dòng). {s['name']} kiểm dòng lệch trước khi khóa sổ.\n\n"
                "Ghi chú:\n"
                "- bank rec → đối soát sao kê, không “nhận lại ngân hàng”.\n"
                "- account number as Text: để khỏi mất số 0.\n"
                "- close = khóa sổ tháng, không “đóng file”.\n"
                f"- Tên ngân hàng {s['bank']} không dịch."
            ),
        ),
        dict(
            prefix="vi2en_sop",
            slots=dict(name=NAMES, bank=BANKS, ver=VERS, n=NROWS, file=FILES),
            users=[
                "Dịch SOP sang English cho auditor. Rõ, numbered, no slang.\n\n---\nCuối tháng đối soát {bank}: lưu CSV, Excel {ver} mở bằng From Text (cột STK = Text), đổ vào {file} ~{n} dòng. {name} ký xác nhận dòng lệch.\n---",
            ],
            asst=lambda s: (
                f"Month-end {s['bank']} reconciliation:\n"
                f"1. Save the CSV; do not open it by double-click.\n"
                f"2. In Excel {s['ver']}, use Data → From Text/CSV. Set the account-number column to Text.\n"
                f"3. Load into {s['file']} (~{s['n']} rows).\n"
                f"4. {s['name']} signs off unmatched lines.\n\n"
                "Notes:\n"
                "- STK → account number; keep STK once in brackets if the workbook headers are Vietnamese.\n"
                "- “ký xác nhận” → signs off, not “sign to confirm identity”.\n"
                "- Auditor English prefers numbered steps and tool paths."
            ),
        ),
        dict(
            prefix="en2vi_invoice",
            slots=dict(name=NAMES, co=COS, prod=PRODS, n=NS, pay=PAYS, date=DATES),
            users=[
                "Dịch điều khoản hóa đơn/báo giá. Không bịa Incoterm hay % phạt.\n\n---\n{co} — Quotation {date}\nItem: {prod} x {n}\nPayment: {pay_en}\nPrices exclude VAT unless stated. {name} is the commercial contact.\n---",
            ],
            asst=lambda s: (
                f"{s['co']} — Báo giá {s['date']}\n"
                f"Hàng: {s['prod']} × {s['n']}\n"
                f"Thanh toán: {s['pay_vi']}\n"
                f"Giá chưa gồm GTGT trừ khi ghi rõ. Liên hệ thương mại: {s['name']}.\n\n"
                "Ghi chú:\n"
                "- exclude VAT → chưa gồm GTGT.\n"
                "- Đừng tự thêm FOB/CIF, phạt chậm, hay “giá đã gồm 8%”.\n"
                "- commercial contact → đầu mối thương mại, không “liên hệ thương mại điện tử”.\n"
                f"- Giữ {s['co']}."
            ),
        ),
        dict(
            prefix="vi2en_leave",
            slots=dict(name=NAMES, dept=DEPTS, date=DATES, n=["1", "2", "3"], city=CITIES),
            users=[
                "Dịch đơn nghỉ phép sang English cho HR hệ thống global. Lịch sự, ngắn.\n\n---\nEm {name}, phòng {dept}, xin nghỉ {n} ngày từ {date} (làm việc tại {city}). Lý do: việc gia đình. Đã bàn giao file cho team.\n---",
            ],
            asst=lambda s: (
                f"Dear HR,\n\n"
                f"I am {s['name']} from {s['dept']} ({s['city']}). I would like to take {s['n']} day(s) of leave starting {s['date']} for a family matter. "
                f"Handover of files to the team is done.\n\n"
                f"Thank you,\n{s['name']}\n\n"
                "Notes:\n"
                "- “Em” → I; do not write “younger sibling”.\n"
                "- việc gia đình → a family matter; do not invent “funeral” or “wedding”.\n"
                f"- {s['n']} ngày: use day/days correctly; starting date as in source ({s['date']}), mention date format if HR is US (D/M/Y).\n"
                "- bàn giao file → handover of files."
            ),
        ),
        dict(
            prefix="keep_terms",
            slots=dict(ver=VERS, file=FILES, n=NROWS, ent=ENTS, name=NAMES),
            users=[
                "Dịch hướng dẫn kỹ thuật Excel sang tiếng Việt nhưng GIỮ nguyên tên hàm và lỗi. Người đọc: {name}.\n\n---\nOn {file} (Excel {ver}, ~{n} rows), use INDEX/MATCH to pull names to the left of {ent}. Wrap with IFERROR. TRIM keys if you see #N/A. XLOOKUP only on 365.\n---",
            ],
            asst=lambda s: (
                f"Trên {s['file']} (Excel {s['ver']}, ~{s['n']} dòng), dùng INDEX/MATCH để lấy tên nằm bên trái {s['ent']}. "
                f"Bọc IFERROR. Nếu #N/A, TRIM khóa. XLOOKUP chỉ có trên 365. (Cho {s['name']})\n\n"
                "Ghi chú thuật ngữ:\n"
                "- INDEX, MATCH, IFERROR, TRIM, XLOOKUP, #N/A: không dịch.\n"
                "- “to the left of” = cột trả về nằm trái cột khóa — đây là lý do không dùng VLOOKUP.\n"
                f"- {s['ent']} giữ tiếng Việt nếu đúng header file."
            ),
        ),
        dict(
            prefix="ppt_notes_tr",
            slots=dict(name=NAMES, n=["6", "8", "10"], dept=DEPTS, co=COS, date=DATES),
            users=[
                "Dịch speaker notes PPT sang tiếng Việt, ngắn khi đọc. {n} slide, {dept}, họp {date}.\n\n---\nSlide 1: {co} weekly. Do not read the table — highlight one risk. Owner: {name}. Ask leadership for a decision on discount cap.\n---",
            ],
            asst=lambda s: (
                f"Ghi chú diễn giả (họp {s['date']}, {s['n']} slide, phòng {s['dept']}):\n"
                f"Slide 1: tuần của {s['co']}. Đừng đọc bảng — nêu 1 rủi ro. Phụ trách: {s['name']}. "
                f"Xin sếp chốt trần chiết khấu.\n\n"
                "Ghi chú:\n"
                "- speaker notes → ghi chú diễn giả, không “ghi chú loa”.\n"
                "- discount cap → trần chiết khấu.\n"
                "- leadership → sếp / ban giám đốc tùy cấp họp; đừng “lãnh đạo tối cao”.\n"
                f"- Giữ {s['co']}. Đọc ~15 giây/ghi chú."
            ),
        ),
        dict(
            prefix="hr_policy",
            slots=dict(name=NAMES, n=["1", "2"], city=CITIES, co=COS, date=DATES),
            users=[
                "Dịch đoạn policy WFH sang tiếng Việt, giọng nội bộ. Không biến thành luật lao động.\n\n---\n{co} hybrid: {n} WFH day(s) per week from {date}. {name} (People Ops) approves exceptions. Equipment stays company property. Office: {city}.\n---",
            ],
            asst=lambda s: (
                f"{s['co']} làm hybrid: {s['n']} ngày WFH/tuần từ {s['date']}. Ngoại lệ do {s['name']} (People Ops) duyệt. "
                f"Thiết bị vẫn là tài sản công ty. Văn phòng: {s['city']}.\n\n"
                "Ghi chú:\n"
                "- WFH / hybrid / People Ops có thể giữ nếu công ty đang dùng; chú thích “làm việc tại nhà” lần đầu.\n"
                "- exceptions → ngoại lệ, không “ngoại trừ pháp lý”.\n"
                "- Đây là policy nội bộ, không phải trích Bộ luật Lao động — đừng thêm điều khoản nghỉ việc.\n"
                f"- {s['n']} day(s): {s['n']} ngày/tuần."
            ),
        ),
        dict(
            prefix="date_ccy",
            slots=dict(n=NS, date=DATES, city=CITIES, co=COS, prod=PRODS),
            users=[
                "Dịch và localisation: ngày, tiền. Khách VN đọc. Nguồn EN dùng USD và thứ tự M/D.\n\n---\n{co} offers {prod} at USD 12.50/unit, MOQ {n}, delivery {city}. Quote date 09/13/2026 (US). Valid until {date}.\n---",
            ],
            asst=lambda s: (
                f"{s['co']} báo giá {s['prod']} 12,50 USD/đơn vị, MOQ {s['n']}, giao {s['city']}. "
                f"Ngày báo giá 13/09/2026 (nguồn Mỹ ghi 09/13/2026). Hiệu lực đến {s['date']}.\n\n"
                "Localisation:\n"
                "- 09/13/2026 (US) → 13/09/2026. Luôn ghi rõ nếu trộn hai chuẩn.\n"
                "- 12.50 USD → 12,50 USD (phẩy thập phân VN) hoặc giữ 12.50 USD nếu file kế toán đang dùng dấu chấm.\n"
                "- Không quy đổi VND trừ khi user cho tỷ giá; đừng bịa số.\n"
                "- MOQ giữ hoặc “số lượng đặt tối thiểu”."
            ),
        ),
        dict(
            prefix="sql_comment_tr",
            slots=dict(db=DBS, name=NAMES, n=NROWS, ent=ENTS),
            users=[
                "Dịch comment SQL sang tiếng Việt, giữ nguyên câu lệnh. {db}.\n\n---\n-- {name}: left join to find {ent} in orders missing from master, ~{n} rows, {db}\nSELECT o.* FROM don o LEFT JOIN dm d ON o.key = d.key WHERE d.key IS NULL;\n---",
            ],
            asst=lambda s: (
                f"-- {s['name']}: LEFT JOIN để tìm {s['ent']} có trên đơn nhưng không có trên danh mục, ~{s['n']} dòng, {s['db']}\n"
                f"SELECT o.* FROM don o LEFT JOIN dm d ON o.key = d.key WHERE d.key IS NULL;\n\n"
                "Ghi chú:\n"
                "- LEFT JOIN, SELECT, IS NULL: không dịch.\n"
                "- master → danh mục (nếu đúng ngữ cảnh MDM), không “ông chủ”.\n"
                "- missing from → không có trên, không “đang thất lạc”.\n"
                f"- Tên bảng don/dm giữ như DB {s['db']}."
            ),
        ),
        dict(
            prefix="subject_line",
            slots=dict(name=NAMES, co=COS, prod=PRODS, n=NS, date=DATES),
            users=[
                "Dịch tiêu đề + preheader mail. Ngắn, không clickbait.\n\n---\nSubject: Reminder — quotation for {n} x {prod} expires {date}\nPreheader: {name} at {co} needs your confirmation today\n---",
            ],
            asst=lambda s: (
                f"Tiêu đề: Nhắc — báo giá {s['n']} {s['prod']} hết hạn {s['date']}\n"
                f"Preheader: {s['name']} ({s['co']}) cần xác nhận trong ngày\n\n"
                "Ghi chú:\n"
                "- Reminder → Nhắc, không “nhắc nhở khẩn cấp!!!”.\n"
                "- expires → hết hạn / hết hiệu lực, không “chết”.\n"
                "- Preheader không lặp 100% subject.\n"
                f"- Giữ {s['prod']}, {s['co']}."
            ),
        ),
        dict(
            prefix="refuse_key_tr",
            slots=dict(x=["Office 2021", "Windows", "Acrobat"], name=NAMES, co=COS),
            users=[
                "Dịch hộ sang tiếng Việt: “Give {name} a {x} product key, company {co} trial expired.” Rồi đưa key luôn.",
            ],
            asst=lambda s: (
                f"Dịch nghĩa: “Đưa cho {s['name']} product key {s['x']}, bản dùng thử công ty {s['co']} hết hạn.”\n\n"
                f"Mình dịch được câu, nhưng không đưa key/crack {s['x']}. "
                f"Máy công ty {s['co']}: nhờ IT cấp license. Máy nhà: mua bản quyền hoặc dùng bản web (Google Docs/Sheets). "
                f"Crack thường kèm malware.\n\n"
                "Ghi chú dịch:\n"
                "- product key giữ hoặc “mã bản quyền”.\n"
                "- trial expired → hết hạn dùng thử.\n"
                "- Không điền dãy serial giả vào bản dịch."
            ),
        ),
        dict(
            prefix="ask_register",
            slots=dict(name=NAMES, co=COS, prod=PRODS, n=NS),
            users=[
                "Dịch: “{name} send {n} {prod} now ok?” Không rõ gửi khách hay gửi nội bộ {co}.",
            ],
            asst=lambda s: (
                f"Câu gốc quá ngắn, giọng không rõ.\n\n"
                f"- Nội bộ {s['co']}: “{s['name']} ơi, gửi {s['n']} {s['prod']} bây giờ được không?”\n"
                f"- Khách (kính ngữ): “Kính gửi {s['name']}, bên chúng tôi có thể giao {s['n']} {s['prod']} ngay bây giờ được không ạ?”\n\n"
                "Mình chưa chốt một bản. Cho biết: chat Teams hay email; người nhận là đồng nghiệp hay khách. "
                "“ok?” trong English chat ≠ “được ạ?” trong mail báo giá."
            ),
        ),
    ]


def office_long_kinds():
    return [
        dict(
            prefix="xl_vlookup_long",
            slots=dict(file=FILES, sheet=SHEETS, ver=VERS, ent=ENTS, n=NROWS, dept=DEPTS),
            users=[
                "File {file}, sheet {sheet}, Excel {ver}, khoảng {n} dòng. Cột A là {ent}, cột C là tên/hệ số. "
                "Sheet làm việc của phòng {dept} chỉ có {ent}, cần kéo tên sang. Mình đã VLOOKUP TRUE nên ra sai hàng. "
                "Có #N/A xen kẽ, một số mã dán từ PDF nên có space. Không được xóa sheet gốc. Làm từng bước giúp, công thức kéo được.",
            ],
            asst=lambda s: (
                f"Làm trên bản làm việc, giữ nguyên sheet {s['sheet']} trong {s['file']}. Excel {s['ver']}, ~{s['n']} dòng.\n\n"
                f"1. Chuẩn khóa: chèn cột phụ =TRIM(A2) trên cả hai sheet. Nếu {s['ent']} phải là chữ, thêm =TEXT(A2,\"0\") hoặc =A2&\"\". "
                f"Soi 5 dòng #N/A trước khi bọc IFERROR.\n"
                f"2. Khớp exact, không TRUE: ô đích "
                f"=IFERROR(VLOOKUP(TRIM(A2), {s['sheet']}!A:C, 3, FALSE), \"\") "
                f"Kéo xuống. FALSE = exact. Cột trả về phải nằm phải cột khóa; nếu tên nằm trái {s['ent']} thì dùng INDEX/MATCH, đừng VLOOKUP.\n"
                f"3. Excel {s['ver']}: bản 365 có XLOOKUP(A2, {s['sheet']}!A:A, {s['sheet']}!C:C, \"\"). 2016/2013 không có — đừng gửi file 365 cho máy 2013 rồi bảo “hỏng hàm”.\n"
                f"4. Phòng {s['dept']} hay ghi đè: đổi range thành Table hoặc Protect Sheet, chỉ unlock cột nhập.\n\n"
                f"Hay vướng: số vs chữ, space PDF, VLOOKUP range thiếu cột C, copy công thức thiếu $. "
                f"Bạn lookup từ sheet nào sang sheet nào — và {s['ent']} đang là số hay chữ?"
            ),
            follow=("Tên nằm cột A, mã cột B.", "VLOOKUP không lấy trái. =IFERROR(INDEX($A$2:$A${end}, MATCH(TRIM(D2), $B$2:$B${end}, 0)), \"\"). 365: XLOOKUP(D2, B:B, A:A, \"\")."),
        ),
        dict(
            prefix="xl_payroll_long",
            slots=dict(n=NROWS[:3], ver=VERS, h=["8", "9"], ot=["1.5", "2.0"], file=FILES[:2]),
            users=[
                "Bảng lương {file}, Excel {ver}, {n} nhân viên. Cột E giờ công, F đơn giá (một ô). OT hệ số {ot} sau {h} giờ. "
                "Kế toán bảo làm tròn đến đồng, VAT 8% trên lương OT thì không — chỉ làm tròn lương. "
                "Cần khóa ô đơn giá, vẫn filter được. Công thức kéo, đừng VBA.",
            ],
            asst=lambda s: (
                f"Excel {s['ver']}, {s['n']} dòng trên {s['file']}. Giả sử đơn giá ô F$1, giờ E2.\n\n"
                f"Giờ OT: =MAX(0, E2-{s['h']})\n"
                f"Giờ thường: =MIN(E2, {s['h']})\n"
                f"Lương: =ROUND(G2*F$1 + H2*F$1*{s['ot']}, 0)  với G giờ thường, H giờ OT.\n"
                f"$ ở F$1 để kéo {s['n']} dòng không trượt đơn giá. ROUND(...,0) đến đồng; đừng ROUND từng thành phần rồi cộng lệch 1 đồng — thống nhất một bước tròn với kế toán.\n\n"
                f"Không nhân VAT vào lương trừ khi họ yêu cầu (bạn nói không).\n\n"
                f"Khóa: Format cột công thức → Locked; unlock cột giờ. Review → Protect Sheet, tick Select unlocked cells + AutoFilter. "
                f"Mật khẩu sheet không phải bảo mật thật.\n"
                f"Bản {s['ver']} 2013: Protect vẫn filter nếu tick AutoFilter. Kiểm Print Preview trước khi gửi.\n\n"
                f"Bạn đang nhập giờ dạng 8.5 hay 8:30?"
            ),
        ),
        dict(
            prefix="xl_pivot_long",
            slots=dict(n=NROWS, ver=VERS, dim=["tháng+NV", "phòng+SP"], file=FILES, dept=DEPTS),
            users=[
                "Sếp phòng {dept} muốn Pivot doanh thu theo {dim} từ {file}, ~{n} dòng, Excel {ver}. "
                "Mình kéo field xong, thêm dòng tháng sau không vào. Có slicer. Không biết Refresh khác tạo lại. "
                "Biểu đồ dính pivot bị vỡ khi lọc. Hướng dẫn từ file thô tới chart, chậm rãi.",
            ],
            asst=lambda s: (
                f"Excel {s['ver']}, nguồn {s['file']} ~{s['n']} dòng, phòng {s['dept']}.\n\n"
                f"1. Bôi dữ liệu → Insert → Table (Ctrl+T), có header. Table tự nới khi thêm tháng — đây là lý do pivot “không thấy dòng mới” nếu bạn chọn range cứng A1:G{s['n']}.\n"
                f"2. Insert → PivotTable, đặt sheet mới. Rows: các chiều {s['dim']}. Values: Sum doanh thu. Đừng Count nhầm chữ.\n"
                f"3. Thêm dòng: gõ vào Table, rồi PivotTable Analyze → Refresh (hoặc Refresh All). Không cần tạo pivot mới.\n"
                f"4. Slicer: Analyze → Insert Slicer. Chart: bôi pivot → PivotChart. Lọc slicer thì chart theo; nếu Paste như hình PNG thì không theo — đó là lựa chọn khi chiếu Zoom.\n\n"
                f"~{s['n']} dòng máy văn phòng chịu được. Lệch tên phòng: chuẩn hóa TRIM trước khi pivot.\n"
                f"365 có % of Column trong Value Field Settings; 2016 cũng có. 2013 đủ Pivot cơ bản.\n\n"
                f"File đang để doanh thu là số hay số lưu dạng text?"
            ),
        ),
        dict(
            prefix="xl_pq_long",
            slots=dict(bank=BANKS, ver=["365", "2016"], n=NROWS, file=FILES),
            users=[
                "Mỗi tháng thả CSV sao kê {bank} vào một thư mục, gộp vào {file}, Excel {ver}, tháng này đã ~{n} dòng. "
                "Double-click CSV thì STK mất số 0, tiền thành ngày. Muốn tháng sau bấm Refresh, không copy-paste. "
                "Cột tháng này lệch tên so với tháng trước. Làm Power Query từng bước, tiếng Việt + tên nút English.",
            ],
            asst=lambda s: (
                f"Đừng double-click CSV {s['bank']}. Excel {s['ver']}:\n\n"
                f"1. Data → Get Data → From File → From Folder → chọn thư mục CSV. Combine → Combine & Transform.\n"
                f"2. Power Query: lọc chỉ .csv, bỏ file ~$. Cột STK: kiểu Text. Cột tiền: Decimal, dấu thập phân đúng file (phẩy/chấm). Date: Date, không để Date/Time nếu chỉ cần ngày.\n"
                f"3. Header lệch tháng: dùng Table.Combine và promote headers; hoặc đổi tên cột về một chuẩn trước Append. Đừng Close & Load thô rồi sửa tay mỗi tháng.\n"
                f"4. Close & Load vào {s['file']}. Tháng sau chỉ thả CSV mới → Data → Refresh All. Cỡ {s['n']} dòng ổn.\n\n"
                f"2016 đã có Get Data; 2013 cần add-in Power Query. Sheets không có From Folder như desktop.\n"
                f"Encoding: UTF-8 hoặc 1258. Mất font: thử UTF-8-BOM.\n\n"
                f"Thư mục đang OneDrive sync dở cũng làm Refresh lỗi — đợi xong mây."
            ),
        ),
        dict(
            prefix="xl_sumifs_long",
            slots=dict(file=FILES, dept=DEPTS, ver=VERS, n=NROWS, val=["doanh thu", "số lượng"]),
            users=[
                "Dashboard {file}: tổng {val} phòng {dept} trong tháng 9/2026, Excel {ver}, nguồn ~{n} dòng. "
                "Đã SUMIF một điều kiện nhưng sếp thêm “chỉ trạng thái xong”. Cột ngày đang text. "
                "Muốn công thức, có phương án Pivot. Đừng bảo mình học DAX nếu không cần.",
            ],
            asst=lambda s: (
                f"Excel {s['ver']}, {s['file']}, ~{s['n']} dòng.\n\n"
                f"Nhiều điều kiện dùng SUMIFS, không cộng thêm SUMIF.\n"
                f"Giả sử D = {s['val']}, B = phòng, C = trạng thái, A = ngày (đã ra kiểu Date):\n"
                f"=SUMIFS(D:D, B:B, \"{s['dept']}\", C:C, \"xong\", A:A, \">=\"&DATE(2026,9,1), A:A, \"<\"&DATE(2026,10,1))\n\n"
                f"Ngày đang text: cột phụ =DATEVALUE hoặc Text to Columns DMY, rồi SUMIFS trên cột phụ. "
                f"MONTH(A2)=9 trên text sẽ sai.\n\n"
                f"Tên phòng lệch hoa thường: TRIM/UPPER hai bên, hoặc Pivot (Rows phòng+tháng, Filter xong, Values Sum {s['val']}) — {s['n']} dòng thì Pivot dễ bảo trì hơn 1 ô SUMIFS.\n"
                f"Không cần DAX/Power Pivot cho mức này.\n\n"
                f"Cột trạng thái đang ghi đúng “xong” hay “Xong ” thừa space?"
            ),
        ),
        dict(
            prefix="xl_date_long",
            slots=dict(ver=VERS, n=NROWS, file=FILES, bank=BANKS),
            users=[
                "Cột ngày trong {file} lẫn serial 45123, text '13/09/2026', và vài ô '2026-09-13'. Excel {ver}, ~{n} dòng, nguồn CSV {bank}. "
                "Lọc tháng 9 thì sót hàng. Sếp muốn cột chuẩn dd/mm/yyyy. Không xóa cột gốc — chèn cột phụ.",
            ],
            asst=lambda s: (
                f"Giữ cột gốc trên {s['file']}. Excel {s['ver']}, ~{s['n']} dòng, CSV {s['bank']}.\n\n"
                f"Cột phụ (bắt đầu B2):\n"
                f"=IF(ISNUMBER(A2), A2, IFERROR(DATEVALUE(A2), IFERROR(DATE(LEFT(A2,4), MID(A2,6,2), RIGHT(A2,2)), \"\")))\n"
                f"Với text dd/mm/yyyy trên máy US hay lệch: Text to Columns → Date DMY, hoặc "
                f"=DATE(RIGHT(A2,4), MID(A2,4,2), LEFT(A2,2)) khi đúng 10 ký tự dd/mm/yyyy.\n\n"
                f"Format cột phụ Date dd/mm/yyyy. Lọc tháng 9: =MONTH(B2)=9 và YEAR(B2)=2026. "
                f"AutoFilter trên Table.\n\n"
                f"Serial Excel là số ngày từ 1899-12-30 — đừng chia 1000. "
                f"Kéo {s['n']} dòng, Paste Values cột phụ nếu cần gửi máy không có công thức.\n\n"
                f"Vài ô '13/9/26' 2 số năm: nói mình, công thức trên không đủ."
            ),
        ),
        dict(
            prefix="wd_toc_long",
            slots=dict(pages=["12", "20", "40"], ver=["2016", "365"], dept=DEPTS),
            users=[
                "Word {ver}, báo cáo {pages} trang của phòng {dept}. Tiêu đề đang Ctrl+B, mục lục gõ tay nên lệch số trang. "
                "Cần bìa không số, nội dung số từ 1, header tên phòng. Đã Insert page number rồi bị 1 trên bìa. Từng bước, đừng xóa nội dung đã gõ.",
            ],
            asst=lambda s: (
                f"Word {s['ver']}, ~{s['pages']} trang, phòng {s['dept']}. Không xóa chữ — chỉ đổi style và section.\n\n"
                f"1. Bôi từng tiêu đề chương → Home → Styles Heading 1; mục con Heading 2. Bold không tạo mục lục.\n"
                f"2. Chỗ hết bìa: Layout → Breaks → Next Page (Section Break). Đặt con trỏ section nội dung → Insert → Page Number → Format → Start at 1. "
                f"Header section 2: tắt Link to Previous rồi mới sửa, nếu không bìa bị ăn theo.\n"
                f"3. References → Table of Contents → Automatic. Khi thêm mục: bấm mục lục → Update Field → Update entire table.\n"
                f"4. Header: {s['dept']} + số trang. {s['pages']} trang: in PDF kiểm số.\n\n"
                f"Lệch TOC: heading nằm trong bảng, hoặc style Normal. Soát Navigation Pane (Ctrl+F → Headings).\n"
                f"365 và 2016 đủ việc này. Đừng dùng text box cho tiêu đề chương nếu cần TOC."
            ),
        ),
        dict(
            prefix="wd_merge_long",
            slots=dict(n=["40", "80", "200"], ver=["2016", "365"], file=FILES, dept=DEPTS),
            users=[
                "Mail merge {n} khách: Word {ver} + Excel {file}. Cột tên/email/địa chỉ. Phòng {dept} cần PDF từng người hoặc in. "
                "Tên bị 123E+10 vì mã khách. Một số dòng trống email. Outlook công ty có. Hướng dẫn từ Select Recipients tới Finish, lỗi hay gặp.",
            ],
            asst=lambda s: (
                f"Word {s['ver']}, {s['n']} dòng từ {s['file']}, phòng {s['dept']}.\n\n"
                f"1. Excel: cột mã khách / STK / điện thoại = Text trước khi gõ hoặc thêm dấu '. Đóng {s['file']} trước khi merge.\n"
                f"2. Word: Mailings → Start Mail Merge → Letters. Select Recipients → Use an Existing List → {s['file']} (sheet đúng).\n"
                f"3. Insert Merge Field. Preview Results, lướt vài bản trong {s['n']}. Rule: Skip nếu email trống (nếu gửi mail).\n"
                f"4. In: Finish & Merge → Print Documents. PDF từng người: Finish → Edit Individual Documents rồi Save As PDF từng file, hoặc merge to PDF nếu bản Word hỗ trợ. "
                f"Email: Merge to E-Mail, cần Outlook — body là bản Word.\n\n"
                f"Lỗi: file Excel đang mở; sheet ẩn; field «Tên» không khớp header; 123E+10 do cột General.\n"
                f"Đừng gửi BCC 200 người một SMTP lạ — dùng Merge to Email hoặc công cụ công ty."
            ),
        ),
        dict(
            prefix="ppt_week_long",
            slots=dict(n=["6", "8", "10"], dept=DEPTS, ver=["Windows", "Mac"], app=["Zoom", "Teams"]),
            users=[
                "Mai họp sếp 15 phút trên {app}, máy {ver}. Cần {n} slide phòng {dept}: tuần này, blocker, tuần sau. "
                "Sếp ghét chữ nhỏ. Chart Excel paste vào bị vỡ font. Notes chỉ mình thấy. Đừng gợi ý 20 slide.",
            ],
            asst=lambda s: (
                f"Họp 15 phút: tối đa {s['n']} slide, phòng {s['dept']}, {s['app']}, máy {s['ver']}.\n\n"
                f"Bố cục gợi ý: (1) 1 câu mục tiêu tuần, (2–3) một số/KPI + 1 chart, (4) blocker + owner + cần sếp gì, "
                f"(5) 3 việc tuần sau, (6) phụ lục nếu còn chỗ — không đọc phụ lục trừ khi được hỏi.\n"
                f"Mỗi bullet ≤ ~8 chữ. Chi tiết để Notes.\n\n"
                f"Chart: họp {s['app']} nên Paste Special → Picture (PNG) để khỏi vỡ font. Cần sửa số live thì Keep Source Formatting và kiểm máy chiếu.\n"
                f"Presenter View: Slide Show → Use Presenter View. {s['app']}: share cửa sổ slideshow, không share desktop (tránh lộ notes). "
                f"Một laptop không HDMI: notes in giấy hoặc cửa sổ nhỏ.\n\n"
                f"15 phút: chốt 1 quyết định, đừng đọc {s['n']} slide chữ.\n"
                f"Theme công ty: Slide Master, đừng sửa logo từng slide."
            ),
        ),
        dict(
            prefix="gs_long",
            slots=dict(place=CITIES, n=NROWS, file=["bán", "kho"], dept=DEPTS),
            users=[
                "Google Sheets, dữ liệu {file} ~{n} dòng. Sheet khác chỉ hiện khu vực {place} và tổng, phòng {dept} không được sửa gốc. "
                "Đã IMPORTRANGE #REF vì quyền. FILTER vs QUERY, cái nào bền. Excel desktop của kế toán không mở được IMPORTRANGE — cần đường lui.",
            ],
            asst=lambda s: (
                f"Hai việc tách nhau: lọc, và lấy data từ file khác.\n\n"
                f"Cùng file, ~{s['n']} dòng, khu {s['place']}:\n"
                f"=QUERY({s['file']}!A:D, \"select A,B,D where C = '{s['place']}'\", 1)\n"
                f"hoặc =FILTER({s['file']}!A:D, {s['file']}!C:C=\"{s['place']}\"). Tổng: SUMIF. "
                f"Lệch hoa thường: where lower(C)=lower('{s['place']}').\n\n"
                f"Khác file: =IMPORTRANGE(\"url\", \"{s['file']}!A:D\") — user có quyền xem file nguồn phải Allow access lần đầu. "
                f"Phòng {s['dept']} chỉ edit file đích; khóa sheet nguồn.\n"
                f"#REF quyền: sai tài khoản Google, hoặc range sai tên sheet (có dấu).\n\n"
                f"Excel desktop không chạy IMPORTRANGE. Đường lui: File → Download .xlsx định kỳ, hoặc kế toán dùng Sheets. "
                f"{s['n']} dòng Sheets chịu được; quá chậm thì thu range, đừng import cả workbook.\n\n"
                f"Header dòng 1 phải ổn định, kẻ QUERY."
            ),
        ),
        dict(
            prefix="xl_protect_long",
            slots=dict(ver=VERS, file=FILES, dept=DEPTS, n=NROWS),
            users=[
                "Gửi {file} cho phòng {dept}, Excel {ver}, ~{n} dòng. Họ chỉ được sửa cột chiết khấu. Công thức giá phải khóa. "
                "Vẫn cần AutoFilter. Sợ họ copy file là gỡ được mật khẩu — có cách nào chắc hơn không. Sheets thì sao.",
            ],
            asst=lambda s: (
                f"Excel {s['ver']}, {s['file']}, ~{s['n']} dòng, phòng {s['dept']}.\n\n"
                f"1. Unlock cột chiết khấu: Format Cells → Protection → bỏ Locked. Các cột công thức để Locked (mặc định).\n"
                f"2. Review → Protect Sheet: mật khẩu vừa; tick Select unlocked cells + Use AutoFilter (+ Sort nếu cần). "
                f"Không tick Edit objects nếu không muốn họ phá chart.\n"
                f"3. Protect Sheet không phải bảo mật: ai cũng gỡ được. Chắc hơn: chia sẻ OneDrive/SharePoint quyền View, hoặc chỉ unlock trên bản Google Sheets Protect range + quyền editor hạn chế. "
                f"Hợp đồng giá: PDF snapshot, Excel chỉ để tính nội bộ.\n\n"
                f"Sheets: Data → Protected sheets and ranges. Excel 2013 đủ Protect Sheet + AutoFilter nếu tick.\n"
                f"Filter sau protect: nếu quên tick AutoFilter thì họ bảo “không lọc được” — không phải hỏng file.\n\n"
                f"Cột chiết khấu đang số hay %? Công thức giá = đơn_giá*(1-chiết_khấu) cần thống nhất 0.1 vs 10%."
            ),
        ),
        dict(
            prefix="wd_track_long",
            slots=dict(ver=["2016", "365"], pages=["8", "16", "24"], dept=DEPTS),
            users=[
                "Hợp đồng Word {ver}, {pages} trang, Track Changes đỏ, comment của phòng {dept} và luật. "
                "Sếp bảo Accept All cho sạch. Mình sợ nuốt điều khoản. Cần bản sạch gửi khách + giữ bản markup. Compare hai file nữa.",
            ],
            asst=lambda s: (
                f"Word {s['ver']}, ~{s['pages']} trang. Đừng Accept All khi chưa đọc.\n\n"
                f"1. Save As “HĐ_markup_{s['dept']}.docx” rồi mới làm bản sạch.\n"
                f"2. Review → Display for Review: All Markup. Next + Accept hoặc Reject từng. Comment: Reply rồi Resolve; xóa hàng loạt chỉ khi đã chốt.\n"
                f"3. Bản gửi khách: Accept những chỗ đã đồng ý, Reject chỗ chưa. Tắt Track Changes. Delete All Comments nếu khách không cần thấy thảo luận nội bộ.\n"
                f"4. So hai phiên bản: Review → Compare → hai file. Kết quả là file thứ ba — không ghi đè markup.\n\n"
                f"{s['pages']} trang: lọc markup theo reviewer (luật vs {s['dept']}) để đọc nhanh.\n"
                f"365/2016 đủ. PDF: xuất bản sạch; đừng gửi PDF còn comment ẩn nếu không muốn lộ.\n\n"
                f"Chỗ tiền/phạt/thời hạn: đọc chậm, không Accept vì “đỏ khó chịu”."
            ),
        ),
        dict(
            prefix="ol_meet_long",
            slots=dict(n=["4", "8", "12"], app=["Teams", "Zoom"], dept=DEPTS, city=CITIES),
            users=[
                "Lịch họp {n} người Outlook, phòng {dept}, {app}, khách ở {city} và một bạn nước ngoài. "
                "Lần trước gửi Appointment nên không có To, người kia không thấy. PPT 25MB. Múi giờ VN. ICS cho người không Outlook. Từng bước.",
            ],
            asst=lambda s: (
                f"Dùng New Meeting, không phải Appointment — Appointment không có người nhận.\n\n"
                f"1. Outlook: New Meeting. To: {s['n']} người, Required vs Optional. Location: {s['city']} / {s['app']}. Time zone GMT+7 (Indochina). "
                f"Khách nước ngoài: hiện hai múi trong Scheduling Assistant; đừng ghi “9h” không múi.\n"
                f"2. {s['app']}: bật link trong meeting options (Teams Add Teams Meeting / Zoom add-in). Kiểm Join trước giờ.\n"
                f"3. PPT 25MB: đừng đính. Để SharePoint/Drive, link trong body. Đính < ~10MB nếu bắt buộc.\n"
                f"4. Người không Outlook: File → Save As ICS, hoặc họ Accept từ mail. Phòng {s['dept']}: ghi agenda 3 dòng trong body, không để trống.\n\n"
                f"Recurring: kiểm exception khi đổi một buổi. Private nếu họp lương.\n"
                f"Nhầm Appointment là lỗi hay gặp — xóa appointment, tạo Meeting mới, đừng forward appointment rồi hy vọng."
            ),
        ),
        dict(
            prefix="xl_dups_long",
            slots=dict(ver=VERS, n=NROWS, ent=ENTS, file=FILES),
            users=[
                "Cột {ent} trên {file}, Excel {ver}, {n} dòng, trùng nhiều. Sếp muốn danh sách unique để dropdown, nhưng không được xóa dòng gốc (còn doanh thu). "
                "365 máy mình có, máy kế toán 2013. Conditional formatting tô trùng nữa.",
            ],
            asst=lambda s: (
                f"Không Remove Duplicates trên sheet gốc {s['file']} — mất doanh thu. ~{s['n']} dòng, khóa {s['ent']}, Excel {s['ver']}.\n\n"
                f"365/Sheets: sheet khác =UNIQUE(A2:A) hoặc =UNIQUE(A:A). Đếm lần =COUNTIF(A:A, E2).\n"
                f"2013/2016 không UNIQUE: copy cột {s['ent']} sang sheet phụ → Data → Remove Duplicates trên bản copy. Máy 2013 nhận sheet phụ, không nhận hàm UNIQUE từ máy bạn.\n\n"
                f"Tô trùng trên gốc: Home → Conditional Formatting → Highlight → Duplicate Values. Hoặc công thức =COUNTIF($A$2:$A$2, A2)>1 trên cả cột (sửa hết dải).\n"
                f"Dropdown: Data Validation → List → trỏ cột unique sheet phụ.\n\n"
                f"Trùng vì space: TRIM trước UNIQUE. Hoa thường: UNIQUE không gộp “SP01”/“sp01” — chuẩn UPPER nếu nghiệp vụ coi là một.\n"
                f"Bạn cần unique để dropdown hay để cộng gộp doanh thu? Hai việc khác nhau (UNIQUE vs Pivot/SUMIF)."
            ),
        ),
        dict(
            prefix="xl_vat_long",
            slots=dict(n=NROWS[:3], vat=["8", "10"], file=FILES, ver=VERS),
            users=[
                "Báo giá {file}, Excel {ver}, {n} dòng. Thành tiền = SL × đơn giá, VAT {vat}%. "
                "Làm tròn đến đồng. File kế toán lệch 1–2 đồng. Khách hỏi tròn trước hay sau thuế. Có dòng chiết khấu %. Giải thích và công thức.",
            ],
            asst=lambda s: (
                f"Excel {s['ver']}, {s['file']}, {s['n']} dòng, VAT {s['vat']}%.\n\n"
                f"Thường khớp hóa đơn VN: tròn thành tiền chưa thuế, rồi tính thuế, rồi cộng.\n"
                f"Chiết khấu % trên đơn giá hoặc trên thành tiền — chốt 1 kiểu:\n"
                f"Thành tiền = ROUND(SL * đơn_giá * (1-chiết_khấu), 0)\n"
                f"VAT = ROUND(thành_tiền * {s['vat']}/100, 0)\n"
                f"Tổng = thành_tiền + VAT\n\n"
                f"Lệch 1 đồng vs phần mềm kế toán: họ có thể ROUND từng dòng rồi cộng, hoặc cộng rồi ROUND. "
                f"Hỏi họ một hóa đơn mẫu 3 dòng. Đừng sửa {s['n']} dòng tay.\n"
                f"Chiết khấu 10% nhập 0.1 hay 10: nếu ô format % thì 10% là 0.1 trong công thức — nhân (1-E2) đúng.\n\n"
                f"Không bịa “luật phải tròn bước X”; bước tròn là thỏa thuận với kế toán/hóa đơn điện tử đang dùng."
            ),
        ),
        dict(
            prefix="refuse_office_long",
            slots=dict(x=["Office 2021", "Windows 11", "Acrobat"], dept=DEPTS, ver=VERS),
            users=[
                "Máy phòng {dept} hết hạn {x}, Excel {ver} báo unlicensed. Sếp bảo tìm key / crack cho nhanh, mai có họp. "
                "Nhờ hướng dẫn cài và đưa luôn serial. Công ty chưa mua.",
            ],
            asst=lambda s: (
                f"Mình không đưa key, serial hay crack {s['x']}. Crack hay kèm malware, vi phạm bản quyền, IT có thể chặn máy.\n\n"
                f"Việc mai họp phòng {s['dept']} vẫn làm được:\n"
                f"- Excel {s['ver']} online / Google Sheets / LibreOffice để sửa số và chiếu PDF.\n"
                f"- Máy công ty: ticket IT xin license (Microsoft 365). Đừng cài tool crack trên domain.\n"
                f"- Máy nhà: mua bản quyền hoặc dùng bản web.\n\n"
                f"Cảnh unlicensed: vẫn mở file xem, một số lệnh bị hạn chế — xuất PDF/Sheets là đường lui họp.\n"
                f"Mình giúp được: chuyển file, công thức, PPT 15 phút, không phải bỏ qua license."
            ),
        ),
    ]


def code_long_kinds():
    return [
        dict(
            prefix="py_merge_long",
            slots=dict(n=NROWS[1:], key=["ma_sp", "sku", "MST"], os=PYOS, file=FILES),
            users=[
                "pandas merge đơn {file} với kho theo {key}, ~{n} dòng, máy {os}. "
                "Một số {key} trùng bên kho, left merge nở hàng. Đơn thiếu hàng cần list. "
                "Mã lệch hoa thường và space. File Excel đang mở thì lần trước PermissionError. Code chạy được, giải thích how=left, đừng xóa kho.",
            ],
            asst=lambda s: (
                f"Máy {s['os']}, đóng {s['file']} trên Excel trước khi đọc. ~{s['n']} dòng, khóa {s['key']}.\n\n"
                f"import pandas as pd\n"
                f"don = pd.read_excel(\"{s['file']}\")\n"
                f"kho = pd.read_excel(\"kho.xlsx\")\n"
                f"for df in (don, kho):\n"
                f"    df[\"{s['key']}\"] = df[\"{s['key']}\"].astype(str).str.strip().str.upper()\n"
                f"kho_u = kho.drop_duplicates(\"{s['key']}\", keep=\"first\")  # hoặc groupby sum số lượng nếu trùng là tồn\n"
                f"out = don.merge(kho_u, on=\"{s['key']}\", how=\"left\", indicator=True)\n"
                f"thieu = out[out[\"_merge\"] == \"left_only\"]\n"
                f"thieu.to_excel(\"don_thieu.xlsx\", index=False)\n\n"
                f"how='left' giữ mọi dòng đơn; kho không khớp → NaN. inner sẽ mất đúng đơn thiếu. "
                f"Kho trùng {s['key']} làm nở hàng — drop_duplicates hoặc cộng tồn trước merge.\n"
                f"Đừng overwrite kho.xlsx. In out['_merge'].value_counts() để soi.\n\n"
                f"Bạn trùng kho là cùng mã nhiều vị trí (cộng SL) hay dữ liệu rác (giữ first)?"
            ),
        ),
        dict(
            prefix="py_excel_lock_long",
            slots=dict(file=FILES, sheet=SHEETS, n=NROWS, os=PYOS),
            users=[
                "Script pandas đọc {file} sheet '{sheet}', ~{n} dòng, {os}. Lúc FileNotFound, lúc PermissionError vì kế toán đang mở Excel. "
                "Cần thông báo tiếng Việt, không traceback Anh cho họ. In head và danh sách cột. Đường dẫn có tiếng Việt.",
            ],
            asst=lambda s: (
                f"import sys\nfrom pathlib import Path\nimport pandas as pd\n\n"
                f"path = Path(\"{s['file']}\")  # {s['os']}: raw string nếu có \\ \n"
                f"if not path.exists():\n"
                f"    sys.exit(f\"Không thấy file: {{path.resolve()}}\")\n"
                f"try:\n"
                f"    df = pd.read_excel(path, sheet_name=\"{s['sheet']}\")\n"
                f"except PermissionError:\n"
                f"    sys.exit(\"Đóng file trên Excel rồi chạy lại. OneDrive đang sync cũng hay khóa.\")\n"
                f"except ValueError as e:\n"
                f"    sys.exit(f\"Không có sheet {s['sheet']}: {{e}}\")\n"
                f"print(df.head())\nprint(df.columns.tolist())\n"
                f"print(len(df), \"dòng\")  # kỳ vọng ~{s['n']}\n\n"
                f"openpyxl đọc xlsx. csv thì encoding utf-8-sig/cp1258, không dùng read_excel.\n"
                f"Sheet index 0 nếu tên có space. {s['os']}+OneDrive: chờ icon mây xong.\n"
                f"Đừng except Exception: pass — nuốt lỗi."
            ),
        ),
        dict(
            prefix="py_groupby_long",
            slots=dict(file=FILES[:4], key=["phong", "NV", "thang"], val=["doanh_thu", "so_luong"], n=NROWS, os=PYOS),
            users=[
                "{file}: groupby {key} tổng {val}, ~{n} dòng, {os}. Ghi sheet 'Tổng' trong file có sẵn, không xóa sheet khác. "
                "Có NaN {key}. Xuất cả CSV utf-8-sig để Excel {os} khỏi lỗi font. Code rõ.",
            ],
            asst=lambda s: (
                f"import pandas as pd\n\n"
                f"df = pd.read_excel(\"{s['file']}\")\n"
                f"df[\"{s['val']}\"] = pd.to_numeric(df[\"{s['val']}\"], errors=\"coerce\")\n"
                f"g = df.groupby(\"{s['key']}\", dropna=False)[\"{s['val']}\"].sum().reset_index()\n"
                f"g.to_csv(\"tong_{s['key']}.csv\", index=False, encoding=\"utf-8-sig\")\n"
                f"with pd.ExcelWriter(\"{s['file']}\", engine=\"openpyxl\", mode=\"a\", if_sheet_exists=\"replace\") as w:\n"
                f"    g.to_excel(w, sheet_name=\"Tổng\", index=False)\n\n"
                f"~{s['n']} dòng, máy {s['os']}: đóng Excel trước writer. mode='a' giữ sheet khác; replace chỉ sheet 'Tổng'.\n"
                f"dropna=False để nhóm NaN không biến mất. Tên cột lệch: in df.columns, đừng đoán.\n"
                f"Không dùng g.to_excel(file) không mode a — ghi đè cả workbook."
            ),
        ),
        dict(
            prefix="sql_left_long",
            slots=dict(db=DBS, key=["ma_kh", "ma_sp"], n=NROWS),
            users=[
                "{db}: đơn không khớp danh mục theo {key}, khoảng {n} dòng. Người khác viết INNER JOIN nên mất đúng dòng cần. "
                "Có space, NULL, hoa thường. Cần câu SELECT, giải thích IS NULL, không UPDATE production.",
            ],
            asst=lambda s: (
                f"-- {s['db']}, ~{s['n']} dòng, tìm {s['key']} có trên đơn không có trên danh mục\n"
                f"SELECT d.*\n"
                f"FROM don d\n"
                f"LEFT JOIN dm k\n"
                f"  ON TRIM(UPPER(d.{s['key']})) = TRIM(UPPER(k.{s['key']}))\n"
                f"WHERE k.{s['key']} IS NULL;\n\n"
                f"LEFT JOIN giữ đơn; INNER mất đơn không khớp — đó là bug báo cáo \"thiếu khách\". "
                f"WHERE k.{s['key']} IS NULL lọc phía phải trống. Không viết <> '' cho NULL: NULL <> '' ra UNKNOWN, dòng bị loại oan.\n"
                f"{s['db']}: TRIM/UPPER sẵn; SQL Server dùng LTRIM(RTRIM(...)). Collation hoa thường tùy DB — UPPER cho chắc.\n"
                f"Chạy SELECT trên replica/staging. Không UPDATE/DELETE từ câu này. Backup trước khi sửa dữ liệu.\n\n"
                f"Hai bên {s['key']} đang VARCHAR hay INT? Số vs chữ cũng làm JOIN trượt."
            ),
        ),
        dict(
            prefix="sql_update_long",
            slots=dict(db=DBS, grp=["PK", "NB", "KM"], n=["50", "200", "800"]),
            users=[
                "UPDATE giá +5% nhóm {grp} trên {db} production, ước {n} dòng. Sợ quên WHERE. "
                "Đồng nghiệp bảo DROP TABLE cho sạch. Cần transaction, SELECT trước, cách rollback. Không đưa câu DROP.",
            ],
            asst=lambda s: (
                f"Không DROP TABLE production. {s['db']}, nhóm {s['grp']}, ~{s['n']} dòng.\n\n"
                f"-- 1) Đếm và soi\n"
                f"SELECT COUNT(*), MIN(gia), MAX(gia) FROM sp WHERE nhom = '{s['grp']}';\n"
                f"SELECT * FROM sp WHERE nhom = '{s['grp']}' LIMIT 20;  -- SQL Server: TOP 20\n\n"
                f"START TRANSACTION;  -- SQL Server: BEGIN TRAN\n"
                f"UPDATE sp SET gia = gia * 1.05 WHERE nhom = '{s['grp']}';\n"
                f"-- SELECT lại COUNT/MIN/MAX; sai thì ROLLBACK; đúng thì COMMIT;\n\n"
                f"Backup trước. Privileges: user app không nên DROP. "
                f"LIMIT không có trên UPDATE mọi {s['db']} — WHERE phải đúng. "
                f"Thử staging clone {s['n']} dòng trước live.\n"
                f"1.05 là hệ số; làm tròn DECIMAL nếu cột tiền.\n\n"
                f"Ai yêu cầu DROP: từ chối, đề nghị DELETE/UPDATE có WHERE trên bản sao."
            ),
        ),
        dict(
            prefix="git_stash_long",
            slots=dict(f=["app.py", "loc.py", "report.py"], os=PYOS),
            users=[
                "Đang sửa dở {f} trên {os}, sếp kêu git pull. Có file untracked. Sợ mất công. "
                "Nhánh feature, không phải main. Conflict thì làm gì. Đừng bảo force.",
            ],
            asst=lambda s: (
                f"# máy {s['os']}, file {s['f']}\n"
                f"git status\n"
                f"git stash push -u -m \"wip {s['f']}\"   # -u gồm untracked\n"
                f"git pull --rebase\n"
                f"git stash pop\n\n"
                f"Conflict sau pop: mở {s['f']}, xóa marker <<<<<< ===== >>>>>>, giữ đúng ý, git add {s['f']}. "
                f"Đang rebase: git rebase --continue (sau add). Không git stash drop trước khi chắc pop ổn.\n"
                f"Không git pull --force, không push --force main/master.\n"
                f"Binary .xlsx conflict: chọn một bản, không merge tay.\n"
                f"{s['os']}: line ending CRLF — đừng reformat cả file khi stash, khó diff.\n\n"
                f"Chưa commit bao giờ: stash -u vẫn giữ; hoặc commit tạm trên nhánh feature rồi pull --rebase."
            ),
        ),
        dict(
            prefix="git_force_long",
            slots=dict(br=["main", "master", "develop"], f=["app.py", "README.md"]),
            users=[
                "git push --force nhánh {br} vì rebase rồi reject. Có {f} vừa sửa. Team 4 người dùng chung {br}. "
                "Muốn lịch sử thẳng. Có được force không, làm gì thay.",
            ],
            asst=lambda s: (
                f"Không force push {s['br']} dùng chung — ghi đè commit đồng nghiệp, mất {s['f']} của họ trên remote.\n\n"
                f"Sau rebase nhánh riêng: git push --force-with-lease origin ten-nhanh-cua-ban "
                f"(lease an hơn --force; vẫn chỉ nhánh riêng).\n"
                f"Nhánh {s['br']}: git pull --rebase origin {s['br']}, giải conflict, test, git push thường.\n"
                f"Lịch sử thẳng không đáng bằng mất code team 4 người.\n\n"
                f"Đã lỡ force và người khác pull lỗi: họ không reset --hard theo bạn trừ khi team thống nhất; "
                f"thường họ merge/rebase lại. Báo ngay trên chat.\n"
                f"Hook reject force {s['br']} là đúng — đừng tắt hook.\n\n"
                f"Bạn đang đứng nhánh nào (git branch --show-current)?"
            ),
        ),
        dict(
            prefix="py_csv_long",
            slots=dict(os=PYOS, src=["sao kê ngân hàng", "phần mềm kế toán", "Excel CSV"], n=NROWS),
            users=[
                "pandas đọc CSV {src} trên {os}, ~{n} dòng. UnicodeDecodeError, rồi mở được thì cột tiền NaN, STK mất 0. "
                "Cần thử encoding, dtype STK=str, decimal phẩy. Đừng đoán xóa cột.",
            ],
            asst=lambda s: (
                f"import pandas as pd\n\n"
                f"encodings = [\"utf-8-sig\", \"utf-8\", \"cp1258\", \"cp1252\"]\n"
                f"df = None\n"
                f"for e in encodings:\n"
                f"    try:\n"
                f"        df = pd.read_csv(\n"
                f"            \"data.csv\",\n"
                f"            encoding=e,\n"
                f"            dtype={{\"STK\": str}},\n"
                f"            decimal=\",\",\n"
                f"            thousands=\".\",\n"
                f"        )\n"
                f"        break\n"
                f"    except UnicodeDecodeError:\n"
                f"        continue\n"
                f"if df is None:\n"
                f"    raise SystemExit(\"Không đọc được encoding. Mở Notepad++ xem Encoding.\")\n"
                f"print(e, df.head(), df.dtypes)\n\n"
                f"~{s['n']} dòng, {s['os']}, nguồn {s['src']}: Windows hay cp1258; Excel CSV thường utf-8-sig. "
                f"STK dtype str trước khi Excel biến thành số. decimal=',' nếu file VN dùng 1.234,56 — nếu file dùng 1234.56 thì bỏ thousands/decimal.\n"
                f"In 3 dòng thô open(...,'rb') nếu vẫn NaN.\n"
                f"Đừng to_excel rồi đọc lại CSV làm mất 0."
            ),
        ),
        dict(
            prefix="py_datetime_long",
            slots=dict(os=PYOS, n=NROWS, file=FILES),
            users=[
                "Cột ngày {file} lẫn '13/09/2026', '2026-09-13' và serial Excel, pandas trên {os}, ~{n} dòng. "
                "to_datetime ra NaT, lọc tháng 9 sót. dayfirst, origin serial. Code + giải thích.",
            ],
            asst=lambda s: (
                f"import pandas as pd\n\n"
                f"df = pd.read_excel(\"{s['file']}\")  # {s['os']}, ~{s['n']} dòng\n"
                f"s = df[\"ngay\"]\n"
                f"num = pd.to_numeric(s, errors=\"coerce\")\n"
                f"from_serial = pd.to_datetime(num, unit=\"d\", origin=\"1899-12-30\", errors=\"coerce\")\n"
                f"from_text = pd.to_datetime(s.astype(str), dayfirst=True, errors=\"coerce\")\n"
                f"df[\"ngay\"] = from_serial.fillna(from_text)\n"
                f"thang9 = df[(df[\"ngay\"].dt.month == 9) & (df[\"ngay\"].dt.year == 2026)]\n\n"
                f"dayfirst=True cho dd/mm. Serial Excel ≠ Unix timestamp. "
                f"errors='coerce' cho NaT; đếm df['ngay'].isna().sum() — đừng im xóa.\n"
                f"Timezone VN không cần nếu chỉ ngày. Cột mixed làm 1 to_datetime thường fail im.\n\n"
                f"In s.head(20) nếu vẫn NaT: có thể '13/9/26' hoặc 'tháng 9'."
            ),
        ),
        dict(
            prefix="sql_having_long",
            slots=dict(db=DBS, x=["50", "80", "100"], key=["nv", "ma_kh"]),
            users=[
                "{db}: {key} có SUM(doanh_thu) > {x} triệu tháng 9/2026. Mình để SUM trong WHERE nên lỗi. "
                "Giải thích WHERE vs HAVING, đơn vị triệu, Top sau gộp. Câu chạy được.",
            ],
            asst=lambda s: (
                f"-- {s['db']}: lọc dòng trước (tháng), gộp, rồi lọc tổng\n"
                f"SELECT {s['key']}, SUM(doanh_thu) AS tong\n"
                f"FROM don\n"
                f"WHERE ngay >= '2026-09-01' AND ngay < '2026-10-01'\n"
                f"GROUP BY {s['key']}\n"
                f"HAVING SUM(doanh_thu) > {s['x']} * 1000000\n"
                f"ORDER BY tong DESC;\n\n"
                f"WHERE không dùng SUM. HAVING sau GROUP BY. "
                f"{s['x']} triệu → nhân 1e6 nếu cột là đồng; nếu cột đã triệu thì so sánh > {s['x']}.\n"
                f"SELECT * đi với GROUP BY sẽ lỗi trên {s['db']} nghiêm (Postgres). Chỉ chọn {s['key']} và aggregate.\n"
                f"Top 5: LIMIT 5; SQL Server: TOP 5 hoặc OFFSET FETCH.\n"
                f"Index (ngay, {s['key']}) nếu bảng lớn. Chạy EXPLAIN trên staging, không lock production."
            ),
        ),
        dict(
            prefix="py_glob_long",
            slots=dict(n=["8", "12", "24"], sheet=SHEETS[:3], os=PYOS),
            users=[
                "Thư mục {n} file xlsx trên {os}, gộp sheet '{sheet}'. Có file Excel đang mở (~$) và file tạm. "
                "pathlib, concat, cột lệch file. Xuất gop.xlsx. Đừng đọc cả workbook.",
            ],
            asst=lambda s: (
                f"from pathlib import Path\nimport pandas as pd\n\n"
                f"files = [\n"
                f"    p for p in Path(\".\").glob(\"*.xlsx\")\n"
                f"    if not p.name.startswith(\"~$\") and not p.name.startswith(\".\")\n"
                f"]\n"
                f"dfs = []\n"
                f"for p in files:  # kỳ vọng ~{s['n']} file, {s['os']}\n"
                f"    try:\n"
                f"        d = pd.read_excel(p, sheet_name=\"{s['sheet']}\")\n"
                f"    except ValueError:\n"
                f"        print(\"bỏ, không sheet\", p)\n"
                f"        continue\n"
                f"    d[\"_file\"] = p.name\n"
                f"    dfs.append(d)\n"
                f"out = pd.concat(dfs, ignore_index=True, sort=False)\n"
                f"out.to_excel(\"gop.xlsx\", index=False)\n\n"
                f"Cột lệch: sort=False giữ union cột, NaN chỗ thiếu — soi out.isna().mean(). "
                f"Đừng sheet_name=None nếu chỉ cần '{s['sheet']}' (chậm, dict).\n"
                f"File đang mở: bỏ ~$ ; PermissionError: đóng Excel.\n"
                f"{s['os']}: Path('.') là cwd VS Code, không hẳn thư mục data."
            ),
        ),
        dict(
            prefix="sql_param_long",
            slots=dict(db=DBS[:3], os=PYOS, key=["ma_sp", "MST", "ma_kh"]),
            users=[
                "User gõ {key} từ ô Excel, mình f-string vào SQL {db} trên {os}. Sợ injection và mã có dấu nháy. "
                "Viết execute an toàn, ví dụ Python. Đừng bảo “tin user nội bộ là đủ”.",
            ],
            asst=lambda s: (
                f"# {s['db']} trên {s['os']}: placeholder, không cộng chuỗi\n"
                f"sql = \"SELECT * FROM sp WHERE ma = %s\"  # Postgres: %s với psycopg; SQL Server: ? pyodbc\n"
                f"cur.execute(sql, (ma,))\n"
                f"rows = cur.fetchall()\n\n"
                f"Đừng: cur.execute(f\"SELECT * FROM sp WHERE ma = '{{ma}}'\") — nháy trong mã, hoặc `; DROP`.\n"
                f"Excel đưa ma: str.strip(); reject ký tự lạ nếu mã chỉ [A-Z0-9]. "
                f"IN list: placeholder lặp, vẫn tuple params, không join thô.\n"
                f"User nội bộ vẫn paste nhầm. Log query có param, đừng log password.\n"
                f"{s['db']}: đọc doc driver (pymysql %s, psycopg %s, sqlite ?).\n\n"
                f"Thiếu: bạn dùng thư viện kết nối nào? Mình khớp placeholder đúng driver."
            ),
        ),
        dict(
            prefix="py_venv_long",
            slots=dict(os=PYOS, pkg=["pandas openpyxl", "pandas sqlalchemy"]),
            users=[
                "pip install {pkg} đụng Python khác, {os}. VS Code chạy kernel không phải máy mình. "
                "Không quyền admin. Cần venv, requirements, chọn interpreter. Giải thích vì sao pandas-not-found.",
            ],
            asst=lambda s: (
                f"# {s['os']}, không cần admin\n"
                f"python -m venv .venv\n"
                f"{'.venv\\Scripts\\activate' if s['os']=='Windows' else 'source .venv/bin/activate'}\n"
                f"python -m pip install -U pip\n"
                f"pip install {s['pkg']}\n"
                f"pip freeze > requirements.txt\n\n"
                f"VS Code: Python: Select Interpreter → .venv. Terminal hiện (.venv). "
                f"pandas-not-found = kernel/terminal đang Python khác (App Store / Anaconda / WindowsApps).\n"
                f"which python / where python để soi. Không pip install --user song song hai chỗ.\n"
                f"Commit requirements.txt, không commit .venv. "
                f"{s['os']} company policy: nếu cấm pip, nhờ IT wheel nội bộ — đừng tắt SSL verify bừa.\n\n"
                f"Chạy lại: python -c \"import pandas; print(pandas.__file__)\""
            ),
        ),
        dict(
            prefix="py_argparse_long",
            slots=dict(ot=["8", "10"], os=PYOS, file=FILES[:3]),
            users=[
                "CLI: python loc.py --file {file} --ot {ot} trên {os}. Thiếu file phải thoát tiếng Việt. "
                "Đường dẫn có space. Default OT. Không hard-code. argparse, pathlib.",
            ],
            asst=lambda s: (
                f"import argparse, sys\nfrom pathlib import Path\nimport pandas as pd\n\n"
                f"p = argparse.ArgumentParser(description=\"Lọc OT\")\n"
                f"p.add_argument(\"--file\", required=True)\n"
                f"p.add_argument(\"--ot\", type=float, default={s['ot']})\n"
                f"a = p.parse_args()\n"
                f"path = Path(a.file)\n"
                f"if not path.exists():\n"
                f"    sys.exit(f\"Không thấy file: {{path}}\")\n"
                f"df = pd.read_excel(path)\n"
                f"# ... lọc OT > {s['ot']}\n\n"
                f"Máy {s['os']}: python loc.py --file \"C:\\Users\\...\\{s['file']}\"  — quote vì space. "
                f"type=float: --ot 8,5 có thể lỗi locale; dùng 8.5.\n"
                f"Không parse tay sys.argv. required=True cho --file.\n"
                f"Exit code khác 0 khi thiếu file, để scheduler biết fail."
            ),
        ),
        dict(
            prefix="git_ignore_long",
            slots=dict(x=["*.xlsx", "*.csv", ".env"], os=PYOS),
            users=[
                "Repo đang track {x} trên {os}. Muốn .gitignore, gỡ khỏi Git, giữ file máy. "
                "Đồng nghiệp pull sẽ mất file trên GitHub — cần biết nói gì. Có file mẫu.xlsx cần giữ.",
            ],
            asst=lambda s: (
                f"# .gitignore\n"
                f"{s['x']}\n"
                f"!mau.xlsx\n\n"
                f"git rm --cached {s['x']}\n"
                f"git add .gitignore\n"
                f"git commit -m \"Stop tracking {s['x']}\"\n\n"
                f"File trên đĩa {s['os']} còn. git rm --cached không xóa local. "
                f"Đồng nghiệp: sau pull, file biến khỏi working tree nếu họ chưa có bản local — copy backup/mẫu trước, nói trên chat.\n"
                f"Đã nằm history: --cached không xóa lịch sử; secret trong .env cũ cần rotate, không chỉ gitignore.\n"
                f"Đừng git rm không --cached — xóa file máy.\n"
                f"Excel lương: {s['x']} ignore là đúng; không force-add."
            ),
        ),
        dict(
            prefix="py_ask_long",
            slots=dict(x=["lương", "kho", "công nợ"], os=PYOS),
            users=[
                "pandas đọc file {x} lỗi trên {os}. Không gửi được file. Sửa giúp. Traceback để trong đầu, đại khái PermissionError hoặc dtype.",
            ],
            asst=lambda s: (
                f"Thiếu dữ liệu để sửa file {s['x']} trên {s['os']}. Đừng đoán xóa cột hay dropna cả khung.\n\n"
                f"Nhờ gửi: đuôi .xlsx/.csv; tên sheet; máy {s['os']}; traceback đầy đủ; df.columns nếu đọc được; "
                f"file có đang mở Excel không; encoding nếu CSV.\n"
                f"CSV: thử utf-8-sig rồi cp1258. Excel: pd.read_excel(path, sheet_name=0) khi không nhớ tên sheet.\n"
                f"PermissionError: đóng Excel / chờ OneDrive. FileNotFound: Path.cwd().\n"
                f"Không DROP bảng, không ghi đè file gốc — Save As bản làm việc.\n\n"
                f"Dán 5 dòng mẫu (che PII) + traceback, mình viết đúng một lệnh đọc."
            ),
        ),
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-dich", type=int, default=10000)
    ap.add_argument("--n-off", type=int, default=5000)
    ap.add_argument("--n-code", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    dich = fill(dich_kinds(), "dich_thuat", "dich-synth", a.n_dich, a.seed, min_u=80, min_a=280)
    off = fill(office_long_kinds(), "van_phong", "office-long", a.n_off, a.seed + 3, min_u=140, min_a=520)
    code = fill(code_long_kinds(), "lap_trinh", "code-long", a.n_code, a.seed + 5, min_u=120, min_a=520)
    write_jsonl(OUT_DICH, dich)
    write_jsonl(OUT_OFF, off)
    write_jsonl(OUT_CODE, code)
    print(f"dich   {len(dich)} → {OUT_DICH}")
    print(f"office {len(off)} → {OUT_OFF}")
    print(f"code   {len(code)} → {OUT_CODE}")
    print("mẫu dịch :", dich[0]["messages"][0]["content"][:90].replace("\n", " "))
    print("mẫu office:", off[0]["messages"][0]["content"][:90].replace("\n", " "))
    print("mẫu code  :", code[0]["messages"][0]["content"][:90].replace("\n", " "))


if __name__ == "__main__":
    main()
