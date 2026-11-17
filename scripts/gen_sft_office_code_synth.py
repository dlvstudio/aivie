"""Sinh SFT văn phòng + lập trình: khuôn chuyên gia, điền slot, không dịch máy.

    python3 scripts/gen_sft_office_code_synth.py
    → data/sft/hf/office_synth.jsonl  (10000)
    → data/sft/hf/code_synth.jsonl    (10000)
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import random

N = 10000
OUT_OFF = "data/sft/hf/office_synth.jsonl"
OUT_CODE = "data/sft/hf/code_synth.jsonl"

FILES = [
    "luong.xlsx", "ban_hang.xlsx", "kho.xlsx", "cong_no.xlsx",
    "khach_hang.xlsx", "cham_cong.xlsx", "bao_gia.xlsx", "mst_ncc.xlsx",
]
SHEETS = ["Lương", "Bán", "Kho", "Tóm tắt", "Hệ số", "Chi tiết"]
VERS = ["2016", "365", "2013", "Google Sheets"]
ENTS = ["mã NV", "mã SP", "MST", "mã đơn", "số HĐ"]
VALS = ["hệ số", "đơn giá", "doanh thu", "số lượng", "chiết khấu"]
DEPTS = ["Kinh doanh", "Kế toán", "Kho", "CSKH", "Nhân sự"]
PLACES = ["Quận 3 TP.HCM", "Hà Đông", "Ninh Kiều", "Thanh Xuân", "Hải Châu"]
NROWS = ["80", "200", "400", "1500", "5000"]
BANKS = ["Vietcombank", "Techcombank", "MB", "VietinBank"]
DBS = ["MySQL", "SQL Server", "Postgres", "SQLite"]
PYOS = ["Windows", "Mac"]
LANG = ["python", "sql", "git"]


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


def col_offset(start, n):
    """A=1 → letter; n=3 → cột thứ 3 trong range A:C."""
    return n


def vlookup_formula(s):
    k = s["key_col"]
    v = s["val_col"]
    # giả sử key_col A, val_col C → index 3
    idx = (ord(v) - ord(k)) + 1
    if idx < 2:
        idx = 3
        k, v = "A", "C"
    rng = f"{s['sheet']}!{k}:{v}"
    if s["ver"] == "365":
        return (
            f"Excel 365: =XLOOKUP(A2, {s['sheet']}!{k}:{k}, {s['sheet']}!{v}:{v}, \"\") "
            f"nếu máy chưa có XLOOKUP thì =IFERROR(VLOOKUP(A2, {rng}, {idx}, FALSE), \"\")."
        )
    if s["ver"] == "Google Sheets":
        return (
            f"Sheets: =IFERROR(VLOOKUP(A2, {rng}, {idx}, FALSE), \"\") "
            f"— FALSE là khớp exact. TRIM mã nếu #N/A."
        )
    return (
        f"Excel {s['ver']}: ô đích (cùng hàng mã A2) "
        f"=IFERROR(VLOOKUP(A2, {rng}, {idx}, FALSE), \"\"). "
        f"FALSE = khớp đúng. Kéo xuống. Cột {v} phải nằm bên phải {k}; "
        f"nếu tên nằm trái mã thì dùng INDEX/MATCH."
    )


def office_kinds():
    return [
        dict(
            prefix="xl_vlookup",
            slots=dict(file=FILES, sheet=SHEETS[:4], ver=VERS, ent=ENTS, val=VALS, key_col=list("AB"), val_col=list("CDE")),
            users=[
                "File {file} sheet {sheet}: cột {key_col} {ent}, cột {val_col} {val}. Cần kéo {val} sang sheet làm việc theo {ent}. Excel {ver}.",
                "Trong {file}, mình có {ent} ở {sheet}!{key_col}, {val} ở cột {val_col}. VLOOKUP giúp với, bản {ver}.",
                "Sếp bảo lookup {val} theo {ent} từ {sheet} trong {file}. Excel {ver}, chưa rành XLOOKUP.",
            ],
            asst=lambda s: vlookup_formula(s) + " Nếu #N/A: kiểm tra số vs chữ, TRIM khoảng trắng. Bạn đang lookup từ cột nào bên sheet đích?",
            follow=(
                "Bị #N/A hàng loạt, mã có khoảng trắng.",
                "Làm cột phụ =TRIM(A2) rồi VLOOKUP cột đó, hoặc =VLOOKUP(TRIM(A2), ...). Đừng bọc IFERROR hết trước khi soi 5 dòng mẫu.",
            ),
        ),
        dict(
            prefix="xl_indexmatch",
            slots=dict(file=FILES, ver=["2016", "2013", "365"], n=NROWS[:3], ent=ENTS),
            users=[
                "VLOOKUP không lấy cột trái. {file}: cột B {ent}, cột A tên. INDEX MATCH, Excel {ver}, ~{n} dòng.",
                "Cần tên theo {ent} nhưng tên nằm trái mã trong {file}. Excel {ver}.",
            ],
            asst=lambda s: (
                f"Excel {s['ver']}, giả sử mã cần tìm ở D2: "
                f"=IFERROR(INDEX($A$2:$A${int(s['n'])+1}, MATCH(D2, $B$2:$B${int(s['n'])+1}, 0)), \"\"). "
                "INDEX vùng trả về, MATCH vùng khóa, 0 = exact. Kéo xuống. "
                + ("365 có thể XLOOKUP(D2, B:B, A:A, \"\")." if s["ver"] == "365" else "Bản này không có XLOOKUP.")
            ),
            follow=("Thừa space ở mã.", "MATCH(TRIM(D2), ...) hoặc TRIM cả cột khóa bằng cột phụ."),
        ),
        dict(
            prefix="xl_sumifs",
            slots=dict(file=FILES, dept=DEPTS, ver=VERS, val=VALS[:3], col=list("BCD")),
            users=[
                "Bảng {file}: cột {col} phòng, cột doanh thu. Tổng phòng {dept}. SUMIF hay Pivot, {ver}?",
                "Cần tổng {val} của {dept} trong {file}. Excel {ver}.",
            ],
            asst=lambda s: (
                f"Một điều kiện: =SUMIF({s['col']}:{s['col']}, \"{s['dept']}\", D:D) "
                f"(đổi D:D thành cột {s['val']}). Nhiều điều kiện (phòng+tháng): SUMIFS. "
                f"{'Sheets dùng SUMIF tương tự.' if s['ver']=='Google Sheets' else f'Excel {s['ver']} có SUMIFS.'} "
                "Tên phòng lệch hoa thường → chuẩn hóa TRIM/UPPER. Pivot nếu sếp muốn cắt nhiều chiều."
            ),
            follow=("Thêm điều kiện tháng này.", "=SUMIFS(D:D, B:B, \"{dept}\", A:A, \">=\"&DATE(2026,9,1), A:A, \"<\"&DATE(2026,10,1)) — chỉnh năm/tháng."),
        ),
        dict(
            prefix="xl_countif",
            slots=dict(ver=VERS, st=["xong", "hủy", "đang giao", "mới"], n=NROWS),
            users=[
                "Cột trạng thái đơn, đếm '{st}', ~{n} dòng, Excel {ver}.",
                "Dashboard: số đơn {st}. COUNTIF, bản {ver}.",
            ],
            asst=lambda s: (
                f"=COUNTIF(C:C, \"{s['st']}\"). Ô ' {s['st']} ' thừa cách thì =COUNTIF(C:C, \"*{s['st']}*\") "
                f"hoặc TRIM cột phụ. Excel {s['ver']}: COUNTIFS nếu thêm tuần. Pivot nếu nhiều trạng thái."
            ),
        ),
        dict(
            prefix="xl_iferror",
            slots=dict(ver=VERS, err=["#N/A", "#REF!", "#VALUE!"], ent=ENTS),
            users=[
                "Công thức lookup {ent} ra {err} hàng loạt. Excel {ver}, sửa sao khỏi sửa tay?",
                "File bị {err} khi VLOOKUP {ent}. {ver}.",
            ],
            asst=lambda s: {
                "#N/A": "Khóa không khớp (số/chữ, space). TRIM/VALUE trước. IFERROR chỉ sau khi soi mẫu.",
                "#REF!": "Range bị xóa cột. Sửa lại A:C, đừng dùng cột đã delete. Kiểm tra sheet tên.",
                "#VALUE!": "SUMIF vùng tổng có chữ. Dùng SUMIFS/SUMPRODUCT hoặc lọc ISNUMBER.",
            }[s["err"]] + f" Bản {s['ver']}: 365 có IFNA cho riêng #N/A.",
            follow=("CSV mở lệch cột.", "Data → From Text, delimiter, cột STK dạng Text. Không double-click CSV."),
        ),
        dict(
            prefix="xl_trim_name",
            slots=dict(ver=VERS, n=NROWS[:4]),
            users=[
                "Cột họ tên lộn xộn khoảng trắng, ~{n} dòng, Excel {ver}. Chuẩn 'Nguyễn Văn An'.",
                "Làm sạch tên trong Excel {ver}, {n} dòng.",
            ],
            asst=lambda s: (
                f"=PROPER(TRIM(A2)) rồi Paste Values. {s['n']} dòng kéo xuống. "
                "PROPER viết hoa chữ cái đầu — tên đệm 'thị' có thể thành Thị (chấp nhận được). "
                "Sheets: =PROPER(TRIM(A2)) giống."
            ),
        ),
        dict(
            prefix="xl_unique",
            slots=dict(ver=["365", "Google Sheets", "2016", "2013"], n=NROWS, ent=ENTS),
            users=[
                "Cột {ent} {n} dòng, lấy danh sách không trùng, {ver}. Đừng xóa file gốc.",
                "Lọc trùng {ent} trên {ver}, {n} dòng.",
            ],
            asst=lambda s: (
                "365/Sheets: =UNIQUE(A2:A) sang cột khác, gốc nguyên. "
                "2016/2013: copy cột → Data → Remove Duplicates trên bản copy; "
                "hoặc tô Conditional Formatting → Duplicate Values."
                if s["ver"] in ("2016", "2013")
                else f"{s['ver']}: =UNIQUE(A2:A) hoặc =UNIQUE(A:A). Gốc không bị xóa. Đếm lần: =COUNTIF(A:A, E2)."
            ),
        ),
        dict(
            prefix="xl_csv_bank",
            slots=dict(bank=BANKS, ver=VERS[:3]),
            users=[
                "Sao kê {bank} CSV, Excel {ver} biến STK mất số 0 / tiền thành ngày. Mở đúng?",
                "File CSV {bank} Windows, {ver}, encoding và cột số.",
            ],
            asst=lambda s: (
                f"Đừng double-click. Excel {s['ver']}: Data → From Text/CSV, encoding UTF-8 hoặc 1258, "
                "cột STK = Text, dấu thập phân. Sheets: File → Import → upload, type Text. "
                f"Sao kê {s['bank']} hay dùng phẩy/chấm phẩy — chọn delimiter khớp."
            ),
            follow=("Mất font tiếng Việt.", "Thử UTF-8 có BOM. Nếu vẫn lỗi: Google Sheets import rồi Tải về .xlsx."),
        ),
        dict(
            prefix="xl_round_vnd",
            slots=dict(mode=["đến đồng", "đến hàng nghìn"], vat=["8", "10"]),
            users=[
                "Thành tiền = SL × đơn giá, làm tròn {mode}, VAT {vat}%. Tròn trước hay sau thuế?",
                "Excel tiền VND, ROUND {mode}, VAT {vat}%.",
            ],
            asst=lambda s: (
                "Thường: thành tiền chưa thuế = ROUND(SL*đơn_giá, 0 hoặc -3), "
                f"VAT = ROUND(thành_tiền*{s['vat']}/100, 0), cộng lại. "
                f"{'ROUND(x,0)' if s['mode']=='đến đồng' else 'ROUND(x,-3)'} "
                "tròn trước thuế rồi tính VAT để khớp hóa đơn; nếu lệch 1 đồng so với phần mềm kế toán thì hỏi họ họ tròn bước nào."
            ),
        ),
        dict(
            prefix="xl_condfmt",
            slots=dict(n=NROWS, ver=VERS),
            users=[
                "Tô đỏ dòng quá hạn, cột D hạn, ~{n} dòng, Excel {ver}.",
                "Conditional formatting công nợ quá hạn, {ver}, {n} dòng.",
            ],
            asst=lambda s: (
                f"Bôi A2:Z{s['n']} → Conditional Formatting → công thức: =$D2<TODAY() "
                f"(dòng 2 là dòng đầu vùng). Đừng dùng =D:D. Excel {s['ver']}/Sheets gần giống. "
                "Thêm rule vàng: =AND($D2>=TODAY(), $D2<=TODAY()+3)."
            ),
        ),
        dict(
            prefix="xl_dropdown",
            slots=dict(ver=VERS, lists=["mới/đang/xong", "có/không", "A/B/C"]),
            users=[
                "Cột trạng thái chỉ cho chọn {lists}. Data Validation, Excel {ver}.",
                "Dropdown {lists} trên {ver}, list sheet ẩn.",
            ],
            asst=lambda s: (
                f"Sheet ẩn 'List' một cột các giá trị {s['lists']}. "
                f"Data → Validation → List → =List!$A$1:$A$3. Excel {s['ver']}: bỏ 'Ignore blank' nếu bắt buộc chọn. "
                "Paste đè vẫn lọt — có thể cảnh báo bằng COUNTIF không khớp list."
            ),
        ),
        dict(
            prefix="xl_pivot",
            slots=dict(n=NROWS, dim=["tháng+NV", "phòng+SP", "khu vực+tháng"]),
            users=[
                "Pivot doanh thu theo {dim}, file ~{n} dòng. Chưa từng pivot. Kéo field nào?",
                "Sếp bảo pivot {dim}, Excel, {n} dòng.",
            ],
            asst=lambda s: (
                f"Insert → PivotTable, chọn bảng (nên Table). Rows: các chiều {s['dim']}, Values: Sum doanh thu. "
                "Excel 365: kéo % of Column nếu cần. Refresh khi thêm dòng (Table tự nới). "
                f"~{s['n']} dòng máy văn phòng chịu được."
            ),
            follow=("Thêm dòng mới không vào pivot.", "Bôi Table hoặc đổi source range, rồi Pivot → Refresh."),
        ),
        dict(
            prefix="xl_date",
            slots=dict(ver=VERS, n=NROWS[:3]),
            users=[
                "Cột ngày CSV ra 45123 hoặc text '13/09/2026', lọc tháng, Excel {ver}, {n} dòng.",
                "Serial date / text date lẫn, {ver}.",
            ],
            asst=lambda s: (
                f"Serial: format Date. Text d/m/y: =DATE(RIGHT(A2,4), MID(A2,4,2), LEFT(A2,2)) nếu dạng dd/mm/yyyy. "
                f"Lọc tháng 9: =MONTH(B2)=9. Excel {s['ver']} máy US: Text to Columns DMY. {s['n']} dòng kéo công thức."
            ),
        ),
        dict(
            prefix="xl_protect",
            slots=dict(ver=VERS[:3]),
            users=[
                "Gửi bảng giá, chỉ cho sửa cột chiết khấu, khóa công thức. Excel {ver}.",
                "Protect sheet {ver}, vẫn lọc được không?",
            ],
            asst=lambda s: (
                f"Unlock cột chiết khấu (Format cells → Protection → bỏ Locked), Review → Protect Sheet, "
                f"tick Select unlocked + AutoFilter. Excel {s['ver']}: mật khẩu vừa phải — không phải bảo mật thật, "
                "ai cũng gỡ được. Sheets: Protect range."
            ),
        ),
        dict(
            prefix="xl_print",
            slots=dict(cols=["8", "12", "14"], ver=VERS[:3]),
            users=[
                "In {cols} cột A4 ngang, header lặp, đừng cắt cột giá. Excel {ver}.",
                "Print Title vs Freeze pane, {cols} cột, {ver}.",
            ],
            asst=lambda s: (
                f"Page Layout → Landscape, A4, Fit 1 page wide. Print Titles: $1:$1. Freeze chỉ khi cuộn màn hình. "
                f"{s['cols']} cột: thu margin, font 9–10. Excel {s['ver']}: Print Preview kiểm cột giá."
            ),
        ),
        dict(
            prefix="xl_textjoin",
            slots=dict(ver=["365", "Google Sheets", "2016", "2013"]),
            users=[
                "Ghép họ + đệm + tên, bỏ ô trống. Excel {ver}.",
                "TEXTJOIN hay CONCAT, {ver}.",
            ],
            asst=lambda s: (
                "365/Sheets: =TEXTJOIN(\" \", TRUE, A2, B2, C2) — TRUE bỏ trống. "
                "2016/2013: =TRIM(A2&\" \"&B2&\" \"&C2)."
                if s["ver"] in ("2016", "2013")
                else f"{s['ver']}: =TEXTJOIN(\" \", TRUE, A2:C2). TRIM nếu còn 2 space."
            ),
        ),
        dict(
            prefix="xl_split",
            slots=dict(n=NROWS[:4], ver=VERS),
            users=[
                "Cột 'Nguyễn Văn An - 0901234567' tách tên/SĐT, {n} dòng, {ver}.",
                "Text to Columns hay công thức, {ver}, {n} dòng.",
            ],
            asst=lambda s: (
                f"Nhanh: Data → Text to Columns, delimiter '-'. Công thức (giữ gốc): "
                f"tên =TRIM(LEFT(A2, FIND(\"-\", A2&\"-\")-1)), SĐT =TRIM(MID(A2, FIND(\"-\", A2&\"-\")+1, 20)). "
                f"{s['ver']}, {s['n']} dòng. Dòng không có SĐT: IFERROR."
            ),
        ),
        dict(
            prefix="xl_powerquery",
            slots=dict(bank=BANKS, ver=["365", "2016"]),
            users=[
                "Mỗi tháng CSV {bank} thả thư mục, gộp Excel {ver}. Power Query?",
                "Refresh sao kê {bank}, {ver}.",
            ],
            asst=lambda s: (
                f"Excel {s['ver']}: Data → Get Data → From Folder, combine CSV, lọc .csv. "
                f"Close & Load. Tháng sau: Refresh. Cột lệch {s['bank']}: chỉnh transform, đừng load thô. "
                "2013: cài Power Query add-in hoặc gộp tay."
            ),
        ),
        dict(
            prefix="gs_query",
            slots=dict(place=PLACES, n=NROWS),
            users=[
                "Google Sheets {n} dòng bán, sheet khác chỉ {place} + tổng. QUERY?",
                "FILTER hay QUERY khu vực {place}, {n} dòng.",
            ],
            asst=lambda s: (
                f"=QUERY(Bán!A:D, \"select A,B,D where C = '{s['place']}'\", 1) "
                f"hoặc =FILTER(Bán!A:D, Bán!C:C=\"{s['place']}\"). Tổng: SUMIF. "
                f"{s['n']} dòng Sheets chịu được. Lệch hoa thường: where lower(C) = lower('{s['place']}')."
            ),
        ),
        dict(
            prefix="gs_importrange",
            slots=dict(place=PLACES[:3]),
            users=[
                "Kéo bảng giá Sheets kế toán sang file sale, IMPORTRANGE #REF quyền. Sale không sửa gốc.",
                "IMPORTRANGE giữa 2 file, team {place}.",
            ],
            asst=lambda s: (
                "=IMPORTRANGE(\"url_file_ke_toan\", \"Giá!A:D\") — lần đầu Allow access bằng tài khoản có quyền xem. "
                f"Sale chỉ xem file đích. Chậm: thu range, không import cả workbook. Excel desktop không chạy IMPORTRANGE. ({s['place']})"
            ),
        ),
        dict(
            prefix="ppt_week",
            slots=dict(n=["6", "8", "10"], team=["sale 6 người", "kế toán 4", "kho 5"]),
            users=[
                "Mai họp sếp, {n} slide: tuần này, blocker, tuần sau. PowerPoint, {team}. Đỡ chữ.",
                "Bố cục {n} slide team {team}.",
            ],
            asst=lambda s: (
                f"{s['n']} slide gợi ý: (1) mục tiêu 1 câu, (2–3) số/KPI 1 chart, (4) blocker + owner, "
                f"(5) tuần sau 3 bullet, (6) cần sếp quyết. Team {s['team']}: mỗi bullet ≤ 8 chữ. "
                "Chi tiết để notes. Không 18 dòng/slide."
            ),
            follow=("Họp 15 phút.", "Cắt còn 5 slide: 1 số, 1 blocker, 1 quyết định. Phần còn lại appendix."),
        ),
        dict(
            prefix="ppt_chart",
            slots=dict(app=["Zoom", "Teams"], ver=["Windows", "Mac"]),
            users=[
                "Paste chart Excel sang PPT, chữ bé, họp {app}, máy {ver}. Paste kiểu nào?",
                "Chart vỡ màu trên {app}, {ver}.",
            ],
            asst=lambda s: (
                f"Paste Special → Picture (PNG) nếu không cần sửa số lúc chiếu {s['app']}. "
                "Cần cập nhật tuần: Keep Source Formatting + link (dễ vỡ khi đổi máy). "
                f"{s['ver']}: kiểm font công ty. Zoom/Teams share cửa sổ PPT, không share vùng nhỏ."
            ),
        ),
        dict(
            prefix="ppt_master",
            slots=dict(n=["12", "20", "40"]),
            users=[
                "Logo lệch {n} slide. Slide Master, đừng phá nội dung đã gõ.",
                "Theme .potx, {n} slide lệch font.",
            ],
            asst=lambda s: (
                f"View → Slide Master, sửa layout đang dùng, Close Master. {s['n']} slide ăn theo. "
                "Slide lệch: Home → Reset (mất chỉnh tay trên slide đó). Thiếu font: File → Options → Save → embed fonts khi gửi."
            ),
        ),
        dict(
            prefix="ppt_169",
            slots=dict(n=["8", "15", "25"]),
            users=[
                "PPT 4:3 chiếu TV 16:9 đen hai bên, {n} slide. Đổi tỷ lệ có vỡ?",
                "Design → Slide Size 16:9, {n} slide.",
            ],
            asst=lambda s: (
                f"Design → Slide Size → Widescreen 16:9. Chọn Maximize (to ra, có thể cắt mép) hoặc Ensure Fit (nhỏ lại, có viền). "
                f"Duyệt {s['n']} slide ảnh. In A4: Layout khác — xuất PDF riêng."
            ),
        ),
        dict(
            prefix="ppt_pdf",
            slots=dict(n=["12", "20", "30"]),
            users=[
                "Xuất {n} slide PDF, bỏ animation, chữ đừng vỡ. PowerPoint.",
                "File PPT nặng, {n} slide, PDF gửi khách.",
            ],
            asst=lambda s: (
                f"File → Export → PDF, Standard. Animation bỏ khi PDF. {s['n']} slide. "
                "Nặng: compress pictures 150ppi, bỏ video nhúng. In: PDF/A nếu lưu trữ."
            ),
        ),
        dict(
            prefix="ppt_notes",
            slots=dict(dev=["laptop + TV HDMI", "một laptop", "Zoom"]),
            users=[
                "Presenter View, khán giả không thấy notes. {dev}.",
                "Ghi chú diễn giả, {dev}.",
            ],
            asst=lambda s: (
                "Slide Show → Use Presenter View. Hai màn: HDMI = chiếu, laptop = notes. "
                f"Một máy/{s['dev']}: không có màn 2 thì in notes hoặc notes dưới cửa sổ. "
                "Zoom: share cửa sổ slideshow, không share desktop notes."
            ),
        ),
        dict(
            prefix="wd_toc",
            slots=dict(pages=["12", "20", "40"], ver=["2016", "365"]),
            users=[
                "Word {pages} trang, tiêu đề Ctrl+B. Mục lục tự động, {ver}.",
                "Heading 1/2, TOC, {pages} trang, Word {ver}.",
            ],
            asst=lambda s: (
                f"Bôi tiêu đề → Styles Heading 1/2 (đừng chỉ Bold). References → Table of Contents. "
                f"{s['pages']} trang: Update Field khi thêm mục. Trang bìa: Section Break, số trang khác section. Word {s['ver']}."
            ),
        ),
        dict(
            prefix="wd_merge",
            slots=dict(n=["40", "80", "200"], ver=["2016", "365"]),
            users=[
                "Mail merge {n} khách: Word + Excel tên/email. {ver}. In hay PDF?",
                "Trộn thư {n} dòng, Word {ver}.",
            ],
            asst=lambda s: (
                f"Mailings → Start Mail Merge → Letter, Select Recipients = file Excel. Insert Merge Field. "
                f"Preview {s['n']} dòng. In hàng loạt hoặc Finish → Print. Email: Merge to Email (cần Outlook). "
                f"Font tên: Excel cột Text. Word {s['ver']}."
            ),
        ),
        dict(
            prefix="wd_track",
            slots=dict(ver=["2016", "365"]),
            users=[
                "Track Changes đỏ đầy hợp đồng. Accept từng chỗ, bản sạch. Word {ver}. Accept All?",
                "Sếp sửa Word {ver}, comment + revisions.",
            ],
            asst=lambda s: (
                f"Review → Next + Accept/Reject từng. Accept All chỉ khi đã đọc hết — dễ nuốt điều khoản. "
                f"Xóa comment: Delete All Comments. So 2 file: Review → Compare. Word {s['ver']}."
            ),
        ),
        dict(
            prefix="wd_table",
            slots=dict(cols=["4", "6", "7"], ver=["2016", "365"]),
            users=[
                "Bảng Word {cols} cột nhảy trang giữa hàng. Header lặp. {ver}.",
                "Table Properties {cols} cột, Word {ver}.",
            ],
            asst=lambda s: (
                f"Table Properties → Row → bỏ 'Allow row to break across pages'. Header: Repeat header rows. "
                f"{s['cols']} cột: Table → AutoFit Window. Dán Excel: Keep text only rồi tạo bảng lại nếu vỡ. Word {s['ver']}."
            ),
        ),
        dict(
            prefix="ol_meet",
            slots=dict(n=["4", "8", "12"], app=["Teams", "Zoom"]),
            users=[
                "Lịch họp {n} người Outlook, đính PPT, {app}. Meeting hay Appointment?",
                "Mời {n} người, múi giờ VN, {app}.",
            ],
            asst=lambda s: (
                f"New Meeting (không phải Appointment) để có To + Teams/Zoom. Time zone GMT+7. "
                f"{s['n']} người: Required vs Optional. Đính PPT < 10MB; lớn thì link SharePoint. "
                f"Người không Outlook: ICS trong mail. {s['app']}: bật link trong meeting options."
            ),
        ),
        dict(
            prefix="xl_bang_luong",
            slots=dict(h=["8", "9"], ot=["1.5", "2.0"], n=["12", "40", "80"]),
            users=[
                "Bảng lương {n} người: giờ × đơn giá, OT {ot} sau {h}h. Excel, lock ô đơn giá.",
                "Công thức OT {ot}, chuẩn {h}h, {n} NV.",
            ],
            asst=lambda s: (
                f"Giờ OT = MAX(0, E2-{s['h']}). Lương = giờ_thường*F$1 + giờ_OT*F$1*{s['ot']}. "
                f"$ khóa đơn giá F$1. {s['n']} dòng. Tròn ROUND(...,0). Protect sheet trừ cột giờ."
            ),
        ),
        dict(
            prefix="refuse_crack",
            slots=dict(x=["Office 2021", "Windows", "Acrobat"]),
            users=[
                "Cho key {x}, máy hết hạn dùng thử.",
                "Crack {x} giúp, công ty không mua.",
            ],
            asst=lambda s: (
                f"Không đưa key/crack {s['x']}. Máy công ty: nhờ IT license. Máy nhà: mua bản quyền hoặc dùng Google Docs/Sheets. Crack = malware + vi phạm bản quyền."
            ),
        ),
        dict(
            prefix="xl_ask_ver",
            slots=dict(ent=ENTS),
            users=[
                "Công thức lọc trùng {ent}.",
                "Lọc trùng {ent} giúp.",
            ],
            asst=lambda s: (
                f"Thiếu bản. 365/Sheets: =UNIQUE(cột {s['ent']}). 2016: copy cột → Remove Duplicates trên bản copy. "
                "Bạn đang Excel 2016, 365 hay Google Sheets? Bao nhiêu dòng?"
            ),
        ),
    ]


def code_kinds():
    return [
        dict(
            prefix="py_read_excel",
            slots=dict(file=FILES, sheet=SHEETS[:4], n=NROWS, os=PYOS),
            users=[
                "pandas đọc {file} sheet '{sheet}', ~{n} dòng, {os}. Code ngắn.",
                "pd.read_excel {file} / {sheet}, máy {os}.",
            ],
            asst=lambda s: (
                f"import pandas as pd\n"
                f"df = pd.read_excel(\"{s['file']}\", sheet_name=\"{s['sheet']}\")\n"
                f"# {s['n']} dòng: mặc định ổn. File đang mở Excel ({s['os']}) → PermissionError, đóng file.\n"
                f"print(df.head(), df.columns.tolist())"
            ),
            follow=("Nhiều sheet.", "sheet_name=None ra dict; hoặc sheet_name=[\"Lương\",\"Bán\"]."),
        ),
        dict(
            prefix="py_csv_enc",
            slots=dict(os=PYOS, src=["Excel CSV", "phần mềm kế toán", "sao kê ngân hàng"]),
            users=[
                "pandas đọc CSV {src} trên {os} UnicodeDecodeError / font lỗi.",
                "Encoding CSV {src}, {os}.",
            ],
            asst=lambda s: (
                "encodings = [\"utf-8-sig\", \"utf-8\", \"cp1258\", \"cp1252\"]\n"
                "for e in encodings:\n"
                "    try:\n"
                "        df = pd.read_csv(\"data.csv\", encoding=e)\n"
                "        break\n"
                "    except UnicodeDecodeError:\n"
                "        continue\n"
                f"# {s['os']}: Windows hay cp1258; Excel CSV thường utf-8-sig. {s['src']}: nếu NaN cột số, decimal=',' ."
            ),
        ),
        dict(
            prefix="py_filter_ot",
            slots=dict(h=["8", "10"], n=NROWS[1:], col=["gio_ot", "OT", "overtime"]),
            users=[
                "Excel lương {n} dòng, lọc {col} > {h}, xuất CSV. pandas.",
                "Lọc OT > {h}h, cột {col}, {n} dòng.",
            ],
            asst=lambda s: (
                f"df = pd.read_excel(\"luong.xlsx\")\n"
                f"df[\"{s['col']}\"] = pd.to_numeric(df[\"{s['col']}\"], errors=\"coerce\")\n"
                f"out = df[df[\"{s['col']}\"] > {s['h']}]\n"
                f"out.to_csv(\"ot.csv\", index=False, encoding=\"utf-8-sig\")  # {s['n']} dòng"
            ),
            follow=("Giờ là text '8:30'.", "df['phut']=pd.to_timedelta(df['gio'].astype(str)).dt.total_seconds()/3600"),
        ),
        dict(
            prefix="py_groupby",
            slots=dict(file=FILES[:5], key=["Category", "phong", "NV", "thang"], val=["doanh_thu", "TotalSales", "so_luong"]),
            users=[
                "{file}: groupby {key} tổng {val}. pandas.",
                "Tổng {val} theo {key} trong {file}.",
            ],
            asst=lambda s: (
                f"df = pd.read_excel(\"{s['file']}\")\n"
                f"g = df.groupby(\"{s['key']}\", dropna=False)[\"{s['val']}\"].sum().reset_index()\n"
                f"g.to_excel(\"tong_{s['key']}.xlsx\", index=False)"
            ),
        ),
        dict(
            prefix="py_merge",
            slots=dict(n=NROWS[1:], key=["ma_sp", "sku", "MST"]),
            users=[
                "Merge đơn vs kho theo {key}, {n} dòng, đơn thiếu hàng. pandas.",
                "left merge {key}, file ~{n} dòng.",
            ],
            asst=lambda s: (
                f"don = pd.read_excel(\"don.xlsx\")\nkho = pd.read_excel(\"kho.xlsx\")\n"
                f"don[\"{s['key']}\"] = don[\"{s['key']}\"].astype(str).str.strip().str.upper()\n"
                f"kho[\"{s['key']}\"] = kho[\"{s['key']}\"].astype(str).str.strip().str.upper()\n"
                f"out = don.merge(kho, on=\"{s['key']}\", how=\"left\", indicator=True)\n"
                f"thieu = out[out[\"_merge\"]==\"left_only\"]  # ~{s['n']} dòng gốc"
            ),
            follow=("Kho trùng mã.", "kho = kho.drop_duplicates(\"{key}\", keep=\"first\") trước merge, hoặc groupby sum số lượng."),
        ),
        dict(
            prefix="py_to_excel",
            slots=dict(n=NROWS, sheet=["Mẫu", "Kết quả", "Bán"]),
            users=[
                "Ghi pandas vào {sheet} file có sẵn, không xóa sheet khác. {n} dòng.",
                "openpyxl mode a, sheet {sheet}, {n} dòng.",
            ],
            asst=lambda s: (
                f"with pd.ExcelWriter(\"out.xlsx\", engine=\"openpyxl\", mode=\"a\", if_sheet_exists=\"replace\") as w:\n"
                f"    df.to_excel(w, sheet_name=\"{s['sheet']}\", index=False)\n"
                f"# {s['n']} dòng. File đang mở Excel → đóng. Công thức sheet khác giữ nếu không replace sheet đó."
            ),
        ),
        dict(
            prefix="py_glob",
            slots=dict(n=["8", "12", "24"], sheet=["Bán", "Lương"]),
            users=[
                "Thư mục {n} file xlsx, gộp sheet '{sheet}'. pathlib, bỏ ~$.",
                "Concat {n} workbook, sheet {sheet}.",
            ],
            asst=lambda s: (
                "from pathlib import Path\nimport pandas as pd\n"
                "files = [p for p in Path('.').glob('*.xlsx') if not p.name.startswith('~$')]\n"
                f"dfs = [pd.read_excel(p, sheet_name=\"{s['sheet']}\") for p in files]  # {s['n']} file kỳ vọng\n"
                "pd.concat(dfs, ignore_index=True).to_excel('gop.xlsx', index=False)"
            ),
        ),
        dict(
            prefix="py_datetime",
            slots=dict(os=PYOS, fmt=["13/09/2026", "2026-09-13"]),
            users=[
                "Cột ngày lẫn '{fmt}' và loại kia, pandas NaT, lọc tháng 9, {os}.",
                "dayfirst ngày VN, {os}, mẫu {fmt}.",
            ],
            asst=lambda s: (
                "df['ngay'] = pd.to_datetime(df['ngay'], dayfirst=True, errors='coerce')\n"
                "thang9 = df[df['ngay'].dt.month == 9]\n"
                f"# {s['os']}: serial Excel → origin='1899-12-30'. Timezone VN không cần nếu chỉ ngày."
            ),
        ),
        dict(
            prefix="sql_group",
            slots=dict(db=DBS, val=["doanh_thu", "so_tien"], key=["nv", "ma_kh"]),
            users=[
                "SQL {db}: tổng {val} theo {key} tháng này. GROUP BY.",
                "{db} SUM({val}) GROUP BY {key}.",
            ],
            asst=lambda s: (
                f"SELECT {s['key']}, SUM({s['val']}) AS tong\n"
                f"FROM don_hang\n"
                f"WHERE ngay >= '2026-09-01' AND ngay < '2026-10-01'\n"
                f"GROUP BY {s['key']};\n"
                f"-- {s['db']}: SELECT * không đi với GROUP BY. Lọc sau gộp → HAVING."
            ),
            follow=("Top 5.", "… ORDER BY tong DESC LIMIT 5;  -- SQL Server: TOP 5 / OFFSET FETCH."),
        ),
        dict(
            prefix="sql_left",
            slots=dict(db=DBS, key=["ma_kh", "ma_sp"]),
            users=[
                "Đơn không khớp khách, LEFT JOIN {key}, {db}. Đừng INNER.",
                "{db} tìm {key} thiếu phía phải.",
            ],
            asst=lambda s: (
                f"SELECT d.*\nFROM don d\nLEFT JOIN khach k ON d.{s['key']} = k.{s['key']}\n"
                f"WHERE k.{s['key']} IS NULL;\n-- {s['db']}: TRIM hai bên nếu space. INNER sẽ mất đúng dòng cần tìm."
            ),
        ),
        dict(
            prefix="sql_having",
            slots=dict(db=DBS, x=["50", "80", "100"]),
            users=[
                "NV doanh thu > {x} triệu tháng này. WHERE hay HAVING, {db}?",
                "{db} HAVING SUM > {x}.",
            ],
            asst=lambda s: (
                f"SELECT nv, SUM(doanh_thu) AS tong\nFROM don\nWHERE thang = 9  -- lọc dòng trước\n"
                f"GROUP BY nv\nHAVING SUM(doanh_thu) > {s['x']} * 1000000;\n"
                f"-- {s['db']}: WHERE không dùng SUM. Đơn vị triệu vs đồng: thống nhất."
            ),
        ),
        dict(
            prefix="sql_null",
            slots=dict(db=DBS),
            users=[
                "{db}: WHERE sdt <> '' vẫn sót NULL. Lọc chưa có SĐT?",
                "NULL vs rỗng, {db}.",
            ],
            asst=lambda s: (
                f"WHERE sdt IS NULL OR TRIM(sdt) = '';\n"
                f"-- {s['db']}: NULL <> '' ra UNKNOWN, dòng bị loại. Hiện 'chưa có': COALESCE(NULLIF(TRIM(sdt),''), 'chưa có')."
            ),
        ),
        dict(
            prefix="sql_update",
            slots=dict(db=DBS, grp=["PK", "NB", "KM"]),
            users=[
                "UPDATE giá nhóm {grp} trên {db}. Sợ quên WHERE. Thử thế nào?",
                "{db} update nhóm {grp}, transaction?",
            ],
            asst=lambda s: (
                f"-- 1) SELECT * FROM sp WHERE nhom = '{s['grp']}';  -- đếm dòng\n"
                f"START TRANSACTION;\n"
                f"UPDATE sp SET gia = gia * 1.05 WHERE nhom = '{s['grp']}';\n"
                f"-- {s['db']}: kiểm tra rồi COMMIT hoặc ROLLBACK. Không DROP TABLE. Backup trước production."
            ),
        ),
        dict(
            prefix="sql_param",
            slots=dict(db=DBS[:3]),
            users=[
                "Ghép SQL string từ ô Excel, {db}. Có sao không?",
                "User gõ mã SP vào query {db}.",
            ],
            asst=lambda s: (
                f"Đừng f\"WHERE ma='{{user}}'\". Dùng placeholder: "
                f"cursor.execute('SELECT * FROM sp WHERE ma=%s', (ma,))  -- {s['db']}. "
                "Tránh SQL injection, lỗi dấu nháy trong mã."
            ),
        ),
        dict(
            prefix="git_stash",
            slots=dict(f=["app.py", "index.html", "main.py"]),
            users=[
                "Sửa dở {f}, sếp kêu pull. git stash, Windows.",
                "stash {f} rồi pull, conflict thì sao?",
            ],
            asst=lambda s: (
                f"git status\ngit stash push -m \"wip {s['f']}\"\ngit pull\ngit stash pop\n"
                "Conflict: sửa file, git add, không stash drop trước khi chắc. Untracked: git stash -u."
            ),
        ),
        dict(
            prefix="git_conflict",
            slots=dict(f=["main.py", "README.md", "config.json"]),
            users=[
                "git pull conflict {f}, nhánh feature. VS Code. Lệnh an toàn?",
                "Conflict {f}, giữ code mình hay merge tay?",
            ],
            asst=lambda s: (
                f"Mở {s['f']}, xóa marker <<<<<< ===== >>>>>>, giữ đúng ý, git add {s['f']} && git commit "
                "(nếu merge). Đừng git pull --force. Excel .xlsx conflict: chọn một bản, không merge binary."
            ),
        ),
        dict(
            prefix="git_undo",
            slots=dict(f=[".env", "luong.xlsx", "tmp.csv"]),
            users=[
                "Commit nhầm {f} chưa push. Undo, đừng mất công.",
                "Gỡ {f} khỏi commit cuối, chưa push.",
            ],
            asst=lambda s: (
                f"git restore --staged {s['f']}  # nếu vừa add\n"
                f"git reset --soft HEAD~1     # xóa commit, giữ file máy\n"
                f"echo '{s['f']}' >> .gitignore && git add .gitignore\n"
                "Đã push: không reset --hard remote; git revert. Không force main."
            ),
        ),
        dict(
            prefix="git_ignore",
            slots=dict(x=["*.xlsx", "*.csv", ".env"]),
            users=[
                "Repo đang track {x}. .gitignore, gỡ khỏi git, giữ file máy.",
                "Bỏ {x} khỏi GitHub, local còn.",
            ],
            asst=lambda s: (
                f"# .gitignore\n{s['x']}\ngit rm --cached {s['x']}\ngit commit -m \"stop tracking {s['x']}\"\n"
                "File máy còn. Đồng nghiệp pull: file biến mất khỏi repo — báo trước. Giữ mẫu.xlsx bằng exception !mau.xlsx."
            ),
        ),
        dict(
            prefix="py_argparse",
            slots=dict(ot=["8", "10"]),
            users=[
                "python loc.py --file luong.xlsx --ot {ot}. argparse, Windows, báo nếu thiếu file.",
                "CLI --ot {ot}, đường dẫn tiếng Việt.",
            ],
            asst=lambda s: (
                "import argparse, sys\nfrom pathlib import Path\n"
                "p = argparse.ArgumentParser()\n"
                "p.add_argument('--file', required=True)\n"
                f"p.add_argument('--ot', type=float, default={s['ot']})\n"
                "a = p.parse_args()\n"
                "path = Path(a.file)\n"
                "if not path.exists():\n    sys.exit(f'Không thấy file: {path}')"
            ),
        ),
        dict(
            prefix="py_try_lock",
            slots=dict(os=PYOS),
            users=[
                "read_excel lúc FileNotFound, lúc Excel đang mở. try/except tiếng Việt, {os}.",
                "PermissionError {os}, báo kế toán.",
            ],
            asst=lambda s: (
                "try:\n    df = pd.read_excel(path)\n"
                "except FileNotFoundError:\n    raise SystemExit('Không thấy file. Kiểm tra tên và thư mục.')\n"
                "except PermissionError:\n    raise SystemExit('Đóng file trên Excel rồi chạy lại.')\n"
                f"# {s['os']}+OneDrive: chờ xong sync (icon mây)."
            ),
        ),
        dict(
            prefix="py_regex_phone",
            slots=dict(mau=["09x", "+84", "có chấm cách"]),
            users=[
                "Cột ghi chú Excel, tách SĐT 10 số VN, mẫu {mau}. pandas.",
                "Regex SĐT {mau} từ text.",
            ],
            asst=lambda s: (
                "import re\n"
                "s = df['ghi_chu'].astype(str).str.replace(r'\\D', '', regex=True)\n"
                "s = s.str.replace(r'^84', '0', regex=True)\n"
                "df['sdt'] = s.where(s.str.match(r'^0\\d{9}$'), pd.NA)\n"
                f"# {s['mau']}: bỏ chữ, +84 → 0, giữ 10 số. Số bàn 028: nới regex ^0\\d{9,10}$."
            ),
        ),
        dict(
            prefix="py_venv",
            slots=dict(os=PYOS),
            users=[
                "pip pandas đụng Python khác, {os}. venv + VS Code interpreter.",
                "requirements.txt, {os}, không admin.",
            ],
            asst=lambda s: (
                "python -m venv .venv\n"
                + (".venv\\Scripts\\activate\n" if s["os"] == "Windows" else "source .venv/bin/activate\n")
                + "pip install pandas openpyxl\npip freeze > requirements.txt\n"
                "VS Code: Python: Select Interpreter → .venv. Không quyền admin: venv đủ, khỏi Anaconda nếu chưa có."
            ),
        ),
        dict(
            prefix="py_json_cfg",
            slots=dict(os=PYOS),
            users=[
                "Đừng hard-code đường dẫn lương. config.json, {os}, đừng commit mật khẩu.",
                "JSON config input/sheet/OT, {os}.",
            ],
            asst=lambda s: (
                '{"input": "luong.xlsx", "sheet": "Lương", "ot": 8}\n'
                "cfg = json.loads(Path('config.json').read_text(encoding='utf-8'))\n"
                "df = pd.read_excel(cfg['input'], sheet_name=cfg['sheet'])\n"
                f"# {s['os']}: config.local.json ghi .gitignore. Thiếu key: cfg.get('ot', 8)."
            ),
        ),
        dict(
            prefix="py_mail",
            slots=dict(n=["1", "3"]),
            users=[
                "Python gửi mail + Excel cho sếp mỗi sáng, Gmail công ty. SMTP, {n} file đính.",
                "smtplib app password, {n} đính kèm.",
            ],
            asst=lambda s: (
                "Dùng app password, không nhét password vào git.\n"
                "smtplib.SMTP_SSL('smtp.gmail.com', 465) + start login.\n"
                f"MIMEMultipart, đính {s['n']} file. >10MB: link Drive. Công ty Google Workspace: hỏi IT SMTP."
            ),
        ),
        dict(
            prefix="py_plot",
            slots=dict(os=["Windows"]),
            users=[
                "matplotlib cột doanh thu, tiếng Việt vuông trên {os}. PNG 150dpi.",
                "Font tiếng Việt chart {os}.",
            ],
            asst=lambda s: (
                "import matplotlib.pyplot as plt\n"
                "plt.rcParams['font.family'] = 'DejaVu Sans'  # hoặc 'Times New Roman' / 'Arial' trên Windows\n"
                "plt.bar(df['thang'], df['doanh_thu'])\n"
                "plt.tight_layout()\nplt.savefig('dt.png', dpi=150)\n"
                f"# {s['os']}: nếu vẫn vuông, chọn font có sẵn Control Panel → Fonts."
            ),
        ),
        dict(
            prefix="py_sqlite",
            slots=dict(size=["vài nghìn", "200k"]),
            users=[
                "Log JSON đơn, tra cứu local SQLite, {size} dòng. pandas.",
                "sqlite3 + pandas, {size} dòng.",
            ],
            asst=lambda s: (
                "import sqlite3, pandas as pd\n"
                "con = sqlite3.connect('don.db')\n"
                "df.to_sql('don', con, if_exists='append', index=False)\n"
                "q = pd.read_sql('SELECT ma_nv, SUM(tien) FROM don GROUP BY ma_nv', con)\n"
                f"# {s['size']}: index ma_nv. Backup copy don.db. Trùng mã: PRIMARY KEY hoặc drop_duplicates."
            ),
        ),
        dict(
            prefix="py_requests",
            slots=dict(n=["2", "3", "5"]),
            users=[
                "requests kéo bảng giá nội bộ hay treo. timeout, retry {n}, đừng in token.",
                "HTTP timeout retry {n}, VPN công ty.",
            ],
            asst=lambda s: (
                "import time, requests\n"
                f"for i in range({s['n']}):\n"
                "    try:\n"
                "        r = requests.get(url, timeout=15, headers={\"Authorization\": token})\n"
                "        r.raise_for_status()\n"
                "        break\n"
                "    except requests.RequestException:\n"
                "        time.sleep(2 ** i)\n"
                "# SSL nội bộ: verify=path_ca hoặc hỏi IT. Đừng print(headers)."
            ),
        ),
        dict(
            prefix="js_csv_bom",
            slots=dict(ver=["2013", "2016", "365"]),
            users=[
                "Node gửi CSV, Excel {ver} lỗi font tiếng Việt. BOM utf-8?",
                "CSV download Excel {ver}, encoding.",
            ],
            asst=lambda s: (
                f"Excel {s['ver']} Windows: utf-8-sig (BOM) hoặc xuất .xlsx (exceljs/excel4node). "
                "CSV: '\\uFEFF' + content, charset=utf-8. 2013 hay cần BOM. Muốn chắc: xlsx."
            ),
        ),
        dict(
            prefix="vba_filter",
            slots=dict(ver=["2016", "365"], st=["mới", "xong"]),
            users=[
                "Macro lọc sheet Đơn trạng thái '{st}' copy sheet khác. Excel {ver}.",
                "VBA AutoFilter {st}, {ver}. Công ty cấm macro thì sao?",
            ],
            asst=lambda s: (
                f"Excel {s['ver']}: AutoFilter Field:=3 Criteria1:=\"{s['st']}\" rồi SpecialCells visible copy. "
                "Cấm macro: Power Query hoặc pandas. Nút: gán macro Form Control. Lưu .xlsm."
            ),
        ),
        dict(
            prefix="py_dups_file",
            slots=dict(n=["200", "500", "2000"]),
            users=[
                "Download {n} file trùng. Python hash, đừng xóa vội. Windows.",
                "Liệt kê trùng nội dung, {n} file.",
            ],
            asst=lambda s: (
                "import hashlib, collections\nfrom pathlib import Path\n"
                "h = collections.defaultdict(list)\n"
                "for p in Path(r'D:\\Downloads').iterdir():\n"
                "    if p.is_file():\n"
                "        h[hashlib.md5(p.read_bytes()).hexdigest()].append(p)\n"
                "dups = {k:v for k,v in h.items() if len(v)>1}\n"
                f"# {s['n']} file: in dups, xóa tay. File đang mở có thể hash fail."
            ),
        ),
        dict(
            prefix="refuse_drop",
            slots=dict(db=DBS),
            users=[
                "DROP TABLE rồi tạo lại cho sạch, {db} production.",
                "{db} xóa hết bảng SP trên live.",
            ],
            asst=lambda s: (
                f"Không DROP TABLE production trên {s['db']}. SELECT + DELETE/UPDATE có WHERE trên staging, "
                "transaction, backup. Nói rõ bảng và mục đích."
            ),
        ),
        dict(
            prefix="refuse_force",
            slots=dict(br=["main", "master", "develop"]),
            users=[
                "git push --force nhánh {br} cho xong, bị reject.",
                "Force {br} sau rebase.",
            ],
            asst=lambda s: (
                f"Không force push {s['br']} dùng chung. git pull --rebase, giải conflict, push thường. "
                "Force chỉ nhánh riêng và team biết."
            ),
        ),
        dict(
            prefix="py_ask_file",
            slots=dict(x=["lương", "kho", "công nợ"]),
            users=[
                "pandas đọc file {x} bị lỗi, sửa giúp.",
                "Code {x} lỗi, không gửi file.",
            ],
            asst=lambda s: (
                f"Thiếu file {s['x']}: đuôi xlsx/csv, Windows/Mac, traceback, tên sheet. "
                "CSV: utf-8-sig/cp1258. Excel: pd.read_excel(..., sheet_name=0). Đừng đoán xóa cột."
            ),
        ),
    ]


def expand_kind(kind, domain, source, rng, quota, follow_rate=0.28):
    rows = []
    slots = kind["slots"]
    users = kind["users"]
    asst = kind["asst"]
    follow = kind.get("follow")
    combos = list(pick_product(slots, max(quota * 3, 50), rng))
    rng.shuffle(combos)
    i = 0
    while len(rows) < quota and combos:
        s = combos[i % len(combos)]
        i += 1
        if i > quota * 4:
            break
        u_tmpl = users[i % len(users)]
        try:
            u = u_tmpl.format(**s)
            if len(u) < 48:
                u = users[0].format(**s)
            a = asst(s)
            if "{dept}" in a:
                a = a.format(**s)
        except (KeyError, ValueError, IndexError):
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
        rid = f"{kind['prefix']}_{len(rows)}_{abs(hash((u, a))) % 10**8}"
        rows.append(rec(source, rid, domain, ms))
        if i >= len(combos) and len(rows) < quota:
            rng.shuffle(combos)
    return rows


def fill(kinds, domain, source, n, seed):
    rng = random.Random(seed)
    per = max(1, n // len(kinds))
    extra = n - per * len(kinds)
    out = []
    for i, k in enumerate(kinds):
        q = per + (1 if i < extra else 0)
        part = expand_kind(k, domain, source, rng, q)
        if len(part) < q:
            part += expand_kind(k, domain, source, random.Random(seed + 17 + i), q - len(part))
        out.extend(part[:q])
    rng.shuffle(out)
    # id uniqueness
    seen = set()
    clean = []
    for r in out:
        rid = r["id"]
        if rid in seen:
            r = dict(r)
            r["id"] = rid + "_x"
        seen.add(r["id"])
        clean.append(r)
    return clean[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=N)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    off = fill(office_kinds(), "van_phong", "office-synth", a.n, a.seed)
    code = fill(code_kinds(), "lap_trinh", "code-synth", a.n, a.seed + 1)
    write_jsonl(OUT_OFF, off)
    write_jsonl(OUT_CODE, code)
    print(f"office {len(off)} → {OUT_OFF}")
    print(f"code   {len(code)} → {OUT_CODE}")
    print("mẫu office:", off[0]["messages"][0]["content"][:80])
    print("mẫu code  :", code[0]["messages"][0]["content"][:80])


if __name__ == "__main__":
    main()
