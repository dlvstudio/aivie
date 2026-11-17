# TK_108: thử model bằng 8 đề THỰC TẾ nhiều ràng buộc (TK_108 §13) qua /api/chat của serve_hybrid_ui.py.
# Bài code có test → trích khối python dài nhất và chạy thật (python -I, 20 giây); đề khác lưu đáp án để đọc tay.
#   python scripts/m4_probe_real.py http://localhost:8801 out/m4_probe_cx3b.jsonl [ten1,ten2]
import json, re, subprocess, sys, time, urllib.request, tempfile, os

URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8801"
OUT = sys.argv[2] if len(sys.argv) > 2 else "out/m4_probe.jsonl"
P = [
 ("py_invoice", """Viết hàm Python `tong_hoa_don(items, vat=0.1, giam_gia=None)`:
- items là list dict {"ten": str, "don_gia": int, "so_luong": int}
- giam_gia là None hoặc dict {"loai": "phan_tram"|"tien", "gia_tri": số}; giảm giá áp dụng TRƯỚC thuế, không để tạm tính âm
- trả về dict {"tam_tinh", "giam", "thue", "tong"} làm tròn đến đồng (int)
- items rỗng → mọi trường = 0; so_luong âm → raise ValueError
Chỉ trả về code.""",
  """
r = tong_hoa_don([{"ten":"A","don_gia":100000,"so_luong":2},{"ten":"B","don_gia":50000,"so_luong":1}])
assert r == {"tam_tinh":250000,"giam":0,"thue":25000,"tong":275000}, r
r = tong_hoa_don([{"ten":"A","don_gia":100000,"so_luong":2}], giam_gia={"loai":"phan_tram","gia_tri":10})
assert r["giam"] == 20000 and r["thue"] == 18000 and r["tong"] == 198000, r
r = tong_hoa_don([{"ten":"A","don_gia":10000,"so_luong":1}], giam_gia={"loai":"tien","gia_tri":50000})
assert r["tong"] == 0, r
assert tong_hoa_don([]) == {"tam_tinh":0,"giam":0,"thue":0,"tong":0}
try:
    tong_hoa_don([{"ten":"A","don_gia":1,"so_luong":-1}]); assert False
except ValueError: pass
"""),
 ("py_logparse", """Tôi có file log nginx dạng:
192.168.1.10 - - [21/Oct/2026:10:15:32 +0700] "GET /api/orders?page=2 HTTP/1.1" 200 5123 "-" "Mozilla/5.0"
Viết hàm Python `top_loi(lines, n=3)` nhận list các dòng log, trả về list n tuple (path, so_lan) của các path (bỏ query string) có status >= 500 nhiều nhất, sắp giảm dần theo số lần, bằng nhau thì theo path tăng dần. Dòng sai định dạng thì bỏ qua.""",
  """
L = ['1.1.1.1 - - [21/Oct/2026:10:15:32 +0700] "GET /api/orders?page=2 HTTP/1.1" 500 12 "-" "x"',
     '1.1.1.1 - - [21/Oct/2026:10:15:33 +0700] "GET /api/orders HTTP/1.1" 502 12 "-" "x"',
     '1.1.1.2 - - [21/Oct/2026:10:15:34 +0700] "POST /api/pay HTTP/1.1" 503 12 "-" "x"',
     '1.1.1.2 - - [21/Oct/2026:10:15:35 +0700] "GET /home HTTP/1.1" 200 12 "-" "x"',
     'rac rac',
     '1.1.1.3 - - [21/Oct/2026:10:15:36 +0700] "GET /api/a HTTP/1.1" 500 1 "-" "x"']
assert top_loi(L, 3) == [("/api/orders", 2), ("/api/a", 1), ("/api/pay", 1)], top_loi(L, 3)
assert top_loi([], 3) == []
"""),
 ("py_debug", """Hàm này đôi khi trả kết quả sai khi gọi nhiều lần, tìm lỗi và sửa:
```python
def them_vao_gio(sp, gio=[]):
    for x in gio:
        if x["id"] == sp["id"]:
            x["sl"] += sp["sl"]
            return gio
    gio.append(sp)
    return gio
```
Giải thích ngắn gọn nguyên nhân rồi đưa code đã sửa.""",
  """
a = them_vao_gio({"id":1,"sl":1})
b = them_vao_gio({"id":2,"sl":1})
assert len(b) == 1, b
g = them_vao_gio({"id":1,"sl":2}, [{"id":1,"sl":1}])
assert g[0]["sl"] == 3
"""),
 ("sql_report", """CSDL SQL Server có bảng:
DonHang(MaDH int, MaKH int, NgayDat date, TrangThai nvarchar(20))
ChiTietDH(MaDH int, MaSP int, SoLuong int, DonGia decimal(18,2))
KhachHang(MaKH int, TenKH nvarchar(100), TinhThanh nvarchar(50))
Viết câu T-SQL lấy top 5 khách hàng có tổng doanh thu cao nhất trong quý 3/2026 (chỉ tính đơn TrangThai = N'Hoàn thành'), kèm số đơn và giá trị đơn trung bình, sắp giảm dần theo doanh thu.""", None),
 ("excel", """Trong Excel, cột A là ngày (A2:A500), cột B là mã nhân viên, cột C là doanh số. Tôi muốn ở ô F2 tính tổng doanh số của nhân viên có mã ở E2 trong tháng 10/2026. Viết công thức và giải thích ngắn.""", None),
 ("js_debounce", """Viết hàm JavaScript `debounce(fn, ms)` trả về hàm mới chỉ gọi fn sau khi ngừng gọi ms mili-giây, giữ đúng `this` và tham số của lần gọi cuối, và có thêm phương thức `.cancel()`. Kèm ví dụ dùng cho ô tìm kiếm.""", None),
 ("arch", """Công ty tôi có ứng dụng bán hàng monolith .NET + SQL Server, khoảng 2.000 đơn/ngày, giờ cao điểm trang danh sách sản phẩm chậm (3–5 giây). Hãy đề xuất các bước tối ưu theo thứ tự ưu tiên, cụ thể, có thể làm trong 2 tuần, không viết lại hệ thống.""", None),
 ("email", """Soạn email gửi khách hàng thông báo hệ thống bảo trì từ 22:00 ngày 25/10 đến 02:00 ngày 26/10, trong thời gian này không đặt hàng được, đơn đang xử lý không bị ảnh hưởng. Giọng chuyên nghiệp, ngắn gọn, có lời xin lỗi.""", None),
]


def chat(q, max_new=700):
    body = json.dumps({"messages": [{"role": "user", "content": q}], "max_new": max_new, "temperature": 0.2,
                       "top_p": 0.9, "rep_pen": 1.1, "conv_id": f"p{time.time()}"}).encode()
    req = urllib.request.Request(URL + "/api/chat", body, {"Content-Type": "application/json"})
    done = {}
    with urllib.request.urlopen(req, timeout=1800) as r:
        for line in r:
            line = line.decode().strip()
            if line.startswith("data: "):
                o = json.loads(line[6:])
                if o["type"] in ("done", "error"):
                    done = o
    return done


def run_test(ans, test):
    blocks = re.findall(r"```(?:python|py)?\n(.*?)```", ans, re.S) or [ans]
    code = max(blocks, key=len)
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(code + "\n" + test)
        p = f.name
    try:
        r = subprocess.run([sys.executable, "-I", p], capture_output=True, text=True, timeout=20)
        return "PASS" if r.returncode == 0 else "FAIL: " + (r.stderr.strip().splitlines() or ["?"])[-1][:200]
    except subprocess.TimeoutExpired:
        return "FAIL: timeout"
    finally:
        os.unlink(p)


only = sys.argv[3].split(",") if len(sys.argv) > 3 else None
with open(OUT, "a") as fo:
    for name, q, test in P:
        if only and name not in only:
            continue
        d = chat(q)
        ans = d.get("text", d.get("error", ""))
        res = run_test(ans, test) if test else "-"
        rec = {"name": name, "tok": d.get("tokens"), "tok_s": d.get("tok_s"), "test": res, "answer": ans}
        fo.write(json.dumps(rec, ensure_ascii=False) + "\n")
        fo.flush()
        print(f"=== {name} · {d.get('tokens')} tok · {d.get('tok_s')} tok/s · test {res}", flush=True)
