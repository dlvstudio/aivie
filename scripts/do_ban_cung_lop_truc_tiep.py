"""§104 — HUB có tự chứa BẠN CÙNG LỚP không? (bỏ hẳn bước gọi tên lớp)

§103.2 thấy một mảnh dương duy nhất: gieo `cáo·chuột·cầy` với `typed_first`,
`gidf≥6` thì hub trả về có `sói`, `chồn` — bạn cùng lớp THẬT, mà KHÔNG đi qua
nhãn lớp nào. Đó đúng là «nhóm ngầm» người dùng đề xuất, nhưng trên kênh CÓ KIỂU
chứ không phải kênh đồng hiện.

Đo: gieo từ VẬT MANG của mỗi bộ → hub trả về có chứa «khỉ» (thành viên cùng lớp
mà bộ đó giấu đi) không? So với ĐỐI CHỨNG HOÁN VỊ khoá.

Khác mọi phép đo trước: KHÔNG dùng nhãn lớp ở giữa. Hub chính là đề cử.
"""
from __future__ import annotations
import gc, json, random, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
gc.disable()
from nn_aivie.core.checkpoint_io import load_checkpoint  # noqa: E402
from nn_aivie.hca import hub_spread as H  # noqa: E402
from nn_aivie.hca.config import HCAConfig  # noqa: E402

T0 = time.time()

def main() -> int:
    ck = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/network_v12_sent_ls.pkl"
    c = HCAConfig()
    st, _ = load_checkpoint(ck); v = st.vocab
    idx = H.build_hub_spread_index(
        st, d_max=c.hub_chain_d_max, min_count=c.hub_chain_min_count,
        head_rule=c.hub_chain_head_rule, head_rule_trai=c.hub_chain_head_rule_trai,
        label_gidf_q=c.hub_chain_gidf_q, head_gidf_q=c.hub_chain_head_gidf_q,
        label_indeg_q=c.hub_chain_indeg_q, gop_manh_tu=c.hub_chain_gop_manh_tu,
        pos_map=H.load_pos_map(c.hub_chain_pos_dict),
        kenh_vi_ngu=bool(c.hub_chain_kenh_vi_ngu),
        chu_ngu_ket_hop_q=float(c.hub_chain_chu_gia_q))
    kh = [k for k in json.load(open("out/loai_suy_that_khoa.json")) if k["khi"]]
    print(f"\n### {len(kh)} bộ · gieo từ VẬT MANG · hub CHÍNH LÀ đề cử\n")

    def cham(bo, g, kenh, ten):
        d1 = d5 = d10 = n = 0
        for k in bo:
            ids = [i for m in k["gieo"][:3]
                   for i in (H.ids_theo_be_mat(idx, v, m) or [])[:1]]
            # §104.2 — nới xuống ≥1 hạt: chỉ 27/198 bộ có ≥2 vật mang (trần dữ
            # liệu, §88.12), mà n=27 với 3 lần trúng thì không tách được khỏi
            # nhiễu — đúng bẫy §101 đã sập ở n=44.
            if len(ids) < int(sys.argv[2] if len(sys.argv) > 2 else 2):
                continue
            hs = H.score_hubs_from(st, idx, ids, topk=10, gidf_min=g, channel=kenh)
            top = [h.surface.strip() for h in hs]
            if not top:
                continue
            n += 1
            khi = set(k["khi"])
            if top[:1] and top[0] in khi: d1 += 1
            if any(x in khi for x in top[:5]): d5 += 1
            if any(x in khi for x in top[:10]): d10 += 1
        if n:
            print(f"   {ten:<16} gidf≥{g:<4} {kenh:<12} {n:>3} bộ · "
                  f"top-1 {100*d1/n:4.1f}% · top-5 {100*d5/n:4.1f}% · top-10 {100*d10/n:4.1f}%",
                  flush=True)

    rng = random.Random(0)
    khis = [k["khi"] for k in kh]; rng.shuffle(khis)
    hv = [dict(k, khi=x) for k, x in zip(kh, khis)]
    for kenh in ("typed_first", "typed"):
        for g in (0.0, 4.0, 6.0, 8.0):
            cham(kh, g, kenh, "THẬT")
            cham(hv, g, kenh, "  hoán vị")
        print()
    print(f"[{time.time()-T0:.0f}s] xong")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
