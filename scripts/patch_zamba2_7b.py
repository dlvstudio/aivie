"""TK_105 · vá transformers 5.16.1 để nạp Zamba2-7B (chạy 1 lần trong venv có transformers).

Lỗi: `get_layers` chọn khối dùng chung theo `layer_id % num_mem_blocks`, còn vòng buộc trọng số + adapter LoRA theo
tầng đi theo THỨ TỰ tầng lai. Zamba2-7B có hybrid_layer_ids 6, 11, 17, 23… (xen lẻ/chẵn) → tầng 17 dựng nhầm khối 1
→ ValueError `tie_weights_keys` (hoặc gắn nhầm adapter). Sửa: block_id = (số tầng lai đứng trước) % num_mem_blocks.
    /workspace/zenv/bin/python scripts/patch_zamba2_7b.py
"""
import transformers.models.zamba2.modeling_zamba2 as M

p = M.__file__
s = open(p).read()
old = "                block_id = layer_id % self.config.num_mem_blocks"
new = ("                # vá TK_105: khối dùng chung theo THỨ TỰ tầng lai (khớp vòng buộc trọng-số + adapter theo tầng)\n"
       "                block_id = sum(t == \"hybrid\" for t in self.layers_block_type[:layer_id]) % self.config.num_mem_blocks")
if new.splitlines()[1] in s:
    print("đã vá từ trước:", p)
else:
    assert old in s, "không thấy dòng cần vá — transformers đã đổi?"
    open(p, "w").write(s.replace(old, new))
    print("đã vá:", p)
