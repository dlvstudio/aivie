from pathlib import Path

import torch

from nn_aivie.core.neural_lm import NeuralLM
from nn_aivie.core.tokenizer import BPETokenizer


root = Path(".").resolve()
tok = BPETokenizer.load(root / "checkpoints/bpe_v1.json")
V = len(tok)
ckpt = root / "out/lm_v2_v11.pt"
lm = NeuralLM.load(ckpt, V, device="cpu")
lm.model.eval()

prompts = [
    "Việt Nam nằm ở khu vực",
    "Đồng bằng sông Hồng có đặc điểm",
    "Khí hậu nhiệt đới gió mùa là",
    "Dãy Hoàng Liên Sơn thuộc",
    "Sông Mekong chảy qua",
    "Thủ đô của Lào là",
    "Biển Đông có vai trò",
    "Tây Nguyên nổi bật với",
    "Vùng duyên hải miền Trung thường",
    "Bản đồ địa lý giúp học sinh",
]


def gen_greedy(prompt: str, n: int = 60) -> str:
    ids = tok.encode(prompt)
    p_len = len(ids)
    for _ in range(n):
        x = torch.tensor(ids[-lm.ctx:], dtype=torch.long)[None]
        with torch.no_grad():
            out = lm.model(x)[0, -1]
            nxt = int(torch.argmax(out).item())
        ids.append(nxt)
    return tok.decode(ids[p_len:])


print("=== Prompt eval v11 (geography, greedy, +60 tokens) ===")
print(f"checkpoint: {ckpt.name}")
for p in prompts:
    out = gen_greedy(p).replace("\n", "\\n")
    print(f"PROMPT: {p}")
    print(f"GEN   : {out}")
    print()