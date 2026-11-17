from dataclasses import replace
from pathlib import Path
from nn_aivie.core.checkpoint_io import load_checkpoint
from nn_aivie.hca.config import HCAConfig
from nn_aivie.hca.generate_hebbian import generate
from nn_aivie.hca.readout import decode_surface
from nn_aivie.hca.w_skip import clear_w_skip_cache

clear_w_skip_cache()
state, _ = load_checkpoint("checkpoints/network_v12_sent.pkl")
cfg = replace(HCAConfig(), ar_w_skip_use_mz_agenda=False, ar_multizone=False)
prompt = "Năm 1010, vua Lý Thái Tổ"
lines = ["=== EOS+dual+hist-after-boundary smoke (mz off) ===", ""]
for seed in range(5):
    text = decode_surface(state, generate(state, prompt, max_len=48, config=cfg, seed=seed))
    lines.append(f"seed{seed}: {text}")
    print(f"seed{seed}: {text}", flush=True)
Path("out/w_skip_primary_smoke.log").write_text("\n".join(lines) + "\n", encoding="utf-8")
