import json,os,mlx.core as mx
B="out/exp2_mamba_layer"; S=[5,11,17,23,29,35,41,47]; src={}
for r in ["gla","glb"]:
    lay=json.load(open(f"{B}/{r}/train_log.json"))["layers"]; w=mx.load(f"{B}/{r}/mixers.safetensors")
    for j,i in enumerate(lay): src[i]={k.split(".",2)[2]:v for k,v in w.items() if k.startswith(f"mixers.{j}.")}
out={f"mixers.{j}.{k}":v for j,i in enumerate(S) for k,v in src[i].items()}
os.makedirs(f"{B}/gl",exist_ok=True); mx.save_safetensors(f"{B}/gl/mixers.safetensors",out)
json.dump({"layers":S,"note":"gla (L2048×500) + glb (L1024×1000)"},open(f"{B}/gl/train_log.json","w")); print("merged",len(out))
