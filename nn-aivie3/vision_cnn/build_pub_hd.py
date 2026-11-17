#!/usr/bin/env python
# Nhúng cnnw_hd.json + ảnh mẫu (planar RGB base64, 96×96) vào vision_hd.html → vision_hd_pub.html
import os, json, base64, numpy as np
ROOT=os.path.dirname(os.path.abspath(__file__))
cnnw=open(os.path.join(ROOT,'cnnw_hd.json')).read()
d=np.load(os.path.join(ROOT,'samples_hd.npz')); X,y=d['X'],d['y']     # X:(K,96,96,3) HWC uint8
samples=[]
for i in range(len(X)):
    planar=np.ascontiguousarray(X[i].transpose(2,0,1)).reshape(-1).astype(np.uint8)   # CHW → c*S*S+y*S+x
    samples.append({'y':int(y[i]),'d':base64.b64encode(planar.tobytes()).decode()})
html=open(os.path.join(ROOT,'vision_hd.html')).read()
assert '/*__CNNW__*/ null' in html and '/*__SAMPLES__*/ null' in html
html=html.replace('/*__CNNW__*/ null','/*__CNNW__*/ '+cnnw).replace('/*__SAMPLES__*/ null','/*__SAMPLES__*/ '+json.dumps(samples))
open(os.path.join(ROOT,'vision_hd_pub.html'),'w').write(html)
print(f'ĐÃ DỰNG vision_hd_pub.html ({len(html)/1e6:.2f}MB) · {len(samples)} ảnh mẫu {X.shape[1]}×{X.shape[2]}')
