#!/usr/bin/env python
# Nhúng cnnw.json + vài ảnh mẫu (base64 planar RGB) vào vision.html → vision_pub.html
import os, json, base64, numpy as np
ROOT = os.path.dirname(os.path.abspath(__file__))
cnnw = open(os.path.join(ROOT,'cnnw.json')).read()
d = np.load(os.path.join(ROOT,'samples.npz'))
X, y = d['X'], d['y']                       # X:(N,3,32,32) float 0..1
PER = 2                                      # ảnh mẫu mỗi lớp
ncls = int(y.max())+1
picks = []
for c in range(ncls):
    ids = np.where(y==c)[0][:PER]
    for i in ids: picks.append(int(i))
samples = []
for i in picks:
    planar = (np.clip(X[i],0,1)*255).astype(np.uint8).reshape(-1)   # c*1024+h*32+w
    samples.append({'y': int(y[i]), 'd': base64.b64encode(planar.tobytes()).decode()})
html = open(os.path.join(ROOT,'vision.html')).read()
assert '/*__CNNW__*/ null' in html and '/*__SAMPLES__*/ null' in html, 'thiếu placeholder'
html = html.replace('/*__CNNW__*/ null', '/*__CNNW__*/ '+cnnw)
html = html.replace('/*__SAMPLES__*/ null', '/*__SAMPLES__*/ '+json.dumps(samples))
open(os.path.join(ROOT,'vision_pub.html'),'w').write(html)
print(f'ĐÃ DỰNG vision_pub.html ({len(html)/1e6:.2f}MB) · {len(samples)} ảnh mẫu')
