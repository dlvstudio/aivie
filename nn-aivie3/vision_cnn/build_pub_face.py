#!/usr/bin/env python
# Nhúng facew.json + vài mặt mẫu (planar base64) vào face.html → face_pub.html
import os, json, base64, numpy as np
ROOT=os.path.dirname(os.path.abspath(__file__))
facew=open(os.path.join(ROOT,'facew.json')).read()
d=np.load(os.path.join(ROOT,'faces_sample.npz'),allow_pickle=True); X,y,names=d['X'],d['y'],d['names']
samples=[]
for i in range(len(X)):
    planar=np.ascontiguousarray(X[i].transpose(2,0,1)).reshape(-1).astype(np.uint8)
    nm=str(names[int(y[i])]).replace('_',' ')
    samples.append({'name':nm,'d':base64.b64encode(planar.tobytes()).decode()})
html=open(os.path.join(ROOT,'face.html')).read()
assert '/*__FACEW__*/ null' in html and '/*__SAMPLES__*/ null' in html
html=html.replace('/*__FACEW__*/ null','/*__FACEW__*/ '+facew).replace('/*__SAMPLES__*/ null','/*__SAMPLES__*/ '+json.dumps(samples))
open(os.path.join(ROOT,'face_pub.html'),'w').write(html)
print(f'ĐÃ DỰNG face_pub.html ({len(html)/1e6:.2f}MB) · {len(samples)} mặt mẫu / {len(set(s["name"] for s in samples))} người')
