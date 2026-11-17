#!/usr/bin/env python
# Gộp: nhúng cnnw_hd.json (vật) + facew.json (mặt) + mẫu vật + mẫu mặt vào brain.html → brain_pub.html
import os, json, base64, numpy as np
ROOT=os.path.dirname(os.path.abspath(__file__))
objw=open(os.path.join(ROOT,'cnnw_hd.json')).read()
facew=open(os.path.join(ROOT,'facew.json')).read()
detw=open(os.path.join(ROOT,'detw.json')).read()
mfw=open(os.path.join(ROOT,'mfw.json')).read() if os.path.exists(os.path.join(ROOT,'mfw.json')) else 'null'
celeb=open(os.path.join(ROOT,'celeb_book.json')).read() if os.path.exists(os.path.join(ROOT,'celeb_book.json')) else 'null'
objmeta=json.loads(objw)['meta']; VIO=objmeta['classes_vi']

def planar_b64(hwc):
    return base64.b64encode(np.ascontiguousarray(hwc.transpose(2,0,1)).reshape(-1).astype(np.uint8).tobytes()).decode()

# mẫu VẬT (không phải người): 1 ảnh cho vài lớp
d=np.load(os.path.join(ROOT,'samples_hd.npz')); Xo,yo=d['X'],d['y']; So=Xo.shape[1]
objsamp=[]; seen=set()
for i in range(len(Xo)):
    c=int(yo[i])
    if c==10 or c in seen: continue           # bỏ 'người' (đã có mẫu mặt), 1 ảnh/lớp
    seen.add(c); objsamp.append({'label':VIO[c],'S':int(So),'d':planar_b64(Xo[i])})
    if len(objsamp)>=6: break

# mẫu MẶT: nạp LẠI từ LFW ở 160px FULL-FRAME (khớp cách mạng-vật học "người" — ảnh đủ, không cắt sát)
from PIL import Image
LFW=os.path.join(ROOT,'..','..','data','lfw')
f=np.load(os.path.join(ROOT,'faces_sample.npz'),allow_pickle=True); names=f['names']
FS=160; facesamp=[]                                  # full-frame 160 (khớp mạng-vật; mạng-mặt tự crop sát khi prep)
for pid in [str(n) for n in names]:
    pd=os.path.join(LFW,pid)
    if not os.path.isdir(pd): continue
    for fn in sorted(os.listdir(pd))[:2]:
        im=Image.open(os.path.join(pd,fn)).convert('RGB').resize((FS,FS),Image.BILINEAR)
        facesamp.append({'name':pid.replace('_',' '),'S':FS,'d':planar_b64(np.asarray(im,np.uint8))})

html=open(os.path.join(ROOT,'brain.html')).read()
for ph,val in [('/*__OBJW__*/ null','/*__OBJW__*/ '+objw),('/*__FACEW__*/ null','/*__FACEW__*/ '+facew),('/*__DETW__*/ null','/*__DETW__*/ '+detw),('/*__MFW__*/ null','/*__MFW__*/ '+mfw),('/*__CELEBS__*/ null','/*__CELEBS__*/ '+celeb),
               ('/*__OBJSAMP__*/ null','/*__OBJSAMP__*/ '+json.dumps(objsamp)),('/*__FACESAMP__*/ null','/*__FACESAMP__*/ '+json.dumps(facesamp))]:
    assert ph in html, 'thiếu '+ph; html=html.replace(ph,val)
open(os.path.join(ROOT,'brain_pub.html'),'w').write(html)
print(f'ĐÃ DỰNG brain_pub.html ({len(html)/1e6:.2f}MB) · {len(objsamp)} mẫu vật + {len(facesamp)} mẫu mặt / {len(set(s["name"] for s in facesamp))} người')
