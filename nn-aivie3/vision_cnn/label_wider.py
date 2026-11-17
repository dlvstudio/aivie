#!/usr/bin/env python
# YuNet gán nhãn WIDER_val: mỗi MẶT rõ → cắt ô vuông quanh mặt (có ngữ-cảnh) làm 1 mẫu-1-mặt + nhãn hộp/5-mốc.
# Lưu wider_faces.npz {X:(N,160,160,3), box:(N,4), lm:(N,10)}. Bổ-sung độ ĐA-DẠNG cho detector (khác LFW toàn frontal).
import os, glob, numpy as np, cv2
ROOT=os.path.dirname(os.path.abspath(__file__)); SRC=160
WID=os.path.join(ROOT,'..','..','data','WIDER_val','images'); MODEL=os.path.join(ROOT,'..','..','models','yunet.onnx')
det=cv2.FaceDetectorYN.create(MODEL,'',(320,320),0.85)   # ngưỡng cao → chỉ mặt rõ
paths=glob.glob(os.path.join(WID,'*','*.jpg')); rng=np.random.RandomState(0); rng.shuffle(paths)
X=[]; B=[]; L=[]; MAXP=4000
for p in paths:
    if len(X)>=12000: break
    im=cv2.imread(p)
    if im is None: continue
    H,W=im.shape[:2]; det.setInputSize((W,H)); _,f=det.detect(im)
    if f is None: continue
    for r in f:
        x,y,w,h=r[:4]; sc=r[14]
        if sc<0.9 or w<40 or h<40: continue
        cxp,cyp=x+w/2,y+h/2; side=max(w,h)*rng.uniform(1.4,3.0)     # ô ngữ-cảnh quanh mặt
        x0=int(cxp-side/2); y0=int(cyp-side/2); s=int(side)
        if x0<0 or y0<0 or x0+s>W or y0+s>H: continue               # bỏ ô tràn biên (giữ nhãn sạch)
        crop=cv2.cvtColor(cv2.resize(im[y0:y0+s,x0:x0+s],(SRC,SRC)),cv2.COLOR_BGR2RGB)
        bx=(x-x0)/s; by=(y-y0)/s; bw=w/s; bh=h/s
        lm=((r[4:14].reshape(5,2)-[x0,y0])/s).reshape(-1)
        if bx<0 or by<0 or bx+bw>1 or by+bh>1: continue
        X.append(crop); B.append([bx,by,bw,bh]); L.append(lm)
        if len(X)>=12000: break
np.savez(os.path.join(ROOT,'wider_faces.npz'), X=np.array(X,np.uint8), box=np.array(B,np.float32), lm=np.array(L,np.float32))
print(f'ĐÃ GÁN NHÃN {len(X)} mặt WIDER → wider_faces.npz')
