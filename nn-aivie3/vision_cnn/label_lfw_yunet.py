#!/usr/bin/env python
# Dùng YuNet (OpenCV DNN) tự gán nhãn HỘP + 5 ĐIỂM-MỐC (2 mắt, mũi, 2 khoé miệng) lên ảnh LFW.
# Lưu nhãn (toạ-độ chuẩn-hoá theo 250px) → detlabels.npz. Chạy 1 lần (~vài chục giây).
import os, glob, numpy as np, cv2
ROOT=os.path.dirname(os.path.abspath(__file__))
LFW=os.path.join(ROOT,'..','..','data','lfw'); MODEL=os.path.join(ROOT,'..','..','models','yunet.onnx')
det=cv2.FaceDetectorYN.create(MODEL,'',(250,250),0.7)
paths=sorted(glob.glob(os.path.join(LFW,'*','*.jpg')))
np.random.RandomState(0).shuffle(paths); paths=paths[:9000]
out_p=[]; out_box=[]; out_lm=[]
for p in paths:
    im=cv2.imread(p)
    if im is None: continue
    H,W=im.shape[:2]; det.setInputSize((W,H)); _,f=det.detect(im)
    if f is None or not len(f): continue
    r=max(f,key=lambda a:a[2]*a[3])         # mặt lớn nhất
    x,y,w,h=r[:4]; lm=r[4:14].reshape(5,2)
    out_p.append(os.path.relpath(p,ROOT)); out_box.append([x/W,y/H,w/W,h/H]); out_lm.append((lm/[W,H]).reshape(-1))
np.savez(os.path.join(ROOT,'detlabels.npz'), paths=np.array(out_p), box=np.array(out_box,np.float32), lm=np.array(out_lm,np.float32))
print(f'ĐÃ GÁN NHÃN {len(out_p)}/{len(paths)} ảnh có mặt → detlabels.npz')
