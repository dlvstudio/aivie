#!/usr/bin/env python
# KIỂM mfw.json bằng numpy (khớp detectMulti JS): dựng vài ảnh ghép 1-3 mặt → dò → so đỉnh với nhãn.
import os, json, numpy as np
from train_multiface import composite_multi, S, FS
from train_detector import load_sources, load_bg
import train_detector as td
ROOT=os.path.dirname(os.path.abspath(__file__))

def relu(x): return np.maximum(x,0)
def conv3(x,w,b):                       # x:(C,H,W) w:(O,C,3,3) pad1
    C,H,W=x.shape; O=w.shape[0]; xp=np.pad(x,((0,0),(1,1),(1,1)))
    out=np.zeros((O,H,W),np.float32)
    for ky in range(3):
        for kx in range(3):
            patch=xp[:,ky:ky+H,kx:kx+W]                       # (C,H,W)
            out+=np.tensordot(w[:,:,ky,kx],patch,axes=([1],[0]))
    return out+b[:,None,None]
def pool2(x):
    C,H,W=x.shape; return x[:,:H//2*2,:W//2*2].reshape(C,H//2,2,W//2,2).max(axis=(2,4))

def forward(mf,img,thr=0.30):           # img:(H,W,3) uint8 → faces
    x=img.transpose(2,0,1).astype(np.float32)/255.0
    e=mf['enc']
    x=pool2(relu(conv3(x,np.array(e[0]['w'],np.float32),np.array(e[0]['b'],np.float32))))
    x=pool2(relu(conv3(x,np.array(e[1]['w'],np.float32),np.array(e[1]['b'],np.float32))))
    x=relu(conv3(x,np.array(e[2]['w'],np.float32),np.array(e[2]['b'],np.float32)))
    hm=conv3(x,np.array(mf['hm']['w'],np.float32),np.array(mf['hm']['b'],np.float32))[0]
    wh=conv3(x,np.array(mf['wh']['w'],np.float32),np.array(mf['wh']['b'],np.float32))
    lm=conv3(x,np.array(mf['lm']['w'],np.float32),np.array(mf['lm']['b'],np.float32))
    p=1/(1+np.exp(-hm)); faces=[]
    for y in range(FS):
        for xx in range(FS):
            v=p[y,xx]
            if v<thr: continue
            y0,y1,x0,x1=max(0,y-1),min(FS,y+2),max(0,xx-1),min(FS,xx+2)
            if v<p[y0:y1,x0:x1].max()-1e-9: continue
            cw,ch=wh[0,y,xx],wh[1,y,xx]; cx=(xx+0.5)/FS; cy=(y+0.5)/FS
            faces.append((v,[cx-cw/2,cy-ch/2,cw,ch]))
    faces.sort(key=lambda t:-t[0]); keep=[]
    def iou(a,b):
        ax1,ay1,bx1,by1=a[0]+a[2],a[1]+a[3],b[0]+b[2],b[1]+b[3]
        iw=max(0,min(ax1,bx1)-max(a[0],b[0])); ih=max(0,min(ay1,by1)-max(a[1],b[1])); inter=iw*ih
        return inter/(a[2]*a[3]+b[2]*b[3]-inter+1e-9)
    for v,bx in faces:
        if all(iou(k,bx)<0.3 for k in keep): keep.append(bx)
    return keep

def main():
    mf=json.load(open(os.path.join(ROOT,'mfw.json')))
    print('meta',mf['meta'])
    load_bg(); BG=td.BG; X,box,lm=load_sources()
    def iou(a,b):
        ax1,ay1,bx1,by1=a[0]+a[2],a[1]+a[3],b[0]+b[2],b[1]+b[3]
        iw=max(0,min(ax1,bx1)-max(a[0],b[0])); ih=max(0,min(ay1,by1)-max(a[1],b[1])); inter=iw*ih
        return inter/(a[2]*a[3]+b[2]*b[3]-inter+1e-9)
    imgs=[composite_multi(X,box,lm,BG,True) for _ in range(80)]
    for thr in (0.30,0.40,0.50,0.60,0.70):
        tp=0;fp=0;gt=0
        for img,fs in imgs:
            pred=forward(mf,img,thr); gtb=[f[0] for f in fs]; gt+=len(gtb); used=[False]*len(pred)
            for g in gtb:
                best=-1;bi=-1
                for i,p in enumerate(pred):
                    if used[i]:continue
                    v=iou(g,p)
                    if v>best:best=v;bi=i
                if best>=0.4 and bi>=0: used[bi]=True; tp+=1
            fp+=used.count(False)
        print(f'thr {thr:.2f} · GT {gt} · TP {tp} FP {fp} · recall {tp/max(1,gt):.2f} precision {tp/max(1,tp+fp):.2f}')

if __name__=='__main__': main()
