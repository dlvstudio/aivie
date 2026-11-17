#!/usr/bin/env python
# Thử 3 mạng trên ảnh người NỔI-TIẾNG VN (Sơn Tùng, Hoàng Thùy Linh) — kiểm dò-mặt + nhận-diện one-shot chéo.
# Trung-thực: embedder luyện LFW (đa số mặt phương-Tây) → xem có tổng-quát sang mặt Việt không.
import os, json, glob, numpy as np, cv2
R=os.path.dirname(os.path.abspath(__file__))
OBJ=json.load(open(f'{R}/cnnw_hd.json')); FACE=json.load(open(f'{R}/facew.json')); DET=json.load(open(f'{R}/detw.json'))
FACE_CF=FACE['meta']['crop']/250.0

def conv(inp,w,b,relu=True):
    Ci,H,W=inp.shape;Co=len(w);P=np.pad(inp,((0,0),(1,1),(1,1)));o=np.zeros((Co,H,W),np.float32);w=np.array(w,np.float32);b=np.array(b,np.float32)
    for ky in range(3):
        for kx in range(3): o+=np.tensordot(w[:,:,ky,kx],P[:,ky:ky+H,kx:kx+W],axes=([1],[0]))
    o+=b[:,None,None]; return np.maximum(o,0) if relu else o
def pool(a):C,H,W=a.shape;return a.reshape(C,H//2,2,W//2,2).max((2,4))
def prep(imgRGB,S,cf=1.0):                      # center-crop (cf) → SxS float CHW [0,1] (mean0/std tuỳ mạng ngoài)
    H,W=imgRGB.shape[:2]; s=int(min(H,W)*cf); y0=(H-s)//2; x0=(W-s)//2
    c=cv2.resize(imgRGB[y0:y0+s,x0:x0+s],(S,S)); return c
def norm(cHWC,meta):
    x=cHWC.astype(np.float32).transpose(2,0,1)/255.0; m=np.array(meta['mean']); st=np.array(meta['std'])
    return (x-m[:,None,None])/st[:,None,None]
def obj_pred(imgRGB):
    x=norm(prep(imgRGB,OBJ['meta']['input_size']),OBJ['meta']); h=x
    for c in OBJ['convs']: h=pool(conv(h,c['w'],c['b']))
    g=h.mean((1,2)); lo=np.array(OBJ['fc']['w'])@g+np.array(OBJ['fc']['b']); e=np.exp(lo-lo.max()); p=e/e.sum()
    return OBJ['meta']['classes_vi'][int(p.argmax())], float(p.max()), float(p[OBJ['meta']['classes_en'].index('person')])
def detect(canvasRGB):                          # canvas SxS RGB → box,lm [0,1]
    S=DET['meta']['input_size']; x=canvasRGB.astype(np.float32).transpose(2,0,1)/255.0; h=x
    for enc in DET['enc']: h=pool(conv(h,enc['w'],enc['b']))
    gap=h.mean((1,2)); box=1/(1+np.exp(-(np.array(DET['boxfc']['w'])@gap+np.array(DET['boxfc']['b']))))
    up=np.repeat(np.repeat(h,2,1),2,2); d=conv(up,DET['dec']['w'],DET['dec']['b']); hm=conv(d,DET['head']['w'],DET['head']['b'],relu=False)
    H2=hm.shape[1]; lm=[]
    for k in range(5):
        z=hm[k]-hm[k].max(); e=np.exp(z); e/=e.sum(); yy,xx=np.mgrid[0:H2,0:H2]
        lm+= [float((e*xx).sum()/(H2-1)), float((e*yy).sum()/(H2-1))]
    return box,lm
def detect2(imgRGB):                                      # dò 2 TẦNG (như demo): thô → zoom → tinh
    S=DET['meta']['input_size']; box,lm=detect(prep(imgRGB,S))
    cx=box[0]+box[2]/2; cy=box[1]+box[3]/2; cs=min(max(max(box[2],box[3])*1.8,0.05),1.0)
    cx0=min(max(cx-cs/2,0),1-cs); cy0=min(max(cy-cs/2,0),1-cs)
    H,W=imgRGB.shape[:2]; sq=min(H,W); ox=(W-sq)//2; oy=(H-sq)//2
    px=int(ox+cx0*sq); py=int(oy+cy0*sq); ps=max(1,int(cs*sq))
    crop=cv2.resize(imgRGB[py:py+ps,px:px+ps],(S,S)); _,l2=detect(crop)
    return [ (cx0+l2[k]*cs) if k%2==0 else (cy0+l2[k]*cs) for k in range(10) ]
def face_emb(imgRGB, adaptive=False):
    lm=detect2(imgRGB)
    Lx=lm[0::2]; Ly=lm[1::2]; cx=sum(Lx)/5; cy=sum(Ly)/5   # tâm mốc
    H,W=imgRGB.shape[:2]; s=min(H,W); y0=(H-s)//2; x0=(W-s)//2      # ô-giữa trên ảnh gốc
    if adaptive:
        lw=max(Lx)-min(Lx); lh=max(Ly)-min(Ly); side=min(max(2.2*max(lw,lh,1e-3),0.2),0.5)   # cỡ theo mặt (kẹp)
    else: side=FACE_CF
    nx=min(max(cx-side/2,0),1-side); ny=min(max(cy-side/2,0),1-side)
    fx=int(x0+nx*s); fy=int(y0+ny*s); fs=int(side*s)
    crop=cv2.resize(imgRGB[fy:fy+fs,fx:fx+fs],(FACE['meta']['input_size'],)*2)
    x=norm(crop,FACE['meta']); h=x
    for c in FACE['convs']: h=pool(conv(h,c['w'],c['b']))
    g=h.mean((1,2)); return g/(np.linalg.norm(g)+1e-9), lm, crop

names={'sontung':['sontung_a.png','sontung_b.png','sontung_c.jpg'],'htl':['htl_a.jpg','htl_b.jpg','htl_c.jpg'],
       'hhen':['hhen_a.png','hhen_b.png'],'chipu':['chipu_a.png','chipu_b.png'],'dongnhi':['dongnhi_a.jpg']}
VN={'sontung':'Sơn Tùng','htl':'Hoàng Thùy Linh','hhen':"H'Hen Niê",'chipu':'Chi Pu','dongnhi':'Đông Nhi'}
emb={}
print('=== DÒ-MẶT + VẬT-THỂ ===')
for who,fs in names.items():
    for fn in fs:
        im=cv2.cvtColor(cv2.imread(f'{R}/celeb_test/{fn}'),cv2.COLOR_BGR2RGB)
        v,lm,crop=face_emb(im); emb[fn]=(who,v,crop)
        oc,op,pp=obj_pred(im)
        print(f'  {fn:16s} mắt(0.{int(lm[0]*100):02d},0.{int(lm[1]*100):02d}) mũi(0.{int(lm[4]*100):02d},0.{int(lm[5]*100):02d}) | vật-thể={oc}({op*100:.0f}%) P(người)={pp*100:.0f}%')

print('\n=== ONE-SHOT: ghi-danh ảnh _a, nhận ảnh _b,_c ===')
gal={who:emb[fs[0]][1] for who,fs in names.items()}
cor=tot=0
for who,fs in names.items():
    for fn in fs[1:]:
        v=emb[fn][1]; sims={w:float(gal[w]@v) for w in gal}; best=max(sims,key=sims.get)
        ok=best==who; cor+=ok; tot+=1
        print(f'  {fn:16s} → đoán "{VN[best]}" ({sims[best]*100:.0f}%)  [ST {sims["sontung"]*100:.0f}% · HTL {sims["htl"]*100:.0f}%]  {"✓" if ok else "✗ (thật:"+VN[who]+")"}')
print(f'\nĐúng {cor}/{tot} · gallery {len(names)} người.')

# ===== XUẤT SỔ MẶT CHUNG (nhúng vào demo) =====
import base64
book=[]
for who,fs in names.items():
    embs=[emb[fn][1].round(4).tolist() for fn in fs]                 # nhiều ảnh/người
    thumb=np.ascontiguousarray(emb[fs[0]][2].transpose(2,0,1)).reshape(-1).astype(np.uint8)  # crop 96 planar
    book.append({'name':VN[who],'embs':embs,'thumb':base64.b64encode(thumb.tobytes()).decode(),'S':FACE['meta']['input_size']})
json.dump(book,open(f'{R}/celeb_book.json','w'),separators=(',',':'))
print(f'ĐÃ XUẤT celeb_book.json — {len(book)} người nổi-tiếng ({sum(len(b["embs"]) for b in book)} ảnh)')

# ===== SO-SÁNH: cắt cố-định vs cắt-theo-cỡ-mặt · ảnh thường vs ảnh XA =====
def imgf(fn): return cv2.cvtColor(cv2.imread(f'{R}/celeb_test/{fn}'),cv2.COLOR_BGR2RGB)
def far(im,k=2.4):
    H,W=im.shape[:2]; Sz=int(max(H,W)*k); c=np.full((Sz,Sz,3),120,np.uint8); y0=(Sz-H)//2; x0=(Sz-W)//2; c[y0:y0+H,x0:x0+W]=im; return c
def run(adaptive,farq):
    gal={who:face_emb(imgf(fs[0]),adaptive)[0] for who,fs in names.items()}; cor=tot=0
    for who,fs in names.items():
        for fn in fs[1:]:
            im=far(imgf(fn)) if farq else imgf(fn); v=face_emb(im,adaptive)[0]
            best=max(gal,key=lambda w:float(gal[w]@v)); cor+=best==who; tot+=1
    return f'{cor}/{tot}'
print('\n=== cắt CỐ-ĐỊNH vs cắt THEO-CỠ-MẶT ===')
print(f'  ảnh thường: cố-định {run(False,False)} · theo-cỡ {run(True,False)}')
print(f'  ảnh XA    : cố-định {run(False,True)} · theo-cỡ {run(True,True)}')
