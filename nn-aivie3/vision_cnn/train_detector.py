#!/usr/bin/env python
# Mạng DÒ-MẶT tự-viết: ảnh → HỘP (4) + 5 ĐIỂM-MỐC (10), hồi-quy thuần (chạy MPS).
# Nhãn từ YuNet (detlabels.npz). Tăng-cường: dán mặt vào vị-trí/kích-thước ngẫu-nhiên → học ĐỊNH-VỊ.
import os, sys, json, time, argparse, glob
import numpy as np, cv2, torch, torch.nn as nn, torch.nn.functional as F
ROOT=os.path.dirname(os.path.abspath(__file__)); S=128; SRC=160
BG=None                                          # ảnh nền thật (để dán mặt lên → giống webcam)
def load_bg():
    global BG
    ps=glob.glob(os.path.join(ROOT,'..','..','data','imagenette2-160','val','*','*.JPEG'))[:800]
    B=np.empty((len(ps),S,S,3),np.uint8)
    for i,p in enumerate(ps): B[i]=cv2.cvtColor(cv2.resize(cv2.imread(p),(S,S)),cv2.COLOR_BGR2RGB)
    BG=B; print('nền:',B.shape)

def load_sources():
    d=np.load(os.path.join(ROOT,'detlabels.npz'),allow_pickle=True)
    paths,box,lm=d['paths'],d['box'],d['lm']
    X=np.empty((len(paths),SRC,SRC,3),np.uint8)
    for i,p in enumerate(paths):
        im=cv2.imread(os.path.join(ROOT,str(p))); X[i]=cv2.cvtColor(cv2.resize(im,(SRC,SRC)),cv2.COLOR_BGR2RGB)
    box=box.astype(np.float32); lm=lm.astype(np.float32)
    wf=os.path.join(ROOT,'wider_faces.npz')
    if os.path.exists(wf):                                  # thêm mặt WIDER (đa-dạng: nghiêng, cận, ánh-sáng)
        w=np.load(wf); X=np.concatenate([X,w['X']]); box=np.concatenate([box,w['box']]); lm=np.concatenate([lm,w['lm']])
        print('  + WIDER',w['X'].shape[0],'mặt')
    return X,box,lm

def composite(img,box,lm,train):
    # img HWC uint8 (SRC); box[x,y,w,h] & lm[10] chuẩn-hoá [0,1]. Trả canvas SxS + nhãn chuẩn-hoá theo S.
    fcx=(box[0]+box[2]/2); fcy=(box[1]+box[3]/2)                 # tâm mặt (0..1 theo SRC)
    if train: tw=np.random.uniform(0.15,1.15); rx=np.random.uniform(0.15,0.85); ry=np.random.uniform(0.15,0.85)   # tw 0.15 = mặt XA/nhỏ … 1.15 = cận-cảnh lấp khung
    else:     tw=0.55; rx=0.5; ry=0.5
    s=(tw)/max(box[2],1e-3)                                      # tỉ-lệ để bề-ngang mặt = tw (theo S)
    s*= S/SRC                                                    # từ px-SRC sang px-S
    tx=rx*S - s*fcx*SRC; ty=ry*S - s*fcy*SRC
    M=np.array([[s,0,tx],[0,s,ty]],np.float32)
    # nền THẬT (ảnh phòng/vật) thay nền xám → giống webcam
    if train and BG is not None: base=BG[np.random.randint(len(BG))].copy()
    else: base=np.full((S,S,3),128,np.uint8)
    fw=cv2.warpAffine(img,M,(S,S),borderValue=(0,0,0))
    mask=cv2.warpAffine(np.full((SRC,SRC),255,np.uint8),M,(S,S))>0
    canvas=np.where(mask[...,None],fw,base)
    if train:                                                    # rung sáng/màu (ánh-sáng webcam)
        canvas=np.clip(canvas.astype(np.float32)*np.random.uniform(0.7,1.3)+np.random.uniform(-20,20),0,255).astype(np.uint8)
    def tf(px,py): return (s*px*SRC+tx)/S,(s*py*SRC+ty)/S
    bx,by=tf(box[0],box[1]); bw=s*box[2]*SRC/S; bh=s*box[3]*SRC/S
    L=lm.reshape(5,2).copy()
    for k in range(5): L[k,0],L[k,1]=tf(L[k,0],L[k,1])
    if train and np.random.rand()<0.5:                          # lật ngang
        canvas=canvas[:,::-1].copy(); bx=1-bx-bw
        L[:,0]=1-L[:,0]; L=L[[1,0,2,4,3]]                       # đổi mắt-P/T, miệng-P/T
    tb=np.array([bx,by,bw,bh],np.float32); tl=L.reshape(-1).astype(np.float32)
    return canvas, np.clip(tb,0,1), np.clip(tl,0,1)

class Det(nn.Module):
    def __init__(s,ch=(32,64,128,256)):                    # to hơn cho khít
        super().__init__(); c=[3]+list(ch); s.cv=nn.ModuleList(); s.bn=nn.ModuleList()
        for i in range(len(ch)): s.cv.append(nn.Conv2d(c[i],c[i+1],3,padding=1,bias=False)); s.bn.append(nn.BatchNorm2d(c[i+1]))
        s.box=nn.Linear(ch[-1],4); s.lm=nn.Linear(ch[-1],10)
    def feat(s,x):
        for cv,bn in zip(s.cv,s.bn): x=F.max_pool2d(F.relu(bn(cv(x))),2)
        return x.mean((2,3))
    def forward(s,x): f=s.feat(x); return torch.sigmoid(s.box(f)), torch.sigmoid(s.lm(f))

def fold_bn(conv,bn):
    W=conv.weight.detach().cpu().numpy(); g=bn.weight.detach().cpu().numpy(); be=bn.bias.detach().cpu().numpy()
    mu=bn.running_mean.detach().cpu().numpy(); var=bn.running_var.detach().cpu().numpy(); sc=g/np.sqrt(var+bn.eps)
    return (W*sc[:,None,None,None]).astype(np.float32),(be-mu*sc).astype(np.float32)

def batch(X,box,lm,idx,train):
    cv=np.empty((len(idx),3,S,S),np.float32); tb=np.empty((len(idx),4),np.float32); tl=np.empty((len(idx),10),np.float32)
    for j,i in enumerate(idx):
        c,b,l=composite(X[i],box[i],lm[i],train); cv[j]=c.transpose(2,0,1)/255.0; tb[j]=b; tl[j]=l
    return cv,tb,tl

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--epochs',type=int,default=50); ap.add_argument('--bs',type=int,default=128)
    ap.add_argument('--lr',type=float,default=0.05); ap.add_argument('--device',default='mps'); ap.add_argument('--out',default=os.path.join(ROOT,'detw.json'))
    a=ap.parse_args(); dev=torch.device(a.device if (torch.backends.mps.is_available() or a.device=='cpu') else 'cpu'); print('device',dev)
    load_bg(); X,box,lm=load_sources(); n=len(X); print('ảnh',X.shape)
    ne=n//10; ev=np.arange(ne); tr=np.arange(ne,n)
    net=Det().to(dev); opt=torch.optim.SGD(net.parameters(),lr=a.lr,momentum=0.9,weight_decay=5e-4,nesterov=True)
    sched=torch.optim.lr_scheduler.OneCycleLR(opt,max_lr=a.lr*3,epochs=a.epochs,steps_per_epoch=(len(tr)+a.bs-1)//a.bs)
    for ep in range(a.epochs):
        net.train(); t0=time.time(); np.random.shuffle(tr)
        for i in range(0,len(tr),a.bs):
            cv,tb,tl=batch(X,box,lm,tr[i:i+a.bs],True)
            xb=torch.tensor(cv,device=dev); bb=torch.tensor(tb,device=dev); ll=torch.tensor(tl,device=dev)
            pb,pl=net(xb); loss=F.mse_loss(pb,bb)+3*F.mse_loss(pl,ll)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if (ep+1)%6==0 or ep==a.epochs-1:
            net.eval(); errs=[]
            with torch.no_grad():
                for i in range(0,len(ev),256):
                    cv,tb,tl=batch(X,box,lm,ev[i:i+256],False)
                    _,pl=net(torch.tensor(cv,device=dev))
                    errs.append((pl.cpu().numpy()-tl).reshape(-1,5,2))
            e=np.concatenate(errs); mlme=float(np.sqrt((e**2).sum(-1)).mean())   # sai-số mốc trung-bình (theo cạnh ảnh)
            print(f'ep{ep+1:02d}/{a.epochs} sai-số-mốc {mlme*100:.1f}% cạnh ({time.time()-t0:.1f}s)',flush=True)
    convs=[]
    for cv,bn in zip(net.cv,net.bn):
        W,b=fold_bn(cv,bn); convs.append({'w':W.tolist(),'b':b.tolist()})
    def lin(l):return {'w':l.weight.detach().cpu().numpy().tolist(),'b':l.bias.detach().cpu().numpy().tolist()}
    out={'meta':{'arch':'det-gap','input_size':S,'mean':[0,0,0],'std':[1,1,1],'lm_err':round(mlme,4)},
         'convs':convs,'boxhead':lin(net.box),'lmhead':lin(net.lm)}
    def rnd(x):return [rnd(v) for v in x] if isinstance(x,list) else (round(x,5) if isinstance(x,float) else x)
    for c in out['convs']: c['w']=rnd(c['w']);c['b']=rnd(c['b'])
    for h in ['boxhead','lmhead']: out[h]['w']=rnd(out[h]['w']);out[h]['b']=rnd(out[h]['b'])
    json.dump(out,open(a.out,'w'),separators=(',',':')); print('ĐÃ XUẤT',a.out,f'({os.path.getsize(a.out)/1e6:.2f}MB)',flush=True)

if __name__=='__main__': main()
