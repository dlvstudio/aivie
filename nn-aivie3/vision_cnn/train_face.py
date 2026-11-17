#!/usr/bin/env python
# "Vân-tay khuôn-mặt" v2: CNN map mặt → vector (chuẩn-hoá L2 lúc SUY-LUẬN).
# LUYỆN kiểu classifier thường (conv→GAP→Linear→CE) — GIỐNG object-model nên CHẠY MPS (né bug metric-learning).
# Chuẩn-hoá + cosine chỉ ở lúc eval/suy-luận (không backward). Test ONE-SHOT trên người CHƯA thấy.
import os, sys, json, time, argparse, random
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from PIL import Image

ROOT=os.path.dirname(os.path.abspath(__file__)); LFW=os.path.join(ROOT,'..','..','data','lfw')
S=96; CROP=110; OFF=(250-CROP)//2; CH=(48,96,160,256); EMB=CH[-1]

def gather(min_imgs):
    d={}
    for p in os.listdir(LFW):
        pd=os.path.join(LFW,p)
        if not os.path.isdir(pd): continue
        fs=[os.path.join(pd,f) for f in os.listdir(pd) if f.lower().endswith('.jpg')]
        if len(fs)>=min_imgs: d[p]=sorted(fs)
    return d

def loadimg(path):
    im=Image.open(path).convert('RGB').crop((OFF,OFF,OFF+CROP,OFF+CROP)).resize((S,S),Image.BILINEAR)
    return np.asarray(im,np.uint8).transpose(2,0,1)

def load_set(id2,ids):
    X=[];y=[]
    for lab,pid in enumerate(ids):
        for p in id2[pid]: X.append(loadimg(p)); y.append(lab)
    return np.stack(X), np.array(y,np.int64)

class Emb(nn.Module):
    def __init__(s, ch=CH):
        super().__init__(); c=[3]+list(ch); s.cv=nn.ModuleList(); s.bn=nn.ModuleList()
        for i in range(len(ch)): s.cv.append(nn.Conv2d(c[i],c[i+1],3,padding=1,bias=False)); s.bn.append(nn.BatchNorm2d(c[i+1]))
    def features(s,x):
        for cv,bn in zip(s.cv,s.bn): x=F.max_pool2d(F.relu(bn(cv(x))),2)
        return x.mean((2,3))                       # GAP (KHÔNG chuẩn-hoá — dùng lúc TRAIN)
    def forward(s,x):                              # embedding chuẩn-hoá (dùng lúc EVAL/suy-luận)
        f=s.features(x); return f/(f*f).sum(1,keepdim=True).add(1e-9).sqrt()

def fold_bn(conv,bn):
    W=conv.weight.detach().cpu().numpy(); g=bn.weight.detach().cpu().numpy(); be=bn.bias.detach().cpu().numpy()
    mu=bn.running_mean.detach().cpu().numpy(); var=bn.running_var.detach().cpu().numpy(); sc=g/np.sqrt(var+bn.eps)
    return (W*sc[:,None,None,None]).astype(np.float32),(be-mu*sc).astype(np.float32)

def crop_flip(X,train):
    n=len(X); out=np.empty_like(X); P=np.pad(X,((0,0),(0,0),(6,6),(6,6)),mode='reflect')
    if train: ox=np.random.randint(0,13,n); oy=np.random.randint(0,13,n); fl=np.random.rand(n)<0.5
    else: ox=np.full(n,6); oy=np.full(n,6); fl=np.zeros(n,bool)
    for i in range(n):
        c=P[i,:,oy[i]:oy[i]+S,ox[i]:ox[i]+S]; out[i]=np.ascontiguousarray(c[:,:,::-1]) if fl[i] else c
    return out

def embed_all(net,X,m3,s3,dev):
    net.eval(); outs=[]
    with torch.no_grad():
        for i in range(0,len(X),256):
            xb=torch.tensor((crop_flip(X[i:i+256],False)/255.0-m3)/s3,dtype=torch.float32,device=dev)
            outs.append(net(xb).cpu().numpy())
    return np.concatenate(outs).astype(np.float64)

def evaluate(net,Xe,ye,m3,s3,dev):
    E=embed_all(net,Xe,m3,s3,dev); rng=np.random.RandomState(0); idx_by=[np.where(ye==c)[0] for c in np.unique(ye)]
    pos=[];neg=[]
    for _ in range(4000):
        c=rng.randint(len(idx_by))
        if len(idx_by[c])<2: continue
        a,b=rng.choice(idx_by[c],2,replace=False); pos.append(float(E[a]@E[b]))
        c2=rng.randint(len(idx_by))
        if c2==c: continue
        neg.append(float(E[rng.choice(idx_by[c])]@E[rng.choice(idx_by[c2])]))
    pos=np.array(pos); neg=np.array(neg); best=0; bth=0
    for t in np.linspace(-0.2,0.95,140):
        acc=((pos>t).mean()+(neg<=t).mean())/2
        if acc>best: best=acc; bth=t
    gal=[];gy=[];q=[];qy=[]
    for c in np.unique(ye):
        ids=np.where(ye==c)[0]; gal.append(E[ids[0]]); gy.append(c)
        for j in ids[1:]: q.append(E[j]); qy.append(c)
    gal=np.stack(gal); gy=np.array(gy); q=np.stack(q); qy=np.array(qy)
    one=(gy[(q@gal.T).argmax(1)]==qy).mean()
    return best,bth,one,len(np.unique(ye))

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--epochs',type=int,default=45); ap.add_argument('--bs',type=int,default=128)
    ap.add_argument('--lr',type=float,default=0.05); ap.add_argument('--min_imgs',type=int,default=3)
    ap.add_argument('--margin',type=float,default=0.2); ap.add_argument('--scale',type=float,default=30)
    ap.add_argument('--device',default='cpu'); ap.add_argument('--out',default=os.path.join(ROOT,'facew.json'))
    a=ap.parse_args(); dev=torch.device(a.device if (torch.backends.mps.is_available() or a.device=='cpu') else 'cpu'); print('device',dev,'| CH',CH,'crop',CROP)

    id2=gather(a.min_imgs); ids=sorted(id2); random.Random(1).shuffle(ids)
    n_eval=80; eval_ids=ids[:n_eval]; train_ids=ids[n_eval:]
    print(f'người: train {len(train_ids)}  eval(CHƯA thấy) {len(eval_ids)}  (>= {a.min_imgs} ảnh)')
    Xtr,ytr=load_set(id2,train_ids); Xte,yte=load_set(id2,eval_ids); print('train',Xtr.shape,'eval',Xte.shape)
    mean=(Xtr/255.0).mean((0,2,3)).astype(np.float32); std=(Xtr/255.0).std((0,2,3)).astype(np.float32)
    m3=mean[None,:,None,None]; s3=std[None,:,None,None]; N=len(train_ids)

    net=Emb().to(dev); Wc=nn.Parameter(torch.randn(N,EMB,device=dev)*0.01)   # tâm-lớp (ArcFace)
    opt=torch.optim.SGD(list(net.parameters())+[Wc],lr=a.lr,momentum=0.9,weight_decay=5e-4,nesterov=True)
    n=len(Xtr); sched=torch.optim.lr_scheduler.OneCycleLR(opt,max_lr=a.lr*1.5,epochs=a.epochs,steps_per_epoch=(n+a.bs-1)//a.bs)
    warm=max(1,int(0.4*a.epochs))
    for ep in range(a.epochs):
        net.train(); t0=time.time(); idx=np.random.permutation(n); Xa=crop_flip(Xtr[idx],True); ya=ytr[idx]
        mg=a.margin*min(1.0,(ep+1)/warm)                                  # HÂM margin 0→0.2 (chống sụp từ-đầu)
        for i in range(0,n,a.bs):
            xb=torch.tensor((Xa[i:i+a.bs]/255.0-m3)/s3,dtype=torch.float32,device=dev); yb=torch.tensor(ya[i:i+a.bs],device=dev)
            emb=net(xb)                                                    # đã chuẩn-hoá
            Wn=Wc/(Wc*Wc).sum(1,keepdim=True).add(1e-9).sqrt()
            cos=torch.mm(emb,Wn.t()).clamp(-1+1e-6,1-1e-6)
            tgt=torch.cos(torch.acos(cos)+mg)                             # ArcFace: cộng biên góc vào lớp ĐÚNG
            oh=F.one_hot(yb,N).float(); logits=a.scale*(oh*tgt+(1-oh)*cos)
            opt.zero_grad(); F.cross_entropy(logits,yb).backward(); opt.step(); sched.step()
        if (ep+1)%9==0 or ep==a.epochs-1:
            v,th,one,ne=evaluate(net,Xte,yte,m3,s3,dev); print(f'ep{ep+1:02d}/{a.epochs} verify {v:.3f}@th{th:.2f} · 1-shot {one:.3f}/{ne} lạ ({time.time()-t0:.1f}s)',flush=True)

    v,th,one,ne=evaluate(net,Xte,yte,m3,s3,dev)
    print(f'== KẾT-QUẢ (người CHƯA thấy): verify {v:.3f}@ngưỡng {th:.2f} · one-shot {one:.3f}/{ne} người ==',flush=True)
    convs=[]
    for cv,bn in zip(net.cv,net.bn):
        W,b=fold_bn(cv,bn); convs.append({'w':W.tolist(),'b':b.tolist()})
    out={'meta':{'arch':'face-emb-gap','input_size':S,'crop':CROP,'emb_dim':EMB,'mean':mean.tolist(),'std':std.tolist(),
                 'verif_acc':round(float(v),4),'oneshot_acc':round(float(one),4),'threshold':round(float(th),3),'eval_people':ne},
         'convs':convs}
    def rnd(x): return [rnd(v) for v in x] if isinstance(x,list) else (round(x,5) if isinstance(x,float) else x)
    for cv in out['convs']: cv['w']=rnd(cv['w']); cv['b']=rnd(cv['b'])
    json.dump(out,open(a.out,'w'),separators=(',',':')); print('ĐÃ XUẤT',a.out,f'({os.path.getsize(a.out)/1e6:.2f}MB)',flush=True)
    sids=eval_ids[:8]; Xs=[];ys=[]
    for lab,pid in enumerate(sids):
        for p in id2[pid][:3]: Xs.append(loadimg(p).transpose(1,2,0)); ys.append(lab)
    np.savez(os.path.join(ROOT,'faces_sample.npz'),X=np.stack(Xs).astype(np.uint8),y=np.array(ys),names=np.array(sids))
    print('ĐÃ LƯU faces_sample.npz',len(Xs),'ảnh /',len(sids),'người',flush=True)

if __name__=='__main__': main()
