#!/usr/bin/env python
# CNN từ-đầu trên Imagenette (10 lớp ImageNet, ảnh NÉT ~160px) — input 96×96, 4 conv + GAP.
# Nét thật hơn CIFAR 32×32. Xuất trọng-số (BN gập) cho forward JS. Train PyTorch/MPS trên M4.
import os, sys, json, time, argparse
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from PIL import Image

ROOT = os.path.dirname(os.path.abspath(__file__))
IMG  = os.path.join(ROOT, '..', '..', 'data', 'imagenette2-160')
S    = 96          # kích-thước làm-việc
LOAD = 108         # nạp hơi to hơn để random-crop
WNIDS = ['n01440764','n02102040','n02979186','n03000684','n03028079',
         'n03394916','n03417042','n03425413','n03445777','n03888257']
EN = ['tench','English springer','cassette player','chain saw','church',
      'French horn','garbage truck','gas pump','golf ball','parachute']
VI = ['cá tench','chó springer','máy cát-sét','cưa máy','nhà thờ',
      'kèn Pháp','xe rác','trụ bơm xăng','bóng gôn','dù']

def load_split(split):
    cache = os.path.join(ROOT, f'imnet_{split}_{LOAD}.npz')
    if os.path.exists(cache):
        d = np.load(cache); return d['X'], d['y']
    base = os.path.join(IMG, split)
    if not os.path.isdir(base): sys.exit(f'CHƯA CÓ Imagenette: {base}')
    X=[]; y=[]
    for ci,wnid in enumerate(WNIDS):
        dd = os.path.join(base, wnid)
        for fn in os.listdir(dd):
            if not fn.lower().endswith(('.jpeg','.jpg','.png')): continue
            im = Image.open(os.path.join(dd,fn)).convert('RGB').resize((LOAD,LOAD), Image.BILINEAR)
            X.append(np.asarray(im,dtype=np.uint8)); y.append(ci)
    X = np.stack(X).transpose(0,3,1,2)                 # (N,3,LOAD,LOAD) uint8 CHW
    y = np.array(y, np.int64)
    np.savez(cache, X=X, y=y); print(f'{split}: {X.shape} (đã cache)')
    return X, y

def crop_flip(X, train):                                # (N,3,LOAD,LOAD) uint8 → (N,3,S,S) uint8
    n=len(X); out=np.empty((n,3,S,S), np.uint8); m=LOAD-S
    if train:
        ox=np.random.randint(0,m+1,n); oy=np.random.randint(0,m+1,n); fl=np.random.rand(n)<0.5
    else:
        ox=np.full(n,m//2); oy=np.full(n,m//2); fl=np.zeros(n,bool)
    for i in range(n):
        c=X[i,:,oy[i]:oy[i]+S, ox[i]:ox[i]+S]
        out[i]=c[:,:,::-1] if fl[i] else c
    return out

class CNN(nn.Module):
    def __init__(s, ncls=10, ch=(24,48,96,128)):
        super().__init__()
        c=[3]+list(ch); s.convs=nn.ModuleList(); s.bns=nn.ModuleList()
        for i in range(len(ch)):
            s.convs.append(nn.Conv2d(c[i],c[i+1],3,padding=1,bias=False)); s.bns.append(nn.BatchNorm2d(c[i+1]))
        s.fc=nn.Linear(ch[-1], ncls)
    def forward(s,x):
        for cv,bn in zip(s.convs,s.bns): x=F.max_pool2d(F.relu(bn(cv(x))),2)
        x=x.mean((2,3))                                 # global average pool
        return s.fc(x)

def fold_bn(conv, bn):
    W=conv.weight.detach().cpu().numpy(); g=bn.weight.detach().cpu().numpy(); beta=bn.bias.detach().cpu().numpy()
    mean=bn.running_mean.detach().cpu().numpy(); var=bn.running_var.detach().cpu().numpy()
    sc=g/np.sqrt(var+bn.eps); return (W*sc[:,None,None,None]).astype(np.float32), (beta-mean*sc).astype(np.float32)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--epochs',type=int,default=34); ap.add_argument('--bs',type=int,default=128)
    ap.add_argument('--lr',type=float,default=0.05); ap.add_argument('--device',default='mps')
    ap.add_argument('--out',default=os.path.join(ROOT,'cnnw_hd.json'))
    a=ap.parse_args()
    dev=torch.device(a.device if (torch.backends.mps.is_available() or a.device=='cpu') else 'cpu'); print('device',dev)

    Xtr,ytr=load_split('train'); Xte,yte=load_split('val')
    print('train',Xtr.shape,'val',Xte.shape)
    mean=(Xtr/255.0).mean((0,2,3)).astype(np.float32); std=(Xtr/255.0).std((0,2,3)).astype(np.float32)
    print('mean',mean.round(3),'std',std.round(3))
    m3=mean[None,:,None,None]; s3=std[None,:,None,None]
    Xte_s=crop_flip(Xte,False); Xte_t=torch.tensor((Xte_s/255.0-m3)/s3,dtype=torch.float32,device=dev); yte_t=torch.tensor(yte,device=dev)

    net=CNN().to(dev)
    opt=torch.optim.SGD(net.parameters(),lr=a.lr,momentum=0.9,weight_decay=5e-4,nesterov=True)
    n=len(Xtr); sched=torch.optim.lr_scheduler.OneCycleLR(opt,max_lr=a.lr*3,epochs=a.epochs,steps_per_epoch=(n+a.bs-1)//a.bs)
    for ep in range(a.epochs):
        net.train(); t0=time.time(); idx=np.random.permutation(n)
        Xa=crop_flip(Xtr[idx],True); ya=ytr[idx]
        for i in range(0,n,a.bs):
            xb=torch.tensor((Xa[i:i+a.bs]/255.0-m3)/s3,dtype=torch.float32,device=dev)
            yb=torch.tensor(ya[i:i+a.bs],device=dev)
            opt.zero_grad(); F.cross_entropy(net(xb),yb).backward(); opt.step(); sched.step()
        net.eval()
        with torch.no_grad():
            pred=torch.cat([net(Xte_t[i:i+500]).argmax(1) for i in range(0,len(Xte_t),500)])
            acc=(pred==yte_t).float().mean().item()
        print(f'ep{ep+1:02d}/{a.epochs} val {acc:.4f} ({time.time()-t0:.1f}s)')

    net.eval()
    with torch.no_grad():
        pred=torch.cat([net(Xte_t[i:i+500]).argmax(1) for i in range(0,len(Xte_t),500)]).cpu().numpy()
    per={EN[c]:round(float((pred[yte==c]==c).mean()),3) for c in range(10)}
    overall=round(float((pred==yte).mean()),4); print('VAL tổng',overall); print('theo lớp',per)

    convs=[]
    for cv,bn in zip(net.convs,net.bns):
        W,b=fold_bn(cv,bn); convs.append({'w':W.tolist(),'b':b.tolist()})
    out={'meta':{'arch':'cnnN-gap','input_size':S,'classes_en':EN,'classes_vi':VI,
                 'mean':mean.tolist(),'std':std.tolist(),'test_acc':overall,'per_class':per,'dataset':'Imagenette-160'},
         'convs':convs, 'fc':{'w':net.fc.weight.detach().cpu().numpy().tolist(),'b':net.fc.bias.detach().cpu().numpy().tolist()}}
    def rnd(x): return [rnd(v) for v in x] if isinstance(x,list) else (round(x,5) if isinstance(x,float) else x)
    for cv in out['convs']: cv['w']=rnd(cv['w']); cv['b']=rnd(cv['b'])
    out['fc']['w']=rnd(out['fc']['w']); out['fc']['b']=rnd(out['fc']['b'])
    json.dump(out,open(a.out,'w'),separators=(',',':')); print('ĐÃ XUẤT',a.out,f'({os.path.getsize(a.out)/1e6:.2f}MB)')

    # mẫu: 2 ảnh/lớp từ val (uint8 HWC 96×96) cho demo
    picks=[]
    for c in range(10): picks += list(np.where(yte==c)[0][:2])
    Xs=Xte_s[picks].transpose(0,2,3,1)     # (K,96,96,3) HWC uint8
    np.savez(os.path.join(ROOT,'samples_hd.npz'), X=Xs, y=yte[picks]); print('ĐÃ LƯU samples_hd.npz', Xs.shape)

if __name__=='__main__': main()
