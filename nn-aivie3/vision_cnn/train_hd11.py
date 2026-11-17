#!/usr/bin/env python
# 11 lớp @128px: Imagenette (10) + "người" từ LFW (khuôn-mặt thật ~250px). 4 conv+GAP. Cân lớp bằng weight.
import os, sys, json, time, argparse, random
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from PIL import Image
from train_imagenette import CNN, fold_bn, WNIDS, EN as EN10, VI as VI10

ROOT = os.path.dirname(os.path.abspath(__file__))
IMG  = os.path.join(ROOT, '..', '..', 'data', 'imagenette2-160')
LFW  = os.path.join(ROOT, '..', '..', 'data', 'lfw')
S, LOAD = 128, 144
EN = EN10 + ['person']; VI = VI10 + ['người']

def load_dir_images(paths, label):
    X=[]
    for p in paths:
        try: im = Image.open(p).convert('RGB').resize((LOAD,LOAD), Image.BILINEAR)
        except Exception: continue
        X.append(np.asarray(im, np.uint8))
    X = np.stack(X).transpose(0,3,1,2)      # (N,3,LOAD,LOAD)
    return X, np.full(len(X), label, np.int64)

def load_imagenette(split):
    cache=os.path.join(ROOT,f'imnet_{split}_{LOAD}.npz')
    if os.path.exists(cache): d=np.load(cache); return d['X'],d['y']
    base=os.path.join(IMG,split); X=[];y=[]
    for ci,wnid in enumerate(WNIDS):
        dd=os.path.join(base,wnid)
        for fn in os.listdir(dd):
            if not fn.lower().endswith(('.jpeg','.jpg','.png')): continue
            im=Image.open(os.path.join(dd,fn)).convert('RGB').resize((LOAD,LOAD),Image.BILINEAR)
            X.append(np.asarray(im,np.uint8)); y.append(ci)
    X=np.stack(X).transpose(0,3,1,2); y=np.array(y,np.int64); np.savez(cache,X=X,y=y); return X,y

def load_person(n_train=1500, n_val=450):
    if not os.path.isdir(LFW): sys.exit(f'CHƯA CÓ LFW: {LFW}')
    paths=[]
    for root,_,files in os.walk(LFW):
        for f in files:
            if f.lower().endswith('.jpg'): paths.append(os.path.join(root,f))
    random.Random(0).shuffle(paths)
    tr = paths[:n_train]; va = paths[n_train:n_train+n_val]
    Xtr,ytr = load_dir_images(tr,10); Xva,yva = load_dir_images(va,10)
    print(f"'người' (LFW): train {len(Xtr)}  val {len(Xva)}")
    return Xtr,ytr,Xva,yva

def crop_flip(X, train):
    n=len(X); out=np.empty((n,3,S,S),np.uint8); m=LOAD-S
    if train: ox=np.random.randint(0,m+1,n); oy=np.random.randint(0,m+1,n); fl=np.random.rand(n)<0.5
    else: ox=np.full(n,m//2); oy=np.full(n,m//2); fl=np.zeros(n,bool)
    for i in range(n):
        c=X[i,:,oy[i]:oy[i]+S, ox[i]:ox[i]+S]; out[i]=c[:,:,::-1] if fl[i] else c
    return out

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--epochs',type=int,default=34); ap.add_argument('--bs',type=int,default=96)
    ap.add_argument('--lr',type=float,default=0.05); ap.add_argument('--device',default='mps')
    ap.add_argument('--size',type=int,default=128); ap.add_argument('--load',type=int,default=144)
    ap.add_argument('--out',default=os.path.join(ROOT,'cnnw_hd.json'))
    a=ap.parse_args()
    global S, LOAD; S=a.size; LOAD=a.load
    dev=torch.device(a.device if (torch.backends.mps.is_available() or a.device=='cpu') else 'cpu'); print('device',dev,'| S',S,'LOAD',LOAD)

    Xi_tr,yi_tr=load_imagenette('train'); Xi_va,yi_va=load_imagenette('val')
    Xp_tr,yp_tr,Xp_va,yp_va=load_person()
    Xtr=np.concatenate([Xi_tr,Xp_tr]); ytr=np.concatenate([yi_tr,yp_tr])
    Xva=np.concatenate([Xi_va,Xp_va]); yva=np.concatenate([yi_va,yp_va])
    print('train',Xtr.shape,'val',Xva.shape)
    mean=(Xtr/255.0).mean((0,2,3)).astype(np.float32); std=(Xtr/255.0).std((0,2,3)).astype(np.float32)
    m3=mean[None,:,None,None]; s3=std[None,:,None,None]
    Xva_s=crop_flip(Xva,False); Xva_t=torch.tensor((Xva_s/255.0-m3)/s3,dtype=torch.float32,device=dev); yva_t=torch.tensor(yva,device=dev)
    cnt=np.bincount(ytr,minlength=11).astype(np.float32); w=cnt.mean()/cnt; wt=torch.tensor(w,dtype=torch.float32,device=dev)
    print('số ảnh/lớp',cnt.astype(int).tolist(),'\nweight',w.round(2).tolist())

    net=CNN(ncls=11).to(dev)
    opt=torch.optim.SGD(net.parameters(),lr=a.lr,momentum=0.9,weight_decay=5e-4,nesterov=True)
    n=len(Xtr); sched=torch.optim.lr_scheduler.OneCycleLR(opt,max_lr=a.lr*3,epochs=a.epochs,steps_per_epoch=(n+a.bs-1)//a.bs)
    for ep in range(a.epochs):
        net.train(); t0=time.time(); idx=np.random.permutation(n); Xa=crop_flip(Xtr[idx],True); ya=ytr[idx]
        for i in range(0,n,a.bs):
            xb=torch.tensor((Xa[i:i+a.bs]/255.0-m3)/s3,dtype=torch.float32,device=dev); yb=torch.tensor(ya[i:i+a.bs],device=dev)
            opt.zero_grad(); F.cross_entropy(net(xb),yb,weight=wt).backward(); opt.step(); sched.step()
        net.eval()
        with torch.no_grad():
            pred=torch.cat([net(Xva_t[i:i+400]).argmax(1) for i in range(0,len(Xva_t),400)])
            acc=(pred==yva_t).float().mean().item()
        print(f'ep{ep+1:02d}/{a.epochs} val {acc:.4f} ({time.time()-t0:.1f}s)')

    net.eval()
    with torch.no_grad():
        pred=torch.cat([net(Xva_t[i:i+400]).argmax(1) for i in range(0,len(Xva_t),400)]).cpu().numpy()
    per={EN[c]:round(float((pred[yva==c]==c).mean()),3) for c in range(11)}
    overall=round(float((pred==yva).mean()),4); print('VAL tổng',overall); print('theo lớp',per)

    convs=[]
    for cv,bn in zip(net.convs,net.bns):
        W,b=fold_bn(cv,bn); convs.append({'w':W.tolist(),'b':b.tolist()})
    out={'meta':{'arch':'cnnN-gap','input_size':S,'classes_en':EN,'classes_vi':VI,'mean':mean.tolist(),'std':std.tolist(),
                 'test_acc':overall,'per_class':per,'dataset':'Imagenette-160 + LFW(người)'},
         'convs':convs,'fc':{'w':net.fc.weight.detach().cpu().numpy().tolist(),'b':net.fc.bias.detach().cpu().numpy().tolist()}}
    def rnd(x): return [rnd(v) for v in x] if isinstance(x,list) else (round(x,5) if isinstance(x,float) else x)
    for cv in out['convs']: cv['w']=rnd(cv['w']); cv['b']=rnd(cv['b'])
    out['fc']['w']=rnd(out['fc']['w']); out['fc']['b']=rnd(out['fc']['b'])
    json.dump(out,open(a.out,'w'),separators=(',',':')); print('ĐÃ XUẤT',a.out,f'({os.path.getsize(a.out)/1e6:.2f}MB)')

    picks=[]
    for c in range(11): picks += list(np.where(yva==c)[0][:2])
    Xs=Xva_s[picks].transpose(0,2,3,1)
    np.savez(os.path.join(ROOT,'samples_hd.npz'),X=Xs,y=yva[picks]); print('ĐÃ LƯU samples_hd.npz',Xs.shape)

if __name__=='__main__': main()
