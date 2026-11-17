#!/usr/bin/env python
# 11 lớp: CIFAR-10 (10 vật) + "người" từ siêu-lớp 'people' của CIFAR-100 (cùng gốc, cùng 32×32).
# Cân lớp bằng weight (người ~2500 ảnh vs 5000/lớp). Tái-dùng CNN/augment/fold_bn của train_cnn.py.
import os, sys, json, pickle, time, argparse
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from train_cnn import CNN, augment, fold_bn, load_batch, ROOT, DATA

C100 = os.path.join(ROOT, '..', '..', 'data', 'cifar-100-python')
CLASSES_EN = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck','person']
CLASSES_VI = ['máy bay','ô tô','chim','mèo','hươu','chó','ếch','ngựa','tàu thủy','xe tải','người']

def load_cifar10():
    Xtr=[]; ytr=[]
    for i in range(1,6):
        X,y = load_batch(os.path.join(DATA, f'data_batch_{i}')); Xtr.append(X); ytr.append(y)
    Xtr=np.concatenate(Xtr); ytr=np.concatenate(ytr)
    Xte,yte = load_batch(os.path.join(DATA,'test_batch'))
    return Xtr,ytr,Xte,yte

def load_person():
    if not os.path.isdir(C100): sys.exit(f'CHƯA CÓ CIFAR-100: {C100}')
    meta = pickle.load(open(os.path.join(C100,'meta'),'rb'), encoding='bytes')
    coarse = [c.decode() for c in meta[b'coarse_label_names']]
    pidx = coarse.index('people')
    def grab(fn):
        d = pickle.load(open(os.path.join(C100,fn),'rb'), encoding='bytes')
        X = d[b'data'].reshape(-1,3,32,32).astype(np.float32)/255.0
        cl = np.array(d[b'coarse_labels'])
        m = cl==pidx
        return X[m]
    Xp_tr = grab('train'); Xp_te = grab('test')
    print(f"'người' (people idx {pidx}): train {len(Xp_tr)}  test {len(Xp_te)}")
    return Xp_tr, Xp_te

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=26)
    ap.add_argument('--bs', type=int, default=128)
    ap.add_argument('--lr', type=float, default=0.05)
    ap.add_argument('--device', default='mps')
    ap.add_argument('--out', default=os.path.join(ROOT,'cnnw.json'))
    a = ap.parse_args()
    dev = torch.device(a.device if (torch.backends.mps.is_available() or a.device=='cpu') else 'cpu'); print('device',dev)

    X10tr,y10tr,X10te,y10te = load_cifar10()
    Xptr,Xpte = load_person()
    Xtr = np.concatenate([X10tr, Xptr]); ytr = np.concatenate([y10tr, np.full(len(Xptr),10,np.int64)])
    Xte = np.concatenate([X10te, Xpte]); yte = np.concatenate([y10te, np.full(len(Xpte),10,np.int64)])
    print('train', Xtr.shape, 'test', Xte.shape)

    mean = X10tr.mean((0,2,3)); std = X10tr.std((0,2,3))   # chuẩn-hoá theo CIFAR-10 (như bản 10 lớp)
    m3=mean[None,:,None,None]; s3=std[None,:,None,None]
    Xte_t = torch.tensor((Xte-m3)/s3, device=dev); yte_t = torch.tensor(yte, device=dev)

    cnt = np.bincount(ytr, minlength=11).astype(np.float32)
    w = (cnt.mean()/cnt); wt = torch.tensor(w, device=dev, dtype=torch.float32)
    print('số ảnh/lớp', cnt.astype(int).tolist(), '\nweight', w.round(2).tolist())

    net = CNN(ncls=11).to(dev)
    opt = torch.optim.SGD(net.parameters(), lr=a.lr, momentum=0.9, weight_decay=5e-4, nesterov=True)
    n=len(Xtr); sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr*3, epochs=a.epochs, steps_per_epoch=(n+a.bs-1)//a.bs)
    for ep in range(a.epochs):
        net.train(); t0=time.time(); idx=np.random.permutation(n); Xa=augment(Xtr[idx]); ya=ytr[idx]; Xa=(Xa-m3)/s3
        for i in range(0,n,a.bs):
            xb=torch.tensor(Xa[i:i+a.bs],device=dev); yb=torch.tensor(ya[i:i+a.bs],device=dev)
            opt.zero_grad(); loss=F.cross_entropy(net(xb),yb,weight=wt); loss.backward(); opt.step(); sched.step()
        net.eval()
        with torch.no_grad():
            pred=torch.cat([net(Xte_t[i:i+1000]).argmax(1) for i in range(0,len(Xte_t),1000)])
            acc=(pred==yte_t).float().mean().item()
        print(f'ep{ep+1:02d}/{a.epochs} TEST {acc:.4f} ({time.time()-t0:.1f}s)')

    net.eval()
    with torch.no_grad():
        pred=torch.cat([net(Xte_t[i:i+1000]).argmax(1) for i in range(0,len(Xte_t),1000)]).cpu().numpy()
    per={}
    for c in range(11):
        msk=yte==c; per[CLASSES_EN[c]]=round(float((pred[msk]==c).mean()),3)
    overall=round(float((pred==yte).mean()),4)
    print('TEST tổng', overall); print('theo lớp', per)

    W1,b1=fold_bn(net.c1,net.b1); W2,b2=fold_bn(net.c2,net.b2); W3,b3=fold_bn(net.c3,net.b3)
    lin=lambda l:{'w':l.weight.detach().cpu().numpy().tolist(),'b':l.bias.detach().cpu().numpy().tolist()}
    out={'meta':{'arch':'cnn3-bnfold','classes_en':CLASSES_EN,'classes_vi':CLASSES_VI,
                 'mean':mean.tolist(),'std':std.tolist(),'test_acc':overall,'per_class':per},
         'conv1':{'w':W1.tolist(),'b':b1.tolist()},'conv2':{'w':W2.tolist(),'b':b2.tolist()},
         'conv3':{'w':W3.tolist(),'b':b3.tolist()},'fc1':lin(net.f1),'fc2':lin(net.f2)}
    # round 5 chữ-số để nhẹ (accuracy không đổi)
    def rnd(x):
        if isinstance(x,list): return [rnd(v) for v in x]
        return round(x,5) if isinstance(x,float) else x
    for k in ['conv1','conv2','conv3','fc1','fc2']: out[k]['w']=rnd(out[k]['w']); out[k]['b']=rnd(out[k]['b'])
    json.dump(out, open(a.out,'w'), separators=(',',':'))
    print('ĐÃ XUẤT', a.out, f'({os.path.getsize(a.out)/1e6:.2f}MB)')

    # mẫu cho demo: trộn ít vật + ít người (lấy từ test)
    pick = list(range(0,180,2))                              # 90 ảnh vật đầu (test CIFAR-10)
    Xperson = Xpte[:20]
    Xs = np.concatenate([X10te[pick], Xperson]); ys = np.concatenate([y10te[pick], np.full(len(Xperson),10)])
    np.savez(os.path.join(ROOT,'samples.npz'), X=Xs.astype(np.float32), y=ys.astype(np.int64))
    print('ĐÃ LƯU samples.npz', Xs.shape)

if __name__=='__main__': main()
