#!/usr/bin/env python
# CNN từ-đầu phân-loại CIFAR-10, train trên Apple M4 (MPS), KHÔNG cần torchvision.
# Xuất trọng-số ra JSON với BatchNorm ĐÃ GẬP vào conv → forward JS chỉ cần conv+relu+pool+fc.
import os, sys, json, pickle, tarfile, argparse, time, urllib.request
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, '..', '..', 'data', 'cifar-10-batches-py')
CLASSES_EN = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
CLASSES_VI = ['máy bay','ô tô','chim','mèo','hươu','chó','ếch','ngựa','tàu thủy','xe tải']

def load_batch(fn):
    with open(fn,'rb') as f: d = pickle.load(f, encoding='bytes')
    X = d[b'data'].reshape(-1,3,32,32).astype(np.float32)/255.0
    y = np.array(d[b'labels'], dtype=np.int64)
    return X, y

def load_cifar():
    if not os.path.isdir(DATA):
        sys.exit(f'CHƯA CÓ dữ-liệu: {DATA} (chạy download trước)')
    Xtr, ytr = [], []
    for i in range(1,6):
        X,y = load_batch(os.path.join(DATA, f'data_batch_{i}')); Xtr.append(X); ytr.append(y)
    Xtr = np.concatenate(Xtr); ytr = np.concatenate(ytr)
    Xte, yte = load_batch(os.path.join(DATA, 'test_batch'))
    return Xtr, ytr, Xte, yte

class CNN(nn.Module):
    def __init__(s, ncls=10):
        super().__init__()
        s.c1 = nn.Conv2d(3,16,3,padding=1,bias=False);  s.b1 = nn.BatchNorm2d(16)
        s.c2 = nn.Conv2d(16,32,3,padding=1,bias=False); s.b2 = nn.BatchNorm2d(32)
        s.c3 = nn.Conv2d(32,64,3,padding=1,bias=False); s.b3 = nn.BatchNorm2d(64)
        s.f1 = nn.Linear(64*4*4, 64); s.f2 = nn.Linear(64, ncls)
    def forward(s, x):
        x = F.max_pool2d(F.relu(s.b1(s.c1(x))), 2)   # 32->16
        x = F.max_pool2d(F.relu(s.b2(s.c2(x))), 2)   # 16->8
        x = F.max_pool2d(F.relu(s.b3(s.c3(x))), 2)   # 8->4
        x = x.flatten(1)
        x = F.relu(s.f1(x))
        return s.f2(x)

def augment(X):  # random crop (pad 4) + horizontal flip, trên numpy
    n = X.shape[0]
    out = np.empty_like(X)
    P = np.pad(X, ((0,0),(0,0),(4,4),(4,4)), mode='reflect')
    ox = np.random.randint(0,9,n); oy = np.random.randint(0,9,n)
    flip = np.random.rand(n) < 0.5
    for i in range(n):
        c = P[i,:,oy[i]:oy[i]+32, ox[i]:ox[i]+32]
        out[i] = c[:,:,::-1] if flip[i] else c
    return out

def fold_bn(conv, bn):   # trả (W_fold [O,I,kh,kw], b_fold [O]) sao cho conv+affine ≡ 1 conv
    W = conv.weight.detach().cpu().numpy()
    g = bn.weight.detach().cpu().numpy(); beta = bn.bias.detach().cpu().numpy()
    mean = bn.running_mean.detach().cpu().numpy(); var = bn.running_var.detach().cpu().numpy()
    scale = g / np.sqrt(var + bn.eps)
    Wf = W * scale[:,None,None,None]
    bf = beta - mean * scale              # conv không bias → shift = beta - mean*scale
    return Wf.astype(np.float32), bf.astype(np.float32)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=24)
    ap.add_argument('--bs', type=int, default=128)
    ap.add_argument('--lr', type=float, default=0.05)
    ap.add_argument('--device', default='mps')
    ap.add_argument('--out', default=os.path.join(ROOT,'cnnw.json'))
    a = ap.parse_args()
    dev = torch.device(a.device if torch.backends.mps.is_available() or a.device=='cpu' else 'cpu')
    print('device', dev)

    Xtr, ytr, Xte, yte = load_cifar()
    mean = Xtr.mean((0,2,3)); std = Xtr.std((0,2,3))
    print('mean', mean.round(4), 'std', std.round(4), '| train', Xtr.shape, 'test', Xte.shape)
    m3 = mean[None,:,None,None]; s3 = std[None,:,None,None]
    Xte_t = torch.tensor((Xte-m3)/s3, device=dev); yte_t = torch.tensor(yte, device=dev)

    net = CNN().to(dev)
    opt = torch.optim.SGD(net.parameters(), lr=a.lr, momentum=0.9, weight_decay=5e-4, nesterov=True)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr*3, epochs=a.epochs,
                steps_per_epoch=(len(Xtr)+a.bs-1)//a.bs)
    n = len(Xtr)
    for ep in range(a.epochs):
        net.train(); t0=time.time(); idx = np.random.permutation(n); Xa = augment(Xtr[idx]); ya = ytr[idx]
        Xa = (Xa - m3)/s3; tot=0; cor=0; los=0.0
        for i in range(0, n, a.bs):
            xb = torch.tensor(Xa[i:i+a.bs], device=dev); yb = torch.tensor(ya[i:i+a.bs], device=dev)
            opt.zero_grad(); out = net(xb); loss = F.cross_entropy(out, yb); loss.backward(); opt.step(); sched.step()
            los += loss.item()*len(yb); cor += (out.argmax(1)==yb).sum().item(); tot += len(yb)
        net.eval()
        with torch.no_grad():
            pred = []
            for i in range(0, len(Xte_t), 1000): pred.append(net(Xte_t[i:i+1000]).argmax(1))
            pred = torch.cat(pred); acc = (pred==yte_t).float().mean().item()
        print(f'ep{ep+1:02d}/{a.epochs} loss {los/tot:.3f} train {cor/tot:.3f} TEST {acc:.4f} ({time.time()-t0:.1f}s)')

    # per-class + xuất
    net.eval()
    with torch.no_grad():
        pred = torch.cat([net(Xte_t[i:i+1000]).argmax(1) for i in range(0,len(Xte_t),1000)]).cpu().numpy()
    per = {}
    for c in range(10):
        msk = yte==c; per[CLASSES_EN[c]] = round(float((pred[msk]==c).mean()),3)
    overall = round(float((pred==yte).mean()),4)
    print('TEST tổng', overall); print('theo lớp', per)

    W1,b1 = fold_bn(net.c1, net.b1); W2,b2 = fold_bn(net.c2, net.b2); W3,b3 = fold_bn(net.c3, net.b3)
    def lin(l): return {'w': l.weight.detach().cpu().numpy().tolist(), 'b': l.bias.detach().cpu().numpy().tolist()}
    out = {
        'meta': {'arch':'cnn3-bnfold','classes_en':CLASSES_EN,'classes_vi':CLASSES_VI,
                 'mean':mean.tolist(),'std':std.tolist(),'test_acc':overall,'per_class':per},
        'conv1': {'w': W1.tolist(), 'b': b1.tolist()},
        'conv2': {'w': W2.tolist(), 'b': b2.tolist()},
        'conv3': {'w': W3.tolist(), 'b': b3.tolist()},
        'fc1': lin(net.f1), 'fc2': lin(net.f2),
    }
    with open(a.out,'w') as f: json.dump(out, f)
    print('ĐÃ XUẤT', a.out, f'({os.path.getsize(a.out)/1e6:.2f}MB)')

    # lưu vài ảnh test làm mẫu cho demo (10 lớp, mỗi lớp 3 ảnh) + 1 bộ để kiểm parity
    np.savez(os.path.join(ROOT,'samples.npz'),
             X=Xte[:200].astype(np.float32), y=yte[:200])
    print('ĐÃ LƯU samples.npz (200 ảnh test đầu)')

if __name__ == '__main__':
    main()
