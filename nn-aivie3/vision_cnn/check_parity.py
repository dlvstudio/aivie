#!/usr/bin/env python
# Forward numpy KHỚP CHÍNH-XÁC forward JS (conv pad1+relu, maxpool2, fc, softmax) từ cnnw.json.
# So với PyTorch (train_cnn in ra) để xác-nhận export + layout đúng trước khi ship.
import os, json, numpy as np
ROOT = os.path.dirname(os.path.abspath(__file__))
W = json.load(open(os.path.join(ROOT,'cnnw.json')))
d = np.load(os.path.join(ROOT,'samples.npz')); X=d['X']; y=d['y']

def conv_relu(inp, w, b):        # inp (Ci,H,W); w (Co,Ci,3,3); pad1 stride1 + relu
    Ci,H,Wd = inp.shape; Co=len(w)
    P = np.pad(inp, ((0,0),(1,1),(1,1)))
    out = np.zeros((Co,H,Wd), np.float32)
    w = np.array(w, np.float32); b=np.array(b,np.float32)
    for ky in range(3):
        for kx in range(3):
            patch = P[:, ky:ky+H, kx:kx+Wd]                 # (Ci,H,W)
            out += np.tensordot(w[:,:,ky,kx], patch, axes=([1],[0]))
    out += b[:,None,None]
    return np.maximum(out,0)
def pool(inp):                    # maxpool 2x2 s2
    C,H,Wd=inp.shape
    return inp.reshape(C,H//2,2,Wd//2,2).max(axis=(2,4))
def fc(v,w,b,relu):
    o = np.array(w,np.float32)@v + np.array(b,np.float32)
    return np.maximum(o,0) if relu else o
def softmax(a): e=np.exp(a-a.max()); return e/e.sum()

def predict(planar_float):        # (3,32,32) float 0..1
    mean=np.array(W['meta']['mean']); std=np.array(W['meta']['std'])
    x=((planar_float - mean[:,None,None])/std[:,None,None]).astype(np.float32)
    h=pool(conv_relu(x, W['conv1']['w'], W['conv1']['b']))
    h=pool(conv_relu(h, W['conv2']['w'], W['conv2']['b']))
    h=pool(conv_relu(h, W['conv3']['w'], W['conv3']['b']))
    h=h.reshape(-1)                                          # c*16 + y*4 + x
    h=fc(h, W['fc1']['w'], W['fc1']['b'], True)
    return softmax(fc(h, W['fc2']['w'], W['fc2']['b'], False))

vi=W['meta']['classes_vi']
print('KIỂM 8 ảnh test đầu (numpy-từ-JSON = forward JS):')
cor=0
for i in range(8):
    p=predict(X[i]); k=int(p.argmax())
    ok = (k==y[i]); cor+=ok
    print(f'  #{i} thật={vi[y[i]]:8s} đoán={vi[k]:8s} {p[k]*100:5.1f}%  {"✓" if ok else "✗"}')
print(f'(khớp nhãn {cor}/8 — chỉ để mắt thường; parity thật = số này TRÙNG log PyTorch)')
