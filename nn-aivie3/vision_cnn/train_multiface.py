#!/usr/bin/env python
# NHẬN NHIỀU MẶT: detector kiểu CenterNet-lite. Xuất bản-đồ TÂM-MẶT (40×40) + hộp(wh) + 5 mốc(offset) tại mỗi ô.
# Suy-luận: tìm ĐỈNH trên bản-đồ tâm → mỗi đỉnh 1 mặt → đọc hộp+mốc → NMS. 1 lần forward ra MỌI mặt.
# Dữ-liệu: ghép 1-3 mặt (LFW+WIDER, nhãn YuNet) lên nền thật, vị-trí/cỡ ngẫu-nhiên.
import os, json, time, argparse
import numpy as np, cv2, torch, torch.nn as nn, torch.nn.functional as F
from train_detector import load_sources, load_bg
ROOT=os.path.dirname(os.path.abspath(__file__)); S=160; FS=40   # ảnh 160, bản-đồ tâm 40 (÷4)

def iou(a,b):
    ax0,ay0,ax1,ay1=a[0],a[1],a[0]+a[2],a[1]+a[3]; bx0,by0,bx1,by1=b[0],b[1],b[0]+b[2],b[1]+b[3]
    ix0,iy0,ix1,iy1=max(ax0,bx0),max(ay0,by0),min(ax1,bx1),min(ay1,by1)
    iw=max(0,ix1-ix0); ih=max(0,iy1-iy0); inter=iw*ih; u=a[2]*a[3]+b[2]*b[3]-inter+1e-9
    return inter/u

def composite_multi(X,box,lm,BG,train):
    base=BG[np.random.randint(len(BG))].copy() if (train and BG is not None) else np.full((S,S,3),120,np.uint8)
    base=cv2.resize(base,(S,S)); K=np.random.randint(1,4) if train else 1; faces=[]
    for _ in range(K*3):
        if len(faces)>=K: break
        i=np.random.randint(len(X)); b=box[i]; L=lm[i].reshape(5,2)
        tw=np.random.uniform(0.12,0.4) if train else 0.3
        fcx=(b[0]+b[2]/2); fcy=(b[1]+b[3]/2)
        s=tw/max(b[2],1e-3)                                       # src(160)→canvas theo bề-ngang mặt
        rx=np.random.uniform(0.12,0.88) if train else 0.5; ry=np.random.uniform(0.12,0.88) if train else 0.5
        ang=np.deg2rad(np.random.uniform(-32,32)) if train else 0.0   # NGHIÊNG (roll): mốc+hộp phải XOAY theo
        ca,sa=np.cos(ang),np.sin(ang); cxs,cys=fcx*S,fcy*S            # xoay+co quanh tâm mặt → đặt tâm tới (rx,ry)
        M=np.array([[s*ca,-s*sa, rx*S-s*(ca*cxs-sa*cys)],
                    [s*sa, s*ca, ry*S-s*(sa*cxs+ca*cys)]],np.float32)
        apM=lambda p: (np.concatenate([p*S,np.ones((len(p),1),np.float32)],1)@M.T)/S   # (N,2)[0,1] nguồn→canvas
        cor=apM(np.array([[b[0],b[1]],[b[0]+b[2],b[1]],[b[0]+b[2],b[1]+b[3]],[b[0],b[1]+b[3]]],np.float32))
        x0c,y0c,x1c,y1c=cor[:,0].min(),cor[:,1].min(),cor[:,0].max(),cor[:,1].max()
        nb=[x0c,y0c,x1c-x0c,y1c-y0c]                             # hộp BAO mặt đã xoay [0,1]
        if any(iou(nb,f[0])>0.15 for f in faces): continue       # tránh mặt chồng mặt
        fw=cv2.warpAffine(X[i],M,(S,S),borderValue=(0,0,0)); mask=cv2.warpAffine(np.full((S,S),255,np.uint8),M,(S,S))>0
        base=np.where(mask[...,None],fw,base)
        nl=apM(L)                                                # 5 mốc XOAY theo mặt
        faces.append((np.clip(nb,0,1), np.clip(nl.reshape(-1),0,1)))
    if train: base=np.clip(base.astype(np.float32)*np.random.uniform(0.7,1.3)+np.random.uniform(-18,18),0,255).astype(np.uint8)
    return base, faces

def make_targets(batch_faces):                                   # → hm(B,1,40,40), wh(B,2,40,40), lm(B,10,40,40), mask(B,1,40,40)
    B=len(batch_faces); hm=np.zeros((B,1,FS,FS),np.float32); wh=np.zeros((B,2,FS,FS),np.float32)
    lmt=np.zeros((B,10,FS,FS),np.float32); mask=np.zeros((B,1,FS,FS),np.float32)
    yy,xx=np.mgrid[0:FS,0:FS]
    for bi,faces in enumerate(batch_faces):
        for nb,nl in faces:
            cx=(nb[0]+nb[2]/2)*FS; cy=(nb[1]+nb[3]/2)*FS; ix=min(FS-1,int(cx)); iy=min(FS-1,int(cy))
            sig=max(1.0,(nb[2]*FS)*0.3); g=np.exp(-((xx-cx)**2+(yy-cy)**2)/(2*sig*sig))
            hm[bi,0]=np.maximum(hm[bi,0],g)
            wh[bi,0,iy,ix]=nb[2]; wh[bi,1,iy,ix]=nb[3]; mask[bi,0,iy,ix]=1
            for k in range(5):
                lmt[bi,2*k,iy,ix]=nl[2*k]-(ix+0.5)/FS
                lmt[bi,2*k+1,iy,ix]=nl[2*k+1]-(iy+0.5)/FS
    return hm,wh,lmt,mask

class MF(nn.Module):
    def __init__(s,ch=(24,48,96)):
        super().__init__(); s.c1=nn.Conv2d(3,ch[0],3,padding=1,bias=False);s.b1=nn.BatchNorm2d(ch[0])
        s.c2=nn.Conv2d(ch[0],ch[1],3,padding=1,bias=False);s.b2=nn.BatchNorm2d(ch[1])
        s.c3=nn.Conv2d(ch[1],ch[2],3,padding=1,bias=False);s.b3=nn.BatchNorm2d(ch[2])
        s.hm=nn.Conv2d(ch[2],1,3,padding=1); s.wh=nn.Conv2d(ch[2],2,3,padding=1); s.lm=nn.Conv2d(ch[2],10,3,padding=1)
    def forward(s,x):
        x=F.max_pool2d(F.relu(s.b1(s.c1(x))),2)     #160->80
        x=F.max_pool2d(F.relu(s.b2(s.c2(x))),2)     #80->40
        x=F.relu(s.b3(s.c3(x)))                      #40 (giữ)
        return s.hm(x), s.wh(x), s.lm(x)

def fold_bn(conv,bn):
    W=conv.weight.detach().cpu().numpy();g=bn.weight.detach().cpu().numpy();be=bn.bias.detach().cpu().numpy()
    mu=bn.running_mean.detach().cpu().numpy();var=bn.running_var.detach().cpu().numpy();sc=g/np.sqrt(var+bn.eps)
    return (W*sc[:,None,None,None]).astype(np.float32),(be-mu*sc).astype(np.float32)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--epochs',type=int,default=40);ap.add_argument('--bs',type=int,default=64)
    ap.add_argument('--lr',type=float,default=0.05);ap.add_argument('--device',default='mps');ap.add_argument('--out',default=os.path.join(ROOT,'mfw.json'))
    a=ap.parse_args();dev=torch.device(a.device if(torch.backends.mps.is_available() or a.device=='cpu')else 'cpu');print('device',dev)
    load_bg();import train_detector as td; BG=td.BG; X,box,lm=load_sources();n=len(X);print('mặt-nguồn',X.shape)
    net=MF().to(dev);opt=torch.optim.SGD(net.parameters(),lr=a.lr,momentum=0.9,weight_decay=5e-4,nesterov=True)
    steps=300; sched=torch.optim.lr_scheduler.OneCycleLR(opt,max_lr=a.lr*3,epochs=a.epochs,steps_per_epoch=steps)
    for ep in range(a.epochs):
        net.train();t0=time.time();tot=0
        for _ in range(steps):
            cvs=[];fcs=[]
            for _ in range(a.bs):
                c,f=composite_multi(X,box,lm,BG,True);cvs.append(c.transpose(2,0,1)/255.0);fcs.append(f)
            xb=torch.tensor(np.array(cvs,np.float32),device=dev)
            hm,wh,lmt,mask=make_targets(fcs)
            hm=torch.tensor(hm,device=dev);wh=torch.tensor(wh,device=dev);lmt=torch.tensor(lmt,device=dev);mask=torch.tensor(mask,device=dev)
            phm,pwh,plm=net(xb)
            lh=F.mse_loss(phm.sigmoid(),hm)*20
            m=mask; denom=m.sum()+1e-6
            lw=(F.l1_loss(pwh*m,wh*m,reduction='sum'))/denom
            ll=(F.l1_loss(plm*m.repeat(1,10,1,1),lmt*m.repeat(1,10,1,1),reduction='sum'))/(denom*10)*5
            loss=lh+lw+ll;opt.zero_grad();loss.backward();opt.step();sched.step();tot+=float(loss.item())
        if (ep+1)%5==0 or ep==a.epochs-1: print(f'ep{ep+1:02d}/{a.epochs} loss {tot/steps:.3f} ({time.time()-t0:.1f}s)',flush=True)
    W1,b1=fold_bn(net.c1,net.b1);W2,b2=fold_bn(net.c2,net.b2);W3,b3=fold_bn(net.c3,net.b3)
    raw=lambda c:{'w':c.weight.detach().cpu().numpy().tolist(),'b':c.bias.detach().cpu().numpy().tolist()}
    out={'meta':{'arch':'centernet-lite','input_size':S,'fs':FS},
         'enc':[{'w':W1.tolist(),'b':b1.tolist()},{'w':W2.tolist(),'b':b2.tolist()},{'w':W3.tolist(),'b':b3.tolist()}],
         'hm':raw(net.hm),'wh':raw(net.wh),'lm':raw(net.lm)}
    rnd=lambda x:[rnd(v) for v in x] if isinstance(x,list) else (round(x,5) if isinstance(x,float) else x)
    for p in ['enc','hm','wh','lm']:
        o=out[p]
        for o2 in (o if isinstance(o,list) else [o]): o2['w']=rnd(o2['w']);o2['b']=rnd(o2['b'])
    json.dump(out,open(a.out,'w'),separators=(',',':'));print('ĐÃ XUẤT',a.out,f'({os.path.getsize(a.out)/1e6:.2f}MB)',flush=True)

if __name__=='__main__': main()
