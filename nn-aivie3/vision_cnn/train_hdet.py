#!/usr/bin/env python
# Dò-mặt HEATMAP: encoder→decoder xuất 5 BẢN-ĐỒ NHIỆT (mắt/mũi/miệng) 32×32 (giữ toạ-độ không-gian → KHÍT)
# + đầu hồi-quy hộp. Nhãn từ YuNet (LFW+WIDER), tăng-cường nền thật/màu/cỡ. Chạy MPS (hồi-quy, không dính bug metric).
import os, json, time, argparse
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from train_detector import load_sources, load_bg, batch, fold_bn, S, SRC
HM=32; SIG=1.5

class HDet(nn.Module):
    def __init__(s,ch=(24,48,96)):
        super().__init__()
        s.c1=nn.Conv2d(3,ch[0],3,padding=1,bias=False); s.b1=nn.BatchNorm2d(ch[0])
        s.c2=nn.Conv2d(ch[0],ch[1],3,padding=1,bias=False); s.b2=nn.BatchNorm2d(ch[1])
        s.c3=nn.Conv2d(ch[1],ch[2],3,padding=1,bias=False); s.b3=nn.BatchNorm2d(ch[2])
        s.dec=nn.Conv2d(ch[2],ch[1],3,padding=1,bias=False); s.bd=nn.BatchNorm2d(ch[1])
        s.head=nn.Conv2d(ch[1],5,3,padding=1)          # 5 heatmap (có bias, KHÔNG bn)
        s.boxfc=nn.Linear(ch[2],4)
    def forward(s,x):
        x=F.max_pool2d(F.relu(s.b1(s.c1(x))),2)        #128->64
        x=F.max_pool2d(F.relu(s.b2(s.c2(x))),2)        #64->32
        f=F.max_pool2d(F.relu(s.b3(s.c3(x))),2)        #32->16
        box=torch.sigmoid(s.boxfc(f.mean((2,3))))
        u=F.interpolate(f,scale_factor=2,mode='nearest')  #16->32
        d=F.relu(s.bd(s.dec(u)))
        return box, s.head(d)                          # hm (B,5,32,32) logits

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--epochs',type=int,default=45); ap.add_argument('--bs',type=int,default=128)
    ap.add_argument('--lr',type=float,default=0.05); ap.add_argument('--device',default='mps'); ap.add_argument('--out',default=os.path.join(os.path.dirname(__file__),'detw.json'))
    a=ap.parse_args(); dev=torch.device(a.device if (torch.backends.mps.is_available() or a.device=='cpu') else 'cpu'); print('device',dev)
    load_bg(); X,box,lm=load_sources(); n=len(X); print('ảnh',X.shape)
    ne=n//10; ev=np.arange(ne); tr=np.arange(ne,n)
    gy,gx=torch.meshgrid(torch.arange(HM,dtype=torch.float32,device=dev),torch.arange(HM,dtype=torch.float32,device=dev),indexing='ij')
    idxx=(torch.arange(HM*HM,device=dev)%HM).float(); idxy=(torch.arange(HM*HM,device=dev)//HM).float()
    def target(tl):                                    # tl (B,10) [0,1] → heatmap Gaussian (B,5,32,32)
        p=tl.reshape(-1,5,2)*HM; px=p[:,:,0:1].unsqueeze(-1); py=p[:,:,1:2].unsqueeze(-1)
        return torch.exp(-((gx.view(1,1,HM,HM)-px)**2+(gy.view(1,1,HM,HM)-py)**2)/(2*SIG*SIG))
    def softargmax(hm):
        pr=F.softmax(hm.reshape(-1,5,HM*HM),2)
        xs=(pr*idxx).sum(2)/(HM-1); ys=(pr*idxy).sum(2)/(HM-1)
        return torch.stack([xs,ys],2).reshape(-1,10)
    net=HDet().to(dev); opt=torch.optim.SGD(net.parameters(),lr=a.lr,momentum=0.9,weight_decay=5e-4,nesterov=True)
    sched=torch.optim.lr_scheduler.OneCycleLR(opt,max_lr=a.lr*3,epochs=a.epochs,steps_per_epoch=(len(tr)+a.bs-1)//a.bs)
    for ep in range(a.epochs):
        net.train(); t0=time.time(); np.random.shuffle(tr)
        for i in range(0,len(tr),a.bs):
            cv,tb,tl=batch(X,box,lm,tr[i:i+a.bs],True)
            xb=torch.tensor(cv,device=dev); bb=torch.tensor(tb,device=dev); ll=torch.tensor(tl,device=dev)
            pbox,hm=net(xb); tgt=target(ll)
            loss=F.mse_loss(pbox,bb)+8*F.mse_loss(hm.sigmoid(),tgt)+2*F.mse_loss(softargmax(hm),ll)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if (ep+1)%6==0 or ep==a.epochs-1:
            net.eval(); errs=[]
            with torch.no_grad():
                for i in range(0,len(ev),256):
                    cv,tb,tl=batch(X,box,lm,ev[i:i+256],False)
                    _,hm=net(torch.tensor(cv,device=dev)); errs.append((softargmax(hm).cpu().numpy()-tl).reshape(-1,5,2))
            e=np.concatenate(errs); print(f'ep{ep+1:02d}/{a.epochs} sai-số-mốc {np.sqrt((e**2).sum(-1)).mean()*100:.1f}% cạnh ({time.time()-t0:.1f}s)',flush=True)
    mlme=float(np.sqrt((np.concatenate(errs)**2).sum(-1)).mean())
    def foldc(c,b): W,bb=fold_bn(c,b); return {'w':W.tolist(),'b':bb.tolist()}
    def rawc(c): return {'w':c.weight.detach().cpu().numpy().tolist(),'b':c.bias.detach().cpu().numpy().tolist()}
    out={'meta':{'arch':'hdet','input_size':S,'hm':HM,'mean':[0,0,0],'std':[1,1,1],'lm_err':round(mlme,4)},
         'enc':[foldc(net.c1,net.b1),foldc(net.c2,net.b2),foldc(net.c3,net.b3)], 'dec':foldc(net.dec,net.bd),
         'head':rawc(net.head), 'boxfc':{'w':net.boxfc.weight.detach().cpu().numpy().tolist(),'b':net.boxfc.bias.detach().cpu().numpy().tolist()}}
    def rnd(x): return [rnd(v) for v in x] if isinstance(x,list) else (round(x,5) if isinstance(x,float) else x)
    for part in ['enc','dec','head','boxfc']:
        o=out[part]
        for o2 in (o if isinstance(o,list) else [o]): o2['w']=rnd(o2['w']); o2['b']=rnd(o2['b'])
    json.dump(out,open(a.out,'w'),separators=(',',':')); print('ĐÃ XUẤT',a.out,f'({os.path.getsize(a.out)/1e6:.2f}MB)',flush=True)

if __name__=='__main__': main()
