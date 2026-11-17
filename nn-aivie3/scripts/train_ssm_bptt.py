"""nn-aivie3 · Luyện SSM (hồi-quy, tự nhớ) đá bóng bằng BPTT — PyTorch autograd, chạy MPS/GPU trên M4.
Giám-sát DÀY từ thầy: chuỗi (camera-obs 51 → hành-động [steer,thr]) dump từ sim JS (dump_seqs.mjs).
SSM chéo: h=a·h+(1-a)(W_in x+b) ; đọc-ra MLP. Toán KHỚP ssmStep trong robotball.html (parity Δ~1e-16).
Xuất ssmw.json → dán vào /*__SSMW__*/ để chạy trong trình duyệt.
CHẠY:  python train_ssm_bptt.py --device mps --epochs 120
"""
import argparse, json, math, time, os
import torch, torch.nn as nn, torch.nn.functional as F

SP='/private/tmp/claude-501/-Users-user-Documents-learn-nn-aivie/bbbd11fd-4078-4299-874a-72c1068fd8b1/scratchpad/'
H, HID = 64, 48

class SSM(nn.Module):
    def __init__(self, IN):
        super().__init__()
        self.W_in=nn.Parameter(torch.randn(H,IN)/math.sqrt(IN)); self.b_in=nn.Parameter(torch.zeros(H))
        self.a_raw=nn.Parameter(torch.ones(H))                    # a≈0.73 ban đầu
        self.W1=nn.Parameter(torch.randn(HID,H)*math.sqrt(2/H)); self.b1=nn.Parameter(torch.zeros(HID))
        self.W2=nn.Parameter(torch.randn(2,HID)/math.sqrt(HID)); self.b2=nn.Parameter(torch.zeros(2))
    def forward(self, X):   # X [B,T,IN] → Y [B,T,2]
        B,T,IN=X.shape; h=torch.zeros(B,H,device=X.device); a=torch.sigmoid(self.a_raw); ys=[]
        Wi=self.W_in.t()
        for t in range(T):
            u=X[:,t]@Wi + self.b_in
            h=a*h+(1-a)*u
            o=torch.relu(h@self.W1.t()+self.b1)
            ys.append(torch.tanh(o@self.W2.t()+self.b2))
        return torch.stack(ys,1)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--device',default='mps' if torch.backends.mps.is_available() else 'cpu')
    ap.add_argument('--epochs',type=int,default=120); ap.add_argument('--bs',type=int,default=64); ap.add_argument('--lr',type=float,default=2e-3)
    ap.add_argument('--seqs',default=SP+'seqs.json'); ap.add_argument('--out',default=SP+'ssmw.json')
    a=ap.parse_args(); dev=torch.device(a.device)
    d=json.load(open(a.seqs)); IN=d['IN']; chunks=d['chunks']
    X=torch.tensor([[f[0] for f in c] for c in chunks],dtype=torch.float32)   # [N,T,IN]
    Y=torch.tensor([[f[1] for f in c] for c in chunks],dtype=torch.float32)   # [N,T,2]
    N=X.shape[0]; ntr=int(N*0.9)
    perm=torch.randperm(N); tr,va=perm[:ntr],perm[ntr:]
    Xtr,Ytr,Xva,Yva=X[tr].to(dev),Y[tr].to(dev),X[va].to(dev),Y[va].to(dev)
    print(f'device={dev} · {N} chunk × T={X.shape[1]} · IN={IN} · H={H} HID={HID}')
    net=SSM(IN).to(dev); opt=torch.optim.Adam(net.parameters(),lr=a.lr)
    t0=time.time()
    for ep in range(a.epochs):
        net.train(); pm=torch.randperm(ntr,device=dev); L=0;nb=0
        for i in range(0,ntr,a.bs):
            idx=pm[i:i+a.bs]; opt.zero_grad()
            loss=F.mse_loss(net(Xtr[idx]), Ytr[idx]); loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(),5.0); opt.step(); L+=loss.item();nb+=1
        if (ep+1)%10==0:
            net.eval()
            with torch.no_grad(): vl=F.mse_loss(net(Xva),Yva).item()
            print(f'ep{ep+1} train {L/nb:.4f} · val {vl:.4f} · {time.time()-t0:.0f}s',flush=True)
    # xuất đúng schema ssmStep
    W={k:getattr(net,k).detach().cpu() for k in ['W_in','b_in','a_raw','W1','b1','W2','b2']}
    obj={'meta':{'IN':IN,'H':H,'HID':HID,'note':'PyTorch BPTT imitation SSM (M4/MPS)'}}
    for k in ['W_in','W1','W2']: obj[k]=W[k].tolist()
    for k in ['b_in','a_raw','b1','b2']: obj[k]=W[k].tolist()
    json.dump(obj,open(a.out,'w')); print('XUẤT →',a.out, f'({os.path.getsize(a.out)//1024} KB)')

if __name__=='__main__': main()
