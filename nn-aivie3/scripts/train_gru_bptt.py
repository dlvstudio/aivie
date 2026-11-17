"""nn-aivie3 · Luyện GRU (hồi-quy CÓ CỔNG) đá bóng — BPTT PyTorch/MPS. GRU đủ mạnh để "xoay toạ-độ" egocentric→world.
Xuất gruw.json (schema tự-định-nghĩa) → cần cài GRU-forward trong JS để chạy trình duyệt.
CHẠY: python train_gru_bptt.py --device mps --epochs 140
"""
import argparse, json, math, time, os
import torch, torch.nn as nn, torch.nn.functional as F
SP='/private/tmp/claude-501/-Users-user-Documents-learn-nn-aivie/bbbd11fd-4078-4299-874a-72c1068fd8b1/scratchpad/'

class GRUPolicy(nn.Module):
    def __init__(self, IN, H=64, HID=48):
        super().__init__(); self.gru=nn.GRU(IN,H,batch_first=True)
        self.f1=nn.Linear(H,HID); self.f2=nn.Linear(HID,2)
    def forward(self,X):
        y,_=self.gru(X); return torch.tanh(self.f2(torch.relu(self.f1(y))))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--device',default='mps' if torch.backends.mps.is_available() else 'cpu')
    ap.add_argument('--epochs',type=int,default=140); ap.add_argument('--bs',type=int,default=64); ap.add_argument('--lr',type=float,default=2e-3)
    ap.add_argument('--H',type=int,default=64); ap.add_argument('--HID',type=int,default=48)
    ap.add_argument('--seqs',default=SP+'seqs.json'); ap.add_argument('--out',default=SP+'gruw.json')
    a=ap.parse_args(); dev=torch.device(a.device)
    d=json.load(open(a.seqs)); IN=d['IN']; chunks=d['chunks']
    X=torch.tensor([[f[0] for f in c] for c in chunks],dtype=torch.float32)
    Y=torch.tensor([[f[1] for f in c] for c in chunks],dtype=torch.float32)
    N=X.shape[0]; ntr=int(N*0.9); perm=torch.randperm(N); tr,va=perm[:ntr],perm[ntr:]
    Xtr,Ytr,Xva,Yva=X[tr].to(dev),Y[tr].to(dev),X[va].to(dev),Y[va].to(dev)
    print(f'device={dev} · {N} chunk × T={X.shape[1]} · IN={IN} · GRU H={a.H} HID={a.HID}')
    net=GRUPolicy(IN,a.H,a.HID).to(dev); opt=torch.optim.Adam(net.parameters(),lr=a.lr); t0=time.time()
    for ep in range(a.epochs):
        net.train(); pm=torch.randperm(ntr,device=dev); L=0;nb=0
        for i in range(0,ntr,a.bs):
            idx=pm[i:i+a.bs]; opt.zero_grad()
            loss=F.mse_loss(net(Xtr[idx]),Ytr[idx]); loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(),5.0); opt.step(); L+=loss.item();nb+=1
        if (ep+1)%10==0:
            net.eval()
            with torch.no_grad(): vl=F.mse_loss(net(Xva),Yva).item()
            print(f'ep{ep+1} train {L/nb:.4f} · val {vl:.4f} · {time.time()-t0:.0f}s',flush=True)
    # xuất: GRU (W_ih[3H,IN],W_hh[3H,H],b_ih[3H],b_hh[3H]) + f1,f2
    g=net.gru
    obj={'meta':{'IN':IN,'H':a.H,'HID':a.HID,'arch':'gru','gate_order':'r,z,n'},
         'W_ih':g.weight_ih_l0.detach().cpu().tolist(),'W_hh':g.weight_hh_l0.detach().cpu().tolist(),
         'b_ih':g.bias_ih_l0.detach().cpu().tolist(),'b_hh':g.bias_hh_l0.detach().cpu().tolist(),
         'f1w':net.f1.weight.detach().cpu().tolist(),'f1b':net.f1.bias.detach().cpu().tolist(),
         'f2w':net.f2.weight.detach().cpu().tolist(),'f2b':net.f2.bias.detach().cpu().tolist()}
    json.dump(obj,open(a.out,'w')); print('XUẤT →',a.out,f'({os.path.getsize(a.out)//1024} KB)')

if __name__=='__main__': main()
