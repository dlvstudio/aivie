"""nn-aivie3 · Luyện MẠNG HỒI-QUY/SSM đá bóng NGOÀI trình duyệt (PyTorch, chạy GPU).

Vì sim gốc ở JS (robotball.html) và ES tại-chỗ thiếu compute, script này:
  • PORT sim sang Python, VECTOR-HOÁ theo lô B trận song-song (GPU → hàng trăm/nghìn trận cùng lúc).
  • Luyện policy SSM chéo (h = a·h + (1-a)·(W_in x + b) ; đọc-ra MLP) bằng ES (OpenAI-ES mirrored) — KHÔNG cần BPTT,
    hợp mạng hồi-quy, scale tuyến-tính theo GPU.
  • Camera 120° + la-bàn hướng + trí-nhớ NẰM TRONG trạng-thái h (mạng TỰ học nhớ). Đối-thủ THỤ-ĐỘNG (bootstrap).
  • XUẤT ssmw.json ĐÚNG schema `ssmStep` trong robotball.html → dán vào chỗ `/*__SSMW__*/` là chạy trong trình duyệt.

Toán & hằng-số KHỚP CHÍNH-XÁC bản JS (nếu lệch, policy sẽ không chuyển được sang trình duyệt).

CHẠY (máy có GPU):  python train_ssm_torch.py --pop 256 --gens 2000 --device cuda
Smoke (CPU):        python train_ssm_torch.py --pop 8 --gens 3 --steps 300 --device cpu
"""
import argparse, json, math, os, time
import torch

# ---- HẰNG-SỐ (khớp G trong robotball.html) ----
MX,MY,FW,FH = 40,34,640,392
BR,ROLL,WALLB = 9.0,78.0,0.72
RR,SPD,TURN,KICK = 13.0,172.0,5.4,1.35
GOALW,DT = 90.0,0.05
L,R,TOP,BOT = MX, MX+FW, MY, MY+FH
MIDY = TOP+FH/2
GY0,GY1 = MIDY-GOALW/2, MIDY+GOALW/2
CAM_N,CAM_FOV,CAM_R = 16,2.094,760.0
NS = 4                      # substep/tick
IN = 1 + 2 + 3*CAM_N        # v + sin/cos th + (ball|opp|goal)×CAM_N = 51
H, HID = 48, 32

def wrap(a): return (a+math.pi*3) % (2*math.pi) - math.pi

# ---------- ENV vector-hoá (B trận). Robot0=XANH (gôn phải), Robot1=ĐỎ thụ-động ----------
class Env:
    def __init__(self, B, device):
        self.B=B; self.dev=device; self.reset(torch.arange(B,device=device))
    def reset(self, idx):
        B=idx.numel(); dev=self.dev
        if not hasattr(self,'bx'):
            z=lambda: torch.zeros(self.B,device=dev)
            self.bx,self.by,self.bvx,self.bvy=z(),z(),z(),z()
            self.rx=torch.zeros(self.B,2,device=dev); self.ry=torch.zeros(self.B,2,device=dev)
            self.rth=torch.zeros(self.B,2,device=dev); self.rv=torch.zeros(self.B,2,device=dev)
            self.blue=z(); self.red=z(); self.stuck=z()
        self.bx[idx]=MX+FW*0.5
        self.by[idx]=MIDY+(torch.rand(B,device=dev)-0.5)*FH*0.3
        self.bvx[idx]=(torch.rand(B,device=dev)-0.5)*120; self.bvy[idx]=(torch.rand(B,device=dev)-0.5)*120
        self.rx[idx,0]=L+70; self.ry[idx,0]=MIDY; self.rth[idx,0]=0.0; self.rv[idx,0]=0.0
        self.rx[idx,1]=R-70; self.ry[idx,1]=MIDY; self.rth[idx,1]=math.pi; self.rv[idx,1]=0.0
        self.blue[idx]=0; self.red[idx]=0; self.stuck[idx]=0
    def kickoff(self, idx):  # sau bàn: bóng về giữa, robot về chỗ (giữ điểm)
        self.reset(idx)  # (điểm đã cộng trước khi gọi)
    def cam(self, i):
        """camView cho robot i: [B, 3*CAM_N] (bóng|đối-thủ|gôn)."""
        B=self.B; dev=self.dev; j=1-i
        rx,ry,th=self.rx[:,i],self.ry[:,i],self.rth[:,i]
        goalx = R+2 if i==0 else L-2   # gôn tấn-công của robot i
        goaly = MIDY
        tgts=[(self.bx,self.by),(self.rx[:,j],self.ry[:,j]),
              (torch.full((B,),goalx,device=dev),torch.full((B,),goaly,device=dev))]
        outs=[]
        for (tx,ty) in tgts:
            dx=tx-rx; dy=ty-ry; dist=torch.hypot(dx,dy)
            rel=wrap(torch.atan2(dy,dx)-th)
            inview=(dist<CAM_R)&(rel.abs()<CAM_FOV/2)
            b=torch.round((rel+CAM_FOV/2)/CAM_FOV*(CAM_N-1)).long().clamp(0,CAM_N-1)
            depth=(dist/CAM_R).clamp(0,1)
            bins=torch.ones(B,CAM_N,device=dev)
            val=torch.where(inview,depth,torch.ones_like(depth))
            bins.scatter_reduce_(1,b.unsqueeze(1),val.unsqueeze(1),reduce='amin',include_self=True)
            outs.append(bins)
        return torch.cat(outs,1)
    def obs(self,i):
        v=self.rv[:,i:i+1]/SPD; th=self.rth[:,i]
        return torch.cat([v, torch.sin(th).unsqueeze(1), torch.cos(th).unsqueeze(1), self.cam(i)],1)
    def apply_and_phys(self, act0, act1=None):
        """act = [B,2] steer,thr ∈ đã map. Robot0 dùng act0; robot1 thụ-động nếu act1 None."""
        for i,act in [(0,act0),(1,act1)]:
            if act is None:  # thụ-động: giảm tốc
                self.rv[:,i]*=0.9; continue
            steer=act[:,0].clamp(-1,1); thr=((act[:,1]+1)/2).clamp(0,1)
            self.rth[:,i]=self.rth[:,i]+steer*TURN*DT
            self.rv[:,i]=self.rv[:,i]+(thr*SPD-self.rv[:,i])*0.28
        h=DT/NS
        for _ in range(NS):
            for i in range(2):
                self.rx[:,i]=(self.rx[:,i]+torch.cos(self.rth[:,i])*self.rv[:,i]*h).clamp(L+RR,R-RR)
                self.ry[:,i]=(self.ry[:,i]+torch.sin(self.rth[:,i])*self.rv[:,i]*h).clamp(TOP+RR,BOT-RR)
            self._ball(h); self._kick(); self._sep()
        # trọng-tài: bóng đứng im lâu → đá về giữa
        slow=torch.hypot(self.bvx,self.bvy)<9
        self.stuck=torch.where(slow,self.stuck+1,torch.zeros_like(self.stuck))
        fire=self.stuck>90
        if fire.any():
            dx=(L+FW/2)-self.bx; dy=MIDY-self.by; d=torch.hypot(dx,dy).clamp(min=1e-3)
            self.bvx=torch.where(fire,dx/d*150,self.bvx); self.bvy=torch.where(fire,dy/d*150,self.bvy); self.stuck=torch.where(fire,torch.zeros_like(self.stuck),self.stuck)
    def _ball(self,h):
        self.bx=self.bx+self.bvx*h; self.by=self.by+self.bvy*h
        sp=torch.hypot(self.bvx,self.bvy); k=torch.where(sp>0,(sp*0.999-ROLL*h).clamp(min=0)/sp.clamp(min=1e-6),torch.ones_like(sp))
        self.bvx*=k; self.bvy*=k
        top=self.by<TOP+BR; self.by=torch.where(top,torch.full_like(self.by,TOP+BR),self.by); self.bvy=torch.where(top,self.bvy.abs()*WALLB,self.bvy)
        bot=self.by>BOT-BR; self.by=torch.where(bot,torch.full_like(self.by,BOT-BR),self.by); self.bvy=torch.where(bot,-self.bvy.abs()*WALLB,self.bvy)
        inG=(self.by>GY0)&(self.by<GY1)
        left=self.bx<L+BR; goalL=left&inG; bounceL=left&(~inG)
        self.bx=torch.where(bounceL,torch.full_like(self.bx,L+BR),self.bx); self.bvx=torch.where(bounceL,self.bvx.abs()*WALLB,self.bvx)
        right=self.bx>R-BR; goalR=right&inG; bounceR=right&(~inG)
        self.bx=torch.where(bounceR,torch.full_like(self.bx,R-BR),self.bx); self.bvx=torch.where(bounceR,-self.bvx.abs()*WALLB,self.bvx)
        self.blue=self.blue+goalR.float(); self.red=self.red+goalL.float()
        scored=goalR|goalL
        if scored.any(): self.kickoff(torch.nonzero(scored,as_tuple=True)[0])
        self._last_touch=torch.zeros(self.B,device=self.dev)
    def _kick(self):
        for i in range(2):
            dx=self.bx-self.rx[:,i]; dy=self.by-self.ry[:,i]; d=torch.hypot(dx,dy)
            hit=d<(RR+BR); dd=d.clamp(min=1e-6); nx=dx/dd; ny=dy/dd; ov=(RR+BR-d).clamp(min=0)
            self.bx=torch.where(hit,self.bx+nx*ov,self.bx); self.by=torch.where(hit,self.by+ny*ov,self.by)
            along=(self.rv[:,i]*(torch.cos(self.rth[:,i])*nx+torch.sin(self.rth[:,i])*ny)).clamp(min=0)
            power=torch.where(hit,along*KICK,torch.zeros_like(along))
            self.bvx=self.bvx+nx*power; self.bvy=self.bvy+ny*power
            self._last_touch = getattr(self,'_last_touch',torch.zeros(self.B,device=self.dev)) + (hit&(power>40)).float()
    def _sep(self):
        dx=self.rx[:,1]-self.rx[:,0]; dy=self.ry[:,1]-self.ry[:,0]; d=torch.hypot(dx,dy).clamp(min=1e-6)
        mn=2*RR; ov=((mn-d)/2).clamp(min=0); nx=dx/d; ny=dy/d
        self.rx[:,0]=(self.rx[:,0]-nx*ov).clamp(L+RR,R-RR); self.ry[:,0]=(self.ry[:,0]-ny*ov).clamp(TOP+RR,BOT-RR)
        self.rx[:,1]=(self.rx[:,1]+nx*ov).clamp(L+RR,R-RR); self.ry[:,1]=(self.ry[:,1]+ny*ov).clamp(TOP+RR,BOT-RR)

# ---------- SSM policy (per-env params: [B,...]) ----------
def ssm_init(B, dev, base=None):
    def mk(shape, s): return (torch.randn(shape,device=dev)*s)
    if base is None:
        return dict(W_in=mk((H,IN),1/math.sqrt(IN)),b_in=torch.zeros(H,device=dev),a_raw=torch.ones(H,device=dev),
                    W1=mk((HID,H),math.sqrt(2/H)),b1=torch.zeros(HID,device=dev),W2=mk((2,HID),1/math.sqrt(HID)),b2=torch.zeros(2,device=dev))
    return base
def ssm_forward(P, x, h):
    # P: dict per-env [B,...]; x:[B,IN]; h:[B,H] → h', y[B,2]
    u = torch.einsum('bhi,bi->bh', P['W_in'], x) + P['b_in']
    a = torch.sigmoid(P['a_raw'])
    h = a*h + (1-a)*u
    o = torch.relu(torch.einsum('bkh,bh->bk', P['W1'], h) + P['b1'])
    y = torch.tanh(torch.einsum('bmk,bk->bm', P['W2'], o) + P['b2'])
    return h, y

def rollout(P, B, dev, steps, selfplay=False):
    """Chạy B trận với per-env params P; robot0=policy. Trả fitness[B] (thưởng dày)."""
    env=Env(B,dev); h0=torch.zeros(B,H,device=dev); h1=torch.zeros(B,H,device=dev)
    prevbx=env.bx.clone(); prevblue=env.blue.clone()
    prog=torch.zeros(B,device=dev); appr=torch.zeros(B,device=dev); view=torch.zeros(B,device=dev); tou=torch.zeros(B,device=dev)
    for t in range(steps):
        x0=env.obs(0); h0,y0=ssm_forward(P,x0,h0)
        y1=None
        if selfplay:
            x1=env.obs(1); h1,y1=ssm_forward(P,x1,h1)
        env.apply_and_phys(y0,y1)
        gained=(env.blue-prevblue)
        prog=prog+torch.where(gained>0,torch.zeros_like(prog),env.bx-prevbx)
        prevbx=env.bx.clone(); prevblue=env.blue.clone()
        appr=appr - torch.hypot(env.bx-env.rx[:,0],env.by-env.ry[:,0])
        rel=wrap(torch.atan2(env.by-env.ry[:,0],env.bx-env.rx[:,0])-env.rth[:,0])
        view=view+(rel.abs()<1.047).float()
        tou=tou+getattr(env,'_last_touch',torch.zeros(B,device=dev))
    F = 60*env.blue + 0.03*prog + 0.12*tou + 0.02*view + 0.0006*appr
    return F, env.blue.detach(), tou.detach()

def flat(P): return torch.cat([P[k].reshape(-1) for k in KEYS])
KEYS=['W_in','b_in','a_raw','W1','b1','W2','b2']
SHAPES={'W_in':(H,IN),'b_in':(H,),'a_raw':(H,),'W1':(HID,H),'b1':(HID,),'W2':(2,HID),'b2':(2,)}
def unflat(v, dev):
    P={}; o=0
    for k in KEYS:
        n=1
        for d in SHAPES[k]: n*=d
        P[k]=v[o:o+n].reshape(SHAPES[k]).to(dev); o+=n
    return P
def batch_params(theta_flat, eps_flat, dev):
    # theta[D], eps[N2,D] → per-env dict [N2,...]
    N2=eps_flat.shape[0]; full=theta_flat.unsqueeze(0)+eps_flat  # [N2,D]
    P={}; o=0
    for k in KEYS:
        n=1
        for d in SHAPES[k]: n*=d
        P[k]=full[:,o:o+n].reshape((N2,)+SHAPES[k]); o+=n
    return P

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--pop',type=int,default=256); ap.add_argument('--gens',type=int,default=2000)
    ap.add_argument('--steps',type=int,default=1400); ap.add_argument('--sigma',type=float,default=0.08)
    ap.add_argument('--lr',type=float,default=0.05); ap.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--selfplay',action='store_true'); ap.add_argument('--out',default='ssmw.json')
    a=ap.parse_args(); dev=torch.device(a.device); torch.manual_seed(0)
    print(f'device={dev} pop={a.pop} gens={a.gens} steps={a.steps} · IN={IN} H={H} HID={HID}')
    theta=flat(ssm_init(1,dev)).to(dev); D=theta.numel(); print('params D =',D)
    N=a.pop; t0=time.time()
    for gen in range(a.gens):
        eps=torch.randn(N,D,device=dev)
        eps2=torch.cat([eps,-eps],0)                    # mirrored → 2N
        P=batch_params(theta,eps2*a.sigma,dev)
        F,_,_=rollout(P, 2*N, dev, a.steps, a.selfplay)  # [2N]
        Fp,Fm=F[:N],F[N:]
        adv=(Fp-Fm)
        adv=(adv-adv.mean())/(adv.std()+1e-8)
        grad=(eps*adv.unsqueeze(1)).mean(0)             # [D]
        theta=theta + a.lr/a.sigma*grad
        theta*=0.9995
        if (gen+1)%20==0:
            Pe=batch_params(theta,torch.zeros(96,theta.numel(),device=dev),dev)   # đánh-giá theta hiện-tại
            _,gb,gt=rollout(Pe,96,dev,a.steps,a.selfplay)
            print(f'gen {gen+1} · fit {F.mean():.0f} · BÀN/trận {gb.mean():.2f} (max {gb.max():.0f}) · chạm {gt.mean():.1f} · {time.time()-t0:.0f}s', flush=True)
            _save(theta,dev,a.out)
    _save(theta,dev,a.out); print('XONG →',a.out)

def _save(theta,dev,out):
    P=unflat(theta,dev)
    obj={'meta':{'IN':IN,'H':H,'HID':HID,'note':'PyTorch ES self-play SSM (diagonal state-space)'}}
    for k in KEYS: obj[k]=P[k].detach().cpu().tolist()
    json.dump(obj,open(out,'w'))

if __name__=='__main__': main()
