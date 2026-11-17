import time,numpy as np
from nn_aivie.hca.ctx_encoder import load_ctx_encoder
from nn_aivie.hca.knn_lm import load_knn_lm
enc=load_ctx_encoder(); knn=load_knn_lm(enc)
print("knn=",bool(knn),"dev=",knn._dev if knn else None)
ctx=[100,200,300,400,500]
# warm
knn.next_dist(ctx,32002); 
# enc forward alone
t=time.time()
for _ in range(24): _=enc.causal_last(ctx+[i for i in range(600,600+10)])
te=(time.time()-t)/24*1000
# full next_dist
t=time.time()
for i in range(24): _=knn.next_dist(ctx+list(range(600,600+i)),32002)
tf=(time.time()-t)/24*1000
print(f"encoder.causal_last: {te:.1f} ms/bước")
print(f"next_dist toàn phần : {tf:.1f} ms/bước")
print(f"→ ~24 bước thêm    : {tf*24/1000:.2f} s")
