"""Independent contraction controls against native autograd, CPU only."""
import os
os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
import json,hashlib
from pathlib import Path
import torch
from torch import nn
from v46_factorized_jacobian import FactorizedJacobian
from v46_factorized_exact import ExactFactorGram
from structured_gram import structured_gram
torch.set_num_threads(1)

class Tiny(nn.Module):
 def __init__(self,tied=False):
  super().__init__();self.embedding=nn.Embedding(17,8,padding_idx=0);self.norm=nn.LayerNorm(8)
  self.linear=nn.Linear(8,17)
  if tied:self.linear.weight=self.embedding.weight
 def forward(self,x):return self.linear(self.norm(self.embedding(x))).tanh()

rows=[]
for dtype in [torch.float32,torch.float64]:
 for tied in [False,True]:
  for seed in [19,20,21]:
   torch.manual_seed(seed);model=Tiny(tied).to(dtype);x=torch.randint(0,17,(9,5));y=torch.randn(9,5,17,dtype=dtype)
   owner=torch.tensor([0,2,1,1,2,2,0,2,2]);m=3
   def losses():return (model(x)-y).square().mean((1,2))
   with FactorizedJacobian(model,owner,m,compute_dtype=dtype) as cache:cache.capture(losses())
   l=losses();params=tuple(model.parameters())
   G=torch.stack([torch.cat([g.flatten() for g in torch.autograd.grad(l[owner==j].mean(),params,retain_graph=True)]) for j in range(m)])
   native=G.double()@G.double().T
   direct=ExactFactorGram(cache).gram()
   direct_error=float(torch.linalg.vector_norm(direct.double()-native)/torch.linalg.vector_norm(native))
   if direct_error>2e-5:raise ArithmeticError('Fast Direct native audit failed')
   for order in ['parameter','token','auto']:
    K,census=structured_gram(cache,order,scratch_bytes=4096)
    error=float(torch.linalg.vector_norm(K.double()-native)/torch.linalg.vector_norm(native))
    row={'dtype':str(dtype),'tied':tied,'seed':seed,'order':order,'native_relative_error':error,'direct_relative_error':direct_error}
    rows.append(row);print(json.dumps(row),flush=True)
    if error>2e-5:raise ArithmeticError('Structured control native audit failed')
root=Path(__file__).parent
(root/'structured_cpu_audit.json').write_text(json.dumps({'complete':True,'rows':rows,
 'source_sha256':hashlib.sha256((root/'structured_gram.py').read_bytes()).hexdigest()},indent=2))
