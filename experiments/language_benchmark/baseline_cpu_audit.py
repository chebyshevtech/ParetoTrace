"""No GPU use: validate the official Ghost column adapter against autograd."""
import os
os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
import hashlib,json,subprocess,sys
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from v46_factorized_jacobian import FactorizedJacobian
from v46_factorized_exact import ExactFactorGram
from ghost_adapter import load_upstream,ghost_columns,GHOST_COMMIT

root=Path(__file__).parent;ghost=root/'GhostSuite'
if not ghost.exists():subprocess.run(['git','clone','--depth','1','https://github.com/Jiachen-T-Wang/GhostSuite.git',str(ghost)],check=True)
head=subprocess.check_output(['git','-C',str(ghost),'rev-parse','HEAD'],text=True).strip()
if head!=GHOST_COMMIT:
 subprocess.run(['git','-C',str(ghost),'fetch','--depth','1','origin',GHOST_COMMIT],check=True)
 subprocess.run(['git','-C',str(ghost),'checkout','--detach',GHOST_COMMIT],check=True)
upstream=load_upstream(ghost);torch.set_num_threads(1)

class Tiny(nn.Module):
 def __init__(self,tied=False):
  super().__init__();self.embedding=nn.Embedding(17,8,padding_idx=0);self.norm=nn.LayerNorm(8)
  self.linear=nn.Linear(8,17)
  if tied:self.linear.weight=self.embedding.weight
 def forward(self,x):return self.linear(self.norm(self.embedding(x))).tanh()

rows=[]
for dtype in [torch.float32,torch.float64]:
 for tied in [False,True]:
  torch.manual_seed(19);model=Tiny(tied).to(dtype);x=torch.randint(0,17,(12,5));y=torch.randn(12,5,17,dtype=dtype)
  owner=torch.arange(3).repeat_interleave(4)
  def losses():return (model(x)-y).square().mean((1,2))
  with FactorizedJacobian(model,owner,3,compute_dtype=dtype) as cache:cache.capture(losses())
  H,census=ghost_columns(cache,upstream);K=ExactFactorGram(cache).gram();l=losses().reshape(3,4).mean(1)
  params=tuple(model.parameters());G=torch.stack([torch.cat([g.flatten() for g in torch.autograd.grad(li,params,retain_graph=True)]) for li in l])
  native=G.double()@G.double().T
  error=float(torch.linalg.vector_norm(H.double()-native)/torch.linalg.vector_norm(native))
  row={'dtype':str(dtype),'tied':tied,'native_relative_error':error,
       'direct_relative_error':float(torch.linalg.vector_norm(K.double()-native)/torch.linalg.vector_norm(native)),**census}
  rows.append(row);print(json.dumps(row),flush=True)
  if error>2e-5:raise ArithmeticError('Ghost adapter native audit failed')
result={'ghost_commit':GHOST_COMMIT,'rows':rows,'complete':True,
 'sources':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in root.glob('*.py')}}
(root/'baseline_cpu_audit.json').write_text(json.dumps(result,indent=2))
