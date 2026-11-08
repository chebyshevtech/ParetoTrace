"""Exact full-coverage recovery for shared-input recurrent graph models.

Linear layers use task-axis adjoints and bounded parameter-row contractions.
Repeated occurrences are summed before forming products. Unsupported parameters
(e.g. fused GRU/LSTM weights) use an explicit native-AD fallback. No all-layer
factorization or speed advantage is asserted; profile against pure native AD.

V2 acquires Linear adjoints and unsupported parameter gradients in ONE reverse
traversal per task block. A singleton uses ordinary VJP, permitting fused RNN
backward where batched VJP has no rule. Contraction arithmetic is unchanged.

Scope: every use of a covered parameter must occur through the registered
Linear module. Functional reuse of its weight outside that module is not
supported. Captured inputs/outputs must not be modified in place. Official
QM9 satisfies these conditions; each prospective configuration is audited.
"""
import math
import torch
from torch import nn

class SharedMultioutputGram:
 def __init__(self,module,task_chunk=2,row_chunk=64):
  self.module=module;self.params={id(p):p for p in module.parameters() if p.requires_grad}
  self.names={id(p):n for n,p in module.named_parameters()};self.task_chunk=task_chunk;self.row_chunk=row_chunk
  self.entries=[];self.handles=[];self.covered=set();self.storage_versions=[]
  self.unsupported={id(p) for mod in module.modules() if not isinstance(mod,nn.Linear)
                    for p in mod.parameters(recurse=False) if id(p) in self.params}
 def __enter__(self):
  def hook(mod,args,out):
   if not isinstance(out,torch.Tensor):raise TypeError('Linear output must be one tensor')
   self.entries.append((mod,args[0].detach(),out))
   self.storage_versions.extend([(args[0],args[0]._version),(out,out._version)])
   for p in (mod.weight,mod.bias):
    if p is not None and id(p) in self.params and id(p) not in self.unsupported:self.covered.add(id(p))
  for mod in self.module.modules():
   if isinstance(mod,nn.Linear) and any(id(p) in self.params for p in mod.parameters(recurse=False)):
    self.handles.append(mod.register_forward_hook(hook))
  return self
 def __exit__(self,*args):
  for h in self.handles:h.remove()
 def capture(self,losses):
  if any(t._version!=version for t,version in self.storage_versions):
   raise RuntimeError('In-place modification of captured Linear input/output unsupported')
  self.losses=losses;self.m=len(losses);eye=torch.eye(self.m,device=losses.device,dtype=losses.dtype)
  self.fallback=[p for pid,p in self.params.items() if pid not in self.covered]
  collected=[out.new_empty((self.m,)+tuple(out.shape)) for _,_,out in self.entries]
  self.fallback_rows=[p.new_empty((self.m,)+tuple(p.shape)) for p in self.fallback]
  targets=tuple(e[2] for e in self.entries)+tuple(self.fallback)
  if targets:
   for start in range(0,self.m,self.task_chunk):
    stop=min(start+self.task_chunk,self.m)
    if stop-start==1:
     values=tuple(g.unsqueeze(0) for g in torch.autograd.grad(losses[start],targets,retain_graph=True))
    else:
     values=torch.autograd.grad(losses,targets,eye[start:stop],is_grads_batched=True,retain_graph=True)
    for rows,g in zip(collected+self.fallback_rows,values):rows[start:stop].copy_(g.detach())
    del values,g
  self.adjoints=collected
 def gram(self,dtype=torch.float64):
  out=torch.zeros(self.m,self.m,device=self.losses.device,dtype=dtype)
  for pid in self.covered:
   param=self.params[pid];terms=[]
   for (mod,x,_),adj in zip(self.entries,self.adjoints):
    if mod.weight is param:terms.append(('weight',x,adj))
    if mod.bias is param:terms.append(('bias',x,adj))
   if param.ndim not in (1,2):raise ValueError('Unexpected Linear parameter shape')
   for start in range(0,param.shape[0],self.row_chunk):
    stop=min(start+self.row_chunk,param.shape[0]);shape=(self.m,stop-start)+tuple(param.shape[1:])
    gradient=out.new_zeros(shape)
    for kind,x,adj in terms:
     b=adj.reshape(self.m,-1,adj.shape[-1])[:,:,start:stop].to(dtype)
     if kind=='weight':gradient+=b.transpose(1,2)@x.reshape(-1,x.shape[-1]).to(dtype)
     else:gradient+=b.sum(1)
    flat=gradient.reshape(self.m,-1);out.addmm_(flat,flat.T)
    del gradient,flat,b
  if self.fallback:
   for tensor in self.fallback_rows:
    g=tensor.reshape(self.m,-1).to(dtype);out.addmm_(g,g.T);del g
  self.census={'factorized_parameter_elements':sum(self.params[p].numel() for p in self.covered),
    'native_fallback_parameter_elements':sum(p.numel() for p in self.fallback),
    'native_fallback_names':[self.names[id(p)] for p in self.fallback],
    'linear_occurrences':len(self.entries),'task_chunk':self.task_chunk,'row_chunk':self.row_chunk,
    'task_axis_capture_passes':math.ceil(self.m/self.task_chunk) if self.entries else 0,
    'native_fallback_passes':0,
    'joint_adjoint_and_fallback_passes':math.ceil(self.m/self.task_chunk),
    'native_fallback_acquisition':'same traversal as Linear adjoints; no additional reverse pass',
    'singleton_reverse_mode':'ordinary native VJP'}
  return out
