"""Shared answer-only loss, data packing, and audited QP for rebuttal runs."""
import hashlib,json
from pathlib import Path
import numpy as np
import torch
from v46_supervised_head import prepare_supervised_positions,supervised_token_outputs,per_example_cross_entropy
from simplex_qp import qp

def load_data(root):
 root=Path(root);manifest=json.loads((root/'data_manifest.json').read_text());p=root/'official_split_pool.pt'
 if hashlib.sha256(p.read_bytes()).hexdigest()!=manifest['pool_sha256']:raise ValueError('Language pool hash mismatch')
 data=torch.load(p,map_location='cpu',weights_only=True)
 seen=set()
 for split in ['train','validation','test']:
  current={r['prompt_sha256'] for rows in data[split].values() for r in rows}
  if current&seen:raise ValueError('Truncated prompt overlap across dataset splits')
  seen.update(current)
 return data,manifest

def pack(rows,owners,m,eos,device='cuda'):
 width=max(len(r['ids']) for r in rows);n=len(rows)
 x=torch.full((n,width),eos,dtype=torch.long);attention=torch.zeros_like(x);labels=torch.full_like(x,-100)
 for i,r in enumerate(rows):
  x[i,:len(r['ids'])]=torch.tensor(r['ids']);attention[i,:len(r['ids'])]=1;labels[i,:len(r['labels'])]=torch.tensor(r['labels'])
 prepared=prepare_supervised_positions(labels)
 return {'inputs':x.to(device),'attention':attention.to(device),'labels':labels.to(device),
  'prepared':tuple(t.to(device) for t in prepared),'owner':torch.tensor(owners,device=device),'M':m,
  'identities':[r['identity'] for r in rows]}

def make_batch(data,tasks,eos,seed,step=0,batch_size=32,objectives=8,device='cuda'):
 if objectives%len(tasks) or batch_size%objectives:raise ValueError('Task/stratum count must divide batch size')
 strata=objectives//len(tasks);per_objective=batch_size//objectives;rows=[];owners=[]
 for j,t in enumerate(tasks):
  rng=np.random.default_rng(seed*100000+step+int(hashlib.sha256(t.encode()).hexdigest()[:8],16))
  chosen=rng.choice(len(data[t]),strata*per_objective,replace=False)
  rows.extend(data[t][int(i)] for i in chosen)
  owners.extend(j*strata+k//per_objective for k in range(len(chosen)))
 return pack(rows,owners,objectives,eos,device)

def losses(model,b):
 logits,targets,_=supervised_token_outputs(model,b['inputs'],b['attention'],b['labels'],prepared=b['prepared'])
 return per_example_cross_entropy(logits,targets)

def task_losses(model,b):
 values=losses(model,b);counts=torch.bincount(b['owner'],minlength=b['M']).to(values.dtype)
 return values.new_zeros(b['M']).index_add(0,b['owner'],values)/counts

def solve(K):
 fit=qp(K,tolerance=2e-12,max_iterations=3000)
 if not fit['converged']:raise ArithmeticError('Unresolved MGDA simplex QP')
 return fit

def native_direction(model,b,w):
 params={n:p for n,p in model.named_parameters() if p.requires_grad}
 l=task_losses(model,b);weight=torch.as_tensor(w,device=l.device,dtype=l.dtype)
 g=torch.autograd.grad(l@weight,tuple(params.values()))
 return {n:t.detach() for n,t in zip(params,g)}

def save(path,obj):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(obj,indent=2,allow_nan=False));tmp.replace(path)
