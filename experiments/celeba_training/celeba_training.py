"""Paired-seed CelebA training with actual shared-batch Direct.

Preflight/development and confirmatory seeds are explicitly separate. This
script never selects a learning rate or stops seeds based on test outcomes.
Run one configuration per process; checkpoints make interruptions resumable.
"""
import os
os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
import argparse, gc, hashlib, json, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet18,ResNet18_Weights
from preflight import SharedCapture
from simplex_qp import qp

class NonfiniteTrainingError(ArithmeticError):pass
RUN_CONTEXT={}

def atomic_json(path,obj):
 tmp=path.with_suffix('.json.tmp');tmp.write_text(json.dumps(obj,indent=2,allow_nan=False));tmp.replace(path)

def solve(K):
 K=np.asarray(K,dtype=np.float64)
 if not np.isfinite(K).all():raise NonfiniteTrainingError("Nonfinite training Gram")
 K=(K+K.T)/2
 scale=max(float(np.abs(K).max()),1e-30)
 fit=qp(K,tolerance=2e-14,max_iterations=3000)
 if not fit['converged']:raise ArithmeticError('simplex active-set QP unresolved')
 w=fit['weights']
 if not np.isfinite(w).all() or w.min() < -1e-12 or abs(w.sum()-1)>1e-10:raise ArithmeticError('QP weights infeasible')
 u=K@w;q=float(w@u);lower=max(0.,2*float(u.min())-q)
 if not np.isfinite([q,lower]).all():raise ArithmeticError('QP objective unresolved')
 if q-lower>1e-6*max(q,scale*1e-10):raise ArithmeticError('QP dual gap unresolved')
 return w,{'objective':q,'lower':lower,'gap':q-lower}

class Model(nn.Module):
 def __init__(self):
  super().__init__();self.trunk=resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
  self.trunk.fc=nn.Identity();self.head=nn.Linear(512,40)
  for m in self.trunk.modules():
   if isinstance(m,nn.BatchNorm2d):
    for p in m.parameters(recurse=False):p.requires_grad_(False)
   if isinstance(m,nn.ReLU):m.inplace=False
 def forward(self,x):return self.head(self.trunk(x))

def require_finite(tensors,label):
 values=[torch.isfinite(t.detach()).all() for t in tensors]
 if values and not bool(torch.stack(values).all()):raise NonfiniteTrainingError("Nonfinite "+label)

def load_split(root,split):
 meta=json.loads((root/'celeba_manifest.json').read_text())
 if not meta['complete'] or split not in meta['splits']:raise ValueError('Incomplete dataset manifest')
 path=root/('celeba_'+split+'.pt')
 if hashlib.sha256(path.read_bytes()).hexdigest()!=meta['splits'][split]['sha256']:
  raise ValueError('Dataset tensor SHA256 mismatch: '+split)
 d=torch.load(path,map_location='cpu',weights_only=True)
 if d['revision']!=meta['revision'] or d['split']!=split or d['attrs']!=meta['attributes']:
  raise ValueError('Dataset tensor metadata mismatch')
 if len(d['X'])!=meta['splits'][split]['images'] or d['Y'].shape!=(len(d['X']),40):
  raise ValueError('Dataset shape mismatch')
 return d['X'],d['Y']

def batch(X,Y,idx):
 x=X[idx].cuda().float()/255
 mean=torch.tensor([.485,.456,.406],device='cuda')[None,:,None,None]
 std=torch.tensor([.229,.224,.225],device='cuda')[None,:,None,None]
 return (x-mean)/std,Y[idx].cuda()

@torch.no_grad()
def evaluate(model,X,Y):
 model.eval();correct=torch.zeros(40,device='cuda');loss=torch.zeros(40,device='cuda',dtype=torch.float64)
 tp=torch.zeros(40,device='cuda');tn=tp.clone();pos=tp.clone();neg=tp.clone()
 for st in range(0,len(X),256):
  x,y=batch(X,Y,slice(st,st+256));z=model(x);require_finite([z],'evaluation logits');pred=z>0;truth=y>0
  correct+=(pred==truth).sum(0);loss+=F.binary_cross_entropy_with_logits(z,y,reduction='none').double().sum(0)
  tp+=(pred&truth).sum(0);tn+=((~pred)&(~truth)).sum(0);pos+=truth.sum(0);neg+=(~truth).sum(0)
 acc=(correct/len(X)).cpu().numpy();bce=(loss/len(X)).cpu().numpy();ba=(.5*(tp/pos+tn/neg)).cpu().numpy()
 return {'n':len(X),'mean_accuracy':float(acc.mean()),'worst_accuracy':float(acc.min()),
   'per_attribute_accuracy':acc.tolist(),'mean_bce':float(bce.mean()),'worst_bce':float(bce.max()),
   'per_attribute_bce':bce.tolist(),'mean_balanced_accuracy':float(ba.mean()),
   'worst_balanced_accuracy':float(ba.min()),'per_attribute_balanced_accuracy':ba.tolist()}

def native_gram(losses,params):
 G=torch.stack([torch.cat([g.detach().reshape(-1) for g in torch.autograd.grad(l,params,retain_graph=True)]) for l in losses])
 K=torch.zeros(40,40,device='cuda',dtype=torch.float64)
 for st in range(0,G.shape[1],65536):
  g=G[:,st:st+65536].double();K.addmm_(g,g.T)
 return K

def main():
 p=argparse.ArgumentParser();p.add_argument('--arm',choices=['direct','mgda_ub','uniform','serial'],required=True)
 p.add_argument('--attempt-id',default=None)
 p.add_argument('--seed',type=int,required=True);p.add_argument('--lr',type=float,default=.02)
 p.add_argument('--steps',type=int,default=1200);p.add_argument('--batch',type=int,default=128)
 p.add_argument('--phase',choices=['preflight','development','confirmatory'],default='preflight')
 p.add_argument('--data',default='/content/rebuttal/data');p.add_argument('--out',default='/content/rebuttal/runs')
 p.add_argument('--checkpoint-every',type=int,default=100);p.add_argument('--evaluate-every',type=int,default=300)
 args=p.parse_args();cfg=vars(args).copy();cfg.pop("attempt_id")
 if args.phase=='preflight' and args.seed not in range(8001,8100):raise ValueError('preflight seeds reserved as 8001..8099')
 if args.phase=='development' and args.seed!=9001:raise ValueError('development seed fixed at 9001')
 if args.phase=='confirmatory' and args.seed not in range(1001,1021):raise ValueError('report seeds fixed at 1001..1020')
 root=Path(args.out);root.mkdir(parents=True,exist_ok=True)
 stem=f'{args.phase}_{args.arm}_seed{args.seed}_lr{args.lr:g}'
 dest=root/(stem+'.json');ckpt=root/(stem+'.pt')
 RUN_CONTEXT.update(phase=args.phase,arm=args.arm,seed=args.seed,lr=args.lr,step=None,failure_path=str(root/(stem+'.failure.json')))
 cfg['source_sha256']={n:hashlib.sha256(Path(__file__).with_name(n).read_bytes()).hexdigest() for n in ['celeba_training.py','preflight.py','simplex_qp.py']}
 cfg['data_sha256']=hashlib.sha256((Path(args.data)/'celeba_manifest.json').read_bytes()).hexdigest()
 cfg_hash=hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest()
 RUN_CONTEXT.update(attempt_id=args.attempt_id,config_hash=cfg_hash,source_sha256=cfg['source_sha256'],data_sha256=cfg['data_sha256'])
 torch.set_num_threads(1);torch.manual_seed(args.seed);np.random.seed(args.seed)
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 torch.backends.cudnn.benchmark=False
 model=Model().cuda().eval();params=tuple(p for p in model.trunk.parameters() if p.requires_grad)
 opt=torch.optim.SGD(model.parameters(),lr=args.lr,momentum=.9)
 X,Y=load_split(Path(args.data),'train');VX,VY=load_split(Path(args.data),'valid')
 result={'config':cfg,'config_hash':cfg_hash,'gpu':torch.cuda.get_device_name(),
  'torch':torch.__version__,'protocol':'shared_trunk_40_attributes_official_splits_v1',
  'batchnorm':'fixed pretrained eval statistics; affine frozen',
  'head_update':'own unweighted mean BCE gradient, same for every arm',
  'guarantee_scope':'raw shared-trunk MGDA direction; momentum update not asserted common descent',
  'rows':[],'evaluations':[],'audits':[],'complete':False,'update_seconds':0.,'wall_seconds':0.,
  'process_peak_bytes':torch.cuda.max_memory_allocated()}
 start_step=0
 if ckpt.exists():
  saved=torch.load(ckpt,map_location='cpu',weights_only=False)
  if saved['config_hash']!=cfg_hash:raise ValueError('checkpoint configuration/source mismatch')
  model.load_state_dict(saved['model']);opt.load_state_dict(saved['optimizer'])
  result=saved['result'];start_step=saved['next_step']
  if result.get('invalid'):
   raise RuntimeError('Checkpoint is quarantined after a failed fidelity audit; do not resume it')
  if result.get('complete'):
   atomic_json(dest,result)
   print('ALREADY_COMPLETE',str(dest),flush=True);return
 begin=time.perf_counter();previous_wall=result['wall_seconds']
 def save(next_step):
  result['wall_seconds']=previous_wall+time.perf_counter()-begin
  tmp=ckpt.with_suffix('.pt.tmp');torch.save({'config_hash':cfg_hash,'model':model.state_dict(),
    'optimizer':opt.state_dict(),'result':result,'next_step':next_step},tmp);tmp.replace(ckpt)
  atomic_json(dest,result)
 for step in range(start_step,args.steps):
  RUN_CONTEXT["step"]=step
  idx=torch.randint(len(X),(args.batch,),generator=torch.Generator().manual_seed(args.seed*100000+step))
  torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();tick=time.perf_counter()
  x,y=batch(X,Y,idx);opt.zero_grad(set_to_none=True);meta={};K=None
  if args.arm=='direct':
   with SharedCapture(model.trunk) as cap:
    losses=F.binary_cross_entropy_with_logits(model(x),y,reduction='none').mean(0)
   cap.capture(losses);K=cap.gram(torch.float64);del cap
   w,meta=solve(K.detach().cpu().numpy())
  else:
   z=model.trunk(x);logits=model.head(z);losses=F.binary_cross_entropy_with_logits(logits,y,reduction='none').mean(0)
   if args.arm=='mgda_ub':
    with torch.no_grad():
     delta=(torch.sigmoid(logits)-y)/args.batch;W=model.head.weight.double()
     K=(delta.double().T@delta.double())*(W@W.T)
    w,meta=solve(K.cpu().numpy())
   elif args.arm=='serial':K=native_gram(losses,params);w,meta=solve(K.cpu().numpy())
   else:w=np.full(40,1/40)
  if not bool(torch.isfinite(losses).all()):raise NonfiniteTrainingError("Nonfinite minibatch losses")
  # Heads remain unweighted in every arm, as in standard shared-gradient MTL.
  hg=torch.autograd.grad(losses.sum(),tuple(model.head.parameters()),retain_graph=True)
  wt=torch.as_tensor(w,device='cuda',dtype=losses.dtype)
  (losses*wt).sum().backward()
  for par,g in zip(model.head.parameters(),hg):par.grad=g
  require_finite([p.grad for p in model.parameters() if p.grad is not None],'gradients')
  opt.step()
  require_finite(list(model.parameters())+[v for state in opt.state.values() for v in state.values() if isinstance(v,torch.Tensor)],'parameters or optimizer state')
  torch.cuda.synchronize();elapsed=time.perf_counter()-tick
  result['update_seconds']+=elapsed
  result['rows'].append({'step':step,'seconds':elapsed,'peak_bytes':torch.cuda.max_memory_allocated(),
      'mean_batch_bce':float(losses.mean()),'worst_batch_bce':float(losses.max()),'weights':w.tolist(),**meta})
  # Independent, post-decision audit at new model state; never changes weights.
  if step in {0,args.steps//2,args.steps-1} and args.arm=='direct':
   del losses,K;gc.collect();torch.cuda.empty_cache();audit_start=time.perf_counter()
   with SharedCapture(model.trunk) as cap:
    al=F.binary_cross_entropy_with_logits(model(x),y,reduction='none').mean(0)
   cap.capture(al);Ka=cap.gram(torch.float64);del cap
   Kn=native_gram(al,params);wa,ma=solve(Ka.cpu().numpy());wn,mn=solve(Kn.cpu().numpy())
   ua=Kn@torch.as_tensor(wa,device='cuda',dtype=torch.float64);qa=float(torch.as_tensor(wa,device='cuda',dtype=torch.float64)@ua)
   gram_error=float(torch.linalg.vector_norm(Ka-Kn)/torch.linalg.vector_norm(Kn).clamp_min(1e-100))
   scale=max(float(Kn.abs().max()),1e-100)
   resolved=mn['lower']>scale*1e-10
   factor=qa/mn['lower'] if resolved else None
   objective_pass=(factor<=1.001 if resolved else qa-mn['objective']<=1e-7*scale)
   finite=bool(np.isfinite([gram_error,qa,float(ua.min())]).all())
   result['audits'].append({'step_after':step,'gram_relative_error':gram_error,'native_objective_factor_upper':factor,
        'objective_audit_scope':'relative' if resolved else 'absolute near stationarity; no relative certificate',
        'native_objective_excess':qa-mn['objective'],'native_min_response':float(ua.min()),'seconds':time.perf_counter()-audit_start})
   if not finite or gram_error>2e-5 or not objective_pass:
    result['invalid']={'reason':'Direct native fidelity audit failed','step_after':step}
    save(step+1);raise ArithmeticError('Direct native fidelity audit failed; run quarantined')
   del al,Ka,Kn
  if (step+1)%args.evaluate_every==0 or step+1==args.steps:
   metrics=evaluate(model,VX,VY);result['evaluations'].append({'step':step+1,'split':'validation','metrics':metrics,
     'update_seconds':result['update_seconds'],'wall_seconds':previous_wall+time.perf_counter()-begin})
  result['process_peak_bytes']=max(result['process_peak_bytes'],torch.cuda.max_memory_allocated())
  if (step+1)%args.checkpoint_every==0 or step+1==args.steps:
   save(step+1);print(json.dumps({'arm':args.arm,'seed':args.seed,'step':step+1,'updates_s':result['update_seconds'],
     'median_step_s':float(np.median([r['seconds'] for r in result['rows'][-100:]]))}),flush=True)
 if args.phase=='confirmatory':
  TX,TY=load_split(Path(args.data),'test');result['test']=evaluate(model,TX,TY)
 result['process_peak_bytes']=max(result['process_peak_bytes'],torch.cuda.max_memory_allocated())
 result['complete']=True;save(args.steps);print('COMPLETE',str(dest),flush=True)

if __name__=='__main__':
 try:main()
 except NonfiniteTrainingError as exc:
  record=dict(RUN_CONTEXT,kind='nonfinite_training',error=str(exc),complete=False)
  if RUN_CONTEXT.get('failure_path'):atomic_json(Path(RUN_CONTEXT['failure_path']),record)
  print('NONFINITE_TRIAL',json.dumps(record),flush=True)
  raise

