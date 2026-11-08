"""Dual-reference MGDA benchmark with accelerated independent audit reductions.

The adapted structured controls use established identities, not unmodified
Neural Tangents. Ghost columns call pinned upstream kernels. No 14.9x claim
against these methods is assumed. Every method includes acquisition, solve and
reconstruction; native fidelity audits occur outside the timed boundary.
"""
import os
os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
import argparse,gc,hashlib,json,time,traceback,subprocess,inspect
from pathlib import Path
import numpy as np
import torch
import transformers
from transformers import AutoModelForCausalLM,AutoTokenizer
from language_common import load_data,make_batch,losses,task_losses,solve,native_direction,save as strict_save
from v46_factorized_jacobian import FactorizedJacobian
from v46_factorized_exact import ExactFactorGram
from paretotrace_online_factor import OnlineFactorGram
from ghost_adapter import load_upstream,GHOST_COMMIT,ghost_columns
from structured_gram import structured_gram
import v46_supervised_head
from attention_precision import shared_attention_precision
from native_language_dual_reference_gpu_v6 import native_reference,audit_directions,classify_contracts,gram64
from language_tokenizer_contract import load_tokenizer

METHODS=['direct_cached','direct_online','ghost_columns','structured_auto','structured_parameter','structured_token','native_serial','native_batched']
START_CONTEXT={}

def save(path,value):
 invalid=[]
 def clean(obj,where=''):
  if isinstance(obj,float) and not np.isfinite(obj):invalid.append(where);return None
  if isinstance(obj,dict):return {k:clean(v,where+'/'+str(k)) for k,v in obj.items()}
  if isinstance(obj,list):return [clean(v,where+'/'+str(i)) for i,v in enumerate(obj)]
  return obj
 copied=clean(value)
 if invalid:copied['nonfinite_diagnostic_fields_replaced_by_null']=invalid
 strict_save(path,copied)

def sync():torch.cuda.synchronize()
def cleanup():gc.collect();torch.cuda.empty_cache();sync()
def timed(f):
 sync();start=time.perf_counter();v=f();sync();return v,time.perf_counter()-start

def run_method(model,b,method,upstream,dtype=torch.float32,row_chunk=256,scratch_mib=128,native_chunk=4,gram_precision="fp64_tiled",reconstruction="operator",reference=None,diagnostics=None):
 if reconstruction not in ('operator','weighted_native'):raise ValueError('Unknown reconstruction')
 if reference is None:raise ValueError('Independent retained task table required')
 if gram_precision not in ('fp32','fp64_tiled'):raise ValueError('Unknown native contraction precision')
 cleanup();torch.cuda.reset_peak_memory_stats();base=torch.cuda.memory_allocated();sync();start=time.perf_counter();times={};metadata={}
 if diagnostics is None:diagnostics={}
 diagnostics.update(stage='capture',times=times,peak_reset=True,base_allocated_bytes=base)
 params={n:p for n,p in model.named_parameters() if p.requires_grad}
 if method.startswith('native_'):
  def materialize():
   l=task_losses(model,b)
   table=l.new_empty((b['M'],sum(p.numel() for p in params.values())))
   chunk=1 if method=='native_serial' else native_chunk;eye=torch.eye(b['M'],device=l.device,dtype=l.dtype)
   for j in range(0,len(l),chunk):
    stop=min(j+chunk,len(l))
    grads=(torch.autograd.grad(l[j],tuple(params.values()),retain_graph=stop<len(l)) if chunk==1 else
           torch.autograd.grad(l,tuple(params.values()),eye[j:stop],retain_graph=stop<len(l),is_grads_batched=True))
    offset=0
    for param,g in zip(params.values(),grads):
     n=param.numel();table[j:stop,offset:offset+n].copy_(g.detach().reshape(stop-j,n));offset+=n
    del grads,g
   return table
  G,times['capture_seconds']=timed(materialize)
  diagnostics['stage']='gram'
  K,times['gram_seconds']=timed(lambda:gram64(G) if gram_precision=='fp64_tiled' else G@G.T)
 elif method=='direct_online':
  def online():
   with OnlineFactorGram(model,b['owner'],b['M'],compute_dtype=dtype,scratch_bytes=scratch_mib*2**20,row_chunk=row_chunk) as obj:obj.capture(losses(model,b))
   return obj.gram(),obj.last_plan
  (K,metadata['contraction_plan']),times['capture_and_gram_seconds']=timed(online)
 else:
  def capture():
   with FactorizedJacobian(model,b['owner'],b['M'],compute_dtype=dtype,compact_zero_rows=True) as obj:obj.capture(losses(model,b))
   return obj
  cache,times['capture_seconds']=timed(capture)
  diagnostics['stage']='gram'
  if method=='direct_cached':
   engine=ExactFactorGram(cache,row_chunk=row_chunk,scratch_bytes=scratch_mib*2**20)
   K,times['gram_seconds']=timed(engine.gram);metadata['contraction_plan']=engine.last_plan
  elif method=='ghost_columns':
   (K,metadata['census']),times['gram_seconds']=timed(lambda:ghost_columns(cache,upstream))
  else:
   (K,metadata['contraction_plan']),times['gram_seconds']=timed(lambda:structured_gram(cache,method.removeprefix('structured_'),scratch_mib*2**20,row_chunk))
 def fit():
  matrix=K.double().cpu().numpy();matrix=(matrix+matrix.T)/2
  return matrix,solve(matrix)
 diagnostics['stage']='transfer_and_qp'
 (matrix,solution),times['transfer_and_qp_seconds']=timed(fit);w=solution['weights']
 retired=False
 if reconstruction=='weighted_native' or method=='direct_online':
  if method.startswith('native_'):del G
  elif method!='direct_online':
   if method=='direct_cached':del engine
   del cache
  retired=True
  reconstruct=lambda:native_direction(model,b,w)
 elif method.startswith('native_'):
  def reconstruct():
   v=torch.as_tensor(w,device=G.device,dtype=G.dtype)@G;offset=0;out={}
   for n,p in params.items():out[n]=v[offset:offset+p.numel()].reshape_as(p);offset+=p.numel()
   return out
 else:reconstruct=lambda:{n:g.float() for n,g in cache.direction(w).items()}
 diagnostics['stage']='reconstruction'
 direction,times['reconstruction_seconds']=timed(reconstruct)
 # Retire every method's table/cache inside the same measured boundary.
 del reconstruct
 if not retired:
  if method.startswith('native_'):del G
  else:
   if method=='direct_cached':del engine
   del cache
 sync();elapsed=time.perf_counter()-start;peak=torch.cuda.max_memory_allocated()
 diagnostics.update(stage='independent_audit',timed_recovery_complete=True,recovery_seconds=elapsed,recovery_peak_allocated_bytes=peak)
 row={'method':method,'seconds':elapsed,'times':times,'peak_allocated_bytes':peak,
      'configuration':{'row_chunk':row_chunk,'scratch_mib':scratch_mib,'native_chunk':native_chunk,'gram_precision':gram_precision if method.startswith('native_') else None,'reconstruction':reconstruction},
      'incremental_peak_bytes':peak-base,
      'weights':w.tolist(),'gram':matrix.tolist(),'objective':solution['objective'],
      'qp_gap':solution['dual_gap'],'metadata':metadata,'timed_recovery_complete':True,
      'execution_variant':'online retirement + second native reconstruction' if method=='direct_online' else 'native task table' if method.startswith('native_') else 'cached contraction control'}
 # Audit memory is separate. No retained GPU factor/table survives into it.
 del K
 try:
  native,audit_time=timed(lambda:native_direction(model,b,w))
  table,Kref,fitref=reference
  extra,audit_dual_time=timed(lambda:audit_directions(direction,native,params,table,w,Kref,fitref))
  row.update(extra,audit_weighted_seconds=audit_time,audit_dual_seconds=audit_dual_time)
  classify_contracts(row,Kref,fitref)
 except Exception:
  row.update(passes_table_contract=False,passes_weighted_contract=False,passes_joint_contract=False,
      audit_failure=traceback.format_exc())
 return row

CONTRACTS=('table','weighted','joint')
DEV=5000
HELDOUT=(5011,5012,5013)

def configurations(method,quick=False):
 if method.startswith('direct_') or method.startswith('structured_'):
  base=[{'row_chunk':r,'scratch_mib':s} for r,s in [(64,32),(256,32),(64,128),(256,128)]]
 elif method=='native_batched':base=[{'native_chunk':n,'gram_precision':p} for p in ['fp64_tiled','fp32'] for n in [2,4,8]]
 elif method=='native_serial':base=[{'gram_precision':p} for p in ['fp64_tiled','fp32']]
 else:base=[{}]
 if quick:base=base[:1]
 choices=['weighted_native'] if method=='direct_online' else ['operator','weighted_native']
 return [{**cfg,'reconstruction':r} for cfg in base for r in choices]

def benchmark(precision):
 p=argparse.ArgumentParser();p.add_argument('--model',default='EleutherAI/pythia-160m-deduped')
 p.add_argument('--root',type=Path,default=Path('/content/rebuttal/language'));p.add_argument('--output',type=Path,required=True)
 p.add_argument('--objectives',default='8,32');p.add_argument('--repeats',type=int,default=3)
 p.add_argument('--mode',choices=['preflight','reporting'],default='reporting');p.add_argument('--quick',action='store_true')
 a=p.parse_args();ms=list(map(int,a.objectives.split(',')))
 if not ms or len(set(ms))!=len(ms) or any(m not in (8,32) for m in ms) or a.repeats<1:p.error('Valid distinct objectives and positive repeats required')
 if a.quick and a.mode!='preflight':p.error('Quick mode is development-only')
 if a.output.exists():p.error('Do not overwrite retained benchmark output')
 START_CONTEXT.update(output=str(a.output),stage='data_and_tokenizer',started=time.perf_counter())
 save(a.output,{'complete':False,'mode':a.mode,'model_requested':a.model,'failures':[],'stage':'setup'})
 meta=json.loads((a.root/(a.model.split('/')[-1]+'_manifest.json')).read_text())
 data,manifest=load_data(a.root);tok,tokenizer_contract=load_tokenizer(a.model,meta['revision'],manifest)
 torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 START_CONTEXT['stage']='model_loading'
 model=AutoModelForCausalLM.from_pretrained(a.model,revision=meta['revision'],torch_dtype=torch.float32,attn_implementation='eager',token=False).cuda().eval()
 ghost_path=Path(__file__).parent/'GhostSuite'
 START_CONTEXT['stage']='upstream_loading'
 if subprocess.check_output(['git','-C',str(ghost_path),'rev-parse','HEAD'],text=True).strip()!=GHOST_COMMIT:raise ValueError('Ghost revision mismatch')
 upstream=load_upstream(ghost_path)
 out={'complete':False,'mode':a.mode,'quick':a.quick,'model':meta,'data':manifest,'tokenizer_contract':tokenizer_contract,
  'gpu':torch.cuda.get_device_name(),'gpu_total_bytes':torch.cuda.get_device_properties(0).total_memory,
  'torch':str(torch.__version__),'cuda':torch.version.cuda,'transformers':transformers.__version__,'ghost_commit':GHOST_COMMIT,
  'development_seed':DEV,'reporting_seeds':list(HELDOUT) if a.mode=='reporting' else [],'objectives':ms,'batch':32,
  'attention_precision':precision,'model_dtype':'float32','tf32':False,'calibration':[],'rows':[],'failures':[],
  'selected':{},'native_references':{},'coverage':{},'contracts':{
    'table':'Gram2e-5, solver-coefficient simplex1e-10, objective1.001 (additive near stationarity), emitted vs native-task-table direction2e-5; executed FP32 coefficients audited separately',
    'weighted':'same geometry gates, emitted vs separately evaluated weighted native backward2e-5',
    'joint':'both table and weighted contracts; not an either-or success rule'},
  'reference_scope':'FP64 audits of stored FP32 native task gradients; no outward enclosure for exact-real neural gradients',
  'selection':'fastest median of two qualifying development repeats after warmup, separately per method and reference contract; all candidates/failures retained',
  'timing_scope':'capture+Gram+transfer/QP+emitted direction including any new forward/backward; independent audits outside timing',
  'baseline_scope':'pinned Ghost column-kernel adaptation; independent structured identities; every acquisition method has the same weighted-native reconstruction option',
  'objective_scope':'M8 is8source tasks; M32 is32singleton example objectives from those8sources',
  'sources':{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in Path(__file__).parent.glob('*.py')}}
 out['executed_imports']={}
 for obj in [FactorizedJacobian,ExactFactorGram,OnlineFactorGram,losses,task_losses,solve,native_reference,audit_directions,classify_contracts,load_tokenizer]:
  f=Path(inspect.getfile(obj));out['executed_imports'][f.name]=hashlib.sha256(f.read_bytes()).hexdigest()
 f=Path(v46_supervised_head.__file__);out['executed_imports'][f.name]=hashlib.sha256(f.read_bytes()).hexdigest()
 out['ghost_sampler_sha256']=hashlib.sha256(Path(upstream.__file__).read_bytes()).hexdigest()
 save(a.output,out)
 START_CONTEXT['stage']='benchmark'
 def reference(batch,m,seed):
  try:
   K,fit,record,table=native_reference(model,batch,retain_table=True)
   out['native_references'][f'{m}:{seed}']={'complete':True,**record};save(a.output,out)
   return table,K,fit
  except Exception:
   failure={'phase':'reference','M':m,'seed':seed,'error':traceback.format_exc()}
   out['failures'].append(failure);out['native_references'][f'{m}:{seed}']={'complete':False,**failure}
   save(a.output,out);cleanup();return None
 def attempt(batch,method,cfg,ref,m,seed,repeat,phase):
  diagnostics={};attempt_start=time.perf_counter()
  try:
   row=run_method(model,batch,method,upstream,reference=ref,diagnostics=diagnostics,**cfg)
   row.update(M=m,seed=seed,repeat=repeat,warmup=repeat==0,configuration=cfg,identities=batch['identities'])
   out['calibration' if phase=='calibration' else 'rows'].append(row)
   if not row['passes_joint_contract']:
    out['failures'].append({'phase':phase,'M':m,'seed':seed,'method':method,'configuration':cfg,'repeat':repeat,
       'kind':'contract_gate_failure','table':row['passes_table_contract'],'weighted':row['passes_weighted_contract']})
   save(a.output,out);return row
  except Exception as exc:
   diagnostics['attempt_wall_seconds_including_cleanup']=time.perf_counter()-attempt_start
   if diagnostics.get('peak_reset'):
    diagnostics['peak_allocated_bytes_until_failure']=torch.cuda.max_memory_allocated()
   out['failures'].append({'phase':phase,'M':m,'seed':seed,'method':method,'configuration':cfg,'repeat':repeat,
      'kind':'CUDA_OOM' if isinstance(exc,torch.cuda.OutOfMemoryError) else 'runtime_failure','error':traceback.format_exc(),'cost_diagnostics':diagnostics})
   save(a.output,out);cleanup();return None
 for m in ms:
  batch=make_batch(data['train'],manifest['tasks'],tok.eos_token_id,DEV,objectives=m)
  ref=reference(batch,m,DEV);selected={method:{c:None for c in CONTRACTS} for method in METHODS}
  if ref is not None:
   for method in METHODS:
    candidates={c:[] for c in CONTRACTS}
    for cfg in configurations(method,a.quick):
     trials=[]
     for repeat in range(3):
      row=attempt(batch,method,cfg,ref,m,DEV,repeat,'calibration')
      if row is None:break
      trials.append(row)
     if len(trials)==3:
      for contract in CONTRACTS:
       if all(r['passes_'+contract+'_contract'] for r in trials):
        candidates[contract].append((float(np.median([r['seconds'] for r in trials[1:]])),cfg))
    for contract in CONTRACTS:
     if candidates[contract]:selected[method][contract]=min(candidates[contract],key=lambda z:z[0])[1]
   del ref
  out['selected'][str(m)]=selected;save(a.output,out);del batch;cleanup()
  if a.mode=='preflight':continue
  expected={c:len(METHODS)*len(HELDOUT)*a.repeats for c in CONTRACTS}
  successful={c:0 for c in CONTRACTS};retained={c:0 for c in CONTRACTS}
  for seed in HELDOUT:
   batch=make_batch(data['train'],manifest['tasks'],tok.eos_token_id,seed,objectives=m)
   ref=reference(batch,m,seed)
   if ref is None:del batch;continue
   for repeat in range(a.repeats+1):
    shift=(seed+repeat)%len(METHODS);order=METHODS[shift:]+METHODS[:shift]
    for method in order:
     unique={}
     for contract,cfg in selected[method].items():
      if cfg is not None:unique.setdefault(json.dumps(cfg,sort_keys=True),{'configuration':cfg,'roles':[]})['roles'].append(contract)
     for item in unique.values():
      row=attempt(batch,method,item['configuration'],ref,m,seed,repeat,'reporting')
      if row is None:continue
      row['selected_for_contracts']=item['roles']
      if repeat:
       for contract in item['roles']:
        retained[contract]+=1;successful[contract]+=int(row['passes_'+contract+'_contract'])
      save(a.output,out)
   del ref,batch;cleanup()
  out['coverage'][str(m)]={'expected_nonwarmup_roles':expected,'retained_nonwarmup_roles':retained,'successful_nonwarmup_roles':successful}
  save(a.output,out)
 out['complete']=True;out['completion_scope']='all scheduled states/candidates attempted or explicitly unresolved; completion does not mean all contracts pass'
 save(a.output,out);print('DUAL_REFERENCE_BENCHMARK_COMPLETE',flush=True)

def main():
 probe=argparse.ArgumentParser(add_help=False);probe.add_argument('--output',type=Path)
 args,_=probe.parse_known_args()
 if args.output is not None and not args.output.exists():
  START_CONTEXT.update(output=str(args.output),stage='attention_setup',started=time.perf_counter())
 try:
  with shared_attention_precision() as precision:benchmark(precision)
 except BaseException:
  if START_CONTEXT.get('output'):
   path=Path(START_CONTEXT['output']);out=json.loads(path.read_text()) if path.exists() else {'complete':False}
   out.setdefault('failures',[]).append({'phase':START_CONTEXT['stage'],'error':traceback.format_exc(),
       'attempt_wall_seconds':time.perf_counter()-START_CONTEXT['started']})
   save(path,out)
  raise

if __name__=='__main__':main()
