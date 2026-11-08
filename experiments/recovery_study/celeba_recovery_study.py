"""Same-state CelebA tolerance, query-cost, and Armijo study.

Uses only the development checkpoint and new training batches. It does not
inspect reporting/test outcomes. Retained recovery code is identified by hash.
Native audits are outside timed recovery; every failed query and Direct fallback
is inside. This is a recovery/progress diagnostic, not a downstream trajectory.
"""
import os
os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
import argparse,gc,hashlib,json,sys,time,traceback
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from shared_native_actions import SharedNativeActions

def sync():torch.cuda.synchronize()
def clean():gc.collect();torch.cuda.empty_cache();sync()
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,value):
    invalid=[]
    def safe(obj,where=''):
        if isinstance(obj,float) and not np.isfinite(obj):invalid.append(where);return None
        if isinstance(obj,dict):return {k:safe(v,where+'/'+str(k)) for k,v in obj.items()}
        if isinstance(obj,list):return [safe(v,where+'/'+str(i)) for i,v in enumerate(obj)]
        return obj
    value=safe(value)
    if invalid:value['nonfinite_diagnostic_fields_replaced_by_null']=invalid
    tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False));tmp.replace(p)

def direct(model,x,y,solve,SharedCapture):
    with SharedCapture(model.trunk) as capture:
        l=F.binary_cross_entropy_with_logits(model(x),y,reduction='none').mean(0)
    capture.capture(l);K=capture.gram(torch.float64).cpu().numpy();del capture
    w,fit=solve(K)
    params={n:p for n,p in model.named_parameters() if n.startswith('trunk.') and p.requires_grad}
    gs=torch.autograd.grad(l,tuple(params.values()),torch.as_tensor(w,device=l.device,dtype=l.dtype))
    return w,dict(zip(params,[g.detach() for g in gs])),K,fit

def native_table(model,x,y,params,chunk=1):
    l=F.binary_cross_entropy_with_logits(model(x),y,reduction='none').mean(0)
    G=l.new_empty((40,sum(p.numel() for p in params.values())))
    basis=torch.eye(40,device=l.device,dtype=l.dtype)
    for j in range(0,40,chunk):
        stop=min(j+chunk,40)
        if chunk==1:gs=tuple(g[None] for g in torch.autograd.grad(l[j],tuple(params.values()),retain_graph=stop<40))
        else:gs=torch.autograd.grad(l,tuple(params.values()),basis[j:stop],is_grads_batched=True,retain_graph=stop<40)
        offset=0
        for p,g in zip(params.values(),gs):
            G[j:stop,offset:offset+p.numel()].copy_(g.reshape(stop-j,-1));offset+=p.numel()
        del gs,g
    K=torch.zeros(40,40,device='cuda',dtype=torch.float64)
    for st in range(0,G.shape[1],65536):
        tile=G[:,st:st+65536].double();K.addmm_(tile,tile.T)
    return G,K.cpu().numpy()

def native_full(model,x,y,params,solve,chunk):
    table,K=native_table(model,x,y,params,chunk);w,fit=solve(K)
    vector=torch.as_tensor(w,device='cuda',dtype=table.dtype)@table
    out={};offset=0
    for n,p in params.items():
        out[n]=vector[offset:offset+p.numel()].reshape_as(p);offset+=p.numel()
    return w,out,K,fit

def reference_state(model,x,y,params,solve):
    table,K=native_table(model,x,y,params)
    weights,fit=solve(K)
    return table.cpu(),K,weights,fit

def audit_direction(function,params,direction,w,G,K,reference,eps):
    # All large audit temporaries are scoped here, including exception paths.
    d=torch.cat([direction[n].flatten() for n in params]).detach().cpu().double()
    table=G.double();w=np.asarray(w)
    feasible=bool(w.shape==(40,) and np.isfinite(w).all() and w.min()>=-1e-12 and abs(w.sum()-1)<=1e-10)
    target=torch.as_tensor(w,dtype=torch.float64)@table
    responses=(table@d).numpy();q=float(w@K@w)
    native=torch.autograd.grad(function(params),tuple(params.values()),torch.as_tensor(w,device='cuda',dtype=torch.float32))
    actual=torch.cat([v.flatten() for v in native]).detach().cpu().double()
    err=float((d-actual).norm()/actual.norm().clamp_min(1e-30))
    table_error=float((d-target).norm()/target.norm().clamp_min(1e-30))
    resolved=reference['lower']>max(float(np.abs(K).max()),1e-100)*1e-10
    factor=q/reference['lower'] if resolved else None
    norm2=float(d@d);response_min=float(responses.min())
    ratio=float(response_min/np.sqrt(norm2*reference['objective'])) if norm2*reference['objective']>0 else None
    gain=ratio if response_min>0 else None
    gain_pass=bool(gain is not None and np.isfinite(gain) and gain>=.95-1e-5)
    passed=bool(feasible and factor is not None and np.isfinite([factor,err,table_error]).all()
                and factor<=1+(eps or .001)+1e-5 and err<=2e-5 and table_error<=2e-5 and gain_pass)
    return {'native_objective_factor_upper':factor,'relative_reference_resolved':resolved,'native_direction_relative_error':err,
        'native_table_direction_relative_error':table_error,'passed_simplex_gate':feasible,
        'executed_min_response':response_min,'executed_norm_squared':norm2,
        'normalized_gain_to_reference_primal':ratio,'native_normalized_gain_lower':gain,
        'passed_gain_audit':gain_pass,'passed_audit':passed},responses

def warm_configuration(rule,model,x,y,params,function,solve,SharedCapture):
    if rule=='direct':direct(model,x,y,solve,SharedCapture)
    elif rule.startswith('native_'):native_full(model,x,y,params,solve,int(rule.split('_')[1]))
    else:
        oracle=SharedNativeActions(params,function)
        try:oracle.action(np.full(40,1/40));oracle.direction(np.full(40,1/40))
        finally:oracle.close()

def armijo(function,params,direction,before,response,initial):
    # Functional trial points preserve the checkpoint exactly.
    rows=[]
    if min(response)<=0 or not np.isfinite(response).all():
        return [{'eligible':False,'reason':'positive native task-response gate failed'}]
    device=before.device;response=torch.as_tensor(response,device=device,dtype=torch.float64)
    for c in [.5,.1,.01,.0001]:
        start=time.perf_counter();trials=[];accepted=False
        for k in range(30):
            eta=initial*2.**(-k)
            with torch.no_grad():
                trial={n:p-eta*direction[n] for n,p in params.items()};after=function(trial)
            sync();changes=(after.double()-before.double())
            finite=bool(torch.isfinite(after).all())
            passed=finite and bool((changes<=-c*eta*response).all())
            trials.append({'eta':eta,'passed':passed,'max_task_change':float(changes.max()) if finite else None,
                           'mean_task_change':float(changes.mean()) if finite else None})
            if passed:accepted=True;break
        rows.append({'eligible':True,'c':c,'accepted':accepted,'accepted_eta':eta if accepted else 0.,
                     'trials':trials,'seconds':time.perf_counter()-start,
                     'scope':'rounded same-minibatch checks; not an interval enclosure or population guarantee'})
    # The first trial in each row evaluates exactly the same fixed-step update.
    rows.append({'comparator':'unprotected fixed step','eta':initial,
                 'max_task_change':rows[0]['trials'][0]['max_task_change'],
                 'mean_task_change':rows[0]['trials'][0]['mean_task_change']})
    return rows

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('/content/rebuttal'))
    p.add_argument('--seeds',default='9101,9102,9103,9104,9105');p.add_argument('--output',type=Path)
    p.add_argument('--repeats',type=int,default=3)
    a=p.parse_args();root=a.root;c=root/'celeba_v2'
    if a.repeats<1 or a.seeds!='9101,9102,9103,9104,9105':p.error('Frozen seeds and positive repeat count required')
    sys.path.insert(0,str(c));sys.path.insert(0,str(Path(__file__).parent/'retained_sources'))
    from celeba_training import Model,batch,load_split,solve
    from preflight import SharedCapture
    from stable_recovery import recover
    selection=json.loads((c/'selected_learning_rates.json').read_text())
    checkpoint=c/'runs'/f'development_direct_seed9001_lr{selection["rates"]["direct"]:g}.pt'
    state=torch.load(checkpoint,map_location='cpu',weights_only=False)
    if not state['result']['complete'] or state['result']['config']['phase']!='development':
        raise ValueError('Complete development checkpoint required')
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    model=Model().cuda().eval();model.load_state_dict(state['model']);del state
    params={n:p for n,p in model.named_parameters() if n.startswith('trunk.') and p.requires_grad}
    X,Y=load_split(c/'data','train')
    dest=a.output or root/'celeba_recovery_study.json'
    out={'complete':False,'checkpoint_sha256':digest(checkpoint),'selection_sha256':digest(c/'selected_learning_rates.json'),
        'gpu':torch.cuda.get_device_name(),'torch':str(torch.__version__),'batch':128,
        'scope':'same fixed development model; independently seeded shared-task minibatches, no reporting/test outcomes',
        'precision':'FP32 model/AD; FP64 contractions/solves; numerical diagnostics with independent native audit, not outward-rounded guarantees',
        'comparison':'completion versus scalar stopping endpoints with the SAME lower-completion query policy and solver; not a scalar-only acquisition algorithm',
        'margin_coupling':'fixed95% response margin itself implies at most1/.95^2 objective factor in exact positive-energy arithmetic; eps.25 can therefore be margin-limited',
        'repeats':a.repeats,'baseline_selection':'lowest median successful full-recovery runtime on seed9101 only; all configurations retained on later states',
        'sampling_unit':'five minibatches at ONE development checkpoint;20 transfer pairs are dependent; seed9101 calibrates runtime settings',
        'model_scope':'shared trunk; frozen pretrained BN statistics and affine; private heads fixed during recovery study',
        'training_data_manifest_sha256':digest(c/'data/celeba_manifest.json'),
        'frozen_imports':{n:digest(c/n) for n in ['celeba_training.py','preflight.py','simplex_qp.py']},
        'retained_manifest_sha256':digest(Path(__file__).parent/'retained_sources/manifest.json'),
        'sources':{str(f.relative_to(Path(__file__).parent)):digest(f) for f in [Path(__file__),Path(__file__).with_name('shared_native_actions.py'),
                    Path(__file__).parent/'retained_sources/stable_recovery.py',Path(__file__).parent/'retained_sources/v46_baseline_frontier.py']},
        'states':[],'rows':[],'line_search':[],'failures':[]}
    save(dest,out)
    for seed in map(int,a.seeds.split(',')):
        idx=torch.randint(len(X),(128,),generator=torch.Generator().manual_seed(seed))
        x,y=batch(X,Y,idx)
        def function(parameters):
            return F.binary_cross_entropy_with_logits(torch.func.functional_call(model,parameters,(x,),strict=False),y,reduction='none').mean(0)
        # Warm-up is outside timing and does not yield a solver start. An
        # unavailable action backend must not erase the other planned arms.
        warm=None
        try:
            warm=SharedNativeActions(params,function);warm.action(np.full(40,1/40))
        except Exception:
            out['failures'].append({'seed':seed,'phase':'action_warmup','error':traceback.format_exc()})
        finally:
            if warm is not None:warm.close()
            del warm;clean()
        try:
            G,K,wn,reference=reference_state(model,x,y,params,solve)
        except Exception:
            out['failures'].append({'seed':seed,'phase':'native_reference','error':traceback.format_exc()})
            save(dest,out);clean();continue
        # Keep reference gradients on CPU so they cannot inflate measured peaks.
        clean()
        u=K@np.full(40,1/40);q=float(u.mean())
        resolved=reference['lower']>max(float(np.abs(K).max()),1e-100)*1e-10
        out['states'].append({'seed':seed,'indices':idx.tolist(),'gram':K.tolist(),'reference':reference,'reference_weights':wn.tolist(),'relative_reference_resolved':resolved,
             'uniform_objective_factor_upper':q/reference['lower'] if resolved else None,
             'uniform_min_response':float(u.min()),'negative_pair_fraction':float(np.mean(K[np.triu_indices(40,1)]<0))})
        cases=[('direct',None)]+[(f'native_{b}',None) for b in [1,2,4,8,40]]+[(rule,eps) for eps in [.01,.05,.1,.25] for rule in ['completion','margin']]
        shift=seed%len(cases);cases=cases[shift:]+cases[:shift]
        warm_done=set()
        for rule,eps,repeat in [(r,e,k) for k in range(a.repeats) for r,e in cases]:
            oracle=direction=None;row={'seed':seed,'rule':rule,'epsilon':eps,'repeat':repeat,'complete':False}
            start=None;action_calls=0;action_seconds=0.
            try:
                if rule not in warm_done:
                    tick=time.perf_counter()
                    try:warm_configuration(rule,model,x,y,params,function,solve,SharedCapture)
                    except Exception:row['configuration_warmup_failure']=traceback.format_exc()
                    row['untimed_configuration_warmup_seconds']=time.perf_counter()-tick
                    warm_done.add(rule)
                clean();torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
                if rule=='direct' or rule.startswith('native_'):
                    if rule=='direct':w,direction,_,fit=direct(model,x,y,solve,SharedCapture)
                    else:w,direction,_,fit=native_full(model,x,y,params,solve,int(rule.split('_')[1]))
                    result={'accepted':True,'actions':0,'action_scope':'adaptive Gram-oracle calls only; full recovery still performs AD',
                        'task_adjoint_rows':40,'acquisition_backward_batches':1 if rule=='direct' else int(np.ceil(40/int(rule.split('_')[1]))),
                        'reconstruction_backward_calls':1 if rule=='direct' else 0,
                        'status':'full_contraction','objective':fit['objective'],'lower':fit['lower']}
                else:
                    result={'accepted':False,'status':'capture_pending'}
                    try:
                        oracle=SharedNativeActions(params,function)
                        def action(v):
                            nonlocal action_calls,action_seconds
                            action_calls+=1;tick=time.perf_counter()
                            try:
                                z=oracle.action(v);sync();return z.double().cpu().numpy()
                            finally:action_seconds+=time.perf_counter()-tick
                        result=recover(action,40,eps=eps,gamma=.05,lower_rule=rule,max_actions=40)
                        w=result.pop('weights');result.pop('response',None)
                        if result['accepted']:direction=oracle.direction(w)
                    except Exception:
                        result.update(accepted=False,status='backend_or_reconstruction_failure',error=traceback.format_exc())
                    result.update(attempted_action_calls=action_calls,all_attempted_action_seconds=action_seconds)
                    row['recovery']=result
                    if not result['accepted']:
                        if oracle is not None:oracle.close();oracle=None
                        clean()
                        w,direction,_,fit=direct(model,x,y,solve,SharedCapture)
                        result['fallback']={'kind':'Direct','objective':fit['objective'],'lower':fit['lower']}
                sync();row.update(seconds=time.perf_counter()-start,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                    weights=np.asarray(w).tolist(),recovery=result,timed_scope='capture, every action and solve, reconstruction, and any Direct fallback')
                if oracle is not None:oracle.close();oracle=None
                row['failure_phase']='independent_audit'
                # Independent table and weighted backward, outside timing.
                audit,responses=audit_direction(function,params,direction,w,G,K,reference,eps)
                row.update(audit,complete=True)
                if rule=='direct' and repeat==0 and row['passed_audit']:
                    row['failure_phase']='armijo_diagnostic'
                    before=function(params).detach()
                    out['line_search'].append({'seed':seed,'initial_step':selection['rates']['direct'],
                        'rows':armijo(function,params,direction,before,responses,selection['rates']['direct'])})
                elif rule=='direct' and repeat==0:out['line_search'].append({'seed':seed,'eligible':False,'reason':'Direct native audit failed'})
                row.pop('failure_phase',None)
            except Exception:
                row['error']=traceback.format_exc();out['failures'].append({'seed':seed,'rule':rule,'epsilon':eps,'error':row['error']})
                if start is not None:
                    row.update(total_elapsed_through_failure_seconds=time.perf_counter()-start,
                        failed_attempt_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                        attempted_action_calls=action_calls,all_attempted_action_seconds=action_seconds)
                    row.setdefault('failure_phase','recovery_or_fallback')
            finally:
                if oracle is not None:oracle.close()
                del oracle,direction;clean()
            out['rows'].append(row);save(dest,out)
            print(json.dumps({k:row.get(k) for k in ['seed','rule','epsilon','complete','seconds','peak_allocated_bytes','passed_audit']}),flush=True)
        if seed==9101:
            timings={}
            for name in ['direct']+[f'native_{b}' for b in [1,2,4,8,40]]:
                rr=[r for r in out['rows'] if r['seed']==9101 and r['rule']==name]
                if len(rr)==a.repeats and all(r.get('passed_audit') for r in rr):
                    timings[name]=float(np.median([r['seconds'] for r in rr]))
            out['baseline_calibration']={'seed':9101,'successful_medians':timings,
                'selected':min(timings,key=timings.get) if timings else None}
            save(dest,out)
        del G;clean()
    out['batch_transfer']=[]
    for target in out['states']:
        H=np.asarray(target['gram']);lower=target['reference']['lower']
        for source in out['states']:
            if source['seed']==target['seed']:continue
            w=np.asarray(source['reference_weights']);q=float(w@H@w)
            out['batch_transfer'].append({'weight_source_seed':source['seed'],'target_seed':target['seed'],
                'objective_factor_upper':q/lower if target['relative_reference_resolved'] else None,
                'scope':'cross-minibatch transfer suboptimality, not a population noise floor'})
    out['complete']=True
    out['all_rows_audited']=bool(not out['failures'] and all(r.get('passed_audit') for r in out['rows']))
    save(dest,out)

if __name__=='__main__':main()
