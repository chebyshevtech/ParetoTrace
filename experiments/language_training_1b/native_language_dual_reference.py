"""Prospective dual-reference audit: independent native task table in host RAM.

Excluded from timed recovery. Host storage permits auditing methods even when
an M-by-P native table does not fit on the GPU. It is not a performance method.
"""
import gc
import time
import numpy as np
import torch
from language_common import task_losses,solve

def gram64(table,columns=262144):
    device=table.device if table.is_cuda else torch.device('cuda')
    K=torch.zeros((len(table),len(table)),device=device,dtype=torch.float64)
    for first in range(0,table.shape[1],columns):
        block=table[:,first:first+columns].to(device=device,dtype=torch.float64)
        K.addmm_(block,block.T)
    return K

def native_reference(model,b,retain_table=False):
    import psutil
    params=tuple(p for p in model.parameters() if p.requires_grad)
    size=sum(p.numel() for p in params)
    required=b['M']*size*params[0].element_size()
    if required>psutil.virtual_memory().available*.70:
        raise MemoryError('Independent host table exceeds 70% available RAM: '+str(required))
    torch.cuda.synchronize();start=time.perf_counter()
    table=torch.empty((b['M'],size),device='cpu',dtype=params[0].dtype)
    l=task_losses(model,b)
    for j in range(b['M']):
        grads=torch.autograd.grad(l[j],params,retain_graph=j+1<b['M'])
        offset=0
        for p,g in zip(params,grads):
            n=p.numel();table[j,offset:offset+n].copy_(g.detach().reshape(-1));offset+=n
        del grads,g
    del l
    K=gram64(table).cpu().numpy();K=(K+K.T)/2
    if not retain_table:del table
    gc.collect();torch.cuda.empty_cache()
    if not np.isfinite(K).all():raise FloatingPointError('Nonfinite native reference Gram')
    fit=solve(K);torch.cuda.synchronize()
    metadata={'seconds':time.perf_counter()-start,'host_table_bytes':required,
        'scope':'independent native task gradients, host table, tiled GPU FP64 contractions; outside timed recovery',
        'lower':fit['lower'],'objective':fit['objective'],'qp_gap':fit['dual_gap'],
        'gram':K.tolist()}
    return (K,fit,metadata,table) if retain_table else (K,fit,metadata)

def audit_geometry(row,K,fit):
    K=np.asarray(K)
    H=np.asarray(row['gram']);w=np.asarray(row['weights'])
    if K.ndim!=2 or K.shape[0]!=K.shape[1] or H.shape!=K.shape or not np.isfinite(K).all() or not np.isfinite(H).all():
        row.update(passed_all_gates=False,passed_native_gram_gate=False,passed_native_objective_gate=False,
            audit_failure='Invalid reference or recovered Gram shape/finiteness')
        return row
    feasible=bool(w.shape==(len(K),) and np.isfinite(w).all() and w.min()>=-1e-12 and abs(w.sum()-1)<=1e-10)
    row['passed_weight_feasibility_gate']=feasible
    if w.shape!=(len(K),):
        row.update(passed_all_gates=False,passed_native_gram_gate=False,passed_native_objective_gate=False,
            audit_failure='Invalid weight dimension')
        return row
    u=K@w;q=float(w@u)
    scale=max(float(np.abs(K).max()),1e-100)
    error=float(np.linalg.norm(H-K)/max(np.linalg.norm(K),1e-100))
    lower=fit['lower'];relative=lower>scale*1e-10
    factor=q/lower if relative else None
    excess=q-lower
    passed=(np.isfinite(factor) and factor<=1.001) if relative else (np.isfinite(excess) and excess<=1e-7*scale)
    row.update(native_gram_relative_error=error,passed_native_gram_gate=bool(np.isfinite(error) and error<=2e-5),
        native_objective_factor_upper=factor,native_objective_excess_above_lower=excess,
        native_reference_lower=lower,native_reference_qp_gap=fit['dual_gap'],
        native_table_min_response=float(u.min()),passed_native_objective_gate=bool(passed),
        objective_audit_scope='relative native task-table objective' if relative else 'additive gap to native lower bound near stationarity',
        common_descent_scope='table response only; actual reconstructed direction response not certified')
    row['passed_all_gates']=bool(feasible and row.get('passed_direction_gate') and row['passed_native_gram_gate'] and passed)
    return row

def audit_directions(emitted,weighted,parameters,table,w,K,fit,columns=262144):
    """Independent parameter-space and executed-response audits, outside timing.

    The host reference table is built from native per-task backwards. Solver
    weights and actual FP32 coefficients are both audited. Arithmetic here is
    FP64 diagnostics; no outward-rounded network-gradient claim is made.
    """
    w=np.asarray(w,dtype=np.float64);wc=w.astype(np.float32).astype(np.float64)
    if table.device.type!='cpu' or table.shape[0]!=len(w):raise ValueError('Host reference layout')
    if set(emitted)!=set(parameters) or set(weighted)!=set(parameters):raise ValueError('Direction parameter coverage')
    if table.shape[1]!=sum(p.numel() for p in parameters.values()):raise ValueError('Reference parameter dimension')
    coeff=torch.as_tensor(w,dtype=torch.float64);cast_coeff=torch.as_tensor(wc,dtype=torch.float64)
    sums={k:0. for k in ['table_norm2','cast_table_norm2','emitted_norm2','weighted_norm2','table_error2',
        'cast_table_error2','weighted_error2','backward_table_error2']}
    responses=torch.zeros(len(w),dtype=torch.float64);offset=0
    for name,p in parameters.items():
        d=emitted[name].detach().reshape(-1).cpu();b=weighted[name].detach().reshape(-1).cpu()
        if d.numel()!=p.numel() or b.numel()!=p.numel():raise ValueError('Direction shape')
        for first in range(0,p.numel(),columns):
            stop=min(first+columns,p.numel());T=table[:,offset+first:offset+stop].double()
            target=coeff@T;cast_target=cast_coeff@T;out=d[first:stop].double();back=b[first:stop].double()
            values={'table_norm2':target,'cast_table_norm2':cast_target,'emitted_norm2':out,'weighted_norm2':back,
                'table_error2':out-target,'cast_table_error2':out-cast_target,'weighted_error2':out-back,
                'backward_table_error2':back-cast_target}
            for key,value in values.items():sums[key]+=float(value.square().sum())
            responses+=T@out
        offset+=p.numel()
        del d,b
    if not np.isfinite(list(sums.values())).all() or not bool(torch.isfinite(responses).all()):raise FloatingPointError('Nonfinite dual-reference audit')
    root=lambda key:float(np.sqrt(max(sums[key],0)))
    table_error=root('table_error2')/max(root('table_norm2'),1e-100)
    weighted_error=root('weighted_error2')/max(root('weighted_norm2'),1e-100)
    backward_error=root('backward_table_error2')/max(root('cast_table_norm2'),1e-100)
    response=responses.numpy();K=np.asarray(K,dtype=np.float64);pred=K@w
    conservative=pred-np.sqrt(np.maximum(np.diag(K),0))*root('table_error2')
    denominator=np.sqrt(sums['emitted_norm2']*max(fit['objective'],0))
    return {'emitted_table_direction_relative_error':table_error,'emitted_table_direction_absolute_error':root('table_error2'),
        'emitted_cast_table_direction_relative_error':root('cast_table_error2')/max(root('cast_table_norm2'),1e-100),
        'emitted_weighted_backward_relative_error':weighted_error,'weighted_backward_cast_table_relative_error':backward_error,
        'emitted_direction_norm':root('emitted_norm2'),'table_direction_norm':root('table_norm2'),
        'task_gradient_norm_max':float(np.sqrt(np.maximum(np.diag(K),0)).max()),
        'mixture_cancellation_condition':float(np.abs(w)@np.sqrt(np.maximum(np.diag(K),0))/max(root('table_norm2'),1e-100)),
        'executed_coefficients':wc.tolist(),'executed_coefficient_sum':float(wc.sum()),
        'solver_coefficient_sum':float(w.sum()),'cast_table_objective':float(wc@K@wc),
        'executed_native_table_responses':response.tolist(),'executed_native_table_min_response':float(response.min()),
        'table_response_error_bound_diagnostic':conservative.tolist(),
        'normalized_gain_to_reference_primal':float(response.min()/denominator) if denominator>0 else None,
        'passes_table_direction':bool(table_error<=2e-5),'passes_weighted_backward_direction':bool(weighted_error<=2e-5),
        'direction_audit_scope':'FP64 diagnostics against an independently acquired FP32 native task table and separately against weighted native backward; no outward AD enclosure'}

def classify_contracts(row,K,fit):
    # Reuse the independently audited geometry predicate, replacing only its
    # old single-reference conjunction with explicit named contracts.
    proxy=dict(row,passed_direction_gate=True);audit_geometry(proxy,K,fit)
    row.update({k:v for k,v in proxy.items() if k not in ('passed_all_gates','passed_direction_gate')})
    geometry=bool(proxy.get('passed_weight_feasibility_gate') and proxy.get('passed_native_gram_gate') and proxy.get('passed_native_objective_gate'))
    row['passes_table_contract']=bool(geometry and row.get('passes_table_direction'))
    row['passes_weighted_contract']=bool(geometry and row.get('passes_weighted_backward_direction'))
    row['passes_joint_contract']=bool(row['passes_table_contract'] and row['passes_weighted_contract'])
    return row
