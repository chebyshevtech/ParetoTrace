"""Same independent dual-reference checks, using bounded GPU audit scratch.

The independent native task table remains on the host. Transfer retains source precision; conversion to FP64 occurs on the GPU.
Only FP64 audit
reductions move to tiled GPU computation, outside every method's timed scope.
The CPU implementation is retained for direct parity checks.
"""
import numpy as np
import torch
from native_language_dual_reference import native_reference,audit_geometry,classify_contracts,gram64
from native_language_dual_reference import audit_directions as audit_directions_cpu

def audit_directions(emitted,weighted,parameters,table,w,K,fit,columns=1048576):
    """Independent parameter-space and executed-response audits, outside timing.

    The host reference table is built from native per-task backwards. Solver
    weights and actual FP32 coefficients are both audited. Arithmetic here is
    Tiled GPU FP64 diagnostics; no outward-rounded network-gradient claim is made.
    """
    w=np.asarray(w,dtype=np.float64);wc=w.astype(np.float32).astype(np.float64)
    if table.device.type!='cpu' or table.shape[0]!=len(w):raise ValueError('Host reference layout')
    if set(emitted)!=set(parameters) or set(weighted)!=set(parameters):raise ValueError('Direction parameter coverage')
    if table.shape[1]!=sum(p.numel() for p in parameters.values()):raise ValueError('Reference parameter dimension')
    device=next(iter(emitted.values())).device
    coeff=torch.as_tensor(w,dtype=torch.float64,device=device);cast_coeff=torch.as_tensor(wc,dtype=torch.float64,device=device)
    sums={k:torch.zeros((),dtype=torch.float64,device=device) for k in ['table_norm2','cast_table_norm2','emitted_norm2','weighted_norm2','table_error2',
        'cast_table_error2','weighted_error2','backward_table_error2']}
    responses=torch.zeros(len(w),dtype=torch.float64,device=device);offset=0
    for name,p in parameters.items():
        d=emitted[name].detach().reshape(-1);b=weighted[name].detach().reshape(-1)
        if d.numel()!=p.numel() or b.numel()!=p.numel():raise ValueError('Direction shape')
        for first in range(0,p.numel(),columns):
            stop=min(first+columns,p.numel());T=table[:,offset+first:offset+stop].to(device=device).to(dtype=torch.float64)
            target=coeff@T;cast_target=cast_coeff@T;out=d[first:stop].double();back=b[first:stop].double()
            values={'table_norm2':target,'cast_table_norm2':cast_target,'emitted_norm2':out,'weighted_norm2':back,
                'table_error2':out-target,'cast_table_error2':out-cast_target,'weighted_error2':out-back,
                'backward_table_error2':back-cast_target}
            for key,value in values.items():sums[key]+=value.square().sum()
            responses+=T@out
        offset+=p.numel()
        del d,b
    sums={key:float(value) for key,value in sums.items()}
    if not np.isfinite(list(sums.values())).all() or not bool(torch.isfinite(responses).all()):raise FloatingPointError('Nonfinite dual-reference audit')
    root=lambda key:float(np.sqrt(max(sums[key],0)))
    table_error=root('table_error2')/max(root('table_norm2'),1e-100)
    weighted_error=root('weighted_error2')/max(root('weighted_norm2'),1e-100)
    backward_error=root('backward_table_error2')/max(root('cast_table_norm2'),1e-100)
    response=responses.cpu().numpy();K=np.asarray(K,dtype=np.float64);pred=K@w
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
        'direction_audit_backend':'tiled GPU FP64, independent host task table, outside recovery timing',
        'direction_audit_scope':'FP64 diagnostics against an independently acquired FP32 native task table and separately against weighted native backward; no outward AD enclosure'}

