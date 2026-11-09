"""Independent task-mean implementations of established contraction identities.

These are adapted arithmetic controls, not upstream Neural Tangents/GhostSuite
executions. Both orders are mathematically equivalent to Direct's contraction.
All methods use the same captured factors. Tied and non-linear-weight terms use
the disclosed shared exact fallback. Compare both orders; do not use a slow
fixed order as the only prior-work baseline.
"""
import math
import torch
from v46_factorized_exact import ExactFactorGram


@torch.no_grad()
def structured_gram(cache, order='auto', scratch_bytes=128*2**20,row_chunk=256):
    if order not in ('auto','parameter','token'):raise ValueError(order)
    if hasattr(cache,'caches'):raise ValueError('Single capture required')
    cache._check_current()
    m=cache.num_tasks
    result=torch.zeros(m,m,device=cache.device,dtype=cache.dtype)
    census=[]
    common=ExactFactorGram(cache,scratch_bytes=scratch_bytes,row_chunk=row_chunk)
    for pid,p in cache.parameters.items():
        terms=cache.terms[pid]
        if not terms:continue
        if len(terms)!=1 or terms[0].kind!='linear_weight':
            # Common optimized embeddings and all occurrence cross terms. This
            # isolates linear contraction order, avoiding a slow dense-vocab
            # fallback only in the comparison arm.
            contribution,plan=common.parameter_gram(pid)
            result+=contribution
            census.append({'parameter':cache.names[pid],'order':'shared_optimized_non_linear_weight','common_plan':plan})
            del contribution
            continue
        term=terms[0];a=term.right.reshape(-1,term.right.shape[-1])
        d=term.left.reshape(-1,term.left.shape[-1])
        owner=cache.task_ids[:,None].expand(term.left.shape[:2]).reshape(-1)
        active=d.ne(0).any(1)
        a=a[active];d=d[active];owner=owner[active]
        d=d/cache.counts[owner,None]
        n,di=len(owner),a.shape[1];do=d.shape[1];item=a.element_size()
        pcost=2*(n+m*m)*di*do;tcost=2*n*n*(di+do)
        route=('token' if tcost<pcost else 'parameter') if order=='auto' else order
        if route=='token':
            # <sum d a^T, sum d' a'^T> = sum <d,d'><a,a'>.
            tile=max(1,math.isqrt(max(1,scratch_bytes//(3*item))))
            for first in range(0,n,tile):
                stop=min(first+tile,n)
                for second in range(first,n,tile):
                    end=min(second+tile,n)
                    block=(a[first:stop]@a[second:end].T)*(d[first:stop]@d[second:end].T)
                    indices=(owner[first:stop,None]*m+owner[None,second:end]).reshape(-1)
                    reduced=result.new_zeros(m*m)
                    reduced.index_add_(0,indices,block.reshape(-1))
                    reduced=reduced.reshape(m,m);result+=reduced
                    if first!=second:result+=reduced.T
                    del block,indices,reduced
        else:
            # Build bounded parameter-row tiles of task means, never an M x P table.
            lengths=torch.bincount(owner,minlength=m).cpu().tolist();width=max(lengths)
            aa=a.new_zeros(m,width,di);dd=d.new_zeros(m,width,do)
            for task in range(m):
                mask=owner==task;aa[task,:lengths[task]]=a[mask];dd[task,:lengths[task]]=d[mask]
            tile=max(1,min(row_chunk,scratch_bytes//max(1,m*di*item)))
            for first in range(0,do,tile):
                h=torch.bmm(dd[:,:,first:first+tile].transpose(1,2),aa).reshape(m,-1)
                result.addmm_(h,h.T)
                del h
            del aa,dd,mask
        census.append({'parameter':cache.names[pid],'order':route,'active_tokens':n,
                       'parameter_flops':pcost,'token_flops':tcost})
        del a,d,owner,active
    return result,census
