"""Task-mean Gram adapter for the official GhostSuite product kernels.

The upstream routines return products with one validation-gradient aggregate.
Here each task mean is that aggregate in turn, and training-example products
are reduced to task means. Factor acquisition is common to every compared arm.
This column adapter is not claimed to be an optimized all-pairs Ghost method;
the matched suite also needs a blocked all-pairs contraction control.

No norm-only/projection approximation is substituted for a full Gram. Shared
parameter occurrences are combined before contraction via the shared exact
fallback; every such fallback is counted and disclosed.
"""
from types import SimpleNamespace
import importlib.util
from pathlib import Path
import torch

GHOST_COMMIT='72ac352ae05436275e154b59334e5cb343b96339'

def load_upstream(root):
 path=Path(root)/'ghostEngines/supported_layers_grad_samplers_dotprod.py'
 spec=importlib.util.spec_from_file_location('official_ghost_samplers',path)
 module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
 return module

@torch.no_grad()
def ghost_columns(cache,upstream):
 if hasattr(cache,'caches'):raise ValueError('single capture only in this adapter')
 out=torch.zeros(cache.num_tasks,cache.num_tasks,device=cache.device,dtype=cache.dtype)
 census={'upstream_parameters':[],'shared_exact_fallback_parameters':[]}
 for pid,param in cache.parameters.items():
  terms=cache.terms[pid]
  if not terms:continue
  if len(terms)!=1 or terms[0].kind not in ('linear_weight','embedding'):
   # Norm/bias and tied-parameter contributions retain all occurrence cross terms.
   out+=cache.grouped_gram(memory_budget_bytes=128*2**20,exclude_parameter_ids=set(cache.parameters)-{pid})
   census['shared_exact_fallback_parameters'].append(cache.names[pid]);continue
  term=terms[0];A,B=term.right,term.left
  layer=SimpleNamespace(weight=param,bias=None,padding_idx=term.padding_idx)
  kernel=upstream._compute_linear_dot_product if term.kind=='linear_weight' else upstream._compute_embedding_dot_product
  for j,indices in enumerate(cache.task_examples):
   for attr in ['grad_dot_prod','_ghost_grad_val']:
    if hasattr(param,attr):delattr(param,attr)
   # Reference factors sum to task mean, train factors retain per-example scale.
   aa=torch.cat([A,A[indices]],0)
   bb=torch.cat([B,B[indices]/cache.counts[j]],0)
   try:
    kernel(layer,aa,bb,len(indices),compute_dtype=cache.dtype,accum_dtype=cache.dtype)
    score=param.grad_dot_prod
    column=torch.zeros(cache.num_tasks,device=cache.device,dtype=cache.dtype)
    column.index_add_(0,cache.task_ids,score);out[:,j]+=column/cache.counts
   finally:
    for attr in ['grad_dot_prod','_ghost_grad_val']:
     if hasattr(param,attr):delattr(param,attr)
   del aa,bb,score,column
  census['upstream_parameters'].append(cache.names[pid])
 return out,census
