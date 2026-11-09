"""Matched language-training rules with full, explicit update lifecycles.

All model updates use plain SGD without momentum, decay or clipping. CAGrad
uses the published mean-gradient scaling (not the optional M-scaled wrapper).
FAMO follows the pinned author's log-loss rule without its optional clipping.
FA uses the published same-batch finite-difference recursion. Periodic is the
retained held-weight/refinement heuristic, not a per-step fidelity certificate.
"""
import copy,time
import numpy as np
import torch
from scipy.optimize import minimize
from language_common import losses,task_losses,solve,native_direction
from simplex_qp import project_simplex
from v46_factorized_jacobian import FactorizedJacobian
from v46_factorized_exact import ExactFactorGram
from paretotrace_online_factor import OnlineFactorGram

class LossWrapper(torch.nn.Module):
    def __init__(self,model):super().__init__();self.model=model
    def forward(self,b):return losses(self.model,b)

def native_periodic(model,b,weights,scale):
    """Retained V31 action/segment rule on the common answer-only objective."""
    from language_retained_v31 import exact_action
    wrapper=LossWrapper(model);params=dict(wrapper.named_parameters())
    function=lambda p:torch.func.functional_call(wrapper,p,(b,))
    counts=torch.bincount(b['owner'],minlength=b['M'])
    direction,response,before=exact_action(function,params,b['owner'],counts,weights)
    if response.shape!=(b['M'],) or not np.isfinite(response).all() or not np.isfinite(before).all():
        raise FloatingPointError('Nonfinite native periodic action/losses')
    selected=int(np.argmin(response));take=(b['owner']==selected).nonzero(as_tuple=True)[0]
    sparse={k:b[k].index_select(0,take) for k in ['inputs','attention','labels','owner']}
    sparse.update(M=b['M'],prepared=tuple(t.index_select(0,take) for t in b['prepared']))
    selected_loss,pullback=torch.func.vjp(lambda p:torch.func.functional_call(wrapper,p,(sparse,)).mean(),params)
    selected_gradient=pullback(torch.ones_like(selected_loss))[0]
    diagonal=sum(float(g.detach().double().square().sum()) for g in selected_gradient.values())
    q=float(weights@response);denominator=diagonal-2*float(response[selected])+q
    if not np.isfinite([diagonal,q,denominator]).all():raise FloatingPointError('Nonfinite native segment geometry')
    exact=float(np.clip((q-float(response[selected]))/denominator,0,1)) if denominator>0 else 0.
    executed=scale*exact;w=weights*(1-executed);w[selected]+=executed
    out={n.removeprefix('model.'):(1-executed)*g+executed*selected_gradient[n].detach() for n,g in direction.items()}
    meta={'action_response':response.tolist(),'selected_diagonal':diagonal,'exact_segment_step':exact,'executed_segment_step':executed,
          'refinement_applied':executed>0,'periodic_kernel':'retained native VJP/JVP and sparse-VJP action, damped exact segment; FP64 norm reduction'}
    return out,w,torch.as_tensor(before,device=next(model.parameters()).device,dtype=torch.float32),meta

def sync(model):
    if next(model.parameters()).is_cuda:torch.cuda.synchronize()

def finite(values,label):
    groups={}
    for v in values:
        if v is None:raise FloatingPointError('Missing '+label)
        groups.setdefault(v.device,[]).append(torch.isfinite(v.detach()).all())
    if any(not bool(torch.stack(g).all()) for g in groups.values()):raise FloatingPointError('Nonfinite '+label)

def reduce_tasks(values,b):
    counts=torch.bincount(b['owner'],minlength=b['M']).to(values.dtype)
    return values.new_zeros(b['M']).index_add(0,b['owner'],values)/counts

def cagrad_weights(K,alpha=.4):
    """Author's simplex problem and rescale=1, replacing only G^T G acquisition.

    The objective x'Kb + c sqrt(x'Kx), with c proportional to sqrt(mean K), is positively
    homogeneous in K, so the minimizer is unchanged by solving on K/scale. Upstream CAGrad
    accepts SciPy's iterate as returned. We accept it only within 1e-4 of the simplex and
    then project it exactly, which is stricter than upstream. Rescaling fixes the v1 failure
    at 1B-scale Gram magnitudes ('Optimization terminated successfully' with a 1e-7 sum miss).
    """
    m=len(K);b=np.full(m,1/m);c=alpha*np.sqrt(K.mean()+1e-8)+1e-8
    scale=float(np.mean(np.diag(K)));scale=scale if np.isfinite(scale) and scale>0 else 1.
    Ks=np.asarray(K,float)/scale;cs=alpha*np.sqrt(max(float(Ks.mean()),0.)+1e-8)+1e-8
    def objective(x):return float(x@Ks@b+cs*np.sqrt(max(float(x@Ks@x),0.)+1e-8))
    fit=minimize(objective,b,method='SLSQP',bounds=[(0,1)]*m,constraints={'type':'eq','fun':lambda x:1-x.sum()},options={'ftol':1e-12,'maxiter':500})
    raw=np.asarray(fit.x,float)
    if not np.isfinite(raw).all() or raw.min()<-1e-4 or abs(raw.sum()-1)>1e-4:
        raise ArithmeticError('CAGrad simplex solve unresolved: '+str(fit.message))
    x=np.clip(raw,0.,None);x/=x.sum()
    w=(b+(c/(np.sqrt(max(float(x@K@x),0))+1e-8))*x)/(1+alpha**2)
    return w,{'success':bool(fit.success),'iterations':int(fit.nit),'objective':float(fit.fun)*scale,'simplex_weights':x.tolist(),
              'gram_scale':scale,'raw_simplex_sum_error':float(abs(raw.sum()-1)),'raw_simplex_min':float(raw.min())}

class LanguageRule:
    METHODS=('direct','uniform','cagrad','fa','periodic','famo')
    def __init__(self,method,m,device,system,options=None):
        if method not in self.METHODS:raise ValueError(method)
        self.method,self.m,self.system=method,m,copy.deepcopy(system)
        defaults={'fa_beta':.5,'fa_rho':.5,'fa_warm_steps':40,'period':5,'period_damping':.5,'periodic_backend':'native',
                  'cagrad_alpha':.4,'famo_weight_lr':.025,'famo_gamma':.01}
        if set(options or {})-set(defaults):raise ValueError('Unknown method option')
        self.options={**defaults,**(options or {})};self.next_step=0;self.weights=np.full(m,1/m)
        if not isinstance(self.options['period'],int) or self.options['period']<1 or not 0<self.options['period_damping']<=1:
            raise ValueError('Invalid periodic refinement configuration')
        if self.options['periodic_backend'] not in ('native','factor'):raise ValueError('Unknown periodic backend')
        if not isinstance(self.options['fa_warm_steps'],int) or self.options['fa_warm_steps']<0 or self.options['fa_beta']<=0 or self.options['fa_rho']<0:
            raise ValueError('Invalid FA configuration')
        if self.options['cagrad_alpha']<0 or self.options['famo_weight_lr']<=0 or self.options['famo_gamma']<0:
            raise ValueError('Invalid baseline configuration')
        self.incomplete=False
        self.gram_captures=0;self.periodic_refinements=0;self.post_forwards=0
        self.famo_logits=None;self.famo_opt=None
        if method=='famo':
            self.famo_logits=torch.zeros(m,device=device,requires_grad=True)
            self.famo_opt=torch.optim.Adam([self.famo_logits],lr=self.options['famo_weight_lr'],weight_decay=self.options['famo_gamma'])

    def state(self):
        if self.incomplete:raise RuntimeError('Cannot checkpoint a failed/incomplete update; restore the complete model and rule checkpoint')
        return {'method':self.method,'m':self.m,'system':copy.deepcopy(self.system),'options':copy.deepcopy(self.options),
                'next_step':self.next_step,'weights':self.weights.copy(),'gram_captures':self.gram_captures,
                'periodic_refinements':self.periodic_refinements,'post_forwards':self.post_forwards,
                'famo_logits':None if self.famo_logits is None else self.famo_logits.detach().clone(),
                'famo_optimizer':None if self.famo_opt is None else copy.deepcopy(self.famo_opt.state_dict())}

    def restore(self,s):
        for k in ('method','m','system','options'):
            if s[k]!=getattr(self,k):raise ValueError('Method identity changed: '+k)
        for k in ('next_step','weights','gram_captures','periodic_refinements','post_forwards'):setattr(self,k,copy.deepcopy(s[k]))
        if self.famo_logits is not None:
            with torch.no_grad():self.famo_logits.copy_(s['famo_logits'])
            self.famo_opt.load_state_dict(s['famo_optimizer'])
        self.incomplete=False

    def step(self,model,b,optimizer,step):
        if self.incomplete:raise RuntimeError('Failed/incomplete update; restore complete model and rule state before retry')
        self.incomplete=True
        result=self._step(model,b,optimizer,step)
        self.incomplete=False
        return result

    def _step(self,model,b,optimizer,step):
        if step!=self.next_step:raise ValueError('Method cursor mismatch')
        if not isinstance(optimizer,torch.optim.SGD):raise ValueError('Plain SGD required')
        rates={g['lr'] for g in optimizer.param_groups}
        if len(rates)!=1 or next(iter(rates))<=0 or any(g.get('momentum',0) or g.get('weight_decay',0) or g.get('nesterov',False) or g.get('maximize',False) for g in optimizer.param_groups):
            raise ValueError('Matched positive-LR SGD without momentum/decay required')
        lr=next(iter(rates));sync(model);begin=time.perf_counter()
        model.eval();optimizer.zero_grad(set_to_none=True);times={};meta={};cache=None;K=None
        def timed(name,f):
            sync(model);tick=time.perf_counter();v=f();sync(model);times[name]=time.perf_counter()-tick;return v
        full=self.method in ('direct','cagrad') or (self.method=='fa' and step==0 and self.options['fa_warm_steps']>0)
        refine=self.method=='periodic' and step%self.options['period']==0
        if refine and self.options['periodic_backend']=='native':
            direction,w,before,detail=timed('native_action_and_segment',lambda:native_periodic(model,b,self.weights,self.options['period_damping']))
            meta.update(detail);self.periodic_refinements+=int(detail['refinement_applied'])
            finite([before],'task losses')
            for n,p in model.named_parameters():
                if p.requires_grad:p.grad=direction[n].detach().to(dtype=p.dtype)
            del direction
        elif full or refine:
            def capture():
                cls=OnlineFactorGram if full and self.system['mode']=='online' else FactorizedJacobian
                kwargs={'compute_dtype':torch.float32}
                if cls is OnlineFactorGram:kwargs.update(row_chunk=self.system['row_chunk'],scratch_bytes=self.system['scratch_mib']*2**20)
                else:kwargs['compact_zero_rows']=True
                with cls(model,b['owner'],self.m,**kwargs) as obj:obj.capture(losses(model,b))
                return obj
            cache=timed('capture',capture);before=reduce_tasks(cache.losses,b).detach();finite([before],'task losses')
            if full:
                K=timed('gram',lambda:cache.gram() if self.system['mode']=='online' else ExactFactorGram(cache,row_chunk=self.system['row_chunk'],scratch_bytes=self.system['scratch_mib']*2**20).gram())
                K=timed('transfer',lambda:K.double().cpu().numpy());K=(K+K.T)/2
                if not np.isfinite(K).all():raise FloatingPointError('Nonfinite Gram')
                self.gram_captures+=1
            elif refine:
                response=timed('action',lambda:cache.matvec(self.weights).double().cpu().numpy())
                if not np.isfinite(response).all():raise FloatingPointError('Nonfinite periodic action')
                q=float(self.weights@response);j=int(np.argmin(response));w=self.weights.copy()
                def diagonal():
                    basis=np.zeros(self.m);basis[j]=1
                    selected=cache.direction(basis)
                    value=sum(float(g.double().square().sum()) for g in selected.values())
                    del selected
                    return value
                diag=timed('selected_gradient_diagonal',diagonal)
                denominator=diag-2*float(response[j])+q
                if not np.isfinite([diag,q,denominator]).all():raise FloatingPointError('Nonfinite factor segment geometry')
                exact_step=float(np.clip((q-float(response[j]))/denominator,0,1)) if denominator>0 else 0.
                executed=self.options['period_damping']*exact_step
                w*=1-executed;w[j]+=executed;self.periodic_refinements+=int(executed>0)
                meta.update(action_response=response.tolist(),refinement_applied=executed>0,
                    selected_diagonal=diag,exact_segment_step=exact_step,executed_segment_step=executed,
                    periodic_kernel='retained damped exact segment rule; factor-cache acquisition replaces native VJP/JVP/sparse-VJP plumbing')
            if full:
                def choose():
                    if self.method=='direct':
                        fit=solve(K);meta.update(qp_gap=fit['dual_gap'],recovered_gram=K.tolist());return fit['weights']
                    if self.method=='cagrad':
                        result,detail=cagrad_weights(K,self.options['cagrad_alpha']);meta['cagrad_solver']=detail;return result
                    w=self.weights.copy();rho=self.options['fa_rho'];rate=1/max(np.abs(K).sum(1).max()+rho,1e-100)
                    for _ in range(self.options['fa_warm_steps']):w=project_simplex(w-rate*(K@w+rho*w))
                    return w
                w=timed('weight_solve',choose)
            if full and (self.system['mode']=='online' or self.system.get('reconstruction')=='weighted_native'):
                del cache
                direction=timed('reconstruction',lambda:native_direction(model,b,w))
            else:
                direction=timed('reconstruction',lambda:cache.direction(w));del cache
            for n,p in model.named_parameters():
                if p.requires_grad:p.grad=direction[n].detach().to(dtype=p.dtype)
            del direction
        else:
            l=timed('forward',lambda:task_losses(model,b));finite([l],'task losses');before=l.detach()
            if self.method=='famo':
                z=self.famo_logits.softmax(-1);D=l+1e-8;c=(z/D).sum().detach()
                objective=(D.log()*z/c).sum();w=((z/D)/c).detach().cpu().double().numpy()
            else:
                w=self.weights.copy();objective=l@torch.as_tensor(w,device=l.device,dtype=l.dtype)
            timed('backward',objective.backward)
            del l,objective
        finite([p.grad for p in model.parameters() if p.requires_grad],'model gradients')
        timed('sgd',optimizer.step);finite([p for p in model.parameters() if p.requires_grad],'updated parameters')
        self.weights=np.asarray(w,dtype=np.float64).copy()
        if self.method in ('fa','famo'):
            def post_forward():
                with torch.no_grad():return task_losses(model,b)
            after=timed('same_batch_post_forward',post_forward);finite([after],'post-update losses');self.post_forwards+=1
            def update_rule():
                if self.method=='fa':
                    secant=(before-after).double().cpu().numpy()/lr
                    self.weights=project_simplex(self.weights-self.options['fa_beta']*(secant+self.options['fa_rho']*self.weights))
                else:
                    delta=(before+1e-8).log()-(after+1e-8).log()
                    grad=torch.autograd.grad(self.famo_logits.softmax(-1),self.famo_logits,grad_outputs=delta.detach())[0]
                    self.famo_opt.zero_grad(set_to_none=True);self.famo_logits.grad=grad;self.famo_opt.step()
                    finite([self.famo_logits]+[v for st in self.famo_opt.state.values() for v in st.values() if torch.is_tensor(v)],'FAMO state')
            timed('method_state_update',update_rule)
        if not np.isfinite(self.weights).all():raise FloatingPointError('Nonfinite weight state')
        self.next_step+=1;sync(model)
        return {'step':step,'weights':np.asarray(w).tolist(),'losses_before':before.cpu().tolist(),'update_seconds':time.perf_counter()-begin,
                'phase_seconds':times,'gram_captures':self.gram_captures,'periodic_refinements':self.periodic_refinements,
                'post_update_forwards':self.post_forwards,'adaptive_actions_this_step':int(refine),
                'per_step_interval_certificate':False,'scope':'all update arithmetic including finite checks, solve and extra forwards; excludes batch preparation/evaluation',**meta}
