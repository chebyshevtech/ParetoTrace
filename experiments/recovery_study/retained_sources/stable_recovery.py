"""Recovery-driven ParetoTrace with orthonormal actions and full-span correction.

Exact theory: acquisition_guarantee.tex. Floating-point gates are diagnostics,
not an outward-rounded enclosure. No nonpositive energy is a relative stop.
"""
import time
import numpy as np
from scipy.optimize import minimize,linprog
from v46_baseline_frontier import qp

def upper_span(Q,Z,start,anchor):
    raw=Q.T@Z;H=(raw+raw.T)/2
    scale=max(float(np.linalg.norm(H,2)),1e-100)
    if np.linalg.norm(raw-raw.T)>2e-5*scale or np.linalg.eigvalsh(H)[0]<-2e-6*scale:
        raise ArithmeticError('inconsistent acquired symmetric-PSD core')
    a=Q.sum(0);c0=Q.T@start;Hn=H/scale
    fit=minimize(lambda c:float(c@Hn@c),c0,jac=lambda c:2*Hn@c,method='SLSQP',constraints=[
        {'type':'eq','fun':lambda c:a@c-1,'jac':lambda c:a},
        {'type':'ineq','fun':lambda c:Q@c,'jac':lambda c:Q}],options={'ftol':1e-12,'maxiter':300})
    if not fit.success:raise ArithmeticError('upper QP failed: '+fit.message)
    c=fit.x;x=Q@c
    E=np.vstack([a,Q[x<1e-7]])
    system=np.block([[Hn,E.T],[E,np.zeros((len(E),len(E)))]])
    polished=np.linalg.lstsq(system,np.r_[np.zeros(len(c)),1.,np.zeros(len(E)-1)],rcond=1e-13)[0][:len(c)]
    if abs(a@polished-1)<1e-10 and (Q@polished).min()>=-1e-10 and polished@Hn@polished<=c@Hn@c+1e-12:
        c=polished;x=Q@c
    if x.min() < -1e-8 or abs(x.sum()-1)>1e-8:raise ArithmeticError('upper infeasibility')
    # Start is a known feasible point in this nested span. Repair tiny
    # negatives via that point only when it has positive entries there.
    if x.min()<0:
        bad=x<0
        if (anchor[bad]<=0).any():raise ArithmeticError('cannot repair upper in span')
        mix=max(-x[bad]/(anchor[bad]-x[bad]))+1e-14
        c=(1-mix)*c+mix*(Q.T@anchor)
    c/=a@c;x=Q@c;y=Z@c
    return x,y,float(x@y),H

def recover(action,M,*,start=None,lower_rule='completion',eps=.05,gamma=.05,max_actions=None):
    if lower_rule not in ('completion','margin'):raise ValueError(lower_rule)
    anchor=np.full(M,1/M) if start is None else np.asarray(start,float).copy()
    # Positive anchor permits only roundoff repair; mixing is charged as the
    # actual initial query, not an unobserved change to the candidate.
    anchor=np.maximum(anchor,0);anchor/=anchor.sum();anchor=(1-1e-6)*anchor+1e-6/M
    Q=np.zeros((M,0));Z=np.zeros((M,0));mu=anchor.copy();previous=anchor.copy()
    history=[];action_seconds=solver_seconds=0.;query_calls=0;best_lower=0.;budget=M if max_actions is None else max_actions
    begin=time.perf_counter();status='action_budget';x=anchor;y=None;U=None
    try:
      for _ in range(budget):
        v=mu-Q@(Q.T@mu);v-=Q@(Q.T@v);norm=np.linalg.norm(v)
        if norm<1e-11:raise ArithmeticError('nonterminal query in acquired span')
        v/=norm;tick=time.perf_counter();query_calls+=1;z=np.asarray(action(v),float).reshape(M);action_seconds+=time.perf_counter()-tick
        if not np.isfinite(z).all():raise ArithmeticError('nonfinite action')
        Q=np.column_stack([Q,v]);Z=np.column_stack([Z,z]);tick=time.perf_counter()
        x,y,U,H=upper_span(Q,Z,previous,anchor)
        # Never certify a nonpositive or unresolved-scale energy.
        if U<=0 or not np.isfinite(U):raise ArithmeticError('nonpositive candidate energy')
        margin=float(y.min());cheap=max(0.,margin)**2/U
        best_lower=max(best_lower,cheap)
        ok_gain=margin>=(1-gamma)*U
        if best_lower>0 and U<=(1+eps)*best_lower and ok_gain:
            status='accepted';solver_seconds+=time.perf_counter()-tick
            history.append({'actions':Q.shape[1],'U':U,'lower':best_lower,'min_response':margin,'lower_qp_used':False});break
        eig,V=np.linalg.eigh(H);cut=max(eig[-1]*1e-12,1e-100)
        keep=eig>cut
        F=(Z@V[:,keep])/np.sqrt(eig[keep]);A=F@F.T
        residual=np.linalg.norm(A@Q-Z)/max(np.linalg.norm(Z),1e-100)
        if residual>2e-5:raise ArithmeticError('completion interpolation failed')
        lo=qp(A,tolerance=2e-12,max_iterations=1000)
        if not lo['converged']:raise ArithmeticError('lower QP failed')
        if lower_rule=='completion':best_lower=max(best_lower,float(lo['lower']))
        mu=lo['weights'];previous=x
        solver_seconds+=time.perf_counter()-tick
        history.append({'actions':Q.shape[1],'U':U,'lower':best_lower,'min_response':margin,'lower_qp_used':True,'completion_lower':float(lo['lower']),'interpolation':float(residual)})
        if best_lower>0 and U<=(1+eps)*best_lower and ok_gain:status='accepted';break
    except (ArithmeticError,np.linalg.LinAlgError) as exc:
      status='unresolved_numerics';error=str(exc)
    out={'weights':x,'response':y,'objective':U,'lower':best_lower,'actions':query_calls,'status':status,'accepted':status=='accepted','history':history,'action_seconds':action_seconds,'solver_seconds':solver_seconds,'total_seconds':time.perf_counter()-begin,'precision_contract':'numerical diagnostics; independent native audits required'}
    if status=='unresolved_numerics':out['error']=error
    return out
