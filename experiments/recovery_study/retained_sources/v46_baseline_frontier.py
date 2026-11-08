"""Action-oracle MGDA frontier. NumPy only; no neural-training claims.

The oracle sees every query, including unsuccessful line-search queries.  No
method reads the hidden Gram, its spectrum, or the reference solution.  Dense
CPU timings quantify task-space overhead, not GPU backward-pass savings.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np


def project_simplex(v):
    v = np.asarray(v, dtype=np.float64)
    ordered = np.sort(v)[::-1]
    offsets = (np.cumsum(ordered) - 1) / np.arange(1, len(v) + 1)
    rho = np.flatnonzero(ordered > offsets)[-1]
    return np.maximum(v - offsets[rho], 0)


def qp(matrix, start=None, tolerance=2e-11, max_iterations=10000):
    """Wolfe active-set quadratic solve with affine minor cycles.

    This is a strong classical baseline, not a proposed research contribution.
    The returned dual gap exposes unsuccessful convergence independently of
    the internal stopping flag, including singular and near-stationary cases.
    """
    K = np.asarray(matrix, dtype=np.float64)
    K = (K + K.T) / 2
    m = len(K)
    scale = max(float(np.trace(K) / m), 1e-100)
    H = K / scale
    x = np.zeros(m)
    if start is None:
        x[int(np.argmin(np.diag(H)))] = 1
    else:
        x[:] = project_simplex(start)
    support = list(np.flatnonzero(x > 1e-14))
    iterations = 0
    for iterations in range(1, max_iterations + 1):
        y = H @ x
        q = float(x @ y)
        entering = int(np.argmin(y))
        gap = max(0.0, 2 * (q - y[entering]))
        if gap <= tolerance * max(1.0, abs(q)):
            break
        if entering not in support:
            support.append(entering)
        old = x[support].copy()
        for _ in range(2 * m + 10):
            block = H[np.ix_(support, support)]
            n = len(support)
            system = np.empty((n + 1, n + 1))
            system[:n, :n] = block
            system[:n, n] = -1
            system[n, :n] = 1
            system[n, n] = 0
            rhs = np.zeros(n + 1)
            rhs[-1] = 1
            try:
                affine = np.linalg.solve(system, rhs)[:n]
            except np.linalg.LinAlgError:
                affine = np.linalg.lstsq(system, rhs, rcond=1e-13)[0][:n]
            if affine.min() >= -2e-13:
                old = np.maximum(affine, 0)
                old /= old.sum()
                break
            negative = affine < 0
            alpha = float(np.min(old[negative] / (old[negative] - affine[negative])))
            old += alpha * (affine - old)
            keep = old > 1e-13
            support = [j for j, retained in zip(support, keep) if retained]
            old = old[keep]
        else:
            raise RuntimeError("active-set minor cycle failed")
        x.fill(0)
        x[support] = old
    y = K @ x
    objective = float(x @ y)
    lower = max(0.0, min(objective, 2 * float(y.min()) - objective))
    return dict(weights=x, response=y, objective=objective, lower=lower,
                dual_gap=objective-lower, iterations=iterations,
                converged=bool(objective-lower <= tolerance*max(scale, objective)*1.05))


class ActionOracle:
    def __init__(self, apply, dimension):
        self.apply = apply
        self.dimension = dimension
        self.actions = 0
        self.calls = 0
        self.seconds = 0.0
        self.query_records = []

    def __call__(self, v):
        start = time.perf_counter()
        output = np.asarray(self.apply(v), dtype=np.float64)
        self.seconds += time.perf_counter() - start
        self.calls += 1
        self.actions += 1 if v.ndim == 1 else v.shape[1]
        columns = v[:,None] if v.ndim==1 else v
        for column in columns.T:
            self.query_records.append(dict(nonzero_objectives=int(np.count_nonzero(np.abs(column)>1e-14)),
                                           l1_norm=float(np.abs(column).sum()),norm=float(np.linalg.norm(column))))
        return output


def fw_lower(x, y):
    q = float(x @ y)
    # Sharp one-action bound from Cauchy--Schwarz. This dominates the
    # conventional Frank--Wolfe linearization lower bound for PSD quadratics.
    minimum = max(0.0, float(y.min()))
    return min(q, minimum*minimum/q) if q > 0 else 0.0


def stopped(x, y, lower, rel_tol, abs_tol):
    q = float(x @ y)
    return q - lower <= max(abs_tol, rel_tol * lower)


def coordinate_span_basis(dimension, selected):
    """Vertices of Δ intersect span(uniform, queried coordinates)."""
    remaining = np.setdiff1d(np.arange(dimension), selected)
    B = np.eye(dimension)[:, selected]
    if len(remaining):
        residual = np.zeros(dimension)
        residual[remaining] = 1 / len(remaining)
        B = np.column_stack([B, residual])
    return B


def _shorted_lower(Omega, Y, start=None):
    W = (Omega.T @ Y + Y.T @ Omega) / 2
    values, vectors = np.linalg.eigh(W)
    keep = values > max(float(values[-1]) * 2e-13, 1e-100)
    factor = (Y @ vectors[:, keep]) / np.sqrt(values[keep])[None, :]
    completion = factor @ factor.T
    result = qp(completion, start=start)
    return result["lower"], result["weights"]


def _general_upper(Omega, Y, previous):
    from scipy.optimize import minimize
    U,s,Vt=np.linalg.svd(Omega,full_matrices=False)
    keep=s>s[0]*1e-12
    Q=U[:,keep]
    response_basis=Y@((Vt[keep,:].T)/s[keep][None,:])
    H=(Q.T@response_basis+response_basis.T@Q)/2
    scale=max(float(np.trace(H)/len(H)),1e-30)
    equality=Q.sum(axis=0)
    solution=minimize(lambda c:float(c@H@c)/scale,Q.T@previous,
                      jac=lambda c:2*H@c/scale,method="SLSQP",
                      constraints=[dict(type="eq",fun=lambda c:equality@c-1,jac=lambda c:equality),
                                   dict(type="ineq",fun=lambda c:Q@c,jac=lambda c:Q)],
                      options=dict(ftol=2e-13,maxiter=2000))
    if not solution.success: raise RuntimeError(solution.message)
    c=solution.x
    candidate=Q@c
    if candidate.min() < -1e-8: raise RuntimeError("infeasible span candidate")
    # Uniform anchor repairs roundoff while preserving query span.
    if candidate.min()<0:
        anchor=np.full(len(candidate),1/len(candidate))
        alpha=max(-candidate[candidate<0]/(anchor[candidate<0]-candidate[candidate<0]))+1e-14
        c=(1-alpha)*c+alpha*(Q.T@anchor)
        candidate=Q@c
    return candidate,response_basis@c


def solve_action(K_apply, M, method, rel_tol=0.05, max_actions=None,
                 abs_tol=1e-12, start=None):
    """Solve using only action access; return raw, auditable accounting.

    Methods: full_gram, pg, fista, fw, fully_corrective, span_fw,
    minimax_coordinate. span_fw has the minimax upper decision but cheap FW
    stopping; minimax_coordinate adds the full post-sketch lower endpoint.
    """
    max_actions = max_actions or 4 * M
    oracle = ActionOracle(K_apply, M)
    beginning = time.perf_counter()
    x = np.full(M, 1/M) if start is None else project_simplex(start)
    history = []
    stored_bytes = 0
    if method == "full_gram":
        K = oracle(np.eye(M))
        answer = qp(K)
        x, y, lower = answer["weights"], answer["response"], answer["lower"]
        stored_bytes = int(K.nbytes)
    elif method in ("pg", "fista"):
        y = oracle(x)
        previous_x, previous_y = x.copy(), y.copy()
        acceleration = 1.0
        L = max(np.linalg.norm(y) * np.sqrt(M), 1e-10)
        lower = fw_lower(x, y)
        while oracle.actions < max_actions and not stopped(x, y, lower, rel_tol, abs_tol):
            next_acceleration = (1 + np.sqrt(1 + 4*acceleration**2))/2
            beta = ((acceleration-1)/next_acceleration) if method == "fista" else 0
            z = x + beta*(x-previous_x)
            yz = y + beta*(y-previous_y)  # exact action interpolation is free
            L *= 0.5
            while True:
                trial = project_simplex(z - yz/L)
                yt = oracle(trial)
                delta = trial-z
                curvature = float(delta @ (yt-yz))
                if curvature <= L*float(delta@delta) + 1e-14*max(1.0, abs(float(z@yz))):
                    break
                L *= 2
                if oracle.actions >= max_actions:
                    break
            previous_x, previous_y, x, y = x, y, trial, yt
            acceleration = next_acceleration
            if float((x-previous_x) @ (z-x)) > 0:
                acceleration = 1.0
            lower = max(lower, fw_lower(x, y))
            history.append([oracle.actions, float(x@y), lower])
        stored_bytes = 8*M*8
    elif method in ("lower_minimizer","lower_minimizer_hull","random_nystrom"):
        y=oracle(x)
        Omega,Y=x[:,None],y[:,None]
        lower=fw_lower(x,y)
        lower_weights=None
        rng=np.random.default_rng(771)
        while True:
            bound,lower_weights=_shorted_lower(Omega,Y,lower_weights)
            lower=max(lower,bound)
            history.append([oracle.actions,float(x@y),lower])
            if stopped(x,y,lower,rel_tol,abs_tol) or oracle.actions>=max_actions:break
            if method=="random_nystrom" and oracle.actions+1<max_actions:
                probe=rng.normal(size=M)
                probe/=np.linalg.norm(probe)
                Omega=np.column_stack([Omega,probe])
                Y=np.column_stack([Y,oracle(probe)])
                bound,lower_weights=_shorted_lower(Omega,Y,lower_weights)
            proposal=lower_weights
            coefficients=np.linalg.lstsq(Omega,proposal,rcond=1e-12)[0]
            if np.linalg.norm(Omega@coefficients-proposal)<=1e-11:
                proposal_response=Y@coefficients
                # No information is gained, but a previously unavailable
                # convex-hull upper point can close the interval for free.
                x,y=proposal,proposal_response
                lower=max(lower,fw_lower(x,y))
                if stopped(x,y,lower,rel_tol,abs_tol):break
                raise RuntimeError("dependent lower minimizer did not close interval")
            proposal_response=oracle(proposal)
            Omega=np.column_stack([Omega,proposal])
            Y=np.column_stack([Y,proposal_response])
            if method=="lower_minimizer_hull":
                correction=qp(Omega.T@Y)
                x,y=Omega@correction["weights"],Y@correction["weights"]
            else:
                x,y=_general_upper(Omega,Y,proposal)
            lower=max(lower,fw_lower(x,y))
            stored_bytes=max(stored_bytes,int(Omega.nbytes+Y.nbytes))
    elif method in ("fw", "fully_corrective", "span_fw", "minimax_coordinate"):
        u = x.copy()
        yu = oracle(u)
        y = yu.copy()
        Omega, Y = u[:, None], yu[:, None]
        selected = []
        lower_weights = None
        lower = fw_lower(x, y)
        while True:
            if method == "minimax_coordinate":
                bound, lower_weights = _shorted_lower(Omega, Y, lower_weights)
                lower = max(lower, bound)
            history.append([oracle.actions, float(x@y), lower])
            if stopped(x, y, lower, rel_tol, abs_tol) or oracle.actions >= max_actions:
                break
            remaining = np.setdiff1d(np.arange(M), selected)
            if len(remaining) == 0:
                break
            entering = int(remaining[np.argmin(y[remaining])])
            e = np.eye(1, M, entering).ravel()
            ye = oracle(e)
            selected.append(entering)
            Omega = np.column_stack([Omega, e])
            Y = np.column_stack([Y, ye])
            if method == "fw":
                d, yd = e-x, ye-y
                curvature = float(d@yd)
                alpha = np.clip(-float(d@y)/max(curvature, 1e-100), 0, 1)
                x, y = x+alpha*d, y+alpha*yd
                # Classical FW may revisit known coordinates at zero action
                # cost. Optimize over cached columns until another is needed.
                for _ in range(100):
                    j = int(np.argmin(y))
                    if j not in selected:
                        break
                    d = np.eye(1, M, j).ravel()-x
                    yd = Y[:, selected.index(j)+1]-y
                    curvature = float(d@yd)
                    alpha = np.clip(-float(d@y)/max(curvature, 1e-100), 0, 1)
                    x, y = x+alpha*d, y+alpha*yd
                    if stopped(x,y,fw_lower(x,y),rel_tol,abs_tol):
                        break
            else:
                if method == "fully_corrective":
                    B, KB = Omega, Y
                else:
                    if not np.allclose(u, np.full(M,1/M), rtol=0, atol=1e-14):
                        raise ValueError("coordinate-span fast path requires uniform start")
                    B = coordinate_span_basis(M, selected)
                    KB = Y[:,1:].copy()
                    if len(selected) < M:
                        residual_response = (M*yu - Y[:,1:].sum(axis=1))/(M-len(selected))
                        KB = np.column_stack([KB, residual_response])
                small = qp(B.T @ KB)
                x, y = B@small["weights"], KB@small["weights"]
            lower = max(lower,fw_lower(x,y))
            stored_bytes = max(stored_bytes, int(Omega.nbytes+Y.nbytes))
    else:
        raise ValueError(method)
    wall = time.perf_counter()-beginning
    objective = float(x@y)
    return dict(method=method, weights=x, response=y, objective=objective,
                lower=lower, actions=oracle.actions, oracle_calls=oracle.calls,
                wall_seconds=wall, oracle_seconds=oracle.seconds,
                query_records=oracle.query_records,
                task_space_bytes=stored_bytes, history=history,
                stopping_passed=stopped(x,y,lower,rel_tol,abs_tol),
                observed_factor=objective/lower if lower>0 else None)


def synthetic_case(M, rank, condition, support, geometry, seed):
    rng = np.random.default_rng(seed)
    if geometry == "diagonal":
        G = np.diag(np.geomspace(1, 1/np.sqrt(condition), M))
    else:
        rank = max(2, min(rank, M))
        support = max(2, min(support, rank))
        G = rng.normal(size=(rank, M))
        scale = np.geomspace(1,1/np.sqrt(condition),rank-1)
        G[1:] *= scale[:,None]
        G[1:,:support] -= G[1:,:support].mean(axis=1,keepdims=True)
        offset = 0.01 if geometry == "conflicting" else 1.0
        G[0,:] = offset
        G[0,support:] += (0.01 if geometry == "small_margin" else 0.3)
        permutation = rng.permutation(M)
        G = G[:,permutation]
    K = G.T@G
    K /= np.trace(K)/M
    return K


def evaluate(K, result, reference):
    x = result["weights"]
    y = K@x
    q = float(x@y)
    star = reference["objective"]
    delta = x-reference["weights"]
    direction_error = np.sqrt(max(0.0,float(delta@K@delta)))
    result.update(true_objective=q, objective_factor=q/star if star>1e-20 else None,
                  objective_error=q-star, direction_error=direction_error,
                  direction_relative_error=direction_error/np.sqrt(star) if star>1e-20 else None,
                  common_descent_margin=float(y.min()), strict_common_descent=bool(y.min()>0),
                  normalized_worst_gain=float(y.min())/np.sqrt(max(q*star,1e-100)),
                  response_relative_error=float(np.linalg.norm(y-result["response"])/max(np.linalg.norm(y),1e-100)))
    return result


def native(value):
    if isinstance(value,np.ndarray): return value.tolist()
    if isinstance(value,np.generic): return value.item()
    if isinstance(value,dict): return {k:native(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)): return [native(v) for v in value]
    return value


def benchmark(output, quick=False, neural_path=None):
    import scipy.optimize  # exclude first-import overhead from solver timing
    methods = ("full_gram","pg","fista","fw","fully_corrective","span_fw","minimax_coordinate","lower_minimizer","lower_minimizer_hull","random_nystrom")
    specs = []
    sizes = (32,) if quick else (32,64,128)
    for M in sizes:
        for rank in (4,min(16,M),M):
            for condition in (1.0,1e4):
                for support in (2,min(rank,M//2)):
                    for geometry in ("planted","conflicting","small_margin"):
                        specs.append(dict(M=M,rank=rank,condition=condition,support=support,geometry=geometry,
                                          seed=2026090800+len(specs)))
        for condition in (1.0,1e4,1e8):
            specs.append(dict(M=M,rank=M,condition=condition,support=M,geometry="diagonal",seed=2026090999+M))
    cases = [(s,synthetic_case(**s)) for s in specs]
    if neural_path:
        for path in sorted(Path(neural_path).glob("coordinate_interior*json")):
            payload = json.loads(path.read_text())
            for audit in payload.get("audits",[]):
                K = np.asarray(audit["exact_gram"])
                K /= np.trace(K)/len(K)
                cases.append((dict(M=len(K),geometry="stored_neural",seed=payload.get("seed"),source=str(path),
                                   step=audit.get("step")),K))
    rows = []
    output = Path(output)
    output.parent.mkdir(parents=True,exist_ok=True)
    for index,(spec,K) in enumerate(cases):
        reference=qp(K)
        values=np.linalg.eigvalsh(K)
        diagnostics=dict(reference_objective=reference["objective"],reference_dual_gap=reference["dual_gap"],
                         reference_support=int(np.count_nonzero(reference["weights"]>1e-8)),
                         effective_rank=float(np.trace(K)/max(values[-1],1e-100)),
                         numerical_rank=int(np.count_nonzero(values>values[-1]*1e-10)))
        for method in methods:
            try:
                result=solve_action(lambda v:K@v,len(K),method,max_actions=4*len(K))
                result=evaluate(K,result,reference)
                row=dict(case=index,spec=spec,diagnostics=diagnostics,result=result)
            except Exception as exc:
                row=dict(case=index,spec=spec,diagnostics=diagnostics,error=repr(exc),method=method)
            rows.append(native(row))
        print(f"case {index+1}/{len(cases)} {spec['geometry']} M={len(K)} "+
              " ".join(f"{r.get('result',{}).get('method',r.get('method'))}:{r.get('result',{}).get('actions','ERR')}" for r in rows[-len(methods):]),flush=True)
        payload=dict(protocol="v46_numpy_baseline_frontier_v2",scope="CPU action-oracle diagnostic; not neural training",rel_tol=0.05,
                     methods=list(methods),platform=platform.platform(),numpy=np.__version__,
                     source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),rows=rows)
        output.write_text(json.dumps(payload,indent=2,allow_nan=False))
    return payload


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",default="results/v46/baseline_frontier.json")
    parser.add_argument("--quick",action="store_true")
    parser.add_argument("--neural-path")
    args=parser.parse_args()
    benchmark(args.output,args.quick,args.neural_path)
