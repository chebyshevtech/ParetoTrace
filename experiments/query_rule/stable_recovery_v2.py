"""ParetoTrace-Stream recovery with a target-aware query rule (rebuttal improvement).

Identical to retained_sources/stable_recovery.recover except for the choice of the
next query vector. The upper solve (stable_recovery.upper_span), the completion
lower QP (v46_baseline_frontier.qp), the interpolation check and the stopping test
are imported or copied unchanged, so validity is unaffected: the completion lower
bound is valid for any transcript, whatever queries produced it.

query_rule='completion'  current rule: query the off-span part of the completion lower minimizer.
query_rule='gate'        target-aware rule: as above until the energy test U <= (1+eps) L holds,
                         then, while the response gate fails, query the off-span part of the
                         coordinate of the task with the smallest exact response.
gamma sets the response gate min_i (K lam)_i >= (1-gamma) U. gamma=1 asks only for
non-negative responses; acceptance then also requires L > 0, so every accepted stop
certifies the energy target, and strict descent is checked by the independent audit.
"""
import time
import numpy as np
from stable_recovery import upper_span
from v46_baseline_frontier import qp


def recover(action, M, *, start=None, lower_rule='completion', query_rule='completion', eps=.05, gamma=.05, max_actions=None):
    if lower_rule not in ('completion', 'margin'):
        raise ValueError(lower_rule)
    if query_rule not in ('completion', 'gate'):
        raise ValueError(query_rule)
    anchor = np.full(M, 1 / M) if start is None else np.asarray(start, float).copy()
    anchor = np.maximum(anchor, 0); anchor /= anchor.sum(); anchor = (1 - 1e-6) * anchor + 1e-6 / M
    Q = np.zeros((M, 0)); Z = np.zeros((M, 0)); mu = anchor.copy(); previous = anchor.copy()
    history = []; action_seconds = solver_seconds = 0.; query_calls = 0; best_lower = 0.
    budget = M if max_actions is None else max_actions
    begin = time.perf_counter(); status = 'action_budget'; x = anchor; y = None; U = None; gate_queries = 0
    try:
        for _ in range(budget):
            v = None
            if query_rule == 'gate' and y is not None and best_lower > 0 and U <= (1 + eps) * best_lower and float(y.min()) < (1 - gamma) * U:
                for j in np.argsort(y):
                    e = np.zeros(M); e[j] = 1.
                    cand = e - Q @ (Q.T @ e); cand -= Q @ (Q.T @ cand); n = np.linalg.norm(cand)
                    if n > 1e-8:
                        v = cand / n; gate_queries += 1
                        break
            if v is None:
                v = mu - Q @ (Q.T @ mu); v -= Q @ (Q.T @ v); norm = np.linalg.norm(v)
                if norm < 1e-11:
                    raise ArithmeticError('nonterminal query in acquired span')
                v /= norm
            tick = time.perf_counter(); query_calls += 1; z = np.asarray(action(v), float).reshape(M); action_seconds += time.perf_counter() - tick
            if not np.isfinite(z).all():
                raise ArithmeticError('nonfinite action')
            Q = np.column_stack([Q, v]); Z = np.column_stack([Z, z]); tick = time.perf_counter()
            x, y, U, H = upper_span(Q, Z, previous, anchor)
            if U <= 0 or not np.isfinite(U):
                raise ArithmeticError('nonpositive candidate energy')
            margin = float(y.min()); cheap = max(0., margin) ** 2 / U
            best_lower = max(best_lower, cheap)
            ok_gain = margin >= (1 - gamma) * U
            if best_lower > 0 and U <= (1 + eps) * best_lower and ok_gain:
                status = 'accepted'; solver_seconds += time.perf_counter() - tick
                history.append({'actions': Q.shape[1], 'U': U, 'lower': best_lower, 'min_response': margin, 'lower_qp_used': False}); break
            eig, V = np.linalg.eigh(H); cut = max(eig[-1] * 1e-12, 1e-100)
            keep = eig > cut
            F = (Z @ V[:, keep]) / np.sqrt(eig[keep]); A = F @ F.T
            residual = np.linalg.norm(A @ Q - Z) / max(np.linalg.norm(Z), 1e-100)
            if residual > 2e-5:
                raise ArithmeticError('completion interpolation failed')
            lo = qp(A, tolerance=2e-12, max_iterations=1000)
            if not lo['converged']:
                raise ArithmeticError('lower QP failed')
            if lower_rule == 'completion':
                best_lower = max(best_lower, float(lo['lower']))
            mu = lo['weights']; previous = x
            solver_seconds += time.perf_counter() - tick
            history.append({'actions': Q.shape[1], 'U': U, 'lower': best_lower, 'min_response': margin, 'lower_qp_used': True,
                            'completion_lower': float(lo['lower']), 'interpolation': float(residual)})
            if best_lower > 0 and U <= (1 + eps) * best_lower and ok_gain:
                status = 'accepted'; break
    except (ArithmeticError, np.linalg.LinAlgError) as exc:
        status = 'unresolved_numerics'; error = str(exc)
    out = {'weights': x, 'response': y, 'objective': U, 'lower': best_lower, 'actions': query_calls, 'status': status,
           'accepted': status == 'accepted', 'history': history, 'action_seconds': action_seconds, 'solver_seconds': solver_seconds,
           'total_seconds': time.perf_counter() - begin, 'gate_queries': gate_queries, 'query_rule': query_rule, 'gamma': gamma}
    if status == 'unresolved_numerics':
        out['error'] = error
    return out
