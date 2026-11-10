"""Numerical check of the residual-rank action bound (theory/residual_rank_theorem.md).

Runs the completion query rule on random instances K = C + R, with C known and R reached only through actions
v -> R v, and checks that (a) every query made while L < U lies outside the acquired span S (Lemma 3), (b) every
query that does not close the interval raises dim F S by one (Lemma 4), and (c) the rule stops after at most
min(M, rank R + 1) actions, initialization included. Floating-point solves stand in for the exact solves of the
theorem, so closure L = U is tested to 1e-8 of the Gram's mean diagonal (this also covers q* = 0).

Run from the repository root:  python3 audits/verify_residual_rank.py
"""
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'paretotrace' / 'stream'))
from v46_baseline_frontier import qp  # noqa: E402  exact Wolfe active-set solve over the simplex

TOL = 1e-8


def lower(C, Q, Y):
    """L = min over the simplex of mu' A mu, A = C + Y (Q'Y)^+ Y' (shorted completion), and dim F S = rank Q'Y."""
    W = (Q.T @ Y + Y.T @ Q) / 2
    values, vectors = np.linalg.eigh(W)
    keep = values > max(float(values[-1]) * 1e-12, 1e-300)
    factor = (Y @ vectors[:, keep]) / np.sqrt(values[keep])[None, :]
    A = C + factor @ factor.T
    mu = np.asarray(qp(A)['weights'], float)
    return float(mu @ A @ mu), mu, int(keep.sum())


def upper(C, Q, Y, start):
    """U = min over the simplex intersected with span Q of lam' K lam, computed from the transcript only."""
    H = Q.T @ C @ Q + (Q.T @ Y + Y.T @ Q) / 2
    scale = max(float(np.trace(H)) / len(H), 1e-300)
    ones = Q.sum(axis=0)
    sol = minimize(lambda c: float(c @ H @ c) / scale, Q.T @ start, jac=lambda c: 2 * H @ c / scale, method='SLSQP',
                   constraints=[dict(type='eq', fun=lambda c: ones @ c - 1, jac=lambda c: ones),
                                dict(type='ineq', fun=lambda c: Q @ c, jac=lambda c: Q)],
                   options=dict(ftol=1e-15, maxiter=5000))
    c = sol.x
    return float(c @ H @ c), Q @ c


def run(C, R):
    """Completion query rule. Returns actions, whether every query left the span, the dim F S trace, and L, U."""
    M = len(C)
    u = np.asarray(qp(C)['weights'], float) if np.abs(C).max() > 0 else np.full(M, 1.0 / M)
    Q = (u / np.linalg.norm(u))[:, None]
    Y = R @ Q
    scale = max(float(np.trace(C + R)) / M, 1e-300)  # closure is judged on the Gram's scale, so q* = 0 is handled
    lam, actions, outside, dims = u.copy(), 1, True, []
    while True:
        L, mu, dimT = lower(C, Q, Y)
        U, lam = upper(C, Q, Y, lam)
        dims.append(dimT)
        if L >= U - TOL * scale or Q.shape[1] == M:
            return actions, outside, dims, L, U
        off = mu - Q @ (Q.T @ mu)
        if np.linalg.norm(off) <= 1e-10 * np.linalg.norm(mu):
            return actions, False, dims, L, U  # a lower minimizer inside S while L < U would contradict Lemma 3
        v = off / np.linalg.norm(off)
        Q = np.column_stack([Q, v])
        Y = np.column_stack([Y, R @ v])
        actions += 1


def instance(rng, M, r, c_rank, conflict):
    F = rng.standard_normal((M, r))
    if conflict:  # opposing task gradients, so that uniform weighting does not descend on every task
        F[: M // 2] *= -1.0
        F += 0.3 * rng.standard_normal((M, r))
    C = np.zeros((M, M))
    if c_rank:
        G = rng.standard_normal((M, c_rank))
        C = G @ G.T * rng.uniform(0.05, 2.0)
    return (C + C.T) / 2, F @ F.T


def check():
    """Run the 608 instances. Returns (instances, failures, attained, smallest slack, failure records)."""
    rng = np.random.default_rng(20261110)
    trials, smallest_slack, fails, used = 0, None, [], {}
    for M in (5, 8, 12, 20):
        for r in (1, 2, 3, 4, 6):
            if r > M:
                continue
            for c_rank in sorted({0, 1, M // 2, M}):
                for conflict in (False, True):
                    for _ in range(4):
                        C, R = instance(rng, M, r, c_rank, conflict)
                        actions, outside, dims, L, U = run(C, R)
                        bound = min(M, r + 1)
                        grows = all(dims[j + 1] == dims[j] + 1 for j in range(len(dims) - 2))
                        closed = L >= U - TOL * max(float(np.trace(C + R)) / M, 1e-300)
                        trials += 1
                        used[actions - bound] = used.get(actions - bound, 0) + 1
                        smallest_slack = bound - actions if smallest_slack is None else min(smallest_slack, bound - actions)
                        if not (outside and grows and closed and actions <= bound):
                            fails.append(dict(M=M, r=r, c_rank=c_rank, conflict=conflict, actions=actions, bound=bound,
                                              outside=outside, grows=grows, L=L, U=U, dims=dims))
    return trials, len(fails), used.get(0, 0), smallest_slack, fails, used, rng


def main():
    trials, nfail, attained, smallest_slack, fails, used, rng = check()
    print('instances', trials, '| failures', nfail, '| bound attained', attained, '| smallest slack, min(M, r+1) - actions:', smallest_slack)
    print('actions minus bound, counts:', dict(sorted(used.items())))
    for f in fails[:10]:
        print('  FAIL', f)
    full = []
    for M in (8, 16):  # nothing contracted (C = 0) and a full-rank R: the bound is M, vacuous for live training
        F = rng.standard_normal((M, M))
        full.append((M, run(np.zeros((M, M)), F @ F.T)[0]))
    print('full rank R, C = 0, actions (bound M):', full)
    sys.exit(1 if fails else 0)


if __name__ == '__main__':
    main()
