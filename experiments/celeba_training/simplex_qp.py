"""Classical Wolfe simplex solver from src/v46_baseline_frontier.py.
Source SHA256: 64a9fc44b6fef852ef245fb23b8797cffbfb64bbf3694477149e3a76956b8bda
No new research contribution claimed for this QP solver.
"""
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
