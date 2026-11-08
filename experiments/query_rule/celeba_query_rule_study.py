"""Live CelebA-40 timing of the target-aware query rule at the recovery-study states.

Same development checkpoint, the same five minibatches (seeds 9101-9105), the same
native action backend, timing scope and independent native audits as
celeba_recovery_study.py, whose helpers are imported unchanged. Only the Stream
query rule and the response-gate level vary. Development-only: no test data.
"""
import os
os.environ['OMP_NUM_THREADS'] = '1'; os.environ['OPENBLAS_NUM_THREADS'] = '1'
import argparse, hashlib, json, sys, time, traceback
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, default=Path('/content/rebuttal'))
    p.add_argument('--study', type=Path, default=Path('/content/rebuttal/recovery_study'))
    p.add_argument('--output', type=Path, default=HERE / 'query_rule_study.json')
    p.add_argument('--repeats', type=int, default=3)
    a = p.parse_args()
    c = a.root / 'celeba_v2'
    sys.path.insert(0, str(c)); sys.path.insert(0, str(a.study / 'retained_sources')); sys.path.insert(0, str(a.study)); sys.path.insert(0, str(HERE))
    import celeba_recovery_study as base
    from celeba_training import Model, batch, load_split, solve
    from preflight import SharedCapture
    from shared_native_actions import SharedNativeActions
    from stable_recovery_v2 import recover
    from torch.nn import functional as F
    selection = json.loads((c / 'selected_learning_rates.json').read_text())
    checkpoint = c / 'runs' / f'development_direct_seed9001_lr{selection["rates"]["direct"]:g}.pt'
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if not state['result']['complete'] or state['result']['config']['phase'] != 'development':
        raise ValueError('Complete development checkpoint required')
    torch.set_num_threads(1); torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    model = Model().cuda().eval(); model.load_state_dict(state['model']); del state
    params = {n: q for n, q in model.named_parameters() if n.startswith('trunk.') and q.requires_grad}
    X, Y = load_split(c / 'data', 'train')
    sha = lambda f: hashlib.sha256(Path(f).read_bytes()).hexdigest()
    out = {'complete': False, 'checkpoint_sha256': base.digest(checkpoint), 'gpu': torch.cuda.get_device_name(), 'torch': str(torch.__version__),
           'scope': 'development checkpoint, recovery-study minibatches 9101-9105; query rule and gate level vary; no test outcomes',
           'sources': {f.name: sha(f) for f in [Path(__file__), HERE / 'stable_recovery_v2.py', a.study / 'celeba_recovery_study.py',
                                                a.study / 'shared_native_actions.py', a.study / 'retained_sources/stable_recovery.py',
                                                a.study / 'retained_sources/v46_baseline_frontier.py']},
           'cases': 'direct; stream completion and gate-aware queries at eps 0.05 with gate 0.95 (gamma .05) and with non-negative-response gate (gamma 1)',
           'states': [], 'rows': [], 'failures': []}
    base.save(a.output, out)
    cases = [('direct', None, None), ('completion', .05, .05), ('gate', .05, .05), ('completion', .05, 1.0), ('gate', .05, 1.0)]
    for seed in (9101, 9102, 9103, 9104, 9105):
        idx = torch.randint(len(X), (128,), generator=torch.Generator().manual_seed(seed))
        x, y = batch(X, Y, idx)

        def function(parameters):
            return F.binary_cross_entropy_with_logits(torch.func.functional_call(model, parameters, (x,), strict=False), y, reduction='none').mean(0)
        warm = None
        try:
            warm = SharedNativeActions(params, function); warm.action(np.full(40, 1 / 40))
        finally:
            if warm is not None: warm.close()
            del warm; base.clean()
        G, K, wn, reference = base.reference_state(model, x, y, params, solve)
        base.clean()
        out['states'].append({'seed': seed, 'reference': reference, 'gram': K.tolist()})
        shift = seed % len(cases); order = cases[shift:] + cases[:shift]
        warm_done = set()
        for rule, eps, gamma, repeat in [(r, e, g, k) for k in range(a.repeats) for r, e, g in order]:
            row = {'seed': seed, 'rule': rule, 'epsilon': eps, 'gamma': gamma, 'repeat': repeat, 'complete': False}
            oracle = direction = None
            try:
                key = 'direct' if rule == 'direct' else 'stream'
                if key not in warm_done:
                    base.warm_configuration('direct' if rule == 'direct' else 'completion', model, x, y, params, function, solve, SharedCapture)
                    warm_done.add(key)
                base.clean(); torch.cuda.reset_peak_memory_stats(); start = time.perf_counter()
                if rule == 'direct':
                    w, direction, _, fit = base.direct(model, x, y, solve, SharedCapture)
                    result = {'accepted': True, 'actions': 0, 'status': 'full_contraction', 'objective': fit['objective'], 'lower': fit['lower']}
                else:
                    oracle = SharedNativeActions(params, function)

                    def action(v):
                        z = oracle.action(v); base.sync(); return z.double().cpu().numpy()
                    result = recover(action, 40, eps=eps, gamma=gamma, lower_rule='completion', query_rule=rule, max_actions=40)
                    w = result.pop('weights'); result.pop('response', None)
                    if result['accepted']:
                        direction = oracle.direction(w)
                    else:
                        oracle.close(); oracle = None; base.clean()
                        w, direction, _, fit = base.direct(model, x, y, solve, SharedCapture)
                        result['fallback'] = {'kind': 'Direct', 'objective': fit['objective'], 'lower': fit['lower']}
                base.sync()
                row.update(seconds=time.perf_counter() - start, peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                           weights=np.asarray(w).tolist(), recovery=result)
                if oracle is not None: oracle.close(); oracle = None
                audit, _ = base.audit_direction(function, params, direction, w, G, K, reference, eps)
                fid = bool(audit['passed_simplex_gate'] and audit['native_direction_relative_error'] <= 2e-5
                           and audit['native_table_direction_relative_error'] <= 2e-5 and audit['native_objective_factor_upper'] is not None
                           and audit['native_objective_factor_upper'] <= 1 + (eps or .001) + 1e-5)
                row.update(audit, passed_fidelity=fid, passed_strict_descent=bool(audit['executed_min_response'] > 0), complete=True)
            except Exception:
                row['error'] = traceback.format_exc(); out['failures'].append({'seed': seed, 'rule': rule, 'gamma': gamma, 'error': row['error']})
            finally:
                if oracle is not None: oracle.close()
                del oracle, direction; base.clean()
            out['rows'].append(row); base.save(a.output, out)
            print(json.dumps({k: row.get(k) for k in ['seed', 'rule', 'gamma', 'repeat', 'complete', 'seconds', 'passed_fidelity']}
                             | {'actions': (row.get('recovery') or {}).get('actions')}), flush=True)
    out['complete'] = not out['failures']; base.save(a.output, out)
    print('QUERY_RULE_STUDY_DONE', out['complete'], flush=True)


if __name__ == '__main__':
    main()
