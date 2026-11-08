"""Certified-step CelebA-40 study (rebuttal, 8 November 2026).

Question. When no update may raise any task's minibatch loss (the guarantee of Proposition 1), which
direction still makes progress, and how often does an unprotected step raise a task's loss?

Same frozen model, data, selected learning rate (0.02), initialization and data order as the sharp-training
and target-aware studies (seeds 62001-62003, 300 updates). Every arm takes plain SGD steps without momentum,
which the guarantee does not cover. The trunk moves along -G w for the arm's weighting w, and each private
head along its own unweighted gradient, as in every arm of the earlier CelebA studies.

Arms (weighting x step rule):
  uniform_fixed, direct_fixed
      step 0.02 at every update. The realized change of every task loss on the minibatch is recorded.
  uniform_certified, direct_certified, stream_certified, mgda_ub_certified
      certified step. The exact first-order responses s_i = (K w)_i + |grad of head i|^2 must all be positive.
      They are read from the captured Gram (Direct), from the transcript (target-aware Stream), or from one
      native Gram action (uniform, MGDA-UB). Then the largest step 0.02 * 2^-k, k < 20, that passes
      Equation (27) of the paper, L_i(after) <= L_i(before) - c * eta * s_i with c = 1/2, for every task is
      taken. Otherwise the parameters are held.

Rounded same-minibatch checks, not an interval enclosure or a population guarantee.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import argparse
import fcntl
import json
import random
import sys
import time
import traceback
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))
import target_aware_training as ta
from target_aware_training import (Model, batch, load_split, evaluate, solve, require_finite, SharedNativeActions,
                                   reference_state, plain, json_write, sync, memory, cleanup, task_losses, scope,
                                   model_hash, sha, validate_events)

ARMS = ('uniform_fixed', 'direct_fixed', 'uniform_certified', 'direct_certified', 'stream_certified', 'mgda_ub_certified')
LR, C, HALVINGS = .02, .5, 20
SOURCE_FILES = ['certified_step_training.py', 'target_aware_training.py', 'shared_native_actions.py',
                'celeba_recovery_study.py', 'celeba_v2/celeba_training.py', 'celeba_v2/preflight.py',
                'celeba_v2/simplex_qp.py', 'retained_sources/stable_recovery.py',
                'retained_sources/v46_baseline_frontier.py', 'retained_sources/stable_recovery_v2.py']


def native_function(model, x, y):
    return lambda parameters: F.binary_cross_entropy_with_logits(
        torch.func.functional_call(model, parameters, (x,), strict=False), y, reduction='none').mean(0)


def direct(model, x, y, shared, heads):
    """ParetoTrace-Direct as in the earlier studies, also returning the exact trunk responses K w."""
    with ta.SharedCapture(model.trunk) as capture:
        losses = task_losses(model, x, y)
    capture.capture(losses)
    K = capture.gram(torch.float64).detach().cpu().numpy()
    del capture
    w, fit = solve(K)
    g = torch.autograd.grad(losses, tuple(shared.values()),
                            torch.as_tensor(w, device=losses.device, dtype=losses.dtype), retain_graph=True)
    hg = torch.autograd.grad(losses.sum(), tuple(heads.values()))
    K = (K + K.T) / 2
    extra = {'fit': fit, 'uniform_trunk_response': K @ np.full(len(w), 1 / len(w))}
    return w, dict(zip(shared, g)), dict(zip(heads, hg)), losses.detach(), K @ np.asarray(w, float), extra


def mgda_ub_weights(model, x, y):
    """The MGDA-UB rule of celeba_training.py: the Gram of gradients at the shared representation."""
    with torch.no_grad():
        logits = model.head(model.trunk(x))
        delta = (torch.sigmoid(logits) - y) / x.shape[0]
        W = model.head.weight.double()
        K = (delta.double().T @ delta.double()) * (W @ W.T)
    return solve(K.cpu().numpy())


def weigh(model, x, y, arm, row):
    """Weights, trunk direction G w, unweighted head gradients, losses and exact trunk responses K w."""
    shared, heads = scope(model)
    kind = arm.rsplit('_', 1)[0]
    row.update(fallback=False, adaptive_actions=0)
    if kind == 'direct':
        w, d, hg, losses, response, extra = direct(model, x, y, shared, heads)
        row.update(full_recovery=plain(extra['fit']), uniform_trunk_response=plain(extra['uniform_trunk_response']),
                   response_source='captured Gram')
        return w, d, hg, losses, response
    if kind == 'stream':
        oracle = None
        try:
            oracle = SharedNativeActions(shared, native_function(model, x, y))
            result = ta.traced_recovery(oracle, 'completion', row)
            w = result.pop('weights')
            response = np.asarray(result.pop('response'), float)
            row['recovery'] = plain(result)
            if not result['accepted']:
                raise ArithmeticError('Recovery did not accept: ' + result['status'])
            first = row['queries'][0]
            scale = (1 / oracle.M) / float(np.asarray(first['query'], float)[0])
            row['uniform_trunk_response'] = plain(np.asarray(first['response'], float) * scale)
            d = oracle.direction(w)
            hg = torch.autograd.grad(oracle.losses.sum(), tuple(heads.values()))
            losses = oracle.losses.detach()
            row['response_source'] = 'Stream transcript (in-span exact responses)'
            return w, d, dict(zip(heads, hg)), losses, response
        except Exception:
            row.update(fallback=True, adaptive_error=traceback.format_exc())
        finally:
            row['adaptive_actions'] = len(row.get('queries', []))
            if oracle is not None:
                oracle.close()
            del oracle
        cleanup(model)
        w, d, hg, losses, response, extra = direct(model, x, y, shared, heads)
        row.update(fallback_full_recovery=plain(extra['fit']), response_source='captured Gram (fallback)')
        return w, d, hg, losses, response
    if kind not in ('uniform', 'mgda_ub'):
        raise ValueError('Unknown arm')
    oracle = SharedNativeActions(shared, native_function(model, x, y))
    try:
        if kind == 'uniform':
            w = np.full(oracle.M, 1 / oracle.M)
        else:
            w, fit = mgda_ub_weights(model, x, y)
            row['mgda_ub_fit'] = plain(fit)
        response = oracle.action(w).double().cpu().numpy()
        row['adaptive_actions'] = 1
        d = oracle.direction(w)
        hg = torch.autograd.grad(oracle.losses.sum(), tuple(heads.values()))
        losses = oracle.losses.detach()
        if kind == 'uniform':
            row['uniform_trunk_response'] = plain(response)
        row['response_source'] = 'one native Gram action'
        return w, d, dict(zip(heads, hg)), losses, response
    finally:
        oracle.close()
        del oracle


def head_terms(hg):
    W, b = hg['head.weight'].double(), hg['head.bias'].double()
    return (W.square().sum(1) + b.square()).cpu().numpy()


def native_audit(model, x, y, w, response, heads_first_order, shared, heads):
    """Independent per-task autodiff check of the responses that decide eligibility."""
    _, K, _, reference = reference_state(model, x, y, shared, solve)
    K = (np.asarray(K, float) + np.asarray(K, float).T) / 2
    native_response = K @ np.asarray(w, float)
    error = float(np.linalg.norm(native_response - response) / max(np.linalg.norm(native_response), 1e-30))
    losses = task_losses(model, x, y)
    native_heads = head_terms(dict(zip(heads, torch.autograd.grad(losses.sum(), tuple(heads.values())))))
    head_error = float(np.abs(native_heads - heads_first_order).max() / max(np.abs(native_heads).max(), 1e-30))
    s_native, s = native_response + native_heads, response + heads_first_order
    return {'native_trunk_response_relative_error': error, 'native_head_term_relative_error': head_error,
            'native_min_first_order_response': float(s_native.min()), 'native_min_trunk_response': float(native_response.min()),
            'eligibility_agrees': bool((s_native.min() > 0) == (s.min() > 0)),
            'uniform_native_trunk_response': (K @ np.full(len(w), 1 / len(w))).tolist(),
            'native_reference': plain(reference), 'native_gram': K.tolist(),
            'passed': bool(error <= 1e-4 and head_error <= 1e-4 and (s_native.min() > 0) == (s.min() > 0))}


def update(model, batch_factory, arm, audit, row):
    shared, heads = scope(model)
    params = dict(shared)
    params.update(heads)
    sync(model)
    memory(model, reset=True)
    start = time.perf_counter()
    x, y = batch_factory()
    w, d, hg, losses, response = weigh(model, x, y, arm, row)
    grads = dict(d)
    grads.update(hg)
    require_finite([losses] + list(grads.values()), 'losses/direction')
    h = head_terms(hg)
    response = np.asarray(response, float)
    s = response + h
    sync(model)
    direction_seconds = time.perf_counter() - start
    direction_peak = memory(model)['peak']
    row.update(weights=plain(w), trunk_response=plain(response), head_terms=plain(h), first_order_response=plain(s),
               min_first_order_response=float(s.min()), min_trunk_response=float(response.min()),
               nonpositive_first_order_tasks=int((s <= 0).sum()), nonpositive_trunk_tasks=int((response <= 0).sum()),
               direction_seconds=direction_seconds, direction_peak_bytes=direction_peak)
    if audit:
        memory(model, reset=True)
        tick = time.perf_counter()
        row['native_audit'] = native_audit(model, x, y, w, response, h, shared, heads)
        sync(model)
        row['native_audit'].update(seconds=time.perf_counter() - tick, peak_allocated_bytes=memory(model)['peak'])
        cleanup(model)
    memory(model, reset=True)
    tick = time.perf_counter()
    with torch.no_grad():
        saved = {n: p.detach().clone() for n, p in params.items()}
        L0 = task_losses(model, x, y).double().cpu().numpy()

    def put(eta):
        with torch.no_grad():
            for n, p in params.items():
                p.copy_(saved[n] - eta * grads[n])
            return task_losses(model, x, y).double().cpu().numpy()

    trials, eta_accepted, after, eligible = [], None, L0, None
    if arm.endswith('_fixed'):
        after, eta_accepted, status = put(LR), LR, 'fixed'
        trials.append({'eta': LR, 'max_task_change': float((after - L0).max())})
    else:
        eligible = bool(np.isfinite(s).all() and s.min() > 0)
        status = 'no_admissible_step' if eligible else 'not_certified'
        if eligible:
            for k in range(HALVINGS):
                eta = LR * .5 ** k
                trial = put(eta)
                bound = L0 - C * eta * s
                ok = bool(np.isfinite(trial).all() and (trial <= bound).all())
                trials.append({'eta': eta, 'passed': ok, 'max_task_change': float((trial - L0).max()),
                               'tasks_failing': int((trial > bound).sum())})
                if ok:
                    eta_accepted, after, status = eta, trial, 'accepted'
                    break
        if eta_accepted is None:
            with torch.no_grad():
                for n, p in params.items():
                    p.copy_(saved[n])
            after = L0
    require_finite(list(model.parameters()), 'updated parameters')
    sync(model)
    step_seconds = time.perf_counter() - tick
    change = after - L0
    row.update(step_rule='fixed' if arm.endswith('_fixed') else 'certified', status=status, eligible=eligible,
               accepted_eta=eta_accepted, trials=trials, task_losses_before=plain(L0), task_losses_after=plain(after),
               realized_change=plain(change), raised_tasks=int((change > 0).sum()), max_task_change=float(change.max()),
               worst_task_before=float(L0.max()), worst_task_after=float(after.max()),
               loss_recompute_max_abs=float(np.abs(L0 - losses.double().cpu().numpy()).max()),
               step_seconds=step_seconds, update_seconds=direction_seconds + step_seconds,
               peak_allocated_bytes=max(direction_peak, memory(model)['peak']), applied=True, complete=True,
               timing_scope='direction (transfer, weighting, actions or capture, reconstruction, head gradients) plus the step rule; the separate native audit is untimed')
    del saved


def source_identity(protocol):
    current = {name: sha(BASE / name) for name in SOURCE_FILES}
    if current != protocol['source_sha256']:
        raise ValueError('Source differs from frozen protocol')
    return current


def final_metrics(root, state):
    result = dict(state)
    result['attempts'] = [json.loads(p.read_text()) for p in sorted((root / 'attempts').glob('*.json'))]
    json_write(root / 'metrics.json', result)


def main():
    begin = time.perf_counter()
    p = argparse.ArgumentParser()
    p.add_argument('--protocol', type=Path, required=True)
    p.add_argument('--phase', choices=['pilot', 'reporting'], required=True)
    p.add_argument('--arm', choices=ARMS, required=True)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--selection', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    protocol = json.loads(args.protocol.read_text())
    constants = {'arms': list(ARMS), 'lr': LR, 'batch_size': 128, 'momentum': 0., 'armijo_c': C, 'halvings': HALVINGS}
    if any(protocol[k] != value for k, value in constants.items()):
        raise ValueError('Scientific constants differ from implemented protocol')
    if args.seed not in protocol[args.phase]['seeds']:
        raise ValueError('Unplanned seed')
    steps = protocol[args.phase]['updates']
    sources = source_identity(protocol)
    if sha(args.selection) != protocol['prior_selection_sha256']:
        raise ValueError('Prior development-selection identity differs')
    selection = json.loads(args.selection.read_text())
    if selection['rates']['direct'] != LR or sha(args.data / 'celeba_manifest.json') != protocol['data_manifest_sha256']:
        raise ValueError('Prior LR/data contract differs')
    identity = {'protocol_sha256': sha(args.protocol), 'sources': sources,
                'data_manifest_sha256': protocol['data_manifest_sha256'], 'selection_sha256': sha(args.selection)}
    config = {'phase': args.phase, 'arm': args.arm, 'seed': args.seed, 'steps': steps, 'lr': LR, 'batch_size': 128}
    root = args.out.resolve()
    root.mkdir(parents=True, exist_ok=True)
    lock = (root / 'run.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    json_write(root / 'run_identity.json', {'identity': identity, 'config': config}, immutable=True)
    if (root / 'failure.json').exists():
        raise RuntimeError('Failed run is quarantined; retain it and use a prospective revision')
    latest = root / 'latest.pt'
    saved = torch.load(latest, map_location='cpu', weights_only=False) if latest.exists() else None
    if saved and (saved['identity'] != identity or saved['config'] != config):
        raise ValueError('Checkpoint configuration/source identity differs')
    state = saved['state'] if saved else {'next_step': 0, 'events': [], 'complete': False}
    validate_events(root, state)
    if state['complete']:
        final_metrics(root, state)
        print('ALREADY_COMPLETE', flush=True)
        return
    attempts = root / 'attempts'
    attempts.mkdir(exist_ok=True)
    number = len(list(attempts.glob('*.json'))) + 1
    attempt_path = attempts / f'{number:04d}.json'
    attempt = {'attempt': number, 'complete': False, 'started_utc_seconds': time.time(), 'start_step': state['next_step']}
    json_write(attempt_path, attempt)
    row = None
    try:
        torch.set_num_threads(1)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        model = Model().cuda().eval()
        X, Y = load_split(args.data, 'train')
        VX, VY = load_split(args.data, 'valid')
        if saved:
            model.load_state_dict(saved['model'])
            torch.set_rng_state(saved['rng'])
            torch.cuda.set_rng_state_all(saved['cuda_rng'])
            np.random.set_state(saved['numpy_rng'])
            random.setstate(saved['python_rng'])
        else:
            state['initial_model_sha256'] = model_hash(model)
            state['initial_train'] = evaluate(model, X, Y)
            state['initial_validation'] = evaluate(model, VX, VY)
        del saved
        state.update(config=config, identity=identity, torch=str(torch.__version__), cuda=torch.version.cuda,
                     gpu=torch.cuda.get_device_name(),
                     scope='Plain-SGD CelebA-40 study of the certified step (Proposition 1, Equation 27). Same-minibatch rounded checks; no momentum, population or exact-real claim.')

        def checkpoint():
            tmp = latest.with_suffix('.pt.tmp')
            torch.save({'identity': identity, 'config': config, 'state': state, 'model': model.state_dict(),
                        'rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all(),
                        'numpy_rng': np.random.get_state(), 'python_rng': random.getstate()}, tmp)
            with tmp.open('rb') as handle:
                os.fsync(handle.fileno())
            tmp.replace(latest)
            attempt.update(last_checkpoint_step=state['next_step'], recorded_elapsed_seconds=time.perf_counter() - begin)
            json_write(attempt_path, attempt)

        for step in range(state['next_step'], steps):
            indices = torch.randint(len(X), (128,), generator=torch.Generator().manual_seed(args.seed * 100000 + step))
            row = {'step': step, 'seed': args.seed, 'arm': args.arm, 'indices': indices.tolist(), 'complete': False}
            cleanup(model)
            update(model, lambda: batch(X, Y, indices), args.arm, step in {0, steps // 2, steps - 1}, row)
            path = root / 'events' / f'attempt{number:04d}_step{step:06d}.json'
            json_write(path, row, immutable=True)
            state['events'].append({'file': str(path.relative_to(root)), 'sha256': sha(path)})
            state['next_step'] = step + 1
            if (step + 1) % 25 == 0 or step + 1 == steps:
                checkpoint()
                print(json.dumps({'arm': args.arm, 'seed': args.seed, 'step': step + 1}), flush=True)
        cleanup(model)
        state['final_train'] = evaluate(model, X, Y)
        state['final_validation'] = evaluate(model, VX, VY)
        state['heldout_test_evaluated'] = args.phase == 'reporting'
        if args.phase == 'reporting':
            TX, TY = load_split(args.data, 'test')
            state['final_test'] = evaluate(model, TX, TY)
        records = [json.loads((root / ref['file']).read_text()) for ref in state['events']]
        state.update(update_seconds=sum(r['update_seconds'] for r in records),
                     peak_update_allocated_bytes=max(r['peak_allocated_bytes'] for r in records),
                     adaptive_actions=sum(r['adaptive_actions'] for r in records),
                     fallback_count=sum(r['fallback'] for r in records),
                     status_counts={k: sum(r['status'] == k for r in records) for k in ('fixed', 'accepted', 'no_admissible_step', 'not_certified')},
                     updates_raising_a_task=sum(r['raised_tasks'] > 0 for r in records),
                     audits_passed=sum(r['native_audit']['passed'] for r in records if 'native_audit' in r),
                     audits=sum('native_audit' in r for r in records), complete=True)
        checkpoint()
        attempt.update(complete=True, final_step=steps, recorded_elapsed_seconds=time.perf_counter() - begin)
        json_write(attempt_path, attempt)
        final_metrics(root, state)
        print('CERTIFIED_STEP_COMPLETE', flush=True)
    except BaseException:
        failure = {'config': config, 'identity': identity, 'attempt': number, 'complete': False,
                   'error': traceback.format_exc(), 'current_update': row,
                   'recorded_elapsed_seconds': time.perf_counter() - begin}
        json_write(root / 'failure.json', failure)
        attempt.update(error=failure['error'])
        json_write(attempt_path, attempt)
        raise


if __name__ == '__main__':
    main()
