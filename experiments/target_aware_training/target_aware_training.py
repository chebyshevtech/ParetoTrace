"""Target-aware arm (rebuttal improvement): stable_recovery_v2, gate-aware queries, strict-descent target.
Bounded shared-batch CelebA sharp-endpoint training study; disabled until frozen.

One process runs one prespecified arm/seed. Existing scientific implementations
are imported unchanged. No service launcher, Drive lifecycle, or model selection.
Numerical stopping and achieved-fidelity audits are not exact-real AD certificates.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import argparse
import fcntl
import gc
import hashlib
import inspect
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
sys.path.insert(0, str(BASE / 'celeba_v2'))
from celeba_training import Model, batch, load_split, evaluate, solve, require_finite
from preflight import SharedCapture
sys.path.insert(0, str(BASE / 'retained_sources'))
import stable_recovery
import stable_recovery_v2
from shared_native_actions import SharedNativeActions
from celeba_recovery_study import reference_state, audit_direction

ARMS = ('stream_target_aware',)
SOURCE_FILES = ['target_aware_training.py', 'shared_native_actions.py', 'celeba_recovery_study.py',
                'celeba_v2/celeba_training.py', 'celeba_v2/preflight.py',
                'celeba_v2/simplex_qp.py', 'retained_sources/stable_recovery.py',
                'retained_sources/v46_baseline_frontier.py', 'retained_sources/stable_recovery_v2.py']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def plain(value):
    """Retain nonfinite failure evidence without emitting invalid JSON."""
    if isinstance(value, (np.ndarray, torch.Tensor)):
        return plain(value.detach().cpu().tolist() if torch.is_tensor(value) else value.tolist())
    if isinstance(value, np.generic):
        return plain(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return {'nonfinite': str(value)}
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(v) for v in value]
    return value


def json_write(path, value, immutable=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(plain(value), indent=2, allow_nan=False) + '\n'
    if immutable and path.exists():
        if path.read_text() != payload:
            raise ValueError('Immutable record differs: ' + str(path))
        return
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w') as out:
        out.write(payload)
        out.flush()
        os.fsync(out.fileno())
    tmp.replace(path)


def sync(model):
    if next(model.parameters()).is_cuda:
        torch.cuda.synchronize()


def memory(model, reset=False):
    if not next(model.parameters()).is_cuda:
        return {'allocated': 0, 'peak': 0}
    if reset:
        torch.cuda.reset_peak_memory_stats()
    return {'allocated': torch.cuda.memory_allocated(), 'peak': torch.cuda.max_memory_allocated()}


def cleanup(model):
    gc.collect()
    if next(model.parameters()).is_cuda:
        torch.cuda.empty_cache()
    sync(model)


def task_losses(model, x, y):
    return F.binary_cross_entropy_with_logits(model(x), y, reduction='none').mean(0)


def scope(model):
    shared = {n: p for n, p in model.named_parameters() if n.startswith('trunk.') and p.requires_grad}
    heads = {n: p for n, p in model.named_parameters() if n.startswith('head.') and p.requires_grad}
    if not shared or not heads or set(shared) | set(heads) != {n for n, p in model.named_parameters() if p.requires_grad}:
        raise ValueError('Unexpected shared/private parameter coverage')
    return shared, heads


def direct_gradients(model, x, y, shared, heads):
    with SharedCapture(model.trunk) as capture:
        losses = task_losses(model, x, y)
    capture.capture(losses)
    K = capture.gram(torch.float64).detach().cpu().numpy()
    del capture
    weights, fit = solve(K)
    gradients = torch.autograd.grad(losses, tuple(shared.values()),
                                   torch.as_tensor(weights, device=losses.device, dtype=losses.dtype), retain_graph=True)
    head_gradients = torch.autograd.grad(losses.sum(), tuple(heads.values()))
    return weights, dict(zip(shared, gradients)), dict(zip(heads, head_gradients)), losses.detach(), fit


def native_gradients(model, x, y, shared, heads):
    losses = task_losses(model, x, y)
    table = losses.new_empty((len(losses), sum(p.numel() for p in shared.values())))
    for i, loss in enumerate(losses):
        gradients = torch.autograd.grad(loss, tuple(shared.values()), retain_graph=True)
        offset = 0
        for parameter, gradient in zip(shared.values(), gradients):
            table[i, offset:offset + parameter.numel()].copy_(gradient.detach().flatten())
            offset += parameter.numel()
        del gradients, gradient
    K = torch.zeros(len(losses), len(losses), device=losses.device, dtype=torch.float64)
    for first in range(0, table.shape[1], 65536):
        tile = table[:, first:first + 65536].double()
        K.addmm_(tile, tile.T)
        del tile
    weights, fit = solve(K.cpu().numpy())
    vector = torch.as_tensor(weights, device=table.device, dtype=table.dtype) @ table
    del table, K
    direction = {}
    offset = 0
    for name, parameter in shared.items():
        direction[name] = vector[offset:offset + parameter.numel()].reshape_as(parameter)
        offset += parameter.numel()
    head_gradients = torch.autograd.grad(losses.sum(), tuple(heads.values()))
    return weights, direction, dict(zip(heads, head_gradients)), losses.detach(), fit


def traced_recovery(oracle, rule, trace):
    """Observation-only wrappers preserve the common acquisition and QP code."""
    original_upper, original_lower = stable_recovery_v2.upper_span, stable_recovery_v2.qp
    trace.update(queries=[], prefixes=[])

    def action(query):
        record = {'index': len(trace['queries']) + 1, 'query': plain(query)}
        trace['queries'].append(record)
        tick = time.perf_counter()
        try:
            response = oracle.action(query)
            if response.is_cuda:
                torch.cuda.synchronize()
            value = response.double().cpu().numpy()
            record['response'] = plain(value)
            return value
        except Exception:
            record['error'] = traceback.format_exc()
            raise
        finally:
            record['seconds'] = time.perf_counter() - tick

    def upper(*args, **kwargs):
        record = {'actions': len(trace['queries'])}
        trace['prefixes'].append(record)
        try:
            result = original_upper(*args, **kwargs)
            weights, response, objective, _ = result
            record['upper'] = plain({'weights': weights, 'response': response, 'objective': objective})
            return result
        except Exception:
            record['upper_error'] = traceback.format_exc()
            raise

    def lower(*args, **kwargs):
        try:
            result = original_lower(*args, **kwargs)
            trace['prefixes'][-1]['completion_qp'] = plain(result)
            return result
        except Exception:
            trace['prefixes'][-1]['completion_qp_error'] = traceback.format_exc()
            raise

    stable_recovery_v2.upper_span, stable_recovery_v2.qp = upper, lower
    try:
        return stable_recovery_v2.recover(action, oracle.M, lower_rule='completion', query_rule='gate', eps=.05,
                                          gamma=1.0, max_actions=oracle.M)
    finally:
        stable_recovery_v2.upper_span, stable_recovery_v2.qp = original_upper, original_lower


def gradients(model, x, y, arm, trace):
    shared, heads = scope(model)
    trace.update(arm=arm, fallback=False, adaptive_actions=0)
    if arm in ('direct', 'native_chunk1'):
        result = (direct_gradients if arm == 'direct' else native_gradients)(model, x, y, shared, heads)
        trace['full_recovery'] = plain(result[-1])
        return result[:4]
    if arm not in ARMS:
        raise ValueError('Unknown arm')
    oracle = None
    function = lambda parameters: F.binary_cross_entropy_with_logits(
        torch.func.functional_call(model, parameters, (x,), strict=False), y, reduction='none').mean(0)
    try:
        oracle = SharedNativeActions(shared, function)
        result = traced_recovery(oracle, 'completion', trace)
        weights = result.pop('weights')
        trace['recovery'] = plain(result)
        if not result['accepted']:
            raise ArithmeticError('Recovery did not accept: ' + result['status'])
        direction = oracle.direction(weights)
        hg = torch.autograd.grad(oracle.losses.sum(), tuple(heads.values()))
        losses = oracle.losses.detach()
        require_finite([losses] + list(direction.values()) + list(hg), 'Stream reconstructed losses/gradients')
        return weights, direction, dict(zip(heads, hg)), losses
    except Exception:
        trace.update(fallback=True, adaptive_error=traceback.format_exc())
    finally:
        trace['adaptive_actions'] = len(trace.get('queries', []))
        if oracle is not None:
            oracle.close()
        del oracle
    # All failed actions, solver work, cleanup and full fallback remain timed.
    direction = hg = losses = None
    cleanup(model)
    result = direct_gradients(model, x, y, shared, heads)
    trace['fallback_full_recovery'] = plain(result[-1])
    return result[:4]


def native_audit(model, x, y, weights, arm, fallback):
    shared, heads = scope(model)
    direction = {name: p.grad.detach() for name, p in shared.items()}
    table, K, _, reference = reference_state(model, x, y, shared, solve)
    function = lambda parameters: F.binary_cross_entropy_with_logits(
        torch.func.functional_call(model, parameters, (x,), strict=False), y, reduction='none').mean(0)
    tolerance = .05 if arm.startswith('stream_') and not fallback else .001
    record, _ = audit_direction(function, shared, direction, weights, table, K, reference, tolerance)
    losses = task_losses(model, x, y)
    expected = torch.autograd.grad(losses.sum(), tuple(heads.values()))
    error2 = sum(float((p.grad.double() - g.double()).square().sum()) for p, g in zip(heads.values(), expected))
    norm2 = sum(float(g.double().square().sum()) for g in expected)
    head_error = np.sqrt(error2) / max(np.sqrt(norm2), 1e-30)
    emitted_factor = record['executed_norm_squared'] / reference['lower'] if record['relative_reference_resolved'] else None
    record.update(head_relative_error=float(head_error), native_emitted_objective_factor_upper=emitted_factor,
                  reference=plain(reference), reference_gram=K.tolist(),
                  scope='Actual pre-SGD shared gradient and unweighted private-head gradients; numerical native checks, no real-AD or momentum-descent certificate')
    # Unresolved reference energy cannot support the requested relative audit.
    # Target-aware arm: strict descent replaces the 0.95 normalized-gain target; fidelity gates unchanged.
    record['passed_target_audit'] = bool(record['passed_simplex_gate'] and record['native_direction_relative_error'] <= 2e-5
        and record['native_table_direction_relative_error'] <= 2e-5 and record['native_objective_factor_upper'] is not None
        and record['native_objective_factor_upper'] <= 1 + tolerance + 1e-5 and record['executed_min_response'] > 0)
    record['passed'] = bool(record['passed_target_audit'] and np.isfinite(head_error) and head_error <= 2e-5
                            and emitted_factor is not None and np.isfinite(emitted_factor)
                            and emitted_factor <= 1 + tolerance + 1e-5)
    return record


def update(model, optimizer, batch_factory, arm, audit, row):
    sync(model)
    base = memory(model, reset=True)['allocated']
    start = time.perf_counter()
    row.update(applied=False, base_allocated_bytes=base)
    x, y = batch_factory()  # Transfer/preprocessing is part of the update.
    optimizer.zero_grad(set_to_none=True)
    weights, direction, heads, losses = gradients(model, x, y, arm, row)
    require_finite([losses] + list(direction.values()) + list(heads.values()), 'emitted losses/gradients')
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            parameter.grad = (direction[name] if name in direction else heads[name]).detach()
    row.update(weights=plain(weights), executed_weights=plain(torch.as_tensor(weights, dtype=losses.dtype)),
               task_losses_before=plain(losses))
    del direction, heads, losses
    sync(model)
    before_seconds = time.perf_counter() - start
    before_peak = memory(model)['peak']
    row.update(recovery_and_head_seconds=before_seconds, recovery_and_head_peak_bytes=before_peak)
    if audit:
        memory(model, reset=True)
        tick = time.perf_counter()
        row['native_audit'] = native_audit(model, x, y, weights, arm, row['fallback'])
        sync(model)
        row['native_audit'].update(seconds=time.perf_counter() - tick, peak_allocated_bytes=memory(model)['peak'])
        cleanup(model)  # Audit tables/temporaries leave scope before the next reset.
        if not row['native_audit']['passed']:
            raise ArithmeticError('Independent emitted-gradient/head audit failed; run quarantined')
    memory(model, reset=True)
    tick = time.perf_counter()
    optimizer.step()
    row['applied'] = True
    require_finite(list(model.parameters()) + [v for state in optimizer.state.values() for v in state.values()
                                              if torch.is_tensor(v)], 'updated parameters/optimizer')
    sync(model)
    optimizer_seconds = time.perf_counter() - tick
    peak = max(before_peak, memory(model)['peak'])
    row.update(optimizer_seconds=optimizer_seconds, update_seconds=before_seconds + optimizer_seconds,
               peak_allocated_bytes=peak, incremental_peak_bytes=max(0, peak - base), complete=True,
               timing_scope='Transfer/preprocessing, recovery, head gradients, finite checks and actual SGD; two measured segments exclude the separate pre-SGD native audit')


def source_identity(protocol):
    current = {name: sha(BASE / name) for name in SOURCE_FILES}
    if current != protocol['source_sha256']:
        raise ValueError('Source differs from frozen protocol')
    expected = {Model: BASE / 'celeba_v2/celeba_training.py', SharedCapture: BASE / 'celeba_v2/preflight.py',
                stable_recovery.recover: BASE / 'retained_sources/stable_recovery.py',
                stable_recovery.qp: BASE / 'retained_sources/v46_baseline_frontier.py',
                stable_recovery_v2.recover: BASE / 'retained_sources/stable_recovery_v2.py',
                solve.__globals__['qp']: BASE / 'celeba_v2/simplex_qp.py',
                SharedNativeActions: BASE / 'shared_native_actions.py',
                reference_state: BASE / 'celeba_recovery_study.py'}
    if any(Path(inspect.getfile(obj)).resolve() != path.resolve() for obj, path in expected.items()):
        raise ValueError('Unexpected imported scientific implementation')
    return current


def validate_events(root, state):
    for step, record in enumerate(state['events']):
        relative = Path(record['file'])
        path = root / relative
        if relative.is_absolute() or '..' in relative.parts or relative.parent != Path('events') or sha(path) != record['sha256']:
            raise ValueError('Committed update identity differs')
        row = json.loads(path.read_text())
        if (row['step'] != step or not row['complete'] or not row['applied']
                or row['arm'] != state['config']['arm'] or row['seed'] != state['config']['seed']):
            raise ValueError('Committed update coverage differs')
    if state['next_step'] != len(state['events']):
        raise ValueError('Checkpoint/event cursor mismatch')


def model_hash(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def final_metrics(root, state):
    result = dict(state)
    result['attempts'] = [json.loads(p.read_text()) for p in sorted((root / 'attempts').glob('*.json'))]
    result['recorded_process_seconds'] = sum(a.get('recorded_elapsed_seconds', 0.) for a in result['attempts'])
    result['unrecorded_interrupted_tails'] = sum(not a['complete'] for a in result['attempts'])
    result['latest_checkpoint_sha256'] = sha(root / 'latest.pt')
    committed = {ref['file'] for ref in state['events']}
    extra = [p for p in sorted((root / 'events').glob('*.json')) if str(p.relative_to(root)) not in committed]
    result['uncommitted_attempt_events'] = [{'file': str(p.relative_to(root)), 'sha256': sha(p)} for p in extra]
    result['uncommitted_recorded_update_seconds'] = sum(json.loads(p.read_text()).get('update_seconds', 0.) for p in extra)
    json_write(root / 'metrics.json', result)


def main():
    begin = time.perf_counter()
    p = argparse.ArgumentParser()
    p.add_argument('--protocol', type=Path, default=BASE / 'sharp_training_protocol.json')
    p.add_argument('--phase', choices=['pilot', 'reporting'], required=True)
    p.add_argument('--arm', choices=ARMS, required=True)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--selection', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    protocol = json.loads(args.protocol.read_text())
    if args.phase not in protocol['enabled_phases'] or (args.phase == 'reporting' and not protocol['reporting_budget_frozen']):
        raise ValueError('Training disabled: protocol must be audited and explicitly frozen/enabled first')
    constants = {'arms': list(ARMS), 'lr': .02, 'batch_size': 128, 'momentum': .9,
                 'weight_decay': 0., 'epsilon': .05, 'response_margin_fraction': 0.}
    if any(protocol[k] != value for k, value in constants.items()):
        raise ValueError('Scientific constants differ from implemented protocol')
    if (protocol['pilot']['seeds'] != [61900] or protocol['reporting']['seeds'] != [62001, 62002, 62003]
            or args.seed not in protocol[args.phase]['seeds']):
        raise ValueError('Unplanned arm/seed')
    steps = protocol[args.phase]['updates']
    if steps != (100 if args.phase == 'pilot' else 300) or protocol['lr'] != .02 or protocol['batch_size'] != 128:
        raise ValueError('Unexpected study budget or fixed hyperparameters')
    sources = source_identity(protocol)
    if sha(args.selection) != protocol['prior_selection_sha256']:
        raise ValueError('Prior development-selection identity differs')
    selection = json.loads(args.selection.read_text())
    if selection['rates']['direct'] != .02 or sha(args.data / 'celeba_manifest.json') != protocol['data_manifest_sha256']:
        raise ValueError('Prior LR/data contract differs')
    identity = {'protocol_sha256': sha(args.protocol), 'sources': sources,
                'data_manifest_sha256': protocol['data_manifest_sha256'], 'selection_sha256': sha(args.selection)}
    config = {'phase': args.phase, 'arm': args.arm, 'seed': args.seed, 'steps': steps, 'lr': .02, 'batch_size': 128}
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
    state = saved['state'] if saved else {'next_step': 0, 'events': [], 'validation': [], 'complete': False}
    validate_events(root, state)
    if state['complete']:
        final_metrics(root, state)
        print('ALREADY_COMPLETE', flush=True)
        return
    attempts = root / 'attempts'
    attempts.mkdir(exist_ok=True)
    number = len(list(attempts.glob('*.json'))) + 1
    attempt_path = attempts / f'{number:04d}.json'
    attempt = {'attempt': number, 'complete': False, 'started_utc_seconds': time.time(),
               'start_step': state['next_step'], 'checkpoint_seconds': 0., 'event_write_seconds': 0.,
               'wall_scope': 'main entry through latest recorded event; excludes imports/final manifest write; interrupted tails are unrecorded'}
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
        optimizer = torch.optim.SGD(model.parameters(), lr=.02, momentum=.9)
        if not saved:
            tick = time.perf_counter()
            state['initial_model_sha256'] = model_hash(model)
            attempt['initial_model_hash_seconds'] = time.perf_counter() - tick
        X, Y = load_split(args.data, 'train')
        VX, VY = load_split(args.data, 'valid')
        if saved:
            model.load_state_dict(saved['model'])
            optimizer.load_state_dict(saved['optimizer'])
            torch.set_rng_state(saved['rng'])
            torch.cuda.set_rng_state_all(saved['cuda_rng'])
            np.random.set_state(saved['numpy_rng'])
            random.setstate(saved['python_rng'])
        del saved
        state.update(config=config, identity=identity, torch=str(torch.__version__),
                     cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(),
                     scope='Complete bounded early-training numerical fidelity/memory comparison; no convergence, superiority, exact-real AD or momentum-descent claim')

        def checkpoint():
            tick = time.perf_counter()
            tmp = latest.with_suffix('.pt.tmp')
            torch.save({'identity': identity, 'config': config, 'state': state,
                        'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                        'rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all(),
                        'numpy_rng': np.random.get_state(), 'python_rng': random.getstate()}, tmp)
            with tmp.open('rb') as handle:
                os.fsync(handle.fileno())
            tmp.replace(latest)
            attempt['checkpoint_seconds'] += time.perf_counter() - tick
            attempt.update(last_checkpoint_step=state['next_step'], recorded_elapsed_seconds=time.perf_counter() - begin)
            json_write(attempt_path, attempt)

        for step in range(state['next_step'], steps):
            indices = torch.randint(len(X), (128,), generator=torch.Generator().manual_seed(args.seed * 100000 + step))
            row = {'step': step, 'seed': args.seed, 'arm': args.arm,
                   'indices': indices.tolist(), 'indices_sha256': hashlib.sha256(indices.numpy().tobytes()).hexdigest(),
                   'complete': False}
            cleanup(model)
            update(model, optimizer, lambda: batch(X, Y, indices), args.arm,
                   step in {0, steps // 2, steps - 1}, row)
            path = root / 'events' / f'attempt{number:04d}_step{step:06d}.json'
            tick = time.perf_counter()
            json_write(path, row, immutable=True)
            attempt['event_write_seconds'] += time.perf_counter() - tick
            state['events'].append({'file': str(path.relative_to(root)), 'sha256': sha(path)})
            state['next_step'] = step + 1
            if (step + 1) % 25 == 0 or step + 1 == steps:
                checkpoint()
                print(json.dumps({'arm': args.arm, 'seed': args.seed, 'step': step + 1}), flush=True)
        cleanup(model)
        tick = time.perf_counter()
        state['final_validation'] = evaluate(model, VX, VY)
        state['heldout_test_evaluated'] = args.phase == 'reporting'
        if args.phase == 'reporting':
            TX, TY = load_split(args.data, 'test')
            state['final_test'] = evaluate(model, TX, TY)
        sync(model)
        state['evaluation_seconds_this_attempt'] = time.perf_counter() - tick
        for key in ('final_validation', 'final_test'):
            if key in state and not all(np.isfinite(np.asarray(value, dtype=float)).all() for value in state[key].values()):
                raise ArithmeticError('Nonfinite final evaluation')
        records = [json.loads((root / ref['file']).read_text()) for ref in state['events']]
        state.update(update_seconds=sum(r['update_seconds'] for r in records),
                     peak_update_allocated_bytes=max(r['peak_allocated_bytes'] for r in records),
                     peak_update_incremental_bytes=max(r['incremental_peak_bytes'] for r in records),
                     adaptive_actions=sum(r['adaptive_actions'] for r in records),
                     fallback_count=sum(r['fallback'] for r in records),
                     audited_steps=[r['step'] for r in records if 'native_audit' in r], complete=True)
        checkpoint()
        attempt.update(complete=True, final_step=steps, recorded_elapsed_seconds=time.perf_counter() - begin)
        json_write(attempt_path, attempt)
        final_metrics(root, state)
        print('SHARP_TRAINING_COMPLETE', flush=True)
    except BaseException:
        failure = {'config': config, 'identity': identity, 'attempt': number, 'complete': False,
                   'error': traceback.format_exc(), 'current_update': row, 'recorded_elapsed_seconds': time.perf_counter() - begin}
        if 'model' in locals():
            failure['peak_allocated_bytes_at_failure'] = memory(model)['peak']
        json_write(root / 'failure.json', failure)
        attempt.update(error=failure['error'], recorded_elapsed_seconds=failure['recorded_elapsed_seconds'])
        json_write(attempt_path, attempt)
        raise


if __name__ == '__main__':
    main()
