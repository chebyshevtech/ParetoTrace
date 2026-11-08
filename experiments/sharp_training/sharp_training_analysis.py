"""Strict completed sharp-training evidence validator; CPU only, no model loading.

Pilot validation deliberately does not access any validation-quality values.
A complete reporting analysis requires every frozen seed/arm and all native gates.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import torch

ARMS = ['direct', 'native_chunk1', 'stream_completion', 'stream_scalar']
SEEDS = [62001, 62002, 62003]
GATES = {'shared_native_and_table_direction_relative_error': 2e-5,
         'head_gradient_relative_error': 2e-5, 'full_recovery_objective_factor': 1.001,
         'stream_objective_factor': 1.05, 'objective_factor_roundoff_tolerance': 1e-5,
         'recorded_normalized_gain_min': .95, 'gain_tolerance': 1e-5}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    def pairs(items):
        out = {}
        for key, value in items:
            require(key not in out, 'Duplicate JSON key')
            out[key] = value
        return out
    return json.loads(Path(path).read_text(), object_pairs_hook=pairs,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError('Nonfinite JSON')))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 2**20), b''):
            digest.update(block)
    return digest.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def file(root, relative):
    root, rel = Path(root), Path(relative)
    require(not rel.is_absolute() and '..' not in rel.parts, 'Unsafe relative path')
    result = root / rel
    require(not any(p.is_symlink() for p in [result, *result.parents] if p == root or root in p.parents), 'Symlink in evidence path')
    return result


def number(value, minimum=0):
    require(type(value) in (int, float) and math.isfinite(value) and value >= minimum, 'Invalid finite scalar')
    return value


def close(left, right, tolerance=1e-8):
    require(math.isclose(float(left), float(right), rel_tol=tolerance, abs_tol=tolerance), 'Recomputed scalar differs')


def vector(value, simplex=False):
    array = np.asarray(value, dtype=float)
    require(array.shape == (40,) and np.isfinite(array).all(), 'Invalid forty-task vector')
    if simplex:
        require(array.min() >= -1e-8 and abs(array.sum() - 1) <= 1e-7, 'Invalid simplex weights')
    return array


def nonfinite_evidence(value):
    if isinstance(value, dict):
        return value.get('nonfinite') in ('nan', 'inf', '-inf') if set(value) == {'nonfinite'} else any(nonfinite_evidence(v) for v in value.values())
    return isinstance(value, list) and any(nonfinite_evidence(v) for v in value)


def validate_protocol(protocol, source_root):
    expected = {'arms': ARMS, 'lr': .02, 'momentum': .9, 'weight_decay': 0., 'batch_size': 128,
                'epsilon': .05, 'response_margin_fraction': .95, 'audit_gates': GATES}
    require(all(protocol[k] == value for k, value in expected.items()), 'Frozen scientific constants differ')
    require(protocol['pilot']['seeds'] == [61900] and protocol['pilot']['updates'] == 100, 'Pilot budget differs')
    require(protocol['reporting']['seeds'] == SEEDS and protocol['reporting']['updates'] == 300
            and protocol['reporting']['expected_runs'] == 12, 'Reporting population differs')
    for name, expected_sha in protocol['source_sha256'].items():
        require(sha(file(source_root, name)) == expected_sha, 'Scientific source hash differs: ' + name)


def fit(record):
    q, lower = [number(record[k]) for k in ('objective', 'lower')]
    gap = number(record['gap'], -1e-12 * max(q, 1.))
    close(q - lower, gap)
    require(lower <= q + 1e-12 and gap <= 1e-6 * max(q, 1e-20) + 1e-12, 'Unresolved full recovery')


def audit(row):
    a = row['native_audit']
    require(all(a[k] is True for k in ('passed', 'passed_audit', 'passed_gain_audit',
                'passed_simplex_gate', 'relative_reference_resolved')), 'Native audit boolean gate failed')
    for key in ('native_direction_relative_error', 'native_table_direction_relative_error', 'head_relative_error'):
        require(number(a[key]) <= 2e-5, 'Native gradient fidelity gate failed')
    limit = 1.05 if row['arm'].startswith('stream_') and not row['fallback'] else 1.001
    for key in ('native_objective_factor_upper', 'native_emitted_objective_factor_upper'):
        require(number(a[key]) <= limit + 1e-5, 'Native objective gate failed')
    gain = number(a['native_normalized_gain_lower'])
    require(gain >= .95 - 1e-5, 'Native response gain gate failed')
    close(gain, a['normalized_gain_to_reference_primal'])
    K = np.asarray(a['reference_gram'], dtype=float)
    require(K.shape == (40, 40) and np.isfinite(K).all() and np.allclose(K, K.T, rtol=1e-8, atol=1e-12), 'Native Gram shape/symmetry')
    fit(a['reference'])
    lower, q = a['reference']['lower'], a['reference']['objective']
    require(lower > max(float(np.abs(K).max()), 1e-100) * 1e-10, 'Unresolved reference energy')
    w = vector(row['weights'], True)
    close(float(w @ K @ w) / lower, a['native_objective_factor_upper'], 1e-6)
    norm = number(a['executed_norm_squared'])
    close(norm / lower, a['native_emitted_objective_factor_upper'], 1e-6)
    require(norm > 0 and q > 0 and a['executed_min_response'] > 0, 'Unresolved native gain')
    close(a['executed_min_response'] / math.sqrt(norm * q), gain, 1e-6)
    number(a['seconds']); number(a['peak_allocated_bytes'])


def trace(row):
    require(type(row['fallback']) is bool and type(row['adaptive_actions']) is int, 'Invalid adaptive counters')
    w = vector(row['weights'], True)
    require(np.array_equal(np.asarray(row['executed_weights']), w.astype(np.float32).astype(float)), 'Executed weight cast differs')
    require(vector(row['task_losses_before']).min() >= 0, 'Negative BCE')
    queries, prefixes = row.get('queries', []), row.get('prefixes', [])
    require(row['adaptive_actions'] == len(queries) <= 40, 'Adaptive action coverage')
    failures = 0
    for i, query in enumerate(queries, 1):
        require(query['index'] == i, 'Query order')
        vector(query['query']); number(query['seconds'])
        if 'error' in query:
            require(row['fallback'] and isinstance(query['error'], str) and query['error'], 'Unaccounted failed query')
            failures += 1
        else:
            if nonfinite_evidence(query['response']):
                require(row['fallback'] and row.get('adaptive_error'), 'Nonfinite action without retained fallback')
                failures += 1
            else: vector(query['response'])
    solver_failures = 0
    for i, prefix in enumerate(prefixes, 1):
        require(prefix['actions'] == i <= len(queries), 'Endpoint prefix order')
        if 'upper' in prefix:
            upper = prefix['upper']
            if nonfinite_evidence(upper):
                require(row['fallback'], 'Nonfinite endpoint without fallback'); solver_failures += 1
            else:
                weights = vector(upper['weights'], True); response = vector(upper['response'])
                close(float(weights @ response), upper['objective'], 1e-7)
        else:
            require(row['fallback'] and prefix.get('upper_error'), 'Missing upper endpoint')
        for key in ('upper_error', 'completion_qp_error'):
            solver_failures += int(key in prefix)
        if 'completion_qp' in prefix:
            qp = prefix['completion_qp']
            if nonfinite_evidence(qp):
                require(row['fallback'], 'Nonfinite lower solver without fallback'); solver_failures += 1
                continue
            qw = vector(qp['weights'], True); response = vector(qp['response'])
            close(float(qw @ response), qp['objective'])
            close(max(0., min(qp['objective'], 2 * float(response.min()) - qp['objective'])), qp['lower'])
            close(qp['objective'] - qp['lower'], qp['dual_gap'])
            require(type(qp['converged']) is bool, 'QP status missing')
            if not qp['converged']:
                require(row['fallback'], 'Failed solver without fallback')
                solver_failures += 1
    if row['arm'].startswith('stream_'):
        require(queries or row['fallback'], 'Stream omitted acquisition')
        if row['fallback']:
            require(row.get('adaptive_error') and 'fallback_full_recovery' in row, 'Missing fallback provenance')
            fit(row['fallback_full_recovery'])
        else:
            recovery = row['recovery']
            require(recovery['accepted'] is True and recovery['status'] == 'accepted'
                    and recovery['actions'] == len(queries) == len(prefixes), 'Unaccepted Stream recovery')
            require(len(recovery['history']) == len(prefixes), 'Missing stopping histories')
            lower = 0.
            for history, prefix in zip(recovery['history'], prefixes):
                require(history['actions'] == prefix['actions'], 'Stopping history order')
                close(history['U'], prefix['upper']['objective'])
                close(history['min_response'], min(prefix['upper']['response']))
                require(history['lower_qp_used'] is ('completion_qp' in prefix), 'Lower solver accounting differs')
                number(history['lower'])
                require(history['U'] > 0, 'Nonpositive stopping objective')
                lower = max(lower, max(0., history['min_response'])**2 / history['U'])
                if 'completion_qp' in prefix:
                    close(history['completion_lower'], prefix['completion_qp']['lower'])
                    require(number(history['interpolation']) <= 2e-5, 'Unresolved completion interpolation')
                    if row['arm'] == 'stream_completion': lower = max(lower, prefix['completion_qp']['lower'])
                close(lower, history['lower'])
            close(recovery['objective'], prefixes[-1]['upper']['objective'])
            close(recovery['lower'], lower)
            require(np.allclose(w, prefixes[-1]['upper']['weights'], rtol=1e-8, atol=1e-10)
                    and np.allclose(recovery['response'], prefixes[-1]['upper']['response'], rtol=1e-8, atol=1e-10), 'Executed/terminal endpoint differs')
            require(recovery['lower'] > 0 and recovery['objective'] <= 1.05 * recovery['lower'] + 1e-12,
                    'Stream stopping energy failed')
            require(min(recovery['response']) >= .95 * recovery['objective'] - 1e-12, 'Stream stopping margin failed')
            vector(recovery['response'])
    else:
        require(not row['fallback'] and not queries and not prefixes, 'Unexpected control adaptive work')
        fit(row['full_recovery'])
    return failures, solver_failures


def evaluation(value, count):
    require(value['n'] == count, 'Evaluation population differs')
    for kind, maximize in [('accuracy', False), ('balanced_accuracy', False), ('bce', True)]:
        vals = vector(value['per_attribute_' + kind])
        require(vals.min() >= 0 and (kind == 'bce' or vals.max() <= 1), 'Evaluation range')
        close(vals.mean(), value['mean_' + kind], 2e-7)
        close(vals.max() if maximize else vals.min(), value['worst_' + kind], 2e-7)


def validate_run(out, protocol, protocol_sha, phase, arm, seed, *, test_count=19962):
    out = Path(out)
    require(not (out / 'failure.json').exists(), 'Quarantined scientific failure')
    d = read(out / 'metrics.json'); steps = 100 if phase == 'pilot' else 300
    cfg = dict(phase=phase, arm=arm, seed=seed, steps=steps, lr=.02, batch_size=128)
    identity = dict(protocol_sha256=protocol_sha, sources=protocol['source_sha256'],
                    data_manifest_sha256=protocol['data_manifest_sha256'], selection_sha256=protocol['prior_selection_sha256'])
    require(d['config'] == cfg and d['identity'] == identity and d['complete'] is True and d['next_step'] == steps, 'Run identity/completion')
    require(read(out / 'run_identity.json') == {'config': cfg, 'identity': identity}, 'Run ownership')
    require(sha(out / 'latest.pt') == d['latest_checkpoint_sha256'], 'Missing/corrupted final model')
    require(len(d['initial_model_sha256']) == 64 and all(c in '0123456789abcdef' for c in d['initial_model_sha256']), 'Initial model hash')
    require(d['heldout_test_evaluated'] is (phase == 'reporting'), 'Heldout policy')
    if phase == 'pilot':
        require('final_test' not in d and 'final_validation' in d, 'Pilot test leakage')
        # Deliberately never index final_validation: no pilot quality selection.
    else:
        evaluation(d['final_validation'], 4000); evaluation(d['final_test'], test_count)
    require(len(d['events']) == steps, 'Incomplete event population')
    rows, batches, refs, failed_queries, solver_failures = [], [], set(), 0, 0
    for step, ref in enumerate(d['events']):
        path = file(out, ref['file'])
        require(path.parent == out / 'events' and ref['file'] not in refs and sha(path) == ref['sha256'], 'Event path/hash/duplicate')
        refs.add(ref['file']); row = read(path)
        require(row['step'] == step and row['seed'] == seed and row['arm'] == arm
                and row['complete'] is True and row['applied'] is True, 'Event order/identity')
        indices = torch.randint(16000, (128,), generator=torch.Generator().manual_seed(seed * 100000 + step))
        require(row['indices'] == indices.tolist() and row['indices_sha256'] == hashlib.sha256(indices.numpy().tobytes()).hexdigest(), 'Deterministic sample order differs')
        batches.append(row['indices_sha256'])
        for key in ('recovery_and_head_seconds', 'optimizer_seconds', 'update_seconds', 'peak_allocated_bytes', 'incremental_peak_bytes', 'base_allocated_bytes'):
            number(row[key])
        close(row['recovery_and_head_seconds'] + row['optimizer_seconds'], row['update_seconds'])
        require(row['peak_allocated_bytes'] >= row['base_allocated_bytes'], 'Peak below baseline')
        close(max(0, row['peak_allocated_bytes'] - row['base_allocated_bytes']), row['incremental_peak_bytes'])
        f, s = trace(row); failed_queries += f; solver_failures += s
        require(('native_audit' in row) == (step in (0, steps // 2, steps - 1)), 'Audit milestone coverage')
        if 'native_audit' in row:
            audit(row)
        rows.append(row)
    require(d['audited_steps'] == [0, steps // 2, steps - 1], 'Audit summary differs')
    for key, expected in [('update_seconds', sum(r['update_seconds'] for r in rows)),
            ('adaptive_actions', sum(r['adaptive_actions'] for r in rows)),
            ('fallback_count', sum(r['fallback'] for r in rows)),
            ('peak_update_allocated_bytes', max(r['peak_allocated_bytes'] for r in rows)),
            ('peak_update_incremental_bytes', max(r['incremental_peak_bytes'] for r in rows))]:
        close(d[key], expected)
    attempts = [read(p) for p in sorted((out / 'attempts').glob('*.json'))]
    require(attempts == d['attempts'] and attempts and [a['attempt'] for a in attempts] == list(range(1, len(attempts) + 1)), 'Attempt coverage')
    for a in attempts:
        require(type(a['complete']) is bool and not a.get('error'), 'Failed scientific attempt')
        number(a.get('recorded_elapsed_seconds', 0.))
    close(d['recorded_process_seconds'], sum(a.get('recorded_elapsed_seconds', 0.) for a in attempts))
    require(d['unrecorded_interrupted_tails'] == sum(not a['complete'] for a in attempts), 'Interrupted tail accounting')
    extras = sorted(p for p in (out / 'events').glob('*.json') if str(p.relative_to(out)) not in refs)
    require(d['uncommitted_attempt_events'] == [{'file': str(p.relative_to(out)), 'sha256': sha(p)} for p in extras], 'Orphan event accounting')
    close(d['uncommitted_recorded_update_seconds'], sum(read(p).get('update_seconds', 0.) for p in extras))
    summary = {key: d[key] for key in ('initial_model_sha256', 'update_seconds', 'peak_update_allocated_bytes',
               'peak_update_incremental_bytes', 'adaptive_actions', 'fallback_count', 'recorded_process_seconds',
               'unrecorded_interrupted_tails', 'uncommitted_recorded_update_seconds', 'latest_checkpoint_sha256')}
    summary.update(arm=arm, seed=seed, metrics_sha256=sha(out / 'metrics.json'), batch_sequence_sha256=digest(batches),
                   failed_queries=failed_queries, solver_failures=solver_failures,
                   audit_seconds=sum(r.get('native_audit', {}).get('seconds', 0.) for r in rows),
                   max_audit_peak_bytes=max(r.get('native_audit', {}).get('peak_allocated_bytes', 0) for r in rows),
                   audit_records=[r['native_audit'] for r in rows if 'native_audit' in r], attempt_count=len(attempts))
    if phase == 'reporting':
        summary.update(final_validation=d['final_validation'], final_test=d['final_test'])
    return summary


def population(base, protocol, protocol_sha, phase):
    base = Path(base); results = []
    seeds = [61900] if phase == 'pilot' else SEEDS
    for seed in seeds:
        group = [validate_run(base / (arm if phase == 'pilot' else f'seed{seed}/{arm}'), protocol, protocol_sha, phase, arm, seed) for arm in ARMS]
        require(len({r['initial_model_sha256'] for r in group}) == len({r['batch_sequence_sha256'] for r in group}) == 1, 'Unpaired initial models/batches')
        results.extend(group)
    return results


def paired(values):
    x = np.asarray(values, dtype=float)
    require(x.shape == (3,) and np.isfinite(x).all(), 'Three paired outcomes required')
    mean = float(x.mean()); half = 4.302652729911275 * float(x.std(ddof=1)) / math.sqrt(3)
    return {'paired_seed_differences': x.tolist(), 'mean': mean, 'descriptive_t95': [mean - half, mean + half],
            'interpretation': 'Descriptive n=3 paired interval; no convergence or superiority claim'}


def process_wall(root, protocol, protocol_sha, rows):
    attempts = sorted((Path(root) / 'reporting_control').glob('attempt_*.json'))
    require(attempts, 'Missing full-process reporting attempts')
    successes, seconds, by_run, active_seconds = {}, 0., {}, 0.
    identity = {'protocol_sha256': protocol_sha, 'control_sources': protocol['reporting_freeze']['control_sources']}
    require(read(Path(root) / 'reporting_control/identity.json') == identity, 'Controller ownership differs')
    for path in attempts:
        attempt = read(path)
        active_seconds += number(attempt['control_active_seconds'])
        require(attempt['identity'] == identity and (attempt['complete'] is True or attempt.get('interrupted') is True), 'Unmeasured/failed controller attempt')
        for child in attempt['children']:
            require(child['complete'] is True, 'Unmeasured subprocess tail')
            key = child['seed'], child['arm']
            require(key[0] in SEEDS and key[1] in ARMS, 'Unplanned reporting subprocess')
            seconds += number(child['wall_seconds'])
            by_run[key] = by_run.get(key, 0.) + child['wall_seconds']
            if child['returncode'] == 0:
                require(key not in successes, 'Repeated successful reporting subprocess')
                successes[key] = child
            else:
                require(child.get('interrupted') is True, 'Failed reporting subprocess')
    require(set(successes) == {(seed, arm) for seed in SEEDS for arm in ARMS}, 'Missing successful reporting subprocesses')
    require(seconds <= 10800, 'Reporting exceeded frozen total runtime allowance')
    require(seconds <= active_seconds <= 10800, 'Recorded exclusive control wall/budget differs')
    require(all(by_run[r['seed'], r['arm']] >= r['recorded_process_seconds'] for r in rows), 'Subprocess wall omits recorded scientific work')
    return seconds


def analyze(root, protocol_path, source_root, *, completed_marker=True):
    root, source_root = Path(root), Path(source_root)
    protocol = read(protocol_path); validate_protocol(protocol, source_root)
    require(protocol['enabled_phases'] == ['reporting'] and protocol['reporting_budget_frozen'] is True, 'Reporting not frozen')
    freeze = protocol['reporting_freeze']
    require(freeze['quality_values_used'] is False and freeze['budget_seconds'] == 10800, 'Reporting freeze policy differs')
    require(all(sha(source_root / name) == value for name, value in freeze['control_sources'].items()), 'Frozen analysis/controller source differs')
    require(sha(root / 'sharp_training_pilot_protocol.json') == freeze['pilot_protocol_sha256']
            and sha(root / 'pilot_complete.json') == freeze['pilot_complete_sha256']
            and sha(root / 'sharp_training_pilot_wall_observation.json') == freeze['observer_sha256']
            and sha(source_root / 'sharp_training_reporting_plan.json') == freeze['plan_sha256'], 'Frozen pilot/control evidence differs')
    pilot = read(root / 'sharp_training_pilot_protocol.json')
    require(population(root / 'pilot', pilot, freeze['pilot_protocol_sha256'], 'pilot') == freeze['pilot_diagnostics'], 'Frozen pilot diagnostics changed')
    raw = 36 * max(freeze['pilot_full_process_upper_seconds'].values())
    close(raw, freeze['extrapolated_seconds']); close(max(1800., .25 * raw), freeze['reserve_seconds'])
    close(raw + freeze['reserve_seconds'], freeze['conservative_reporting_seconds'])
    require(freeze['conservative_reporting_seconds'] <= 10800, 'Pilot projection exceeded budget')
    rows = population(Path(root) / 'reporting', protocol, sha(protocol_path), 'reporting')
    by = {(r['seed'], r['arm']): r for r in rows}
    contrasts = {arm: paired([by[seed, 'stream_completion']['final_test']['worst_accuracy'] - by[seed, arm]['final_test']['worst_accuracy'] for seed in SEEDS]) for arm in ARMS if arm != 'stream_completion'}
    result = {'complete': True, 'runs': rows, 'expected_runs': 12, 'protocol_sha256': sha(protocol_path),
            'subprocess_wall_seconds': process_wall(root, protocol, sha(protocol_path), rows),
            'primary_stream_completion_minus_arm_worst_test_accuracy': contrasts,
            'scope': 'Final300-update bounded early training; all40 metrics retained; numerical fidelity only, no full-AD certificate or momentum descent claim',
            'timing': 'Measured update includes SGD; audits separate; recorded main-entry wall excludes imports and interrupted tails. Controller subprocess attempt wall is reported separately.'}
    if completed_marker:
        attempts = [read(p) for p in sorted((root / 'reporting_control').glob('attempt_*.json'))]
        require(all(p.get('control_wall_finalized') is True for p in attempts), 'Final control wall not yet recorded')
        active_wall = sum(number(p['control_active_seconds']) for p in attempts)
        require(result['subprocess_wall_seconds'] <= active_wall <= 10800, 'Final control wall/budget differs')
        marker = read(root / 'reporting_complete.json')
        require(marker == {'complete': True, 'protocol_sha256': sha(protocol_path),
                          'analysis_sha256': sha(root / 'sharp_training_reporting_analysis.json'),
                          'runs': [{k: r[k] for k in ('seed', 'arm', 'metrics_sha256', 'latest_checkpoint_sha256')} for r in rows]}, 'Terminal reporting marker differs')
        require(read(root / 'sharp_training_reporting_analysis.json') == result, 'Saved complete analysis differs')
        result['finalized_exclusive_control_seconds'] = active_wall
        result['control_wall_scope'] = 'After stable idle through completed validation/result and metadata publication; excludes the final timing receipt, launcher-release bookkeeping and subsequent CPU-only backup after launcher release.'
    return result


def main():
    p = argparse.ArgumentParser(); p.add_argument('--root', type=Path, required=True); p.add_argument('--protocol', type=Path, required=True)
    p.add_argument('--source-root', type=Path, default=Path(__file__).resolve().parent); p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(); result = analyze(a.root, a.protocol, a.source_root)
    a.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
