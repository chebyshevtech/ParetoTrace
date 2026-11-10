"""Analysis of the certified-step Pythia-1B study. Usage: analyze_certified_1b.py <study dir> <campaign_analysis.json> [--out out.json]

Reads runs/*/metrics.json and their hash-bound update records. Reports the development table over (c, eta0), per-arm
counts on the reporting seeds, the frozen primary contrast, the secondary contrasts, the price of the guarantee against
the unprotected uniform step, and the primary rule against its companion at the other sufficient-decrease constant
(paired by seed, descriptive Student-t 95% intervals with df=2), and each arm against the campaign's unprotected runs of
the same seed and batch sequence.
"""
import hashlib
import json
import math
import statistics as st
import sys
from pathlib import Path

import numpy as np

T95 = 4.302652729911275  # Student t, 0.975 quantile, df = 2
study, campaign = Path(sys.argv[1]), json.loads(Path(sys.argv[2]).read_text())
out_path = Path(sys.argv[sys.argv.index('--out') + 1]) if '--out' in sys.argv else None
P = json.loads((study / 'protocol.json').read_text()) if (study / 'protocol.json').exists() else None
assert abs(T95 - math.sqrt(2 * .95**2 / (1 - .95**2))) < 1e-9


def load_run(d):
    m = json.loads((d / 'metrics.json').read_text())
    rows = []
    for blk in m['update_blocks']:
        f = d / blk['file']
        if hashlib.sha256(f.read_bytes()).hexdigest() != blk['sha256']:
            raise ValueError('update block hash differs: %s' % f)
        rows += json.loads(f.read_text())
    if [r['step'] for r in rows] != list(range(m['steps'])):
        raise ValueError('update records incomplete: %s' % d)
    return m, rows


def step_length(r):
    """eta times the direction norm, sqrt(w . K w), for an accepted certified step."""
    if r['status'] != 'accepted' or r.get('first_order_response') is None:
        return None
    return r['accepted_eta'] * math.sqrt(max(float(np.dot(r['weights'], r['first_order_response'])), 0.))


runs, failed_runs = {}, []
for d in sorted(p for p in (study / 'runs').iterdir() if p.is_dir()):
    if (d / 'metrics.json').exists():
        m, rows = load_run(d)
        c = m['config']
        runs[(c['phase'], c['arm'], c['seed'], c['eta0'], c['c'])] = (m, rows)
    elif (d / 'failure.json').exists():
        f = json.loads((d / 'failure.json').read_text())
        failed_runs.append({'run': d.name, 'phase': f['config']['phase'], 'arm': f['config']['arm'], 'seed': f['config']['seed'],
                            'eta0': f['config']['eta0'], 'c': f['config']['c'], 'error': f['error'].strip().splitlines()[-1][:300]})

report = {'protocol_sha256': hashlib.sha256((study / 'protocol.json').read_bytes()).hexdigest() if P else None,
          'development': {}, 'confirmatory': {}, 'failed_runs': failed_runs}
for (ph, arm, seed, eta, c), (m, rows) in sorted(runs.items()):
    if ph != 'development':
        continue
    lengths = [x for x in map(step_length, rows) if x is not None]
    report['development'].setdefault(arm, []).append({
        'c': c, 'eta0': eta, 'final_validation_worst_ce': m['final_validation']['worst_ce'],
        'final_validation_mean_ce': m['final_validation']['mean_ce'], 'status_counts': m['status_counts'],
        'updates_raising_a_task': m['updates_raising_a_task'], 'weighting_counts': m['weighting_counts'],
        'accepted_eta_median': m['accepted_eta_median'], 'accepted_step_length_median': float(np.median(lengths)) if lengths else None,
        'update_seconds_median': m['update_seconds_median'], 'study_source_sha256': m['identity']['sources']['certified_language_1b.py']})

SEEDS = tuple(P['confirmatory_seeds']) if P else (49201, 49202, 49203)
configs = {}
if P:
    for arm in P['arms']:
        configs[(arm, P['c'][arm], P['eta0'][arm])] = arm
    for x in P.get('companions', []):
        configs[(x['arm'], x['c'], x['eta0'])] = '%s_c%g' % (x['arm'], x['c'])
conf = {}
for (ph, arm, seed, eta, c), v in runs.items():
    if ph != 'confirmatory':
        continue
    key = configs.get((arm, c, eta))
    if key is None:
        raise ValueError('confirmatory run outside the frozen protocol: %s' % ((arm, seed, eta, c),))
    if v[0]['identity'].get('protocol_sha256') != report['protocol_sha256']:
        raise ValueError('confirmatory run bound to a different protocol: %s' % ((arm, seed),))
    conf[(key, seed)] = v
keys = sorted({k for k, s in conf})
for key in keys:
    rs = [conf[(key, s)] for s in SEEDS if (key, s) in conf]
    if len(rs) < len(SEEDS):
        report['confirmatory'][key] = {'incomplete': True, 'seeds_done': len(rs),
                                       'failed': [f['run'] for f in failed_runs if f['phase'] == 'confirmatory' and configs.get((f['arm'], f['c'], f['eta0'])) == key]}
        continue
    ms = [r[0] for r in rs]
    allrows = [x for r in rs for x in r[1]]
    lengths = [x for x in map(step_length, allrows) if x is not None]
    certified = ms[0]['config']['arm'] != 'uniform_fixed'
    report['confirmatory'][key] = {
        'arm': ms[0]['config']['arm'], 'c': ms[0]['config']['c'], 'eta0': ms[0]['config']['eta0'],
        'updates': len(allrows),
        'status_counts': {k: sum(m['status_counts'][k] for m in ms) for k in ms[0]['status_counts']},
        'updates_raising_a_task': sum(m['updates_raising_a_task'] for m in ms),
        'largest_single_task_increase': max(x['max_task_change'] for x in allrows),
        'weighting_counts': {k: sum(m['weighting_counts'][k] for m in ms) for k in ms[0]['weighting_counts']},
        'uniform_response_negative': sum(1 for x in allrows if x.get('uniform_response') is not None and min(x['uniform_response']) < 0),
        'uniform_response_recorded': sum(1 for x in allrows if x.get('uniform_response') is not None),
        'accepted_eta_median': float(np.median([x['accepted_eta'] for x in allrows if x['status'] == 'accepted'])) if any(x['status'] == 'accepted' for x in allrows) else None,
        'accepted_step_length_median': float(np.median(lengths)) if lengths else None,
        'accepted_at_full_step': sum(m['accepted_at_full_step'] for m in ms),
        'update_seconds_median': float(np.median([x['update_seconds'] for x in allrows])),
        'peak_gib': max(m['peak_update_allocated_bytes'] for m in ms) / 2**30,
        'audits': sum(m['audits'] for m in ms), 'audits_passed': sum(m['audits_passed'] for m in ms),
        'audit_max_relative_error': max([m['audit_max_relative_error'] for m in ms if m['audit_max_relative_error'] is not None], default=None),
        'max_realized_change_certified': max(x['max_task_change'] for x in allrows) if certified else None,
        'loss_recompute_max_abs': max((x['loss_recompute_max_abs'] for x in allrows if 'loss_recompute_max_abs' in x), default=None),
        'final_test': {k: {'mean': st.mean(m['final_test'][k] for m in ms), 'per_seed': [m['final_test'][k] for m in ms]}
                       for k in ('worst_ce', 'mean_ce', 'worst_answer_exact_match', 'mean_answer_exact_match')},
        'final_validation_worst_ce_mean': st.mean(m['final_validation']['worst_ce'] for m in ms),
        'worst_train_task_after_last_update_mean': st.mean(r[1][-1]['worst_task_after'] for r in rs)}


def contrast(a, b, key='worst_ce'):
    if not all((x, s) in conf for x in (a, b) for s in SEEDS):
        if any((x, s) in conf for x in (a, b) for s in SEEDS):
            return {'contrast': '%s-minus-%s' % (a, b), 'complete': False,
                    'per_seed': [conf[(a, s)][0]['final_test'][key] - conf[(b, s)][0]['final_test'][key]
                                 if (a, s) in conf and (b, s) in conf else None for s in SEEDS]}
        return None
    d = [conf[(a, s)][0]['final_test'][key] - conf[(b, s)][0]['final_test'][key] for s in SEEDS]
    mu, sd = st.mean(d), st.stdev(d)
    h = T95 * sd / math.sqrt(len(d))
    return {'contrast': '%s-minus-%s' % (a, b), 'endpoint': 'final held-out %s' % key, 'per_seed': d, 'mean': mu,
            't95': [mu - h, mu + h], 'first_lower_seeds': sum(x < 0 for x in d)}


report['primary'] = contrast('auditfirst_certified', 'uniform_certified')
report['secondary'] = [c for c in (contrast('direct_certified', 'uniform_certified'), contrast('famo_certified', 'uniform_certified'),
                                   contrast('auditfirst_certified', 'direct_certified')) if c]
report['price_of_the_guarantee'] = [c for c in (contrast(k, 'uniform_fixed') for k in keys if k != 'uniform_fixed') if c]
report['price_of_the_guarantee_mean_ce'] = [c for c in (contrast(k, 'uniform_fixed', 'mean_ce') for k in keys if k != 'uniform_fixed') if c]
report['constant_contrast'] = [c for c in (contrast('auditfirst_certified', k) for k in keys if k.startswith('auditfirst_certified_c')) if c]
camp = {(r['method'], r['seed']): r for r in campaign['runs']['confirmatory']}
pairs = []
for key in keys:
    for ref in ('uniform', 'direct'):
        if all((key, s) in conf and (ref, s) in camp for s in SEEDS):
            d = [conf[(key, s)][0]['final_test']['worst_ce'] - camp[(ref, s)]['final_test']['worst_ce'] for s in SEEDS]
            mu, h = st.mean(d), T95 * st.stdev(d) / math.sqrt(len(d))
            pairs.append({'contrast': '%s-minus-campaign-%s' % (key, ref), 'per_seed': d, 'mean': mu, 't95': [mu - h, mu + h]})
report['against_campaign_unprotected'] = pairs
if all(('uniform_fixed', s) in conf for s in SEEDS) and all(conf[('uniform_fixed', s)][0]['config']['eta0'] == camp[('uniform', s)]['lr'] for s in SEEDS):
    report['reproduction_uniform_fixed_vs_campaign_uniform'] = {
        'note': 'same seeds, batch sequence, rate and 200 updates as the campaign uniform runs',
        'campaign_steps': [camp[('uniform', s)]['steps'] for s in SEEDS],
        'final_test_worst_ce_difference_per_seed': [conf[('uniform_fixed', s)][0]['final_test']['worst_ce'] - camp[('uniform', s)]['final_test']['worst_ce'] for s in SEEDS]}
print(json.dumps(report, indent=1)[:9000])
if out_path:
    out_path.write_text(json.dumps(report, indent=1))
