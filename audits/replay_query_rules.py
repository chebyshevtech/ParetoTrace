"""Exact replay of Stream acquisition on stored Grams (CPU), reproducing the query-rule claims.

Uses paretotrace/stream/stable_recovery_v2.recover, whose upper solve, completion QP and
stopping test are the ones used live. Actions are exact products K v. Writes
results/query_rule/replay_results.json and checks that replay predicts every live action
count in results/query_rule/query_rule_study.json.

Rules: queries 'completion' (original) or 'gate' (target-aware); endpoint 'completion'
(sharp, L_sk) or 'margin' (scalar, max(m,0)^2/U); gamma .05 (0.95 gate) or 1.0 (strict
descent: non-negative responses, with L > 0 and U <= 1.05 L required for acceptance).
"""
import json, statistics as st, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'paretotrace/stream'))
from stable_recovery_v2 import recover  # noqa: E402
from v46_baseline_frontier import qp  # noqa: E402

CONFIGS = {'original': ('completion', 'completion', .05), 'target_aware_0.95': ('gate', 'completion', .05),
           'target_aware_strict': ('gate', 'completion', 1.0), 'scalar_endpoint_strict': ('gate', 'margin', 1.0)}


def celeba_grams():
    out = []
    rec = json.loads((ROOT / 'results/recovery_study/results.json').read_text())
    out += [('recovery_%d' % s['seed'], np.asarray(s['gram'], float)) for s in rec['states']]
    ana = json.loads((ROOT / 'results/sharp_training/sharp_training_reporting_analysis.json').read_text())
    for r in ana['runs']:
        for a, step in zip(r['audit_records'], (0, 150, 299)):
            out.append(('train_%d_%s_step%d' % (r['seed'], r['arm'], step), np.asarray(a['reference_gram'], float)))
    return out


def distinct(grams):
    seen, keep = set(), []
    for name, K in grams:
        key = np.round(K / np.abs(K).max(), 6).tobytes()
        if key not in seen:
            seen.add(key); keep.append((name, K))
    return keep


def run(grams):
    rows = []
    for name, K in grams:
        K = (K + K.T) / 2
        q = float(qp(K, tolerance=2e-13, max_iterations=20000)['objective'])
        row = {'gram': name, 'qstar': q}
        for label, (qr, lr, g) in CONFIGS.items():
            r = recover(lambda v: K @ v, len(K), eps=.05, gamma=g, lower_rule=lr, query_rule=qr, max_actions=len(K))
            row[label] = {'actions': r['actions'], 'accepted': r['accepted'], 'factor': (r['objective'] / q) if (r['accepted'] and q > 0) else None}
        rows.append(row)
    summary = {}
    for label in CONFIGS:
        a = [r[label]['actions'] for r in rows]
        summary[label] = {'n': len(rows), 'accepted': sum(r[label]['accepted'] for r in rows), 'mean_actions': st.mean(a),
                          'median_actions': st.median(a), 'max_factor': max(r[label]['factor'] for r in rows if r[label]['factor'] is not None)}
    return rows, summary


def main():
    sets = {'celeba_all41': celeba_grams()}
    sets['celeba_distinct'] = distinct(sets['celeba_all41'])
    for nm, f in (('cifar10_m100', 'cifar10_m100.npz'), ('recorded103', 'recorded103.npz')):
        z = np.load(ROOT / 'results/grams' / f)
        sets[nm] = [(k, np.asarray(z[k], float)) for k in z.keys()]
    out = {'configs': {k: list(v) for k, v in CONFIGS.items()}, 'sets': {}}
    for nm, grams in sets.items():
        rows, summary = run(grams)
        out['sets'][nm] = {'summary': summary, 'rows': rows}
        print(nm, {k: (v['accepted'], v['n'], round(v['mean_actions'], 2)) for k, v in summary.items()}, flush=True)
    live = json.loads((ROOT / 'results/query_rule/query_rule_study.json').read_text())
    states = {s['seed']: np.asarray(s['gram'], float) for s in live['states']}
    mism, n = 0, 0
    for r in live['rows']:
        if r['rule'] == 'direct' or not r['complete']:
            continue
        K = states[r['seed']]; K = (K + K.T) / 2
        p = recover(lambda v: K @ v, 40, eps=r['epsilon'], gamma=r['gamma'], lower_rule='completion', query_rule=r['rule'], max_actions=40)
        n += 1; mism += p['actions'] != r['recovery']['actions']
    out['live_action_count_check'] = {'live_stream_rows': n, 'mismatches': mism}
    print('live action counts predicted by replay:', n - mism, 'of', n)
    (ROOT / 'results/query_rule/replay_results.json').write_text(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
