"""Uniform weighting's first-order trunk responses at every update of the Stream training runs (CPU only).

Every Stream recovery starts from uniform weighting, so its first Gram action is K v with v = 1/sqrt(M).
The response of the uniform direction, K (1/M), is that action rescaled. This script reads only metrics.json
and events/*.json, and for each run reports how often the uniform trunk direction would have raised at least
one task's loss to first order, next to the margin of the direction the run actually executed.

Usage: python3 -I uniform_along_training.py <run_dir> [<run_dir> ...] > out.json
"""
import json
import statistics as st
import sys
from pathlib import Path


def load(p):
    return json.loads(Path(p).read_text())


def run_summary(run):
    run = Path(run)
    refs = load(run / 'metrics.json')['events']
    rows = [load(run / r['file']) for r in refs]
    assert [r['step'] for r in rows] == list(range(len(rows))), 'step order'
    neg_updates, neg_counts, norm_margins, exec_margins, exec_positive, fallbacks, checked = 0, [], [], [], 0, 0, 0
    for r in rows:
        q0 = r['queries'][0]
        query, response = [float(v) for v in q0['query']], [float(v) for v in q0['response']]
        M = len(query)
        assert max(query) - min(query) <= 1e-12 * max(query), 'first query is not uniform'
        scale = (1.0 / M) / query[0]
        ub = [v * scale for v in response]
        qb = sum(ub) / M
        neg = sum(v < 0 for v in ub)
        neg_updates += neg > 0
        neg_counts.append(neg)
        norm_margins.append(min(ub) / qb)
        checked += 1
        if r['fallback']:
            fallbacks += 1
            continue
        rec = r['recovery']
        m = min(float(v) for v in rec['response'])
        exec_margins.append(m / float(rec['objective']))
        exec_positive += m > 0
    return {'run': '/'.join(run.parts[-3:]), 'arm': rows[0]['arm'], 'seed': rows[0]['seed'], 'updates': len(rows),
            'uniform_checked': checked, 'uniform_raises_a_task': neg_updates,
            'uniform_tasks_raised_median': st.median(neg_counts), 'uniform_tasks_raised_max': max(neg_counts),
            'uniform_normalized_margin_median': st.median(norm_margins), 'uniform_normalized_margin_max': max(norm_margins),
            'executed_stream_updates': len(exec_margins), 'executed_margin_positive': exec_positive,
            'executed_normalized_margin_min': min(exec_margins) if exec_margins else None,
            'fallbacks_exact': fallbacks}


def main():
    runs = [run_summary(p) for p in sys.argv[1:]]
    total = {k: sum(r[k] for r in runs) for k in ('updates', 'uniform_checked', 'uniform_raises_a_task',
                                                    'executed_stream_updates', 'executed_margin_positive', 'fallbacks_exact')}
    total['uniform_tasks_raised_median_of_run_medians'] = st.median(r['uniform_tasks_raised_median'] for r in runs)
    print(json.dumps({'scope': 'shared-trunk first-order responses on each update minibatch; private heads not included',
                      'runs': runs, 'total': total}, indent=1))


if __name__ == '__main__':
    main()
