"""Independent event-level summary of the sharp-endpoint training runs (CPU only).

Reads only metrics.json and events/*.json (no checkpoints, no project imports).
For each run directory given, recomputes action counts, early stops, stopping
conditions, fallbacks, timing, memory and native audit figures directly from
the per-update records, and prints a JSON summary.

Usage: python3 -I sharp_events_summary.py <run_dir> [<run_dir> ...] > out.json
"""
import json
import math
import statistics as st
import sys
from pathlib import Path

EPS, MARGIN, M = 0.05, 0.95, 40


def load(p):
    return json.loads(Path(p).read_text())


def first_prefix(history, cond):
    for h in history:
        if cond(h):
            return h['actions']
    return None


def run_summary(run):
    run = Path(run)
    metrics = load(run / 'metrics.json')
    refs = metrics['events']
    rows = [load(run / r['file']) for r in refs]
    assert [r['step'] for r in rows] == list(range(len(rows))), 'step order'
    arm = rows[0]['arm']
    out = {'run': str(run.relative_to(run.parents[2])) if len(run.parents) > 2 else str(run),
           'arm': arm, 'seed': rows[0]['seed'], 'updates': len(rows),
           'fallbacks': sum(bool(r['fallback']) for r in rows),
           'update_seconds_mean': st.mean(r['update_seconds'] for r in rows),
           'update_seconds_median': st.median(r['update_seconds'] for r in rows),
           'update_seconds_total': sum(r['update_seconds'] for r in rows),
           'peak_allocated_gib_max': max(r['peak_allocated_bytes'] for r in rows) / 2**30,
           'incremental_peak_gib_max': max(r['incremental_peak_bytes'] for r in rows) / 2**30,
           'base_allocated_gib': rows[0]['base_allocated_bytes'] / 2**30}
    if arm.startswith('stream_'):
        acts = [r['adaptive_actions'] for r in rows]
        accepted = [r for r in rows if not r['fallback']]
        early = [r for r in accepted if r['adaptive_actions'] < M]
        energy_first, margin_first, binding = [], [], {'energy': 0, 'margin': 0, 'both_same_prefix': 0, 'full_rank': 0}
        ratios, lowers, uppers, gains = [], [], [], []
        for r in accepted:
            rec = r['recovery']
            hist = rec['history']
            U, L = rec['objective'], rec['lower']
            ratios.append(U / L)
            lowers.append(L)
            uppers.append(U)
            gains.append(min(rec['response']) / U)
            e = first_prefix(hist, lambda h: h['lower'] > 0 and h['U'] <= (1 + EPS) * h['lower'])
            m = first_prefix(hist, lambda h: h['min_response'] >= MARGIN * h['U'])
            energy_first.append(e)
            margin_first.append(m)
            stop = rec['actions']
            if stop >= M and (e is None or e >= M) and (m is None or m >= M):
                binding['full_rank'] += 1
            elif e is not None and m is not None and e == m:
                binding['both_same_prefix'] += 1
            elif m is None or (e is not None and e < m):
                binding['margin'] += 1  # energy met first, margin condition decided the stop
            else:
                binding['energy'] += 1  # margin met first, energy condition decided the stop
        out.update({
            'actions_mean': st.mean(acts), 'actions_median': st.median(acts),
            'actions_min': min(acts), 'actions_max': max(acts), 'actions_total': sum(acts),
            'early_stops_below_40': len(early),
            'accepted_without_fallback': len(accepted),
            'final_U_over_L_max': max(ratios) if ratios else None,
            'final_lower_min': min(lowers) if lowers else None,
            'final_upper_min': min(uppers) if uppers else None,
            'final_margin_over_U_min': min(gains) if gains else None,
            'stop_decided_by': binding,
            'energy_condition_first_met_median': st.median([x for x in energy_first if x is not None]) if any(x is not None for x in energy_first) else None,
            'failed_query_records': sum(1 for r in rows for q in r.get('queries', []) if 'error' in q),
        })
    audits = []
    for r in rows:
        a = r.get('native_audit')
        if a:
            audits.append({'step': r['step'],
                           'objective_factor': a['native_objective_factor_upper'],
                           'emitted_objective_factor': a['native_emitted_objective_factor_upper'],
                           'normalized_gain': a['native_normalized_gain_lower'],
                           'direction_rel_error': a['native_direction_relative_error'],
                           'table_direction_rel_error': a['native_table_direction_relative_error'],
                           'head_rel_error': a['head_relative_error'],
                           'passed': a['passed']})
    out['audits'] = audits
    if 'final_test' in metrics:
        t = metrics['final_test']
        out['final_test'] = {k: t[k] for k in ('worst_accuracy', 'mean_accuracy', 'mean_balanced_accuracy',
                                               'worst_balanced_accuracy', 'mean_bce', 'worst_bce') if k in t}
    out['initial_model_sha256'] = metrics.get('initial_model_sha256')
    return out


if __name__ == '__main__':
    print(json.dumps([run_summary(p) for p in sys.argv[1:]], indent=1, allow_nan=False))
