"""Summarize the live query-rule study: per-rule medians over held-out states (9102-9105)."""
import json, statistics as st, sys
from pathlib import Path
d = json.loads(Path(sys.argv[1]).read_text())
assert d['complete'] and not d['failures'], 'incomplete or failed study'
rows = [r for r in d['rows'] if r['complete']]
assert len(rows) == 75, len(rows)
held = (9102, 9103, 9104, 9105)
out = {}
for rule, gamma in [('direct', None), ('completion', .05), ('gate', .05), ('completion', 1.0), ('gate', 1.0)]:
    rr = [r for r in rows if r['rule'] == rule and r['gamma'] == gamma]
    per_state = {}
    for s in held:
        x = [r for r in rr if r['seed'] == s]
        per_state[s] = {'median_seconds': st.median(r['seconds'] for r in x),
                        'actions': sorted({(r['recovery'] or {}).get('actions', 0) for r in x}),
                        'peak_gib': max(r['peak_allocated_bytes'] for r in x) / 2**30}
    out[f'{rule}@{gamma}'] = {
        'median_of_state_medians_seconds': st.median(v['median_seconds'] for v in per_state.values()),
        'actions_by_state': {s: v['actions'] for s, v in per_state.items()},
        'max_peak_gib': max(v['peak_gib'] for v in per_state.values()),
        'all_passed_fidelity': all(r['passed_fidelity'] for r in rr),
        'all_strict_descent': all(r['passed_strict_descent'] for r in rr),
        'max_objective_factor': max(r['native_objective_factor_upper'] for r in rr),
        'min_normalized_gain': min((r['normalized_gain_to_reference_primal'] or 0) for r in rr),
        'fallbacks': sum(1 for r in rr if 'fallback' in (r['recovery'] or {})),
        'per_state_median_seconds': {s: v['median_seconds'] for s, v in per_state.items()}}
base = out['completion@0.05']['per_state_median_seconds']
for k, v in out.items():
    if k.startswith('direct'): continue
    ratios = [v['per_state_median_seconds'][s] / base[s] for s in held]
    v['paired_time_ratio_to_completion@0.05_geomean'] = float(__import__('math').exp(sum(map(__import__('math').log, ratios)) / len(ratios)))
print(json.dumps(out, indent=1))
