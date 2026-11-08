"""Analysis of the certified-step study from certified_step_summary.json (prespecified endpoints only).

Usage: python3 analyze_certified_step.py certified_step_summary.json > certified_step_analysis.json
"""
import json
import statistics as st
import sys

ARMS = ('uniform_fixed', 'direct_fixed', 'uniform_certified', 'direct_certified', 'stream_certified', 'mgda_ub_certified')
SEEDS = (62001, 62002, 62003)


def main():
    runs = {(r['arm'], r['seed']): r for r in json.load(open(sys.argv[1]))['runs']}
    missing = [(a, s) for a in ARMS for s in SEEDS if (a, s) not in runs]
    if missing:
        raise SystemExit('missing runs: %s' % missing)
    per_arm = {}
    for a in ARMS:
        rs = [runs[(a, s)] for s in SEEDS]
        etas = [e for r in rs for e, k in zip(r['per_update']['accepted_eta'], r['per_update']['status']) if k in ('accepted', 'fixed')]
        per_arm[a] = {
            'updates': sum(r['updates'] for r in rs),
            'updates_raising_a_task': sum(r['updates_raising_a_task'] for r in rs),
            'status_counts': {k: sum(r['status_counts'][k] for r in rs) for k in rs[0]['status_counts']},
            'accepted_at_full_step': sum(r['accepted_at_full_step'] for r in rs),
            'accepted_eta_median': st.median(etas) if etas else None,
            'max_task_increase': max(r['max_task_increase'] for r in rs),
            'first_order_nonpositive_updates': sum(r['first_order_nonpositive_updates'] for r in rs),
            'trunk_nonpositive_updates': sum(r['trunk_nonpositive_updates'] for r in rs),
            'uniform_trunk_raises_a_task': sum(r['uniform_trunk_raises_a_task'] for r in rs),
            'uniform_trunk_checked': sum(r['uniform_trunk_checked'] for r in rs),
            'initial_train_worst_bce': [r['initial_train']['worst_bce'] for r in rs],
            'final_train_worst_bce': [r['final_train']['worst_bce'] for r in rs],
            'final_train_mean_bce': [r['final_train']['mean_bce'] for r in rs],
            'final_validation_worst_bce': [r['final_validation']['worst_bce'] for r in rs],
            'final_test_worst_bce': [r['final_test']['worst_bce'] for r in rs],
            'final_test_worst_accuracy': [r['final_test']['worst_accuracy'] for r in rs],
            'final_test_mean_accuracy': [r['final_test']['mean_accuracy'] for r in rs],
            'update_seconds_median': st.median(r['update_seconds_median'] for r in rs),
            'peak_gib_max': max(r['peak_gib_max'] for r in rs),
            'fallbacks': sum(r['fallbacks'] for r in rs),
            'audits': sum(r['audits'] for r in rs), 'audits_passed': sum(r['audits_passed'] for r in rs),
            'audit_max_trunk_error': max(r['audit_max_trunk_error'] for r in rs),
            'audit_max_head_error': max(r['audit_max_head_error'] for r in rs)}
        for k in ('final_train_worst_bce', 'final_train_mean_bce', 'final_validation_worst_bce', 'final_test_worst_bce',
                  'final_test_worst_accuracy', 'final_test_mean_accuracy'):
            per_arm[a][k + '_mean'] = st.mean(per_arm[a][k])

    def paired(a, b, key='final_train', sub='worst_bce'):
        d = [runs[(a, s)][key][sub] - runs[(b, s)][key][sub] for s in SEEDS]
        return {'per_seed': d, 'mean': st.mean(d), 'first_lower_on_seeds': sum(x < 0 for x in d)}

    cert_etas = [e for a in ARMS if a.endswith('_certified') for s in SEEDS
                 for e, k in zip(runs[(a, s)]['per_update']['accepted_eta'], runs[(a, s)]['per_update']['status']) if k == 'accepted']
    out = {'per_arm': per_arm,
           'certified_accepted': len(cert_etas), 'certified_accepted_eta_median': st.median(cert_etas),
           'certified_accepted_at_full_step': sum(e == 0.02 for e in cert_etas),
           'primary_direct_certified_minus_uniform_certified_train_worst_bce': paired('direct_certified', 'uniform_certified'),
           'secondary': {
               'stream_certified_minus_uniform_certified_train_worst_bce': paired('stream_certified', 'uniform_certified'),
               'mgda_ub_certified_minus_uniform_certified_train_worst_bce': paired('mgda_ub_certified', 'uniform_certified'),
               'direct_certified_minus_direct_fixed_train_worst_bce': paired('direct_certified', 'direct_fixed'),
               'uniform_certified_minus_uniform_fixed_train_worst_bce': paired('uniform_certified', 'uniform_fixed'),
               'stream_certified_minus_direct_certified_train_worst_bce': paired('stream_certified', 'direct_certified'),
               'direct_certified_minus_uniform_certified_test_worst_accuracy': paired('direct_certified', 'uniform_certified', 'final_test', 'worst_accuracy'),
               'uniform_certified_minus_uniform_fixed_test_worst_accuracy': paired('uniform_certified', 'uniform_fixed', 'final_test', 'worst_accuracy'),
               'direct_certified_minus_direct_fixed_test_worst_accuracy': paired('direct_certified', 'direct_fixed', 'final_test', 'worst_accuracy')},
           'same_initialization': len({runs[(a, s)]['initial_train']['worst_bce'] for a in ARMS for s in SEEDS}) == len(SEEDS)}
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
