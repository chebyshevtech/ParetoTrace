"""Event-level summary of the certified-step study (CPU only; reads metrics.json and events/*.json).

For each run: final metrics, status counts, updates whose executed step raised a task loss on its minibatch,
first-order responses along the trajectory (trunk only, and with private heads), accepted step sizes, cost
and audits, plus per-update scalars so every count can be re-derived.

Usage: python3 -I certified_step_summary.py <run_dir> [<run_dir> ...] > certified_step_summary.json
"""
import json
import statistics as st
import sys
from pathlib import Path

SCALARS = ('status', 'raised_tasks', 'max_task_change', 'accepted_eta', 'min_trunk_response', 'min_first_order_response',
           'nonpositive_trunk_tasks', 'nonpositive_first_order_tasks', 'worst_task_before', 'worst_task_after',
           'update_seconds', 'peak_allocated_bytes', 'fallback', 'adaptive_actions', 'loss_recompute_max_abs')


def summarize(run):
    run = Path(run)
    m = json.loads((run / 'metrics.json').read_text())
    rows = [json.loads((run / ref['file']).read_text()) for ref in m['events']]
    assert [r['step'] for r in rows] == list(range(len(rows)))
    per = {k: [r.get(k) for r in rows] for k in SCALARS}
    uniform_trunk = [min(r['uniform_trunk_response']) for r in rows if r.get('uniform_trunk_response') is not None]
    audits = [r['native_audit'] for r in rows if 'native_audit' in r]
    accepted = [r['accepted_eta'] for r in rows if r['status'] in ('accepted', 'fixed')]
    out = {'run': '/'.join(run.parts[-2:]), 'arm': m['config']['arm'], 'seed': m['config']['seed'], 'updates': len(rows),
           'initial_model_sha256': m['initial_model_sha256'],
           'initial_train': {k: m['initial_train'][k] for k in ('worst_bce', 'mean_bce', 'worst_accuracy', 'mean_accuracy')},
           'final_train': {k: m['final_train'][k] for k in ('worst_bce', 'mean_bce', 'worst_accuracy', 'mean_accuracy')},
           'final_validation': {k: m['final_validation'][k] for k in ('worst_bce', 'mean_bce', 'worst_accuracy', 'mean_accuracy')},
           'final_test': {k: m['final_test'][k] for k in ('worst_bce', 'mean_bce', 'worst_accuracy', 'mean_accuracy', 'mean_balanced_accuracy')} if 'final_test' in m else None,
           'status_counts': m['status_counts'], 'updates_raising_a_task': sum(r['raised_tasks'] > 0 for r in rows),
           'train_attributes_risen': sum(f > i for f, i in zip(m['final_train']['per_attribute_bce'], m['initial_train']['per_attribute_bce'])),
           'train_max_attribute_rise': max(f - i for f, i in zip(m['final_train']['per_attribute_bce'], m['initial_train']['per_attribute_bce'])),
           'validation_attributes_risen': sum(f > i for f, i in zip(m['final_validation']['per_attribute_bce'], m['initial_validation']['per_attribute_bce'])),
           'validation_max_attribute_rise': max(f - i for f, i in zip(m['final_validation']['per_attribute_bce'], m['initial_validation']['per_attribute_bce'])),
           'raised_tasks_median_when_raised': st.median([r['raised_tasks'] for r in rows if r['raised_tasks'] > 0]) if any(r['raised_tasks'] > 0 for r in rows) else 0,
           'max_task_increase': max(r['max_task_change'] for r in rows),
           'trunk_nonpositive_updates': sum(r['min_trunk_response'] <= 0 for r in rows),
           'first_order_nonpositive_updates': sum(r['min_first_order_response'] <= 0 for r in rows),
           'trunk_nonpositive_tasks_median': st.median(r['nonpositive_trunk_tasks'] for r in rows),
           'first_order_nonpositive_tasks_median': st.median(r['nonpositive_first_order_tasks'] for r in rows),
           'uniform_trunk_raises_a_task': sum(v < 0 for v in uniform_trunk), 'uniform_trunk_checked': len(uniform_trunk),
           'accepted_eta_median': st.median(accepted) if accepted else None,
           'accepted_at_full_step': sum(e == 0.02 for e in accepted),
           'update_seconds_median': st.median(r['update_seconds'] for r in rows),
           'peak_gib_max': max(r['peak_allocated_bytes'] for r in rows) / 2**30,
           'fallbacks': sum(bool(r['fallback']) for r in rows), 'audits': len(audits), 'audits_passed': sum(a['passed'] for a in audits),
           'audit_max_trunk_error': max(a['native_trunk_response_relative_error'] for a in audits) if audits else None,
           'audit_max_head_error': max(a['native_head_term_relative_error'] for a in audits) if audits else None,
           'loss_recompute_max_abs': max(r['loss_recompute_max_abs'] for r in rows),
           'audit_records': [{'step': r['step'], 'trunk_error': r['native_audit']['native_trunk_response_relative_error'],
                              'head_error': r['native_audit']['native_head_term_relative_error'],
                              'native_min_trunk': r['native_audit']['native_min_trunk_response'], 'event_min_trunk': r['min_trunk_response'],
                              'native_min_first_order': r['native_audit']['native_min_first_order_response'],
                              'event_min_first_order': r['min_first_order_response'], 'heads_frozen': r.get('heads') is not None,
                              'passed_as_recorded': r['native_audit']['passed']} for r in rows if 'native_audit' in r],
           'per_update': per}
    return out


def main():
    print(json.dumps({'scope': 'plain-SGD CelebA-40 certified-step study; same-minibatch rounded checks',
                      'runs': [summarize(p) for p in sys.argv[1:]]}))


if __name__ == '__main__':
    main()
