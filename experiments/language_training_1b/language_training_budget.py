"""Prospective fixed 36-run budget gate; CPU-only, no quality-based choices.

The existing training pilot excludes model loading, evaluation, independent
native audits and persistence. Those costs MUST come from the bounded cost probe.
FA's 40 warm steps are inner simplex iterations in update zero, not 40 updates.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path

METHODS = ('direct', 'uniform', 'cagrad', 'fa', 'periodic', 'famo')
DEV_STEPS, REPORT_STEPS, WALL_LIMIT = 100, 200, 10800
MARGIN = 1.5
MODEL = 'EleutherAI/pythia-1b-deduped'
TIMINGS = ('setup', 'method_setup', 'validation', 'test', 'direct_audit',
           'latest_save', 'selected_save', 'metadata_publish', 'selected_load',
           'retirement', 'final_archive_verify', 'raw_pair_verify', 'event_write')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 * 2**20), b''): h.update(b)
    return h.hexdigest()


def require_drive(backup):
    mount = Path('/content/drive')
    home = mount / 'MyDrive'
    if not os.path.ismount(mount) or not home.is_dir():
        raise RuntimeError('Real mounted Drive/MyDrive required; refuse local-directory fallback')
    destination = Path(backup).resolve()
    if destination == home.resolve() or home.resolve() not in destination.parents:
        raise ValueError('Backup must be a distinct directory beneath mounted MyDrive')


def read(path):
    def pairs(items):
        value = {}
        for key, item in items:
            if key in value: raise ValueError('Duplicate JSON key: ' + key)
            value[key] = item
        return value
    return json.loads(Path(path).read_text(), object_pairs_hook=pairs,
                      parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))


def positive(value):
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError('Positive finite measurement required')
    return float(value)


def freeze(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n')


def verify_staged(root, manifest):
    root = Path(root)
    for entry in manifest['files']:
        name = entry['destination']
        if Path(name).name != name or sha(root / name) != entry['sha256']:
            raise ValueError('Staged audited source differs: ' + name)
    for entry in manifest.get('data_files', []):
        rel = Path(entry['destination'])
        if rel.is_absolute() or '..' in rel.parts or sha(root / rel) != entry['sha256']:
            raise ValueError('Staged data identity differs: ' + str(rel))
    return {entry['destination']: entry['sha256'] for entry in manifest['files']}


def select_pilot(pilot, expected_sources):
    if (pilot.get('complete') is not True or pilot.get('seed') != 49100 or
            pilot.get('lr') != 1e-4 or pilot.get('batch') != 32 or
            pilot.get('warmup_updates') != 10 or pilot.get('measured_updates') != 10 or
            pilot['model']['model'] != MODEL or 'A100' not in pilot.get('gpu', '')):
        raise ValueError('Complete matching A100 1B pilot required')
    # An independently frozen source map binds the older pilot lifecycle. The
    # campaign's separately audited persistence changes do not replace this pin.
    if not expected_sources or any(pilot['sources'].get(n) != h for n, h in expected_sources.items()):
        raise ValueError('Frozen pilot source provenance differs')
    if not {'language_training_preflight.py', 'language_training_methods.py',
            'language_training_rebuttal.py', 'language_baseline_v6.py'} <= set(expected_sources):
        raise ValueError('Incomplete pilot source provenance')
    eligible = {m: [] for m in METHODS}
    seen = set()
    for row in pilot['rows']:
        cfg = row['configuration']; method = cfg['method']
        if method not in METHODS: raise ValueError('Unexpected pilot method')
        key = json.dumps(cfg, sort_keys=True)
        if key in seen: raise ValueError('Duplicate pilot candidate')
        seen.add(key)
        if row.get('complete') is not True: continue
        updates = row['updates']
        if len(updates) != 20 or [u['step'] for u in updates] != list(range(20)):
            raise ValueError('Incomplete complete-update pilot coverage')
        if any(u.get('warmup') is not (i < 10) for i, u in enumerate(updates)):
            raise ValueError('Pilot warmup labels differ')
        times = [positive(u['total_update_and_preparation_seconds']) for u in updates]
        measured = sum(times[10:]) / 10
        if not math.isclose(measured, row['mean_update_and_preparation_seconds'], rel_tol=1e-10):
            raise ValueError('Pilot mean differs from complete update records')
        peak = max(u['peak_allocated_bytes'] for u in updates)
        if peak != row['peak_allocated_bytes']: raise ValueError('Pilot memory summary differs')
        if method == 'direct':
            audits = row['native_audits']
            if len(audits) != 2 or [a['after_updates'] for a in audits] != [1, 20] or any(a.get('passes_table_contract') is not True for a in audits):
                raise ValueError('Direct native-table pilot qualification differs')
        if peak <= 34 * 2**30: eligible[method].append((measured, key, row))
    selected = {}
    for method in METHODS:
        if not eligible[method]: raise ValueError('No qualified pilot candidate: ' + method)
        # Python min's stable input-order tie rule matches the frozen preflight.
        row = min(eligible[method], key=lambda x: x[0])[2]
        if pilot['selected'].get(method) != row['configuration']:
            raise ValueError('Pilot choice is not its fastest qualified candidate')
        selected[method] = row
    return selected


def project(pilot, pilot_sha, probe, probe_sha, expected_sources):
    selected = select_pilot(pilot, expected_sources)
    costs = {}
    for method, row in selected.items():
        values = [positive(x['total_update_and_preparation_seconds']) for x in row['updates']]
        # Max of complete 5-update cycles respects periodic refresh frequency;
        # the first block includes update-zero FA's full Gram + 40 inner steps.
        cycle = max(sum(values[s:s+5]) / 5 for s in (0, 5, 10, 15))
        startup = values[0]
        costs[method] = {'per_update_cycle_envelope_seconds': cycle,
                         'once_per_run_update_zero_seconds': startup,
                         'six_run_update_seconds': 6 * startup + (3*99 + 3*199) * cycle}
    updates = sum(c['six_run_update_seconds'] for c in costs.values())
    common = {'version': 'language_training_budget_v1', 'development_steps': DEV_STEPS,
              'reporting_steps': REPORT_STEPS, 'runs': 36, 'updates': 5400,
              'wall_limit_seconds': WALL_LIMIT, 'safety_factor': MARGIN,
              'pilot_sha256': pilot_sha, 'probe_sha256': probe_sha,
              'systems': {m: r['configuration'] for m, r in selected.items()},
              'checkpoint_backup': 'metadata', 'completed_checkpoint_policy': 'final-only',
              'update_projection': costs, 'projected_update_seconds_before_margin': updates,
              'FA_scope': '40 inner simplex iterations plus full Gram occur only at update zero; charged once in every run and conservatively included in cycle envelope.',
              'quality_selection': 'No validation/test quality values consumed by this budget/backend gate.'}
    if updates * MARGIN > WALL_LIMIT:
        return dict(common, outcome='budget_unaffordable', launch_authorized=False,
                    reason='Complete-update projection alone exceeds fixed three-hour allowance')
    if probe is None:
        return dict(common, outcome='cost_measurements_required', launch_authorized=False,
                    reason='Pilot excludes setup, audits, full evaluation, checkpointing, archival and retirement')
    if (probe.get('complete') is not True or probe.get('purpose') != 'cost_only' or
            probe.get('quality_values_retained') is not False or probe['pilot_sha256'] != pilot_sha or
            probe['model'] != pilot['model'] or probe['systems'] != common['systems'] or
            probe['data'] != pilot['data'] or probe['gpu'] != pilot['gpu'] or
            probe['torch'] != pilot['torch'] or probe['cuda'] != pilot['cuda'] or
            probe.get('storage') != {'checkpoint_backup': 'metadata', 'completed_checkpoints': 'final-only'}):
        raise ValueError('Completed matching overhead probe required')
    if set(probe.get('method_startup_update_seconds', {})) != set(METHODS):
        raise ValueError('All six measured method startups required')
    for method in METHODS:
        observed = positive(probe['method_startup_update_seconds'][method])
        extra = 6 * max(0., observed - costs[method]['once_per_run_update_zero_seconds'])
        costs[method]['six_run_update_seconds'] += extra
        costs[method]['cost_probe_startup_seconds'] = observed
        updates += extra
    common['projected_update_seconds_before_margin'] = updates
    t = {key: positive(probe['seconds'][key]) for key in TIMINGS}
    # 18 dev runs: 2 validation/latest snapshots, 1 selected save, 3 metadata
    # publications. 18 report: 3,2,4 respectively; 2 full test evaluations.
    counts = {'setup': 40, 'method_setup': 36, 'validation': 90, 'test': 36,
              'direct_audit': 18, 'latest_save': 90, 'selected_save': 54,
              'metadata_publish': 126, 'selected_load': 18, 'retirement': 36,
              'final_archive_verify': 54, 'raw_pair_verify': 36, 'event_write': 5400}
    # Extra setup charges cover campaign/analysis imports, source/data hashing.
    overhead = sum(counts[k] * t[k] for k in TIMINGS)
    probe_wall = positive(probe['total_process_seconds'])
    projected = MARGIN * (updates + overhead) + probe_wall
    sizes = probe['sizes']
    for k in ('latest_raw_bytes', 'selected_raw_bytes', 'latest_gzip_bytes', 'selected_gzip_bytes', 'max_update_json_bytes'):
        if type(sizes[k]) is not int or sizes[k] <= 0: raise ValueError('Measured checkpoint/trace sizes required')
    # Use measured raw sizes and a conservative gzip expansion bound; model
    # entropy may change across updates, so initialization compressibility is not
    # treated as an upper bound. All actual archive measurements remain recorded.
    cf = max(sizes['latest_gzip_bytes'], math.ceil(sizes['latest_raw_bytes'] * 1.002) + 2**20)
    cs = max(sizes['selected_gzip_bytes'], math.ceil(sizes['selected_raw_bytes'] * 1.002) + 2**20)
    raw_pair = sizes['latest_raw_bytes'] + sizes['selected_raw_bytes']
    vm_required = 36 * cf + cs + 2 * raw_pair + 10 * 2**30
    drive_required = 4 * 5400 * sizes['max_update_json_bytes'] + 36 * 10 * 2**20 + 256 * 2**20
    capacity = {'vm_required_bytes': vm_required, 'drive_required_bytes': drive_required,
                'vm_free_bytes': probe['vm_free_bytes'], 'drive_free_bytes': probe['drive_free_bytes'],
                'actual_probe_sizes': sizes, 'scope': 'All36 final archives, one transient selected gzip, overlapping raw pair, 10GiB VM reserve; no pending Trash cleanup credit.'}
    outcome = 'ready'
    if projected > WALL_LIMIT: outcome = 'budget_unaffordable'
    elif vm_required > probe['vm_free_bytes'] or drive_required > probe['drive_free_bytes']: outcome = 'storage_unaffordable'
    return dict(common, outcome=outcome, launch_authorized=outcome == 'ready',
                measured_overhead_seconds=t, operation_counts=counts,
                projected_overhead_seconds_before_margin=overhead,
                projected_total_seconds_including_probe=projected, capacity=capacity,
                interrupted_attempt_costs='Not assumed free. Controller stops this campaign at the fixed wall deadline; no automatic retry/reduced budget.')


def main():
    p = argparse.ArgumentParser(); p.add_argument('--pilot', type=Path, required=True)
    p.add_argument('--pilot-sources', type=Path, required=True); p.add_argument('--probe', type=Path)
    p.add_argument('--output', type=Path, required=True); a = p.parse_args()
    result = project(read(a.pilot), sha(a.pilot), read(a.probe) if a.probe else None,
                     sha(a.probe) if a.probe else None, read(a.pilot_sources))
    freeze(a.output, result); print(json.dumps(result, indent=2))

if __name__ == '__main__': main()
