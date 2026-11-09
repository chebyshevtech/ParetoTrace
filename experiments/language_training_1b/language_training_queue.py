"""Exclusive, prospective 36-run LM queue. No work occurs on import.

Waits for all named upstream terminal markers AND exact known controller/worker
paths to disappear. Never stops upstream jobs. No campaign starts without the
bounded cost probe and fixed100/200 three-hour affordability gate.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import traceback
from language_training_budget import read, sha, freeze, project, verify_staged, WALL_LIMIT, require_drive

REQUIRED = {'v6', 'sharp_pilot', 'sharp_reporting', 'celeba', 'qm9', 'qm9_waiter'}


def atomic(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.json.tmp')
    with temp.open('w') as f:
        json.dump(value, f, indent=2, allow_nan=False); f.flush(); os.fsync(f.fileno())
    temp.replace(path)


def copy_json(source, target):
    source, target = Path(source), Path(target); target.parent.mkdir(parents=True, exist_ok=True)
    digest = sha(source); temp = target.with_suffix('.json.tmp')
    shutil.copyfile(source, temp)
    if sha(temp) != digest: raise IOError('Numerical backup copy differs')
    temp.replace(target)


def command(pid):
    try: return (Path('/proc') / str(pid) / 'cmdline').read_bytes().split(b'\0')
    except FileNotFoundError: return []


def active(paths):
    needles = {str(Path(p)).encode() for p in paths}
    return [int(p.name) for p in Path('/proc').iterdir()
            if p.name.isdigit() and int(p.name) != os.getpid() and needles.intersection(command(p.name))]


def dependencies(plan):
    result = {}
    for item in plan['dependencies']:
        for failure in item.get('failure_files', []):
            if Path(failure).exists(): raise RuntimeError('Upstream failure requires inspection: ' + failure)
        path = Path(item['marker'])
        if not path.exists(): return None
        marker = read(path)
        if item['name'] == 'sharp_reporting':
            if marker.get('outcome') == 'failed': raise RuntimeError('Sharp reporting failed; diagnosis required')
            if marker.get('outcome') not in ('complete', 'no_launch'): return None
            if marker.get('terminal') is not True or marker.get('safe_for_downstream') is not True:
                raise RuntimeError('Sharp controller has not safely released its workers')
        elif marker.get('complete') is not True: raise RuntimeError('Invalid upstream completion: ' + str(path))
        for key, value in item.get('equals', {}).items():
            if marker.get(key) != value: raise RuntimeError('Upstream marker identity differs: ' + str(path))
        for key, count in item.get('lengths', {}).items():
            if len(marker.get(key, [])) != count: raise RuntimeError('Upstream run coverage differs: ' + str(path))
        for field, artifact in item.get('references', {}).items():
            if marker.get(field) != sha(artifact): raise RuntimeError('Upstream referenced identity differs: ' + str(path))
        result[item['name']] = {'marker': str(path), 'sha256': sha(path)}
    return result


def validate_plan(plan):
    if plan.get('enabled') is not True: raise ValueError('Queue plan is disabled pending source audit and explicit staging freeze')
    if (plan.get('development_steps'), plan.get('reporting_steps'), plan.get('wall_limit_seconds'),
        plan.get('checkpoint_backup'), plan.get('completed_checkpoints')) != (100, 200, 10800, 'metadata', 'final-only'):
        raise ValueError('Prospective fixed budget/storage policy differs')
    if {d['name'] for d in plan['dependencies']} != REQUIRED or len(plan['dependencies']) != len(REQUIRED):
        raise ValueError('All six upstream workflow barriers required')
    if not plan.get('known_gpu_scripts') or not plan.get('pilot_sources') or not plan.get('operation_sources'):
        raise ValueError('Explicit source and process pins required')
    locations = {d['name']: Path(d['marker']).parent for d in plan['dependencies']}
    required_scripts = {
        locations['v6']/'language_v6_interlude.py', locations['v6']/'language_baseline_v6.py',
        locations['v6']/'language_training_preflight.py',
        locations['sharp_pilot']/'sharp_training_pilot_interlude.py',
        locations['sharp_reporting']/'sharp_training_reporting.py', locations['sharp_reporting']/'sharp_training.py',
        locations['celeba']/'run_campaign.py', locations['celeba']/'celeba_queue.py', locations['celeba']/'celeba_training.py',
        locations['qm9']/'qm9_short_waiter.py', locations['qm9']/'qm9_short_campaign.py', locations['qm9']/'qm9_training_rebuttal.py',
        Path(plan['root'])/'language_campaign.py', Path(plan['root'])/'language_training_rebuttal.py',
        Path(plan['root'])/'language_training_cost_probe.py'}
    if not {str(p) for p in required_scripts} <= set(plan['known_gpu_scripts']):
        raise ValueError('Missing known upstream/owned GPU controller or worker path')
    for value in plan['known_gpu_scripts']:
        if not Path(value).is_absolute() or Path(value).suffix != '.py': raise ValueError('Exact absolute Python process paths required')
    root, backup = Path(plan['root']).resolve(), Path(plan['backup']).resolve()
    if root == backup or root in backup.parents or backup in root.parents: raise ValueError('VM and Drive destinations must be disjoint')


def verify(plan):
    require_drive(plan['backup'])
    root = Path(plan['root'])
    if sha(plan['staging_manifest']) != plan['staging_manifest_sha256']: raise ValueError('Staging manifest changed')
    pins = verify_staged(root, read(plan['staging_manifest']))
    for name, digest in plan['operation_sources'].items():
        if Path(name).name != name or sha(root / name) != digest: raise ValueError('Operational source changed: ' + name)
    for path, digest in plan.get('upstream_source_pins', {}).items():
        if sha(path) != digest: raise ValueError('Upstream source changed: ' + path)
    return pins


def idle(plan, timeout=300):
    deadline = time.monotonic() + timeout; samples = 0
    while samples < 3:
        if time.monotonic() > deadline: raise TimeoutError('No stable unoccupied GPU boundary')
        if dependencies(plan) is None: raise RuntimeError('Upstream terminal marker disappeared')
        busy = active(plan['known_gpu_scripts'])
        gpu = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True, timeout=10).strip()
        samples = samples + 1 if not busy and not gpu else 0
        time.sleep(2)


def group_alive(pgid):
    try: os.killpg(pgid, 0); return True
    except ProcessLookupError: return False


def stop_group(child):
    # Every child starts a fresh session, so only this controller's owned work is
    # signalled, including campaign subprocess descendants.
    for sig in (signal.SIGTERM, signal.SIGKILL):
        if not group_alive(child.pid): break
        try: os.killpg(child.pid, sig)
        except ProcessLookupError: break
        end = time.monotonic() + 30
        while group_alive(child.pid) and time.monotonic() < end:
            child.poll(); time.sleep(.2)
    child.poll()
    if group_alive(child.pid): raise RuntimeError('Owned worker group survived bounded termination: ' + str(child.pid))


def run_owned(args, log, deadline):
    child = None
    try:
        with Path(log).open('a') as f:
            child = subprocess.Popen(args, stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
            while child.poll() is None:
                if time.monotonic() > deadline: raise TimeoutError('Fixed cost/campaign wall allowance exhausted')
                time.sleep(1)
        if child.returncode: raise RuntimeError('Owned command failed: ' + str(args))
        # A successful parent must not leave a GPU child behind.
        if group_alive(child.pid): raise RuntimeError('Owned command left descendants running')
    finally:
        if child is not None and group_alive(child.pid): stop_group(child)


def main():
    p = argparse.ArgumentParser(); p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--plan-sha256', required=True); a = p.parse_args()
    if sha(a.plan) != a.plan_sha256: raise ValueError('Queue-plan pin differs')
    plan = read(a.plan); validate_plan(plan); pins = verify(plan)
    root, backup = Path(plan['root']), Path(plan['backup'])
    result_path = root / 'language_queue_outcome.json'
    if result_path.exists(): raise RuntimeError('Retain prior terminal queue outcome; no automatic replay')
    lock = (root/'language_queue.lock').open('a'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    def interrupted(signum, frame): raise InterruptedError('LM queue interrupted: ' + str(signum))
    signal.signal(signal.SIGTERM, interrupted); signal.signal(signal.SIGINT, interrupted)
    result = {'complete': False, 'plan_sha256': a.plan_sha256, 'outcome': 'waiting', 'started_utc_seconds': time.time()}
    def publish():
        atomic(result_path, result)
        try:
            require_drive(backup)
            copy_json(result_path, backup / result_path.name)
        except InterruptedError: raise
        except Exception: atomic(root / 'queue_backup_error.json', {'error': traceback.format_exc()})
    try:
        end = time.monotonic() + plan.get('wait_timeout_seconds', 86400)
        while True:
            if sha(a.plan) != a.plan_sha256: raise ValueError('Queue plan changed while waiting')
            done = dependencies(plan)
            if done is not None and not active(plan['known_gpu_scripts']): break
            if time.monotonic() > end: raise TimeoutError('Upstream completion wait expired')
            time.sleep(10)
        # Never hold the shared GPU lease while waiting for upstream workflows.
        # That would deadlock any upstream controller using the same lease.
        gpu_lock = Path(plan['gpu_lease']).open('a')
        fcntl.flock(gpu_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        verify(plan); idle(plan); result['upstream'] = done
        pilot_path = Path(plan['pilot']); pilot = read(pilot_path)
        # Copy the complete existing pilot unchanged into the campaign directory.
        target = root / 'training_preflight.json'
        if target.exists() and sha(target) != sha(pilot_path): raise ValueError('Campaign pilot already differs')
        if not target.exists(): copy_json(pilot_path, target)
        pilot_sources = root / 'budget_pilot_sources.json'
        if pilot_sources.exists():
            if read(pilot_sources) != plan['pilot_sources']: raise ValueError('Pilot source map changed')
        else: freeze(pilot_sources, plan['pilot_sources'])
        preliminary = project(pilot, sha(target), None, None, plan['pilot_sources'])
        if preliminary['outcome'] == 'budget_unaffordable':
            freeze(root / 'budget_freeze.json', preliminary)
            result.update(outcome='budget_unaffordable', budget=preliminary); publish(); return
        # Probe time is charged against the same three-hour total allowance.
        campaign_start = time.monotonic(); deadline = campaign_start + WALL_LIMIT
        python = str(root / 'venv/bin/python'); work = root / 'cost_probe_work'
        run_owned([python, str(root/'language_training_cost_probe.py'), '--root', str(root),
                   '--pilot', str(target), '--pilot-sources', str(pilot_sources),
                   '--manifest', plan['staging_manifest'], '--work', str(work),
                   '--backup', str(backup/'cost_probe'), '--drive-free-ceiling-bytes', str(plan['drive_free_ceiling_bytes'])],
                  root/'cost_probe.log', min(deadline, time.monotonic()+1200))
        probe_path = work/'cost_probe.json'; probe = read(probe_path)
        if probe['sources'] != pins: raise ValueError('Cost probe source pins differ')
        budget = project(pilot, sha(target), probe, sha(probe_path), plan['pilot_sources'])
        freeze(root/'budget_freeze.json', budget)
        require_drive(backup)
        copy_json(root/'budget_freeze.json', backup/'budget_freeze.json')
        if budget['launch_authorized'] is not True:
            result.update(outcome=budget['outcome'], budget=budget); publish(); return
        # Recheck actual capacity at the final decision, never pending Trash space.
        cap = budget['capacity']
        if (shutil.disk_usage(root).free < cap['vm_required_bytes'] or
            min(shutil.disk_usage(backup).free, plan['drive_free_ceiling_bytes']) < cap['drive_required_bytes']):
            raise RuntimeError('Capacity fell below frozen all-36 requirement')
        for stage in ('development', 'confirmatory'):
            verify(plan); idle(plan)
            run_owned([python, str(root/'language_campaign.py'), '--root', str(root), '--backup', str(backup),
                       '--stage', stage, '--development-steps', '100', '--reporting-steps', '200',
                       '--completed-checkpoints', 'final-only', '--checkpoint-backup', 'metadata'], root/(stage+'_campaign.log'), deadline)
            marker = read(root/(stage+'_complete.json'))
            if marker.get('complete') is not True or len(marker.get('runs', {})) != 18:
                raise RuntimeError('All18 stage runs required')
        verify(plan)
        run_owned([python, str(root/'analyze_language_training.py'), '--root', str(root),
                   '--output', str(root/'campaign_analysis.json')], root/'analysis.log', deadline)
        result.update(complete=True, outcome='complete', total_active_seconds=time.monotonic()-campaign_start,
                      analysis_sha256=sha(root/'campaign_analysis.json'), budget_sha256=sha(root/'budget_freeze.json'))
        publish()
    except BaseException:
        result.update(outcome='failed', error=traceback.format_exc(), ended_utc_seconds=time.time())
        publish(); raise

if __name__ == '__main__': main()
