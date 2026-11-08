"""Prospective pilot wall observer, deterministic freeze, and reporting controller.

No work occurs on import. Run only after independent source/CPU audit. The
scientific driver and queued pilot controller remain unchanged. Metadata-only
backup cannot recover model bytes after VM loss.
"""
import argparse
import copy
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import traceback
import uuid
from sharp_training_analysis import (ARMS, SEEDS, analyze, digest, file, number,
    population, read, require, sha, validate_protocol, validate_run)

BASE = Path(__file__).resolve().parent
CONTROL_FILES = ['sharp_training_analysis.py', 'sharp_training_reporting.py', 'sharp_training_reporting_plan.json']


@contextmanager
def defer_signals():
    previous = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGINT, signal.SIGTERM})
    try: yield previous
    finally: signal.pthread_sigmask(signal.SIG_SETMASK, previous)


def write(path, value, immutable=False):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    if immutable and path.exists():
        require(read(path) == value, 'Immutable control record changed: ' + str(path)); return
    fd, name = tempfile.mkstemp(prefix=path.name + '.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as handle:
            json.dump(value, handle, indent=2, allow_nan=False); handle.write('\n'); handle.flush(); os.fsync(handle.fileno())
        os.replace(name, path)
        fd = os.open(path.parent, os.O_RDONLY)
        try: os.fsync(fd)
        finally: os.close(fd)
    finally:
        if Path(name).exists(): Path(name).unlink()


def boot_seconds():
    return time.clock_gettime(time.CLOCK_BOOTTIME)


def process(pid, proc=Path('/proc')):
    try:
        data = (proc / str(pid) / 'stat').read_text().rsplit(')', 1)[1].split()
        command = (proc / str(pid) / 'cmdline').read_bytes().split(b'\0')
        return {'pid': int(pid), 'parent': int(data[1]), 'state': data[0], 'start_ticks': int(data[19]),
                'command': [os.fsdecode(part) for part in command if part]}
    except (FileNotFoundError, ProcessLookupError):
        return None


def processes():
    return [p for path in Path('/proc').iterdir() if path.name.isdigit() and (p := process(path.name)) is not None]


def same_process(record):
    current = process(record['pid'])
    return current is not None and current['start_ticks'] == record['start_ticks'] and current['command'] == record['command'] and current['state'] != 'Z'


def controller_identity(source_root):
    return {name: sha(source_root / name) for name in CONTROL_FILES}


def checked_plan(plan_path, source_root):
    plan = read(plan_path)
    require(plan['arms'] == ARMS and plan['pilot_seeds'] == [61900] and plan['pilot_updates'] == 100
            and plan['reporting_seeds'] == SEEDS and plan['reporting_updates'] == 300 and plan['reporting_runs'] == 12,
            'Control plan population changed')
    require(plan['total_reporting_budget_seconds'] == 10800 and plan['reserve_minimum_seconds'] == 1800
            and plan['reserve_fraction_of_projection'] == .25, 'Control cost rule changed')
    require(sha(source_root / 'sharp_training.py') == plan['scientific_driver_sha256'], 'Scientific driver changed')
    require(sha(source_root / 'sharp_training_pilot_interlude.py') == plan['pilot_controller_sha256'], 'Pilot controller changed')
    return plan


def expected_pilot_command(source_root, celeba, arm, python):
    return [python, '-u', str(source_root / 'sharp_training.py'), '--protocol', str(source_root / 'sharp_training_pilot_protocol.json'),
            '--phase', 'pilot', '--arm', arm, '--seed', '61900', '--data', str(celeba / 'data'),
            '--selection', str(celeba / 'selected_learning_rates.json'), '--out', str(source_root / 'pilot' / arm)]


def observe_pilot(source_root, celeba, pilot_pid, plan, state_path):
    require(not state_path.exists(), 'Observer evidence already exists; do not overwrite/re-observe')
    parent = process(pilot_pid)
    require(parent is not None and str(source_root / 'sharp_training_pilot_interlude.py') in parent['command'], 'Pilot controller identity')
    parent['executable'] = os.readlink(Path('/proc') / str(pilot_pid) / 'exe')
    require(not any((source_root / 'pilot').glob('*/attempts/*.json')), 'Observer started after a pilot attempt; full coverage cannot be established')
    protocol_path = source_root / 'sharp_training_pilot_protocol.json'
    protocol_sha = sha(protocol_path); ticks = os.sysconf('SC_CLK_TCK')
    state = {'complete': False, 'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
             'parent': parent, 'protocol_sha256': protocol_sha, 'plan_sha256': sha(source_root / 'sharp_training_reporting_plan.json'),
             'controller_sha256': plan['pilot_controller_sha256'], 'clock_ticks_per_second': ticks,
             'control_sources': controller_identity(source_root),
             'observer_started_boot_seconds': boot_seconds(), 'records': []}
    write(state_path, state)
    while True:
        now = boot_seconds(); active = processes()
        for child in active:
            if child['parent'] != pilot_pid or str(source_root / 'sharp_training.py') not in child['command']:
                continue
            try: child_executable = os.readlink(Path('/proc') / str(child['pid']) / 'exe')
            except FileNotFoundError: continue  # Reaped between procfs snapshots; tracked exit is handled below.
            require(child_executable == parent['executable'], 'Pilot child interpreter differs')
            matches = [arm for arm in ARMS if child['command'] == expected_pilot_command(source_root, celeba, arm, child['command'][0])]
            require(len(matches) == 1, 'Unexpected pilot child command')
            arm = matches[0]
            existing = [r for r in state['records'] if r['process']['pid'] == child['pid'] and r['process']['start_ticks'] == child['start_ticks']]
            if not existing:
                require(not any(r['arm'] == arm for r in state['records']), 'Repeated pilot arm/attempt')
                require(child['start_ticks'] / ticks >= state['observer_started_boot_seconds'] - 1. / ticks, 'Unobserved pilot process birth')
                state['records'].append({'arm': arm, 'process': child, 'first_seen_boot_seconds': boot_seconds(),
                                         'kernel_start_boot_seconds': child['start_ticks'] / ticks, 'complete': False})
        for record in state['records']:
            if not record['complete'] and not same_process(record['process']):
                record.update(complete=True, exit_observed_boot_seconds=boot_seconds())
                record['process_wall_upper_seconds'] = record['exit_observed_boot_seconds'] - record['kernel_start_boot_seconds']
        write(state_path, state)
        require(sha(protocol_path) == protocol_sha and sha(source_root / 'sharp_training_pilot_interlude.py') == plan['pilot_controller_sha256'], 'Pilot source/protocol changed')
        require(not (source_root / 'pilot_interlude_failed.txt').exists(), 'Pilot controller failed')
        if not same_process(parent):
            break
        time.sleep(.1)
    require(len(state['records']) == 4 and {r['arm'] for r in state['records']} == set(ARMS)
            and all(r['complete'] for r in state['records']), 'Missed/incomplete pilot subprocess coverage')
    marker = read(source_root / 'pilot_complete.json')
    require(marker['complete'] is True and marker['protocol_sha256'] == protocol_sha, 'Pilot terminal marker')
    state.update(complete=True, pilot_complete_sha256=sha(source_root / 'pilot_complete.json'))
    write(state_path, state)
    return state


def freeze(source_root, plan_path, observer_path):
    plan = checked_plan(plan_path, source_root)
    path = source_root / 'sharp_training_pilot_protocol.json'; pilot = read(path)
    validate_protocol(pilot, source_root)
    require(pilot['enabled_phases'] == ['pilot'] and pilot['reporting_budget_frozen'] is False, 'Not the frozen pilot protocol')
    observation = read(observer_path); marker_path = source_root / 'pilot_complete.json'; marker = read(marker_path)
    require(observation['complete'] is True and observation['protocol_sha256'] == sha(path)
            and observation['controller_sha256'] == plan['pilot_controller_sha256']
            and observation['plan_sha256'] == sha(plan_path)
            and observation['pilot_complete_sha256'] == sha(marker_path), 'Observer evidence identity')
    require(observation['control_sources'] == controller_identity(source_root), 'Observer control source changed')
    require(marker['complete'] is True and marker['protocol_sha256'] == sha(path), 'Pilot terminal marker differs')
    records = population(source_root / 'pilot', pilot, sha(path), 'pilot')
    require(marker['records'] == [{k: r[k] for k in ('arm', 'metrics_sha256', 'initial_model_sha256')} for r in records], 'Pilot marker population/hashes')
    watched = observation['records']
    require(len(watched) == 4 and {r['arm'] for r in watched} == set(ARMS) and all(r['complete'] is True for r in watched), 'Pilot observation coverage')
    walls = {}
    for record in records:
        require(record['unrecorded_interrupted_tails'] == 0 and record['attempt_count'] == 1
                and record['uncommitted_recorded_update_seconds'] == 0, 'Interrupted/repeated pilot attempts cannot freeze cost')
        watched_run = next(r for r in watched if r['arm'] == record['arm'])
        child = watched_run['process']; command = child['command']
        require(child['parent'] == observation['parent']['pid'] and len(command) == 17
                and command[1:9] == ['-u', str(source_root / 'sharp_training.py'), '--protocol', str(path),
                                      '--phase', 'pilot', '--arm', record['arm']]
                and command[9:11] == ['--seed', '61900']
                and command[15:] == ['--out', str(source_root / 'pilot' / record['arm'])], 'Observed worker command differs')
        wall = number(watched_run['process_wall_upper_seconds'])
        require(wall >= record['recorded_process_seconds'] and wall > 0, 'Full-process wall does not cover recorded scientific work')
        require(watched_run['kernel_start_boot_seconds'] >= observation['observer_started_boot_seconds'] - 1. / observation['clock_ticks_per_second'], 'Pilot birth coverage differs')
        require(watched_run['kernel_start_boot_seconds'] <= watched_run['first_seen_boot_seconds'] <= watched_run['exit_observed_boot_seconds'], 'Observation chronology differs')
        require(abs(watched_run['kernel_start_boot_seconds'] - child['start_ticks'] / observation['clock_ticks_per_second']) < 1e-7, 'Kernel start identity differs')
        require(abs(wall - watched_run['exit_observed_boot_seconds'] + watched_run['kernel_start_boot_seconds']) < 1e-7, 'Observer wall differs')
        walls[record['arm']] = wall
    raw = 12 * 3 * max(walls.values()); reserve = max(1800., .25 * raw); total = raw + reserve
    require(total <= 10800, 'Conservative reporting projection exceeds3A100hours; reporting remains disabled')
    evidence = {'pilot_protocol_sha256': sha(path), 'pilot_complete_sha256': sha(marker_path),
                'observer_sha256': sha(observer_path), 'plan_sha256': sha(plan_path),
                'control_sources': controller_identity(source_root), 'pilot_diagnostics': records,
                'pilot_full_process_upper_seconds': walls, 'extrapolated_seconds': raw,
                'reserve_seconds': reserve, 'conservative_reporting_seconds': total,
                'budget_seconds': 10800, 'quality_values_used': False}
    protocol = copy.deepcopy(pilot)
    protocol.update(status='reporting_frozen_after_complete_pilot_correctness_and_cost_gate', enabled_phases=['reporting'],
                    reporting_budget_frozen=True, reporting_freeze=evidence)
    target = source_root / 'sharp_training_reporting_protocol.json'
    write(target, protocol, immutable=True)
    return target, protocol


def metadata_backup(source_root, destination):
    require(destination.parent.is_dir(), 'Metadata backup parent is unavailable')
    destination.mkdir(exist_ok=True)
    candidates = []
    for directory in ('pilot', 'reporting', 'reporting_control'):
        root = source_root / directory
        candidates.extend(p for p in root.rglob('*') if p.is_file() and p.suffix in ('.json', '.log', '.txt'))
    candidates.extend(p for p in source_root.glob('sharp_training*.json'))
    candidates.extend(source_root / name for name in ('pilot_complete.json', 'reporting_complete.json', 'sharp_reporting_outcome.json') if (source_root / name).exists())
    candidates.sort(key=lambda p: (p.name == 'reporting_complete.json', str(p)))
    for source in candidates:
        relative = source.relative_to(source_root); target = file(destination, relative)
        require(not source.is_symlink(), 'Metadata source symlink')
        expected = sha(source)
        if target.exists() and sha(target) == expected: continue
        target.parent.mkdir(parents=True, exist_ok=True)
        fd, name = tempfile.mkstemp(prefix=target.name + '.', suffix='.tmp', dir=target.parent); os.close(fd)
        try:
            shutil.copy2(source, name)
            require(sha(name) == expected, 'Metadata changed during copy')
            with open(name, 'rb') as stream: os.fsync(stream.fileno())
            os.replace(name, target)
        finally:
            if Path(name).exists(): Path(name).unlink()


def blockers(workspace, own_pid, queue_script):
    result = []
    for p in processes():
        if p['pid'] == own_pid or p['state'] == 'Z': continue
        scripts = [Path(token) for token in p['command'] if token.endswith('.py') and str(workspace) + '/' in token]
        downstream = workspace / 'language_training_v1/language_training_queue.py'
        if any(script not in (queue_script, downstream) and any(word in script.name for word in ('queue', 'interlude', 'waiter', 'campaign', 'reporting')) for script in scripts):
            result.append(p)
    return result


def gpu_pids():
    output = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True, timeout=10)
    return {int(line.strip()) for line in output.splitlines() if line.strip()}


def descendant_pids(pid):
    active = processes(); allowed = {pid}
    while True:
        expanded = allowed | {p['pid'] for p in active if p['parent'] in allowed}
        if expanded == allowed: return allowed
        allowed = expanded


def stop_child(child):
    if child is None or child.poll() is not None: return
    child.terminate()
    try: child.wait(timeout=30)
    except subprocess.TimeoutExpired:
        child.kill(); child.wait(timeout=30)


def run_reporting(source_root, celeba, workspace, backup, protocol_path, resume=False):
    control = source_root / 'reporting_control'; control.mkdir(exist_ok=True)
    protocol = read(protocol_path); identity = {'protocol_sha256': sha(protocol_path), 'control_sources': controller_identity(source_root)}
    write(control / 'identity.json', identity, immutable=True)
    previous = sorted(control.glob('attempt_*.json'))
    require(resume or not previous, 'Prior controller attempt exists; explicit --resume required')
    # Unobserved killed-controller tails cannot be converted into a cost claim.
    for path in previous:
        old = read(path)
        require(old['complete'] or old.get('interrupted') is True, 'Prior controller has an unmeasured tail; inspect before proceeding')
        require(not old.get('error') or old.get('interrupted') is True, 'Prior controller failure is quarantined')
    attempt_path = control / f'attempt_{len(previous)+1:04d}.json'
    attempt = {'complete': False, 'started_utc_seconds': time.time(), 'identity': identity, 'children': []}
    write(attempt_path, attempt)
    queue_script = celeba / 'celeba_queue.py'; queue = None; paused = False; child = None; child_record = None; active_start = None
    prior_active = sum(read(p).get('control_active_seconds', 0.) for p in previous)
    def backup_now(required=False):
        try: metadata_backup(source_root, backup)
        except InterruptedError: raise
        except Exception:
            write(control / ('backup_error_' + uuid.uuid4().hex + '.json'), {'error': traceback.format_exc(), 'utc_seconds': time.time()})
            if required: raise
    try:
        while blockers(workspace, os.getpid(), queue_script): time.sleep(2)
        candidates = [p for p in processes() if str(queue_script) in p['command'] and p['state'] != 'Z']
        require(len(candidates) <= 1, 'Multiple CelebA launchers')
        if candidates:
            queue = candidates[0]; require(queue['state'] not in ('T', 't'), 'CelebA launcher already stopped by another owner')
            write(control / 'launcher_lease.json', {'queue': queue, 'owner_pid': os.getpid(), 'active': True})
            with defer_signals():
                os.kill(queue['pid'], signal.SIGSTOP); paused = True
            deadline = time.monotonic() + 10
            while same_process(queue) and process(queue['pid'])['state'] not in ('T', 't'):
                require(time.monotonic() < deadline, 'Launcher stop confirmation timeout'); time.sleep(.05)
            require(same_process(queue), 'Launcher exited during stop')
        idle = 0; deadline = time.monotonic() + 1800
        while idle < 3:
            require(time.monotonic() < deadline, 'No clean GPU boundary within30minutes')
            require(not blockers(workspace, os.getpid(), queue_script), 'New competing campaign controller')
            if queue: require(same_process(queue) and process(queue['pid'])['state'] in ('T', 't'), 'Launcher stop lost')
            workers = [p for p in processes() if str(celeba / 'celeba_training.py') in p['command'] and p['state'] != 'Z']
            idle = idle + 1 if not workers and not gpu_pids() else 0
            time.sleep(2)
        require(not (celeba / 'campaign_failed.txt').exists(), 'CelebA failure needs inspection')
        active_start = time.monotonic()
        backup_now(True)
        spent = sum(c['wall_seconds'] for p in previous for c in read(p)['children'])
        for seed in SEEDS:
            order = ARMS[seed % 4:] + ARMS[:seed % 4]
            for arm in order:
                require(controller_identity(source_root) == identity['control_sources'] and sha(protocol_path) == identity['protocol_sha256'], 'Frozen control/source changed')
                validate_protocol(protocol, source_root)
                out = source_root / 'reporting' / f'seed{seed}' / arm
                if (out / 'metrics.json').exists() and read(out / 'metrics.json').get('complete') is True:
                    validate_run(out, protocol, identity['protocol_sha256'], 'reporting', arm, seed); continue
                require(not (out / 'failure.json').exists(), 'Scientific run failure is quarantined')
                require(not gpu_pids() and not blockers(workspace, os.getpid(), queue_script), 'Competing GPU/campaign before launch')
                remaining = 10800 - prior_active - (time.monotonic() - active_start)
                require(remaining > 0, 'Reporting runtime budget exhausted; preserve incomplete population')
                command = [sys.executable, '-u', str(source_root / 'sharp_training.py'), '--protocol', str(protocol_path),
                           '--phase', 'reporting', '--arm', arm, '--seed', str(seed), '--data', str(celeba / 'data'),
                           '--selection', str(celeba / 'selected_learning_rates.json'), '--out', str(out)]
                child_record = {'seed': seed, 'arm': arm, 'command': command, 'complete': False, 'start_utc_seconds': time.time()}
                attempt['children'].append(child_record); write(attempt_path, attempt)
                log = control / f'seed{seed}_{arm}.log'; start = time.monotonic()
                with log.open('a') as stream:
                    with defer_signals() as inherited_mask:
                        child = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                            preexec_fn=lambda: signal.pthread_sigmask(signal.SIG_SETMASK, inherited_mask))
                        child_record['pid'] = child.pid; write(attempt_path, attempt)
                    next_backup = time.monotonic() + 30
                    while child.poll() is None:
                        require(time.monotonic() - start < remaining, 'Reporting runtime budget exhausted')
                        require(not blockers(workspace, os.getpid(), queue_script), 'Competing campaign appeared')
                        require(not (gpu_pids() - descendant_pids(child.pid)), 'Competing GPU worker appeared')
                        if queue: require(same_process(queue) and process(queue['pid'])['state'] in ('T', 't'), 'CelebA launcher resumed unexpectedly')
                        if time.monotonic() >= next_backup: backup_now(); next_backup = time.monotonic() + 30
                        time.sleep(1)
                child_record.update(complete=True, returncode=child.returncode, wall_seconds=time.monotonic()-start, end_utc_seconds=time.time())
                spent += child_record['wall_seconds']; write(attempt_path, attempt)
                require(child.returncode == 0, 'Reporting child failed; no cell omitted')
                validate_run(out, protocol, identity['protocol_sha256'], 'reporting', arm, seed)
                backup_now(True); child = None; child_record = None
        attempt.update(complete=True, control_active_seconds=time.monotonic()-active_start, finished_utc_seconds=time.time()); write(attempt_path, attempt)
        result = analyze(source_root, protocol_path, source_root, completed_marker=False)
        require(prior_active + time.monotonic() - active_start <= 10800, 'Reporting active wall budget exhausted during validation')
        write(source_root / 'sharp_training_reporting_analysis.json', result)
        backup_now(True)
        require(prior_active + time.monotonic() - active_start <= 10800, 'Reporting active wall budget exhausted during persistence')
        marker = {'complete': True, 'protocol_sha256': sha(protocol_path), 'analysis_sha256': sha(source_root / 'sharp_training_reporting_analysis.json'),
                  'runs': [{k: r[k] for k in ('seed', 'arm', 'metrics_sha256', 'latest_checkpoint_sha256')} for r in result['runs']]}
        write(source_root / 'reporting_complete.json', marker, immutable=True); backup_now(True)
    except BaseException as error:
        stop_child(child)
        if child_record is not None and not child_record['complete']:
            child_record.update(complete=True, returncode=child.returncode if child is not None else None,
                                launch_failed=child is None, wall_seconds=time.monotonic()-start,
                                end_utc_seconds=time.time(), interrupted=isinstance(error, (KeyboardInterrupt, InterruptedError)))
        attempt.update(complete=False, interrupted=isinstance(error, (KeyboardInterrupt, InterruptedError)), error=traceback.format_exc(), finished_utc_seconds=time.time(),
                       control_active_seconds=time.monotonic()-active_start if active_start is not None else 0.)
        write(attempt_path, attempt); raise
    finally:
        try: stop_child(child)
        finally:
            over_budget = False
            try:
                # A handled pre-boundary interruption has exactly zero active
                # GPU-control time; an unhandled kill never reaches this receipt.
                attempt.update(control_active_seconds=time.monotonic()-active_start if active_start is not None else 0.,
                               control_wall_finalized=True)
                write(attempt_path, attempt)
                over_budget = prior_active + attempt['control_active_seconds'] > 10800
                if over_budget:
                    for directory in (source_root, backup):
                        marker = directory / 'reporting_complete.json'
                        if marker.exists(): marker.replace(directory / ('reporting_complete_overbudget_' + uuid.uuid4().hex + '.json'))
            finally:
                if paused and queue and same_process(queue):
                    if child is None or child.poll() is not None:
                        with defer_signals():
                            os.kill(queue['pid'], signal.SIGCONT)
                            write(control / 'launcher_lease.json', {'queue': queue, 'owner_pid': os.getpid(), 'active': False})
                    else:
                        write(control / 'launcher_held.json', {'queue': queue, 'child_pid': child.pid, 'reason': 'Own worker survived bounded termination'})
                backup_now()
            require(not over_budget, 'Final exclusive reporting wall exceeded3A100hours; completion retired')


def main():
    p = argparse.ArgumentParser(); p.add_argument('--mode', choices=['observe-freeze', 'run', 'observe-freeze-run'], required=True)
    p.add_argument('--source-root', type=Path, default=BASE); p.add_argument('--workspace', type=Path, required=True)
    p.add_argument('--celeba', type=Path, required=True); p.add_argument('--backup', type=Path, required=True)
    p.add_argument('--pilot-pid', type=int); p.add_argument('--resume', action='store_true'); a = p.parse_args()
    root = a.source_root.resolve(); root.mkdir(exist_ok=True)
    lock = (root / 'reporting_controller.lock').open('a'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    global_lock = (a.workspace / 'sharp_reporting_gpu_control.lock').open('a'); fcntl.flock(global_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    def interrupted(signum, frame): raise InterruptedError('Controller interrupted: ' + str(signum))
    signal.signal(signal.SIGTERM, interrupted); signal.signal(signal.SIGINT, interrupted)
    plan_path = root / 'sharp_training_reporting_plan.json'; plan = checked_plan(plan_path, root)
    observer = root / 'sharp_training_pilot_wall_observation.json'
    try:
        if a.mode.startswith('observe'):
            require(a.pilot_pid is not None, 'Pilot PID required before observation')
            observe_pilot(root, a.celeba, a.pilot_pid, plan, observer)
            protocol_path, _ = freeze(root, plan_path, observer)
        else:
            protocol_path, _ = freeze(root, plan_path, observer)
        metadata_backup(root, a.backup)
        if a.mode.endswith('run') or a.mode == 'run':
            run_reporting(root, a.celeba, a.workspace, a.backup, protocol_path, a.resume)
        if a.mode != 'observe-freeze':
            write(root / 'sharp_reporting_outcome.json', {'outcome': 'complete', 'terminal': True,
                  'safe_for_downstream': True, 'reporting_complete_sha256': sha(root / 'reporting_complete.json'),
                  'protocol_sha256': sha(protocol_path), 'utc_seconds': time.time()})
            metadata_backup(root, a.backup)
    except BaseException:
        error = traceback.format_exc()
        write(root / 'reporting_control' / ('entry_failure_' + uuid.uuid4().hex + '.json'), {'error': error, 'utc_seconds': time.time()})
        active = [p for p in processes() if str(root / 'sharp_training.py') in p['command'] and p['state'] != 'Z']
        safe = not active and not (root / 'reporting_control/launcher_held.json').exists()
        lease_path = root / 'reporting_control/launcher_lease.json'
        if lease_path.exists():
            lease = read(lease_path)
            if lease['active'] and same_process(lease['queue']): safe = False
        write(root / 'sharp_reporting_outcome.json', {'outcome': 'no_launch' if 'Conservative reporting projection exceeds3A100hours' in error else 'failed',
              'terminal': True, 'safe_for_downstream': safe, 'error': error, 'utc_seconds': time.time()})
        try: metadata_backup(root, a.backup)
        except Exception: pass
        raise


if __name__ == '__main__':
    main()
