"""Driver for the certified-step Pythia-1B study. Usage: certified_1b_driver.py --stage smoke|development|freeze|confirmatory

development: every certified rule gets the same search over eta0 in ETA_GRID on the development seed (validation only),
             plus the unprotected uniform control at the campaign's selected rate.
freeze:      selects each rule's eta0 by final development validation worst-task CE (ties: smaller eta0) and writes the
             frozen protocol, including the primary endpoint, before any confirmatory run.
confirmatory: every arm on the three reporting seeds, seed-major, at the frozen eta0. Held-out test only here.
"""
import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

STUDY = Path(__file__).resolve().parent
ROOT = STUDY.parent / 'language_training_v2'
OUT = STUDY / 'runs'
CERTIFIED = ('uniform_certified', 'direct_certified', 'auditfirst_certified', 'famo_certified')
ARMS = ('uniform_fixed',) + CERTIFIED
ETA_GRID = (1e-3, 3e-3, 1e-2)
UNIFORM_FIXED_ETA = 1e-3  # the campaign's selected plain-SGD rate for uniform
DEV_SEED, SEEDS = 49100, (49201, 49202, 49203)
DEV_STEPS, CONF_STEPS = 100, 200


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def run(arm, seed, phase, eta0, steps, validation_every, protocol=None):
    label = '%s_%s_seed%d_eta%g' % (phase, arm, seed, eta0)
    out = OUT / label
    if (out / 'metrics.json').exists():
        return label
    if (out / 'failure.json').exists():
        raise RuntimeError('Retained failure, diagnose before rerunning: ' + label)
    cmd = [sys.executable, '-u', str(STUDY / 'certified_language_1b.py'), '--root', str(ROOT), '--study', str(STUDY), '--arm', arm,
           '--seed', str(seed), '--phase', phase, '--eta0', repr(eta0), '--steps', str(steps), '--validation-every', str(validation_every),
           '--out', str(out)]
    if protocol:
        cmd += ['--protocol', str(protocol)]
    print(json.dumps({'start': label, 'utc_seconds': time.time()}), flush=True)
    log = (STUDY / 'logs' / (label + '.log'))
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open('w') as f:
        rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT).returncode
    if rc or not (out / 'metrics.json').exists():
        raise RuntimeError('Run failed: %s (return code %d)' % (label, rc))
    print(json.dumps({'done': label, 'utc_seconds': time.time()}), flush=True)
    return label


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--stage', choices=['smoke', 'development', 'freeze', 'confirmatory'], required=True)
    a = p.parse_args()
    if a.stage == 'smoke':
        for arm in ARMS:
            run(arm, DEV_SEED, 'smoke', ETA_GRID[0], 3, 3)
    elif a.stage == 'development':
        run('uniform_fixed', DEV_SEED, 'development', UNIFORM_FIXED_ETA, DEV_STEPS, DEV_STEPS)
        for eta0 in ETA_GRID:
            for arm in CERTIFIED:
                run(arm, DEV_SEED, 'development', eta0, DEV_STEPS, DEV_STEPS)
        (STUDY / 'development_complete.json').write_text(json.dumps({'complete': True, 'utc_seconds': time.time()}))
    elif a.stage == 'freeze':
        if (STUDY / 'protocol.json').exists():
            raise RuntimeError('Protocol already frozen')
        candidates, eta0 = {}, {'uniform_fixed': UNIFORM_FIXED_ETA}
        for arm in CERTIFIED:
            rows = []
            for e in ETA_GRID:
                m = json.loads((OUT / ('development_%s_seed%d_eta%g' % (arm, DEV_SEED, e)) / 'metrics.json').read_text())
                rows.append({'eta0': e, 'final_validation_worst_ce': m['final_validation']['worst_ce']})
            candidates[arm] = rows
            eta0[arm] = min(rows, key=lambda r: (r['final_validation_worst_ce'], r['eta0']))['eta0']
        from certified_language_1b import STUDY_FILES, CAMPAIGN_FILES
        sources = {n: sha(STUDY / n) for n in STUDY_FILES}
        sources.update({n: sha(ROOT / n) for n in CAMPAIGN_FILES})
        protocol = {
            'version': 'certified_1b_v1', 'arms': list(ARMS), 'eta0': eta0, 'eta_grid': list(ETA_GRID), 'candidates': candidates,
            'development_seed': DEV_SEED, 'development_steps': DEV_STEPS, 'confirmatory_seeds': list(SEEDS),
            'confirmatory_steps': CONF_STEPS, 'validation_every': 100, 'c': .5, 'halvings': 20, 'sources': sources,
            'selection': 'per certified rule, minimum final development validation worst-task answer CE over eta0; ties to the smaller eta0; no test access',
            'primary_endpoint': 'final held-out worst-task answer CE, paired by seed: auditfirst_certified minus uniform_certified',
            'secondary': ['direct_certified minus uniform_certified and famo_certified minus uniform_certified on the same endpoint',
                          'refusals (not_certified) and holds (no_admissible_step) per certified arm',
                          'updates raising some task minibatch loss: uniform_fixed against every certified arm (0 by construction)',
                          'each certified arm against the campaign unprotected arms of the same seed and batch sequence (descriptive)',
                          'accepted step sizes, seconds per update, peak memory, native audits'],
            'scope': 'plain-SGD same-minibatch guarantee in floating point at Pythia-1B; no momentum, population or held-out guarantee',
            'utc_seconds': time.time()}
        (STUDY / 'protocol.json').write_text(json.dumps(protocol, indent=1))
        print(json.dumps({'frozen': eta0, 'protocol_sha256': sha(STUDY / 'protocol.json')}), flush=True)
    else:
        protocol = STUDY / 'protocol.json'
        P = json.loads(protocol.read_text())
        for seed in P['confirmatory_seeds']:
            for arm in P['arms']:
                run(arm, seed, 'confirmatory', P['eta0'][arm], P['confirmatory_steps'], P['validation_every'], protocol)
        (STUDY / 'confirmatory_complete.json').write_text(json.dumps({'complete': True, 'protocol_sha256': sha(protocol), 'utc_seconds': time.time()}))


if __name__ == '__main__':
    main()
