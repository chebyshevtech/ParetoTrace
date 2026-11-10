"""Driver for the certified-step Pythia-1B study.
Usage: certified_1b_driver.py --stage smoke|development|development_relaxed|freeze|confirmatory

development:         every certified rule gets the same search over eta0 in ETA_GRID at c = 1/2 (Equation 27) on the
                     development seed (validation only), plus the unprotected uniform control at the campaign's selected rate.
development_relaxed: amendment of 10 November 2026, decided after the c = 1/2 development runs (accepted steps 2 to 15 times
                     shorter than the unprotected step) and before any confirmatory run: the same eta0 search at the relaxed
                     constant c = 1e-4 that the paper's all-task Armijo paragraph allows.
freeze:              selects each rule's (c, eta0) by final development validation worst-task CE (ties: c = 1/2, then the
                     smaller eta0) and writes the frozen protocol, including the primary endpoint, before any confirmatory
                     run. The primary rule also keeps a companion at the other constant, at that constant's selected eta0.
development_extension: second amendment of 10 November 2026, before any confirmatory run: the development runs listed in
                     extension_plan.json (uploaded with the sources and hash-checked by the runner). A run that fails, for
                     example by diverging, is recorded and excluded from selection rather than stopping the stage.
confirmatory:        every arm on the three reporting seeds, seed-major, at the frozen (c, eta0), then the companion.
                     Held-out test only here.
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
C_EQ27, C_RELAXED = .5, 1e-4
C_GRID = (C_EQ27, C_RELAXED)
PRIMARY_RULE = 'auditfirst_certified'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def name(phase, arm, seed, eta0, c):
    return '%s_%s_seed%d_eta%g' % (phase, arm, seed, eta0) + ('' if c == C_EQ27 else '_c%g' % c)


def run(arm, seed, phase, eta0, steps, validation_every, c=C_EQ27, protocol=None, tolerate_failure=False):
    label = name(phase, arm, seed, eta0, c)
    out = OUT / label
    if (out / 'metrics.json').exists():
        return label
    if (out / 'failure.json').exists():
        if tolerate_failure:
            return None
        raise RuntimeError('Retained failure, diagnose before rerunning: ' + label)
    cmd = [sys.executable, '-u', str(STUDY / 'certified_language_1b.py'), '--root', str(ROOT), '--study', str(STUDY), '--arm', arm,
           '--seed', str(seed), '--phase', phase, '--eta0', repr(eta0), '--steps', str(steps), '--validation-every', str(validation_every),
           '--out', str(out), '--c', repr(c)]
    if protocol:
        cmd += ['--protocol', str(protocol)]
    print(json.dumps({'start': label, 'utc_seconds': time.time()}), flush=True)
    log = (STUDY / 'logs' / (label + '.log'))
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open('w') as f:
        rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT).returncode
    if rc or not (out / 'metrics.json').exists():
        if tolerate_failure and (out / 'failure.json').exists():
            print(json.dumps({'failed': label, 'utc_seconds': time.time()}), flush=True)
            return None
        raise RuntimeError('Run failed: %s (return code %d)' % (label, rc))
    print(json.dumps({'done': label, 'utc_seconds': time.time()}), flush=True)
    return label


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--stage', choices=['smoke', 'development', 'development_relaxed', 'development_extension', 'freeze', 'confirmatory'], required=True)
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
    elif a.stage == 'development_relaxed':
        if (STUDY / 'protocol.json').exists():
            raise RuntimeError('Protocol already frozen')
        for eta0 in ETA_GRID:
            for arm in CERTIFIED:
                run(arm, DEV_SEED, 'development', eta0, DEV_STEPS, DEV_STEPS, c=C_RELAXED)
        (STUDY / 'development_relaxed_complete.json').write_text(json.dumps({'complete': True, 'utc_seconds': time.time()}))
    elif a.stage == 'development_extension':
        if (STUDY / 'protocol.json').exists():
            raise RuntimeError('Protocol already frozen')
        plan = json.loads((STUDY / 'extension_plan.json').read_text())
        done, failed = [], []
        for x in plan['runs']:
            if x['arm'] not in ARMS:
                raise RuntimeError('Unknown arm in the extension plan: %s' % x['arm'])
            label = run(x['arm'], DEV_SEED, 'development', x['eta0'], DEV_STEPS, DEV_STEPS, c=x['c'], tolerate_failure=True)
            (done if label else failed).append(name('development', x['arm'], DEV_SEED, x['eta0'], x['c']))
        (STUDY / 'development_extension_complete.json').write_text(json.dumps(
            {'complete': True, 'plan_sha256': sha(STUDY / 'extension_plan.json'), 'completed': done, 'failed': failed, 'utc_seconds': time.time()}))
    elif a.stage == 'freeze':
        if (STUDY / 'protocol.json').exists():
            raise RuntimeError('Protocol already frozen')
        plan_file = STUDY / 'extension_plan.json'
        plan = json.loads(plan_file.read_text()) if plan_file.exists() else {'runs': []}
        if plan['runs'] and not (STUDY / 'development_extension_complete.json').exists():
            raise RuntimeError('Extension plan present but the extension stage has not completed')
        configs = {arm: [(c, e) for c in C_GRID for e in ETA_GRID] for arm in CERTIFIED}
        configs['uniform_fixed'] = [(C_EQ27, UNIFORM_FIXED_ETA)]
        for x in plan['runs']:
            if (x['c'], x['eta0']) not in configs[x['arm']]:
                configs[x['arm']].append((x['c'], x['eta0']))
        candidates, eta0, cs = {}, {}, {}
        for arm in ARMS:
            rows = []
            for c, e in configs[arm]:
                d = OUT / name('development', arm, DEV_SEED, e, c)
                if (d / 'metrics.json').exists():
                    m = json.loads((d / 'metrics.json').read_text())
                    if m['config']['c'] != c or m['config']['eta0'] != e or m['heldout_test_evaluated'] or m['steps'] != DEV_STEPS:
                        raise RuntimeError('Development record differs from its label: ' + d.name)
                    rows.append({'c': c, 'eta0': e, 'completed': True, 'final_validation_worst_ce': m['final_validation']['worst_ce'],
                                 'study_source_sha256': m['identity']['sources']['certified_language_1b.py']})
                elif (d / 'failure.json').exists():
                    err = json.loads((d / 'failure.json').read_text())['error'].strip().splitlines()[-1][:300]
                    rows.append({'c': c, 'eta0': e, 'completed': False, 'failure': err})
                else:
                    raise RuntimeError('Missing development record: ' + d.name)
            candidates[arm] = rows
            best = min((r for r in rows if r['completed']), key=lambda r: (r['final_validation_worst_ce'], r['c'] != C_EQ27, r['eta0']))
            eta0[arm], cs[arm] = best['eta0'], best['c']
        other = [r for r in candidates[PRIMARY_RULE] if r['c'] != cs[PRIMARY_RULE] and r['completed']]
        best_other = min(other, key=lambda r: (r['final_validation_worst_ce'], r['eta0']))
        companions = [{'arm': PRIMARY_RULE, 'c': best_other['c'], 'eta0': best_other['eta0'],
                       'role': 'the primary rule at the other sufficient-decrease constant, for the descriptive c contrast'}]
        from certified_language_1b import STUDY_FILES, CAMPAIGN_FILES
        sources = {n: sha(STUDY / n) for n in STUDY_FILES}
        sources.update({n: sha(ROOT / n) for n in CAMPAIGN_FILES})
        protocol = {
            'version': 'certified_1b_v3', 'arms': list(ARMS), 'eta0': eta0, 'c': cs, 'companions': companions,
            'eta_grid': list(ETA_GRID), 'c_grid': list(C_GRID), 'candidates': candidates, 'extension_plan': plan,
            'extension_plan_sha256': sha(plan_file) if plan_file.exists() else None,
            'development_seed': DEV_SEED, 'development_steps': DEV_STEPS, 'confirmatory_seeds': list(SEEDS),
            'confirmatory_steps': CONF_STEPS, 'validation_every': 100, 'halvings': 20, 'sources': sources,
            'selection': 'per arm, the unprotected uniform step included, minimum final development validation worst-task answer CE over '
                         'its development configurations (c_grid x eta_grid for certified rules, plus the extension plan); failed runs '
                         'excluded; ties to c = 1/2, then the smaller eta0; no test access',
            'amendments': ['c = 1e-4 added to the development grid on 10 November 2026, after the c = 1/2 development runs showed accepted '
                           'steps 2 to 15 times shorter than the unprotected step and before any confirmatory run; the c = 1/2 runs used '
                           'the previous study source, which differs only by the --c argument (study_source_sha256 per candidate)',
                           plan.get('amendment')],
            'primary_endpoint': 'final held-out worst-task answer CE, paired by seed: auditfirst_certified minus uniform_certified',
            'secondary': ['direct_certified minus uniform_certified and famo_certified minus uniform_certified on the same endpoint',
                          'refusals (not_certified) and holds (no_admissible_step) per certified arm',
                          'updates raising some task minibatch loss: uniform_fixed against every certified arm (0 by construction)',
                          'each certified arm against the campaign unprotected arms of the same seed and batch sequence (descriptive)',
                          'accepted step sizes, seconds per update, peak memory, native audits',
                          'the primary rule at its selected constant minus its companion at the other constant (descriptive)'],
            'scope': 'plain-SGD same-minibatch guarantee in floating point at Pythia-1B; no momentum, population or held-out guarantee',
            'utc_seconds': time.time()}
        (STUDY / 'protocol.json').write_text(json.dumps(protocol, indent=1))
        print(json.dumps({'frozen_eta0': eta0, 'frozen_c': cs, 'companions': companions, 'protocol_sha256': sha(STUDY / 'protocol.json')}), flush=True)
    else:
        protocol = STUDY / 'protocol.json'
        P = json.loads(protocol.read_text())
        for seed in P['confirmatory_seeds']:
            for arm in P['arms']:
                run(arm, seed, 'confirmatory', P['eta0'][arm], P['confirmatory_steps'], P['validation_every'], P['c'][arm], protocol)
            for x in P['companions']:
                run(x['arm'], seed, 'confirmatory', x['eta0'], P['confirmatory_steps'], P['validation_every'], x['c'], protocol)
        (STUDY / 'confirmatory_complete.json').write_text(json.dumps({'complete': True, 'protocol_sha256': sha(protocol), 'utc_seconds': time.time()}))


if __name__ == '__main__':
    main()
