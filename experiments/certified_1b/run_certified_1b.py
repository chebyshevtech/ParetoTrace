"""Runner for the certified-step Pythia-1B study on the Colab VM. Usage: run_certified_1b.py <stage>[,<stage>...]

Verifies every uploaded source against expected_sources.json, pins the campaign's library versions, rebuilds and verifies
the frozen language pool if needed, downloads the pinned Pythia-1B revision, then runs the requested driver stages.
Writes runner_status.json throughout and DONE_<stages>.json at the end, also on failure.
"""
import hashlib
import json
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

R = Path('/content/rebuttal')
V2 = R / 'language_training_v2'
S = R / 'certified_1b'
POOL_SHA = '7168eea3ca22eca16fc3877cb22ec8ec0d45d27c38a03730155adfdf7d5afbed'
PINS = ['transformers==4.57.1', 'tokenizers==0.22.2', 'huggingface_hub==0.36.2', 'accelerate==1.15.0', 'safetensors==0.8.0']
stages = sys.argv[1].split(',')
tag = '_'.join(stages)
status = {'stages': stages, 'stage': 'setup', 'utc': time.time()}


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda: f.read(8 * 2**20), b''):
            h.update(block)
    return h.hexdigest()


def put(**k):
    status.update(k)
    status['utc'] = time.time()
    (S / ('runner_status_%s.json' % tag)).write_text(json.dumps(status, indent=1))
    (S / 'runner_status.json').write_text(json.dumps(status, indent=1))


def run(name, cmd, log, cwd):
    put(stage=name, command=' '.join(cmd))
    with open(S / log, 'w') as f:
        rc = subprocess.run(cmd, cwd=str(cwd), stdout=f, stderr=subprocess.STDOUT).returncode
    put(**{name + '_returncode': rc})
    if rc:
        raise RuntimeError(name + ' failed with return code ' + str(rc))


try:
    expected = json.loads((S / 'expected_sources.json').read_text())
    got = {rel: (sha(R / rel) if (R / rel).exists() else None) for rel in expected}
    mism = [rel for rel, h in expected.items() if got[rel] != h]
    put(sources_match=not mism, source_mismatches=mism)
    if mism:
        raise RuntimeError('sources differ from the upload record: ' + str(mism))
    r = subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '--no-deps'] + PINS, capture_output=True, text=True)
    ver = subprocess.run([sys.executable, '-c', 'import transformers, tokenizers, huggingface_hub, torch; '
                          'print(transformers.__version__, tokenizers.__version__, huggingface_hub.__version__, torch.__version__)'],
                         capture_output=True, text=True).stdout.strip()
    put(pip_returncode=r.returncode, versions=ver)
    if not ver.startswith('4.57.1 0.22.2 0.36.2'):
        raise RuntimeError('environment pin failed: ' + ver)
    dst = V2 / 'data' / 'official_split_pool.pt'
    if not dst.exists() or sha(dst) != POOL_SHA:
        run('prepare_language', [sys.executable, '-u', str(R / 'prepare_language.py')], 'prepare_language.log', R)
        pool = R / 'language' / 'official_split_pool.pt'
        if sha(pool) != POOL_SHA:
            raise RuntimeError('language pool differs from the frozen protocol')
        shutil.copy2(pool, dst)
    put(pool_matches_frozen=True, stage='download_pythia_1b')
    run('download_pythia_1b', [sys.executable, '-c', "from huggingface_hub import snapshot_download; "
                               "print(snapshot_download('EleutherAI/pythia-1b-deduped', revision='d1988145ec1cf1e76d786cd05af9f53e8b1c95cc'))"],
        'download.log', S)
    while subprocess.run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], capture_output=True, text=True).stdout.strip():
        time.sleep(15)
    for stage in stages:
        run('stage_' + stage, [sys.executable, '-u', str(S / 'certified_1b_driver.py'), '--stage', stage], 'driver_%s.log' % stage, S)
    put(stage='done', complete=True)
except Exception:
    put(stage='failed', complete=False, error=traceback.format_exc()[-3000:])
finally:
    (S / ('DONE_%s.json' % tag)).write_text(json.dumps(status, indent=1))
