"""Frozen two-epoch, three-paired-seed QM9 screen at the published learning rate.

This is early-training evidence, not a converged 300-epoch comparison. All
seven methods and every seed are retained, including baseline solver fallbacks.
"""
import argparse,fcntl,json,shutil,subprocess,time
from pathlib import Path
from qm9_campaign import METHODS,sha,save,frozen_json,select_systems,validate_result

def protocol(systems,detail,root):
    return {'version':'qm9_rebuttal_short_v1','arms':METHODS,'reporting_seeds':[38201,38202,38203],
        'epochs':2,'learning_rate':1e-4,'batch_size':128,'optimizer':'Adam, weight_decay1e-5',
        'systems':systems,'systems_pilot_summary':detail,
        'systems_memory_rule':'Direct fastest qualified pilot configuration under30GiB allocated; not an OOM guarantee',
        'method_hyperparameters':'pinned LibMTL defaults; fixed published learning rate for every method; no LR tuning',
        'normalization':'published whole-QM9 label means/std, including heldout labels',
        'scheduler':'shared validation mean normalized MAE, factor.7/patience5/min_lr1e-5; no reduction in two epochs',
        'primary_endpoint':'paired Direct-minus-MGDA-UB final test mean normalized MAE after two full-data epochs',
        'secondary':'all seven methods and11targets; final and validation-selected test metrics; all seed differences; cost/memory',
        'inference':'three paired seeds; descriptive differences and t95% intervals; no power or convergence claim',
        'scope':'bounded early-training screen using110831training graphs; not the standard300epoch comparison',
        'budget':'approximately2.5A100hours of updates plus evaluation, loading, packing, checkpoints; replaces unstarted long development campaign',
        'systems_inputs':{n:sha(root/n) for n in ['original_preflight.json','joint_preflight.json']}}

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();root=a.root
    lock=(root/'campaign.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    backup=Path('/content/drive/MyDrive/ParetoTrace/rebuttal_20261107/qm9_short')
    if not backup.parent.exists():raise RuntimeError('Verified Drive destination required')
    backup.mkdir(exist_ok=True)
    for base in [root,backup]:
        marker=base/'complete.json'
        if marker.exists():marker.rename(base/f'complete_previous_{time.time_ns()}.json')
    original=json.loads((root/'original_preflight.json').read_text());joint=json.loads((root/'joint_preflight.json').read_text())
    systems,detail=select_systems(original,joint);frozen_json(root/'protocol.json',protocol(systems,detail,root))
    identity={str(f.relative_to(root)):sha(f) for f in list(root.glob('*.py'))+list((root/'weighting_upstream').glob('*.py'))}
    frozen_json(root/'campaign_source_identity.json',identity)
    for name in ['protocol.json','campaign_source_identity.json','original_preflight.json','joint_preflight.json',*identity]:
        target=backup/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(root/name,target)
    completed={};campaign_start=time.time()
    for seed in [38201,38202,38203]:
        order=METHODS[seed%len(METHODS):]+METHODS[:seed%len(METHODS)]
        for method in order:
            for name,digest in identity.items():
                if sha(root/name)!=digest:raise RuntimeError('Campaign source mutated: '+name)
            label=f'confirmatory_{method}_seed{seed}_lr0.0001'
            cfg={'phase':'confirmatory','seed':seed,'lr':1e-4,'epochs':2,'batch_size':128,'workers':2,
                'device':'cuda','weight_decay':1e-5,'system':systems[method],'protocol_sha256':sha(root/'protocol.json')}
            config=root/'configs'/f'{label}.json';frozen_json(config,cfg)
            (backup/'configs').mkdir(exist_ok=True);shutil.copy2(config,backup/'configs'/config.name)
            out=root/'runs'/label;out.mkdir(parents=True,exist_ok=True);dest=backup/'runs'/label;dest.mkdir(parents=True,exist_ok=True)
            metrics=out/'metrics.json'
            if (out/'failure.json').exists():raise RuntimeError('Retained failure requires diagnosis: '+label)
            if not (metrics.exists() and json.loads(metrics.read_text()).get('complete')):
                cmd=[str(root/'venv/bin/python'),str(root/'qm9_training_rebuttal.py'),'--root',str(root),
                    '--config',str(config),'--output',str(out),'--backup',str(dest)]
                if (out/'latest.pt').exists():cmd.append('--resume')
                attempts_file=out/'process_attempts.json'
                attempts=json.loads(attempts_file.read_text()) if attempts_file.exists() else []
                attempts.append({'start_utc_seconds':time.time(),'complete':False,'resume':'--resume' in cmd})
                save(attempts_file,attempts);shutil.copy2(attempts_file,dest/attempts_file.name)
                print(json.dumps({'start':label,'utc_seconds':time.time()}),flush=True);tick=time.perf_counter()
                with (out/'run.log').open('a') as log:child=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
                attempts[-1].update(complete=True,end_utc_seconds=time.time(),subprocess_wall_seconds=time.perf_counter()-tick,returncode=child.returncode)
                save(attempts_file,attempts);shutil.copy2(attempts_file,dest/attempts_file.name);shutil.copy2(out/'run.log',dest/'run.log')
                if child.returncode:raise RuntimeError('Campaign halted; no omitted arm or seed: '+label)
            result=json.loads(metrics.read_text());validate_result(result,cfg,root,out)
            completed[label]={'metrics_sha256':sha(metrics),'config_sha256':sha(config)}
    if len(completed)!=21:raise ValueError('All21runs required')
    save(root/'complete.json',{'complete':True,'runs':completed,'protocol_sha256':sha(root/'protocol.json'),
        'source_identity_sha256':sha(root/'campaign_source_identity.json'),'utc_seconds':time.time(),
        'current_invocation_wall_seconds':time.time()-campaign_start,'scope':'two-epoch early-training evidence, not convergence'})
    shutil.copy2(root/'complete.json',backup/'complete.json')

if __name__=='__main__':main()
