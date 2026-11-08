"""Two-stage full-data QM9 campaign; fixed systems pilot, then equal LR tuning.

The stage argument is explicit so development can proceed before committing
to a reporting compute budget. No reporting test metric enters selection.
All reporting configurations must finish before an aggregate analysis.
"""
import argparse,fcntl,hashlib,json,math,shutil,subprocess,time,traceback
from pathlib import Path

METHODS=['direct_factorized','mgda_ub','uniform','Nash_MTL','FairGrad','Aligned_MTL','FAMO']
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,obj):
    p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj,indent=2,allow_nan=False));tmp.replace(p)
def frozen_json(p,obj):
    if p.exists():
        if json.loads(p.read_text())!=obj:raise ValueError('Frozen identity changed: '+str(p))
    else:save(p,obj)

def validate_metrics(metrics,graphs,contract):
    if metrics['graphs']!=graphs or metrics['target_order']!=contract['target_order']:raise ValueError('Evaluation coverage/order')
    for key in ('normalized_mae','physical_mae'):
        values=metrics[key]
        if len(values)!=11 or any(not isinstance(x,(int,float)) or not math.isfinite(x) or x<0 for x in values):raise ValueError('Invalid target metric')
    values=metrics['normalized_mae']
    if not math.isclose(sum(values)/11,metrics['mean_normalized_mae'],rel_tol=1e-12,abs_tol=1e-14) or max(values)!=metrics['worst_normalized_mae']:
        raise ValueError('Incorrect aggregate metric')
    energy={2,3,6,12,13,14,15}
    for i,target in enumerate(contract['target_order']):
        expected=values[i]*contract['std'][0][target]*(1000 if target in energy else 1)
        if not math.isclose(expected,metrics['physical_mae'][i],rel_tol=1e-12,abs_tol=1e-14):raise ValueError('Physical unit conversion')

def select_systems(original,joint):
    if not original.get('complete') or not joint.get('complete'):raise ValueError('Both completed systems pilots required')
    selected={};detail={}
    for method in METHODS:
        source=joint if method=='direct_factorized' else original
        # Nash's pinned previous-alpha fallback is executed baseline behavior,
        # retained and counted. It is never relabelled an optimal solver call.
        good=[r for r in source['rows'] if r['config']['method']==method and
              (r.get('executed') if method=='Nash_MTL' else r.get('complete')) and
              isinstance(r.get('median_update_seconds'),(int,float)) and
              math.isfinite(r['median_update_seconds']) and r['median_update_seconds']>0]
        # Molecular batch graph sizes vary. Reserve headroom prospectively;
        # do not discover a smaller tile by selecting surviving reporting runs.
        if method=='direct_factorized':good=[r for r in good if r['peak_allocated_bytes']<=30*2**30]
        if not good:raise ValueError('No executable system configuration: '+method)
        winner=min(good,key=lambda r:r['median_update_seconds'])
        selected[method]=winner['config'];detail[method]={'median_update_seconds':winner['median_update_seconds'],
            'pilot':'joint' if method=='direct_factorized' else 'original',
            'development_peak_cap_bytes':30*2**30 if method=='direct_factorized' else None,
            'pilot_unsuccessful_solver_calls':winner.get('unsuccessful_solver_calls',0)}
    return selected,detail

def validate_result(result,cfg,root,out,complete=True):
    import torch
    required={'qm9_training_rebuttal.py','qm9_methods_v2.py','shared_direct_v2.py','simplex_qp.py','qm9_native_actions.py','qm9_data_verified.py'}
    if result.get('config')!=cfg:raise ValueError('Run configuration differs')
    if complete and result.get('complete') is not True:raise ValueError('Incomplete run')
    if result['identity']['config_sha256']!=sha(root/'configs'/f"{out.name}.json"):raise ValueError('Config hash differs')
    if set(result['identity']['sources'])!=required:raise ValueError('Incomplete source identity')
    if set(result['identity']['upstream'])!={f.name for f in (root/'weighting_upstream').glob('*.py')}:raise ValueError('Incomplete upstream identity')
    contract=json.loads((root/'original_preflight.json').read_text())['data']
    if result['identity']['data']!=contract:raise ValueError('Data contract mismatch')
    split_path=root/'qm9_upstream/random_split.t'
    if sha(split_path)!=contract['split_sha256']:raise ValueError('Split hash mismatch')
    split=torch.as_tensor(torch.load(split_path,map_location='cpu',weights_only=True)).long()[20000:]
    for name,digest in result['identity']['sources'].items():
        if sha(root/name)!=digest:raise ValueError('Scientific source changed: '+name)
    for name,digest in result['identity']['upstream'].items():
        if sha(root/'weighting_upstream'/name)!=digest:raise ValueError('Upstream source changed')
    if complete:
        if result['epochs_finished']!=cfg['epochs'] or len(result['history'])!=cfg['epochs']:
            raise ValueError('Incomplete epoch coverage')
        scores=[x['validation']['mean_normalized_mae'] for x in result['history']]
        if [x['epoch'] for x in result['history']]!=list(range(cfg['epochs'])):raise ValueError('Validation epoch order')
        for row in result['history']:validate_metrics(row['validation'],10000,contract)
        if any(not math.isfinite(x) or x<0 for x in scores):raise ValueError('Invalid validation score')
        best=min(range(len(scores)),key=lambda i:scores[i])
        if best!=result['best_epoch'] or scores[best]!=result['best_validation_mean_normalized_mae']:
            raise ValueError('Incorrect validation selection')
        if cfg['phase']=='development':
            if result.get('heldout_test_evaluated') is not False or any('test' in k and k!='heldout_test_evaluated' for k in result):
                raise ValueError('Development test metric access')
        elif result.get('heldout_test_evaluated') is not True or any(k not in result for k in ['final_test','validation_selected_test']):
            raise ValueError('Missing reporting metrics')
        else:
            for key in ['final_test','validation_selected_test']:validate_metrics(result[key],10000,contract)
        expected=cfg['epochs']*math.ceil(110831/cfg['batch_size'])
        if result['updates']!=expected:raise ValueError('Training update coverage mismatch')
        cursor=0;seconds=0.;solver_calls=0;solver_failures=0;orders={}
        for block in result['update_blocks']:
            path=out/block['file']
            if sha(path)!=block['sha256'] or block['first_step']!=cursor:raise ValueError('Event block identity/order')
            rows=json.loads(path.read_text())
            if len(rows)!=block['rows'] or [r['step'] for r in rows]!=list(range(cursor,cursor+len(rows))):raise ValueError('Event coverage')
            if rows[-1]['step']!=block['last_step']:raise ValueError('Event endpoint')
            for row in rows:
                epoch,within=divmod(row['step'],math.ceil(len(split)/cfg['batch_size']))
                offset=within*cfg['batch_size'];count=min(cfg['batch_size'],len(split)-offset)
                if (row['epoch'],row['offset'],row['graphs'])!=(epoch,offset,count):raise ValueError('Wrong batch cardinality/cursor')
                if epoch not in orders:
                    orders={epoch:split[torch.randperm(len(split),generator=torch.Generator().manual_seed(cfg['seed']*100000+epoch))]}
                expected_hash=hashlib.sha256(orders[epoch][offset:offset+count].numpy().tobytes()).hexdigest()
                if row['indices_sha256']!=expected_hash:raise ValueError('Wrong paired sample order')
                elapsed=row['update_seconds']
                if not isinstance(elapsed,(int,float)) or not math.isfinite(elapsed) or elapsed<0:raise ValueError('Invalid update timing')
                seconds+=elapsed;events=row.get('solver_events',[]);solver_calls+=len(events)
                failed=sum(not(e.get('success') is True if 'success' in e else e.get('status')=='optimal') for e in events)
                if failed!=row['solver_fallback_or_failure_calls']:raise ValueError('Solver event count differs')
                solver_failures+=failed
            cursor+=len(rows)
        if cursor!=expected:raise ValueError('Incomplete update trace')
        if not math.isclose(seconds,result['update_seconds'],rel_tol=1e-9,abs_tol=1e-6):raise ValueError('Timing total differs')
        if (solver_calls,solver_failures)!=(result['solver_calls'],result['solver_fallback_or_failure_calls']):raise ValueError('Solver totals differ')
        ref=result['selected_checkpoint']
        if sha(out/ref['file'])!=ref['sha256'] or ref['epoch']!=best or ref['metric']!=scores[best]:raise ValueError('Selected model identity')
        if cfg['system']['method']=='direct_factorized' and (not result['native_audits'] or any(not a['passed'] for a in result['native_audits'])):
            raise ValueError('Unresolved Direct native audit')
        if cfg['system']['method']=='direct_factorized':
            expected_audits={1,*[(e+1)*math.ceil(len(split)/cfg['batch_size']) for e in {cfg['epochs']//2,cfg['epochs']-1}]}
            if {a['after_updates'] for a in result['native_audits']}!=expected_audits:raise ValueError('Missing native audit milestone')
            for audit in result['native_audits']:
                for key in ('direction_relative_error','gram_relative_error'):
                    if not math.isfinite(audit[key]) or not 0<=audit[key]<=2e-5:raise ValueError('Native fidelity gate')
                if audit['relative_reference_resolved'] and (not math.isfinite(audit['objective_factor_upper']) or audit['objective_factor_upper']>1.001):raise ValueError('Native objective gate')

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--stage',choices=['development','confirmatory'],required=True)
    p.add_argument('--reporting-epochs',type=int,choices=[50,100,300],default=50)
    a=p.parse_args();root=a.root;root.mkdir(parents=True,exist_ok=True)
    lock=(root/'campaign.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    marker=root/f'{a.stage}_complete.json'
    if marker.exists():marker.rename(root/f'{a.stage}_complete_previous_{time.time_ns()}.json')
    backup=Path('/content/drive/MyDrive/ParetoTrace/rebuttal_20261107/qm9_training')
    if not backup.parent.exists():raise RuntimeError('Verified Drive destination required')
    backup.mkdir(exist_ok=True)
    stale=backup/f'{a.stage}_complete.json'
    if stale.exists():stale.rename(backup/f'{a.stage}_complete_previous_{time.time_ns()}.json')
    original=json.loads((root/'original_preflight.json').read_text());joint=json.loads((root/'joint_preflight.json').read_text())
    systems,detail=select_systems(original,joint)
    protocol={'version':'qm9_rebuttal_v1','arms':METHODS,'development_seed':38102,'reporting_seeds':[38201,38202,38203],
        'development_learning_rates':[1e-4,3e-4,1e-3],'development_epochs':3,
        'reporting_budget':'frozen separately before any confirmatory run; it does not alter validation tuning',
        'selection':'lowest best validation mean-normalized MAE over three development epochs, then lower learning rate',
        'systems':systems,'systems_pilot_summary':detail,'batch_size':128,'optimizer':'Adam, weight_decay1e-5',
        'systems_memory_rule':'Direct fastest qualifying pilot configuration under30GiB allocated; headroom for variable-size graph batches, not an OOM guarantee',
        'method_hyperparameters':'pinned LibMTL adapter defaults; identical three-LR validation budget for every method',
        'normalization':'published whole-QM9 label means/std, including heldout labels; no development test-metric access',
        'scheduler':'shared validation mean normalized MAE, factor.7/patience5/min_lr1e-5; not upstream relative-first-epoch score',
        'primary_endpoint':'paired Direct-minus-UB difference in validation-selected test mean normalized MAE',
        'secondary':'per-target physical and normalized MAE; all methods; final/selected models; update/wall time and peak allocation',
        'inference':'all three paired seeds; report all differences and descriptive t95% intervals, no power/separation claim',
        'scope':'fixed-budget full-dataset training; 50/100 epochs are not the standard300-epoch convergence protocol',
        'systems_inputs':{n:sha(root/n) for n in ['original_preflight.json','joint_preflight.json']}}
    frozen_json(root/'protocol.json',protocol)
    source_identity={str(f.relative_to(root)):sha(f) for f in list(root.glob('*.py'))+list((root/'weighting_upstream').glob('*.py'))}
    frozen_json(root/'campaign_source_identity.json',source_identity)
    for name in ['protocol.json','campaign_source_identity.json','original_preflight.json','joint_preflight.json']:
        shutil.copy2(root/name,backup/name)
    if (root/'reporting_budget.json').exists():shutil.copy2(root/'reporting_budget.json',backup/'reporting_budget.json')
    for name in source_identity:
        target=backup/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(root/name,target)
    completed={}
    def run(method,seed,lr,phase):
        for name,digest in source_identity.items():
            if sha(root/name)!=digest:raise RuntimeError('Campaign source mutated')
        label=f'{phase}_{method}_seed{seed}_lr{lr:g}'
        cfg={'phase':phase,'seed':seed,'lr':lr,'epochs':3 if phase=='development' else a.reporting_epochs,
            'batch_size':128,'workers':2,'device':'cuda','weight_decay':1e-5,'system':systems[method],
            'protocol_sha256':sha(root/'protocol.json')}
        if phase=='confirmatory':cfg.update(reporting_budget_sha256=sha(root/'reporting_budget.json'),selection_sha256=sha(root/'selection.json'))
        config=root/'configs'/f'{label}.json';frozen_json(config,cfg)
        (backup/'configs').mkdir(exist_ok=True);shutil.copy2(config,backup/'configs'/config.name)
        out=root/'runs'/label;out.mkdir(parents=True,exist_ok=True)
        metrics=out/'metrics.json'
        if metrics.exists() and json.loads(metrics.read_text()).get('complete'):
            result=json.loads(metrics.read_text());validate_result(result,cfg,root,out)
            completed[label]={'metrics_sha256':sha(metrics),'config_sha256':sha(config)};return result
        if (out/'failure.json').exists():raise RuntimeError('Retained failure requires explicit diagnosis: '+label)
        cmd=[str(root/'venv/bin/python'),str(root/'qm9_training_rebuttal.py'),'--root',str(root),
             '--config',str(config),'--output',str(out),'--backup',str(backup/'runs'/label)]
        if (out/'latest.pt').exists():cmd.append('--resume')
        print(json.dumps({'start':label,'utc_seconds':time.time()}),flush=True)
        attempts_file=out/'process_attempts.json'
        attempts=json.loads(attempts_file.read_text()) if attempts_file.exists() else []
        attempts.append({'start_utc_seconds':time.time(),'complete':False,'resume':'--resume' in cmd})
        save(attempts_file,attempts);(backup/'runs'/label).mkdir(parents=True,exist_ok=True)
        shutil.copy2(attempts_file,backup/'runs'/label/attempts_file.name)
        tick=time.perf_counter()
        with (out/'run.log').open('a') as log:
            child=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
        attempts[-1].update(complete=True,end_utc_seconds=time.time(),subprocess_wall_seconds=time.perf_counter()-tick,returncode=child.returncode)
        save(attempts_file,attempts);shutil.copy2(attempts_file,backup/'runs'/label/attempts_file.name)
        (backup/'runs'/label).mkdir(parents=True,exist_ok=True);shutil.copy2(out/'run.log',backup/'runs'/label/'run.log')
        if child.returncode:raise RuntimeError('Campaign halted; no omitted arm/seed: '+label)
        result=json.loads(metrics.read_text());validate_result(result,cfg,root,out)
        completed[label]={'metrics_sha256':sha(metrics),'config_sha256':sha(config)};return result
    if a.stage=='development':
        rows={m:[] for m in METHODS}
        for lr in protocol['development_learning_rates']:
            for m in METHODS:
                d=run(m,38102,lr,'development');rows[m].append({'lr':lr,'metric':d['best_validation_mean_normalized_mae'],
                    'metrics_sha256':sha(root/'runs'/f'development_{m}_seed38102_lr{lr:g}'/'metrics.json')})
        selected={m:min(rows[m],key=lambda x:(x['metric'],x['lr']))['lr'] for m in METHODS}
        frozen_json(root/'selection.json',{'protocol_sha256':sha(root/'protocol.json'),'rates':selected,'development':rows})
        shutil.copy2(root/'selection.json',backup/'selection.json')
    else:
        selection=json.loads((root/'selection.json').read_text())
        if selection['protocol_sha256']!=sha(root/'protocol.json'):raise ValueError('Selection protocol differs')
        if set(selection['rates'])!=set(METHODS) or set(selection['development'])!=set(METHODS):raise ValueError('Incomplete tuning arms')
        for m in METHODS:
            choices=selection['development'][m]
            if sorted(x['lr'] for x in choices)!=protocol['development_learning_rates']:raise ValueError('Incomplete or duplicated learning-rate grid')
            for row in selection['development'][m]:
                out=root/'runs'/f'development_{m}_seed38102_lr{row["lr"]:g}'
                if sha(out/'metrics.json')!=row['metrics_sha256']:raise ValueError('Development selection changed')
                cfg=json.loads((root/'configs'/f'{out.name}.json').read_text())
                if cfg['phase']!='development' or cfg['seed']!=38102 or cfg['lr']!=row['lr'] or cfg['epochs']!=3 or cfg['system']!=systems[m]:raise ValueError('Development configuration changed')
                d=json.loads((out/'metrics.json').read_text());validate_result(d,cfg,root,out)
                if d['best_validation_mean_normalized_mae']!=row['metric']:raise ValueError('Selection score differs')
            if selection['rates'][m]!=min(choices,key=lambda x:(x['metric'],x['lr']))['lr']:raise ValueError('Incorrect selected rate')
        frozen_json(root/'reporting_budget.json',{'epochs':a.reporting_epochs,'protocol_sha256':sha(root/'protocol.json'),
            'selection_sha256':sha(root/'selection.json'),'source_identity_sha256':sha(root/'campaign_source_identity.json')})
        shutil.copy2(root/'reporting_budget.json',backup/'reporting_budget.json')
        for seed in protocol['reporting_seeds']:
            order=METHODS[seed%len(METHODS):]+METHODS[:seed%len(METHODS)]
            for m in order:run(m,seed,selection['rates'][m],'confirmatory')
    if len(completed)!=21:raise ValueError('All21 stage runs required')
    save(root/f'{a.stage}_complete.json',{'complete':True,'protocol_sha256':sha(root/'protocol.json'),'utc_seconds':time.time(),
        'runs':completed,'source_identity_sha256':sha(root/'campaign_source_identity.json'),'selection_sha256':sha(root/'selection.json'),
        'reporting_budget_sha256':sha(root/'reporting_budget.json') if a.stage=='confirmatory' else None})
    shutil.copy2(root/f'{a.stage}_complete.json',backup/f'{a.stage}_complete.json')

if __name__=='__main__':main()
