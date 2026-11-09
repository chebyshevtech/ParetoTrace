"""Frozen, equally tuned full-parameter language comparison; budgets explicit.

Development never reads test metrics. Completed results remain on Drive;
optionally retain completed models as verified lossless archives on the VM.
An explicit policy chooses full active recovery in Drive slots or numerical
records only with model state on the VM. Colab VM persistence is not guaranteed.
"""
import argparse,fcntl,hashlib,json,math,subprocess,time
from pathlib import Path
import numpy as np
from checkpoint_archive import atomic_json,verified_copy,sha,fsync_path
from retire_completed_language import retire_completed,owned_model,verify_archive as verify_checkpoint_archive
from language_checkpoint_storage import (validate_backup,restore_backup,validate_vm_retention,
    writer as checkpoint_writer,METADATA_FORMAT,FINAL_ONLY_RETENTION)

METHODS=['direct','uniform','cagrad','fa','periodic','famo']
RATES=[1e-4,3e-4,1e-3]
SEEDS=[49201,49202,49203]
SOURCE_NAMES={'language_training_rebuttal.py','language_training_methods.py','language_common.py',
    'attention_precision.py','simplex_qp.py','v46_factorized_jacobian.py','v46_factorized_exact.py',
    'paretotrace_online_factor.py','v46_supervised_head.py','native_language_dual_reference.py',
    'native_language_dual_reference_gpu_v6.py','language_baseline_v6.py','ghost_adapter.py','structured_gram.py',
    'language_retained_v31.py','language_retained_manifest.json','language_tokenizer_contract.py','checkpoint_archive.py',
    'drive_checkpoint_slots.py','language_checkpoint_storage.py','retire_completed_language.py'}
_DATA_CACHE={}
_BATCH_CACHE={}

def run_label(method,seed,lr,phase):
    return f'{phase}_{method}_seed{seed}_lr{lr:g}'

def configuration(protocol,protocol_sha256,method,seed,lr,phase):
    if method not in METHODS or lr not in RATES or phase not in ('development','confirmatory'):
        raise ValueError('Unknown campaign arm/rate/stage')
    if seed not in ([49100] if phase=='development' else SEEDS):raise ValueError('Campaign seed')
    choice=protocol['systems'][method]
    return {'phase':phase,'seed':seed,'method':method,'lr':lr,'model':protocol['model'],'batch_size':32,
        'steps':protocol['development_steps'] if phase=='development' else protocol['reporting_steps'],
        'validation_every':100,'system':choice['system'],'method_options':choice.get('method_options',{}),
        'protocol_sha256':protocol_sha256,'checkpoint_backup':protocol['checkpoint_backup'],
        'checkpoint_slot_budget_bytes':protocol['checkpoint_slot_budget_bytes'],
        'completed_checkpoint_policy':protocol['completed_checkpoint_policy']}

def relative_file(base,value,parent):
    rel=Path(value)
    if rel.is_absolute() or '..' in rel.parts or rel.parent!=Path(parent):raise ValueError('Unsafe artifact reference')
    return base/rel

def data_contract(root):
    manifest=json.loads((root/'data/data_manifest.json').read_text())
    digest=sha(root/'data/official_split_pool.pt')
    if digest!=manifest['pool_sha256']:raise ValueError('Frozen data pool hash')
    key=(str(root.resolve()),sha(root/'data/data_manifest.json'),digest)
    if key not in _DATA_CACHE:
        from language_common import load_data
        data,loaded=load_data(root/'data')
        if loaded!=manifest or len(manifest['tasks'])!=8:raise ValueError('Data objective identity')
        for split in ('train','validation','test'):
            if set(data[split])!=set(manifest['tasks']):raise ValueError('Data task coverage')
            if {t:len(data[split][t]) for t in manifest['tasks']}!=manifest['counts'][split]:
                raise ValueError('Data cardinalities differ from manifest')
        _DATA_CACHE[key]=(data,manifest)
    return _DATA_CACHE[key]

def expected_batch_hashes(data,manifest,cfg):
    key=(manifest['pool_sha256'],tuple(manifest['tasks']),cfg['seed'],cfg['steps'],cfg['batch_size'])
    if key not in _BATCH_CACHE:
        hashes=[]
        for step in range(cfg['steps']):
            identities=[]
            for task in manifest['tasks']:
                rng=np.random.default_rng(cfg['seed']*100000+step+int(hashlib.sha256(task.encode()).hexdigest()[:8],16))
                chosen=rng.choice(len(data['train'][task]),cfg['batch_size']//8,replace=False)
                identities.extend(data['train'][task][int(i)]['identity'] for i in chosen)
            hashes.append(hashlib.sha256(json.dumps(identities,sort_keys=True).encode()).hexdigest())
        _BATCH_CACHE[key]=hashes
    return _BATCH_CACHE[key]

def checkpoint_identity(out,rel,digest,verify_archive=False):
    file=owned_model(out,rel)
    receipt_path=out/'completed_checkpoint_retention.json'
    if receipt_path.exists():
        receipt=json.loads(receipt_path.read_text())
        if receipt.get('storage_format')==FINAL_ONLY_RETENTION:
            validate_vm_retention(out,receipt)
            expected=receipt['local_checkpoint_archives']['latest.pt']['source_sha256'] if rel=='latest.pt' else receipt['selected_model']['source_sha256']
            if digest!=expected or (rel!='latest.pt' and rel!=receipt['selected_model']['file']):raise ValueError('Final-only source identity differs')
            return
    if file.exists():
        if sha(file)!=digest:raise ValueError('Checkpoint source differs')
        return
    retention=json.loads((out/'completed_checkpoint_retention.json').read_text())
    record=retention.get('local_checkpoint_archives',{}).get(rel)
    metrics=json.loads((out/'metrics.json').read_text())
    if (retention.get('complete') is not True or retention['metrics_sha256']!=sha(out/'metrics.json') or
            retention['config']!=metrics['config'] or record is None or record.get('format')!='gzip-lossless-v1' or record['source_sha256']!=digest):
        raise ValueError('Missing verified completed checkpoint')
    name=Path(record['archive_file'])
    if name.name!=str(name):raise ValueError('Unsafe checkpoint archive')
    archive_root=validate_vm_retention(out,retention) if retention.get('storage_format') else out
    archive=archive_root/Path(rel).parent/name
    verify_checkpoint_archive(archive,record,full=verify_archive)

def retire_marker(path):
    if not path.exists():return
    dest=path.parent/'retired_markers'/(path.stem+'_'+sha(path)+'.json')
    verified_copy(path,dest);path.unlink();fsync_path(path.parent)

def freeze(path,value):
    if path.exists():
        if json.loads(path.read_text())!=value:raise ValueError('Frozen identity changed: '+str(path))
    else:atomic_json(path,value)

def validate_metrics(metrics,manifest,split):
    rows=metrics['tasks']
    if [r['task'] for r in rows]!=manifest['tasks']:raise ValueError('Evaluation task order')
    for row in rows:
        if type(row['examples']) is not int or row['examples']!=manifest['counts'][split][row['task']]:raise ValueError('Evaluation cardinality')
        if not math.isfinite(row['answer_ce']) or row['answer_ce']<0 or not 0<=row['teacher_forced_answer_exact_match']<=1:
            raise ValueError('Evaluation metric range')
    ce=[r['answer_ce'] for r in rows];acc=[r['teacher_forced_answer_exact_match'] for r in rows]
    for key,expected in [('mean_ce',np.mean(ce)),('worst_ce',max(ce)),('mean_answer_exact_match',np.mean(acc)),('worst_answer_exact_match',min(acc))]:
        if not math.isclose(metrics[key],float(expected),rel_tol=1e-12,abs_tol=1e-12):raise ValueError('Incorrect aggregate')

def validate_identity(root,cfg,identity):
    label=run_label(cfg['method'],cfg['seed'],cfg['lr'],cfg['phase'])
    if identity['config_sha256']!=sha(root/'configs'/f'{label}.json'):raise ValueError('Configuration hash')
    if set(identity['source_sha256'])!=SOURCE_NAMES:raise ValueError('Executed source coverage')
    for name,digest in identity['source_sha256'].items():
        if sha(root/name)!=digest:raise ValueError('Executed source changed: '+name)
    _,manifest=data_contract(root)
    if identity['data']!=manifest:raise ValueError('Data identity')
    if identity['model']!=json.loads((root/'data'/f'{cfg["model"].split("/")[-1]}_manifest.json').read_text()):raise ValueError('Model identity')

def recover_active(root,out,dest,cfg):
    """Restore only an owned committed run, preserving incomplete local files."""
    owner=json.loads((dest/'run_identity.json').read_text())
    if owner['config']!=cfg:raise ValueError('Recovery configuration differs')
    validate_identity(root,cfg,owner['identity'])
    commit=validate_backup(dest,owner)
    if commit['storage_format']==METADATA_FORMAT:
        raise RuntimeError('Metadata-only backup cannot recover model/rule/RNG state after VM checkpoint loss; restore verified local model archives before continuing')
    if commit.get('state')=='retired':raise RuntimeError('Completed Drive slots are retired; restore retained VM model archives before continuing')
    if out.is_symlink():raise ValueError('Unexpected recovery output location')
    if out.exists() and any(out.iterdir()):
        if not (out/'run_identity.json').exists() or json.loads((out/'run_identity.json').read_text())!=owner:
            raise ValueError('Partial recovery output ownership differs')
        orphan=root/'recovery_orphans'/f'{out.name}_{time.time_ns()}'
        orphan.parent.mkdir(parents=True,exist_ok=True);fsync_path(root)
        out.rename(orphan);fsync_path(out.parent);fsync_path(orphan.parent)
    restore_backup(dest,out,owner)

def validate_result(root,out,cfg,verify_archives=False):
    protocol=json.loads((root/'protocol.json').read_text());protocol_digest=sha(root/'protocol.json')
    if cfg!=configuration(protocol,protocol_digest,cfg['method'],cfg['seed'],cfg['lr'],cfg['phase']):raise ValueError('Run differs from frozen campaign')
    if out.name!=run_label(cfg['method'],cfg['seed'],cfg['lr'],cfg['phase']):raise ValueError('Run label identity')
    if (out/'failure.json').exists():raise ValueError('Unresolved run failure')
    d=json.loads((out/'metrics.json').read_text())
    if d.get('complete') is not True or d['config']!=cfg or d['steps']!=cfg['steps']:raise ValueError('Incomplete or changed run')
    validate_identity(root,cfg,d['identity'])
    data,manifest=data_contract(root)
    if json.loads((out/'run_identity.json').read_text())!={'config':cfg,'identity':d['identity']}:raise ValueError('Run ownership')
    batches=expected_batch_hashes(data,manifest,cfg)
    cursor=0;seconds=0.;scheduled=0;applied=0;captures=0;post=0
    for block in d['update_blocks']:
        file=relative_file(out,block['file'],'updates')
        if sha(file)!=block['sha256'] or block['first_step']!=cursor:raise ValueError('Trace identity/order')
        rows=json.loads(file.read_text())
        if not rows or len(rows)!=block['rows'] or [r['step'] for r in rows]!=list(range(cursor,cursor+len(rows))):raise ValueError('Trace coverage')
        for row in rows:
            step=row['step']
            if step>=len(batches) or row['batch_identity_sha256']!=batches[step]:raise ValueError('Training batch pairing/identity')
            if not math.isfinite(row['update_seconds']) or row['update_seconds']<0:raise ValueError('Timing')
            seconds+=row['update_seconds']
            expected=cfg['method']=='periodic' and step%5==0
            if row['adaptive_actions_this_step']!=int(expected):raise ValueError('Periodic action schedule')
            if cfg['method']=='periodic':
                scheduled+=expected
                if expected:
                    if type(row['refinement_applied']) is not bool:raise ValueError('Periodic refinement metadata')
                    applied+=row['refinement_applied']
            captures=row['gram_captures'];post=row['post_update_forwards']
            expected_captures=step+1 if cfg['method'] in ('direct','cagrad') else int(cfg['method']=='fa')
            expected_post=step+1 if cfg['method'] in ('fa','famo') else 0
            if (captures,post,row['periodic_refinements'])!=(expected_captures,expected_post,applied):raise ValueError('Per-update method counters')
        if rows[-1]['step']!=block['last_step']:raise ValueError('Trace endpoint')
        cursor+=len(rows)
    if cursor!=cfg['steps'] or not math.isclose(seconds,d['update_seconds'],rel_tol=1e-10,abs_tol=1e-7):raise ValueError('Total update coverage/cost')
    if (captures,post)!=(d['gram_captures'],d['post_update_forwards']):raise ValueError('Phase counters')
    if applied!=d['periodic_refinements']:raise ValueError('Refinement counter')
    wanted=[0,*range(cfg['validation_every'],cfg['steps']+1,cfg['validation_every'])]
    if wanted[-1]!=cfg['steps']:wanted.append(cfg['steps'])
    if [r['after_updates'] for r in d['history']]!=wanted:raise ValueError('Validation coverage')
    for row in d['history']:validate_metrics(row['metrics'],manifest,'validation')
    best=min(d['history'][1:],key=lambda r:r['metrics']['worst_ce']);selected=d['selected_checkpoint']
    if Path(selected['file']).parent!=Path('selected'):raise ValueError('Selected checkpoint must remain separate from latest')
    if selected['after_updates']!=best['after_updates'] or selected['worst_ce']!=best['metrics']['worst_ce']:
        raise ValueError('Validation selection')
    running=None
    for event in d['history']:
        improved=event['after_updates']>0 and (running is None or event['metrics']['worst_ce']<running)
        if event['selected'] is not improved:raise ValueError('Selection history')
        if improved:running=event['metrics']['worst_ce']
    checkpoint_identity(out,selected['file'],selected['sha256'],verify_archives)
    if cfg['method']=='direct':
        if [r['after_updates'] for r in d['native_audits']]!=[1,cfg['steps']//2,cfg['steps']] or any(r['passes_table_contract'] is not True for r in d['native_audits']):
            raise ValueError('Direct independent audits')
        for audit in d['native_audits']:
            if type(audit['passes_weighted_contract']) is not bool or audit['passes_joint_contract']!=(audit['passes_table_contract'] and audit['passes_weighted_contract']):raise ValueError('Dual audit contracts')
            if json.loads((out/'audits'/f'step{audit["after_updates"]:08d}.json').read_text())!=audit:raise ValueError('Independent audit artifact')
    elif d['native_audits']:raise ValueError('Unexpected method audits')
    if cfg['phase']=='development':
        if d.get('heldout_test_evaluated') is not False or 'final_test' in d or 'validation_selected_test' in d:raise ValueError('Development test access')
    else:
        if d.get('heldout_test_evaluated') is not True:raise ValueError('Missing heldout evaluation')
        for key in ['final_test','validation_selected_test']:validate_metrics(d[key],manifest,'test')
    d['validated_periodic_scheduled_opportunities']=scheduled
    d['validated_batch_sequence_sha256']=hashlib.sha256(json.dumps(batches).encode()).hexdigest()
    return d

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--backup',type=Path,default=Path('/content/drive/MyDrive/ParetoTrace/rebuttal_20261107/language_training'))
    p.add_argument('--stage',choices=['development','confirmatory'],required=True)
    p.add_argument('--development-steps',type=int,required=True);p.add_argument('--reporting-steps',type=int,required=True)
    p.add_argument('--completed-checkpoints',choices=['keep','vm','final-only'],default='keep')
    p.add_argument('--checkpoint-backup',choices=['slots','metadata'],required=True,
        help='Explicit storage policy: slots requires measured Drive capacity; metadata cannot recover models after VM loss')
    p.add_argument('--slot-budget-bytes',type=int,help='Measured capacity available to the four model slots; required for slots policy')
    a=p.parse_args();root=a.root
    if (a.checkpoint_backup=='slots' and (a.slot_budget_bytes is None or a.slot_budget_bytes<=0)) or (a.checkpoint_backup=='metadata' and a.slot_budget_bytes is not None):
        raise ValueError('Slots require a positive measured slot budget; metadata mode must omit it')
    if min(a.development_steps,a.reporting_steps)<100 or a.development_steps%100 or a.reporting_steps%100:
        raise ValueError('Budgets must be positive multiples of100 and at least100')
    lock=(root/'campaign.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    backup=a.backup;backup.mkdir(parents=True,exist_ok=True)
    if backup.resolve()==root.resolve():raise ValueError('Campaign output and backup must differ')
    pilot=json.loads((root/'training_preflight.json').read_text())
    if pilot.get('complete') is not True or any(pilot['selected'].get(m) is None for m in METHODS):raise ValueError('All six pilot arms must qualify')
    systems=pilot['selected']
    for method in METHODS:
        choice=systems[method]
        if choice['method']!=method:raise ValueError('Pilot arm identity')
        options=choice.get('method_options',{})
        if options and (method!='periodic' or set(options)!={'periodic_backend'} or options['periodic_backend'] not in ('native','factor')):
            raise ValueError('Pilot changed fixed method defaults')
    data_contract(root)
    protocol={'version':'language_training_v1','model':'EleutherAI/pythia-1b-deduped','arms':METHODS,'development_seed':49100,
        'reporting_seeds':SEEDS,'learning_rates':RATES,'development_steps':a.development_steps,'reporting_steps':a.reporting_steps,
        'batch_size':32,'validation_every':100,'systems':systems,'pilot_sha256':sha(root/'training_preflight.json'),
        'optimizer':'plain SGD; no momentum, weight decay or gradient clipping; full-parameter training in eval mode',
        'method_settings':'fixed published/retained defaults; periodic every5updates, damping.5; each method receives same3LR development budget',
        'loss':'answer-only cross-entropy; EOS/padding ignored; identical shared attention precision for every method',
        'selection':'minimum development validation worst-task CE across post-start checkpoints; lower LR breaks ties; no test access',
        'primary_endpoint':'final heldout worst-task answer CE; paired Direct-minus-Uniform and periodic-minus-FA differences',
        'secondary':'all six methods, all three seeds, per-task CE and teacher-forced answer exact match; validation-selected and final results; update/total wall/memory',
        'scope':'bounded-budget full-parameter adaptation, not convergence or standard full-dataset benchmark training',
        'completed_checkpoint_policy':a.completed_checkpoints,'checkpoint_backup':a.checkpoint_backup,
        'checkpoint_slot_budget_bytes':a.slot_budget_bytes,
        'checkpoint_recovery_scope':'Complete compressed active-run model/rule/RNG state in fixed Drive slots' if a.checkpoint_backup=='slots' else
            'Models/rule/RNG remain on VM; Drive contains numerical/provenance records only; VM loss requires independently retained model bytes'}
    freeze(root/'protocol.json',protocol)
    identities={f.name:sha(f) for f in root.glob('*.py')};freeze(root/'campaign_sources.json',identities)
    for name in ('protocol.json','campaign_sources.json'):
        freeze(backup/name,json.loads((root/name).read_text()))
    for stage in (['development','confirmatory'] if a.stage=='development' else ['confirmatory']):
        for directory in (root,backup):retire_marker(directory/f'{stage}_complete.json')
    for name in ['protocol.json','campaign_sources.json','training_preflight.json',*identities]:verified_copy(root/name,backup/name)
    completed={};pairing={}
    def run(method,seed,lr,phase):
        for name,digest in identities.items():
            if sha(root/name)!=digest:raise ValueError('Campaign source changed')
        label=run_label(method,seed,lr,phase);config=root/'configs'/f'{label}.json'
        cfg=configuration(protocol,sha(root/'protocol.json'),method,seed,lr,phase)
        freeze(config,cfg);verified_copy(config,backup/'configs'/config.name)
        out=root/'runs'/label;dest=backup/'runs'/label;metrics=out/'metrics.json'
        if any((directory/'failure.json').exists() for directory in (out,dest)):raise RuntimeError('Retained failure requires diagnosis: '+label)
        if not (metrics.exists() and json.loads(metrics.read_text()).get('complete') is True):
            if (dest/'completed_backup_commit.json').exists():
                raise RuntimeError('Completed run unavailable locally; restore its retained VM model archives before continuing: '+label)
            if not (out/'latest.pt').exists() and (dest/'backup_commit.json').exists():recover_active(root,out,dest,cfg)
        if not (metrics.exists() and json.loads(metrics.read_text()).get('complete')):
            cmd=[str(root/'venv/bin/python'),str(root/'language_training_rebuttal.py'),'--root',str(root),
                '--config',str(config),'--output',str(out),'--backup',str(dest)]
            if (out/'latest.pt').exists():cmd.append('--resume')
            attempt_path=root/'process_attempts'/f'{label}.json';attempts=json.loads(attempt_path.read_text()) if attempt_path.exists() else []
            attempts.append({'start_utc_seconds':time.time(),'complete':False,'resume':'--resume' in cmd})
            atomic_json(attempt_path,attempts);verified_copy(attempt_path,backup/'process_attempts'/attempt_path.name)
            log_path=root/'logs'/f'{label}.log';log_path.parent.mkdir(exist_ok=True)
            print(json.dumps({'start':label,'utc_seconds':time.time()}),flush=True);tick=time.perf_counter()
            with log_path.open('a') as log:child=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
            attempts[-1].update(complete=True,returncode=child.returncode,subprocess_wall_seconds=time.perf_counter()-tick,end_utc_seconds=time.time())
            atomic_json(attempt_path,attempts);verified_copy(attempt_path,backup/'process_attempts'/attempt_path.name);verified_copy(log_path,backup/'logs'/log_path.name)
            if child.returncode:raise RuntimeError('Campaign halts on failed arm/seed: '+label)
        result=validate_result(root,out,cfg,verify_archives=True)
        owner=json.loads((out/'run_identity.json').read_text())
        committed=validate_backup(dest,owner,full=False) if (dest/'backup_commit.json').exists() else None
        if committed is None or committed['metrics_sha256']!=sha(metrics):
            # A diagnosed interruption can leave a complete local result before
            # its final backup commit. Finish persistence without rerunning it.
            with checkpoint_writer(dest,owner,vm_temp=root/'checkpoint_tmp',mode=cfg['checkpoint_backup'],
                    slot_budget_bytes=cfg['checkpoint_slot_budget_bytes']) as store:
                store.publish(out)
            verified_copy(metrics,dest/'metrics.json')
        batch_digest=result['validated_batch_sequence_sha256'];key=(phase,seed)
        if key in pairing and pairing[key]!=batch_digest:raise ValueError('Unpaired arm batches')
        pairing[key]=batch_digest
        completed[label]={'metrics_sha256':sha(metrics),'config_sha256':sha(config),'batch_sequence_sha256':batch_digest}
        if a.completed_checkpoints in ('vm','final-only'):retire_completed(out,dest,policy=a.completed_checkpoints)
        return result
    if a.stage=='development':
        candidates={m:[] for m in METHODS}
        for lr in RATES:
            for method in METHODS:
                d=run(method,49100,lr,'development')
                candidates[method].append({'lr':lr,'score':d['selected_checkpoint']['worst_ce'],
                    'label':run_label(method,49100,lr,'development')})
        selection={'protocol_sha256':sha(root/'protocol.json'),'candidates':candidates,
            'rates':{m:min(candidates[m],key=lambda r:(r['score'],r['lr']))['lr'] for m in METHODS}}
        freeze(root/'selection.json',selection);verified_copy(root/'selection.json',backup/'selection.json')
    else:
        selection=json.loads((root/'selection.json').read_text())
        if selection['protocol_sha256']!=sha(root/'protocol.json') or set(selection['rates'])!=set(METHODS) or set(selection['candidates'])!=set(METHODS):raise ValueError('Selection identity')
        development_completed={}
        for method in METHODS:
            choices=selection['candidates'][method]
            if sorted(c['lr'] for c in choices)!=RATES:raise ValueError('Incomplete tuning')
            for choice in choices:
                label=run_label(method,49100,choice['lr'],'development')
                if choice['label']!=label:raise ValueError('Tuning arm/seed/rate label differs')
                out=root/'runs'/label;config=root/'configs'/f'{label}.json';cfg=json.loads(config.read_text())
                if cfg!=configuration(protocol,sha(root/'protocol.json'),method,49100,choice['lr'],'development'):raise ValueError('Tuning configuration identity')
                d=validate_result(root,out,cfg,verify_archives=True)
                if d['selected_checkpoint']['worst_ce']!=choice['score']:raise ValueError('Selection score differs')
                development_completed[label]={'metrics_sha256':sha(out/'metrics.json'),'config_sha256':sha(config),
                    'batch_sequence_sha256':d['validated_batch_sequence_sha256']}
            if selection['rates'][method]!=min(choices,key=lambda r:(r['score'],r['lr']))['lr']:raise ValueError('Wrong selected LR')
        development_marker=json.loads((root/'development_complete.json').read_text())
        if (development_marker.get('complete') is not True or development_marker['runs']!=development_completed or
            development_marker['protocol_sha256']!=sha(root/'protocol.json') or
            development_marker['sources_sha256']!=sha(root/'campaign_sources.json') or
            development_marker['selection_sha256']!=sha(root/'selection.json')):raise ValueError('Development completion identity')
        for seed in SEEDS:
            order=METHODS[seed%6:]+METHODS[:seed%6]
            for method in order:run(method,seed,selection['rates'][method],'confirmatory')
    if len(completed)!=18:raise ValueError('All18 stage runs required')
    marker=root/f'{a.stage}_complete.json'
    atomic_json(marker,{'complete':True,'runs':completed,'protocol_sha256':sha(root/'protocol.json'),
        'sources_sha256':sha(root/'campaign_sources.json'),'selection_sha256':sha(root/'selection.json'),'utc_seconds':time.time()})
    verified_copy(marker,backup/marker.name)

if __name__=='__main__':main()
