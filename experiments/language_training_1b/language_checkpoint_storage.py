"""Explicit language-run persistence policies; no optimizer or training changes.

`slots` keeps verified compressed recovery models in four reusable Drive files.
`metadata` keeps models only on the VM and mirrors numerical/provenance JSON;
it cannot recover model/rule/RNG state after VM loss. There is no auto-fallback.
"""
import copy, hashlib, json
from pathlib import Path
from checkpoint_archive import atomic_json, verified_copy, sha, validate_dependencies, restore_run as restore_legacy
from drive_checkpoint_slots import FORMAT as SLOT_FORMAT, SlotWriter, run_lease, validate_commit as validate_slots, restore_run as restore_slots

METADATA_FORMAT='language-metadata-only-v1'
METADATA_RETENTION='language-metadata-retention-v1'
FINAL_ONLY_RETENTION='language-final-only-retention-v1'

def read(path):return json.loads(Path(path).read_text())
def owner_digest(owner):return hashlib.sha256(json.dumps(owner,sort_keys=True,allow_nan=False).encode()).hexdigest()
def owned_path(root,relative):
    rel=Path(relative)
    if rel.is_absolute() or '..' in rel.parts:raise ValueError('Unsafe storage reference')
    p=root/rel
    if any(q.is_symlink() for q in [p,*p.parents] if q!=root and root in q.parents):raise ValueError('Symlink in owned storage')
    return p

def numerical_files(metrics):
    files={b['file']:b['sha256'] for b in metrics['update_blocks']}
    if len(files)!=len(metrics['update_blocks']):raise ValueError('Duplicate event blocks')
    for rel in files:
        if Path(rel).parent!=Path('updates') or Path(rel).suffix!='.json':raise ValueError('Unsafe event block')
    audits={f'audits/step{r["after_updates"]:08d}.json':r for r in metrics.get('native_audits',[])}
    if len(audits)!=len(metrics.get('native_audits',[])):raise ValueError('Duplicate audit steps')
    return files,audits

def validate_metadata(backup,owner):
    backup=Path(backup);commit=read(backup/'backup_commit.json')
    if commit.get('storage_format')!=METADATA_FORMAT:raise ValueError('Not a metadata-only backup')
    if read(backup/'run_identity.json')!=owner:raise ValueError('Backup ownership differs')
    validate_dependencies(backup,commit)
    metrics=read(owned_path(backup,commit['metrics_file']))
    if commit['run_complete'] is not (metrics.get('complete') is True) or commit['selected_checkpoint']!=metrics['selected_checkpoint']:
        raise ValueError('Metadata completion/selection differs')
    refs=commit['checkpoint_sources'];selected=metrics['selected_checkpoint']
    if set(refs)!=({'latest.pt',selected['file']} if selected is not None else {'latest.pt'}):raise ValueError('Metadata checkpoint coverage')
    if refs['latest.pt']['source_sha256']!=commit['latest_sha256']:raise ValueError('Metadata latest identity')
    if selected is not None and (Path(selected['file']).parent!=Path('selected') or refs[selected['file']]['source_sha256']!=selected['sha256']):raise ValueError('Metadata selected identity')
    for rel,entry in refs.items():
        if rel!='latest.pt' and (Path(rel).parent!=Path('selected') or Path(rel).suffix!='.pt'):raise ValueError('Invalid model reference')
        owned_path(backup,rel)
        if type(entry['source_bytes']) is not int or entry['source_bytes']<=0:raise ValueError('Invalid model size')
        if not isinstance(entry['source_sha256'],str) or len(entry['source_sha256'])!=64 or any(c not in '0123456789abcdef' for c in entry['source_sha256']):raise ValueError('Invalid model hash')
    _,audits=numerical_files(metrics)
    for rel,value in audits.items():
        if read(owned_path(backup,rel))!=value:raise ValueError('Committed audit artifact differs')
    return commit

class MetadataWriter:
    def __init__(self,backup,owner):
        self.backup=Path(backup);self.owner=copy.deepcopy(owner);self.guard=None
    def __enter__(self):
        self.backup.mkdir(parents=True,exist_ok=True);self.guard=run_lease(self.backup);self.guard.__enter__()
        try:
            p=self.backup/'run_identity.json'
            if p.exists():
                if read(p)!=self.owner:raise ValueError('Backup ownership differs')
            else:
                if any(p.name!='.checkpoint_slots.lock' for p in self.backup.iterdir()):raise ValueError('Nonempty unowned metadata backup')
                atomic_json(p,self.owner)
            if (self.backup/'backup_commit.json').exists():
                previous=validate_metadata(self.backup,self.owner)
                if previous['run_complete']:raise ValueError('Completed metadata backup cannot resume publication')
            return self
        except BaseException:
            self.guard.__exit__(*__import__('sys').exc_info());self.guard=None;raise
    def __exit__(self,*args):
        guard,self.guard=self.guard,None
        if guard is not None:return guard.__exit__(*args)
    def publish(self,run):
        if self.guard is None:raise ValueError('Publication requires the run lease')
        run=Path(run)
        if run.resolve()==self.backup.resolve() or run.resolve() in self.backup.resolve().parents or self.backup.resolve() in run.resolve().parents:raise ValueError('Run and backup must be disjoint')
        if any((p/'failure.json').exists() for p in (run,self.backup)):raise ValueError('Failed run requires diagnosis')
        if read(run/'run_identity.json')!=self.owner:raise ValueError('Local ownership differs')
        previous=validate_metadata(self.backup,self.owner) if (self.backup/'backup_commit.json').exists() else None
        if previous and previous['run_complete']:raise ValueError('Completed metadata backup is terminal')
        metrics=read(run/'metrics.json');digest=sha(run/'metrics.json')
        if {'config':metrics['config'],'identity':metrics['identity']}!=self.owner:raise ValueError('Metrics ownership differs')
        if type(metrics['steps']) is not int or metrics['steps']<0 or previous and metrics['steps']<previous['steps']:raise ValueError('Checkpoint step regressed')
        selected=metrics['selected_checkpoint'];refs={}
        if metrics.get('complete') is True and selected is None:raise ValueError('Complete run needs a selected model')
        from retire_completed_language import owned_model
        for rel in ['latest.pt']+([selected['file']] if selected is not None else []):
            p=owned_model(run,rel);refs[rel]={'source_sha256':sha(p),'source_bytes':p.stat().st_size}
            if refs[rel]['source_bytes']<=0:raise ValueError('Empty model source')
        if selected is not None and (Path(selected['file']).parent!=Path('selected') or refs[selected['file']]['source_sha256']!=selected['sha256']):raise ValueError('Selected source differs')
        files,audits=numerical_files(metrics)
        for rel,value in audits.items():
            if read(owned_path(run,rel))!=value:raise ValueError('Local audit differs')
            files[rel]=sha(owned_path(run,rel))
        protected={}
        if previous:
            old=read(self.backup/previous['metrics_file']);protected,old_audits=numerical_files(old)
            protected.update({rel:sha(owned_path(self.backup,rel)) for rel in old_audits})
        for rel,expected in files.items():
            source,target=owned_path(run,rel),owned_path(self.backup,rel)
            if sha(source)!=expected:raise ValueError('Local numerical artifact differs')
            if rel in protected and protected[rel]!=expected:raise ValueError('Committed numerical artifact changed')
            if not target.exists() or sha(target)!=expected:verified_copy(source,target)
        generation='generations/metrics_'+digest+'.json';target=owned_path(self.backup,generation)
        if target.exists():
            if sha(target)!=digest:raise ValueError('Immutable metrics conflict')
        else:verified_copy(run/'metrics.json',target)
        if sha(run/'metrics.json')!=digest or any(sha(owned_model(run,rel))!=entry['source_sha256'] for rel,entry in refs.items()):raise ValueError('Sources changed before commit')
        commit={'storage_format':METADATA_FORMAT,'run_complete':metrics.get('complete') is True,'steps':metrics['steps'],
            'selected_checkpoint':selected,'checkpoint_sources':refs,'latest_sha256':refs['latest.pt']['source_sha256'],
            'run_identity_sha256':sha(self.backup/'run_identity.json'),'metrics_file':generation,'metrics_sha256':digest,
            'recovery_scope':'Numerical/provenance records only; active model, optimizer, rule and RNG recovery requires surviving VM checkpoint bytes.'}
        atomic_json(self.backup/'backup_commit.json',commit)
        return validate_metadata(self.backup,self.owner)

def writer(backup,owner,*,vm_temp,mode,slot_budget_bytes=None):
    if mode=='slots':
        if type(slot_budget_bytes) is not int or slot_budget_bytes<=0:raise ValueError('Slots require an explicit measured byte budget')
        return SlotWriter(backup,owner,vm_temp=vm_temp,slot_budget_bytes=slot_budget_bytes)
    if mode=='metadata':
        if slot_budget_bytes is not None:raise ValueError('Metadata-only policy must omit a model-slot budget')
        return MetadataWriter(backup,owner)
    raise ValueError('Choose an explicit slots or metadata checkpoint policy')

def validate_backup(backup,owner,full=True):
    fmt=read(Path(backup)/'backup_commit.json')['storage_format']
    if fmt==SLOT_FORMAT:return validate_slots(backup,owner,full=full)
    if fmt==METADATA_FORMAT:
        with run_lease(backup):return validate_metadata(backup,owner)
    if fmt=='gzip-lossless-v1':
        commit=read(Path(backup)/'backup_commit.json');validate_dependencies(backup,commit)
        if read(Path(backup)/'run_identity.json')!=owner:raise ValueError('Backup ownership differs')
        return commit
    raise ValueError('Unknown backup format')

def restore_backup(backup,output,owner):
    fmt=read(Path(backup)/'backup_commit.json')['storage_format']
    if fmt==SLOT_FORMAT:return restore_slots(backup,output,owner)
    if fmt==METADATA_FORMAT:
        validate_backup(backup,owner)
        raise RuntimeError('Metadata-only backup cannot recover model/rule/RNG state after VM checkpoint loss; restore verified local model archives before continuing')
    if fmt=='gzip-lossless-v1':return restore_legacy(backup,output)
    raise ValueError('Unknown backup format')

def completed_root(out):
    out=Path(out)
    parent=out.parent.parent/'completed_checkpoints';root=parent/out.name
    if parent.is_symlink() or root.is_symlink():raise ValueError('Unexpected completed archive location')
    return root

def validate_vm_retention(out,proof):
    from retire_completed_language import verify_archive
    out=Path(out);root=completed_root(out);owner=read(out/'run_identity.json')
    if proof.get('storage_format')==FINAL_ONLY_RETENTION:return validate_final_retention(out,proof)
    if proof.get('storage_format')=='drive-fixed-slots-retention-v1':
        from drive_checkpoint_slots import validate_retention
        validate_retention(root,proof,owner)
    elif proof.get('storage_format')==METADATA_RETENTION:
        if proof.get('complete') is not True or proof['owner_identity_sha256']!=owner_digest(owner) or proof['config']!=owner['config'] or proof['metrics_sha256']!=sha(out/'metrics.json'):raise ValueError('Metadata VM retention identity differs')
        previous=proof['previous_backup_commit']
        if previous['storage_format']!=METADATA_FORMAT or not previous['run_complete'] or previous['metrics_sha256']!=proof['metrics_sha256']:raise ValueError('Metadata VM retention commit differs')
        if read(root/'retention_owner.json')!={'owner_identity_sha256':owner_digest(owner),'metrics_sha256':proof['metrics_sha256']}:raise ValueError('VM archive ownership differs')
        if set(proof['local_checkpoint_archives'])!=set(previous['checkpoint_sources']):raise ValueError('VM archive coverage differs')
        for rel,record in proof['local_checkpoint_archives'].items():
            old=previous['checkpoint_sources'][rel]
            if record['source_sha256']!=old['source_sha256'] or record['source_bytes']!=old['source_bytes'] or Path(record['archive_file']).name!=record['archive_file']:raise ValueError('VM archive source differs')
            verify_archive(owned_path(root,str(Path(rel).parent/record['archive_file'])),record,full=True)
    else:raise ValueError('Unknown VM retention format')
    if read(root/'slot_retention.json')!=proof:raise ValueError('VM retention manifest differs')
    return root

def final_only_receipt(out,pair):
    """Describe an explicit terminal policy; this does not delete any bytes."""
    out=Path(out);metrics=read(out/'metrics.json');owner=read(out/'run_identity.json')
    if metrics.get('complete') is not True or metrics['config'].get('completed_checkpoint_policy')!='final-only':
        raise ValueError('Final-only retention requires the frozen completed-run policy')
    selected=metrics['selected_checkpoint'];record=pair['local_checkpoint_archives'][selected['file']]
    validation=[event for event in metrics['history'] if event['after_updates']==selected['after_updates']]
    if len(validation)!=1:raise ValueError('Selected evaluation coverage differs')
    return {'storage_format':FINAL_ONLY_RETENTION,'complete':True,'config':metrics['config'],
        'owner_identity_sha256':owner_digest(owner),'metrics_sha256':sha(out/'metrics.json'),
        'verified_pair_retention':pair,'local_checkpoint_archives':{'latest.pt':pair['local_checkpoint_archives']['latest.pt']},
        'selected_checkpoint':selected,
        'selected_model':{'file':selected['file'],'source_sha256':record['source_sha256'],'source_bytes':record['source_bytes'],
            'verified_archive_sha256':record['archive_sha256'],'verified_archive_bytes':record['archive_bytes'],
            'lossless_archive_verified_before_retirement':True,'model_bytes_retained':False},
        'selected_evaluation':{'validation_metrics_sha256':owner_digest(validation[0]['metrics']),
            'heldout_test_metrics_sha256':owner_digest(metrics['validation_selected_test']) if metrics.get('heldout_test_evaluated') is True else None},
        'policy':'Final model checkpoint retained losslessly; selected model bytes intentionally not retained after both archives were verified.',
        'limitation':'All validation-selected scalar/evaluation/provenance records remain; the selected model cannot be re-evaluated without an independently retained copy. VM loss can still destroy final model bytes.'}

def validate_final_retention(out,receipt):
    from retire_completed_language import verify_archive
    out=Path(out);root=completed_root(out);owner=read(out/'run_identity.json');metrics=read(out/'metrics.json')
    pair=receipt['verified_pair_retention']
    if receipt!=final_only_receipt(out,pair):raise ValueError('Final-only retention receipt differs')
    if pair.get('complete') is not True or pair['config']!=metrics['config'] or pair['metrics_sha256']!=sha(out/'metrics.json'):
        raise ValueError('Original pair-verification identity differs')
    previous=pair['previous_backup_commit']
    if pair['storage_format']=='drive-fixed-slots-retention-v1':
        from drive_checkpoint_slots import validate_retention_manifest
        validate_retention_manifest(pair,owner)
        expected_owner={'owner_identity_sha256':pair['owner_identity_sha256'],'active_commit_id':pair['active_commit_id']}
        refs=previous['checkpoint_archives']
    elif pair['storage_format']==METADATA_RETENTION:
        if pair['owner_identity_sha256']!=owner_digest(owner) or previous['storage_format']!=METADATA_FORMAT or previous['run_complete'] is not True:
            raise ValueError('Original metadata retention identity differs')
        expected_owner={'owner_identity_sha256':owner_digest(owner),'metrics_sha256':pair['metrics_sha256']}
        refs=previous['checkpoint_sources']
    else:raise ValueError('Unknown original pair-verification format')
    selected=metrics['selected_checkpoint'];records=pair['local_checkpoint_archives']
    if previous['metrics_sha256']!=pair['metrics_sha256'] or previous['selected_checkpoint']!=selected or set(records)!=set(refs) or set(refs)!={'latest.pt',selected['file']}:
        raise ValueError('Original pair-verification coverage differs')
    for rel,record in records.items():
        if record.get('format')!='gzip-lossless-v1' or any(type(record[k]) is not int or record[k]<=0 for k in ('source_bytes','archive_bytes')):
            raise ValueError('Invalid historical archive verification record')
        if any(not isinstance(record[k],str) or len(record[k])!=64 or any(c not in '0123456789abcdef' for c in record[k]) for k in ('source_sha256','archive_sha256')):
            raise ValueError('Invalid historical archive verification hash')
        if record['source_sha256']!=refs[rel]['source_sha256'] or record['source_bytes']!=refs[rel]['source_bytes'] or Path(record['archive_file']).name!=record['archive_file']:
            raise ValueError('Original verified source differs')
    if records[selected['file']]['source_sha256']!=selected['sha256'] or records['latest.pt']['source_sha256']!=previous['latest_sha256']:
        raise ValueError('Final/selected source identity differs')
    if read(root/'retention_owner.json')!=expected_owner or read(root/'slot_retention.json')!=pair or read(root/'final_only_retention.json')!=receipt:
        raise ValueError('Final-only VM ownership/receipt differs')
    final=records['latest.pt'];verify_archive(owned_path(root,final['archive_file']),final,full=True)
    return root
