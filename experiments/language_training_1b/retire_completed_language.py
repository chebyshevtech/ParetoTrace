"""Retain completed model bytes losslessly on the VM; free redundant Drive copies.

All numerical records remain on Drive. The active run is never eligible. A VM
reset can still lose completed models until an independent local download is
verified; the retention manifest states that limitation explicitly.
"""
import argparse, gzip, hashlib, json, os
from pathlib import Path
from checkpoint_archive import archive_file, atomic_json, fsync_path, sha, retire_archive, verified_copy, validate_dependencies


def verify_archive(path, record, full=True):
    path=Path(path)
    if record.get('format')!='gzip-lossless-v1' or path.is_symlink() or path.name!=record['archive_file'] or path.stat().st_size!=record['archive_bytes'] or sha(path)!=record['archive_sha256']:
        raise IOError('Archive bytes differ')
    if full:
        h=hashlib.sha256();count=0
        with gzip.open(path,'rb') as stream:
            for block in iter(lambda:stream.read(8*2**20),b''):
                h.update(block);count+=len(block)
        if count!=record['source_bytes'] or h.hexdigest()!=record['source_sha256']:
            raise IOError('Restored model bytes differ')


def owned_model(out, rel):
    rel=Path(rel)
    if rel.is_absolute() or '..' in rel.parts or (rel!=Path('latest.pt') and (rel.parent!=Path('selected') or rel.suffix!='.pt')):
        raise ValueError('Invalid completed model reference')
    p=out/rel
    if p.is_symlink() or (out/'selected').is_symlink() or p.resolve().parent not in (out.resolve(),(out/'selected').resolve()):
        raise ValueError('Unexpected model location')
    return p


def _retire_legacy_completed(out, backup):
    out,backup=Path(out),Path(backup)
    if out.resolve()==backup.resolve() or out.resolve() in backup.resolve().parents or backup.resolve() in out.resolve().parents:
        raise ValueError('Output and backup must be disjoint')
    if (backup/'selected').is_symlink():raise ValueError('Unexpected backup model location')
    metrics=json.loads((out/'metrics.json').read_text());owner=json.loads((out/'run_identity.json').read_text())
    if metrics.get('complete') is not True or any((p/'failure.json').exists() for p in (out,backup)):raise ValueError('Incomplete/failed runs retain all recovery state')
    if {'config':metrics['config'],'identity':metrics['identity']}!=owner:raise ValueError('Run ownership differs')
    commit_path=backup/'backup_commit.json'
    if not commit_path.exists():commit_path=backup/'completed_backup_commit.json'
    commit=json.loads(commit_path.read_text())
    validate_dependencies(backup,commit)
    if commit['metrics_sha256']!=sha(out/'metrics.json') or commit['run_identity_sha256']!=sha(out/'run_identity.json'):
        raise ValueError('Only the committed completed run may be archived')
    selected=metrics['selected_checkpoint'];refs=commit['checkpoint_archives']
    if Path(selected['file']).parent!=Path('selected'):raise ValueError('Selected model must remain separate from latest')
    if (commit.get('storage_format')!='gzip-lossless-v1' or commit['selected_checkpoint']!=selected or
            set(refs)!={'latest.pt',selected['file']} or refs[selected['file']]['source_sha256']!=selected['sha256'] or
            refs['latest.pt']['source_sha256']!=commit['latest_sha256']):
        raise ValueError('Completed checkpoint coverage differs')
    previous=out/'completed_checkpoint_retention.json'
    prior=json.loads(previous.read_text()) if previous.exists() else None
    if prior is not None and (prior.get('complete') is not True or prior['metrics_sha256']!=sha(out/'metrics.json') or
            prior['config']!=metrics['config'] or prior['previous_backup_commit']!=commit):
        raise ValueError('Previous retention identity differs')
    records=prior.get('local_checkpoint_archives',{}) if prior else {}
    if records and set(records)!=set(refs):raise ValueError('Previous archive coverage differs')
    # Check every existing source before any raw model or remote archive is removed.
    for rel,entry in refs.items():
        owned_model(out,rel)
        name=Path(entry['archive_file'])
        if name.name!=str(name):raise ValueError('Invalid committed archive path')
        remote=backup/Path(rel).parent/name
        side=remote.with_suffix(remote.suffix+'.json')
        if remote.exists():
            verify_archive(remote,entry,full=False)
            if side.is_symlink() or json.loads(side.read_text())!=entry:raise ValueError('Committed Drive sidecar differs')
        elif prior is None:raise ValueError('Committed Drive archive missing before retirement')
        elif side.exists() and (side.is_symlink() or json.loads(side.read_text())!=entry):raise ValueError('Retired Drive sidecar differs')
    for rel,entry in refs.items():
        p=owned_model(out,rel)
        if rel in records:
            record=records[rel]
            if record['source_sha256']!=entry['source_sha256']:raise ValueError('Previous local archive identity differs')
            verify_archive(p.parent/record['archive_file'],record)
        else:
            if sha(p)!=entry['source_sha256']:raise ValueError('VM model differs from committed Drive copy')
            record=archive_file(p,p.with_suffix(p.suffix+'.gz'))
            if record['source_sha256']!=entry['source_sha256']:raise ValueError('Archived model differs')
            records[rel]=record
    retention={'complete':True,'metrics_sha256':sha(out/'metrics.json'),'config':metrics['config'],
        'local_checkpoint_archives':records,'previous_backup_commit':commit,
        'policy':'lossless verified model archives on VM; numerical results and provenance on Drive; active run recovery retained on Drive',
        'limitation':'completed model archives can be lost on VM reset until independently downloaded; numerical results persist'}
    # Publish the complete verified VM archive map before retiring any copy.
    atomic_json(previous,retention);verified_copy(previous,backup/previous.name)
    for rel,entry in refs.items():
        p=owned_model(out,rel)
        if p.exists():
            if sha(p)!=records[rel]['source_sha256']:raise ValueError('Raw model changed before retirement')
            p.unlink();fsync_path(p.parent)
        remote=backup/Path(rel).parent/entry['archive_file']
        retire_archive(remote,entry['source_sha256'])
        # A crash between the archive and sidecar unlinks must also be resumable.
        side=remote.with_suffix(remote.suffix+'.json')
        if not remote.exists() and side.exists():
            if side.is_symlink() or json.loads(side.read_text())!=entry:raise ValueError('Retired Drive sidecar differs')
            side.unlink()
        fsync_path(remote.parent)
    if (backup/'backup_commit.json').exists():
        os.replace(backup/'backup_commit.json',backup/'completed_backup_commit.json');fsync_path(backup)
    return retention


def _retire_pair_completed(out,backup):
    """Retain VM archives before removing VM raw models or truncating Drive slots."""
    from language_checkpoint_storage import (SLOT_FORMAT,METADATA_FORMAT,METADATA_RETENTION,completed_root,
        validate_metadata,validate_vm_retention,owner_digest,run_lease)
    out,backup=Path(out),Path(backup)
    commit_path=backup/'backup_commit.json'
    if not commit_path.exists() or json.loads(commit_path.read_text()).get('storage_format')=='gzip-lossless-v1':
        return _retire_legacy_completed(out,backup)
    commit=json.loads(commit_path.read_text());owner=json.loads((out/'run_identity.json').read_text())
    vm=completed_root(out)
    if vm.is_symlink() or out.is_symlink() or backup.is_symlink():raise ValueError('Unexpected storage location')
    roots=[out.resolve(),backup.resolve(),vm.resolve()]
    if any(a==b or a in b.parents or b in a.parents for i,a in enumerate(roots) for b in roots[i+1:]):
        raise ValueError('Run, backup and VM archives must be disjoint')
    if commit['storage_format']==SLOT_FORMAT:
        from drive_checkpoint_slots import retire_completed as retire_slots
        retire_slots(backup,out,owner,vm_archive_root=vm)
        retention=json.loads((backup/'slot_retention.json').read_text())
    elif commit['storage_format']==METADATA_FORMAT:
        with run_lease(backup):
            commit=validate_metadata(backup,owner)
            if not commit['run_complete'] or commit['metrics_sha256']!=sha(out/'metrics.json') or any((p/'failure.json').exists() for p in (out,backup)):
                raise ValueError('Only the committed completed run may be archived')
            if out.resolve()==backup.resolve() or out.resolve() in backup.resolve().parents or backup.resolve() in out.resolve().parents:
                raise ValueError('Run and backup must be disjoint')
            vm.mkdir(parents=True,exist_ok=True);owner_file=vm/'retention_owner.json'
            expected_owner={'owner_identity_sha256':owner_digest(owner),'metrics_sha256':commit['metrics_sha256']}
            if owner_file.exists():
                if json.loads(owner_file.read_text())!=expected_owner:raise ValueError('VM archive ownership differs')
            else:
                if any(vm.iterdir()):raise ValueError('Nonempty unowned VM archive directory')
                atomic_json(owner_file,expected_owner)
            proof=vm/'slot_retention.json'
            if proof.exists():
                retention=json.loads(proof.read_text())
                if retention['previous_backup_commit']!=commit:raise ValueError('VM retention commit differs')
                validate_vm_retention(out,retention)
            else:
                records={}
                for rel,entry in commit['checkpoint_sources'].items():
                    source=owned_model(out,rel)
                    if sha(source)!=entry['source_sha256'] or source.stat().st_size!=entry['source_bytes']:raise ValueError('VM model differs from committed metadata')
                    record=archive_file(source,vm/(rel+'.gz'))
                    verify_archive(vm/Path(rel).parent/record['archive_file'],record)
                    records[rel]=record
                retention={'storage_format':METADATA_RETENTION,'complete':True,'owner_identity_sha256':owner_digest(owner),
                    'config':owner['config'],'metrics_sha256':commit['metrics_sha256'],'previous_backup_commit':commit,
                    'local_checkpoint_archives':records,'vm_archive_root_at_retirement':str(vm.resolve()),
                    'policy':'Verified lossless completed models on VM; Drive holds numerical/provenance records only.',
                    'limitation':'VM reset loses models unless an independent model copy has been verified; metadata-only Drive backup cannot restore model/rule/RNG state.'}
                atomic_json(proof,retention)
            validate_vm_retention(out,retention)
            verified_copy(proof,backup/'slot_retention.json')
    else:raise ValueError('Unknown checkpoint storage format')
    # The small projected manifest keeps the campaign validator independent of
    # mutable Drive slot names. Archives live in a separate, relocatable VM tree.
    validate_vm_retention(out,retention)
    atomic_json(out/'completed_checkpoint_retention.json',retention)
    verified_copy(out/'completed_checkpoint_retention.json',backup/'completed_checkpoint_retention.json')
    for rel,record in retention['local_checkpoint_archives'].items():
        source=owned_model(out,rel)
        if source.exists():
            if sha(source)!=record['source_sha256']:raise ValueError('Raw model changed before retirement')
            source.unlink();fsync_path(source.parent)
    return retention


def retire_completed(out,backup,policy=None):
    """Apply the frozen pair or final-only policy to a completed run."""
    from language_checkpoint_storage import (FINAL_ONLY_RETENTION,SLOT_FORMAT,completed_root,
        final_only_receipt,validate_final_retention,validate_backup)
    out,backup=Path(out),Path(backup);owner=json.loads((out/'run_identity.json').read_text())
    if any((p/'failure.json').exists() for p in (out,backup)):raise ValueError('Failed runs require diagnosis before retention changes')
    configured=owner['config'].get('completed_checkpoint_policy','vm')
    policy=configured if policy is None else policy
    if policy!=configured:raise ValueError('Completed model policy differs from frozen configuration')
    if policy=='vm':return _retire_pair_completed(out,backup)
    if policy!='final-only':raise ValueError('Retirement requires vm or final-only policy')
    projected=out/'completed_checkpoint_retention.json'
    receipt=json.loads(projected.read_text()) if projected.exists() else None
    if receipt is not None and receipt.get('storage_format')==FINAL_ONLY_RETENTION:
        vm=validate_final_retention(out,receipt)
        current=validate_backup(backup,owner)
        previous=receipt['verified_pair_retention']['previous_backup_commit']
        canonical=current.get('retired_from_commit') if current['storage_format']==SLOT_FORMAT else current
        if canonical!=previous:raise ValueError('Final-only receipt differs from committed completion')
    else:
        pair=_retire_pair_completed(out,backup)  # verifies BOTH archives before raw retirement
        receipt=final_only_receipt(out,pair);vm=completed_root(out)
        atomic_json(vm/'final_only_retention.json',receipt)
        atomic_json(projected,receipt)
        validate_final_retention(out,receipt)
    # Publish the intentional selected-model nonretention before any selected
    # archive unlink. These are VM files; Drive model slots remain in-place.
    verified_copy(projected,backup/'completed_checkpoint_retention.json')
    selected=receipt['selected_checkpoint']['file']
    record=receipt['verified_pair_retention']['local_checkpoint_archives'][selected]
    archive=vm/Path(selected).parent/record['archive_file'];side=archive.with_suffix(archive.suffix+'.json')
    if archive.exists():
        verify_archive(archive,record,full=True)
        if side.is_symlink() or json.loads(side.read_text())!=record:raise ValueError('Selected VM archive sidecar differs')
        archive.unlink();fsync_path(archive.parent)
    if side.exists():
        if side.is_symlink() or json.loads(side.read_text())!=record:raise ValueError('Selected VM archive sidecar differs')
        side.unlink();fsync_path(side.parent)
    validate_final_retention(out,receipt)
    return receipt


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--backup',type=Path,required=True)
    a=p.parse_args();print(json.dumps(retire_completed(a.output,a.backup),indent=2))
