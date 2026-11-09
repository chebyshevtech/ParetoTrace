"""Lossless, verified checkpoint archives for the small persistent Drive quota.

Compression changes storage only. Tensor values, optimizer/RNG states and
source identities are byte-for-byte recoverable; no quantization is used.
"""
import gzip,hashlib,json,os,re,shutil,tempfile
from pathlib import Path

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*2**20),b''):h.update(b)
    return h.hexdigest()

def fsync_path(path):
    fd=os.open(path,os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)

def atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    with tmp.open('w') as f:f.write(json.dumps(value,indent=2,allow_nan=False));f.flush();os.fsync(f.fileno())
    tmp.replace(path);fsync_path(path.parent)

def verified_copy(source,destination):
    source,destination=Path(source),Path(destination);destination.parent.mkdir(parents=True,exist_ok=True)
    expected=sha(source);fd,name=tempfile.mkstemp(prefix=destination.name+'.',suffix='.tmp',dir=destination.parent);os.close(fd);tmp=Path(name)
    try:
        shutil.copy2(source,tmp)
        if sha(tmp)!=expected:raise IOError('Copied file differs')
        fsync_path(tmp);tmp.replace(destination);fsync_path(destination.parent)
    finally:
        if tmp.exists():tmp.unlink()

def archive_file(source,destination):
    source,destination=Path(source),Path(destination)
    destination.parent.mkdir(parents=True,exist_ok=True)
    before=source.stat();original=hashlib.sha256();count=0
    # A unique temporary name leaves the committed old archive intact on failure.
    fd,name=tempfile.mkstemp(prefix=destination.name+'.',suffix='.tmp',dir=destination.parent);os.close(fd);tmp=Path(name)
    try:
        with source.open('rb') as src,tmp.open('wb') as raw:
            with gzip.GzipFile(fileobj=raw,mode='wb',compresslevel=1,mtime=0,filename='') as dest:
                for block in iter(lambda:src.read(8*2**20),b''):
                    original.update(block);count+=len(block);dest.write(block)
        after=source.stat()
        if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns) or count!=before.st_size:
            raise RuntimeError('Checkpoint changed during archival')
        restored=hashlib.sha256();restored_count=0
        with gzip.open(tmp,'rb') as stream:
            for block in iter(lambda:stream.read(8*2**20),b''):restored.update(block);restored_count+=len(block)
        if restored.hexdigest()!=original.hexdigest() or restored_count!=count:raise IOError('Archive verification failed')
        fsync_path(tmp)
        destination=destination.with_name(destination.stem+'_'+original.hexdigest()[:16]+destination.suffix)
        record={'format':'gzip-lossless-v1','source_sha256':original.hexdigest(),'source_bytes':count,
            'archive_sha256':sha(tmp),'archive_bytes':tmp.stat().st_size,'archive_file':destination.name}
        if destination.exists():
            if sha(destination)!=record['archive_sha256']:raise IOError('Immutable archive identity conflict')
        else:tmp.replace(destination)
        fsync_path(destination.parent)
        atomic_json(destination.with_suffix(destination.suffix+'.json'),record)
        return record
    finally:
        if tmp.exists():tmp.unlink()

def restore_file(archive,destination,record):
    archive,destination=Path(archive),Path(destination)
    if record.get('format')!='gzip-lossless-v1' or sha(archive)!=record['archive_sha256']:
        raise IOError('Archive identity differs')
    destination.parent.mkdir(parents=True,exist_ok=True)
    fd,name=tempfile.mkstemp(prefix=destination.name+'.',suffix='.tmp',dir=destination.parent);os.close(fd);tmp=Path(name)
    try:
        with gzip.open(archive,'rb') as src,tmp.open('wb') as dst:shutil.copyfileobj(src,dst,8*2**20)
        if tmp.stat().st_size!=record['source_bytes'] or sha(tmp)!=record['source_sha256']:
            raise IOError('Restored checkpoint identity differs')
        tmp.replace(destination)
    finally:
        if tmp.exists():tmp.unlink()

def retire_archive(archive,expected_source_sha256):
    archive=Path(archive);side=archive.with_suffix(archive.suffix+'.json')
    if not archive.exists():return
    record=json.loads(side.read_text())
    match=re.fullmatch(r'.+\.pt_([0-9a-f]{16})\.gz',archive.name)
    if (match is None or record.get('archive_file')!=archive.name or
        match.group(1)!=record['source_sha256'][:16] or
        record['source_sha256']!=expected_source_sha256 or sha(archive)!=record['archive_sha256']):
        raise IOError('Refuse to retire a checkpoint with changed identity')
    archive.unlink();side.unlink()

def prune_run_archives(backup):
    """After a durable commit, remove only unreferenced generations owned by it."""
    backup=Path(backup)
    if not (backup/'run_identity.json').is_file():raise ValueError('Missing run ownership')
    commit=json.loads((backup/'backup_commit.json').read_text())
    if commit.get('storage_format')!='gzip-lossless-v1':raise ValueError('Unknown committed format')
    validate_dependencies(backup,commit)
    retained=set()
    for rel,record in commit['checkpoint_archives'].items():
        parent=Path(rel).parent;name=Path(record['archive_file'])
        if parent not in (Path('.'),Path('selected')) or name.name!=str(name):raise ValueError('Invalid owned reference')
        p=backup/parent/name
        if sha(p)!=record['archive_sha256']:raise IOError('Committed generation differs; refuse pruning')
        retained.add(p)
    for parent in [backup,backup/'selected']:
        if not parent.exists():continue
        for p in parent.glob('*.pt_*.gz'):
            if p in retained:continue
            side=p.with_suffix('.gz.json')
            if side.exists():
                record=json.loads(side.read_text());retire_archive(p,record['source_sha256'])
            else:
                # A crash can occur after archive publication but before its
                # sidecar. Validate the embedded content hash before retirement.
                match=re.fullmatch(r'.+\.pt_([0-9a-f]{16})\.gz',p.name)
                if match is None:continue
                digest=hashlib.sha256()
                with gzip.open(p,'rb') as f:
                    for b in iter(lambda:f.read(8*2**20),b''):digest.update(b)
                if digest.hexdigest()[:16]!=match.group(1):raise IOError('Orphan generation identity differs')
                p.unlink()
        for p in parent.glob('*.pt.gz.*.tmp'):p.unlink()
        fsync_path(parent)

def validate_dependencies(backup,commit):
    backup=Path(backup);owner=backup/'run_identity.json'
    if sha(owner)!=commit['run_identity_sha256']:raise IOError('Committed ownership differs')
    rel=Path(commit['metrics_file'])
    if rel.parent!=Path('generations') or '..' in rel.parts:raise ValueError('Invalid metrics reference')
    if sha(backup/rel)!=commit['metrics_sha256']:raise IOError('Committed metrics differ')
    metrics=json.loads((backup/rel).read_text());identity=json.loads(owner.read_text())
    if metrics['config']!=identity['config'] or metrics['identity']!=identity['identity'] or metrics['steps']!=commit['steps']:
        raise IOError('Committed run metadata differs')
    for block in metrics['update_blocks']:
        rel=Path(block['file'])
        if rel.parent!=Path('updates') or '..' in rel.parts:raise ValueError('Invalid event block reference')
        if sha(backup/rel)!=block['sha256']:raise IOError('Committed event block differs')

def _restore_run(backup,output):
    backup,output=Path(backup),Path(output)
    if output.exists() and any(output.iterdir()):raise ValueError('Restore requires an empty destination')
    commit=json.loads((backup/'backup_commit.json').read_text())
    if commit.get('storage_format')!='gzip-lossless-v1':raise ValueError('Unsupported backup format')
    validate_dependencies(backup,commit)
    refs=commit['checkpoint_archives']
    for rel,record in refs.items():
        p=Path(rel)
        if p.is_absolute() or '..' in p.parts or (p!=Path('latest.pt') and p.parent!=Path('selected')):
            raise ValueError('Invalid checkpoint reference')
        filename=Path(record['archive_file'])
        if filename.name!=str(filename):raise ValueError('Invalid archive filename')
        restore_file(backup/p.parent/filename,output/rel,record)
    for p in backup.rglob('*.json'):
        if p.name.endswith('.gz.json'):continue
        target=output/p.relative_to(backup);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
    metrics=Path(commit['metrics_file'])
    if metrics.parent!=Path('generations') or '..' in metrics.parts:raise ValueError('Invalid metrics reference')
    shutil.copy2(backup/metrics,output/'metrics.json')
    if sha(output/'metrics.json')!=commit['metrics_sha256']:raise ValueError('Metrics identity differs')
    if sha(output/'latest.pt')!=commit['latest_sha256']:raise ValueError('Latest checkpoint differs')
    return commit

def restore_run(backup,output):
    backup,output=Path(backup),Path(output)
    if output.exists() and any(output.iterdir()):raise ValueError('Restore requires an empty destination')
    output.parent.mkdir(parents=True,exist_ok=True)
    stage=Path(tempfile.mkdtemp(prefix=output.name+'.restore-',dir=output.parent))
    try:
        commit=_restore_run(backup,stage)
        for p in stage.rglob('*'):
            if p.is_file():fsync_path(p)
        for p in sorted((x for x in stage.rglob('*') if x.is_dir()),key=lambda x:len(x.parts),reverse=True):fsync_path(p)
        fsync_path(stage)
        if output.exists():output.rmdir()
        stage.replace(output);fsync_path(output.parent)
        return commit
    finally:
        if stage.exists():shutil.rmtree(stage)
