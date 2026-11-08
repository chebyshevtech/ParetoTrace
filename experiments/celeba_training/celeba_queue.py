"""Frozen, resumable campaign order; no selection on reporting test outcomes."""
import argparse,hashlib,json,subprocess,sys,time,shutil,uuid
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--stage',choices=['development','confirmatory'],required=True);args=p.parse_args()
def write_json(path,obj):
 tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(obj,indent=2,allow_nan=False));tmp.replace(path)
root=Path(__file__).parent;protocol=json.loads((root/'celeba_protocol.json').read_text())
names=['celeba_training.py','simplex_qp.py','preflight.py','celeba_protocol.json','celeba_queue.py','run_campaign.py']
identity={n:hashlib.sha256((root/n).read_bytes()).hexdigest() for n in names}
identity['dataset']=hashlib.sha256((root/'data/celeba_manifest.json').read_bytes()).hexdigest()
freeze=root/'campaign_identity.json'
if freeze.exists():
 if json.loads(freeze.read_text())!=identity:raise RuntimeError('Frozen campaign changed: choose a new campaign, do not mix runs')
else:write_json(freeze,identity)
out=root/'runs';out.mkdir(exist_ok=True)
def backup():
 drive=Path('/content/drive/MyDrive')
 if drive.is_dir():
  dest=drive/'ParetoTrace/rebuttal_20261107/celeba_v2';dest.mkdir(parents=True,exist_ok=True)
  for f in list(out.glob('*'))+[freeze,root/'celeba_protocol.json',root/'data/celeba_manifest.json',root/'selected_learning_rates.json',root/'selection_lock.json',root/'development_outcomes.json']:
   if f.is_file() and f.suffix in ('.json','.pt'):
    target=dest/f.name
    if not target.exists() or target.stat().st_size!=f.stat().st_size or target.stat().st_mtime<f.stat().st_mtime:
     tmp=target.with_suffix(target.suffix+'.tmp');shutil.copy2(f,tmp);tmp.replace(target)
  for n in names:
   source=root/n;target=dest/n
   if not target.exists():shutil.copy2(source,target)
def run(arm,seed,lr):
 attempt=uuid.uuid4().hex
 failure=out/f'{args.stage}_{arm}_seed{seed}_lr{lr:g}.failure.json'
 if failure.exists():failure.replace(failure.with_name(failure.stem+'.previous.'+attempt+'.json'))
 command=[sys.executable,str(root/'celeba_training.py'),'--arm',arm,'--seed',str(seed),'--lr',str(lr),
  '--attempt-id',attempt,'--out',str(out),'--data',str(root/'data'),'--phase',args.stage,'--steps',str(protocol['development_steps_per_configuration'] if args.stage=='development' else protocol['reporting_steps']),'--batch',str(protocol['batch_size'])]
 print(json.dumps({'start':command,'utc_seconds':time.time()}),flush=True)
 proc=subprocess.Popen(command)
 next_backup=time.monotonic()+20
 while proc.poll() is None:
  time.sleep(1)
  if time.monotonic()>=next_backup:backup();next_backup=time.monotonic()+20
 backup()
 if proc.returncode:
  record=json.loads(failure.read_text()) if failure.exists() else {}
  source_match=all(record.get('source_sha256',{}).get(n)==identity[n] for n in ['celeba_training.py','preflight.py','simplex_qp.py'])
  if args.stage=='development' and record.get('kind')=='nonfinite_training' and record.get('attempt_id')==attempt and source_match:
   print('UNAVAILABLE_DEVELOPMENT_CANDIDATE',json.dumps(record),flush=True)
   return {'success':False,'attempt_id':attempt,'failure':record}
  raise RuntimeError('Campaign halted on failed configuration; no seed omitted')
 return {'success':True,'attempt_id':attempt}
if args.stage=='development':
 outcomes={}
 previous=root/'development_outcomes.json'
 if previous.exists():
  saved=json.loads(previous.read_text())
  if saved['identity']!=identity:raise RuntimeError('Development identity mismatch')
  outcomes=saved['outcomes']
 for lr in protocol['development_learning_rates']:
  for arm in protocol['arms']:
   key=f'{arm}:{lr:g}'
   if key in outcomes:
    if not outcomes[key]['success']:continue
    record=out/f'development_{arm}_seed9001_lr{lr:g}.json'
    if record.exists() and json.loads(record.read_text()).get('complete'):continue
   outcomes[key]=run(arm,protocol['development_seed'],lr)
   write_json(root/'development_outcomes.json',{'identity':identity,'outcomes':outcomes})
 selected={}
 for arm in protocol['arms']:
  choices=[]
  for lr in protocol['development_learning_rates']:
   if not outcomes[f'{arm}:{lr:g}']['success']:continue
   d=json.loads((out/f'development_{arm}_seed9001_lr{lr:g}.json').read_text())
   if not d['complete'] or d.get('invalid'):raise ValueError('Incomplete development run')
   v=d['evaluations'][-1]['metrics'];choices.append((v['worst_accuracy'],v['mean_accuracy'],-lr,lr))
  if not choices:raise RuntimeError('No finite development candidate for '+arm)
  selected[arm]=max(choices)[-1]
 selection={'identity':identity,'rates':selected}
 selected_path=root/'selected_learning_rates.json'
 if selected_path.exists() and json.loads(selected_path.read_text())!=selection:raise RuntimeError('Attempted to change frozen selection')
 write_json(selected_path,selection)
 write_json(root/'selection_lock.json',{'identity':identity,'selection_sha256':hashlib.sha256(selected_path.read_bytes()).hexdigest(),'development_outcomes_sha256':hashlib.sha256((root/'development_outcomes.json').read_bytes()).hexdigest()})
else:
 lock=json.loads((root/'selection_lock.json').read_text())
 if lock['identity']!=identity or lock['selection_sha256']!=hashlib.sha256((root/'selected_learning_rates.json').read_bytes()).hexdigest() or lock['development_outcomes_sha256']!=hashlib.sha256((root/'development_outcomes.json').read_bytes()).hexdigest():raise RuntimeError('Frozen development selection changed')
 selected=json.loads((root/'selected_learning_rates.json').read_text())
 if selected['identity']!=identity:raise RuntimeError('Development configuration mismatch')
 for seed in protocol['confirmatory_seeds']:
  arms=protocol['arms'];order=arms[seed%len(arms):]+arms[:seed%len(arms)]
  for arm in order:run(arm,seed,selected['rates'][arm])
backup();print('CAMPAIGN_STAGE_COMPLETE',args.stage,flush=True)
