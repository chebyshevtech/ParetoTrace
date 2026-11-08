import subprocess,sys,time,traceback,json
from pathlib import Path
root=Path(__file__).parent
try:
 stages=['confirmatory'] if (root/'selection_lock.json').exists() else ['development','confirmatory']
 for stage in stages:
  subprocess.run([sys.executable,str(root/'celeba_queue.py'),'--stage',stage],check=True)
 protocol=json.loads((root/'celeba_protocol.json').read_text());selection=json.loads((root/'selected_learning_rates.json').read_text())
 count=0
 for seed in protocol['confirmatory_seeds']:
  for arm in protocol['arms']:
   path=root/'runs'/f'confirmatory_{arm}_seed{seed}_lr{selection["rates"][arm]:g}.json'
   d=json.loads(path.read_text())
   if not d.get('complete') or d.get('invalid') or 'test' not in d or len(d['rows'])!=protocol['reporting_steps']:raise RuntimeError('Incomplete reporting result: '+str(path))
   count+=1
 (root/'campaign_complete.json').write_text(json.dumps({'complete':True,'utc_seconds':time.time(),'verified_reporting_runs':count,'identity':selection['identity']}))
except BaseException:
 (root/'campaign_failed.txt').write_text(traceback.format_exc())
 raise
