"""Prespecified paired inference, refuses incomplete confirmatory campaigns."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from scipy.stats import t
from validate_celeba import validate_campaign

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',default='/content/rebuttal');a=p.parse_args();root=Path(a.root)
 records,protocol,selected,validation=validate_campaign(root)
 sources=validation['artifact_hashes']
 seeds=protocol['confirmatory_seeds'];rows=[];summary={}
 for arm in protocol['arms']:
  tests=[records[s,arm]['test'] for s in seeds]
  summary[arm]={key:float(np.mean([x[key] for x in tests])) for key in
   ['worst_accuracy','mean_accuracy','worst_bce','mean_bce','worst_balanced_accuracy','mean_balanced_accuracy']}
  summary[arm]['mean_total_update_seconds_per_run']=float(np.mean([records[s,arm]['update_seconds'] for s in seeds]))
  summary[arm]['mean_seconds_per_update']=summary[arm]['mean_total_update_seconds_per_run']/protocol['reporting_steps']
  summary[arm]['mean_recorded_training_wall_seconds_per_run']=float(np.mean([records[s,arm]['wall_seconds'] for s in seeds]))
  summary[arm]['recorded_wall_scope']='checkpointed training-loop time; excludes startup/loading, terminal write, checkpoint writes immediately preceding resumes, and uncheckpointed replayed work; not total end-to-end time'
  summary[arm]['max_process_peak_bytes']=max(records[s,arm]['process_peak_bytes'] for s in seeds)
  audits=[a for s in seeds for a in records[s,arm]['audits']]
  if audits:
   summary[arm]['native_audits']={'count':len(audits),'max_gram_relative_error':max(a['gram_relative_error'] for a in audits),
     'negative_min_response_count':sum(a['native_min_response']<0 for a in audits),
     'minimum_recorded_native_response':min(a['native_min_response'] for a in audits),
     'scope':'Objective-fidelity diagnostics; common descent of native or momentum-executed updates is not certified.'}
 for s in seeds:
  rows.append({'seed':s,'test':{arm:records[s,arm]['test'] for arm in protocol['arms']},
               'primary_difference':records[s,'direct']['test']['worst_accuracy']-records[s,'mgda_ub']['test']['worst_accuracy']})
 delta=np.array([r['primary_difference'] for r in rows]);n=len(delta);mean=float(delta.mean())
 half=float(t.ppf(.975,n-1)*delta.std(ddof=1)/np.sqrt(n))
 rng=np.random.default_rng(271828);boot=delta[rng.integers(n,size=(50000,n))].mean(1)
 out={'complete':True,'n_paired_seeds':n,'primary_endpoint':protocol['primary_endpoint'],
  'mean_difference':mean,'student_t_95_interval':[mean-half,mean+half],
  'paired_bootstrap_percentile_95_interval':np.quantile(boot,[.025,.975]).tolist(),
  'positive_seed_differences':int((delta>0).sum()),'zero_seed_differences':int((delta==0).sum()),
  'negative_seed_differences':int((delta<0).sum()),'all_seed_results':rows,'arm_summary':summary,
  'selected_learning_rates':selected['rates'],'sources':sources,'protocol':protocol,'artifact_validation':validation,
  'analysis_sources':{n:hashlib.sha256(Path(__file__).with_name(n).read_bytes()).hexdigest() for n in ['analyze_celeba.py','validate_celeba.py']},
  'inference_scope':'variation over training seeds conditional on this fixed test set; no dataset-population uncertainty claim',
  'quality_time_scope':'equal-update comparison; cheap arms not yet extended to Direct wall budget'}
 path=root/'celeba_paired_analysis.json';tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(out,indent=2,allow_nan=False));tmp.replace(path)
 print(json.dumps({k:out[k] for k in ['n_paired_seeds','mean_difference','student_t_95_interval','arm_summary']}),flush=True)

if __name__=='__main__':main()
