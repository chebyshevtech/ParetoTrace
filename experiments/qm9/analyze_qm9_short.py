"""All-seed analysis of the bounded QM9 campaign; no new model evaluations."""
import argparse,json,math
from pathlib import Path
import numpy as np
from scipy.stats import t
from qm9_campaign import METHODS,sha,validate_result,save

def paired(values):
    x=np.asarray(values,dtype=float)
    if x.shape!=(3,) or not np.isfinite(x).all():raise ValueError('All three finite paired differences required')
    mean=float(x.mean());half=float(t.ppf(.975,2)*x.std(ddof=1)/np.sqrt(3))
    return {'differences':x.tolist(),'mean':mean,'student_t_95_interval':[mean-half,mean+half],
        'scope':'descriptive uncertainty over three training seeds conditional on a fixed test set; secondary comparisons unadjusted'}

def analyze(root):
    root=Path(root);done=json.loads((root/'complete.json').read_text());protocol=json.loads((root/'protocol.json').read_text())
    if not done.get('complete') or done['protocol_sha256']!=sha(root/'protocol.json') or len(done['runs'])!=21:
        raise ValueError('All21 completed runs required')
    if protocol['version']!='qm9_rebuttal_short_v1' or protocol['epochs']!=2 or protocol['learning_rate']!=1e-4:
        raise ValueError('Unexpected budget/protocol')
    seeds=[38201,38202,38203]
    if protocol['reporting_seeds']!=seeds or protocol['arms']!=METHODS:raise ValueError('Paired population changed')
    identity=json.loads((root/'campaign_source_identity.json').read_text())
    if sha(root/'campaign_source_identity.json')!=done['source_identity_sha256']:raise ValueError('Campaign identity differs')
    for name,digest in identity.items():
        if sha(root/name)!=digest:raise ValueError('Campaign source changed')
    records={};all_rows=[];summary={}
    for method in METHODS:
        per=[]
        for seed in seeds:
            label=f'confirmatory_{method}_seed{seed}_lr0.0001';out=root/'runs'/label
            config=root/'configs'/f'{label}.json';metrics=out/'metrics.json'
            if label not in done['runs'] or sha(metrics)!=done['runs'][label]['metrics_sha256'] or sha(config)!=done['runs'][label]['config_sha256']:
                raise ValueError('Run identity differs')
            cfg=json.loads(config.read_text());d=json.loads(metrics.read_text())
            if cfg['seed']!=seed or cfg['epochs']!=2 or cfg['lr']!=1e-4 or cfg['system']!=protocol['systems'][method]:
                raise ValueError('Run budget/system differs')
            validate_result(d,cfg,root,out);records[method,seed]=d
            attempts=json.loads((out/'process_attempts.json').read_text())
            if not attempts:raise ValueError('Missing process attempt record')
            completed_attempts=[x for x in attempts if x.get('complete')]
            for attempt in completed_attempts:
                duration=attempt.get('subprocess_wall_seconds')
                if not isinstance(duration,(int,float)) or not math.isfinite(duration) or duration<0:
                    raise ValueError('Invalid completed-attempt wall time')
            wall=sum(x['subprocess_wall_seconds'] for x in completed_attempts)
            unknown=any(not x.get('complete') for x in attempts)
            row={'method':method,'seed':seed,'final_test':d['final_test'],'validation_selected_test':d['validation_selected_test'],
                'update_seconds':d['update_seconds'],'seconds_per_update':d['update_seconds']/d['updates'],
                'completed_attempt_wall_seconds':wall,'unrecorded_interrupted_attempt_wall':unknown,
                'last_recorded_returncode':attempts[-1].get('returncode'),
                'peak_allocated_gib':d['process_peak_allocated_bytes']/2**30,
                'solver_calls':d['solver_calls'],'solver_fallback_or_failure_calls':d['solver_fallback_or_failure_calls'],
                'native_audits':d['native_audits'],'metrics_sha256':sha(metrics)}
            per.append(row);all_rows.append(row)
        values=np.array([x['final_test']['mean_normalized_mae'] for x in per])
        summary[method]={'mean_final_test_normalized_mae':float(values.mean()),'sample_sd_final_test_normalized_mae':float(values.std(ddof=1)),
            'mean_final_test_worst_normalized_mae':float(np.mean([x['final_test']['worst_normalized_mae'] for x in per])),
            'mean_seconds_per_update':float(np.mean([x['seconds_per_update'] for x in per])),
            'total_update_seconds':sum(x['update_seconds'] for x in per),'total_completed_attempt_wall_seconds':sum(x['completed_attempt_wall_seconds'] for x in per),
            'any_unrecorded_interrupted_wall':any(x['unrecorded_interrupted_attempt_wall'] for x in per),
            'max_peak_allocated_gib':max(x['peak_allocated_gib'] for x in per),
            'solver_calls':sum(x['solver_calls'] for x in per),'solver_fallback_or_failure_calls':sum(x['solver_fallback_or_failure_calls'] for x in per),
            'physical_mae_mean':np.mean([x['final_test']['physical_mae'] for x in per],axis=0).tolist(),
            'physical_mae_sample_sd':np.std([x['final_test']['physical_mae'] for x in per],axis=0,ddof=1).tolist()}
    comparisons={method:paired([records['direct_factorized',s]['final_test']['mean_normalized_mae']-
        records[method,s]['final_test']['mean_normalized_mae'] for s in seeds]) for method in METHODS if method!='direct_factorized'}
    return {'complete':True,'scope':'two full-data epochs at published fixed learning rate; early training, not300epoch convergence or tuned-best-method ranking',
        'protocol':protocol,'protocol_sha256':sha(root/'protocol.json'),'campaign_complete_sha256':sha(root/'complete.json'),
        'primary':comparisons['mgda_ub'],'all_direct_minus_baseline_comparisons':comparisons,
        'arm_summary':summary,'all_seed_results':all_rows,
        'full_300epoch_update_hours_extrapolated':sum(x['update_seconds'] for x in all_rows)*150/3600,
        'projection_scope':'linear update-time projection only; excludes scaling of evaluation, loading and persistence overhead',
        'analysis_sources':{n:sha(Path(__file__).with_name(n)) for n in ['analyze_qm9_short.py','qm9_campaign.py']}}

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    result=analyze(a.root);save(a.root/'paired_analysis.json',result)
    print(json.dumps({'primary':result['primary'],'arm_summary':result['arm_summary']},indent=2))

if __name__=='__main__':main()
