"""Prospective1B complete-update development pilot; no heldout evaluation.

Candidate Direct configurations come only from development calibration in the
separate recovery benchmark. Periodic compares both existing kernel backends.
Ten warmup updates precede two measured five-step cycles for every candidate.
"""
import os
os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
import argparse,gc,hashlib,json,time,traceback
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM
from language_common import load_data,make_batch,save
from language_tokenizer_contract import load_tokenizer
from attention_precision import shared_attention_precision
from language_training_rebuttal import Trainer,audit_current

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--recovery',type=Path,required=True)
    a=p.parse_args();dest=a.root/'training_preflight.json'
    if dest.exists():raise ValueError('Do not overwrite retained pilot')
    recovery=json.loads(a.recovery.read_text())
    if recovery.get('complete') is not True or recovery['development_seed']!=5000:raise ValueError('Completed frozen development calibration required')
    model_name='EleutherAI/pythia-1b-deduped'
    if recovery['model']['model']!=model_name:raise ValueError('1B calibration required')
    # Reporting outcomes in the source artifact are never used for selection.
    candidates=[]
    for method,mode in [('direct_cached','cached'),('direct_online','online')]:
        cfg=recovery['selected']['8'][method]['table']
        if cfg is not None:candidates.append({'method':'direct','system':{'mode':mode,'row_chunk':cfg['row_chunk'],
            'scratch_mib':cfg['scratch_mib'],'reconstruction':cfg['reconstruction']}})
    if not candidates:raise ValueError('No native-table-qualified Direct development configuration')
    # Shared geometry backend for CAGrad/FA warm start is selected strictly by
    # development recovery timing, before any training-pilot measurements.
    def score(system):
        name='direct_cached' if system['mode']=='cached' else 'direct_online'
        cfg={k:v for k,v in system.items() if k!='mode'}
        rows=[r for r in recovery['calibration'] if r['method']==name and r['M']==8 and r['seed']==5000 and
              r['configuration']==cfg and not r['warmup'] and r['passes_table_contract']]
        if len(rows)!=2:raise ValueError('Incomplete selected development timing')
        return float(np.median([r['seconds'] for r in rows]))
    shared=min(candidates,key=lambda x:score(x['system']))['system']
    for method in ['uniform','cagrad','fa','famo']:candidates.append({'method':method,'system':shared})
    for backend in ['native','factor']:candidates.append({'method':'periodic','system':shared,'method_options':{'periodic_backend':backend}})
    data,manifest=load_data(a.root/'data');meta=recovery['model'];tok,token_contract=load_tokenizer(model_name,meta['revision'],manifest)
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    model=AutoModelForCausalLM.from_pretrained(model_name,revision=meta['revision'],torch_dtype=torch.float32,attn_implementation='eager',token=False).cuda().eval()
    initial={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    out={'complete':False,'purpose':'complete-update1Bdevelopment pilot; no validation/test evaluation',
        'seed':49100,'lr':1e-4,'batch':32,'warmup_updates':10,'measured_updates':10,'model':meta,'data':manifest,
        'tokenizer_contract':token_contract,'recovery_sha256':hashlib.sha256(a.recovery.read_bytes()).hexdigest(),
        'selection_source':'only seed5000 calibration/configurations; no recovery reporting metrics consumed',
        'gpu':torch.cuda.get_device_name(),'gpu_total_bytes':torch.cuda.get_device_properties(0).total_memory,
        'torch':str(torch.__version__),'cuda':torch.version.cuda,
        'rows':[],'sources':{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in a.root.glob('*.py')}}
    save(dest,out)
    with shared_attention_precision() as precision:
        out['attention_precision']=precision
        for candidate in candidates:
            cfg={'lr':1e-4,**candidate};trainer=None
            row={'configuration':candidate,'complete':False,'updates':[],'native_audits':[],'stage':'setup'}
            attempt_start=time.perf_counter()
            try:
                model.load_state_dict(initial);model.zero_grad(set_to_none=True)
                trainer=Trainer(model,cfg,{'pilot':True})
                for step in range(20):
                    row['stage']='update_'+str(step)
                    tick=time.perf_counter();b=make_batch(data['train'],manifest['tasks'],tok.eos_token_id,49100,step=step)
                    torch.cuda.synchronize();prep=time.perf_counter()-tick
                    torch.cuda.reset_peak_memory_stats();trainer.update(b)
                    record=trainer.pending[-1].copy();record.update(batch_preparation_seconds=prep,warmup=step<10,
                        total_update_and_preparation_seconds=record['update_seconds']+prep,
                        peak_allocated_bytes=torch.cuda.max_memory_allocated())
                    row['updates'].append(record)
                    if candidate['method']=='direct' and step in (0,19):
                        row['stage']='audit_'+str(step)
                        audit=audit_current(trainer,b);row['native_audits'].append(audit)
                        if not audit['passes_table_contract']:raise ArithmeticError('Direct training-pilot native-table audit failed')
                measured=row['updates'][10:]
                row.update(complete=True,mean_update_seconds=float(np.mean([x['update_seconds'] for x in measured])),
                    mean_update_and_preparation_seconds=float(np.mean([x['total_update_and_preparation_seconds'] for x in measured])),
                    peak_allocated_bytes=max(x['peak_allocated_bytes'] for x in row['updates']),
                    measured_peak_allocated_bytes=max(x['peak_allocated_bytes'] for x in measured),
                    scope='two complete five-update cycles, including periodic refresh/held steps and all extra forwards; loading/audits/evaluation/checkpointing excluded')
            except Exception:
                row['error']=traceback.format_exc()
            row['attempt_wall_seconds_including_setup_and_audits']=time.perf_counter()-attempt_start
            out['rows'].append(row);save(dest,out)
            print(json.dumps({k:v for k,v in row.items() if k not in ('updates','native_audits')}),flush=True)
            del trainer;model.zero_grad(set_to_none=True);gc.collect();torch.cuda.empty_cache()
    out['selected']={}
    for method in ('direct','uniform','cagrad','fa','famo','periodic'):
        eligible=[r for r in out['rows'] if r['configuration']['method']==method and r['complete'] and r['peak_allocated_bytes']<=34*2**30]
        out['selected'][method]=min(eligible,key=lambda r:r['mean_update_and_preparation_seconds'])['configuration'] if eligible else None
    out['training_memory_rule']='fastest completed candidate at most34GiB allocated over ALL20updates including FA warm start, leaving headroom on40GiB A100; not an OOM guarantee'
    out['complete']=True;save(dest,out)

if __name__=='__main__':main()
