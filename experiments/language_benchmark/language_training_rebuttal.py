"""Resumable full-parameter language training; no GPU work on import.

Checkpoint policy retains current/selected models for the active run. Numerical
results and traces are permanent; an external campaign may retire completed
model checkpoints only after validating and durably archiving their manifests.
"""
import os
os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
import argparse,copy,hashlib,json,random,shutil,time,traceback
from pathlib import Path
import numpy as np
import torch
from language_common import load_data,make_batch,pack,losses,task_losses,native_direction
from language_training_methods import LanguageRule,finite,sync
from attention_precision import shared_attention_precision
from native_language_dual_reference import native_reference
from language_tokenizer_contract import load_tokenizer
from v46_supervised_head import supervised_token_outputs,per_example_cross_entropy

CONTEXT={}
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(8*2**20),b''):h.update(block)
    return h.hexdigest()
def save(p,obj):
    p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj,indent=2,allow_nan=False));tmp.replace(p)
def save_torch(p,obj):
    p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp');torch.save(obj,tmp);tmp.replace(p)

@torch.no_grad()
def evaluate(model,split,tasks,eos,batch_size=32):
    model.eval();rows=[]
    for task in tasks:
        total=0.;correct=0;count=0
        for first in range(0,len(split[task]),batch_size):
            values=split[task][first:first+batch_size]
            b=pack(values,[0]*len(values),1,eos,device=str(next(model.parameters()).device))
            logits,targets,_=supervised_token_outputs(model,b['inputs'],b['attention'],b['labels'],prepared=b['prepared'])
            loss=per_example_cross_entropy(logits,targets);finite([loss],'evaluation losses')
            exact=((logits.argmax(-1)==targets)|(targets==-100)).all(1)
            total+=float(loss.double().sum());correct+=int(exact.sum());count+=len(values)
        if count!=len(split[task]) or count==0:raise ValueError('Evaluation coverage')
        rows.append({'task':task,'examples':count,'answer_ce':total/count,'teacher_forced_answer_exact_match':correct/count})
    ce=np.array([r['answer_ce'] for r in rows]);acc=np.array([r['teacher_forced_answer_exact_match'] for r in rows])
    return {'tasks':rows,'mean_ce':float(ce.mean()),'worst_ce':float(ce.max()),'mean_answer_exact_match':float(acc.mean()),
        'worst_answer_exact_match':float(acc.min()),'accuracy_scope':'all supervised answer tokens correct under teacher forcing; not free-generation accuracy'}

class Trainer:
    def __init__(self,model,cfg,identity):
        self.model=model.eval();self.cfg=copy.deepcopy(cfg);self.identity=copy.deepcopy(identity)
        self.optimizer=torch.optim.SGD(model.parameters(),lr=cfg['lr'],momentum=0,weight_decay=0)
        self.rule=LanguageRule(cfg['method'],8,next(model.parameters()).device,cfg['system'],cfg.get('method_options'))
        self.step=0;self.pending=[];self.blocks=[];self.history=[];self.best=None;self.audits=[];self.superseded=[]
        self.update_seconds=0.;self.preparation_seconds=0.;self.recorded_wall_seconds=0.;self.attempts=0;self.peak_bytes=0
    def update(self,b):
        row=self.rule.step(self.model,b,self.optimizer,self.step)
        row['batch_identity_sha256']=hashlib.sha256(json.dumps(b['identities'],sort_keys=True).encode()).hexdigest()
        self.pending.append(row);self.step+=1;self.update_seconds+=row['update_seconds']
        if next(self.model.parameters()).is_cuda:self.peak_bytes=max(self.peak_bytes,torch.cuda.max_memory_allocated())
    def state(self):
        counters={k:copy.deepcopy(getattr(self,k)) for k in ['step','pending','blocks','history','best','audits','superseded','update_seconds',
            'preparation_seconds','recorded_wall_seconds','attempts','peak_bytes']}
        rng={'torch':torch.get_rng_state(),'numpy':np.random.get_state(),'python':random.getstate()}
        if next(self.model.parameters()).is_cuda:rng['cuda']=torch.cuda.get_rng_state_all()
        return {'config':self.cfg,'identity':self.identity,'model':self.model.state_dict(),'optimizer':self.optimizer.state_dict(),
            'rule':self.rule.state(),'counters':counters,'rng':rng}
    def restore(self,state):
        if state['config']!=self.cfg or state['identity']!=self.identity:raise ValueError('Resume identity differs')
        self.model.load_state_dict(state['model']);self.optimizer.load_state_dict(state['optimizer']);self.rule.restore(state['rule'])
        for k,v in state['counters'].items():setattr(self,k,copy.deepcopy(v))
        rng=state['rng'];torch.set_rng_state(rng['torch']);np.random.set_state(rng['numpy']);random.setstate(rng['python'])
        if 'cuda' in rng:torch.cuda.set_rng_state_all(rng['cuda'])
        if self.rule.next_step!=self.step:raise ValueError('Restored method/model cursor mismatch')
    def flush(self,out):
        if not self.pending:return
        name=f'updates/{self.pending[0]["step"]:08d}_{self.pending[-1]["step"]:08d}.json';p=out/name;save(p,self.pending)
        self.blocks.append({'file':name,'sha256':sha(p),'first_step':self.pending[0]['step'],'last_step':self.pending[-1]['step'],'rows':len(self.pending)})
        self.pending=[]
    def summary(self):
        return {'complete':False,'config':self.cfg,'identity':self.identity,'steps':self.step,'update_blocks':self.blocks,
            'history':self.history,'selected_checkpoint':self.best,'native_audits':self.audits,'update_seconds':self.update_seconds,
            'preparation_seconds':self.preparation_seconds,'recorded_wall_seconds':self.recorded_wall_seconds,'attempts':self.attempts,
            'peak_update_allocated_bytes':self.peak_bytes,'gram_captures':self.rule.gram_captures,
            'periodic_refinements':self.rule.periodic_refinements,'post_update_forwards':self.rule.post_forwards,
            'wall_scope':'main through latest recorded event, including loading, preparation, audits, evaluation and earlier checkpoint copies; excludes imports/final persistence; resumed uncheckpointed work may be missing',
            'model_mode':'full-parameter fine-tuning in eval mode, disabling dropout for every rule',
            'certificate_scope':'raw computational fidelity diagnostics; no population guarantee or per-step interval certificate'}

def audit_current(trainer,b):
    # Independent of the training update and excluded from update timing.
    # Replay the selected Direct implementation without mutating model/rule.
    from language_baseline_v6 import run_method
    system=trainer.cfg['system'];method='direct_online' if system['mode']=='online' else 'direct_cached'
    trainer.optimizer.zero_grad(set_to_none=True)
    K,fit,reference,table=native_reference(trainer.model,b,retain_table=True)
    row=run_method(trainer.model,b,method,None,row_chunk=system['row_chunk'],scratch_mib=system['scratch_mib'],
        reconstruction=system.get('reconstruction','weighted_native' if system['mode']=='online' else 'operator'),
        reference=(table,K,fit))
    row.update(after_updates=trainer.step,reference=reference)
    trainer.optimizer.zero_grad(set_to_none=True)
    return row

def main():
    start=time.perf_counter()
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--config',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--backup',type=Path,required=True);p.add_argument('--resume',action='store_true')
    a=p.parse_args();out=a.output;out.mkdir(parents=True,exist_ok=True);CONTEXT['output']=str(out)
    cfg=json.loads(a.config.read_text());phase=cfg['phase']
    seeds=[49100] if phase=='development' else [49201,49202,49203]
    if phase not in ('development','confirmatory') or cfg['seed'] not in seeds or cfg['batch_size']!=32 or cfg['steps']<1 or cfg['validation_every']<1:
        raise ValueError('Frozen language training configuration required')
    from transformers import AutoModelForCausalLM,AutoTokenizer
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    random.seed(cfg['seed']);np.random.seed(cfg['seed']);torch.manual_seed(cfg['seed'])
    data,manifest=load_data(a.root/'data');model_meta=json.loads((a.root/'data'/f'{cfg["model"].split("/")[-1]}_manifest.json').read_text())
    tok,tokenizer_contract=load_tokenizer(cfg['model'],model_meta['revision'],manifest)
    model=AutoModelForCausalLM.from_pretrained(cfg['model'],revision=model_meta['revision'],torch_dtype=torch.float32,attn_implementation='eager').cuda().eval()
    names=['language_training_rebuttal.py','language_training_methods.py','language_common.py','attention_precision.py','simplex_qp.py',
        'v46_factorized_jacobian.py','v46_factorized_exact.py','paretotrace_online_factor.py','v46_supervised_head.py',
        'native_language_dual_reference.py','language_baseline_v4.py','ghost_adapter.py','structured_gram.py']
    names+=['language_retained_v31.py','language_retained_manifest.json','language_tokenizer_contract.py']
    identity={'config_sha256':sha(a.config),'source_sha256':{n:sha(a.root/n) for n in names},'data':manifest,'model':model_meta,'tokenizer_contract':tokenizer_contract}
    with shared_attention_precision() as precision:
        identity['attention']={k:v for k,v in precision.items() if k!='calls'}
        owner={'config':cfg,'identity':identity}
        for base in (out,a.backup):
            if base.exists() and any(base.iterdir()):
                record=base/'run_identity.json'
                if not record.exists() or json.loads(record.read_text())!=owner:raise ValueError('Output/backup is not owned by this run: '+str(base))
            if (base/'failure.json').exists():raise ValueError('Retained failure requires explicit diagnosis before resume')
        for base in (out,a.backup):save(base/'run_identity.json',owner)
        trainer=Trainer(model,cfg,identity);latest=out/'latest.pt'
        if a.resume:
            trainer.restore(torch.load(latest,map_location='cpu',weights_only=False))
            for block in trainer.blocks:
                if sha(out/block['file'])!=block['sha256']:raise ValueError('Event block mismatch')
            if trainer.best is not None and sha(out/trainer.best['file'])!=trainer.best['sha256']:raise ValueError('Selected model mismatch')
        elif latest.exists():raise ValueError('Existing run requires resume')
        trainer.attempts+=1;prior=trainer.recorded_wall_seconds;copied=set()
        def backup():
            files=[out/b['file'] for b in trainer.blocks]
            if trainer.best is not None:files.append(out/trainer.best['file'])
            files.extend([out/'metrics.json',latest])
            for f in files:
                if not f.exists() or (str(f) in copied and f.parent.name in ('updates','selected')):continue
                target=a.backup/f.relative_to(out);target.parent.mkdir(parents=True,exist_ok=True)
                tmp=target.with_suffix('.tmp');shutil.copy2(f,tmp);tmp.replace(target)
                if f.parent.name in ('updates','selected'):copied.add(str(f))
            save(a.backup/'backup_commit.json',{'steps':trainer.step,'selected_checkpoint':trainer.best,
                'latest_sha256':sha(a.backup/'latest.pt') if (a.backup/'latest.pt').exists() else None,
                'metrics_sha256':sha(a.backup/'metrics.json') if (a.backup/'metrics.json').exists() else None})
        def checkpoint():
            trainer.flush(out);trainer.recorded_wall_seconds=prior+time.perf_counter()-start
            save_torch(latest,trainer.state());save(out/'metrics.json',trainer.summary());backup()
            # Remove only superseded selections owned by THIS run, after both
            # latest checkpoints reference and contain the new chosen model.
            for base in (out,a.backup):
                for ref in trainer.superseded:
                    rel=Path(ref['file'])
                    if rel.parent!=Path('selected') or '..' in rel.parts or ref['file']==trainer.best['file']:raise ValueError('Unsafe superseded-selection reference')
                    f=base/rel
                    if f.exists():
                        if sha(f)!=ref['sha256']:raise ValueError('Superseded-selection identity changed')
                        f.unlink()
        def validation():
            metrics=evaluate(model,data['validation'],manifest['tasks'],tok.eos_token_id)
            record={'after_updates':trainer.step,'metrics':metrics,'update_seconds':trainer.update_seconds,
                'recorded_wall_seconds':prior+time.perf_counter()-start}
            # Initial evaluation describes the start but cannot win selection.
            if trainer.step>0 and (trainer.best is None or metrics['worst_ce']<trainer.best['worst_ce']):
                pending=out/'selected/pending.pt';save_torch(pending,{'model':model.state_dict(),'after_updates':trainer.step,'config':cfg,'identity':identity})
                digest=sha(pending);chosen=out/'selected'/f'step{trainer.step:08d}_{digest[:16]}.pt';pending.replace(chosen)
                if trainer.best is not None:trainer.superseded.append(copy.deepcopy(trainer.best))
                trainer.best={'file':str(chosen.relative_to(out)),'sha256':digest,'after_updates':trainer.step,'worst_ce':metrics['worst_ce']}
                record['selected']=True
            else:record['selected']=False
            trainer.history.append(record)
        try:
            if trainer.step==0 and not trainer.history:validation();checkpoint()
            while trainer.step<cfg['steps']:
                tick=time.perf_counter();b=make_batch(data['train'],manifest['tasks'],tok.eos_token_id,cfg['seed'],step=trainer.step)
                sync(model);trainer.preparation_seconds+=time.perf_counter()-tick
                torch.cuda.reset_peak_memory_stats();trainer.update(b)
                if cfg['method']=='direct' and trainer.step in {1,cfg['steps']//2,cfg['steps']}:
                    audit=audit_current(trainer,b);trainer.audits.append(audit)
                    audit_name=f'audits/step{trainer.step:08d}.json'
                    save(out/audit_name,audit);save(a.backup/audit_name,audit)
                    if not audit['passes_table_contract']:raise ArithmeticError('Independent native-task-table MGDA fidelity audit failed')
                if trainer.step%cfg['validation_every']==0 or trainer.step==cfg['steps']:
                    validation();checkpoint()
                if trainer.step%50==0:print(json.dumps({'method':cfg['method'],'seed':cfg['seed'],'step':trainer.step,'update_seconds':trainer.update_seconds}),flush=True)
            if phase=='development':
                result=trainer.summary();result.update(complete=True,heldout_test_evaluated=False)
            else:
                final=evaluate(model,data['test'],manifest['tasks'],tok.eos_token_id)
                ref=trainer.best
                if ref is None or sha(out/ref['file'])!=ref['sha256']:raise ValueError('Missing selected model')
                saved=torch.load(out/ref['file'],map_location='cpu',weights_only=False)
                if saved['config']!=cfg or saved['identity']!=identity or saved['after_updates']!=ref['after_updates']:raise ValueError('Selected model identity')
                model.load_state_dict(saved['model']);del saved
                selected=evaluate(model,data['test'],manifest['tasks'],tok.eos_token_id)
                # No checkpoint after loading selected model: latest remains the
                # complete final-training state paired with final rule/counters.
                result=trainer.summary();result.update(complete=True,heldout_test_evaluated=True,final_test=final,validation_selected_test=selected)
            result['recorded_wall_seconds']=prior+time.perf_counter()-start
            result['uninterrupted_training_evaluation_seconds']=result['recorded_wall_seconds'] if trainer.attempts==1 else None
            save(out/'metrics.json',result);backup();print('LANGUAGE_TRAINING_COMPLETE',str(out),flush=True)
        except BaseException:
            failure={'complete':False,'step':trainer.step,'identity':identity,'error':traceback.format_exc(),
                'elapsed_current_attempt_seconds':time.perf_counter()-start,'native_audits':trainer.audits}
            save(out/'failure.json',failure);save(a.backup/'failure.json',failure);CONTEXT['failure_recorded']=True
            raise

if __name__=='__main__':
    try:main()
    except BaseException:
        if CONTEXT.get('output') and not CONTEXT.get('failure_recorded'):save(Path(CONTEXT['output'])/'failure.json',{'complete':False,'phase':'startup','error':traceback.format_exc()})
        raise
