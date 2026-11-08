"""Resumable, matched full-dataset QM9 training with explicit method adapters.

No GPU work on import. A frozen campaign config supplies method, system
configuration, seeds and training budget. Upstream Nash fallback is retained
and counted; it is not silently called a successful optimization solve.
"""
import os
os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
import argparse,copy,hashlib,json,random,shutil,time,traceback
from contextlib import contextmanager
from pathlib import Path
import numpy as np
import torch
from torch_geometric.loader import DataLoader
from qm9_data_verified import load_real,complete,TARGETS
from qm9_native_actions import OfficialQM9Model
from qm9_methods_v2 import OfficialWeighting,common_update,native_table,gram64,checked_qp

ENERGY={2,3,6,12,13,14,15}
UNITS=['D','a0^3','meV','meV','a0^2','meV','meV','meV','meV','meV','cal/(mol K)']
START_CONTEXT={}

@contextmanager
def kernels(fused):
    with torch.backends.cudnn.flags(enabled=fused),torch.backends.mkldnn.flags(enabled=False):yield

def require_finite(values,label):
    grouped={}
    for value in values:
        if value is None:raise FloatingPointError('Missing '+label)
        grouped.setdefault(value.device,[]).append(torch.isfinite(value.detach()).all())
    if any(not bool(torch.stack(group).all()) for group in grouped.values()):raise FloatingPointError('Nonfinite '+label)

def solver_success(events):
    return all(e.get('success') is True if 'success' in e else e.get('status')=='optimal' for e in events)

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def json_write(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(obj,indent=2,allow_nan=False));tmp.replace(path)
def torch_write(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp');torch.save(obj,tmp);tmp.replace(path)
def sync(device):
    if device=='cuda':torch.cuda.synchronize()

class Graphs(torch.utils.data.Dataset):
    def __init__(self,data,indices,mean,std):
        self.data,self.indices,self.mean,self.std=data,indices,mean,std
    def __len__(self):return len(self.indices)
    def __getitem__(self,j):
        graph=self.data[int(self.indices[j])]
        graph.y=((graph.y-self.mean)/self.std)[:,TARGETS]
        return complete(graph)

def loader(data,indices,mean,std,batch_size,workers,seed):
    return DataLoader(Graphs(data,indices,mean,std),batch_size=batch_size,shuffle=False,
        num_workers=workers,pin_memory=True,drop_last=False,
        generator=torch.Generator().manual_seed(seed))

def epoch_indices(train,seed,epoch):
    order=torch.randperm(len(train),generator=torch.Generator().manual_seed(seed*100000+epoch))
    return train[order]

@torch.no_grad()
def evaluate(model,data,indices,mean,std,cfg):
    was=model.training;model.eval();tot=torch.zeros(11,device=cfg['device'],dtype=torch.float64);count=0
    try:
        with kernels(cfg['system']['fused']):
            for batch in loader(data,indices,mean,std,cfg['batch_size'],cfg['workers'],cfg['seed']):
                batch=batch.to(cfg['device'],non_blocking=True);pred=model(batch)
                require_finite([pred],'evaluation predictions')
                tot+=(pred-batch.y).abs().double().sum(0);count+=batch.num_graphs
    finally:model.train(was)
    if count!=len(indices):raise RuntimeError('Evaluation coverage mismatch')
    normalized=(tot/count).cpu().numpy()
    physical=normalized*std.flatten()[TARGETS].numpy()*np.array([1000. if t in ENERGY else 1. for t in TARGETS])
    if not np.isfinite(normalized).all() or not np.isfinite(physical).all():raise FloatingPointError('Nonfinite evaluation metrics')
    return {'graphs':count,'mean_normalized_mae':float(normalized.mean()),'worst_normalized_mae':float(normalized.max()),
        'normalized_mae':normalized.tolist(),'physical_mae':physical.tolist(),'target_order':TARGETS,'units':UNITS}

class Trainer:
    def __init__(self,cfg,identity,upstream,model_factory=OfficialQM9Model):
        self.cfg=copy.deepcopy(cfg);self.identity=copy.deepcopy(identity)
        random.seed(cfg['seed']);np.random.seed(cfg['seed']);torch.manual_seed(cfg['seed'])
        self.model=model_factory().to(cfg['device']).train()
        self.opt=torch.optim.Adam(self.model.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay'])
        self.scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(self.opt,mode='min',factor=.7,patience=5,min_lr=1e-5)
        method=cfg['system']['method']
        self.rule=(OfficialWeighting(self.model,method,upstream,cfg['system'].get('chunk',1))
                   if method in ('Nash_MTL','FairGrad','Aligned_MTL','FAMO') else None)
        self.epoch=0;self.offset=0;self.step=0;self.best_metric=None;self.best_epoch=None;self.best_checkpoint=None
        self.history=[];self.pending=[];self.blocks=[];self.update_seconds=0.;self.solver_failures=0;self.solver_calls=0
        self.recorded_execution_seconds=0.;self.attempts=0;self.peak_bytes=0;self.audits=[]

    def update(self,batch,indices):
        cfg=self.cfg;system=cfg['system'];self.model.train();sync(cfg['device']);tick=time.perf_counter()
        self.opt.zero_grad(set_to_none=True)
        with kernels(system['fused']):
            if self.rule is None:
                info=common_update(self.model,batch,system['method'],system.get('chunk',1),system.get('row_chunk',64))
            else:info=self.rule.backward((self.model(batch)-batch.y).square().mean(0))
            require_finite([p.grad for p in self.model.parameters()],'executed gradients')
            self.opt.step()
            require_finite(list(self.model.parameters())+[v for state in self.opt.state.values() for v in state.values()
                if isinstance(v,torch.Tensor)],'model/optimizer state')
            if self.rule is not None and self.rule.method=='FAMO':
                with torch.no_grad():post=(self.model(batch)-batch.y).square().mean(0)
                require_finite([post],'FAMO post-update losses');self.rule.after_step(post)
                require_finite([self.rule.rule.w]+[v for state in self.rule.rule.w_opt.state.values() for v in state.values()
                    if isinstance(v,torch.Tensor)],'FAMO weight state')
        sync(cfg['device']);elapsed=time.perf_counter()-tick
        events=info.get('solver_events',[]);failed=sum(not solver_success([e]) for e in events)
        self.solver_calls+=len(events);self.solver_failures+=failed;self.update_seconds+=elapsed
        self.pending.append({'step':self.step,'epoch':self.epoch,'offset':self.offset,'graphs':len(indices),
            'indices_sha256':hashlib.sha256(indices.numpy().tobytes()).hexdigest(),
            'update_seconds':elapsed,'solver_fallback_or_failure_calls':failed,**info})
        self.step+=1;self.offset+=len(indices)
        if cfg['device']=='cuda':self.peak_bytes=max(self.peak_bytes,torch.cuda.max_memory_allocated())

    def state(self):
        counters={k:copy.deepcopy(getattr(self,k)) for k in ['epoch','offset','step','best_metric','best_epoch','best_checkpoint','history',
            'pending','blocks','update_seconds','solver_failures','solver_calls','recorded_execution_seconds','attempts','peak_bytes','audits']}
        rng={'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state()}
        if self.cfg['device']=='cuda':rng['cuda']=torch.cuda.get_rng_state_all()
        return {'config':self.cfg,'identity':self.identity,'model':self.model.state_dict(),'optimizer':self.opt.state_dict(),
            'scheduler':self.scheduler.state_dict(),'rule':self.rule.state() if self.rule is not None else None,'counters':counters,'rng':rng}

    def restore(self,state):
        if state['config']!=self.cfg or state['identity']!=self.identity:raise RuntimeError('Checkpoint protocol/source identity differs')
        self.model.load_state_dict(state['model']);self.opt.load_state_dict(state['optimizer']);self.scheduler.load_state_dict(state['scheduler'])
        if self.rule is not None:self.rule.restore(state['rule'])
        for k,v in state['counters'].items():setattr(self,k,copy.deepcopy(v))
        rng=state['rng'];random.setstate(rng['python']);np.random.set_state(rng['numpy']);torch.set_rng_state(rng['torch'])
        if self.cfg['device']=='cuda':torch.cuda.set_rng_state_all(rng['cuda'])

    def finish_epoch(self,metrics):
        score=metrics['mean_normalized_mae']
        if not np.isfinite(score) or score<0:raise FloatingPointError('Invalid validation selection metric')
        improved=self.best_metric is None or score<self.best_metric
        if improved:self.best_metric=score;self.best_epoch=self.epoch
        self.scheduler.step(score)
        self.history.append({'epoch':self.epoch,'validation':metrics,'selected':improved,'learning_rate':self.opt.param_groups[0]['lr']})
        self.epoch+=1;self.offset=0
        return improved

    def flush(self,out):
        if not self.pending:return
        name=f'updates_{self.pending[0]["step"]:08d}_{self.pending[-1]["step"]:08d}.json'
        path=out/'updates'/name;json_write(path,self.pending)
        self.blocks.append({'file':str(path.relative_to(out)),'sha256':sha(path),'rows':len(self.pending),
                            'first_step':self.pending[0]['step'],'last_step':self.pending[-1]['step']})
        self.pending=[]

    def summary(self):
        return {'config':self.cfg,'identity':self.identity,'complete':False,'epochs_finished':self.epoch,'updates':self.step,
            'best_epoch':self.best_epoch,'best_validation_mean_normalized_mae':self.best_metric,'history':self.history,
            'selected_checkpoint':self.best_checkpoint,
            'update_blocks':self.blocks,'update_seconds':self.update_seconds,'process_peak_allocated_bytes':self.peak_bytes,
            'solver_calls':self.solver_calls,'solver_fallback_or_failure_calls':self.solver_failures,
            'recorded_execution_seconds':self.recorded_execution_seconds,'attempts':self.attempts,'native_audits':self.audits,
            'update_scope':'forward, gradient acquisition/solve/reconstruction, finite checks, Adam, FAMO post-update; excludes input preparation',
            'wall_scope':'main() wall time through latest recorded event, including input preparation, validation, audits and earlier checkpoint/backup work; excludes interpreter/import startup and final summary persistence; uncheckpointed interrupted work may be missing after resume',
            'scheduler':'all methods: validation mean normalized MAE, ReduceLROnPlateau min/factor.7/patience5/min_lr1e-5; differs from upstream improvement-relative-to-first-epoch score',
            'Nash_scope':'pinned upstream previous-alpha fallback is part of executed baseline; every unsuccessful solve is counted'}

def native_audit(trainer,batch):
    system=trainer.cfg['system']
    if system['method']!='direct_factorized':return None
    model=trainer.model;model.zero_grad(set_to_none=True)
    with kernels(system['fused']):
        info=common_update(model,batch,'direct_factorized',system['chunk'],system['row_chunk'],return_gram=True)
        direction=torch.cat([p.grad.flatten() for p in model.encoder.parameters()]).detach().double()
        l=(model(batch)-batch.y).square().mean(0)
        table=native_table(l,tuple(model.encoder.parameters()),1)
        K=gram64(table);fit=checked_qp(K)
        candidate=torch.as_tensor(info['audit_gram'],device=K.device,dtype=torch.float64)
        gram_error=float((candidate-K).norm()/K.norm().clamp_min(1e-30))
        w=torch.tensor(info['weights'],device=K.device,dtype=torch.float64)
        target=w@table.double();error=float((direction-target).norm()/target.norm().clamp_min(1e-30))
        q=float(w@K@w);scale=max(float(K.abs().max()),1e-100)
        relative=fit['lower']>scale*1e-10;factor=q/fit['lower'] if relative else None
        passed=bool(np.isfinite([error,gram_error]).all() and error<=2e-5 and gram_error<=2e-5 and (factor<=1.001 if relative else q-fit['lower']<=1e-7*scale))
        record={'after_updates':trainer.step,'direction_relative_error':error,'gram_relative_error':gram_error,'objective_factor_upper':factor,
            'objective_excess_above_lower':q-fit['lower'],'native_lower':fit['lower'],'native_qp_gap':fit['dual_gap'],
            'relative_reference_resolved':relative,'passed':passed,
            'scope':'raw shared gradient in matched arithmetic; Adam update not asserted common descent'}
    model.zero_grad(set_to_none=True)
    return record

def main():
    attempt_begin=time.perf_counter()
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--backup',type=Path);p.add_argument('--resume',action='store_true')
    args=p.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=True)
    START_CONTEXT.update(output=str(out),phase='startup',failure_recorded=False)
    cfg=json.loads(args.config.read_text())
    phase=cfg.get('phase','confirmatory')
    allowed_seeds=[38102] if phase=='development' else [38201,38202,38203]
    if phase not in ('development','confirmatory') or cfg['epochs']<1 or cfg['batch_size']!=128 or cfg['device']!='cuda' or cfg['seed'] not in allowed_seeds:
        raise ValueError('Frozen reporting config required')
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    data,split,mean,std,contract=load_real(args.root/'data')
    names=['qm9_training_rebuttal.py','qm9_methods_v2.py','shared_direct_v2.py','simplex_qp.py',
           'qm9_native_actions.py','qm9_data_verified.py']
    identity={'config_sha256':sha(args.config),'data':contract,'sources':{n:sha(args.root/n) for n in names},
        'upstream':{f.name:sha(f) for f in (args.root/'weighting_upstream').glob('*.py')}}
    trainer=Trainer(cfg,identity,args.root/'weighting_upstream');latest=out/'latest.pt'
    if args.resume:
        trainer.restore(torch.load(latest,map_location='cpu',weights_only=False))
        for block in trainer.blocks:
            if sha(out/block['file'])!=block['sha256']:raise RuntimeError('Checkpoint event-block hash mismatch')
        if trainer.best_checkpoint is not None:
            ref=trainer.best_checkpoint
            if sha(out/ref['file'])!=ref['sha256'] or ref['epoch']!=trainer.best_epoch or ref['metric']!=trainer.best_metric:
                raise RuntimeError('Selected checkpoint reference inconsistent')
    elif latest.exists():raise RuntimeError('Existing run requires explicit resume')
    trainer.attempts+=1;prior=trainer.recorded_execution_seconds
    copied=set()
    def backup():
        if args.backup is None:return
        # Immutable event blocks and selected model precede latest checkpoint,
        # so a durable checkpoint never points to data not yet copied.
        files=[out/b['file'] for b in trainer.blocks]
        if trainer.best_checkpoint is not None:files.append(out/trainer.best_checkpoint['file'])
        files.extend([out/'metrics.json',latest])
        for f in files:
            if not f.exists():continue
            immutable=f.parent.name in ('updates','selected')
            if immutable and str(f) in copied:continue
            target=args.backup/f.relative_to(out);target.parent.mkdir(parents=True,exist_ok=True)
            tmp=target.with_suffix('.tmp');shutil.copy2(f,tmp);tmp.replace(target)
            if immutable:copied.add(str(f))
        json_write(args.backup/'backup_commit.json',{'step':trainer.step,'epoch':trainer.epoch,
            'selected_checkpoint':trainer.best_checkpoint,
            'mutable_files':{f.name:sha(args.backup/f.name) for f in [latest,out/'metrics.json'] if (args.backup/f.name).exists()}})
    def checkpoint():
        trainer.flush(out);trainer.recorded_execution_seconds=prior+time.perf_counter()-attempt_begin
        torch_write(latest,trainer.state());json_write(out/'metrics.json',trainer.summary());backup()
    try:
        START_CONTEXT['phase']='training'
        train=split[20000:];valid=split[10000:20000];test=split[:10000]
        while trainer.epoch<cfg['epochs']:
            indices=epoch_indices(train,cfg['seed'],trainer.epoch);start=trainer.offset
            dl=loader(data,indices[start:],mean,std,cfg['batch_size'],cfg['workers'],cfg['seed']*1000+trainer.epoch)
            batch=None
            for batch in dl:
                used=indices[trainer.offset:trainer.offset+batch.num_graphs]
                batch=batch.to(cfg['device'],non_blocking=True);trainer.update(batch,used)
                if trainer.step==1:
                    audit=native_audit(trainer,batch)
                    if audit is not None:
                        trainer.audits.append(audit)
                        if not audit['passed']:raise ArithmeticError('Independent native audit failed')
                if trainer.step%100==0:checkpoint()
            if trainer.offset!=len(train):raise RuntimeError('Training epoch coverage mismatch')
            if trainer.epoch in {cfg['epochs']//2,cfg['epochs']-1}:
                if batch is None:
                    last=(len(train)-1)//cfg['batch_size']*cfg['batch_size']
                    batch=next(iter(loader(data,indices[last:],mean,std,cfg['batch_size'],0,cfg['seed']))).to(cfg['device'])
                audit=native_audit(trainer,batch)
                if audit is not None:
                    trainer.audits.append(audit)
                    if not audit['passed']:raise ArithmeticError('Independent native audit failed')
            metrics=evaluate(trainer.model,data,valid,mean,std,cfg)
            if trainer.finish_epoch(metrics):
                pending=out/'selected/pending.pt'
                torch_write(pending,{'model':trainer.model.state_dict(),'epoch':trainer.best_epoch,
                    'metric':trainer.best_metric,'config':cfg,'identity':identity})
                checksum=sha(pending)
                selected=out/'selected'/f'epoch{trainer.best_epoch:04d}_step{trainer.step:08d}_{checksum[:16]}.pt'
                pending.replace(selected)
                trainer.best_checkpoint={'file':str(selected.relative_to(out)),'sha256':checksum,
                    'epoch':trainer.best_epoch,'metric':trainer.best_metric}
            checkpoint();print(json.dumps({'epoch':trainer.epoch,'steps':trainer.step,'method':cfg['system']['method'],
                'seed':cfg['seed'],'update_seconds':trainer.update_seconds,'recorded_wall_seconds':trainer.recorded_execution_seconds}),flush=True)
        if phase=='development':
            checkpoint();result=trainer.summary()
            result.update(complete=True,final_validation=trainer.history[-1]['validation'],
                heldout_test_evaluated=False,recorded_execution_seconds=prior+time.perf_counter()-attempt_begin)
            json_write(out/'metrics.json',result);backup();print('DEVELOPMENT_COMPLETE',str(out),flush=True)
            return
        final=evaluate(trainer.model,data,test,mean,std,cfg)
        final_model=copy.deepcopy(trainer.model.state_dict())
        selected_path=out/trainer.best_checkpoint['file']
        if sha(selected_path)!=trainer.best_checkpoint['sha256']:raise RuntimeError('Selected checkpoint hash mismatch')
        selected=torch.load(selected_path,map_location='cpu',weights_only=False)
        if selected['config']!=cfg or selected['identity']!=identity or selected['epoch']!=trainer.best_epoch or selected['metric']!=trainer.best_metric:
            raise RuntimeError('Selected checkpoint identity/selection mismatch')
        trainer.model.load_state_dict(selected['model']);best=evaluate(trainer.model,data,test,mean,std,cfg)
        trainer.model.load_state_dict(final_model);del final_model,selected
        checkpoint();result=trainer.summary();result.update(complete=True,heldout_test_evaluated=True,final_test=final,validation_selected_test=best)
        result['recorded_execution_seconds']=prior+time.perf_counter()-attempt_begin
        result['uninterrupted_training_evaluation_seconds']=result['recorded_execution_seconds'] if trainer.attempts==1 else None
        json_write(out/'metrics.json',result);backup();print('COMPLETE',str(out),flush=True)
    except BaseException:
        failure={'complete':False,'error':traceback.format_exc(),'step':trainer.step,'epoch':trainer.epoch,
            'identity':identity,'elapsed_current_attempt_seconds':time.perf_counter()-attempt_begin}
        json_write(out/'failure.json',failure)
        START_CONTEXT['failure_recorded']=True
        # Keep the last fully checkpointed state. Do not save partially mutated
        # model/optimizer state after a failed update as resumable.
        if args.backup:
            args.backup.mkdir(parents=True,exist_ok=True);shutil.copy2(out/'failure.json',args.backup/'failure.json')
        raise

if __name__=='__main__':
    try:main()
    except BaseException:
        if START_CONTEXT.get('output') and not START_CONTEXT.get('failure_recorded'):
            json_write(Path(START_CONTEXT['output'])/'failure.json',{'complete':False,'phase':START_CONTEXT['phase'],'error':traceback.format_exc()})
        raise
