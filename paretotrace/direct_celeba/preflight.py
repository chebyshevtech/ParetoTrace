"""A100 preflight: shared-batch multihead Direct, independently audited.

This is a systems preflight, not a downstream result. Unlike disjoint example
groups, every one of the 40 attribute losses uses the same input minibatch.
The contraction streams parameter tiles after a task-batched adjoint capture.
No benchmark score is computed on random labels.
"""
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import argparse, gc, hashlib, json, platform, time
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet18


class SharedCapture:
    """Linear/Conv2d identity with shared activations and task-axis adjoints.

    Current contract: no tied/reused trainable module, no trainable BatchNorm,
    no in-place modification of captured outputs. Reject other parameters.
    All operations are exact identities in real arithmetic; reported errors
    are empirical FP32/FP64 comparisons, not interval bounds.
    """
    def __init__(self, trunk):
        self.trunk = trunk
        self.entries = []
        self.handles = []
        self.seen = set()
        self.params = {id(p): p for p in trunk.parameters() if p.requires_grad}
        self.names = {id(p): n for n,p in trunk.named_parameters()}
        for m in trunk.modules():
            own = [p for p in m.parameters(recurse=False) if p.requires_grad]
            if own and not isinstance(m, (nn.Linear, nn.Conv2d)):
                raise TypeError(type(m).__name__)
            if isinstance(m, nn.Conv2d) and m.groups != 1:
                raise ValueError('grouped convolution unsupported')

    def __enter__(self):
        def hook(m, args, out):
            if id(m) in self.seen:
                raise ValueError('reused module requires summed occurrence gradients')
            self.seen.add(id(m))
            self.entries.append((m,args[0].detach(),out))
        for m in self.trunk.modules():
            if any(id(p) in self.params for p in m.parameters(recurse=False)):
                self.handles.append(m.register_forward_hook(hook))
        return self

    def __exit__(self, *exc):
        for h in self.handles: h.remove()

    def capture(self, losses):
        m = len(losses)
        eye = torch.eye(m, device=losses.device, dtype=losses.dtype)
        outputs = tuple(o for _,_,o in self.entries)
        self.adjoints = torch.autograd.grad(losses, outputs, eye,
                            is_grads_batched=True, retain_graph=True)
        self.M = m

    def tiles(self, dtype=torch.float64, tile_bytes=64*1024**2):
        for (mod, x, _), adj in zip(self.entries,self.adjoints):
            if isinstance(mod, nn.Conv2d):
                a = F.unfold(x,mod.kernel_size,dilation=mod.dilation,
                             padding=mod.padding,stride=mod.stride)
                # B,I,L -> (B L),I; M,B,O,L -> M,O,(B L).
                a = a.permute(0,2,1).reshape(-1,a.shape[1]).to(dtype)
                b = adj.flatten(3).permute(0,2,1,3).reshape(self.M,mod.out_channels,-1)
                rows = mod.out_channels
            else:
                a = x.reshape(-1,x.shape[-1]).to(dtype)
                b = adj.reshape(self.M,-1,mod.out_features).transpose(1,2)
                rows = mod.out_features
            chunk = max(1,min(rows,tile_bytes//(self.M*a.shape[1]*torch.empty((),dtype=dtype).element_size())))
            for st in range(0,rows,chunk):
                stop = min(st+chunk,rows)
                g = torch.matmul(b[:,st:stop].to(dtype),a).reshape(self.M,-1)
                yield mod.weight, st, stop, g
            if mod.bias is not None and id(mod.bias) in self.params:
                yield mod.bias, 0, rows, b.to(dtype).sum(-1)

    def gram(self, dtype=torch.float64, tile_bytes=64*1024**2):
        out = torch.zeros(self.M,self.M,device=self.adjoints[0].device,dtype=dtype)
        for _,_,_,g in self.tiles(dtype,tile_bytes): out.addmm_(g,g.T)
        return out


def sync():
    if torch.cuda.is_available():torch.cuda.synchronize()


def main():
    p=argparse.ArgumentParser();p.add_argument('--batch',type=int,default=32)
    p.add_argument('--tasks',type=int,default=40);p.add_argument('--repeats',type=int,default=2)
    p.add_argument('--output',default='/content/rebuttal/preflight.json')
    p.add_argument('--device',default='cuda');a=p.parse_args()
    torch.set_num_threads(1);torch.manual_seed(1701)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    model=resnet18(weights=None);model.fc=nn.Identity();model=model.to(a.device).eval()
    for m in model.modules():
        if isinstance(m,nn.BatchNorm2d):
            for t in m.parameters(recurse=False):t.requires_grad_(False)
        if isinstance(m,nn.ReLU):m.inplace=False
    heads=nn.Linear(512,a.tasks).to(a.device)
    x=torch.randn(a.batch,3,64,64,device=a.device)
    y=torch.randint(0,2,(a.batch,a.tasks),device=a.device).float()
    params=tuple(p for p in model.parameters() if p.requires_grad)
    result={'protocol':'shared_attribute_direct_preflight_v1','batch':a.batch,
       'tasks':a.tasks,'input':'random; systems/fidelity preflight only',
       'gpu':torch.cuda.get_device_name() if a.device=='cuda' else None,
       'torch':torch.__version__,'python':platform.python_version(),
       'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'rows':[],'complete':False}
    path=Path(a.output);path.parent.mkdir(parents=True,exist_ok=True)
    for arm in ['serial','direct_fp32','direct_fp64']:
      for rep in range(a.repeats+1):
        gc.collect()
        if a.device=='cuda':torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
        sync();start=time.perf_counter()
        if arm=='serial':
            losses=F.binary_cross_entropy_with_logits(heads(model(x)),y,reduction='none').mean(0)
            G=torch.stack([torch.cat([g.reshape(-1) for g in torch.autograd.grad(losses[i],params,retain_graph=True)]) for i in range(a.tasks)])
            K=torch.zeros(a.tasks,a.tasks,device=a.device,dtype=torch.float64)
            for st in range(0,G.shape[1],65536):
                t=G[:,st:st+65536].double();K.addmm_(t,t.T)
            del G,t
        else:
            with SharedCapture(model) as cap:
                losses=F.binary_cross_entropy_with_logits(heads(model(x)),y,reduction='none').mean(0)
            cap.capture(losses)
            K=cap.gram(torch.float32 if arm=='direct_fp32' else torch.float64)
            del cap
        # Common second backward reconstructs the same known uniform direction.
        # QP time is deliberately not included: this preflight validates kernels.
        d=torch.autograd.grad(losses.mean(),params)
        sync();seconds=time.perf_counter()-start
        kc=K.detach().double().cpu()
        if arm=='serial':reference=kc
        err=float(torch.linalg.vector_norm(kc-reference)/torch.linalg.vector_norm(reference))
        row={'arm':arm,'repeat':rep,'warmup':rep==0,'seconds':seconds,
          'peak_bytes':torch.cuda.max_memory_allocated() if a.device=='cuda' else None,
          'gram_relative_error':err,'positive_uniform_energy':float(kc.mean())}
        result['rows'].append(row);print(json.dumps(row),flush=True)
        path.write_text(json.dumps(result,indent=2,allow_nan=False))
        if not bool(torch.isfinite(kc).all()) or not bool(torch.isfinite(torch.tensor(err))) or err>2e-5:raise ArithmeticError('native Gram fidelity gate failed')
        del d,K,losses
    result['complete']=True;path.write_text(json.dumps(result,indent=2,allow_nan=False))

if __name__=='__main__':main()
