"""Native shared-task Gram actions without an M-by-P parameter-gradient table.

One retained forward provides VJPs. A forward-mode pass computes J(J^T v).
The supplied functional loss must be deterministic, with frozen model state.
Floating-point actions are empirically audited, not outward-rounded oracles.
"""
import torch

class SharedNativeActions:
    def __init__(self,parameters,function):
        if not parameters or any(not p.requires_grad for p in parameters.values()):
            raise ValueError('Pass only differentiated parameters')
        self.parameters=dict(parameters);self.function=function
        self.versions={n:p._version for n,p in parameters.items()}
        self.losses=function(self.parameters)
        if self.losses.ndim!=1 or not torch.isfinite(self.losses).all():
            raise ValueError('Finite one-dimensional loss vector required')
        self.M=len(self.losses);self.action_count=0;self.reconstruction_count=0

    def _check(self):
        if any(p._version!=self.versions[n] for n,p in self.parameters.items()):
            raise RuntimeError('Parameters changed during retained-graph recovery')

    def _direction(self,weights):
        self._check()
        w=torch.as_tensor(weights,device=self.losses.device,dtype=self.losses.dtype)
        if w.shape!=(self.M,) or not torch.isfinite(w).all():
            raise ValueError('Finite query of objective dimension required')
        values=torch.autograd.grad(self.losses,tuple(self.parameters.values()),w,retain_graph=True)
        return {n:v.detach() for n,v in zip(self.parameters,values)}

    def action(self,query):
        direction=self._direction(query)
        with torch.no_grad():
            _,response=torch.func.jvp(self.function,(self.parameters,),(direction,))
        self._check();self.action_count+=1
        if not torch.isfinite(response).all():raise FloatingPointError('Nonfinite native Gram action')
        return response.detach()

    def direction(self,weights):
        self.reconstruction_count+=1
        return self._direction(weights)

    def close(self):
        del self.losses

