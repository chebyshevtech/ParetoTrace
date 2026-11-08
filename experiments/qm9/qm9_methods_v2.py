"""QM9 method adapters; preserve weighting rules and expose implementation changes.

Pinned LibMTL formulas remain in weighting_upstream/. The optimized retrieval
uses a common native batched-VJP table; it does not rename that path as a new
factorization. Aligned-MTL's removed symeig call is replaced by eigh(UPLO='U').
All task-specific head conventions are preserved, including weighted heads for
Nash/Fair/FAMO and unweighted heads for MGDA/Aligned.
"""
import copy
import importlib.util
import sys
import types
from pathlib import Path
import numpy as np
import torch
from simplex_qp import qp
from shared_direct_v2 import SharedMultioutputGram


def native_table(losses, parameters, chunk=1):
    """Preallocate once; retain graph for the executed weighted backward."""
    parameters = tuple(parameters)
    size = sum(p.numel() for p in parameters)
    table = losses.new_empty((len(losses), size))
    basis = torch.eye(len(losses), device=losses.device, dtype=losses.dtype)
    for start in range(0, len(losses), chunk):
        stop = min(start + chunk, len(losses))
        if chunk == 1:
            gs = torch.autograd.grad(losses[start], parameters, retain_graph=True)
            gs = tuple(g.unsqueeze(0) for g in gs)
        else:
            gs = torch.autograd.grad(losses, parameters, basis[start:stop],
                                     retain_graph=True, is_grads_batched=True)
        offset = 0
        for p, g in zip(parameters, gs):
            table[start:stop, offset:offset+p.numel()].copy_(g.reshape(stop-start, -1))
            offset += p.numel()
        del gs, g
    return table


def gram64(table, tile=65536):
    K = torch.zeros(len(table), len(table), device=table.device, dtype=torch.float64)
    for start in range(0, table.shape[1], tile):
        block = table[:, start:start+tile].double()
        K.addmm_(block, block.T)
    return K


def set_vector(parameters, vector):
    offset = 0
    for p in parameters:
        p.grad = vector[offset:offset+p.numel()].reshape_as(p).detach().clone()
        offset += p.numel()
    if offset != vector.numel():
        raise ValueError('Shared-gradient layout mismatch')


def checked_qp(K):
    matrix = K.detach().double().cpu().numpy()
    if not np.isfinite(matrix).all():
        raise FloatingPointError('Nonfinite native Gram')
    fit = qp(matrix)
    if not fit['converged'] or not np.isfinite(fit['weights']).all():
        raise ArithmeticError('Unresolved simplex QP')
    return fit


def common_update(model, batch, method, chunk=1, row_chunk=64, return_gram=False):
    """Set gradients; caller performs and times optimizer step too."""
    shared = tuple(model.encoder.parameters())
    heads = tuple(model.decoders.parameters())
    extra = {}
    if method == 'direct_factorized':
        with SharedMultioutputGram(model.encoder, task_chunk=chunk, row_chunk=row_chunk) as capture:
            predictions = model(batch)
            losses = (predictions-batch.y).square().mean(0)
        capture.capture(losses)
        K = capture.gram()
        extra['coverage'] = capture.census
        fit = checked_qp(K)
        if return_gram: extra['audit_gram'] = K.detach().double().cpu().tolist()
        del capture, K
    elif method == 'exact_native':
        predictions = model(batch)
        losses = (predictions-batch.y).square().mean(0)
        table = native_table(losses, shared, chunk)
        K = gram64(table)
        fit = checked_qp(K)
        if return_gram: extra['audit_gram'] = K.detach().double().cpu().tolist()
        del K
        del table
    elif method in ('uniform', 'mgda_ub'):
        representation = model.encoder(batch)
        predictions = torch.cat([h(representation) for h in model.decoders], 1)
        losses = (predictions-batch.y).square().mean(0)
        if method == 'mgda_ub':
            # Pinned LibMTL representation surrogate: sum batch coordinates
            # before forming the Gram. Explicitly differs from full B*D UB.
            grads = torch.stack([torch.autograd.grad(l, representation, retain_graph=True)[0]
                                 for l in losses])
            summed = grads.sum(1)
            fit = checked_qp(gram64(summed))
            extra['representation_reduction'] = 'LibMTL batch-summed representation surrogate'
            del grads, summed
        else:
            fit = {'weights': np.full(len(losses), 1/len(losses))}
    else:
        raise ValueError(method)
    weights = torch.as_tensor(fit['weights'], device=losses.device, dtype=losses.dtype)
    shared_grad = torch.autograd.grad(losses @ weights, shared, retain_graph=True)
    head_grad = torch.autograd.grad(losses.sum(), heads)
    for p, g in zip(shared+heads, shared_grad+head_grad):
        p.grad = g.detach()
    return {'weights': weights.detach().cpu().tolist(),
            'losses': losses.detach().cpu().tolist(),
            'qp_gap': float(fit.get('dual_gap', 0)), **extra}


def load_upstream(root, selected=None):
    """Load retained source without importing unrelated LibMTL trainers."""
    root = Path(root)
    for name in ('LibMTL', 'LibMTL.weighting'):
        if name not in sys.modules:
            module = types.ModuleType(name)
            module.__path__ = [str(root)]
            sys.modules[name] = module
    modules = {}
    for name in ('abstract_weighting', *(selected or ('Nash_MTL', 'FairGrad', 'Aligned_MTL', 'FAMO'))):
        full = 'LibMTL.weighting.'+name
        module = types.ModuleType(full)
        module.__file__ = str(root/(name+'.py'))
        source = Path(module.__file__).read_text()
        if name == 'Aligned_MTL':
            old = 'torch.symeig(M, eigenvectors=True)'
            if source.count(old) != 1:
                raise ValueError('Unexpected pinned Aligned-MTL source')
            source = source.replace(old, "torch.linalg.eigh(M, UPLO='U')")
        exec(compile(source, module.__file__, 'exec'), module.__dict__)
        sys.modules[full] = module
        modules[name] = module
    return modules


class OfficialWeighting:
    """Upstream algorithms with an auditable shared retrieval optimization."""
    def __init__(self, model, method, upstream, chunk=1, optimized=True):
        modules = load_upstream(upstream, (method,))
        cls = getattr(modules[method], method)
        shared = tuple(model.encoder.parameters())
        heads = tuple(model.decoders.parameters())
        device = next(model.parameters()).device
        self.method, self.model, self.modules = method, model, modules
        class Adapter(cls):
            def get_share_params(self):
                return shared
            def zero_grad_share_params(self):
                for p in shared:
                    p.grad = None
            def _grad2vec(self):
                # Same extraction as upstream, on-device to avoid CPU roundtrips.
                return torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1)
                                  for p in shared])
            def _compute_grad(self, losses, mode, rep_grad=False):
                if not optimized:
                    return super()._compute_grad(losses, mode, rep_grad)
                if rep_grad or mode not in ('backward', 'autograd'):
                    raise ValueError('Unsupported retrieval convention')
                table = native_table(losses, shared, chunk)
                if mode == 'backward':
                    # Aligned's serial upstream backwards accumulate all private
                    # heads without weighting. The vector reset changes shared only.
                    for p, g in zip(heads, torch.autograd.grad(losses.sum(), heads, retain_graph=True)):
                        p.grad = g.detach()
                return table
            def _reset_grad(self, new_grads):
                set_vector(shared, new_grads)
        self.rule = Adapter()
        self.rule.task_num = len(model.decoders)
        self.rule.device = device
        self.rule.rep_grad = False
        self.rule.task_name = [str(j) for j in range(len(model.decoders))]
        self.rule.init_param()
        self.solver_events = []
        if method == 'Nash_MTL':
            original_init = self.rule._init_optim_problem
            def observed_init():
                original_init()
                original_solve = self.rule.prob.solve
                def observed_solve(*args, **kwargs):
                    try:
                        result = original_solve(*args, **kwargs)
                    except Exception as exc:
                        self.solver_events.append({'status': 'exception', 'error': repr(exc)})
                        raise
                    self.solver_events.append({'status': self.rule.prob.status})
                    return result
                self.rule.prob.solve = observed_solve
            self.rule._init_optim_problem = observed_init
        if method == 'FairGrad':
            original = modules[method].least_squares
            def observed_solve(*args, **kwargs):
                result = original(*args, **kwargs)
                self.solver_events.append({'success': bool(result.success), 'status': int(result.status),
                    'residual_norm': float(np.linalg.norm(result.fun)), 'nfev': int(result.nfev)})
                return result
            modules[method].least_squares = observed_solve

    def backward(self, losses):
        self.solver_events.clear()
        weights = self.rule.backward(losses, update_weights_every=1, optim_niter=20, max_norm=1.,
                                     FairGrad_alpha=1., FAMO_w_lr=.025, FAMO_w_gamma=.001)
        return {'weights': None if weights is None else np.asarray(weights).tolist(),
                'losses': losses.detach().cpu().tolist(), 'solver_events': copy.deepcopy(self.solver_events)}

    def after_step(self, new_losses):
        if self.method == 'FAMO':
            self.rule.update_w(new_losses.detach())
            self.rule.prev_losses = self.rule.prev_losses.detach()

    def state(self):
        if self.method == 'FAMO':
            return {'step': self.rule.step, 'w': self.rule.w.detach().cpu(),
                    'min_losses': self.rule.min_losses.cpu(), 'w_opt': self.rule.w_opt.state_dict()}
        if self.method == 'Nash_MTL':
            return {'step': self.rule.step, 'prvs_alpha': self.rule.prvs_alpha.copy(),
                    'normalization_factor': self.rule.normalization_factor.copy()}
        return {}

    def restore(self, state):
        if self.method == 'FAMO':
            self.rule.step = state['step']
            with torch.no_grad():
                self.rule.w.copy_(state['w'].to(self.rule.device))
                self.rule.min_losses.copy_(state['min_losses'].to(self.rule.device))
            self.rule.w_opt.load_state_dict(state['w_opt'])
        elif self.method == 'Nash_MTL':
            self.rule.prvs_alpha = state['prvs_alpha'].copy()
            self.rule.normalization_factor = state['normalization_factor'].copy()
            self.rule._init_optim_problem()
            self.rule.step = state['step']
