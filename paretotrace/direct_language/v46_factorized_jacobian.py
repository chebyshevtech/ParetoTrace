"""Exact structured per-example Jacobian actions from one adjoint capture.

Contract: losses are a vector of independent per-example scalar losses; every
parameter occurrence is inside a supported Linear, Embedding, or LayerNorm
module; batch is the first axis. No cross-example BatchNorm, functional uses of
parameters, checkpoint recomputation, or parameter updates during cache reuse.
Dropout is permitted: the cache describes the one realized forward pass.

Example::

    with FactorizedJacobian(model, task_ids, freeze_parameters=True) as cache:
        cache.capture(per_example_loss(model(inputs), targets))
    response = cache.matvec(weights)      # K @ weights, no autograd
    direction = cache.direction(weights)  # name -> weighted parameter gradient

The original trainable parameters are temporarily frozen during capture. Any
supported module output without an autograd history becomes a leaf anchor, so
adjoints propagate while parameter-gradient GEMMs are skipped. Shared/tied
parameter occurrences are added BEFORE dual contraction, preserving cross
terms. An unsupported module owning a trainable parameter fails immediately.

This is an exact-real-arithmetic identity, not a finite-precision PSD guarantee.
Compute with one stated dtype and audit numerical actions before certification.
Layerwise Gram and TorchJD autogram are required strong comparison baselines.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F


def _parameter_row_ranges(parameter, num_tasks, dtype, row_chunk, memory_budget_bytes):
    rows = parameter.shape[0]
    chunk = rows if row_chunk is None else min(rows, int(row_chunk))
    if chunk < 1:
        raise ValueError("row_chunk must be positive")
    if memory_budget_bytes is not None:
        bytes_per_row = (num_tasks * (parameter.numel() // rows)
                         * torch.empty((), dtype=dtype).element_size())
        available = int(memory_budget_bytes) // bytes_per_row
        if available < 1:
            raise ValueError("Gram memory budget cannot hold one parameter row per task")
        chunk = min(chunk, available)
    return range(0, rows, chunk), chunk


def _parameter_tiles(parameter, num_tasks, dtype, row_chunk, memory_budget_bytes, subset):
    starts, chunk = _parameter_row_ranges(parameter, num_tasks, dtype, row_chunk,
                                          memory_budget_bytes)
    if subset is None:
        for start in starts:
            yield start, min(start + chunk, parameter.shape[0]), None
        return
    rows = torch.as_tensor(subset, device=parameter.device, dtype=torch.long)
    if rows.ndim != 1:
        raise ValueError("row subsets must be one-dimensional")
    rows = torch.unique(rows, sorted=True)
    if rows.numel() and bool(((rows < 0) | (rows >= parameter.shape[0])).any()):
        raise ValueError("selected parameter row is out of range")
    for start in range(0, rows.numel(), chunk):
        yield 0, rows[start:start + chunk].numel(), rows[start:start + chunk]


@dataclass
class _Term:
    parameter_id: int
    kind: str
    left: torch.Tensor
    right: Optional[torch.Tensor] = None
    padding_idx: Optional[int] = None


class FactorizedJacobian:
    """Cache a separable loss Jacobian without its task-by-parameter table.

    ``compute_dtype`` defaults to the first trainable parameter dtype. FP64 is
    recommended for the identity audit; FP32/BF16 require measured error bars.
    Features are converted once to this dtype at capture, so storage is honest.
    ``direction`` uses O(P) output; ``matvec`` streams one shared parameter at a
    time, using O(max_parameter_size) gradient scratch beyond cached features.
    """

    def __init__(self, model: nn.Module, task_ids: torch.Tensor,
                 num_tasks: Optional[int] = None, *, freeze_parameters=True,
                 compute_dtype=None, allow_empty_tasks=False, compact_zero_rows=False):
        self.model = model
        self.parameters = {id(p): p for p in model.parameters() if p.requires_grad}
        if not self.parameters:
            raise ValueError("model has no trainable parameters")
        self.names = {id(p): n for n, p in model.named_parameters()
                      if id(p) in self.parameters}
        first = next(iter(self.parameters.values()))
        self.dtype = first.dtype if compute_dtype is None else compute_dtype
        self.device = first.device
        self.task_ids = torch.as_tensor(task_ids, device=self.device, dtype=torch.long)
        if self.task_ids.ndim != 1 or self.task_ids.numel() == 0:
            raise ValueError("task_ids must be a nonempty vector")
        self.batch_size = self.task_ids.numel()
        self.num_tasks = int(self.task_ids.max().item()) + 1 if num_tasks is None else int(num_tasks)
        if self.task_ids.min().item() < 0 or self.task_ids.max().item() >= self.num_tasks:
            raise ValueError("task_ids out of range")
        self.counts = torch.bincount(self.task_ids, minlength=self.num_tasks).to(self.dtype)
        if not allow_empty_tasks and bool((self.counts == 0).any()):
            raise ValueError("every objective needs at least one example")
        self.allow_empty_tasks = bool(allow_empty_tasks)
        self.task_examples = [(self.task_ids == i).nonzero(as_tuple=True)[0]
                              for i in range(self.num_tasks)]
        self._task_counts_cpu = [int(examples.numel()) for examples in self.task_examples]
        self._equal_task_count = (self._task_counts_cpu[0]
                                  if len(set(self._task_counts_cpu)) == 1 else None)
        self._task_order = torch.cat(self.task_examples)
        self.freeze_parameters = bool(freeze_parameters)
        self.compact_zero_rows = bool(compact_zero_rows)
        self.linear_rows_before = 0
        self.linear_rows_after = 0
        self.terms = {pid: [] for pid in self.parameters}
        self._hooks = []
        self._entered = False
        self._captured = False
        self._pending = 0
        self._anchors = 0
        self._anchor_tensors = []
        self.losses = None
        self._validate_modules()

    def _validate_modules(self):
        for module in self.model.modules():
            own = [p for p in module.parameters(recurse=False) if id(p) in self.parameters]
            if not own:
                continue
            if not isinstance(module, (nn.Linear, nn.Embedding, nn.LayerNorm,
                                       nn.Conv2d)):
                raise TypeError("unsupported trainable module: " + type(module).__name__)
            if isinstance(module, nn.Conv2d) and module.groups != 1:
                raise ValueError("Conv2d groups != 1 unsupported")
            if isinstance(module, nn.Embedding):
                if module.scale_grad_by_freq or module.max_norm is not None or module.sparse:
                    raise ValueError("Embedding scale_grad_by_freq/max_norm/sparse unsupported")

    def __enter__(self):
        if self._entered or self._captured:
            raise RuntimeError("a cache captures exactly one forward/backward")
        if not torch.is_grad_enabled():
            raise RuntimeError("capture requires autograd enabled")
        self._entered = True
        if self.freeze_parameters:
            for parameter in self.parameters.values():
                parameter.requires_grad_(False)
        for module in self.model.modules():
            if any(id(p) in self.parameters for p in module.parameters(recurse=False)):
                self._hooks.append(module.register_forward_hook(self._forward_hook))
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()
        self._anchor_tensors.clear()
        if self.freeze_parameters:
            for parameter in self.parameters.values():
                parameter.requires_grad_(True)
        self._entered = False
        return False

    def _convert(self, tensor):
        return tensor.detach().to(dtype=self.dtype)

    def _append(self, parameter, kind, left, right=None, padding_idx=None):
        if parameter is not None and id(parameter) in self.terms:
            self.terms[id(parameter)].append(_Term(id(parameter), kind, left, right, padding_idx))

    def _forward_hook(self, module, inputs, output):
        if self._captured:
            raise RuntimeError("new forward while capture context remains open")
        if not isinstance(output, torch.Tensor) or not inputs:
            raise TypeError("supported modules must have a single tensor output")
        x = inputs[0].detach()
        if x.shape[0] != self.batch_size or output.shape[0] != self.batch_size:
            raise ValueError("every supported module must preserve the first batch axis")
        if not output.requires_grad:
            output.requires_grad_(True)
            self._anchors += 1
            self._anchor_tensors.append(output)
        self._pending += 1

        def adjoint_hook(adjoint):
            if self._captured:
                raise RuntimeError("cache cannot capture a second backward")
            delta = self._convert(adjoint)
            batch = self.batch_size
            if isinstance(module, nn.Linear):
                a = self._convert(x).reshape(batch, -1, module.in_features)
                d = delta.reshape(batch, -1, module.out_features)
                self.linear_rows_before += batch * d.shape[1]
                if self.compact_zero_rows:
                    # Drop EXACT zero adjoint rows. Different examples may
                    # retain different positions; pad only up to the largest
                    # retained count so grouped GEMMs keep a regular batch.
                    active = d.ne(0).any(dim=-1)
                    kept = int(active.sum(dim=1).max().item())
                    if kept < d.shape[1]:
                        positions = active.to(torch.int8).argsort(
                            dim=1, descending=True, stable=True)[:, :kept]
                        a = a.gather(1, positions[:, :, None].expand(-1, -1, a.shape[-1]))
                        d = d.gather(1, positions[:, :, None].expand(-1, -1, d.shape[-1]))
                self.linear_rows_after += batch * d.shape[1]
                self._append(module.weight, "linear_weight", d, a)
                self._append(module.bias, "direct", d.sum(dim=1))
            elif isinstance(module, nn.Conv2d):
                # Per-example weight gradient is sum_l delta[:,co,l] * patch[:,d,l]
                # with patch = unfold(input). That is the same bilinear form as
                # Linear, with l ranging over output positions, so the existing
                # linear_weight contraction applies once the weight is viewed
                # as (out_channels, in_channels * kh * kw).
                a = F.unfold(self._convert(x), module.kernel_size,
                             dilation=module.dilation, padding=module.padding,
                             stride=module.stride).transpose(1, 2)
                d = delta.reshape(batch, module.out_channels, -1).transpose(1, 2)
                self.linear_rows_before += batch * d.shape[1]
                self.linear_rows_after += batch * d.shape[1]
                self._append(module.weight, "linear_weight", d, a)
                self._append(module.bias, "direct", d.sum(dim=1))
            elif isinstance(module, nn.Embedding):
                indices = x.reshape(batch, -1).to(dtype=torch.long)
                d = delta.reshape(batch, -1, module.embedding_dim)
                self._append(module.weight, "embedding", d, indices, module.padding_idx)
            elif isinstance(module, nn.LayerNorm):
                shape = tuple(module.normalized_shape)
                norm_axes = tuple(range(x.ndim - len(shape), x.ndim))
                xx = self._convert(x)
                mean = xx.mean(dim=norm_axes, keepdim=True)
                variance = ((xx - mean) ** 2).mean(dim=norm_axes, keepdim=True)
                xhat = (xx - mean) * torch.rsqrt(variance + module.eps)
                # Positions are summed; the normalized parameter axes remain.
                reduce_axes = tuple(range(1, x.ndim - len(shape)))
                gamma = delta * xhat
                beta = delta
                if reduce_axes:
                    gamma, beta = gamma.sum(dim=reduce_axes), beta.sum(dim=reduce_axes)
                self._append(module.weight, "direct", gamma)
                self._append(module.bias, "direct", beta)
            self._pending -= 1
            return None

        output.register_hook(adjoint_hook)
        return output

    def capture(self, per_example_losses: torch.Tensor):
        """Perform the sole backward; losses must already normalize within example."""
        if not self._entered or self._captured:
            raise RuntimeError("capture must occur once inside its context")
        if per_example_losses.shape != (self.batch_size,):
            raise ValueError("losses must have shape (batch_size,)")
        self.losses = per_example_losses.detach().to(self.dtype)
        per_example_losses.sum().backward()
        for anchor in self._anchor_tensors:
            anchor.grad = None
        self._anchor_tensors.clear()
        # An unused module invocation can have no adjoint. It contributes zero.
        self.unused_invocations = self._pending
        self._captured = True
        self._parameter_versions = {pid: p._version for pid, p in self.parameters.items()}
        return self

    def _check_current(self):
        if not self._captured:
            raise RuntimeError("capture a forward/backward first")
        if any(p._version != self._parameter_versions[pid]
               for pid, p in self.parameters.items()):
            raise RuntimeError("parameters changed after capture; rebuild the Jacobian cache")

    def _coefficients(self, weights):
        self._check_current()
        weights = torch.as_tensor(weights, device=self.device, dtype=self.dtype)
        if weights.shape != (self.num_tasks,):
            raise ValueError("weights must have shape (num_tasks,)")
        return weights[self.task_ids] / self.counts[self.task_ids]

    def _gradient(self, pid, coefficients, examples=None, out=None):
        parameter = self.parameters[pid]
        result = (torch.zeros(parameter.shape, device=self.device, dtype=self.dtype)
                  if out is None else out)
        for term in self.terms[pid]:
            left = term.left if examples is None else term.left.index_select(0, examples)
            right = term.right
            if examples is not None and right is not None:
                right = right.index_select(0, examples)
            if term.kind == "linear_weight":
                d, a = left, right
                # A convolution weight arrives as (out, in, kh, kw); the
                # contraction addresses it as (out, in * kh * kw).
                flat = result.reshape(result.shape[0], -1)
                # Scale the smaller factor: a vocabulary head should never
                # allocate an N-by-vocabulary weighted adjoint temporary.
                if a.shape[-1] <= d.shape[-1]:
                    weighted = a * coefficients[:, None, None]
                    flat.addmm_(d.reshape(-1, d.shape[-1]).T,
                                weighted.reshape(-1, a.shape[-1]))
                else:
                    weighted = d * coefficients[:, None, None]
                    flat.addmm_(weighted.reshape(-1, d.shape[-1]).T,
                                a.reshape(-1, a.shape[-1]))
            elif term.kind == "direct":
                contribution = torch.tensordot(coefficients, left, dims=([0], [0]))
                result.add_(contribution)
            elif term.kind == "embedding":
                d, indices = left, right
                weighted = d * coefficients[:, None, None]
                if term.padding_idx is not None:
                    weighted = weighted * (indices != term.padding_idx).unsqueeze(-1)
                result.index_add_(0, indices.reshape(-1), weighted.reshape(-1, d.shape[-1]))
        return result

    def _dual(self, pid, gradient):
        response = torch.zeros(self.batch_size, device=self.device, dtype=self.dtype)
        for term in self.terms[pid]:
            if term.kind == "linear_weight":
                gradient = gradient.reshape(gradient.shape[0], -1)
                if term.right.shape[-1] <= term.left.shape[-1]:
                    response.add_((term.right * (term.left @ gradient)).sum(dim=(1, 2)))
                else:
                    response.add_((term.left * (term.right @ gradient.T)).sum(dim=(1, 2)))
            elif term.kind == "direct":
                response.add_((term.left * gradient).reshape(self.batch_size, -1).sum(dim=1))
            elif term.kind == "embedding":
                inner = (term.left * gradient[term.right]).sum(dim=-1)
                if term.padding_idx is not None:
                    inner = inner * (term.right != term.padding_idx)
                response.add_(inner.sum(dim=1))
        return response

    def _task_gradient_sums(self, pid, out=None, row_start=0, row_stop=None,
                            row_indices=None):
        """Strong grouped kernel: batched GEMM when objective counts match.

        Returns unnormalized per-task gradient SUMS; callers divide once by
        local or global counts. Shared occurrences and fragments accumulate
        into the same tensor before Gram products.
        """
        parameter = self.parameters[pid]
        row_stop = parameter.shape[0] if row_stop is None else row_stop
        rows = row_stop - row_start if row_indices is None else row_indices.numel()
        tile_shape = (rows, *parameter.shape[1:])
        result = (torch.zeros((self.num_tasks, *tile_shape),
                              device=self.device, dtype=self.dtype)
                  if out is None else out)
        for term in self.terms[pid]:
            if term.kind == "linear_weight":
                flat = result.reshape(result.shape[0], result.shape[1], -1)
                d = (term.left[..., row_start:row_stop] if row_indices is None
                     else term.left.index_select(-1, row_indices))
                a = term.right
                count = self._equal_task_count
                if count is not None:
                    dg = d.index_select(0, self._task_order).reshape(self.num_tasks, -1, d.shape[-1])
                    ag = a.index_select(0, self._task_order).reshape(self.num_tasks, -1, a.shape[-1])
                    flat.baddbmm_(dg.transpose(1, 2), ag)
                else:
                    for i, examples in enumerate(self.task_examples):
                        if examples.numel():
                            di = d.index_select(0, examples).reshape(-1, d.shape[-1])
                            ai = a.index_select(0, examples).reshape(-1, a.shape[-1])
                            flat[i].addmm_(di.T, ai)
            elif term.kind == "direct":
                contribution = (term.left[:, row_start:row_stop] if row_indices is None
                                else term.left.index_select(1, row_indices))
                result.index_add_(0, self.task_ids, contribution)
            elif term.kind == "embedding":
                d, indices = term.left, term.right
                if row_indices is None:
                    mask = (indices >= row_start) & (indices < row_stop)
                    tile_indices = (indices - row_start).clamp(0, rows - 1)
                else:
                    locations = torch.searchsorted(row_indices, indices)
                    tile_indices = locations.clamp_max(rows - 1)
                    mask = (locations < rows) & (row_indices[tile_indices] == indices)
                if term.padding_idx is not None:
                    mask = mask & (indices != term.padding_idx)
                d = d * mask.unsqueeze(-1)
                flat_indices = tile_indices + self.task_ids[:, None] * rows
                result.reshape(-1, parameter.shape[-1]).index_add_(
                    0, flat_indices.reshape(-1), d.reshape(-1, d.shape[-1]))
        return result

    @torch.no_grad()
    def direction(self, weights):
        """Reconstruct G weights, with every shared occurrence counted once."""
        coefficients = self._coefficients(weights)
        return {self.names[pid]: self._gradient(pid, coefficients) for pid in self.parameters}

    @torch.no_grad()
    def matvec(self, weights, *, return_direction=False):
        """Compute K weights with one gradient scratch tensor at a time."""
        coefficients = self._coefficients(weights)
        example_response = torch.zeros(self.batch_size, device=self.device, dtype=self.dtype)
        direction = {} if return_direction else None
        for pid in self.parameters:
            gradient = self._gradient(pid, coefficients)
            example_response.add_(self._dual(pid, gradient))
            if return_direction:
                direction[self.names[pid]] = gradient
            del gradient
        response = torch.zeros(self.num_tasks, device=self.device, dtype=self.dtype)
        response.index_add_(0, self.task_ids, example_response)
        response.div_(self.counts.clamp_min(1))
        return (response, direction) if return_direction else response

    @torch.no_grad()
    def grouped_gram(self, *, row_chunk=None, memory_budget_bytes=None, parameter_chunk=None,
                     exclude_parameter_ids=None, row_subsets=None):
        """Strong layer/parameter-streamed full-Gram baseline, no full M*P table.

        Form output-row tiles of M task gradients and add H H^T. ``row_chunk``
        or ``memory_budget_bytes`` bounds the gradient tile during CONSTRUCTION,
        so each parameter tile is computed once without an M*P allocation.
        The budget excludes features, Gram output, and GEMM workspace. A minimum
        one-row-per-task tile must fit. Legacy ``parameter_chunk`` only slices
        the Gram multiplication within an already constructed tile.
        """
        self._check_current()
        excluded = set() if exclude_parameter_ids is None else set(exclude_parameter_ids)
        subsets = {} if row_subsets is None else row_subsets
        if not excluded.issubset(self.parameters) or not set(subsets).issubset(self.parameters):
            raise ValueError("unknown parameter identity in Gram selection")
        gram = torch.zeros((self.num_tasks, self.num_tasks), device=self.device, dtype=self.dtype)
        for pid in self.parameters:
            if pid in excluded:
                continue
            parameter = self.parameters[pid]
            for start, stop, selected in _parameter_tiles(
                    parameter, self.num_tasks, self.dtype, row_chunk,
                    memory_budget_bytes, subsets.get(pid)):
                h = self._task_gradient_sums(pid, row_start=start,
                                            row_stop=stop, row_indices=selected).reshape(self.num_tasks, -1)
                h.div_(self.counts.clamp_min(1)[:, None])
                if parameter_chunk is None:
                    gram.addmm_(h, h.T)
                else:
                    for column in range(0, h.shape[1], parameter_chunk):
                        block = h[:, column:column + parameter_chunk]
                        gram.addmm_(block, block.T)
                    del block
                del h
        return gram

    def memory_report(self):
        """Tensor payload accounting, not allocator peak or backward peak."""
        tensors = [self.task_ids, self.counts, self._task_order, *self.task_examples]
        if self.losses is not None:
            tensors.append(self.losses)
        for terms in self.terms.values():
            for term in terms:
                tensors.append(term.left)
                if term.right is not None:
                    tensors.append(term.right)
        storages = {}
        for tensor in tensors:
            storage = tensor.untyped_storage()
            storages[(str(tensor.device), storage.data_ptr())] = storage.nbytes()
        element_size = torch.empty((), dtype=self.dtype).element_size()
        return {
            "feature_tensor_bytes": sum(t.numel() * t.element_size() for t in tensors),
            "feature_unique_storage_bytes": sum(storages.values()),
            "direction_bytes": sum(p.numel() for p in self.parameters.values()) * element_size,
            "largest_parameter_scratch_bytes": max(p.numel() for p in self.parameters.values()) * element_size,
            "task_gradient_table_bytes": self.num_tasks * sum(p.numel() for p in self.parameters.values()) * element_size,
            "module_terms": sum(map(len, self.terms.values())),
            "anchors": self._anchors,
            "capture_backward_passes": int(self._captured),
            "linear_token_rows_before": self.linear_rows_before,
            "linear_token_rows_after": self.linear_rows_after,
        }


class MicrobatchedFactorizedJacobian:
    """Global exact operator assembled from independent capture fragments.

    Construct each fragment using ``allow_empty_tasks=True`` and the same
    ``num_tasks``. Global task means use the sum of all fragment counts. A
    parameter's contributions are summed across ALL fragments before any dual
    contraction. Summing the fragments' separate Grams would be incorrect.
    ``matvec`` streams one parameter tensor, plus cached features and O(N+M)
    responses. Capture requires one backward per microbatch, not one per task.
    """

    def __init__(self, caches):
        self.caches = list(caches)
        if not self.caches:
            raise ValueError("at least one captured fragment is required")
        first = self.caches[0]
        self.parameters, self.names = first.parameters, first.names
        self.device, self.dtype, self.num_tasks = first.device, first.dtype, first.num_tasks
        for cache in self.caches:
            if not cache._captured:
                raise ValueError("all fragments must already be captured")
            if (cache.model is not first.model or cache.parameters.keys() != self.parameters.keys()
                    or cache.device != self.device or cache.dtype != self.dtype
                    or cache.num_tasks != self.num_tasks):
                raise ValueError("fragments must share model parameters, dtype, device and objectives")
            cache._check_current()
        self.counts = torch.stack([cache.counts for cache in self.caches]).sum(dim=0)
        if bool((self.counts == 0).any()):
            raise ValueError("every global objective needs at least one example")
        self.batch_size = sum(cache.batch_size for cache in self.caches)

    def _coefficients(self, weights):
        self.caches[0]._check_current()
        weights = torch.as_tensor(weights, device=self.device, dtype=self.dtype)
        if weights.shape != (self.num_tasks,):
            raise ValueError("weights must have shape (num_tasks,)")
        return [weights[c.task_ids] / self.counts[c.task_ids] for c in self.caches]

    def _gradient(self, pid, coefficients):
        gradient = torch.zeros(self.parameters[pid].shape, device=self.device, dtype=self.dtype)
        for cache, coeff in zip(self.caches, coefficients):
            cache._gradient(pid, coeff, out=gradient)
        return gradient

    @torch.no_grad()
    def direction(self, weights):
        coefficients = self._coefficients(weights)
        return {self.names[pid]: self._gradient(pid, coefficients) for pid in self.parameters}

    @torch.no_grad()
    def matvec(self, weights, *, return_direction=False):
        coefficients = self._coefficients(weights)
        examples = [torch.zeros(cache.batch_size, device=self.device, dtype=self.dtype)
                    for cache in self.caches]
        direction = {} if return_direction else None
        for pid in self.parameters:
            gradient = self._gradient(pid, coefficients)
            for cache, example_response in zip(self.caches, examples):
                example_response.add_(cache._dual(pid, gradient))
            if return_direction:
                direction[self.names[pid]] = gradient
            del gradient
        response = torch.zeros(self.num_tasks, device=self.device, dtype=self.dtype)
        for cache, example_response in zip(self.caches, examples):
            response.index_add_(0, cache.task_ids, example_response)
        response.div_(self.counts)
        return (response, direction) if return_direction else response

    @torch.no_grad()
    def grouped_gram(self, *, row_chunk=None, memory_budget_bytes=None,
                     exclude_parameter_ids=None, row_subsets=None):
        """Parameter-streamed strong full Gram, including cross-fragment terms."""
        self.caches[0]._check_current()
        excluded = set() if exclude_parameter_ids is None else set(exclude_parameter_ids)
        subsets = {} if row_subsets is None else row_subsets
        if not excluded.issubset(self.parameters) or not set(subsets).issubset(self.parameters):
            raise ValueError("unknown parameter identity in Gram selection")
        gram = torch.zeros((self.num_tasks, self.num_tasks), device=self.device, dtype=self.dtype)
        for pid, parameter in self.parameters.items():
            if pid in excluded:
                continue
            for start, stop, selected in _parameter_tiles(
                    parameter, self.num_tasks, self.dtype, row_chunk,
                    memory_budget_bytes, subsets.get(pid)):
                h = torch.zeros((self.num_tasks, stop - start, *parameter.shape[1:]),
                                device=self.device, dtype=self.dtype)
                for cache in self.caches:
                    cache._task_gradient_sums(pid, out=h, row_start=start, row_stop=stop,
                                              row_indices=selected)
                h = h.reshape(self.num_tasks, -1)
                h.div_(self.counts[:, None])
                gram.addmm_(h, h.T)
                del h
        return gram

    def memory_report(self):
        reports = [cache.memory_report() for cache in self.caches]
        result = {
            "feature_tensor_bytes": sum(r["feature_tensor_bytes"] for r in reports),
            # This sum can conservatively double-count storage shared by slices.
            "feature_storage_bytes_upper_bound": sum(r["feature_unique_storage_bytes"] for r in reports),
            "direction_bytes": reports[0]["direction_bytes"],
            "largest_parameter_scratch_bytes": reports[0]["largest_parameter_scratch_bytes"],
            "task_gradient_table_bytes": reports[0]["task_gradient_table_bytes"],
            "module_terms": sum(r["module_terms"] for r in reports),
            "capture_backward_passes": len(self.caches),
            "microbatches": len(self.caches),
            "linear_token_rows_before": sum(r["linear_token_rows_before"] for r in reports),
            "linear_token_rows_after": sum(r["linear_token_rows_after"] for r in reports),
        }
        return result
