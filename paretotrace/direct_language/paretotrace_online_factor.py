"""Single-capture Direct Gram with per-parameter factor retirement.

The forward census must finish before capture(). All occurrences of a shared
parameter are contracted together. This bounds retained factors by the live
parameter frontier, NOT universally by the largest layer. An optional payload
limit aborts captures with an excessively wide frontier; it is not an allocator
peak limit. The normal autograd tape and contraction preparation still exist.
"""
from __future__ import annotations

import torch
from torch import nn

from v46_factorized_jacobian import FactorizedJacobian
from v46_factorized_exact import ExactFactorGram


class OnlineFactorGram(FactorizedJacobian):
    """Retire completed parameter factors while backward is running.

    One full logical batch is supported, with local task counts defining its
    objective means. Do not sum separately captured microbatch Grams. Losses
    must already normalize within example, as for FactorizedJacobian.

    ``on_parameter(pid, contribution, plan)`` runs under no_grad before factor
    retirement. Callbacks retaining tensors increase memory outside our budget.
    A post-solve direction requires another forward/backward, with identical
    randomness if the model is stochastic. This object is not a Jacobian cache.
    """

    def __init__(self, model, task_ids, num_tasks=None, *, compute_dtype=None,
                 compact_zero_rows=True, row_chunk=256,
                 scratch_bytes=128 * 2**20, max_pending_factor_bytes=None,
                 method_overrides=None, on_parameter=None):
        super().__init__(model, task_ids, num_tasks, freeze_parameters=True,
                         compute_dtype=compute_dtype,
                         compact_zero_rows=compact_zero_rows)
        if row_chunk < 1 or scratch_bytes < 1:
            raise ValueError("row_chunk and scratch_bytes must be positive")
        if max_pending_factor_bytes is not None and max_pending_factor_bytes < 1:
            raise ValueError("max_pending_factor_bytes must be positive")
        self.max_pending_factor_bytes = max_pending_factor_bytes
        self.method_overrides = {} if method_overrides is None else dict(method_overrides)
        if not set(self.method_overrides).issubset(self.parameters):
            raise ValueError("unknown parameter identity in method overrides")
        self.on_parameter = on_parameter
        self._engine = ExactFactorGram(self, row_chunk=row_chunk, scratch_bytes=scratch_bytes)
        self._remaining = {pid: 0 for pid in self.parameters}
        self._retired = set()
        self._output_hooks = {}
        self._payloads = {}
        self._sealed = False
        self._running = False
        self._failed = False
        self._gram = torch.zeros((self.num_tasks, self.num_tasks), device=self.device, dtype=self.dtype)
        self.last_plan = []
        self.pending_factor_bytes = 0
        self.peak_pending_factor_bytes = 0
        self.peak_pending_terms = 0
        self._pending_terms = 0
        self._bytes_by_pid = {pid: 0 for pid in self.parameters}

    def _append(self, parameter, kind, left, right=None, padding_idx=None):
        if parameter is None or id(parameter) not in self.terms:
            return
        pid = id(parameter)
        if pid in self._retired:
            raise RuntimeError("late contribution after parameter retirement")
        size = left.numel() * left.element_size()
        if right is not None:
            size += right.numel() * right.element_size()
        self.pending_factor_bytes += size
        self._bytes_by_pid[pid] += size
        self._pending_terms += 1
        self.peak_pending_factor_bytes = max(self.peak_pending_factor_bytes, self.pending_factor_bytes)
        self.peak_pending_terms = max(self.peak_pending_terms, self._pending_terms)
        if (self.max_pending_factor_bytes is not None
                and self.pending_factor_bytes > self.max_pending_factor_bytes):
            raise MemoryError("online pending-factor payload limit exceeded; shared/reused parameter frontier is too large")
        super()._append(parameter, kind, left, right, padding_idx)

    def _forward_hook(self, module, inputs, output):
        if self._sealed:
            raise RuntimeError("forward/recomputation during or after capture is unsupported")
        if not isinstance(output, torch.Tensor) or not inputs:
            raise TypeError("supported modules must have a single tensor output")
        x = inputs[0].detach()
        if x.shape[0] != self.batch_size or output.shape[0] != self.batch_size:
            raise ValueError("every supported module must preserve the first batch axis")
        if not output.requires_grad:
            output.requires_grad_(True)
            self._anchors += 1
            self._anchor_tensors.append(output)
        pids = {id(p) for p in module.parameters(recurse=False) if id(p) in self.parameters}
        for pid in pids:
            self._remaining[pid] += 1
        self._pending += 1
        token = object()
        # Clearing this payload and removing the hook releases input references,
        # including for invocations disconnected from the supplied loss.
        self._payloads[token] = x

        def adjoint_hook(adjoint):
            if not self._running:
                raise RuntimeError("use capture() for the sole backward")
            x = self._payloads.pop(token)
            self._output_hooks.pop(token).remove()
            with torch.no_grad():
                delta = self._convert(adjoint)
                batch = self.batch_size
                if isinstance(module, nn.Linear):
                    a = self._convert(x).reshape(batch, -1, module.in_features)
                    d = delta.reshape(batch, -1, module.out_features)
                    self.linear_rows_before += batch * d.shape[1]
                    if self.compact_zero_rows:
                        active = d.ne(0).any(dim=-1)
                        kept = int(active.sum(dim=1).max().item())
                        if kept < d.shape[1]:
                            positions = active.to(torch.int8).argsort(dim=1, descending=True, stable=True)[:, :kept]
                            a = a.gather(1, positions[:, :, None].expand(-1, -1, a.shape[-1]))
                            d = d.gather(1, positions[:, :, None].expand(-1, -1, d.shape[-1]))
                    self.linear_rows_after += batch * d.shape[1]
                    self._append(module.weight, "linear_weight", d, a)
                    self._append(module.bias, "direct", d.sum(dim=1))
                elif isinstance(module, nn.Embedding):
                    indices = x.reshape(batch, -1).to(dtype=torch.long)
                    d = delta.reshape(batch, -1, module.embedding_dim)
                    self._append(module.weight, "embedding", d, indices, module.padding_idx)
                elif isinstance(module, nn.LayerNorm):
                    axes = tuple(range(x.ndim - len(module.normalized_shape), x.ndim))
                    xx = self._convert(x)
                    mean = xx.mean(dim=axes, keepdim=True)
                    variance = ((xx - mean) ** 2).mean(dim=axes, keepdim=True)
                    gamma = delta * (xx - mean) * torch.rsqrt(variance + module.eps)
                    beta = delta
                    reduce = tuple(range(1, x.ndim - len(module.normalized_shape)))
                    if reduce:
                        gamma, beta = gamma.sum(dim=reduce), beta.sum(dim=reduce)
                    self._append(module.weight, "direct", gamma)
                    self._append(module.bias, "direct", beta)
                # Both weight and bias terms have been added before either is
                # retired, including pathological sharing between those fields.
                for pid in pids:
                    self._remaining[pid] -= 1
                    if self._remaining[pid] == 0:
                        self._retire(pid)
                self._pending -= 1
            return None

        self._output_hooks[token] = output.register_hook(adjoint_hook)
        return output

    def _check_current(self):
        if self._failed:
            raise RuntimeError("capture failed; discard the online collector")
        if not (self._running or self._captured):
            raise RuntimeError("capture a forward/backward first")
        if any(p._version != self._parameter_versions[pid] for pid, p in self.parameters.items()):
            raise RuntimeError("parameters changed during or after capture")

    @torch.no_grad()
    def _retire(self, pid):
        contribution, plan = self._engine.parameter_gram(pid, method=self.method_overrides.get(pid, "auto"))
        plan["retired_during_backward"] = self._running
        self._gram.add_(contribution)
        if self.on_parameter is not None:
            self.on_parameter(pid, contribution, dict(plan))
        self.last_plan.append(plan)
        self._pending_terms -= len(self.terms[pid])
        self.terms[pid].clear()
        self.pending_factor_bytes -= self._bytes_by_pid[pid]
        self._bytes_by_pid[pid] = 0
        self._retired.add(pid)

    def capture(self, per_example_losses):
        if not self._entered or self._sealed:
            raise RuntimeError("capture must occur once inside its context")
        if per_example_losses.shape != (self.batch_size,):
            raise ValueError("losses must have shape (batch_size,)")
        self._sealed = True
        self._parameter_versions = {pid: p._version for pid, p in self.parameters.items()}
        self.losses = per_example_losses.detach().to(self.dtype)
        self._running = True
        try:
            per_example_losses.sum().backward()
            self._running = False
            self._captured = True
            self.unused_invocations = self._pending
            # Only now are absent adjoints known to be zero. Parameters shared
            # with an unused invocation cannot safely retire earlier.
            for pid in self.parameters:
                if pid not in self._retired:
                    self._retire(pid)
        except BaseException:
            self._failed = True
            raise
        finally:
            self._running = False
            self._release_capture_state()
        return self

    def _release_capture_state(self):
        for handle in self._output_hooks.values():
            handle.remove()
        self._output_hooks.clear()
        self._payloads.clear()
        for anchor in self._anchor_tensors:
            anchor.grad = None
        self._anchor_tensors.clear()
        if self._failed:
            for terms in self.terms.values():
                terms.clear()
            self.pending_factor_bytes = 0
            self._pending_terms = 0

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is not None:
            self._failed = True
        self._release_capture_state()
        return super().__exit__(exc_type, exc_value, traceback)

    def gram(self):
        self._check_current()
        if not self._captured:
            raise RuntimeError("Gram is only complete after capture")
        return self._gram.clone()

    def matvec(self, weights, *, return_direction=False):
        if return_direction:
            raise RuntimeError("factors were retired; recompute the weighted gradient")
        weights = torch.as_tensor(weights, device=self.device, dtype=self.dtype)
        if weights.shape != (self.num_tasks,):
            raise ValueError("weights must have shape (num_tasks,)")
        return self.gram() @ weights

    def direction(self, weights):
        raise RuntimeError("factors were retired; recompute the weighted gradient")

    def grouped_gram(self, **kwargs):
        # ExactFactorGram's mixed/direct fallback calls this before retirement.
        if not self._running and self._captured and len(self._retired) == len(self.parameters):
            raise RuntimeError("factors were retired; use gram()")
        return super().grouped_gram(**kwargs)

    def memory_report(self):
        return dict(pending_factor_payload_bytes=self.pending_factor_bytes,
                    peak_pending_factor_payload_bytes=self.peak_pending_factor_bytes,
                    peak_pending_terms=self.peak_pending_terms,
                    retired_parameters=len(self._retired),
                    capture_backward_passes=int(self._captured),
                    gram_bytes=self._gram.numel() * self._gram.element_size(),
                    payload_is_allocator_peak=False)
