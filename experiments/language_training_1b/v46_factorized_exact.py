"""ParetoTrace-Direct: structured exact Gram recovery from cached factors.

Pure embeddings use only observed vocabulary rows and scatter each token once
into a bounded row tile. Pure affine weights may use a genuinely ragged token
Hadamard contraction. Mixed tied embedding/head parameters retain the general
row-tiled implementation; their cross terms are never silently omitted.
Preparation/copy work occurs within each call and must be timed. The recovered
parameter contributions are known PSD geometry in ParetoTrace's residual
version space; complete coverage leaves no unknown residual. Contraction
identities themselves are prior machinery. ExactFactorGram is the historical
class name retained for compatibility with immutable experiment bundles.
"""
from __future__ import annotations

import math
import torch


class ExactFactorGram:
    def __init__(self, cache, *, row_chunk=256, scratch_bytes=128 * 2**20):
        self.cache = cache
        self.fragments = getattr(cache, "caches", [cache])
        self.parameters, self.names = cache.parameters, cache.names
        self.counts, self.num_tasks = cache.counts, cache.num_tasks
        self.device, self.dtype = cache.device, cache.dtype
        self.row_chunk, self.scratch_bytes = int(row_chunk), int(scratch_bytes)
        self.last_plan = []

    def _terms(self, pid):
        return [(fragment, term) for fragment in self.fragments
                for term in fragment.terms[pid]]

    def _fallback(self, pid, rows=None):
        subsets = None if rows is None else {pid: rows}
        return self.cache.grouped_gram(row_chunk=self.row_chunk,
                                      memory_budget_bytes=self.scratch_bytes,
                                      exclude_parameter_ids=set(self.parameters) - {pid},
                                      row_subsets=subsets)

    def _embedding(self, pid, rows=None):
        indices, values, tasks = [], [], []
        for fragment, term in self._terms(pid):
            ids = term.right.reshape(-1)
            d = term.left.reshape(-1, term.left.shape[-1])
            t = fragment.task_ids[:, None].expand(term.right.shape).reshape(-1)
            active = d.ne(0).any(dim=1)
            if term.padding_idx is not None:
                active &= ids != term.padding_idx
            if rows is not None:
                selected = torch.as_tensor(rows, device=self.device, dtype=torch.long)
                active &= torch.isin(ids, selected)
            take = active.nonzero(as_tuple=True)[0]
            indices.append(ids.index_select(0, take))
            tasks.append(t.index_select(0, take))
            values.append(d.index_select(0, take))
        ids, tasks, values = torch.cat(indices), torch.cat(tasks), torch.cat(values)
        gram = torch.zeros((self.num_tasks, self.num_tasks), device=self.device, dtype=self.dtype)
        if ids.numel() == 0:
            return gram, {"active_tokens": 0, "observed_rows": 0}
        vocabulary, inverse = torch.unique(ids, sorted=True, return_inverse=True)
        width = values.shape[1]
        per_row_bytes = self.num_tasks * width * values.element_size()
        if self.scratch_bytes < per_row_bytes:
            raise ValueError("embedding tile budget cannot hold one observed row per task")
        row_chunk = min(self.row_chunk, self.scratch_bytes // per_row_bytes)
        order = inverse.argsort()
        inverse, tasks = inverse[order], tasks[order]
        values = values[order] / self.counts[tasks, None]
        # One small CPU transfer sets every row-tile boundary. Subsequent tiles
        # process disjoint token slices, rather than rescanning all examples.
        ends = torch.bincount(inverse, minlength=vocabulary.numel()).cumsum(0)
        tile_ends = ends[row_chunk - 1::row_chunk]
        if vocabulary.numel() % row_chunk:
            tile_ends = torch.cat((tile_ends, ends[-1:]))
        boundaries = [0, *tile_ends.cpu().tolist()]
        for tile, first in enumerate(range(0, vocabulary.numel(), row_chunk)):
            rows_in_tile = min(row_chunk, vocabulary.numel() - first)
            start, stop = boundaries[tile:tile + 2]
            h = torch.zeros((self.num_tasks * rows_in_tile, width),
                            device=self.device, dtype=self.dtype)
            local = tasks[start:stop] * rows_in_tile + inverse[start:stop] - first
            h.index_add_(0, local, values[start:stop])
            h = h.reshape(self.num_tasks, -1)
            gram.addmm_(h, h.T)
            del h
        return gram, {"active_tokens": ids.numel(), "observed_rows": vocabulary.numel()}

    def _linear_tokens(self, pid, rows=None):
        aa, dd, tt = [], [], []
        selected = None if rows is None else torch.unique(
            torch.as_tensor(rows, device=self.device, dtype=torch.long), sorted=True)
        for fragment, term in self._terms(pid):
            d = term.left if selected is None else term.left.index_select(-1, selected)
            a = term.right
            t = fragment.task_ids[:, None].expand(d.shape[:2]).reshape(-1)
            d, a = d.reshape(-1, d.shape[-1]), a.reshape(-1, a.shape[-1])
            active = d.ne(0).any(dim=1).nonzero(as_tuple=True)[0]
            aa.append(a.index_select(0, active))
            dd.append(d.index_select(0, active))
            tt.append(t.index_select(0, active))
        a, d, task_ids = torch.cat(aa), torch.cat(dd), torch.cat(tt)
        d = d / self.counts[task_ids, None]
        n = len(task_ids)
        gram = torch.zeros((self.num_tasks, self.num_tasks), device=self.device, dtype=self.dtype)
        element_bytes = torch.empty((), dtype=self.dtype).element_size()
        token_tile = max(1, math.isqrt(max(1, self.scratch_bytes // (3 * element_bytes))))
        # Task aggregation matrices are small (N*M). Their GEMMs avoid an
        # M²-loop over irregular token counts and support arbitrary fragments.
        aggregation = torch.zeros((n, self.num_tasks), device=self.device, dtype=self.dtype)
        aggregation.scatter_(1, task_ids[:, None], 1)
        for first in range(0, n, token_tile):
            end_i = min(first + token_tile, n)
            for second in range(first, n, token_tile):
                end_j = min(second + token_tile, n)
                block = d[first:end_i] @ d[second:end_j].T
                block.mul_(a[first:end_i] @ a[second:end_j].T)
                reduced = aggregation[first:end_i].T @ block @ aggregation[second:end_j]
                gram.add_(reduced)
                if first != second:
                    gram.add_(reduced.T)
                del block, reduced
        return gram, {"active_tokens": n, "token_tile": token_tile,
                      "output_rows": d.shape[1]}

    def _linear_merged(self, pid, rows=None):
        """Merge microbatches once, then use one batched GEMM per row tile.

        The scratch budget bounds task-gradient tiles. Packed factor copies
        are additional preparation storage and must be included in peak memory.
        This avoids one small GEMM per present task per microbatch per tile.
        """
        aa, dd, tt = [], [], []
        selected = None if rows is None else torch.unique(
            torch.as_tensor(rows, device=self.device, dtype=torch.long), sorted=True)
        for fragment, term in self._terms(pid):
            d = term.left if selected is None else term.left.index_select(-1, selected)
            aa.append(term.right.reshape(-1, term.right.shape[-1]))
            dd.append(d.reshape(-1, d.shape[-1]))
            tt.append(fragment.task_ids[:, None].expand(d.shape[:2]).reshape(-1))
        tasks = torch.cat(tt)
        lengths = torch.bincount(tasks, minlength=self.num_tasks)
        lengths_cpu = lengths.cpu().tolist()
        largest = max(lengths_cpu)
        order = tasks.argsort(stable=True)
        a = torch.cat(aa).index_select(0, order)
        d = torch.cat(dd).index_select(0, order)
        if min(lengths_cpu) == largest:
            a = a.reshape(self.num_tasks, largest, -1)
            d = d.reshape(self.num_tasks, largest, -1)
        else:
            # Concatenate arbitrary fragment/task layouts without dropping any
            # tied occurrence or cross-microbatch inner product.
            offsets = lengths.cumsum(0) - lengths
            sorted_tasks = tasks.index_select(0, order)
            positions = torch.arange(len(tasks), device=self.device) - offsets[sorted_tasks]
            indices = sorted_tasks * largest + positions
            ag = a.new_zeros((self.num_tasks * largest, a.shape[-1]))
            dg = d.new_zeros((self.num_tasks * largest, d.shape[-1]))
            ag.index_copy_(0, indices, a); dg.index_copy_(0, indices, d)
            a, d = ag.reshape(self.num_tasks, largest, -1), dg.reshape(self.num_tasks, largest, -1)
        gram = a.new_zeros((self.num_tasks, self.num_tasks))
        bytes_per_row = self.num_tasks * a.shape[-1] * a.element_size()
        tile = min(self.row_chunk, self.scratch_bytes // bytes_per_row)
        if tile < 1:
            raise ValueError('linear tile budget cannot hold one row per task')
        for first in range(0, d.shape[-1], tile):
            h = torch.bmm(d[:, :, first:first + tile].transpose(1, 2), a)
            h = h.reshape(self.num_tasks, -1) / self.counts[:, None]
            gram.addmm_(h, h.T)
        return gram, dict(tokens=len(tasks), padded_tokens=self.num_tasks * largest,
                          row_tile=tile, packed_factor_bytes=(a.numel() + d.numel()) * a.element_size())

    @torch.no_grad()
    def parameter_gram(self, pid, *, rows=None, method="auto"):
        if pid not in self.parameters:
            raise ValueError("unknown parameter identity")
        if rows is not None and len(rows) == 0:
            return torch.zeros((self.num_tasks, self.num_tasks), device=self.device, dtype=self.dtype), {
                "parameter": self.names[pid], "method": "empty_row_selection"}
        self.fragments[0]._check_current()
        terms = self._terms(pid)
        kinds = {term.kind for _, term in terms}
        plan = {"parameter": self.names[pid], "method": method}
        if not terms:
            return torch.zeros((self.num_tasks, self.num_tasks), device=self.device, dtype=self.dtype), plan
        if kinds == {"embedding"} and method in ("auto", "embedding"):
            gram, details = self._embedding(pid, rows)
            plan.update(method="observed_embedding_tiles", **details)
            return gram, plan
        if kinds == {"linear_weight"}:
            n = sum(term.left.shape[0] * term.left.shape[1] for _, term in terms)
            output, width = self.parameters[pid].shape
            output = output if rows is None else len(rows)
            parameter_cost = 2 * (n + self.num_tasks ** 2) * width * output
            token_cost = 2 * n * n * (width + output)
            use_token = method == "token" or (method == "auto" and token_cost < parameter_cost)
            if use_token:
                gram, details = self._linear_tokens(pid, rows)
                plan.update(method="ragged_token_hadamard", **details,
                            parameter_flops_padded=parameter_cost, token_flops_padded=token_cost)
                return gram, plan
            if method in ('auto', 'merged'):
                gram, details = self._linear_merged(pid, rows)
                plan.update(method='merged_task_parameter_tiles', **details)
                return gram, plan
        if method not in ("auto", "parameter"):
            raise ValueError("requested contraction does not support this parameter's occurrences")
        gram = self._fallback(pid, rows)
        plan["method"] = "grouped_parameter_tiles"
        return gram, plan

    @torch.no_grad()
    def gram(self, *, exclude_parameter_ids=None, row_subsets=None,
             method_overrides=None):
        excluded = set() if exclude_parameter_ids is None else set(exclude_parameter_ids)
        subsets = {} if row_subsets is None else row_subsets
        overrides = {} if method_overrides is None else method_overrides
        if not (excluded | set(subsets) | set(overrides)).issubset(self.parameters):
            raise ValueError("unknown parameter identity")
        gram = torch.zeros((self.num_tasks, self.num_tasks), device=self.device, dtype=self.dtype)
        self.last_plan = []
        for pid in self.parameters:
            if pid in excluded:
                continue
            contribution, plan = self.parameter_gram(pid, rows=subsets.get(pid),
                                                      method=overrides.get(pid, "auto"))
            gram.add_(contribution)
            self.last_plan.append(plan)
            del contribution
        return gram


ParetoTraceDirect = ExactFactorGram
