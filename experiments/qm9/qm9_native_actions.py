"""Research-only fixed-batch, multi-output native Gram actions.

No factorized Direct support. Matmat shares one VJP capture, then loops JVPs;
this is a blocked API, not vectorized forward AD. See accompanying report.
"""
from contextlib import contextmanager
import numpy as np
import torch
from torch.func import functional_call, jvp, vjp


class OfficialQM9Model(torch.nn.Module):
    """LibMTL QM9 64-wide NNConv/GRU/Set2Set encoder and 11 linear heads.

    Architecture reference: LibMTL/examples/qm9/main.py. Inputs must already
    have the official complete-graph+distance transform and normalized labels.
    """
    def __init__(self):
        super().__init__()
        from torch_geometric.nn import NNConv
        from torch_geometric.nn.aggr import Set2Set
        self.encoder = QM9Encoder(NNConv, Set2Set)
        self.decoders = torch.nn.ModuleList([torch.nn.Linear(64, 1) for _ in range(11)])

    def forward(self, data):
        features = self.encoder(data)
        return torch.cat([head(features) for head in self.decoders], dim=1)


class QM9Encoder(torch.nn.Module):
    def __init__(self, nnconv, set2set):
        super().__init__()
        self.lin0 = torch.nn.Linear(11, 64)
        edge_net = torch.nn.Sequential(torch.nn.Linear(5, 128), torch.nn.ReLU(),
                                       torch.nn.Linear(128, 64 * 64))
        self.conv = nnconv(64, 64, edge_net, aggr='mean')
        self.gru = torch.nn.GRU(64, 64)
        self.set2set = set2set(64, processing_steps=3)
        self.lin1 = torch.nn.Linear(128, 64)

    def forward(self, data):
        features = self.lin0(data.x).relu()
        hidden = features.unsqueeze(0)
        for _ in range(3):
            messages = self.conv(features, data.edge_index, data.edge_attr).relu()
            features, hidden = self.gru(messages.unsqueeze(0), hidden)
            features = features.squeeze(0)
        return self.lin1(self.set2set(features, data.batch)).relu()


def qm9_mean_losses(model, batch):
    """Eleven simultaneous mean property MSEs, not example/task strata."""
    prediction = model(batch)
    if batch.y.shape != prediction.shape:
        raise ValueError('expected normalized labels of shape [graphs, 11]')
    return (prediction - batch.y).square().mean(dim=0)


class _LossVector(torch.nn.Module):
    def __init__(self, model, batch, loss_fn):
        super().__init__()
        self.model, self.batch, self.loss_fn = model, batch, loss_fn

    def forward(self):
        return self.loss_fn(self.model, self.batch)


class NativeMultioutputActions:
    """K=J J^T for a frozen batch/model snapshot and named parameter scope.

    loss_fn(model, batch) must return a 1D vector of mean task losses. Batch is
    borrowed and must not be mutated. Params/buffers/RNG are snapshotted; model
    training modes must stay fixed. Head exclusion matches LibMTL shared MGDA.
    """
    def __init__(self, model, batch, loss_fn, *, parameter_names=None):
        self.wrapper = _LossVector(model, batch, loss_fn)
        named = dict(model.named_parameters())
        names = list(named) if parameter_names is None else list(parameter_names)
        if not names or len(set(names)) != len(names) or any(n not in named for n in names):
            raise ValueError('parameter_names must be unique existing model names')
        self.names = tuple(names)
        self.params = {'model.' + n: named[n].detach().clone() for n in names}
        self.fixed = {'model.' + n: p.detach().clone() for n, p in named.items() if n not in names}
        self.buffers = {n: b.detach().clone() for n, b in self.wrapper.named_buffers()}
        self.training_modes = tuple(m.training for m in self.wrapper.modules())
        devices = {p.device.index for p in named.values() if p.device.type == 'cuda'}
        self.cuda_devices = sorted(devices)
        self.cpu_rng = torch.get_rng_state().clone()
        self.cuda_rng = {d: torch.cuda.get_rng_state(d).clone() for d in self.cuda_devices}
        self.counters = dict(vjp_captures=0, pullbacks=0, jvps=0, directions=0)
        with self._context():
            values = self._loss(self.params)
        if values.ndim != 1 or not values.numel() or not torch.isfinite(values).all():
            raise ValueError('loss_fn must return a finite nonempty 1D task vector')
        self.m = values.numel()
        self.dtype, self.device = values.dtype, values.device
        self.initial_losses = values.detach().cpu().numpy()

    @contextmanager
    def _context(self):
        if tuple(m.training for m in self.wrapper.modules()) != self.training_modes:
            raise RuntimeError('model training modes changed after snapshot')
        # Native RNN kernels provide forward AD; fused cuDNN/MKLDNN may not.
        with torch.random.fork_rng(devices=self.cuda_devices), \
                torch.backends.cudnn.flags(enabled=False), \
                torch.backends.mkldnn.flags(enabled=False):
            torch.set_rng_state(self.cpu_rng)
            for device, state in self.cuda_rng.items():
                torch.cuda.set_rng_state(state, device)
            yield

    def _loss(self, selected):
        return functional_call(self.wrapper,
            ({**self.fixed, **selected}, {n: b.clone() for n, b in self.buffers.items()}),
            (), strict=True)

    def _queries(self, weights):
        q = torch.as_tensor(weights, dtype=self.dtype, device=self.device)
        if q.ndim != 2 or q.shape[0] != self.m or not torch.isfinite(q).all():
            raise ValueError('finite M by b weights required')
        return q

    def matvec(self, weights):
        q = np.asarray(weights)
        if q.ndim != 1:
            raise ValueError('M-vector weights required')
        return self.matmat(q[:, None])[:, 0]

    def matmat(self, weights):
        q = self._queries(weights)
        if q.shape[1] == 0:
            return np.empty((self.m, 0))
        with self._context():
            _, pullback = vjp(self._loss, self.params)
        self.counters['vjp_captures'] += 1
        answers = []
        for column in q.unbind(1):
            direction = {n: p.detach() for n, p in pullback(column)[0].items()}
            self.counters['pullbacks'] += 1
            with self._context(), torch.no_grad():
                _, response = jvp(self._loss, (self.params,), (direction,))
            self.counters['jvps'] += 1
            answers.append(response.detach())
        return torch.stack(answers, dim=1).cpu().numpy()

    def direction(self, weights):
        q = self._queries(np.asarray(weights)[:, None])[:, 0]
        with self._context():
            _, pullback = vjp(self._loss, self.params)
        self.counters['vjp_captures'] += 1
        values = pullback(q)[0]
        self.counters['pullbacks'] += 1
        self.counters['directions'] += 1
        return {n.removeprefix('model.'): p.detach() for n, p in values.items()}
