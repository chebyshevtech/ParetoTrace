"""CPU toy integration checks; no datasets, pretrained weights or CUDA calls."""
import copy
import io
import importlib.util
import json
import sys
import time
import types
import numpy as np
import torch
from torch import nn

# This CPU environment need not install torchvision merely to import the frozen
# ResNet factory. The factory is never called: all numerical tests use TinyModel.
VISION_STUB = importlib.util.find_spec('torchvision') is None
if VISION_STUB:
    package = types.ModuleType('torchvision')
    models = types.ModuleType('torchvision.models')
    def unused_factory(*args, **kwargs):
        raise RuntimeError('CPU fixture must not instantiate the real ResNet factory')
    models.resnet18 = unused_factory
    models.ResNet18_Weights = types.SimpleNamespace(IMAGENET1K_V1=None)
    package.models = models
    sys.modules['torchvision'] = package
    sys.modules['torchvision.models'] = models
import sharp_training as study


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(3, 5), nn.Tanh(), nn.Linear(5, 4))
        self.head = nn.Linear(4, 40)
        with torch.no_grad():
            self.head.weight.copy_(torch.linspace(.2, .8, 40)[:, None] * torch.tensor([[.1, .15, .12, .08]]))
            self.head.bias.zero_()

    def forward(self, x):
        return self.head(self.trunk(x))


def check_update(model, optimizer, x, y, arm):
    reference = copy.deepcopy(model)
    before = {name: p.detach().clone() for name, p in model.named_parameters()}
    old_buffers = {name: optimizer.state[p].get('momentum_buffer', torch.zeros_like(p)).clone()
                   for name, p in model.named_parameters()}
    row = {}
    study.update(model, optimizer, lambda: (x, y), arm, False, row)
    losses = study.task_losses(reference, x, y)
    shared, heads = study.scope(reference)
    weights = torch.as_tensor(row['weights'], dtype=losses.dtype)
    expected_shared = torch.autograd.grad(losses @ weights, tuple(shared.values()), retain_graph=True)
    expected_heads = torch.autograd.grad(losses.sum(), tuple(heads.values()), retain_graph=True)
    expected = dict(zip(list(shared) + list(heads), expected_shared + expected_heads))
    table = torch.stack([torch.cat([g.flatten() for g in torch.autograd.grad(
        loss, tuple(shared.values()), retain_graph=True)]) for loss in losses]).double()
    K = (table @ table.T).numpy()
    _, fit = study.solve(K)
    w = np.asarray(row['weights'])
    factor = float(w @ K @ w / fit['lower'])
    assert factor <= (1.051 if arm.startswith('stream_') else 1.001), (arm, factor)
    worst_error = head_error = 0.
    for name, p in model.named_parameters():
        error = float((p.grad - expected[name]).norm() / expected[name].norm().clamp_min(1e-20))
        worst_error = max(worst_error, error)
        if name.startswith('head.'):
            head_error = max(head_error, error)
        assert error < 2e-5, (arm, name, error)
        wanted = before[name] - .02 * (.9 * old_buffers[name] + expected[name])
        assert torch.allclose(p, wanted, atol=2e-7, rtol=2e-6), (arm, name, 'SGD update')
    assert row['complete'] and row['applied']
    if arm.startswith('stream_'):
        assert not row['fallback'], (arm, row.get('adaptive_error'))
        assert row['adaptive_actions'] == len(row['queries']) == len(row['prefixes'])
        assert [q['index'] for q in row['queries']] == list(range(1, len(row['queries']) + 1))
        assert all('response' in q for q in row['queries'])
        assert all('upper' in prefix for prefix in row['prefixes'])
    return row, {'factor': factor, 'gradient_error': worst_error, 'head_error': head_error}


def main():
    torch.set_num_threads(1)
    torch.manual_seed(1709)
    initial = TinyModel().eval().state_dict()
    x = torch.tensor([[.2, -.4, .1], [.5, .1, -.3], [-.2, .3, .6], [.1, .2, .3]])
    y = torch.zeros(4, 40)
    records = []
    stream_traces = {}
    for arm in study.ARMS:
        model = TinyModel().eval()
        model.load_state_dict(initial)
        optimizer = torch.optim.SGD(model.parameters(), lr=.02, momentum=.9)
        first, first_check = check_update(model, optimizer, x, y, arm)
        checkpoint = io.BytesIO()
        torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                    'rng': torch.get_rng_state()}, checkpoint)
        second, second_check = check_update(model, optimizer, x * .9, y, arm)
        checkpoint.seek(0)
        saved = torch.load(checkpoint, weights_only=False)
        resumed = TinyModel().eval()
        resumed.load_state_dict(saved['model'])
        resumed_optimizer = torch.optim.SGD(resumed.parameters(), lr=.02, momentum=.9)
        resumed_optimizer.load_state_dict(saved['optimizer'])
        torch.set_rng_state(saved['rng'])
        replay, _ = check_update(resumed, resumed_optimizer, x * .9, y, arm)
        assert all(torch.equal(a, b) for a, b in zip(model.parameters(), resumed.parameters())), arm
        assert all(torch.equal(optimizer.state[a]['momentum_buffer'], resumed_optimizer.state[b]['momentum_buffer'])
                   for a, b in zip(model.parameters(), resumed.parameters())), arm
        assert second['weights'] == replay['weights'] and second['adaptive_actions'] == replay['adaptive_actions']
        records.append({'arm': arm, 'first': first_check, 'second': second_check,
                        'actions': [first['adaptive_actions'], second['adaptive_actions']],
                        'exact_cpu_resume_parameters_and_momentum': True})
        if arm.startswith('stream_'):
            stream_traces[arm] = first
    completion, scalar = (stream_traces[a] for a in ('stream_completion', 'stream_scalar'))
    for left, right in zip(completion['queries'], scalar['queries']):
        assert left['query'] == right['query'] and left['response'] == right['response']
    for left, right in zip(completion['prefixes'], scalar['prefixes']):
        assert left['upper'] == right['upper']
        if 'completion_qp' in left and 'completion_qp' in right:
            assert left['completion_qp']['weights'] == right['completion_qp']['weights']
    # Backend failure must retain the attempted query and execute a full fallback.
    original = study.SharedNativeActions.action
    def broken(self, query):
        raise ArithmeticError('injected CPU action failure')
    study.SharedNativeActions.action = broken
    try:
        model = TinyModel().eval()
        model.load_state_dict(initial)
        optimizer = torch.optim.SGD(model.parameters(), lr=.02, momentum=.9)
        row = {}
        study.update(model, optimizer, lambda: (x, y), 'stream_completion', False, row)
        assert row['fallback'] and row['adaptive_actions'] == 1 and 'error' in row['queries'][0]
        assert row['complete'] and row['applied'] and 'fallback_full_recovery' in row
    finally:
        study.SharedNativeActions.action = original
    # Accepted recovery can still fail during reconstruction. The failed work
    # remains inside the measured update and must lead to a valid Direct step.
    direct = TinyModel().eval()
    direct.load_state_dict(initial)
    direct_optimizer = torch.optim.SGD(direct.parameters(), lr=.02, momentum=.9)
    study.update(direct, direct_optimizer, lambda: (x, y), 'direct', False, {})
    original_direction = study.SharedNativeActions.direction
    injected_delay = .01
    def nonfinite_direction(self, weights):
        result = original_direction(self, weights)
        time.sleep(injected_delay)
        next(iter(result.values())).fill_(float('nan'))
        return result
    study.SharedNativeActions.direction = nonfinite_direction
    try:
        for arm in ('stream_completion', 'stream_scalar'):
            model = TinyModel().eval()
            model.load_state_dict(initial)
            optimizer = torch.optim.SGD(model.parameters(), lr=.02, momentum=.9)
            row = {}
            study.update(model, optimizer, lambda: (x, y), arm, False, row)
            assert row['recovery']['accepted'] and row['fallback'] and row['adaptive_actions'] > 0
            assert 'Nonfinite Stream reconstructed losses/gradients' in row['adaptive_error']
            assert row['recovery_and_head_seconds'] >= injected_delay
            assert row['complete'] and row['applied'] and 'fallback_full_recovery' in row
            assert all(torch.equal(a, b) for a, b in zip(model.parameters(), direct.parameters())), arm
            assert all(torch.equal(optimizer.state[a]['momentum_buffer'], direct_optimizer.state[b]['momentum_buffer'])
                       for a, b in zip(model.parameters(), direct.parameters())), arm
    finally:
        study.SharedNativeActions.direction = original_direction
    # Neither phase can create an output directory or touch data in this draft.
    arguments = sys.argv
    try:
        for phase, seed in [('pilot', 61900), ('reporting', 62001)]:
            sys.argv = ['sharp_training.py', '--phase', phase, '--arm', 'direct', '--seed', str(seed),
                        '--data', '/must-not-open-data', '--selection', '/must-not-open-selection',
                        '--out', '/must-not-create-output']
            try:
                study.main()
            except ValueError as error:
                assert 'Training disabled' in str(error)
            else:
                raise AssertionError('Draft protocol permitted training')
    finally:
        sys.argv = arguments
    print(json.dumps({'complete': True, 'scope': 'CPU toy gradients/heads/SGD/momentum restart and failed-action fallback; no CUDA timing or large-model certification',
                      'unused_torchvision_factory_stub': VISION_STUB,
                      'records': records, 'failed_action_preserved_and_fallback_executed': True,
                      'nonfinite_reconstruction_timed_fallback_matches_direct_sgd': True,
                      'shared_query_and_qp_policy_verified': True, 'both_training_phases_disabled': True}, indent=2))


if __name__ == '__main__':
    main()
