"""Certified-step Pythia-1B study (rebuttal, 10 November 2026). One arm, one seed per invocation.

Question. At Pythia-1B, where every parameter is shared by the eight tasks and native per-task differentiation ran
out of memory, which weighting can train under the per-update guarantee (Proposition 1, Equation 27), how often does an
unprotected step raise a task's minibatch loss, and what does the guarantee cost in held-out loss?

Same pool, pinned model, attention precision, seeded batch construction and seeds as the 1B training campaign, so each
arm is paired by seed and batch sequence with the campaign's unprotected runs. Plain SGD, no momentum, decay or clipping.

Arms
  uniform_fixed         uniform weighting, plain SGD at eta0 through the campaign's own backward pass; the realized change
                        of every task loss on the minibatch is recorded.
  uniform_certified     uniform; one Gram action gives its exact responses s = K u.
  direct_certified      exact MGDA by ParetoTrace-Direct; s = K w* from the exact Gram.
  auditfirst_certified  uniform when its one-action audit shows every response positive, otherwise exact MGDA
                        (Corollary 4 audit, then Direct on refusal, as Algorithm 1 does for a held weighting).
  famo_certified        FAMO's weighting and logit update; one Gram action gives s = K w.
Certified rule (as in the CelebA study): if every s_i > 0, take the largest eta = eta0 * 2^-k, k < 20, with
L_i(after) <= L_i(before) - c * eta * s_i for every task, otherwise hold the parameters. Equation (27) uses c = 1/2. The
paper's all-task Armijo relaxation allows any 0 < c < 1 (--c), and every accepted step still strictly lowers every task's
minibatch loss. Amendment, 10 November 2026, after the c = 1/2 development runs and before any confirmatory run: the
development grid adds c = 1e-4, and each rule's (c, eta0) is selected on development validation.
Rounded same-minibatch checks in floating point, not an interval enclosure or a population guarantee.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import argparse
import hashlib
import json
import random
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

ARMS = ('uniform_fixed', 'uniform_certified', 'direct_certified', 'auditfirst_certified', 'famo_certified')
C, HALVINGS, M = .5, 20, 8
SYSTEM = {'row_chunk': 256, 'scratch_mib': 32}
FAMO = {'weight_lr': .025, 'gamma': .01}
MODEL = 'EleutherAI/pythia-1b-deduped'
STUDY_FILES = ['certified_language_1b.py', 'certified_1b_driver.py']
CAMPAIGN_FILES = ['language_common.py', 'language_training_methods.py', 'language_training_rebuttal.py', 'attention_precision.py',
                  'simplex_qp.py', 'v46_factorized_jacobian.py', 'v46_factorized_exact.py', 'v46_supervised_head.py',
                  'language_tokenizer_contract.py', 'checkpoint_archive.py', 'language_checkpoint_storage.py',
                  'native_language_dual_reference.py', 'paretotrace_online_factor.py', 'drive_checkpoint_slots.py']


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda: f.read(8 * 2**20), b''):
            h.update(block)
    return h.hexdigest()


def save(p, obj):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=1, allow_nan=False))
    tmp.replace(p)


def plain(x):
    return np.asarray(x, float).tolist()


def fin(x):
    x = float(x)
    return x if np.isfinite(x) else None


def capture(model, b):
    from v46_factorized_jacobian import FactorizedJacobian
    from language_common import losses
    with FactorizedJacobian(model, b['owner'], M, compute_dtype=torch.float32, compact_zero_rows=True) as obj:
        obj.capture(losses(model, b))
    return obj


def exact_gram(cache):
    from v46_factorized_exact import ExactFactorGram
    K = ExactFactorGram(cache, row_chunk=SYSTEM['row_chunk'], scratch_bytes=SYSTEM['scratch_mib'] * 2**20).gram()
    K = K.double().cpu().numpy()
    return (K + K.T) / 2


def native_audit(model, params, b, w, s):
    """Independent autodiff check of the responses that decide the gate: s_i = g_i . (G w), one task gradient at a time."""
    from language_common import task_losses
    l = task_losses(model, b)
    wt = torch.as_tensor(np.asarray(w, float), device=l.device, dtype=l.dtype)
    d = torch.autograd.grad(l @ wt, tuple(params.values()), retain_graph=True)
    native = []
    for i in range(M):
        g = torch.autograd.grad(l[i], tuple(params.values()), retain_graph=i < M - 1)
        native.append(float(sum((gi.double() * di.double()).sum() for gi, di in zip(g, d))))
        del g
    del d, l
    native = np.asarray(native)
    s = np.asarray(s, float)
    error = float(np.linalg.norm(native - s) / max(np.linalg.norm(native), 1e-30))
    agree = bool((native.min() > 0) == (s.min() > 0))
    return {'native_response': native.tolist(), 'relative_error': error, 'gate_agrees': agree,
            'passed': bool(error <= 1e-3 and agree)}


class FamoState:
    def __init__(self, device):
        self.logits = torch.zeros(M, device=device, requires_grad=True)
        self.opt = torch.optim.Adam([self.logits], lr=FAMO['weight_lr'], weight_decay=FAMO['gamma'])

    def weights(self, before):
        with torch.no_grad():
            z = self.logits.softmax(-1)
            D = before + 1e-8
            c = (z / D).sum()
            return ((z / D) / c).double().cpu().numpy()

    def update(self, before, after):
        delta = (before + 1e-8).log() - (after + 1e-8).log()
        grad = torch.autograd.grad(self.logits.softmax(-1), self.logits, grad_outputs=delta.detach())[0]
        self.opt.zero_grad(set_to_none=True)
        self.logits.grad = grad
        self.opt.step()


def update(model, params, b, arm, eta0, famo, audit, c):
    from language_common import task_losses, solve
    from language_training_methods import reduce_tasks, finite, sync
    sync(model)
    torch.cuda.reset_peak_memory_stats()
    begin = time.perf_counter()
    times, row = {}, {'arm': arm}
    u = np.full(M, 1 / M)

    def timed(name, f):
        sync(model)
        tick = time.perf_counter()
        v = f()
        sync(model)
        times[name] = time.perf_counter() - tick
        return v

    actions, s, K, before = 0, None, None, None
    if arm == 'uniform_fixed':
        l = task_losses(model, b)
        w = u
        grads = timed('backward', lambda: torch.autograd.grad(l @ torch.as_tensor(w, device=l.device, dtype=l.dtype), tuple(params.values())))
        d = dict(zip(params, grads))
        del grads, l
        row['weighting'] = 'uniform'
    else:
        cache = timed('capture', lambda: capture(model, b))
        before = reduce_tasks(cache.losses, b).detach()
        finite([before], 'task losses')
        if arm in ('uniform_certified', 'auditfirst_certified'):
            su = timed('uniform_action', lambda: cache.matvec(u).double().cpu().numpy())
            actions += 1
            row['uniform_response'] = plain(su)
            if arm == 'uniform_certified' or (np.isfinite(su).all() and su.min() > 0):
                w, s, row['weighting'] = u, su, 'uniform'
        if arm == 'direct_certified' or (arm == 'auditfirst_certified' and s is None):
            K = timed('gram', lambda: exact_gram(cache))
            fit = timed('qp', lambda: solve(K))
            w = np.asarray(fit['weights'], float)
            s = K @ w
            row.update(weighting='exact MGDA', qp_gap=float(fit['dual_gap']), recovered_gram=K.tolist())
            if arm == 'direct_certified':
                row['uniform_response'] = plain(K @ u)
        if arm == 'famo_certified':
            w = famo.weights(before)
            s = timed('famo_action', lambda: cache.matvec(w).double().cpu().numpy())
            actions += 1
            row['weighting'] = 'FAMO'
        d = timed('direction', lambda: cache.direction(w))
        del cache
    finite(list(d.values()), 'direction')
    direction_seconds = time.perf_counter() - begin
    if audit and s is not None:
        tick = time.perf_counter()
        row['native_audit'] = native_audit(model, params, b, w, s)
        row['native_audit']['seconds'] = time.perf_counter() - tick
        model.zero_grad(set_to_none=True)
    tick = time.perf_counter()
    with torch.no_grad():
        saved = {n: p.detach().clone() for n, p in params.items()}
        L0 = task_losses(model, b).double().cpu().numpy()

    def put(eta):
        with torch.no_grad():
            for n, p in params.items():
                p.copy_(saved[n] - eta * d[n].to(p.dtype))
            return task_losses(model, b).double().cpu().numpy()

    trials, eta_accepted, after, eligible = [], None, L0, None
    if arm.endswith('_fixed'):
        after, eta_accepted, status = put(eta0), eta0, 'fixed'
        trials.append({'eta': eta0, 'max_task_change': fin((after - L0).max())})
    else:
        eligible = bool(np.isfinite(s).all() and s.min() > 0)
        status = 'no_admissible_step' if eligible else 'not_certified'
        if eligible:
            for k in range(HALVINGS):
                eta = eta0 * .5 ** k
                trial = put(eta)
                bound = L0 - c * eta * s
                ok = bool(np.isfinite(trial).all() and (trial <= bound).all())
                trials.append({'eta': eta, 'passed': ok, 'max_task_change': fin((trial - L0).max()),
                               'tasks_failing': int((trial > bound).sum())})
                if ok:
                    eta_accepted, after, status = eta, trial, 'accepted'
                    break
        if eta_accepted is None:
            with torch.no_grad():
                for n, p in params.items():
                    p.copy_(saved[n])
            after = L0
    del saved, d
    finite(list(params.values()), 'updated parameters')
    if arm == 'famo_certified':
        dev = next(model.parameters()).device
        famo.update(torch.as_tensor(L0, device=dev, dtype=torch.float32), torch.as_tensor(after, device=dev, dtype=torch.float32))
    sync(model)
    step_seconds = time.perf_counter() - tick
    change = after - L0
    if before is not None:
        row['loss_recompute_max_abs'] = float(np.abs(L0 - before.double().cpu().numpy()).max())
    row.update(weights=plain(w), first_order_response=None if s is None else plain(s),
               min_first_order_response=None if s is None else float(np.min(s)), adaptive_actions=actions,
               status=status, eligible=eligible, accepted_eta=eta_accepted, trials=trials,
               task_losses_before=plain(L0), task_losses_after=plain(after), realized_change=plain(change),
               raised_tasks=int((change > 0).sum()), max_task_change=float(change.max()),
               worst_task_before=float(L0.max()), worst_task_after=float(after.max()),
               phase_seconds=times, direction_seconds=direction_seconds, step_seconds=step_seconds,
               update_seconds=direction_seconds + step_seconds, peak_allocated_bytes=int(torch.cuda.max_memory_allocated()),
               timing_scope='direction (capture, actions or Gram and QP, reconstruction) plus the step rule; the native audit is untimed')
    return row


def main():
    begin = time.perf_counter()
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True, help='campaign root with sources and data')
    p.add_argument('--study', type=Path, required=True, help='directory holding the study sources')
    p.add_argument('--arm', choices=ARMS, required=True)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--phase', choices=['smoke', 'development', 'confirmatory'], required=True)
    p.add_argument('--eta0', type=float, required=True)
    p.add_argument('--steps', type=int, required=True)
    p.add_argument('--validation-every', type=int, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--protocol', type=Path)
    p.add_argument('--c', type=float, default=C, help='sufficient-decrease constant, 0 < c < 1; Equation (27) uses 1/2')
    a = p.parse_args()
    if not 0 < a.c < 1:
        raise ValueError('c must lie in (0, 1)')
    sys.path.insert(0, str(a.root))
    out = a.out
    if (out / 'metrics.json').exists():
        print('ALREADY_COMPLETE', out, flush=True)
        return
    out.mkdir(parents=True, exist_ok=True)
    sources = {n: sha(a.study / n) for n in STUDY_FILES}
    sources.update({n: sha(a.root / n) for n in CAMPAIGN_FILES})
    config = {'arm': a.arm, 'seed': a.seed, 'phase': a.phase, 'eta0': a.eta0, 'steps': a.steps,
              'validation_every': a.validation_every, 'c': a.c, 'halvings': HALVINGS, 'batch_size': 32, 'model': MODEL,
              'optimizer': 'plain SGD; no momentum, weight decay or clipping; full-parameter training in eval mode'}
    identity = {'sources': sources}
    if a.phase == 'confirmatory':
        protocol = json.loads(a.protocol.read_text())
        identity['protocol_sha256'] = sha(a.protocol)
        if protocol['sources'] != sources:
            raise ValueError('Sources differ from the frozen protocol')
        allowed = [(arm, protocol['c'][arm], protocol['eta0'][arm]) for arm in protocol['arms']]
        allowed += [(x['arm'], x['c'], x['eta0']) for x in protocol.get('companions', [])]
        if a.seed not in protocol['confirmatory_seeds'] or a.steps != protocol['confirmatory_steps'] \
                or a.validation_every != protocol['validation_every'] or (a.arm, a.c, a.eta0) not in allowed:
            raise ValueError('Configuration differs from the frozen protocol')
    save(out / 'run_identity.json', {'config': config, 'identity': identity})
    try:
        from transformers import AutoModelForCausalLM
        from language_common import load_data, make_batch
        from language_tokenizer_contract import load_tokenizer
        from language_training_rebuttal import evaluate
        from attention_precision import shared_attention_precision
        torch.set_num_threads(1)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        random.seed(a.seed)
        np.random.seed(a.seed)
        torch.manual_seed(a.seed)
        data, manifest = load_data(a.root / 'data')
        model_meta = json.loads((a.root / 'data' / 'pythia-1b-deduped_manifest.json').read_text())
        tok, contract = load_tokenizer(MODEL, model_meta['revision'], manifest)
        model = AutoModelForCausalLM.from_pretrained(MODEL, revision=model_meta['revision'], torch_dtype=torch.float32,
                                                     attn_implementation='eager', token=False).cuda().eval()
        params = {n: q for n, q in model.named_parameters() if q.requires_grad}
        famo = FamoState(next(model.parameters()).device) if a.arm == 'famo_certified' else None
        rows, blocks, history = [], [], []
        audit_steps = {0, a.steps // 2, a.steps - 1}
        with shared_attention_precision() as precision:
            identity['attention'] = {k: v for k, v in precision.items() if k != 'calls'}
            identity.update(data=manifest, model=model_meta, tokenizer_contract=contract)
            history.append({'after_updates': 0, 'validation': evaluate(model, data['validation'], manifest['tasks'], tok.eos_token_id)})
            for step in range(a.steps):
                b = make_batch(data['train'], manifest['tasks'], tok.eos_token_id, a.seed, step=step)
                row = update(model, params, b, a.arm, a.eta0, famo, a.arm != 'uniform_fixed' and step in audit_steps, a.c)
                row['step'] = step
                row['batch_identity_sha256'] = hashlib.sha256(json.dumps(b['identities'], sort_keys=True).encode()).hexdigest()
                rows.append(row)
                del b
                if (step + 1) % 50 == 0 or step + 1 == a.steps:
                    name = 'updates/%08d_%08d.json' % (rows[0]['step'], rows[-1]['step'])
                    save(out / name, rows)
                    blocks.append({'file': name, 'sha256': sha(out / name), 'first_step': rows[0]['step'], 'last_step': rows[-1]['step'], 'rows': len(rows)})
                    rows = []
                    print(json.dumps({'arm': a.arm, 'seed': a.seed, 'step': step + 1, 'elapsed': round(time.perf_counter() - begin)}), flush=True)
                if (step + 1) % a.validation_every == 0 or step + 1 == a.steps:
                    history.append({'after_updates': step + 1, 'validation': evaluate(model, data['validation'], manifest['tasks'], tok.eos_token_id)})
            result = {'complete': True, 'config': config, 'identity': identity, 'steps': a.steps, 'update_blocks': blocks, 'history': history,
                      'final_validation': history[-1]['validation'], 'heldout_test_evaluated': a.phase == 'confirmatory'}
            if a.phase == 'confirmatory':
                result['final_test'] = evaluate(model, data['test'], manifest['tasks'], tok.eos_token_id)
        records = [r for blk in blocks for r in json.loads((out / blk['file']).read_text())]
        statuses = ('fixed', 'accepted', 'no_admissible_step', 'not_certified')
        etas = [r['accepted_eta'] for r in records if r['status'] == 'accepted']
        result.update(status_counts={k: sum(r['status'] == k for r in records) for k in statuses},
                      updates_raising_a_task=sum(r['raised_tasks'] > 0 for r in records),
                      weighting_counts={k: sum(r.get('weighting') == k for r in records) for k in ('uniform', 'exact MGDA', 'FAMO')},
                      accepted_eta_median=float(np.median(etas)) if etas else None,
                      accepted_at_full_step=sum(e == a.eta0 for e in etas),
                      update_seconds_sum=sum(r['update_seconds'] for r in records),
                      update_seconds_median=float(np.median([r['update_seconds'] for r in records])),
                      peak_update_allocated_bytes=max(r['peak_allocated_bytes'] for r in records),
                      adaptive_actions=sum(r['adaptive_actions'] for r in records),
                      audits=sum('native_audit' in r for r in records),
                      audits_passed=sum(r['native_audit']['passed'] for r in records if 'native_audit' in r),
                      audit_max_relative_error=max([r['native_audit']['relative_error'] for r in records if 'native_audit' in r], default=None),
                      wall_seconds=time.perf_counter() - begin, torch=str(torch.__version__), gpu=torch.cuda.get_device_name())
        save(out / 'metrics.json', result)
        print('CERTIFIED_1B_COMPLETE', a.arm, a.seed, a.eta0, a.c, flush=True)
    except BaseException:
        save(out / 'failure.json', {'config': config, 'error': traceback.format_exc(), 'elapsed': time.perf_counter() - begin})
        raise


if __name__ == '__main__':
    main()
