"""Bounded, development cost-only 1B overhead probe. Never a campaign result.

Run under language_training_queue.py's exclusive GPU boundary and 20-minute
process-group deadline. Uses actual train/validation/test cardinalities, but
retains no validation/test quality values and makes no quality-based choice.
Only this fresh probe's own temporary model pair is retired by existing helpers.
"""
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import argparse
import gc
import json
from pathlib import Path
import shutil
import sys
import time
import traceback
from language_training_budget import read, sha, select_pilot, verify_staged, require_drive


def main():
    begin = time.perf_counter()
    p = argparse.ArgumentParser(); p.add_argument('--root', type=Path, required=True)
    p.add_argument('--pilot', type=Path, required=True); p.add_argument('--pilot-sources', type=Path, required=True)
    p.add_argument('--manifest', type=Path, required=True); p.add_argument('--work', type=Path, required=True)
    p.add_argument('--backup', type=Path, required=True)
    p.add_argument('--drive-free-ceiling-bytes', type=int, required=True)
    a = p.parse_args()
    require_drive(a.backup)
    if a.work.exists() or a.backup.exists(): raise ValueError('Probe requires fresh isolated local and backup directories')
    if a.drive_free_ceiling_bytes <= 0: raise ValueError('Measured Drive capacity ceiling required')
    pins = verify_staged(a.root, read(a.manifest))
    pilot = read(a.pilot); selected = select_pilot(pilot, read(a.pilot_sources))
    a.work.mkdir(parents=True); a.backup.mkdir(parents=True)
    result = {'complete': False, 'purpose': 'cost_only', 'quality_values_retained': False,
              'pilot_sha256': sha(a.pilot), 'model': pilot['model'], 'data': pilot['data'],
              'systems': {m: r['configuration'] for m, r in selected.items()},
              'sources': pins, 'probe_sha256': sha(__file__),
              'storage': {'checkpoint_backup': 'metadata', 'completed_checkpoints': 'final-only'},
              'seconds': {}, 'method_startup_update_seconds': {}}
    output = a.work / 'cost_probe.json'
    # No torch/CUDA import occurs until this explicit main invocation.
    import torch
    from transformers import AutoModelForCausalLM
    from language_common import load_data, make_batch
    from language_tokenizer_contract import load_tokenizer
    from language_training_rebuttal import Trainer, evaluate, audit_current, save_torch
    from attention_precision import shared_attention_precision
    from checkpoint_archive import atomic_json, verified_copy
    from language_checkpoint_storage import writer, completed_root, validate_final_retention
    from retire_completed_language import retire_completed
    def record():
        atomic_json(output, result)
        require_drive(a.backup)
        verified_copy(output, a.backup / 'cost_probe.json')
    def timed(name, action):
        torch.cuda.synchronize(); tick = time.perf_counter()
        value = action(); torch.cuda.synchronize()
        result['seconds'][name] = max(result['seconds'].get(name, 0.), time.perf_counter() - tick)
        return value
    try:
        torch.set_num_threads(1); torch.manual_seed(49100)
        torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
        result.update(gpu=torch.cuda.get_device_name(), torch=str(torch.__version__), cuda=torch.version.cuda)
        if 'A100' not in result['gpu']: raise ValueError('A100 cost evidence required')
        data, manifest = load_data(a.root / 'data')
        if manifest != pilot['data']: raise ValueError('Pilot/probe data differs')
        for split in ('train', 'validation', 'test'):
            if {t: len(data[split][t]) for t in manifest['tasks']} != manifest['counts'][split]:
                raise ValueError('Actual full evaluation cardinalities differ')
        tok, contract = load_tokenizer(pilot['model']['model'], pilot['model']['revision'], manifest)
        model = AutoModelForCausalLM.from_pretrained(pilot['model']['model'], revision=pilot['model']['revision'],
                torch_dtype=torch.float32, attn_implementation='eager', token=False).cuda().eval()
        torch.cuda.synchronize(); result['seconds']['setup'] = time.perf_counter() - begin
        initial = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        trainer = None; largest_row = {}
        with shared_attention_precision() as precision:
            result['attention_precision'] = precision
            for method, row in selected.items():
                if trainer is not None: del trainer
                model.zero_grad(set_to_none=True); gc.collect(); torch.cuda.empty_cache()
                cfg = dict(row['configuration'], lr=1e-4, phase='cost_probe', seed=49100,
                           completed_checkpoint_policy='final-only', checkpoint_backup='metadata')
                identity = {'purpose': 'cost_only', 'pilot_sha256': sha(a.pilot), 'sources': pins}
                def setup():
                    model.load_state_dict(initial)
                    return Trainer(model, cfg, identity)
                trainer = timed('method_setup', setup)
                torch.cuda.synchronize(); tick = time.perf_counter()
                b = make_batch(data['train'], manifest['tasks'], tok.eos_token_id, 49100, step=0)
                trainer.update(b); torch.cuda.synchronize()
                result['method_startup_update_seconds'][method] = time.perf_counter() - tick
                update = trainer.pending[-1]
                if len(json.dumps(update)) > len(json.dumps(largest_row)): largest_row = update
                if method == 'direct':
                    audit = timed('direct_audit', lambda: audit_current(trainer, b))
                    if audit.get('passes_table_contract') is not True: raise ValueError('Cost probe native-table audit failed')
                del b; record()
            # FAMO is last: its optimizer state also exercises the small auxiliary
            # state absent from Uniform/Direct. All six startups were measured.
            for unused in range(2):
                for split in ('validation', 'test'):
                    timed(split, lambda split=split: evaluate(model, data[split], manifest['tasks'], tok.eos_token_id))
            out = a.work / 'runs' / 'cost_only'; out.mkdir(parents=True)
            backup = a.backup / 'run'
            owner = {'config': trainer.cfg, 'identity': trainer.identity}
            atomic_json(out / 'run_identity.json', owner)
            timed('event_write', lambda: atomic_json(a.work / 'largest_update.json', largest_row))
            trainer.flush(out)
            selected_path = out / 'selected' / 'probe.pt'
            timed('selected_save', lambda: save_torch(selected_path, {'model': model.state_dict(),
                  'after_updates': 1, 'config': trainer.cfg, 'identity': trainer.identity}))
            trainer.best = {'file': 'selected/probe.pt', 'sha256': sha(selected_path), 'after_updates': 1,
                            'worst_ce': 0., 'cost_probe_only': True}
            # Receipt plumbing retains only a redacted cost-only history entry;
            # no model quality values enter source artifacts or budget decisions.
            trainer.history = [{'after_updates': 1, 'metrics': {'cost_only': True, 'quality_values_retained': False}}]
            timed('latest_save', lambda: save_torch(out / 'latest.pt', trainer.state()))
            def reload_selected():
                saved = torch.load(selected_path, map_location='cpu', weights_only=False)
                model.load_state_dict(saved['model'])
            timed('selected_load', reload_selected)
            metrics = trainer.summary(); metrics.update(complete=True, heldout_test_evaluated=False, purpose='cost_only')
            atomic_json(out / 'metrics.json', metrics)
            raw_sizes = {'latest_raw_bytes': (out / 'latest.pt').stat().st_size,
                         'selected_raw_bytes': selected_path.stat().st_size}
            timed('raw_pair_verify', lambda: (sha(out / 'latest.pt'), sha(selected_path)))
            require_drive(backup)
            with writer(backup, owner, vm_temp=a.work / 'checkpoint_tmp', mode='metadata') as store:
                timed('metadata_publish', lambda: store.publish(out))
            # Exact existing archival + verified pair + final-only receipt path.
            require_drive(backup)
            receipt = timed('retirement', lambda: retire_completed(out, backup, policy='final-only'))
            timed('final_archive_verify', lambda: validate_final_retention(out, receipt))
            pair = receipt['verified_pair_retention']['local_checkpoint_archives']
            all_rows = [u for r in pilot['rows'] for u in r.get('updates', [])] + [largest_row]
            result['sizes'] = dict(raw_sizes, latest_gzip_bytes=pair['latest.pt']['archive_bytes'],
                selected_gzip_bytes=pair['selected/probe.pt']['archive_bytes'],
                max_update_json_bytes=max(len(json.dumps(u, indent=2).encode()) for u in all_rows))
            result['retention_receipt_sha256'] = sha(out / 'completed_checkpoint_retention.json')
            result['retained_probe_archive_root'] = str(completed_root(out))
        result.update(complete=True, total_process_seconds=time.perf_counter()-begin,
            vm_free_bytes=shutil.disk_usage(a.root).free,
            drive_free_bytes=min(a.drive_free_ceiling_bytes, shutil.disk_usage(a.backup).free),
            evaluation_examples={s: sum(manifest['counts'][s].values()) for s in ('validation', 'test')},
            scope='Actual full cardinality evaluation timings only; all quality outputs discarded. Existing helpers measure real metadata, model saves, verified compression, retirement and archive reads. Probe model bytes remain VM-only.')
        record()
    except BaseException:
        result.update(error=traceback.format_exc(), total_process_seconds=time.perf_counter()-begin)
        record(); raise

if __name__ == '__main__': main()
