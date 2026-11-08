"""Extract, for every update of the Stream training runs, the uniform direction's trunk responses K(1/M)
(from the first Gram action, which every Stream recovery spends on the uniform weighting) and the exact
responses of the executed weighting. Writes one compact .npz so the counts can be re-derived without the
raw events. CPU only; reads metrics.json and events/*.json.

Usage: python3 -I extract_uniform_responses.py <out.npz> <run_dir> [<run_dir> ...]
"""
import json
import sys
from pathlib import Path
import numpy as np


def main():
    out, runs = sys.argv[1], sys.argv[2:]
    names, uniform, executed, executed_energy, fallback, step, seed = [], [], [], [], [], [], []
    for run in runs:
        run = Path(run)
        for ref in json.loads((run / 'metrics.json').read_text())['events']:
            r = json.loads((run / ref['file']).read_text())
            q0 = r['queries'][0]
            query, response = np.asarray(q0['query'], float), np.asarray(q0['response'], float)
            assert np.ptp(query) <= 1e-12 * query.max(), 'first query is not uniform'
            uniform.append(response * ((1 / len(query)) / query[0]))
            if r['fallback']:
                executed.append(np.full(len(query), np.nan)); executed_energy.append(np.nan); fallback.append(True)
            else:
                executed.append(np.asarray(r['recovery']['response'], float))
                executed_energy.append(float(r['recovery']['objective'])); fallback.append(False)
            names.append('/'.join(run.parts[-3:])); step.append(r['step']); seed.append(r['seed'])
    np.savez_compressed(out, run=np.asarray(names), seed=np.asarray(seed), step=np.asarray(step),
                        uniform_trunk_response=np.asarray(uniform), executed_trunk_response=np.asarray(executed),
                        executed_energy=np.asarray(executed_energy), fallback=np.asarray(fallback))
    print('WROTE', out, len(names))


if __name__ == '__main__':
    main()
