"""Protected fine-tuning study, version 2 (rebuttal, 9 November 2026).

Version 1 (protected_finetune.py) completed four arms of one seed before its cloud runtime was reclaimed, and
its recorded audit flags compared the head first-order terms against fresh head gradients, which the frozen
design never uses. Version 2 reruns the whole study on a new runtime with one change: the head first-order term
is zero everywhere, including inside the independent audit, so the audit checks exactly the responses that decide
eligibility (the trunk responses K w).

A CelebA-40 model trained with uniform weighting (the development run at the selected learning rate, 1200
updates, retrained on the new runtime with the frozen code, seed and data) is fine-tuned for 300 more updates
with its 40 private heads frozen, so every attribute depends only on the shared trunk. Arms, step rules, seeds,
data order and audits are those of certified_step_training.py, imported unchanged.

Rounded same-minibatch checks, not an interval enclosure or a population guarantee.
"""
import os
import sys
from pathlib import Path
import numpy as np
import torch

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))
import certified_step_training as cst

INIT = Path(os.environ.get('PROTECTED_INIT', '/content/rebuttal/celeba_v2/runs/development_uniform_seed9001_lr0.02.pt'))


class TrainedModel(cst.Model):
    """The frozen CelebA model class, initialized from the uniform-trained development checkpoint."""

    def __init__(self):
        super().__init__()
        state = torch.load(INIT, map_location='cpu', weights_only=False)
        if not state['result']['complete'] or state['result']['config']['phase'] != 'development' or state['result']['config']['arm'] != 'uniform':
            raise ValueError('Complete uniform development checkpoint required')
        self.load_state_dict(state['model'])


_weigh = cst.weigh


def frozen_heads_weigh(model, x, y, arm, row):
    w, d, hg, losses, response = _weigh(model, x, y, arm, row)
    row['heads'] = 'frozen: zero head update and zero head first-order term'
    return w, d, {n: torch.zeros_like(g) for n, g in hg.items()}, losses, response


def zero_head_terms(hg):
    """Frozen heads contribute nothing to the executed update, so their first-order term is zero, in the audit too."""
    return np.zeros(hg['head.weight'].shape[0])


cst.Model = TrainedModel
cst.weigh = frozen_heads_weigh
cst.head_terms = zero_head_terms
cst.SOURCE_FILES = ['protected_finetune_v2.py'] + cst.SOURCE_FILES

if __name__ == '__main__':
    cst.main()
