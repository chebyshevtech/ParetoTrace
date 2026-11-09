"""Protected fine-tuning study (rebuttal, 9 November 2026): the certified-step study in a fully shared setting.

A CelebA-40 model trained with uniform weighting (the development run at the selected learning rate, 1200 updates)
is fine-tuned for 300 more updates on the same training distribution with its 40 private heads frozen, so every
attribute depends only on the shared trunk and nothing private can offset a conflict there. Arms, step rules,
seeds, data order and audits are exactly those of certified_step_training.py, which is imported unchanged. Two
things differ: the starting model, and the head updates, which are set to zero (so the first-order responses are
the exact trunk responses K w).

Rounded same-minibatch checks, not an interval enclosure or a population guarantee.
"""
import os
import sys
from pathlib import Path
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


cst.Model = TrainedModel
cst.weigh = frozen_heads_weigh
cst.SOURCE_FILES = ['protected_finetune.py'] + cst.SOURCE_FILES

if __name__ == '__main__':
    cst.main()
