"""Exact supervised-position LM head evaluation shared by every baseline.

Pythia/GPTNeoX's body is evaluated normally. Its actual output-head module sees
only predictive positions with supervised next-token labels, packed/padded per
example so module-hook Jacobian capture retains the first batch axis. Softmax
still uses the FULL vocabulary. Unknown model architectures fall back to their
ordinary forward. This is an exact computation elimination, not a new loss.
"""
from __future__ import annotations

import torch
from torch.nn import functional as F


def prepare_supervised_positions(labels):
    """Prepare once on CPU before GPU timing when the batch labels are known.

    Returns predictive position indices and their next-token targets, including
    -100 padding. Index k predicts labels[:,k+1]. Moving these small tensors to
    the model device is charged to batch preparation just like input transfer.
    """
    shifted = labels[:, 1:]
    active = shifted != -100
    counts = active.sum(dim=1)
    if not bool((counts > 0).all()):
        raise ValueError("each example requires a supervised next-token label")
    width = int(counts.max().item())
    positions = active.to(torch.int8).argsort(dim=1, descending=True, stable=True)[:, :width]
    targets = shifted.gather(1, positions)
    return positions, targets


def supervised_token_outputs(model, inputs, attention, labels, *, prepared=None,
                             force_full=False):
    """Return ``(logits[B,S,V], targets[B,S], original_positions[B,S])``.

    ``prepared`` may be the device-resident pair from
    :func:`prepare_supervised_positions`. For exact Pythia/GPTNeoX support the
    model must expose its normal gpt_neox body and output embedding module.
    Other architectures use their standard causal forward with all positions.
    """
    is_neox = (getattr(getattr(model, "config", None), "model_type", None) == "gpt_neox"
               and hasattr(model, "gpt_neox") and callable(getattr(model, "get_output_embeddings", None)))
    if force_full or not is_neox:
        output = model(inputs, attention_mask=attention, use_cache=False, return_dict=True)
        logits = output.logits[:, :-1]
        if logits.dtype != torch.float64:
            logits = logits.float()
        positions = torch.arange(logits.shape[1], device=inputs.device).expand(inputs.shape[0], -1)
        return logits, labels[:, 1:], positions
    if prepared is None:
        prepared = prepare_supervised_positions(labels)
    positions, targets = prepared
    positions = positions.to(device=inputs.device)
    targets = targets.to(device=inputs.device)
    output = model.gpt_neox(inputs, attention_mask=attention,
                            use_cache=False, return_dict=True)
    hidden = output.last_hidden_state
    gathered = hidden.gather(1, positions[:, :, None].expand(-1, -1, hidden.shape[-1]))
    logits = model.get_output_embeddings()(gathered)
    if logits.dtype != torch.float64:
        logits = logits.float()
    return logits, targets, positions


def per_example_cross_entropy(logits, targets):
    """Same per-example token-mean objective for packed and ordinary logits."""
    mask = targets != -100
    losses = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1),
                             ignore_index=-100, reduction="none").view(targets.shape)
    return losses.sum(dim=1) / mask.sum(dim=1).clamp_min(1)
