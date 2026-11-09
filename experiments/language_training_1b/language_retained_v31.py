"""Graph-free ParetoTrace proposal/refinement kernels.

This module is the production successor to the retained-graph prototype.  A
normal update uses the same fused reverse pass as MGDA-FA.  A refresh adds one
loss JVP and one sparse reverse pass, then reuses the two parameter directions
for a scaled exact line search on the selected simplex segment.  No
parameter-by-objective gradient table is formed.

The scale is deliberately explicit.  ``scale=1`` is the exact minimizer of the
observed segment.  Values in ``(0, 1)`` damp the local refinement and are an
outer-training hyperparameter.  Damping does not turn this practical hybrid
into the minimax recovery rule from the paper's main theorem.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping

import numpy as np
import torch
from torch.func import jvp, vjp


TensorTree = Mapping[str, torch.Tensor]


@dataclass(frozen=True)
class ScalarRefresh:
    """Outputs and diagnostics for one graph-free scalar refinement."""

    direction: dict[str, torch.Tensor]
    weights: np.ndarray
    response: np.ndarray
    selected_index: int
    exact_segment_step: float
    executed_step: float
    quadratic_value: float
    selected_diagonal: float


def _task_means(
    values: torch.Tensor,
    task_ids: torch.Tensor,
    counts: torch.Tensor,
    objective_count: int,
) -> torch.Tensor:
    totals = torch.zeros(
        objective_count,
        device=values.device,
        dtype=torch.float64,
    )
    totals.scatter_add_(0, task_ids, values.detach().double())
    return totals / counts.double()


def fused_weighted_reverse(
    loss_fn: Callable[[TensorTree], torch.Tensor],
    params: TensorTree,
    task_ids: torch.Tensor,
    counts: torch.Tensor,
    weights: np.ndarray,
) -> tuple[dict[str, torch.Tensor], np.ndarray, Callable]:
    """Return the weighted gradient and task losses from one reverse graph.

    ``loss_fn`` returns one loss per example.  Fusing task-loss collection into
    the VJP avoids a second forward pass and makes the no-refresh path the same
    computational baseline used by the optimized MGDA-FA implementation.
    """

    objective_count = int(counts.numel())
    weight_array = np.asarray(weights, dtype=np.float64)
    if weight_array.shape != (objective_count,):
        raise ValueError("weights must have shape (objective_count,)")
    if np.any(weight_array < 0.0) or not np.isclose(weight_array.sum(), 1.0):
        raise ValueError("weights must lie on the probability simplex")

    losses, pullback = vjp(loss_fn, params)
    weight_tensor = torch.as_tensor(
        weight_array,
        device=losses.device,
        dtype=losses.dtype,
    )
    cotangent = weight_tensor[task_ids] / counts.float()[task_ids]
    direction = pullback(cotangent)[0]
    task_losses = _task_means(losses, task_ids, counts, objective_count)
    return (
        {name: value.detach() for name, value in direction.items()},
        task_losses.cpu().numpy(),
        loss_fn,
    )


def exact_action(
    loss_fn: Callable[[TensorTree], torch.Tensor],
    params: TensorTree,
    task_ids: torch.Tensor,
    counts: torch.Tensor,
    weights: np.ndarray,
) -> tuple[dict[str, torch.Tensor], np.ndarray, np.ndarray]:
    """Return ``G weights`` and its exact Gram response without forming ``G``."""

    direction, task_losses, differentiable_loss_fn = fused_weighted_reverse(
        loss_fn, params, task_ids, counts, weights
    )
    _, per_example_response = jvp(
        differentiable_loss_fn,
        (params,),
        (direction,),
    )
    response = _task_means(
        per_example_response,
        task_ids,
        counts,
        int(counts.numel()),
    )
    return direction, response.cpu().numpy(), task_losses


def scaled_scalar_refresh(
    loss_fn: Callable[[TensorTree], torch.Tensor],
    sparse_loss_fn: Callable[[TensorTree, int], torch.Tensor],
    params: TensorTree,
    task_ids: torch.Tensor,
    counts: torch.Tensor,
    weights: np.ndarray,
    *,
    scale: float = 1.0,
) -> ScalarRefresh:
    """Refine a proposal with one exact action and one sparse reverse pass.

    ``sparse_loss_fn(params, i)`` must return the mean loss for objective ``i``.
    The full exact segment minimizer is computed first, then multiplied by
    ``scale``.  This preserves the low-memory kernel while allowing a
    prespecified damping factor for noisy stochastic training.
    """

    if not 0.0 < float(scale) <= 1.0:
        raise ValueError("scale must lie in (0, 1]")
    proposal = np.asarray(weights, dtype=np.float64)
    direction, response, _ = exact_action(
        loss_fn, params, task_ids, counts, proposal
    )
    selected = int(np.argmin(response))
    selected_loss, pullback = vjp(
        lambda current: sparse_loss_fn(current, selected), params
    )
    selected_gradient = pullback(torch.ones_like(selected_loss))[0]
    diagonal = float(
        sum(
            torch.sum(value.detach().float().square()).item()
            for value in selected_gradient.values()
        )
    )
    q_value = float(proposal @ response)
    denominator = diagonal - 2.0 * float(response[selected]) + q_value
    if denominator <= 0.0:
        exact_step = 0.0
    else:
        exact_step = float(
            np.clip(
                (q_value - float(response[selected])) / denominator,
                0.0,
                1.0,
            )
        )
    executed_step = float(scale) * exact_step
    refined = proposal * (1.0 - executed_step)
    refined[selected] += executed_step
    refined_direction = {
        name: (1.0 - executed_step) * direction[name]
        + executed_step * selected_gradient[name].detach()
        for name in direction
    }
    return ScalarRefresh(
        direction=refined_direction,
        weights=refined,
        response=response,
        selected_index=selected,
        exact_segment_step=exact_step,
        executed_step=executed_step,
        quadratic_value=q_value,
        selected_diagonal=diagonal,
    )


def amortized_update_seconds(
    base_seconds: float,
    refresh_seconds: float,
    cadence: int,
) -> float:
    """Return the measured-kernel time model for a periodic refresh schedule."""

    if base_seconds <= 0.0 or refresh_seconds < base_seconds:
        raise ValueError("times must satisfy 0 < base_seconds <= refresh_seconds")
    if cadence <= 0:
        raise ValueError("cadence must be positive")
    return float(base_seconds + (refresh_seconds - base_seconds) / cadence)
