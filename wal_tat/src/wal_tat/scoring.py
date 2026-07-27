"""Group damage estimates and candidate selection."""
from __future__ import annotations

from typing import Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from .quantization import padded_grouped


def group_weight_mse(
    weight: torch.Tensor,
    *,
    group_size: int,
    scales: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    grouped, _, _ = padded_grouped(weight.detach(), group_size)
    effective_scales = (
        grouped.abs().mean(-1).clamp_min(1e-5)
        if scales is None
        else scales.detach().float().abs().clamp_min(1e-5)
    )
    codes = (grouped / effective_scales.unsqueeze(-1)).round().clamp(-1, 1)
    return (codes * effective_scales.unsqueeze(-1) - grouped).square().sum(-1)


def activation_fisher_group_damage(
    weight: torch.Tensor,
    input_second_moment: torch.Tensor,
    output_grad_second_moment: torch.Tensor,
    *,
    group_size: int,
    scales: Optional[torch.Tensor] = None,
    relative: bool = True,
) -> torch.Tensor:
    """Diagonal Fisher proxy for changing one output-row/input-group pair."""
    if input_second_moment.numel() != weight.shape[1]:
        raise ValueError("input moment does not match in_features")
    if output_grad_second_moment.numel() != weight.shape[0]:
        raise ValueError("output-gradient moment does not match out_features")
    grouped, padding, size = padded_grouped(weight.detach(), group_size)
    input_moment = input_second_moment.detach().float().clamp_min(0)
    if padding:
        input_moment = F.pad(input_moment, (0, padding))
    input_moment = input_moment.view(1, -1, size)
    output_moment = output_grad_second_moment.detach().float().clamp_min(0).view(-1, 1)
    effective_scales = (
        grouped.abs().mean(-1).clamp_min(1e-5)
        if scales is None
        else scales.detach().float().abs().clamp_min(1e-5)
    )
    codes = (grouped / effective_scales.unsqueeze(-1)).round().clamp(-1, 1)
    delta = codes * effective_scales.unsqueeze(-1) - grouped
    damage = (delta.square() * input_moment).sum(-1) * output_moment
    if relative:
        row_energy = (grouped.square() * input_moment).sum((-1, -2)).clamp_min(1e-12)
        damage = damage / row_energy.unsqueeze(-1)
    return damage


@torch.no_grad()
def diagonal_ternary_search(
    weight: torch.Tensor,
    input_second_moment: torch.Tensor,
    *,
    group_size: int,
    threshold_multipliers: Sequence[float] = (0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 2),
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Search ternary support and solve its diagonal-WLS scale per group."""
    if not threshold_multipliers:
        raise ValueError("threshold_multipliers must not be empty")
    grouped, padding, size = padded_grouped(weight.detach(), group_size)
    moment = input_second_moment.detach().float().clamp_min(0)
    if moment.numel() != weight.shape[1]:
        raise ValueError("input moment does not match in_features")
    if padding:
        moment = F.pad(moment, (0, padding))
    moment = moment.view(1, -1, size)
    mean_abs = grouped.abs().mean(-1, keepdim=True).clamp_min(1e-8)
    best_error = torch.full(grouped.shape[:2], float("inf"), device=grouped.device)
    best_scale = torch.ones_like(best_error)
    best_codes = torch.zeros_like(grouped, dtype=torch.int8)
    sign = grouped.sign()
    for multiplier in threshold_multipliers:
        codes = sign * (grouped.abs() >= mean_abs * float(multiplier)).float()
        denominator = (moment * codes.square()).sum(-1)
        numerator = (moment * codes * grouped).sum(-1)
        scale = (numerator / denominator.clamp_min(1e-12)).abs().clamp_min(1e-5)
        error = (moment * (grouped - codes * scale.unsqueeze(-1)).square()).sum(-1)
        use = error < best_error
        best_error = torch.where(use, error, best_error)
        best_scale = torch.where(use, scale, best_scale)
        best_codes = torch.where(use.unsqueeze(-1), codes.to(torch.int8), best_codes)
    return best_codes, best_scale, best_error


@torch.no_grad()
def exact_diagonal_ternary_project(
    weight: torch.Tensor,
    input_second_moment: torch.Tensor,
    *,
    group_size: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Globally solve diagonal-weighted symmetric ternary projection.

    For a fixed positive scale, a coordinate is non-zero exactly when
    ``abs(weight) > scale / 2``; the diagonal sensitivity cancels from this
    comparison.  Therefore every optimum support is a prefix after sorting a
    group by absolute magnitude.  Evaluating all prefix lengths gives the
    exact solution in ``O(group_size log group_size)`` rather than relying on
    a fixed threshold grid.
    """
    grouped, padding, size = padded_grouped(weight.detach(), group_size)
    moment = input_second_moment.detach().float().clamp_min(0)
    if moment.numel() != weight.shape[1]:
        raise ValueError("input moment does not match in_features")
    if padding:
        moment = F.pad(moment, (0, padding))
    moment = moment.view(1, -1, size).expand_as(grouped)

    magnitude, order = grouped.abs().sort(dim=-1, descending=True)
    sorted_weight = grouped.gather(-1, order)
    sorted_moment = moment.gather(-1, order)
    weighted_abs_prefix = (sorted_moment * magnitude).cumsum(-1)
    weight_prefix = sorted_moment.cumsum(-1)
    gain = weighted_abs_prefix.square() / weight_prefix.clamp_min(1e-12)
    best_gain, best_index = gain.max(-1)
    best_scale = (
        weighted_abs_prefix.gather(-1, best_index.unsqueeze(-1)).squeeze(-1)
        / weight_prefix.gather(-1, best_index.unsqueeze(-1))
        .squeeze(-1)
        .clamp_min(1e-12)
    ).abs().clamp_min(1e-5)
    best_size = best_index + 1
    active_sorted = (
        torch.arange(size, device=grouped.device).view(1, 1, -1)
        < best_size.unsqueeze(-1)
    )
    sorted_codes = torch.where(
        active_sorted, sorted_weight.sign(), torch.zeros_like(sorted_weight)
    ).to(torch.int8)
    best_codes = torch.zeros_like(sorted_codes)
    best_codes.scatter_(-1, order, sorted_codes)
    total_error = (moment * grouped.square()).sum(-1)
    best_error = (total_error - best_gain).clamp_min(0)
    return best_codes, best_scale, best_error


def select_group_mask(
    scores: torch.Tensor,
    count: int,
    *,
    eligible: Optional[torch.Tensor] = None,
    strategy: str = "lowest",
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    if scores.ndim != 2:
        raise ValueError("scores must have shape [out_features, groups]")
    eligible = torch.ones_like(scores, dtype=torch.bool) if eligible is None else eligible
    if eligible.shape != scores.shape or eligible.dtype != torch.bool:
        raise ValueError("eligible must be a matching bool tensor")
    candidates = torch.where(eligible.reshape(-1))[0]
    if not 0 <= count <= candidates.numel():
        raise ValueError(f"count {count} is outside [0, {candidates.numel()}]")
    result = torch.zeros_like(eligible.reshape(-1))
    if count == 0:
        return result.view_as(eligible)
    if strategy == "random":
        order = torch.randperm(candidates.numel(), device=candidates.device, generator=generator)
        chosen = candidates[order[:count]]
    elif strategy in {"lowest", "highest"}:
        values = scores.reshape(-1).index_select(0, candidates).float()
        local = torch.topk(values, count, largest=strategy == "highest", sorted=False).indices
        chosen = candidates.index_select(0, local)
    else:
        raise ValueError(f"unknown strategy: {strategy}")
    result[chosen] = True
    return result.view_as(eligible)


def sensitivity_decile_mask(
    scores: torch.Tensor,
    eligible: torch.Tensor,
    *,
    decile: int,
    count: int,
) -> torch.Tensor:
    """Select deterministic, evenly spaced groups from one sensitivity decile.

    Decile 1 contains the lowest-damage eligible groups and decile 10 the
    highest-damage groups.  Even spacing avoids reporting only the easiest edge
    of a requested bucket while keeping the exact mask reproducible.
    """
    if scores.ndim != 2 or eligible.shape != scores.shape or eligible.dtype != torch.bool:
        raise ValueError("scores and eligible must be matching 2D tensors")
    if not 1 <= int(decile) <= 10:
        raise ValueError("decile must be in [1, 10]")
    if count < 1:
        raise ValueError("count must be positive")
    flat_eligible = torch.where(eligible.reshape(-1))[0]
    if flat_eligible.numel() < 10:
        raise ValueError("at least ten eligible groups are required")
    values = scores.reshape(-1).index_select(0, flat_eligible).float()
    order = torch.argsort(values, stable=True)
    ranked = flat_eligible.index_select(0, order)
    lower = ranked.numel() * (int(decile) - 1) // 10
    upper = ranked.numel() * int(decile) // 10
    bucket = ranked[lower:upper]
    if count > bucket.numel():
        raise ValueError(f"count {count} exceeds decile size {bucket.numel()}")
    if count == 1:
        chosen = bucket[bucket.numel() // 2].reshape(1)
    else:
        positions = torch.linspace(
            0,
            bucket.numel() - 1,
            steps=count,
            device=bucket.device,
        ).round().long()
        chosen = bucket.index_select(0, positions)
    result = torch.zeros_like(eligible.reshape(-1))
    result[chosen] = True
    return result.view_as(eligible)
