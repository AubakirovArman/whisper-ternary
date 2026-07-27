"""Second-order proposals for strict ternary code repair.

The routines in this module never introduce a continuous deployment residual.
They use a gradient and a non-negative diagonal-curvature estimate only to rank
discrete changes to an already materialized ternary matrix.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


@torch.no_grad()
def ternary_newton_proposals(
    codes: torch.Tensor,
    scales: torch.Tensor,
    gradient: torch.Tensor,
    curvature: torch.Tensor,
    *,
    in_features: int,
    curvature_multiplier: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return the best adjacent ternary move and its predicted improvement.

    For ``-1`` and ``+1`` the only adjacent destination is zero.  A zero code
    evaluates both signs.  The returned score is ``-delta_loss`` under the
    diagonal Taylor model, so positive values predict an improvement.
    """
    if codes.ndim != 3 or scales.shape != codes.shape[:2]:
        raise ValueError("codes/scales shape mismatch")
    if gradient.ndim != 2 or gradient.shape != curvature.shape:
        raise ValueError("gradient/curvature shape mismatch")
    if gradient.shape[0] != codes.shape[0]:
        raise ValueError("gradient output dimension does not match codes")
    if not torch.all((codes >= -1) & (codes <= 1)):
        raise ValueError("codes must be ternary")
    if not torch.isfinite(gradient).all():
        raise ValueError("gradient must be finite")
    if not torch.isfinite(curvature).all() or torch.any(curvature < 0):
        raise ValueError("curvature must be finite and non-negative")
    if curvature_multiplier < 0:
        raise ValueError("curvature_multiplier must be non-negative")

    group_size = int(codes.shape[-1])
    full_in_features = int(codes.shape[1]) * group_size
    if not 0 < in_features <= full_in_features:
        raise ValueError("invalid input feature count")
    if gradient.shape[1] != in_features:
        raise ValueError("gradient input dimension does not match in_features")

    padded_gradient = F.pad(
        gradient.float(), (0, full_in_features - in_features)
    ).view_as(codes)
    padded_curvature = F.pad(
        curvature.float(), (0, full_in_features - in_features)
    ).view_as(codes)
    scale = scales.float().unsqueeze(-1)

    negative_delta = -scale
    positive_delta = scale
    to_zero_delta = -codes.float() * scale

    def predicted(delta: torch.Tensor) -> torch.Tensor:
        return padded_gradient * delta + (
            0.5
            * float(curvature_multiplier)
            * padded_curvature
            * delta.square()
        )

    negative_loss = predicted(negative_delta)
    positive_loss = predicted(positive_delta)
    choose_negative = negative_loss <= positive_loss
    zero_destination = torch.where(
        choose_negative,
        -torch.ones_like(codes),
        torch.ones_like(codes),
    )
    zero_delta = torch.where(choose_negative, negative_delta, positive_delta)
    zero_loss = torch.where(choose_negative, negative_loss, positive_loss)

    destination = torch.where(codes == 0, zero_destination, torch.zeros_like(codes))
    delta = torch.where(codes == 0, zero_delta, to_zero_delta)
    delta_loss = torch.where(codes == 0, zero_loss, predicted(to_zero_delta))
    score = -delta_loss

    if in_features < full_in_features:
        column = torch.arange(full_in_features, device=codes.device).view(
            1, codes.shape[1], group_size
        )
        valid = column < in_features
        score = torch.where(valid, score, torch.full_like(score, -torch.inf))
    return destination.to(torch.int8), delta, score


@torch.no_grad()
def apply_top_groupwise_ternary_moves(
    source_codes: torch.Tensor,
    proposed_codes: torch.Tensor,
    scores: torch.Tensor,
    *,
    budget: int,
    require_positive_score: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply at most one proposed code change in each selected g-group."""
    if budget < 1:
        raise ValueError("budget must be positive")
    if proposed_codes.shape != source_codes.shape or scores.shape != source_codes.shape:
        raise ValueError("proposal shape mismatch")
    best_score, best_position = scores.max(dim=-1)
    flat_score = best_score.reshape(-1)
    eligible = torch.isfinite(flat_score)
    if require_positive_score:
        eligible &= flat_score > 0
    count = min(int(budget), int(eligible.sum().item()))
    if count == 0:
        raise ValueError("no eligible ternary moves")
    masked = torch.where(
        eligible, flat_score, torch.full_like(flat_score, -torch.inf)
    )
    chosen_score, chosen_group = torch.topk(masked, count, sorted=True)
    flat_position = best_position.reshape(-1).index_select(0, chosen_group)
    candidate = source_codes.clone().reshape(-1, source_codes.shape[-1])
    proposals = proposed_codes.reshape_as(candidate)
    candidate[chosen_group, flat_position] = proposals[
        chosen_group, flat_position
    ]
    return (
        candidate.view_as(source_codes),
        chosen_group,
        flat_position,
        chosen_score,
    )


@torch.no_grad()
def apply_top_groupwise_ternary_moves_across_matrices(
    source_codes: dict[str, torch.Tensor],
    proposed_codes: dict[str, torch.Tensor],
    scores: dict[str, torch.Tensor],
    *,
    budget: int,
    require_positive_score: bool = True,
) -> tuple[
    dict[str, torch.Tensor], list[str], torch.Tensor, torch.Tensor, torch.Tensor
]:
    """Apply a global move budget across multiple grouped T3 matrices.

    At most one code is changed in any g-group.  The budget is global, so a
    structurally linked pair such as Whisper ``fc1``/``fc2`` can spend more
    moves in whichever matrix has the stronger worst-shard evidence.
    """
    if budget < 1:
        raise ValueError("budget must be positive")
    names = list(source_codes)
    if not names:
        raise ValueError("at least one matrix is required")
    if set(proposed_codes) != set(names) or set(scores) != set(names):
        raise ValueError("matrix keys must match")

    flat_scores: list[torch.Tensor] = []
    flat_positions: list[torch.Tensor] = []
    flat_groups: list[torch.Tensor] = []
    flat_names: list[str] = []
    device = source_codes[names[0]].device
    for name in names:
        source = source_codes[name]
        proposed = proposed_codes[name]
        score = scores[name]
        if proposed.shape != source.shape or score.shape != source.shape:
            raise ValueError(f"proposal shape mismatch for {name}")
        if source.device != device or proposed.device != device or score.device != device:
            raise ValueError("all matrices must use the same device")
        best_score, best_position = score.max(dim=-1)
        flat = best_score.reshape(-1)
        flat_scores.append(flat)
        flat_positions.append(best_position.reshape(-1))
        flat_groups.append(torch.arange(flat.numel(), device=device))
        flat_names.extend([name] * flat.numel())

    all_scores = torch.cat(flat_scores)
    all_positions = torch.cat(flat_positions)
    all_groups = torch.cat(flat_groups)
    eligible = torch.isfinite(all_scores)
    if require_positive_score:
        eligible &= all_scores > 0
    count = min(int(budget), int(eligible.sum().item()))
    if count == 0:
        raise ValueError("no eligible ternary moves")
    masked = torch.where(
        eligible, all_scores, torch.full_like(all_scores, -torch.inf)
    )
    chosen_score, chosen_global = torch.topk(masked, count, sorted=True)
    chosen_position = all_positions.index_select(0, chosen_global)
    chosen_group = all_groups.index_select(0, chosen_global)
    chosen_names = [flat_names[index] for index in chosen_global.cpu().tolist()]

    candidates = {name: value.clone() for name, value in source_codes.items()}
    for index, name in enumerate(chosen_names):
        grouped = candidates[name].reshape(-1, candidates[name].shape[-1])
        proposal = proposed_codes[name].reshape_as(grouped)
        group_index = int(chosen_group[index].item())
        position = int(chosen_position[index].item())
        grouped[group_index, position] = proposal[group_index, position]
    return candidates, chosen_names, chosen_group, chosen_position, chosen_score


@torch.no_grad()
def consensus_ternary_move_scores(
    delta: torch.Tensor,
    gradients: torch.Tensor,
    curvatures: torch.Tensor,
    *,
    in_features: int,
    curvature_multiplier: float = 1.0,
) -> torch.Tensor:
    """Worst-shard predicted improvement for one fixed move per code slot.

    ``gradients`` and ``curvatures`` are ``[replica, out, in]``.  A positive
    result means the same discrete move is predicted to reduce the objective
    on every replica; taking the minimum prevents a large gain on one shard
    from hiding a loss on another.
    """
    if delta.ndim != 3:
        raise ValueError("delta must use grouped [out, groups, group_size] shape")
    if gradients.ndim != 3 or gradients.shape != curvatures.shape:
        raise ValueError("gradient/curvature replicas must have identical rank-3 shape")
    if gradients.shape[1] != delta.shape[0] or gradients.shape[2] != in_features:
        raise ValueError("replica gradient shape does not match grouped delta")
    if gradients.shape[0] < 1:
        raise ValueError("at least one replica is required")
    if not torch.isfinite(gradients).all():
        raise ValueError("replica gradients must be finite")
    if not torch.isfinite(curvatures).all() or torch.any(curvatures < 0):
        raise ValueError("replica curvatures must be finite and non-negative")
    if curvature_multiplier < 0:
        raise ValueError("curvature_multiplier must be non-negative")
    full_in_features = delta.shape[1] * delta.shape[2]
    if not 0 < in_features <= full_in_features:
        raise ValueError("invalid input feature count")
    padded_gradient = F.pad(
        gradients.float(), (0, full_in_features - in_features)
    ).view(gradients.shape[0], *delta.shape)
    padded_curvature = F.pad(
        curvatures.float(), (0, full_in_features - in_features)
    ).view(curvatures.shape[0], *delta.shape)
    expanded_delta = delta.float().unsqueeze(0)
    predicted = padded_gradient * expanded_delta + (
        0.5
        * float(curvature_multiplier)
        * padded_curvature
        * expanded_delta.square()
    )
    score = (-predicted).amin(dim=0)
    if in_features < full_in_features:
        column = torch.arange(full_in_features, device=delta.device).view(
            1, delta.shape[1], delta.shape[2]
        )
        score = torch.where(
            column < in_features, score, torch.full_like(score, -torch.inf)
        )
    return score
