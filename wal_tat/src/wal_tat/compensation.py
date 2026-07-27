"""Structured MLP channel windows for atomic SwiGLU compensation."""
from __future__ import annotations

import math
from typing import Tuple

import torch

from .scoring import select_group_mask


def structured_channel_candidate_mask(
    scores: torch.Tensor,
    eligible: torch.Tensor,
    *,
    count: int,
    channel_block_size: int = 128,
    block_count: int = 4,
) -> Tuple[torch.Tensor, Tuple[int, ...]]:
    """Select low-damage up-projection groups inside a few channel blocks.

    Rows of an up projection are intermediate MLP channels. Constraining them
    to a small number of g128 row blocks bounds the linked down-projection
    compensation window.
    """
    if scores.ndim != 2 or eligible.shape != scores.shape or eligible.dtype != torch.bool:
        raise ValueError("scores and eligible must be matching two-dimensional tensors")
    if channel_block_size <= 0 or block_count <= 0:
        raise ValueError("channel_block_size and block_count must be positive")
    available = int(eligible.sum().item())
    if not 0 < count <= available:
        raise ValueError(f"count {count} is outside [1, {available}]")
    blocks = math.ceil(scores.shape[0] / channel_block_size)
    if block_count > blocks:
        raise ValueError("block_count exceeds the number of channel blocks")

    quota = math.ceil(count / block_count)
    ranked = []
    for block in range(blocks):
        start = block * channel_block_size
        stop = min(start + channel_block_size, scores.shape[0])
        values = scores[start:stop][eligible[start:stop]].float()
        if values.numel() == 0:
            continue
        local = torch.topk(values, min(quota, values.numel()), largest=False).values
        ranked.append((float(local.mean().item()), block, int(values.numel())))
    ranked.sort()
    if len(ranked) < block_count:
        raise ValueError("not enough eligible channel blocks")

    chosen = [block for _, block, _ in ranked[:block_count]]
    capacity = sum(available_in_block for _, block, available_in_block in ranked if block in chosen)
    cursor = block_count
    while capacity < count and cursor < len(ranked):
        _, block, available_in_block = ranked[cursor]
        chosen.append(block)
        capacity += available_in_block
        cursor += 1
    if capacity < count:
        raise ValueError("selected channel blocks do not contain enough eligible groups")

    restricted = torch.zeros_like(eligible)
    for block in chosen:
        start = block * channel_block_size
        restricted[start : start + channel_block_size] = eligible[
            start : start + channel_block_size
        ]
    mask = select_group_mask(scores, count, eligible=restricted, strategy="lowest")
    return mask, tuple(sorted(chosen))


def structured_down_candidate_mask(
    scores: torch.Tensor,
    eligible: torch.Tensor,
    *,
    count: int,
    block_count: int = 1,
) -> Tuple[torch.Tensor, Tuple[int, ...]]:
    """Select low-damage down-projection groups in few input-channel blocks.

    A g128 group column of a down projection consumes one 128-channel SwiGLU
    block. Restricting a transaction to a small number of columns therefore
    bounds the matching BF16 ``up_proj`` and ``gate_proj`` compensation rows.
    """
    if scores.ndim != 2 or eligible.shape != scores.shape or eligible.dtype != torch.bool:
        raise ValueError("scores and eligible must be matching two-dimensional tensors")
    if block_count <= 0:
        raise ValueError("block_count must be positive")
    available = int(eligible.sum().item())
    if not 0 < count <= available:
        raise ValueError(f"count {count} is outside [1, {available}]")
    if block_count > scores.shape[1]:
        raise ValueError("block_count exceeds the number of down input blocks")

    quota = math.ceil(count / block_count)
    ranked = []
    for block in range(scores.shape[1]):
        values = scores[:, block][eligible[:, block]].float()
        if values.numel() == 0:
            continue
        local = torch.topk(values, min(quota, values.numel()), largest=False).values
        ranked.append((float(local.mean().item()), block, int(values.numel())))
    ranked.sort()
    if len(ranked) < block_count:
        raise ValueError("not enough eligible down input blocks")

    chosen = [block for _, block, _ in ranked[:block_count]]
    capacity = sum(
        available_in_block
        for _, block, available_in_block in ranked
        if block in chosen
    )
    cursor = block_count
    while capacity < count and cursor < len(ranked):
        _, block, available_in_block = ranked[cursor]
        chosen.append(block)
        capacity += available_in_block
        cursor += 1
    if capacity < count:
        raise ValueError("selected down input blocks do not contain enough eligible groups")

    restricted = torch.zeros_like(eligible)
    for block in chosen:
        restricted[:, block] = eligible[:, block]
    mask = select_group_mask(scores, count, eligible=restricted, strategy="lowest")
    return mask, tuple(sorted(chosen))


def linked_down_group_mask(
    down_group_state: torch.Tensor,
    channel_blocks: Tuple[int, ...],
) -> torch.Tensor:
    """Reopen all down-projection output rows for linked input-channel blocks."""
    if down_group_state.ndim != 2 or down_group_state.dtype != torch.bool:
        raise ValueError("down_group_state must be a two-dimensional bool tensor")
    result = torch.zeros_like(down_group_state)
    for block in channel_blocks:
        if not 0 <= block < result.shape[1]:
            raise ValueError(f"channel block {block} is outside down group range")
        result[:, block] = True
    return result


def linked_output_group_mask(
    group_state: torch.Tensor,
    channel_blocks: Tuple[int, ...],
    *,
    channel_block_size: int = 128,
) -> torch.Tensor:
    """Reopen committed output rows belonging to intermediate-channel blocks.

    Gate and up projections share their output coordinate: each row is one
    SwiGLU intermediate channel.  A candidate gate row block can therefore be
    compensated by the matching rows of an already ternary up projection.
    Only committed groups are returned so the helper never expands coverage as
    a side effect of a compensation transaction.
    """
    if group_state.ndim != 2 or group_state.dtype != torch.bool:
        raise ValueError("group_state must be a two-dimensional bool tensor")
    if channel_block_size <= 0:
        raise ValueError("channel_block_size must be positive")
    block_count = math.ceil(group_state.shape[0] / channel_block_size)
    result = torch.zeros_like(group_state)
    for block in channel_blocks:
        if not 0 <= block < block_count:
            raise ValueError(f"channel block {block} is outside output row range")
        start = block * channel_block_size
        stop = min(start + channel_block_size, group_state.shape[0])
        result[start:stop] = group_state[start:stop]
    return result


def linked_gqa_output_group_mask(
    eligible: torch.Tensor,
    kv_head_blocks: Tuple[int, ...],
    *,
    query_heads_per_kv: int,
) -> torch.Tensor:
    """Select O-projection input groups fed by chosen GQA value heads.

    With g128 and head_dim=128, each V output row block is one KV head and
    each O input group is one query head.  GQA repeats a KV value head across
    ``query_heads_per_kv`` adjacent query heads before the O projection.
    """
    if eligible.ndim != 2 or eligible.dtype != torch.bool:
        raise ValueError("eligible must be a two-dimensional bool tensor")
    if query_heads_per_kv <= 0:
        raise ValueError("query_heads_per_kv must be positive")
    if eligible.shape[1] % query_heads_per_kv:
        raise ValueError("O input groups are not divisible by query-head ratio")
    kv_heads = eligible.shape[1] // query_heads_per_kv
    result = torch.zeros_like(eligible)
    for block in kv_head_blocks:
        if not 0 <= block < kv_heads:
            raise ValueError(f"KV head block {block} is outside O input range")
        start = block * query_heads_per_kv
        stop = start + query_heads_per_kv
        result[:, start:stop] = eligible[:, start:stop]
    return result


def linked_gqa_query_group_mask(
    eligible: torch.Tensor,
    kv_head_blocks: Tuple[int, ...],
    *,
    query_heads_per_kv: int,
    head_block_size: int = 128,
) -> torch.Tensor:
    """Select Q-projection rows paired with chosen GQA key heads."""
    if eligible.ndim != 2 or eligible.dtype != torch.bool:
        raise ValueError("eligible must be a two-dimensional bool tensor")
    if query_heads_per_kv <= 0 or head_block_size <= 0:
        raise ValueError("head ratio and head block size must be positive")
    if eligible.shape[0] % head_block_size:
        raise ValueError("Q output rows are not divisible by head block size")
    query_heads = eligible.shape[0] // head_block_size
    if query_heads % query_heads_per_kv:
        raise ValueError("query heads are not divisible by GQA ratio")
    kv_heads = query_heads // query_heads_per_kv
    result = torch.zeros_like(eligible)
    for block in kv_head_blocks:
        if not 0 <= block < kv_heads:
            raise ValueError(f"KV head block {block} is outside Q output range")
        first_head = block * query_heads_per_kv
        start = first_head * head_block_size
        stop = (first_head + query_heads_per_kv) * head_block_size
        result[start:stop] = eligible[start:stop]
    return result
