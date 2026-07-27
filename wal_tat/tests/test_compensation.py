import pytest
import torch

from wal_tat import (
    linked_down_group_mask,
    linked_gqa_output_group_mask,
    linked_gqa_query_group_mask,
    linked_output_group_mask,
    structured_channel_candidate_mask,
    structured_down_candidate_mask,
)


def test_structured_selection_bounds_linked_blocks():
    scores = torch.arange(32, dtype=torch.float32).view(8, 4)
    eligible = torch.ones_like(scores, dtype=torch.bool)
    mask, blocks = structured_channel_candidate_mask(
        scores,
        eligible,
        count=6,
        channel_block_size=2,
        block_count=2,
    )
    assert blocks == (0, 1)
    assert mask.sum() == 6
    selected_rows = torch.where(mask.any(dim=1))[0]
    assert int(selected_rows.max()) < 4


def test_structured_selection_respects_eligibility():
    scores = torch.arange(16, dtype=torch.float32).view(4, 4)
    eligible = torch.ones_like(scores, dtype=torch.bool)
    eligible[0] = False
    mask, _ = structured_channel_candidate_mask(
        scores, eligible, count=3, channel_block_size=2, block_count=1
    )
    assert not mask[0].any()
    assert mask.sum() == 3


def test_structured_down_selection_stays_in_best_input_block():
    scores = torch.tensor(
        [[9.0, 1.0, 8.0], [9.0, 2.0, 7.0], [9.0, 3.0, 6.0], [9.0, 4.0, 5.0]]
    )
    eligible = torch.ones_like(scores, dtype=torch.bool)
    mask, blocks = structured_down_candidate_mask(
        scores, eligible, count=2, block_count=1
    )
    assert blocks == (1,)
    assert mask.sum() == 2
    assert mask[:2, 1].all()
    assert not mask[:, 0].any() and not mask[:, 2].any()


def test_linked_down_mask_selects_whole_input_groups():
    state = torch.ones((3, 5), dtype=torch.bool)
    mask = linked_down_group_mask(state, (1, 4))
    assert mask.sum() == 6
    assert mask[:, 1].all() and mask[:, 4].all()
    assert not mask[:, 0].any()


def test_linked_down_mask_rejects_bad_block():
    with pytest.raises(ValueError, match="outside"):
        linked_down_group_mask(torch.ones((2, 3), dtype=torch.bool), (3,))


def test_linked_output_mask_selects_committed_channel_rows():
    state = torch.ones((7, 3), dtype=torch.bool)
    state[3, 1] = False
    mask = linked_output_group_mask(state, (1,), channel_block_size=2)
    assert mask.sum() == 5
    assert mask[2].all()
    assert torch.equal(mask[3], torch.tensor([True, False, True]))
    assert not mask[:2].any() and not mask[4:].any()


def test_linked_output_mask_rejects_bad_block():
    with pytest.raises(ValueError, match="outside"):
        linked_output_group_mask(
            torch.ones((5, 2), dtype=torch.bool), (3,), channel_block_size=2
        )


def test_linked_gqa_mask_repeats_kv_head_into_query_heads():
    eligible = torch.ones((3, 8), dtype=torch.bool)
    eligible[1, 3] = False
    mask = linked_gqa_output_group_mask(
        eligible, (1, 3), query_heads_per_kv=2
    )
    assert mask.sum() == 11
    assert mask[:, 2].all()
    assert torch.equal(mask[:, 3], torch.tensor([True, False, True]))
    assert mask[:, 6:8].all()
    assert not mask[:, :2].any() and not mask[:, 4:6].any()


def test_linked_gqa_mask_validates_head_mapping():
    with pytest.raises(ValueError, match="divisible"):
        linked_gqa_output_group_mask(
            torch.ones((2, 5), dtype=torch.bool), (0,), query_heads_per_kv=2
        )
    with pytest.raises(ValueError, match="outside"):
        linked_gqa_output_group_mask(
            torch.ones((2, 6), dtype=torch.bool), (3,), query_heads_per_kv=2
        )


def test_linked_gqa_query_mask_expands_each_kv_head():
    eligible = torch.ones((16, 3), dtype=torch.bool)
    eligible[5, 1] = False
    mask = linked_gqa_query_group_mask(
        eligible, (1,), query_heads_per_kv=2, head_block_size=2
    )
    assert mask.sum() == 11
    assert mask[4].all()
    assert torch.equal(mask[5], torch.tensor([True, False, True]))
    assert mask[6:8].all()
    assert not mask[:4].any() and not mask[8:].any()


def test_linked_gqa_query_mask_rejects_bad_mapping():
    with pytest.raises(ValueError, match="outside"):
        linked_gqa_query_group_mask(
            torch.ones((8, 2), dtype=torch.bool),
            (2,),
            query_heads_per_kv=2,
            head_block_size=2,
        )
