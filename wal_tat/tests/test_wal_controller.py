import json

import pytest
import torch

from wal_tat import (
    AdaptiveTransactionSizer,
    HashChainWAL,
    RatioGate,
    TransactionController,
    TransactionalTernaryMatrix,
    WALIntegrityError,
)


def test_adaptive_sizer_shrinks_tight_pass_and_rollback():
    sizer = AdaptiveTransactionSizer(
        1 / 128, minimum_fraction=1 / 1024, maximum_fraction=1 / 32
    )
    tight = sizer.observe(passed=True, worst_ratio=1.00161, gate_ratio=1.00194)
    assert tight.reason == "tight_gate_shrink"
    assert tight.next_fraction == pytest.approx(1 / 256)
    assert tight.normalized_headroom == pytest.approx((1.00194 - 1.00161) / 0.00194)

    failed = sizer.observe(passed=False, worst_ratio=1.0021, gate_ratio=1.00195)
    assert failed.reason == "rollback_shrink"
    assert failed.next_fraction == pytest.approx(1 / 512)
    assert sizer.group_count(98_304) == 192


def test_adaptive_sizer_grows_only_after_roomy_streak():
    sizer = AdaptiveTransactionSizer(
        1 / 256,
        minimum_fraction=1 / 1024,
        maximum_fraction=1 / 32,
        grow_after=2,
    )
    first = sizer.observe(passed=True, worst_ratio=0.999, gate_ratio=1.002)
    second = sizer.observe(passed=True, worst_ratio=0.999, gate_ratio=1.002)
    assert first.reason == "safe_streak_hold"
    assert second.reason == "safe_streak_grow"
    assert second.next_fraction == pytest.approx(1 / 128)


def test_adaptive_sizer_roomy_streak_survives_process_boundary():
    first_process = AdaptiveTransactionSizer(
        1 / 256,
        minimum_fraction=1 / 1024,
        maximum_fraction=1 / 32,
        grow_after=2,
    )
    first = first_process.observe(
        passed=True, worst_ratio=0.999, gate_ratio=1.002
    )
    assert first.reason == "safe_streak_hold"
    assert first_process.roomy_passes == 1

    second_process = AdaptiveTransactionSizer(
        first.next_fraction,
        minimum_fraction=1 / 1024,
        maximum_fraction=1 / 32,
        grow_after=2,
        roomy_passes=first_process.roomy_passes,
    )
    second = second_process.observe(
        passed=True, worst_ratio=0.999, gate_ratio=1.002
    )
    assert second.reason == "safe_streak_grow"
    assert second.next_fraction == pytest.approx(1 / 128)
    assert second_process.roomy_passes == 0


def test_hash_chain_round_trip(tmp_path):
    path = tmp_path / "transactions.jsonl"
    wal = HashChainWAL(path, fsync=False)
    first = wal.append("begin", "tx-1", {"groups": 2})
    second = wal.append("commit", "tx-1", {"ratios": {"wiki": 1.01}})
    records = HashChainWAL(path, fsync=False).verify()
    assert [record.kind for record in records] == ["begin", "commit"]
    assert second.previous_hash == first.digest


def test_hash_chain_detects_tampering(tmp_path):
    path = tmp_path / "transactions.jsonl"
    wal = HashChainWAL(path, fsync=False)
    wal.append("begin", "tx-1", {"groups": 2})
    raw = json.loads(path.read_text())
    raw["payload"]["groups"] = 999
    path.write_text(json.dumps(raw) + "\n")
    with pytest.raises(WALIntegrityError, match="digest mismatch"):
        HashChainWAL(path, fsync=False)


def test_controller_commits_passing_candidate(tmp_path):
    matrix = TransactionalTernaryMatrix(torch.randn(2, 4), group_size=2)
    controller = TransactionController(
        HashChainWAL(tmp_path / "pass.jsonl", fsync=False), "toy", RatioGate(1.02)
    )
    mask = torch.tensor([[True, False], [False, False]])
    controller.begin(matrix, mask, selector="causal")
    decision = controller.decide(
        matrix, baseline={"wiki": 2.0, "code": 3.0}, candidate={"wiki": 2.01, "code": 3.05}
    )
    assert decision.passed
    assert matrix.committed_mask.sum() == 1
    assert controller.wal.verify()[-1].kind == "commit"
    assert [record.kind for record in controller.wal.verify()] == [
        "begin", "commit_intent", "commit"
    ]


def test_controller_rolls_back_failing_candidate(tmp_path):
    weight = torch.randn(2, 4)
    matrix = TransactionalTernaryMatrix(weight, group_size=2)
    controller = TransactionController(HashChainWAL(tmp_path / "fail.jsonl", fsync=False), "toy")
    mask = torch.tensor([[True, False], [False, False]])
    controller.begin(matrix, mask, selector="causal")
    with torch.no_grad():
        matrix.master_weight[0, :2] += 5
    decision = controller.decide(
        matrix, baseline={"wiki": 2.0}, candidate={"wiki": 2.2}
    )
    assert not decision.passed
    assert torch.equal(matrix.master_weight, weight)
    assert controller.wal.verify()[-1].kind == "rollback"
