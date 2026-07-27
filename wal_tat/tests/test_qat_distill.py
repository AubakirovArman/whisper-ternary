"""Unit tests for the QAT distillation trainer."""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from wal_tat.qat.convert import convert_model_to_qat, qat_modules
from wal_tat.qat.distill import (
    IGNORE_INDEX,
    CodeFlipTracker,
    DistillConfig,
    QATDistiller,
    cosine_lr_multiplier,
    distillation_loss,
    parameter_groups,
    shift_labels_right,
    transformer_block_modules,
)

VOCAB = 24
DIM = 32
FEATURES = 16
TARGETS = ("fc1", "fc2")


# --------------------------------------------------------------------------- #
# A minimal model with the Hugging Face seq2seq calling convention
# --------------------------------------------------------------------------- #


class ToyBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(DIM, DIM)
        self.fc2 = nn.Linear(DIM, DIM)
        self.norm = nn.LayerNorm(DIM)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.norm(value + self.fc2(torch.relu(self.fc1(value))))


class ToyModel(nn.Module):
    """Smallest thing that behaves like ``WhisperForConditionalGeneration``."""

    def __init__(self, blocks: int = 2) -> None:
        super().__init__()
        self.config = SimpleNamespace(decoder_start_token_id=1, pad_token_id=0)
        self.embed = nn.Embedding(VOCAB, DIM)
        self.encode = nn.Linear(FEATURES, DIM)
        self.blocks = nn.ModuleList(ToyBlock() for _ in range(blocks))
        self.head = nn.Linear(DIM, VOCAB)

    def forward(
        self,
        input_features: torch.Tensor,
        decoder_input_ids: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        use_cache: bool | None = None,
    ) -> SimpleNamespace:
        if decoder_input_ids is None:
            assert labels is not None
            decoder_input_ids = shift_labels_right(
                labels,
                decoder_start_token_id=self.config.decoder_start_token_id,
                pad_token_id=self.config.pad_token_id,
            )
        hidden = self.embed(decoder_input_ids) + torch.tanh(
            self.encode(input_features)
        ).unsqueeze(1)
        for block in self.blocks:
            hidden = block(hidden)
        return SimpleNamespace(logits=self.head(hidden))


def toy_batches(count: int = 4, batch: int = 3, tokens: int = 5) -> list[dict]:
    generator = torch.Generator().manual_seed(7)
    batches = []
    for _ in range(count):
        labels = torch.randint(2, VOCAB, (batch, tokens), generator=generator)
        labels[:, -1] = IGNORE_INDEX  # every batch has padding to ignore
        batches.append(
            {
                "input_features": torch.randn(batch, FEATURES, generator=generator),
                "labels": labels,
            }
        )
    return batches


def converted_student(**kwargs):
    torch.manual_seed(0)
    student = ToyModel()
    manifest = convert_model_to_qat(student, targets=TARGETS, **kwargs)
    return student, manifest


# --------------------------------------------------------------------------- #
# Schedule
# --------------------------------------------------------------------------- #


def test_cosine_warms_up_then_decays_to_the_floor():
    values = [
        cosine_lr_multiplier(step, total_steps=10, warmup_steps=3, min_ratio=0.1)
        for step in range(10)
    ]
    assert values[:3] == pytest.approx([0.25, 0.5, 0.75])
    assert values[3] == pytest.approx(1.0)
    assert values[-1] == pytest.approx(0.1)
    assert all(a >= b for a, b in zip(values[3:], values[4:]))


def test_cosine_without_warmup_starts_at_one():
    assert cosine_lr_multiplier(0, total_steps=5, warmup_steps=0) == pytest.approx(1.0)


def test_cosine_rejects_impossible_schedules():
    with pytest.raises(ValueError):
        cosine_lr_multiplier(0, total_steps=0, warmup_steps=0)
    with pytest.raises(ValueError):
        cosine_lr_multiplier(0, total_steps=4, warmup_steps=4)


# --------------------------------------------------------------------------- #
# Teacher forcing
# --------------------------------------------------------------------------- #


def test_shift_labels_right_prepends_start_and_masks_padding():
    labels = torch.tensor([[5, 6, IGNORE_INDEX], [7, IGNORE_INDEX, IGNORE_INDEX]])
    shifted = shift_labels_right(labels, decoder_start_token_id=9, pad_token_id=0)
    assert shifted.tolist() == [[9, 5, 6], [9, 7, 0]]
    assert shifted.shape == labels.shape


def test_shift_labels_right_rejects_non_matrices():
    with pytest.raises(ValueError):
        shift_labels_right(
            torch.tensor([1, 2]), decoder_start_token_id=0, pad_token_id=0
        )


# --------------------------------------------------------------------------- #
# Loss
# --------------------------------------------------------------------------- #


def test_identical_logits_have_zero_kl():
    logits = torch.randn(2, 3, VOCAB)
    labels = torch.randint(0, VOCAB, (2, 3))
    terms = distillation_loss(logits, logits.clone(), labels, ce_weight=0.0)
    assert terms.kl == pytest.approx(0.0, abs=1e-6)
    assert terms.tokens == 6


def test_ignored_positions_do_not_contribute():
    torch.manual_seed(0)
    student = torch.randn(1, 4, VOCAB)
    teacher = torch.randn(1, 4, VOCAB)
    labels = torch.tensor([[3, 4, IGNORE_INDEX, IGNORE_INDEX]])
    masked = distillation_loss(student, teacher, labels)
    kept = distillation_loss(
        student[:, :2], teacher[:, :2], labels[:, :2]
    )
    assert masked.tokens == 2
    assert masked.kl == pytest.approx(kept.kl, rel=1e-6)
    assert float(masked.total) == pytest.approx(float(kept.total), rel=1e-6)

    # Corrupting only the ignored positions must not change the loss.
    student[:, 2:] += 10.0
    again = distillation_loss(student, teacher, labels)
    assert float(again.total) == pytest.approx(float(masked.total), rel=1e-6)


def test_temperature_squared_keeps_the_kl_gradient_scale():
    torch.manual_seed(0)
    labels = torch.randint(0, VOCAB, (2, 3))
    teacher = torch.randn(2, 3, VOCAB)
    gradients = []
    for temperature in (1.0, 4.0):
        student = torch.randn(2, 3, VOCAB, generator=torch.Generator().manual_seed(1))
        student.requires_grad_(True)
        terms = distillation_loss(
            student, teacher, labels, temperature=temperature, ce_weight=0.0
        )
        terms.total.backward()
        gradients.append(float(student.grad.norm()))
    # Without the T^2 factor the T=4 gradient would be ~16x smaller.
    assert gradients[1] / gradients[0] > 0.25


def test_weights_select_the_reported_terms():
    torch.manual_seed(0)
    student = torch.randn(1, 2, VOCAB)
    teacher = torch.randn(1, 2, VOCAB)
    labels = torch.randint(0, VOCAB, (1, 2))
    only_ce = distillation_loss(student, teacher, labels, kl_weight=0.0, ce_weight=1.0)
    assert only_ce.kl == 0.0
    assert float(only_ce.total) == pytest.approx(only_ce.ce)
    feature = torch.tensor(0.5)
    with_feature = distillation_loss(
        student,
        teacher,
        labels,
        kl_weight=0.0,
        ce_weight=1.0,
        feature_loss=feature,
        feature_weight=2.0,
    )
    assert float(with_feature.total) == pytest.approx(only_ce.ce + 1.0)


def test_loss_rejects_fully_masked_batches_and_mismatched_shapes():
    logits = torch.randn(1, 2, VOCAB)
    with pytest.raises(ValueError):
        distillation_loss(logits, logits, torch.full((1, 2), IGNORE_INDEX))
    with pytest.raises(ValueError):
        distillation_loss(logits, torch.randn(1, 3, VOCAB), torch.zeros(1, 2).long())
    with pytest.raises(ValueError):
        distillation_loss(logits, logits, torch.zeros(1, 3).long())


# --------------------------------------------------------------------------- #
# Parameter groups
# --------------------------------------------------------------------------- #


def test_parameter_groups_isolate_scales_and_full_precision_weights():
    student, _ = converted_student()
    config = DistillConfig(lr=1e-3, scale_lr_mult=0.1, fp_lr_mult=0.5, weight_decay=0.05)
    groups = {group["name"]: group for group in parameter_groups(student, config)}
    assert set(groups) == {"latent", "scale", "fp16"}
    assert groups["latent"]["lr"] == pytest.approx(1e-3)
    assert groups["scale"]["lr"] == pytest.approx(1e-4)
    assert groups["fp16"]["lr"] == pytest.approx(5e-4)
    assert groups["latent"]["weight_decay"] == pytest.approx(0.05)
    assert groups["scale"]["weight_decay"] == 0.0
    assert groups["fp16"]["weight_decay"] == 0.0

    scale_ids = {id(p) for p in groups["scale"]["params"]}
    modules = qat_modules(student)
    assert scale_ids == {id(m.group_scale) for m in modules.values()}
    assert {id(p) for p in groups["latent"]["params"]} == {
        id(m.weight) for m in modules.values()
    }
    # A layer norm gain must land in the undecayed group, never with the latents.
    norm = student.blocks[0].norm.weight
    assert id(norm) in {id(p) for p in groups["fp16"]["params"]}

    every = sum(len(group["params"]) for group in groups.values())
    assert every == len(list(student.parameters()))


def test_derived_scales_produce_no_scale_group():
    student, _ = converted_student(learnable_scales=False)
    names = {group["name"] for group in parameter_groups(student, DistillConfig())}
    assert names == {"latent", "fp16"}


def test_freezing_non_quantized_parameters_drops_the_fp_group():
    student, _ = converted_student()
    config = DistillConfig(train_non_quantized=False)
    names = {group["name"] for group in parameter_groups(student, config)}
    assert names == {"latent", "scale"}


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "changes",
    [
        {"steps": 0},
        {"batch_size": 0},
        {"grad_accum": 0},
        {"lr": 0.0},
        {"warmup_steps": 100, "steps": 100},
        {"temperature": 0.0},
        {"kl_weight": 0.0, "ce_weight": 0.0, "feature_weight": 0.0},
        {"kl_weight": -1.0},
        {"autocast": "int8"},
        {"eval_batches": 0},
        {"min_lr_ratio": 1.5},
    ],
)
def test_config_rejects_impossible_settings(changes):
    with pytest.raises(ValueError):
        DistillConfig(**changes)


def test_config_round_trips_through_json():
    config = DistillConfig(steps=10, warmup_steps=2, lr=3e-4, scale_lr=1e-5)
    payload = config.to_dict()
    assert json.loads(json.dumps(payload)) == payload
    assert payload["effective_scale_lr"] == pytest.approx(1e-5)
    assert payload["effective_fp_lr"] == pytest.approx(3e-5)
    assert config.replace(steps=20).steps == 20


# --------------------------------------------------------------------------- #
# Code-flip tracking
# --------------------------------------------------------------------------- #


def test_flip_tracker_counts_only_real_code_changes():
    student, _ = converted_student()
    tracker = CodeFlipTracker(student)
    first = tracker.measure()
    assert first["flips_since_last"] == 0
    assert first["flips_since_start"] == 0
    assert first["matrices"] == len(qat_modules(student))
    fractions = (
        first["code_fraction_zero"]
        + first["code_fraction_plus"]
        + first["code_fraction_minus"]
    )
    assert fractions == pytest.approx(1.0)

    module = next(iter(qat_modules(student).values()))
    codes, scales = module.export_codes()
    flat = codes.reshape(-1)
    victim = int((flat == 0).nonzero()[0])
    with torch.no_grad():
        # Push one latent weight well past the +s/2 decision boundary.
        module.weight.view(-1)[victim] = 4.0 * float(scales.max())
    second = tracker.measure()
    assert second["flips_since_last"] == 1
    assert second["flips_since_start"] == 1
    assert tracker.measure()["flips_since_last"] == 0


def test_flip_tracker_probe_samples_a_subset():
    student, _ = converted_student()
    tracker = CodeFlipTracker(student, probe=2)
    assert len(tracker.names) == 2
    assert tracker.measure()["matrices"] == 2


def test_flip_tracker_needs_quantized_layers():
    with pytest.raises(ValueError):
        CodeFlipTracker(ToyModel())


# --------------------------------------------------------------------------- #
# Feature taps
# --------------------------------------------------------------------------- #


def test_block_discovery_is_empty_for_unknown_architectures():
    assert transformer_block_modules(ToyModel()) == {}


def test_feature_matching_requires_discoverable_blocks():
    student, manifest = converted_student()
    with pytest.raises(ValueError, match="no transformer blocks"):
        QATDistiller(
            student,
            ToyModel(),
            config=DistillConfig(steps=2, warmup_steps=0, feature_weight=1.0),
            train_loader=toy_batches(),
        ).run()


# --------------------------------------------------------------------------- #
# The loop
# --------------------------------------------------------------------------- #


def test_short_run_reduces_the_loss_and_moves_codes(tmp_path):
    student, manifest = converted_student()
    torch.manual_seed(0)
    teacher = ToyModel()
    with torch.no_grad():
        # Give the teacher a distinctly different head so matching it is work.
        teacher.head.weight.mul_(1.5)
    batches = toy_batches(count=6)
    distiller = QATDistiller(
        student,
        teacher,
        config=DistillConfig(
            steps=25,
            batch_size=3,
            lr=5e-3,
            warmup_steps=2,
            log_every=5,
            eval_every=10,
            eval_batches=2,
            autocast="fp32",
        ),
        train_loader=batches,
        val_loader=batches[:2],
        manifest=manifest,
        checkpoint_dir=tmp_path,
        log=lambda _message: None,
    )
    report = distiller.run()

    assert report.steps_completed == 25
    assert not report.diverged
    assert report.final_validation["loss"] < report.initial_validation["loss"]
    assert report.history[-1]["loss"] < report.history[0]["loss"]
    assert report.history[-1]["flip_rate_since_start"] > 0
    assert 0.0 < report.history[-1]["code_fraction_zero"] < 1.0
    assert report.history[-1]["quant_error_mean"] > 0

    # The teacher must be untouched and excluded from the optimiser.
    assert all(not p.requires_grad for p in teacher.parameters())
    assert not teacher.training

    lines = [json.loads(line) for line in (tmp_path / "history.jsonl").read_text().splitlines()]
    assert {line["kind"] for line in lines} == {"step", "validation"}
    assert (tmp_path / "best.pt").exists()
    payload = json.loads(json.dumps(report.to_dict()))
    assert payload["config"]["steps"] == 25


def test_scales_stay_positive_and_finite_after_training():
    student, manifest = converted_student()
    distiller = QATDistiller(
        student,
        ToyModel(),
        config=DistillConfig(
            steps=10, batch_size=3, lr=1e-1, warmup_steps=1, log_every=100,
            eval_every=0, autocast="fp32",
        ),
        train_loader=toy_batches(),
        manifest=manifest,
        log=lambda _message: None,
    )
    distiller.run()
    for module in qat_modules(student).values():
        assert torch.all(module.group_scale > 0)
        assert torch.isfinite(module.group_scale).all()
        assert torch.isfinite(module.weight).all()


def test_gradient_accumulation_matches_a_single_large_batch():
    torch.manual_seed(0)
    teacher = ToyModel()
    batches = toy_batches(count=2, batch=4, tokens=4)
    combined = [
        {
            "input_features": torch.cat([b["input_features"] for b in batches]),
            "labels": torch.cat([b["labels"] for b in batches]),
        }
    ]

    def trained_latents(loader, accum):
        student, manifest = converted_student()
        QATDistiller(
            student,
            teacher,
            config=DistillConfig(
                steps=1, batch_size=4, grad_accum=accum, lr=1e-6, warmup_steps=0,
                log_every=100, eval_every=0, autocast="fp32", grad_clip=1e9,
            ),
            train_loader=loader,
            manifest=manifest,
            log=lambda _message: None,
        ).run()
        return torch.cat(
            [
                module.weight.detach().reshape(-1)
                for _name, module in sorted(qat_modules(student).items())
            ]
        )

    accumulated = trained_latents(batches, 2)
    single = trained_latents(combined, 1)
    # Both paths take one AdamW step from the same start; identical batches must
    # therefore land on (numerically) the same weights.
    assert torch.allclose(accumulated, single, atol=1e-6)


def test_distiller_requires_a_converted_student():
    with pytest.raises(ValueError, match="convert it first"):
        QATDistiller(
            ToyModel(),
            ToyModel(),
            config=DistillConfig(steps=1, warmup_steps=0),
            train_loader=toy_batches(),
        )


def test_non_finite_loss_stops_the_run():
    student, manifest = converted_student()
    teacher = ToyModel()
    batches = toy_batches()
    batches[0]["input_features"] = torch.full_like(
        batches[0]["input_features"], float("nan")
    )
    distiller = QATDistiller(
        student,
        teacher,
        config=DistillConfig(
            steps=5, batch_size=3, lr=1e-4, warmup_steps=0, log_every=100,
            eval_every=0, autocast="fp32",
        ),
        train_loader=batches,
        manifest=manifest,
        log=lambda _message: None,
    )
    report = distiller.run()
    assert report.diverged
    assert report.steps_completed == 0
