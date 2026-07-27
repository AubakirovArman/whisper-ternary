"""Distillation trainer for full-coverage quantisation-aware training.

The trainer takes a model whose target matrices have already been swapped for
:class:`~wal_tat.qat.quant.QuantLinear` (see
:func:`~wal_tat.qat.convert.convert_model_to_qat`) together with a frozen
full-precision copy of the *same* checkpoint, and trains the latent weights so
the quantised network reproduces the teacher's behaviour.

Design commitments, all deliberate:

Teacher forced only
    Every forward pass is a single teacher-forced pass over cached labels.
    There is no autoregressive decoding anywhere in the loop -- decoding is the
    single most expensive thing an ASR training loop can do and it is not
    needed to match a teacher's next-token distribution.

Simultaneous coverage
    All target matrices are quantised from step zero.  Nothing is staged,
    screened or promoted; the network co-adapts as one system.

Straight-through latent weights
    The optimiser updates the fp32 latent weight (and, optionally, the group
    scales).  Codes flip as a *consequence* of latent motion, which is what
    lets a matrix reorganise instead of being frozen at its initial support.

Loss
    ``kl_weight * T^2 * KL(teacher || student) + ce_weight * CE(labels)``
    plus an optional normalised feature-matching term on transformer block
    outputs.  ``KL(teacher || student)`` is the standard distillation
    direction: it is mode-covering and its gradient is the teacher's
    probability mass, which is exactly the signal a quantised student lacks.

The module is deliberately model-agnostic: it only assumes the model is called
as ``model(input_features=..., labels=...)`` and returns ``.logits``, which is
the Hugging Face sequence-to-sequence contract.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
import json
import math
from pathlib import Path
import time
from typing import Any, Callable, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .convert import QATManifest, qat_modules, save_qat_checkpoint
from .quant import QuantLinear

__all__ = [
    "AUTOCAST_DTYPES",
    "CodeFlipTracker",
    "DistillConfig",
    "LossTerms",
    "QATDistiller",
    "StepRecord",
    "TrainingReport",
    "cosine_lr_multiplier",
    "distillation_loss",
    "parameter_groups",
    "shift_labels_right",
    "transformer_block_modules",
]

#: Autocast dtypes selectable from a command line.
AUTOCAST_DTYPES: Mapping[str, torch.dtype] = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
}

#: Label id ignored by the loss; matches ``WhisperCollator.label_pad_id``.
IGNORE_INDEX = -100


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class DistillConfig:
    """Every knob of a distillation run, in one serialisable record.

    Learning rates
        ``lr`` applies to the latent weights of the quantised matrices and is
        the one that matters: it must be far larger than a normal finetuning
        rate because a latent weight has to travel a sizeable fraction of a
        group scale before its code changes.  ``scale_lr`` and ``fp_lr``
        default to ``lr`` scaled by their multipliers -- group scales and the
        surviving 16-bit parameters (layer norms, biases, embeddings, the
        convolutional frontend) are already well placed and only need gentle
        adaptation, so they are trained an order of magnitude slower.

    Weight decay
        Applied to latent weights only.  Decaying a group scale drives it to
        zero and silently deletes the matrix, and decaying a layer norm gain is
        a well-known mistake; :func:`parameter_groups` enforces both.
    """

    steps: int = 2000
    batch_size: int = 32
    grad_accum: int = 1

    lr: float = 2e-4
    scale_lr: Optional[float] = None
    scale_lr_mult: float = 0.1
    fp_lr: Optional[float] = None
    fp_lr_mult: float = 0.1
    weight_decay: float = 0.0
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8

    warmup_steps: int = 100
    min_lr_ratio: float = 0.05
    grad_clip: float = 1.0

    temperature: float = 2.0
    kl_weight: float = 1.0
    ce_weight: float = 0.1
    feature_weight: float = 0.0

    autocast: str = "bf16"
    train_non_quantized: bool = True
    seed: int = 0

    log_every: int = 25
    eval_every: int = 250
    eval_batches: int = 16
    save_every: int = 0
    flip_probe: int = 0  # 0 -> track every matrix

    def __post_init__(self) -> None:
        if self.steps < 1:
            raise ValueError("steps must be positive")
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        if self.grad_accum < 1:
            raise ValueError("grad_accum must be positive")
        if self.lr <= 0:
            raise ValueError("lr must be positive")
        if self.warmup_steps < 0 or self.warmup_steps >= self.steps:
            raise ValueError("warmup_steps must be in [0, steps)")
        if not 0.0 <= self.min_lr_ratio <= 1.0:
            raise ValueError("min_lr_ratio must be in [0, 1]")
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")
        if min(self.kl_weight, self.ce_weight, self.feature_weight) < 0:
            raise ValueError("loss weights must be non-negative")
        if max(self.kl_weight, self.ce_weight, self.feature_weight) <= 0:
            raise ValueError("at least one loss term must be enabled")
        if self.autocast not in AUTOCAST_DTYPES:
            raise ValueError(f"autocast must be one of {sorted(AUTOCAST_DTYPES)}")
        if self.eval_batches < 1:
            raise ValueError("eval_batches must be positive")
        if self.flip_probe < 0:
            raise ValueError("flip_probe must be non-negative")

    @property
    def autocast_dtype(self) -> torch.dtype:
        return AUTOCAST_DTYPES[self.autocast]

    @property
    def effective_scale_lr(self) -> float:
        return self.lr * self.scale_lr_mult if self.scale_lr is None else self.scale_lr

    @property
    def effective_fp_lr(self) -> float:
        return self.lr * self.fp_lr_mult if self.fp_lr is None else self.fp_lr

    @property
    def utterances_per_step(self) -> int:
        return self.batch_size * self.grad_accum

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["effective_scale_lr"] = self.effective_scale_lr
        payload["effective_fp_lr"] = self.effective_fp_lr
        payload["utterances_per_step"] = self.utterances_per_step
        return payload

    def replace(self, **changes: Any) -> "DistillConfig":
        return replace(self, **changes)


# --------------------------------------------------------------------------- #
# Schedules and parameter groups
# --------------------------------------------------------------------------- #


def shift_labels_right(
    labels: torch.Tensor, *, decoder_start_token_id: int, pad_token_id: int
) -> torch.Tensor:
    """Teacher-forced decoder inputs for a label tensor.

    Passing explicit ``decoder_input_ids`` instead of ``labels`` keeps the
    Hugging Face model from computing its own cross entropy over the whole
    vocabulary, which would double the cost of the output projection and hold
    an extra ``[batch, tokens, vocab]`` fp32 tensor alive for the backward
    pass.  The shift is the standard one: prepend the start token, drop the
    last label, and replace ignored positions by the pad id.
    """
    if labels.ndim != 2:
        raise ValueError("labels must be [batch, tokens]")
    shifted = labels.new_empty(labels.shape)
    shifted[:, 0] = decoder_start_token_id
    shifted[:, 1:] = labels[:, :-1]
    return shifted.masked_fill(shifted.eq(IGNORE_INDEX), pad_token_id)


def cosine_lr_multiplier(
    step: int, *, total_steps: int, warmup_steps: int, min_ratio: float = 0.0
) -> float:
    """Linear warmup then cosine decay, as a multiplier in ``[min_ratio, 1]``.

    ``step`` is zero-based: ``step=0`` is the first optimiser update.  The
    multiplier reaches ``min_ratio`` exactly at ``total_steps - 1``.
    """
    if total_steps < 1:
        raise ValueError("total_steps must be positive")
    if warmup_steps < 0 or warmup_steps >= total_steps:
        raise ValueError("warmup_steps must be in [0, total_steps)")
    if step < warmup_steps:
        return (step + 1) / (warmup_steps + 1)
    span = max(total_steps - 1 - warmup_steps, 1)
    progress = min((step - warmup_steps) / span, 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_ratio + (1.0 - min_ratio) * cosine


def parameter_groups(
    model: nn.Module,
    config: DistillConfig,
) -> List[Dict[str, Any]]:
    """Split trainable parameters into latent / scale / 16-bit groups.

    Three groups are produced, each with its own learning rate:

    ``latent``
        the fp32 weights of every :class:`QuantLinear`; the only group that
        carries weight decay.
    ``scale``
        the learned per-group scales; never decayed.
    ``fp16``
        everything that stays in the baseline dtype (layer norms, biases, the
        convolutional frontend, embeddings, the tied head); never decayed, and
        dropped entirely when ``train_non_quantized`` is false.
    """
    latent: List[nn.Parameter] = []
    scales: List[nn.Parameter] = []
    seen: set[int] = set()
    for module in model.modules():
        if not isinstance(module, QuantLinear):
            continue
        if module.weight.requires_grad and id(module.weight) not in seen:
            latent.append(module.weight)
            seen.add(id(module.weight))
        if module.group_scale is not None and module.group_scale.requires_grad:
            if id(module.group_scale) not in seen:
                scales.append(module.group_scale)
                seen.add(id(module.group_scale))
    rest = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad and id(parameter) not in seen
    ]
    groups: List[Dict[str, Any]] = []
    if latent:
        groups.append(
            {
                "name": "latent",
                "params": latent,
                "lr": config.lr,
                "weight_decay": config.weight_decay,
            }
        )
    if scales:
        groups.append(
            {
                "name": "scale",
                "params": scales,
                "lr": config.effective_scale_lr,
                "weight_decay": 0.0,
            }
        )
    if rest and config.train_non_quantized:
        groups.append(
            {
                "name": "fp16",
                "params": rest,
                "lr": config.effective_fp_lr,
                "weight_decay": 0.0,
            }
        )
    if not groups:
        raise ValueError("no trainable parameters were found")
    return groups


# --------------------------------------------------------------------------- #
# Loss
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class LossTerms:
    """One evaluation of the distillation objective."""

    total: torch.Tensor
    kl: float
    ce: float
    feature: float
    tokens: int

    def to_dict(self) -> Dict[str, float]:
        return {
            "loss": float(self.total.detach().item()),
            "kl": self.kl,
            "ce": self.ce,
            "feature": self.feature,
            "tokens": self.tokens,
        }


def distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    temperature: float = 2.0,
    kl_weight: float = 1.0,
    ce_weight: float = 0.1,
    feature_loss: Optional[torch.Tensor] = None,
    feature_weight: float = 0.0,
) -> LossTerms:
    """Token-averaged ``KL(teacher || student)`` plus ground-truth cross entropy.

    Only positions with a real label contribute: padding is gathered away
    before the softmax, which both removes the mask bookkeeping and cuts the
    cost of the vocabulary-sized reduction on short batches.  The KL term is
    multiplied by ``temperature ** 2`` so its gradient magnitude is independent
    of the temperature, the usual Hinton scaling.
    """
    if student_logits.shape != teacher_logits.shape:
        raise ValueError("student and teacher logits must have the same shape")
    if labels.shape != student_logits.shape[:-1]:
        raise ValueError("labels must match the logits' leading dimensions")
    vocabulary = student_logits.shape[-1]
    keep = labels.reshape(-1).ne(IGNORE_INDEX)
    tokens = int(keep.sum().item())
    if tokens == 0:
        raise ValueError("batch contains no supervised tokens")
    index = keep.nonzero(as_tuple=True)[0]
    student = student_logits.reshape(-1, vocabulary).index_select(0, index).float()
    targets = labels.reshape(-1).index_select(0, index)

    total = student.new_zeros(())
    kl_value = 0.0
    ce_value = 0.0
    if kl_weight > 0:
        teacher = (
            teacher_logits.reshape(-1, vocabulary).index_select(0, index).float()
        )
        student_log = F.log_softmax(student / temperature, dim=-1)
        teacher_log = F.log_softmax(teacher / temperature, dim=-1)
        kl = F.kl_div(
            student_log, teacher_log, reduction="batchmean", log_target=True
        ) * (temperature**2)
        kl_value = float(kl.detach().item())
        total = total + kl_weight * kl
    if ce_weight > 0:
        ce = F.cross_entropy(student, targets)
        ce_value = float(ce.detach().item())
        total = total + ce_weight * ce
    feature_value = 0.0
    if feature_weight > 0 and feature_loss is not None:
        feature_value = float(feature_loss.detach().item())
        total = total + feature_weight * feature_loss
    return LossTerms(
        total=total,
        kl=kl_value,
        ce=ce_value,
        feature=feature_value,
        tokens=tokens,
    )


# --------------------------------------------------------------------------- #
# Feature matching
# --------------------------------------------------------------------------- #


def transformer_block_modules(model: nn.Module) -> Dict[str, nn.Module]:
    """Encoder and decoder block modules of a Hugging Face seq2seq model.

    Returns an empty mapping for architectures that do not expose
    ``model.encoder.layers`` / ``model.decoder.layers``, which simply disables
    feature matching instead of failing.
    """
    inner = getattr(model, "model", model)
    blocks: Dict[str, nn.Module] = {}
    for side in ("encoder", "decoder"):
        stack = getattr(inner, side, None)
        layers = getattr(stack, "layers", None)
        if layers is None:
            continue
        for index, layer in enumerate(layers):
            blocks[f"{side}.{index}"] = layer
    return blocks


class _BlockTap:
    """Capture the hidden state produced by each transformer block."""

    def __init__(self, model: nn.Module):
        self.outputs: Dict[str, torch.Tensor] = {}
        self._handles = [
            module.register_forward_hook(self._make_hook(name))
            for name, module in transformer_block_modules(model).items()
        ]

    def _make_hook(self, name: str) -> Callable[..., None]:
        def hook(_module, _args, output) -> None:
            value = output[0] if isinstance(output, tuple) else output
            self.outputs[name] = value

        return hook

    def __len__(self) -> int:
        return len(self._handles)

    def clear(self) -> None:
        self.outputs.clear()

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def __enter__(self) -> "_BlockTap":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()


def _feature_mse(
    student: Mapping[str, torch.Tensor],
    teacher: Mapping[str, torch.Tensor],
) -> Optional[torch.Tensor]:
    """Mean over blocks of the teacher-energy-normalised MSE.

    Normalising by the teacher's own second moment makes the term comparable
    across blocks whose activations differ in scale by orders of magnitude, and
    keeps its weight interpretable as "fraction of block energy".
    """
    shared = [name for name in student if name in teacher]
    if not shared:
        return None
    total: Optional[torch.Tensor] = None
    for name in shared:
        target = teacher[name].detach().float()
        delta = student[name].float() - target
        value = delta.square().mean() / target.square().mean().clamp_min(1e-12)
        total = value if total is None else total + value
    assert total is not None
    return total / len(shared)


# --------------------------------------------------------------------------- #
# Code-flip tracking
# --------------------------------------------------------------------------- #


class CodeFlipTracker:
    """Count how many discrete codes change between two points in training.

    Snapshots live on the CPU as ``int8``, so tracking every matrix of a
    whisper-small (198M codes) costs 189 MiB of host memory and one device to
    host copy per probe -- negligible next to the training step itself.
    """

    def __init__(self, model: nn.Module, *, probe: int = 0):
        modules = qat_modules(model)
        if not modules:
            raise ValueError("model has no QuantLinear layers to track")
        names = sorted(modules)
        if probe:
            # Evenly spaced sample so the probe spans the whole depth of the
            # network rather than the first few blocks.
            stride = max(len(names) // probe, 1)
            names = names[::stride][:probe]
        self.names: Tuple[str, ...] = tuple(names)
        self._modules = {name: modules[name] for name in self.names}
        self._initial = self._snapshot()
        self._previous = {name: value.clone() for name, value in self._initial.items()}
        self.total_codes = sum(value.numel() for value in self._initial.values())

    @torch.no_grad()
    def _snapshot(self) -> Dict[str, torch.Tensor]:
        return {
            name: module.export_codes()[0].to("cpu")
            for name, module in self._modules.items()
        }

    @torch.no_grad()
    def measure(self) -> Dict[str, float]:
        """Flips since the previous call and since initialisation.

        Also reports the ternary occupancy of the current codes, which is the
        cheapest honest check that the network has not collapsed to all-zero or
        saturated to a pure sign code.
        """
        current = self._snapshot()
        since_previous = 0
        since_start = 0
        zeros = 0
        positives = 0
        negatives = 0
        for name, codes in current.items():
            since_previous += int(codes.ne(self._previous[name]).sum().item())
            since_start += int(codes.ne(self._initial[name]).sum().item())
            zeros += int(codes.eq(0).sum().item())
            positives += int(codes.eq(1).sum().item())
            negatives += int(codes.eq(-1).sum().item())
        self._previous = current
        total = float(self.total_codes)
        return {
            "matrices": len(self.names),
            "codes": self.total_codes,
            "flips_since_last": since_previous,
            "flip_rate_since_last": since_previous / total,
            "flips_since_start": since_start,
            "flip_rate_since_start": since_start / total,
            "code_fraction_zero": zeros / total,
            "code_fraction_plus": positives / total,
            "code_fraction_minus": negatives / total,
        }


@torch.no_grad()
def _quant_error(model: nn.Module, names: Sequence[str]) -> Dict[str, float]:
    """Mean and worst round-trip error of the tracked matrices."""
    modules = qat_modules(model)
    errors = [
        modules[name].quant_error()["relative_frobenius"]
        for name in names
        if name in modules
    ]
    if not errors:
        return {}
    return {
        "quant_error_mean": sum(errors) / len(errors),
        "quant_error_max": max(errors),
    }


# --------------------------------------------------------------------------- #
# Records
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class StepRecord:
    """One logged training step."""

    step: int
    loss: float
    kl: float
    ce: float
    feature: float
    lr: float
    grad_norm: float
    tokens: int
    seconds: float
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "step": self.step,
            "loss": self.loss,
            "kl": self.kl,
            "ce": self.ce,
            "feature": self.feature,
            "lr": self.lr,
            "grad_norm": self.grad_norm,
            "tokens": self.tokens,
            "seconds": self.seconds,
        }
        payload.update(self.extra)
        return payload


@dataclass
class TrainingReport:
    """Everything a run produced, ready to be written next to the checkpoint."""

    config: Dict[str, Any]
    history: List[Dict[str, Any]]
    validations: List[Dict[str, Any]]
    initial_validation: Optional[Dict[str, Any]] = None
    final_validation: Optional[Dict[str, Any]] = None
    best_validation: Optional[Dict[str, Any]] = None
    elapsed_seconds: float = 0.0
    steps_completed: int = 0
    diverged: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "config": self.config,
            "steps_completed": self.steps_completed,
            "elapsed_seconds": self.elapsed_seconds,
            "diverged": self.diverged,
            "initial_validation": self.initial_validation,
            "final_validation": self.final_validation,
            "best_validation": self.best_validation,
            "validations": self.validations,
            "history": self.history,
        }


# --------------------------------------------------------------------------- #
# Trainer
# --------------------------------------------------------------------------- #


def _cycle(loader: Iterable[Mapping[str, Any]]) -> Iterator[Mapping[str, Any]]:
    """Endless stream over a finite loader, reshuffling every epoch."""
    while True:
        for batch in loader:
            yield batch


class QATDistiller:
    """Train a quantised student to match a frozen full-precision teacher.

    Parameters
    ----------
    student
        Model already converted by :func:`convert_model_to_qat`; its latent
        weights must still be fp32.
    teacher
        Frozen copy of the same checkpoint.  It is put in ``eval`` mode and its
        parameters have ``requires_grad`` cleared, so it never appears in the
        optimiser and never allocates activations for backward.
    train_loader
        Yields ``{"input_features": ..., "labels": ...}`` batches; see
        :class:`~wal_tat.qat.data.WhisperCollator`.
    val_loader
        Held-out slice of the *training* corpus.  Evaluation datasets are never
        touched here: they exist to report a final number, not to steer a run.
    """

    def __init__(
        self,
        student: nn.Module,
        teacher: nn.Module,
        *,
        config: DistillConfig,
        train_loader: Iterable[Mapping[str, Any]],
        val_loader: Optional[Iterable[Mapping[str, Any]]] = None,
        device: Optional[torch.device | str] = None,
        manifest: Optional[QATManifest] = None,
        checkpoint_dir: Optional[Path | str] = None,
        log: Optional[Callable[[str], None]] = None,
    ):
        self.student = student
        self.teacher = teacher.eval()
        for parameter in self.teacher.parameters():
            parameter.requires_grad_(False)
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.manifest = manifest
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        if self.checkpoint_dir is not None:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log = log or (lambda message: print(message, flush=True))
        self.device = torch.device(
            device
            if device is not None
            else next(student.parameters()).device
        )
        torch.manual_seed(config.seed)

        model_config = getattr(student, "config", None)
        self._decoder_start = getattr(model_config, "decoder_start_token_id", None)
        self._pad_id = getattr(model_config, "pad_token_id", None)

        self.quant_modules: Dict[str, QuantLinear] = qat_modules(student)
        if not self.quant_modules:
            raise ValueError("student has no QuantLinear layers; convert it first")
        if not config.train_non_quantized:
            quantized = {
                id(parameter)
                for module in self.quant_modules.values()
                for parameter in module.parameters()
            }
            for parameter in student.parameters():
                if id(parameter) not in quantized:
                    parameter.requires_grad_(False)
        self.groups = parameter_groups(student, config)
        self.optimizer = torch.optim.AdamW(
            self.groups,
            lr=config.lr,
            betas=(config.beta1, config.beta2),
            eps=config.eps,
        )
        self._base_lrs = [group["lr"] for group in self.optimizer.param_groups]
        self.flips = CodeFlipTracker(student, probe=config.flip_probe)
        self._trainable = [
            parameter for group in self.groups for parameter in group["params"]
        ]
        self.history: List[Dict[str, Any]] = []
        self.validations: List[Dict[str, Any]] = []
        self._history_path = (
            self.checkpoint_dir / "history.jsonl" if self.checkpoint_dir else None
        )
        if self._history_path is not None and self._history_path.exists():
            self._history_path.unlink()

    # -------------------------------------------------------------- utilities

    def describe(self) -> str:
        parts = " | ".join(
            f"{group['name']} "
            f"{sum(p.numel() for p in group['params']):,} @ lr {group['lr']:.2e}"
            for group in self.groups
        )
        return (
            f"{len(self.quant_modules)} quantized matrices | {parts} | "
            f"autocast {self.config.autocast} | "
            f"batch {self.config.batch_size}x{self.config.grad_accum}"
        )

    def _set_lr(self, step: int) -> float:
        multiplier = cosine_lr_multiplier(
            step,
            total_steps=self.config.steps,
            warmup_steps=self.config.warmup_steps,
            min_ratio=self.config.min_lr_ratio,
        )
        for group, base in zip(self.optimizer.param_groups, self._base_lrs):
            group["lr"] = base * multiplier
        return self._base_lrs[0] * multiplier

    def _to_device(self, batch: Mapping[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
        features = batch["input_features"].to(self.device, non_blocking=True)
        labels = batch["labels"].to(self.device, non_blocking=True)
        return features, labels

    def _autocast(self):
        return torch.autocast(
            device_type=self.device.type,
            dtype=self.config.autocast_dtype,
            enabled=self.config.autocast != "fp32" and self.device.type == "cuda",
        )

    # ------------------------------------------------------------------ steps

    def _decoder_inputs(self, labels: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Model kwargs that select the teacher-forced decoder inputs."""
        if self._decoder_start is None or self._pad_id is None:
            return {"labels": labels}
        return {
            "decoder_input_ids": shift_labels_right(
                labels,
                decoder_start_token_id=self._decoder_start,
                pad_token_id=self._pad_id,
            )
        }

    def _forward(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        *,
        student_tap: Optional[_BlockTap],
        teacher_tap: Optional[_BlockTap],
    ) -> LossTerms:
        decoder = self._decoder_inputs(labels)
        with torch.no_grad(), self._autocast():
            if teacher_tap is not None:
                teacher_tap.clear()
            teacher_logits = self.teacher(
                input_features=features.to(self._teacher_dtype),
                use_cache=False,
                **decoder,
            ).logits
        with self._autocast():
            if student_tap is not None:
                student_tap.clear()
            student_logits = self.student(
                input_features=features, use_cache=False, **decoder
            ).logits
        feature_loss = None
        if student_tap is not None and teacher_tap is not None:
            feature_loss = _feature_mse(student_tap.outputs, teacher_tap.outputs)
        return distillation_loss(
            student_logits,
            teacher_logits,
            labels,
            temperature=self.config.temperature,
            kl_weight=self.config.kl_weight,
            ce_weight=self.config.ce_weight,
            feature_loss=feature_loss,
            feature_weight=self.config.feature_weight,
        )

    @property
    def _teacher_dtype(self) -> torch.dtype:
        for parameter in self.teacher.parameters():
            return parameter.dtype
        return torch.float32

    def _constrain(self) -> None:
        for module in self.quant_modules.values():
            module.constrain_()

    # ------------------------------------------------------------- validation

    @torch.no_grad()
    def validate(self, *, max_batches: Optional[int] = None) -> Dict[str, Any]:
        """Distillation loss on the held-out slice of the training corpus."""
        if self.val_loader is None:
            return {}
        limit = max_batches if max_batches is not None else self.config.eval_batches
        was_training = self.student.training
        self.student.eval()
        taps = self._make_taps()
        totals = {"loss": 0.0, "kl": 0.0, "ce": 0.0, "feature": 0.0}
        tokens = 0
        batches = 0
        started = time.perf_counter()
        try:
            for batch in self.val_loader:
                if batches >= limit:
                    break
                features, labels = self._to_device(batch)
                terms = self._forward(
                    features, labels, student_tap=taps[0], teacher_tap=taps[1]
                )
                weight = terms.tokens
                totals["loss"] += float(terms.total.item()) * weight
                totals["kl"] += terms.kl * weight
                totals["ce"] += terms.ce * weight
                totals["feature"] += terms.feature * weight
                tokens += weight
                batches += 1
        finally:
            for tap in taps:
                if tap is not None:
                    tap.close()
            self.student.train(was_training)
        if tokens == 0:
            return {}
        payload = {key: value / tokens for key, value in totals.items()}
        payload.update(
            {
                "batches": batches,
                "tokens": tokens,
                "seconds": time.perf_counter() - started,
            }
        )
        return payload

    def _make_taps(self) -> Tuple[Optional[_BlockTap], Optional[_BlockTap]]:
        if self.config.feature_weight <= 0:
            return None, None
        student_tap = _BlockTap(self.student)
        teacher_tap = _BlockTap(self.teacher)
        if len(student_tap) == 0 or len(teacher_tap) == 0:
            student_tap.close()
            teacher_tap.close()
            raise ValueError(
                "feature matching was requested but no transformer blocks were found"
            )
        return student_tap, teacher_tap

    # ------------------------------------------------------------------- loop

    def run(self) -> TrainingReport:
        """Run the configured number of optimiser steps and report."""
        config = self.config
        self.log(f"[train] {self.describe()}")
        self.log(
            f"[train] tracking {self.flips.total_codes:,} codes across "
            f"{len(self.flips.names)} matrices"
        )
        report = TrainingReport(config=config.to_dict(), history=[], validations=[])
        report.initial_validation = self._record_validation(step=0)
        stream = iter(_cycle(self.train_loader))
        taps = self._make_taps()
        self.student.train()
        best = math.inf
        started = time.perf_counter()
        window: List[float] = []
        try:
            for step in range(config.steps):
                lr = self._set_lr(step)
                step_started = time.perf_counter()
                self.optimizer.zero_grad(set_to_none=True)
                accumulated = {"loss": 0.0, "kl": 0.0, "ce": 0.0, "feature": 0.0}
                tokens = 0
                for _ in range(config.grad_accum):
                    features, labels = self._to_device(next(stream))
                    terms = self._forward(
                        features, labels, student_tap=taps[0], teacher_tap=taps[1]
                    )
                    (terms.total / config.grad_accum).backward()
                    accumulated["loss"] += float(terms.total.detach().item())
                    accumulated["kl"] += terms.kl
                    accumulated["ce"] += terms.ce
                    accumulated["feature"] += terms.feature
                    tokens += terms.tokens
                grad_norm = float(
                    torch.nn.utils.clip_grad_norm_(
                        self._trainable, config.grad_clip
                    ).item()
                )
                self.optimizer.step()
                self._constrain()
                loss = accumulated["loss"] / config.grad_accum
                window.append(loss)
                if not math.isfinite(loss):
                    report.diverged = True
                    self.log(f"[train] step {step}: non-finite loss, stopping")
                    break
                if step % config.log_every == 0 or step == config.steps - 1:
                    record = StepRecord(
                        step=step,
                        loss=loss,
                        kl=accumulated["kl"] / config.grad_accum,
                        ce=accumulated["ce"] / config.grad_accum,
                        feature=accumulated["feature"] / config.grad_accum,
                        lr=lr,
                        grad_norm=grad_norm,
                        tokens=tokens,
                        seconds=time.perf_counter() - step_started,
                        extra=self.flips.measure()
                        | _quant_error(self.student, self.flips.names)
                        | {"loss_window": sum(window) / len(window)},
                    )
                    window.clear()
                    self._emit(record)
                if config.eval_every and (step + 1) % config.eval_every == 0:
                    validation = self._record_validation(step=step + 1)
                    if validation and validation["loss"] < best:
                        best = validation["loss"]
                        report.best_validation = validation
                        self._save("best.pt", step=step + 1, validation=validation)
                if config.save_every and (step + 1) % config.save_every == 0:
                    self._save(f"step{step + 1:06d}.pt", step=step + 1)
                report.steps_completed = step + 1
        finally:
            for tap in taps:
                if tap is not None:
                    tap.close()
        report.elapsed_seconds = time.perf_counter() - started
        report.final_validation = self._record_validation(step=report.steps_completed)
        if report.final_validation and report.final_validation["loss"] < best:
            report.best_validation = report.final_validation
        report.history = list(self.history)
        report.validations = list(self.validations)
        return report

    # ---------------------------------------------------------------- logging

    def _emit(self, record: StepRecord) -> None:
        payload = record.to_dict()
        self.history.append(payload)
        self._append_jsonl({"kind": "step", **payload})
        self.log(
            f"[step {record.step:>6}] loss {record.loss:.4f} "
            f"(kl {record.kl:.4f} ce {record.ce:.4f}) "
            f"lr {record.lr:.2e} gnorm {record.grad_norm:.2f} "
            f"flips {payload['flip_rate_since_last'] * 100:.3f}% "
            f"(cum {payload['flip_rate_since_start'] * 100:.2f}%) "
            f"zero {payload['code_fraction_zero'] * 100:.1f}% "
            f"qerr {payload.get('quant_error_mean', float('nan')):.4f} "
            f"{record.seconds:.3f}s"
        )

    def _record_validation(self, *, step: int) -> Optional[Dict[str, Any]]:
        payload = self.validate()
        if not payload:
            return None
        payload = {"step": step, **payload}
        self.validations.append(payload)
        self._append_jsonl({"kind": "validation", **payload})
        self.log(
            f"[valid {step:>6}] loss {payload['loss']:.4f} "
            f"(kl {payload['kl']:.4f} ce {payload['ce']:.4f}) "
            f"over {payload['batches']} batches / {payload['tokens']:,} tokens "
            f"in {payload['seconds']:.1f}s"
        )
        return payload

    def _append_jsonl(self, payload: Mapping[str, Any]) -> None:
        if self._history_path is None:
            return
        with self._history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload) + "\n")

    def _save(
        self,
        filename: str,
        *,
        step: int,
        validation: Optional[Mapping[str, Any]] = None,
    ) -> Optional[Path]:
        if self.checkpoint_dir is None or self.manifest is None:
            return None
        path = save_qat_checkpoint(
            self.student,
            self.checkpoint_dir / filename,
            manifest=self.manifest,
            extra={
                "step": step,
                "config": self.config.to_dict(),
                "validation": dict(validation or {}),
            },
        )
        self.log(f"[save] {path}")
        return path
