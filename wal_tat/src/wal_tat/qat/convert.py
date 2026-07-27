"""Model surgery: swap every target ``nn.Linear`` for a :class:`QuantLinear`.

The conversion is *global and simultaneous*: all target matrices become
quantized at step zero so the network can co-adapt, which is the opposite of
the incremental per-matrix pipeline.  Everything outside the target set --
the convolutional frontend, embeddings, positional tables, layer norms,
biases and the tied output head -- is left untouched in its original dtype,
so the honest compression ceiling is reported by :class:`QATManifest`
instead of being assumed.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Callable, Dict, Iterator, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from ..binary_packing import pack_binary_matrix, write_packed_binary_matrix
from ..moments import InputSecondMomentCollector
from ..packing import pack_partial_matrix, write_packed_matrix
from ..whisper import set_module
from .quant import (
    PRECISIONS,
    IdentityQuantizer,
    QuantLinear,
    SCALE_EPS,
    get_quantizer,
)

__all__ = [
    "DEFAULT_TARGETS",
    "QATEntry",
    "QATManifest",
    "collect_input_second_moments",
    "convert_model_to_qat",
    "export_packed_artifact",
    "load_qat_checkpoint",
    "qat_modules",
    "save_qat_checkpoint",
]

#: Leaf module names quantized by default: attention projections and MLP.
#: Covers Whisper self-attention, cross-attention and both MLP matrices.
DEFAULT_TARGETS: Tuple[str, ...] = (
    "q_proj",
    "k_proj",
    "v_proj",
    "out_proj",
    "fc1",
    "fc2",
)

CHECKPOINT_FORMAT = "wal-tat-qat-1"


@dataclass(frozen=True)
class QATEntry:
    """One converted matrix."""

    name: str
    out_features: int
    in_features: int
    groups: int
    group_size: int

    @property
    def weights(self) -> int:
        return self.out_features * self.in_features

    @property
    def num_groups(self) -> int:
        return self.out_features * self.groups

    def to_dict(self) -> Dict[str, int | str]:
        return {
            "name": self.name,
            "out_features": self.out_features,
            "in_features": self.in_features,
            "groups": self.groups,
            "group_size": self.group_size,
            "weights": self.weights,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "QATEntry":
        return cls(
            name=str(payload["name"]),
            out_features=int(payload["out_features"]),  # type: ignore[arg-type]
            in_features=int(payload["in_features"]),  # type: ignore[arg-type]
            groups=int(payload["groups"]),  # type: ignore[arg-type]
            group_size=int(payload["group_size"]),  # type: ignore[arg-type]
        )


@dataclass(frozen=True)
class QATManifest:
    """Coverage and size accounting for a converted model."""

    precision: str
    group_size: int
    learnable_scales: bool
    entries: Tuple[QATEntry, ...]
    untouched_linear: Tuple[str, ...] = ()
    #: Parameters of the equivalent dense model, i.e. excluding the learned
    #: group scales, which are billed separately as quantization metadata.
    model_parameters: int = 0
    #: Weights of every ``nn.Linear``-shaped matrix, converted or not.
    linear_parameters: int = 0
    baseline_bits: int = 16
    scale_bits: int = 16

    # ------------------------------------------------------------- coverage

    @property
    def num_matrices(self) -> int:
        return len(self.entries)

    @property
    def quantized_weights(self) -> int:
        return sum(entry.weights for entry in self.entries)

    @property
    def quantized_groups(self) -> int:
        return sum(entry.num_groups for entry in self.entries)

    @property
    def linear_coverage(self) -> float:
        """Fraction of all ``nn.Linear`` weights in the model that are low-bit."""
        if self.linear_parameters <= 0:
            return 0.0
        return self.quantized_weights / self.linear_parameters

    @property
    def parameter_coverage(self) -> float:
        """Fraction of *all* model parameters that are low-bit."""
        if self.model_parameters <= 0:
            return 0.0
        return self.quantized_weights / self.model_parameters

    # ----------------------------------------------------------------- size

    @property
    def quantized_payload_bits(self) -> int:
        code_bits = get_quantizer(self.precision).code_bits
        return sum(
            entry.num_groups * entry.group_size * code_bits
            + entry.num_groups * self.scale_bits
            for entry in self.entries
        )

    @property
    def residual_bits(self) -> int:
        """Bits spent on everything that stays in the baseline dtype."""
        residual = self.model_parameters - self.quantized_weights
        return max(residual, 0) * self.baseline_bits

    @property
    def total_bits(self) -> int:
        return self.quantized_payload_bits + self.residual_bits

    @property
    def effective_bpw(self) -> float:
        if self.model_parameters <= 0:
            return 0.0
        return self.total_bits / self.model_parameters

    @property
    def matrix_bpw(self) -> float:
        """Bits per weight inside the converted matrices only."""
        if self.quantized_weights <= 0:
            return 0.0
        return self.quantized_payload_bits / self.quantized_weights

    @property
    def compression_ratio(self) -> float:
        """Whole-model compression against the ``baseline_bits`` checkpoint."""
        if self.total_bits <= 0:
            return 0.0
        return self.model_parameters * self.baseline_bits / self.total_bits

    # --------------------------------------------------------- serialization

    def to_dict(self) -> Dict[str, object]:
        return {
            "format": CHECKPOINT_FORMAT,
            "precision": self.precision,
            "group_size": self.group_size,
            "learnable_scales": self.learnable_scales,
            "baseline_bits": self.baseline_bits,
            "scale_bits": self.scale_bits,
            "model_parameters": self.model_parameters,
            "linear_parameters": self.linear_parameters,
            "num_matrices": self.num_matrices,
            "quantized_weights": self.quantized_weights,
            "quantized_groups": self.quantized_groups,
            "linear_coverage": self.linear_coverage,
            "parameter_coverage": self.parameter_coverage,
            "matrix_bpw": self.matrix_bpw,
            "effective_bpw": self.effective_bpw,
            "compression_ratio": self.compression_ratio,
            "untouched_linear": list(self.untouched_linear),
            "entries": [entry.to_dict() for entry in self.entries],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "QATManifest":
        return cls(
            precision=str(payload["precision"]),
            group_size=int(payload["group_size"]),  # type: ignore[arg-type]
            learnable_scales=bool(payload["learnable_scales"]),
            entries=tuple(
                QATEntry.from_dict(item)
                for item in payload["entries"]  # type: ignore[union-attr]
            ),
            untouched_linear=tuple(payload.get("untouched_linear", ())),  # type: ignore[arg-type]
            model_parameters=int(payload.get("model_parameters", 0)),  # type: ignore[arg-type]
            linear_parameters=int(payload.get("linear_parameters", 0)),  # type: ignore[arg-type]
            baseline_bits=int(payload.get("baseline_bits", 16)),  # type: ignore[arg-type]
            scale_bits=int(payload.get("scale_bits", 16)),  # type: ignore[arg-type]
        )

    def summary(self) -> str:
        return (
            f"{self.num_matrices} matrices / {self.quantized_weights:,} weights "
            f"@ {self.precision} g{self.group_size} | "
            f"linear coverage {100 * self.linear_coverage:.2f}% | "
            f"parameter coverage {100 * self.parameter_coverage:.2f}% | "
            f"matrix {self.matrix_bpw:.3f} bpw | "
            f"model {self.effective_bpw:.3f} bpw "
            f"({self.compression_ratio:.2f}x vs {self.baseline_bits}-bit)"
        )


def _matches(name: str, targets: Sequence[str], exclude: Sequence[str]) -> bool:
    leaf = name.rsplit(".", 1)[-1]
    if leaf not in targets:
        return False
    return not any(pattern in name for pattern in exclude)


def iter_target_linears(
    model: nn.Module,
    *,
    targets: Sequence[str] = DEFAULT_TARGETS,
    exclude: Sequence[str] = (),
    predicate: Optional[Callable[[str, nn.Module], bool]] = None,
) -> Iterator[Tuple[str, nn.Module]]:
    """Yield ``(name, module)`` for every linear layer selected for conversion.

    Selection is by leaf module name so that the tied output head
    (``proj_out``) and every non-linear parameter are structurally excluded.
    Already converted :class:`QuantLinear` layers are yielded too, which lets
    calibration hooks be attached before or after conversion.
    """
    for name, module in model.named_modules():
        if not isinstance(module, (nn.Linear, QuantLinear)):
            continue
        if not _matches(name, targets, exclude):
            continue
        if predicate is not None and not predicate(name, module):
            continue
        yield name, module


def qat_modules(model: nn.Module) -> Dict[str, QuantLinear]:
    """Every :class:`QuantLinear` currently installed, keyed by dotted name."""
    return {
        name: module
        for name, module in model.named_modules()
        if isinstance(module, QuantLinear)
    }


def _linear_parameter_count(model: nn.Module) -> int:
    total = 0
    for module in model.modules():
        if isinstance(module, nn.Linear):
            total += module.weight.numel()
        elif isinstance(module, QuantLinear):
            total += module.num_weights
    return total


def _dense_parameter_count(model: nn.Module) -> int:
    """Parameters of the equivalent dense model.

    Learned group scales are excluded: they are quantization metadata, already
    budgeted at ``scale_bits`` per group inside
    :attr:`QATManifest.quantized_payload_bits`.  Counting them here would both
    inflate the coverage denominator and bill them twice.
    """
    scales = {
        id(module.group_scale)
        for module in model.modules()
        if isinstance(module, QuantLinear) and module.group_scale is not None
    }
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if id(parameter) not in scales
    )


def convert_model_to_qat(
    model: nn.Module,
    *,
    precision: str = "t3",
    group_size: int = 128,
    targets: Sequence[str] = DEFAULT_TARGETS,
    exclude: Sequence[str] = (),
    predicate: Optional[Callable[[str, nn.Module], bool]] = None,
    learnable_scales: bool = True,
    scale_eps: float = SCALE_EPS,
    fake_fp16_scale: bool = True,
    input_second_moments: Optional[Mapping[str, torch.Tensor]] = None,
    freeze_non_quantized: bool = False,
    baseline_bits: int = 16,
) -> QATManifest:
    """Swap every target ``nn.Linear`` for a :class:`QuantLinear` in place.

    Call this *after* the model has been moved to its device and compute
    dtype: latent weights are always fp32 and a later ``model.half()`` would
    silently destroy them.

    ``input_second_moments`` maps module names to per-input-feature activation
    second moments (see :func:`collect_input_second_moments`) and enables the
    exact activation-weighted scale initialization.
    """
    # "fp" is the identity control arm: a real quantizer object, but kept out
    # of PRECISIONS because it is a measurement device, not a shippable format.
    if precision not in PRECISIONS and precision != IdentityQuantizer.name:
        raise ValueError(
            f"precision must be one of {PRECISIONS} "
            f"(or {IdentityQuantizer.name!r} for the full-precision control)"
        )
    selected = [
        (name, module)
        for name, module in iter_target_linears(
            model, targets=targets, exclude=exclude, predicate=predicate
        )
        if isinstance(module, nn.Linear)
    ]
    entries: list[QATEntry] = []
    for name, linear in selected:
        moment = None if input_second_moments is None else input_second_moments.get(name)
        quantized = QuantLinear.from_linear(
            linear,
            precision=precision,
            group_size=group_size,
            learnable_scales=learnable_scales,
            scale_eps=scale_eps,
            fake_fp16_scale=fake_fp16_scale,
            input_second_moment=moment,
        )
        set_module(model, name, quantized)
        entries.append(
            QATEntry(
                name=name,
                out_features=quantized.out_features,
                in_features=quantized.in_features,
                groups=quantized.groups,
                group_size=quantized.group_size,
            )
        )
    converted = {entry.name for entry in entries}
    untouched = tuple(
        name
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear) and name not in converted
    )
    manifest = QATManifest(
        precision=precision,
        group_size=group_size,
        learnable_scales=learnable_scales,
        entries=tuple(entries),
        untouched_linear=untouched,
        model_parameters=_dense_parameter_count(model),
        linear_parameters=_linear_parameter_count(model),
        baseline_bits=baseline_bits,
    )
    if freeze_non_quantized:
        quantized_parameters = {
            id(parameter)
            for module in qat_modules(model).values()
            for parameter in module.parameters()
        }
        for parameter in model.parameters():
            if id(parameter) not in quantized_parameters:
                parameter.requires_grad_(False)
    return manifest


@torch.no_grad()
def collect_input_second_moments(
    model: nn.Module,
    run_forward: Callable[[], object],
    *,
    targets: Sequence[str] = DEFAULT_TARGETS,
    exclude: Sequence[str] = (),
    predicate: Optional[Callable[[str, nn.Module], bool]] = None,
) -> Dict[str, torch.Tensor]:
    """Diagonal activation second moments for every target linear layer.

    ``run_forward`` runs one or more calibration forward passes over ``model``
    (teacher forced -- no decoding is required).  The returned tensors are the
    per-input-feature mean of ``x**2`` and can be handed straight to
    :func:`convert_model_to_qat`.
    """
    collectors = {
        name: InputSecondMomentCollector(module)
        for name, module in iter_target_linears(
            model, targets=targets, exclude=exclude, predicate=predicate
        )
    }
    if not collectors:
        raise ValueError("no target linear layers were found")
    try:
        run_forward()
        return {
            name: collector.moment().detach().float()
            for name, collector in collectors.items()
        }
    finally:
        for collector in collectors.values():
            collector.close()


def save_qat_checkpoint(
    model: nn.Module,
    path: str | Path,
    *,
    manifest: QATManifest,
    extra: Optional[Mapping[str, object]] = None,
) -> Path:
    """Write the full model state plus the conversion manifest.

    The whole ``state_dict`` is stored -- latent fp32 weights, group scales
    and every parameter that stayed in the baseline dtype -- so that
    :func:`load_qat_checkpoint` restores a bit-identical training state.
    """
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": CHECKPOINT_FORMAT,
        "manifest": manifest.to_dict(),
        "state_dict": {
            key: value.detach().cpu() for key, value in model.state_dict().items()
        },
        "extra": dict(extra or {}),
    }
    temporary = output.with_suffix(output.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(output)
    return output


def load_qat_checkpoint(
    model: nn.Module,
    path: str | Path,
    *,
    convert: bool = True,
    map_location: str | torch.device = "cpu",
    strict: bool = True,
    weights_only: bool = True,
) -> Tuple[QATManifest, Dict[str, object]]:
    """Restore a checkpoint written by :func:`save_qat_checkpoint`.

    When ``convert`` is set the target layers of a plain ``model`` are swapped
    first, using the precision and group size recorded in the checkpoint.
    Returns the manifest and the ``extra`` payload.  ``weights_only`` may be
    disabled for checkpoints whose ``extra`` payload holds non-primitive
    objects; keep it enabled for untrusted files.
    """
    payload = torch.load(
        Path(path), map_location=map_location, weights_only=weights_only
    )
    if payload.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(f"unsupported checkpoint format: {payload.get('format')!r}")
    manifest = QATManifest.from_dict(payload["manifest"])
    if convert and not qat_modules(model):
        names = {entry.name for entry in manifest.entries}
        convert_model_to_qat(
            model,
            precision=manifest.precision,
            group_size=manifest.group_size,
            learnable_scales=manifest.learnable_scales,
            predicate=lambda name, _module: name in names,
            targets=tuple({entry.name.rsplit(".", 1)[-1] for entry in manifest.entries}),
            baseline_bits=manifest.baseline_bits,
        )
    model.load_state_dict(payload["state_dict"], strict=strict)
    return manifest, dict(payload.get("extra", {}))


def export_packed_artifact(
    model: nn.Module,
    directory: str | Path,
    *,
    manifest: Optional[QATManifest] = None,
) -> Dict[str, object]:
    """Write every converted matrix in the deployable packing format.

    Ternary matrices use :func:`wal_tat.packing.pack_partial_matrix` with all
    groups committed; binary matrices use
    :func:`wal_tat.binary_packing.pack_binary_matrix`.  A ``manifest.json``
    records the byte budget of the artifact next to the coverage numbers.
    """
    output = Path(directory)
    output.mkdir(parents=True, exist_ok=True)
    modules = qat_modules(model)
    if not modules:
        raise ValueError("model has no QuantLinear layers to export")
    files: list[Dict[str, object]] = []
    payload_bytes = 0
    for name, module in sorted(modules.items()):
        codes, scales = module.export_codes()
        if module.precision == "t3":
            packed = pack_partial_matrix(
                module.weight.detach(),
                codes,
                scales,
                torch.ones(codes.shape[:2], dtype=torch.bool),
                group_size=module.group_size,
            )
            suffix = ".waltq2"
            writer = write_packed_matrix
        else:
            packed = pack_binary_matrix(
                codes,
                scales,
                shape=(module.out_features, module.in_features),
                group_size=module.group_size,
            )
            suffix = ".waltb1"
            writer = write_packed_binary_matrix
        target = output / f"{name}{suffix}"
        if target.exists():
            target.unlink()
        size = writer(target, packed)
        payload_bytes += size
        files.append(
            {
                "name": name,
                "file": target.name,
                "bytes": size,
                "true_bpw": packed.true_bpw(),
            }
        )
    summary: Dict[str, object] = {
        "format": CHECKPOINT_FORMAT,
        "precision": next(iter(modules.values())).precision,
        "matrices": len(files),
        "artifact_bytes": payload_bytes,
        "files": files,
    }
    if manifest is not None:
        summary["manifest"] = manifest.to_dict()
    (output / "manifest.json").write_text(json.dumps(summary, indent=2))
    return summary
