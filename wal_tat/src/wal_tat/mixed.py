"""Validation and installation of composable mixed Q2/Q4 artifacts."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

from .quantization import (
    q2_g128_physical_bpw,
    q4_g128_physical_bpw,
    q8_g128_physical_bpw,
)


class FixedMixedQ2Q4Linear(nn.Module):
    """Evaluation linear reconstructed from strict Q2 and signed Q4 groups."""

    def __init__(self, weight: torch.Tensor, bias: torch.Tensor | None):
        super().__init__()
        self.register_buffer("weight", weight.detach().clone())
        self.bias = None if bias is None else nn.Parameter(
            bias.detach().clone(), requires_grad=False
        )
        self.in_features = weight.shape[1]
        self.out_features = weight.shape[0]

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return F.linear(value, self.weight.to(value.dtype), self.bias)


@dataclass(frozen=True)
class MixedArtifactInstallResult:
    new_q2_weights: int
    nz4_weights: int
    q4_weights: int
    q8_weights: int
    matrix_statistics: dict[str, dict[str, int]]


def valid_group_weight_count(
    mask: torch.Tensor, columns: int, group_size: int
) -> int:
    """Count real, non-padding weights selected by a grouped boolean mask."""
    full_groups, remainder = divmod(columns, group_size)
    count = int(mask[:, :full_groups].sum().item()) * group_size
    if remainder:
        count += int(mask[:, full_groups].sum().item()) * remainder
    return count


def _set_submodule(model: nn.Module, name: str, module: nn.Module) -> None:
    parent_name, _, child_name = name.rpartition(".")
    parent = model.get_submodule(parent_name) if parent_name else model
    setattr(parent, child_name, module)


def _install_tied_embedding_head(
    model: nn.Module,
    *,
    embedding_name: str,
    head_name: str,
    weight: torch.Tensor,
    device: str | torch.device,
) -> None:
    """Install one frozen reconstructed tensor in both tied model roles."""
    embedding = model.get_submodule(embedding_name)
    head = model.get_submodule(head_name)
    if not isinstance(embedding, nn.Embedding):
        raise TypeError(f"{embedding_name} is not an embedding layer")
    if not isinstance(head, nn.Linear):
        raise TypeError(f"{head_name} is not a linear layer")
    if (embedding.num_embeddings, embedding.embedding_dim) != tuple(weight.shape):
        raise ValueError("tied embedding shape mismatch")
    if (head.out_features, head.in_features) != tuple(weight.shape):
        raise ValueError("tied output-head shape mismatch")
    if embedding.weight is not head.weight:
        raise ValueError("requested embedding and output head are not tied")

    shared = nn.Parameter(
        weight.to(device=device, dtype=embedding.weight.dtype), requires_grad=False
    )
    replacement_embedding = nn.Embedding(
        embedding.num_embeddings,
        embedding.embedding_dim,
        padding_idx=embedding.padding_idx,
        max_norm=embedding.max_norm,
        norm_type=embedding.norm_type,
        scale_grad_by_freq=embedding.scale_grad_by_freq,
        sparse=embedding.sparse,
        device=device,
        dtype=shared.dtype,
    )
    replacement_embedding.weight = shared
    replacement_head = nn.Linear(
        head.in_features,
        head.out_features,
        bias=head.bias is not None,
        device=device,
        dtype=shared.dtype,
    )
    replacement_head.weight = shared
    if head.bias is not None:
        replacement_head.bias = nn.Parameter(
            head.bias.detach().to(device=device, dtype=shared.dtype),
            requires_grad=False,
        )
    replacement_embedding.eval()
    replacement_head.eval()
    _set_submodule(model, embedding_name, replacement_embedding)
    _set_submodule(model, head_name, replacement_head)


def _source_grouped_codes(
    entry: Mapping, *, rows: int, columns: int, group_size: int
) -> torch.Tensor:
    codes = torch.as_tensor(entry["ternary_codes_int8"], dtype=torch.int8)
    if tuple(codes.shape) != (rows, columns):
        raise ValueError("source ternary code shape mismatch")
    padding = (-columns) % group_size
    if padding:
        codes = F.pad(codes, (0, padding))
    return codes.view(rows, -1, group_size)


@torch.no_grad()
def install_mixed_q2_q4_artifact(
    model: nn.Module,
    artifact: Mapping,
    source_checkpoint: Mapping,
    *,
    device: str | torch.device,
    expected_source_sha256: str | None = None,
) -> MixedArtifactInstallResult:
    """Validate and install a consolidated mixed artifact.

    Artifact matrices may either overlap the strict source checkpoint or be
    entirely new.  New matrices must declare an all-false source committed
    mask.  This makes one artifact a reusable parent for subsequent blocks
    without weakening the immutable strict-Q2 source invariant.
    """
    artifact_format = artifact.get("format")
    if artifact_format not in {
        "wal-tat-mixed-q2-q4-v1",
        "wal-tat-mixed-q2-q4-q8-v1",
    }:
        raise ValueError("unsupported mixed artifact format")
    if expected_source_sha256 is not None and artifact.get(
        "source_checkpoint_sha256"
    ) != expected_source_sha256:
        raise ValueError("mixed artifact source checkpoint mismatch")
    group_size = int(artifact.get("group_size", 0))
    if group_size != 128:
        raise ValueError("mixed artifact group size must be 128")
    if artifact.get("q2_physical_bpw") != q2_g128_physical_bpw():
        raise ValueError("mixed artifact Q2 format mismatch")
    if artifact.get("q4_physical_bpw") != q4_g128_physical_bpw():
        raise ValueError("mixed artifact Q4 format mismatch")
    has_q8 = artifact_format == "wal-tat-mixed-q2-q4-q8-v1"
    allow_source_q2_scale_recovery = bool(
        artifact.get("allow_source_q2_scale_recovery", False)
    )
    if has_q8 and artifact.get("q8_physical_bpw") != q8_g128_physical_bpw():
        raise ValueError("mixed artifact Q8 format mismatch")

    source_entries = source_checkpoint.get("matrices", {})
    new_q2_weights = 0
    nz4_weights = 0
    q4_weights = 0
    q8_weights = 0
    matrix_statistics: dict[str, dict[str, int]] = {}
    artifact_entries = artifact.get("matrices", {})
    tied_linear_names = {
        str(entry["tied_linear_name"])
        for entry in artifact_entries.values()
        if entry.get("kind", "linear") == "tied_embedding_head"
    }
    if tied_linear_names & set(artifact_entries):
        raise ValueError("a tied output head must not have a duplicate matrix entry")

    for name, entry in artifact_entries.items():
        target = model.get_submodule(name)
        kind = entry.get("kind", "linear")
        if kind == "linear":
            if not hasattr(target, "in_features") or not hasattr(target, "out_features"):
                raise TypeError(f"{name} is not a linear-compatible layer")
            target_rows = target.out_features
            target_columns = target.in_features
        elif kind == "tied_embedding_head":
            if not isinstance(target, nn.Embedding):
                raise TypeError(f"{name} is not an embedding layer")
            tied_linear_name = str(entry.get("tied_linear_name", ""))
            if not tied_linear_name:
                raise ValueError(f"tied output-head name missing for {name}")
            tied_target = model.get_submodule(tied_linear_name)
            if not isinstance(tied_target, nn.Linear):
                raise TypeError(f"{tied_linear_name} is not a linear layer")
            if target.weight is not tied_target.weight:
                raise ValueError(f"{name} and {tied_linear_name} are not tied")
            target_rows = target.num_embeddings
            target_columns = target.embedding_dim
        else:
            raise ValueError(f"unsupported mixed matrix kind for {name}: {kind!r}")
        rows, columns = map(int, entry["shape"])
        if (rows, columns) != (target_rows, target_columns):
            raise ValueError(f"artifact matrix shape mismatch for {name}")
        groups = (columns + group_size - 1) // group_size
        code_shape = (rows, groups, group_size)
        scale_shape = (rows, groups)

        source_mask = torch.as_tensor(
            entry["source_committed_mask"], dtype=torch.bool
        )
        q4_mask = torch.as_tensor(entry["q4_mask"], dtype=torch.bool)
        q2_codes = torch.as_tensor(entry["q2_codes_int8"], dtype=torch.int8)
        q2_scales = torch.as_tensor(entry["q2_scales_fp16"], dtype=torch.float16)
        q4_codes = torch.as_tensor(entry["q4_codes_int8"], dtype=torch.int8)
        q4_scales = torch.as_tensor(entry["q4_scales_fp16"], dtype=torch.float16)
        q8_mask = (
            torch.as_tensor(entry["q8_mask"], dtype=torch.bool)
            if has_q8
            else torch.zeros_like(q4_mask)
        )
        q8_codes = (
            torch.as_tensor(entry["q8_codes_int8"], dtype=torch.int8)
            if has_q8
            else torch.zeros_like(q4_codes)
        )
        q8_scales = (
            torch.as_tensor(entry["q8_scales_fp16"], dtype=torch.float16)
            if has_q8
            else torch.ones_like(q4_scales)
        )
        nz4_mask = torch.as_tensor(
            entry.get("nz4_mask", torch.zeros_like(q4_mask)), dtype=torch.bool
        )
        if tuple(source_mask.shape) != scale_shape:
            raise ValueError(f"artifact source mask shape mismatch for {name}")
        if tuple(q4_mask.shape) != scale_shape:
            raise ValueError(f"artifact Q4 mask shape mismatch for {name}")
        if tuple(q8_mask.shape) != scale_shape:
            raise ValueError(f"artifact Q8 mask shape mismatch for {name}")
        if tuple(nz4_mask.shape) != scale_shape:
            raise ValueError(f"artifact NZ4 mask shape mismatch for {name}")
        if (
            tuple(q2_codes.shape) != code_shape
            or tuple(q4_codes.shape) != code_shape
            or tuple(q8_codes.shape) != code_shape
        ):
            raise ValueError(f"artifact code shape mismatch for {name}")
        if (
            tuple(q2_scales.shape) != scale_shape
            or tuple(q4_scales.shape) != scale_shape
            or tuple(q8_scales.shape) != scale_shape
        ):
            raise ValueError(f"artifact scale shape mismatch for {name}")
        q2_mask = ~(q4_mask | q8_mask)
        if torch.any(nz4_mask & ~q2_mask):
            raise ValueError(f"NZ4 mask overlaps a higher-precision format in {name}")
        if torch.any(nz4_mask & source_mask):
            raise ValueError(f"NZ4 mask overwrites an accepted ternary group in {name}")
        real_positions = (
            torch.arange(groups * group_size).view(1, groups, group_size) < columns
        )
        ternary_positions = (
            (q2_mask & ~nz4_mask).unsqueeze(-1) & real_positions
        )
        nz4_positions = nz4_mask.unsqueeze(-1) & real_positions
        if ternary_positions.any() and not set(
            q2_codes[ternary_positions].unique().tolist()
        ) <= {-1, 0, 1}:
            raise ValueError(f"non-ternary Q2 code in {name}")
        if nz4_positions.any() and not set(
            q2_codes[nz4_positions].unique().tolist()
        ) <= {-3, -1, 1, 3}:
            raise ValueError(f"invalid no-zero Q2 code in {name}")
        if not set(q4_codes.unique().tolist()) <= set(range(-8, 8)):
            raise ValueError(f"out-of-range signed Q4 code in {name}")
        if not set(q8_codes.unique().tolist()) <= set(range(-128, 128)):
            raise ValueError(f"out-of-range signed Q8 code in {name}")
        if (
            not torch.isfinite(q2_scales).all()
            or not torch.isfinite(q4_scales).all()
            or not torch.isfinite(q8_scales).all()
        ):
            raise ValueError(f"non-finite scale in {name}")
        if torch.any(q4_mask & q8_mask):
            raise ValueError(f"Q4 and Q8 masks overlap in {name}")
        if torch.any((q4_mask | q8_mask) & source_mask):
            raise ValueError(f"rescue overwrites an accepted ternary group in {name}")

        source_entry = source_entries.get(name)
        if source_entry is None:
            if source_mask.any():
                raise ValueError(f"new artifact matrix has source commitments: {name}")
            source_ternary_weights = 0
        else:
            if int(source_entry["group_size"]) != group_size:
                raise ValueError(f"source group size mismatch for {name}")
            checkpoint_mask = torch.as_tensor(
                source_entry["committed_mask"], dtype=torch.bool
            )
            if not torch.equal(source_mask, checkpoint_mask):
                raise ValueError(f"source committed mask mismatch for {name}")
            source_codes = _source_grouped_codes(
                source_entry, rows=rows, columns=columns, group_size=group_size
            )
            source_scales = torch.as_tensor(
                source_entry["scales_fp16"], dtype=torch.float16
            )
            if tuple(source_scales.shape) != scale_shape:
                raise ValueError(f"source scale shape mismatch for {name}")
            if not torch.equal(q2_codes[source_mask], source_codes[source_mask]):
                raise ValueError(f"accepted source codes changed in {name}")
            source_scales_changed = not torch.equal(
                q2_scales[source_mask], source_scales[source_mask]
            )
            entry_allows_scale_recovery = bool(
                entry.get("source_q2_scale_recovery", False)
            )
            if source_scales_changed and not (
                allow_source_q2_scale_recovery and entry_allows_scale_recovery
            ):
                raise ValueError(f"accepted source scales changed in {name}")
            source_ternary_weights = valid_group_weight_count(
                source_mask, columns, group_size
            )

        q2_value = q2_codes.float() * q2_scales.float().unsqueeze(-1)
        q4_value = q4_codes.float() * q4_scales.float().unsqueeze(-1)
        q8_value = q8_codes.float() * q8_scales.float().unsqueeze(-1)
        grouped = torch.where(q4_mask.unsqueeze(-1), q4_value, q2_value)
        grouped = torch.where(q8_mask.unsqueeze(-1), q8_value, grouped)
        weight = grouped.reshape(rows, -1)[:, :columns]
        if kind == "tied_embedding_head":
            _install_tied_embedding_head(
                model,
                embedding_name=name,
                head_name=tied_linear_name,
                weight=weight,
                device=device,
            )
        else:
            bias = getattr(target, "bias", None)
            compute_dtype = getattr(
                getattr(target, "matrix", None), "compute_dtype", weight.dtype
            )
            _set_submodule(
                model,
                name,
                FixedMixedQ2Q4Linear(weight.to(compute_dtype), bias).to(device),
            )

        new_ternary_mask = q2_mask & (~source_mask) & (~nz4_mask)
        matrix_new_ternary = valid_group_weight_count(
            new_ternary_mask, columns, group_size
        )
        matrix_nz4 = valid_group_weight_count(nz4_mask, columns, group_size)
        matrix_q4 = valid_group_weight_count(q4_mask, columns, group_size)
        matrix_q8 = valid_group_weight_count(q8_mask, columns, group_size)
        new_q2_weights += matrix_new_ternary + matrix_nz4
        nz4_weights += matrix_nz4
        q4_weights += matrix_q4
        q8_weights += matrix_q8
        matrix_statistics[name] = {
            "source_ternary_weights": source_ternary_weights,
            "new_ternary_weights": matrix_new_ternary,
            "nz4_weights": matrix_nz4,
            "q4_weights": matrix_q4,
            "q8_weights": matrix_q8,
        }

    norm_extras = artifact.get("norm_extras", {})
    if norm_extras:
        if not artifact.get("allow_norm_recovery", False):
            raise ValueError("mixed artifact norm recovery is not explicitly enabled")
        parameters = dict(model.named_parameters())
        for name, value in norm_extras.items():
            if "norm" not in name:
                raise ValueError(f"mixed artifact extra is not a norm parameter: {name}")
            parameter = parameters.get(name)
            tensor = torch.as_tensor(value)
            if parameter is None or tuple(parameter.shape) != tuple(tensor.shape):
                raise ValueError(f"mixed artifact norm shape mismatch for {name}")
            if not torch.isfinite(tensor).all():
                raise ValueError(f"non-finite mixed artifact norm for {name}")
            parameter.copy_(tensor.to(parameter.device, parameter.dtype))

    return MixedArtifactInstallResult(
        new_q2_weights=new_q2_weights,
        nz4_weights=nz4_weights,
        q4_weights=q4_weights,
        q8_weights=q8_weights,
        matrix_statistics=matrix_statistics,
    )
