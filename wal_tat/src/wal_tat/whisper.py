"""Hugging Face Whisper architecture adapter for WAL-TAT campaigns."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import torch
import torch.nn as nn

from .transaction import TransactionalTernaryLinear


@dataclass(frozen=True)
class WhisperLinearSpec:
    name: str
    stack: str
    block: int
    category: str


@dataclass(frozen=True)
class WhisperTransactionGroup:
    name: str
    stack: str
    block: int
    category: str
    module_names: tuple[str, ...]


def _resolve_parent(root: nn.Module, dotted_name: str) -> tuple[nn.Module, str]:
    parts = dotted_name.split(".")
    parent = root
    for part in parts[:-1]:
        if part.isdigit():
            parent = parent[int(part)]  # type: ignore[index]
        else:
            parent = getattr(parent, part)
    return parent, parts[-1]


def get_module(root: nn.Module, dotted_name: str) -> nn.Module:
    parent, leaf = _resolve_parent(root, dotted_name)
    return getattr(parent, leaf)


def set_module(root: nn.Module, dotted_name: str, module: nn.Module) -> None:
    parent, leaf = _resolve_parent(root, dotted_name)
    setattr(parent, leaf, module)


class WhisperFamilySpec:
    """Enumerate structural low-bit transactions in an HF Whisper model."""

    def __init__(self, encoder_layers: int, decoder_layers: int):
        if encoder_layers < 1 or decoder_layers < 1:
            raise ValueError("Whisper must have positive encoder and decoder depths")
        self.encoder_layers = int(encoder_layers)
        self.decoder_layers = int(decoder_layers)

    @classmethod
    def from_model(cls, model: nn.Module) -> "WhisperFamilySpec":
        try:
            encoder_layers = len(model.model.encoder.layers)  # type: ignore[attr-defined]
            decoder_layers = len(model.model.decoder.layers)  # type: ignore[attr-defined]
        except (AttributeError, TypeError) as error:
            raise TypeError("model does not expose the Hugging Face Whisper layout") from error
        result = cls(encoder_layers, decoder_layers)
        result.validate(model)
        return result

    def linear_specs(self) -> tuple[WhisperLinearSpec, ...]:
        values: list[WhisperLinearSpec] = []
        for block in range(self.encoder_layers):
            prefix = f"model.encoder.layers.{block}"
            values.extend(
                WhisperLinearSpec(f"{prefix}.{suffix}", "encoder", block, category)
                for category, suffix in (
                    ("self_q", "self_attn.q_proj"),
                    ("self_k", "self_attn.k_proj"),
                    ("self_v", "self_attn.v_proj"),
                    ("self_o", "self_attn.out_proj"),
                    ("mlp_in", "fc1"),
                    ("mlp_out", "fc2"),
                )
            )
        for block in range(self.decoder_layers):
            prefix = f"model.decoder.layers.{block}"
            values.extend(
                WhisperLinearSpec(f"{prefix}.{suffix}", "decoder", block, category)
                for category, suffix in (
                    ("self_q", "self_attn.q_proj"),
                    ("self_k", "self_attn.k_proj"),
                    ("self_v", "self_attn.v_proj"),
                    ("self_o", "self_attn.out_proj"),
                    ("cross_q", "encoder_attn.q_proj"),
                    ("cross_k", "encoder_attn.k_proj"),
                    ("cross_v", "encoder_attn.v_proj"),
                    ("cross_o", "encoder_attn.out_proj"),
                    ("mlp_in", "fc1"),
                    ("mlp_out", "fc2"),
                )
            )
        return tuple(values)

    def transaction_groups(self) -> tuple[WhisperTransactionGroup, ...]:
        values: list[WhisperTransactionGroup] = []
        for block in range(self.encoder_layers):
            prefix = f"model.encoder.layers.{block}"
            values.extend(
                (
                    WhisperTransactionGroup(
                        f"encoder.{block}.mlp",
                        "encoder",
                        block,
                        "mlp",
                        (f"{prefix}.fc1", f"{prefix}.fc2"),
                    ),
                    WhisperTransactionGroup(
                        f"encoder.{block}.mlp_in",
                        "encoder",
                        block,
                        "mlp_in",
                        (f"{prefix}.fc1",),
                    ),
                    WhisperTransactionGroup(
                        f"encoder.{block}.mlp_out",
                        "encoder",
                        block,
                        "mlp_out",
                        (f"{prefix}.fc2",),
                    ),
                    WhisperTransactionGroup(
                        f"encoder.{block}.self_vo",
                        "encoder",
                        block,
                        "self_vo",
                        (
                            f"{prefix}.self_attn.v_proj",
                            f"{prefix}.self_attn.out_proj",
                        ),
                    ),
                    WhisperTransactionGroup(
                        f"encoder.{block}.self_v",
                        "encoder",
                        block,
                        "self_v",
                        (f"{prefix}.self_attn.v_proj",),
                    ),
                    WhisperTransactionGroup(
                        f"encoder.{block}.self_o",
                        "encoder",
                        block,
                        "self_o",
                        (f"{prefix}.self_attn.out_proj",),
                    ),
                    WhisperTransactionGroup(
                        f"encoder.{block}.self_qk",
                        "encoder",
                        block,
                        "self_qk",
                        (
                            f"{prefix}.self_attn.q_proj",
                            f"{prefix}.self_attn.k_proj",
                        ),
                    ),
                    WhisperTransactionGroup(
                        f"encoder.{block}.self_q",
                        "encoder",
                        block,
                        "self_q",
                        (f"{prefix}.self_attn.q_proj",),
                    ),
                    WhisperTransactionGroup(
                        f"encoder.{block}.self_k",
                        "encoder",
                        block,
                        "self_k",
                        (f"{prefix}.self_attn.k_proj",),
                    ),
                )
            )
        for block in range(self.decoder_layers):
            prefix = f"model.decoder.layers.{block}"
            values.extend(
                (
                    WhisperTransactionGroup(
                        f"decoder.{block}.mlp",
                        "decoder",
                        block,
                        "mlp",
                        (f"{prefix}.fc1", f"{prefix}.fc2"),
                    ),
                    WhisperTransactionGroup(
                        f"decoder.{block}.mlp_in",
                        "decoder",
                        block,
                        "mlp_in",
                        (f"{prefix}.fc1",),
                    ),
                    WhisperTransactionGroup(
                        f"decoder.{block}.mlp_out",
                        "decoder",
                        block,
                        "mlp_out",
                        (f"{prefix}.fc2",),
                    ),
                    WhisperTransactionGroup(
                        f"decoder.{block}.self_vo",
                        "decoder",
                        block,
                        "self_vo",
                        (
                            f"{prefix}.self_attn.v_proj",
                            f"{prefix}.self_attn.out_proj",
                        ),
                    ),
                    WhisperTransactionGroup(
                        f"decoder.{block}.self_v",
                        "decoder",
                        block,
                        "self_v",
                        (f"{prefix}.self_attn.v_proj",),
                    ),
                    WhisperTransactionGroup(
                        f"decoder.{block}.self_o",
                        "decoder",
                        block,
                        "self_o",
                        (f"{prefix}.self_attn.out_proj",),
                    ),
                    WhisperTransactionGroup(
                        f"decoder.{block}.self_qk",
                        "decoder",
                        block,
                        "self_qk",
                        (
                            f"{prefix}.self_attn.q_proj",
                            f"{prefix}.self_attn.k_proj",
                        ),
                    ),
                    WhisperTransactionGroup(
                        f"decoder.{block}.self_q",
                        "decoder",
                        block,
                        "self_q",
                        (f"{prefix}.self_attn.q_proj",),
                    ),
                    WhisperTransactionGroup(
                        f"decoder.{block}.self_k",
                        "decoder",
                        block,
                        "self_k",
                        (f"{prefix}.self_attn.k_proj",),
                    ),
                    WhisperTransactionGroup(
                        f"decoder.{block}.cross_vo",
                        "decoder",
                        block,
                        "cross_vo",
                        (
                            f"{prefix}.encoder_attn.v_proj",
                            f"{prefix}.encoder_attn.out_proj",
                        ),
                    ),
                    WhisperTransactionGroup(
                        f"decoder.{block}.cross_v",
                        "decoder",
                        block,
                        "cross_v",
                        (f"{prefix}.encoder_attn.v_proj",),
                    ),
                    WhisperTransactionGroup(
                        f"decoder.{block}.cross_o",
                        "decoder",
                        block,
                        "cross_o",
                        (f"{prefix}.encoder_attn.out_proj",),
                    ),
                    WhisperTransactionGroup(
                        f"decoder.{block}.cross_qk",
                        "decoder",
                        block,
                        "cross_qk",
                        (
                            f"{prefix}.encoder_attn.q_proj",
                            f"{prefix}.encoder_attn.k_proj",
                        ),
                    ),
                    WhisperTransactionGroup(
                        f"decoder.{block}.cross_q",
                        "decoder",
                        block,
                        "cross_q",
                        (f"{prefix}.encoder_attn.q_proj",),
                    ),
                    WhisperTransactionGroup(
                        f"decoder.{block}.cross_k",
                        "decoder",
                        block,
                        "cross_k",
                        (f"{prefix}.encoder_attn.k_proj",),
                    ),
                )
            )
        return tuple(values)

    def validate(self, model: nn.Module) -> None:
        for spec in self.linear_specs():
            module = get_module(model, spec.name)
            if not isinstance(module, nn.Linear):
                raise TypeError(f"{spec.name} is not an nn.Linear")
            if module.weight.ndim != 2:
                raise ValueError(f"{spec.name} does not have a matrix weight")

    def group(self, name: str) -> WhisperTransactionGroup:
        for group in self.transaction_groups():
            if group.name == name:
                return group
        raise KeyError(name)


def install_transactional_group(
    model: nn.Module,
    group: WhisperTransactionGroup,
    *,
    group_size: int = 128,
) -> Mapping[str, TransactionalTernaryLinear]:
    """Replace one structural group with WAL transactional linear wrappers."""
    installed: dict[str, TransactionalTernaryLinear] = {}
    for name in group.module_names:
        module = get_module(model, name)
        if not isinstance(module, nn.Linear):
            raise TypeError(f"{name} is not an nn.Linear")
        wrapper = TransactionalTernaryLinear.from_linear(
            module, group_size=group_size
        ).to(device=module.weight.device)
        set_module(model, name, wrapper)
        installed[name] = wrapper
    return installed


def freeze_except_transactional(
    model: nn.Module,
    installed: Iterable[TransactionalTernaryLinear],
) -> None:
    """Freeze the model, then enable only selected proxy master/scales."""
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for linear in installed:
        linear.matrix.master_weight.requires_grad_(True)
        linear.matrix.group_scale.requires_grad_(True)
