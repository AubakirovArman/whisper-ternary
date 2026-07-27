"""Hooks for collecting diagonal causal moments."""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn


class InputSecondMomentCollector:
    """Forward-only diagonal activation moment collector for any linear layer."""

    def __init__(self, module: nn.Module):
        self.input_sum: Optional[torch.Tensor] = None
        self.input_count = 0
        self._pre = module.register_forward_pre_hook(self._capture_input)

    def _capture_input(self, _module, args) -> None:
        if not args:
            raise RuntimeError("target module received no positional input")
        value = args[0].detach().float().reshape(-1, args[0].shape[-1])
        current = value.square().sum(0)
        self.input_sum = current if self.input_sum is None else self.input_sum + current
        self.input_count += value.shape[0]

    def moment(self) -> torch.Tensor:
        if self.input_sum is None or self.input_count == 0:
            raise RuntimeError("no forward samples were collected")
        return self.input_sum / self.input_count

    def close(self) -> None:
        self._pre.remove()

    def __enter__(self):
        return self

    def __exit__(self, *_exc) -> None:
        self.close()


class CausalMomentCollector:
    def __init__(self, module: nn.Module):
        self.input_sum: Optional[torch.Tensor] = None
        self.output_grad_sum: Optional[torch.Tensor] = None
        self.input_count = 0
        self.output_count = 0
        self._pre = module.register_forward_pre_hook(self._capture_input)
        self._post = module.register_forward_hook(self._capture_output)

    def _capture_input(self, _module, args) -> None:
        value = args[0].detach().float().reshape(-1, args[0].shape[-1])
        current = value.square().sum(0)
        self.input_sum = current if self.input_sum is None else self.input_sum + current
        self.input_count += value.shape[0]

    def _capture_output(self, _module, _args, output) -> None:
        if not output.requires_grad:
            raise RuntimeError("target output has no gradient path")

        def capture(gradient: torch.Tensor) -> torch.Tensor:
            value = gradient.detach().float().reshape(-1, gradient.shape[-1])
            current = value.square().sum(0)
            self.output_grad_sum = current if self.output_grad_sum is None else self.output_grad_sum + current
            self.output_count += value.shape[0]
            return gradient

        output.register_hook(capture)

    def moments(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.input_sum is None or self.output_grad_sum is None:
            raise RuntimeError("no complete forward/backward samples were collected")
        return (
            self.input_sum / max(self.input_count, 1),
            self.output_grad_sum / max(self.output_count, 1),
        )

    def close(self) -> None:
        self._pre.remove()
        self._post.remove()

    def __enter__(self):
        return self

    def __exit__(self, *_exc) -> None:
        self.close()
