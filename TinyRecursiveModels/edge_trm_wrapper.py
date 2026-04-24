
"""Thin adapters for reusing the official TRM implementation in notebook workflows.

This module intentionally avoids re-implementing recursion logic. It wraps:
- models.recursive_reasoning.trm.TinyRecursiveReasoningModel_ACTV1
- models.losses.ACTLossHead
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import torch

from models.losses import ACTLossHead
from models.recursive_reasoning.trm import TinyRecursiveReasoningModel_ACTV1



import inspect


# --- Hotfix for PyTorch optimizer version mismatch error ---
if hasattr(torch, '_dynamo') and hasattr(torch._dynamo, 'disable'):
    _disable_sig = inspect.signature(torch._dynamo.disable)
    if 'wrapping' not in _disable_sig.parameters:
        _orig_disable = torch._dynamo.disable
        def _patched_disable(*args, **kwargs):
            kwargs.pop('wrapping', None)
            return _orig_disable(*args, **kwargs)
        torch._dynamo.disable = _patched_disable
        if hasattr(torch, '_compile') and hasattr(torch._compile, 'disable'):
            torch._compile.disable = _patched_disable
# ---------------------------------------------------------


@dataclass
class EdgeTRMBatch:
    """Canonical TRM batch contract."""

    inputs: torch.Tensor
    labels: torch.Tensor
    puzzle_identifiers: torch.Tensor

    def as_dict(self) -> Dict[str, torch.Tensor]:
        return {
            "inputs": self.inputs.to(torch.int32),
            "labels": self.labels.to(torch.int32),
            "puzzle_identifiers": self.puzzle_identifiers.to(torch.int32),
        }


class EdgeTRMAdapter(torch.nn.Module):
    """Notebook-friendly adapter around official TRM + ACT loss head."""

    def __init__(self, model_cfg: dict, loss_type: str = "stablemax_cross_entropy") -> None:
        super().__init__()
        self.vocab_size = int(model_cfg["vocab_size"])
        core = TinyRecursiveReasoningModel_ACTV1(model_cfg)
        self.model = ACTLossHead(core, loss_type=loss_type)

    def _sanitize_batch(self, batch_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Guard against token/index issues that otherwise crash CUDA kernels.

        - inputs: clamp to [0, vocab_size-1]
        - labels: convert out-of-range values to ignore index (-100)
        """
        x = batch_dict["inputs"]
        y = batch_dict["labels"]

        x = torch.clamp(x, min=0, max=self.vocab_size - 1)
        valid_y = (y >= 0) & (y < self.vocab_size)
        y = torch.where(valid_y | (y == -100), y, torch.full_like(y, -100))

        batch_dict["inputs"] = x
        batch_dict["labels"] = y
        return batch_dict

    def initial_carry(self, batch: EdgeTRMBatch):
        return self.model.initial_carry(batch=batch.as_dict())

    def train_step(
        self,
        batch: EdgeTRMBatch,
        carry,
        return_keys: Optional[Sequence[str]] = None,
    ) -> Tuple[object, torch.Tensor, Dict[str, torch.Tensor], Dict[str, torch.Tensor], torch.Tensor]:
        """Run one official TRM step.

        Returns `(new_carry, total_loss, metrics, detached_outputs, done)`.
        """
        keys = return_keys or ("logits", "preds", "q_halt_logits", "q_continue_logits")
        batch_dict = self._sanitize_batch(batch.as_dict())

        new_carry, loss, metrics, detached_outputs, done = self.model(
            return_keys=keys,
            carry=carry,
            batch=batch_dict,
        )

        return new_carry, loss, metrics, detached_outputs, done
