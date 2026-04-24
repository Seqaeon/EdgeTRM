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
        core = TinyRecursiveReasoningModel_ACTV1(model_cfg)
        self.model = ACTLossHead(core, loss_type=loss_type)

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

        new_carry, loss, metrics, detached_outputs, done = self.model(
            return_keys=keys,
            carry=carry,
            batch=batch.as_dict(),
        )

        return new_carry, loss, metrics, detached_outputs, done
