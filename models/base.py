"""Base abstractions for IF-DP models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelMetadata:
    """Lightweight description exposed through the model registry."""

    name: str
    task_type: str
    num_labels: int
    description: str


class ModelBase(nn.Module):
    """Shared behavior for all models used in this project."""

    def __init__(
        self,
        name: str,
        task_type: str,
        num_labels: int,
        description: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.metadata = ModelMetadata(
            name=name,
            task_type=task_type,
            num_labels=num_labels,
            description=description or name,
        )

    def forward(self, features: Any) -> torch.Tensor:  # type: ignore[override]
        """PyTorch entry point delegates to forward_features for clarity."""

        return self.forward_features(features)

    def forward_features(self, features: Any) -> torch.Tensor:
        """Sub-classes implement their task-specific forward pass."""

        raise NotImplementedError

    def compute_loss(
        self, logits: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        """Default single-label cross-entropy."""

        return F.cross_entropy(logits, labels)

    @property
    def num_labels(self) -> int:
        return self.metadata.num_labels

    @property
    def task_type(self) -> str:
        return self.metadata.task_type

    def extra_state(self) -> Dict[str, Any]:
        """Allow derived classes to expose lightweight metadata."""

        return {}
