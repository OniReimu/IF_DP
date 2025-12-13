"""Utility helpers shared across model implementations."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from .base import ModelBase


def compute_loss(model: torch.nn.Module, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Use model-specific loss when available, otherwise fall back to cross entropy."""

    if isinstance(model, ModelBase):
        return model.compute_loss(logits, labels)
    return F.cross_entropy(logits, labels)


def forward_features(model: torch.nn.Module, features: Any) -> torch.Tensor:
    """Dispatch helper that keeps legacy models compatible with dict-based inputs."""

    if isinstance(model, ModelBase):
        return model.forward_features(features)
    # Legacy torchvision-style modules expect a tensor; forward directly.
    return model(features)
