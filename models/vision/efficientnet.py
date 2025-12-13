"""EfficientNet-B0 wrapper."""

from __future__ import annotations

import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

from .. import ModelBase, register_model


@register_model("efficientnet_b0")
@register_model("efficientnet")
class EfficientNetB0Classifier(ModelBase):
    def __init__(self, num_labels: int = 10, pretrained: bool = False) -> None:
        super().__init__(
            name="efficientnet_b0",
            task_type="vision",
            num_labels=num_labels,
            description="EfficientNet-B0 with configurable classifier",
        )
        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        backbone = efficientnet_b0(weights=weights)
        in_features = backbone.classifier[1].in_features
        backbone.classifier[1] = nn.Linear(in_features, self.num_labels)
        self.backbone = backbone

    def forward_features(self, features):  # type: ignore[override]
        return self.backbone(features)
