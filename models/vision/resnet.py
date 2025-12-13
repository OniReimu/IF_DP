"""ResNet-18 wrapper aligned with the unified model interface."""

from __future__ import annotations

import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18

from .. import ModelBase, register_model


@register_model("resnet18")
@register_model("resnet")
class ResNet18Classifier(ModelBase):
    def __init__(self, num_labels: int = 10, pretrained: bool = False) -> None:
        super().__init__(
            name="resnet18",
            task_type="vision",
            num_labels=num_labels,
            description="Torchvision ResNet-18 tailored for 32x32 inputs",
        )
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        backbone = resnet18(weights=weights)
        backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        backbone.maxpool = nn.Identity()
        backbone.fc = nn.Linear(backbone.fc.in_features, self.num_labels)
        self.backbone = backbone

    def forward_features(self, features):  # type: ignore[override]
        return self.backbone(features)
