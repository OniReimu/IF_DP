"""Vision Transformer support."""

from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vit_b_16, ViT_B_16_Weights

from .. import ModelBase, register_model


@register_model("vit_b16")
@register_model("vit")
class VisionTransformerClassifier(ModelBase):
    def __init__(self, num_labels: int = 10, pretrained: bool = False) -> None:
        super().__init__(
            name="vit_b16",
            task_type="vision",
            num_labels=num_labels,
            description="Vision Transformer (ViT-B/16) classifier",
        )
        weights = ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1 if pretrained else None
        backbone = vit_b_16(weights=weights)
        backbone.heads.head = nn.Linear(backbone.heads.head.in_features, self.num_labels)
        self.backbone = backbone
        self.preferred_image_size = getattr(backbone, "image_size", 224)

    def forward_features(self, features):  # type: ignore[override]
        if features.dim() == 4:
            _, _, h, w = features.shape
            if h != self.preferred_image_size or w != self.preferred_image_size:
                features = F.interpolate(
                    features,
                    size=(self.preferred_image_size, self.preferred_image_size),
                    mode="bilinear",
                    align_corners=False,
                )
        return self.backbone(features)
