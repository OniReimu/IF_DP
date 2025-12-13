"""Legacy model aliases backed by the unified registry."""

from typing import Any

from .registry import available_models, build_model
from .vision.cnn import CNNClassifier as CNN
from .vision.resnet import ResNet18Classifier as ResNet18
from .vision.efficientnet import EfficientNetB0Classifier as EfficientNetB0
from .vision.vit import VisionTransformerClassifier as VisionTransformer

__all__ = [
    "CNN",
    "ResNet18",
    "EfficientNetB0",
    "VisionTransformer",
    "create_model",
    "available_models",
]


def create_model(model_type: str = "cnn", **kwargs: Any):
    """
    Backwards-compatible factory that now proxies the global model registry.
    """

    return build_model(model_type, **kwargs)
