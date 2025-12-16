"""Dataset builders and shared data utilities.

This package intentionally lives alongside the on-disk dataset cache folders
(`data/cifar-10-batches-py/`, `data/stl10_binary/`, ...).

When the project root is on `sys.path`, importing `data.*` should resolve to
this package (not any third-party `data` module).
"""

from .registry import DATASET_REGISTRY, available_datasets, build_dataset_builder, register_dataset
from .base import DatasetConfig, DatasetLoaders, DatasetBuilder

# Import dataset modules for side-effect registration.
from . import vision as _vision  # noqa: F401
from . import text as _text  # noqa: F401

__all__ = [
    "DATASET_REGISTRY",
    "available_datasets",
    "register_dataset",
    "build_dataset_builder",
    "DatasetConfig",
    "DatasetLoaders",
    "DatasetBuilder",
]

