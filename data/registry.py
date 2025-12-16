"""Dataset registry utilities (mirrors `models/registry.py`)."""

from __future__ import annotations

from typing import Any, Callable

from core.registry import Registry

DATASET_REGISTRY = Registry("dataset")


def register_dataset(name: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator for registering dataset builders."""

    def decorator(builder: Callable[..., Any]) -> Callable[..., Any]:
        DATASET_REGISTRY.register(name)(builder)
        return builder

    return decorator


def build_dataset_builder(name: str, **kwargs: Any):
    """Instantiate a dataset builder by name."""

    builder = DATASET_REGISTRY.get(name)
    if isinstance(builder, type):
        return builder(**kwargs)
    return builder(**kwargs)


def available_datasets():
    return DATASET_REGISTRY.keys()


