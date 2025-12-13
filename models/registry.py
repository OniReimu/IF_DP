"""Model registry utilities."""

from __future__ import annotations

from typing import Any, Callable, Type

from core.registry import Registry

MODEL_REGISTRY = Registry("model")


def register_model(name: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator for registering model builders."""

    def decorator(builder: Callable[..., Any]) -> Callable[..., Any]:
        MODEL_REGISTRY.register(name)(builder)
        return builder

    return decorator


def build_model(name: str, **kwargs: Any):
    builder = MODEL_REGISTRY.get(name)
    if isinstance(builder, type):
        return builder(**kwargs)
    return builder(**kwargs)


def available_models():
    return MODEL_REGISTRY.keys()
