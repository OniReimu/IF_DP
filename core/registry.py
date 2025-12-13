"""Generic lightweight registry helpers used across IF-DP modules."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Tuple, TypeVar


T = TypeVar("T")


@dataclass(frozen=True)
class RegistryItem:
    """Metadata describing a registered builder."""

    key: str
    builder: Callable[..., Any]


class Registry:
    """Simple string -> callable registry with helpful error messaging."""

    def __init__(self, name: str) -> None:
        self._name = name
        self._items: Dict[str, Callable[..., Any]] = {}

    def register(self, key: str) -> Callable[[Callable[..., T]], Callable[..., T]]:
        """Decorator used to register a callable or class constructor."""

        normalized = key.lower()

        def decorator(builder: Callable[..., T]) -> Callable[..., T]:
            if normalized in self._items:
                raise ValueError(
                    f"Duplicate registration for '{normalized}' in registry '{self._name}'"
                )
            self._items[normalized] = builder
            return builder

        return decorator

    def get(self, key: str) -> Callable[..., Any]:
        """Return the registered callable for `key`."""

        normalized = key.lower()
        if normalized not in self._items:
            available = ", ".join(sorted(self._items)) or "<empty>"
            raise KeyError(
                f"Unknown {self._name} '{key}'. Available options: {available}"
            )
        return self._items[normalized]

    def contains(self, key: str) -> bool:
        return key.lower() in self._items

    def items(self) -> Iterable[RegistryItem]:
        for key in sorted(self._items):
            yield RegistryItem(key=key, builder=self._items[key])

    def keys(self) -> List[str]:
        return sorted(self._items)

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self) -> Iterator[RegistryItem]:
        return iter(self.items())
