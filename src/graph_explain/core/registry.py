from __future__ import annotations

from collections.abc import Callable
from typing import Any

_ALGORITHMS: dict[str, type] = {}
_ALIASES: dict[str, str] = {}


def register(name: str, *aliases: str) -> Callable[[type], type]:
    def decorator(cls: type) -> type:
        _ALGORITHMS[name] = cls
        for alias in aliases:
            _ALIASES[alias] = name
        cls.name = name
        return cls

    return decorator


def get_algorithm(name: str) -> type:
    registered = _ALGORITHMS.get(name) or _ALGORITHMS.get(_ALIASES.get(name, ""))
    if registered is None:
        from ..methods import _available_methods

        raise ValueError(
            f"Algoritmo desconocido: {name}. Disponibles: {sorted(_available_methods())}"
        )
    return registered


def instantiate(name: str, **kwargs) -> Any:
    cls = get_algorithm(name)
    return cls(**kwargs)


def _available_methods() -> set[str]:
    return set(_ALGORITHMS.keys()) | set(_ALIASES.keys())