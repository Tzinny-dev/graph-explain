from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class ExplanationAlgorithm(ABC):
    name = "base"
    graph_level = False

    @abstractmethod
    def explain(
        self,
        backend: Any,
        model: Any,
        data: Any,
        index: Any = None,
        target_class: int | None = None,
        **kwargs,
    ) -> Any: ...

    def validate(self, backend: Any, model: Any, data: Any) -> None:
        return None
