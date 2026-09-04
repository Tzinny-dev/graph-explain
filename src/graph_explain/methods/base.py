from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch


class ExplanationAlgorithm(ABC):
    name = "base"

    @abstractmethod
    def explain(
        self,
        backend: Any,
        model: Any,
        data: Any,
        index: int | list[int] | torch.Tensor,
        target_class: int | None = None,
        **kwargs,
    ) -> Any:
        ...

    def validate(self, backend: Any, model: Any, data: Any) -> None:
        return None