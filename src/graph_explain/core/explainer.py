from __future__ import annotations

from typing import Any

import torch

from ..backends.base import Backend, get_backend
from ..methods.base import ExplanationAlgorithm
from .explanation import Explanation


class Explainer:
    def __init__(
        self,
        algorithm: ExplanationAlgorithm,
        backend: Backend | str = "pyg",
        node_mask_type: str | None = "attributes",
        edge_mask_type: str | None = "object",
        mask_threshold: float = 0.5,
        explanation_type: str = "model",
        **kwargs,
    ):
        if isinstance(backend, str):
            backend = get_backend(backend)
        self.backend = backend
        self.algorithm = algorithm
        self.node_mask_type = node_mask_type
        self.edge_mask_type = edge_mask_type
        self.mask_threshold = mask_threshold
        self.explanation_type = explanation_type
        self._extra = kwargs

    def explain(
        self,
        data: Any,
        model: Any,
        index: int | list[int] | torch.Tensor | None = None,
        target_class: int | None = None,
        **kwargs,
    ) -> Explanation:
        explanation = self.algorithm.explain(
            backend=self.backend,
            model=model,
            data=data,
            index=index,
            target_class=target_class,
            node_mask_type=self.node_mask_type,
            edge_mask_type=self.edge_mask_type,
            **kwargs,
        )
        explanation.node_idx = index
        explanation.mask_threshold = self.mask_threshold
        explanation.metadata.setdefault("backend", self.backend)
        explanation.metadata.setdefault("backing_data", data)
        return explanation

    def explain_node(
        self,
        data: Any,
        model: Any,
        node_idx: int,
        target_class: int | None = None,
        **kwargs,
    ) -> Explanation:
        return self.explain(
            data, model, index=node_idx, target_class=target_class, **kwargs
        )

    def explain_graph(
        self,
        data: Any,
        model: Any,
        target_class: int | None = None,
        **kwargs,
    ) -> Explanation:
        return self.explain(
            data, model, index=None, target_class=target_class, **kwargs
        )

    def __call__(
        self, data: Any, model: Any, index: int | list[int] | None = None, **kwargs
    ):
        return self.explain(data, model, index=index, **kwargs)
