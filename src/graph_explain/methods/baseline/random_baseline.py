from __future__ import annotations

from typing import Any

import torch

from ...core.explanation import Explanation
from ...core.registry import register
from ..base import ExplanationAlgorithm


@register("random", "random_baseline", "rand")
class RandomBaseline(ExplanationAlgorithm):
    """Random: línea base de importancia uniforme aleatoria (seed-able).

    Asigna importancias aleatorias en [0, 1] a nodos, aristas y features sin
    ningún vínculo con el modelo; útil como referencia de escenario nulo en
    benchmarks comparativos.
    """

    def __init__(self, seed: int | None = 0, **kwargs):
        self.seed = seed

    def explain(
        self,
        backend: Any,
        model: Any,
        data: Any,
        index: int | torch.Tensor | None = None,
        target_class: int | None = None,
        **kwargs,
    ) -> Explanation:
        x = backend.node_features(data)
        edge_index = backend.edge_index(data)
        num_nodes = int(x.size(0))
        num_edges = int(edge_index.size(1))

        if self.seed is not None:
            torch.manual_seed(self.seed)

        node_importance = torch.rand(num_nodes).cpu()
        edge_importance = torch.rand(num_edges).cpu()
        feature_importance = torch.rand_like(x).cpu()

        nodes = self._to_node_ids(index, num_nodes)
        root = int(nodes[0])
        with torch.no_grad():
            logits = backend.forward(model, x, edge_index)
        target = target_class
        if logits.dim() == 2:
            if target is None:
                target = int(logits[root].argmax().item())
            target = max(0, min(int(target), logits.size(1) - 1))
            pred = logits[root].detach().cpu()
        else:
            target = None
            pred = logits.detach().cpu()

        return Explanation(
            node_importance=node_importance,
            edge_importance=edge_importance,
            feature_importance=feature_importance,
            prediction_original=pred,
            prediction_explanation=None,
            node_idx=root,
            target_class=target,
        )

    @staticmethod
    def _to_node_ids(index, num_nodes: int) -> torch.Tensor:
        if index is None:
            return torch.zeros(1, dtype=torch.long)
        if isinstance(index, int):
            return torch.tensor([index], dtype=torch.long)
        idx = torch.as_tensor(index, dtype=torch.long)
        return idx.reshape(-1) if idx.numel() else torch.zeros(1, dtype=torch.long)
