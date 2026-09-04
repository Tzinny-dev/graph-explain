from __future__ import annotations

from typing import Any

import torch

from ...core.explanation import Explanation
from ...core.registry import register
from ..base import ExplanationAlgorithm


def _is_gat(module) -> bool:
    try:
        from torch_geometric.nn import GATConv

        return isinstance(module, GATConv)
    except ImportError:
        return False


@register("attention", "gat", "attention_explainer")
class AttentionExplainer(ExplanationAlgorithm):
    """Explicación basada en los pesos de atención de modelos GAT.

    Captura los coeficientes de atención (pre-softmax) de cada `GATConv`
    durante un único forward y los normaliza con softmax por vecino. La
    importancia de arista es la media de los coeficientes a través de las
    cabezas de atención y de las capas GAT; la importancia de nodo agrega la
    atención de las aristas incidentes.
    """

    def __init__(
        self,
        head_aggregate: str = "mean",
        layer_aggregate: str = "mean",
        node_aggregate: str = "sum",
    ):
        self.head_aggregate = head_aggregate
        self.layer_aggregate = layer_aggregate
        self.node_aggregate = node_aggregate

    def explain(
        self,
        backend: Any,
        model: Any,
        data: Any,
        index: int | list[int] | torch.Tensor | None = None,
        target_class: int | None = None,
        **kwargs,
    ) -> Explanation:
        model.eval()
        x = backend.node_features(data)
        edge_index = backend.edge_index(data)
        num_nodes = int(x.size(0))
        convs = [m for m in model.modules() if _is_gat(m)]
        if not convs:
            raise ValueError(
                "AttentionExplainer requiere un modelo con capas GATConv "
                "(torch_geometric.nn.GATConv)."
            )

        nodes = self._to_node_ids(index, num_nodes)
        captured = self._capture_attention(model, backend, x, edge_index)
        logits = captured["logits"]
        # GATConv normaliza sobre las aristas + self-loops (añadidos al final)
        alphas = [a[: edge_index.size(1)] for a in captured["alphas"]]

        if logits.dim() != 2:
            raise ValueError(
                "AttentionExplainer requiere predicciones node-level (logits (N, C))."
            )
        target = target_class
        if target is None:
            target = int(logits[nodes[0]].argmax().item())
        target_cls = max(0, min(int(target), logits.size(1) - 1))

        per_layer: list[torch.Tensor] = []
        for layer_idx, alpha in enumerate(alphas):
            att = torch.softmax(alpha, dim=0)  # (E, heads) normalizada por vecino
            if self.head_aggregate == "mean":
                att = att.mean(dim=1)
            elif self.head_aggregate == "max":
                att = att.max(dim=1).values
            elif self.head_aggregate == "sum":
                att = att.sum(dim=1)
            else:
                raise ValueError(f"head_aggregate desconocido: {self.head_aggregate}")
            weight = 1.0 / len(convs) if self.layer_aggregate == "mean" else 1.0
            per_layer.append(weight * att)

        edge_importance = torch.stack(per_layer).sum(dim=0).clamp(min=0.0)

        node_importance = torch.zeros(num_nodes, device=x.device)
        if self.node_aggregate == "sum":
            node_importance = node_importance.index_add(
                0, edge_index[0], edge_importance
            )
            node_importance = node_importance.index_add(
                0, edge_index[1], edge_importance
            )
        elif self.node_aggregate == "in":
            node_importance = node_importance.index_add(
                0, edge_index[1], edge_importance
            )
        elif self.node_aggregate == "out":
            node_importance = node_importance.index_add(
                0, edge_index[0], edge_importance
            )
        else:
            raise ValueError(f"node_aggregate desconocido: {self.node_aggregate}")

        return Explanation(
            node_importance=node_importance.detach().cpu(),
            edge_importance=edge_importance.detach().cpu(),
            feature_importance=None,
            prediction_original=logits[nodes[0]].detach().cpu(),
            prediction_explanation=None,
            node_idx=int(nodes[0]) if nodes.shape[0] == 1 else index,
            target_class=target_cls,
        )

    @staticmethod
    def _to_node_ids(index, num_nodes: int) -> torch.Tensor:
        if index is None:
            return torch.zeros(1, dtype=torch.long)
        if isinstance(index, int):
            return torch.tensor([index], dtype=torch.long)
        idx = torch.as_tensor(index, dtype=torch.long)
        return idx.reshape(-1) if idx.numel() else torch.zeros(1, dtype=torch.long)

    def _capture_attention(self, model, backend, x, edge_index):
        from torch_geometric import nn as pyg_nn

        captured: list[torch.Tensor] = []
        real = pyg_nn.conv.gat_conv.softmax

        def _wrapped(alpha, index, ptr=None, num_nodes=None):
            captured.append(alpha.detach())
            return real(alpha, index, ptr, num_nodes)

        pyg_nn.conv.gat_conv.softmax = _wrapped
        try:
            with torch.no_grad():
                logits = backend.forward(model, x, edge_index)
        finally:
            pyg_nn.conv.gat_conv.softmax = real
        # un solo softmax por capa GAT en el forward
        return {"logits": logits, "alphas": captured}
