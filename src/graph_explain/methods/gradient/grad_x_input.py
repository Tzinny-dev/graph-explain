from __future__ import annotations

from typing import Any

import torch

from ...core.explanation import Explanation
from ...core.registry import register
from ..base import ExplanationAlgorithm


@register("grad_x_input", "gradient_x_input", "gx")
class GradXInput(ExplanationAlgorithm):
    """Gradient x Input: attribution as gradient scaled by the activation.

    The importance of each feature (and of each edge, if the backend supports
    edge weights) is the gradient of the target-class logit multiplied by the
    input-baseline difference (zero baseline by default). Node importance is the
    sum of |grad × Δx| over features.
    """

    graph_level = True

    def __init__(
        self,
        baseline: str = "zero",
        edge_grads: bool = True,
        node_mask_type: str | None = "attributes",
        **kwargs,
    ):
        self.baseline_name = baseline
        self.edge_grads = edge_grads
        self.node_mask_type = node_mask_type

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

        nodes = self._to_node_ids(index, num_nodes)
        with torch.no_grad():
            logits = backend.forward(model, x, edge_index)
        if logits.dim() != 2:
            raise ValueError(
                "GradXInput requiere predicciones node-level (logits (N, C))."
            )
        target = target_class
        if target is None:
            target = int(logits[nodes[0]].argmax().item())
        target_cls = max(0, min(int(target), logits.size(1) - 1))

        x_in = x.detach().clone().requires_grad_(True)
        ew = None
        compute_edge = self.edge_grads and backend.supports_edge_weight(model)
        if compute_edge:
            ew = torch.ones(
                edge_index.size(1), dtype=torch.float32, device=x.device
            ).requires_grad_(True)

        out = backend.forward(model, x_in, edge_index, edge_weight=ew)
        if out.dim() != 2:
            raise ValueError("GradXInput requiere predicciones node-level.")
        score = out[nodes, target_cls].sum()
        model.zero_grad()
        score.backward()

        grad_x = x_in.grad.detach()
        baseline = torch.zeros_like(x_in)
        contrib = grad_x * (x_in - baseline)

        node_importance = contrib.abs().sum(dim=-1)
        feature_importance = contrib.detach()

        edge_importance = None
        if compute_edge and ew is not None and ew.grad is not None:
            edge_importance = ew.grad.detach().abs()

        return Explanation(
            node_importance=node_importance.cpu(),
            edge_importance=(
                edge_importance.cpu() if edge_importance is not None else None
            ),
            feature_importance=(
                feature_importance.cpu()
                if self.node_mask_type == "attributes"
                else None
            ),
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
