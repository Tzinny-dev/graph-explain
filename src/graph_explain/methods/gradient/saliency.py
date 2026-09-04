from __future__ import annotations

from typing import Any

import torch

from ...core.explanation import Explanation
from ...core.registry import register
from ..base import ExplanationAlgorithm


@register("saliency", "gradient", "grad")
class Saliency(ExplanationAlgorithm):
    graph_level = True

    def __init__(
        self,
        absolute: bool = True,
        aggregate: str = "sum",
        node_mask_type: str | None = None,
    ):
        self.absolute = absolute
        self.aggregate = aggregate
        self.node_mask_type = node_mask_type

    def explain(
        self,
        backend: Any,
        model: Any,
        data: Any,
        index: int | list[int] | torch.Tensor,
        target_class: int | None = None,
        **kwargs,
    ) -> Explanation:
        model.eval()
        x = backend.node_features(data).detach().clone().requires_grad_(True)
        edge_index = backend.edge_index(data)
        edge_weight = backend.edge_weight(data)

        out = backend.forward(model, x, edge_index, edge_weight=edge_weight)
        logits = out

        if index is None:
            idx = torch.zeros(1, dtype=torch.long, device=x.device)
        elif isinstance(index, int):
            idx = torch.tensor([index], device=x.device)
        else:
            idx = torch.as_tensor(index, device=x.device)

        if target_class is None and logits.dim() == 2:
            target_class = int(logits[idx].argmax(dim=-1)[0].item())

        if logits.dim() == 2:
            score = logits[idx, target_class].sum()
        else:
            score = logits[idx].sum()

        model.zero_grad()
        score.backward()

        grad = x.grad
        if grad is None:
            raise RuntimeError("No se obtuvieron gradientes del modelo.")

        if self.absolute:
            grad = grad.abs()
        if self.aggregate == "sum":
            node_importance = grad.sum(dim=-1)
        elif self.aggregate == "mean":
            node_importance = grad.mean(dim=-1)
        elif self.aggregate == "max":
            node_importance = grad.max(dim=-1).values
        else:
            raise ValueError(f"aggregate desconocido: {self.aggregate}")

        prediction_original = logits[idx].detach()
        feature_importance = (
            grad.detach() if self.node_mask_type == "attributes" else None
        )

        return Explanation(
            node_importance=node_importance.detach().cpu(),
            feature_importance=(
                feature_importance.detach().cpu()
                if feature_importance is not None
                else None
            ),
            edge_importance=None,
            prediction_original=prediction_original.cpu(),
            prediction_explanation=None,
            node_idx=int(idx[0].item()) if isinstance(index, int) else index,
            target_class=target_class,
        )
