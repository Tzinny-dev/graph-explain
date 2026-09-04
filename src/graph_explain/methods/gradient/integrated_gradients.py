from __future__ import annotations

from typing import Any

import torch

from ...core.explanation import Explanation
from ...core.registry import register
from ..base import ExplanationAlgorithm


@register("integrated_gradients", "ig")
class IntegratedGradients(ExplanationAlgorithm):
    def __init__(
        self,
        steps: int = 50,
        method: str = "riemann",
        edge_grads: bool = True,
        **kwargs,
    ):
        self.steps = steps
        self.method = method
        self.edge_grads = edge_grads

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
        x = backend.node_features(data)
        edge_index = backend.edge_index(data)

        if isinstance(index, int):
            idx = torch.tensor([index], device=x.device)
        else:
            idx = torch.as_tensor(index, device=x.device)

        with torch.no_grad():
            logits = backend.forward(model, x, edge_index)
        if target_class is None and logits.dim() == 2:
            target_class = int(logits[idx[0]].argmax().item())

        baseline = torch.zeros_like(x)
        alphas, weights = self._alphas(self.steps, self.method, device=x.device)

        ig_feat = torch.zeros_like(x)
        ig_edge = None
        compute_edge = self.edge_grads and backend.supports_edge_weight(model)
        if compute_edge:
            ig_edge = torch.zeros(edge_index.size(1), dtype=torch.float32, device=x.device)

        for alpha, w in zip(alphas, weights):
            x_step = (baseline + alpha * (x - baseline)).requires_grad_(True)
            ew = None
            if compute_edge:
                ew = (torch.ones(edge_index.size(1), device=x.device) * (1.0 - alpha)).requires_grad_(True)
            out = backend.forward(model, x_step, edge_index, edge_weight=ew)
            if out.dim() == 2:
                score = out[idx, target_class].sum()
            else:
                score = out[idx].sum()
            model.zero_grad()
            score.backward()
            ig_feat = ig_feat + w * x_step.grad
            if compute_edge and ew is not None and ew.grad is not None:
                ig_edge = ig_edge + w * ew.grad

        ig_feat = ig_feat * (x - baseline)
        if compute_edge and ig_edge is not None:
            ig_edge = ig_edge * (-1.0)

        grad = ig_feat.detach()
        node_importance = grad.abs().sum(dim=-1)
        feature_importance = grad

        return Explanation(
            node_importance=node_importance.cpu(),
            edge_importance=(
                ig_edge.detach().abs().cpu() if ig_edge is not None else None
            ),
            feature_importance=feature_importance.cpu(),
            prediction_original=logits[idx[0]].detach().reshape(1, -1).cpu(),
            prediction_explanation=None,
            node_idx=int(idx[0].item()) if isinstance(index, int) else index,
            target_class=target_class,
        )

    def _alphas(self, steps: int, method: str, device):
        if method in ("riemann", "left"):
            alphas = torch.arange(0.0, 1.0, 1.0 / steps, device=device)
            return alphas, torch.full_like(alphas, 1.0 / steps)
        if method == "right":
            alphas = torch.arange(1.0 / steps, 1.0 + 1e-6, 1.0 / steps, device=device)
            return alphas, torch.full_like(alphas, 1.0 / steps)
        if method == "gausslegendre":
            from numpy.polynomial.legendre import leggauss

            xs, ws = leggauss(steps)
            alphas = torch.as_tensor((xs + 1) / 2, device=device, dtype=torch.float32)
            weights = torch.as_tensor(ws / 2, device=device, dtype=torch.float32)
            return alphas, weights
        raise ValueError(f"method desconocido: {method}")