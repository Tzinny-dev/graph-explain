from __future__ import annotations

from typing import Any

import torch
from torch import nn

from ...core.explanation import Explanation
from ...core.registry import register
from ..base import ExplanationAlgorithm


@register("guided_backprop", "guided-backprop", "gbp")
class GuidedBackprop(ExplanationAlgorithm):
    """Guided Backpropagation: gradients guided by the ReLU mask.

    During backpropagation the gradient is filtered: it only propagates where
    the ReLU activation was positive (negative gradients are discarded),
    highlighting the features that positively contribute to the class. Temporary
    hooks are registered on the `nn.ReLU` modules; if the model has none, it
    falls back to standard gradients (metadata `guided=False`).
    """

    graph_level = True

    def __init__(self, fallback_to_gradient: bool = True, **kwargs):
        self.fallback_to_gradient = fallback_to_gradient

    def explain(
        self,
        backend: Any,
        model: Any,
        data: Any,
        index: int | torch.Tensor | None = None,
        target_class: int | None = None,
        **kwargs,
    ) -> Explanation:
        model.eval()
        x = backend.node_features(data)
        edge_index = backend.edge_index(data)
        num_nodes = int(x.size(0))

        nodes = self._to_node_ids(index, num_nodes)
        root = int(nodes[0])
        with torch.no_grad():
            logits = backend.forward(model, x, edge_index)
        if logits.dim() != 2:
            raise ValueError(
                "GuidedBackprop requiere predicciones node-level (logits (N, C))."
            )
        target = target_class
        if target is None:
            target = int(logits[root].argmax().item())
        target = max(0, min(int(target), logits.size(1) - 1))

        relus = [m for m in model.modules() if isinstance(m, nn.ReLU)]
        hooks, guided = [], False
        if relus and self.fallback_to_gradient:
            guided = True
            for module in relus:
                fw = module.register_forward_hook(self._mask_forward)
                bw = module.register_full_backward_hook(self._guide_backward)
                hooks.extend((fw, bw))

        try:
            x_in = x.detach().clone().requires_grad_(True)
            out = backend.forward(model, x_in, edge_index)
            if out.dim() != 2:
                raise ValueError("GuidedBackprop requiere predicciones node-level.")
            score = out[root, target]
            model.zero_grad()
            score.backward()
            grad = (
                x_in.grad.detach() if x_in.grad is not None else torch.zeros_like(x_in)
            )
        finally:
            for hook in hooks:
                hook.remove()

        node_importance = grad.abs().sum(dim=-1)
        return Explanation(
            node_importance=node_importance.cpu(),
            edge_importance=None,
            feature_importance=grad.cpu(),
            prediction_original=logits[root].detach().cpu(),
            prediction_explanation=None,
            node_idx=root,
            target_class=target,
            metadata={"guided": guided},
        )

    @staticmethod
    def _mask_forward(module, inp, out):
        mask = torch.where(out > 0, torch.ones_like(out), torch.zeros_like(out))
        module._gbp_mask = mask.detach()

    @staticmethod
    def _guide_backward(module, grad_input, grad_output):
        mask = module._gbp_mask
        if mask is None or not grad_output or grad_output[0] is None:
            return grad_input
        g = grad_output[0]
        if g.shape != mask.shape:
            return grad_input
        guided = (g * mask).clamp(min=0)
        if len(grad_input) == 1:
            return (guided,)
        return grad_input

    @staticmethod
    def _to_node_ids(index, num_nodes: int) -> torch.Tensor:
        if index is None:
            return torch.zeros(1, dtype=torch.long)
        if isinstance(index, int):
            return torch.tensor([index], dtype=torch.long)
        idx = torch.as_tensor(index, dtype=torch.long)
        return idx.reshape(-1) if idx.numel() else torch.zeros(1, dtype=torch.long)
