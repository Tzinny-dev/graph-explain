from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from ...core.explanation import Explanation
from ...core.registry import register
from ..base import ExplanationAlgorithm


@register("node_mask", "nodemask", "nm")
class NodeMask(ExplanationAlgorithm):
    """NodeMask: node mask learned by optimization.

    Optimizes a (sigmoid) mask over the nodes of the target node's k-hop
    subgraph so the model keeps its prediction, with an entropy regularizer to
    force sparsity. The resulting node importance is re-projected onto the full
    graph (0 outside the neighborhood).
    """

    def __init__(
        self,
        epochs: int = 200,
        lr: float = 0.05,
        hops: int = 3,
        suppress_ratio: float = 0.8,
        entropy: float = 0.05,
        **kwargs,
    ):
        self.epochs = epochs
        self.lr = lr
        self.hops = hops
        self.suppress_ratio = suppress_ratio
        self.entropy = entropy

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
        num_nodes = backend.num_nodes(data)

        nodes = self._to_node_ids(index, num_nodes)
        root = int(nodes[0])

        from torch_geometric.utils import k_hop_subgraph

        device = x.device
        sub_nodes, sub_edge_index, mapping, _ = k_hop_subgraph(
            [root],
            num_hops=self.hops,
            edge_index=edge_index,
            relabel_nodes=True,
        )
        x_sub = x[sub_nodes].to(device)

        with torch.no_grad():
            orig = backend.forward(model, x_sub, sub_edge_index)
        ni = int(mapping.item() if torch.is_tensor(mapping) else mapping)
        if target_class is None and orig.dim() == 2:
            target_class = int(orig[ni].argmax().item())
        target = target_class if target_class is not None else 0

        if self.suppress_ratio > 0:
            k = int(self.suppress_ratio * sub_nodes.numel())
            k = max(0, k)
        else:
            k = max(0, sub_nodes.numel() - 1)

        mask = torch.nn.Parameter(torch.zeros(sub_nodes.numel(), device=device))
        optimizer = torch.optim.Adam([mask], lr=self.lr)

        for _ in range(self.epochs):
            optimizer.zero_grad()
            node_mask = torch.sigmoid(mask)
            pred = backend.forward(model, x_sub, sub_edge_index, node_mask=node_mask)
            topk = node_mask.topk(max(k, 1)).values.min()
            loss = self._loss(pred, ni, target, node_mask, topk)
            loss.backward()
            optimizer.step()

        final = torch.sigmoid(mask).detach()
        full = torch.zeros(num_nodes, dtype=torch.float32)
        full[sub_nodes.cpu()] = final.cpu()

        with torch.no_grad():
            pred_masked = backend.forward(
                model, x_sub, sub_edge_index, node_mask=final
            )[ni]

        return Explanation(
            node_importance=full,
            edge_importance=None,
            feature_importance=None,
            prediction_original=orig[ni].detach().cpu(),
            prediction_explanation=pred_masked.detach().cpu(),
            node_idx=root,
            target_class=target,
        )

    def _loss(self, pred, node_idx, target_class, node_mask, topk) -> torch.Tensor:
        if pred.dim() == 2:
            log_logits = pred.log_softmax(dim=-1)
        else:
            log_logits = pred
        idx = torch.as_tensor([node_idx], device=pred.device).reshape(-1)
        loss = F.nll_loss(
            log_logits[idx],
            torch.tensor([target_class], device=pred.device),
        )
        loss += self.entropy * self._entropy(node_mask)
        loss += (node_mask - topk.detach()).relu().mean()
        return loss

    @staticmethod
    def _entropy(p: torch.Tensor) -> torch.Tensor:
        eps = 1e-8
        return -(p * torch.log(p + eps) + (1 - p) * torch.log(1 - p + eps)).mean()

    @staticmethod
    def _to_node_ids(index, num_nodes: int) -> torch.Tensor:
        if index is None:
            return torch.zeros(1, dtype=torch.long)
        if isinstance(index, int):
            return torch.tensor([index], dtype=torch.long)
        idx = torch.as_tensor(index, dtype=torch.long)
        return idx.reshape(-1) if idx.numel() else torch.zeros(1, dtype=torch.long)
