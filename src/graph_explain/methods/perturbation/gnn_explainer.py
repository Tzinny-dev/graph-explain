from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from ...core.explanation import Explanation
from ...core.registry import register
from ..base import ExplanationAlgorithm


@register("gnn_explainer", "gnnexplainer")
class GNNExplainer(ExplanationAlgorithm):
    def __init__(
        self,
        epochs: int = 200,
        lr: float = 0.01,
        edge_entropy: float = 0.001,
        node_entropy: float = 0.001,
        node_mask_type: str | None = "attributes",
        edge_mask_type: str | None = "object",
        prints: int = 20,
        **kwargs,
    ):
        self.epochs = epochs
        self.lr = lr
        self.edge_entropy = edge_entropy
        self.node_entropy = node_entropy
        self.node_mask_type = node_mask_type
        self.edge_mask_type = edge_mask_type
        self.prints = prints

    def explain(
        self,
        backend: Any,
        model: Any,
        data: Any,
        index: int | torch.Tensor,
        target_class: int | None = None,
        **kwargs,
    ) -> Explanation:
        node_mask_type = kwargs.get("node_mask_type", self.node_mask_type)
        edge_mask_type = kwargs.get("edge_mask_type", self.edge_mask_type)

        model.eval()
        x = backend.node_features(data)
        edge_index = backend.edge_index(data)
        num_nodes = backend.num_nodes(data)

        sub_nodes, sub_edge_index, mapping, sub_edge_mask = self._extract_subgraph(
            backend, data, index, edge_index
        )
        device = x.device
        x_sub = x[sub_nodes].to(device)
        sub_graph_level = index is None

        node_mask = None
        if node_mask_type is not None and not sub_graph_level:
            node_mask = torch.nn.Parameter(torch.randn(x_sub.size(0), device=device))

        edge_mask = None
        if edge_mask_type is not None:
            edge_mask = torch.nn.Parameter(
                torch.randn(sub_edge_index.size(1), device=device)
            )

        params = [p for p in (node_mask, edge_mask) if p is not None]
        optimizer = torch.optim.Adam(params, lr=self.lr)

        with torch.no_grad():
            orig_logits = backend.forward(model, x_sub, sub_edge_index)
        if sub_graph_level:
            if target_class is None:
                target_class = (
                    int(orig_logits[0].argmax().item()) if orig_logits.dim() == 2 else 0
                )
            tgt_idx = 0
        else:
            ni = int(mapping.item() if torch.is_tensor(mapping) else mapping)
            if target_class is None and orig_logits.dim() == 2:
                target_class = int(orig_logits[ni].argmax().item())
            tgt_idx = ni

        for epoch in range(self.epochs):
            optimizer.zero_grad()
            mask_node = None
            if node_mask is not None:
                mask_node = torch.sigmoid(node_mask)
            eweight = None
            if edge_mask is not None:
                eweight = torch.sigmoid(edge_mask)

            pred = backend.forward(
                model, x_sub, sub_edge_index, edge_weight=eweight, node_mask=mask_node
            )
            loss = self._loss(
                pred, tgt_idx, sub_graph_level, target_class, node_mask, edge_mask
            )
            loss.backward()
            optimizer.step()

        logits = backend.forward(model, x_sub, sub_edge_index)

        if sub_graph_level:
            target_class = target_class or (int(logits.argmax(-1)[0].item()) if logits.dim() == 2 else 0)
            ni = 0
            pred_orig = logits[0].detach()
            final_mask_node = None
            final_mask_edge = None
            if node_mask is not None:
                final_mask_node = torch.sigmoid(node_mask)
            if edge_mask is not None:
                final_mask_edge = torch.sigmoid(edge_mask)
            pred_masked = backend.forward(
                model, x_sub, sub_edge_index, edge_weight=final_mask_edge, node_mask=final_mask_node
            )[0].detach()
        else:
            ni = int(mapping.item() if torch.is_tensor(mapping) else mapping)
            if target_class is None and logits.dim() == 2:
                target_class = int(logits[ni].argmax().item())
            pred_orig = logits[ni].detach()
            final_mask_node = None
            final_mask_edge = None
            if node_mask is not None:
                final_mask_node = torch.sigmoid(node_mask)
            if edge_mask is not None:
                final_mask_edge = torch.sigmoid(edge_mask)
            pred_masked = backend.forward(
                model, x_sub, sub_edge_index, edge_weight=final_mask_edge, node_mask=final_mask_node
            )[ni].detach()

        full_num_nodes = num_nodes
        full_edge_count = edge_index.size(1)

        node_full = self._scatter_node(node_mask, sub_nodes, full_num_nodes)
        edge_full = self._scatter_edge(edge_mask, sub_edge_mask, full_edge_count)

        return Explanation(
            node_importance=node_full,
            edge_importance=edge_full,
            feature_importance=None,
            prediction_original=pred_orig.cpu(),
            prediction_explanation=pred_masked.cpu(),
            node_idx=None if sub_graph_level else (int(index[0]) if torch.is_tensor(index) else int(index)),
            target_class=target_class,
            metadata={
                "sub_nodes": sub_nodes,
                "sub_edge_index": sub_edge_index,
                "sub_edge_mask": sub_edge_mask,
            },
        )

    @staticmethod
    def _scatter_node(
        mask: torch.nn.Parameter | None,
        sub_nodes: torch.Tensor,
        num_nodes: int,
    ) -> torch.Tensor | None:
        if mask is None:
            return None
        full = torch.zeros(num_nodes, dtype=torch.float32)
        vals = torch.sigmoid(mask).detach().cpu()
        full[sub_nodes.cpu()] = vals
        return full

    @staticmethod
    def _scatter_edge(
        mask: torch.nn.Parameter | None,
        sub_edge_mask: torch.Tensor,
        num_edges: int,
    ) -> torch.Tensor | None:
        if mask is None:
            return None
        full = torch.zeros(num_edges, dtype=torch.float32)
        idx = sub_edge_mask.nonzero(as_tuple=False).view(-1)
        full[idx.cpu()] = torch.sigmoid(mask).detach().cpu()
        return full

    def _loss(
        self,
        pred: torch.Tensor,
        node_idx: int | torch.Tensor,
        sub_graph_level: bool,
        target_class: int | None,
        node_mask: torch.nn.Parameter | None,
        edge_mask: torch.nn.Parameter | None,
    ) -> torch.Tensor:
        if pred.dim() == 2:
            log_logits = pred.log_softmax(dim=-1)
        else:
            log_logits = pred

        if sub_graph_level:
            idx = torch.zeros(1, dtype=torch.long, device=pred.device)
            if target_class is None:
                target_class = int(pred[0].argmax().item())
            loss = F.nll_loss(log_logits[0].unsqueeze(0), torch.tensor([target_class], device=pred.device))
        else:
            if isinstance(node_idx, torch.Tensor) and node_idx.dim() == 0:
                idx = node_idx.unsqueeze(0)
            else:
                idx = torch.as_tensor([node_idx], device=pred.device) if not torch.is_tensor(node_idx) else node_idx.reshape(-1)
            if target_class is None:
                target_class = int(pred[idx].argmax(dim=-1)[0].item())
            loss = F.nll_loss(log_logits[idx], torch.tensor([target_class], device=pred.device))

        if edge_mask is not None and self.edge_entropy > 0:
            loss += self.edge_entropy * self._entropy(torch.sigmoid(edge_mask))
        if node_mask is not None and self.node_entropy > 0:
            loss += self.node_entropy * self._entropy(torch.sigmoid(node_mask))
        return loss

    @staticmethod
    def _entropy(p: torch.Tensor) -> torch.Tensor:
        eps = 1e-8
        return -(p * torch.log(p + eps) + (1 - p) * torch.log(1 - p + eps)).mean()

    def _extract_subgraph(
        self, backend: Any, data: Any, index: int | torch.Tensor | None, edge_index: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        from torch_geometric.utils import k_hop_subgraph

        if index is None:
            n = edge_index.max().item() + 1
            node_idx = torch.arange(n, device=edge_index.device)
            return (
                node_idx,
                edge_index,
                node_idx,
                torch.ones(edge_index.size(1), dtype=torch.bool, device=edge_index.device),
            )
        node_idx = torch.as_tensor([index], device=edge_index.device).reshape(-1)
        return k_hop_subgraph(node_idx, num_hops=3, edge_index=edge_index, relabel_nodes=True)