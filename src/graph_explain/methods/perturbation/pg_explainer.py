from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from ...core.explanation import Explanation
from ...core.model_utils import capture_node_embeddings, edge_embeddings
from ...core.registry import register
from ..base import ExplanationAlgorithm


class _EdgeMaskMLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, edge_emb: torch.Tensor) -> torch.Tensor:
        return self.mlp(edge_emb).squeeze(-1)


@register("pg_explainer", "pgexplainer")
class PGExplainer(ExplanationAlgorithm):
    def __init__(
        self,
        epochs: int = 100,
        lr: float = 0.01,
        hidden: int = 64,
        temp: float = 1.0,
        loss_coeff: float = 0.5,
        entropy_coeff: float = 0.005,
        batch_nodes: int = 32,
        **kwargs,
    ):
        self.epochs = epochs
        self.lr = lr
        self.hidden = hidden
        self.temp = temp
        self.loss_coeff = loss_coeff
        self.entropy_coeff = entropy_coeff
        self.batch_nodes = batch_nodes

    def explain(
        self,
        backend: Any,
        model: Any,
        data: Any,
        index: int | torch.Tensor | None,
        target_class: int | None = None,
        **kwargs,
    ) -> Explanation:
        model.eval()
        x = backend.node_features(data)
        edge_index = backend.edge_index(data)
        num_nodes = backend.num_nodes(data)

        embeddings = capture_node_embeddings(model, backend, data)
        edge_emb = edge_embeddings(embeddings, edge_index).detach()
        in_dim = edge_emb.size(-1)

        with torch.no_grad():
            base_logits = backend.forward(model, x, edge_index)

        mlp = _EdgeMaskMLP(in_dim, hidden=self.hidden)
        optimizer = torch.optim.Adam(mlp.parameters(), lr=self.lr)

        train_idx = (
            torch.arange(num_nodes, device=x.device)
            if not hasattr(data, "train_mask")
            else data.train_mask.nonzero(as_tuple=False).view(-1)
        )

        for epoch in range(self.epochs):
            mlp.train()
            optimizer.zero_grad()
            idx = train_idx[
                torch.randperm(len(train_idx), device=x.device)[: self.batch_nodes]
            ]
            edge_weight = self._sample_mask(mlp, edge_emb, x.device)
            pred = backend.forward(model, x, edge_index, edge_weight=edge_weight)
            if pred.dim() == 2:
                target = base_logits[idx].argmax(dim=-1)
                ce = F.nll_loss(pred[idx].log_softmax(dim=-1), target)
            else:
                ce = -pred[idx].mean()
            logit = mlp(edge_emb)
            p = torch.sigmoid(logit)
            entropy = (
                -(p * torch.clamp(p, 1e-8, 1).log())
                - ((1 - p) * torch.clamp(1 - p, 1e-8, 1).log())
            ).mean()
            sparsity = edge_weight.mean()
            loss = ce + self.loss_coeff * sparsity + self.entropy_coeff * entropy
            loss.backward()
            optimizer.step()

        mlp.eval()
        with torch.no_grad():
            edge_mask = torch.sigmoid(mlp(edge_emb))

        ni = (
            0
            if index is None
            else (int(index[0]) if torch.is_tensor(index) else int(index))
        )
        pred_masked = backend.forward(model, x, edge_index, edge_weight=edge_mask)
        pred_node = (
            pred_masked[ni] if pred_masked.dim() == 2 else pred_masked.unsqueeze(0)
        )

        node_importance = self._node_importance(edge_mask, edge_index, num_nodes)

        return Explanation(
            node_importance=node_importance,
            edge_importance=edge_mask.detach().cpu(),
            feature_importance=None,
            prediction_original=(
                base_logits[ni].detach().reshape(1, -1).cpu()
                if base_logits.dim() == 2
                else base_logits.detach().reshape(1, -1).cpu()
            ),
            prediction_explanation=pred_node.detach().reshape(1, -1).cpu(),
            node_idx=None if index is None else ni,
            target_class=target_class,
            metadata={},
        )

    def _sample_mask(
        self,
        mlp: _EdgeMaskMLP,
        edge_emb: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        logits = mlp(edge_emb)
        noise = torch.rand_like(logits)
        gumbel_noise = torch.log(noise) - torch.log(1 - noise + 1e-8)
        return torch.sigmoid((logits + gumbel_noise) / self.temp)

    @staticmethod
    def _node_importance(
        edge_mask: torch.Tensor,
        edge_index: torch.Tensor,
        num_nodes: int,
    ) -> torch.Tensor:
        w = edge_mask.detach()
        src, dst = edge_index[0], edge_index[1]
        node_imp = torch.zeros(num_nodes, dtype=torch.float32, device=edge_mask.device)
        node_imp.scatter_add_(0, src, w)
        node_imp.scatter_add_(0, dst, w)
        degree = torch.zeros(num_nodes, dtype=torch.float32, device=edge_mask.device)
        degree.scatter_add_(0, src, torch.ones_like(src, dtype=torch.float32))
        degree.scatter_add_(0, dst, torch.ones_like(dst, dtype=torch.float32))
        degree.clamp_(min=1)
        return (node_imp / degree).cpu()
