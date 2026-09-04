from __future__ import annotations

from collections import defaultdict
from typing import Any

import torch

from ...core.explanation import Explanation
from ...core.registry import register
from ..base import ExplanationAlgorithm


@register("graph_lime", "glime", "gl")
class GraphLIME(ExplanationAlgorithm):
    """GraphLIME: feature attribution via weighted local regression.

    Fits a linear regression (ridge, closed form) over the k-hop neighbors'
    features, weighting each neighbor by its similarity to the target node's
    feature (Gaussian kernel). The coefficients explain the probability
    (softmax) of the target class; node importance matches the kernel
    similarity.
    """

    def __init__(
        self,
        hops: int = 2,
        lambda_: float = 1.0,
        sigma: float | None = None,
        normalize: bool = True,
        **kwargs,
    ):
        self.hops = hops
        self.lambda_ = lambda_
        self.sigma = sigma
        self.normalize = normalize

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
                "GraphLIME requiere predicciones node-level (logits (N, C))."
            )
        target = target_class
        if target is None:
            target = int(logits[root].argmax().item())
        target = max(0, min(int(target), logits.size(1) - 1))

        probs = torch.softmax(logits, dim=-1)[:, target].detach().cpu()

        neighbors = self._khop_neighbors(edge_index, root, self.hops)
        if root not in neighbors:
            neighbors.append(root)
        nb = torch.tensor(neighbors, dtype=torch.long)
        x_nb = x[nb].detach().cpu()
        y = probs[nb]

        dists = torch.norm(x_nb - x[root : root + 1].detach().cpu(), dim=-1)
        sigma = self.sigma
        if sigma is None:
            tail = dists[1:]
            sigma = float(tail.mean().item()) if tail.numel() else 1.0
            sigma = max(sigma, 1e-4)
        weights = torch.exp(-(dists**2) / (2.0 * sigma**2))

        coef = self._ridge(x_nb, y, weights, self.lambda_)

        node_importance = torch.zeros(num_nodes)
        node_importance[nb] = weights
        node_importance = node_importance.cpu()

        if self.normalize and coef.numel():
            denom = coef.abs().max()
            if denom > 1e-12:
                coef = coef / denom

        return Explanation(
            node_importance=node_importance,
            edge_importance=None,
            feature_importance=coef,
            prediction_original=logits[root].detach().cpu(),
            prediction_explanation=None,
            node_idx=root,
            target_class=target,
            metadata={"neighborhood": neighbors},
        )

    @staticmethod
    def _ridge(
        x: torch.Tensor, y: torch.Tensor, weights: torch.Tensor, lambda_: float
    ) -> torch.Tensor:
        ones = torch.ones(x.size(0), 1)
        X = torch.cat([x, ones], dim=-1).double()
        yw = (y * weights).double()
        Xw = X * weights.unsqueeze(-1).double()
        n_features = X.size(1)
        gram = Xw.t() @ X + lambda_ * torch.eye(n_features, dtype=torch.double)
        beta = torch.linalg.solve(gram, X.t() @ yw)
        return beta[:-1].float()

    @staticmethod
    def _khop_neighbors(edge_index: torch.Tensor, node: int, hops: int) -> list[int]:
        adj: dict[int, set[int]] = defaultdict(set)
        src = edge_index[0].tolist()
        dst = edge_index[1].tolist()
        for s, d in zip(src, dst):
            adj[s].add(d)
            adj[d].add(s)
        seen = {node}
        frontier = {node}
        for _ in range(hops):
            nxt: set[int] = set()
            for n in frontier:
                nxt |= adj.get(n, set())
            frontier = nxt - seen
            seen |= frontier
        return sorted(seen)

    @staticmethod
    def _to_node_ids(index, num_nodes: int) -> torch.Tensor:
        if index is None:
            return torch.zeros(1, dtype=torch.long)
        if isinstance(index, int):
            return torch.tensor([index], dtype=torch.long)
        idx = torch.as_tensor(index, dtype=torch.long)
        return idx.reshape(-1) if idx.numel() else torch.zeros(1, dtype=torch.long)
