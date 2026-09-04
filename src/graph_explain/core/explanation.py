from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import networkx as nx

from ..core.evaluation import evaluate_fidelity, evaluate_sparsity


@dataclass
class Explanation:
    node_importance: Any | None = None
    edge_importance: Any | None = None
    feature_importance: Any | None = None
    subgraph: Any | None = None
    prediction_original: Any | None = None
    prediction_explanation: Any | None = None
    node_idx: int | None = None
    target_class: int | None = None
    mask_threshold: float = 0.5
    metadata: dict = field(default_factory=dict)

    def evaluate(self, metrics: list[str] | None = None, **kwargs) -> dict:
        metrics = metrics or ["fidelity", "sparsity"]
        results: dict = {}
        for name in metrics:
            name = name.lower()
            if name == "fidelity":
                results[name] = evaluate_fidelity(self)
            elif name in ("sparsity", "sparsity_ratio"):
                results[name] = evaluate_sparsity(self, **kwargs)
            else:
                raise ValueError(f"Métrica desconocida: {name}")
        return results

    def to_networkx(self, threshold: float | None = None) -> nx.Graph:
        threshold = threshold if threshold is not None else self.mask_threshold
        backend = self.metadata.get("backend")
        data = self.metadata.get("backing_data")
        if backend is None or data is None:
            raise ValueError(
                "La explicación se creó sin backend/backing_data en metadata"
            )
        G = nx.Graph()
        keep_edges = []
        if self.edge_importance is not None:
            edge_index = backend.edge_index(data)
            for e in range(self.edge_importance.shape[0]):
                if float(self.edge_importance[e]) >= threshold:
                    u = int(edge_index[0, e])
                    v = int(edge_index[1, e])
                    keep_edges.append((u, v, float(self.edge_importance[e])))
        if keep_edges:
            G.add_weighted_edges_from(keep_edges)
        nodes = {n for e in keep_edges for n in e[:2]}
        if self.node_importance is not None:
            for n in range(self.node_importance.shape[0]):
                if float(self.node_importance[n]) >= threshold:
                    nodes.add(n)
        G.add_nodes_from(nodes)
        return G

    def __repr__(self) -> str:
        parts = []
        if self.node_importance is not None:
            parts.append(f"node_importance={tuple(self.node_importance.shape)}")
        if self.edge_importance is not None:
            parts.append(f"edge_importance={tuple(self.edge_importance.shape)}")
        if self.feature_importance is not None:
            parts.append(f"feature_importance={tuple(self.feature_importance.shape)}")
        return f"Explanation({', '.join(parts)})"
