from __future__ import annotations

import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv

from graph_explain import (
    AttentionExplainer,
    DeepLift,
    Explainer,
    GradXInput,
)
from graph_explain.backends import PyGAdapter
from graph_explain.benchmarks.synthetic import build_data, ground_truth_nodes


class NNReLUGCN(torch.nn.Module):
    task_level = "node"

    def __init__(self, in_channels, hidden=16, out=4):
        super().__init__()
        self.c1 = GCNConv(in_channels, hidden)
        self.relu1 = torch.nn.ReLU()
        self.c2 = GCNConv(hidden, out)

    def forward(self, x, edge_index, edge_weight=None):
        h = self.relu1(self.c1(x, edge_index, edge_weight=edge_weight))
        return self.c2(h, edge_index, edge_weight=edge_weight)


class GAT(torch.nn.Module):
    task_level = "node"

    def __init__(self, in_channels, hidden=8, out=4):
        super().__init__()
        self.c1 = GATConv(in_channels, hidden, heads=2, concat=False)
        self.relu = torch.nn.ReLU()
        self.c2 = GATConv(hidden, out, heads=1)

    def forward(self, x, edge_index, edge_weight=None):
        h = self.relu(self.c1(x, edge_index))
        return self.c2(h, edge_index)


def _train(model, data, epochs=150, lr=0.01):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        opt.step()
    model.eval()


class TestDeepLift:
    def test_shapes_and_conservation(self):
        torch.manual_seed(0)
        data = build_data(base_nodes=40, num_houses=6, m=2, seed=0)
        model = NNReLUGCN(data.x.size(1))
        node = int(data.house_anchors[0])
        expl = Explainer(algorithm=DeepLift(), backend=PyGAdapter()).explain_node(
            data, model, node
        )

        assert expl.node_importance.shape == (data.num_nodes,)
        assert expl.edge_importance.shape == (data.edge_index.size(1),)
        assert expl.feature_importance.shape == (data.num_nodes, data.x.size(1))

        target = int(expl.prediction_original.argmax().item())
        with torch.no_grad():
            l1 = model(data.x, data.edge_index)[node]
            l0 = model(torch.zeros_like(data.x), data.edge_index)[node]
        delta = float((l1[target] - l0[target]).item())
        contrib_total = float(expl.feature_importance.sum().item())
        assert contrib_total != 0.0
        assert abs(contrib_total - delta) < 1e-3

    def test_trained_model_recovers_house_motif(self):
        torch.manual_seed(0)
        data = build_data(base_nodes=60, num_houses=10, m=3, seed=0)
        model = NNReLUGCN(data.x.size(1))
        _train(model, data, epochs=150)
        node = int(data.house_anchors[0])
        expl = Explainer(
            algorithm=DeepLift(normalize=True), backend=PyGAdapter()
        ).explain_node(data, model, node)

        top_k_nodes = set(expl.node_importance.argsort(descending=True)[:5].tolist())
        gt = set(ground_truth_nodes(data, node))
        # con baseline cero la contribución escala con la magnitud de las
        # features (grado one-hot), así que admitimos solape parcial
        assert len(top_k_nodes & gt) >= 2, f"top-5 vs motivo: {top_k_nodes} vs {gt}"


class TestAttentionExplainer:
    def test_gat_edge_and_node_importance(self):
        torch.manual_seed(0)
        data = build_data(base_nodes=40, num_houses=6, m=2, seed=0)
        model = GAT(data.x.size(1))
        node = 0
        expl = Explainer(
            algorithm=AttentionExplainer(), backend=PyGAdapter()
        ).explain_node(data, model, node)

        assert expl.edge_importance.shape == (data.edge_index.size(1),)
        assert expl.node_importance.shape == (data.num_nodes,)
        assert expl.edge_importance.min().item() >= 0.0
        assert expl.edge_importance.sum().item() > 0.0
        # cada softmax por vecino suma 1 por nodo y cabeza; media sobre 2 capas
        # normalización por vecino (incluye self-loops): total <= num_nodes
        total = expl.edge_importance.sum().item()
        assert 0.0 < total <= data.num_nodes
        # node_aggregate="sum" cuenta cada arista incidente (entrada+salida)
        assert abs(expl.node_importance.sum().item() - 2 * total) < 1e-3

    def test_requires_gat(self):
        import pytest

        data = build_data(base_nodes=30, num_houses=5, m=2, seed=0)
        model = NNReLUGCN(data.x.size(1))
        with pytest.raises(ValueError, match="GATConv"):
            Explainer(
                algorithm=AttentionExplainer(), backend=PyGAdapter()
            ).explain_node(data, model, 0)


class TestGradXInput:
    def test_shapes(self):
        torch.manual_seed(0)
        data = build_data(base_nodes=40, num_houses=6, m=2, seed=0)
        model = NNReLUGCN(data.x.size(1))
        node = int(data.house_anchors[0])
        expl = Explainer(algorithm=GradXInput(), backend=PyGAdapter()).explain_node(
            data, model, node
        )

        assert expl.node_importance.shape == (data.num_nodes,)
        assert expl.feature_importance.shape == (data.num_nodes, data.x.size(1))
        assert expl.edge_importance.shape == (data.edge_index.size(1),)
        assert expl.node_importance.sum().item() > 0.0
