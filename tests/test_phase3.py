from __future__ import annotations

import sys
import types

import torch
from numpy.testing import assert_allclose
from torch import nn

from graph_explain import Explainer, GNNExplainer
from graph_explain.backends import get_backend
from graph_explain.benchmarks.synthetic import build_data, ground_truth_nodes
from graph_explain.core.evaluation import (
    evaluate_fidelity_minus,
    evaluate_fidelity_plus,
    evaluate_gea,
    evaluate_stability,
)
from tests.test_core import make_model


class MockGraph:
    def __init__(self, u, v, num_nodes=None):
        self._u = u.long()
        self._v = v.long()
        n = int(num_nodes) if num_nodes is not None else 0
        for node in torch.cat([self._u, self._v]).unique().tolist():
            n = max(n, node + 1)
        self._n = n
        self.ndata: dict = {}
        self.edata: dict = {}

    def num_nodes(self):
        return self._n

    def num_edges(self):
        return self._u.shape[0]

    def edges(self):
        return self._u.clone(), self._v.clone()

    def to_networkx(self):
        import networkx as nx

        G = nx.Graph()
        G.add_nodes_from(range(self._n))
        G.add_edges_from(zip(self._u.tolist(), self._v.tolist()))
        return G


def _install_dgl_mock():
    mod = types.ModuleType("dgl")
    mod.DGLGraph = MockGraph
    mod.graph = lambda uv, num_nodes=None: MockGraph(uv[0], uv[1], num_nodes=num_nodes)
    sys.modules.setdefault("dgl", mod)


class MockDGLGCN(nn.Module):
    task_level = "node"

    def __init__(self, in_channels):
        super().__init__()
        self.lin = nn.Linear(in_channels, 4)

    def forward(self, g, feat):
        src, dst = g.edges()
        w = g.edata.get("w")
        if w is None:
            w = torch.ones(src.shape[0], device=feat.device)
        n = g.num_nodes()
        agg = torch.zeros(n, feat.size(1), device=feat.device)
        agg.index_add_(0, dst, feat[src] * w[:, None])
        deg = torch.zeros(n, device=feat.device)
        deg.index_add_(0, dst, w)
        deg = deg.clamp(min=1.0)
        h = torch.relu(feat / (deg[:, None] + 1.0))
        return self.lin(agg + h)


_install_dgl_mock()


class TestDGLAdapter:
    def test_roundtrip(self):
        from graph_explain.backends.dgl import DGLAdapter

        adapter = DGLAdapter()
        g = MockGraph(torch.tensor([0, 0, 1]), torch.tensor([1, 2, 2]), num_nodes=3)
        g.ndata["feat"] = torch.randn(3, 4)
        g.ndata["label"] = torch.tensor([0, 1, 0])
        g.edata["w"] = torch.tensor([0.5, 0.7, 1.0])

        assert adapter.num_nodes(g) == 3
        assert adapter.node_features(g).shape == (3, 4)
        assert adapter.node_labels(g).tolist() == [0, 1, 0]
        ei = adapter.edge_index(g)
        assert ei.shape == (2, 3)
        assert_allclose(adapter.edge_weight(g).numpy(), [0.5, 0.7, 1.0], rtol=1e-5)

        model = MockDGLGCN(4)
        x = adapter.node_features(g)
        out = adapter.forward(model, x, ei)
        assert out.shape == (3, 4)

        out_zero = adapter.forward(model, x, ei, edge_weight=torch.zeros(3))
        assert not torch.allclose(out, out_zero)

        mask = torch.tensor([1.0, 1.0, 0.0])
        out_masked = adapter.forward(model, x, ei, node_mask=mask)
        assert not torch.allclose(out, out_masked)

    def test_get_backend(self):
        assert get_backend("dgl").name == "dgl"


class TestMetrics:
    def _setup(self):
        torch.manual_seed(0)
        data = build_data(base_nodes=60, num_houses=10, m=3, seed=0)
        model = make_model()(data.x.size(1))
        node = int(data.house_anchors[0])
        explainer = Explainer(algorithm=GNNExplainer(epochs=25, lr=0.01))
        expl = explainer.explain_node(data, model, node)
        return data, model, expl

    def test_fidelity_plus_minus(self):
        _, model, expl = self._setup()
        plus = evaluate_fidelity_plus(model, expl)
        minus = evaluate_fidelity_minus(model, expl)
        assert -1.0 <= plus <= 1.0
        assert 0.0 <= minus <= 1.0

    def test_stability(self):
        data, model, expl = self._setup()
        explainer = Explainer(algorithm=GNNExplainer(epochs=15, lr=0.01))

        def get_explanation(d):
            return explainer.explain_node(d, model, int(expl.node_idx))

        stab = evaluate_stability(
            get_explanation,
            data,
            num_perturbations=4,
            perturbation="feature",
            noise_std=0.02,
            seed=1,
        )
        assert 0.0 <= stab <= 1.0 + 1e-6

    def test_gea(self):
        data, _, expl = self._setup()
        assert len(ground_truth_nodes(data, int(expl.node_idx))) > 0
        gea = evaluate_gea(expl, data=data)
        assert 0.0 <= gea <= 1.0 + 1e-6
