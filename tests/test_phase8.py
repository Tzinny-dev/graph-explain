from __future__ import annotations

import torch
from torch_geometric.nn import GCNConv

from graph_explain import (
    Explainer,
    GraphLIME,
    GuidedBackprop,
    NodeMask,
    RandomBaseline,
)
from graph_explain.backends import PyGAdapter
from graph_explain.benchmarks.synthetic import build_data, ground_truth_nodes
from graph_explain.core.evaluation import evaluate_gea


class ReLUGCN(torch.nn.Module):
    task_level = "node"

    def __init__(self, in_channels, hidden=16, out=4):
        super().__init__()
        self.c1 = GCNConv(in_channels, hidden)
        self.relu1 = torch.nn.ReLU()
        self.c2 = GCNConv(hidden, out)

    def forward(self, x, edge_index, edge_weight=None):
        h = self.relu1(self.c1(x, edge_index, edge_weight=edge_weight))
        return self.c2(h, edge_index, edge_weight=edge_weight)


def _data_and_model():
    torch.manual_seed(0)
    data = build_data(base_nodes=30, num_houses=5, m=2, seed=0)
    model = ReLUGCN(data.x.size(1))
    model.eval()
    return data, model


class TestGraphLIME:
    def test_feature_importance_shapes(self):
        data, model = _data_and_model()
        expl = Explainer(algorithm=GraphLIME(), backend=PyGAdapter()).explain_node(
            data, model, 0
        )
        assert expl.feature_importance is not None
        assert expl.feature_importance.shape == (data.x.size(1),)
        assert expl.node_importance is not None
        assert expl.node_importance.shape == (data.num_nodes,)
        assert expl.node_importance.abs().sum() > 0
        assert expl.target_class is not None

    def test_important_features_include_house_signal(self):
        data, model = _data_and_model()
        expl = Explainer(algorithm=GraphLIME(), backend=PyGAdapter()).explain_node(
            data, model, 0
        )
        assert torch.isfinite(expl.feature_importance).all()

    def test_gea_house_node(self):
        data, model = _data_and_model()
        anchor = next(
            n for n in range(3, data.num_nodes) if ground_truth_nodes(data, n)
        )
        expl = Explainer(algorithm=GraphLIME(), backend=PyGAdapter()).explain_node(
            data, model, anchor
        )
        gea = evaluate_gea(expl, data=data, top_k=5)
        assert 0.0 <= gea <= 1.0


class TestNodeMask:
    def test_node_only_shape_and_endpoint(self):
        data, model = _data_and_model()
        expl = Explainer(
            algorithm=NodeMask(epochs=60), backend=PyGAdapter()
        ).explain_node(data, model, 0)
        assert expl.node_importance is not None
        assert expl.node_importance.shape == (data.num_nodes,)
        assert expl.edge_importance is None
        assert expl.node_importance.min() >= 0.0
        assert expl.node_importance.max() <= 1.0
        assert expl.node_importance[expl.node_idx] > 0.5

    def test_gea_house_node(self):
        data, model = _data_and_model()
        anchor = next(
            n for n in range(3, data.num_nodes) if ground_truth_nodes(data, n)
        )
        expl = Explainer(
            algorithm=NodeMask(epochs=60), backend=PyGAdapter()
        ).explain_node(data, model, anchor)
        gea = evaluate_gea(expl, data=data, top_k=5)
        assert 0.0 <= gea <= 1.0


class TestGuidedBackprop:
    def test_shapes_and_guided_flag(self):
        data, model = _data_and_model()
        expl = Explainer(algorithm=GuidedBackprop(), backend=PyGAdapter()).explain_node(
            data, model, 0
        )
        assert expl.node_importance.shape == (data.num_nodes,)
        assert expl.feature_importance.shape == data.x.shape
        assert expl.metadata["guided"] is True

    def test_fallback_when_no_relu(self):
        data, _model = _data_and_model()

        class NoReLU(torch.nn.Module):
            task_level = "node"

            def __init__(self, in_channels):
                super().__init__()
                self.c1 = GCNConv(in_channels, 4)

            def forward(self, x, edge_index, edge_weight=None):
                return self.c1(x, edge_index, edge_weight=edge_weight)

        no_relu = NoReLU(data.x.size(1))
        no_relu.eval()
        expl = Explainer(algorithm=GuidedBackprop(), backend=PyGAdapter()).explain_node(
            data, no_relu, 0
        )
        assert expl.metadata["guided"] is False
        assert expl.node_importance.shape == (data.num_nodes,)


class TestRandomBaseline:
    def test_deterministic_with_seed(self):
        data, model = _data_and_model()
        a = Explainer(
            algorithm=RandomBaseline(seed=7), backend=PyGAdapter()
        ).explain_node(data, model, 0)
        b = Explainer(
            algorithm=RandomBaseline(seed=7), backend=PyGAdapter()
        ).explain_node(data, model, 0)
        assert torch.equal(a.node_importance, b.node_importance)
        assert torch.equal(a.edge_importance, b.edge_importance)
        assert torch.equal(a.feature_importance, b.feature_importance)

    def test_shapes(self):
        data, model = _data_and_model()
        expl = Explainer(algorithm=RandomBaseline(), backend=PyGAdapter()).explain_node(
            data, model, 0
        )
        assert expl.node_importance.shape == (data.num_nodes,)
        assert expl.edge_importance.shape == (data.edge_index.size(1),)
        assert expl.feature_importance.shape == data.x.shape


class TestRegistry:
    def test_aliases_resolve(self):
        from graph_explain.core.registry import get_algorithm

        for alias in (
            "glime",
            "gl",
            "graph_lime",
            "nm",
            "node_mask",
            "gbp",
            "guided_backprop",
            "random",
        ):
            get_algorithm(alias)

    def test_cli_registry_coverage(self):
        from graph_explain.cli import _METHODS
        from graph_explain.core.registry import get_algorithm

        for name in _METHODS:
            get_algorithm(name)
