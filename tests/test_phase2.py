from __future__ import annotations

import torch

from graph_explain import (
    Explainer,
    IntegratedGradients,
    PGExplainer,
    SubgraphX,
)
from graph_explain.benchmarks.synthetic import build_data
from tests.test_core import make_model


class TestPGExplainer:
    def test_explain_node(self):
        torch.manual_seed(0)
        data = build_data(base_nodes=60, num_houses=10, m=3, seed=0)
        model = make_model()(data.x.size(1))
        node = int(data.house_anchors[0])
        explainer = Explainer(algorithm=PGExplainer(epochs=15, lr=0.01))
        expl = explainer.explain_node(data, model, node)
        assert expl.edge_importance is not None
        assert expl.edge_importance.shape[0] == data.edge_index.size(1)
        assert expl.node_importance is not None
        assert expl.node_importance.shape[0] == data.num_nodes


class TestSubgraphX:
    def test_explain_node(self):
        torch.manual_seed(0)
        data = build_data(base_nodes=60, num_houses=10, m=3, seed=0)
        model = make_model()(data.x.size(1))
        node = int(data.house_anchors[0])
        explainer = Explainer(
            algorithm=SubgraphX(rollout=5, num_hops=2, max_nodes=10, seed=0)
        )
        expl = explainer.explain_node(data, model, node)
        assert expl.edge_importance is not None
        assert expl.node_importance is not None
        assert (expl.node_importance >= 0).all()
        assert expl.target_class is not None


class TestIntegratedGradients:
    def test_node_feature_and_edge(self):
        torch.manual_seed(0)
        data = build_data(base_nodes=60, num_houses=10, m=3, seed=0)
        model = make_model()(data.x.size(1))
        node = int(data.house_anchors[0])
        explainer = Explainer(algorithm=IntegratedGradients(steps=10))
        expl = explainer.explain_node(data, model, node)
        assert expl.feature_importance is not None
        assert expl.node_importance.shape[0] == data.num_nodes
        assert expl.prediction_original is not None
