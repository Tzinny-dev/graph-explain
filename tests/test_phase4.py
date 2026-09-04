from __future__ import annotations

import torch

from graph_explain import Explainer, GNNGatedLRP
from graph_explain.benchmarks.synthetic import build_data
from graph_explain.core.registry import get_algorithm
from tests.test_core import make_model


class TestGNNGatedLRP:
    def _setup(self):
        torch.manual_seed(0)
        data = build_data(base_nodes=60, num_houses=10, m=3, seed=0)
        model = make_model()(data.x.size(1))
        node = int(data.house_anchors[0])
        return data, model, node

    def test_registered(self):
        assert get_algorithm("gnn_lrp") is GNNGatedLRP
        assert get_algorithm("lrp") is GNNGatedLRP

    def test_explain_node(self):
        data, model, node = self._setup()
        explainer = Explainer(algorithm=GNNGatedLRP())
        expl = explainer.explain_node(data, model, node)
        assert expl.node_importance.shape[0] == data.num_nodes
        assert expl.edge_importance is not None
        assert expl.edge_importance.shape[0] == data.edge_index.size(1)
        assert (expl.node_importance >= 0).all()
        assert (expl.edge_importance >= 0).all()
        assert expl.node_idx == node
        assert expl.target_class is not None
        assert expl.prediction_original is not None

    def test_conservation_and_determinism(self):
        data, model, node = self._setup()
        explainer = Explainer(algorithm=GNNGatedLRP())
        expl_a = explainer.explain_node(data, model, node)
        expl_b = explainer.explain_node(data, model, node)
        assert torch.allclose(expl_a.node_importance, expl_b.node_importance)
        total = float(expl_a.node_importance.sum())
        assert 0.2 <= total <= 2.0

    def test_explain_with_target_class(self):
        data, model, node = self._setup()
        expl = Explainer(algorithm=GNNGatedLRP()).explain_node(
            data, model, node, target_class=0
        )
        assert expl.target_class == 0
        assert expl.node_importance.numel() == data.num_nodes
