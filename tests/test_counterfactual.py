from __future__ import annotations

import warnings

import torch

from graph_explain import Counterfactual, Explainer
from graph_explain.benchmarks.synthetic import build_data
from graph_explain.core.registry import get_algorithm
from tests.test_core import make_model

warnings.filterwarnings("ignore")


class TestCounterfactual:
    def test_registered(self):
        assert get_algorithm("counterfactual") is Counterfactual
        assert get_algorithm("cf") is Counterfactual

    def _untrained_data_model(self):
        torch.manual_seed(0)
        data = build_data(base_nodes=60, num_houses=10, m=3, seed=0)
        model = make_model()(data.x.size(1))
        return data, model, int(data.house_anchors[0])

    def test_edge_mode_untrained(self):
        data, model, node = self._untrained_data_model()
        expl = Explainer(
            algorithm=Counterfactual(mode="edge", max_steps=5, hops=2)
        ).explain_node(data, model, node)
        assert expl.node_importance.shape[0] == data.num_nodes
        assert expl.edge_importance.shape[0] == data.edge_index.size(1)
        assert set(expl.edge_importance.tolist()) <= {0.0, 1.0}
        assert expl.prediction_original is not None
        assert expl.prediction_explanation is not None
        assert expl.metadata.get("counterfactual") is True

    def test_edge_mode_flips(self):
        import warnings

        from examples.model import default_data_and_model, train

        warnings.filterwarnings("ignore")
        torch.manual_seed(0)
        data, model = default_data_and_model(base_nodes=120, num_houses=20)
        train(model, data, epochs=120)
        with torch.no_grad():
            pred = model(data.x, data.edge_index).argmax(dim=-1)
        class2 = (data.y == 2).nonzero(as_tuple=False).flatten()
        correct = class2[pred[class2] == data.y[class2]]
        node = int(correct[0])
        expl = Explainer(
            algorithm=Counterfactual(mode="edge", max_steps=8, hops=2)
        ).explain_node(data, model, node)
        orig = int(expl.prediction_original.argmax())
        new = int(expl.prediction_explanation.argmax())
        removed = int(expl.edge_importance.sum())
        assert new != orig
        assert 1 <= removed <= 8

    def test_feature_mode(self):
        data, model, node = self._untrained_data_model()
        expl = Explainer(
            algorithm=Counterfactual(mode="feature", max_steps=10)
        ).explain_node(data, model, node)
        assert expl.feature_importance.shape[0] == data.x.size(1)
        assert set(expl.feature_importance.tolist()) <= {0.0, 1.0}
        assert expl.node_importance[node] == 1.0
        assert expl.edge_importance is None
