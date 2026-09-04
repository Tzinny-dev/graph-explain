from __future__ import annotations

import torch

from graph_explain import Explainer, GNNExplainer, Saliency
from graph_explain.backends import PyGAdapter
from graph_explain.benchmarks.synthetic import build_data
from graph_explain.core.registry import get_algorithm
from graph_explain.visualization import visualize_static


def make_model():
    import torch.nn.functional as F
    from torch_geometric.nn import GCNConv

    class GCN(torch.nn.Module):
        task_level = "node"

        def __init__(self, in_channels):
            super().__init__()
            self.c1 = GCNConv(in_channels, 16)
            self.c2 = GCNConv(16, 4)

        def forward(self, x, edge_index, edge_weight=None):
            x = F.relu(self.c1(x, edge_index, edge_weight=edge_weight))
            return self.c2(x, edge_index, edge_weight=edge_weight)

    return GCN


class TestBenchmark:
    def test_shapes(self):
        data = build_data(base_nodes=60, num_houses=10, seed=0)
        assert data.x.size(0) == data.num_nodes == 60 + 50
        assert data.x.size(0) == data.y.size(0)
        assert data.edge_index.size(0) == 2
        assert set(data.y.unique().tolist()) == {0, 1, 2, 3}
        assert (data.y[data.house_anchors] == 0).all()


class TestRegistration:
    def test_names(self):
        assert get_algorithm("gnn_explainer") is GNNExplainer
        assert get_algorithm("gnnexplainer") is GNNExplainer
        assert get_algorithm("saliency") is Saliency
        assert get_algorithm("gradient") is Saliency


class TestExplain:
    def test_gnnexplainer_node(self):
        torch.manual_seed(0)
        data = build_data(base_nodes=60, num_houses=10, m=3, seed=0)
        GCN = make_model()
        model = GCN(data.x.size(1))
        node = int(data.house_anchors[0])
        explainer = Explainer(
            algorithm=GNNExplainer(epochs=20, lr=0.01),
            backend=PyGAdapter(),
        )
        expl = explainer.explain_node(data, model, node)
        assert expl.node_importance.shape[0] == data.num_nodes
        assert expl.edge_importance.shape[0] == data.edge_index.size(1)
        assert expl.node_idx == node
        assert 0.0 <= expl.mask_threshold <= 1.0
        metrics = expl.evaluate(metrics=["fidelity", "sparsity"])
        assert "fidelity" in metrics and "sparsity" in metrics
        local = expl.evaluate(metrics=["sparsity"], local=True)
        assert 0.0 <= local["sparsity"] <= 1.0

    def test_saliency_node(self):
        torch.manual_seed(0)
        data = build_data(base_nodes=60, num_houses=10, m=3, seed=0)
        GCN = make_model()
        model = GCN(data.x.size(1))
        node = int(data.house_anchors[0])
        explainer = Explainer(
            algorithm=Saliency(), edge_mask_type=None, node_mask_type=None
        )
        expl = explainer.explain_node(data, model, node)
        assert expl.node_importance is not None
        assert expl.node_importance.shape[0] == data.num_nodes
        assert expl.prediction_original is not None
        assert expl.target_class is not None


class TestVisualization:
    def test_no_edges_when_importance_none(self):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        data = build_data(base_nodes=60, num_houses=10, m=3, seed=0)
        GCN = make_model()
        model = GCN(data.x.size(1))
        explainer = Explainer(
            algorithm=Saliency(), edge_mask_type=None, node_mask_type=None
        )
        expl = explainer.explain_node(data, model, int(data.house_anchors[0]))
        visualize_static(expl, threshold=0.5)
        fig = plt.gcf()
        assert fig is not None
        plt.close(fig)
